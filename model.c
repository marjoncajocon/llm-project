// model.c - GPT model: embeddings, stacked transformer blocks, output head
#include "model.h"

#include <stdlib.h>
#include <string.h>

#include "embedding.h"
#include "attention.h"
#include "layernorm.h"
#include "matrix.h"
#include "sampler.h"

// ---------------------------------------------------------------------------
// Small init helpers
// ---------------------------------------------------------------------------

// Create a randomly initialized weight tensor of the given 2D shape.
static Tensor make_weight2d(int rows, int cols) {
  int shape[2] = {rows, cols};
  Tensor t = tensor_create(shape, 2);
  tensor_init_random(&t);
  return t;
}

// Create a bias/vector tensor of length n, initialized to zeros.
static Tensor make_zeros1d(int n) {
  int shape[1] = {n};
  return tensor_create(shape, 1);  // tensor_create already zero-fills
}

// Create a randomly initialized 1D tensor of length n.
static Tensor make_weight1d(int n) {
  int shape[1] = {n};
  Tensor t = tensor_create(shape, 1);
  tensor_init_random(&t);
  return t;
}

// Create a LayerNorm gamma initialized to ones.
static Tensor make_ones1d(int n) {
  Tensor t = make_zeros1d(n);
  for (int i = 0; i < n; i++) t.data[i] = 1.0f;
  return t;
}

// ---------------------------------------------------------------------------
// Model lifecycle
// ---------------------------------------------------------------------------

GPT gpt_create(int vocab_size, int d_model, int n_heads,
               int n_layers, int d_ff, int max_seq_len) {
  GPT m;
  m.vocab_size = vocab_size;
  m.d_model = d_model;
  m.n_heads = n_heads;
  m.n_layers = n_layers;
  m.d_ff = d_ff;
  m.max_seq_len = max_seq_len;

  m.wte = make_weight2d(vocab_size, d_model);
  m.wpe = make_weight2d(max_seq_len, d_model);

  m.blocks = (BlockWeights*)malloc(n_layers * sizeof(BlockWeights));
  for (int l = 0; l < n_layers; l++) {
    BlockWeights *b = &m.blocks[l];
    b->W_q = make_weight2d(d_model, d_model);
    b->W_k = make_weight2d(d_model, d_model);
    b->W_v = make_weight2d(d_model, d_model);
    b->W_o = make_weight2d(d_model, d_model);
    b->gamma1 = make_ones1d(d_model);
    b->beta1 = make_zeros1d(d_model);
    b->gamma2 = make_ones1d(d_model);
    b->beta2 = make_zeros1d(d_model);
    b->ff_W1 = make_weight2d(d_model, d_ff);
    b->ff_b1 = make_weight1d(d_ff);
    b->ff_W2 = make_weight2d(d_ff, d_model);
    b->ff_b2 = make_weight1d(d_model);
  }

  m.ln_f_gamma = make_ones1d(d_model);
  m.ln_f_beta = make_zeros1d(d_model);
  m.lm_head = make_weight2d(d_model, vocab_size);

  return m;
}

void gpt_free(GPT *m) {
  tensor_free(&m->wte);
  tensor_free(&m->wpe);
  for (int l = 0; l < m->n_layers; l++) {
    BlockWeights *b = &m->blocks[l];
    tensor_free(&b->W_q);
    tensor_free(&b->W_k);
    tensor_free(&b->W_v);
    tensor_free(&b->W_o);
    tensor_free(&b->gamma1);
    tensor_free(&b->beta1);
    tensor_free(&b->gamma2);
    tensor_free(&b->beta2);
    tensor_free(&b->ff_W1);
    tensor_free(&b->ff_b1);
    tensor_free(&b->ff_W2);
    tensor_free(&b->ff_b2);
  }
  free(m->blocks);
  m->blocks = NULL;
  tensor_free(&m->ln_f_gamma);
  tensor_free(&m->ln_f_beta);
  tensor_free(&m->lm_head);
}

// ---------------------------------------------------------------------------
// Forward pass and generation
// ---------------------------------------------------------------------------

void gpt_forward(const GPT *m, const int *token_ids, int seq_len, Tensor *logits) {
  int shape_h[2] = {seq_len, m->d_model};
  Tensor x = tensor_create(shape_h, 2);
  Tensor tmp = tensor_create(shape_h, 2);

  // Token + positional embeddings
  embed_tokens(token_ids, seq_len, &m->wte, &m->wpe, &x);

  // Stacked transformer blocks (result of each block feeds the next)
  for (int l = 0; l < m->n_layers; l++) {
    const BlockWeights *b = &m->blocks[l];
    transformer_block_forward(
        &x,
        &b->W_q, &b->W_k, &b->W_v, &b->W_o,
        &b->gamma1, &b->beta1,
        &b->gamma2, &b->beta2,
        &b->ff_W1, &b->ff_b1,
        &b->ff_W2, &b->ff_b2,
        &tmp,
        m->n_heads, m->d_ff);
    memcpy(x.data, tmp.data, x.size * sizeof(float));
  }

  // Final LayerNorm, then project to vocabulary logits
  layernorm(&x, &m->ln_f_gamma, &m->ln_f_beta, &tmp, 1e-5f);
  matmul(&tmp, &m->lm_head, logits);

  tensor_free(&x);
  tensor_free(&tmp);
}

void gpt_generate(const GPT *m, int *token_ids, int prompt_len, int n_new) {
  for (int step = 0; step < n_new; step++) {
    const int cur_len = prompt_len + step;

    int shape_logits[2] = {cur_len, m->vocab_size};
    Tensor logits = tensor_create(shape_logits, 2);

    gpt_forward(m, token_ids, cur_len, &logits);

    // Next token = greedy argmax over the last position's logits
    token_ids[cur_len] = sample_argmax(&logits, cur_len - 1);

    tensor_free(&logits);
  }
}
