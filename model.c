// model.c - GPT model: embeddings, stacked transformer blocks, output head
#include "model.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "embedding.h"
#include "attention.h"
#include "layernorm.h"
#include "matrix.h"
#include "feedforward.h"
#include "sampler.h"

#define LN_EPS 1e-5f

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
    b->b_q = make_zeros1d(d_model);
    b->b_k = make_zeros1d(d_model);
    b->b_v = make_zeros1d(d_model);
    b->b_o = make_zeros1d(d_model);
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
    tensor_free(&b->b_q);
    tensor_free(&b->b_k);
    tensor_free(&b->b_v);
    tensor_free(&b->b_o);
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
        &b->b_q, &b->b_k, &b->b_v, &b->b_o,
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

// ---------------------------------------------------------------------------
// Training: activation cache
// ---------------------------------------------------------------------------

static Tensor *make_layer_tensors(int n_layers, int rows, int cols) {
  Tensor *arr = (Tensor*)malloc(n_layers * sizeof(Tensor));
  int shape[2] = {rows, cols};
  for (int l = 0; l < n_layers; l++) arr[l] = tensor_create(shape, 2);
  return arr;
}

GPTActs gpt_acts_create(const GPT *m, int seq_len) {
  GPTActs a;
  a.seq_len = seq_len;
  a.n_layers = m->n_layers;
  const int dm = m->d_model, df = m->d_ff, nh = m->n_heads, L = m->n_layers;

  int s2[2] = {seq_len, dm};
  a.emb = tensor_create(s2, 2);
  a.ln_f_out = tensor_create(s2, 2);

  a.ln1 = make_layer_tensors(L, seq_len, dm);
  a.Q = make_layer_tensors(L, seq_len, dm);
  a.K = make_layer_tensors(L, seq_len, dm);
  a.V = make_layer_tensors(L, seq_len, dm);
  a.weights = make_layer_tensors(L, nh * seq_len, seq_len);
  a.concat = make_layer_tensors(L, seq_len, dm);
  a.attn_out = make_layer_tensors(L, seq_len, dm);
  a.a_resid = make_layer_tensors(L, seq_len, dm);
  a.ln2 = make_layer_tensors(L, seq_len, dm);
  a.ff_h1b = make_layer_tensors(L, seq_len, df);
  a.ff_act = make_layer_tensors(L, seq_len, df);
  a.ff_out = make_layer_tensors(L, seq_len, dm);
  a.block_out = make_layer_tensors(L, seq_len, dm);
  return a;
}

static void free_layer_tensors(Tensor *arr, int n_layers) {
  for (int l = 0; l < n_layers; l++) tensor_free(&arr[l]);
  free(arr);
}

void gpt_acts_free(GPTActs *a) {
  tensor_free(&a->emb);
  tensor_free(&a->ln_f_out);
  free_layer_tensors(a->ln1, a->n_layers);
  free_layer_tensors(a->Q, a->n_layers);
  free_layer_tensors(a->K, a->n_layers);
  free_layer_tensors(a->V, a->n_layers);
  free_layer_tensors(a->weights, a->n_layers);
  free_layer_tensors(a->concat, a->n_layers);
  free_layer_tensors(a->attn_out, a->n_layers);
  free_layer_tensors(a->a_resid, a->n_layers);
  free_layer_tensors(a->ln2, a->n_layers);
  free_layer_tensors(a->ff_h1b, a->n_layers);
  free_layer_tensors(a->ff_act, a->n_layers);
  free_layer_tensors(a->ff_out, a->n_layers);
  free_layer_tensors(a->block_out, a->n_layers);
}

static void gpt_acts_zero_grad(GPTActs *a) {
  tensor_zero_grad(&a->emb);
  tensor_zero_grad(&a->ln_f_out);
  for (int l = 0; l < a->n_layers; l++) {
    tensor_zero_grad(&a->ln1[l]);
    tensor_zero_grad(&a->Q[l]);
    tensor_zero_grad(&a->K[l]);
    tensor_zero_grad(&a->V[l]);
    tensor_zero_grad(&a->weights[l]);
    tensor_zero_grad(&a->concat[l]);
    tensor_zero_grad(&a->attn_out[l]);
    tensor_zero_grad(&a->a_resid[l]);
    tensor_zero_grad(&a->ln2[l]);
    tensor_zero_grad(&a->ff_h1b[l]);
    tensor_zero_grad(&a->ff_act[l]);
    tensor_zero_grad(&a->ff_out[l]);
    tensor_zero_grad(&a->block_out[l]);
  }
}

// ---------------------------------------------------------------------------
// Training: forward, backward, parameter helpers
// ---------------------------------------------------------------------------

// Run the stacked blocks -> final LayerNorm -> output head, starting from the
// embeddings already in a->emb. The sequence length is a->emb.shape[0], so this
// works for both plain text and multimodal (image+text) inputs.
static void gpt_run_from_emb(GPT *m, GPTActs *a, Tensor *logits) {
  for (int l = 0; l < m->n_layers; l++) {
    BlockWeights *b = &m->blocks[l];
    const Tensor *inp = (l == 0) ? &a->emb : &a->block_out[l - 1];

    layernorm(inp, &b->gamma1, &b->beta1, &a->ln1[l], LN_EPS);
    multihead_attention_train(&a->ln1[l], &b->W_q, &b->W_k, &b->W_v, &b->W_o,
                              &b->b_q, &b->b_k, &b->b_v, &b->b_o,
                              &a->Q[l], &a->K[l], &a->V[l], &a->weights[l], &a->concat[l],
                              &a->attn_out[l], m->n_heads);
    matadd(inp, &a->attn_out[l], &a->a_resid[l]);
    layernorm(&a->a_resid[l], &b->gamma2, &b->beta2, &a->ln2[l], LN_EPS);
    feedforward_train(&a->ln2[l], &b->ff_W1, &b->ff_b1, &b->ff_W2, &b->ff_b2,
                      &a->ff_h1b[l], &a->ff_act[l], &a->ff_out[l], m->d_ff);
    matadd(&a->a_resid[l], &a->ff_out[l], &a->block_out[l]);
  }

  layernorm(&a->block_out[m->n_layers - 1], &m->ln_f_gamma, &m->ln_f_beta,
            &a->ln_f_out, LN_EPS);
  matmul(&a->ln_f_out, &m->lm_head, logits);
}

void gpt_forward_train(GPT *m, const int *token_ids, int seq_len,
                       GPTActs *a, Tensor *logits) {
  embed_tokens(token_ids, seq_len, &m->wte, &m->wpe, &a->emb);
  gpt_run_from_emb(m, a, logits);
}

// Multimodal forward: a sequence of `n_img` precomputed image-patch embeddings
// (img_emb: (n_img, d_model)) followed by `seq_len` text tokens. Positional
// embeddings are added over the whole combined sequence; the transformer then
// attends over [image patches] + [text tokens]. a must be sized for the total
// length (n_img + seq_len), which must not exceed max_seq_len. logits is
// (n_img + seq_len, vocab_size).
void gpt_forward_multimodal(GPT *m, const Tensor *img_emb, const int *token_ids,
                            int n_img, int seq_len, GPTActs *a, Tensor *logits) {
  const int dm = m->d_model;
  // Image positions 0..n_img-1: patch embedding + positional embedding.
  for (int i = 0; i < n_img; i++) {
    for (int j = 0; j < dm; j++) {
      a->emb.data[i * dm + j] = img_emb->data[i * dm + j] + m->wpe.data[i * dm + j];
    }
  }
  // Text positions n_img..n_img+seq_len-1: token embedding + positional embedding.
  for (int t = 0; t < seq_len; t++) {
    int tok = token_ids[t];
    int pos = n_img + t;
    for (int j = 0; j < dm; j++) {
      a->emb.data[pos * dm + j] = m->wte.data[tok * dm + j] + m->wpe.data[pos * dm + j];
    }
  }
  gpt_run_from_emb(m, a, logits);
}

// Backward through the output head + blocks down to a->emb.grad (the gradient of
// the input embeddings). Shared by the text and multimodal backward paths.
static void gpt_backward_to_emb(GPT *m, GPTActs *a, Tensor *logits) {
  gpt_zero_grad(m);
  gpt_acts_zero_grad(a);

  // logits = ln_f_out @ lm_head
  matmul_backward(&a->ln_f_out, &m->lm_head, logits);
  // ln_f_out = LN_f(block_out[L-1])
  layernorm_backward(&a->block_out[m->n_layers - 1], &m->ln_f_gamma, &a->ln_f_out,
                     &m->ln_f_gamma, &m->ln_f_beta, LN_EPS);

  for (int l = m->n_layers - 1; l >= 0; l--) {
    BlockWeights *b = &m->blocks[l];
    const Tensor *inp = (l == 0) ? &a->emb : &a->block_out[l - 1];

    matadd_backward(&a->block_out[l], &a->a_resid[l], &a->ff_out[l]);
    feedforward_backward(&a->ln2[l], &b->ff_W1, &b->ff_b1, &b->ff_W2, &b->ff_b2,
                         &a->ff_h1b[l], &a->ff_act[l], &a->ff_out[l], m->d_ff);
    layernorm_backward(&a->a_resid[l], &b->gamma2, &a->ln2[l],
                       &b->gamma2, &b->beta2, LN_EPS);
    matadd_backward(&a->a_resid[l], inp, &a->attn_out[l]);
    multihead_attention_backward(&a->ln1[l], &b->W_q, &b->W_k, &b->W_v, &b->W_o,
                                 &b->b_q, &b->b_k, &b->b_v, &b->b_o,
                                 &a->Q[l], &a->K[l], &a->V[l], &a->weights[l], &a->concat[l],
                                 &a->attn_out[l], m->n_heads);
    layernorm_backward(inp, &b->gamma1, &a->ln1[l], &b->gamma1, &b->beta1, LN_EPS);
  }
}

void gpt_backward(GPT *m, const int *token_ids, int seq_len,
                  GPTActs *a, Tensor *logits) {
  gpt_backward_to_emb(m, a, logits);
  embedding_backward(token_ids, seq_len, &a->emb, &m->wte, &m->wpe);
}

void gpt_backward_multimodal(GPT *m, const int *token_ids, int n_img, int seq_len,
                             GPTActs *a, Tensor *logits, Tensor *img_emb) {
  gpt_backward_to_emb(m, a, logits);
  const int dm = m->d_model;
  // Image positions: dL/d(patch embedding) = a->emb.grad rows; also update wpe.
  for (int i = 0; i < n_img; i++) {
    for (int j = 0; j < dm; j++) {
      float g = a->emb.grad[i * dm + j];
      img_emb->grad[i * dm + j] = g;
      m->wpe.grad[i * dm + j] += g;
    }
  }
  // Text positions: scatter into wte (by token) and wpe (by position).
  for (int t = 0; t < seq_len; t++) {
    int tok = token_ids[t];
    int pos = n_img + t;
    for (int j = 0; j < dm; j++) {
      float g = a->emb.grad[pos * dm + j];
      m->wte.grad[tok * dm + j] += g;
      m->wpe.grad[pos * dm + j] += g;
    }
  }
}

int gpt_collect_params(GPT *m, Tensor **list) {
  int n = 0;
  list[n++] = &m->wte;
  list[n++] = &m->wpe;
  for (int l = 0; l < m->n_layers; l++) {
    BlockWeights *b = &m->blocks[l];
    list[n++] = &b->W_q;
    list[n++] = &b->W_k;
    list[n++] = &b->W_v;
    list[n++] = &b->W_o;
    list[n++] = &b->b_q;
    list[n++] = &b->b_k;
    list[n++] = &b->b_v;
    list[n++] = &b->b_o;
    list[n++] = &b->gamma1;
    list[n++] = &b->beta1;
    list[n++] = &b->gamma2;
    list[n++] = &b->beta2;
    list[n++] = &b->ff_W1;
    list[n++] = &b->ff_b1;
    list[n++] = &b->ff_W2;
    list[n++] = &b->ff_b2;
  }
  list[n++] = &m->ln_f_gamma;
  list[n++] = &m->ln_f_beta;
  list[n++] = &m->lm_head;
  return n;
}

void gpt_zero_grad(GPT *m) {
  int cap = 2 + 16 * m->n_layers + 3;
  Tensor **list = (Tensor**)malloc(cap * sizeof(Tensor*));
  int n = gpt_collect_params(m, list);
  for (int i = 0; i < n; i++) tensor_zero_grad(list[i]);
  free(list);
}

// ---------------------------------------------------------------------------
// KV cache + incremental decode
// ---------------------------------------------------------------------------

KVCache gpt_kvcache_create(const GPT *m) {
  KVCache kv;
  kv.n_layers = m->n_layers;
  kv.d_model = m->d_model;
  kv.n_heads = m->n_heads;
  kv.max_seq_len = m->max_seq_len;
  kv.len = 0;
  kv.K = make_layer_tensors(m->n_layers, m->max_seq_len, m->d_model);
  kv.V = make_layer_tensors(m->n_layers, m->max_seq_len, m->d_model);
  return kv;
}

void gpt_kvcache_free(KVCache *kv) {
  free_layer_tensors(kv->K, kv->n_layers);
  free_layer_tensors(kv->V, kv->n_layers);
  kv->K = NULL;
  kv->V = NULL;
  kv->len = 0;
}

void gpt_decode_step(const GPT *m, KVCache *kv, int token, Tensor *logits) {
  const int dm = m->d_model, nh = m->n_heads, hd = dm / nh;
  const int pos = kv->len;
  const float scale = 1.0f / sqrtf((float)hd);

  int s1[2] = {1, dm};
  Tensor x = tensor_create(s1, 2), ln1 = tensor_create(s1, 2);
  Tensor q = tensor_create(s1, 2), k = tensor_create(s1, 2), v = tensor_create(s1, 2);
  Tensor concat = tensor_create(s1, 2), attn = tensor_create(s1, 2);
  Tensor ln2 = tensor_create(s1, 2), ff = tensor_create(s1, 2), lnf = tensor_create(s1, 2);

  // Embed the new token at its position
  for (int j = 0; j < dm; j++) {
    x.data[j] = m->wte.data[token * dm + j] + m->wpe.data[pos * dm + j];
  }

  float *scores = (float*)malloc((pos + 1) * sizeof(float));
  for (int l = 0; l < m->n_layers; l++) {
    const BlockWeights *b = &m->blocks[l];

    layernorm(&x, &b->gamma1, &b->beta1, &ln1, LN_EPS);
    matmul(&ln1, &b->W_q, &q); bias_add(&q, &b->b_q, &q);
    matmul(&ln1, &b->W_k, &k); bias_add(&k, &b->b_k, &k);
    matmul(&ln1, &b->W_v, &v); bias_add(&v, &b->b_v, &v);

    // Append this position's K/V to the cache
    memcpy(&kv->K[l].data[pos * dm], k.data, dm * sizeof(float));
    memcpy(&kv->V[l].data[pos * dm], v.data, dm * sizeof(float));

    // Single-query attention over cached positions 0..pos, per head
    for (int h = 0; h < nh; h++) {
      const int off = h * hd;
      float mx = -INFINITY;
      for (int j = 0; j <= pos; j++) {
        float dot = 0.0f;
        for (int d = 0; d < hd; d++) {
          dot += q.data[off + d] * kv->K[l].data[j * dm + off + d];
        }
        scores[j] = dot * scale;
        if (scores[j] > mx) mx = scores[j];
      }
      float sum = 0.0f;
      for (int j = 0; j <= pos; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
      for (int d = 0; d < hd; d++) {
        float acc = 0.0f;
        for (int j = 0; j <= pos; j++) {
          acc += (scores[j] / sum) * kv->V[l].data[j * dm + off + d];
        }
        concat.data[off + d] = acc;
      }
    }

    matmul(&concat, &b->W_o, &attn); bias_add(&attn, &b->b_o, &attn);
    matadd(&x, &attn, &x);  // residual (in place)
    layernorm(&x, &b->gamma2, &b->beta2, &ln2, LN_EPS);
    feedforward(&ln2, &b->ff_W1, &b->ff_b1, &b->ff_W2, &b->ff_b2, &ff, m->d_ff);
    matadd(&x, &ff, &x);    // residual (in place)
  }
  free(scores);

  layernorm(&x, &m->ln_f_gamma, &m->ln_f_beta, &lnf, LN_EPS);
  matmul(&lnf, &m->lm_head, logits);
  kv->len++;

  tensor_free(&x); tensor_free(&ln1); tensor_free(&q); tensor_free(&k); tensor_free(&v);
  tensor_free(&concat); tensor_free(&attn); tensor_free(&ln2); tensor_free(&ff); tensor_free(&lnf);
}
