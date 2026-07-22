// llama.c - A modern (LLaMA-family) transformer block (forward / inference)
#include "llama.h"

#include <stdlib.h>

#include "matrix.h"
#include "rmsnorm.h"
#include "rope.h"
#include "gqa.h"
#include "swiglu.h"

static Tensor vec(int n) { int s[1] = {n}; return tensor_create(s, 1); }
static Tensor mat(int r, int c) { int s[2] = {r, c}; return tensor_create(s, 2); }

LlamaBlock llama_block_create(int d_model, int n_heads, int n_kv, int head_dim,
                              int d_ff, float rope_base, float eps) {
  LlamaBlock b;
  b.d_model = d_model; b.n_heads = n_heads; b.n_kv = n_kv; b.head_dim = head_dim;
  b.d_ff = d_ff; b.rope_base = rope_base; b.eps = eps;
  const int qd = n_heads * head_dim, kvd = n_kv * head_dim;
  b.attn_norm = vec(d_model);
  b.Wq = mat(d_model, qd);
  b.Wk = mat(d_model, kvd);
  b.Wv = mat(d_model, kvd);
  b.Wo = mat(qd, d_model);
  b.ffn_norm = vec(d_model);
  b.Wgate = mat(d_model, d_ff);
  b.Wup = mat(d_model, d_ff);
  b.Wdown = mat(d_ff, d_model);
  return b;
}

void llama_block_free(LlamaBlock *b) {
  tensor_free(&b->attn_norm); tensor_free(&b->Wq); tensor_free(&b->Wk);
  tensor_free(&b->Wv); tensor_free(&b->Wo); tensor_free(&b->ffn_norm);
  tensor_free(&b->Wgate); tensor_free(&b->Wup); tensor_free(&b->Wdown);
}

static void fill_rand(Tensor *t, float scale) {
  int n = 1;
  for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
  for (int i = 0; i < n; i++)
    t->data[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
}

void llama_block_randomize(LlamaBlock *b) {
  // Norm gains ~1 (RMSNorm gamma), projections small.
  int dn = b->attn_norm.shape[0];
  for (int i = 0; i < dn; i++) { b->attn_norm.data[i] = 1.0f; b->ffn_norm.data[i] = 1.0f; }
  float s = 0.15f;
  fill_rand(&b->Wq, s); fill_rand(&b->Wk, s); fill_rand(&b->Wv, s); fill_rand(&b->Wo, s);
  fill_rand(&b->Wgate, s); fill_rand(&b->Wup, s); fill_rand(&b->Wdown, s);
}

void llama_block_forward(const LlamaBlock *b, const Tensor *x, Tensor *out) {
  const int n = x->shape[0];
  const int dm = b->d_model;
  const int qd = b->n_heads * b->head_dim, kvd = b->n_kv * b->head_dim;

  Tensor xn = mat(n, dm);
  Tensor q = mat(n, qd), k = mat(n, kvd), v = mat(n, kvd);
  Tensor o = mat(n, qd), ao = mat(n, dm), h = mat(n, dm);
  Tensor hn = mat(n, dm), f = mat(n, dm);
  GqaCache gc = gqa_cache_create(n, b->n_heads, b->n_kv, b->head_dim);
  SwigluCache sc = swiglu_cache_create(n, b->d_ff);

  // --- Attention sub-layer ---
  rmsnorm(x, &b->attn_norm, &xn, b->eps);
  matmul(&xn, &b->Wq, &q);
  matmul(&xn, &b->Wk, &k);
  matmul(&xn, &b->Wv, &v);
  rope_heads(&q, b->n_heads, b->rope_base);
  rope_heads(&k, b->n_kv, b->rope_base);
  gqa_forward(&q, &k, &v, &gc, &o);
  matmul(&o, &b->Wo, &ao);
  matadd(x, &ao, &h);                       // residual

  // --- Feed-forward sub-layer ---
  rmsnorm(&h, &b->ffn_norm, &hn, b->eps);
  swiglu_forward(&hn, &b->Wgate, &b->Wup, &b->Wdown, &sc, &f);
  matadd(&h, &f, out);                       // residual

  swiglu_cache_free(&sc); gqa_cache_free(&gc);
  tensor_free(&xn); tensor_free(&q); tensor_free(&k); tensor_free(&v);
  tensor_free(&o); tensor_free(&ao); tensor_free(&h); tensor_free(&hn); tensor_free(&f);
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

LlamaModel llama_model_create(int vocab, int d_model, int n_layers, int n_heads,
                              int n_kv, int head_dim, int d_ff, int max_seq,
                              float rope_base, float eps) {
  LlamaModel m;
  m.vocab = vocab; m.d_model = d_model; m.n_layers = n_layers;
  m.n_heads = n_heads; m.n_kv = n_kv; m.head_dim = head_dim; m.d_ff = d_ff;
  m.max_seq = max_seq; m.rope_base = rope_base; m.eps = eps;
  m.wte = mat(vocab, d_model);
  m.blocks = (LlamaBlock*)malloc((size_t)n_layers * sizeof(LlamaBlock));
  for (int l = 0; l < n_layers; l++)
    m.blocks[l] = llama_block_create(d_model, n_heads, n_kv, head_dim, d_ff, rope_base, eps);
  m.final_norm = vec(d_model);
  m.lm_head = mat(d_model, vocab);
  return m;
}

void llama_model_free(LlamaModel *m) {
  tensor_free(&m->wte);
  for (int l = 0; l < m->n_layers; l++) llama_block_free(&m->blocks[l]);
  free(m->blocks);
  tensor_free(&m->final_norm); tensor_free(&m->lm_head);
}

void llama_model_randomize(LlamaModel *m) {
  fill_rand(&m->wte, 0.05f);
  for (int l = 0; l < m->n_layers; l++) llama_block_randomize(&m->blocks[l]);
  for (int j = 0; j < m->d_model; j++) m->final_norm.data[j] = 1.0f;
  fill_rand(&m->lm_head, 0.05f);
}

void llama_model_forward(const LlamaModel *m, const int *ids, int seq_len, Tensor *logits) {
  const int dm = m->d_model;
  Tensor a = mat(seq_len, dm), b = mat(seq_len, dm), hn = mat(seq_len, dm);

  // Token embedding (no positional term — RoPE carries position).
  for (int i = 0; i < seq_len; i++)
    for (int j = 0; j < dm; j++)
      a.data[i * dm + j] = m->wte.data[ids[i] * dm + j];

  Tensor *cur = &a, *nxt = &b;
  for (int l = 0; l < m->n_layers; l++) {
    llama_block_forward(&m->blocks[l], cur, nxt);
    Tensor *t = cur; cur = nxt; nxt = t;      // ping-pong buffers
  }

  rmsnorm(cur, &m->final_norm, &hn, m->eps);
  matmul(&hn, &m->lm_head, logits);            // (seq_len, vocab)

  tensor_free(&a); tensor_free(&b); tensor_free(&hn);
}
