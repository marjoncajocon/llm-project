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
