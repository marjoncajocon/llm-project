// llama.c - A modern (LLaMA-family) transformer block (forward / inference)
#include "llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

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

// ---------------------------------------------------------------------------
// Checkpoint I/O ('LLMA' flat binary)
// ---------------------------------------------------------------------------

#define LLMA_MAGIC   0x414D4C4Cu  // 'LLMA' little-endian bytes L,L,M,A
#define LLMA_VERSION 1u

// Every weight tensor in a fixed canonical order shared by save and load, so the
// two can never drift. 1 (wte) + 9 per layer + 2 (final_norm, lm_head).
static int llama_collect(const LlamaModel *m, const Tensor **list) {
  int n = 0;
  list[n++] = &m->wte;
  for (int l = 0; l < m->n_layers; l++) {
    const LlamaBlock *b = &m->blocks[l];
    list[n++] = &b->attn_norm;
    list[n++] = &b->Wq;
    list[n++] = &b->Wk;
    list[n++] = &b->Wv;
    list[n++] = &b->Wo;
    list[n++] = &b->ffn_norm;
    list[n++] = &b->Wgate;
    list[n++] = &b->Wup;
    list[n++] = &b->Wdown;
  }
  list[n++] = &m->final_norm;
  list[n++] = &m->lm_head;
  return n;
}

int llama_model_save(const char *path, const LlamaModel *m) {
  FILE *f = fopen(path, "wb");
  if (!f) { fprintf(stderr, "Error: cannot open '%s' for writing\n", path); return 1; }

  uint32_t header[2] = {LLMA_MAGIC, LLMA_VERSION};
  int32_t cfg[8] = {m->vocab, m->d_model, m->n_layers, m->n_heads,
                    m->n_kv, m->head_dim, m->d_ff, m->max_seq};
  float fcfg[2] = {m->rope_base, m->eps};

  int ok = 1;
  ok &= (fwrite(header, sizeof(uint32_t), 2, f) == 2);
  ok &= (fwrite(cfg, sizeof(int32_t), 8, f) == 8);
  ok &= (fwrite(fcfg, sizeof(float), 2, f) == 2);

  const Tensor **list = (const Tensor**)malloc((size_t)(1 + 9 * m->n_layers + 2) * sizeof(const Tensor*));
  int count = llama_collect(m, list);
  for (int i = 0; i < count && ok; i++) {
    size_t sz = (size_t)list[i]->size;
    ok &= (fwrite(list[i]->data, sizeof(float), sz, f) == sz);
  }
  free(list);
  fclose(f);
  if (!ok) { fprintf(stderr, "Error: failed while writing '%s'\n", path); return 1; }
  return 0;
}

int llama_model_load(const char *path, LlamaModel *out) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "Error: cannot open '%s' for reading\n", path); return 1; }

  uint32_t header[2];
  int32_t cfg[8];
  float fcfg[2];
  if (fread(header, sizeof(uint32_t), 2, f) != 2 ||
      fread(cfg, sizeof(int32_t), 8, f) != 8 ||
      fread(fcfg, sizeof(float), 2, f) != 2) {
    fprintf(stderr, "Error: '%s' is truncated (bad header)\n", path);
    fclose(f); return 1;
  }
  if (header[0] != LLMA_MAGIC || header[1] != LLMA_VERSION) {
    fprintf(stderr, "Error: '%s' is not a v%u LLMA checkpoint\n", path, LLMA_VERSION);
    fclose(f); return 1;
  }

  *out = llama_model_create(cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[5],
                            cfg[6], cfg[7], fcfg[0], fcfg[1]);

  int ok = 1;
  const Tensor **list = (const Tensor**)malloc((size_t)(1 + 9 * out->n_layers + 2) * sizeof(const Tensor*));
  int count = llama_collect(out, list);
  for (int i = 0; i < count && ok; i++) {
    size_t sz = (size_t)list[i]->size;
    ok &= (fread(list[i]->data, sizeof(float), sz, f) == sz);
  }
  free(list);
  fclose(f);
  if (!ok) {
    fprintf(stderr, "Error: '%s' is truncated (bad weights)\n", path);
    llama_model_free(out);
    return 1;
  }
  return 0;
}
