// llama.c - A modern (LLaMA-family) transformer block (forward / inference)
#include "llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "matrix.h"
#include "rmsnorm.h"
#include "rope.h"
#include "gqa.h"
#include "swiglu.h"
#include "quant.h"

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
  b.b_q = vec(qd); b.b_k = vec(kvd); b.b_v = vec(kvd);
  b.Wo = mat(qd, d_model);
  b.ffn_norm = vec(d_model);
  b.Wgate = mat(d_model, d_ff);
  b.Wup = mat(d_model, d_ff);
  b.Wdown = mat(d_ff, d_model);
  return b;
}

void llama_block_free(LlamaBlock *b) {
  tensor_free(&b->attn_norm); tensor_free(&b->Wq); tensor_free(&b->Wk);
  tensor_free(&b->Wv); tensor_free(&b->b_q); tensor_free(&b->b_k); tensor_free(&b->b_v);
  tensor_free(&b->Wo); tensor_free(&b->ffn_norm);
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
  fill_rand(&b->b_q, s); fill_rand(&b->b_k, s); fill_rand(&b->b_v, s);   // exercise the bias path
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
  matmul(&xn, &b->Wq, &q); bias_add(&q, &b->b_q, &q);
  matmul(&xn, &b->Wk, &k); bias_add(&k, &b->b_k, &k);
  matmul(&xn, &b->Wv, &v); bias_add(&v, &b->b_v, &v);
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

#define LLMA_MAGIC   0x414D4C4Cu  // 'LLMA' little-endian bytes L,L,M,A (f32 weights)
#define LLMQ_MAGIC   0x514D4C4Cu  // 'LLMQ' little-endian bytes L,L,M,Q (int8 weights)
#define LLMA_VERSION 2u           // v2 adds q/k/v attention biases (zero for LLaMA/SmolLM)
#define TENSORS_PER_LAYER 12      // attn_norm, Wq,Wk,Wv, b_q,b_k,b_v, Wo, ffn_norm, Wgate,Wup,Wdown

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
    list[n++] = &b->b_q;
    list[n++] = &b->b_k;
    list[n++] = &b->b_v;
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

  const Tensor **list = (const Tensor**)malloc((size_t)(1 + TENSORS_PER_LAYER * m->n_layers + 2) * sizeof(const Tensor*));
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

// ---------------------------------------------------------------------------
// KV cache + incremental decode
// ---------------------------------------------------------------------------

LlamaKV llama_kv_create(const LlamaModel *m) {
  LlamaKV kv;
  kv.n_layers = m->n_layers;
  kv.max_seq = m->max_seq;
  kv.kv_dim = m->n_kv * m->head_dim;
  kv.len = 0;
  kv.k = (float**)malloc((size_t)m->n_layers * sizeof(float*));
  kv.v = (float**)malloc((size_t)m->n_layers * sizeof(float*));
  for (int l = 0; l < m->n_layers; l++) {
    kv.k[l] = (float*)malloc((size_t)m->max_seq * kv.kv_dim * sizeof(float));
    kv.v[l] = (float*)malloc((size_t)m->max_seq * kv.kv_dim * sizeof(float));
  }
  return kv;
}

void llama_kv_free(LlamaKV *kv) {
  for (int l = 0; l < kv->n_layers; l++) { free(kv->k[l]); free(kv->v[l]); }
  free(kv->k); free(kv->v);
  kv->k = kv->v = NULL;
}

void llama_decode_step(const LlamaModel *m, LlamaKV *kv, int token, Tensor *logits) {
  const int dm = m->d_model, hd = m->head_dim;
  const int qd = m->n_heads * hd, kvd = m->n_kv * hd;
  const int pos = kv->len;                 // this token's position
  const int group = m->n_heads / m->n_kv;
  const float scale = 1.0f / sqrtf((float)hd);

  Tensor x = mat(1, dm);                     // running residual stream (1 row)
  for (int j = 0; j < dm; j++) x.data[j] = m->wte.data[token * dm + j];

  Tensor xn = mat(1, dm), q = mat(1, qd), k = mat(1, kvd), v = mat(1, kvd);
  Tensor o = mat(1, qd), ao = mat(1, dm), hn = mat(1, dm), f = mat(1, dm);
  float *sc = (float*)malloc((size_t)(pos + 1) * sizeof(float));
  SwigluCache sc_ff = swiglu_cache_create(1, m->d_ff);

  for (int l = 0; l < m->n_layers; l++) {
    const LlamaBlock *b = &m->blocks[l];
    // Attention
    rmsnorm(&x, &b->attn_norm, &xn, m->eps);
    matmul(&xn, &b->Wq, &q); bias_add(&q, &b->b_q, &q);
    matmul(&xn, &b->Wk, &k); bias_add(&k, &b->b_k, &k);
    matmul(&xn, &b->Wv, &v); bias_add(&v, &b->b_v, &v);
    rope_heads_pos(&q, m->n_heads, m->rope_base, pos);
    rope_heads_pos(&k, m->n_kv, m->rope_base, pos);
    memcpy(&kv->k[l][pos * kvd], k.data, (size_t)kvd * sizeof(float));   // append to cache
    memcpy(&kv->v[l][pos * kvd], v.data, (size_t)kvd * sizeof(float));

    for (int head = 0; head < m->n_heads; head++) {
      const int kvh = head / group;
      const float *qh = &q.data[head * hd];
      float maxs = -INFINITY;
      for (int jj = 0; jj <= pos; jj++) {
        const float *kj = &kv->k[l][jj * kvd + kvh * hd];
        float dot = 0.0f;
        for (int d = 0; d < hd; d++) dot += qh[d] * kj[d];
        float s = dot * scale;
        sc[jj] = s;
        if (s > maxs) maxs = s;
      }
      float sum = 0.0f;
      for (int jj = 0; jj <= pos; jj++) { float e = expf(sc[jj] - maxs); sc[jj] = e; sum += e; }
      float inv = 1.0f / sum;
      float *oh = &o.data[head * hd];
      for (int d = 0; d < hd; d++) oh[d] = 0.0f;
      for (int jj = 0; jj <= pos; jj++) {
        float w = sc[jj] * inv;
        const float *vj = &kv->v[l][jj * kvd + kvh * hd];
        for (int d = 0; d < hd; d++) oh[d] += w * vj[d];
      }
    }
    matmul(&o, &b->Wo, &ao);
    for (int j = 0; j < dm; j++) x.data[j] += ao.data[j];       // residual

    // Feed-forward
    rmsnorm(&x, &b->ffn_norm, &hn, m->eps);
    swiglu_forward(&hn, &b->Wgate, &b->Wup, &b->Wdown, &sc_ff, &f);
    for (int j = 0; j < dm; j++) x.data[j] += f.data[j];        // residual
  }

  rmsnorm(&x, &m->final_norm, &hn, m->eps);
  matmul(&hn, &m->lm_head, logits);          // (1, vocab)
  kv->len = pos + 1;

  free(sc); swiglu_cache_free(&sc_ff);
  tensor_free(&x); tensor_free(&xn); tensor_free(&q); tensor_free(&k); tensor_free(&v);
  tensor_free(&o); tensor_free(&ao); tensor_free(&hn); tensor_free(&f);
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
  const int quant = (header[0] == LLMQ_MAGIC);
  if ((header[0] != LLMA_MAGIC && !quant) || header[1] != LLMA_VERSION) {
    fprintf(stderr, "Error: '%s' is not a v%u LLMA/LLMQ checkpoint\n", path, LLMA_VERSION);
    fclose(f); return 1;
  }

  *out = llama_model_create(cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[5],
                            cfg[6], cfg[7], fcfg[0], fcfg[1]);

  int ok = 1;
  const Tensor **list = (const Tensor**)malloc((size_t)(1 + TENSORS_PER_LAYER * out->n_layers + 2) * sizeof(const Tensor*));
  int count = llama_collect(out, list);
  for (int i = 0; i < count && ok; i++) {
    const Tensor *t = list[i];
    if (quant && t->ndim == 2) {
      // int8: per-column scales (cols float32) then rows*cols int8; dequantize in.
      int rows = t->shape[0], cols = t->shape[1];
      float *scale = (float*)malloc((size_t)cols * sizeof(float));
      signed char *q = (signed char*)malloc((size_t)rows * cols);
      ok &= (fread(scale, sizeof(float), (size_t)cols, f) == (size_t)cols);
      ok &= (fread(q, 1, (size_t)rows * cols, f) == (size_t)rows * cols);
      if (ok) dequantize_int8_percol(q, scale, rows, cols, t->data);
      free(scale); free(q);
    } else {
      size_t sz = (size_t)t->size;                 // f32 (LLMA, or 1D norms in LLMQ)
      ok &= (fread(t->data, sizeof(float), sz, f) == sz);
    }
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

// ---------------------------------------------------------------------------
// Quantized (int8, in-memory) model
// ---------------------------------------------------------------------------

static QLlamaBlock q_block_create(int d_model, int n_heads, int n_kv, int head_dim, int d_ff) {
  QLlamaBlock b;
  const int qd = n_heads * head_dim, kvd = n_kv * head_dim;
  b.attn_norm = vec(d_model); b.ffn_norm = vec(d_model);
  b.b_q = vec(qd); b.b_k = vec(kvd); b.b_v = vec(kvd);
  b.Wq = qtensor_create(d_model, qd);
  b.Wk = qtensor_create(d_model, kvd);
  b.Wv = qtensor_create(d_model, kvd);
  b.Wo = qtensor_create(qd, d_model);
  b.Wgate = qtensor_create(d_model, d_ff);
  b.Wup = qtensor_create(d_model, d_ff);
  b.Wdown = qtensor_create(d_ff, d_model);
  return b;
}

static void q_block_free(QLlamaBlock *b) {
  tensor_free(&b->attn_norm); tensor_free(&b->ffn_norm);
  tensor_free(&b->b_q); tensor_free(&b->b_k); tensor_free(&b->b_v);
  qtensor_free(&b->Wq); qtensor_free(&b->Wk); qtensor_free(&b->Wv); qtensor_free(&b->Wo);
  qtensor_free(&b->Wgate); qtensor_free(&b->Wup); qtensor_free(&b->Wdown);
}

static QLlamaModel q_model_alloc(int vocab, int d_model, int n_layers, int n_heads, int n_kv,
                                 int head_dim, int d_ff, int max_seq, float rope_base, float eps) {
  QLlamaModel m;
  m.vocab = vocab; m.d_model = d_model; m.n_layers = n_layers; m.n_heads = n_heads;
  m.n_kv = n_kv; m.head_dim = head_dim; m.d_ff = d_ff; m.max_seq = max_seq;
  m.rope_base = rope_base; m.eps = eps;
  m.wte = qtensor_create(vocab, d_model);
  m.blocks = (QLlamaBlock*)malloc((size_t)n_layers * sizeof(QLlamaBlock));
  for (int l = 0; l < n_layers; l++) m.blocks[l] = q_block_create(d_model, n_heads, n_kv, head_dim, d_ff);
  m.final_norm = vec(d_model);
  m.lm_head = qtensor_create(d_model, vocab);
  return m;
}

void q_llama_model_free(QLlamaModel *m) {
  qtensor_free(&m->wte);
  for (int l = 0; l < m->n_layers; l++) q_block_free(&m->blocks[l]);
  free(m->blocks);
  tensor_free(&m->final_norm); qtensor_free(&m->lm_head);
}

QLlamaModel q_llama_quantize_from(const LlamaModel *s) {
  QLlamaModel m = q_model_alloc(s->vocab, s->d_model, s->n_layers, s->n_heads, s->n_kv,
                                s->head_dim, s->d_ff, s->max_seq, s->rope_base, s->eps);
  const size_t dmb = (size_t)s->d_model * sizeof(float);
  qtensor_quantize(&m.wte, s->wte.data);
  for (int l = 0; l < s->n_layers; l++) {
    const LlamaBlock *a = &s->blocks[l];
    QLlamaBlock *b = &m.blocks[l];
    memcpy(b->attn_norm.data, a->attn_norm.data, dmb);
    memcpy(b->ffn_norm.data, a->ffn_norm.data, dmb);
    memcpy(b->b_q.data, a->b_q.data, (size_t)a->b_q.size * sizeof(float));
    memcpy(b->b_k.data, a->b_k.data, (size_t)a->b_k.size * sizeof(float));
    memcpy(b->b_v.data, a->b_v.data, (size_t)a->b_v.size * sizeof(float));
    qtensor_quantize(&b->Wq, a->Wq.data); qtensor_quantize(&b->Wk, a->Wk.data);
    qtensor_quantize(&b->Wv, a->Wv.data); qtensor_quantize(&b->Wo, a->Wo.data);
    qtensor_quantize(&b->Wgate, a->Wgate.data); qtensor_quantize(&b->Wup, a->Wup.data);
    qtensor_quantize(&b->Wdown, a->Wdown.data);
  }
  memcpy(m.final_norm.data, s->final_norm.data, dmb);
  qtensor_quantize(&m.lm_head, s->lm_head.data);
  return m;
}

LlamaKV q_llama_kv_create(const QLlamaModel *m) {
  LlamaKV kv;
  kv.n_layers = m->n_layers;
  kv.max_seq = m->max_seq;
  kv.kv_dim = m->n_kv * m->head_dim;
  kv.len = 0;
  kv.k = (float**)malloc((size_t)m->n_layers * sizeof(float*));
  kv.v = (float**)malloc((size_t)m->n_layers * sizeof(float*));
  for (int l = 0; l < m->n_layers; l++) {
    kv.k[l] = (float*)malloc((size_t)m->max_seq * kv.kv_dim * sizeof(float));
    kv.v[l] = (float*)malloc((size_t)m->max_seq * kv.kv_dim * sizeof(float));
  }
  return kv;
}

void q_llama_decode_step(const QLlamaModel *m, LlamaKV *kv, int token, Tensor *logits) {
  const int dm = m->d_model, hd = m->head_dim, dff = m->d_ff;
  const int qd = m->n_heads * hd, kvd = m->n_kv * hd;
  const int pos = kv->len, group = m->n_heads / m->n_kv;
  const float scale = 1.0f / sqrtf((float)hd);

  Tensor x = mat(1, dm);
  { const signed char *qr = &m->wte.q[(size_t)token * dm]; const float *sq = m->wte.scale;
    for (int j = 0; j < dm; j++) x.data[j] = (float)qr[j] * sq[j]; }   // dequant embedding row

  Tensor xn = mat(1, dm), q = mat(1, qd), k = mat(1, kvd), v = mat(1, kvd);
  Tensor o = mat(1, qd), ao = mat(1, dm), hn = mat(1, dm), f = mat(1, dm);
  Tensor g = mat(1, dff), u = mat(1, dff);
  float *sc = (float*)malloc((size_t)(pos + 1) * sizeof(float));

  for (int l = 0; l < m->n_layers; l++) {
    const QLlamaBlock *b = &m->blocks[l];
    rmsnorm(&x, &b->attn_norm, &xn, m->eps);
    qmatmul(&xn, &b->Wq, &q); bias_add(&q, &b->b_q, &q);
    qmatmul(&xn, &b->Wk, &k); bias_add(&k, &b->b_k, &k);
    qmatmul(&xn, &b->Wv, &v); bias_add(&v, &b->b_v, &v);
    rope_heads_pos(&q, m->n_heads, m->rope_base, pos);
    rope_heads_pos(&k, m->n_kv, m->rope_base, pos);
    memcpy(&kv->k[l][pos * kvd], k.data, (size_t)kvd * sizeof(float));
    memcpy(&kv->v[l][pos * kvd], v.data, (size_t)kvd * sizeof(float));

    for (int head = 0; head < m->n_heads; head++) {
      const int kvh = head / group;
      const float *qh = &q.data[head * hd];
      float maxs = -INFINITY;
      for (int jj = 0; jj <= pos; jj++) {
        const float *kj = &kv->k[l][jj * kvd + kvh * hd];
        float dot = 0.0f;
        for (int d = 0; d < hd; d++) dot += qh[d] * kj[d];
        float s = dot * scale;
        sc[jj] = s;
        if (s > maxs) maxs = s;
      }
      float sum = 0.0f;
      for (int jj = 0; jj <= pos; jj++) { float e = expf(sc[jj] - maxs); sc[jj] = e; sum += e; }
      float inv = 1.0f / sum;
      float *oh = &o.data[head * hd];
      for (int d = 0; d < hd; d++) oh[d] = 0.0f;
      for (int jj = 0; jj <= pos; jj++) {
        float w = sc[jj] * inv;
        const float *vj = &kv->v[l][jj * kvd + kvh * hd];
        for (int d = 0; d < hd; d++) oh[d] += w * vj[d];
      }
    }
    qmatmul(&o, &b->Wo, &ao);
    for (int j = 0; j < dm; j++) x.data[j] += ao.data[j];

    rmsnorm(&x, &b->ffn_norm, &hn, m->eps);
    qmatmul(&hn, &b->Wgate, &g);
    qmatmul(&hn, &b->Wup, &u);
    for (int j = 0; j < dff; j++) {           // h = SiLU(g) (*) u, in place in g
      float z = g.data[j];
      float s = 1.0f / (1.0f + expf(-z));
      g.data[j] = (z * s) * u.data[j];
    }
    qmatmul(&g, &b->Wdown, &f);
    for (int j = 0; j < dm; j++) x.data[j] += f.data[j];
  }

  rmsnorm(&x, &m->final_norm, &hn, m->eps);
  qmatmul(&hn, &m->lm_head, logits);
  kv->len = pos + 1;

  free(sc);
  tensor_free(&x); tensor_free(&xn); tensor_free(&q); tensor_free(&k); tensor_free(&v);
  tensor_free(&o); tensor_free(&ao); tensor_free(&hn); tensor_free(&f);
  tensor_free(&g); tensor_free(&u);
}

// Read one int8 tensor: cols scales (f32) then rows*cols int8.
static int q_read(FILE *f, QTensor *t) {
  int ok = 1;
  ok &= (fread(t->scale, sizeof(float), (size_t)t->cols, f) == (size_t)t->cols);
  ok &= (fread(t->q, 1, (size_t)t->rows * t->cols, f) == (size_t)t->rows * t->cols);
  return ok;
}

int q_llama_model_load(const char *path, QLlamaModel *out) {
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
  if (header[0] != LLMQ_MAGIC || header[1] != LLMA_VERSION) {
    fprintf(stderr, "Error: '%s' is not a v%u int8 LLMQ checkpoint\n", path, LLMA_VERSION);
    fclose(f); return 1;
  }

  *out = q_model_alloc(cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[5], cfg[6], cfg[7], fcfg[0], fcfg[1]);
  const size_t dmb = (size_t)out->d_model;

  int ok = q_read(f, &out->wte);
  for (int l = 0; l < out->n_layers && ok; l++) {
    QLlamaBlock *b = &out->blocks[l];
    ok &= (fread(b->attn_norm.data, sizeof(float), dmb, f) == dmb);
    ok &= q_read(f, &b->Wq); ok &= q_read(f, &b->Wk); ok &= q_read(f, &b->Wv);
    ok &= (fread(b->b_q.data, sizeof(float), (size_t)b->b_q.size, f) == (size_t)b->b_q.size);
    ok &= (fread(b->b_k.data, sizeof(float), (size_t)b->b_k.size, f) == (size_t)b->b_k.size);
    ok &= (fread(b->b_v.data, sizeof(float), (size_t)b->b_v.size, f) == (size_t)b->b_v.size);
    ok &= q_read(f, &b->Wo);
    ok &= (fread(b->ffn_norm.data, sizeof(float), dmb, f) == dmb);
    ok &= q_read(f, &b->Wgate); ok &= q_read(f, &b->Wup); ok &= q_read(f, &b->Wdown);
  }
  ok &= (fread(out->final_norm.data, sizeof(float), dmb, f) == dmb);
  ok &= q_read(f, &out->lm_head);
  fclose(f);

  if (!ok) {
    fprintf(stderr, "Error: '%s' is truncated (bad weights)\n", path);
    q_llama_model_free(out);
    return 1;
  }
  return 0;
}
