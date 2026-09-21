// llama.c - A modern (LLaMA-family) transformer block (forward / inference)
#include "llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <limits.h>

#include "matrix.h"
#include "rmsnorm.h"
#include "rope.h"
#include "gqa.h"
#include "swiglu.h"
#include "quant.h"

// Inference is the common case for the LLaMA-family path, so by default none of
// its tensors — weights, KV cache, or decode scratch — carry a gradient array;
// that halves the memory of a loaded fp32 model. The *_create_train variants
// allocate the same weights WITH grads so the model can be trained (llamatrain.c).
static Tensor vec_of(int n, int grad) {
  int s[1] = {n};
  return grad ? tensor_create(s, 1) : tensor_create_nograd(s, 1);
}
static Tensor mat_of(int r, int c, int grad) {
  int s[2] = {r, c};
  return grad ? tensor_create(s, 2) : tensor_create_nograd(s, 2);
}
static Tensor vec(int n) { return vec_of(n, 0); }
static Tensor mat(int r, int c) { return mat_of(r, c, 0); }

static LlamaBlock block_create_ex(int d_model, int n_heads, int n_kv, int head_dim,
                                  int d_ff, float rope_base, float eps, int grad) {
  LlamaBlock b;
  b.d_model = d_model; b.n_heads = n_heads; b.n_kv = n_kv; b.head_dim = head_dim;
  b.d_ff = d_ff; b.rope_base = rope_base; b.eps = eps;
  b.qk_norm = 0;                           // off until llama_model_set_qk_norm
  b.window = 0;                               // full causal attention by default
  const int qd = n_heads * head_dim, kvd = n_kv * head_dim;
  b.attn_norm = vec_of(d_model, grad);
  b.Wq = mat_of(d_model, qd, grad);
  b.Wk = mat_of(d_model, kvd, grad);
  b.Wv = mat_of(d_model, kvd, grad);
  b.b_q = vec_of(qd, grad); b.b_k = vec_of(kvd, grad); b.b_v = vec_of(kvd, grad);
  b.q_norm = vec_of(head_dim, grad); b.k_norm = vec_of(head_dim, grad);
  b.Wo = mat_of(qd, d_model, grad);
  b.ffn_norm = vec_of(d_model, grad);
  b.Wgate = mat_of(d_model, d_ff, grad);
  b.Wup = mat_of(d_model, d_ff, grad);
  b.Wdown = mat_of(d_ff, d_model, grad);
  return b;
}

LlamaBlock llama_block_create(int d_model, int n_heads, int n_kv, int head_dim,
                              int d_ff, float rope_base, float eps) {
  return block_create_ex(d_model, n_heads, n_kv, head_dim, d_ff, rope_base, eps, 0);
}

LlamaBlock llama_block_create_train(int d_model, int n_heads, int n_kv, int head_dim,
                                    int d_ff, float rope_base, float eps) {
  return block_create_ex(d_model, n_heads, n_kv, head_dim, d_ff, rope_base, eps, 1);
}

void llama_block_free(LlamaBlock *b) {
  tensor_free(&b->attn_norm); tensor_free(&b->Wq); tensor_free(&b->Wk);
  tensor_free(&b->Wv); tensor_free(&b->b_q); tensor_free(&b->b_k); tensor_free(&b->b_v);
  tensor_free(&b->q_norm); tensor_free(&b->k_norm);
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
  // QK-norm gains ~1 but not exactly (so a test sees them). Only drawn when the
  // feature is on, so the rand() stream — and every QK-norm-free test — is unchanged.
  if (b->qk_norm)
    for (int i = 0; i < b->head_dim; i++) {
      b->q_norm.data[i] = 1.0f + ((float)rand() / (float)RAND_MAX - 0.5f) * 0.5f;
      b->k_norm.data[i] = 1.0f + ((float)rand() / (float)RAND_MAX - 0.5f) * 0.5f;
    }
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
  gc.window = b->window;                     // sliding window (0 = full causal)
  SwigluCache sc = swiglu_cache_create(n, b->d_ff);

  // --- Attention sub-layer ---
  rmsnorm(x, &b->attn_norm, &xn, b->eps);
  matmul(&xn, &b->Wq, &q); bias_add(&q, &b->b_q, &q);
  matmul(&xn, &b->Wk, &k); bias_add(&k, &b->b_k, &k);
  matmul(&xn, &b->Wv, &v); bias_add(&v, &b->b_v, &v);
  if (b->qk_norm) {                         // Qwen3: per-head RMSNorm, before RoPE
    rmsnorm_heads(&q, &b->q_norm, &q, b->eps);
    rmsnorm_heads(&k, &b->k_norm, &k, b->eps);
  }
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

static LlamaModel model_create_ex(int vocab, int d_model, int n_layers, int n_heads,
                                  int n_kv, int head_dim, int d_ff, int max_seq,
                                  float rope_base, float eps, int grad) {
  LlamaModel m;
  m.vocab = vocab; m.d_model = d_model; m.n_layers = n_layers;
  m.n_heads = n_heads; m.n_kv = n_kv; m.head_dim = head_dim; m.d_ff = d_ff;
  m.max_seq = max_seq; m.rope_base = rope_base; m.eps = eps;
  m.qk_norm = 0;
  m.wte = mat_of(vocab, d_model, grad);
  m.blocks = (LlamaBlock*)malloc((size_t)n_layers * sizeof(LlamaBlock));
  for (int l = 0; l < n_layers; l++)
    m.blocks[l] = block_create_ex(d_model, n_heads, n_kv, head_dim, d_ff, rope_base, eps, grad);
  m.final_norm = vec_of(d_model, grad);
  m.lm_head = mat_of(d_model, vocab, grad);
  return m;
}

LlamaModel llama_model_create(int vocab, int d_model, int n_layers, int n_heads,
                              int n_kv, int head_dim, int d_ff, int max_seq,
                              float rope_base, float eps) {
  return model_create_ex(vocab, d_model, n_layers, n_heads, n_kv, head_dim,
                         d_ff, max_seq, rope_base, eps, 0);
}

LlamaModel llama_model_create_train(int vocab, int d_model, int n_layers, int n_heads,
                                    int n_kv, int head_dim, int d_ff, int max_seq,
                                    float rope_base, float eps) {
  return model_create_ex(vocab, d_model, n_layers, n_heads, n_kv, head_dim,
                         d_ff, max_seq, rope_base, eps, 1);
}

void llama_model_free(LlamaModel *m) {
  tensor_free(&m->wte);
  for (int l = 0; l < m->n_layers; l++) llama_block_free(&m->blocks[l]);
  free(m->blocks);
  tensor_free(&m->final_norm); tensor_free(&m->lm_head);
}

void llama_model_set_qk_norm(LlamaModel *m, int on) {
  m->qk_norm = on ? 1 : 0;
  for (int l = 0; l < m->n_layers; l++) {
    LlamaBlock *b = &m->blocks[l];
    b->qk_norm = m->qk_norm;
    if (on)
      for (int i = 0; i < b->head_dim; i++) b->q_norm.data[i] = b->k_norm.data[i] = 1.0f;
  }
}

void llama_model_randomize(LlamaModel *m) {
  fill_rand(&m->wte, 0.05f);
  for (int l = 0; l < m->n_layers; l++) llama_block_randomize(&m->blocks[l]);
  for (int j = 0; j < m->d_model; j++) m->final_norm.data[j] = 1.0f;
  fill_rand(&m->lm_head, 0.05f);
}

void llama_model_set_window(LlamaModel *m, int window, int global_every) {
  for (int l = 0; l < m->n_layers; l++) {
    const int global = (window <= 0) || (global_every > 0 && (l + 1) % global_every == 0);
    m->blocks[l].window = global ? 0 : window;
  }
}

void llama_model_copy_window(LlamaModel *dst, const LlamaModel *src) {
  for (int l = 0; l < dst->n_layers && l < src->n_layers; l++)
    dst->blocks[l].window = src->blocks[l].window;
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
#define LLMA_VERSION_WINDOW 3u    // v3 = v2 + n_layers int32 sliding windows after the
                                  // float config; written only when a window is set
#define LLMA_VERSION_FLAGS 4u     // v4 = v2 + a u32 feature-flags word after the float
                                  // config, then the windows iff LLMA_FLAG_WINDOWS
#define LLMA_FLAG_QK_NORM 1u      // bit 0: per-head q/k RMSNorm (Qwen3) — q_norm, k_norm
#define LLMA_FLAG_WINDOWS 2u      // bit 1: the n_layers int32 window block is present
#define LLMA_KNOWN_FLAGS  (LLMA_FLAG_QK_NORM | LLMA_FLAG_WINDOWS)
#define TENSORS_PER_LAYER 14      // attn_norm, Wq,Wk,Wv, b_q,b_k,b_v, [q_norm,k_norm], Wo,
                                  // ffn_norm, Wgate,Wup,Wdown  (the [..] only if QK-norm)

// Every weight tensor in a fixed canonical order shared by save and load, so the
// two can never drift. 1 (wte) + 12 per layer (+2 with QK-norm) + 2 (final_norm,
// lm_head). TENSORS_PER_LAYER is the upper bound, for sizing the list.
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
    if (m->qk_norm) {
      list[n++] = &b->q_norm;
      list[n++] = &b->k_norm;
    }
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

// Atomically replace dst with tmp: rename is atomic on POSIX; on Windows rename
// fails if dst exists, so fall back to remove+rename. Returns 0 on success.
static int replace_file(const char *tmp, const char *dst) {
  if (rename(tmp, dst) == 0) return 0;
  remove(dst);
  return rename(tmp, dst);
}

int llama_model_save(const char *path, const LlamaModel *m) {
  // Write to a temp file, then atomically rename over `path`, so a failed or
  // interrupted save never truncates an existing good checkpoint.
  char *tmp = (char*)malloc(strlen(path) + 5);
  if (!tmp) { fprintf(stderr, "Error: out of memory saving '%s'\n", path); return 1; }
  sprintf(tmp, "%s.tmp", path);

  FILE *f = fopen(tmp, "wb");
  if (!f) { fprintf(stderr, "Error: cannot open '%s' for writing\n", tmp); free(tmp); return 1; }

  // The lowest version that can hold the model, so older builds keep reading
  // what they can: no feature -> v2 (byte-identical to a pre-feature file);
  // windows only -> v3 (unchanged); QK-norm -> v4, windows then via flag bit 1.
  int windowed = 0;
  for (int l = 0; l < m->n_layers; l++) if (m->blocks[l].window > 0) windowed = 1;
  const uint32_t flags = m->qk_norm ? (LLMA_FLAG_QK_NORM | (windowed ? LLMA_FLAG_WINDOWS : 0u)) : 0u;
  uint32_t header[2] = {LLMA_MAGIC, flags ? LLMA_VERSION_FLAGS
                                          : windowed ? LLMA_VERSION_WINDOW : LLMA_VERSION};
  int32_t cfg[8] = {m->vocab, m->d_model, m->n_layers, m->n_heads,
                    m->n_kv, m->head_dim, m->d_ff, m->max_seq};
  float fcfg[2] = {m->rope_base, m->eps};

  int ok = 1;
  ok &= (fwrite(header, sizeof(uint32_t), 2, f) == 2);
  ok &= (fwrite(cfg, sizeof(int32_t), 8, f) == 8);
  ok &= (fwrite(fcfg, sizeof(float), 2, f) == 2);
  if (flags) ok &= (fwrite(&flags, sizeof(uint32_t), 1, f) == 1);
  for (int l = 0; l < m->n_layers && windowed && ok; l++) {
    int32_t w = m->blocks[l].window > 0 ? m->blocks[l].window : 0;   // normalize "off" to 0
    ok &= (fwrite(&w, sizeof(int32_t), 1, f) == 1);
  }

  const Tensor **list = (const Tensor**)malloc((size_t)(1 + TENSORS_PER_LAYER * m->n_layers + 2) * sizeof(const Tensor*));
  int count = llama_collect(m, list);
  for (int i = 0; i < count && ok; i++) {
    size_t sz = (size_t)list[i]->size;
    ok &= (fwrite(list[i]->data, sizeof(float), sz, f) == sz);
  }
  free(list);
  ok &= (fflush(f) == 0);
  ok &= (fclose(f) == 0);          // catch deferred write errors (e.g. disk full)

  if (ok) ok = (replace_file(tmp, path) == 0);
  if (!ok) {
    remove(tmp);
    fprintf(stderr, "Error: failed while writing '%s'\n", path);
    free(tmp);
    return 1;
  }
  free(tmp);
  return 0;
}

// ---------------------------------------------------------------------------
// KV cache + incremental decode
// ---------------------------------------------------------------------------

// Shared by the fp32 and int8 caches (their block types differ; the per-layer
// window, passed in as a plain array, does not).
static LlamaKV kv_create_ex(int n_layers, int max_seq, int kv_dim, const int *windows) {
  LlamaKV kv;
  kv.n_layers = n_layers;
  kv.max_seq = max_seq;
  kv.kv_dim = kv_dim;
  kv.len = 0;
  kv.cap = (int*)malloc((size_t)n_layers * sizeof(int));
  kv.k = (float**)malloc((size_t)n_layers * sizeof(float*));
  kv.v = (float**)malloc((size_t)n_layers * sizeof(float*));
  for (int l = 0; l < n_layers; l++) {
    // A windowed layer only ever reads its last W positions: a W-slot ring.
    kv.cap[l] = (windows[l] > 0 && windows[l] < max_seq) ? windows[l] : max_seq;
    kv.k[l] = (float*)malloc((size_t)kv.cap[l] * kv_dim * sizeof(float));
    kv.v[l] = (float*)malloc((size_t)kv.cap[l] * kv_dim * sizeof(float));
  }
  return kv;
}

LlamaKV llama_kv_create(const LlamaModel *m) {
  int *w = (int*)malloc((size_t)m->n_layers * sizeof(int));
  for (int l = 0; l < m->n_layers; l++) w[l] = m->blocks[l].window;
  LlamaKV kv = kv_create_ex(m->n_layers, m->max_seq, m->n_kv * m->head_dim, w);
  free(w);
  return kv;
}

void llama_kv_free(LlamaKV *kv) {
  for (int l = 0; l < kv->n_layers; l++) { free(kv->k[l]); free(kv->v[l]); }
  free(kv->k); free(kv->v); free(kv->cap);
  kv->k = kv->v = NULL; kv->cap = NULL;
}

// Single-query GQA attention of one new token (position pos) over one layer's
// cache (kc, vc) of `cap` ring slots, honouring the sliding window; writes o
// (n_heads*hd). Shared by the fp32 and int8 decode paths. Keys are visited in
// ascending POSITION order lo..pos — the same order and arithmetic as
// gqa_forward — which keeps decode bit-exact vs the full forward even after the
// ring has wrapped. `sc` is scratch of at least pos+1 floats.
static void decode_attention(const float *q, const float *kc, const float *vc,
                             int cap, int window, int pos, int n_heads, int n_kv,
                             int hd, float *sc, float *o) {
  const int group = n_heads / n_kv, kvd = n_kv * hd;
  const float scale = 1.0f / sqrtf((float)hd);
  const int lo = gqa_window_lo(pos, window);
  for (int head = 0; head < n_heads; head++) {
    const int kvh = head / group;
    const float *qh = &q[head * hd];
    float maxs = -INFINITY;
    for (int jj = lo; jj <= pos; jj++) {
      const float *kj = &kc[(size_t)(jj % cap) * kvd + kvh * hd];
      float dot = 0.0f;
      for (int d = 0; d < hd; d++) dot += qh[d] * kj[d];
      float s = dot * scale;
      sc[jj] = s;
      if (s > maxs) maxs = s;
    }
    float sum = 0.0f;
    for (int jj = lo; jj <= pos; jj++) { float e = expf(sc[jj] - maxs); sc[jj] = e; sum += e; }
    float inv = 1.0f / sum;
    float *oh = &o[head * hd];
    for (int d = 0; d < hd; d++) oh[d] = 0.0f;
    for (int jj = lo; jj <= pos; jj++) {
      float w = sc[jj] * inv;
      const float *vj = &vc[(size_t)(jj % cap) * kvd + kvh * hd];
      for (int d = 0; d < hd; d++) oh[d] += w * vj[d];
    }
  }
}

void llama_decode_step(const LlamaModel *m, LlamaKV *kv, int token, Tensor *logits) {
  llama_decode_step_lora(m, kv, token, logits, NULL);
}

void llama_decode_step_lora(const LlamaModel *m, LlamaKV *kv, int token,
                            Tensor *logits, const LlamaDecodeLora *lo) {
  const int dm = m->d_model, hd = m->head_dim;
  const int qd = m->n_heads * hd, kvd = m->n_kv * hd;
  const int pos = kv->len;                 // this token's position

  Tensor x = mat(1, dm);                     // running residual stream (1 row)
  for (int j = 0; j < dm; j++) x.data[j] = m->wte.data[token * dm + j];

  Tensor xn = mat(1, dm), q = mat(1, qd), k = mat(1, kvd), v = mat(1, kvd);
  Tensor o = mat(1, qd), ao = mat(1, dm), hn = mat(1, dm), f = mat(1, dm);
  float *sc = (float*)malloc((size_t)(pos + 1) * sizeof(float));
  SwigluCache sc_ff = swiglu_cache_create(1, m->d_ff);

  // One-row adapter scratch, reused across layers (decode never backpropagates,
  // so the cache is pure scratch). Only allocated when adapters are in play.
  LoraCache lcq, lck, lcv, lco;
  if (lo) {
    lcq = lora_cache_create(1, lo->q[0].r, qd);
    lck = lora_cache_create(1, lo->k[0].r, kvd);
    lcv = lora_cache_create(1, lo->v[0].r, kvd);
    lco = lora_cache_create(1, lo->o[0].r, dm);
  }

  for (int l = 0; l < m->n_layers; l++) {
    const LlamaBlock *b = &m->blocks[l];
    // Attention
    rmsnorm(&x, &b->attn_norm, &xn, m->eps);
    matmul(&xn, &b->Wq, &q); bias_add(&q, &b->b_q, &q);
    matmul(&xn, &b->Wk, &k); bias_add(&k, &b->b_k, &k);
    matmul(&xn, &b->Wv, &v); bias_add(&v, &b->b_v, &v);
    // Adapters land here — after the bias, BEFORE RoPE — matching
    // llama_forward_train_lora exactly.
    if (lo) {
      lora_forward(&lo->q[l], &xn, &q, &lcq);
      lora_forward(&lo->k[l], &xn, &k, &lck);
      lora_forward(&lo->v[l], &xn, &v, &lcv);
    }
    if (b->qk_norm) {                        // Qwen3 QK-norm: after adapters, before RoPE
      rmsnorm_heads(&q, &b->q_norm, &q, m->eps);
      rmsnorm_heads(&k, &b->k_norm, &k, m->eps);
    }
    rope_heads_pos(&q, m->n_heads, m->rope_base, pos);
    rope_heads_pos(&k, m->n_kv, m->rope_base, pos);
    const size_t slot = (size_t)(pos % kv->cap[l]) * kvd;   // ring slot (== pos if full)
    memcpy(&kv->k[l][slot], k.data, (size_t)kvd * sizeof(float));   // append to cache
    memcpy(&kv->v[l][slot], v.data, (size_t)kvd * sizeof(float));
    decode_attention(q.data, kv->k[l], kv->v[l], kv->cap[l], b->window, pos,
                     m->n_heads, m->n_kv, hd, sc, o.data);
    matmul(&o, &b->Wo, &ao);
    if (lo) lora_forward(&lo->o[l], &o, &ao, &lco);
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
  if (lo) {
    lora_cache_free(&lcq); lora_cache_free(&lck);
    lora_cache_free(&lcv); lora_cache_free(&lco);
  }
  tensor_free(&x); tensor_free(&xn); tensor_free(&q); tensor_free(&k); tensor_free(&v);
  tensor_free(&o); tensor_free(&ao); tensor_free(&hn); tensor_free(&f);
}

// Read the optional per-layer window block that follows the float config (and,
// in v4, the flags word): absent in v2 and in v4 without LLMA_FLAG_WINDOWS
// (every window 0 = full attention), n_layers int32 in v3 and in v4 with it.
// Rejects a negative or absurd window. Returns 1 on success, 0 on truncation/bad value.
static int read_windows(FILE *f, uint32_t version, uint32_t flags, int n_layers, int32_t *win) {
  for (int l = 0; l < n_layers; l++) win[l] = 0;
  const int present = (version == LLMA_VERSION_WINDOW) ||
                      (version == LLMA_VERSION_FLAGS && (flags & LLMA_FLAG_WINDOWS));
  if (!present) return 1;
  if (fread(win, sizeof(int32_t), (size_t)n_layers, f) != (size_t)n_layers) return 0;
  for (int l = 0; l < n_layers; l++)
    if (win[l] < 0 || win[l] > 1000000) return 0;
  return 1;
}

// Validate the 8 config ints of an LLMA/LLMQ header before building a model.
// cfg = {vocab, d_model, n_layers, n_heads, n_kv, head_dim, d_ff, max_seq}.
// Guards against corrupt/hostile checkpoints (the primary input path of this
// tool): non-positive/huge dims, n_kv=0 (division by zero in decode), GQA groups
// that don't divide evenly (out-of-bounds K/V reads), and 2D weight element
// counts above INT_MAX (the forward/embedding path still indexes with int
// arithmetic, even though Tensor.size is now size_t). Returns 1 if sane, else 0.
static int llama_config_valid(const int32_t *cfg) {
  const int32_t MAXD = 1000000;
  for (int i = 0; i < 8; i++)
    if (cfg[i] <= 0 || cfg[i] > MAXD) return 0;
  int vocab = cfg[0], d_model = cfg[1], n_heads = cfg[3];
  int n_kv = cfg[4], head_dim = cfg[5], d_ff = cfg[6];
  if (n_heads % n_kv != 0) return 0;                    // GQA groups divide evenly
  long long qd = (long long)n_heads * head_dim;
  long long kvd = (long long)n_kv * head_dim;
  if (qd > MAXD || kvd > MAXD) return 0;
  // Every 2D weight's element count must fit in int (forward path indexes in int).
  if ((long long)vocab * d_model > (long long)INT_MAX) return 0;   // wte / lm_head
  if ((long long)d_model * d_ff > (long long)INT_MAX) return 0;    // SwiGLU weights
  if (qd * d_model > (long long)INT_MAX) return 0;                 // Wq / Wo
  return 1;
}

// Read an LLMA/LLMQ header — magic + version, the 8 config ints, the 2 floats
// and, for v4 only, the feature-flags word (v2/v3 have none: flags = 0; v3's
// windows are implied by the version). The window block is read separately.
// Rejects unknown versions and unknown flag bits (a feature this build cannot
// run must fail loudly, not load as a silently different model). The caller
// checks the magic. Returns 0 on success, else prints why and returns nonzero.
static int read_header(FILE *f, const char *path, uint32_t *header, int32_t *cfg,
                       float *fcfg, uint32_t *flags) {
  *flags = 0;
  if (fread(header, sizeof(uint32_t), 2, f) != 2 ||
      fread(cfg, sizeof(int32_t), 8, f) != 8 ||
      fread(fcfg, sizeof(float), 2, f) != 2 ||
      (header[1] == LLMA_VERSION_FLAGS && fread(flags, sizeof(uint32_t), 1, f) != 1)) {
    fprintf(stderr, "Error: '%s' is truncated (bad header)\n", path);
    return 1;
  }
  if (header[1] != LLMA_VERSION && header[1] != LLMA_VERSION_WINDOW &&
      header[1] != LLMA_VERSION_FLAGS) {
    fprintf(stderr, "Error: '%s' is checkpoint version %u (this build reads v%u-v%u)\n",
            path, header[1], LLMA_VERSION, LLMA_VERSION_FLAGS);
    return 1;
  }
  if (*flags & ~(uint32_t)LLMA_KNOWN_FLAGS) {
    fprintf(stderr, "Error: '%s' uses unsupported feature flags 0x%x\n", path, *flags);
    return 1;
  }
  return 0;
}

int llama_model_load(const char *path, LlamaModel *out) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "Error: cannot open '%s' for reading\n", path); return 1; }

  uint32_t header[2], flags;
  int32_t cfg[8];
  float fcfg[2];
  if (read_header(f, path, header, cfg, fcfg, &flags) != 0) { fclose(f); return 1; }
  const int quant = (header[0] == LLMQ_MAGIC);
  if (header[0] != LLMA_MAGIC && !quant) {
    fprintf(stderr, "Error: '%s' is not an LLMA/LLMQ checkpoint\n", path);
    fclose(f); return 1;
  }
  if (!llama_config_valid(cfg)) {
    fprintf(stderr, "Error: '%s' has an invalid/unsupported config\n", path);
    fclose(f); return 1;
  }
  int32_t *win = (int32_t*)malloc((size_t)cfg[2] * sizeof(int32_t));
  if (!win || !read_windows(f, header[1], flags, cfg[2], win)) {
    fprintf(stderr, "Error: '%s' has a truncated/invalid sliding-window block\n", path);
    free(win); fclose(f); return 1;
  }

  *out = llama_model_create(cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[5],
                            cfg[6], cfg[7], fcfg[0], fcfg[1]);
  // Before llama_collect: the flag decides which tensors the file holds.
  llama_model_set_qk_norm(out, (flags & LLMA_FLAG_QK_NORM) != 0);
  for (int l = 0; l < out->n_layers; l++) out->blocks[l].window = win[l];
  free(win);

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
  b.window = 0;
  b.attn_norm = vec(d_model); b.ffn_norm = vec(d_model);
  b.b_q = vec(qd); b.b_k = vec(kvd); b.b_v = vec(kvd);
  b.q_norm = vec(head_dim); b.k_norm = vec(head_dim);
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
  tensor_free(&b->q_norm); tensor_free(&b->k_norm);
  qtensor_free(&b->Wq); qtensor_free(&b->Wk); qtensor_free(&b->Wv); qtensor_free(&b->Wo);
  qtensor_free(&b->Wgate); qtensor_free(&b->Wup); qtensor_free(&b->Wdown);
}

static QLlamaModel q_model_alloc(int vocab, int d_model, int n_layers, int n_heads, int n_kv,
                                 int head_dim, int d_ff, int max_seq, float rope_base, float eps) {
  QLlamaModel m;
  m.vocab = vocab; m.d_model = d_model; m.n_layers = n_layers; m.n_heads = n_heads;
  m.n_kv = n_kv; m.head_dim = head_dim; m.d_ff = d_ff; m.max_seq = max_seq;
  m.rope_base = rope_base; m.eps = eps;
  m.qk_norm = 0;
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
  const size_t hdb = (size_t)s->head_dim * sizeof(float);
  m.qk_norm = s->qk_norm;
  qtensor_quantize(&m.wte, s->wte.data);
  for (int l = 0; l < s->n_layers; l++) {
    const LlamaBlock *a = &s->blocks[l];
    QLlamaBlock *b = &m.blocks[l];
    b->window = a->window;
    memcpy(b->attn_norm.data, a->attn_norm.data, dmb);
    memcpy(b->ffn_norm.data, a->ffn_norm.data, dmb);
    memcpy(b->b_q.data, a->b_q.data, (size_t)a->b_q.size * sizeof(float));
    memcpy(b->b_k.data, a->b_k.data, (size_t)a->b_k.size * sizeof(float));
    memcpy(b->b_v.data, a->b_v.data, (size_t)a->b_v.size * sizeof(float));
    memcpy(b->q_norm.data, a->q_norm.data, hdb);    // norm gains stay f32
    memcpy(b->k_norm.data, a->k_norm.data, hdb);
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
  int *w = (int*)malloc((size_t)m->n_layers * sizeof(int));
  for (int l = 0; l < m->n_layers; l++) w[l] = m->blocks[l].window;
  LlamaKV kv = kv_create_ex(m->n_layers, m->max_seq, m->n_kv * m->head_dim, w);
  free(w);
  return kv;
}

void q_llama_decode_step(const QLlamaModel *m, LlamaKV *kv, int token, Tensor *logits) {
  const int dm = m->d_model, hd = m->head_dim, dff = m->d_ff;
  const int qd = m->n_heads * hd, kvd = m->n_kv * hd;
  const int pos = kv->len;

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
    if (m->qk_norm) {                        // Qwen3 QK-norm (f32 gains), before RoPE
      rmsnorm_heads(&q, &b->q_norm, &q, m->eps);
      rmsnorm_heads(&k, &b->k_norm, &k, m->eps);
    }
    rope_heads_pos(&q, m->n_heads, m->rope_base, pos);
    rope_heads_pos(&k, m->n_kv, m->rope_base, pos);
    const size_t slot = (size_t)(pos % kv->cap[l]) * kvd;   // ring slot (== pos if full)
    memcpy(&kv->k[l][slot], k.data, (size_t)kvd * sizeof(float));   // append to cache
    memcpy(&kv->v[l][slot], v.data, (size_t)kvd * sizeof(float));
    decode_attention(q.data, kv->k[l], kv->v[l], kv->cap[l], b->window, pos,
                     m->n_heads, m->n_kv, hd, sc, o.data);
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

  uint32_t header[2], flags;
  int32_t cfg[8];
  float fcfg[2];
  if (read_header(f, path, header, cfg, fcfg, &flags) != 0) { fclose(f); return 1; }
  if (header[0] != LLMQ_MAGIC) {
    fprintf(stderr, "Error: '%s' is not an int8 LLMQ checkpoint\n", path);
    fclose(f); return 1;
  }
  if (!llama_config_valid(cfg)) {
    fprintf(stderr, "Error: '%s' has an invalid/unsupported config\n", path);
    fclose(f); return 1;
  }
  int32_t *win = (int32_t*)malloc((size_t)cfg[2] * sizeof(int32_t));
  if (!win || !read_windows(f, header[1], flags, cfg[2], win)) {
    fprintf(stderr, "Error: '%s' has a truncated/invalid sliding-window block\n", path);
    free(win); fclose(f); return 1;
  }

  *out = q_model_alloc(cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[5], cfg[6], cfg[7], fcfg[0], fcfg[1]);
  out->qk_norm = (flags & LLMA_FLAG_QK_NORM) != 0;
  for (int l = 0; l < out->n_layers; l++) out->blocks[l].window = win[l];
  free(win);
  const size_t dmb = (size_t)out->d_model, hdn = (size_t)out->head_dim;

  int ok = q_read(f, &out->wte);
  for (int l = 0; l < out->n_layers && ok; l++) {
    QLlamaBlock *b = &out->blocks[l];
    ok &= (fread(b->attn_norm.data, sizeof(float), dmb, f) == dmb);
    ok &= q_read(f, &b->Wq); ok &= q_read(f, &b->Wk); ok &= q_read(f, &b->Wv);
    ok &= (fread(b->b_q.data, sizeof(float), (size_t)b->b_q.size, f) == (size_t)b->b_q.size);
    ok &= (fread(b->b_k.data, sizeof(float), (size_t)b->b_k.size, f) == (size_t)b->b_k.size);
    ok &= (fread(b->b_v.data, sizeof(float), (size_t)b->b_v.size, f) == (size_t)b->b_v.size);
    if (out->qk_norm) {                      // same slot as llama_collect: after b_v
      ok &= (fread(b->q_norm.data, sizeof(float), hdn, f) == hdn);
      ok &= (fread(b->k_norm.data, sizeof(float), hdn, f) == hdn);
    }
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
