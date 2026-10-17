// load_cli_stubs.c - test harness for main.S (the CLI) BEFORE / WITHOUT the other
// agents' modules: implements the tokenizer / decode / sampler / kernel API of
// DESIGN.md on top of the C engine (hftok.c, llama.c, sampler.c, quant.c), so a
// build of main.S + this file must print exactly what `transformer hfgen` prints.
// It checks the CLI logic (args, stdin, prompt fitting, sampling loop, stop rules,
// JSON escaping, error paths) independently of the asm decode path.
//
// Build from asm_port/: main.S is assembled with -DLLM_NO_ENTRY (this file's
// main() calls llm_main) and -Dsample_token=stub_sample_token (the C sampler
// already owns the name sample_token); flags match the C engine's -O3 build so
// the float math is identical:
//   Z=/d/env/zig/zig.exe; T="-target x86_64-windows-gnu"
//   for f in mem io load; do $Z cc $T -c $f.S -o build/obj/$f.o; done
//   $Z cc $T -DLLM_NO_ENTRY -Dsample_token=stub_sample_token -c main.S -o build/obj/main_stub.o
//   $Z cc $T -std=c11 -O3 -march=native -I.. -o build/llm_stub.exe tests/load_cli_stubs.c \
//     ../llama.c ../quant.c ../tensor.c ../matrix.c ../rmsnorm.c ../rope.c ../gqa.c \
//     ../swiglu.c ../lora.c ../backend.c ../backend_mock.c ../backend_opencl.c \
//     ../backend_cuda.c ../backend_hip.c ../activation.c ../hftok.c ../sampler.c \
//     build/obj/main_stub.o build/obj/mem.o build/obj/io.o build/obj/load.o -lkernel32 -lshell32
// Then compare `printf PROMPT | build/llm_stub.exe ARGS` with
// `printf PROMPT | transformer hfgen ARGS` (a transformer.exe built from the current
// sources with the same flags): stdout, stderr and exit code must be identical.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>

#include "hftok.h"
#include "llama.h"
#include "quant.h"
#include "sampler.h"

typedef struct { int64_t kind, K, N; void *data; float *scale; void *packed, *colsum, *aux; } Mat;
typedef struct {
  int64_t window;
  float *attn_norm, *ffn_norm, *b_q, *b_k, *b_v, *q_norm, *k_norm;
  Mat Wq, Wk, Wv, Wo, Wgate, Wup, Wdown;
} Layer;
typedef struct {
  int64_t vocab, d_model, n_layers, n_heads, n_kv, head_dim, d_ff, max_seq;
  float rope_base, eps;
  int64_t qk_norm, quant, tied;
  Layer *layers;
  float *final_norm;
  Mat wte, lm_head;
  void *file_base, *spare;
} Model;
typedef struct { int64_t n_layers, max_seq, kv_dim, len; int64_t *cap; float **k, **v; void *spare; } KV;

int64_t llm_main(void);

// ---- kernels -------------------------------------------------------------------
int64_t cpu_init(void) { return 3; }
int64_t pool_init(int64_t n) { (void)n; return 1; }
int64_t pool_threads(void) { return 1; }
void *mat_q8_pack(Mat *m) { (void)m; return 0; }           // keep raw int8 (used below)
void *mat_q8_from_f32(Mat *m, const float *w, int64_t K, int64_t N) {
  m->kind = 2; m->K = K; m->N = N;
  m->data = malloc((size_t)(K * N)); m->scale = malloc((size_t)N * 4);
  quantize_int8_percol(w, (int)K, (int)N, (signed char *)m->data, m->scale);
  return 0;
}

// ---- tokenizer -----------------------------------------------------------------
void *tok_load(const wchar_t *path) {
  char p[2048];
  WideCharToMultiByte(CP_ACP, 0, path, -1, p, sizeof p, NULL, NULL);
  HFTok *t = (HFTok *)malloc(sizeof(HFTok));
  if (hftok_load(t, p) != 0) { free(t); return NULL; }
  return t;
}
int64_t tok_encode(void *tk, const unsigned char *text, int64_t len, int32_t *ids, int64_t cap) {
  char *s = (char *)malloc((size_t)len + 1);
  memcpy(s, text, (size_t)len); s[len] = 0;
  int n = hftok_encode((HFTok *)tk, s, ids, (int)cap);
  free(s);
  return n;
}
int64_t tok_piece(void *tk, int64_t id, unsigned char *out) {
  HFTok *t = (HFTok *)tk;
  if (id < 0 || id >= t->n_vocab) return 0;
  memcpy(out, t->tok_bytes[id], (size_t)t->tok_len[id]);
  return t->tok_len[id];
}
int64_t tok_is_eog(void *tk, int64_t id) { return hftok_is_eog((HFTok *)tk, (int)id); }
int64_t tok_vocab(void *tk) { return ((HFTok *)tk)->n_vocab; }
int64_t tok_max_piece(void *tk) {
  HFTok *t = (HFTok *)tk;
  int mx = 0;
  for (int i = 0; i < t->n_vocab; i++) if (t->tok_len[i] > mx) mx = t->tok_len[i];
  return mx;
}

// ---- sampler -------------------------------------------------------------------
void rng_seed(void *rng, uint64_t seed) { *(Rng *)rng = rng_create(seed); }
int64_t stub_sample_token(const float *logits, int64_t V, float temp, int64_t top_k, float top_p, void *rng) {
  int shape[2] = {1, (int)V};
  Tensor t; memset(&t, 0, sizeof t);
  t.data = (float *)logits; t.shape = shape; t.ndim = 2; t.size = (size_t)V;
  return sample_token(&t, 0, temp, (int)top_k, top_p, (Rng *)rng);
}

// ---- decode: a C model VIEW of the asm-loaded Model ------------------------------
static Tensor tv(float *d, int r, int c) {   // r == 0: 1-D tensor of c
  Tensor t; memset(&t, 0, sizeof t);
  t.shape = (int *)malloc(2 * sizeof(int));
  t.data = d;
  if (r == 0) { t.ndim = 1; t.shape[0] = c; t.size = (size_t)c; }
  else { t.ndim = 2; t.shape[0] = r; t.shape[1] = c; t.size = (size_t)r * c; }
  return t;
}
static Tensor tvm(const Mat *m) { return tv((float *)m->data, (int)m->K, (int)m->N); }
static QTensor qv(const Mat *m) { QTensor q; q.rows = (int)m->K; q.cols = (int)m->N; q.q = m->data; q.scale = m->scale; return q; }

static const Model *g_m;
static QLlamaModel g_q;
static LlamaModel g_f;
static LlamaKV g_ckv;
static float *g_scratch;

static void build_view(const Model *m) {
  g_m = m;
  const int qd = (int)(m->n_heads * m->head_dim), kvd = (int)(m->n_kv * m->head_dim);
  const int hd = (int)m->head_dim, D = (int)m->d_model;
  if (m->quant) {
    QLlamaModel *q = &g_q;
    q->vocab = (int)m->vocab; q->d_model = D; q->n_layers = (int)m->n_layers; q->n_heads = (int)m->n_heads;
    q->n_kv = (int)m->n_kv; q->head_dim = hd; q->d_ff = (int)m->d_ff; q->max_seq = (int)m->max_seq;
    q->rope_base = m->rope_base; q->eps = m->eps; q->qk_norm = (int)m->qk_norm;
    q->wte = qv(&m->wte);
    q->blocks = (QLlamaBlock *)calloc((size_t)m->n_layers, sizeof(QLlamaBlock));
    for (int l = 0; l < m->n_layers; l++) {
      const Layer *a = &m->layers[l];
      QLlamaBlock *b = &q->blocks[l];
      b->window = (int)a->window;
      b->attn_norm = tv(a->attn_norm, 0, D); b->ffn_norm = tv(a->ffn_norm, 0, D);
      b->b_q = tv(a->b_q, 0, qd); b->b_k = tv(a->b_k, 0, kvd); b->b_v = tv(a->b_v, 0, kvd);
      b->q_norm = tv(a->q_norm, 0, hd); b->k_norm = tv(a->k_norm, 0, hd);
      b->Wq = qv(&a->Wq); b->Wk = qv(&a->Wk); b->Wv = qv(&a->Wv); b->Wo = qv(&a->Wo);
      b->Wgate = qv(&a->Wgate); b->Wup = qv(&a->Wup); b->Wdown = qv(&a->Wdown);
    }
    q->final_norm = tv(m->final_norm, 0, D);
    q->lm_head = qv(&m->lm_head);
  } else {
    LlamaModel *f = &g_f;
    memset(f, 0, sizeof *f);
    f->vocab = (int)m->vocab; f->d_model = D; f->n_layers = (int)m->n_layers; f->n_heads = (int)m->n_heads;
    f->n_kv = (int)m->n_kv; f->head_dim = hd; f->d_ff = (int)m->d_ff; f->max_seq = (int)m->max_seq;
    f->rope_base = m->rope_base; f->eps = m->eps; f->qk_norm = (int)m->qk_norm; f->tied = (int)m->tied;
    f->wte = tvm(&m->wte);
    f->blocks = (LlamaBlock *)calloc((size_t)m->n_layers, sizeof(LlamaBlock));
    for (int l = 0; l < m->n_layers; l++) {
      const Layer *a = &m->layers[l];
      LlamaBlock *b = &f->blocks[l];
      b->d_model = D; b->n_heads = (int)m->n_heads; b->n_kv = (int)m->n_kv; b->head_dim = hd;
      b->d_ff = (int)m->d_ff; b->rope_base = m->rope_base; b->eps = m->eps; b->qk_norm = (int)m->qk_norm;
      b->window = (int)a->window;
      b->attn_norm = tv(a->attn_norm, 0, D); b->ffn_norm = tv(a->ffn_norm, 0, D);
      b->b_q = tv(a->b_q, 0, qd); b->b_k = tv(a->b_k, 0, kvd); b->b_v = tv(a->b_v, 0, kvd);
      b->q_norm = tv(a->q_norm, 0, hd); b->k_norm = tv(a->k_norm, 0, hd);
      b->Wq = tvm(&a->Wq); b->Wk = tvm(&a->Wk); b->Wv = tvm(&a->Wv); b->Wo = tvm(&a->Wo);
      b->Wgate = tvm(&a->Wgate); b->Wup = tvm(&a->Wup); b->Wdown = tvm(&a->Wdown);
    }
    f->final_norm = tv(m->final_norm, 0, D);
    if (!m->tied) f->lm_head = tvm(&m->lm_head);
  }
  g_scratch = (float *)malloc((size_t)m->vocab * sizeof(float));
}

static void sync_kv(KV *kv) {
  if (!g_ckv.cap) {
    g_ckv.n_layers = (int)kv->n_layers; g_ckv.max_seq = (int)kv->max_seq; g_ckv.kv_dim = (int)kv->kv_dim;
    g_ckv.cap = (int *)malloc((size_t)kv->n_layers * sizeof(int));
    for (int l = 0; l < kv->n_layers; l++) g_ckv.cap[l] = (int)kv->cap[l];
    g_ckv.k = kv->k; g_ckv.v = kv->v;
  }
  g_ckv.len = (int)kv->len;
}

int64_t decode_step(Model *m, KV *kv, int64_t token, float *logits) {
  if (g_m != m) build_view(m);
  if (kv->len >= kv->max_seq) return 1;
  sync_kv(kv);
  int shape[2] = {1, (int)m->vocab};
  Tensor lt; memset(&lt, 0, sizeof lt);
  lt.data = logits ? logits : g_scratch; lt.shape = shape; lt.ndim = 2; lt.size = (size_t)m->vocab;
  if (m->quant) q_llama_decode_step(&g_q, &g_ckv, (int)token, &lt);
  else llama_decode_step(&g_f, &g_ckv, (int)token, logits ? &lt : NULL);
  kv->len = g_ckv.len;
  return 0;
}

int64_t prefill(Model *m, KV *kv, const int32_t *ids, int64_t n) {
  for (int64_t i = 0; i < n; i++)
    if (decode_step(m, kv, ids[i], NULL)) return 1;
  return 0;
}

int main(void) { return (int)llm_main(); }
