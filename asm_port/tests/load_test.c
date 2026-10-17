// load_test.c - checks the asm loader (load.S / mem.S / io.S) against the C engine.
//
// Build + run from asm_port/:
//   ./build.sh test load_test ../llama.c ../quant.c ../tensor.c ../matrix.c \
//       ../rmsnorm.c ../rope.c ../gqa.c ../swiglu.c ../lora.c ../backend.c \
//       ../backend_mock.c ../backend_opencl.c ../backend_cuda.c ../backend_hip.c ../activation.c
//
// For every checkpoint it loads the model with the C loader (llama_model_load /
// q_llama_model_load) and with model_load, then compares the config, the windows,
// every norm / bias vector and every matrix byte for byte (Q8: raw int8 + scale,
// never the kernel's packed layout). Also: fp32 -> Q8 at load (flag 1) against
// q_llama_quantize_from, synthetic v3 / v4 (QK-norm, tied, windows) checkpoints in
// both formats, kv_create against llama_kv_create, the io.S helpers, and error
// cases. Prints load times (asm vs C).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <windows.h>

#include "llama.h"
#include "quant.h"

// ---- mirrors of model.inc ----------------------------------------------------
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

_Static_assert(sizeof(Mat) == 64, "Mat");
_Static_assert(sizeof(Layer) == 512, "Layer");
_Static_assert(offsetof(Layer, Wq) == 64 && offsetof(Layer, Wdown) == 448, "Layer offs");
_Static_assert(sizeof(Model) == 256, "Model");
_Static_assert(offsetof(Model, wte) == 112 && offsetof(Model, lm_head) == 176, "Model offs");
_Static_assert(offsetof(Model, rope_base) == 64 && offsetof(Model, layers) == 96, "Model offs2");
_Static_assert(sizeof(KV) == 64, "KV");

// ---- asm API --------------------------------------------------------------------
int64_t model_load(const wchar_t *path, Model *m, int64_t flags);
int64_t model_peek_magic(const wchar_t *path);
void model_free(Model *m);
int64_t kv_create(const Model *m, KV *kv);
int64_t kv_create_len(const Model *m, KV *kv, int64_t max_len);
void kv_free(KV *kv);
void q8_quant_percol(const float *w, int64_t rows, int64_t cols, signed char *q, float *scale);
void *mem_alloc(int64_t n);
void mem_free(void *p);
int64_t fmt_i64(char *b, int64_t v);
int64_t fmt_u64(char *b, uint64_t v);
int64_t fmt_hex(char *b, uint64_t v);
int64_t fmt_f64(char *b, double v, int64_t dec);
int64_t parse_i64(const wchar_t *s);
uint64_t parse_u64(const wchar_t *s);
double parse_f64(const wchar_t *s);
int64_t json_escape(unsigned char *dst, const unsigned char *src, int64_t n);
int64_t io_crt_text(unsigned char *buf, int64_t len);
int64_t io_time_ns(void);
unsigned char *io_read_file(const wchar_t *path, int64_t *len);

// Fallbacks so the test links before/without the kernels' matvec.S. `weak`: the
// real asm functions win when they exist.
__attribute__((weak)) void *mat_q8_pack(Mat *m) { (void)m; return 0; }
__attribute__((weak)) void *mat_q8_from_f32(Mat *m, const float *w, int64_t K, int64_t N) {
  m->kind = 2; m->K = K; m->N = N;
  m->data = malloc((size_t)(K * N)); m->scale = malloc((size_t)N * 4);
  quantize_int8_percol(w, (int)K, (int)N, (signed char *)m->data, m->scale);
  return 0;
}

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { g_fail++; printf("  FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static double now_s(void) { return (double)io_time_ns() * 1e-9; }

static int same(const void *a, const void *b, size_t n) {
  return a && b && memcmp(a, b, n) == 0;
}

static void cmp_vec(const char *what, int l, const float *asmv, const Tensor *t) {
  CHECK(same(asmv, t->data, t->size * 4), "%s layer %d differs", what, l);
}
static void cmp_f32mat(const char *what, int l, const Mat *m, const Tensor *t) {
  CHECK(m->kind == 0 && m->K == t->shape[0] && m->N == t->shape[1], "%s layer %d shape/kind (%lld %lld %lld)",
        what, l, (long long)m->kind, (long long)m->K, (long long)m->N);
  CHECK(same(m->data, t->data, t->size * 4), "%s layer %d f32 data differs", what, l);
}
static void cmp_q8mat(const char *what, int l, const Mat *m, const QTensor *q) {
  CHECK(m->kind == 2 && m->K == q->rows && m->N == q->cols, "%s layer %d shape/kind", what, l);
  CHECK(same(m->scale, q->scale, (size_t)q->cols * 4), "%s layer %d scale differs", what, l);
  CHECK(same(m->data, q->q, (size_t)q->rows * q->cols), "%s layer %d int8 data differs", what, l);
}
static void cmp_cfg(const Model *m, int vocab, int d, int nl, int nh, int nkv, int hd, int dff, int ms,
                    float rb, float eps, int qk) {
  CHECK(m->vocab == vocab && m->d_model == d && m->n_layers == nl && m->n_heads == nh &&
        m->n_kv == nkv && m->head_dim == hd && m->d_ff == dff && m->max_seq == ms, "config ints");
  CHECK(m->rope_base == rb && m->eps == eps, "rope/eps");
  CHECK(m->qk_norm == qk, "qk_norm %lld vs %d", (long long)m->qk_norm, qk);
}

// Compare an int8 model (C QLlamaModel) with an asm Q8 model.
static void cmp_q(const Model *m, const QLlamaModel *c) {
  cmp_cfg(m, c->vocab, c->d_model, c->n_layers, c->n_heads, c->n_kv, c->head_dim, c->d_ff, c->max_seq,
          c->rope_base, c->eps, c->qk_norm);
  CHECK(m->quant == 1, "quant flag");
  cmp_q8mat("wte", -1, &m->wte, &c->wte);
  for (int l = 0; l < c->n_layers; l++) {
    const Layer *a = &m->layers[l];
    const QLlamaBlock *b = &c->blocks[l];
    CHECK(a->window == b->window, "window layer %d", l);
    cmp_vec("attn_norm", l, a->attn_norm, &b->attn_norm);
    cmp_vec("ffn_norm", l, a->ffn_norm, &b->ffn_norm);
    cmp_vec("b_q", l, a->b_q, &b->b_q);
    cmp_vec("b_k", l, a->b_k, &b->b_k);
    cmp_vec("b_v", l, a->b_v, &b->b_v);
    if (c->qk_norm) {
      cmp_vec("q_norm", l, a->q_norm, &b->q_norm);
      cmp_vec("k_norm", l, a->k_norm, &b->k_norm);
    } else {
      CHECK(a->q_norm == NULL && a->k_norm == NULL, "q/k_norm set without QK-norm");
    }
    cmp_q8mat("Wq", l, &a->Wq, &b->Wq); cmp_q8mat("Wk", l, &a->Wk, &b->Wk);
    cmp_q8mat("Wv", l, &a->Wv, &b->Wv); cmp_q8mat("Wo", l, &a->Wo, &b->Wo);
    cmp_q8mat("Wgate", l, &a->Wgate, &b->Wgate); cmp_q8mat("Wup", l, &a->Wup, &b->Wup);
    cmp_q8mat("Wdown", l, &a->Wdown, &b->Wdown);
  }
  cmp_vec("final_norm", -1, m->final_norm, &c->final_norm);
  cmp_q8mat("lm_head", -1, &m->lm_head, &c->lm_head);
}

// Compare an fp32 model.
static void cmp_f(const Model *m, const LlamaModel *c) {
  cmp_cfg(m, c->vocab, c->d_model, c->n_layers, c->n_heads, c->n_kv, c->head_dim, c->d_ff, c->max_seq,
          c->rope_base, c->eps, c->qk_norm);
  CHECK(m->quant == 0, "quant flag");
  CHECK(m->tied == c->tied, "tied");
  cmp_f32mat("wte", -1, &m->wte, &c->wte);
  for (int l = 0; l < c->n_layers; l++) {
    const Layer *a = &m->layers[l];
    const LlamaBlock *b = &c->blocks[l];
    CHECK(a->window == b->window, "window layer %d", l);
    cmp_vec("attn_norm", l, a->attn_norm, &b->attn_norm);
    cmp_vec("ffn_norm", l, a->ffn_norm, &b->ffn_norm);
    cmp_vec("b_q", l, a->b_q, &b->b_q);
    cmp_vec("b_k", l, a->b_k, &b->b_k);
    cmp_vec("b_v", l, a->b_v, &b->b_v);
    if (c->qk_norm) {
      cmp_vec("q_norm", l, a->q_norm, &b->q_norm);
      cmp_vec("k_norm", l, a->k_norm, &b->k_norm);
    }
    cmp_f32mat("Wq", l, &a->Wq, &b->Wq); cmp_f32mat("Wk", l, &a->Wk, &b->Wk);
    cmp_f32mat("Wv", l, &a->Wv, &b->Wv); cmp_f32mat("Wo", l, &a->Wo, &b->Wo);
    cmp_f32mat("Wgate", l, &a->Wgate, &b->Wgate); cmp_f32mat("Wup", l, &a->Wup, &b->Wup);
    cmp_f32mat("Wdown", l, &a->Wdown, &b->Wdown);
  }
  cmp_vec("final_norm", -1, m->final_norm, &c->final_norm);
  if (c->tied) {
    CHECK(m->lm_head.kind == 1 && m->lm_head.K == c->d_model && m->lm_head.N == c->vocab &&
          m->lm_head.data == m->wte.data, "tied lm_head must be F32T aliasing wte");
  } else {
    cmp_f32mat("lm_head", -1, &m->lm_head, &c->lm_head);
  }
}

static void cmp_kv(const Model *m, const LlamaKV *c, int64_t max_len) {
  KV kv;
  int64_t rc = max_len > 0 ? kv_create_len(m, &kv, max_len) : kv_create(m, &kv);
  CHECK(rc == 0, "kv_create rc");
  if (rc) return;
  CHECK(kv.n_layers == c->n_layers && kv.max_seq == c->max_seq && kv.kv_dim == c->kv_dim && kv.len == 0,
        "kv header (%lld %lld %lld)", (long long)kv.n_layers, (long long)kv.max_seq, (long long)kv.kv_dim);
  for (int l = 0; l < c->n_layers; l++) {
    CHECK(kv.cap[l] == c->cap[l], "kv cap layer %d: %lld vs %d", l, (long long)kv.cap[l], c->cap[l]);
    CHECK(((uintptr_t)kv.k[l] & 63) == 0 && ((uintptr_t)kv.v[l] & 63) == 0, "kv alignment");
    // touch the whole slot range (catches overlapping slices)
    memset(kv.k[l], 0x11, (size_t)kv.cap[l] * kv.kv_dim * 4);
    memset(kv.v[l], 0x22, (size_t)kv.cap[l] * kv.kv_dim * 4);
  }
  for (int l = 0; l < c->n_layers; l++)
    CHECK(((unsigned char *)kv.k[l])[0] == 0x11 && ((unsigned char *)kv.v[l])[(size_t)kv.cap[l] * kv.kv_dim * 4 - 1] == 0x22,
          "kv slices overlap");
  kv_free(&kv);
}

static wchar_t *wpath(const char *p) {
  static wchar_t buf[4][1024];
  static int i = 0;
  wchar_t *w = buf[i++ & 3];
  MultiByteToWideChar(CP_UTF8, 0, p, -1, w, 1024);
  return w;
}

// ---- real checkpoints -----------------------------------------------------------
static void test_q_file(const char *path) {
  printf("[LLMQ] %s\n", path);
  double t0 = now_s();
  QLlamaModel c;
  if (q_llama_model_load(path, &c) != 0) { printf("  (C loader failed; skipping)\n"); return; }
  double t1 = now_s();
  Model m;
  int64_t rc = model_load(wpath(path), &m, 2);   // 2 = keep raw int8 for the comparison
  double t2 = now_s();
  CHECK(rc == 0, "model_load rc=%lld", (long long)rc);
  printf("  load: C %.3f s, asm %.3f s (asm incl. mat_q8_pack)\n", t1 - t0, t2 - t1);
  if (rc == 0) {
    cmp_q(&m, &c);
    LlamaKV ckv = q_llama_kv_create(&c);
    cmp_kv(&m, &ckv, 0);
    llama_kv_free(&ckv);
    model_free(&m);
  }
  // second asm load, default flags (raw dropped after packing), warm cache
  t1 = now_s();
  rc = model_load(wpath(path), &m, 0);
  t2 = now_s();
  CHECK(rc == 0, "model_load (flags 0)");
  printf("  load (flags 0, warm): asm %.3f s\n", t2 - t1);
  if (rc == 0) {
    CHECK(m.wte.data != NULL, "wte raw must stay");
    CHECK(same(m.wte.data, c.wte.q, (size_t)c.vocab * c.d_model), "wte after flags 0");
    model_free(&m);
  }
  q_llama_model_free(&c);
}

static void test_f_file(const char *path, int do_q8) {
  printf("[LLMA] %s\n", path);
  double t0 = now_s();
  LlamaModel c;
  int crc = llama_model_load(path, &c);
  double t1 = now_s();
  Model m;
  int64_t rc = model_load(wpath(path), &m, 0);
  double t2 = now_s();
  CHECK((rc == 0) == (crc == 0), "C rc %d vs asm rc %lld", crc, (long long)rc);
  if (crc != 0) { printf("  (both loaders rejected it, as expected)\n"); if (rc == 0) model_free(&m); return; }
  printf("  load: C %.3f s, asm %.3f s\n", t1 - t0, t2 - t1);
  if (rc == 0) {
    cmp_f(&m, &c);
    LlamaKV ckv = llama_kv_create(&c);
    cmp_kv(&m, &ckv, 0);
    llama_kv_free(&ckv);
    ckv = llama_kv_create_len(&c, 37);
    cmp_kv(&m, &ckv, 37);
    llama_kv_free(&ckv);
    model_free(&m);
  }
  if (do_q8) {
    double t3 = now_s();
    QLlamaModel q = q_llama_quantize_from(&c);
    double t4 = now_s();
    rc = model_load(wpath(path), &m, 1 | 2);
    double t5 = now_s();
    CHECK(rc == 0, "model_load flag 1");
    printf("  fp32 -> Q8: C quantize %.3f s, asm load+quantize+pack %.3f s\n", t4 - t3, t5 - t4);
    if (rc == 0) {
      CHECK(m.tied == c.tied, "tied (flag 1)");
      cmp_q(&m, &q);
      model_free(&m);
    }
    q_llama_model_free(&q);
  }
  llama_model_free(&c);
}

// ---- synthetic checkpoints ------------------------------------------------------
static void fill_model(LlamaModel *c) {
  llama_model_randomize(c);
  // non-trivial norms / biases so a mix-up shows
  unsigned s = 12345;
  for (int l = 0; l < c->n_layers; l++) {
    LlamaBlock *b = &c->blocks[l];
    Tensor *ts[] = {&b->attn_norm, &b->ffn_norm, &b->b_q, &b->b_k, &b->b_v, &b->q_norm, &b->k_norm};
    for (int t = 0; t < 7; t++)
      for (size_t i = 0; i < ts[t]->size; i++) { s = s * 1103515245u + 12345u; ts[t]->data[i] = (float)(s >> 8) / 16777216.0f - 0.5f; }
  }
  for (size_t i = 0; i < c->final_norm.size; i++) c->final_norm.data[i] = 1.0f + 0.01f * (float)i;
}

static void write_q_tensor(FILE *f, const QTensor *t) {
  fwrite(t->scale, 4, (size_t)t->cols, f);
  fwrite(t->q, 1, (size_t)t->rows * t->cols, f);
}

// Write an LLMQ file by hand (no C writer exists; tools/convert_llama.py makes them).
static void save_llmq(const char *path, const QLlamaModel *q, uint32_t version, uint32_t flags, const int32_t *win) {
  FILE *f = fopen(path, "wb");
  uint32_t h[2] = {0x514D4C4Cu, version};
  int32_t cfg[8] = {q->vocab, q->d_model, q->n_layers, q->n_heads, q->n_kv, q->head_dim, q->d_ff, q->max_seq};
  float fc[2] = {q->rope_base, q->eps};
  fwrite(h, 4, 2, f); fwrite(cfg, 4, 8, f); fwrite(fc, 4, 2, f);
  if (version == 4) fwrite(&flags, 4, 1, f);
  if (version == 3 || (version == 4 && (flags & 2))) fwrite(win, 4, (size_t)q->n_layers, f);
  write_q_tensor(f, &q->wte);
  for (int l = 0; l < q->n_layers; l++) {
    const QLlamaBlock *b = &q->blocks[l];
    fwrite(b->attn_norm.data, 4, b->attn_norm.size, f);
    write_q_tensor(f, &b->Wq); write_q_tensor(f, &b->Wk); write_q_tensor(f, &b->Wv);
    fwrite(b->b_q.data, 4, b->b_q.size, f); fwrite(b->b_k.data, 4, b->b_k.size, f); fwrite(b->b_v.data, 4, b->b_v.size, f);
    if (flags & 1) { fwrite(b->q_norm.data, 4, b->q_norm.size, f); fwrite(b->k_norm.data, 4, b->k_norm.size, f); }
    write_q_tensor(f, &b->Wo);
    fwrite(b->ffn_norm.data, 4, b->ffn_norm.size, f);
    write_q_tensor(f, &b->Wgate); write_q_tensor(f, &b->Wup); write_q_tensor(f, &b->Wdown);
  }
  fwrite(q->final_norm.data, 4, q->final_norm.size, f);
  write_q_tensor(f, &q->lm_head);
  fclose(f);
}

static void truncate_copy(const char *src, const char *dst, long keep) {
  FILE *f = fopen(src, "rb"); FILE *g = fopen(dst, "wb");
  char *b = malloc((size_t)keep);
  long n = (long)fread(b, 1, (size_t)keep, f);
  fwrite(b, 1, (size_t)n, g);
  fclose(f); fclose(g); free(b);
}

static void test_synthetic(const char *dir) {
  char p[1024];
  // (1) v4: QK-norm + tied + windows (fp32)
  {
    LlamaModel c = llama_model_create(301, 48, 3, 6, 2, 8, 70, 40, 10000.0f, 1e-5f);
    llama_model_set_qk_norm(&c, 1);
    llama_model_set_tied(&c, 1);
    llama_model_set_window(&c, 7, 2);          // layer 1 global, others window 7
    fill_model(&c);
    snprintf(p, sizeof p, "%s/syn_v4.bin", dir);
    llama_model_save(p, &c);
    llama_model_free(&c);
    test_f_file(p, 1);
    // truncated variants: header, windows, weights
    char t[1024];
    long cuts[] = {3, 30, 50, 54, 4000};
    for (int i = 0; i < 5; i++) {
      snprintf(t, sizeof t, "%s/syn_trunc%d.bin", dir, i);
      truncate_copy(p, t, cuts[i]);
      LlamaModel cc;
      Model m;
      int crc = llama_model_load(t, &cc);
      int64_t arc = model_load(wpath(t), &m, 0);
      CHECK(crc != 0 && arc != 0, "truncated at %ld must fail in both (C %d asm %lld)", cuts[i], crc, (long long)arc);
      if (crc == 0) llama_model_free(&cc);
    }
  }
  // (2) v3: windows only (fp32), untied
  {
    LlamaModel c = llama_model_create(257, 64, 4, 4, 4, 16, 96, 33, 500000.0f, 1e-6f);
    llama_model_set_window(&c, 5, 0);          // every layer windowed
    fill_model(&c);
    snprintf(p, sizeof p, "%s/syn_v3.bin", dir);
    llama_model_save(p, &c);
    llama_model_free(&c);
    test_f_file(p, 1);
  }
  // (3) LLMQ v4 with QK-norm + windows, and LLMQ v3
  {
    LlamaModel c = llama_model_create(263, 40, 2, 5, 1, 8, 72, 50, 10000.0f, 1e-5f);
    llama_model_set_qk_norm(&c, 1);
    fill_model(&c);
    QLlamaModel q = q_llama_quantize_from(&c);
    int32_t win[2] = {9, 0};
    snprintf(p, sizeof p, "%s/syn_q4.bin", dir);
    save_llmq(p, &q, 4, 1 | 2, win);
    test_q_file(p);
    q.qk_norm = 0;
    snprintf(p, sizeof p, "%s/syn_q3.bin", dir);
    save_llmq(p, &q, 3, 0, win);
    test_q_file(p);
    // tied LLMQ must be rejected by both
    snprintf(p, sizeof p, "%s/syn_qtied.bin", dir);
    save_llmq(p, &q, 4, 4, win);
    QLlamaModel qq; Model m;
    CHECK(q_llama_model_load(p, &qq) != 0 && model_load(wpath(p), &m, 0) != 0, "tied LLMQ must be rejected");
    q_llama_model_free(&q);
    llama_model_free(&c);
  }
  // (4) tiny-context models with SmolLM's vocab size, for the CLI tests (prompt
  //     fitting, windows, QK-norm, tied head through main.S): syn_cli_f.bin (LLMA
  //     v4 tied+QK-norm+windows) and syn_cli_q.bin (LLMQ v4 QK-norm+windows)
  {
    LlamaModel c = llama_model_create(49152, 32, 2, 4, 2, 8, 48, 40, 10000.0f, 1e-5f);
    llama_model_set_qk_norm(&c, 1);
    llama_model_set_tied(&c, 1);
    llama_model_set_window(&c, 6, 2);
    fill_model(&c);
    snprintf(p, sizeof p, "%s/syn_cli_f.bin", dir);
    llama_model_save(p, &c);
    llama_model_set_tied(&c, 0);
    QLlamaModel q = q_llama_quantize_from(&c);
    int32_t win[2] = {6, 0};
    snprintf(p, sizeof p, "%s/syn_cli_q.bin", dir);
    save_llmq(p, &q, 4, 1 | 2, win);
    test_q_file(p);
    q_llama_model_free(&q);
    llama_model_free(&c);
  }
  // (5) peek / missing file
  {
    CHECK(model_peek_magic(L"definitely_missing_file.bin") == -1, "peek missing");
    snprintf(p, sizeof p, "%s/syn_q3.bin", dir);
    CHECK(model_peek_magic(wpath(p)) == 0x514D4C4C, "peek LLMQ");
    Model m;
    CHECK(model_load(L"definitely_missing_file.bin", &m, 0) != 0, "missing file must fail");
  }
}

// ---- io.S helpers ---------------------------------------------------------------
static void test_io(void) {
  printf("[io]\n");
  char b[128];
  int64_t n;
  n = fmt_i64(b, -1234567890123LL); CHECK(n == 14 && !strcmp(b, "-1234567890123"), "fmt_i64 '%s'", b);
  n = fmt_i64(b, 0); CHECK(n == 1 && !strcmp(b, "0"), "fmt_i64 0");
  n = fmt_u64(b, 18446744073709551615ULL); CHECK(!strcmp(b, "18446744073709551615") && n == 20, "fmt_u64 max");
  n = fmt_hex(b, 0xdeadbeefULL); CHECK(!strcmp(b, "deadbeef") && n == 8, "fmt_hex");
  n = fmt_hex(b, 0); CHECK(!strcmp(b, "0"), "fmt_hex 0");
  double fv[] = {3.14159, -2.5, 0.0, 1234.5678, 0.0049, 99.995, 1e15};
  int fd[] = {2, 1, 3, 0, 2, 4, 1};
  for (int i = 0; i < 7; i++) {
    char ref[64];
    fmt_f64(b, fv[i], fd[i]);
    snprintf(ref, sizeof ref, "%.*f", fd[i], fv[i]);
    CHECK(!strcmp(b, ref), "fmt_f64(%g,%d) '%s' vs '%s'", fv[i], fd[i], b, ref);
  }
  CHECK(parse_i64(L"  -42xyz") == -42 && parse_i64(L"+7") == 7 && parse_i64(L"abc") == 0, "parse_i64");
  CHECK(parse_u64(L"1234") == 1234 && parse_u64(L"99999999999999999999999") == ~0ULL &&
        parse_u64(L"-1") == ~0ULL && parse_u64(L"18446744073709551615") == ~0ULL, "parse_u64");
  const wchar_t *fs[] = {L"0.8", L"0", L"0.1", L"1e-3", L"  -2.5", L"123.456e2", L".5", L"7.", L"0.95", L"1.7976931348623157e308",
                         L"3.14159265358979323846", L"x", L"1e", L"0.000001"};
  const char *fsa[] = {"0.8", "0", "0.1", "1e-3", "  -2.5", "123.456e2", ".5", "7.", "0.95", "1.7976931348623157e308",
                       "3.14159265358979323846", "x", "1e", "0.000001"};
  for (int i = 0; i < 14; i++) {
    double a = parse_f64(fs[i]), r = atof(fsa[i]);
    // exact (= strtod) on the fast path; the 1e308 case is outside it and only
    // promised to within a few ulp
    if (i == 9) CHECK(a > r * (1 - 1e-15) && a < r * (1 + 1e-15), "parse_f64 big %.17g", a);
    else CHECK(a == r, "parse_f64('%s') %.17g vs %.17g", fsa[i], a, r);
    CHECK((float)a == (float)r, "parse_f64 float '%s'", fsa[i]);
  }
  unsigned char src[] = "a\"b\\c\nd\re\tf\x01g\x1f\x7f\xc3\xa9";
  unsigned char dst[128];
  n = json_escape(dst, src, (int64_t)sizeof(src) - 1);
  const char *want = "a\\\"b\\\\c\\nd\\re\\tf\\u0001g\\u001f\x7f\xc3\xa9";
  CHECK(n == (int64_t)strlen(want) && !memcmp(dst, want, (size_t)n), "json_escape '%.*s'", (int)n, dst);
  unsigned char t[] = "a\r\nb\rc\r\r\nd\x1a" "e";
  n = io_crt_text(t, (int64_t)sizeof(t) - 1);
  CHECK(n == 8 && !memcmp(t, "a\nb\rc\r\nd", 8), "io_crt_text n=%lld", (long long)n);
  // mem_alloc: alignment, zeroing, big + small
  int64_t sizes[] = {1, 100, 4096, 1 << 20, (1 << 22) + 3, 0};
  for (int i = 0; i < 6; i++) {
    unsigned char *q = mem_alloc(sizes[i]);
    CHECK(q && ((uintptr_t)q & 63) == 0, "mem_alloc align %lld", (long long)sizes[i]);
    int z = 1;
    for (int64_t k = 0; k < sizes[i]; k++) z &= (q[k] == 0);
    CHECK(z, "mem_alloc zero");
    memset(q, 0xAB, (size_t)sizes[i]);
    mem_free(q);
  }
  int64_t len = 0;
  unsigned char *fb = io_read_file(L"../llama.h", &len);
  FILE *f = fopen("../llama.h", "rb");
  fseek(f, 0, SEEK_END); long fl = ftell(f); fclose(f);
  CHECK(fb && len == fl && fb[len] == 0, "io_read_file");
  mem_free(fb);
  CHECK(io_read_file(L"nope_missing", &len) == NULL, "io_read_file missing");
}

int main(int argc, char **argv) {
  const char *dir = argc > 1 ? argv[1] : "build";
  test_io();
  test_synthetic(dir);
  test_q_file("../model/smollm/model_int8.bin");
  test_q_file("../model/qwen/model_int8.bin");
  test_f_file("../model/smollm/model.bin", 0);       // v1: both must reject it
  test_f_file("../model/smollm/model_v2.bin", 1);
  printf(g_fail ? "load_test: %d FAILURES\n" : "load_test: all OK\n", g_fail);
  return g_fail != 0;
}
