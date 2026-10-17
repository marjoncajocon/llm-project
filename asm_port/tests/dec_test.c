// dec_test.c - checks the decode agent's asm (ops.S, decode.S, sample.S) against
// the C engine, and benchmarks it.
//
// Build + run from asm_port/ (the -D flags rename the C reference functions whose
// names the asm exports reuse; the asm ones are bound below with __asm__ labels):
//
//   ./build.sh test dec_test -Drmsnorm=c_rmsnorm -Drmsnorm_heads=c_rmsnorm_heads \
//       -Drng_uniform=c_rng_uniform -Dsample_token=c_sample_token \
//       -Dsample_argmax=c_sample_argmax \
//       ../llama.c ../quant.c ../tensor.c ../matrix.c ../rmsnorm.c ../rope.c \
//       ../gqa.c ../swiglu.c ../lora.c ../sampler.c ../backend.c ../backend_mock.c \
//       ../backend_opencl.c ../backend_cuda.c ../backend_hip.c ../activation.c
//
// Environment:
//   DEC_ONLY=ops|sample|decode|bench   run one part (default: ops, sample, decode)
//   DEC_ONLY=prof   per-token breakdown: decode_step vs matvecs alone vs attention alone
//   DEC_ONLY=dbg / dbgp   layer-truncated Qwen comparison / prefill size sweep
//   DEC_MODEL=path  model for prof / dbgp
//   DEC_BENCH=1                        also run the benchmarks
//
// Parts:
//   ops     expf (bit-exact), SiLU (bit-exact), RMSNorm, RoPE (powf bit-exact,
//           sin/cos ulps, rotated vectors), attention (vs llama.c decode_attention,
//           incl. sliding-window ring wrap, multi-row, threaded, generic head_dim)
//   sample  sample_token / sample_argmax / RNG vs sampler.c: same token + RNG state
//   decode  decode_step vs q_llama_decode_step (Qwen int8) and llama_decode_step
//           (SmolLM fp32): logits diff + argmax every step over a greedy run;
//           prefill (batched) KV cache == decode_step KV cache, bit for bit.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <windows.h>

#include "llama.h"
#include "quant.h"
#include "rmsnorm.h"
#include "rope.h"
#include "gqa.h"
#include "sampler.h"

// ---- mirrors of model.inc -----------------------------------------------------
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
_Static_assert(sizeof(Layer) == 512 && sizeof(Model) == 256 && sizeof(KV) == 64, "layout");

// AttnArgs (ops.S)
typedef struct {
  const float *q, *kc, *vc;
  int64_t cap, window, pos0, nrows, n_heads, n_kv, hd;
  float *sc; int64_t scstride; float *o;
  float scale; int32_t pad_; int64_t group, pad2_;
} AttnArgs;
_Static_assert(sizeof(AttnArgs) == 128 && offsetof(AttnArgs, o) == 96 && offsetof(AttnArgs, scale) == 104, "AttnArgs");

// ---- asm under test (explicit symbol names: immune to the -D renames) ----------
void A_rmsnorm(const float *x, const float *g, float *out, int64_t n, float eps) __asm__("rmsnorm");
void A_rmsnorm_rows(const float *x, const float *g, float *out, int64_t n, float eps, int64_t rows) __asm__("rmsnorm_rows");
void A_rmsnorm_heads(float *x, const float *g, int64_t nh, int64_t hd, float eps) __asm__("rmsnorm_heads");
void A_vec_add(float *y, const float *x, int64_t n) __asm__("vec_add");
void A_add_rows(float *y, const float *b, int64_t n, int64_t rows) __asm__("add_rows");
void A_silu_mul(float *g, const float *u, int64_t n) __asm__("silu_mul");
void A_expf_vec(float *d, const float *s, int64_t n) __asm__("expf_vec");
void A_rope_inv_freq(float *inv, int64_t hd, float base) __asm__("rope_inv_freq");
void A_rope_table(float *cs, const float *inv, int64_t half, int64_t pos) __asm__("rope_table");
void A_rope_apply(float *v, int64_t nh, int64_t hd, const float *cs) __asm__("rope_apply");
void A_rope_pos(float *v, int64_t nh, int64_t hd, float base, int64_t pos) __asm__("rope_pos");
void A_attention(AttnArgs *a) __asm__("attention");
void A_kv_put(float *cache, int64_t cap, int64_t slot, const float *row, int64_t kvd) __asm__("kv_put");
int64_t A_decode_init(Model *m) __asm__("decode_init");
int64_t A_decode_step(Model *m, KV *kv, int64_t tok, float *logits) __asm__("decode_step");
int64_t A_prefill(Model *m, KV *kv, const int32_t *ids, int64_t n) __asm__("prefill");
void A_rng_seed(uint64_t *r, uint64_t seed) __asm__("rng_seed");
float A_rng_uniform(uint64_t *r) __asm__("rng_uniform");
int64_t A_sample_argmax(const float *l, int64_t V) __asm__("sample_argmax");
int64_t A_sample_token(const float *l, int64_t V, float T, int64_t top_k, float top_p, uint64_t *rng) __asm__("sample_token");

int64_t cpu_init(void);
int64_t pool_init(int64_t n);
int64_t pool_threads(void);
int64_t model_load(const wchar_t *path, Model *m, int64_t flags);
void model_free(Model *m);
int64_t kv_create_len(const Model *m, KV *kv, int64_t max_len);
void kv_free(KV *kv);

// ---- helpers ------------------------------------------------------------------------
static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { g_fail++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint64_t rs = 0x1234567887654321ULL;
static uint64_t xr(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static float fr(float a, float b) { return a + (b - a) * (float)((xr() >> 40) * (1.0 / 16777216.0)); }
static float gauss(void) {
  float u = fr(1e-7f, 1.0f), v = fr(0.0f, 1.0f);
  return sqrtf(-2.0f * logf(u)) * cosf(6.2831853f * v);
}
static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static double now_s(void) {
  LARGE_INTEGER c, f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f);
  return (double)c.QuadPart / (double)f.QuadPart;
}
static Tensor tview(float *d, int *shape, int r, int c) {
  Tensor t; memset(&t, 0, sizeof t);
  shape[0] = r; shape[1] = c; t.data = d; t.shape = shape; t.ndim = 2; t.size = (size_t)r * c;
  return t;
}
static float *falloc(size_t n) { float *p = (float*)_aligned_malloc(n * sizeof(float) + 64, 64); memset(p, 0, n * 4 + 64); return p; }
static int ulp_diff(float a, float b) {
  int32_t x = (int32_t)fbits(a), y = (int32_t)fbits(b);
  if (x < 0) x = (int32_t)0x80000000 - x;
  if (y < 0) y = (int32_t)0x80000000 - y;
  int64_t d = (int64_t)x - y; return (int)(d < 0 ? -d : d);
}

// ============================================================================
// (a) ops
// ============================================================================
static void test_expf(void) {
  printf("[expf] bit-exact vs C expf (zig compiler_rt)\n");
  const int N = 1 << 16;
  float *x = falloc(N), *y = falloc(N);
  long long tested = 0, bad = 0;
  for (int round = 0; round < 80; round++) {
    for (int i = 0; i < N; i++) {
      if (round < 40) x[i] = fr(-110.0f, 95.0f);
      else if (round < 60) x[i] = fr(-1.2f, 1.2f);
      else { uint32_t u = (uint32_t)xr(); memcpy(&x[i], &u, 4); }   // any bit pattern
    }
    int n = N - (round % 7);                    // exercise the tail
    A_expf_vec(y, x, n);
    for (int i = 0; i < n; i++) {
      float r = expf(x[i]);
      tested++;
      if (fbits(r) != fbits(y[i]) && !(isnan(r) && isnan(y[i]))) {
        if (bad < 5) printf("  mismatch x=%a: C %a asm %a\n", x[i], r, y[i]);
        bad++;
      }
    }
  }
  // special values
  float sp[] = {0.0f, -0.0f, 1e-30f, -1e-30f, 0x1p-13f, -0x1p-13f, 0x1.000002p-13f, 88.7228f, 88.72284f,
                89.0f, -87.3f, -103.97f, -103.98f, -104.0f, INFINITY, -INFINITY, NAN, 0.3465f, 1.0397f, -1.0397f};
  int ns = (int)(sizeof sp / sizeof sp[0]);
  A_expf_vec(y, sp, ns);
  for (int i = 0; i < ns; i++) {
    float r = expf(sp[i]); tested++;
    if (fbits(r) != fbits(y[i]) && !(isnan(r) && isnan(y[i]))) { printf("  special x=%a: C %a asm %a\n", sp[i], r, y[i]); bad++; }
  }
  printf("  %lld values, %lld mismatches\n", tested, bad);
  CHECK(bad == 0, "expf not bit-exact");
  _aligned_free(x); _aligned_free(y);
}

static void test_silu(void) {
  printf("[silu_mul] bit-exact vs C (z*s)*u, s = 1/(1+expf(-z))\n");
  const int N = 10007;
  float *g = falloc(N), *u = falloc(N), *ref = falloc(N);
  long bad = 0;
  for (int r = 0; r < 20; r++) {
    float sc = (r % 4 == 0) ? 40.0f : 4.0f;
    for (int i = 0; i < N; i++) { g[i] = gauss() * sc; u[i] = gauss(); }
    for (int i = 0; i < N; i++) { float z = g[i]; float s = 1.0f / (1.0f + expf(-z)); ref[i] = (z * s) * u[i]; }
    int n = N - r;
    A_silu_mul(g, u, n);
    for (int i = 0; i < n; i++) if (fbits(g[i]) != fbits(ref[i])) bad++;
  }
  printf("  mismatches: %ld\n", bad);
  CHECK(bad == 0, "silu not bit-exact");
  _aligned_free(g); _aligned_free(u); _aligned_free(ref);
}

static void test_rmsnorm(void) {
  printf("[rmsnorm] vs rmsnorm.c (max |diff| / max |y|)\n");
  int ns[] = {896, 576, 64, 37, 4864, 8};
  for (int t = 0; t < 6; t++) {
    int n = ns[t];
    float *x = falloc(n), *g = falloc(n), *yc = falloc(n), *ya = falloc(n);
    for (int i = 0; i < n; i++) { x[i] = gauss() * 3.0f; g[i] = 1.0f + 0.3f * gauss(); }
    int s1[2], s2[2], s3[2];
    Tensor tx = tview(x, s1, 1, n), tg = tview(g, s2, 1, n), ty = tview(yc, s3, 1, n);
    tg.shape[0] = n; tg.ndim = 1;
    c_rmsnorm(&tx, &tg, &ty, 1e-6f);
    A_rmsnorm(x, g, ya, n, 1e-6f);
    int nbad = 0;
    for (int i = 0; i < n; i++) nbad += fbits(yc[i]) != fbits(ya[i]);
    printf("  n=%5d  bit mismatches %d\n", n, nbad);
    CHECK(nbad == 0, "rmsnorm n=%d not bit-exact", n);
    _aligned_free(x); _aligned_free(g); _aligned_free(yc); _aligned_free(ya);
  }
  // per-head (QK-norm) and rows
  {
    const int nh = 14, hd = 64, n = nh * hd;
    float *x = falloc(n), *xa = falloc(n), *g = falloc(hd);
    for (int i = 0; i < n; i++) x[i] = xa[i] = gauss() * 2.0f;
    for (int i = 0; i < hd; i++) g[i] = 1.0f + 0.2f * gauss();
    int s1[2], s2[2];
    Tensor tx = tview(x, s1, 1, n), tg = tview(g, s2, hd, 1);
    tg.ndim = 1; tg.shape[0] = hd;
    c_rmsnorm_heads(&tx, &tg, &tx, 1e-6f);
    A_rmsnorm_heads(xa, g, nh, hd, 1e-6f);
    int nbad = 0;
    for (int i = 0; i < n; i++) nbad += fbits(x[i]) != fbits(xa[i]);
    printf("  heads 14x64 bit mismatches %d\n", nbad);
    CHECK(nbad == 0, "rmsnorm_heads not bit-exact");
    // rows == row by row
    float *r1 = falloc(n), *r2 = falloc(n);
    A_rmsnorm_rows(xa, g, r1, hd, 1e-6f, nh);
    for (int h = 0; h < nh; h++) A_rmsnorm(xa + h * hd, g, r2 + h * hd, hd, 1e-6f);
    CHECK(memcmp(r1, r2, n * 4) == 0, "rmsnorm_rows != rmsnorm per row");
    _aligned_free(x); _aligned_free(xa); _aligned_free(g); _aligned_free(r1); _aligned_free(r2);
  }
}

static void test_rope(void) {
  printf("[rope] vs rope.c rope_heads_pos\n");
  float bases[] = {1000000.0f, 100000.0f, 10000.0f};
  int hds[] = {64, 128, 48};
  long powbad = 0, trig = 0, trigbad = 0, ropebad = 0; int trigmaxu = 0;
  for (int b = 0; b < 3; b++) for (int h = 0; h < 3; h++) {
    int hd = hds[h], half = hd / 2;
    float inv[256], cs[512];
    A_rope_inv_freq(inv, hd, bases[b]);
    for (int k = 0; k < half; k++) {
      float r = powf(bases[b], -2.0f * (float)k / (float)hd);
      if (fbits(r) != fbits(inv[k])) { if (powbad < 5) printf("  powf mismatch base %g k %d: %a vs %a\n", bases[b], k, r, inv[k]); powbad++; }
    }
    int poss[] = {0, 1, 2, 17, 100, 1000, 4095, 8191, 20000, 32767};
    for (int pi = 0; pi < 10; pi++) {
      int pos = poss[pi];
      A_rope_table(cs, inv, half, pos);
      for (int k = 0; k < half; k++) {
        float ang = (float)pos * inv[k];
        float c = cosf(ang), s = sinf(ang);
        trig += 2;
        int u1 = ulp_diff(c, cs[k]), u2 = ulp_diff(s, cs[half + k]);
        if (u1) trigbad++;
        if (u2) trigbad++;
        if (u1 > trigmaxu) trigmaxu = u1;
        if (u2 > trigmaxu) trigmaxu = u2;
      }
      // full rotation of 14 heads vs rope_heads_pos
      const int nh = 14, n = nh * hd;
      float *v = falloc(n), *va = falloc(n);
      for (int i = 0; i < n; i++) v[i] = va[i] = gauss();
      int s1[2]; Tensor tv = tview(v, s1, 1, n);
      rope_heads_pos(&tv, nh, bases[b], pos);
      A_rope_apply(va, nh, hd, cs);
      int nbad = 0;
      for (int i = 0; i < n; i++) nbad += fbits(v[i]) != fbits(va[i]);
      ropebad += nbad;
      CHECK(nbad == 0, "rope hd=%d pos=%d: %d values not bit-exact", hd, pos, nbad);
      // rope_pos (one-shot) == table + apply
      float *vb = falloc(n);
      for (int i = 0; i < n; i++) vb[i] = v[i];
      memcpy(va, vb, n * 4);
      A_rope_pos(vb, nh, hd, bases[b], pos);
      A_rope_apply(va, nh, hd, cs);
      CHECK(memcmp(va, vb, n * 4) == 0, "rope_pos != table+apply");
      _aligned_free(v); _aligned_free(va); _aligned_free(vb);
    }
  }
  printf("  powf mismatches %ld ; sinf/cosf mismatches %ld of %ld ; rotated values not bit-exact %ld\n",
         powbad, trigbad, trig, ropebad);
  CHECK(powbad == 0, "powf");
  CHECK(trigbad == 0, "sinf/cosf not bit-exact (max %d ulp)", trigmaxu);
  // the trig ports over a dense sweep of angles
  {
    long bad = 0, tot = 0; float inv1[1] = {1.0f}, cs2[2];
    for (uint32_t u = 0x30000000u; u < 0x47000000u; u += 4099) {   // 4.7e-10 .. 32768
      float a; memcpy(&a, &u, 4);
      A_rope_table(cs2, inv1, 1, 0);           // (pos 0 path)
      // table for "pos" = a via inv = a, pos = 1
      float inva[1] = {a};
      A_rope_table(cs2, inva, 1, 1);
      tot += 2;
      bad += fbits(cs2[0]) != fbits(cosf(a));
      bad += fbits(cs2[1]) != fbits(sinf(a));
    }
    printf("  sinf/cosf sweep: %ld of %ld differ\n", bad, tot);
    CHECK(bad == 0, "trig sweep");
  }
}

// llama.c decode_attention, verbatim (it is static there)
static void ref_attention(const float *q, const float *kc, const float *vc,
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

static void test_attention(void) {
  printf("[attention] vs llama.c decode_attention (max |diff| / max |o|)\n");
  struct { int nh, nkv, hd, cap, window, pos, nrows; } cs[] = {
    {14, 2, 64, 512, 0, 0, 1}, {14, 2, 64, 512, 0, 1, 1}, {14, 2, 64, 512, 0, 7, 1},
    {14, 2, 64, 512, 0, 8, 1}, {14, 2, 64, 512, 0, 9, 1}, {14, 2, 64, 512, 0, 100, 1},
    {14, 2, 64, 512, 0, 511, 1}, {9, 3, 64, 4096, 0, 3000, 1}, {14, 2, 64, 4096, 0, 4095, 1},
    {14, 2, 64, 100, 100, 350, 1}, {14, 2, 64, 37, 37, 1000, 1}, {14, 2, 64, 512, 60, 300, 1},
    {8, 8, 128, 300, 0, 299, 1}, {4, 2, 32, 64, 0, 50, 1}, {4, 1, 16, 64, 20, 63, 1},
    {14, 2, 64, 512, 0, 37, 5}, {9, 3, 64, 1024, 0, 900, 64}, {14, 2, 64, 512, 0, 0, 64},
    {16, 4, 128, 256, 0, 100, 7}, {14, 2, 64, 64, 64, 1000, 1}, {14, 2, 64, 16, 16, 333, 1},
    {9, 3, 64, 32, 20, 101, 9}, {14, 2, 64, 24, 24, 200, 1},
  };
  int nc = (int)(sizeof cs / sizeof cs[0]);
  for (int c = 0; c < nc; c++) {
    int nh = cs[c].nh, nkv = cs[c].nkv, hd = cs[c].hd, cap = cs[c].cap, W = cs[c].window;
    int pos0 = cs[c].pos, nr = cs[c].nrows, qd = nh * hd, kvd = nkv * hd;
    float *q = falloc((size_t)nr * qd), *kc = falloc((size_t)cap * kvd), *vc = falloc((size_t)cap * kvd);
    float *o = falloc((size_t)nr * qd), *oref = falloc((size_t)nr * qd);
    int maxlen = pos0 + nr + 16;
    float *scr = falloc((size_t)maxlen);
    int64_t stride = ((pos0 + nr + 14) & ~7) + 8;
    float *sc = falloc((size_t)nr * nh * stride);
    for (size_t i = 0; i < (size_t)nr * qd; i++) q[i] = gauss();
    for (size_t i = 0; i < (size_t)cap * kvd; i++) { kc[i] = gauss(); vc[i] = gauss(); }
    float *kca = falloc((size_t)cap * kvd);                 // asm layout (blocked if cap % 8 == 0)
    for (int sl = 0; sl < cap; sl++) A_kv_put(kca, cap, sl, kc + (size_t)sl * kvd, kvd);
    for (int r = 0; r < nr; r++)
      ref_attention(q + (size_t)r * qd, kc, vc, cap, W, pos0 + r, nh, nkv, hd, scr, oref + (size_t)r * qd);
    AttnArgs a; memset(&a, 0, sizeof a);
    a.q = q; a.kc = kca; a.vc = vc; a.cap = cap; a.window = W; a.pos0 = pos0; a.nrows = nr;
    a.n_heads = nh; a.n_kv = nkv; a.hd = hd; a.sc = sc; a.scstride = stride; a.o = o;
    A_attention(&a);
    double md = 0, mo = 0;
    for (size_t i = 0; i < (size_t)nr * qd; i++) { md = fmax(md, fabs((double)o[i] - oref[i])); mo = fmax(mo, fabs(oref[i])); }
    printf("  nh=%2d kv=%d hd=%3d cap=%4d W=%3d pos=%4d rows=%2d : max|diff| %.2e (max|o| %.2f)\n", nh, nkv, hd, cap, W, pos0, nr, md, mo);
    CHECK(md == 0, "attention case %d not bit-exact", c);
    _aligned_free(kca);
    _aligned_free(q); _aligned_free(kc); _aligned_free(vc); _aligned_free(o); _aligned_free(oref);
    _aligned_free(scr); _aligned_free(sc);
  }
}

// ============================================================================
// (b) sampler
// ============================================================================
static void test_sampler(void) {
  printf("[sampler] sample_token / argmax / rng vs sampler.c\n");
  int Vs[] = {151936, 49152, 1000, 10, 3};
  float temps[] = {0.0f, -1.0f, 0.3f, 0.7f, 1.0f, 1.5f};
  int ks[] = {0, 1, 2, 40, 100, 50000, 200000};
  float ps[] = {0.0f, 0.5f, 0.9f, 0.95f, 0.999f, 1.0f};
  long trials = 0, bad = 0, badstate = 0;
  double tc = 0, ta = 0;
  float *l = falloc(151936 + 16);
  for (int it = 0; it < 600; it++) {
    int V = Vs[it % 5];
    int kind = (it / 5) % 4;
    for (int j = 0; j < V; j++) {
      float g = gauss();
      switch (kind) {
        case 0: l[j] = g * 2.0f; break;                          // smooth
        case 1: l[j] = g * 6.0f + 3.0f; break;                   // peaked
        case 2: l[j] = roundf(g * 4.0f) * 0.5f; break;           // many exact ties
        default: l[j] = (j % 97 == 0) ? 20.0f + g : g; break;    // a few dominant tokens
      }
    }
    float T = temps[xr() % 6];
    int K = ks[xr() % 7];
    float P = ps[xr() % 6];
    uint64_t seed = xr() % 5 == 0 ? 0 : xr();
    Rng cr = rng_create(seed);
    uint64_t ar; A_rng_seed(&ar, seed);
    CHECK(ar == cr.state, "rng_seed");
    int sh[2]; Tensor t = tview(l, sh, 1, V);
    for (int d = 0; d < 3; d++) {
      double t0 = now_s();
      int c = c_sample_token(&t, 0, T, K, P, &cr);
      double t1 = now_s();
      int64_t a = A_sample_token(l, V, T, K, P, &ar);
      double t2 = now_s();
      if (V == 151936) { tc += t1 - t0; ta += t2 - t1; }
      trials++;
      if (c != a) { if (bad < 8) printf("  token mismatch V=%d kind=%d T=%g k=%d p=%g: C %d asm %lld\n", V, kind, T, K, P, c, (long long)a); bad++; }
      if (cr.state != ar) badstate++;
    }
    int c = c_sample_argmax(&t, 0); int64_t a = A_sample_argmax(l, V);
    if (c != a) { bad++; printf("  argmax mismatch V=%d: %d vs %lld\n", V, c, (long long)a); }
  }
  // rng_uniform stream
  Rng cr = rng_create(42); uint64_t ar; A_rng_seed(&ar, 42);
  for (int i = 0; i < 100000; i++) {
    float x = c_rng_uniform(&cr), y = A_rng_uniform(&ar);
    if (fbits(x) != fbits(y)) { bad++; break; }
  }
  printf("  %ld draws: %ld token mismatches, %ld rng-state mismatches\n", trials, bad, badstate);
  printf("  time per draw at V=151936: C %.3f ms, asm %.3f ms\n", tc * 1e3 / (trials / 5.0), ta * 1e3 / (trials / 5.0));
  CHECK(bad == 0 && badstate == 0, "sampler");
  _aligned_free(l);
}

// ============================================================================
// (c) full decode vs the C engine
// ============================================================================
static LlamaKV make_ckv(int n_layers, int max_len, int kvd, const int *windows) {
  LlamaKV kv; kv.n_layers = n_layers; kv.max_seq = max_len; kv.kv_dim = kvd; kv.len = 0;
  kv.cap = (int*)malloc(n_layers * sizeof(int));
  kv.k = (float**)malloc(n_layers * sizeof(float*)); kv.v = (float**)malloc(n_layers * sizeof(float*));
  for (int l = 0; l < n_layers; l++) {
    kv.cap[l] = (windows[l] > 0 && windows[l] < max_len) ? windows[l] : max_len;
    kv.k[l] = (float*)calloc((size_t)kv.cap[l] * kvd, 4); kv.v[l] = (float*)calloc((size_t)kv.cap[l] * kvd, 4);
  }
  return kv;
}
static void free_ckv(LlamaKV *kv) { llama_kv_free(kv); }

static int argmax_f(const float *l, int V) { int b = 0; for (int j = 1; j < V; j++) if (l[j] > l[b]) b = j; return b; }

static const wchar_t *wpath(const char *p) {
  static wchar_t w[512]; int i = 0; for (; p[i] && i < 511; i++) w[i] = (wchar_t)p[i]; w[i] = 0; return w;
}

// K[slot][c] in the asm cache layout (ops.S: blocked when cap % 8 == 0)
static float kget(const KV *kv, int l, int64_t slot, int c) {
  int64_t cap = kv->cap[l], d = kv->kv_dim;
  if (cap % 8) return kv->k[l][slot * d + c];
  return kv->k[l][(slot >> 3) * 8 * d + (int64_t)c * 8 + (slot & 7)];
}
// compare positions < n of two asm caches bit for bit (K through the layout)
static int kv_equal(const KV *a, const KV *b, int n) {
  for (int l = 0; l < a->n_layers; l++) {
    int used = n < a->cap[l] ? n : (int)a->cap[l];
    for (int s = 0; s < used; s++)
      for (int c = 0; c < a->kv_dim; c++)
        if (fbits(kget(a, l, s, c)) != fbits(kget(b, l, s, c))) return 0;
    if (memcmp(a->v[l], b->v[l], (size_t)used * a->kv_dim * 4)) return 0;
  }
  return 1;
}

static void test_decode(const char *path, int q8) {
  printf("[decode] %s (%s)\n", path, q8 ? "int8, vs q_llama_decode_step" : "fp32, vs llama_decode_step");
  QLlamaModel cq; LlamaModel cf; int V, L, kvd; int *win;
  if (q8) { if (q_llama_model_load(path, &cq) != 0) { printf("  (C load failed, skipped)\n"); return; } V = cq.vocab; L = cq.n_layers; kvd = cq.n_kv * cq.head_dim; }
  else    { if (llama_model_load(path, &cf) != 0) { printf("  (C load failed, skipped)\n"); return; } V = cf.vocab; L = cf.n_layers; kvd = cf.n_kv * cf.head_dim; }
  win = (int*)malloc(L * sizeof(int));
  for (int l = 0; l < L; l++) win[l] = q8 ? cq.blocks[l].window : cf.blocks[l].window;
  Model m; memset(&m, 0, sizeof m);
  if (model_load(wpath(path), &m, 0) != 0) { printf("  model_load failed\n"); g_fail++; return; }
  const int MAXLEN = 512;
  LlamaKV ckv = make_ckv(L, MAXLEN, kvd, win);
  KV kv, kv2; kv_create_len(&m, &kv, MAXLEN); kv_create_len(&m, &kv2, MAXLEN);
  int sh[2] = {1, V};
  Tensor cl = tensor_create_nograd(sh, 2);
  float *al = falloc(V);
  // prompt: 12 fixed pseudo-random ids, then 40 greedy steps (both fed C's argmax)
  int toks[64]; uint64_t s = 777;
  for (int i = 0; i < 12; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; toks[i] = (int)(s % (uint64_t)(V < 30000 ? V : 30000)); }
  const int NP = 12, NSTEP = 40;
  double maxd = 0, maxl = 0; int amis = 0; double tc = 0, ta = 0;
  for (int step = 0; step < NP + NSTEP - 1; step++) {
    int tok = toks[step];
    double t0 = now_s();
    if (q8) q_llama_decode_step(&cq, &ckv, tok, &cl); else llama_decode_step(&cf, &ckv, tok, &cl);
    double t1 = now_s();
    A_decode_step(&m, &kv, tok, al);
    double t2 = now_s();
    if (step >= NP) { tc += t1 - t0; ta += t2 - t1; }
    double d = 0, mx = 0;
    for (int j = 0; j < V; j++) { d = fmax(d, fabs((double)cl.data[j] - al[j])); mx = fmax(mx, fabs(cl.data[j])); }
    int ca = argmax_f(cl.data, V), aa = argmax_f(al, V);
    if (ca != aa) amis++;
    if (getenv("DEC_VERBOSE")) printf("    step %2d tok %6d diff %.4g argmax C %d asm %d\n", step, tok, d, ca, aa);
    if (d > maxd) maxd = d;
    if (mx > maxl) maxl = mx;
    if (step + 1 >= NP) toks[step + 1] = ca;          // greedy continuation (C's choice)
  }
  printf("  %d steps: max |logit diff| %.3g (max |logit| %.3g), argmax mismatches %d\n", NP + NSTEP - 1, maxd, maxl, amis);
  printf("  per token (incl. head): C %.1f ms, asm %.1f ms\n", tc * 1e3 / (NSTEP - 1), ta * 1e3 / (NSTEP - 1));
  CHECK(amis == 0, "argmax differs");
  CHECK(maxd < (q8 ? 2e-2 : 5e-3) * (maxl > 1 ? maxl : 1), "logits drift");
  CHECK(kv.len == NP + NSTEP - 1, "kv len");

  // prefill == decode_step, bit for bit: prefill the first n tokens of the same
  // stream (n chosen to cross a batch boundary), then compare the caches and the
  // next decode_step's logits.
  int n = NP + NSTEP - 2;   // all but the last token fed above
  int32_t ids[64]; for (int i = 0; i < n; i++) ids[i] = toks[i];
  kv2.len = 0;
  int64_t rc = A_prefill(&m, &kv2, ids, n);
  CHECK(rc == 0 && kv2.len == n, "prefill rc/len");
  // reference: a fresh decode_step(.., 0) run
  KV kv3; kv_create_len(&m, &kv3, MAXLEN);
  for (int i = 0; i < n; i++) A_decode_step(&m, &kv3, ids[i], NULL);
  int eq = kv_equal(&kv2, &kv3, n);
  printf("  prefill(%d) KV cache == decode_step KV cache: %s\n", n, eq ? "bit-exact" : "DIFFERENT");
  CHECK(eq, "prefill KV != decode KV");
  // and with logits on (decode_step with logits writes the same cache)
  int eq2 = kv_equal(&kv3, &kv, n);
  printf("  decode_step(logits=0) cache == decode_step(logits) cache: %s\n", eq2 ? "bit-exact" : "DIFFERENT");
  CHECK(eq2, "logits=0 cache differs");
  float *l2 = falloc(V), *l3 = falloc(V);
  A_decode_step(&m, &kv2, toks[n], l2); A_decode_step(&m, &kv3, toks[n], l3);
  CHECK(memcmp(l2, l3, V * 4) == 0, "logits after prefill differ");

  // windowed ring: run a model copy with a sliding window of 7 on every layer
  // through prefill (row-wise path) and decode_step, compare with C.
  {
    for (int l = 0; l < L; l++) { m.layers[l].window = 7; if (q8) cq.blocks[l].window = 7; else cf.blocks[l].window = 7; }
    int w7[256]; for (int l = 0; l < L; l++) w7[l] = 7;
    LlamaKV ck = make_ckv(L, MAXLEN, kvd, w7);
    KV ka, kb; kv_create_len(&m, &ka, MAXLEN); kv_create_len(&m, &kb, MAXLEN);
    int okcap = 1; for (int l = 0; l < L; l++) okcap &= (ka.cap[l] == 7);
    int nn = 30;
    for (int i = 0; i < nn; i++) { if (q8) q_llama_decode_step(&cq, &ck, ids[i], &cl); else llama_decode_step(&cf, &ck, ids[i], &cl); }
    A_prefill(&m, &kb, ids, nn - 1);
    A_decode_step(&m, &kb, ids[nn - 1], l2);
    for (int i = 0; i < nn - 1; i++) A_decode_step(&m, &ka, ids[i], NULL);
    A_decode_step(&m, &ka, ids[nn - 1], l3);
    double d = 0; for (int j = 0; j < V; j++) d = fmax(d, fabs((double)cl.data[j] - l2[j]));
    printf("  window=7 (ring cap %s): prefill==decode %s, |diff vs C| %.3g, argmax %s\n", okcap ? "7" : "?",
           memcmp(l2, l3, V * 4) == 0 ? "bit-exact" : "DIFFERENT", d, argmax_f(cl.data, V) == argmax_f(l2, V) ? "same" : "DIFFERENT");
    CHECK(memcmp(l2, l3, V * 4) == 0, "windowed prefill != decode");
    CHECK(argmax_f(cl.data, V) == argmax_f(l2, V), "windowed argmax");
    for (int l = 0; l < L; l++) { m.layers[l].window = win[l]; if (q8) cq.blocks[l].window = win[l]; else cf.blocks[l].window = win[l]; }
    kv_free(&ka); kv_free(&kb); free_ckv(&ck);
  }
  _aligned_free(l2); _aligned_free(l3);
  kv_free(&kv3);
  kv_free(&kv); kv_free(&kv2); free_ckv(&ckv);
  tensor_free(&cl); _aligned_free(al);
  model_free(&m);
  if (q8) q_llama_model_free(&cq); else llama_model_free(&cf);
  free(win);
}

// appended debug: layer-truncated comparison
static void dbg_layers(void) {
  const char *path = "../model/qwen/model_int8.bin";
  QLlamaModel cq; q_llama_model_load(path, &cq);
  Model m; memset(&m, 0, sizeof m); model_load(wpath(path), &m, 0);
  int V = cq.vocab, kvd = cq.n_kv * cq.head_dim, L0 = cq.n_layers;
  int sh[2] = {1, V}; Tensor cl = tensor_create_nograd(sh, 2); float *al = falloc(V), *al2 = falloc(V);
  int toks[4] = {1983, 19261, 19255, 28365};
  for (int nl = 1; nl <= 3; nl++) {
    cq.n_layers = nl; m.n_layers = nl;
    int w[64] = {0};
    LlamaKV ck = make_ckv(nl, 64, kvd, w);
    KV ka, kb; kv_create_len(&m, &ka, 64); kv_create_len(&m, &kb, 64);
    for (int s = 0; s < 4; s++) {
      q_llama_decode_step(&cq, &ck, toks[s], &cl);
      A_decode_step(&m, &ka, toks[s], al);
      A_decode_step(&m, &kb, toks[s], al2);
      double d = 0, d2 = 0; for (int j = 0; j < V; j++) { d = fmax(d, fabs((double)cl.data[j] - al[j])); d2 = fmax(d2, fabs((double)al2[j] - al[j])); }
      double kd = 0; for (int l = 0; l < nl; l++) for (int i = 0; i < kvd; i++) { kd = fmax(kd, fabs((double)ck.k[l][s*kvd+i] - ka.k[l][s*kvd+i])); kd = fmax(kd, fabs((double)ck.v[l][s*kvd+i] - ka.v[l][s*kvd+i])); }
      printf("  layers %d step %d: logit diff %.4g  (asm run-to-run %.3g)  K/V diff %.3g\n", nl, s, d, d2, kd);
    }
    kv_free(&ka); kv_free(&kb); free_ckv(&ck);
  }
  cq.n_layers = L0;
}

static void dbg_prefill(void) {
  const char *path = getenv("DEC_MODEL") ? getenv("DEC_MODEL") : "../model/smollm/model_int8.bin";
  Model m; memset(&m, 0, sizeof m); model_load(wpath(path), &m, 0);
  KV kv; kv_create_len(&m, &kv, 2048);
  int32_t ids[1024]; for (int i = 0; i < 1024; i++) ids[i] = (i * 7919 + 13) % 30000;
  int ns[] = {63, 64, 65, 100, 128, 129, 200, 512};
  for (int t = 0; t < 8; t++) {
    kv.len = 0; printf("  prefill %d ...", ns[t]);
    double t0 = now_s(); int64_t rc = A_prefill(&m, &kv, ids, ns[t]); printf(" rc %lld len %lld %.3f s\n", (long long)rc, (long long)kv.len, now_s() - t0);
  }
}

void matvec(const Mat *m, const float *x, float *y);
static void prof(void) {
  const char *path = getenv("DEC_MODEL") ? getenv("DEC_MODEL") : "../model/qwen/model_int8.bin";
  Model m; memset(&m, 0, sizeof m); model_load(wpath(path), &m, 0);
  int maxlen = 4200; KV kv; kv_create_len(&m, &kv, maxlen);
  int V = (int)m.vocab; float *l = falloc(V);
  int32_t *ids = (int32_t*)malloc(maxlen * 4); for (int i = 0; i < maxlen; i++) ids[i] = (i * 7919 + 13) % 30000;
  float *x = falloc(8192), *y = falloc(V + 64);
  for (int i = 0; i < 8192; i++) x[i] = gauss();
  int ctxs[] = {16, 1024, 4096};
  printf("[profile] %s\n", path);
  for (int c = 0; c < 3; c++) {
    kv.len = 0; A_prefill(&m, &kv, ids, ctxs[c]);
    const int NS = 20; double best_full = 1e9, best_mv = 1e9, best_at = 1e9;
    for (int rep = 0; rep < 3; rep++) {
      int64_t len0 = kv.len;
      double t0 = now_s();
      for (int s = 0; s < NS; s++) A_decode_step(&m, &kv, ids[s], l);
      double t1 = now_s(); kv.len = len0;
      for (int s = 0; s < NS; s++) {
        for (int L = 0; L < m.n_layers; L++) {
          Layer *ly = &m.layers[L];
          matvec(&ly->Wq, x, y); matvec(&ly->Wk, x, y); matvec(&ly->Wv, x, y); matvec(&ly->Wo, x, y);
          matvec(&ly->Wgate, x, y); matvec(&ly->Wup, x, y); matvec(&ly->Wdown, x, y);
        }
        matvec(&m.lm_head, x, y);
      }
      double t2 = now_s();
      // attention alone at this context, all layers
      AttnArgs a; memset(&a, 0, sizeof a);
      int64_t qd = m.n_heads * m.head_dim; float *q = falloc(qd), *o = falloc(qd);
      for (int i = 0; i < qd; i++) q[i] = gauss() * 0.1f;
      int64_t stride = ((len0 + 14) & ~7) + 8; float *sc = falloc(m.n_heads * stride);
      double t3 = now_s();
      for (int s = 0; s < NS; s++) for (int L = 0; L < m.n_layers; L++) {
        a.q = q; a.kc = kv.k[L]; a.vc = kv.v[L]; a.cap = kv.cap[L]; a.window = 0; a.pos0 = len0 - 1; a.nrows = 1;
        a.n_heads = m.n_heads; a.n_kv = m.n_kv; a.hd = m.head_dim; a.sc = sc; a.scstride = stride; a.o = o;
        A_attention(&a);
      }
      double t4 = now_s();
      _aligned_free(q); _aligned_free(o); _aligned_free(sc);
      best_full = fmin(best_full, (t1 - t0) / NS); best_mv = fmin(best_mv, (t2 - t1) / NS); best_at = fmin(best_at, (t4 - t3) / NS);
    }
    printf("  ctx %5d: decode_step %6.2f ms | matvecs alone %6.2f ms | attention alone %5.2f ms | rest %5.2f ms\n",
           ctxs[c], best_full * 1e3, best_mv * 1e3, best_at * 1e3, (best_full - best_mv - best_at) * 1e3);
  }
  kv_free(&kv); model_free(&m);
}

// ============================================================================
// benchmarks
// ============================================================================
static void bench(const char *path) {
  printf("[bench] %s  (%lld threads)\n", path, (long long)pool_threads());
  Model m; memset(&m, 0, sizeof m);
  if (model_load(wpath(path), &m, 0) != 0) { printf("  model_load failed\n"); return; }
  int ctxs[] = {16, 256, 1024, 4096};
  int maxlen = 4096 + 64;
  KV kv; kv_create_len(&m, &kv, maxlen);
  int V = (int)m.vocab;
  float *l = falloc(V);
  int32_t *ids = (int32_t*)malloc(maxlen * 4);
  for (int i = 0; i < maxlen; i++) ids[i] = (int32_t)((i * 7919u + 13) % (unsigned)(V < 30000 ? V : 30000));
  A_decode_init(&m);
  // prefill throughput
  int np[] = {64, 512};
  for (int t = 0; t < 2; t++) {
    kv.len = 0;
    double t0 = now_s(); A_prefill(&m, &kv, ids, np[t]); double t1 = now_s();
    kv.len = 0;
    double t2 = now_s(); for (int i = 0; i < np[t]; i++) A_decode_step(&m, &kv, ids[i], NULL); double t3 = now_s();
    printf("  prefill %4d tokens: batched %7.1f tok/s   (token-by-token decode_step: %6.1f tok/s)\n",
           np[t], np[t] / (t1 - t0), np[t] / (t3 - t2));
  }
  for (int c = 0; c < 4; c++) {
    kv.len = 0;
    A_prefill(&m, &kv, ids, ctxs[c]);
    const int NS = 32;
    A_decode_step(&m, &kv, ids[ctxs[c]], l);           // warm-up
    double t0 = now_s();
    for (int s = 0; s < NS; s++) A_decode_step(&m, &kv, ids[(ctxs[c] + s + 1) % maxlen], l);
    double t1 = now_s();
    printf("  decode at ctx %5d: %6.2f ms/token = %6.1f tok/s\n", ctxs[c], (t1 - t0) * 1e3 / NS, NS / (t1 - t0));
  }
  // sampler cost at this vocab
  uint64_t r; A_rng_seed(&r, 1);
  double t0 = now_s(); for (int i = 0; i < 50; i++) A_sample_token(l, V, 0.7f, 40, 0.9f, &r); double t1 = now_s();
  double t2 = now_s(); for (int i = 0; i < 50; i++) A_sample_token(l, V, 0.8f, 0, 0.95f, &r); double t3 = now_s();
  double t4 = now_s(); for (int i = 0; i < 50; i++) A_sample_argmax(l, V); double t5 = now_s();
  printf("  sample_token: T=0.7 k=40 p=0.9 %.3f ms, T=0.8 p=0.95 %.3f ms, argmax %.3f ms\n",
         (t1 - t0) * 20, (t3 - t2) * 20, (t5 - t4) * 20);
  kv_free(&kv); model_free(&m); free(ids); _aligned_free(l);
}

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  int64_t f = cpu_init();
  int64_t nt = pool_init(0);
  printf("cpu features 0x%llx, pool threads %lld\n", (long long)f, (long long)nt);
  const char *only = getenv("DEC_ONLY");
  int all = !only;
  #define RUN(name) (all || strcmp(only, name) == 0)
  if (RUN("ops")) { test_expf(); test_silu(); test_rmsnorm(); test_rope(); test_attention(); }
  if (only && strcmp(only, "dbg") == 0) dbg_layers();
  if (only && strcmp(only, "dbgp") == 0) dbg_prefill();
  if (only && strcmp(only, "prof") == 0) prof();
  if (RUN("sample")) test_sampler();
  if (RUN("decode")) {
    test_decode("../model/qwen/model_int8.bin", 1);
    test_decode("../model/smollm/model_int8.bin", 1);
    test_decode("../model/smollm/model_v2.bin", 0);
  }
  if ((only && strcmp(only, "bench") == 0) || getenv("DEC_BENCH")) {
    bench("../model/qwen/model_int8.bin");
    bench("../model/smollm/model_v2.bin");
    bench("../model/smollm/model_int8.bin");
  }
  printf(g_fail ? "\nFAILED: %d check(s)\n" : "\nALL PASSED\n", g_fail);
  return g_fail != 0;
}
