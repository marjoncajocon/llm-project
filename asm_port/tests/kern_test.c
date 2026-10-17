// kern_test.c - correctness tests + benchmarks for cpu.S / pool.S / matvec.S.
//
// Build & run (from asm_port/):
//   ASM="cpu pool matvec" ./build.sh test kern_test ../quant.c tests/kern_stream.S
//                                       correctness (bit-exactness etc.)
//   ./build/kern_test.exe bench         benchmarks (bandwidth, decode, prefill)
//   ./build/kern_test.exe tune          decode threads x chunks sweep (KT_MODEL=smol, KT_F32=1)
//   ./build/kern_test.exe threads       Qwen decode vs pool size 1..16
//   ./build/kern_test.exe stress [N]    N random pool_run/pool_run_n calls (run several
//                                       instances at once to oversubscribe the CPU)
//   ./build/kern_test.exe prefill       prefill vs thread count
//   KT_WATCHDOG=s  dump every thread's RIP + pool state after s seconds (hang hunting)
//
// (a) Q8 matvec / matmul_rows are BIT-EXACT vs qmatmul() from ../quant.c
//     (VNNI and AVX2 paths, 1 and N threads, odd shapes), and mat_q8_from_f32 is
//     bit-exact vs quantize_int8_percol().
// (b) F32 / F32T vs a double-precision reference (tolerance).
// (c) matmul_rows row i == matvec of row i, bitwise, for all kinds.
// (d) benchmarks (see bench()).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <windows.h>
#include "quant.h"

typedef struct {
  int64_t kind, K, N;
  void *data;
  float *scale;
  void *packed;
  int32_t *colsum;
  int64_t aux;
} Mat;

extern int64_t cpu_init(void);
extern int64_t cpu_features;
extern int64_t pool_init(int64_t n);
extern void pool_run(void (*fn)(void *, int64_t, int64_t), void *ctx, int64_t n, int64_t grain);
extern int64_t pool_threads(void);
extern void pool_run_n(void (*fn)(void *, int64_t, int64_t), void *ctx, int64_t n, int64_t grain, int64_t nthreads);
extern int64_t pool_mem_threads;
extern void *mat_q8_pack(Mat *m);
extern void *mat_q8_from_f32(Mat *out, const float *w, int64_t K, int64_t N);
extern void matvec(const Mat *m, const float *x, float *y);
extern void matmul_rows(const Mat *m, const float *X, int64_t M, float *Y);
extern int64_t mv_small_bytes, mv_chunks_per_thread, mv_min_chunk_bytes, mv_mem_threads, mv_mem_rows;
extern uint64_t stream_read(const void *p, int64_t bytes);

// qmatmul's integer core (matrix.c is not linked; integer sums are exact in
// any order, so this simple loop is a faithful stand-in).
void qacc_columns(const signed char *qa, const signed char *qw, int K, int N, int32_t *acc) {
  for (int j = 0; j < N; j++) acc[j] = 0;
  for (int k = 0; k < K; k++) {
    int av = qa[k];
    if (!av) continue;
    const signed char *r = qw + (size_t)k * N;
    for (int j = 0; j < N; j++) acc[j] += av * r[j];
  }
}

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static float frand(void) { return (float)((xr() >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; }
static float fnormal(void) { float s = 0; for (int i = 0; i < 4; i++) s += frand(); return s * 0.8f; }

static int fails = 0, checks = 0;
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static double now_s(void);
static void *xalloc(size_t n) { void *p = _aligned_malloc(n ? n : 64, 64); if (!p) { printf("OOM\n"); exit(1); } return p; }

static void fill_x(float *x, int64_t n, int mode) {
  for (int64_t i = 0; i < n; i++) x[i] = fnormal();
  if (mode == 1) for (int64_t i = 0; i < n; i++) x[i] = 0.0f;          // all zero -> as = 1
  if (mode == 2 && n > 3) { x[n / 2] = 37.5f; x[n - 1] = -41.0f; }     // outliers
  if (mode == 3) for (int64_t i = 0; i < n; i++) x[i] = (float)((int)(xr() % 255) - 127) * 0.5f; // ties
}

// C reference: qmatmul over M rows
static void ref_q8(const signed char *q, const float *scale, int K, int N, const float *X, int M, float *Y) {
  int shapeA[2] = {M, K};
  Tensor A = {0}; A.data = (float *)X; A.shape = shapeA; A.ndim = 2; A.size = (size_t)M * K;
  Tensor C = {0}; C.data = Y;
  QTensor W = {K, N, (signed char *)q, (float *)scale};
  qmatmul(&A, &W, &C);
}

static void make_q8(Mat *m, int64_t K, int64_t N) {
  memset(m, 0, sizeof *m);
  m->kind = 2; m->K = K; m->N = N;
  signed char *q = xalloc((size_t)K * N);
  float *s = xalloc((size_t)N * 4);
  for (int64_t i = 0; i < K * N; i++) q[i] = (signed char)((int)(xr() % 255) - 127);
  for (int64_t j = 0; j < N; j++) s[j] = 0.001f + 0.02f * (frand() + 1.0f);
  m->data = q; m->scale = s;
  if (!mat_q8_pack(m)) { printf("pack failed\n"); exit(1); }
}
static void free_packed(Mat *m) { if (m->packed) VirtualFree(m->packed, 0, MEM_RELEASE); m->packed = 0; }

// ---------------------------------------------------------------------------
static void test_q8_shape(int64_t K, int64_t N) {
  Mat m; make_q8(&m, K, N);
  float *x = xalloc(K * 4), *y = xalloc(N * 4 + 64), *yr = xalloc(N * 4 + 64);
  int64_t feats = cpu_features;
  for (int path = 0; path < 2; path++) {
    if (path == 0 && !(feats & 2)) continue;
    cpu_features = path == 0 ? feats : (feats & ~2);
    for (int mode = 0; mode < 4; mode++) {
      fill_x(x, K, mode);
      ref_q8(m.data, m.scale, (int)K, (int)N, x, 1, yr);
      y[N] = 12345.0f;
      matvec(&m, x, y);
      CHECK(memcmp(y, yr, N * 4) == 0, "q8 matvec %lldx%lld %s mode %d not bit-exact",
            (long long)K, (long long)N, path ? "avx2" : "vnni", mode);
      CHECK(y[N] == 12345.0f, "q8 matvec %lldx%lld wrote past y[N]", (long long)K, (long long)N);
      if (memcmp(y, yr, N * 4)) {
        for (int64_t j = 0; j < N; j++) if (y[j] != yr[j]) { printf("   first diff j=%lld got %.9g want %.9g\n", (long long)j, y[j], yr[j]); break; }
      }
    }
  }
  cpu_features = feats;
  free_packed(&m); _aligned_free(m.data); _aligned_free(m.scale);
  _aligned_free(x); _aligned_free(y); _aligned_free(yr);
}

static void test_q8_rows(int64_t K, int64_t N, int64_t M) {
  Mat m; make_q8(&m, K, N);
  float *X = xalloc(M * K * 4), *Y = xalloc(M * N * 4), *Yr = xalloc(M * N * 4), *y1 = xalloc(N * 4);
  for (int64_t i = 0; i < M; i++) fill_x(X + i * K, K, (int)(i % 4));
  ref_q8(m.data, m.scale, (int)K, (int)N, X, (int)M, Yr);
  int64_t feats = cpu_features;
  for (int path = 0; path < 2; path++) {
    if (path == 0 && !(feats & 2)) continue;
    cpu_features = path == 0 ? feats : (feats & ~2);
    matmul_rows(&m, X, M, Y);
    CHECK(memcmp(Y, Yr, M * N * 4) == 0, "q8 matmul_rows %lldx%lld M=%lld %s not bit-exact vs qmatmul",
          (long long)K, (long long)N, (long long)M, path ? "avx2" : "vnni");
    for (int64_t i = 0; i < M; i++) {
      matvec(&m, X + i * K, y1);
      CHECK(memcmp(y1, Y + i * N, N * 4) == 0, "q8 rows!=matvec %lldx%lld row %lld", (long long)K, (long long)N, (long long)i);
    }
  }
  cpu_features = feats;
  free_packed(&m); _aligned_free(m.data); _aligned_free(m.scale);
  _aligned_free(X); _aligned_free(Y); _aligned_free(Yr); _aligned_free(y1);
}

static void test_quantize(int64_t K, int64_t N) {
  float *w = xalloc(K * N * 4);
  for (int64_t i = 0; i < K * N; i++) w[i] = fnormal() * 0.05f;
  if (N > 2) for (int64_t k = 0; k < K; k++) w[k * N + 1] = 0.0f;              // an all-zero column
  if (K > 3) w[2 * N] = 0.75f;                                                  // outlier
  signed char *q = xalloc(K * N); float *s = xalloc(N * 4);
  quantize_int8_percol(w, (int)K, (int)N, q, s);
  Mat m; memset(&m, 0, sizeof m);
  CHECK(mat_q8_from_f32(&m, w, K, N) != 0, "mat_q8_from_f32 failed");
  CHECK(m.kind == 2 && m.K == K && m.N == N, "mat_q8_from_f32 header");
  CHECK(memcmp(m.data, q, K * N) == 0, "mat_q8_from_f32 %lldx%lld data != quantize_int8_percol", (long long)K, (long long)N);
  CHECK(memcmp(m.scale, s, N * 4) == 0, "mat_q8_from_f32 %lldx%lld scale != quantize_int8_percol", (long long)K, (long long)N);
  float *x = xalloc(K * 4), *y = xalloc(N * 4), *yr = xalloc(N * 4);
  fill_x(x, K, 0);
  ref_q8(q, s, (int)K, (int)N, x, 1, yr);
  matvec(&m, x, y);
  CHECK(memcmp(y, yr, N * 4) == 0, "matvec after mat_q8_from_f32 %lldx%lld", (long long)K, (long long)N);
  free_packed(&m); VirtualFree(m.data, 0, MEM_RELEASE);
  _aligned_free(w); _aligned_free(q); _aligned_free(s); _aligned_free(x); _aligned_free(y); _aligned_free(yr);
}

// F32 (kind 0: W (K,N)) and F32T (kind 1: W (N,K)) vs double reference
static void test_f32(int kind, int64_t K, int64_t N, int64_t M) {
  float *W = xalloc(K * N * 4);
  for (int64_t i = 0; i < K * N; i++) W[i] = fnormal() * 0.1f;
  Mat m = {kind, K, N, W};
  float *X = xalloc(M * K * 4), *Y = xalloc(M * N * 4 + 64), *y1 = xalloc(N * 4 + 64);
  for (int64_t i = 0; i < M * K; i++) X[i] = fnormal();
  Y[M * N] = 777.0f;
  matmul_rows(&m, X, M, Y);
  CHECK(Y[M * N] == 777.0f, "f32 kind %d wrote past Y", kind);
  double worst = 0;
  for (int64_t r = 0; r < M; r++) {
    for (int64_t j = 0; j < N; j++) {
      double s = 0, a = 0;
      for (int64_t k = 0; k < K; k++) {
        double w = kind == 0 ? W[k * N + j] : W[j * K + k];
        s += (double)X[r * K + k] * w; a += fabs((double)X[r * K + k] * w);
      }
      double e = fabs(Y[r * N + j] - s) / (a + 1e-30);
      if (e > worst) worst = e;
    }
    y1[N] = 555.0f;
    matvec(&m, X + r * K, y1);
    CHECK(memcmp(y1, Y + r * N, N * 4) == 0, "f32 kind %d %lldx%lld: rows!=matvec at row %lld", kind, (long long)K, (long long)N, (long long)r);
    CHECK(y1[N] == 555.0f, "f32 matvec wrote past y");
  }
  CHECK(worst < 2e-6, "f32 kind %d %lldx%lld M=%lld rel err %.3g", kind, (long long)K, (long long)N, (long long)M, worst);
  _aligned_free(W); _aligned_free(X); _aligned_free(Y); _aligned_free(y1);
}

// pool sanity: every index visited exactly once
static volatile LONG visit[100003];
static void visit_fn(void *ctx, int64_t i0, int64_t i1) { (void)ctx; for (int64_t i = i0; i < i1; i++) InterlockedIncrement(&visit[i]); }
static void test_pool(void) {
  int64_t ns[] = {1, 2, 7, 16, 17, 1000, 100003};
  int64_t gs[] = {1, 3, 64, 1000};
  for (int a = 0; a < 7; a++) for (int b = 0; b < 4; b++) for (int rep = 0; rep < 3; rep++) {
    int64_t n = ns[a];
    memset((void *)visit, 0, sizeof visit);
    pool_run(visit_fn, 0, n, gs[b]);
    int ok = 1;
    for (int64_t i = 0; i < n; i++) if (visit[i] != 1) ok = 0;
    CHECK(ok, "pool_run n=%lld grain=%lld threads=%lld", (long long)n, (long long)gs[b], (long long)pool_threads());
  }
}

static void run_correctness(void) {
  static const int64_t qs[][2] = {
    {896, 896}, {896, 128}, {896, 4864}, {4864, 896}, {896, 151936}, {1, 1}, {13, 37},
    {576, 576}, {576, 192}, {576, 1536}, {1536, 576}, {3, 17}, {17, 3}, {100, 31}, {33, 16},
    {16, 33}, {7, 8}, {64, 1}, {1, 64}, {4095, 129},
  };
  int nq = sizeof qs / sizeof qs[0];
  int64_t tcs[] = {1, 16};
  for (int t = 0; t < 2; t++) {
    pool_init(tcs[t]);
    int64_t small = mv_small_bytes;
    for (int force = 0; force < 2; force++) {
      mv_small_bytes = force ? 0 : small;             // force = thread even tiny shapes
      if (force) mv_min_chunk_bytes = 1;
      test_pool();
      for (int i = 0; i < nq; i++) { test_q8_shape(qs[i][0], qs[i][1]); if (getenv("KT_TRACE")) fprintf(stderr, "[%d:%d:%.2f]", force, i, now_s()); }
      int64_t rows[] = {1, 2, 3, 5, 6, 7, 13, 20};
      for (int i = 0; i < 8; i++) {
        test_q8_rows(896, 128, rows[i]);
        test_q8_rows(13, 37, rows[i]);
        test_q8_rows(576, 1536, rows[i]);
      }
      test_q8_rows(100, 47, 77);
      test_quantize(13, 37); test_quantize(896, 128); test_quantize(64, 1000);
      test_quantize(100, 7); test_quantize(1, 1); test_quantize(5, 9);
      int64_t fs[][3] = {{896, 896, 1}, {13, 37, 3}, {1, 1, 1}, {576, 1536, 7}, {1536, 576, 13},
                         {100, 200, 6}, {17, 70, 5}, {64, 256, 2}, {33, 65, 9}, {7, 3, 4}, {896, 128, 1}};
      for (int i = 0; i < 11; i++) {
        test_f32(0, fs[i][0], fs[i][1], fs[i][2]);
        test_f32(1, fs[i][0], fs[i][1], fs[i][2]);
      }
      mv_small_bytes = small; mv_min_chunk_bytes = 16384;
    }
    printf("correctness with %lld thread(s): %s\n", (long long)pool_threads(), fails ? "FAILURES" : "ok");
  }
}

// ===========================================================================
// benchmarks
// ===========================================================================
static double now_s(void) {
  static LARGE_INTEGER f; LARGE_INTEGER c;
  if (!f.QuadPart) QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return (double)c.QuadPart / (double)f.QuadPart;
}

typedef struct { const char *p; int64_t unit; volatile uint64_t sink; } SR;
static void sr_fn(void *ctx, int64_t i0, int64_t i1) {
  SR *s = ctx;
  s->sink ^= stream_read(s->p + i0 * s->unit, (i1 - i0) * s->unit);
}
static double stream_bw(const char *buf, int64_t bytes, int reps) {
  SR s = {buf, 1 << 20, 0};
  int64_t units = bytes >> 20;
  pool_run(sr_fn, &s, units, 1);
  double t0 = now_s();
  for (int r = 0; r < reps; r++) pool_run(sr_fn, &s, units, 4);
  double t = now_s() - t0;
  return (double)bytes * reps / t / 1e9;
}

typedef struct { const char *name; int64_t d, nh, nkv, hd, ff, V, L; } Cfg;
static const Cfg QWEN = {"Qwen2.5-0.5B", 896, 14, 2, 64, 4864, 151936, 24};
static const Cfg SMOL = {"SmolLM2-135M", 576, 9, 3, 64, 1536, 49152, 30};

typedef struct { Mat w[7]; } LayerM;   // q k v o gate up down
static const char *wname[8] = {"Wq", "Wk", "Wv", "Wo", "Wgate", "Wup", "Wdown", "lm_head"};

static void layer_shapes(const Cfg *c, int64_t sh[8][2]) {
  int64_t kv = c->nkv * c->hd;
  int64_t s[8][2] = {{c->d, c->nh * c->hd}, {c->d, kv}, {c->d, kv}, {c->nh * c->hd, c->d},
                     {c->d, c->ff}, {c->d, c->ff}, {c->ff, c->d}, {c->d, c->V}};
  memcpy(sh, s, sizeof s);
}

// Build a whole model's worth of matrices (random weights) so that a "token"
// streams every weight from DRAM exactly once, like a real decode step.
static Mat *build_model(const Cfg *c, int kind, int64_t *total_bytes) {
  int64_t sh[8][2]; layer_shapes(c, sh);
  int64_t n = c->L * 7 + 1;
  Mat *ms = calloc(n, sizeof(Mat));
  *total_bytes = 0;
  for (int64_t i = 0; i < n; i++) {
    int w = i == n - 1 ? 7 : (int)(i % 7);
    int64_t K = sh[w][0], N = sh[w][1];
    Mat *m = &ms[i];
    if (kind == 2) {
      make_q8(m, K, N);
      _aligned_free(m->data); m->data = 0;            // product frees the raw copy too
      *total_bytes += ((K + 15) & ~15) * ((N + 15) & ~15);
    } else {
      float *W = xalloc(K * N * 4);
      for (int64_t t = 0; t < K * N; t++) W[t] = (float)((int)(xr() & 255) - 128) * 1e-3f;
      m->kind = (w == 7 && kind == 1) ? 1 : 0;        // kind 1: tied head (N,K)
      m->K = K; m->N = N; m->data = W;
      *total_bytes += K * N * 4;
    }
  }
  return ms;
}
static void free_model(Mat *ms, const Cfg *c) {
  int64_t n = c->L * 7 + 1;
  for (int64_t i = 0; i < n; i++) {
    if (ms[i].kind == 2) { free_packed(&ms[i]); _aligned_free(ms[i].scale); }
    else _aligned_free(ms[i].data);
  }
  free(ms);
}

// time `tokens` decode steps worth of matvecs; returns ms/token, fills per-weight ms
static double run_tokens(const Cfg *c, Mat *ms, int tokens, double per[8]) {
  int64_t n = c->L * 7 + 1;
  int64_t maxdim = c->V > c->ff ? c->V : c->ff;
  float *x = xalloc(maxdim * 4), *y = xalloc(maxdim * 4);
  fill_x(x, maxdim, 0);
  memset(per, 0, 8 * sizeof(double));
  for (int64_t i = 0; i < n; i++) matvec(&ms[i], x, y);   // warm-up
  double t0 = now_s();
  for (int t = 0; t < tokens; t++)
    for (int64_t i = 0; i < n; i++) {
      double a = now_s();
      matvec(&ms[i], x, y);
      per[i == n - 1 ? 7 : i % 7] += now_s() - a;
    }
  double tt = (now_s() - t0) / tokens;
  for (int w = 0; w < 8; w++) per[w] = per[w] / tokens * 1e3;
  _aligned_free(x); _aligned_free(y);
  return tt * 1e3;
}

static void bench_model(const Cfg *c, int kind, int64_t *tcs, int ntc) {
  int64_t bytes;
  Mat *ms = build_model(c, kind, &bytes);
  const char *kn = kind == 2 ? "Q8" : kind == 1 ? "F32 (tied F32T head)" : "F32";
  printf("\n== %s %s: %.1f MB of weights per token ==\n", c->name, kn, bytes / 1e6);
  printf("  %-12s %8s %8s   per-matrix ms/token: %s %s %s %s %s %s %s %s\n", "config", "ms/tok", "GB/s",
         wname[0], wname[1], wname[2], wname[3], wname[4], wname[5], wname[6], wname[7]);
  int64_t feats = cpu_features;
  for (int path = 0; path < (kind == 2 && (feats & 2) ? 2 : 1); path++) {
    cpu_features = path == 0 ? feats : (feats & ~2);
    for (int t = 0; t < ntc; t++) {
      pool_init(tcs[t]); mv_mem_threads = tcs[t];       // decode threads = T exactly
      double per[8];
      int tokens = tcs[t] == 1 ? 3 : 10;
      double ms_tok = run_tokens(c, ms, tokens, per);
      char cfg[32];
      snprintf(cfg, sizeof cfg, "%s T=%lld", kind != 2 ? "f32" : (cpu_features & 2) ? "vnni" : "avx2", (long long)tcs[t]);
      printf("  %-12s %8.2f %8.1f   ", cfg, ms_tok, bytes / (ms_tok * 1e-3) / 1e9);
      for (int w = 0; w < 8; w++) printf("%.3f ", per[w]);
      printf("\n");
    }
  }
  cpu_features = feats; mv_mem_threads = 0;
  free_model(ms, c);
}

// hot (cache-resident) latency of one shape - dispatch overhead dominated
static void bench_hot(int64_t K, int64_t N, int64_t T) {
  Mat m; make_q8(&m, K, N);
  float *x = xalloc(K * 4), *y = xalloc(N * 4);
  fill_x(x, K, 0);
  pool_init(T); mv_mem_threads = T;
  for (int i = 0; i < 100; i++) matvec(&m, x, y);
  int reps = 2000;
  double t0 = now_s();
  for (int i = 0; i < reps; i++) matvec(&m, x, y);
  double us = (now_s() - t0) / reps * 1e6;
  printf("  hot Q8 %5lldx%-6lld T=%-2lld %8.2f us/call  (%.1f GB/s from cache)\n", (long long)K, (long long)N,
         (long long)T, us, (double)K * N / (us * 1e-6) / 1e9);
  free_packed(&m); _aligned_free(m.data); _aligned_free(m.scale); _aligned_free(x); _aligned_free(y);
  mv_mem_threads = 0;
}

static void bench_prefill(int64_t K, int64_t N, int64_t M, int64_t T, int kind) {
  Mat m;
  float *W = 0;
  if (kind == 2) make_q8(&m, K, N);
  else { W = xalloc(K * N * 4); for (int64_t i = 0; i < K * N; i++) W[i] = frand(); Mat t = {kind, K, N, W}; m = t; }
  float *X = xalloc(M * K * 4), *Y = xalloc(M * N * 4);
  for (int64_t i = 0; i < M; i++) fill_x(X + i * K, K, 0);
  pool_init(T);
  matmul_rows(&m, X, M, Y);
  int reps = 5;
  double t0 = now_s();
  for (int i = 0; i < reps; i++) matmul_rows(&m, X, M, Y);
  double s = (now_s() - t0) / reps;
  printf("  prefill %s %5lldx%-6lld M=%-4lld T=%-2lld %8.3f ms  %7.1f GMAC/s\n", kind == 2 ? ((cpu_features & 2) ? "Q8vnni" : "Q8avx2") : "F32   ",
         (long long)K, (long long)N, (long long)M, (long long)T, s * 1e3, (double)K * N * M / s / 1e9);
  if (kind == 2) { free_packed(&m); _aligned_free(m.data); _aligned_free(m.scale); } else _aligned_free(W);
  _aligned_free(X); _aligned_free(Y);
}

static void bench(int argc, char **argv) {
  (void)argc; (void)argv;
  int64_t bytes = 1ll << 29;
  char *buf = xalloc(bytes);
  memset(buf, 1, bytes);
  printf("== streaming read (asm, 512 MB buffer) ==\n");
  int64_t tcs[] = {1, 2, 4, 6, 8, 10, 12, 16};
  for (int i = 0; i < 8; i++) { pool_init(tcs[i]); printf("  T=%-2lld %6.1f GB/s\n", (long long)tcs[i], stream_bw(buf, bytes, 6)); }
  _aligned_free(buf);

  printf("\n== hot (L2/L3-resident) single-matrix latency ==\n");
  int64_t hs[][2] = {{896, 128}, {896, 896}, {576, 576}, {896, 4864}, {4864, 896}};
  for (int i = 0; i < 5; i++) { bench_hot(hs[i][0], hs[i][1], 1); bench_hot(hs[i][0], hs[i][1], 6); bench_hot(hs[i][0], hs[i][1], 16); }

  int64_t mt[] = {1, 4, 6, 8, 12, 16};
  bench_model(&QWEN, 2, mt, 6);
  bench_model(&SMOL, 2, mt, 6);
  int64_t ft[] = {1, 6, 16};
  bench_model(&SMOL, 1, ft, 3);

  printf("\n== prefill (matmul_rows) ==\n");
  bench_prefill(896, 4864, 64, 1, 2);
  bench_prefill(896, 4864, 64, 16, 2);
  bench_prefill(4864, 896, 64, 16, 2);
  bench_prefill(896, 896, 128, 16, 2);
  int64_t feats = cpu_features; cpu_features &= ~2;
  bench_prefill(896, 4864, 64, 16, 2);
  cpu_features = feats;
  bench_prefill(576, 1536, 64, 16, 0);
}

// pool stress: many tiny pool_runs with random n / grain / thread count and
// random idle gaps (some longer than the spin budget, so workers go to sleep
// and must be woken). Run several instances at once to oversubscribe the CPU.
static volatile LONG64 st_sum;
static void st_fn(void *ctx, int64_t i0, int64_t i1) {
  (void)ctx; int64_t s = 0;
  for (int64_t i = i0; i < i1; i++) s += i + 1;
  InterlockedAdd64(&st_sum, s);
}
static void stress(int64_t iters) {
  double t0 = now_s(), last = t0;
  for (int64_t it = 0; it < iters; it++) {
    if (it % 5000 == 0) pool_init((int64_t)(xr() % 16) + 1);
    int64_t n = (int64_t)(xr() % 200) + 1, g = (int64_t)(xr() % 8) + 1;
    st_sum = 0;
    if (xr() & 1) pool_run(st_fn, 0, n, g);
    else pool_run_n(st_fn, 0, n, g, (int64_t)(xr() % 18) - 1);
    if (st_sum != n * (n + 1) / 2) { printf("STRESS FAIL it=%lld n=%lld g=%lld\n", (long long)it, (long long)n, (long long)g); exit(1); }
    uint64_t r = xr() % 1000;
    if (r < 3) Sleep((DWORD)(xr() % 8));                 // past the spin budget
    else if (r < 20) { double e = now_s() + (xr() % 3000) * 1e-6; while (now_s() < e) ; }
    if (now_s() - last > 5) { last = now_s(); printf("  stress it=%lld (%.0f/s)\n", (long long)it, it / (last - t0)); }
  }
  printf("stress ok: %lld pool_runs in %.1f s\n", (long long)iters, now_s() - t0);
}


// tune: pool of 16, decode (memory-bound) participants T = mv_mem_threads,
// chunks/thread cpt; best-of-3 ms/token. KT_MODEL=smol for SmolLM2.
static void tune(void) {
  const Cfg *c = getenv("KT_MODEL") && !strcmp(getenv("KT_MODEL"), "smol") ? &SMOL : &QWEN;
  int64_t bytes; Mat *ms = build_model(c, getenv("KT_F32") ? 1 : 2, &bytes);
  int64_t cpts[] = {1, 2, 4, 8};
  int64_t ts[] = {4, 5, 6, 7, 8, 10, 12, 16};
  pool_init(16);
  for (int ti = 0; ti < 8; ti++) for (int ci = 0; ci < 4; ci++) {
    mv_mem_threads = ts[ti]; mv_chunks_per_thread = cpts[ci];
    double best = 1e9, per[8];
    for (int r = 0; r < 3; r++) { double v = run_tokens(c, ms, 5, per); if (v < best) best = v; }
    printf("T=%-2lld cpt=%lld %7.2f ms/tok %6.1f GB/s\n", (long long)ts[ti], (long long)cpts[ci], best, bytes / (best * 1e-3) / 1e9);
  }
  mv_mem_threads = 0; mv_chunks_per_thread = 4;
  free_model(ms, c);
}

static void thread_sweep(void) {
  int64_t bytes;
  Mat *ms = build_model(&QWEN, 2, &bytes);
  for (int64_t t = 1; t <= 16; t++) {
    pool_init(t);
    double per[8];
    double ms_tok = run_tokens(&QWEN, ms, t == 1 ? 3 : 10, per);
    printf("T=%-2lld %7.2f ms/token %6.1f GB/s\n", (long long)t, ms_tok, bytes / (ms_tok * 1e-3) / 1e9);
  }
  free_model(ms, &QWEN);
}

#include <tlhelp32.h>
// watchdog (debug aid): if the run takes longer than KT_WATCHDOG seconds, dump
// every thread's RIP so a hang can be located in the disassembly.
extern char pool_run_sym[] __asm__("pool_run");
static DWORD WINAPI watchdog(void *arg) {
  Sleep((DWORD)(intptr_t)arg * 1000);
  DWORD pid = GetCurrentProcessId(), me = GetCurrentThreadId();
  HANDLE sn = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
  THREADENTRY32 te; te.dwSize = sizeof te;
  extern volatile int64_t pool_active, pool_gen, pool_claim, pool_done, job_nchunks, pool_slots[];
  fprintf(stderr, "\nWATCHDOG: active %lld gen %llx claim %llx done %lld nchunks %lld sleep:", (long long)pool_active,
          (long long)pool_gen, (long long)pool_claim, (long long)pool_done, (long long)job_nchunks);
  for (int i = 1; i < 16; i++) fprintf(stderr, " %lld", (long long)pool_slots[i * 8]);
  fprintf(stderr, "\nWATCHDOG: image base %p pool_run %p\n", (void *)GetModuleHandleA(0), (void *)pool_run_sym);
  for (BOOL ok = Thread32First(sn, &te); ok; ok = Thread32Next(sn, &te)) {
    if (te.th32OwnerProcessID != pid || te.th32ThreadID == me) continue;
    HANDLE h = OpenThread(THREAD_ALL_ACCESS, FALSE, te.th32ThreadID);
    SuspendThread(h);
    CONTEXT c; c.ContextFlags = CONTEXT_FULL;
    GetThreadContext(h, &c);
    fprintf(stderr, "  tid %5lu rip %p rsp %p rax %llx rbx %llx r12 %llx\n", te.th32ThreadID, (void *)c.Rip, (void *)c.Rsp,
            c.Rax, c.Rbx, c.R12);
    ResumeThread(h); CloseHandle(h);
  }
  ExitProcess(3);
  return 0;
}

int main(int argc, char **argv) {
  if (getenv("KT_WATCHDOG")) CreateThread(0, 0, watchdog, (void *)(intptr_t)atoi(getenv("KT_WATCHDOG")), 0, 0);
  setvbuf(stdout, NULL, _IONBF, 0);
  int64_t f = cpu_init();
  printf("cpu_init: 0x%llx (AVX2+FMA=%d VNNI=%d)  logical CPUs: %lld\n", (long long)f, (int)(f & 1), (int)((f >> 1) & 1),
         (long long)pool_init(0));
  printf("pool_mem_threads (threads used for memory-bound matvec): %lld\n", (long long)pool_mem_threads);
  if (!(f & 1)) { printf("no AVX2/FMA - kernels need it\n"); return 1; }
  if (argc > 1 && !strcmp(argv[1], "bench")) { bench(argc, argv); return 0; }
  if (argc > 1 && !strcmp(argv[1], "threads")) { thread_sweep(); return 0; }
  if (argc > 1 && !strcmp(argv[1], "tune")) { tune(); return 0; }
  if (argc > 1 && !strcmp(argv[1], "prefill")) {
    int64_t ts[] = {1, 6, 10, 12, 16};
    for (int i = 0; i < 5; i++) { bench_prefill(896, 4864, 64, ts[i], 2); bench_prefill(4864, 896, 64, ts[i], 2); bench_prefill(896, 896, 128, ts[i], 2); }
    return 0;
  }
  if (argc > 1 && !strcmp(argv[1], "stress")) { rng ^= GetCurrentProcessId(); stress(argc > 2 ? atoll(argv[2]) : 200000); return 0; }
  run_correctness();
  printf("%s (%d checks, %d failures)\n", fails ? "SOME TESTS FAILED" : "ALL TESTS PASSED", checks, fails);
  return fails ? 1 : 0;
}
