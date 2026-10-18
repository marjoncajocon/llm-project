// att_test.c - checks ops.S attention() against llama.c decode_attention (bit for
// bit) and benchmarks it at short and long context.
//
// Build + run from asm_port/:
//   ASM="ops pool cpu" ./build.sh test att_test
//
// Environment:
//   ATT_ONLY=check|bench|prefill|bw   run one part (default: check + bench)
//   ATT_REPS=n                     timing samples (bench: 41 tokens, prefill: 3 per chunk)
//   ATT_SHAPE=0|1 / ATT_CTX=n / ATT_N=n   bench only Qwen (0) or SmolLM (1) /
//                                  only context n / prefill only an n-token prompt
//   ATT_SPLIT / ATT_CPT / ATT_VPF  override ops.S's att_split_min /
//                                  att_chunks_per_thread / att_vpf_shift
//   ATT_R / ATT_KT / ATT_VT        override att_rows_per_task / att_ktile / att_vtile
//   ATT_VSL / ATT_KPF              override att_vslices / att_kpf_lines
//   ATT_SMAX=bytes                 override att_split_max_bytes
//   ATT_DT=n                       override att_dec_threads_n (decode threads; -1 = all)
//   ATT_THREADS=n                  pool size (default: auto)
//
// check   random q/K/V, many shapes: GQA groups 1..16, head_dim 16..128, full
//         causal and sliding windows with ring wrap, 1 and many query rows,
//         contexts from 1 to 9000, row-major and blocked K, large-magnitude
//         scores (expf underflow to 0). Each case runs with the default tuning,
//         with split decode forced on, and with odd tile sizes, so every code
//         path is covered. Every output float must equal the reference exactly.
// bench   ms per token for attention alone over all layers (each layer has its
//         own K/V cache, so the caches really stream from DRAM like in decode),
//         Qwen2.5-0.5B (24 layers, 14 heads, 2 kv, hd 64) and SmolLM2-135M
//         (30 layers, 9 heads, 3 kv, hd 64) at ctx 256 / 1024 / 4096 / 8000.
//         Each sample is one token; min and median are printed (the machine
//         may be shared with other jobs).
// prefill attention time of a whole long prompt in 64-row chunks like
//         decode.S's prefill (sum over chunks of each chunk's best time).
// bw      streaming-read bandwidth of the machine with 1..16 threads (the
//         floor for long-context decode attention: K+V bytes / bandwidth).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <windows.h>

// AttnArgs (ops.S)
typedef struct {
  const float *q, *kc, *vc;
  int64_t cap, window, pos0, nrows, n_heads, n_kv, hd;
  float *sc; int64_t scstride; float *o;
  float scale; int32_t pad_; int64_t group, pad2_;
} AttnArgs;
_Static_assert(sizeof(AttnArgs) == 128 && offsetof(AttnArgs, o) == 96 && offsetof(AttnArgs, scale) == 104, "AttnArgs");

void A_attention(AttnArgs *a) __asm__("attention");
void A_kv_put(float *cache, int64_t cap, int64_t slot, const float *row, int64_t kvd) __asm__("kv_put");
int64_t cpu_init(void);
int64_t pool_init(int64_t n);
int64_t pool_threads(void);
extern int64_t att_split_min, att_chunks_per_thread, att_vpf_shift;   // ops.S tuning words
extern int64_t att_rows_per_task, att_ktile, att_vtile, att_vslices, att_kpf_lines, att_split_max_bytes, att_dec_threads_n;

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static float *falloc(size_t n) {
  float *p = (float*)_aligned_malloc((n + 16) * sizeof(float), 64);
  memset(p, 0, (n + 16) * sizeof(float));
  return p;
}
static uint64_t rs = 0x9E3779B97F4A7C15ull;
static float gauss(void) {         // sum of 4 uniforms, roughly N(0, 1)
  float s = 0;
  for (int i = 0; i < 4; i++) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; s += (float)((rs >> 40) * (1.0 / 16777216.0)); }
  return (s - 2.0f) * 1.7320508f;
}
static double now_s(void) {
  LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
  return (double)c.QuadPart / (double)f.QuadPart;
}
static int window_lo(int i, int window) { return (window > 0 && i - window + 1 > 0) ? i - window + 1 : 0; }

// llama.c decode_attention, verbatim (it is static there). Compiled by zig cc -O2
// -mfma, so `dot += qh[d]*kj[d]` and `oh[d] += w*vj[d]` contract to FMAs exactly
// like the C engine's build.
static void ref_attention(const float *q, const float *kc, const float *vc,
                          int cap, int window, int pos, int n_heads, int n_kv,
                          int hd, float *sc, float *o) {
  const int group = n_heads / n_kv, kvd = n_kv * hd;
  const float scale = 1.0f / sqrtf((float)hd);
  const int lo = window_lo(pos, window);
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

// score scratch the way decode.S sizes it: per (row, head) task
// roundup8(longest span + 7) + 8 floats
static int64_t sc_stride(int64_t pos0, int64_t nrows) { return ((pos0 + nrows + 14) & ~7ll) + 8; }

static void check(void) {
  printf("[check] attention vs llama.c decode_attention, %lld threads\n", (long long)pool_threads());
  struct { int nh, nkv, hd, cap, window, pos, nrows; float qmul; } cs[] = {
    // short contexts (single-thread path) -- the old dec_test cases
    {14, 2, 64, 512, 0, 0, 1, 1}, {14, 2, 64, 512, 0, 1, 1, 1}, {14, 2, 64, 512, 0, 7, 1, 1},
    {14, 2, 64, 512, 0, 8, 1, 1}, {14, 2, 64, 512, 0, 9, 1, 1}, {14, 2, 64, 512, 0, 100, 1, 1},
    {14, 2, 64, 512, 0, 511, 1, 1}, {9, 3, 64, 4096, 0, 3000, 1, 1}, {14, 2, 64, 4096, 0, 4095, 1, 1},
    {14, 2, 64, 100, 100, 350, 1, 1}, {14, 2, 64, 37, 37, 1000, 1, 1}, {14, 2, 64, 512, 60, 300, 1, 1},
    {8, 8, 128, 300, 0, 299, 1, 1}, {4, 2, 32, 64, 0, 50, 1, 1}, {4, 1, 16, 64, 20, 63, 1, 1},
    {14, 2, 64, 512, 0, 37, 5, 1}, {9, 3, 64, 1024, 0, 900, 64, 1}, {14, 2, 64, 512, 0, 0, 64, 1},
    {16, 4, 128, 256, 0, 100, 7, 1}, {14, 2, 64, 64, 64, 1000, 1, 1}, {14, 2, 64, 16, 16, 333, 1, 1},
    {9, 3, 64, 32, 20, 101, 9, 1}, {14, 2, 64, 24, 24, 200, 1, 1},
    // long contexts (key-split path), Qwen / SmolLM shapes
    {14, 2, 64, 8192, 0, 1023, 1, 1}, {14, 2, 64, 8192, 0, 4095, 1, 1}, {14, 2, 64, 8192, 0, 7999, 1, 1},
    {14, 2, 64, 8192, 0, 8191, 1, 1}, {9, 3, 64, 8192, 0, 8191, 1, 1}, {9, 3, 64, 8192, 0, 4097, 1, 1},
    {9, 3, 64, 8192, 0, 2049, 1, 1}, {9, 3, 64, 8192, 0, 777, 1, 1}, {9, 3, 64, 8192, 0, 6000, 1, 3},
    {14, 2, 64, 8192, 0, 5000, 1, 4}, {9, 3, 64, 4096, 0, 3000, 1, 40}, // big scores: e_j underflow to 0
    // sliding windows that wrap the ring (blocked layout: cap % 8 == 0)
    {14, 2, 64, 4096, 4096, 9000, 1, 1}, {9, 3, 64, 2048, 2000, 7777, 1, 1}, {14, 2, 64, 1024, 1000, 5003, 1, 1},
    {14, 2, 64, 4096, 3000, 4500, 1, 1}, {9, 3, 64, 4096, 4096, 4100, 1, 1},
    // other GQA groups / head dims
    {16, 1, 64, 4096, 0, 3500, 1, 1},  // group 16 (> 14 accumulators)
    {28, 4, 64, 4096, 0, 3333, 1, 1},  // group 7, 4 kv heads
    {32, 8, 128, 4096, 0, 4000, 1, 1}, // group 4, hd 128
    {8, 8, 128, 4096, 0, 2500, 1, 1},  // MHA
    {6, 2, 32, 2048, 0, 2047, 1, 1}, {4, 1, 16, 4096, 0, 3001, 1, 1}, {12, 4, 48, 4096, 0, 3999, 1, 1},
    {15, 3, 80, 4096, 0, 2222, 1, 1},  // hd 80 (not a multiple of 16)
    // row-major K (cap % 8 != 0), long
    {14, 2, 64, 3001, 0, 3000, 1, 1}, {9, 3, 64, 2500, 2500, 6000, 1, 1},
    // prefill-like batches at long context
    {14, 2, 64, 8192, 0, 4000, 64, 1}, {9, 3, 64, 8192, 0, 3000, 17, 1}, {9, 3, 64, 4096, 1500, 5000, 33, 1},
  };
  int nc = (int)(sizeof cs / sizeof cs[0]);
  for (int c = 0; c < nc; c++) {
    int nh = cs[c].nh, nkv = cs[c].nkv, hd = cs[c].hd, cap = cs[c].cap, W = cs[c].window;
    int pos0 = cs[c].pos, nr = cs[c].nrows, qd = nh * hd, kvd = nkv * hd;
    float *q = falloc((size_t)nr * qd), *kc = falloc((size_t)cap * kvd), *vc = falloc((size_t)cap * kvd);
    float *o = falloc((size_t)nr * qd), *oref = falloc((size_t)nr * qd);
    float *scr = falloc((size_t)pos0 + nr + 16);
    int64_t stride = sc_stride(pos0, nr);
    float *sc = falloc((size_t)nr * nh * stride);
    for (size_t i = 0; i < (size_t)nr * qd; i++) q[i] = gauss() * cs[c].qmul;
    for (size_t i = 0; i < (size_t)cap * kvd; i++) { kc[i] = gauss(); vc[i] = gauss(); }
    float *kca = falloc((size_t)cap * kvd);                 // asm layout (blocked if cap % 8 == 0)
    for (int sl = 0; sl < cap; sl++) A_kv_put(kca, cap, sl, kc + (size_t)sl * kvd, kvd);
    for (int r = 0; r < nr; r++)
      ref_attention(q + (size_t)r * qd, kc, vc, cap, W, pos0 + r, nh, nkv, hd, scr, oref + (size_t)r * qd);
    // every case runs under 3 tuning sets, so all code paths get exercised:
    //   0 = the defaults, 1 = split decode forced on (any length), 2 = odd sizes
    //   (3 rows per task, 72-key K tiles, 40-key V tiles, 2 V slices per
    //   pass-C task, 5 chunks per thread, K prefetch 3 lines)
    int64_t save[8] = {att_split_min, att_split_max_bytes, att_rows_per_task, att_ktile,
                       att_vtile, att_vslices, att_chunks_per_thread, att_kpf_lines};
    size_t bad = 0;
    for (int cfg = 0; cfg < 3; cfg++) {
      if (cfg == 1) { att_split_min = 1; att_split_max_bytes = INT64_MAX; }
      if (cfg == 2) { att_rows_per_task = 3; att_ktile = 72; att_vtile = 40; att_vslices = 2;
                      att_chunks_per_thread = 5; att_kpf_lines = 3; att_split_min = 30; }
      for (size_t i = 0; i < (size_t)nr * qd; i++) o[i] = NAN;
      AttnArgs a; memset(&a, 0, sizeof a);
      a.q = q; a.kc = kca; a.vc = vc; a.cap = cap; a.window = W; a.pos0 = pos0; a.nrows = nr;
      a.n_heads = nh; a.n_kv = nkv; a.hd = hd; a.sc = sc; a.scstride = stride; a.o = o;
      A_attention(&a);
      for (size_t i = 0; i < (size_t)nr * qd; i++) bad += memcmp(&o[i], &oref[i], 4) != 0;
    }
    att_split_min = save[0]; att_split_max_bytes = save[1]; att_rows_per_task = save[2]; att_ktile = save[3];
    att_vtile = save[4]; att_vslices = save[5]; att_chunks_per_thread = save[6]; att_kpf_lines = save[7];
    printf("  nh=%2d kv=%d hd=%3d cap=%4d W=%4d pos=%4d rows=%2d qx%.0f : %s\n", nh, nkv, hd, cap, W, pos0, nr,
           cs[c].qmul, bad ? "DIFFERENT" : "bit-exact");
    CHECK(bad == 0, "case %d: %zu floats differ", c, bad);
    _aligned_free(kca); _aligned_free(q); _aligned_free(kc); _aligned_free(vc);
    _aligned_free(o); _aligned_free(oref); _aligned_free(scr); _aligned_free(sc);
  }
}

// A model-shaped cache: L layers of `cap` slots, K in the asm (blocked) layout.
typedef struct { const char *name; int L, nh, nkv, hd; } Shape;
static const Shape shapes[] = { {"Qwen2.5-0.5B", 24, 14, 2, 64}, {"SmolLM2-135M", 30, 9, 3, 64} };

static int dcmp(const void *a, const void *b) { double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y); }

// The machine may be shared (other jobs stealing cores / memory bandwidth), so
// each sample is ONE token (all layers) and we report the minimum and the median.
static void bench(void) {
  int reps = getenv("ATT_REPS") ? atoi(getenv("ATT_REPS")) : 41;
  const int cap = 8192;
  int ctxs[] = {256, 1024, 4096, 8000};
  int only_shape = getenv("ATT_SHAPE") ? atoi(getenv("ATT_SHAPE")) : -1;
  int only_ctx = getenv("ATT_CTX") ? atoi(getenv("ATT_CTX")) : 0;
  printf("[bench] attention alone, all layers, ms per token: min / median of %d tokens\n", reps);
  double *t = malloc(reps * sizeof *t);
  for (int s = 0; s < 2; s++) {
    if (only_shape >= 0 && s != only_shape) continue;
    const Shape *sh = &shapes[s];
    int qd = sh->nh * sh->hd, kvd = sh->nkv * sh->hd;
    float **K = malloc(sh->L * sizeof *K), **V = malloc(sh->L * sizeof *V);
    for (int l = 0; l < sh->L; l++) {
      K[l] = falloc((size_t)cap * kvd); V[l] = falloc((size_t)cap * kvd);
      for (size_t i = 0; i < (size_t)cap * kvd; i++) { K[l][i] = gauss(); V[l][i] = gauss(); }
    }
    float *q = falloc(qd), *o = falloc(qd);
    for (int i = 0; i < qd; i++) q[i] = gauss() * 0.3f;
    int64_t stride = sc_stride(cap, 1);
    float *sc = falloc((size_t)sh->nh * stride);
    printf("  %s (%d layers, %d heads, %d kv, hd %d)\n", sh->name, sh->L, sh->nh, sh->nkv, sh->hd);
    for (int c = 0; c < 4; c++) {
      if (only_ctx && c > 0) break;
      if (only_ctx) ctxs[c] = only_ctx;                // ATT_CTX=n: just that context
      for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        for (int l = 0; l < sh->L; l++) {
          AttnArgs a; memset(&a, 0, sizeof a);
          a.q = q; a.kc = K[l]; a.vc = V[l]; a.cap = cap; a.window = 0; a.pos0 = ctxs[c] - 1; a.nrows = 1;
          a.n_heads = sh->nh; a.n_kv = sh->nkv; a.hd = sh->hd; a.sc = sc; a.scstride = stride; a.o = o;
          A_attention(&a);
        }
        t[r] = now_s() - t0;
      }
      qsort(t, reps, sizeof *t, dcmp);
      printf("    ctx %5d: %7.3f / %7.3f ms\n", ctxs[c], t[0] * 1e3, t[reps / 2] * 1e3);
    }
    for (int l = 0; l < sh->L; l++) { _aligned_free(K[l]); _aligned_free(V[l]); }
    free(K); free(V); _aligned_free(q); _aligned_free(o); _aligned_free(sc);
  }
  free(t);
}

// prefill attention of an n-token prompt, 64 rows per call (decode.S's batch)
static void prefill(void) {
  int reps = getenv("ATT_REPS") ? atoi(getenv("ATT_REPS")) : 3;
  const int cap = 8192, M = 64;
  int ns[] = {1024, 4096, 8000};
  printf("[prefill] attention of the whole prompt, all layers, seconds (sum of per-chunk best of %d)\n", reps);
  for (int s = 0; s < 2; s++) {
    const Shape *sh = &shapes[s];
    int qd = sh->nh * sh->hd, kvd = sh->nkv * sh->hd;
    float *K = falloc((size_t)cap * kvd), *V = falloc((size_t)cap * kvd);   // one layer, reused
    for (size_t i = 0; i < (size_t)cap * kvd; i++) { K[i] = gauss(); V[i] = gauss(); }
    float *q = falloc((size_t)M * qd), *o = falloc((size_t)M * qd);
    for (int i = 0; i < M * qd; i++) q[i] = gauss() * 0.3f;
    int64_t stride = sc_stride(cap, M);
    float *sc = falloc((size_t)M * sh->nh * stride);
    printf("  %s:", sh->name);
    for (int c = 0; c < 3; c++) {
      if (getenv("ATT_N") && atoi(getenv("ATT_N")) != ns[c]) continue;
      if (getenv("ATT_SHAPE") && atoi(getenv("ATT_SHAPE")) != s) continue;
      // sum over the prompt's 64-row chunks of each chunk's best time (robust
      // against other jobs on the machine), times the layer count
      double best = 0;
      for (int p0 = 0; p0 < ns[c]; p0 += M) {
        double bc = 1e9;
        for (int r = 0; r < reps; r++) {
          double t0 = now_s();
          AttnArgs a; memset(&a, 0, sizeof a);
          a.q = q; a.kc = K; a.vc = V; a.cap = cap; a.window = 0; a.pos0 = p0;
          a.nrows = ns[c] - p0 < M ? ns[c] - p0 : M;
          a.n_heads = sh->nh; a.n_kv = sh->nkv; a.hd = sh->hd; a.sc = sc; a.scstride = sc_stride(p0, a.nrows); a.o = o;
          A_attention(&a);
          bc = fmin(bc, now_s() - t0);
        }
        best += bc * sh->L;
      }
      printf("  n %d %.3f s", ns[c], best);
    }
    printf("\n");
    _aligned_free(K); _aligned_free(V); _aligned_free(q); _aligned_free(o); _aligned_free(sc);
  }
}

// memory bandwidth reference: read a 384 MB buffer with n threads (pool_run_n)
void pool_run_n(void (*fn)(void*, int64_t, int64_t), void *ctx, int64_t n, int64_t grain, int64_t nthreads);
typedef struct { const float *p; size_t chunk; float out[64 * 16]; } BwCtx;
static void bw_fn(void *vc, int64_t i0, int64_t i1) {
  BwCtx *c = vc;
  for (int64_t i = i0; i < i1; i++) {
    const float *p = c->p + (size_t)i * c->chunk;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
    for (size_t j = 0; j < c->chunk; j += 8) {
      s0 += p[j]; s1 += p[j+1]; s2 += p[j+2]; s3 += p[j+3]; s4 += p[j+4]; s5 += p[j+5]; s6 += p[j+6]; s7 += p[j+7];
    }
    c->out[(i & 63) * 16] += s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
  }
}
static void bw(void) {
  size_t n = (size_t)96 << 20;            // floats = 384 MB
  float *p = falloc(n);
  for (size_t i = 0; i < n; i++) p[i] = (float)(i & 1023);
  BwCtx c; memset(&c, 0, sizeof c); c.p = p; c.chunk = 64 * 1024;   // 256 KB chunks
  int64_t th[] = {1, 2, 4, 6, 8, 12, 16};
  printf("[bw] streaming read of 384 MB, GB/s (best of 9)\n");
  for (int t = 0; t < 7; t++) {
    double best = 1e9;
    for (int r = 0; r < 9; r++) { double t0 = now_s(); pool_run_n(bw_fn, &c, (int64_t)(n / c.chunk), 1, th[t]); best = fmin(best, now_s() - t0); }
    printf("  %2lld threads: %.1f GB/s\n", (long long)th[t], n * 4.0 / best / 1e9);
  }
  _aligned_free(p);
}

// crash reporter: print the faulting address relative to attention()
static LONG WINAPI on_crash(EXCEPTION_POINTERS *e) {
  printf("\n*** exception %08lx at attention+0x%llx (rsp %p)\n", e->ExceptionRecord->ExceptionCode,
         (unsigned long long)((char*)e->ExceptionRecord->ExceptionAddress - (char*)A_attention), (void*)e->ContextRecord->Rsp);
  ExitProcess(3);
  return EXCEPTION_CONTINUE_SEARCH;
}

int main(void) {
  AddVectoredExceptionHandler(1, on_crash);
  setvbuf(stdout, NULL, _IONBF, 0);
  cpu_init();
  pool_init(getenv("ATT_THREADS") ? atoi(getenv("ATT_THREADS")) : 0);
  if (getenv("ATT_SPLIT")) att_split_min = atoll(getenv("ATT_SPLIT"));
  if (getenv("ATT_CPT")) att_chunks_per_thread = atoll(getenv("ATT_CPT"));
  if (getenv("ATT_VPF")) att_vpf_shift = atoll(getenv("ATT_VPF"));
  if (getenv("ATT_R")) att_rows_per_task = atoll(getenv("ATT_R"));
  if (getenv("ATT_KT")) att_ktile = atoll(getenv("ATT_KT"));
  if (getenv("ATT_VT")) att_vtile = atoll(getenv("ATT_VT"));
  if (getenv("ATT_VSL")) att_vslices = atoll(getenv("ATT_VSL"));
  if (getenv("ATT_KPF")) att_kpf_lines = atoll(getenv("ATT_KPF"));
  if (getenv("ATT_SMAX")) att_split_max_bytes = atoll(getenv("ATT_SMAX"));
  if (getenv("ATT_DT")) att_dec_threads_n = atoll(getenv("ATT_DT"));
  const char *only = getenv("ATT_ONLY");
  if (!only || !strcmp(only, "check")) check();
  if (!only || !strcmp(only, "bench")) bench();
  if (only && !strcmp(only, "prefill")) prefill();
  if (only && !strcmp(only, "bw")) bw();
  printf(fails ? "att_test: %d FAILURES\n" : "att_test: all OK\n", fails);
  return fails != 0;
}
