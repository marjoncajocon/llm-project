// matrix.c - Basic matrix operations (with a persistent thread pool for GEMMs)
#include "matrix.h"

#include <stdlib.h>

#include "check.h"
#include "backend.h"

// ===========================================================================
// Persistent thread pool (also exported as tf_parallel_for, see matrix.h).
//
// A fork-join pool: N-1 worker threads are created once (lazily) and reused for
// every parallel region — no per-call thread spawning (that was measured slower).
// `parallel_for(total, fn, ctx)` splits [0,total) into N contiguous chunks; the
// workers run chunks 0..N-2 and the calling thread runs chunk N-1, then joins.
//
// matmul / matmul_backward parallelize over OUTPUT ROWS (or dB rows), so every
// output element is produced by exactly one thread in the same summation order
// as the single-threaded kernel -> results are bit-identical. A few-row forward
// (decode, M = 1) would leave all but one thread idle that way, so it splits the
// OUTPUT COLUMNS instead -- same argument, still bit-identical. Small matmuls stay
// single-threaded (below MM_THREAD_THRESHOLD), so the demo/tests are unchanged.
// WebAssembly has no threads -> parallel_for runs inline.
// Thread count: $TF_THREADS, else the CPU count (capped).
// ===========================================================================

#if defined(__wasi__) || defined(__wasm__)
#  define POOL_DISABLED 1
#elif defined(_WIN32)
#  include <windows.h>
#else
#  include <pthread.h>
#  include <unistd.h>
#endif

#define POOL_MAX_THREADS 64
#define MM_THREAD_THRESHOLD (1L << 19)   // ~500k MACs; below this, run single-threaded

// Few-row (decode) forward matmuls -- see matmul(): with M < MM_COL_ROWS x the
// thread count, the pool splits the OUTPUT COLUMNS into MM_COL_UNIT-wide pieces
// instead of the rows (M = 1 would otherwise run on one thread).
#define MM_COL_ROWS 4
#define MM_COL_UNIT 64
#ifndef MM_COL_THRESHOLD
// ~130k MACs, so decode's 576x576 q/o projections are threaded too (1 x 576 x
// 576 at 3 threads: 0.007 ms vs 0.015 ms serial; the k/v 576x192 stay serial).
#define MM_COL_THRESHOLD (1L << 17)
#endif

typedef tf_range_fn range_fn;

// Chunk idx of n contiguous chunks of [0, total): how parallel_for splits work.
static void chunk_of(int total, int idx, int n, int *lo, int *hi) {
  int per = (total + n - 1) / n;
  int l = idx * per, h = l + per;
  if (l > total) l = total;
  if (h > total) h = total;
  *lo = l; *hi = h;
}

#if defined(POOL_DISABLED)

static void parallel_for(int total, range_fn fn, void *ctx) { fn(0, total, ctx); }
static int pool_threads(void) { return 1; }

#else

// --- thin per-platform mutex / condition-variable wrappers ---
#if defined(_WIN32)
typedef CRITICAL_SECTION pmtx; typedef CONDITION_VARIABLE pcnd; typedef HANDLE pthr;
static void pmtx_init(pmtx *m) { InitializeCriticalSection(m); }
static void pmtx_lock(pmtx *m) { EnterCriticalSection(m); }
static void pmtx_unlock(pmtx *m) { LeaveCriticalSection(m); }
static void pcnd_init(pcnd *c) { InitializeConditionVariable(c); }
static void pcnd_wait(pcnd *c, pmtx *m) { SleepConditionVariableCS(c, m, INFINITE); }
static void pcnd_signal(pcnd *c) { WakeConditionVariable(c); }
static void pcnd_broadcast(pcnd *c) { WakeAllConditionVariable(c); }
#else
typedef pthread_mutex_t pmtx; typedef pthread_cond_t pcnd; typedef pthread_t pthr;
static void pmtx_init(pmtx *m) { pthread_mutex_init(m, NULL); }
static void pmtx_lock(pmtx *m) { pthread_mutex_lock(m); }
static void pmtx_unlock(pmtx *m) { pthread_mutex_unlock(m); }
static void pcnd_init(pcnd *c) { pthread_cond_init(c, NULL); }
static void pcnd_wait(pcnd *c, pmtx *m) { pthread_cond_wait(c, m); }
static void pcnd_signal(pcnd *c) { pthread_cond_signal(c); }
static void pcnd_broadcast(pcnd *c) { pthread_cond_broadcast(c); }
#endif

static int      P_nthreads = 0;
static int      P_started = 0;
static pmtx     P_mtx;
static pcnd     P_work, P_done;
static int      P_gen = 0;         // job generation counter
static int      P_remaining = 0;   // workers still finishing the current job
static int      P_shutdown = 0;
static range_fn P_fn = NULL;
static void    *P_ctx = NULL;
static int      P_total = 0;
static pthr     P_threads[POOL_MAX_THREADS];
static int      P_ids[POOL_MAX_THREADS];

static void pool_body(int id) {
  int last = 0;
  pmtx_lock(&P_mtx);
  for (;;) {
    while (!P_shutdown && P_gen == last) pcnd_wait(&P_work, &P_mtx);
    if (P_shutdown) { pmtx_unlock(&P_mtx); return; }
    last = P_gen;
    range_fn fn = P_fn; void *ctx = P_ctx; int total = P_total, n = P_nthreads;
    pmtx_unlock(&P_mtx);

    int lo, hi; chunk_of(total, id, n, &lo, &hi);
    fn(lo, hi, ctx);

    pmtx_lock(&P_mtx);
    if (--P_remaining == 0) pcnd_signal(&P_done);
  }
}

#if defined(_WIN32)
static DWORD WINAPI pool_entry(LPVOID a) { pool_body(*(int*)a); return 0; }
#else
static void *pool_entry(void *a) { pool_body(*(int*)a); return NULL; }
#endif

static int detect_threads(void) {
  const char *env = getenv("TF_THREADS");
  if (env && atoi(env) > 0) return atoi(env);
#if defined(_WIN32)
  SYSTEM_INFO si; GetSystemInfo(&si); return (int)si.dwNumberOfProcessors;
#else
  long c = sysconf(_SC_NPROCESSORS_ONLN); return (c > 0) ? (int)c : 4;
#endif
}

static void pool_shutdown(void) {
  if (!P_started || P_nthreads <= 1) return;
  pmtx_lock(&P_mtx); P_shutdown = 1; pcnd_broadcast(&P_work); pmtx_unlock(&P_mtx);
  for (int i = 0; i < P_nthreads - 1; i++) {
#if defined(_WIN32)
    WaitForSingleObject(P_threads[i], INFINITE); CloseHandle(P_threads[i]);
#else
    pthread_join(P_threads[i], NULL);
#endif
  }
}

static void pool_init(void) {
  if (P_started) return;
  int n = detect_threads();
  if (n < 1) n = 1;
  if (n > POOL_MAX_THREADS) n = POOL_MAX_THREADS;
  P_nthreads = n;
  if (n <= 1) { P_started = 1; return; }   // no workers; parallel_for runs inline
  pmtx_init(&P_mtx); pcnd_init(&P_work); pcnd_init(&P_done);
  for (int i = 0; i < n - 1; i++) {
    P_ids[i] = i;
#if defined(_WIN32)
    P_threads[i] = CreateThread(NULL, 0, pool_entry, &P_ids[i], 0, NULL);
#else
    pthread_create(&P_threads[i], NULL, pool_entry, &P_ids[i]);
#endif
  }
  atexit(pool_shutdown);
  P_started = 1;
}

// Threads parallel_for will use (starts the pool on first call).
static int pool_threads(void) {
  if (!P_started) pool_init();
  return P_nthreads;
}

static void parallel_for(int total, range_fn fn, void *ctx) {
  if (!P_started) pool_init();
  if (P_nthreads <= 1 || total < 2) { fn(0, total, ctx); return; }
  pmtx_lock(&P_mtx);
  P_fn = fn; P_ctx = ctx; P_total = total;
  P_remaining = P_nthreads - 1;
  P_gen++;
  pcnd_broadcast(&P_work);
  pmtx_unlock(&P_mtx);

  int lo, hi; chunk_of(total, P_nthreads - 1, P_nthreads, &lo, &hi);  // caller's chunk
  fn(lo, hi, ctx);

  pmtx_lock(&P_mtx);
  while (P_remaining > 0) pcnd_wait(&P_done, &P_mtx);
  pmtx_unlock(&P_mtx);
}

#endif  // POOL_DISABLED

// Exported view of the pool for the non-GEMM training ops (see matrix.h).
static int P_debug_chunks = 0;   // tf_parallel_debug_chunks (tests only)
void tf_parallel_debug_chunks(int n) { P_debug_chunks = (n < 1) ? 0 : n; }

void tf_parallel_for(int total, tf_range_fn fn, void *ctx) {
  if (P_debug_chunks >= 1) {       // test hook: n pieces, one after another, here
    for (int i = 0; i < P_debug_chunks; i++) {
      int lo, hi; chunk_of(total, i, P_debug_chunks, &lo, &hi);
      fn(lo, hi, ctx);
    }
    return;
  }
  parallel_for(total, fn, ctx);
}
int tf_threads(void) { return pool_threads(); }

// ===========================================================================
// AVX2 + FMA kernels (roadmap B4), with RUNTIME dispatch.
//
// WHY RUNTIME DISPATCH, not `#ifdef __AVX2__`: __AVX2__ says what the COMPILER
// was told, not what the CPU has. A native `zig cc` build already targets the
// host CPU (it behaves like -march=native), but every portable binary -- `make
// cross`, the Colab `cross-gpu` artifact, a plain `gcc -O2` -- targets baseline
// x86-64 (SSE2 only) and would never get the fast path; and a binary built WITH
// -mavx2 dies with SIGILL on a CPU without it. So the kernels are compiled with
// __attribute__((target("avx2,fma"))) (no global -mavx2; the rest of the program
// stays baseline), and CPUID + XGETBV decide ONCE, at the first matmul, whether
// to call them. Non-x86 targets (aarch64, wasm32) and compilers without GNU
// attributes compile none of this and run the portable kernels, unchanged.
// $TF_SIMD=0 (or matmul_set_simd(0)) forces the portable path, for A/B runs.
//
// WHY THEY BEAT THE AUTO-VECTORIZED KERNELS: the portable forward already
// vectorizes (its inner loop is an AXPY) -- but it LOADS AND STORES the output
// row once per k, and streams all of B from L3/RAM once per output row. These
// are register-blocked: a 4 x 16 tile of C lives in 8 ymm registers for the
// whole p loop (no C traffic), each B load feeds 4 FMAs, and B is packed into
// small strip-major panels that stay in L1/L2 while every row block passes over
// them. The classic GEMM micro-kernel, just small. (Measured gains and the
// cases where there are none -- M=1 decode -- are in README / notes.md.)
//
// ONE MICRO-KERNEL SERVES ALL THREE PRODUCTS. gemm_tile computes
//     T[r][c] (+)= sum_p  a(r, p) * b(p, c)       p ascending, one FMA per step
// with a(r,p) = a[r*a_rs + p*a_ps] at arbitrary strides, so
//   forward   C  = A   @ B   : a = rows of A     (a_rs = K, a_ps = 1)
//   dB       dB += A^T @ dC  : a = columns of A  (a_rs = 1, a_ps = K), no copy
//   dA       dA += dC  @ B^T : B^T gathered from B while packing
//
// NUMERICS / BIT-EXACTNESS. A forward output element is ONE fused-multiply-add
// chain over k in ascending order, started from 0 -- whichever tile, panel, row
// block, code path (tiles vs the decode AXPY) or thread produced it. So a 1-row
// decode matmul and a T-row prefill matmul give bit-identical rows, and results
// do not depend on $TF_THREADS: selftest parts [4]/[19]/[22]/[26] (KV-cache and
// LoRA decode == full forward, 0.00e+00) rely on exactly that. The backward
// forms sum each MM_KC-deep block the same way and ADD it to the gradient; block
// boundaries depend only on the shape, so they are thread-independent too. What
// is NOT identical is SIMD vs portable: FMA rounds once where mul+add rounds
// twice, and dA/dB associate differently -- ~1e-7..1e-6 relative, gated by
// selftest part [31]. The choice is made once per process, never mixed in a run.
// ===========================================================================

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__)) \
    && !defined(TF_NO_SIMD)
#  define MM_X86_SIMD 1
#  include <immintrin.h>
#  include <cpuid.h>
#endif

static int S_simd = -1;   // -1 = not yet decided, 0 = portable C, 1 = AVX2+FMA

#if defined(MM_X86_SIMD)

// AVX2 and FMA must be in the CPU (CPUID) AND enabled by the OS (XGETBV: YMM
// state is saved on context switch). Checking CPUID alone is the classic bug --
// it crashes under an OS or VM that has AVX turned off.
static int cpu_has_avx2_fma(void) {
  unsigned a, b, c, d;
  if (!__get_cpuid(1, &a, &b, &c, &d)) return 0;
  const int fma = (c >> 12) & 1, osxsave = (c >> 27) & 1, avx = (c >> 28) & 1;
  if (!fma || !osxsave || !avx) return 0;
  unsigned xlo, xhi;
  __asm__ volatile("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0));
  (void)xhi;
  if ((xlo & 6u) != 6u) return 0;                 // XMM (bit 1) + YMM (bit 2) state
  if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return 0;
  return (b >> 5) & 1;                             // AVX2
}

#define SIMD_FN      __attribute__((target("avx2,fma")))
#define SIMD_INLINE  static inline __attribute__((always_inline, target("avx2,fma")))

// maskload/maskstore lane masks: the first w (0..8) of 8 lanes.
static const int32_t mm_mask_src[16] = { -1, -1, -1, -1, -1, -1, -1, -1,
                                          0,  0,  0,  0,  0,  0,  0,  0 };
SIMD_INLINE __m256i lane_mask(int w) {
  if (w < 0) w = 0;
  if (w > 8) w = 8;
  return _mm256_loadu_si256((const __m256i*)(mm_mask_src + 8 - w));
}

// Tile modes. The reduction (p) is split into MM_KC-deep blocks and a tile
// covers one block, so a tile either starts or continues an element's sum:
//   MM_STORE : acc = 0,  C  = acc   (forward, first p block)
//   MM_CONT  : acc = C,  C  = acc   (forward, later p blocks: the running sum
//              was parked in C as a float, so resuming from it is EXACTLY the
//              one unbroken FMA chain -- p-blocking changes no bits)
//   MM_ADD   : acc = 0,  C += acc   (backward: gradients accumulate)
enum { MM_STORE, MM_CONT, MM_ADD };

// The micro-kernel: an R x (8V) tile, R <= 4, V <= 8, R*V <= 8 accumulators.
// Only ever called with compile-time-constant R/V/masked/mode, so after
// inlining acc[][] is plain registers and every branch below folds away.
// `masked`: the ragged last strip (V == 2, only w < 16 columns valid); loads
// and stores are masked so nothing outside b / c is touched.
SIMD_INLINE void gemm_tile(const float *a, size_t a_rs, size_t a_ps,
                           const float *b, size_t ldb, float *c, size_t ldc,
                           int P, int R, int V, int masked, int w, int mode) {
  __m256 acc[4][8];
  __m256i m[8];
  for (int v = 0; v < V; v++) m[v] = lane_mask(masked ? w - 8 * v : 8);
  for (int r = 0; r < R; r++)
    for (int v = 0; v < V; v++) {
      const float *cp = c + (size_t)r * ldc + 8 * v;
      acc[r][v] = (mode != MM_CONT) ? _mm256_setzero_ps()
                : masked ? _mm256_maskload_ps(cp, m[v]) : _mm256_loadu_ps(cp);
    }
  for (int p = 0; p < P; p++) {
    const float *bp = b + (size_t)p * ldb;
    __m256 bv[8];
    for (int v = 0; v < V; v++)
      bv[v] = masked ? _mm256_maskload_ps(bp + 8 * v, m[v]) : _mm256_loadu_ps(bp + 8 * v);
    const float *ap = a + (size_t)p * a_ps;
    for (int r = 0; r < R; r++) {
      const __m256 av = _mm256_broadcast_ss(ap + (size_t)r * a_rs);
      for (int v = 0; v < V; v++) acc[r][v] = _mm256_fmadd_ps(av, bv[v], acc[r][v]);
    }
  }
  for (int r = 0; r < R; r++)
    for (int v = 0; v < V; v++) {
      float *cp = c + (size_t)r * ldc + 8 * v;
      __m256 out = acc[r][v];
      if (mode == MM_ADD)
        out = _mm256_add_ps(masked ? _mm256_maskload_ps(cp, m[v]) : _mm256_loadu_ps(cp), out);
      if (masked) _mm256_maskstore_ps(cp, m[v], out);
      else        _mm256_storeu_ps(cp, out);
    }
}

// One output row as an AXPY sweep: c = 0, then c[:] = fma(a(p), b[p,:], c[:])
// for p ascending, four b rows per pass over c. This is the forward for fewer
// than 4 rows -- in particular single-token decode (M=1), which is bandwidth-
// bound: it reads each b row front to back, the hardware prefetcher's favourite
// pattern, and needs no packing (which would cost as much as the product).
// Parking the running sum in c between passes is exact, so every element is the
// SAME FMA chain gemm_tile produces: a decode row and the matching prefill row
// are bit-identical.
SIMD_FN static void axpy_row(const float *a, size_t a_ps, const float *b, size_t ldb,
                             float *c, int P, int ncols) {
  const int n8 = ncols / 8 * 8;
  const __m256i mt = lane_mask(ncols - n8);
  for (int j = 0; j < ncols; j++) c[j] = 0.0f;
  int p = 0;
  for (; p + 4 <= P; p += 4) {
    const __m256 a0 = _mm256_broadcast_ss(a + (size_t)(p + 0) * a_ps);
    const __m256 a1 = _mm256_broadcast_ss(a + (size_t)(p + 1) * a_ps);
    const __m256 a2 = _mm256_broadcast_ss(a + (size_t)(p + 2) * a_ps);
    const __m256 a3 = _mm256_broadcast_ss(a + (size_t)(p + 3) * a_ps);
    const float *b0 = b + (size_t)p * ldb, *b1 = b0 + ldb, *b2 = b1 + ldb, *b3 = b2 + ldb;
    int j = 0;
    for (; j < n8; j += 8) {
      __m256 cv = _mm256_loadu_ps(c + j);
      cv = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b0 + j), cv);
      cv = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b1 + j), cv);
      cv = _mm256_fmadd_ps(a2, _mm256_loadu_ps(b2 + j), cv);
      cv = _mm256_fmadd_ps(a3, _mm256_loadu_ps(b3 + j), cv);
      _mm256_storeu_ps(c + j, cv);
    }
    if (j < ncols) {
      __m256 cv = _mm256_maskload_ps(c + j, mt);
      cv = _mm256_fmadd_ps(a0, _mm256_maskload_ps(b0 + j, mt), cv);
      cv = _mm256_fmadd_ps(a1, _mm256_maskload_ps(b1 + j, mt), cv);
      cv = _mm256_fmadd_ps(a2, _mm256_maskload_ps(b2 + j, mt), cv);
      cv = _mm256_fmadd_ps(a3, _mm256_maskload_ps(b3 + j, mt), cv);
      _mm256_maskstore_ps(c + j, mt, cv);
    }
  }
  for (; p < P; p++) {
    const __m256 a0 = _mm256_broadcast_ss(a + (size_t)p * a_ps);
    const float *b0 = b + (size_t)p * ldb;
    int j = 0;
    for (; j < n8; j += 8)
      _mm256_storeu_ps(c + j, _mm256_fmadd_ps(a0, _mm256_loadu_ps(b0 + j), _mm256_loadu_ps(c + j)));
    if (j < ncols)
      _mm256_maskstore_ps(c + j, mt, _mm256_fmadd_ps(a0, _mm256_maskload_ps(b0 + j, mt),
                                                     _mm256_maskload_ps(c + j, mt)));
  }
}

// Blocking. MM_KC = depth of a p block, MM_NC = columns per packed panel (a
// multiple of 16); a packed panel is MM_KC x MM_NC floats = 128 KB, L2-sized.
#define MM_KC 256
#define MM_NC 128

// Pack b(p, c) = b[p*b_ps + c*b_cs], p < kc, c < nc, into strip-major order:
// dst[strip][p][16], the last strip zero-padded, so the micro-kernel streams a
// strip as one contiguous kc x 64-byte run. WHY PACK: walked in place, a
// 16-column strip hops across kc rows of b at stride ldb -- 196 KB per row for
// lm_head, so every row is a new page (a TLB miss) and the hardware prefetcher
// sees no stream. Measured before packing: lm_head at a third of the small
// shapes' GFLOP/s, and software prefetch did not recover it. b_cs != 1 is dA's
// transposed operand, gathered here straight from B with no transposed copy.
SIMD_FN static void pack_panel(const float *b, size_t b_ps, size_t b_cs, int kc, int nc,
                               float *dst) {
  for (int s = 0; s * 16 < nc; s++) {
    const int w = (nc - s * 16 < 16) ? nc - s * 16 : 16;
    float *d = dst + (size_t)s * kc * 16;
    if (b_cs == 1) {
      for (int p = 0; p < kc; p++) {
        const float *src = b + (size_t)p * b_ps + (size_t)s * 16;
        float *dp = d + (size_t)p * 16;
        if (w == 16) {
          _mm256_storeu_ps(dp, _mm256_loadu_ps(src));
          _mm256_storeu_ps(dp + 8, _mm256_loadu_ps(src + 8));
        } else {
          for (int x = 0; x < 16; x++) dp[x] = (x < w) ? src[x] : 0.0f;
        }
      }
    } else {                          // transposed: read along p, which is contiguous
      for (int x = 0; x < 16; x++) {
        const float *src = b + (size_t)(s * 16 + x) * b_cs;
        for (int p = 0; p < kc; p++) d[(size_t)p * 16 + x] = (x < w) ? src[(size_t)p * b_ps] : 0.0f;
      }
    }
  }
}

// T[r][c] (+)= sum_{p<P} a(r,p) * b(p,c) for output rows [r0, r1) and columns
// [0, ncols); a(r,p) = a[r*a_rs + p*a_ps], b(p,c) = b[p*b_ps + c*b_cs], T has
// row stride ldc. `accum` selects the backward's MM_ADD over the forward's
// MM_STORE/MM_CONT. For each MM_KC-deep p block and MM_NC-wide panel, b is
// packed once, then each 16-column strip is swept by 4-row tiles (the strip
// stays in L1 across row blocks) and 1-3 leftover rows get an R-row tile.
// Returns 0 only if the panel buffer cannot be allocated (nothing written yet).
SIMD_FN static int gemm_rows(int r0, int r1, int ncols, int P,
                             const float *a, size_t a_rs, size_t a_ps,
                             const float *b, size_t b_ps, size_t b_cs,
                             float *c, size_t ldc, int accum) {
  if (r1 <= r0 || ncols <= 0) return 1;
  if (!accum && b_cs == 1 && r1 - r0 < 4) {          // decode-shaped forward
    // NOT column-blocked for L1, on purpose: measured on lm_head (1 x 576 x
    // 49152, 1 thread), 512/1024/2048/4096-column blocks ran 10.4/8.7/8.2/8.2 ms
    // vs 7.7 ms unblocked. The product is DRAM-bound (113 MB of b at ~15 GB/s);
    // blocking turns each b row's one long stream into K short strided runs,
    // which costs more than re-reading the output row from L2 saves.
    for (int r = r0; r < r1; r++)
      axpy_row(a + (size_t)r * a_rs, a_ps, b, b_ps, c + (size_t)r * ldc, P, ncols);
    return 1;
  }
  if (P <= 0) {                                      // empty sum: C = 0 / C += 0
    if (!accum)
      for (int r = r0; r < r1; r++)
        for (int j = 0; j < ncols; j++) c[(size_t)r * ldc + j] = 0.0f;
    return 1;
  }
  float *buf = (float*)malloc((size_t)MM_KC * MM_NC * sizeof(float));
  if (!buf) return 0;
  const int r4 = r0 + (r1 - r0) / 4 * 4, rl = r1 - r4;
  for (int pc = 0; pc < P; pc += MM_KC) {
    const int kc = (P - pc < MM_KC) ? P - pc : MM_KC;
    const int mode = accum ? MM_ADD : (pc == 0 ? MM_STORE : MM_CONT);
    const float *ap = a + (size_t)pc * a_ps;
    for (int jc = 0; jc < ncols; jc += MM_NC) {
      const int nc = (ncols - jc < MM_NC) ? ncols - jc : MM_NC;
      pack_panel(b + (size_t)pc * b_ps + (size_t)jc * b_cs, b_ps, b_cs, kc, nc, buf);
      for (int s = 0; s * 16 < nc; s++) {
        const float *bs = buf + (size_t)s * kc * 16;
        const int j = jc + s * 16, w = (nc - s * 16 < 16) ? nc - s * 16 : 16;
        // R, masked and mode must reach gemm_tile as constants, so every
        // combination is spelled out; each arm is its own specialized kernel.
#define TILE(r, R, MSK, MODE) gemm_tile(ap + (size_t)(r) * a_rs, a_rs, a_ps, bs, 16,   \
                                        c + (size_t)(r) * ldc + j, ldc, kc, R, 2, MSK, w, MODE)
#define TILES(MSK, MODE)                                                          \
        do {                                                                      \
          for (int r = r0; r < r4; r += 4) TILE(r, 4, MSK, MODE);                 \
          if (rl == 3) TILE(r4, 3, MSK, MODE);                                    \
          else if (rl == 2) TILE(r4, 2, MSK, MODE);                               \
          else if (rl == 1) TILE(r4, 1, MSK, MODE);                               \
        } while (0)
        if (w == 16) {
          if (mode == MM_STORE)     TILES(0, MM_STORE);
          else if (mode == MM_CONT) TILES(0, MM_CONT);
          else                      TILES(0, MM_ADD);
        } else {
          if (mode == MM_STORE)     TILES(1, MM_STORE);
          else if (mode == MM_CONT) TILES(1, MM_CONT);
          else                      TILES(1, MM_ADD);
        }
#undef TILES
#undef TILE
      }
    }
  }
  free(buf);
  return 1;
}

#endif  // MM_X86_SIMD

int matmul_simd_available(void) {
#if defined(MM_X86_SIMD)
  static int avail = -1;
  if (avail < 0) avail = cpu_has_avx2_fma();
  return avail;
#else
  return 0;
#endif
}

// Decided once, at the first matmul, in the calling thread: the pool's workers
// are handed an already-chosen kernel and never read this.
static int simd_on(void) {
  if (S_simd < 0) {
    const char *env = getenv("TF_SIMD");
    S_simd = (matmul_simd_available() && !(env && env[0] == '0')) ? 1 : 0;
  }
  return S_simd;
}

int matmul_simd_enabled(void) { return simd_on(); }

void matmul_set_simd(int on) { S_simd = (on && matmul_simd_available()) ? 1 : 0; }

const char *matmul_simd_name(void) { return simd_on() ? "avx2+fma" : "portable C"; }

// ===========================================================================
// Matrix multiply (forward) - C = A @ B, A:(M,K) B:(K,N) C:(M,N)
// ===========================================================================
typedef struct { const float *A, *B; float *C; int M, K, N; } MMFwd;

static int S_split = 0;   // matmul_debug_split (tests only)
void matmul_debug_split(int mode) { S_split = (mode < 0) ? 0 : mode; }

// ikj kernel for output rows [i0, i1) x columns [j0, j1): cache-friendly,
// auto-vectorizable. Each element is C = 0, then C += a*b for k ascending,
// whatever the row/column range -- so splitting rows or columns across threads
// changes no bits.
static void mm_fwd_block(const MMFwd *t, int i0, int i1, int j0, int j1) {
  const int K = t->K, N = t->N;
  for (int i = i0; i < i1; i++) {
    float *Ci = &t->C[(size_t)i * N];
    for (int j = j0; j < j1; j++) Ci[j] = 0.0f;
    const float *Ai = &t->A[(size_t)i * K];
    for (int k = 0; k < K; k++) {
      const float a = Ai[k];
      const float *Bk = &t->B[(size_t)k * N];
      for (int j = j0; j < j1; j++) Ci[j] += a * Bk[j];
    }
  }
}

// Column units [u0, u1) (MM_COL_UNIT columns each) -> column range [j0, j1).
static void col_range(const MMFwd *t, int u0, int u1, int *j0, int *j1) {
  const long lo = (long)u0 * MM_COL_UNIT, hi = (long)u1 * MM_COL_UNIT;
  *j0 = (lo < t->N) ? (int)lo : t->N;
  *j1 = (hi < t->N) ? (int)hi : t->N;
}

static void mm_fwd_rows(int i0, int i1, void *ctx) {
  const MMFwd *t = (const MMFwd*)ctx;
  mm_fwd_block(t, i0, i1, 0, t->N);
}

static void mm_fwd_cols(int u0, int u1, void *ctx) {
  const MMFwd *t = (const MMFwd*)ctx;
  int j0, j1; col_range(t, u0, u1, &j0, &j1);
  mm_fwd_block(t, 0, t->M, j0, j1);
}

#if defined(MM_X86_SIMD)
// AVX2 forward for output rows [i0, i1) x columns [j0, j1), from 0: the column
// range is just an offset into B and C (row strides stay N).
SIMD_FN static void mm_fwd_block_avx2(const MMFwd *t, int i0, int i1, int j0, int j1) {
  if (j1 <= j0 || i1 <= i0) return;
  if (!gemm_rows(i0, i1, j1 - j0, t->K, t->A, (size_t)t->K, 1, t->B + j0, (size_t)t->N, 1,
                 t->C + j0, (size_t)t->N, 0))
    mm_fwd_block(t, i0, i1, j0, j1);              // OOM: portable kernel instead
}

SIMD_FN static void mm_fwd_rows_avx2(int i0, int i1, void *ctx) {
  const MMFwd *t = (const MMFwd*)ctx;
  mm_fwd_block_avx2(t, i0, i1, 0, t->N);
}

SIMD_FN static void mm_fwd_cols_avx2(int u0, int u1, void *ctx) {
  const MMFwd *t = (const MMFwd*)ctx;
  int j0, j1; col_range(t, u0, u1, &j0, &j1);
  mm_fwd_block_avx2(t, 0, t->M, j0, j1);
}
#endif

void matmul(const Tensor *A, const Tensor *B, Tensor *C) {
  tf_check_matmul(A, B, C);
  const int M = A->shape[0], K = A->shape[1], N = B->shape[1];
  // Optional backend offload (backend.h). Returns 0 — "not mine" — for a CPU-only build,
  // when no device was found, when --no-gpu is in force, or when the problem is
  // too small to be worth the dispatch; the CPU path below is then unchanged.
  if (backend_matmul(A->data, B->data, C->data, M, K, N)) return;
  MMFwd t = { A->data, B->data, C->data, M, K, N };
  range_fn fn = mm_fwd_rows, fc = mm_fwd_cols;
#if defined(MM_X86_SIMD)
  if (simd_on()) { fn = mm_fwd_rows_avx2; fc = mm_fwd_cols_avx2; }
#endif
  const size_t macs = (size_t)M * N * K;
  if (S_split == 1) { fn(0, M, &t); return; }     // test hook: one serial call
  if (S_split >= 2) {                             // test hook: column pieces, serially
    const int units = (N + MM_COL_UNIT - 1) / MM_COL_UNIT;
    for (int i = 0; i < S_split; i++) {
      int lo, hi; chunk_of(units, i, S_split, &lo, &hi);
      fc(lo, hi, &t);
    }
    return;
  }
  // Few rows (decode: M = 1): splitting the rows would leave threads idle, so
  // split the columns into 64-wide units instead -- the int8 qacc_columns does
  // the same. Every output element is still produced by ONE thread with the same
  // FMA chain, so the result is bit-identical to one thread (selftest [38]).
  if (macs >= MM_COL_THRESHOLD && N >= 2 * MM_COL_UNIT) {
    const int nt = pool_threads();
    if (nt > 1 && M < MM_COL_ROWS * nt) {
      parallel_for((N + MM_COL_UNIT - 1) / MM_COL_UNIT, fc, &t);
      return;
    }
  }
  if (macs < MM_THREAD_THRESHOLD) { fn(0, M, &t); return; }
  parallel_for(M, fn, &t);
}

// Matrix Addition: C = A + B (element-wise)
void matadd(const Tensor *A, const Tensor *B, Tensor *C) {
  tf_check_same_size(A, B);
  tf_check_same_size(A, C);
  for (size_t i = 0; i < A->size; i++) {
    C->data[i] = A->data[i] + B->data[i];
  }
}

// Scale matrix: C = A * scalar
void matscale(const Tensor *A, float scalar, Tensor *C) {
  tf_check_same_size(A, C);
  for (size_t i = 0; i < A->size; i++) {
    C->data[i] = A->data[i] * scalar;
  }
}

// Row-broadcast bias add: out[i,j] = x[i,j] + bias[j]
void bias_add(const Tensor *x, const Tensor *bias, Tensor *out) {
  tf_check_bias_add(x, bias, out);
  const int M = x->shape[0];
  const int N = x->shape[1];
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      out->data[(size_t)i * N + j] = x->data[(size_t)i * N + j] + bias->data[j];
    }
  }
}

// ---------------------------------------------------------------------------
// Backward passes
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// The backward kernels were ~10x slower than the forward (measured: ~20 GF/s vs
// 113-216 GF/s) for one reason — the forward's inner loop is an AXPY over
// contiguous j, which the compiler auto-vectorizes, whereas both backward loops
// were floating-point REDUCTIONS, which it may not reassociate at -O2 and so
// cannot vectorize. Both are rewritten below to expose the same parallelism in
// plain portable C (no intrinsics, no target guards, works on WebAssembly too):
//   dB: restructured into an AXPY over j, accumulating i in the SAME order as
//       before -> auto-vectorizes AND stays bit-exact.
//   dA: an unavoidable dot product, given ACC_LANES independent accumulators so
//       the compiler can use one vector FMA chain. This reassociates the sum, so
//       dA is NOT bit-identical to the old scalar order (see matrix.h).
// ---------------------------------------------------------------------------
#define ACC_LANES 8

typedef struct { const Tensor *A, *B, *C; int M, K, N; } MMBwd;

// dA[i,k] += sum_j dC[i,j] * B[k,j]   (parallel over rows i)
static void mm_bwd_dA(int i0, int i1, void *ctx) {
  const MMBwd *t = (const MMBwd*)ctx;
  const int K = t->K, N = t->N;
  for (int i = i0; i < i1; i++) {
    const float *dCi = &t->C->grad[(size_t)i * N];
    float *dAi = &t->A->grad[(size_t)i * K];
    for (int k = 0; k < K; k++) {
      const float *Bk = &t->B->data[(size_t)k * N];
      float acc[ACC_LANES];
      for (int u = 0; u < ACC_LANES; u++) acc[u] = 0.0f;
      int j = 0;
      for (; j + ACC_LANES <= N; j += ACC_LANES)
        for (int u = 0; u < ACC_LANES; u++) acc[u] += dCi[j + u] * Bk[j + u];
      float sum = 0.0f;
      for (int u = 0; u < ACC_LANES; u++) sum += acc[u];
      for (; j < N; j++) sum += dCi[j] * Bk[j];
      dAi[k] += sum;
    }
  }
}

// dB[k,j] += sum_i A[i,k] * dC[i,j]   (parallel over rows k).
// Accumulates a whole row of dB at once so the inner loop is a contiguous AXPY.
// `i` is still summed in ascending order into a zeroed row, exactly as the
// previous per-element `sum` did, so the result is bit-identical to it.
static void mm_bwd_dB(int k0, int k1, void *ctx) {
  const MMBwd *t = (const MMBwd*)ctx;
  const int M = t->M, K = t->K, N = t->N;
  float *acc = (float*)malloc((size_t)N * sizeof(float));
  if (!acc) {                       // fall back to the in-place form if OOM
    for (int k = k0; k < k1; k++)
      for (int i = 0; i < M; i++) {
        const float a = t->A->data[(size_t)i * K + k];
        const float *dCi = &t->C->grad[(size_t)i * N];
        float *dBk = &t->B->grad[(size_t)k * N];
        for (int j = 0; j < N; j++) dBk[j] += a * dCi[j];
      }
    return;
  }
  for (int k = k0; k < k1; k++) {
    for (int j = 0; j < N; j++) acc[j] = 0.0f;
    for (int i = 0; i < M; i++) {
      const float a = t->A->data[(size_t)i * K + k];
      const float *dCi = &t->C->grad[(size_t)i * N];
      for (int j = 0; j < N; j++) acc[j] += a * dCi[j];
    }
    float *dBk = &t->B->grad[(size_t)k * N];
    for (int j = 0; j < N; j++) dBk[j] += acc[j];
  }
  free(acc);
}

#if defined(MM_X86_SIMD)
// AVX2 dA[i,k] += sum_j dC[i,j] * B[k,j]   (parallel over rows i).
// As a GEMM this is dC @ B^T; the micro-kernel's b operand is B^T, which
// pack_panel gathers straight out of B (b_ps = 1 walks j, b_cs = N walks k), so
// there is no transposed copy of B (it would be 113 MB for lm_head). The j sum
// is split into MM_KC-deep blocks that are ADDED to dA one after another; block
// boundaries depend on N alone, never on the row range, so threading stays
// bit-exact.
SIMD_FN static void mm_bwd_dA_avx2(int i0, int i1, void *ctx) {
  const MMBwd *t = (const MMBwd*)ctx;
  const int K = t->K, N = t->N;
  if (!gemm_rows(i0, i1, K, N, t->C->grad, (size_t)N, 1, t->B->data, 1, (size_t)N,
                 t->A->grad, (size_t)K, 1))
    mm_bwd_dA(i0, i1, ctx);                       // OOM: portable kernel instead
}

// AVX2 dB[k,j] += sum_i A[i,k] * dC[i,j]   (parallel over rows k).
// The a operand is A^T, read through strides (a_rs = 1 walks k, a_ps = K walks
// i): again no transpose. For M <= MM_KC this has exactly the portable kernel's
// structure -- a fresh sum over i, ascending, then added to dB.
SIMD_FN static void mm_bwd_dB_avx2(int k0, int k1, void *ctx) {
  const MMBwd *t = (const MMBwd*)ctx;
  const int M = t->M, K = t->K, N = t->N;
  if (!gemm_rows(k0, k1, N, M, t->A->data, 1, (size_t)K, t->C->grad, (size_t)N, 1,
                 t->B->grad, (size_t)N, 1))
    mm_bwd_dB(k0, k1, ctx);
}
#endif

// Backward of C = A @ B: dA += dC @ B^T, dB += A^T @ dC
void matmul_backward(const Tensor *A, const Tensor *B, const Tensor *C) {
  tf_check_matmul(A, B, C);
  const int M = A->shape[0], K = A->shape[1], N = B->shape[1];
  if (backend_matmul_backward(A->data, B->data, C->grad, A->grad, B->grad, M, K, N)) return;
  MMBwd t = { A, B, C, M, K, N };
  range_fn fa = mm_bwd_dA, fb = mm_bwd_dB;
#if defined(MM_X86_SIMD)
  if (simd_on()) { fa = mm_bwd_dA_avx2; fb = mm_bwd_dB_avx2; }
#endif
  if ((size_t)M * N * K < MM_THREAD_THRESHOLD) {
    fa(0, M, &t);
    fb(0, K, &t);
    return;
  }
  parallel_for(M, fa, &t);
  parallel_for(K, fb, &t);
}

int matmul_backward_dw_block(void) {
#if defined(MM_X86_SIMD)
  if (simd_on()) return MM_KC;     // mm_bwd_dB_avx2: gemm_rows over P = M, MM_KC deep
#endif
  return 0;                        // mm_bwd_dB: one ascending sum over all M rows
}

// Input-gradient half only: dA += dC @ B^T, leaving B->grad untouched. For a
// FROZEN weight (LoRA fine-tuning) dB is computed and then discarded, which is
// half the backward's work — and B->grad need not even exist.
void matmul_backward_dx(const Tensor *A, const Tensor *B, const Tensor *C) {
  tf_check_matmul(A, B, C);
  const int M = A->shape[0], K = A->shape[1], N = B->shape[1];
  if (backend_matmul_backward_dx(B->data, C->grad, A->grad, M, K, N)) return;
  MMBwd t = { A, B, C, M, K, N };
  range_fn fa = mm_bwd_dA;
#if defined(MM_X86_SIMD)
  if (simd_on()) fa = mm_bwd_dA_avx2;
#endif
  if ((size_t)M * N * K < MM_THREAD_THRESHOLD) { fa(0, M, &t); return; }
  parallel_for(M, fa, &t);
}

// ===========================================================================
// C = A @ B^T with B stored (N, K) -- a TIED output head (logits = hn @ wte^T)
// without a transposed copy of B (it would be 545 MB for Qwen2.5's 151936 x 896).
// ===========================================================================
typedef struct { const float *A, *B; float *C; int M, K, N; } MMNT;

// C[i,n] = sum_k A[i,k] * B[n,k] for rows [i0,i1) x columns [n0,n1). Each
// element is ONE dot product computed the same way whatever the row/column range,
// so any split (rows, columns, threads, decode M = 1 vs a full forward) gives the
// same bits. ACC_LANES partial sums let the compiler vectorize the reduction.
// B is walked in blocks of MM_NT_NB rows (~230 KB at K = 896: L2-resident) with
// every row of A inside, so a many-row forward streams B from DRAM once, not
// once per row. Blocking changes the visiting order only, never a sum.
#define MM_NT_NB 64
static float mm_nt_dot(const float *a, const float *b, int K) {
  float acc[ACC_LANES];
  for (int u = 0; u < ACC_LANES; u++) acc[u] = 0.0f;
  int k = 0;
  for (; k + ACC_LANES <= K; k += ACC_LANES)
    for (int u = 0; u < ACC_LANES; u++) acc[u] += a[k + u] * b[k + u];
  float sum = 0.0f;
  for (int u = 0; u < ACC_LANES; u++) sum += acc[u];
  for (; k < K; k++) sum += a[k] * b[k];
  return sum;
}

static void mm_nt_block(const MMNT *t, int i0, int i1, int n0, int n1) {
  const int K = t->K, N = t->N;
  for (int nb = n0; nb < n1; nb += MM_NT_NB) {
    const int ne = (n1 - nb < MM_NT_NB) ? n1 : nb + MM_NT_NB;
    for (int i = i0; i < i1; i++)
      for (int n = nb; n < ne; n++)
        t->C[(size_t)i * N + n] = mm_nt_dot(&t->A[(size_t)i * K], &t->B[(size_t)n * K], K);
  }
}

static void mm_nt_rows(int i0, int i1, void *ctx) {
  const MMNT *t = (const MMNT*)ctx;
  mm_nt_block(t, i0, i1, 0, t->N);
}

static void mm_nt_cols(int u0, int u1, void *ctx) {
  const MMNT *t = (const MMNT*)ctx;
  const long lo = (long)u0 * MM_COL_UNIT, hi = (long)u1 * MM_COL_UNIT;
  mm_nt_block(t, 0, t->M, (lo < t->N) ? (int)lo : t->N, (hi < t->N) ? (int)hi : t->N);
}

#if defined(MM_X86_SIMD)
// AVX2: every element is the SAME computation -- one 8-lane FMA chain over k in
// steps of 8 from zero, a fixed horizontal sum, then the K % 8 tail added in k
// order -- whether it is computed in the 4x2 register block or the 1x1 edge
// kernel. So the result does not depend on M (decode vs full forward), on the
// row/column range, or on threads. (Not bit-identical to the portable kernel.)
SIMD_FN static inline float nt_finish(__m256 acc, const float *a, const float *b, int k8, int K) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  float sum = _mm_cvtss_f32(s);
  for (int k = k8; k < K; k++)
    sum = _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a[k]), _mm_set_ss(b[k]), _mm_set_ss(sum)));
  return sum;
}

SIMD_FN static void mm_nt_block_avx2(const MMNT *t, int i0, int i1, int n0, int n1) {
  const int K = t->K, N = t->N, k8 = K & ~7;
  for (int nb = n0; nb < n1; nb += MM_NT_NB) {
    const int ne = (n1 - nb < MM_NT_NB) ? n1 : nb + MM_NT_NB;
    int i = i0;
    for (; i + 4 <= i1; i += 4) {
      const float *a0 = &t->A[(size_t)i * K], *a1 = a0 + K, *a2 = a1 + K, *a3 = a2 + K;
      int n = nb;
      for (; n + 2 <= ne; n += 2) {
        const float *b0 = &t->B[(size_t)n * K], *b1 = b0 + K;
        __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c10 = _mm256_setzero_ps(),
               c11 = _mm256_setzero_ps(), c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps(),
               c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
        for (int k = 0; k < k8; k += 8) {
          const __m256 vb0 = _mm256_loadu_ps(b0 + k), vb1 = _mm256_loadu_ps(b1 + k);
          __m256 va = _mm256_loadu_ps(a0 + k);
          c00 = _mm256_fmadd_ps(va, vb0, c00); c01 = _mm256_fmadd_ps(va, vb1, c01);
          va = _mm256_loadu_ps(a1 + k);
          c10 = _mm256_fmadd_ps(va, vb0, c10); c11 = _mm256_fmadd_ps(va, vb1, c11);
          va = _mm256_loadu_ps(a2 + k);
          c20 = _mm256_fmadd_ps(va, vb0, c20); c21 = _mm256_fmadd_ps(va, vb1, c21);
          va = _mm256_loadu_ps(a3 + k);
          c30 = _mm256_fmadd_ps(va, vb0, c30); c31 = _mm256_fmadd_ps(va, vb1, c31);
        }
        float *C0 = &t->C[(size_t)i * N + n];
        C0[0] = nt_finish(c00, a0, b0, k8, K);         C0[1] = nt_finish(c01, a0, b1, k8, K);
        C0[N] = nt_finish(c10, a1, b0, k8, K);         C0[N + 1] = nt_finish(c11, a1, b1, k8, K);
        C0[2 * N] = nt_finish(c20, a2, b0, k8, K);     C0[2 * N + 1] = nt_finish(c21, a2, b1, k8, K);
        C0[3 * N] = nt_finish(c30, a3, b0, k8, K);     C0[3 * N + 1] = nt_finish(c31, a3, b1, k8, K);
      }
      for (; n < ne; n++)
        for (int r = 0; r < 4; r++) {
          const float *a = &t->A[(size_t)(i + r) * K], *b = &t->B[(size_t)n * K];
          __m256 c = _mm256_setzero_ps();
          for (int k = 0; k < k8; k += 8) c = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), c);
          t->C[(size_t)(i + r) * N + n] = nt_finish(c, a, b, k8, K);
        }
    }
    for (; i < i1; i++) {                            // leftover rows (decode: all of them)
      const float *a = &t->A[(size_t)i * K];
      for (int n = nb; n < ne; n++) {
        const float *b = &t->B[(size_t)n * K];
        __m256 c = _mm256_setzero_ps();
        for (int k = 0; k < k8; k += 8) c = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), c);
        t->C[(size_t)i * N + n] = nt_finish(c, a, b, k8, K);
      }
    }
  }
}

SIMD_FN static void mm_nt_rows_avx2(int i0, int i1, void *ctx) {
  const MMNT *t = (const MMNT*)ctx;
  mm_nt_block_avx2(t, i0, i1, 0, t->N);
}

SIMD_FN static void mm_nt_cols_avx2(int u0, int u1, void *ctx) {
  const MMNT *t = (const MMNT*)ctx;
  const long lo = (long)u0 * MM_COL_UNIT, hi = (long)u1 * MM_COL_UNIT;
  mm_nt_block_avx2(t, 0, t->M, (lo < t->N) ? (int)lo : t->N, (hi < t->N) ? (int)hi : t->N);
}
#endif

void matmul_nt(const Tensor *A, const Tensor *B, Tensor *C) {
  TF_ASSERT(A->ndim == 2 && B->ndim == 2 && C->ndim == 2 && A->shape[1] == B->shape[1] &&
            C->shape[0] == A->shape[0] && C->shape[1] == B->shape[0]);
  const int M = A->shape[0], K = A->shape[1], N = B->shape[0];
  MMNT t = { A->data, B->data, C->data, M, K, N };
  range_fn fr = mm_nt_rows, fc = mm_nt_cols;
#if defined(MM_X86_SIMD)
  if (simd_on()) { fr = mm_nt_rows_avx2; fc = mm_nt_cols_avx2; }
#endif
  const size_t macs = (size_t)M * N * K;
  // Few rows (decode): split the output columns, as matmul does.
  if (macs >= MM_COL_THRESHOLD && N >= 2 * MM_COL_UNIT) {
    const int nt = pool_threads();
    if (nt > 1 && M < MM_COL_ROWS * nt) {
      parallel_for((N + MM_COL_UNIT - 1) / MM_COL_UNIT, fc, &t);
      return;
    }
  }
  if (macs < MM_THREAD_THRESHOLD) { fr(0, M, &t); return; }
  parallel_for(M, fr, &t);
}

void matmul_nt_backward(const Tensor *A, const Tensor *B, const Tensor *C, int want_dB) {
  TF_ASSERT(A->ndim == 2 && B->ndim == 2 && C->ndim == 2 && A->shape[1] == B->shape[1] &&
            C->shape[0] == A->shape[0] && C->shape[1] == B->shape[0]);
  const int M = A->shape[0], K = A->shape[1], N = B->shape[0];
  // Views over the existing buffers, so the ordinary kernels do the work.
  int sMN[2] = {M, N}, sNK[2] = {N, K}, sMK[2] = {M, K};
  // dA += dC @ B: a plain (M,N) @ (N,K) product into scratch, then added.
  {
    Tensor dC = { C->grad, NULL, sMN, 2, (size_t)M * N };
    Tensor Bv = { B->data, NULL, sNK, 2, (size_t)N * K };
    Tensor tmp = tensor_create_nograd(sMK, 2);
    matmul(&dC, &Bv, &tmp);
    for (size_t i = 0; i < tmp.size; i++) A->grad[i] += tmp.data[i];
    tensor_free(&tmp);
  }
  if (!want_dB) return;
  // dB[n,k] += sum_i dC[i,n] * A[i,k]: exactly matmul_backward's dB half for the
  // product X(M,N) @ Y(N,K) with X = dC and output gradient A -- reuse its kernels.
  Tensor Xv = { C->grad, NULL, sMN, 2, (size_t)M * N };   // plays A: data = dC
  Tensor Yg = { NULL, B->grad, sNK, 2, (size_t)N * K };   // plays B: grad = dB
  Tensor Zg = { NULL, A->data, sMK, 2, (size_t)M * K };   // plays C: grad = A
  MMBwd t = { &Xv, &Yg, &Zg, M, N, K };
  range_fn fb = mm_bwd_dB;
#if defined(MM_X86_SIMD)
  if (simd_on()) fb = mm_bwd_dB_avx2;
#endif
  if ((size_t)M * N * K < MM_THREAD_THRESHOLD) { fb(0, N, &t); return; }
  parallel_for(N, fb, &t);
}

// Backward of C = A + B: dA += dC, dB += dC
void matadd_backward(const Tensor *C, const Tensor *A, const Tensor *B) {
  tf_check_same_size(C, A);
  tf_check_same_size(C, B);
  for (size_t i = 0; i < C->size; i++) {
    A->grad[i] += C->grad[i];
    B->grad[i] += C->grad[i];
  }
}

// Backward of bias_add: bias_grad[j] += sum_i out->grad[i,j]
void bias_add_backward(const Tensor *out, const Tensor *bias_grad) {
  const int M = out->shape[0];
  const int N = out->shape[1];
  for (int j = 0; j < N; j++) {
    float sum = 0.0f;
    for (int i = 0; i < M; i++) {
      sum += out->grad[(size_t)i * N + j];
    }
    bias_grad->grad[j] += sum;
  }
}

// ---------------------------------------------------------------------------
// Threaded int8 column-accumulate (for the quantized decode path)
// ---------------------------------------------------------------------------
typedef struct { const signed char *qa, *qw; int K, N; int32_t *acc; } QAcc;

// Compute acc[j] = sum_k qa[k]*qw[k,j] for j in [j0, j1). Each column's full
// k-sum lives entirely in one thread, so no races and bit-identical to serial.
static void qacc_rows(int j0, int j1, void *ctx) {
  const QAcc *t = (const QAcc*)ctx;
  const int K = t->K, N = t->N;
  int32_t *acc = t->acc;
  for (int j = j0; j < j1; j++) acc[j] = 0;
  for (int k = 0; k < K; k++) {
    const int av = t->qa[k];
    const signed char *qr = &t->qw[(size_t)k * N];
    for (int j = j0; j < j1; j++) acc[j] += av * (int)qr[j];
  }
}

#if defined(MM_X86_SIMD)
// AVX2 int8 column strip: acc[j .. j+16U) += sum over k in [k0, k1), U = 1 or
// 4 (U=4 covers one 64-byte line of each weight row). Two k at a time: rows k
// and k+1 are widened to int16 and interleaved, so one vpmaddwd yields
// qa[k]*qw[k,c] + qa[k+1]*qw[k+1,c] per column in int32 -- 16 MACs per
// instruction, where the portable loop needs a vpmulld per 8. `first` starts
// from 0 instead of loading acc.
SIMD_INLINE void qacc_strip(const signed char *qa, const signed char *qw, int k0, int k1,
                            int N, int32_t *acc, int U, int first) {
  __m256i lo[4], hi[4];
  for (int u = 0; u < U; u++) {
    lo[u] = hi[u] = _mm256_setzero_si256();
    if (!first) lo[u] = _mm256_loadu_si256((const __m256i*)(acc + 16 * u));   // cols 0-7
    if (!first) hi[u] = _mm256_loadu_si256((const __m256i*)(acc + 16 * u + 8)); // cols 8-15
    // Re-shuffle to the madd layout (lo = cols 0-3 | 8-11, hi = 4-7 | 12-15).
    const __m256i l = lo[u], h = hi[u];
    lo[u] = _mm256_permute2x128_si256(l, h, 0x20);
    hi[u] = _mm256_permute2x128_si256(l, h, 0x31);
  }
  for (int k = k0; k < k1; k += 2) {
    const int two = (k + 1 < k1);                // odd count: the last step pairs with 0
    const uint32_t pk = (uint16_t)(int16_t)qa[k] |
                        ((uint32_t)(uint16_t)(int16_t)(two ? qa[k + 1] : 0) << 16);
    const __m256i pair = _mm256_set1_epi32((int)pk);
    const signed char *r0 = qw + (size_t)k * N;
    for (int u = 0; u < U; u++) {
      const __m256i w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(r0 + 16 * u)));
      const __m256i w1 = two ? _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(r0 + N + 16 * u)))
                             : _mm256_setzero_si256();
      // per 128-bit half: unpacklo -> columns 0-3 (| 8-11), unpackhi -> 4-7 (| 12-15)
      lo[u] = _mm256_add_epi32(lo[u], _mm256_madd_epi16(_mm256_unpacklo_epi16(w0, w1), pair));
      hi[u] = _mm256_add_epi32(hi[u], _mm256_madd_epi16(_mm256_unpackhi_epi16(w0, w1), pair));
    }
  }
  for (int u = 0; u < U; u++) {
    _mm256_storeu_si256((__m256i*)(acc + 16 * u),     _mm256_permute2x128_si256(lo[u], hi[u], 0x20));
    _mm256_storeu_si256((__m256i*)(acc + 16 * u + 8), _mm256_permute2x128_si256(lo[u], hi[u], 0x31));
  }
}

// The k loop is cut into MM_QKB-row blocks, and each block sweeps ALL of this
// thread's columns before the next. Without that, a strip walks every one of
// the K weight rows at stride N -- for lm_head (N = 49152) a new page per row,
// no prefetcher stream -- and the kernel measured 0.7x the portable one there.
// With it, only MM_QKB rows are live at once, each read front to back. The acc
// round trip per block is cheap (32 k per load/store instead of 1). All
// INTEGER arithmetic, so the result is bit-identical to the portable kernel
// whatever the order: this one needs no tolerance at all.
#define MM_QKB 32
SIMD_FN static void qacc_rows_avx2(int j0, int j1, void *ctx) {
  const QAcc *t = (const QAcc*)ctx;
  const int K = t->K, N = t->N;
  const int j16 = j0 + (j1 - j0) / 16 * 16;
  for (int kb = 0; kb < K; kb += MM_QKB) {
    const int ke = (K - kb < MM_QKB) ? K : kb + MM_QKB;
    int j = j0;
    if (kb == 0) {
      for (; j + 64 <= j16; j += 64) qacc_strip(t->qa, t->qw + j, kb, ke, N, t->acc + j, 4, 1);
      for (; j < j16; j += 16)       qacc_strip(t->qa, t->qw + j, kb, ke, N, t->acc + j, 1, 1);
    } else {
      for (; j + 64 <= j16; j += 64) qacc_strip(t->qa, t->qw + j, kb, ke, N, t->acc + j, 4, 0);
      for (; j < j16; j += 16)       qacc_strip(t->qa, t->qw + j, kb, ke, N, t->acc + j, 1, 0);
    }
  }
  for (int j = j16; j < j1; j++) {               // < 16 ragged columns
    int32_t s = 0;
    for (int k = 0; k < K; k++) s += (int32_t)t->qa[k] * (int32_t)t->qw[(size_t)k * N + j];
    t->acc[j] = s;
  }
}
#endif

void qacc_columns(const signed char *qa, const signed char *qw, int K, int N, int32_t *acc) {
  QAcc t = { qa, qw, K, N, acc };
  range_fn fn = qacc_rows;
#if defined(MM_X86_SIMD)
  if (simd_on()) fn = qacc_rows_avx2;
#endif
  if ((size_t)K * N < MM_THREAD_THRESHOLD) { fn(0, N, &t); return; }
  parallel_for(N, fn, &t);
}
