// matrix.c - Basic matrix operations (with a persistent thread pool for GEMMs)
#include "matrix.h"

#include <stdlib.h>

#include "check.h"
#include "gpu.h"

// ===========================================================================
// Persistent thread pool (internal to matrix.c).
//
// A fork-join pool: N-1 worker threads are created once (lazily) and reused for
// every parallel region — no per-call thread spawning (that was measured slower).
// `parallel_for(total, fn, ctx)` splits [0,total) into N contiguous chunks; the
// workers run chunks 0..N-2 and the calling thread runs chunk N-1, then joins.
//
// matmul / matmul_backward parallelize over OUTPUT ROWS (or dB rows), so every
// output element is produced by exactly one thread in the same summation order
// as the single-threaded kernel -> results are bit-identical. Small matmuls stay
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

typedef void (*range_fn)(int lo, int hi, void *ctx);

#if defined(POOL_DISABLED)

static void parallel_for(int total, range_fn fn, void *ctx) { fn(0, total, ctx); }

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

static void chunk_of(int total, int idx, int n, int *lo, int *hi) {
  int per = (total + n - 1) / n;
  int l = idx * per, h = l + per;
  if (l > total) l = total;
  if (h > total) h = total;
  *lo = l; *hi = h;
}

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

// ===========================================================================
// Matrix multiply (forward) - C = A @ B, A:(M,K) B:(K,N) C:(M,N)
// ===========================================================================
typedef struct { const float *A, *B; float *C; int K, N; } MMFwd;

// ikj kernel for output rows [i0, i1): cache-friendly, auto-vectorizable.
static void mm_fwd_rows(int i0, int i1, void *ctx) {
  const MMFwd *t = (const MMFwd*)ctx;
  const int K = t->K, N = t->N;
  for (int i = i0; i < i1; i++) {
    float *Ci = &t->C[(size_t)i * N];
    for (int j = 0; j < N; j++) Ci[j] = 0.0f;
    const float *Ai = &t->A[(size_t)i * K];
    for (int k = 0; k < K; k++) {
      const float a = Ai[k];
      const float *Bk = &t->B[(size_t)k * N];
      for (int j = 0; j < N; j++) Ci[j] += a * Bk[j];
    }
  }
}

void matmul(const Tensor *A, const Tensor *B, Tensor *C) {
  tf_check_matmul(A, B, C);
  const int M = A->shape[0], K = A->shape[1], N = B->shape[1];
  // Optional GPU offload (gpu.h). Returns 0 — "not mine" — for a CPU-only build,
  // when no device was found, when --no-gpu is in force, or when the problem is
  // too small to be worth the dispatch; the CPU path below is then unchanged.
  if (gpu_matmul(A->data, B->data, C->data, M, K, N)) return;
  MMFwd t = { A->data, B->data, C->data, K, N };
  if ((size_t)M * N * K < MM_THREAD_THRESHOLD) { mm_fwd_rows(0, M, &t); return; }
  parallel_for(M, mm_fwd_rows, &t);
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

// Backward of C = A @ B: dA += dC @ B^T, dB += A^T @ dC
void matmul_backward(const Tensor *A, const Tensor *B, const Tensor *C) {
  tf_check_matmul(A, B, C);
  const int M = A->shape[0], K = A->shape[1], N = B->shape[1];
  if (gpu_matmul_backward(A->data, B->data, C->grad, A->grad, B->grad, M, K, N)) return;
  MMBwd t = { A, B, C, M, K, N };
  if ((size_t)M * N * K < MM_THREAD_THRESHOLD) {
    mm_bwd_dA(0, M, &t);
    mm_bwd_dB(0, K, &t);
    return;
  }
  parallel_for(M, mm_bwd_dA, &t);
  parallel_for(K, mm_bwd_dB, &t);
}

// Input-gradient half only: dA += dC @ B^T, leaving B->grad untouched. For a
// FROZEN weight (LoRA fine-tuning) dB is computed and then discarded, which is
// half the backward's work — and B->grad need not even exist.
void matmul_backward_dx(const Tensor *A, const Tensor *B, const Tensor *C) {
  tf_check_matmul(A, B, C);
  const int M = A->shape[0], K = A->shape[1], N = B->shape[1];
  if (gpu_matmul_backward_dx(B->data, C->grad, A->grad, M, K, N)) return;
  MMBwd t = { A, B, C, M, K, N };
  if ((size_t)M * N * K < MM_THREAD_THRESHOLD) { mm_bwd_dA(0, M, &t); return; }
  parallel_for(M, mm_bwd_dA, &t);
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

void qacc_columns(const signed char *qa, const signed char *qw, int K, int N, int32_t *acc) {
  QAcc t = { qa, qw, K, N, acc };
  if ((size_t)K * N < MM_THREAD_THRESHOLD) { qacc_rows(0, N, &t); return; }
  parallel_for(N, qacc_rows, &t);
}
