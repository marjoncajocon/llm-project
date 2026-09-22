// backend.c - Backend registry, runtime selection, the matmul dispatch point, the
// row-major <-> column-major BLAS mapping and the device-BLAS core (backend.h),
// plus the legacy gpu.h facade.
//
// Always compiled, in every build. With no backend selected (the default build:
// no GPU backend compiled in, `auto` resolves to the CPU) every dispatch is a
// flag test and `return 0`, so matrix.c's arithmetic is exactly what it was.
#include "backend.h"
#include "gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Registry, in `auto` preference order. mock is last and never auto-selected.
// ---------------------------------------------------------------------------
#define N_BACKENDS 4
static TfBackendInfo g_info[N_BACKENDS];
static int g_info_n = 0;

static void registry_init(void) {
  if (g_info_n) return;
  const TfBackendInfo all[N_BACKENDS] = {
    { "cuda",   "make GPU=cuda (-DTF_CUDA)",          tf_backend_cuda()   },
    { "hip",    "make GPU=hip (-DTF_HIP)",            tf_backend_hip()    },
    { "opencl", "make GPU=opencl (-DTF_GPU; GPU=1)",  tf_backend_opencl() },
    { "mock",   "always compiled in",                 tf_backend_mock()   },
  };
  memcpy(g_info, all, sizeof(all));
  g_info_n = N_BACKENDS;
}

const TfBackendInfo *backend_info(int i) {
  registry_init();
  return (i >= 0 && i < g_info_n) ? &g_info[i] : NULL;
}

const TfBackend *backend_find(const char *name) {
  registry_init();
  for (int i = 0; i < g_info_n; i++)
    if (name && strcmp(g_info[i].name, name) == 0) return g_info[i].ops;
  return NULL;
}

static int known_name(const char *n) {
  if (!n) return 0;
  if (strcmp(n, "cpu") == 0 || strcmp(n, "auto") == 0) return 1;
  registry_init();
  for (int i = 0; i < g_info_n; i++)
    if (strcmp(g_info[i].name, n) == 0) return 1;
  return 0;
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------
static char g_request[32] = "";          // CLI request ("" = none: env, then auto)
static int  g_resolved = 0;
static const TfBackend *g_active = NULL;
static int  g_enabled = 1;
static char g_msg[320] = "";
static int  g_min_m = 8;
static long long g_min_macs = (1LL << 22);
static long g_handled = 0, g_declined = 0;

static long env_long(const char *k, long dflt) {
  const char *v = getenv(k);
  if (!v || !*v) return dflt;
  char *end = NULL;
  long n = strtol(v, &end, 10);
  return (end && end != v && n > 0) ? n : dflt;
}

int backend_request(const char *name) {
  if (!name || !*name) { g_request[0] = 0; g_resolved = 0; return 1; }   // back to env / auto
  if (!known_name(name)) {
    fprintf(stderr, "backend: unknown backend '%s' (expected cpu|opencl|cuda|hip|mock|auto)\n",
            name ? name : "(null)");
    return 0;
  }
  snprintf(g_request, sizeof(g_request), "%s", name);
  g_resolved = 0;
  return 1;
}

void backend_reset(void) { g_resolved = 0; g_active = NULL; g_msg[0] = 0; }

static void fallback(const char *fmt, const char *a, const char *b) {
  snprintf(g_msg, sizeof(g_msg), fmt, a, b);
  fprintf(stderr, "backend: %s\n", g_msg);
}

// The active backend's own gates, then the env overrides (real GPUs only: the
// env vars predate the layer and mean "retune the GPU", never "gate the mock").
static void set_gates(const TfBackend *b) {
  TfCaps c;
  memset(&c, 0, sizeof(c));
  b->caps(&c);
  g_min_m = c.min_m;
  g_min_macs = c.min_macs;
  if (b->is_gpu) {
    g_min_m = (int)env_long("TF_GPU_MIN_M", g_min_m);
    g_min_macs = env_long("TF_GPU_MIN_MACS", (long)g_min_macs);
  }
}

static void activate(const TfBackend *b) { g_active = b; set_gates(b); }

static void resolve(void) {
  if (g_resolved) return;
  g_resolved = 1;
  g_active = NULL;
  g_msg[0] = 0;

  const char *req = g_request;
  if (!*req) {
    const char *e = getenv("TF_BACKEND");
    if (e && *e) {
      if (known_name(e)) req = e;
      else fallback("TF_BACKEND='%s' is not a backend (cpu|opencl|cuda|hip|mock|auto)%s -- "
                    "using auto", e, "");
    }
  }
  if (!*req) req = "auto";
  if (strcmp(req, "cpu") == 0) return;

  registry_init();
  if (strcmp(req, "auto") == 0) {
    // First compiled-in real GPU that is actually usable; silently the CPU
    // otherwise (a GPU build on a GPU-less box is a supported state, not an error).
    for (int i = 0; i < g_info_n; i++) {
      const TfBackend *b = g_info[i].ops;
      if (b && b->is_gpu && b->init()) { activate(b); return; }
    }
    return;
  }
  const TfBackend *b = backend_find(req);
  if (!b) {
    for (int i = 0; i < g_info_n; i++)
      if (strcmp(g_info[i].name, req) == 0)
        fallback("'%s' is not compiled into this binary (rebuild with %s) -- using the CPU",
                 req, g_info[i].build_hint);
    return;
  }
  if (!b->init()) {
    fallback("'%s' unavailable: %s -- using the CPU", req, b->device_name());
    return;
  }
  activate(b);
}

const TfBackend *backend_active(void) { resolve(); return g_active; }
const char *backend_active_name(void) { resolve(); return g_active ? g_active->name : "cpu"; }
const char *backend_last_message(void) { return g_msg; }
void backend_set_enabled(int on) { g_enabled = on ? 1 : 0; }
int backend_is_enabled(void) { return g_enabled; }

void backend_gates(int *min_m, long long *min_macs) {
  resolve();
  if (min_m) *min_m = g_min_m;
  if (min_macs) *min_macs = g_min_macs;
}

void backend_set_gates(int min_m, long long min_macs) {
  resolve();
  g_min_m = min_m;
  g_min_macs = min_macs;
}

void backend_shutdown_all(void) {
  registry_init();
  for (int i = 0; i < g_info_n; i++)
    if (g_info[i].ops) g_info[i].ops->shutdown();
}

void backend_stats(long *handled, long *declined) {
  if (handled) *handled = g_handled;
  if (declined) *declined = g_declined;
}
void backend_stats_reset(void) { g_handled = 0; g_declined = 0; }

// ---------------------------------------------------------------------------
// The dispatch point
// ---------------------------------------------------------------------------
// Should the active backend take this problem at all? The old gpu.c gpu_takes()
// in the same order: enabled -> a usable device -> the backend's size gates
// (OpenCL/CUDA/HIP: M >= 8 and M*N*K >= 2^22 unless retuned; mock: everything).
static const TfBackend *takes(int M, int K, int N) {
  if (!g_enabled) return NULL;
  resolve();
  const TfBackend *b = g_active;
  if (!b || !b->available()) return NULL;
  if (M < g_min_m) return NULL;
  if ((long long)M * N * K < g_min_macs) return NULL;
  return b;
}

int backend_matmul(const float *A, const float *B, float *C, int M, int K, int N) {
  const TfBackend *b = takes(M, K, N);
  if (b && b->sgemm(A, B, C, M, K, N)) { g_handled++; return 1; }
  g_declined++;
  return 0;
}

int backend_matmul_backward(const float *A, const float *B, const float *dC,
                            float *dA, float *dB, int M, int K, int N) {
  const TfBackend *b = takes(M, K, N);
  // Both halves must succeed together: a partial result would leave dA updated
  // and dB not, which the CPU fallback would then double-count.
  if (!b || !b->sgemm_nt(B, dC, dA, M, K, N)) { g_declined++; return 0; }
  if (!b->sgemm_tn(A, dC, dB, M, K, N)) {
    fprintf(stderr, "gpu: dB half failed after dA succeeded — gradients for this "
                    "call are INCOMPLETE; aborting rather than training on them\n");
    exit(1);
  }
  g_handled++;
  return 1;
}

int backend_matmul_backward_dx(const float *B, const float *dC, float *dA,
                               int M, int K, int N) {
  const TfBackend *b = takes(M, K, N);
  if (b && b->sgemm_nt(B, dC, dA, M, K, N)) { g_handled++; return 1; }
  g_declined++;
  return 0;
}

// ---------------------------------------------------------------------------
// Row-major <-> column-major mapping (the derivation is in backend.h)
// ---------------------------------------------------------------------------
int tf_colmajor_fwd(TfColMajorSgemm g, void *ctx, const float *dA, const float *dB,
                    float *dC, int M, int K, int N) {
  // C^T(NxM) = B^T(NxK) @ A^T(KxM): the row-major buffers ARE B^T and A^T.
  return g(ctx, 0, 0, N, M, K, 1.0f, dB, N, dA, K, 0.0f, dC, N);
}

int tf_colmajor_nt(TfColMajorSgemm g, void *ctx, const float *dB, const float *ddC,
                   float *ddA, int M, int K, int N) {
  // dA^T(KxM) += B(KxN) @ dC^T(NxM): B's buffer is B^T (NxK, ld N) -> op T.
  return g(ctx, 1, 0, K, M, N, 1.0f, dB, N, ddC, N, 1.0f, ddA, K);
}

int tf_colmajor_tn(TfColMajorSgemm g, void *ctx, const float *dA, const float *ddC,
                   float *ddB, int M, int K, int N) {
  // dB^T(NxK) += dC^T(NxM) @ A(MxK): A's buffer is A^T (KxM, ld K) -> op T.
  return g(ctx, 0, 1, N, K, M, 1.0f, ddC, N, dA, K, 1.0f, ddB, N);
}

// Column-major reference SGEMM with cuBLAS's validity rules.
int tf_fake_cublas_sgemm(void *ctx, int transa, int transb, int m, int n, int k,
                         float alpha, const float *A, int lda,
                         const float *B, int ldb, float beta, float *C, int ldc) {
  (void)ctx;
  if ((transa != 0 && transa != 1) || (transb != 0 && transb != 1)) return 7;  // INVALID_VALUE
  if (m < 0 || n < 0 || k < 0) return 7;
  const int arows = transa ? k : m;          // A as STORED has arows rows
  const int brows = transb ? n : k;
  if (lda < (arows > 1 ? arows : 1)) return 7;
  if (ldb < (brows > 1 ? brows : 1)) return 7;
  if (ldc < (m > 1 ? m : 1)) return 7;
  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++) {
      float acc = 0.0f;
      for (int p = 0; p < k; p++) {
        const float a = transa ? A[p + (size_t)i * lda] : A[i + (size_t)p * lda];
        const float b = transb ? B[j + (size_t)p * ldb] : B[p + (size_t)j * ldb];
        acc += a * b;
      }
      float *c = &C[i + (size_t)j * ldc];
      *c = (beta == 0.0f) ? alpha * acc : alpha * acc + beta * *c;   // beta 0: C never read
    }
  return 0;
}

// ---------------------------------------------------------------------------
// Device-BLAS core
// ---------------------------------------------------------------------------
static float *fptr(const TfBuf *b) { return (float*)((char*)b->dev + b->offset); }

// The slot for this host pointer, at least `bytes` long; *resident tells the
// caller it may skip the upload. Same policy as the OpenCL backend: cached by
// pointer, contents never assumed (unless registered). When the cache is full
// the buffer is a per-call temporary, freed by end_call().
static TfBuf *dev_for(TfDevBlas *d, const void *host, size_t bytes, int *resident) {
  if (resident) *resident = 0;
  for (int i = 0; i < d->nslots; i++) {
    if (d->slot[i].host != host) continue;
    if (d->slot[i].buf.bytes >= bytes) {
      if (resident) *resident = d->slot[i].resident;
      return &d->slot[i].buf;
    }
    d->free_(d->slot[i].buf.dev);                 // same pointer, grew: reallocate
    d->slot[i].resident = 0;                      // (and it is no longer resident)
    d->slot[i].buf.dev = d->alloc(bytes);
    d->slot[i].buf.bytes = d->slot[i].buf.dev ? bytes : 0;
    return d->slot[i].buf.dev ? &d->slot[i].buf : NULL;
  }
  void *p = d->alloc(bytes);
  if (!p) return NULL;
  TfBuf nb = { p, 0, bytes, 0, TF_F32 };
  if (d->nslots < TF_DEVBLAS_SLOTS) {
    d->slot[d->nslots].host = host;
    d->slot[d->nslots].buf = nb;
    d->slot[d->nslots].resident = 0;
    return &d->slot[d->nslots++].buf;
  }
  if (d->ntmp < 3) { d->tmp[d->ntmp] = nb; return &d->tmp[d->ntmp++]; }
  d->free_(p);
  return NULL;
}

static void end_call(TfDevBlas *d) {
  for (int i = 0; i < d->ntmp; i++) d->free_(d->tmp[i].dev);
  d->ntmp = 0;
}

static int fail_call(TfDevBlas *d, const char *where) {
  end_call(d);
  fprintf(stderr, "gpu: %s %s failed — falling back to the CPU for the rest of this run\n",
          d->tag, where);
  return 0;
}

static int up(TfDevBlas *d, TfBuf *b, const void *host, size_t bytes) {
  d->uploads++;
  d->upload_bytes += bytes;
  return d->h2d(fptr(b), host, bytes);
}

// Upload an INPUT operand unless it is a registered (resident) weight.
static int up_in(TfDevBlas *d, TfBuf *b, int resident, const void *host, size_t bytes) {
  return resident ? 0 : up(d, b, host, bytes);
}

int tf_devblas_fwd(TfDevBlas *d, const float *A, const float *B, float *C, int M, int K, int N) {
  const size_t bA = (size_t)M * K * sizeof(float);
  const size_t bB = (size_t)K * N * sizeof(float);
  const size_t bC = (size_t)M * N * sizeof(float);
  int rA, rB;
  TfBuf *xA = dev_for(d, A, bA, &rA), *xB = dev_for(d, B, bB, &rB), *xC = dev_for(d, C, bC, NULL);
  if (!xA || !xB || !xC) return fail_call(d, "buffer allocation");
  if (up_in(d, xA, rA, A, bA) || up_in(d, xB, rB, B, bB)) return fail_call(d, "upload");
  // C is NOT uploaded: beta = 0, so the BLAS never reads it.
  if (tf_colmajor_fwd(d->gemm, d->gemm_ctx, fptr(xA), fptr(xB), fptr(xC), M, K, N))
    return fail_call(d, "sgemm");
  if (d->d2h(C, fptr(xC), bC)) return fail_call(d, "download");
  end_call(d);
  return 1;
}

int tf_devblas_nt(TfDevBlas *d, const float *B, const float *dC, float *dA, int M, int K, int N) {
  const size_t bB = (size_t)K * N * sizeof(float);
  const size_t bC = (size_t)M * N * sizeof(float);
  const size_t bA = (size_t)M * K * sizeof(float);
  int rB, rC;
  TfBuf *xB = dev_for(d, B, bB, &rB), *xC = dev_for(d, dC, bC, &rC), *xA = dev_for(d, dA, bA, NULL);
  if (!xA || !xB || !xC) return fail_call(d, "buffer allocation");
  // dA is read-modify-write (beta = 1): its host contents always go up first.
  if (up_in(d, xB, rB, B, bB) || up_in(d, xC, rC, dC, bC) || up(d, xA, dA, bA))
    return fail_call(d, "upload");
  if (tf_colmajor_nt(d->gemm, d->gemm_ctx, fptr(xB), fptr(xC), fptr(xA), M, K, N))
    return fail_call(d, "sgemm (dA)");
  if (d->d2h(dA, fptr(xA), bA)) return fail_call(d, "download");
  end_call(d);
  return 1;
}

int tf_devblas_tn(TfDevBlas *d, const float *A, const float *dC, float *dB, int M, int K, int N) {
  const size_t bA = (size_t)M * K * sizeof(float);
  const size_t bC = (size_t)M * N * sizeof(float);
  const size_t bB = (size_t)K * N * sizeof(float);
  int rA, rC;
  TfBuf *xA = dev_for(d, A, bA, &rA), *xC = dev_for(d, dC, bC, &rC), *xB = dev_for(d, dB, bB, NULL);
  if (!xA || !xB || !xC) return fail_call(d, "buffer allocation");
  if (up_in(d, xA, rA, A, bA) || up_in(d, xC, rC, dC, bC) || up(d, xB, dB, bB))
    return fail_call(d, "upload");
  if (tf_colmajor_tn(d->gemm, d->gemm_ctx, fptr(xA), fptr(xC), fptr(xB), M, K, N))
    return fail_call(d, "sgemm (dB)");
  if (d->d2h(dB, fptr(xB), bB)) return fail_call(d, "download");
  end_call(d);
  return 1;
}

// ---- L2: explicit device buffers ----
TfBuf *tf_devblas_buf_alloc(TfDevBlas *d, size_t bytes, TfDType dtype) {
  TfBuf *b = (TfBuf*)calloc(1, sizeof(TfBuf));
  if (!b) return NULL;
  b->dev = d->alloc(bytes ? bytes : 1);
  if (!b->dev) { free(b); return NULL; }
  b->bytes = bytes;
  b->dtype = dtype;
  return b;
}

void tf_devblas_buf_free(TfDevBlas *d, TfBuf *b) {
  if (!b) return;
  if (b->dev) d->free_(b->dev);
  free(b);
}

int tf_devblas_buf_upload(TfDevBlas *d, TfBuf *dst, size_t off, const void *host, size_t bytes) {
  if (!dst || off + bytes > dst->bytes) return 0;
  d->uploads++;
  d->upload_bytes += bytes;
  return d->h2d((char*)dst->dev + dst->offset + off, host, bytes) == 0;
}

int tf_devblas_buf_download(TfDevBlas *d, void *host, const TfBuf *src, size_t off, size_t bytes) {
  if (!src || off + bytes > src->bytes) return 0;
  return d->d2h(host, (const char*)src->dev + src->offset + off, bytes) == 0;
}

int tf_devblas_gemm(TfDevBlas *d, int transa, int transb, int m, int n, int k,
                    float alpha, const TfBuf *A, int lda, const TfBuf *B, int ldb,
                    float beta, TfBuf *C, int ldc, TfDType ab_type, TfDType c_type) {
  // TODO(dtypes): fp16/bf16 via cublasGemmEx / hipblasGemmEx; int8/4-bit GEMV.
  if (ab_type != TF_F32 || c_type != TF_F32) return 0;
  if (!A || !B || !C) return 0;
  // Bounds: the last element each operand touches must lie inside its buffer.
  const size_t acols = (size_t)(transa ? m : k), bcols = (size_t)(transb ? k : n);
  const size_t arows = (size_t)(transa ? k : m), brows = (size_t)(transb ? n : k);
  if (m > 0 && n > 0 && k > 0) {
    if (((acols - 1) * lda + arows) * sizeof(float) > A->bytes) return 0;
    if (((bcols - 1) * ldb + brows) * sizeof(float) > B->bytes) return 0;
  }
  if (m > 0 && n > 0 && (((size_t)(n - 1) * ldc + m) * sizeof(float) > C->bytes)) return 0;
  return d->gemm(d->gemm_ctx, transa, transb, m, n, k, alpha, fptr(A), lda, fptr(B), ldb,
                 beta, fptr(C), ldc) == 0;
}

// ---- L3: weight residency ----
const TfBuf *tf_devblas_register(TfDevBlas *d, const void *host, size_t bytes, TfDType dtype) {
  if (dtype != TF_F32 || !host || !bytes) return NULL;       // TODO(dtypes)
  int res = 0;
  TfBuf *b = dev_for(d, host, bytes, &res);
  if (!b) return NULL;
  // A per-call temporary (cache full) cannot be resident: give it back.
  int slot = -1;
  for (int i = 0; i < d->nslots; i++) if (&d->slot[i].buf == b) slot = i;
  if (slot < 0) { end_call(d); return NULL; }
  if (!res && up(d, b, host, bytes)) return NULL;
  d->slot[slot].resident = 1;
  return b;
}

void tf_devblas_invalidate(TfDevBlas *d, const void *host) {
  for (int i = 0; i < d->nslots; i++)
    if (!host || d->slot[i].host == host) d->slot[i].resident = 0;
}

void tf_devblas_release(TfDevBlas *d) {
  end_call(d);
  for (int i = 0; i < d->nslots; i++)
    if (d->slot[i].buf.dev) d->free_(d->slot[i].buf.dev);
  d->nslots = 0;
}

// ---------------------------------------------------------------------------
// The legacy gpu.h facade: "the active REAL GPU backend".
// ---------------------------------------------------------------------------
static const TfBackend *active_gpu(void) {
  const TfBackend *b = backend_active();
  return (b && b->is_gpu) ? b : NULL;
}

int gpu_compiled_in(void) {
  registry_init();
  for (int i = 0; i < g_info_n; i++)
    if (g_info[i].ops && g_info[i].ops->is_gpu) return 1;
  return 0;
}

int gpu_available(void) {
  const TfBackend *b = active_gpu();
  return b ? b->available() : 0;
}

const char *gpu_device_name(void) {
  const TfBackend *b = active_gpu();
  if (b) return b->device_name();
  // No active GPU: explain with the first compiled-in one (why it is not there).
  registry_init();
  for (int i = 0; i < g_info_n; i++)
    if (g_info[i].ops && g_info[i].ops->is_gpu) return g_info[i].ops->device_name();
  return "none (built without a GPU backend)";
}

static TfCaps active_caps(void) {
  TfCaps c;
  memset(&c, 0, sizeof(c));
  const TfBackend *b = active_gpu();
  if (b) b->caps(&c);
  return c;
}

int gpu_compute_units(void) { return active_caps().compute_units; }
size_t gpu_global_mem(void) { return active_caps().mem_bytes; }
void gpu_set_enabled(int on) { backend_set_enabled(on); }
int gpu_is_enabled(void) { return backend_is_enabled(); }
void gpu_shutdown(void) { backend_shutdown_all(); }
