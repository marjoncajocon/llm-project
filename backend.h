// backend.h - The compute-backend layer: one ops table per backend, one runtime
// selection, and ONE dispatch point (matrix.c's matmul / matmul_backward /
// matmul_backward_dx).
//
// BACKENDS
//   cpu     no backend at all -- matrix.c's own kernels (AVX2 / portable). Always.
//   opencl  backend_opencl.c, compiled in with -DTF_GPU  (make GPU=opencl, GPU=1)
//   cuda    backend_cuda.c,   compiled in with -DTF_CUDA (make GPU=cuda)
//   hip     backend_hip.c,    compiled in with -DTF_HIP  (make GPU=hip)
//   mock    backend_mock.c,   ALWAYS compiled in, never auto-selected. It runs the
//           exact CUDA/HIP host path (device-BLAS core below: buffers, upload,
//           column-major BLAS, download) against a fake host-memory "device" and
//           a column-major reference SGEMM ("fake cuBLAS"), so the plumbing, the
//           fallbacks and the row/column-major mapping are testable on a machine
//           without any GPU (selftest part [49]).
//
// No vendor SDK, header or import library is needed for ANY of them: the real
// backends open their driver/BLAS library at runtime (dlopen / LoadLibrary) and
// declare the few types they use locally. A binary built with every backend runs
// fine on a machine with none of the libraries -- it falls back to the CPU.
//
// SELECTION (resolved lazily, once, at the first matmul or query)
//   CLI `--backend X` (or `--backend=X`) > env TF_BACKEND=X > default "auto".
//   X = cpu | opencl | cuda | hip | mock | auto
//   auto   : the first COMPILED-IN and AVAILABLE of cuda, hip, opencl -- silently
//            the CPU if none is (the pre-existing behaviour of a GPU=1 build).
//   named  : that backend if compiled in and available; otherwise ONE clear line
//            on stderr saying why, and the CPU carries every matmul.
//   `--no-gpu` / `--gpu` are kept as aliases: they toggle backend_set_enabled(),
//   i.e. whether the selected backend is used at all (--no-gpu == CPU).
//
// DISPATCH CONTRACT (unchanged from the old gpu.h): every compute call returns 1
// if a backend handled it, 0 if the caller must run its CPU path. With no backend
// selected the call is one flag test and a return 0, so the CPU build's
// arithmetic is bit-identical to what it was.
//
// GROWTH PLAN (docs/ACCELERATORS.md section 4). The table has three levels:
//   L1  host-pointer GEMM trio (sgemm / _nt / _tn) -- what matrix.c calls today.
//   L2  opaque device buffers + a cuBLAS-shaped GEMM on them (transA/transB,
//       ld*, alpha/beta, dtypes) + a stream argument (NULL = default stream).
//   L3  weight residency: register_weight / invalidate.
// L2/L3 are IMPLEMENTED for the device-BLAS core (cuda, hip, mock) and tested on
// the mock; OpenCL provides L1 only for now (NULL hooks). Everything else the
// roadmap lists (non-default streams, strided-batched GEMM, ops beyond GEMM,
// kernel launch, graph capture, low-precision dtypes, multi-device, LRU budget)
// is RESERVED below with TODOs, so it can be added without a redesign.
#ifndef BACKEND_H
#define BACKEND_H

#include <stddef.h>

// Element types. Only TF_F32 is implemented anywhere; the rest are reserved so
// the GEMM signature does not change when fp16/bf16/int8/4-bit arrive.
typedef enum { TF_F32 = 0, TF_F16 = 1, TF_BF16 = 2, TF_I8 = 3, TF_Q4 = 4 } TfDType;

// An opaque device buffer. Host pointers are NOT the identity of device memory
// (residency, PJRT and Metal all need real handles), so everything at L2/L3
// takes one of these. Treat the fields as private to the backend that made it.
typedef struct TfBuf {
  void  *dev;        // backend device pointer (CUdeviceptr / hipDeviceptr / host
                     // memory for the mock) -- never dereference on the host
  size_t offset;     // byte offset of this view into `dev` (views: TODO, always 0)
  size_t bytes;
  int    device;     // device index (multi-device reserved: always 0)
  TfDType dtype;
} TfBuf;

// A queue/stream. NULL = the backend's default stream, which is all that exists
// today (every call is synchronous with respect to the host, as before).
// TODO(streams): stream_create/destroy, upload_async, event_record/wait.
typedef struct TfStream TfStream;

// What a backend/device can do, and when it is worth using. The dispatch gates
// come from here (per backend) instead of global defaults; the env vars
// TF_GPU_MIN_M / TF_GPU_MIN_MACS still override them for real GPUs.
typedef struct TfCaps {
  size_t mem_bytes;        // device memory (0 = unknown)
  size_t max_alloc;        // largest single allocation (0 = unknown)
  int    compute_units;    // SMs / CUs (0 = unknown)
  int    unified_memory;   // 1 = shares host memory (iGPU/Apple): copies could be skipped
  int    fp16, bf16, int8_dot, tensor_cores;   // reserved capability bits (all 0 today)
  int    deterministic;    // 1 = same inputs give bit-identical outputs run to run
  int    n_devices;        // multi-device reserved: we always use device 0
  int    min_m;            // dispatch gate: minimum rows M (decode M=1 stays on CPU)
  long long min_macs;      // dispatch gate: minimum M*N*K
} TfCaps;

typedef struct TfBackend {
  const char *name;          // "opencl", "cuda", "hip", "mock"
  const char *build_hint;    // how to compile it in, e.g. "make GPU=cuda (-DTF_CUDA)"
  int is_gpu;                // a real device (counts for `auto` and gpu_compiled_in)

  // ---- lifecycle ----
  // Probe once (idempotent, cheap to call again); 1 = usable. Never aborts: a
  // missing library is a 0 plus an explanation in device_name().
  int  (*init)(void);
  void (*shutdown)(void);    // free device memory and handles
  int  (*available)(void);   // init() && no device error since
  const char *(*device_name)(void);   // device description, or why there is none
  void (*caps)(TfCaps *out);          // filled after init(); zeros where unknown

  // ---- L1: host-pointer GEMMs (the dispatch today). Row-major, contiguous:
  // A (MxK), B (KxN), C/dC (MxN). Return 1 if handled, 0 = run the CPU path. ----
  int (*sgemm)(const float *A, const float *B, float *C, int M, int K, int N); // C  = A @ B
  int (*sgemm_nt)(const float *B, const float *dC, float *dA, int M, int K, int N); // dA += dC @ B^T
  int (*sgemm_tn)(const float *A, const float *dC, float *dB, int M, int K, int N); // dB += A^T @ dC

  // ---- L2: device buffers + cuBLAS-shaped GEMM (NULL = not supported) ----
  TfBuf *(*buf_alloc)(size_t bytes, TfDType dtype);
  void   (*buf_free)(TfBuf *b);
  int    (*buf_upload)(TfBuf *dst, size_t off, const void *host, size_t bytes, TfStream *s);
  int    (*buf_download)(void *host, const TfBuf *src, size_t off, size_t bytes, TfStream *s);
  // C = alpha*op(A)@op(B) + beta*C, COLUMN-major (cuBLAS semantics; see the
  // mapping notes below for how row-major callers use it). trans: 0 = N, 1 = T.
  // ab_type/c_type/compute: only TF_F32 today. Returns 1 on success.
  int (*gemm)(TfStream *s, int transa, int transb, int m, int n, int k,
              float alpha, const TfBuf *A, int lda, const TfBuf *B, int ldb,
              float beta, TfBuf *C, int ldc, TfDType ab_type, TfDType c_type);
  int (*synchronize)(TfStream *s);    // wait for everything queued on s

  // ---- L3: weight residency ----
  // Declare host memory immutable until invalidate(): it is uploaded once and
  // every later L1 GEMM that reads it skips the upload. This is what will make
  // GPU prompting/decode worthwhile (today every call re-uploads the weights).
  // It CANNOT be inferred from pointers (attention passes activations as B), so
  // the model code must call it explicitly -- nothing does yet.
  // TODO(residency): memory budget + LRU eviction; call sites in llama.c.
  const TfBuf *(*register_weight)(const void *host, size_t bytes, TfDType dtype);
  // Call invalidate BEFORE mutating or freeing registered memory (the allocation
  // cache is keyed by host address, so a reused address would look resident).
  void (*invalidate)(const void *host);   // after an optimizer step / LoRA merge; NULL = all

  // ---- reserved (docs/ACCELERATORS.md section 4), all NULL today ----
  // TODO: stream_create/stream_destroy/upload_async/event_*, buf_copy + views,
  // gemm_strided_batched (attention heads), rmsnorm/rope/softmax/attention/
  // swiglu/add/embed/xent/adamw with capability bits, launch(kernel,grid,args)
  // for string-compiled kernels (OpenCL/NVRTC/hipRTC), begin/end_capture/replay.
  void *reserved[12];
} TfBackend;

// ---- registry ---------------------------------------------------------------
// Every backend this binary knows about, compiled in or not, in `auto` order
// (cuda, hip, opencl, mock). backend_info() returns NULL past the end.
typedef struct {
  const char *name;
  const char *build_hint;
  const TfBackend *ops;      // NULL = not compiled into this binary
} TfBackendInfo;
const TfBackendInfo *backend_info(int i);
const TfBackend *backend_find(const char *name);   // compiled-in backend or NULL

// ---- selection --------------------------------------------------------------
// Request a backend by name (CLI --backend). Takes effect at the next dispatch;
// overrides TF_BACKEND. Returns 0 (and says so on stderr) for an unknown name.
// NULL or "" drops the request: back to TF_BACKEND, then auto.
int backend_request(const char *name);
// Resolve NOW (normally lazy) and return the active backend, NULL = CPU.
const TfBackend *backend_active(void);
const char *backend_active_name(void);             // "cpu" when none
// The one-line explanation of the last resolution that fell back to the CPU
// ("" if none). The same text went to stderr once.
const char *backend_last_message(void);
// Forget the resolution so the next dispatch re-resolves (tests, re-requests).
void backend_reset(void);
// The dispatch gates in force for the active backend (after TF_GPU_MIN_* env).
void backend_gates(int *min_m, long long *min_macs);
// Override them until the next resolution (tests; retuning from code).
void backend_set_gates(int min_m, long long min_macs);

// --gpu / --no-gpu: use the selected backend at all? Default 1.
void backend_set_enabled(int on);
int  backend_is_enabled(void);

// Free every backend's resources (all compiled-in ones that initialized).
void backend_shutdown_all(void);

// Dispatch counters since start (or the last backend_stats_reset): calls a
// backend handled vs calls that reached the dispatch point and went to the CPU.
void backend_stats(long *handled, long *declined);
void backend_stats_reset(void);

// ---- the dispatch point (called from matrix.c only) --------------------------
int backend_matmul(const float *A, const float *B, float *C, int M, int K, int N);
int backend_matmul_backward(const float *A, const float *B, const float *dC,
                            float *dA, float *dB, int M, int K, int N);
int backend_matmul_backward_dx(const float *B, const float *dC, float *dA,
                               int M, int K, int N);

// ============================================================================
// Column-major BLAS mapping (shared by cuda, hip and mock)
// ============================================================================
// cuBLAS / hipBLAS are COLUMN-major; this engine is ROW-major. No transpose is
// ever materialized. The identity used is:
//
//   a row-major (r x c) buffer with row stride c IS, byte for byte, the
//   column-major (c x r) matrix X^T with leading dimension c.
//
// So a row-major product C = A @ B is computed as the column-major product
// C^T = B^T @ A^T, i.e. "swap the operands, keep the ops":
//
//   forward  C(MxN)  = A @ B        ->  gemm(N, N, m=N, n=M, k=K, 1, B,ld=N,  A,ld=K,  0, C, ld=N)
//   dA(MxK) += dC @ B^T             ->  gemm(T, N, m=K, n=M, k=N, 1, B,ld=N,  dC,ld=N, 1, dA,ld=K)
//            (dA^T = B @ dC^T: B's buffer read col-major is B^T, so op T gives B)
//   dB(KxN) += A^T @ dC             ->  gemm(N, T, m=N, n=K, k=M, 1, dC,ld=N, A,ld=K,  1, dB,ld=N)
//            (dB^T = dC^T @ A: dC's buffer is dC^T as-is; A's buffer is A^T, op T gives A)
//
// beta = 0 on the forward (C is write-only: BLAS must not read it -- the mock's
// fake device poisons fresh buffers with NaN to prove that) and beta = 1 on both
// backward forms (they ACCUMULATE, matching matmul_backward's contract).
//
// TfColMajorSgemm is the cublasSgemm shape with the ops abstracted to 0 = N,
// 1 = T; each backend's adapter translates them (CUBLAS_OP_N/T = 0/1,
// HIPBLAS_OP_N/T = 111/112). Pointers are DEVICE pointers. Returns 0 on success
// (a BLAS status), nonzero on error.
typedef int (*TfColMajorSgemm)(void *ctx, int transa, int transb, int m, int n, int k,
                               float alpha, const float *A, int lda,
                               const float *B, int ldb, float beta, float *C, int ldc);

int tf_colmajor_fwd(TfColMajorSgemm g, void *ctx, const float *dA, const float *dB,
                    float *dC, int M, int K, int N);
int tf_colmajor_nt(TfColMajorSgemm g, void *ctx, const float *dB, const float *ddC,
                   float *ddA, int M, int K, int N);
int tf_colmajor_tn(TfColMajorSgemm g, void *ctx, const float *dA, const float *ddC,
                   float *ddB, int M, int K, int N);

// A plain column-major reference SGEMM with cuBLAS's argument rules ("fake
// cuBLAS"): C = alpha*op(A)@op(B) + beta*C, element (i,j) of X at X[i + j*ldx].
// Rejects what cuBLAS rejects (bad op, negative dims, lda/ldb/ldc too small for
// the stored matrix) with status 7 (CUBLAS_STATUS_INVALID_VALUE), and does NOT
// read C when beta == 0. ctx unused.
int tf_fake_cublas_sgemm(void *ctx, int transa, int transb, int m, int n, int k,
                         float alpha, const float *A, int lda,
                         const float *B, int ldb, float beta, float *C, int ldc);

// ============================================================================
// Device-BLAS core: the host-side half of a cuBLAS-style backend
// ============================================================================
// Device allocation cache keyed by host pointer (allocations only -- contents are
// uploaded on every call unless the pointer was register_weight()ed; see gpu.h),
// upload, column-major gemm via the mapping above, download. cuda, hip and mock
// are this core plus a loader and a five-function raw device vtable; the
// TF_DEVBLAS_OPS macro (backend.c users) generates their L1/L2/L3 table entries.
#define TF_DEVBLAS_SLOTS 256
typedef struct {
  const char *tag;                                      // for messages
  void *(*alloc)(size_t bytes);                         // NULL on failure
  void  (*free_)(void *dptr);
  int   (*h2d)(void *dptr, const void *host, size_t bytes);   // 0 = ok
  int   (*d2h)(void *host, const void *dptr, size_t bytes);   // 0 = ok
  TfColMajorSgemm gemm;
  void *gemm_ctx;
  struct { const void *host; TfBuf buf; int resident; } slot[TF_DEVBLAS_SLOTS];
  int nslots;
  TfBuf tmp[3]; int ntmp;                               // uncached, freed per call
  long uploads; size_t upload_bytes;                    // counters (tests, gpu-info)
} TfDevBlas;

int  tf_devblas_fwd(TfDevBlas *d, const float *A, const float *B, float *C, int M, int K, int N);
int  tf_devblas_nt(TfDevBlas *d, const float *B, const float *dC, float *dA, int M, int K, int N);
int  tf_devblas_tn(TfDevBlas *d, const float *A, const float *dC, float *dB, int M, int K, int N);
TfBuf *tf_devblas_buf_alloc(TfDevBlas *d, size_t bytes, TfDType dtype);
void tf_devblas_buf_free(TfDevBlas *d, TfBuf *b);
int  tf_devblas_buf_upload(TfDevBlas *d, TfBuf *dst, size_t off, const void *host, size_t bytes);
int  tf_devblas_buf_download(TfDevBlas *d, void *host, const TfBuf *src, size_t off, size_t bytes);
int  tf_devblas_gemm(TfDevBlas *d, int transa, int transb, int m, int n, int k,
                     float alpha, const TfBuf *A, int lda, const TfBuf *B, int ldb,
                     float beta, TfBuf *C, int ldc, TfDType ab_type, TfDType c_type);
const TfBuf *tf_devblas_register(TfDevBlas *d, const void *host, size_t bytes, TfDType dtype);
void tf_devblas_invalidate(TfDevBlas *d, const void *host);
void tf_devblas_release(TfDevBlas *d);

// Generates the static L1/L2/L3 wrappers for a device-BLAS backend whose core
// instance is `DB`: PFX_sgemm, _nt, _tn, _buf_alloc, _buf_free, _buf_upload,
// _buf_download, _gemm_op, _register, _invalidate. The backend must define
// PFX_available() and PFX_fail() (mark the device dead, return 0) first.
#define TF_DEVBLAS_OPS(PFX, DB)                                                          \
  static int PFX##_sgemm(const float *A, const float *B, float *C, int M, int K, int N) \
    { return PFX##_available() && tf_devblas_fwd(&DB, A, B, C, M, K, N) ? 1 : PFX##_fail(); } \
  static int PFX##_nt(const float *B, const float *dC, float *dA, int M, int K, int N)  \
    { return PFX##_available() && tf_devblas_nt(&DB, B, dC, dA, M, K, N) ? 1 : PFX##_fail(); } \
  static int PFX##_tn(const float *A, const float *dC, float *dB, int M, int K, int N)  \
    { return PFX##_available() && tf_devblas_tn(&DB, A, dC, dB, M, K, N) ? 1 : PFX##_fail(); } \
  static TfBuf *PFX##_buf_alloc(size_t n, TfDType t)                                    \
    { return PFX##_available() ? tf_devblas_buf_alloc(&DB, n, t) : NULL; }               \
  static void PFX##_buf_free(TfBuf *b) { tf_devblas_buf_free(&DB, b); }                 \
  static int PFX##_buf_upload(TfBuf *b, size_t o, const void *h, size_t n, TfStream *s) \
    { (void)s; return tf_devblas_buf_upload(&DB, b, o, h, n); }                          \
  static int PFX##_buf_download(void *h, const TfBuf *b, size_t o, size_t n, TfStream *s) \
    { (void)s; return tf_devblas_buf_download(&DB, h, b, o, n); }                        \
  static int PFX##_gemm_op(TfStream *s, int ta, int tb, int m, int n, int k, float al,     \
                        const TfBuf *A, int lda, const TfBuf *B, int ldb, float be,     \
                        TfBuf *C, int ldc, TfDType abt, TfDType ct)                     \
    { (void)s; return tf_devblas_gemm(&DB, ta, tb, m, n, k, al, A, lda, B, ldb, be, C, ldc, abt, ct); } \
  static const TfBuf *PFX##_register(const void *h, size_t n, TfDType t)                \
    { return PFX##_available() ? tf_devblas_register(&DB, h, n, t) : NULL; }             \
  static void PFX##_invalidate(const void *h) { tf_devblas_invalidate(&DB, h); }

// Per-backend accessors: the ops table, or NULL when not compiled in.
const TfBackend *tf_backend_opencl(void);
const TfBackend *tf_backend_cuda(void);
const TfBackend *tf_backend_hip(void);
const TfBackend *tf_backend_mock(void);

// Mock test hooks (backend_mock.c): fake gemm calls, the mock core's upload
// counters, and a deliberately WRONG mapping switch (operands not swapped) so a
// test can prove it would catch one.
long backend_mock_gemm_calls(void);
void backend_mock_uploads(long *n, size_t *bytes);
void backend_mock_set_broken_mapping(int on);
void backend_mock_set_fail(int on);     // make the fake device fail (fallback test)

// Adapter test hooks: the vendor op code each backend passes for our 0/1 (N/T).
// Pure functions, compiled in every build, so [49] checks them everywhere.
int tf_cuda_op(int t);      // CUBLAS_OP_N = 0,  CUBLAS_OP_T = 1
int tf_hip_op(int t);       // HIPBLAS_OP_N = 111, HIPBLAS_OP_T = 112

#endif // BACKEND_H
