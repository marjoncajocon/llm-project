// backend_cuda.c - Optional CUDA backend: the CUDA driver API + cuBLAS, both
// loaded at RUNTIME. Compiled in only with -DTF_CUDA (make GPU=cuda).
//
// NO CUDA toolkit, header, import library or nvcc is needed to build it: the
// driver (nvcuda.dll / libcuda.so.1) and cuBLAS (cublas64_*.dll /
// libcublas.so.*) are opened with LoadLibrary/dlopen, the ~20 entry points are
// resolved by name, and the handful of types involved are declared below. So a
// CUDA-enabled Linux binary cross-compiles from Windows with `zig cc` alone
// (make cross-cuda), and runs -- falling back to the CPU with one clear line --
// on a machine that has neither library.
//
// There are no kernels here: every GEMM is cublasSgemm, reached through the
// device-BLAS core in backend.c (allocation cache, upload, download) and the
// row-major -> column-major mapping documented in backend.h:
//
//   row-major C = A @ B  ==  column-major C^T = B^T @ A^T
//   -> cublasSgemm(N, N, m=N, n=M, k=K, 1, B, N, A, K, 0, C, N)
//   and the backward forms with one op = T each (dA: (T,N), dB: (N,T)), beta = 1.
//
// That mapping is the part that is easy to get wrong and impossible to test
// without a GPU -- so it is NOT in this file. backend.c's tf_colmajor_* runs it
// for CUDA, HIP and the mock alike, and selftest [49] proves it against a
// column-major reference ("fake cuBLAS") on every build. What remains here and
// is unverified until it runs on a real NVIDIA GPU (Colab/Kaggle): the loader,
// the context handling and the op-code translation below.
//
// Precision: cublasSgemm is plain FP32 unless the process opts into TF32
// (NVIDIA_TF32_OVERRIDE=1 or a math-mode call we never make), so results should
// match the CPU to ~1e-6 relative, like the OpenCL backend's.
#include "backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// cublasOperation_t: CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 (cublas_api.h). Compiled in
// every build so [49] can check the translation even without -DTF_CUDA.
int tf_cuda_op(int t) { return t ? 1 : 0; }

#if !defined(TF_CUDA)
const TfBackend *tf_backend_cuda(void) { return NULL; }
#else  // ==================== TF_CUDA ====================

#include <stdint.h>

#if defined(_WIN32)
#include <windows.h>
typedef HMODULE dynlib;
static dynlib dl_open(const char *n) { return LoadLibraryA(n); }
static void *dl_sym(dynlib h, const char *n) { return (void*)GetProcAddress(h, n); }
#else
#include <dlfcn.h>
typedef void *dynlib;
static dynlib dl_open(const char *n) {
  // A full path to libcublas.so.X in a directory that is not on the loader path
  // (e.g. /usr/local/cuda/lib64) needs its sibling libcublasLt.so.X loaded
  // first, globally, or the dlopen fails on the unresolved dependency.
  const char *slash = strrchr(n, '/'), *so = strstr(n, "libcublas.so");
  if (slash && so && so == slash + 1) {
    char lt[1024];
    snprintf(lt, sizeof(lt), "%.*slibcublasLt.so%s", (int)(so - n), n, so + strlen("libcublas.so"));
    dlopen(lt, RTLD_NOW | RTLD_GLOBAL);
  }
  return dlopen(n, RTLD_NOW | RTLD_LOCAL);
}
static void *dl_sym(dynlib h, const char *n) { return dlsym(h, n); }
#endif

// ---------------------------------------------------------------------------
// The slice of the CUDA driver + cuBLAS ABI we use, declared locally.
// ---------------------------------------------------------------------------
typedef int CUresult;                          // CUDA_SUCCESS = 0
typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
typedef unsigned long long CUdeviceptr;        // 64-bit on every platform we build GPU for
typedef struct cublasContext *cublasHandle_t;
typedef int cublasStatus_t;                    // CUBLAS_STATUS_SUCCESS = 0
typedef int cublasOperation_t;

#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16

// Versioned names (_v2) are what the headers #define the plain names to; they
// are the ones exported for 64-bit sizes/pointers since CUDA 3.2.
#define CU_FN_LIST \
  X(cuInit,                   CUresult, (unsigned int)) \
  X(cuDeviceGetCount,         CUresult, (int*)) \
  X(cuDeviceGet,              CUresult, (CUdevice*, int)) \
  X(cuDeviceGetName,          CUresult, (char*, int, CUdevice)) \
  X(cuDeviceTotalMem_v2,      CUresult, (size_t*, CUdevice)) \
  X(cuDeviceGetAttribute,     CUresult, (int*, int, CUdevice)) \
  X(cuDevicePrimaryCtxRetain, CUresult, (CUcontext*, CUdevice)) \
  X(cuCtxSetCurrent,          CUresult, (CUcontext)) \
  X(cuCtxSynchronize,         CUresult, (void)) \
  X(cuMemAlloc_v2,            CUresult, (CUdeviceptr*, size_t)) \
  X(cuMemFree_v2,             CUresult, (CUdeviceptr)) \
  X(cuMemcpyHtoD_v2,          CUresult, (CUdeviceptr, const void*, size_t)) \
  X(cuMemcpyDtoH_v2,          CUresult, (void*, CUdeviceptr, size_t))

#define CUBLAS_FN_LIST \
  X(cublasCreate_v2,  cublasStatus_t, (cublasHandle_t*)) \
  X(cublasDestroy_v2, cublasStatus_t, (cublasHandle_t)) \
  X(cublasSgemm_v2,   cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, \
                                       int, int, int, const float*, const float*, int, \
                                       const float*, int, const float*, float*, int))

#define X(name, ret, args) typedef ret (*pfn_##name) args; static pfn_##name p_##name;
CU_FN_LIST
CUBLAS_FN_LIST
#undef X
// Optional (only for messages / version display).
typedef CUresult (*pfn_cuGetErrorString)(CUresult, const char**);
static pfn_cuGetErrorString p_cuGetErrorString;
typedef cublasStatus_t (*pfn_cublasGetVersion_v2)(cublasHandle_t, int*);
static pfn_cublasGetVersion_v2 p_cublasGetVersion_v2;
typedef CUresult (*pfn_cuDevicePrimaryCtxRelease)(CUdevice);
static pfn_cuDevicePrimaryCtxRelease p_cuDevicePrimaryCtxRelease;

static const char *const CUDA_LIBS[] = {
#if defined(_WIN32)
  "nvcuda.dll",
#else
  "libcuda.so.1", "libcuda.so",
  // WSL2 keeps the driver's libcuda here, outside the default search path.
  "/usr/lib/wsl/lib/libcuda.so.1",
#endif
  NULL
};

// cuBLAS ships with the TOOLKIT, not the driver: newest ABI first. Colab and
// Kaggle have it (and PyTorch's nvidia-cublas wheel carries one too -- point
// TF_CUBLAS_LIB at it if nothing below is found).
static const char *const CUBLAS_LIBS[] = {
#if defined(_WIN32)
  "cublas64_13.dll", "cublas64_12.dll", "cublas64_11.dll", "cublas64_10.dll",
#else
  "libcublas.so.13", "libcublas.so.12", "libcublas.so.11", "libcublas.so.10", "libcublas.so",
  "/usr/local/cuda/lib64/libcublas.so.13", "/usr/local/cuda/lib64/libcublas.so.12",
  "/usr/local/cuda/lib64/libcublas.so.11", "/usr/local/cuda/lib64/libcublas.so",
#endif
  NULL
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
static int  g_probed = 0, g_ok = 0;
static char g_name[320] = "not initialized";
static int  g_sms = 0;
static size_t g_mem = 0;
static dynlib g_cu, g_blas;
static CUdevice g_dev;
static CUcontext g_ctx;
static cublasHandle_t g_handle;

static void fail(const char *why) { snprintf(g_name, sizeof(g_name), "%s", why); g_ok = 0; }

static const char *cu_err(CUresult r) {
  const char *s = NULL;
  if (p_cuGetErrorString && p_cuGetErrorString(r, &s) == 0 && s) return s;
  return "unknown error";
}

static dynlib open_first(const char *env, const char *const *names, char *tried, size_t cap) {
  const char *e = getenv(env);
  if (e && *e) {
    dynlib h = dl_open(e);
    snprintf(tried, cap, "%s=%s", env, e);
    return h;
  }
  tried[0] = 0;
  for (int i = 0; names[i]; i++) {
    dynlib h = dl_open(names[i]);
    if (h) return h;
    if (i < 4) {                                   // keep the message short
      size_t l = strlen(tried);
      snprintf(tried + l, cap - l, "%s%s", l ? ", " : "", names[i]);
    }
  }
  return NULL;
}

static void init_once(void) {
  if (g_probed) return;
  g_probed = 1;
  char tried[512], msg[320 + 512];

  g_cu = open_first("TF_CUDA_LIB", CUDA_LIBS, tried, sizeof(tried));
  if (!g_cu) {
    snprintf(msg, sizeof(msg), "no CUDA driver library (tried %s) -- no NVIDIA driver installed", tried);
    fail(msg);
    return;
  }
#define X(name, ret, args) \
  p_##name = (pfn_##name)dl_sym(g_cu, #name); \
  if (!p_##name) { fail("CUDA driver library is missing " #name " (driver too old?)"); return; }
  CU_FN_LIST
#undef X
  p_cuGetErrorString = (pfn_cuGetErrorString)dl_sym(g_cu, "cuGetErrorString");
  p_cuDevicePrimaryCtxRelease = (pfn_cuDevicePrimaryCtxRelease)dl_sym(g_cu, "cuDevicePrimaryCtxRelease_v2");
  if (!p_cuDevicePrimaryCtxRelease)
    p_cuDevicePrimaryCtxRelease = (pfn_cuDevicePrimaryCtxRelease)dl_sym(g_cu, "cuDevicePrimaryCtxRelease");

  CUresult r = p_cuInit(0);
  if (r != 0) {
    snprintf(msg, sizeof(msg), "cuInit failed (%d: %s) -- driver present but no usable GPU", r, cu_err(r));
    fail(msg);
    return;
  }
  int ndev = 0;
  if (p_cuDeviceGetCount(&ndev) != 0 || ndev < 1) { fail("CUDA driver loaded but it reports no device"); return; }
  if (p_cuDeviceGet(&g_dev, 0) != 0) { fail("cuDeviceGet(0) failed"); return; }
  char dname[160] = {0};
  p_cuDeviceGetName(dname, (int)sizeof(dname) - 1, g_dev);
  p_cuDeviceTotalMem_v2(&g_mem, g_dev);
  p_cuDeviceGetAttribute(&g_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, g_dev);
  // The PRIMARY context is the one the CUDA runtime -- and therefore cuBLAS --
  // uses, so driver-API allocations and cuBLAS calls share it.
  r = p_cuDevicePrimaryCtxRetain(&g_ctx, g_dev);
  if (r != 0 || !g_ctx) { snprintf(msg, sizeof(msg), "cuDevicePrimaryCtxRetain failed (%d: %s)", r, cu_err(r)); fail(msg); return; }
  if (p_cuCtxSetCurrent(g_ctx) != 0) { fail("cuCtxSetCurrent failed"); return; }

  g_blas = open_first("TF_CUBLAS_LIB", CUBLAS_LIBS, tried, sizeof(tried));
  if (!g_blas) {
    snprintf(msg, sizeof(msg), "%s found, but no cuBLAS library (tried %s; set TF_CUBLAS_LIB=/full/path)",
             dname, tried);
    fail(msg);
    return;
  }
#define X(name, ret, args) \
  p_##name = (pfn_##name)dl_sym(g_blas, #name); \
  if (!p_##name) { fail("cuBLAS library is missing " #name); return; }
  CUBLAS_FN_LIST
#undef X
  p_cublasGetVersion_v2 = (pfn_cublasGetVersion_v2)dl_sym(g_blas, "cublasGetVersion_v2");
  cublasStatus_t s = p_cublasCreate_v2(&g_handle);
  if (s != 0) { snprintf(msg, sizeof(msg), "cublasCreate failed (status %d)", s); fail(msg); return; }
  int ver = 0;
  if (p_cublasGetVersion_v2) p_cublasGetVersion_v2(g_handle, &ver);

  snprintf(g_name, sizeof(g_name), "%s (%d SMs, %.1f GB, cuBLAS %d)", dname, g_sms,
           (double)g_mem / (1024.0 * 1024.0 * 1024.0), ver);
  g_ok = 1;
}

// ---------------------------------------------------------------------------
// The raw device vtable for the device-BLAS core. A CUDA context is current
// PER THREAD, and matmul may be called from any thread, so every entry binds the
// primary context first (a cheap TLS store when it already is current).
// ---------------------------------------------------------------------------
static CUdeviceptr dp(const void *p) { return (CUdeviceptr)(uintptr_t)p; }
static void *cu_alloc(size_t n) {
  CUdeviceptr p = 0;
  if (p_cuCtxSetCurrent(g_ctx) != 0 || p_cuMemAlloc_v2(&p, n ? n : 1) != 0) return NULL;
  return (void*)(uintptr_t)p;
}
static void cu_free(void *p) { p_cuCtxSetCurrent(g_ctx); p_cuMemFree_v2(dp(p)); }
static int cu_h2d(void *dev, const void *host, size_t n) {
  return p_cuCtxSetCurrent(g_ctx) != 0 || p_cuMemcpyHtoD_v2(dp(dev), host, n) != 0;
}
static int cu_d2h(void *host, const void *dev, size_t n) {
  // Synchronize first: cublasSgemm is asynchronous, and this also surfaces any
  // error from it here instead of on some later, unrelated call.
  return p_cuCtxSetCurrent(g_ctx) != 0 || p_cuCtxSynchronize() != 0 ||
         p_cuMemcpyDtoH_v2(host, dp(dev), n) != 0;
}
static int cu_gemm(void *ctx, int ta, int tb, int m, int n, int k, float alpha,
                   const float *A, int lda, const float *B, int ldb,
                   float beta, float *C, int ldc) {
  (void)ctx;
  if (p_cuCtxSetCurrent(g_ctx) != 0) return 1;
  // alpha/beta are HOST pointers: the handle's default pointer mode.
  return p_cublasSgemm_v2(g_handle, tf_cuda_op(ta), tf_cuda_op(tb), m, n, k,
                          &alpha, A, lda, B, ldb, &beta, C, ldc);
}

static TfDevBlas g_cuda_db = {
  "cuda", cu_alloc, cu_free, cu_h2d, cu_d2h, cu_gemm, NULL,
  {{0}}, 0, {{0}}, 0, 0, 0,
};

static int cuda_init(void) { init_once(); return g_ok; }
static int cuda_available(void) { init_once(); return g_ok; }
// Any device error disables the backend for the rest of the run rather than
// killing it (same policy as OpenCL); the core already printed where it failed.
static int cuda_fail(void) { g_ok = 0; return 0; }
static const char *cuda_device_name(void) { init_once(); return g_name; }
static void cuda_caps(TfCaps *c) {
  init_once();
  memset(c, 0, sizeof(*c));
  c->mem_bytes = g_mem;
  c->compute_units = g_sms;
  c->deterministic = 1;      // cuBLAS: bitwise reproducible per arch/SM count, one stream
  c->n_devices = g_ok ? 1 : 0;
  // Starting points copied from OpenCL; retune on a real card with
  // TF_GPU_MIN_M / TF_GPU_MIN_MACS (gpu-bench prints the evidence).
  c->min_m = 8;
  c->min_macs = 1LL << 22;
}
static int cuda_sync(TfStream *s) {
  (void)s;
  return g_ok && p_cuCtxSetCurrent(g_ctx) == 0 && p_cuCtxSynchronize() == 0;
}
static void cuda_shutdown(void) {
  if (!g_probed || !g_ctx) return;
  tf_devblas_release(&g_cuda_db);
  if (g_handle) p_cublasDestroy_v2(g_handle);
  g_handle = NULL;
  if (p_cuDevicePrimaryCtxRelease) p_cuDevicePrimaryCtxRelease(g_dev);
  g_ctx = NULL;
  g_ok = 0;
}

TF_DEVBLAS_OPS(cuda, g_cuda_db)

static const TfBackend CUDA_OPS = {
  "cuda", "make GPU=cuda (-DTF_CUDA)", 1,
  cuda_init, cuda_shutdown, cuda_available, cuda_device_name, cuda_caps,
  cuda_sgemm, cuda_nt, cuda_tn,
  cuda_buf_alloc, cuda_buf_free, cuda_buf_upload, cuda_buf_download, cuda_gemm_op, cuda_sync,
  cuda_register, cuda_invalidate,
  {0},
};
const TfBackend *tf_backend_cuda(void) { return &CUDA_OPS; }

#endif // TF_CUDA
