// backend_hip.c - Optional AMD HIP/ROCm backend: the HIP runtime + hipBLAS, both
// loaded at RUNTIME. Compiled in only with -DTF_HIP (make GPU=hip).
//
// A mirror of backend_cuda.c (read that first): no ROCm SDK/headers needed to
// build, libraries opened with LoadLibrary/dlopen, types declared locally, every
// GEMM is hipblasSgemm reached through backend.c's device-BLAS core and the
// SAME row-major -> column-major mapping (tf_colmajor_*), which selftest [49]
// proves against a column-major reference on every build. hipBLAS is
// column-major with the cuBLAS argument order; only the op codes differ
// (HIPBLAS_OP_N = 111, HIPBLAS_OP_T = 112).
//
// STATUS: compiles and fails gracefully when the libraries are absent (verified);
// never run on AMD hardware (there is no free cloud AMD GPU -- see
// docs/ACCELERATORS.md). Treat the first real run as a bring-up.
#include "backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// hipblasOperation_t (hipblas.h): HIPBLAS_OP_N = 111, HIPBLAS_OP_T = 112.
int tf_hip_op(int t) { return t ? 112 : 111; }

#if !defined(TF_HIP)
const TfBackend *tf_backend_hip(void) { return NULL; }
#else  // ==================== TF_HIP ====================

#include <stdint.h>

#if defined(_WIN32)
#include <windows.h>
typedef HMODULE dynlib;
static dynlib dl_open(const char *n) { return LoadLibraryA(n); }
static void *dl_sym(dynlib h, const char *n) { return (void*)GetProcAddress(h, n); }
#else
#include <dlfcn.h>
typedef void *dynlib;
static dynlib dl_open(const char *n) { return dlopen(n, RTLD_NOW | RTLD_LOCAL); }
static void *dl_sym(dynlib h, const char *n) { return dlsym(h, n); }
#endif

typedef int hipError_t;                        // hipSuccess = 0
typedef struct hipblasContext *hipblasHandle_t; // opaque (void* in hipblas.h)
typedef int hipblasStatus_t;                   // HIPBLAS_STATUS_SUCCESS = 0
typedef int hipblasOperation_t;
#define HIP_MEMCPY_H2D 1                       // hipMemcpyHostToDevice
#define HIP_MEMCPY_D2H 2                       // hipMemcpyDeviceToHost

#define HIP_FN_LIST \
  X(hipInit,               hipError_t, (unsigned int)) \
  X(hipGetDeviceCount,     hipError_t, (int*)) \
  X(hipSetDevice,          hipError_t, (int)) \
  X(hipDeviceGetName,      hipError_t, (char*, int, int)) \
  X(hipDeviceTotalMem,     hipError_t, (size_t*, int)) \
  X(hipMalloc,             hipError_t, (void**, size_t)) \
  X(hipFree,               hipError_t, (void*)) \
  X(hipMemcpy,             hipError_t, (void*, const void*, size_t, int)) \
  X(hipDeviceSynchronize,  hipError_t, (void))

#define HIPBLAS_FN_LIST \
  X(hipblasCreate,  hipblasStatus_t, (hipblasHandle_t*)) \
  X(hipblasDestroy, hipblasStatus_t, (hipblasHandle_t)) \
  X(hipblasSgemm,   hipblasStatus_t, (hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, \
                                      int, int, int, const float*, const float*, int, \
                                      const float*, int, const float*, float*, int))

#define X(name, ret, args) typedef ret (*pfn_##name) args; static pfn_##name p_##name;
HIP_FN_LIST
HIPBLAS_FN_LIST
#undef X
typedef const char *(*pfn_hipGetErrorString)(hipError_t);
static pfn_hipGetErrorString p_hipGetErrorString;

static const char *const HIP_LIBS[] = {
#if defined(_WIN32)
  "amdhip64_6.dll", "amdhip64.dll", "amdhip64_5.dll",
#else
  "libamdhip64.so.6", "libamdhip64.so.5", "libamdhip64.so", "/opt/rocm/lib/libamdhip64.so",
#endif
  NULL
};
static const char *const HIPBLAS_LIBS[] = {
#if defined(_WIN32)
  "hipblas.dll", "libhipblas.dll",
#else
  "libhipblas.so.3", "libhipblas.so.2", "libhipblas.so.1", "libhipblas.so.0", "libhipblas.so",
  "/opt/rocm/lib/libhipblas.so",
#endif
  NULL
};

static int  g_probed = 0, g_ok = 0;
static char g_name[320] = "not initialized";
static size_t g_mem = 0;
static dynlib g_hip, g_blas;
static hipblasHandle_t g_handle;

static void fail(const char *why) { snprintf(g_name, sizeof(g_name), "%s", why); g_ok = 0; }
static const char *hip_err(hipError_t e) {
  const char *s = p_hipGetErrorString ? p_hipGetErrorString(e) : NULL;
  return s ? s : "unknown error";
}

static dynlib open_first(const char *env, const char *const *names, char *tried, size_t cap) {
  const char *e = getenv(env);
  if (e && *e) { snprintf(tried, cap, "%s=%s", env, e); return dl_open(e); }
  tried[0] = 0;
  for (int i = 0; names[i]; i++) {
    dynlib h = dl_open(names[i]);
    if (h) return h;
    if (i < 3) {
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

  g_hip = open_first("TF_HIP_LIB", HIP_LIBS, tried, sizeof(tried));
  if (!g_hip) {
    snprintf(msg, sizeof(msg), "no HIP runtime library (tried %s) -- ROCm / HIP SDK not installed", tried);
    fail(msg);
    return;
  }
#define X(name, ret, args) \
  p_##name = (pfn_##name)dl_sym(g_hip, #name); \
  if (!p_##name) { fail("HIP runtime library is missing " #name); return; }
  HIP_FN_LIST
#undef X
  p_hipGetErrorString = (pfn_hipGetErrorString)dl_sym(g_hip, "hipGetErrorString");
  hipError_t e = p_hipInit(0);
  if (e != 0) { snprintf(msg, sizeof(msg), "hipInit failed (%d: %s)", e, hip_err(e)); fail(msg); return; }
  int ndev = 0;
  if (p_hipGetDeviceCount(&ndev) != 0 || ndev < 1) { fail("HIP runtime loaded but it reports no device"); return; }
  if (p_hipSetDevice(0) != 0) { fail("hipSetDevice(0) failed"); return; }
  char dname[160] = {0};
  p_hipDeviceGetName(dname, (int)sizeof(dname) - 1, 0);
  p_hipDeviceTotalMem(&g_mem, 0);

  g_blas = open_first("TF_HIPBLAS_LIB", HIPBLAS_LIBS, tried, sizeof(tried));
  if (!g_blas) {
    snprintf(msg, sizeof(msg), "%s found, but no hipBLAS library (tried %s; set TF_HIPBLAS_LIB=/full/path)",
             dname, tried);
    fail(msg);
    return;
  }
#define X(name, ret, args) \
  p_##name = (pfn_##name)dl_sym(g_blas, #name); \
  if (!p_##name) { fail("hipBLAS library is missing " #name); return; }
  HIPBLAS_FN_LIST
#undef X
  hipblasStatus_t s = p_hipblasCreate(&g_handle);
  if (s != 0) { snprintf(msg, sizeof(msg), "hipblasCreate failed (status %d)", s); fail(msg); return; }

  snprintf(g_name, sizeof(g_name), "%s (%.1f GB, hipBLAS)", dname,
           (double)g_mem / (1024.0 * 1024.0 * 1024.0));
  g_ok = 1;
}

// The current device is per thread in HIP too: bind device 0 on every entry.
static void *hip_alloc(size_t n) {
  void *p = NULL;
  if (p_hipSetDevice(0) != 0 || p_hipMalloc(&p, n ? n : 1) != 0) return NULL;
  return p;
}
static void hip_free(void *p) { p_hipSetDevice(0); p_hipFree(p); }
static int hip_h2d(void *dev, const void *host, size_t n) {
  return p_hipSetDevice(0) != 0 || p_hipMemcpy(dev, host, n, HIP_MEMCPY_H2D) != 0;
}
static int hip_d2h(void *host, const void *dev, size_t n) {
  return p_hipSetDevice(0) != 0 || p_hipDeviceSynchronize() != 0 ||
         p_hipMemcpy(host, dev, n, HIP_MEMCPY_D2H) != 0;
}
static int hip_gemm(void *ctx, int ta, int tb, int m, int n, int k, float alpha,
                    const float *A, int lda, const float *B, int ldb,
                    float beta, float *C, int ldc) {
  (void)ctx;
  if (p_hipSetDevice(0) != 0) return 1;
  return p_hipblasSgemm(g_handle, tf_hip_op(ta), tf_hip_op(tb), m, n, k,
                        &alpha, A, lda, B, ldb, &beta, C, ldc);
}

static TfDevBlas g_hip_db = {
  "hip", hip_alloc, hip_free, hip_h2d, hip_d2h, hip_gemm, NULL,
  {{0}}, 0, {{0}}, 0, 0, 0,
};

static int hip_init(void) { init_once(); return g_ok; }
static int hip_available(void) { init_once(); return g_ok; }
static int hip_fail(void) { g_ok = 0; return 0; }
static const char *hip_device_name(void) { init_once(); return g_name; }
static void hip_caps(TfCaps *c) {
  init_once();
  memset(c, 0, sizeof(*c));
  c->mem_bytes = g_mem;
  c->n_devices = g_ok ? 1 : 0;
  c->min_m = 8;                 // untuned: same starting point as OpenCL/CUDA
  c->min_macs = 1LL << 22;
}
static int hip_sync(TfStream *s) {
  (void)s;
  return g_ok && p_hipSetDevice(0) == 0 && p_hipDeviceSynchronize() == 0;
}
static void hip_shutdown(void) {
  if (!g_probed || !g_handle) return;
  tf_devblas_release(&g_hip_db);
  p_hipblasDestroy(g_handle);
  g_handle = NULL;
  g_ok = 0;
}

TF_DEVBLAS_OPS(hip, g_hip_db)

static const TfBackend HIP_OPS = {
  "hip", "make GPU=hip (-DTF_HIP)", 1,
  hip_init, hip_shutdown, hip_available, hip_device_name, hip_caps,
  hip_sgemm, hip_nt, hip_tn,
  hip_buf_alloc, hip_buf_free, hip_buf_upload, hip_buf_download, hip_gemm_op, hip_sync,
  hip_register, hip_invalidate,
  {0},
};
const TfBackend *tf_backend_hip(void) { return &HIP_OPS; }

#endif // TF_HIP
