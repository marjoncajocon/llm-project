// fake_gpu.c - Host-memory stand-ins for the CUDA driver, cuBLAS, the HIP runtime
// and hipBLAS, so the REAL backend_cuda.c / backend_hip.c code (LoadLibrary /
// dlopen, symbol resolution, context binding, op-code translation, host-pointer
// alpha/beta, upload/sync/download) can be exercised on a machine with no GPU.
//
// One source, four shared libraries (pick one with -D):
//   -DFAKE_CUDA     exports cuInit, cuMemAlloc_v2, ...      (stands in for nvcuda.dll / libcuda.so.1)
//   -DFAKE_CUBLAS   exports cublasCreate_v2, cublasSgemm_v2 (cublas64_12.dll / libcublas.so.12)
//   -DFAKE_HIP      exports hipInit, hipMalloc, ...          (amdhip64.dll / libamdhip64.so)
//   -DFAKE_HIPBLAS  exports hipblasCreate, hipblasSgemm      (hipblas.dll / libhipblas.so)
//
// Build + use (Windows; on Linux use -o libX.so and -fPIC):
//   zig cc -shared -O2 -DFAKE_CUDA   -o fake_cuda.dll   tools/fake_gpu/fake_gpu.c
//   zig cc -shared -O2 -DFAKE_CUBLAS -o fake_cublas.dll tools/fake_gpu/fake_gpu.c
//   TF_CUDA_LIB=./fake_cuda.dll TF_CUBLAS_LIB=./fake_cublas.dll ./transformer_cuda.exe
//   (the selftest's [30] and [49](g) then run "cuda" end to end on this fake)
//
// The BLAS entry points are a naive COLUMN-major reference that REJECTS what the
// real library rejects -- including op codes from the other vendor (cuBLAS
// N/T = 0/1, hipBLAS N/T = 111/112) -- so a wrong translation or a wrong
// row/column-major mapping fails loudly instead of passing by accident. Fresh
// device allocations are poisoned with NaN, and the driver's context check makes
// a missing cuCtxSetCurrent visible. What this canNOT show: real-device timing,
// async-stream races, driver quirks. That still needs Colab/Kaggle.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

#if defined(FAKE_CUBLAS) || defined(FAKE_HIPBLAS)
static int colmajor_sgemm(int ta, int tb, int m, int n, int k, float alpha,
                          const float *A, int lda, const float *B, int ldb,
                          float beta, float *C, int ldc) {
  if (m < 0 || n < 0 || k < 0) return 7;
  const int ar = ta ? k : m, br = tb ? n : k;
  if (lda < (ar > 1 ? ar : 1) || ldb < (br > 1 ? br : 1) || ldc < (m > 1 ? m : 1)) return 7;
  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++) {
      float acc = 0.0f;
      for (int p = 0; p < k; p++)
        acc += (ta ? A[p + (size_t)i * lda] : A[i + (size_t)p * lda]) *
               (tb ? B[j + (size_t)p * ldb] : B[p + (size_t)j * ldb]);
      float *c = &C[i + (size_t)j * ldc];
      *c = beta == 0.0f ? alpha * acc : alpha * acc + beta * *c;
    }
  return 0;
}
#endif

#if defined(FAKE_CUDA) || defined(FAKE_HIP)
static void *poisoned(size_t n) {
  void *p = malloc(n ? n : 1);
  if (p) memset(p, 0xFF, n);     // NaN: reading a never-written buffer shows up
  return p;
}
#endif

#if defined(FAKE_CUDA)
typedef unsigned long long CUdeviceptr;
static int g_ctx_obj, g_inited;
static void *const CTX = &g_ctx_obj;
static void *g_current;
API int cuInit(unsigned f) { (void)f; g_inited = 1; return 0; }
API int cuDeviceGetCount(int *n) { *n = g_inited ? 1 : 0; return g_inited ? 0 : 3; }
API int cuDeviceGet(int *d, int i) { *d = i; return i == 0 ? 0 : 101; }
API int cuDeviceGetName(char *s, int len, int d) { (void)d; strncpy(s, "FakeCUDA (host memory)", (size_t)len); return 0; }
API int cuDeviceTotalMem_v2(size_t *b, int d) { (void)d; *b = (size_t)1 << 30; return 0; }
API int cuDeviceGetAttribute(int *v, int a, int d) { (void)d; *v = a == 16 ? 2 : 0; return 0; }
API int cuDevicePrimaryCtxRetain(void **c, int d) { (void)d; *c = CTX; return 0; }
API int cuDevicePrimaryCtxRelease_v2(int d) { (void)d; return 0; }
API int cuCtxSetCurrent(void *c) { if (c != CTX) return 201; g_current = c; return 0; }
API int cuCtxSynchronize(void) { return g_current == CTX ? 0 : 201; }
API int cuMemAlloc_v2(CUdeviceptr *p, size_t n) {
  if (g_current != CTX) return 201;                    // CUDA_ERROR_INVALID_CONTEXT
  void *h = poisoned(n);
  *p = (CUdeviceptr)(uintptr_t)h;
  return h ? 0 : 2;
}
API int cuMemFree_v2(CUdeviceptr p) { free((void*)(uintptr_t)p); return 0; }
API int cuMemcpyHtoD_v2(CUdeviceptr d, const void *h, size_t n) {
  if (g_current != CTX) return 201;
  memcpy((void*)(uintptr_t)d, h, n); return 0;
}
API int cuMemcpyDtoH_v2(void *h, CUdeviceptr d, size_t n) {
  if (g_current != CTX) return 201;
  memcpy(h, (const void*)(uintptr_t)d, n); return 0;
}
API int cuGetErrorString(int r, const char **s) { *s = r == 201 ? "invalid context (fake)" : "fake error"; return 0; }
#endif

#if defined(FAKE_CUBLAS)
static int g_handle_obj;
API int cublasCreate_v2(void **h) { *h = &g_handle_obj; return 0; }
API int cublasDestroy_v2(void *h) { return h == &g_handle_obj ? 0 : 1; }
API int cublasGetVersion_v2(void *h, int *v) { (void)h; *v = 99999; return 0; }
API int cublasSgemm_v2(void *h, int ta, int tb, int m, int n, int k, const float *alpha,
                       const float *A, int lda, const float *B, int ldb,
                       const float *beta, float *C, int ldc) {
  if (h != &g_handle_obj) return 1;                    // NOT_INITIALIZED
  if ((ta != 0 && ta != 1) || (tb != 0 && tb != 1)) return 7;   // cuBLAS op codes only
  return colmajor_sgemm(ta, tb, m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
}
#endif

#if defined(FAKE_HIP)
static int g_dev = -1;
API int hipInit(unsigned f) { (void)f; return 0; }
API int hipGetDeviceCount(int *n) { *n = 1; return 0; }
API int hipSetDevice(int d) { if (d != 0) return 101; g_dev = d; return 0; }
API int hipDeviceGetName(char *s, int len, int d) { (void)d; strncpy(s, "FakeHIP (host memory)", (size_t)len); return 0; }
API int hipDeviceTotalMem(size_t *b, int d) { (void)d; *b = (size_t)1 << 30; return 0; }
API int hipMalloc(void **p, size_t n) { if (g_dev != 0) return 101; *p = poisoned(n); return *p ? 0 : 2; }
API int hipFree(void *p) { free(p); return 0; }
API int hipMemcpy(void *d, const void *s, size_t n, int kind) {
  if (g_dev != 0 || (kind != 1 && kind != 2)) return 1;
  memcpy(d, s, n); return 0;
}
API int hipDeviceSynchronize(void) { return 0; }
API const char *hipGetErrorString(int e) { (void)e; return "fake error"; }
#endif

#if defined(FAKE_HIPBLAS)
static int g_handle_obj;
API int hipblasCreate(void **h) { *h = &g_handle_obj; return 0; }
API int hipblasDestroy(void *h) { return h == &g_handle_obj ? 0 : 1; }
API int hipblasSgemm(void *h, int ta, int tb, int m, int n, int k, const float *alpha,
                     const float *A, int lda, const float *B, int ldb,
                     const float *beta, float *C, int ldc) {
  if (h != &g_handle_obj) return 1;
  if ((ta != 111 && ta != 112) || (tb != 111 && tb != 112)) return 3;   // hipBLAS op codes only
  return colmajor_sgemm(ta == 112, tb == 112, m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
}
#endif
