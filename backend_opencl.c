// backend_opencl.c - Optional OpenCL backend for the matmul hot path (formerly
// gpu.c; gpu.h keeps the design notes, backend.h describes the layer it plugs in).
//
// Everything here is behind TF_GPU. Without it this file only reports "not
// compiled in", so the CPU build is byte-for-byte the code it was.
//
// Moved into the backend layer WITHOUT behaviour change: the kernels, device
// selection, buffer cache, upload/launch/download sequence and error handling
// are the old gpu.c's. What moved OUT is generic and now lives in backend.c's
// dispatch, applied in the same order: the --gpu/--no-gpu enabled flag, the
// TF_GPU_MIN_M / TF_GPU_MIN_MACS gates and the "both backward halves or abort"
// rule.
#include "backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(TF_GPU)
// Not compiled in: the registry lists OpenCL as "not compiled into this binary".
const TfBackend *tf_backend_opencl(void) { return NULL; }

#else  // ==================== TF_GPU ====================

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

// ---------------------------------------------------------------------------
// The slice of the OpenCL ABI we use, declared here rather than pulled from
// <CL/cl.h> — that is what lets this build (and CROSS-build) with no SDK at all.
// Every handle is an opaque pointer and every scalar a fixed-width int, so the
// ABI is stable across vendors.
// ---------------------------------------------------------------------------
typedef int32_t  cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint64_t cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_command_queue_properties;
typedef cl_uint  cl_bool;
typedef cl_uint  cl_device_info;
typedef cl_uint  cl_program_build_info;
typedef intptr_t cl_context_properties;

typedef struct _cl_platform_id  *cl_platform_id;
typedef struct _cl_device_id    *cl_device_id;
typedef struct _cl_context      *cl_context;
typedef struct _cl_command_queue*cl_command_queue;
typedef struct _cl_program      *cl_program;
typedef struct _cl_kernel       *cl_kernel;
typedef struct _cl_mem          *cl_mem;
typedef struct _cl_event        *cl_event;

#define CL_SUCCESS                     0
#define CL_TRUE                        1
#define CL_DEVICE_TYPE_GPU             (1 << 2)
#define CL_DEVICE_NAME                 0x102B
#define CL_DEVICE_MAX_COMPUTE_UNITS    0x1002
#define CL_DEVICE_GLOBAL_MEM_SIZE      0x101F
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE   0x1010
#define CL_DEVICE_HOST_UNIFIED_MEMORY  0x1035
#define CL_MEM_READ_WRITE              (1 << 0)
#define CL_PROGRAM_BUILD_LOG           0x1183

#define CL_FN_LIST \
  X(clGetPlatformIDs,        cl_int, (cl_uint, cl_platform_id*, cl_uint*)) \
  X(clGetDeviceIDs,          cl_int, (cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*)) \
  X(clGetDeviceInfo,         cl_int, (cl_device_id, cl_device_info, size_t, void*, size_t*)) \
  X(clCreateContext,         cl_context, (const cl_context_properties*, cl_uint, const cl_device_id*, void*, void*, cl_int*)) \
  X(clCreateCommandQueue,    cl_command_queue, (cl_context, cl_device_id, cl_command_queue_properties, cl_int*)) \
  X(clCreateProgramWithSource, cl_program, (cl_context, cl_uint, const char**, const size_t*, cl_int*)) \
  X(clBuildProgram,          cl_int, (cl_program, cl_uint, const cl_device_id*, const char*, void*, void*)) \
  X(clGetProgramBuildInfo,   cl_int, (cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*)) \
  X(clCreateKernel,          cl_kernel, (cl_program, const char*, cl_int*)) \
  X(clSetKernelArg,          cl_int, (cl_kernel, cl_uint, size_t, const void*)) \
  X(clEnqueueNDRangeKernel,  cl_int, (cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*)) \
  X(clCreateBuffer,          cl_mem, (cl_context, cl_mem_flags, size_t, void*, cl_int*)) \
  X(clEnqueueWriteBuffer,    cl_int, (cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*)) \
  X(clEnqueueReadBuffer,     cl_int, (cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*)) \
  X(clFinish,                cl_int, (cl_command_queue)) \
  X(clReleaseMemObject,      cl_int, (cl_mem)) \
  X(clReleaseKernel,         cl_int, (cl_kernel)) \
  X(clReleaseProgram,        cl_int, (cl_program)) \
  X(clReleaseCommandQueue,   cl_int, (cl_command_queue)) \
  X(clReleaseContext,        cl_int, (cl_context))

#define X(name, ret, args) typedef ret (*pfn_##name) args; static pfn_##name p_##name;
CL_FN_LIST
#undef X

// Candidate sonames, in order. libOpenCL is the ICD loader; the NVIDIA vendor
// library is tried directly last so a Colab image that never installed
// ocl-icd-libopencl1 still works — the NVIDIA ICD exports the standard entry
// points itself.
static const char *const CL_LIBS[] = {
#if defined(_WIN32)
  "OpenCL.dll",
#elif defined(__APPLE__)
  "/System/Library/Frameworks/OpenCL.framework/OpenCL",
#else
  "libOpenCL.so.1", "libOpenCL.so", "libnvidia-opencl.so.1",
#endif
  NULL
};

// ---------------------------------------------------------------------------
// The kernels. Tiled through local memory: a naive one-thread-per-output kernel
// re-reads both operands from global memory K times and ends up bandwidth-bound,
// which on an integrated GPU loses to the CPU outright.
//
// Every tile load is bounds-checked and zero-filled, so M/N/K need not be
// multiples of TS — the failure mode a tiled GEMM classically gets wrong, and
// what selftest part [30] deliberately probes with ragged shapes.
// ---------------------------------------------------------------------------
#define TS 16
#define WPT 4          // must match the kernel source below

static const char *KERNEL_SRC =
"#define TS 16\n"
"#define WPT 4\n"
"// Forward GEMM, the hottest kernel, with REGISTER TILING on top of the local\n"
"// tiling: each work-item computes WPT outputs along N, so one value loaded from\n"
"// local memory feeds WPT fused multiply-adds instead of one. Local tiling alone\n"
"// leaves the kernel local-memory-bandwidth-bound; this is what turns it into an\n"
"// arithmetic-bound one. The WPT columns a thread owns are STRIDED by TS so that\n"
"// consecutive threads still touch consecutive addresses (coalesced global reads).\n"
"__kernel void sgemm(const int M, const int K, const int N,\n"
"                    __global const float *A, __global const float *B,\n"
"                    __global float *C) {\n"
"  const int lr = get_local_id(0), lc = get_local_id(1);\n"
"  const int gr = get_group_id(0) * TS + lr;\n"
"  const int gc0 = get_group_id(1) * (TS * WPT) + lc;\n"
"  __local float As[TS][TS];\n"
"  __local float Bs[TS][TS * WPT];\n"
"  float acc[WPT];\n"
"  for (int w = 0; w < WPT; w++) acc[w] = 0.0f;\n"
"  const int nt = (K + TS - 1) / TS;\n"
"  for (int t = 0; t < nt; t++) {\n"
"    const int ka = t * TS + lc, kb = t * TS + lr;\n"
"    As[lr][lc] = (gr < M && ka < K) ? A[(long)gr * K + ka] : 0.0f;\n"
"    for (int w = 0; w < WPT; w++) {\n"
"      const int c = gc0 + w * TS;\n"
"      Bs[lr][lc + w * TS] = (kb < K && c < N) ? B[(long)kb * N + c] : 0.0f;\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int k = 0; k < TS; k++) {\n"
"      const float a = As[lr][k];\n"
"      for (int w = 0; w < WPT; w++) acc[w] += a * Bs[k][lc + w * TS];\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"  }\n"
"  if (gr < M)\n"
"    for (int w = 0; w < WPT; w++) {\n"
"      const int c = gc0 + w * TS;\n"
"      if (c < N) C[(long)gr * N + c] = acc[w];\n"
"    }\n"
"}\n"
"\n"
"// dA(M,K) += dC(M,N) @ B(K,N)^T   -- reduction over N\n"
"__kernel void sgemm_nt(const int M, const int N, const int K,\n"
"                       __global const float *dC, __global const float *B,\n"
"                       __global float *dA) {\n"
"  const int lr = get_local_id(0), lc = get_local_id(1);\n"
"  const int gr = get_group_id(0) * TS + lr;\n"
"  const int gc = get_group_id(1) * TS + lc;\n"
"  __local float As[TS][TS];\n"
"  __local float Bs[TS][TS];\n"
"  float acc = 0.0f;\n"
"  const int nt = (N + TS - 1) / TS;\n"
"  for (int t = 0; t < nt; t++) {\n"
"    const int na = t * TS + lc, nb = t * TS + lr;\n"
"    As[lr][lc] = (gr < M && na < N) ? dC[(long)gr * N + na] : 0.0f;\n"
"    Bs[lr][lc] = (gc < K && nb < N) ? B[(long)gc * N + nb] : 0.0f;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int k = 0; k < TS; k++) acc += As[lr][k] * Bs[k][lc];\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"  }\n"
"  if (gr < M && gc < K) dA[(long)gr * K + gc] += acc;\n"
"}\n"
"\n"
"// dB(K,N) += A(M,K)^T @ dC(M,N)   -- reduction over M\n"
"__kernel void sgemm_tn(const int M, const int K, const int N,\n"
"                       __global const float *A, __global const float *dC,\n"
"                       __global float *dB) {\n"
"  const int lr = get_local_id(0), lc = get_local_id(1);\n"
"  const int gr = get_group_id(0) * TS + lr;\n"
"  const int gc = get_group_id(1) * TS + lc;\n"
"  __local float As[TS][TS];\n"
"  __local float Bs[TS][TS];\n"
"  float acc = 0.0f;\n"
"  const int nt = (M + TS - 1) / TS;\n"
"  for (int t = 0; t < nt; t++) {\n"
"    const int ma = t * TS + lc, mb = t * TS + lr;\n"
"    As[lr][lc] = (gr < K && ma < M) ? A[(long)ma * K + gr] : 0.0f;\n"
"    Bs[lr][lc] = (mb < M && gc < N) ? dC[(long)mb * N + gc] : 0.0f;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (int k = 0; k < TS; k++) acc += As[lr][k] * Bs[k][lc];\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"  }\n"
"  if (gr < K && gc < N) dB[(long)gr * N + gc] += acc;\n"
"}\n";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
static int  g_probed = 0;        // init attempted
static int  g_ok = 0;            // usable device present
static char g_name[256] = "not initialized";
static int  g_cus = 0;
static size_t g_mem = 0;
static size_t g_max_alloc = 0;  // capabilities only (gpu-info); no effect on compute
static int  g_unified = 0;

static dynlib g_lib;
static cl_context       g_ctx;
static cl_command_queue g_q;
static cl_program       g_prog;
static cl_kernel        g_k_mm, g_k_nt, g_k_tn;
static cl_device_id     g_dev;

// Device buffer cache: allocations only, never contents (see gpu.h).
#define CACHE_SLOTS 256
typedef struct { const void *host; cl_mem buf; size_t bytes; } BufEntry;
static BufEntry g_cache[CACHE_SLOTS];
static int g_cache_n = 0;

static void fail(const char *why) {
  snprintf(g_name, sizeof(g_name), "%s", why);
  g_ok = 0;
}

static int load_symbols(void) {
  for (int i = 0; CL_LIBS[i]; i++) {
    g_lib = dl_open(CL_LIBS[i]);
    if (g_lib) break;
  }
  if (!g_lib) { fail("no OpenCL library found (install an OpenCL ICD loader)"); return 0; }
#define X(name, ret, args) \
  p_##name = (pfn_##name)dl_sym(g_lib, #name); \
  if (!p_##name) { fail("OpenCL library is missing " #name); return 0; }
  CL_FN_LIST
#undef X
  return 1;
}

// Pick the first platform that exposes a GPU.
static int pick_device(void) {
  cl_platform_id plats[16];
  cl_uint nplat = 0;
  if (p_clGetPlatformIDs(16, plats, &nplat) != CL_SUCCESS || nplat == 0) {
    fail("OpenCL loaded but no platform is registered (no vendor ICD installed)");
    return 0;
  }
  for (cl_uint i = 0; i < nplat; i++) {
    cl_uint ndev = 0;
    if (p_clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_GPU, 1, &g_dev, &ndev) == CL_SUCCESS && ndev > 0)
      return 1;
  }
  fail("OpenCL platform present but it exposes no GPU device");
  return 0;
}

static void init_once(void) {
  if (g_probed) return;
  g_probed = 1;

  if (!load_symbols()) return;
  if (!pick_device()) return;

  char dname[192] = {0};
  p_clGetDeviceInfo(g_dev, CL_DEVICE_NAME, sizeof(dname) - 1, dname, NULL);
  cl_uint cus = 0;
  p_clGetDeviceInfo(g_dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, NULL);
  cl_ulong gmem = 0;
  p_clGetDeviceInfo(g_dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(gmem), &gmem, NULL);
  g_cus = (int)cus;
  g_mem = (size_t)gmem;
  cl_ulong maxa = 0;
  cl_bool unified = 0;
  p_clGetDeviceInfo(g_dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxa), &maxa, NULL);
  p_clGetDeviceInfo(g_dev, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified, NULL);
  g_max_alloc = (size_t)maxa;
  g_unified = unified ? 1 : 0;

  cl_int err = CL_SUCCESS;
  g_ctx = p_clCreateContext(NULL, 1, &g_dev, NULL, NULL, &err);
  if (err != CL_SUCCESS || !g_ctx) { fail("clCreateContext failed"); return; }
  g_q = p_clCreateCommandQueue(g_ctx, g_dev, 0, &err);
  if (err != CL_SUCCESS || !g_q) { fail("clCreateCommandQueue failed"); return; }

  const char *src = KERNEL_SRC;
  const size_t len = strlen(src);
  g_prog = p_clCreateProgramWithSource(g_ctx, 1, &src, &len, &err);
  if (err != CL_SUCCESS || !g_prog) { fail("clCreateProgramWithSource failed"); return; }
  if (p_clBuildProgram(g_prog, 1, &g_dev, "-cl-fast-relaxed-math", NULL, NULL) != CL_SUCCESS) {
    // The build log is the only useful diagnostic when a driver rejects a kernel.
    char log[2048] = {0};
    p_clGetProgramBuildInfo(g_prog, g_dev, CL_PROGRAM_BUILD_LOG, sizeof(log) - 1, log, NULL);
    fprintf(stderr, "gpu: kernel build failed:\n%s\n", log);
    fail("OpenCL kernel build failed (see stderr)");
    return;
  }
  g_k_mm = p_clCreateKernel(g_prog, "sgemm", &err);
  if (err != CL_SUCCESS) { fail("clCreateKernel(sgemm) failed"); return; }
  g_k_nt = p_clCreateKernel(g_prog, "sgemm_nt", &err);
  if (err != CL_SUCCESS) { fail("clCreateKernel(sgemm_nt) failed"); return; }
  g_k_tn = p_clCreateKernel(g_prog, "sgemm_tn", &err);
  if (err != CL_SUCCESS) { fail("clCreateKernel(sgemm_tn) failed"); return; }

  snprintf(g_name, sizeof(g_name), "%s (%u compute units, %.1f GB)",
           dname, cus, (double)gmem / (1024.0 * 1024.0 * 1024.0));
  g_ok = 1;
}

static int ocl_init(void) { init_once(); return g_ok; }
static const char *ocl_device_name(void) { init_once(); return g_name; }
static void ocl_caps(TfCaps *c) {
  init_once();
  memset(c, 0, sizeof(*c));
  c->mem_bytes = g_mem;
  c->max_alloc = g_max_alloc;
  c->compute_units = g_cus;
  c->unified_memory = g_unified;   // TODO: skip the copies on a unified-memory iGPU
  c->deterministic = 1;            // fixed kernels, fixed work split
  c->n_devices = g_ok ? 1 : 0;
  // The gates the old gpu.c hard-coded (see gpu.h for why M matters separately).
  c->min_m = 8;
  c->min_macs = 1LL << 22;
}

// A device buffer for this host pointer, at least `bytes` long. Cached by
// pointer so a hot loop does not churn allocations; contents are never assumed.
static cl_mem buffer_for(const void *host, size_t bytes) {
  for (int i = 0; i < g_cache_n; i++) {
    if (g_cache[i].host == host && g_cache[i].bytes >= bytes) return g_cache[i].buf;
    if (g_cache[i].host == host) {                 // same pointer, grew: reallocate
      p_clReleaseMemObject(g_cache[i].buf);
      cl_int e = CL_SUCCESS;
      g_cache[i].buf = p_clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, bytes, NULL, &e);
      g_cache[i].bytes = bytes;
      return (e == CL_SUCCESS) ? g_cache[i].buf : NULL;
    }
  }
  cl_int e = CL_SUCCESS;
  cl_mem m = p_clCreateBuffer(g_ctx, CL_MEM_READ_WRITE, bytes, NULL, &e);
  if (e != CL_SUCCESS || !m) return NULL;
  if (g_cache_n < CACHE_SLOTS) {
    g_cache[g_cache_n].host = host;
    g_cache[g_cache_n].buf = m;
    g_cache[g_cache_n].bytes = bytes;
    g_cache_n++;
  }
  // Cache full: the buffer still works, it just gets re-created next time. The
  // engine touches far fewer than CACHE_SLOTS distinct tensors in practice.
  return m;
}

// (Whether the GPU takes a problem at all -- enabled flag, device present, the
// TF_GPU_MIN_M / TF_GPU_MIN_MACS gates -- is decided in backend.c before any
// function below is called. g_ok is still checked there via the available() hook.)

static void grid_for(int rows, int cols, size_t *gws, size_t *lws) {
  gws[0] = (size_t)((rows + TS - 1) / TS) * TS;
  gws[1] = (size_t)((cols + TS - 1) / TS) * TS;
  lws[0] = TS;
  lws[1] = TS;
}

// The forward kernel gives each work-item WPT columns, so the N dimension needs
// only 1/WPT as many threads — a grid built with grid_for would launch WPT times
// too many and compute the tail columns repeatedly.
static void grid_for_fwd(int rows, int cols, size_t *gws, size_t *lws) {
  gws[0] = (size_t)((rows + TS - 1) / TS) * TS;
  gws[1] = (size_t)((cols + TS * WPT - 1) / (TS * WPT)) * TS;
  lws[0] = TS;
  lws[1] = TS;
}

// Any device error disables the backend for the rest of the run rather than
// killing it: a training run that has been going for two hours should degrade to
// the CPU, not abort.
static int device_failed(const char *where) {
  fprintf(stderr, "gpu: %s failed — falling back to the CPU for the rest of this run\n", where);
  g_ok = 0;
  return 0;
}

static int ocl_sgemm(const float *A, const float *B, float *C, int M, int K, int N) {
  const size_t bA = (size_t)M * K * sizeof(float);
  const size_t bB = (size_t)K * N * sizeof(float);
  const size_t bC = (size_t)M * N * sizeof(float);
  cl_mem dA = buffer_for(A, bA), dB = buffer_for(B, bB), dC = buffer_for(C, bC);
  if (!dA || !dB || !dC) return device_failed("buffer allocation");

  if (p_clEnqueueWriteBuffer(g_q, dA, CL_TRUE, 0, bA, A, 0, NULL, NULL) != CL_SUCCESS ||
      p_clEnqueueWriteBuffer(g_q, dB, CL_TRUE, 0, bB, B, 0, NULL, NULL) != CL_SUCCESS)
    return device_failed("upload");

  cl_int e = CL_SUCCESS;
  e |= p_clSetKernelArg(g_k_mm, 0, sizeof(int), &M);
  e |= p_clSetKernelArg(g_k_mm, 1, sizeof(int), &K);
  e |= p_clSetKernelArg(g_k_mm, 2, sizeof(int), &N);
  e |= p_clSetKernelArg(g_k_mm, 3, sizeof(cl_mem), &dA);
  e |= p_clSetKernelArg(g_k_mm, 4, sizeof(cl_mem), &dB);
  e |= p_clSetKernelArg(g_k_mm, 5, sizeof(cl_mem), &dC);
  if (e != CL_SUCCESS) return device_failed("clSetKernelArg");

  size_t gws[2], lws[2];
  grid_for_fwd(M, N, gws, lws);
  if (p_clEnqueueNDRangeKernel(g_q, g_k_mm, 2, NULL, gws, lws, 0, NULL, NULL) != CL_SUCCESS)
    return device_failed("sgemm launch");
  if (p_clEnqueueReadBuffer(g_q, dC, CL_TRUE, 0, bC, C, 0, NULL, NULL) != CL_SUCCESS)
    return device_failed("download");
  return 1;
}

// dA += dC @ B^T. Shared by matmul_backward and matmul_backward_dx.
static int run_nt(const float *B, const float *dCh, float *dAh, int M, int K, int N) {
  const size_t bB = (size_t)K * N * sizeof(float);
  const size_t bC = (size_t)M * N * sizeof(float);
  const size_t bA = (size_t)M * K * sizeof(float);
  cl_mem dB = buffer_for(B, bB), dC = buffer_for(dCh, bC), dA = buffer_for(dAh, bA);
  if (!dA || !dB || !dC) return device_failed("buffer allocation");
  // dA is READ-MODIFY-WRITE: the kernel accumulates, matching matmul_backward's
  // contract, so the existing host contents must be uploaded first.
  if (p_clEnqueueWriteBuffer(g_q, dB, CL_TRUE, 0, bB, B, 0, NULL, NULL) != CL_SUCCESS ||
      p_clEnqueueWriteBuffer(g_q, dC, CL_TRUE, 0, bC, dCh, 0, NULL, NULL) != CL_SUCCESS ||
      p_clEnqueueWriteBuffer(g_q, dA, CL_TRUE, 0, bA, dAh, 0, NULL, NULL) != CL_SUCCESS)
    return device_failed("upload");

  cl_int e = CL_SUCCESS;
  e |= p_clSetKernelArg(g_k_nt, 0, sizeof(int), &M);
  e |= p_clSetKernelArg(g_k_nt, 1, sizeof(int), &N);
  e |= p_clSetKernelArg(g_k_nt, 2, sizeof(int), &K);
  e |= p_clSetKernelArg(g_k_nt, 3, sizeof(cl_mem), &dC);
  e |= p_clSetKernelArg(g_k_nt, 4, sizeof(cl_mem), &dB);
  e |= p_clSetKernelArg(g_k_nt, 5, sizeof(cl_mem), &dA);
  if (e != CL_SUCCESS) return device_failed("clSetKernelArg");

  size_t gws[2], lws[2];
  grid_for(M, K, gws, lws);
  if (p_clEnqueueNDRangeKernel(g_q, g_k_nt, 2, NULL, gws, lws, 0, NULL, NULL) != CL_SUCCESS)
    return device_failed("sgemm_nt launch");
  if (p_clEnqueueReadBuffer(g_q, dA, CL_TRUE, 0, bA, dAh, 0, NULL, NULL) != CL_SUCCESS)
    return device_failed("download");
  return 1;
}

// dB += A^T @ dC
static int run_tn(const float *A, const float *dCh, float *dBh, int M, int K, int N) {
  const size_t bA = (size_t)M * K * sizeof(float);
  const size_t bC = (size_t)M * N * sizeof(float);
  const size_t bB = (size_t)K * N * sizeof(float);
  cl_mem dA = buffer_for(A, bA), dC = buffer_for(dCh, bC), dB = buffer_for(dBh, bB);
  if (!dA || !dB || !dC) return device_failed("buffer allocation");
  if (p_clEnqueueWriteBuffer(g_q, dA, CL_TRUE, 0, bA, A, 0, NULL, NULL) != CL_SUCCESS ||
      p_clEnqueueWriteBuffer(g_q, dC, CL_TRUE, 0, bC, dCh, 0, NULL, NULL) != CL_SUCCESS ||
      p_clEnqueueWriteBuffer(g_q, dB, CL_TRUE, 0, bB, dBh, 0, NULL, NULL) != CL_SUCCESS)
    return device_failed("upload");

  cl_int e = CL_SUCCESS;
  e |= p_clSetKernelArg(g_k_tn, 0, sizeof(int), &M);
  e |= p_clSetKernelArg(g_k_tn, 1, sizeof(int), &K);
  e |= p_clSetKernelArg(g_k_tn, 2, sizeof(int), &N);
  e |= p_clSetKernelArg(g_k_tn, 3, sizeof(cl_mem), &dA);
  e |= p_clSetKernelArg(g_k_tn, 4, sizeof(cl_mem), &dC);
  e |= p_clSetKernelArg(g_k_tn, 5, sizeof(cl_mem), &dB);
  if (e != CL_SUCCESS) return device_failed("clSetKernelArg");

  size_t gws[2], lws[2];
  grid_for(K, N, gws, lws);
  if (p_clEnqueueNDRangeKernel(g_q, g_k_tn, 2, NULL, gws, lws, 0, NULL, NULL) != CL_SUCCESS)
    return device_failed("sgemm_tn launch");
  if (p_clEnqueueReadBuffer(g_q, dB, CL_TRUE, 0, bB, dBh, 0, NULL, NULL) != CL_SUCCESS)
    return device_failed("download");
  return 1;
}

static void ocl_shutdown(void) {
  if (!g_probed) return;
  for (int i = 0; i < g_cache_n; i++)
    if (g_cache[i].buf) p_clReleaseMemObject(g_cache[i].buf);
  g_cache_n = 0;
  if (g_k_mm) p_clReleaseKernel(g_k_mm);
  if (g_k_nt) p_clReleaseKernel(g_k_nt);
  if (g_k_tn) p_clReleaseKernel(g_k_tn);
  if (g_prog) p_clReleaseProgram(g_prog);
  if (g_q) p_clReleaseCommandQueue(g_q);
  if (g_ctx) p_clReleaseContext(g_ctx);
  g_k_mm = NULL; g_k_nt = NULL; g_k_tn = NULL;
  g_prog = NULL; g_q = NULL; g_ctx = NULL;
  g_ok = 0;
}

// L1 only for now. TODO(P1, docs/ACCELERATORS.md): port onto TfBuf (L2) with a
// row-major->column-major-capable gemm kernel, and weight residency (L3), so
// the iGPU can test residency on a real device.
static const TfBackend OPENCL_OPS = {
  "opencl", "make GPU=opencl (-DTF_GPU; GPU=1)", 1,
  ocl_init, ocl_shutdown, ocl_init, ocl_device_name, ocl_caps,
  ocl_sgemm, run_nt, run_tn,
  NULL, NULL, NULL, NULL, NULL, NULL,    /* L2: device memory stays private */
  NULL, NULL,                            /* L3: no residency yet */
  {0},
};
const TfBackend *tf_backend_opencl(void) { return &OPENCL_OPS; }

#endif // TF_GPU
