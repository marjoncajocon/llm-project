// gpu.h - GPU acceleration for the matmul hot path: the legacy gpu_* API and the
// design notes shared by every GPU backend.
//
// The compute now goes through the backend layer (backend.h): OpenCL
// (backend_opencl.c, -DTF_GPU), CUDA (backend_cuda.c, -DTF_CUDA), HIP/ROCm
// (backend_hip.c, -DTF_HIP) and a CPU-side mock. matrix.c calls backend_matmul*;
// the functions below are kept as a thin facade (implemented in backend.c) for
// `gpu-info`, `gpu-bench`, `--gpu/--no-gpu` and selftest part [30], and they
// mean "the ACTIVE real GPU backend", whichever one that is.
//
// WHY every backend is loaded at RUNTIME. Neither nvcc/.cu files nor an SDK can
// be part of this tree's C-only, flat, cross-compiled-from-Windows build. So each
// backend opens its driver with dlopen/LoadLibrary, resolves its entry points by
// name and declares the handful of types it needs locally (no <CL/cl.h>, no
// <cuda.h>, no <cublas_v2.h>, no <hip/hip_runtime.h>). OpenCL kernels are
// STRINGS compiled by the driver; CUDA and HIP need no kernels at all -- they
// call the vendor BLAS (cuBLAS / hipBLAS). The consequence is the point:
//
//   * NO SDK, header or import library is needed to BUILD, anywhere.
//   * The cross-compile to x86_64-linux-gnu needs nothing but `zig cc`.
//   * A GPU build on a machine with no GPU still links, runs, and falls back.
//
// The dispatch contract: every compute call returns 1 if a backend handled it
// and 0 if it did not, in which case the caller must run its normal CPU path. A
// 0 means any of: no GPU backend compiled in, no device found, the runtime
// override turned it off, the problem is too small to be worth a dispatch, or a
// device error occurred (in which case the backend disables itself permanently
// rather than failing the run).
#ifndef GPU_H
#define GPU_H

#include <stddef.h>

// Report whether the active backend is a usable real GPU (resolving the
// selection on first use -- see backend.h). Always 0 without a GPU backend.
int gpu_available(void);

// 1 if ANY real GPU backend (opencl, cuda, hip) was compiled in, regardless of
// whether a device was found. Lets `gpu-info` tell "CPU-only build" apart from
// "GPU build, no device" — two very different things to debug.
int gpu_compiled_in(void);

// Human-readable device description ("NVIDIA Tesla T4", 40 compute units, ...),
// or an explanation of why there is none. Never NULL.
const char *gpu_device_name(void);
// Compute units and global memory of the active device (0 if none).
int gpu_compute_units(void);
size_t gpu_global_mem(void);

// Runtime override, from `--gpu` / `--no-gpu` (== backend_set_enabled). Passing
// 0 forces the CPU path even on a GPU build, which is what makes an honest A/B
// timing possible from one binary.
void gpu_set_enabled(int on);
int gpu_is_enabled(void);

// Free every backend's device buffers and handles (backend_shutdown_all).
void gpu_shutdown(void);

// ---------------------------------------------------------------------------
// What is cached, and what deliberately is NOT
// ---------------------------------------------------------------------------
// Every backend (OpenCL, and the device-BLAS core behind CUDA/HIP/mock) caches
// device BUFFER ALLOCATIONS keyed by host pointer, so a hot
// loop does not churn clCreateBuffer / cuMemAlloc. It does NOT cache their
// CONTENTS: every input is uploaded on every call.
//
// That is a deliberate correctness decision. A content cache would have to know
// that the host has not written to a buffer since the last upload, and in this
// tree that cannot be inferred from the call site: activations are rewritten
// between matmuls by rmsnorm/rope/swiglu, and attention.c passes ACTIVATIONS as
// the B operand for Q@K^T, so "B is a weight" is not true either. Guessing wrong
// would not crash — it would silently train on stale weights.
//
// The cost is that a GEMM is roughly transfer-bound at parity with its own
// compute on a discrete GPU (~50% of achievable throughput). The fix is a weight
// content-cache driven by an explicit "this tensor is immutable until I say
// otherwise" registration from the model code, which is the obvious next
// optimization and is NOT built here.
//
// Two gates keep the GPU away from work it would lose at (both tunable by env
// var so they can be retuned on a real device without a rebuild):
//   TF_GPU_MIN_M     (default 8)     minimum rows M.  Single-token decode is
//                                    M=1 and memory-bandwidth-bound: uploading a
//                                    113 MB lm_head to multiply one row by it is
//                                    far slower than just doing it on the CPU.
//   TF_GPU_MIN_MACS  (default 1<<22) minimum M*N*K. Below this the dispatch and
//                                    transfer cost dominate the arithmetic.

#endif // GPU_H
