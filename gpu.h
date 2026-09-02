// gpu.h - Optional GPU acceleration for the matmul hot path (OpenCL).
//
// COMPILED IN ONLY WITH -DTF_GPU. Without it every function here is a stub that
// reports "not mine", so the CPU build runs exactly the code it runs today and
// there is no #ifdef anywhere outside gpu.c.
//
// WHY OpenCL, and why loaded at RUNTIME. CUDA would need nvcc and .cu files,
// which breaks this tree's C-only-flat-root rule and cannot be cross-compiled
// from Windows. Vulkan compute would need a SPIR-V toolchain at build time.
// OpenCL is a plain C API whose kernels are STRINGS compiled by the driver at
// runtime, so the kernel source lives as a string literal in gpu.c and the tree
// stays C-only and flat.
//
// The library is opened with dlopen/LoadLibrary and its entry points resolved by
// name; the handful of types we need are declared in gpu.c rather than pulled
// from <CL/cl.h>. The consequence is the point of the whole design:
//
//   * NO OpenCL SDK, header or import library is needed to BUILD, anywhere.
//   * The cross-compile to x86_64-linux-gnu needs nothing but `zig cc`.
//   * A GPU build on a machine with no GPU still links, runs, and falls back.
//
// The dispatch contract: every gpu_* compute function returns 1 if it handled
// the call and 0 if it did not, in which case the caller must run its normal CPU
// path. A 0 means any of: not compiled with TF_GPU, no device found, the runtime
// override turned it off, the problem is too small to be worth a dispatch, or a
// device error occurred (in which case the backend disables itself permanently
// rather than failing the run).
#ifndef GPU_H
#define GPU_H

#include <stddef.h>

// Initialize on first use and report whether a usable device was found. Safe and
// cheap to call repeatedly; the probe happens once. Always 0 without TF_GPU.
int gpu_available(void);

// 1 if the backend was compiled in at all (i.e. TF_GPU was defined), regardless
// of whether a device was found. Lets `gpu-info` tell "CPU-only build" apart
// from "GPU build, no device" — two very different things to debug.
int gpu_compiled_in(void);

// Human-readable device description ("NVIDIA Tesla T4", 40 compute units, ...),
// or an explanation of why there is none. Never NULL.
const char *gpu_device_name(void);
// Compute units and global memory of the selected device (0 if none).
int gpu_compute_units(void);
size_t gpu_global_mem(void);

// Runtime override, from `--gpu` / `--no-gpu`. Call before any compute. Passing 0
// forces the CPU path even on a GPU build, which is what makes an honest A/B
// timing possible from one binary.
void gpu_set_enabled(int on);
int gpu_is_enabled(void);

// Free every device buffer, kernel, queue and context. Optional — the OS reclaims
// on exit — but it keeps a long run's device memory bounded if it is ever called.
void gpu_shutdown(void);

// ---------------------------------------------------------------------------
// The compute entry points. Shapes match matrix.h exactly:
//   A is (M x K), B is (K x N), C is (M x N), all row-major and contiguous.
// ---------------------------------------------------------------------------

// C = A @ B. Returns 1 if the GPU did it.
int gpu_matmul(const float *A, const float *B, float *C, int M, int K, int N);

// dA += dC @ B^T and dB += A^T @ dC, accumulating into the two grad arrays
// (matching matmul_backward's contract — it ACCUMULATES, it does not overwrite).
int gpu_matmul_backward(const float *A, const float *B, const float *dC,
                        float *dA, float *dB, int M, int K, int N);

// Just dA += dC @ B^T, for a frozen B (the LoRA case). Returns 1 if handled.
int gpu_matmul_backward_dx(const float *B, const float *dC, float *dA,
                           int M, int K, int N);

// ---------------------------------------------------------------------------
// What is cached, and what deliberately is NOT
// ---------------------------------------------------------------------------
// The backend caches device BUFFER ALLOCATIONS keyed by host pointer, so a hot
// loop does not churn clCreateBuffer/clReleaseMemObject. It does NOT cache their
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
