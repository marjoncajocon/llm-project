# Accelerators: GPU / TPU roadmap for this engine

Status: research + design, September 2026. Nothing here is built yet except the existing
OpenCL matmul offload (`gpu.c`). A parallel effort is adding a **backend layer** with runtime-loaded
**CUDA/cuBLAS**, **HIP/hipBLAS** and a **mock** backend; this document does not redesign that, it
says what the layer has to grow into (section 4) and in what order to add the remaining targets.

Ground rules this plan respects (from `gpu.h` / notes.md 2026-09-02(b)):
pure C11, flat root; **no SDK, header or import library at build time** — vendor libraries are
opened with `dlopen`/`LoadLibrary` and entry points resolved by name, types re-declared locally;
cross-compile from Windows with `zig cc`; a GPU build on a machine without the device must still
run and fall back to the CPU; every offloaded op returns 1 = handled / 0 = run the CPU path.

## 0. What THIS machine has (checked read-only, 2026-09-22)

* CPU: Intel Core i5-14400 (Raptor Lake, 6P+4E): **AVX2 + FMA + AVX-VNNI; no AVX-512, no AMX**.
* GPU: **Intel UHD Graphics 730** (Xe-LP, 24 EUs, shared DDR memory, driver 32.0.101.7088).
  The driver package (`iigd_dch.inf`) ships all three compute runtimes:
  `igdrcl64.dll` (**OpenCL**, already used — `gpu-info` finds 24 CUs), `igvk64.dll` (**Vulkan**),
  `ze_intel_gpu64.dll` + `System32\ze_loader.dll` (**Level Zero**). `d3d12.dll` and a system
  `DirectML.dll` are also present (**D3D12 / DirectML**).
* So the box can test **OpenCL, Vulkan, Level Zero and D3D12** code paths locally. It cannot
  test CUDA, HIP, Metal or TPU. And per the measured `gpu-bench` (notes.md), the iGPU does
  **~60 GFLOP/s vs the CPU's 170–240 GFLOP/s** — it is a *correctness* test device, not a speedup
  device. The only GEMM it won was the 49k-column `lm_head` (1.77x).

## 1. Per-target summary

Effort: S = days, M = 1–3 weeks, L = a month+, L+ = a project of its own.
"Runtime-loadable" = can be `dlopen`ed with no SDK at build time, like `gpu.c`.

| # | Target | API a C11 engine uses | Runtime-loadable, no SDK? | Kernels come from | Effort | Free/cheap test |
|---|--------|----------------------|---------------------------|-------------------|--------|-----------------|
| 1 | NVIDIA | CUDA **driver** API (`libcuda`/`nvcuda.dll`) + cuBLAS(Lt) + NVRTC | **Yes** (all C APIs) | cuBLAS for GEMM; CUDA C strings via NVRTC → PTX → `cuModuleLoadData` | S (other agent) → M | Colab T4 (free), Kaggle 2xT4 (free) |
| 2 | AMD | HIP runtime + hipBLAS(Lt) + hipRTC | **Yes** on Linux; on Windows hipBLAS/hipRTC exist only if the HIP SDK is installed | hipBLAS; HIP C via hipRTC | S (other agent) → M | no free tier; paid rentals |
| 3 | Intel | **OpenCL** (have it), **Level Zero** (C API); SYCL (C++, unusable) | OpenCL/L0: **Yes** (driver ships them) | OpenCL C strings (have); SPIR-V for L0 | L0: M | **this PC** (UHD 730) |
| 4 | Apple | Metal (ObjC/Swift/metal-cpp) + MPS | Frameworks always on macOS, but **not callable from plain C** without an ObjC shim or `objc_msgSend` | MSL source strings compiled at runtime, or MPS | M | GitHub Actions macOS arm64 runners |
| 5 | Vulkan | Vulkan 1.2/1.3 compute | **Yes** (`vulkan-1.dll`/`libvulkan.so.1`) | **SPIR-V compiled offline**, embedded as `uint32_t[]` | L | **this PC**; Colab/Kaggle T4; lavapipe (CPU) in CI |
| 6 | DirectML / D3D12 | D3D12 compute (COM, C bindings); DirectML (COM, C++ header) | D3D12 **yes** (Windows only); DirectML yes but **maintenance mode** | DXIL compiled offline; or DirectML operators | M–L | **this PC** |
| 7 | TPU | **PJRT C API** (`pjrt_c_api.h`) + StableHLO text | **Yes** (`libtpu.so` exports `GetPjrtApi`) | none — XLA compiles a StableHLO graph the engine emits | L+ | Colab free TPU v5e-1; Kaggle TPU v5e-8 |
| 8 | CPU | AVX2/FMA (have), AVX-VNNI, AVX-512/AMX (servers), NEON/SVE | n/a (runtime CPUID dispatch, as matrix.c does) | intrinsics with `target()` attributes | S–M each | this PC; Colab/Kaggle Xeons |

### 1. NVIDIA — CUDA driver API + cuBLAS + NVRTC
Being built by the other agent (cuBLAS behind the backend layer). The piece to add next is
**NVRTC**: `libnvrtc` accepts CUDA C++ source **as a string** and produces PTX/cubin that
`cuModuleLoadData` loads, and NVIDIA explicitly supports using the shared library via
`dlopen`/`LoadLibrary` ([NVRTC docs](https://docs.nvidia.com/cuda/nvrtc/index.html)). That gives
the same "kernel is a C string literal" property `gpu.c` has for OpenCL, **with no nvcc** — so
custom kernels (RMSNorm, RoPE, softmax, SwiGLU, fused attention, int8 dequant-GEMV) stay in the
flat C tree. Prefer the *driver* API (`cuInit`, `cuMemAlloc`, `cuModuleLoadData`,
`cuLaunchKernel`, `cuStream*`, `cuGraph*`): it lives in the driver (`libcuda.so.1` /
`nvcuda.dll`), so any machine with an NVIDIA driver can create a context. cuBLAS and NVRTC are
toolkit libraries — present on Colab/Kaggle (`/usr/local/cuda/lib64`, and inside PyTorch's
`nvidia/*/lib` wheels); on end-user PCs they must be found or shipped. Caveat: NVRTC and
`nvrtc-builtins` must come from the same toolkit version; probe `libnvrtc.so.13`, `.12`, then
unversioned. Order: cuBLAS GEMM (S) → NVRTC elementwise/norm kernels so activations stay
resident (M) → weights resident + `cublasGemmEx` fp16/bf16 on tensor cores (M) → fused
attention / int8 GEMV for M=1 decode (M) → CUDA Graphs for the per-token decode loop (S).

### 2. AMD — HIP / ROCm + hipBLAS
HIP mirrors the CUDA runtime API nearly 1:1 (`hipMalloc`, `hipModuleLaunchKernel`,
`hiprtcCompileProgram`), so it can be a second instantiation of the CUDA backend's table (the
other agent's plan). Linux: `libamdhip64.so`, `libhipblas.so`, `libhiprtc.so`.
**Windows status:** the HIP SDK is officially supported on Windows 10/11/Server 2022 and ships the
HIP runtime + math libraries (hipBLAS; hipBLASLt from 6.4.2 on some GPUs), while PyTorch on
Windows is only a *public preview* for Radeon 7000/9000 and some Ryzen AI APUs
([HIP SDK for Windows](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/),
[component support](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/component-support.html),
[AMD blog](https://www.amd.com/en/blogs/2025/the-road-to-rocm-on-radeon-for-windows-and-linux.html)).
Implication: on consumer Windows, hipBLAS/hipRTC exist only if the user installed the HIP SDK, so
**Vulkan is the more reliable AMD path on Windows**. There is no free cloud AMD GPU; rely on the
mock backend plus an occasional paid hour on an MI300X / Radeon rental.

### 3. Intel — OpenCL (have), Level Zero, SYCL
* **OpenCL** already works on the UHD 730 and runs on Arc. Cheap wins inside the OpenCL backend:
  `cl_khr_fp16` half storage, `cl_intel_subgroups` block reads, weight residency (section 2).
  It is the only real-device backend testable on this box today, so keep it alive.
* **Level Zero** is a plain C API (`zeInit`, `zeMemAllocDevice`, `zeModuleCreate` from
  **SPIR-V**, `zeCommandListAppendLaunchKernel`), shipped in the driver (`ze_loader.dll` is present
  here). llama.cpp's SYCL docs recommend Level Zero over OpenCL on Intel GPUs for performance and
  list the UHD 730 as a Level Zero device
  ([llama.cpp SYCL.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)).
  It needs SPIR-V kernels (same offline-toolchain issue as Vulkan) and adds little over OpenCL for
  us. **Defer**; revisit only if Arc XMX matrix engines become a target.
* **SYCL / oneAPI DPC++** is C++17 single-source — incompatible with the C-only rule. Not
  recommended.

### 4. Apple — Metal
Metal and MPS are Objective-C/Swift APIs; `metal-cpp` is a header-only C++ wrapper
([Metal (API)](https://en.wikipedia.org/wiki/Metal_(API))). A C11 engine has two options:
(a) one **`metal.m` shim** (~400 lines of Objective-C, compiled only for `METAL=1` on macOS)
exporting a C function table (`mtl_alloc`, `mtl_matmul`, `mtl_launch(name, args)`) — the shape of
llama.cpp's `ggml-metal`; kernels are MSL **source strings** compiled at runtime with
`newLibraryWithSource`, so no offline toolchain; or
(b) pure C through `objc_msgSend` + `sel_registerName` — keeps the tree C-only, but verbose and
fragile. Recommend (a) as the single sanctioned exception to "C-only", or (b) if the rule is
absolute. First offload: `MPSMatrixMultiplication` (fp32/fp16 GEMM, S). Apple GPUs have **unified
memory** (`MTLResourceStorageModeShared`), so uploads are free and residency is trivial — the win
comes early. Cannot be cross-built from Windows (links Apple frameworks); build and test on a
free **GitHub Actions macOS arm64 runner** (M-series GPU, free for public repos). Effort M.

### 5. Vulkan compute — the portable GPU backend
One backend covering **NVIDIA, AMD, Intel (incl. this iGPU), Qualcomm, and Apple via MoltenVK**,
on Windows and Linux, loaded from `vulkan-1.dll`/`libvulkan.so.1`, which GPU drivers install.
llama.cpp's Vulkan backend proves it can be competitive: it targets all those vendors, uses
`VK_KHR_cooperative_matrix` (and NVIDIA coopmat2) for tensor cores, and is reported competitive
with or ahead of ROCm on AMD while CUDA still leads on NVIDIA
([ggml Vulkan backend overview](https://deepwiki.com/ggml-org/llama.cpp/5.3-vulkan-backend-(cross-platform)),
[FOSDEM 2026 talk notes](https://philpax.me/notes/talks/other-people/fosdem-2026/vulkan-api-for-machine-learning-competing-with-cuda-and-rocm-in-llamacpp/),
[int8 coopmat PR](https://github.com/ggml-org/llama.cpp/pull/27952)).
The cost is the **shader toolchain**: Vulkan consumes SPIR-V only. Keep the no-SDK build by
compiling GLSL → SPIR-V *offline* with `glslc`/`glslangValidator` (dev-only, via a
`tools/gen_spirv.ps1`) and committing the output as generated C arrays
(`vk_shaders.c: static const uint32_t k_sgemm[] = {...}`). Normal builds need nothing; only
editing a shader needs the tool. The boilerplate is large (instance, device, queues, descriptor
sets, pipelines, memory types, barriers) — budget ~1500 lines. Order: GEMM (M) → all per-layer
ops so a layer runs on-device (M) → weights resident + fp16 storage (S) → coopmat GEMM / int8
dot via `VK_KHR_shader_integer_dot_product` (M). Effort L overall. Testable **here** (UHD 730),
on Colab/Kaggle T4 (the NVIDIA driver includes a Vulkan ICD — verify with `vulkaninfo` in the
notebook, since containers sometimes omit it), and in CPU-only CI via Mesa **lavapipe**.

### 6. DirectML / Direct3D 12 (Windows)
DirectML runs on **every DX12 GPU including integrated ones** — this machine qualifies — but
Microsoft has put it in **maintenance mode**: security/compliance fixes only, no new features,
and new work is pointed at Windows ML on Windows 11 24H2+
([microsoft/DirectML](https://github.com/microsoft/DirectML),
[Microsoft Learn](https://learn.microsoft.com/en-us/windows/win32/direct3d12/dml-redirect)).
Windows ML runs ONNX models; it is not an operator library, so it does not fit an engine that owns
its ops. DirectML's header is C++/COM (a C caller would hand-declare vtables or add a `.cpp`
shim). **Recommendation: do not build DirectML.** Raw **D3D12 compute** is callable from C
(`COBJMACROS` C bindings; DXIL compiled offline with dxc and embedded like SPIR-V), but it
duplicates Vulkan with Windows-only reach. OpenCL (now) and Vulkan (P4) already cover every
DX12-class GPU on Windows, including this UHD 730.

### 7. TPU — the honest version
TPUs are **not** programmed with hand-written kernels from C. There is no public ISA and no
kernel-launch API; the supported path is **XLA** (Pallas kernels exist, but they are written in
Python/JAX and still go through XLA). The route a C engine can take is the one JAX and
PyTorch/XLA themselves use:

1. `dlopen("libtpu.so")` (pip package `libtpu`; preinstalled on Colab/Kaggle TPU VMs), call
   `GetPjrtApi()` → a `PJRT_Api*` function table
   ([pjrt_c_api.h](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h),
   [PyTorch/XLA plugin docs](https://docs.pytorch.org/xla/master/contribute/plugins.html)).
   This is `gpu.c`'s pattern exactly: **runtime-loadable, no SDK** (the header is plain C).
2. The engine **emits StableHLO (MLIR) as text** for a whole computation — one decoder layer, or
   the full forward + backward + optimizer step — with static shapes. PJRT compiles textual
   StableHLO, so no MLIR libraries are needed; Leela Chess Zero's XLA backend does this from C++
   ([Lc0 XLA backend](https://lczero.org/dev/wiki/xla-backend/)).
3. `PJRT_Client_Compile` → executable; `PJRT_Client_BufferFromHostBuffer` puts weights on the
   TPU **once**; `PJRT_LoadedExecutable_Execute` per step; outputs stay on device.

What that takes (**L+**): a graph builder re-expressing every op this engine has (matmul, RMSNorm,
RoPE, GQA attention + KV cache, SwiGLU, MoE routing, cross-entropy, AdamW/Muon, LoRA) as
StableHLO; static-shape bucketing (XLA recompiles per shape, so decode needs padded KV lengths);
bf16 throughout (the MXUs are bf16); and it abandons the op-at-a-time model — sending single
matmuls to a TPU through PJRT would lose to the CPU. It is a second execution engine, not a
backend plug-in. Testing is free (Colab v5e-1, Kaggle v5e-8), Linux x86-64 only, which the
existing `zig cc` cross-build already produces.
**Pragmatic alternative (recommended):** export weights (the engine already converts HF
safetensors ↔ its own format) and do TPU *training* in a ~300-line JAX/Flax or PyTorch/XLA
re-implementation of the same architecture on Kaggle/Colab TPU; import the trained weights back
for CPU/GPU inference in the C engine. A parity test (same weights + prompt → logits within ~1e-2
in bf16) keeps the two implementations honest. Take the PJRT route only if "the C engine itself
runs on TPU" becomes a goal in its own right.

### 8. CPU work that often beats a weak GPU
Measured here, the CPU is 3–4x the iGPU; and decode (M=1) is memory-bandwidth-bound on every
device, so bytes per weight matter more than FLOPs.
* **int8 with AVX-VNNI** (`vpdpbusd` on 256-bit registers; the i5-14400 has it). The current
  int8 path uses AVX2 widening multiplies. S–M, plausibly 1.5–2x on int8 GEMM/GEMV.
* **4-bit block quantization** (32-weight blocks + fp16 scale): halves decode bandwidth again
  vs int8. M. The same format should be what GPU GEMV kernels consume later.
* **fp16/bf16 weight storage** converted on the fly (F16C `vcvtph2ps`): halves memory for fp32
  models with no inference accuracy cost. S.
* **AVX-512 (VNNI/BF16) and AMX** exist on server Xeons (cloud VMs, possibly the Colab/Kaggle
  hosts), not on Raptor Lake client chips. Add as more `target()` kernels behind matrix.c's CPUID
  dispatch (AMX also needs Linux `arch_prctl(ARCH_REQ_XCOMP_PERM)`). M.
* BLIS-style packed/blocked GEMM, and P-core vs E-core aware thread splitting on hybrid CPUs. M.
* ARM NEON/SVE (`dotprod`, `i8mm`) for Apple Silicon / Graviton / Snapdragon builds. M.

## 2. The offload ladder (same for every GPU backend)

1. **GEMM offload, per-call upload** — `gpu.c` today. Wins only for big M; on a discrete GPU
   ~half the time is transfer.
2. **Weight residency** — weights uploaded once via an explicit *immutable* registration from the
   model code, invalidated after an optimizer step / LoRA merge. notes.md records why this cannot
   be inferred from pointers (attention passes activations as B). ~2x by itself.
3. **Activation residency** — RMSNorm, RoPE, softmax/attention, SwiGLU, residual add, embedding
   gather on-device; a whole layer, then the whole forward, without host round-trips. Only logits
   (or the sampled token) come back.
4. **Low precision** — fp16/bf16 storage + tensor-core/coopmat GEMM; int8/int4 dequant-in-GEMV,
   which is what finally makes M=1 decode worth moving to the GPU.
5. **Fused kernels + graph capture** — flash-style attention, fused dequant-GEMV, resident KV
   cache, CUDA Graphs / prerecorded Vulkan command buffers for the per-token loop.
6. **Training on device** — backward kernels, optimizer on device, gradients resident.

## 3. Recommended phased roadmap

| Phase | Work | Why this order | Test where |
|---|---|---|---|
| **P0** (other agent, now) | Backend layer + CUDA/HIP (cuBLAS/hipBLAS) + mock | NVIDIA on Colab/Kaggle is the only free real speedup target | mock locally; Colab T4 |
| **P1** | Weight-residency API (rung 2); port the OpenCL backend onto the layer | Biggest single win on every backend; OpenCL keeps a real device testable locally | this PC + Colab T4 |
| **P2** | CPU: AVX-VNNI int8, 4-bit blocks, fp16 weight storage | Most users (and this box) have no useful GPU; decode is bandwidth-bound | this PC |
| **P3** | NVRTC (+ hipRTC) kernel path: per-layer ops on device (rung 3); fp16/bf16 GEMM | Turns "matmul offload" into "model on GPU"; kernels stay C strings | Colab T4 / Kaggle 2xT4 |
| **P4** | Vulkan backend (GEMM → full layer), SPIR-V embedded as C arrays | One code base for AMD/Intel/Qualcomm/Apple-via-MoltenVK on Windows + Linux | this PC, Colab T4, lavapipe CI |
| **P5** | GPU decode: int8/int4 GEMV, resident KV, graph capture (rungs 4–5) | Needs P1 + P3 first | Colab / Kaggle |
| **P6** | Metal shim (MPS GEMM → MSL kernels) | Only once there are Mac users; unified memory makes it cheap | GitHub macOS runner |
| **P7** | TPU: JAX/PyTorch-XLA twin + weight round-trip; PJRT/StableHLO C path only if explicitly wanted | TPU needs whole-graph compilation — a different engine shape | Colab v5e-1, Kaggle v5e-8 |
| skip | DirectML, SYCL, Level Zero (unless Arc XMX becomes a target) | Maintenance mode / C++ only / no gain over OpenCL here | — |

## 4. Backend-interface requirements implied by this research

For the other agent's layer to grow into all of this without a redesign, its function table should
have (or reserve) these. Everything keeps the "return 0 → caller runs the CPU path" contract.

1. **Opaque device buffers** (`be_buf *`) with alloc/free/upload/download/copy and offset views.
   Host pointers must not be the identity of device memory — residency, PJRT buffers and Metal
   all need real handles.
2. **Weight registration / residency**: `register_weight(host_ptr, bytes, dtype) -> be_buf*`,
   `invalidate(host_ptr)` (after an optimizer step or LoRA merge), and a memory budget with LRU
   eviction (16 GB on a T4, ~2 GB shared on the UHD 730).
3. **Streams / async**: ops take a stream; `upload_async`, `event_record/wait`, `synchronize`.
   Maps to CUDA/HIP streams, OpenCL queues, Vulkan queues + fences, Metal command buffers, PJRT
   events. A default stream keeps today's synchronous callers unchanged.
4. **Dtypes**: f32, f16, bf16, int8 (+ per-channel scales), 4-bit blocks (+ fp16 scales). The GEMM
   entry should take `(dtypes, compute type, transA, transB, lda/ldb/ldc, alpha, beta)` — the
   cuBLAS/hipBLAS/MPS shape — so today's `matmul` / `_nt` / `_tn` trio and accumulate-into-grad
   (`beta = 1`) are special cases, and add **strided-batched GEMM** (attention heads) as one call.
5. **Ops beyond GEMM**, each optional, advertised by capability bits: rmsnorm, rope,
   softmax (+ causal mask), attention (fused, with KV cache), swiglu/gelu, add, embedding gather,
   cross-entropy, adamw — plus a generic `launch(kernel_name, grid, args)` for backends that
   compile kernel strings (OpenCL, NVRTC, hipRTC, MSL).
6. **Capabilities struct**: memory size, **unified-memory flag** (Apple/iGPU → skip copies),
   fp16/bf16/int8-dot/tensor-core support, max allocation, per-device dispatch thresholds
   (replacing the global `TF_GPU_MIN_M` / `TF_GPU_MIN_MACS` defaults).
7. **Graph capture / whole-graph mode**: `begin_capture(stream)`, `end_capture -> be_graph*`,
   `replay`. CUDA Graphs and Vulkan command buffers implement it directly; for TPU the graph *is*
   the StableHLO module, so an op-recording mode (ops append to a graph instead of executing) is
   the seam a PJRT backend could attach to later.
8. **Determinism + tests**: a flag forcing deterministic reductions; the mock backend checks every
   op against the CPU with the tensor-level `max|diff| / max|value|` metric from part [30]
   (per-element relative error gives false failures — notes.md).
9. **Multi-device** reserved (device index on buffers/streams): Kaggle's default GPU is **2xT4**.

## 5. Test matrix (verified September 2026)

| Platform | Cost | Accelerators today | Backends testable |
|---|---|---|---|
| **This PC** | free | Intel UHD 730 (OpenCL, Vulkan, Level Zero, D3D12); i5-14400 AVX2/AVX-VNNI | OpenCL, Vulkan, L0, CPU, mock |
| **Google Colab free** | free, not guaranteed, ≤12 h sessions | **T4 16 GB** most of the time; **TPU v5e-1** (TPU v2-8 retired) | CUDA/cuBLAS/NVRTC, OpenCL-on-NVIDIA, Vulkan (verify ICD), PJRT/TPU |
| Colab paid (Pro / pay-as-you-go) | ~$10/month+ | L4, A100, H100 subject to availability; **TPU v6e-1** (32 GB HBM) | same, plus bf16/fp8 tensor cores |
| **Kaggle** | free, weekly quota | **GPU T4 x2** (default; single T4 not offered), **TPU v5e-8**; **P100 retired 2026-09-15**; L4 only in some competitions; TPU v3-8 replaced by v5e-8 | CUDA incl. multi-GPU, PJRT/TPU (8 chips) |
| GitHub Actions | free (public repos) | macOS arm64 (Apple M-series GPU); Linux/Windows CPU | Metal, Vulkan via lavapipe/MoltenVK, CPU |
| Paid GPU rental | ~$0.2–2/h | RTX 30/40/50, AMD MI300X / Radeon, Intel Arc | HIP/hipBLAS, Vulkan-on-AMD, tensor-core paths |

Sources: [Colab FAQ](https://research.google.com/colaboratory/faq.html) (types vary, free ≤12 h),
[Colab: TPU v2-8 → v5e-1](https://github.com/googlecolab/colabtools/issues/5566),
[Kaggle CLI accelerator docs, PR of 2026-09-17](https://github.com/Kaggle/kaggle-cli/pull/1198)
(T4x2 default, TpuV5E8, L4),
[Kaggle: P100 sunset 2026-09-15](https://www.kaggle.com/discussions/product-announcements/735239),
[Kaggle: v5e-8 replaces v3-8](https://www.kaggle.com/product-announcements/607202),
[free-tier survey, Sept 2026](https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025).
Availability changes often: every notebook should print `nvidia-smi` / `vulkaninfo --summary` /
`jax.devices()` first, and the engine's `gpu-info` second.

## 6. Key caveats

* A GPU being present is not a speedup: on this PC, use `--no-gpu`. The iGPU is for correctness
  tests.
* Nothing has run on a discrete GPU yet; `tools/colab_gpu_test.ipynb` is still the first
  real-device validation.
* "No SDK" holds at **build** time only. cuBLAS/NVRTC/hipBLAS must exist on the **run** machine
  (true on Colab/Kaggle; often false on consumer Windows, especially AMD).
* Vulkan and D3D12 need an offline shader compiler for whoever edits shaders; the generated arrays
  are committed so normal builds stay SDK-free.
* Metal needs one Objective-C file (or `objc_msgSend` gymnastics) and cannot be cross-built from
  Windows.
* TPU support from C means building a second engine (a StableHLO emitter), not a backend. The
  JAX twin is the realistic path.
