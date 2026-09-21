# Pure-C LLM Engine — GPT-2-style Transformer, Agent & Multimodal (C11)

> **This is a big, actively-developed project — not a one-off demo.** It is a from-scratch,
> dependency-free machine-learning stack written in portable C11, built up over many committed
> steps and still growing along a real roadmap (see `notes.md`). "Minimal" here means *minimal
> dependencies* (no libraries, no BLAS, no framework — just the C standard library and `-lm`),
> **not** minimal scope. The scope is deliberately broad.

**What's built (all from scratch, all in pure C):**
- **Training** — full backpropagation through attention, feed-forward, LayerNorm and embeddings;
  Adam; cross-entropy; **numerically gradient-checked**. Train on your own text (`transformer
  train`) with configurable model size, train/val split, checkpointing and resume.
  The **modern LLaMA-family architecture trains too** (`llamatrain.c`): backward through
  RMSNorm + RoPE + GQA + SwiGLU, gradient-checked end to end — not just inference.
- **Inference** — temperature / top-k / top-p sampling, a **bit-exact KV cache**, and a
  token/text generation CLI.
- **Real GPT-2** — loads actual Hugging Face GPT-2 weights (124M–1.5B) and generates coherent
  text (`tools/gpt2_setup.py` → GPTC checkpoint).
- **A tool-using agent** — a coding-agent loop (parse tool calls → approval gate → execute →
  observe) with a pluggable backend, including a real-LLM backend (`tools/agent_llm.py`).
- **Multimodal** — image / video / audio *understanding* (patch/frame/waveform embeddings, trained
  via a shared multimodal path) **and** discrete-token media *generation* (image/audio/video),
  including text→media and media→text.
- **Portable** — one flat, readable module per concern; builds and runs on Linux, macOS, Windows,
  and **WebAssembly**.

**Where it's honest about scale:** quality is capped by what a from-scratch model on a CPU can be
— outputs are toy-scale, and the media codecs are crude. The deliverable is a *correct, complete,
extensible engine and pipeline*, not production fidelity. The roadmap (`notes.md`) lays out the
path to bigger, genuinely useful models: **SIMD + int8/int4 quantization**, and **modern
LLaMA-family architecture** support (RMSNorm, RoPE, SwiGLU, GQA + a SentencePiece tokenizer) so it
can load small modern models — turning this into a from-scratch mini-`llama.cpp`.

Every module (tensors, matrix math, softmax, GELU, LayerNorm, causal multi-head attention,
feed-forward, embeddings, sampling, checkpointing, BPE tokenizer) is still small enough to read in
one sitting — the codebase stays approachable even as the project grows.

## Why I'm building this

I'm building this because I don't want to be left behind as AI advances. I want to understand it
from the ground up — not just use it, but know how it actually works — so I can help make AI
better and better in the future. My goal is to become one of the people who *raise* AI: who help
it grow, improve, and carry it forward. This project is where I learn that craft, one honest,
from-scratch step at a time. — Marjon

Every stage here is built to be *understood*, not just run — because you can't help raise
something you don't understand.

## Project layout

Inspired by the Lua source tree: **every C source file lives flat in the project root —
there are no subfolders for code.** Each module is one concern, declared in a `.h` and
defined in the matching `.c`.

| File                              | Responsibility                                                |
| --------------------------------- | ------------------------------------------------------------- |
| `tensor.h` / `tensor.c`           | `Tensor` struct + aligned memory management                   |
| `matrix.h` / `matrix.c`           | `matmul`, `matadd`, `matscale`                                |
| `activation.h` / `activation.c`   | `softmax_rowwise`, `gelu`                                      |
| `layernorm.h` / `layernorm.c`     | `layernorm`                                                    |
| `rmsnorm.h` / `rmsnorm.c`         | `rmsnorm` (LLaMA-family; RMS-only norm, no mean/bias) + backward; `rmsnorm_heads` per-head variant (Qwen3 QK-norm) |
| `rope.h` / `rope.c`               | `rope` (LLaMA-family rotary position embeddings) + backward     |
| `swiglu.h` / `swiglu.c`           | `swiglu` (LLaMA-family gated FFN: SiLU gate) + backward         |
| `moe.h` / `moe.c`                 | Mixture-of-Experts FFN: top-k router over SwiGLU experts (+ optional shared expert), load-balancing aux loss, full backward; unchosen experts are never run |
| `gqa.h` / `gqa.c`                 | `gqa` (LLaMA-family grouped-query causal attention, optional sliding window) + backward |
| `quant.h` / `quant.c`             | int8 per-channel weight quantize/dequantize (for the LLMQ checkpoint) |
| `llama.h` / `llama.c`             | Modern (LLaMA-family) block + `LlamaModel` + `LLMA` checkpoint + KV-cache decode. Optional per-model features: q/k/v biases (Qwen2), per-layer sliding window with a ring-buffer cache (LLMA v3), QK-norm (Qwen3: per-head q/k RMSNorm before RoPE, `llama_model_set_qk_norm`; LLMA v4 flags word bit 0). v2/v3 files still load |
| `llamatrain.h` / `llamatrain.c`   | **Training** the LLaMA-family model: cached-activation forward + full backward (incl. QK-norm gains) |
| `lora.h` / `lora.c`               | LoRA low-rank adapters (`y += scale·(x·A)·B`) for fine-tuning a frozen base |
| `feedforward.h` / `feedforward.c` | Position-wise feed-forward MLP: Linear → GELU → Linear        |
| `attention.h` / `attention.c`     | `multihead_attention_forward` (causal), `transformer_block_forward` |
| `embedding.h` / `embedding.c`     | Token + positional embedding lookup                           |
| `sampler.h` / `sampler.c`         | Token sampling: greedy argmax + temperature / top-k / top-p (seeded RNG) |
| `model.h` / `model.c`             | `GPT` model: embeddings, stacked blocks, output head, training, KV-cache decode |
| `checkpoint.h` / `checkpoint.c`   | Save/load model weights (`gpt_save` / `gpt_load`, GPTC format) |
| `loss.h` / `loss.c`               | Softmax cross-entropy (forward + backward)                    |
| `optim.h` / `optim.c`             | Adam optimizer                                                |
| `muon.h` / `muon.c`               | Muon optimizer: Newton-Schulz orthogonalized momentum on hidden matrices, AdamW on the rest (`llama-train --optim muon`) |
| `tokenizer.h` / `tokenizer.c`     | Byte-level BPE: encode/decode + merge-learning (train) + save/load |
| `toolcall.h` / `toolcall.c`       | Parse `<tool …>`/`<done>` blocks from model output (agent)    |
| `toolexec.h` / `toolexec.c`       | Execute a tool call: write_file / read_file / run_command     |
| `risk.h` / `risk.c`               | run_command risk classifier: SAFE / REVIEW / BLOCK (sh + cmd + PowerShell parsing) |
| `approval.h` / `approval.c`       | Approval gate for tool actions (ask y/N, typed `yes` for BLOCK, or auto) |
| `agent.h` / `agent.c`             | Agent loop: parse → approve → execute → observe               |
| `image.h` / `image.c`             | Minimal image loading (binary PPM) — Phase E (multimodal)     |
| `patch.h` / `patch.c`             | ViT-style patch embedding + backward (trainable image front-end) |
| `video.h` / `video.c`             | Video = frames → per-frame patches + temporal encoding (Phase F) |
| `audio.h` / `audio.c`             | Audio = WAV → windowed frames → embeddings + temporal encoding (Phase F) |
| `imagegen.h` / `imagegen.c`       | Image ↔ tokens (color-cube) for autoregressive image **generation** (Phase G) |
| `audiogen.h` / `audiogen.c`       | Audio ↔ tokens (µ-law) for autoregressive audio **generation** (Phase G) |
| `corpus.h` / `corpus.c`           | Read training text from a file or a directory tree (recursive `.txt`) |
| `train.h` / `train.c`             | Train a GPT on a text file/dir (tokenizer + model + Adam loop) |
| `gpu.h` / `gpu.c`                 | Optional OpenCL matmul offload (`-DTF_GPU`); driver loaded at runtime, no SDK |
| `reward.h` / `reward.c`           | Verifiable reward for a sampled agent response: format + grounding + action match |
| `grpo.h` / `grpo.c`               | **GRPO**: RL from verifiable rewards — group-relative advantage, clipped policy gradient |
| `selftest.h` / `selftest.c`       | Built-in smoke test (`run_selftest`): the default `transformer` run           |
| `main.c`                          | Entry point + subcommands (`train`, `llama-train`, `grpo`, `generate`, `agent`, `imggen`, `audgen`, `llama`, `gpu-info`, `gpu-bench`, `simd-bench`) |
| `data/`                           | Your training data (git-ignored; keeps a sample)             |
| `model/`                          | One folder per model (`model/<name>/`); weights git-ignored, READMEs kept |
| `tools/`                          | Non-C helpers: crawlers (`crawl_wikipedia.py`, `crawl_stackoverflow.py`), SFT data (`make_sft.py`, `make_qa.py`, `mix_sft.py`, `sft_memtest.py`), `agent_local.py`, `upgrade_llma.py`, GPT-2/LLaMA converters |
| `Makefile`                        | Build rules                                                   |
| `tools/`                          | Helper scripts in **other** languages (Python/Go/…) — not C   |

## Build & run

```sh
make            # compiles all modules into ./transformer
./transformer   # runs the demo (tokenizer + model + checkpoint round-trips)
make clean      # removes object files and the binary
```

Or compile directly, without make:

```sh
gcc -std=c11 -O3 -march=native -o transformer *.c -lm
```

`clang` works as a drop-in replacement for `gcc`. On a machine without gcc/clang, the
project also builds with `zig cc -std=c11 -O2 -Wall -o transformer *.c -lm`.

### The complete guide (PDF)

`build/guide.pdf` is a 32-page end-to-end manual — compiling, data preparation, pre-training,
SFT, LoRA, GRPO, GPU acceleration, the Colab workflow, and embedding the engine in your own C
program, with every CLI parameter documented alongside its default and its use case. It is
committed, so you can just open it. Regenerate after changing the engine:

```sh
cd tools/guidepdf && go run .     # -> build/guide.pdf
```

### Optional GPU build (`GPU=1`)

```sh
make                # CPU only — the default, unchanged
make GPU=1          # + the OpenCL backend (gpu.c)
make cross-gpu      # a Linux x86_64 GPU binary, for Colab or any Linux GPU box
powershell -File tools/build_gpu.ps1     # all three, from Windows (no make here)

./transformer gpu-info      # what the backend found
./transformer gpu-bench     # CPU vs GPU on this engine's real GEMM shapes
```

**The GPU build needs no OpenCL SDK, header, import library or CUDA toolkit.** The driver is
opened with `dlopen`/`LoadLibrary` at runtime and its ~20 entry points resolved by name; the
handful of OpenCL types are declared in `gpu.c` rather than pulled from `<CL/cl.h>`. That is
what lets a Windows machine with no NVIDIA GPU **cross-compile a working GPU binary for
Colab** with nothing but `zig cc`.

`--gpu` / `--no-gpu` work on every subcommand and are stripped before the subcommand parses
its own flags, so one binary can be A/B'd against itself.

Only `matmul` and its two backward forms are offloaded — that is essentially all the FLOPs in
both inference and training, and it needs no change at any of the 49 call sites. Two gates
(`TF_GPU_MIN_M`, default 8, and `TF_GPU_MIN_MACS`, default 4194304 — both env vars, tunable
without a rebuild) keep small work on the CPU. **Single-token decode is `M=1` and
memory-bandwidth-bound, so it stays on the CPU on purpose**: uploading a 113 MB `lm_head` to
multiply one row by it loses badly.

> **A GPU being present does not mean it is worth using.** Measured here on an Intel UHD 730
> (24 EUs, sharing the CPU's memory bus) against a 10-core i5-14400: the GPU reaches ~62
> GFLOP/s versus the CPU's ~170–240, i.e. **0.27–0.37× — slower** — on the mid-size shapes,
> winning only on the huge `lm_head` GEMM (1.77×) where the CPU does worst. `gpu-bench` prints
> an explicit verdict and tells you to use `--no-gpu` when that is the right call. An
> integrated GPU losing to a good CPU is the normal result; the build exists for discrete
> GPUs, which is what `make cross-gpu` + `tools/colab_gpu_test.ipynb` are for.
>
> Numerically the two paths agree: the forward matmul and `dB` are **bit-identical**, and `dA`
> differs by 2.0e-07 only because `matrix.h` documents that the CPU reassociates that
> reduction to vectorize. Smoke-test part `[30]` gates this on ragged shapes.
> (That was measured against the portable CPU kernels. With the AVX2 kernels active the CPU side
> is a different FMA/association order, so expect ~1e-7 rather than bit-identity; `[30]` is a
> 1e-4 tolerance and was not re-run on a GPU for this change.)
>
> Known headroom (~2×): every operand is uploaded on every call. Caching weight *contents*
> cannot be done safely without an explicit immutability hint from the model code, because
> `attention.c` passes **activations** as the `B` operand for Q·Kᵀ — so "B is a weight" is not
> a safe assumption, and guessing wrong would silently train on stale data rather than crash.

## Supported platforms

Portable C11 with the platform-specific bits guarded (`_WIN32` / POSIX / WASM). Verified by
cross-compiling every source with `zig cc -target <triple>` (all `-Wall`-clean):

| Platform | Target triple | Status |
| --- | --- | --- |
| Linux x86-64 (glibc) | `x86_64-linux-gnu` | builds |
| Linux ARM64 (glibc) | `aarch64-linux-gnu` | builds |
| Linux x86-64 (musl) | `x86_64-linux-musl` | builds |
| Windows x86-64 | `x86_64-windows-gnu` | builds + **runs** (smoke test PASSED) |
| Windows ARM64 | `aarch64-windows-gnu` | builds |
| macOS Intel | `x86_64-macos` | builds |
| macOS Apple Silicon | `aarch64-macos` | builds |
| **WebAssembly (WASI)** | `wasm32-wasi` | builds + **runs** (via Node WASI, PASSED) |

### Threads

`matmul` and `matmul_backward` run on a **persistent thread pool** (auto-enabled for large
matrices; small ones stay single-threaded and bit-exact). It uses all CPU cores by default;
override with the `TF_THREADS` env var (e.g. `TF_THREADS=4`). Measured ~2.7× on 8 threads for
a d_model-256 / 4-layer model. WebAssembly runs single-threaded (no thread model).

### Vectorization (portable C, no intrinsics)

The forward `matmul` was already auto-vectorized at `-O2` — its inner loop is an AXPY over
contiguous columns. The **backward** was not: both of its loops were floating-point *reductions*,
which the compiler may not reassociate, and it measured **~10× slower than the forward**. Both were
restructured in plain portable C (no `<immintrin.h>`, no target guards, so WebAssembly benefits
too):

| kernel (M=128) | before | after | |
| --- | --- | --- | --- |
| backward, 576×576 | 19.9 GF/s | **141.6 GF/s** | 7.1× |
| backward, 576×1536 | 20.0 GF/s | **181.2 GF/s** | 9.1× |
| backward, 1536×576 | 20.1 GF/s | **174.2 GF/s** | 8.7× |
| backward, lm_head 576×49152 | 21.3 GF/s | **39.4 GF/s** | 1.9× (M=16 limits row parallelism) |

**Numerics:** `dB` accumulates in the same order as before and is **bit-exact**; `dA` is a
dot-product reduction given several independent accumulators so it can vectorize, which
reassociates the sum — it is *not* bit-identical to a naive left-to-right scalar sum (typically
slightly more accurate). Threading stays bit-exact either way, since each output row is produced
entirely by one thread. The forward is untouched, so the **KV-cache bit-exactness gate is still
`0.00e+00`** and every gradient check is numerically unchanged.

**Frozen-weight backward.** `matmul_backward_dx` / `swiglu_backward_dx` compute only the input
gradient, skipping `dW` for weights that will never be updated. LoRA fine-tuning uses them
throughout (and skips the embedding scatter): measured **61.2 → 42.4 ms/step, 1.44×**, on a
d_model-256 / 4-layer model. Not a full halving — the forward pass and the adapters' own work
don't shrink.

### SIMD kernels (AVX2 + FMA, runtime dispatch)

On x86 CPUs with AVX2+FMA, `matmul`, `matmul_backward`, `matmul_backward_dx` and the int8
`qacc_columns` (the core of `qmatmul`) switch to hand-written intrinsic kernels in `matrix.c`.
They are selected **at runtime** (CPUID + XGETBV, once, at the first matmul) and compiled with
`__attribute__((target("avx2,fma")))`, so no build flag changes: the default build, the
baseline-x86-64 cross builds and the Colab binary all get them, a CPU without AVX2 falls back to
the portable C, and aarch64 / wasm32 compile none of it. `TF_SIMD=0` forces the portable path.
`./transformer simd-bench [reps]` times one against the other in the same binary.

What makes them faster is **register blocking**, not vector width — a native `zig cc` build
already auto-vectorizes the portable loops with AVX2. A 4×16 tile of the output stays in
registers for the whole reduction, `B` is packed into 128 KB panels (for `lm_head` an unpacked
strip is a TLB miss per row), and `dA = dC @ Bᵀ` gathers `Bᵀ` while packing instead of
transposing. Single-row forwards (decode, M=1) use a streaming AXPY instead: they are
bandwidth-bound and gain little. The int8 kernel does 16 MACs per `vpmaddwd`.

Measured with **`TF_THREADS=1`** (kernels, not the thread pool), i5-14400, median of 3 runs ×
5 interleaved reps, a 3-thread training job sharing the CPU throughout. GFLOP/s in brackets:

| shape (M×K×N), SmolLM2-135M | portable C, native build | AVX2 kernels | speedup | portable C, baseline x86-64 build | speedup there |
| --- | --- | --- | --- | --- | --- |
| fwd 128×576×1536 (gate/up) | 9.00 ms (25) | 2.04 ms (111) | **4.0×** | 10.45 ms (22) | 5.2× |
| fwd 128×1536×576 (down) | 9.13 ms (25) | 2.12 ms (107) | **4.4×** | 10.84 ms (21) | 5.2× |
| fwd 128×576×576 (q/o proj) | 1.75 ms (49) | 0.87 ms (98) | **2.1×** | 3.24 ms (26) | 4.0× |
| fwd 128×576×49152 (lm_head) | 981 ms (7) | 111 ms (65) | **8.3×** | 1169 ms (6) | 9.9× |
| fwd 1×576×1536 (decode) | 0.07 ms (25) | 0.07 ms (25) | 1.0× | 0.08 ms (22) | 1.2× |
| fwd 1×576×49152 (decode lm_head) | 8.72 ms (6.5) | 7.98 ms (7.1) | 1.1× | 9.29 ms (6.1) | 1.3× |
| bwd 128×576×1536 (dA + dB) | 17.1 ms (27) | 4.81 ms (94) | **3.5×** | 18.2 ms (25) | 3.9× |
| bwd 128×1536×576 | 17.6 ms (26) | 5.44 ms (83) | **3.2×** | 18.7 ms (24) | 3.9× |
| bwd 128×576×576 | 5.69 ms (30) | 1.69 ms (101) | **3.4×** | 5.87 ms (29) | 3.7× |
| dx 128×576×49152 (frozen lm_head) | 994 ms (7) | 96 ms (75) | **10.1×** | 1132 ms (6) | 11.1× |
| int8 decode 1×576×1536 | 0.05 ms (35) | 0.02 ms (83) | **2.4×** | 0.19 ms (9) | 7.6× |
| int8 decode 1×576×49152 | 3.15 ms (18) | 1.58 ms (36) | **1.9×** | 5.98 ms (9.5) | 4.0× |

The "native build" columns are the honest ones for this machine's default build; the "baseline"
columns are what a portable binary (`make cross`, `cross-gpu`) gains from the dispatch alone.
These are single-kernel numbers — end-to-end training/decode time was **not** measured here, and
decode (M=1) is essentially unchanged.

**Decode (M=1) threading and sampling.** Row-parallel matmul leaves a 1-row product on one
thread, so a forward with `M < 4 × threads` is split over **output columns** (64-wide units)
instead — still one FMA chain per element, bit-identical to one thread (selftest `[38]`). On a
synthetic 30-layer 135M-shaped model this takes a decode step from 33.1 to 23.3 ms at
`TF_THREADS=3` (1 thread: unchanged, it is DRAM-bound). `sample_token` sorts only the top-k /
top-p candidates (threshold by float exponent, total-order comparator, full-sort fallback), which
gives the same token as a full sort for the same RNG draw: 5.1 → 0.5–2.9 ms per token on
peaked 49152-way rows. `llama_decode_step_lora(…, NULL, …)` skips the last layer's attention/FFN,
the final norm and lm_head, for prompt prefill (GRPO uses it). Details: notes.md.

**Numerics.** Each output element is one FMA chain in reduction order, whichever tile, panel,
row block or thread computes it — so the SIMD path is **bit-exact across row counts and thread
counts**, and the KV-cache / LoRA-decode gates (`[4]`, `[19]`, `[22]`, `[26]`) stay `0.00e+00`
(also on the baseline-x86-64 build, where the rest of the code is compiled without FMA). It is
*not* bit-identical to the portable kernels: ~1e-7 relative on the forward (0 in a native build,
where the portable loop is FMA-contracted too) and up to ~1e-6 on `dA`/`dB`, which associate
differently. Against a double-precision reference the SIMD `dA` is the *more* accurate of the
two (N=49152: 4.4e-06 vs 1.6e-05 relative). Selftest part `[34]` gates all of this on awkward
shapes; the int8 kernel is integer arithmetic and must match exactly.

### Cross-compiling

Build **all** targets at once into `build/` (needs `zig` + a POSIX shell):

```sh
make cross          # -> build/transformer-<target>[.exe|.wasm] for every platform
```

Or cross-compile one target by hand (no OS-specific toolchain needed — `zig` bundles them):

```sh
zig cc -target x86_64-linux-gnu  -std=c11 -O2 -o transformer      *.c -lm   # Linux
zig cc -target aarch64-macos     -std=c11 -O2 -o transformer      *.c       # macOS (libm in libSystem)
zig cc -target wasm32-wasi       -std=c11 -O2 -o transformer.wasm *.c       # WebAssembly
```

Run the WebAssembly build with any WASI runtime (grant filesystem access for `train`):

```sh
wasmtime --dir=. transformer.wasm                       # or wasmer
node --experimental-wasi-unstable-preview1 run.js transformer.wasm   # Node's built-in WASI
```

> WASM note: WebAssembly has no process model, so the agent's `run_command` tool is disabled
> under WASM (it returns "not supported"); `write_file`/`read_file`, training, and generation
> all work. Workspace confinement is lexical-only there (no symlink resolution).

---

## CLI guide

### `transformer` (the demo binary)

Runs the full pipeline and prints a report. It takes no arguments today; edit the config
constants at the top of `run_selftest()` in `selftest.c` to change vocab size, dimensions, heads,
layers, prompt, or number of generated tokens.

```sh
$ ./transformer
=== Minimal GPT-2 Style Transformer Demo ===

[1] Byte-level BPE tokenizer
    text     : "hello hello" (11 bytes)
    encoded  : 5 tokens -> 258 111 32 258 111
    decoded  : "hello hello"
    round-trip: OK | compressed: yes

[2] GPT model
    config: vocab=16 d_model=8 heads=2 layers=2 d_ff=32
    generate: 1 5 9 -> 1 5 9 11 6 11 14 14
    logits  : no NaN/Inf

[3] Checkpoint save/load
    saved 'model.bin': OK | reloaded logits identical: yes

Output validation: PASSED (no NaN/Inf)
```

Exit code is `0` when all checks pass, `1` otherwise — handy in scripts:

```sh
./transformer && echo "smoke test ok"
```

### `transformer train <path> [flags]`

Train the model on **your own text**. `<path>` may be a single file **or a directory** —
a directory is read **recursively**, concatenating every `.txt` under it (including
subfolders like `data/wiki/`). The model size is fully configurable via flags:

```sh
transformer train data/wiki                       # defaults (d_model 128, 3 layers, ...)
transformer train data/wiki --steps 20000 --out ai.bin --tok ai.tok

# Scale the model up (bigger = slower but better):
transformer train data/wiki --d-model 256 --layers 6 --heads 8 --ctx 128 --steps 30000
```

| Flag | Meaning (default) | | Flag | Meaning (default) |
| --- | --- | --- | --- | --- |
| `--out F`   | model out (`model.bin`) | | `--d-model N` | model dim (128) |
| `--tok F`   | tokenizer out (`tok.bin`) | | `--layers N` | transformer blocks (3) |
| `--merges N`| BPE merges (300) | | `--heads N`  | attention heads (4) |
| `--steps N` | Adam steps (1000) | | `--d-ff N`   | FFN dim (4·d_model) |
| `--lr F`    | learning rate (3e-4) | | `--ctx N`    | context window (64) |
| `--seed N`  | RNG seed (1234) | | `--val-frac F` | held-out fraction (0.1) |
| `--checkpoint-every N` | save every N steps | | `--resume F` | continue a saved model |
| `--weight-decay F` | AdamW decay (0 = off) | | `--clip F` | grad-norm clip (0 = off) |
| `--warmup N` | LR warmup steps (0 = off) | | `--min-lr F` | cosine floor LR (<0 = constant) |
| `--batch N` | sequences per step (1) | | | |

**Batching (`--batch`).** Training is one sequence per optimizer step by default. `--batch N`
accumulates N forward+backward passes, scales the summed gradient by 1/N, and then takes a single
step — the gradient of a real batch of N, at N× the cost per step. Verified against its exact
definition in the smoke test (part `[25]`): accumulating two sequences and halving equals the mean
of their separately computed gradients (rel 7.6e-10), and `--batch 1` is **bit-identical** to the
old path.

> **Scale the learning rate with the batch, or it will look like a regression.** Measured on
> `data/sft.txt`, all runs processing the same 400 sequences: `--batch 1 --lr 1e-3` → loss
> **0.0124**; `--batch 8 --lr 1e-3` (only 50 steps) → **1.87**, badly undertrained purely from
> having 8× fewer updates; `--batch 8 --lr 8e-3` → **0.0265**, back in line while taking an eighth
> of the optimizer steps. Spending the extra compute — `--batch 8` for the full 400 steps — gives
> the best result, **0.0093**. A less noisy gradient buys stability and larger usable learning
> rates, not a free win at fixed compute.

**Training recipe options.** `--weight-decay` switches Adam to **AdamW** (decoupled decay, applied
to weight matrices only — biases and norm gains are exempt, since decaying a norm gain fights the
normalization). `--clip` applies one **global** gradient-norm clip across the whole model (the
direction is preserved, only the length is capped; a NaN/Inf norm zeroes the step so one bad batch
can't poison the weights). `--warmup`/`--min-lr` enable **linear warmup then cosine decay**. All
four default to OFF so previously-recorded runs still reproduce exactly. They earn their keep on
long or fine-tuning runs and against divergence — on a short overfit-a-tiny-corpus run, plain Adam
at a constant LR will still reach a lower loss, because warmup and cosine decay spend part of the
budget at a lower LR:

```sh
transformer train data/wiki --steps 20000 --lr 3e-4 \
  --weight-decay 0.01 --clip 1.0 --warmup 200 --min-lr 3e-5
```

`--d-model` must be divisible by `--heads`. Training prints **train + held-out `val`** loss
(so you can tell learning from memorizing); `--checkpoint-every` saves periodically, and
`--resume model.bin` continues training a saved model across sessions (keeps its config +
tokenizer). CPU-only, so bigger/cleaner text and **many more `steps`** help. Each trained
`model.bin`+`tok.bin` pair is a self-contained model you can keep for a specific role.

```sh
# train, checkpointing every 500 steps, holding out 10% for validation:
transformer train data/wiki --out ml.bin --tok ml.tok --steps 5000 --checkpoint-every 500
# later, continue where it left off:
transformer train data/wiki --resume ml.bin --out ml.bin --tok ml.tok --steps 5000
```

**Build a real corpus by crawling** (see `tools/crawl_wikipedia.py`):

```sh
python tools/crawl_wikipedia.py "Machine learning" --max-pages 50   # -> data/wiki/*.txt
transformer train data/wiki --out ml.bin --tok ml.tok --steps 10000
echo "machine learning is" | transformer generate ml.bin ml.tok
```

The crawler fetches plain-text Wikipedia extracts and follows article links (breadth-first)
so the text stays topically *connected*. Then `train` ingests the whole `data/wiki/` folder.

### Instruction tuning (SFT) on your own data

This is **stage 3 of the recipe** (see the "5 stages" in `memory.md`) — the step that makes a
model *follow instructions* instead of just continuing text. `data/instruct.txt` is a tiny
example dataset of `Q: <instruction>` / `A: <response>` pairs whose answers are the agent's
tool-call format (`<tool …>` / `<done>`). Train on it, then feed an instruction and watch the
model produce the matching tool call:

```sh
transformer train data/instruct.txt --name instruct --d-model 96 --layers 3 \
  --heads 4 --ctx 96 --merges 48 --steps 4000 --lr 1e-3
echo "Q: read notes.txt"      | transformer generate --name instruct
#   -> A: <tool name="read_file" path="notes.txt"></tool>
echo "Q: the task is finished" | transformer generate --name instruct
#   -> A: <done>Task complete.</done>
```

Verified: a small model learns to route each held-in instruction to its correct response.
(Note: keep `--merges` low so the BPE tokens stay fine-grained enough for prompts to align to
token boundaries.)

**True masked SFT (`--sft`).** The run above trains on the whole token stream (the model
*memorizes* the corpus). Real SFT supervises **only the response** and masks the prompt — that's
the `--sft` mode. The file is one `prompt<TAB>response` example per line (`data/sft.txt`); the
loss is computed on the response tokens only (prompt + padding get target `-1` = ignored), so the
model is trained to *produce* the answer, not reconstruct the prompt:

```sh
transformer train data/sft.txt --sft --name sft --d-model 96 --layers 3 \
  --heads 4 --merges 48 --steps 3000 --lr 1e-3
echo "read notes.txt" | transformer generate --name sft
#   -> <tool name="read_file" path="notes.txt"></tool>     (stops cleanly at EOS)
```

Verified: every instruction routes to its exact tool-call/`<done>` response, and generation
**stops cleanly** at the end. This is the same loss the smoke-test part `[9]` uses, now available
on your own data. **EOS/stop token:** `--sft` reserves one extra vocab slot as an end-of-sequence
token (`EOS = tokenizer.vocab_size`), appends it to every response and supervises predicting it, so
the model learns to *end* its answer; `generate` halts as soon as it samples an id ≥ the
tokenizer's vocab (non-SFT models have no extra id, so they're unaffected — fully backward
compatible).

**Honest scale:** at this size the model *memorizes* the pairs — the value is seeing the SFT
technique work end-to-end on real data via the CLI; generalizing to unseen instructions needs far
more data + a bigger model.

### `transformer llama-train <path> [flags]` — train the **modern** architecture

Identical flags to `train`, but trains the LLaMA-family architecture (RMSNorm + RoPE + GQA +
SwiGLU, via `llamatrain.c`) and writes an **LLMA** checkpoint that `transformer llama` generates
from. Extra flags: `--kv-heads N` (GQA key/value heads; default = `--heads`, i.e. plain
multi-head) and `--rope-base F` (default 10000). `head_dim` is `d_model / heads` and must be even,
since RoPE rotates feature pairs.

**Sliding-window attention** (Mistral, Gemma 2/3, gpt-oss): `--window W` makes each position attend
only to its `W` most recent positions (itself included); `--global-every N` keeps every Nth layer
full/global (`2` = alternate local/global like Gemma 2 and gpt-oss, `6` = 5 local : 1 global like
Gemma 3, `0` = every layer windowed like Mistral). The window is stored per layer in the checkpoint
(LLMA **v3**; a model without a window is still written as v2, byte-identical to before, and old v2
checkpoints load unchanged). Windowed layers get a `W`-slot ring-buffer KV cache during decode, so
their memory no longer grows with context. Applies to fresh models; `--resume` keeps the window the
checkpoint was saved with. `tools/convert_llama.py` reads `sliding_window` / `layer_types` from an HF
config. Smoke-test part `[32]` gates it (gradient check, `W >= seq` == full attention exactly,
out-of-window tokens have zero influence, decode == forward bit-exact with the ring wrapping).

```sh
# Train a small modern model with GQA (4 query heads sharing 2 KV heads):
transformer llama-train data/sample.txt --steps 600 --d-model 64 --layers 2 \
  --heads 4 --kv-heads 2 --ctx 32 --merges 60 --lr 1e-3 \
  --clip 1.0 --warmup 50 --min-lr 1e-4 --weight-decay 0.01 --out lm.bin --tok lm.tok

# Generate from it — TEXT in / TEXT out (pass the tokenizer as the 3rd argument):
echo "the" | transformer llama lm.bin 30 lm.tok
# -> the cat likes the mat. the dog likes the log. the quick cat sat. ...
```

It supports everything the GPT trainer does: train/val split, `--checkpoint-every`, `--resume`
(rebuilds a trainable model from the inference checkpoint and copies the weights across), the
recipe flags, and **`--sft`** masked instruction tuning:

```sh
transformer llama-train data/sft.txt --sft --steps 400 --d-model 64 --layers 2 \
  --heads 4 --merges 48 --lr 1e-3 --clip 1.0 --out ls.bin --tok ls.tok
echo "read notes.txt"       | transformer llama ls.bin 40 ls.tok
#   -> <tool name="read_file" path="notes.txt"></tool>
echo "the task is finished" | transformer llama ls.bin 40 ls.tok
#   -> <done>Task complete.</done>
```

Verified: both routed to the exact tool call and stopped cleanly at EOS. (At this size the model
memorizes the pairs — the point is the modern-architecture SFT pipeline running end to end.)

#### LoRA fine-tuning (`--lora-r`)

`--lora-r N` **freezes the base** and trains only low-rank adapters (`lora.c`) on `Wq/Wk/Wv/Wo`:
`y = x·W + (alpha/r)·(x·A)·B`, with `A: (d_in, r)` and `B: (r, d_out)`. `B` starts at **zero**, so
an adapted model is bit-identical to its base before the first step. The adapters are **merged into
the weights before saving** (`W += scale·A·B`), so the output is an ordinary LLMA checkpoint that
`transformer llama` runs with no adapter support and no inference-time cost.

```sh
# fine-tune a pretrained checkpoint; --tok must be the BASE's tokenizer
transformer llama-train data/sft.txt --sft --resume base.bin \
  --lora-r 8 --lora-alpha 16 --steps 600 --lr 5e-3 --clip 1.0 \
  --out lora.bin --tok base.tok
```

Fine-tuning reuses the base's tokenizer (a fresh one would assign different ids to the same text,
making the pretrained embeddings meaningless) and grows the vocab by one for the SFT `EOS` slot.
`--checkpoint-every` is ignored under LoRA — merging mid-run would zero `B` and desync Adam's
moments for it, so adapters are folded in only at the end.

#### Muon optimizer (`--optim muon`)

`--optim muon` swaps Adam for **Muon** (Keller Jordan 2024; Moonshot's "Muon is Scalable for LLM
Training" 2025, the optimizer behind Kimi K2) on the **hidden weight matrices** (`Wq/Wk/Wv/Wo`,
`Wgate/Wup/Wdown`): Nesterov momentum (0.95), then 5 quintic Newton-Schulz iterations that push
every singular value of the update to ~1 (orthogonalize it), then
`W -= lr·(0.2·sqrt(max(rows, cols))·O + wd·W)`. The `0.2·sqrt(max)` factor is Moonshot's RMS
matching — it makes Muon's update the same size as AdamW's — so the **same `--lr` and
`--weight-decay`** apply. The token embedding, `lm_head`, norm gains and biases stay on **AdamW**
(1-D tensors have no matrix to orthogonalize; embedding/output rows are per-token, and
orthogonalizing across the vocab spreads updates onto tokens that were never seen). Under
`--lora-r` every adapter `A`/`B` is a genuine linear map, so all of them use Muon. Default stays
`adam`, and the Adam path is **bit-identical** to before (same model bytes, same log). Newton-Schulz
runs through the ordinary matmul, so it gets the AVX2 kernels and the thread pool.

Honest numbers (tiny synthetic corpus, `--d-model 64 --layers 2 --ctx 32 --steps 600 --lr 3e-3`,
three seeds): final train loss Adam 1.20 / 1.21 / 1.41 vs Muon 1.20 / 1.16 / 1.41, val 1.44 / 1.46 /
1.34 vs 1.47 / 1.53 / 1.34 — a **wash** at this scale, and Muon's steps took **~2.2× longer** (the
Newton-Schulz cost is `O(r²c)` per matrix per step and does not shrink with the batch, so it
dominates when a step sees only 32 tokens; at real batch sizes it is a small fraction). Muon's
published wins are at scale (~2× compute efficiency for 1B+ models); none of that is measured
here. Selftest part `[35]` gates the orthogonalization, the RMS matching, the decay, the routing
and the unchanged Adam path.

> **Honest scope.** The mechanism is verified in the smoke test (part `[24]`): zero-init
> equivalence, a numerical gradient check on the adapters, a frozen-base byte-comparison, and
> merge equivalence. But **LoRA only pays off on a genuinely pretrained base.** Adapting one of the
> toy models here (trained on a 440-byte corpus) reaches SFT loss ~2.5 and emits garbage, where a
> full fine-tune of the same data reaches 0.0125 and emits the exact tool call — the frozen
> embeddings and `lm_head` simply cannot represent the new tokens, and attention adapters alone
> can't fix that. The parameter saving is also only visible at scale: on the tiny smoke-test model
> the adapters are 27.9% of the base, whereas at `d_model` 576 with `r=8` they are a few percent.

### `transformer grpo <tasks.txt> [flags]` — RL from **verifiable rewards**

Every other trainer here is supervised: it is shown the right answer and pushed toward it. `grpo`
implements **GRPO** (Group Relative Policy Optimization — the algorithm behind DeepSeek-R1), which
instead lets the model **act**, scores what it did, and pushes it toward its own better-than-average
attempts and away from its worse-than-average ones.

Per task, per step:

1. **sample G rollouts** from the current policy (KV-cache decode *with* the LoRA adapters applied,
   so the samples really do come from the policy being updated);
2. **score each** with a verifiable reward (`reward.h`) — format, grounding, action match;
3. **normalize within the group**: `Â_i = (r_i − mean) / (std + 1e-4)`. This is the "group relative"
   part, and it is what replaces a value network: the group's own mean *is* the baseline;
4. one clipped policy-gradient step, optionally pulled toward a reference policy by a KL penalty.

```sh
# build a task file whose reference actions are all grounded (see the warning below)
python tools/grpo_tasks.py --out data/grpo_tasks.txt --actions-only data/clean_reason.txt

transformer grpo data/grpo_tasks.txt \
  --model model/smollm/reason_e2.bin --hf-tok model/smollm/tok.hf \
  --out model/smollm/grpo.bin \
  --group 8 --steps 200 --max-new 96 --temp 1.0 --lr 1e-5 --lora-r 8 --verbose
```

The task file is the **same `prompt<TAB>response` format** the SFT trainer reads — the response
column becomes the *reference action* the reward scores against — so any corpus from
`tools/clean_sft.py` drops straight in.

**How this differs from the STaR loop** (`tools/self_train.py`): STaR is rejection sampling. It
keeps the successful rollouts as SFT rows and throws the failures away, so a failed rollout teaches
nothing. GRPO uses the failures too — a below-average rollout gets a *negative* advantage, actively
suppressing it. That suppression is the capability rejection sampling does not have.

**Three implementation notes worth knowing:**

- **The whole backward is one function.** `∇_logits log π(o_t) = onehot − p`, so any objective of
  the form `Σ w_t · log π(o_t)` is just the cross-entropy gradient with a per-token weight.
  `pg_backward` (`loss.h`) is `cross_entropy_backward` with `1/N` replaced by `coeff[t]`; everything
  downstream (`llama_backward_accum`, `clip_grad_norm`, `adam_step`, `llama_lora_merge`) is reused
  untouched. The prompt is masked out with `targets[t] = -1`, exactly as in SFT.
- **The reference policy is free.** With LoRA the base is frozen, so running the same model with the
  adapters switched off *is* the reference. No second copy of a 621 MB checkpoint. `--kl` costs one
  extra forward per rollout and no extra memory.
- **Rollouts stop at `</tool>`/`</done>`**, not just at EOS. `toolcall_parse` takes the first block
  and `agent.c` executes only that, so text after it is discarded at runtime — scoring or training
  on tokens the harness throws away would optimize something that does not exist in deployment. It
  also cuts rollout cost 2–3× in practice (a 135M model keeps talking long after it is done).

The run saves **once, at the end**: mid-run saving would mean merging the adapters, and
`llama_lora_merge` resets `B` to zero, after which `dA` is identically zero and training stalls.

> **Honest scope.** GRPO learns from **variance**. If every rollout in a group scores the same, every
> advantage is zero, and the group is skipped before any forward pass (the run reports how many).
> Where a 135M model is *uniformly* right or *uniformly* wrong there is no signal at all — and lever
> #4 measured that its residual errors are systematic and low-variance. On `reason_e2.bin` against
> `data/grpo_tasks.txt` the mean reward is already ~0.94, so most groups are flat: the realistic win
> is sharpening the cases it gets right only sometimes and suppressing the confidently-wrong ones,
> not a new skill. Watch the `--verbose` breakdown for **reward hacking** — a rising total with a
> flat `match` term means the model found a cheap way to satisfy format + grounding — and gate any
> promotion on `tools/remeasure.py`, exactly as `4bea3b9` was gated.
>
> **A bad row in SFT data teaches one wrong answer; a bad row in an RL task file teaches the reward
> to lie.** `tools/grpo_tasks.py` therefore puts every reference action through `clean_sft.verify()`
> and drops what it rejects rather than trying to fix it.

Verified in the smoke test: `[26]` the LoRA decode matches the LoRA training forward to `0.00e+00`
(a wrong adapter/RoPE order would make every importance ratio meaningless), `[27]` the reward on a
10-case adversarial probe, `[28]` the objective arithmetic including one-sided clipping and the
`pg_backward(1/N) == cross_entropy_backward` identity, and `[29]` a numerical gradient check of the
surrogate plus a closed GRPO loop that drives a toy reward from chance (0.083) to 1.000.

#### How to compare two checkpoints (`grpo --eval-only` + `tools/evalcard.py`)

`grpo --lr 0` is **not** an evaluation: it draws tasks *with replacement* (40 draws cover ~36 of 200
tasks), shares one sampling RNG across the run (once one rollout's length differs between two
checkpoints, every later draw is unpaired noise), still runs the backward pass, and never logs
per-task rewards. Use `--eval-only` instead: inference-only load (no LoRA, optimizer, backward or
save), every task visited **once, in file order**, and rollout *(task i, draw k)* sampled from its
**own** RNG seeded by `hash(--seed, i, k)` — so both checkpoints see the *same* noise on every
task and the per-task difference is paired.

```bash
# primary: greedy (the --eval-only default is --temp 0, --k 1)
transformer grpo data/grpo_tasks.txt --model model/A.bin --hf-tok model/smollm/tok.hf \
  --eval-only --reward v2 --dump evalA.jsonl
transformer grpo data/grpo_tasks.txt --model model/B.bin --hf-tok model/smollm/tok.hf \
  --eval-only --reward v2 --dump evalB.jsonl
python tools/evalcard.py evalA.jsonl evalB.jsonl

# secondary: sampled, 4 draws per task (same --seed for both!)
transformer grpo data/grpo_tasks.txt --model model/A.bin --hf-tok model/smollm/tok.hf \
  --eval-only --reward v2 --temp 1 --k 4 --dump sampA.jsonl
#   ... same for B -> sampB.jsonl
python tools/evalcard.py sampA.jsonl sampB.jsonl
```

Each dump line is one rollout: `task`, `k`, `family` (the reference action's kind: write / read /
run / delete / done / prose), `total`, `format`, `grounded`, `match`, `exact` (match term at its
maximum), `action`, `stop` (`eos`/`tag`/`cap`), `tokens`, `task_hash`, `response`. `--limit N`
evaluates only the first N tasks. `evalcard.py` pairs rollouts by (task, k) and prints, for B − A:
mean total and exact-match with a **95% paired bootstrap CI over tasks** (10k resamples, fixed
seed; the k draws are averaged within a task first, because tasks — not rollouts — are the
independent units), an **exact McNemar test** on per-task exact match (k = 1, or the majority of k),
a per-family breakdown, and the share of tasks whose response text changed — `0%` means *the policy
did not move*, whatever the training log said. It refuses dumps whose `task_hash` disagree (different
task files) and warns when reward version / temperature / seed differ. With one file it prints a
summary with CIs.

**Statistical recommendation.** Decide on the **primary** metric — greedy exact-match, McNemar
p < 0.05 — and treat the **secondary** — `--temp 1 --k 4` mean total, paired bootstrap CI over
tasks — as supporting evidence (it sees changes in the *distribution* that greedy decoding hides).
Size the task set before the run: about **125 tasks to detect a 5-point** exact-match change and
**~350 for 3 points** (80% power, α = 0.05, assuming the two checkpoints disagree on ~4% of tasks;
if they disagree on more, you need proportionally more tasks). On 40 tasks only a very large effect
is detectable — a CI that spans 0 there means "not measured", not "no effect".

Verified by `[39]`: two eval runs are bit-identical; evaluating task 1 with a different model *and*
a shorter budget leaves tasks 0 and 2 bit-identical (with the old shared RNG the same change shifts
task 2); parameters are bit-identical and gradients zero afterwards.

### `transformer generate [model.bin] [tok.bin]`

A generation loop driven by the KV cache (`gpt_decode_step`) + sampler (`sample_token`),
one prompt per stdin line:

```sh
# TEXT in / TEXT out — pass a trained model AND its tokenizer:
echo "the cat" | ./transformer generate model.bin tok.bin
# the cat sat on the mat. the dog sat on the log. ...

# Token IDs — loaded weights, no tokenizer:
echo "1 5 9" | ./transformer generate model.bin
# gen: 1 5 9 [15] [15] [5] ...            (bracketed tokens are generated)

# No args -> random toy model (token IDs in):
echo "1 5 9" | ./transformer generate
```

Passing a tokenizer switches to **text-in/text-out**. Defaults: 40 new tokens, temperature
0.8, top-k 40, top-p 0.95 (edit `run_generate` in `main.c`). Pair with `train` above to
generate from a model trained on your own data.

### `transformer agent [--auto] [--workspace DIR]` (agent loop demo)

Demonstrates the coding-agent loop in pure C: **parse → approve → execute → observe**.
The model emits tool blocks; each approved call runs and its result is fed back:

```
<tool name="write_file" path="a.txt">...body...</tool>
<tool name="read_file" path="a.txt"></tool>
<tool name="run_command" command="ls"></tool>
<done>summary</done>
```

The **model backend is pluggable** (`AgentBackend` interface in `agent.h`):

```sh
transformer agent                                   # built-in scripted demo (asks y/N per action)
transformer agent --auto                            # auto-approve file reads/writes
transformer agent --cmd "python tools/agent_mock.py"  # external program IS the model
transformer agent --cmd "python tools/agent_mock.py" --goal "make hello.txt"
transformer agent --workspace scratch/ --timeout 30   # tool root + run_command time limit
```

**Safety model** (the model's output and everything it reads are untrusted):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--workspace DIR` | `./workspace` (created) | Every tool path resolves inside DIR, and `run_command` runs with DIR as its cwd. |
| `--auto` | off (ask) | Auto-approve `write_file`/`read_file`. `run_command` **always** asks y/N. Auto-approved writes also refuse `.git/`, a top-level `tools/`, and executable/script extensions (`.py .ps1 .bat .cmd .sh .exe .dll .so` ...). |
| `--ask` | (default) | Accepted for compatibility; asking is the default. |
| `--timeout SEC` | 60 | Wall-clock limit per `run_command` (Windows Job Object / POSIX process group; killed on expiry). |

- Paths: relative only; `..`, `:` (drive-relative `C:x`, NTFS streams `a:b`), control/wildcard
  characters, Windows device names (`CON`, `nul.txt`, `COM1` ...) and components ending in `.`
  or space are refused; the opened file's canonical path (after symlinks/junctions) must be
  inside the canonical workspace, and hard-linked files are refused.
- Output: tool output is drained to EOF and capped (`[truncated N bytes]`), a failing command
  reports `[exit status N]`, stdin is the null device, stderr is captured.
- Injection: model text and observations are stripped of terminal control sequences (C0/C1,
  ESC, bidi overrides) before printing or storing; each observation is framed as
  `OBSERVATION: [[obs:NONCE]]` ... `[[/obs:NONCE]]` with a per-run random nonce, and role-marker
  lines (`MODEL:`, `GOAL:`, `USER:` ...) inside it are prefixed with `> `. A transcript overflow
  stops the run with an error rather than truncating it.
- Command risk (`risk.h`): every `run_command` is classified **SAFE / REVIEW / BLOCK** from the
  command text alone (never the model's explanation of it) and the prompt shows the level and
  reason. SAFE = read-only (`ls`, `dir`, `cat`, `grep`, `git status/log/diff`, `echo` without
  redirection ...); REVIEW = writes, moves, installs, network, interpreters, and any command it
  does not recognize; BLOCK = recursive/forced/wildcard deletes (`rm -rf`, `del /s`, `rd /s`,
  `Remove-Item -Recurse`, `find -delete`, `xargs rm`), `format`/`mkfs`/`dd of=`, `git reset
  --hard` / `clean -f` / `push --force` / `checkout -- .`, `chmod -R`, download piped into a
  shell, `shutdown`, `reg delete`, and any absolute, `~` or `..`-escaping path. The string is
  tokenized under sh, cmd.exe and PowerShell rules and split on `; & && || |` and newlines;
  wrappers (`sudo`, `env`, `xargs`, `timeout` ...), `$(...)`, `sh -c`, `cmd /c`,
  `powershell -Command`, `eval` and `iex` are followed recursively; the highest level wins.
  BLOCK needs the full word `yes` typed; with `--auto` it is refused without asking.
  **Defence in depth, not a sandbox:** a string classifier can never be complete (a script
  written now and run later, aliases, interpreters, encodings ...). `transformer risk "<cmd>"`
  prints the verdict for one command; `--stdin` classifies one command per line (used by
  `tools/risk_filter.py` to keep BLOCK commands out of the routine training pools). Selftest
  part [42] checks a 300-command table.

Breaking change (Sep 2026): the default used to be auto-approve with tool paths relative to the
current directory (the repo root). Pass `--auto` for unattended runs and `--workspace .` only if
you really want the agent loose in the current directory. Selftest part [36] covers all of this.

With `--cmd`, the agent writes the running transcript to a temp file, passes its path to the
program, and uses the program's **stdout** as the model's next response — so you can plug in a
real LLM with a small script (see `tools/agent_mock.py`, which shows the Anthropic/OpenAI call
to drop in). The pieces: `toolcall.c` (parse) · `toolexec.c` (write/read/run) · `approval.c`
(gate) · `agent.c` (loop + scripted/command backends).

**The agent now runs on this project's own engine** — no API, no network. LoRA-fine-tune a real
model to emit tool calls, then point `--cmd` at it via `tools/agent_local.py`:

```sh
# 1. LoRA-SFT SmolLM2-135M on the tool-call data (~5 min, base frozen)
transformer llama-train data/sft.txt --sft --resume model/smollm/model.bin \
  --hf-tok model/smollm/tok.hf --lora-r 16 --lora-alpha 32 \
  --steps 400 --lr 2e-3 --clip 1.0 --warmup 20 --min-lr 2e-4 --out smol_sft.bin

# 2. drive the agent with it
transformer agent --cmd "python tools/agent_local.py smol_sft.bin model/smollm/tok.hf" \
  --goal "write hello to notes.txt" --auto
#   tool requested: write_file 'notes.txt' (5 bytes)
#   [auto-approved] write_file 'notes.txt' (5 bytes)
#   observation: wrote 5 bytes to notes.txt
```

Measured after 400 LoRA steps (loss 8.65 → 0.10), generating at `temp 0.1`, every instruction
routes to its exact tool call and stops cleanly at EOS:

| prompt | output |
| --- | --- |
| `read notes.txt` | `<tool name="read_file" path="notes.txt"></tool>` |
| `list the files here` | `<tool name="run_command" command="ls"></tool>` |
| `show the git status` | `<tool name="run_command" command="git status"></tool>` |
| `the task is finished` | `<done>Task complete.</done>` |
| `write hello to notes.txt` | `<tool name="write_file" path="notes.txt">hello</tool>` |

> Generation is near-greedy on purpose (`hfgen ... 0.1`); at the default `temp 0.8` the model still
> emits the right call but often rambles past the EOS token. `run_command` is **never**
> auto-approved regardless of approval mode, so that row always requires an interactive `y`.

**Bigger data → generalization, not memorization.** Twelve hand-written examples make the model
*memorize* pairs. `tools/make_sft.py` templates many phrasings of each intent over a pool of
filenames (drawn from crawled Wikipedia titles via `tools/crawl_wikipedia.py`, or a built-in list)
**and real shell commands** (mined from Stack Overflow answers via `tools/crawl_stackoverflow.py`,
`data/so_commands.txt`), so the same intent recurs with different words and arguments and the model
must learn the *mapping*. `data/sft_agent.txt` is a committed 250-example sample.

```sh
# mine real git/npm/docker/... commands from Stack Overflow answers (public API, gzip + backoff)
python tools/crawl_stackoverflow.py git bash docker npm python --pages 1 --commands data/so_commands.txt
python tools/make_sft.py --out data/sft_big.txt --n 800 --wiki data/wiki --commands data/so_commands.txt
```

```sh
python tools/make_sft.py --out data/sft_big.txt --n 600 --wiki data/wiki
transformer llama-train data/sft_big.txt --sft --resume model/smollm/model.bin \
  --hf-tok model/smollm/tok.hf --lora-r 16 --lora-alpha 32 --batch 4 \
  --steps 300 --lr 3e-3 --clip 1.0 --warmup 15 --min-lr 3e-4 --out smol_agent.bin
```

Measured, holding out examples the model never trained on, and then testing wholly **invented**
filenames (`quantum_widget.md`, `snarfblat.json`, `xyzzy.txt` — absent from training in any form):

| model | exact match, unseen prompts | novel filenames |
| --- | --- | --- |
| 12 hand-written examples | memorizes | **62%** — right tool, wrong argument (`blat.json` for `snarfblat.json`) |
| 600 templated examples | **100%** on 60 held-out | **100%** — copies arbitrary filenames + content |
| 720 templated + real commands | **98%** on 80 held-out, 100% right-tool | **80%** — runs unseen `git rebase -i HEAD~3`, `kubectl get pods`, ... |

The small model still doesn't *plan* multi-step tasks — it routes one instruction to one tool call
— but it now does so for filenames and commands it has never seen. The generalization is bounded by
the training vocabulary: asked to run `cargo build` (a CLI absent from the crawled `git/bash/docker/
npm/python` tags), it substitutes the familiar `docker build` — it copies commands from the
distribution it saw, it does not know CLIs it never met.

**Mixed data fixes catastrophic narrowing.** Training on tool calls *alone* narrows the model so
hard it answers **every** prompt with a tool call — `why is the sky blue` →
`<tool name="run_command" command="peter_blue">`. `tools/make_qa.py` turns each crawled Wikipedia
article into a `what is <title>? → definition` pair, and `tools/mix_sft.py` interleaves those with
the tool-call data. A model trained on the 50/50 mix learns the **boundary**:

```sh
python tools/make_qa.py --wiki data/wiki --out data/qa.txt --n 400
python tools/mix_sft.py --out data/mixed.txt data/toolcalls.txt data/qa.txt
transformer llama-train data/mixed.txt --sft --resume model/smollm/model.bin \
  --hf-tok model/smollm/tok.hf --lora-r 16 --lora-alpha 32 --batch 4 --steps 400 --out smol_mixed.bin
```

| model | held-out tool routing | general question → prose |
| --- | --- | --- |
| tool-call only | 100% | **0%** — forces a tool call every time |
| 50/50 tool + Q&A mix | **100%** (kept) | **100%** — `why is the sky blue` → "The sky is blue because…" |

So the mix keeps tool-routing intact *and* restores prose answers. The answers' factual accuracy is
limited by the 135M base (`why is the sky blue → "…a transparent liquid"` is wrong), but the model
now correctly *distinguishes an instruction from a question* — the capability the tool-only fine-tune
destroyed. A real assistant is built exactly this way: instruction-following data mixed with general
data so one model does both.

### `transformer imggen` / `audgen` (generate media to a file)

Discrete-token media generation (Phase G): quantize the input to tokens, train a small GPT to
model it, then autoregressively decode to a real file. Two modes, auto-selected from the input:

```sh
# Single-file mode -- reconstruct one input (overfits it):
python tools/img_to_ppm.py cat.png cat.ppm 32     # any image -> small PPM (<= ~32x32)
transformer imggen cat.ppm out.ppm 4 500          # -> out.ppm   (levels=4, steps=500)
python tools/audio_to_wav.py clip.mp3 clip.wav    # any audio -> 16-bit mono WAV (short)
transformer audgen clip.wav out.wav 500           # -> out.wav

# Dataset mode -- learn from a FOLDER and GENERATE a novel sample:
#   name the files img_0000.ppm, img_0001.ppm, ...  (all the same size)
transformer imggen imgset/ out.ppm 4 800          # -> a generated image
#   name the files aud_0000.wav, aud_0001.wav, ...  (all the same length)
transformer audgen audset/ out.wav 800            # -> a generated clip
```

If the input path is a single PPM/WAV it is reconstructed (overfit); if it is a folder of
`img_0000.ppm,…` / `aud_0000.wav,…` the model trains on the whole set and **generates** a new
sample. Dataset mode prepends a **BOS** token so the model learns the distribution of first
tokens and can sample unconditionally (temperature 0.9 / top-k 40 / top-p 0.95).

Image uses a **color-cube** codec (`levels³` tokens); audio uses **µ-law** (256). O(n²)
attention, so keep inputs small (≤ ~32×32 image, ≤ ~1024 audio samples). Text-conditioned
generation (text→image/audio/video) and captioning (media→text) are in the default smoke test —
see `notes.md` (Phase G). Real quality needs a trained VQ-VAE/diffusion + far more data.

### `tools/convert_to_gptc.py` (weight importer)

Converts a `.npz` of named weight arrays into the C checkpoint format so `gpt_load()`
can read it. This is the bridge for importing external weights (e.g. a converted GPT-2).

```sh
python tools/convert_to_gptc.py <weights.npz> <model.bin> --n-heads <N>
```

| Argument       | Meaning                                                       |
| -------------- | ------------------------------------------------------------- |
| `weights.npz`  | input archive; array names/shapes must match the model params |
| `model.bin`    | output GPTC checkpoint                                         |
| `--n-heads N`  | number of attention heads (can't be inferred from shapes)     |

Expected `.npz` keys: `wte`, `wpe`, `blocks.{l}.{W_q,W_k,W_v,W_o,gamma1,beta1,gamma2,beta2,ff_W1,ff_b1,ff_W2,ff_b2}`, `ln_f_gamma`, `ln_f_beta`, `lm_head` (see the script's docstring for shapes). Load the result in C with `gpt_load("model.bin", &m)`.

### Web testing UI — `tools/webchat/` (Go + WebSocket)

A browser frontend for exercising every modality: text chat across all detected models (a
switcher), image/audio generation, and image/audio/video "understanding" (teach-then-reproduce
captioning, via the new `transformer caption` subcommand). Pure Go standard library; it bridges to
the CLI as a subprocess.

```sh
make                                       # build the engine first
cd tools/webchat && go run . -root ../..   # open http://localhost:8080
```

Needs Go 1.23+; Python (`tiktoken`/`tokenizers`) only for the real GPT-2/SmolLM2/Qwen models,
ffmpeg only for the video panel. Quality is toy-scale by design — it proves the pipelines run on
real media end-to-end. See `tools/webchat/README.md`.

---

## Programmatic guide (using the C API)

### Tokenizer

```c
#include "tokenizer.h"

Tokenizer tok = tokenizer_create_bytelevel();   // 256 byte tokens
int ll = tokenizer_add_merge(&tok, 'l', 'l');   // learn merges -> new token ids
tokenizer_add_merge(&tok, 'e', ll);

int ids[64];
int n = tokenizer_encode(&tok, "hello", ids, 64);

char out[128];
tokenizer_decode(&tok, ids, n, out, sizeof(out)); // "hello" (byte-exact round-trip)

tokenizer_free(&tok);
```

### Model: build, run, generate

```c
#include "model.h"

// vocab, d_model, n_heads, n_layers, d_ff, max_seq_len
GPT m = gpt_create(16, 8, 2, 2, 32, 16);

// Forward pass -> logits (seq_len, vocab_size)
int ids[3] = {1, 5, 9};
int shape[2] = {3, m.vocab_size};
Tensor logits = tensor_create(shape, 2);
gpt_forward(&m, ids, 3, &logits);

// Autoregressive greedy generation (appends in place; buffer must fit prompt+new)
int seq[16] = {1, 5, 9};
gpt_generate(&m, seq, /*prompt_len=*/3, /*n_new=*/5);

tensor_free(&logits);
gpt_free(&m);
```

### Checkpoints

```c
#include "checkpoint.h"

gpt_save("model.bin", &m);        // persist weights

GPT loaded;
gpt_load("model.bin", &loaded);   // rebuilds config + weights; free with gpt_free
```

### Training (backprop + Adam)

Every forward op has a matching `*_backward` that **accumulates** into the `grad` array of
its inputs/parameters (call `tensor_zero_grad` before each step). Cross-entropy is in
`loss.c`, the optimizer in `optim.c`. A training step looks like:

```c
#include "loss.h"
#include "optim.h"

Tensor *params[] = {&W1, &b1, &W2, &b2};
Adam opt = adam_create(params, 4, /*lr=*/0.05f);

for (int step = 0; step < n_steps; step++) {
  // forward: ... -> logits
  float loss = cross_entropy_forward(&logits, targets, &probs);

  // backward: zero grads, then propagate from the loss back to every parameter
  cross_entropy_backward(&probs, targets, &logits);
  // ... matmul_backward / bias_add_backward / gelu_backward / layernorm_backward ...

  adam_step(&opt, params, 4);   // updates params using their .grad
}
adam_free(&opt);
```

The **full `GPT` trains end-to-end**: `gpt_forward_train` (in `model.c`) caches activations,
`gpt_backward` propagates through the output head, final LayerNorm, every stacked block
(multi-head causal attention + FFN + residuals), and the embeddings, and `gpt_collect_params`
feeds Adam. A training step:

```c
#include "model.h"
Tensor *params[64];
int n = gpt_collect_params(&m, params);
Adam opt = adam_create(params, n, 0.02f);
GPTActs acts = gpt_acts_create(&m, seq_len);

gpt_forward_train(&m, ids, seq_len, &acts, &logits);
float loss = cross_entropy_forward(&logits, targets, &probs);
tensor_zero_grad(&logits);
cross_entropy_backward(&probs, targets, &logits);
gpt_backward(&m, ids, seq_len, &acts, &logits);   // grads into every parameter
adam_step(&opt, params, n);
```

Gradients are verified in `selftest.c` by **end-to-end numerical gradient checking** (analytic
`grad` vs `(L(+ε) − L(−ε)) / 2ε`) across the whole model, then it trains the GPT on a
next-token task and overfits it to ~0 loss. (Note: float32 finite-difference checking has a
noise floor for tiny gradients, so the check trusts well-conditioned entries; see `notes.md`.)

### Importing external weights (e.g. GPT-2) — end-to-end

The model is **GPT-2-architecture-compatible**: pre-norm blocks, multi-head causal attention
**with biases** (`b_q/b_k/b_v/b_o`, matching GPT-2's `c_attn`/`c_proj`), GELU FFN, learned
positional embeddings, and a final LayerNorm. The output head is a separate `lm_head`, so
GPT-2's tied head is written as `lm_head = wte^T` at conversion time.

**Real GPT-2 (124M) — verified, generates real text:**
```sh
pip install numpy safetensors huggingface_hub tiktoken
python tools/gpt2_setup.py --out model/gpt2/model.bin      # download + convert (~622 MB), once

python tools/gpt2_chat.py "The meaning of life is"         # one command: tokenize -> C -> text
```
(`gpt2_chat.py` wraps tokenize → `transformer generate --name gpt2` → detokenize. The manual
3-step form — `gpt2_tok.py encode` → `generate` → `gpt2_tok.py decode` — still works too.)
Actual output from that prompt: *"The meaning of life is not to be confused with the fact of
being reborn. Rather, life is to be born and that is what the term means."* GPT-2 uses its own
BPE (via `tiktoken`), so it runs in token-ID mode. See `model/gpt2/README.md`.

**Your own weights:** assemble a `.npz` with the documented key names (incl. `b_q…b_o`) and
run `convert_to_gptc.py` directly. The GPTC format is **v2** (adds the attention biases).

> Not yet wired: GPT-2's real BPE tokenizer (`encoder.json` / `vocab.bpe`) — the byte-level
> BPE here is self-contained. That plus running the full 124M weights is the next milestone
> in `notes.md`.

---

## Coding rules (for contributors)

1. **Indentation is 2 spaces.** Never use tabs.
2. **The source root is C-only, and flat.** All `.c`/`.h` files live directly in the
   project root with no subfolders — do not mix other languages into the root. A pull
   request that adds a code subfolder or non-C source to the root will be rejected.
3. **Non-C helper / third-party scripts go in `tools/`.** Anything in another language
   (Python, Go, shell, …) — converters, downloaders, experiments — belongs under
   `tools/`, never in the root next to the C sources.
4. **One concern per module.** Declare the public API in the `.h`, define it in the
   matching `.c`, and give every header an include guard (`#ifndef MODULE_H`).
5. Keep helpers that are used by only one file `static` inside that `.c`.
6. **Never use untyped `void *` parameters** in a function's public API. Pass concrete,
   typed pointers (`Tensor *`, `float *`, …) instead — untyped pointers defeat the
   compiler's type checking and lead to slow, hard-to-predict runtime errors.
7. Code must compile cleanly under `-std=c11 -Wall` (no warnings).
8. When you add a new module, add its `.o` to `OBJS` in the `Makefile` and list its
   header dependencies there.
