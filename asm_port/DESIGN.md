# asm_port — design contract (read this before touching any .S file)

Goal: a **pure x86-64 assembly** port of the engine's LLaMA-family inference path
(`transformer hfgen`): load an LLMA (fp32) or LLMQ (int8) checkpoint + a `tok.hf`
tokenizer, read a prompt from stdin, decode with a KV cache, sample, print tokens.
**Speed is the main goal**; correctness is checked against the C engine.

## Toolchain / rules

- Sources: `asm_port/*.S` — GNU assembler, **`.intel_syntax noprefix`**, assembled
  and linked by `zig cc` (`/d/env/zig/zig.exe`, LLVM integrated assembler). The `.S`
  files go through the C preprocessor, so `#include "model.inc"` and `#define` work.
- **No C and no CRT in the product.** Link `-nostdlib`, entry `mainCRTStartup`, only
  `kernel32` (+ `shell32` for `CommandLineToArgvW` if wanted). Memory via
  `VirtualAlloc`/`HeapAlloc`, files via `CreateFileW`/`ReadFile`, threads via
  `CreateThread`, output via `WriteFile`.
- Tests MAY use C (`asm_port/tests/*.c`, built with `zig cc`) to compare the asm
  functions against the repo's own C reference code (e.g. link `../quant.c`,
  `../hftok.c`). Tests are not part of the product.
- Build: `asm_port/build.sh` (bash, run from mmc-shell / git-bash). Output in
  `asm_port/build/` (git-ignored).
- Target CPU: Intel i5-14400 (Raptor Lake: AVX2, FMA, **AVX-VNNI** (VEX
  `{vex} vpdpbusd`), no AVX-512; 6P+4E cores / 16 threads). Detect with CPUID at
  startup (`cpu_init`) and fall back to an AVX2 (no VNNI) path; SSE-only machines
  may simply be rejected with a clear message.

## Calling convention (all exported functions)

Microsoft x64: args `rcx, rdx, r8, r9`, then stack (`[rsp+40]` at entry...);
float args in `xmm0..xmm3` **by position** (arg 2 float → xmm1);
return `rax` / `xmm0`. Callee-saved: `rbx rbp rdi rsi r12-r15 xmm6-xmm15`.
Caller reserves 32 bytes shadow space, rsp 16-byte aligned at the call.
**Save/restore xmm6-15 if you use ymm6-15** (upper halves are volatile, lower are not).
Run `vzeroupper` before returning from any function that touched ymm and before
calling Win32 APIs. Ints in this doc are **int64** unless marked (i32).

## Shared data structures (`model.inc` holds the offsets — use the names, never raw numbers)

### Mat (64 bytes) — any weight matrix used as `y = x @ W`, x: K floats, y: N floats
| off | field | meaning |
|---|---|---|
| 0 | kind | 0 = F32 row-major (K,N) `W[k*N+j]`; 1 = F32T row-major (N,K) (tied head: y[j]=dot(W[j,:],x)); 2 = Q8 |
| 8 | K | input dim (rows) |
| 16 | N | output dim (cols) |
| 24 | data | F32/F32T: float*; Q8: int8* raw row-major (K,N) as in the file (kept for embedding lookup, may be freed after packing for non-embedding mats) |
| 32 | scale | Q8: float[N] per-column scale |
| 40 | packed | Q8: kernel-specific packed int8 layout (built by `mat_q8_pack`) |
| 48 | colsum | Q8: int32[N] (or whatever the kernel needs for the u8/s8 VNNI offset fix) |
| 56 | aux | spare for the kernel owner |

### Layer (512 bytes)
`0 window` (int64, <=0 = full), `8 attn_norm`, `16 ffn_norm`, `24 b_q`, `32 b_k`,
`40 b_v`, `48 q_norm`, `56 k_norm` (float* each; q_norm/k_norm only if qk_norm),
then Mats: `64 Wq, 128 Wk, 192 Wv, 256 Wo, 320 Wgate, 384 Wup, 448 Wdown`.

### Model (256 bytes)
`0 vocab, 8 d_model, 16 n_layers, 24 n_heads, 32 n_kv, 40 head_dim, 48 d_ff, 56 max_seq`
(int64), `64 rope_base` (f32), `68 eps` (f32), `72 qk_norm`, `80 quant` (1 = matrices
are Q8), `88 tied`, `96 layers` (Layer*), `104 final_norm` (float*), `112 wte` (Mat,
64 B: K=vocab rows, N=d_model; F32 or Q8 raw — embedding of token t is row t,
Q8 dequant = q[t*N+j]*scale[j]), `176 lm_head` (Mat, 64 B; F32T aliasing wte.data
when tied), `240 file_base` (mapping/buffer), `248 spare`.

### KV (64 bytes)
`0 n_layers, 8 max_seq, 16 kv_dim, 24 len, 32 cap` (int64* per layer, ring slots =
min(window,max_seq) or max_seq), `40 k` (float** per layer, cap*kv_dim),
`48 v` (float**), `56 spare`.

## Module ownership and exported API

| file | owner | exports |
|---|---|---|
| `cpu.S`, `pool.S` | kernels | `cpu_init()`→rax feature bits (1=AVX2+FMA, 2=VNNI); `pool_init(n)` (n<=0 = auto); `pool_run(fn, ctx, n, grain)` runs `fn(ctx, i0, i1)` (Win64) over [0,n) split across threads, returns when all done; `pool_threads()` |
| `matvec.S` | kernels | `mat_q8_pack(Mat*)`; `mat_q8_from_f32(Mat* out, const float* w, K, N)` (= `quantize_int8_percol` exactly, then pack); `matvec(const Mat*, const float* x, float* y)` dispatch on kind, threaded; `matmul_rows(const Mat*, const float* X, M, float* Y)` (M rows, for batched prefill; row i must equal `matvec` of row i bit-for-bit) |
| `mem.S`, `io.S`, `load.S`, `main.S` | loader | `mem_alloc(bytes)` (64-B aligned, zeroed), `mem_free`; `io_write(handle,buf,len)`, `io_stdout()`, `io_stderr()`, `io_read_stdin(&len)`→buf; `fmt_*` helpers; `model_load(const wchar_t* path, Model* m, i64 flags)` (flag 1 = quantize an fp32 LLMA to Q8 at load); `kv_create(const Model*, KV*)`; `main.S` = CLI |
| `tok.S` | tokenizer | `tok_load(const wchar_t* path)`→Tok* (0 on error); `tok_encode(Tok*, const u8* text, len, i32* ids, cap)`→n (special-token aware, == `hftok_encode`); `tok_piece(Tok*, id, u8* out)`→len (bytes of one token, == `hftok_decode` of one id); `tok_is_eog(Tok*, id)`; `tok_vocab(Tok*)` |
| `ops.S`, `decode.S`, `sample.S` | decode | `rmsnorm(x,g,out,n,eps)`, `rope_pos(v, n_heads, head_dim, base, pos)`, attention, `decode_step(Model*, KV*, token, float* logits /*may be 0*/)`, `prefill(Model*, KV*, ids, n)` (batched, via `matmul_rows`), sampler (`rng_seed`, `sample_token(logits, V, temp, top_k, top_p, rng*)` == `sample_token` in sampler.c incl. its xorshift RNG) |

The int8 path must reproduce the C engine's `qmatmul` arithmetic: per-row activation
quantization (`as = max|x|/127` or 1, `v = lrintf(x*(1/as))` clamped ±127), exact
int32 dot products, then `y[j] = as*scale[j]*(float)acc[j]`. Integer sums are exact
in any order, so **Q8 matvec is bit-exact with C**. Everything else is checked with a
tolerance (and greedy decode must produce the same token stream as `transformer.exe hfgen`).

## Rules for agents working in parallel

- Touch only the files you own (plus your own `tests/<area>_*` files). Do not edit
  `model.inc`, `DESIGN.md` or `build.sh` — if you need a change, say so in your report.
- Do not commit; the coordinator commits.
- Every exported function: correct Win64 prologue, stack alignment, saved non-volatiles.
