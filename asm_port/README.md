# asm_port — the LLM engine in pure x86-64 assembly

A from-scratch port of the engine's LLaMA-family inference path (`transformer hfgen`) to
**pure assembly**: ~11,000 lines of hand-written x86-64, no C, no C runtime — the whole
program is a **58 KB** `llm.exe` that talks to Windows only through `kernel32`/`shell32`.

It produces **exactly the same tokens** as the C engine (bit-identical logits, same sampler and
RNG), and runs **5–7x faster end to end**.

## Build and run

From mmc-shell (or git-bash), with zig at `/d/env/zig/zig.exe` (override with `ZIG=`):

```sh
./asm_port/build.sh                       # -> asm_port/build/llm.exe

printf 'The capital of France is' | \
  ./asm_port/build/llm.exe model/qwen/model_int8.bin model/qwen/tok.hf 64 0 1
```

`llm.exe <model.bin> <tok.hf> [n_new] [temp] [seed] [--raw] [--bench] [--threads N] [--q8]`

- Drop-in for `transformer hfgen`: same arguments, same `{"d":"..."}` / `{"done":true}` output,
  same prompt fitting, stop rules and error messages (so `ai.py` can call it unchanged).
- `--raw` plain text instead of JSON lines · `--bench` timings on stderr ·
  `--threads N` pool size · `--q8` quantize an fp32 (LLMA) model to int8 at load (faster, not bit-exact vs C).
- Models: LLMA v2–v4 (fp32; QK-norm, tied head, sliding windows) and LLMQ (int8).
  `model/smollm/model.bin` is a v1 file that the current C engine also rejects — use `model_v2.bin`.
  GPT-2 (GPTC) and `--samples/--constrain/--stop-tags` are not ported yet.

## Speed (i5-14400, DDR4-3200; 424-token prompt + 128 generated, greedy, best of 3)

| model | C engine (`transformer hfgen`) | asm `llm.exe` | speedup |
|---|---|---|---|
| Qwen2.5-0.5B int8 | 24.3 s | **3.4 s** | **7.1x** |
| SmolLM2-135M int8 | 10.5 s | **1.37 s** | **7.6x** |
| SmolLM2-135M fp32 | 21.3 s | **3.9 s** | **5.4x** |

`--bench` breakdown for the same runs:

| model | load | prefill | decode |
|---|---|---|---|
| Qwen2.5-0.5B int8 | 0.33 s | 988 tok/s | 49 tok/s |
| SmolLM2-135M int8 | 0.10 s | 1795 tok/s | 133 tok/s |
| SmolLM2-135M fp32 | 0.11 s | 947 tok/s | 40 tok/s |

Decode is memory-bound: Qwen's int8 matvecs stream 494 MB per token at ~30 GB/s, 90–95% of
what a pure streaming read reaches on this machine.

## Where the speed comes from

- **int8 matvec on AVX-VNNI** (`matvec.S`): weights repacked at load into 16-column panels,
  4 consecutive k per 64-byte line, so one `{vex} vpdpbusd` does 32 multiply-adds; activations
  are offset to u8 and the offset is removed with a precomputed `128*colsum`. Integer sums are
  exact, so it is still bit-identical to C's `qmatmul`. AVX2 fallback without VNNI.
- **Right thread count**: decode matvecs run on the 6 P-cores (16 threads is *slower* — memory
  bound); prefill uses all 16.
- **Batched prefill** (`decode.S`): the prompt runs 64 rows at a time through a 6x16
  register-blocked kernel, each weight line reused across rows — ~50x faster than token by token.
- **Low-latency thread pool** (`pool.S`): spin on a generation word, then sleep on an event; one
  `lock cmpxchg` claims a chunk.
- **Fast exact sampler** (`sample.S`): exponent buckets + partial sort instead of sorting 151,936
  probabilities — same token and RNG state as `sampler.c`, ~5x faster.
- **Tokenizer** (`tok.S`): word and pair caches; ~110 MB/s vs ~17 MB/s for `hftok.c`, identical ids.
- **Loader** (`load.S`): big sequential reads straight into the final buffers; Qwen ready in 0.33 s.

## How it stays identical to C

The int8 path re-quantizes activations every layer, so a last-bit float difference grows into a
different token. The port therefore reproduces C's arithmetic exactly: serial FMA chains where C
has them (RMSNorm sum of squares, attention scores — the 8 vector lanes hold 8 different keys),
a vector port of musl's `expf`, musl `sinf`/`cosf`, x87 `powf`, and the same `as*scale*acc`
order in the int8 epilogue.

## Layout

| file | what |
|---|---|
| `main.S` | CLI (`hfgen`-compatible), generation loop, stop rules |
| `load.S` | LLMA/LLMQ checkpoint loader, KV-cache allocation |
| `tok.S`, `tok_unitab.S` | Hugging Face byte-level BPE tokenizer (port of `hftok.c`) |
| `decode.S` | `decode_step`, batched `prefill` |
| `ops.S` | RMSNorm, RoPE, attention, SiLU, expf/sin/cos/pow |
| `sample.S` | temperature / top-k / top-p sampler + xorshift RNG |
| `matvec.S` | int8 (VNNI/AVX2) and fp32 matvec / matmul, weight packing |
| `pool.S`, `cpu.S` | thread pool, CPU feature detection |
| `mem.S`, `io.S` | memory, files, stdin/stdout, number formatting, JSON escaping |
| `model.inc`, `DESIGN.md` | shared struct layouts and the module contract |
| `tests/` | C test harnesses comparing each module with the repo's C code (tests only) |

## Tests

Each test links the asm module with the repo's C code and compares:

```sh
cd asm_port
ASM="tok tok_unitab"     ./build.sh test tok_test ../hftok.c                        # 2.28M encodes, 0 mismatches
ASM="cpu pool matvec"    ./build.sh test kern_test ../quant.c tests/kern_stream.S   # 4928 checks
ASM="mem io load matvec cpu pool" ./build.sh test load_test <C files, see its header>
# decode: see the build line in tests/dec_test.c's header
```
