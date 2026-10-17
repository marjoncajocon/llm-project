# asm_port progress log

Toolchain: zig cc (D:\env\zig, from mmc-shell) assembling GNU-as Intel-syntax `.S`; no CRT, kernel32 only.
Branch: `asm-port`. Contract: `DESIGN.md` + `model.inc`. Build: `./build.sh`.

## 2026-10-17 22:30 (Manila) — kickoff (started the night before, at the user's request)
- Skeleton committed: DESIGN.md, model.inc, build.sh.
- Baseline (C engine): `transformer.exe hfgen model/qwen/model_int8.bin model/qwen/tok.hf 64 0 1`
  ~7.4 s wall for load + prefill + 64 tokens (to be measured precisely by the bench).
- Four parallel agents launched:
  - kernels: cpu.S, pool.S, matvec.S (AVX-VNNI int8 matvec, bit-exact vs C qmatmul; threadpool)
  - tokenizer: tok.S (exact port of hftok.c)
  - loader/io/cli: mem.S, io.S, load.S, main.S (LLMA/LLMQ loader, hfgen-compatible CLI)
  - decode: ops.S, decode.S, sample.S (rmsnorm/rope/attention/silu, decode_step, batched prefill, sampler)
- Next: integrate, end-to-end greedy test vs transformer.exe, benchmarks, README.
