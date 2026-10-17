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

## 2026-10-17 23:30 — first complete version
- All four modules done and committed; `build/llm.exe` (58 KB, ~11k lines of asm) works end to end.
- **Identical output to the C engine**: greedy and sampled (temp 0.8, seed 7) token streams match
  `transformer hfgen` (current-source C build) on qwen int8, smollm int8 and smollm fp32 (model_v2.bin);
  logits bit-identical (dec_test). CLI: 21/21 byte-identical cases.
- Speed (424-token prompt + 128 tokens, best of 3): Qwen int8 24.3 s -> 3.4 s (7.1x),
  SmolLM int8 10.5 -> 1.37 s (7.6x), SmolLM fp32 21.3 -> 3.9 s (5.4x).
  Qwen decode 49 tok/s, prefill ~990 tok/s, load 0.33 s.
- Fixed: thread-pool hang (phantom chunk race) — stress-tested 4M jobs.
- Notes: repo-root transformer.exe is stale (rejects v2 files) — rebuild it before comparing.
  model/smollm/model.bin is v1 (rejected by both engines).

### Next ideas (morning sessions)
1. Long-context attention: split by key range as well as by head (SmolLM has only 9 heads →
   7 of 16 threads idle); at ctx 4096 attention costs 5-10 ms/token.
2. Decode at 49 tok/s vs ~60 tok/s matvec-only ceiling for Qwen: profile the gap (per-op
   pool dispatch, fusing Wq/Wk/Wv and gate/up into one pool_run each, sampler 0.5 ms).
3. Optional non-exact speed mode: fp16 KV cache, int8 lm_head shortcut for greedy (top-1 only).
4. Port `--samples`, `--stop-tags`, `--constrain`, `hfchat`; GPT-2 (GPTC) models.
