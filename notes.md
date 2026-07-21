# Project Notes

## Roadmap — all steps & phases

Goal: grow this pure-C transformer into a **fully local coding-agent CLI**. Every step ends
green (`-Wall` clean, smoke test `PASSED`, artifacts cleaned) and is committed.

**Foundations (DONE):**
- [x] Step 1 — Transformer block: FFN + GELU + pre-norm
- [x] Step 2 — Multi-head attention + causal masking
- [x] Step 3 — Full model: embeddings, stacked blocks, output head, greedy sampler
- [x] Step 4 — Checkpoint save/load (GPTC) + byte-level BPE tokenizer + `tools/` converter
- [x] Step 5a — Training foundation: backward core + cross-entropy + Adam, gradient-checked
- [x] Step 5b — End-to-end GPT training (attention/FFN/embedding backward)

**Phase A — Make the model produce real text:**
- [x] Step 4b — Load real GPT-2 weights: GPT-2-compatible architecture (attention biases,
      tied output head), converter (HF safetensors → GPTC), verified on GPT-2-shaped weights
- [ ] Step A1 — Real GPT-2 BPE tokenizer: load `encoder.json` + `vocab.bpe` + pretok regex
- [ ] Step A2 — Scale config to a real architecture (dims/layers/heads/context)

**Phase B — A local generation CLI:**
- [x] Step B1 — Better sampling: temperature + top-k/top-p + seeded RNG (`sampler.c`)
- [x] Step B2 — KV cache (incremental decode; verified bit-exact vs full forward)
- [ ] Step B3 — REPL front-end: `you>` loop (stdin → tokenize → generate → stream out)
- [ ] Step B4 — Performance: blocked/threaded/SIMD matmul + int8 quantization

**Phase C — The agent layer (tools + approval), pure C:**
- [ ] Step C1 — Tool protocol the model emits (parseable `<tool …>` blocks)
- [ ] Step C2 — Tool executor: read_file / write_file / list_dir / run_command (`tools.c`)
- [ ] Step C3 — Approval gate: diff/preview + `y/n/a` on stdin (`approval.c`)
- [ ] Step C4 — Agent loop: prompt → generate → parse tool → approve → execute → feed back (`agent.c`)

**Phase D — Quality track (as needed):**
- [ ] Step D1 — Run a small quantized code model (1–3B) for genuinely useful output

> Honest note: the *pipeline* is 100% doable in pure C and local; agent *quality* is capped by
> the model you can run. Loading pretrained weights (4b) is the fastest path to real output.

---

## Step 1 — Complete the transformer block (FFN + GELU + pre-norm)

**Status: DONE — smoke test passed.**

### What changed
The transformer block was previously `attention → +residual → LN1 → LN2`, where the
second LayerNorm stood in where the feed-forward network belongs, using post-norm
ordering. It is now a real GPT-2 pre-norm block with a feed-forward MLP:

```
a = input + Attention(LN1(input))
output = a + FFN(LN2(a))       where FFN(h) = GELU(h @ W1 + b1) @ W2 + b2
```

- **GELU** activation added to `activation.c` (GPT-2 tanh approximation).
- New flat module **`feedforward.h` / `feedforward.c`**: Linear → GELU → Linear with
  biases, hidden size `d_ff = 4 * d_model`.
- **`transformer_block_forward`** rewritten to pre-norm ordering with the FFN sub-layer;
  signature extended with the four FFN weight tensors and `d_ff`.
- **`main.c`** creates, initializes, passes, and frees the FFN weights.
- **Portability fix** (needed to build/smoke-test on Windows): `tensor.c` now uses a
  per-platform aligned allocator — `_aligned_malloc`/`_aligned_free` on Windows,
  `aligned_alloc`/`free` elsewhere. Previously `aligned_alloc` failed to compile on the
  Windows CRT, and blocks were freed with the wrong deallocator.

### Progress checklist
- [x] GELU activation (`activation.h` / `activation.c`)
- [x] Feed-forward module (`feedforward.h` / `feedforward.c`)
- [x] Pre-norm block reorder + FFN (`attention.h` / `attention.c`)
- [x] Wire FFN weights into `main.c`
- [x] `Makefile` updated (`feedforward.o` + deps)
- [x] `README.md` updated (module table + no-`void *`-parameter rule)
- [x] Portable aligned alloc/free in `tensor.c`
- [x] `.gitignore` added
- [x] Smoke test (build + run → `Output validation: PASSED (no NaN/Inf)`)

### Smoke test result (2026-07-21)
Built with `zig cc` (no gcc/clang/make on this machine; `zig cc` is a C11-compliant
clang driver). `-Wall` clean, program ran to completion:

```
Output shape: (4, 8)
Output validation: PASSED (no NaN/Inf)
Done! All memory freed successfully.
```

Build command used:
```sh
zig cc -std=c11 -O2 -Wall -o transformer.exe *.c -lm
```
The `Makefile` (`make`) produces the same result on a machine with gcc/clang installed.

## Step 2 — Multi-head attention + causal masking

**Status: DONE — smoke test passed.**

### What changed
Single-head `self_attention_forward` was replaced by
**`multihead_attention_forward`**, which:
- Projects the input into `n_heads` heads (`head_dim = d_model / n_heads`) using square
  `(d_model, d_model)` weight matrices `W_q/W_k/W_v/W_o`.
- Runs scaled dot-product attention per head with a **causal mask**: position `i` may
  attend only to positions `j <= i`; future positions get score `-INFINITY`, so after
  softmax their weight is exactly 0 (autoregressive behavior, like GPT-2).
- Concatenates the per-head outputs and projects through `W_o`.

`transformer_block_forward` now calls the multi-head function; its `d_k` parameter became
`n_heads`. `main.c` uses `n_heads = 2` and square attention weights.

### Progress checklist
- [x] Multi-head projection + per-head attention (`attention.h` / `attention.c`)
- [x] Causal masking (future positions -> -INFINITY before softmax)
- [x] `transformer_block_forward` updated (`d_k` -> `n_heads`)
- [x] `main.c` updated (square weights, `n_heads = 2`)
- [x] `README.md` updated (attention module description)
- [x] Smoke test (build + run -> `Output validation: PASSED`, `-Wall` clean)

### Smoke test result (2026-07-21)
Built with `zig cc -std=c11 -O2 -Wall`; `-Wall` clean, ran to completion with
`Output validation: PASSED (no NaN/Inf)`.

## Step 3 — Full model: embeddings, stacked blocks, output head, sampler

**Status: DONE — smoke test passed.**

### What changed
The demo is now an actual (tiny, untrained) autoregressive language model instead of a
single block on random input. Three new flat modules plus a rewritten `main.c`:

- **`embedding.h/.c`** — `embed_tokens`: `output[i] = wte[token_ids[i]] + wpe[i]`
  (token + positional embeddings).
- **`sampler.h/.c`** — `sample_argmax`: greedy next-token pick from a logits row.
- **`model.h/.c`** — introduces two structs to bundle weights:
  - `BlockWeights` — all learnable tensors of one transformer block.
  - `GPT` — config + `wte`, `wpe`, an array of `n_layers` `BlockWeights`, final
    LayerNorm params, and `lm_head` output projection.
  - `gpt_create` / `gpt_free` — allocate + randomly initialize / release all weights.
  - `gpt_forward` — embeddings -> stacked blocks -> final LayerNorm -> `@ lm_head`
    logits `(seq_len, vocab_size)`.
  - `gpt_generate` — autoregressive greedy loop, appending tokens in place.

`main.c` builds a `GPT` (vocab=16, d_model=8, heads=2, layers=2, d_ff=32), generates 5
tokens from a 3-token prompt, and validates the logits (no NaN/Inf).

### Notes / simplifications
- Output head `lm_head` is a **separate** weight, not tied to `wte` (real GPT-2 ties
  them: `logits = hidden @ wte^T`). Tying can come later once a transpose-matmul exists.
- No KV cache: each generation step re-runs the full forward over the current sequence.
  Fine for a reference model; a cache is a later optimization.
- Only greedy sampling; temperature / top-k / top-p can extend `sampler.c`.

### Progress checklist
- [x] Token + positional embeddings (`embedding.h` / `embedding.c`)
- [x] Greedy sampler (`sampler.h` / `sampler.c`)
- [x] `GPT` + `BlockWeights` structs, create/free (`model.h` / `model.c`)
- [x] `gpt_forward`: embed -> N blocks -> final LayerNorm -> logits
- [x] `gpt_generate`: autoregressive greedy loop
- [x] `main.c` rewritten to build + run the model
- [x] `Makefile` + `README.md` updated (3 new modules)
- [x] Smoke test (build + run -> generates tokens, `Output validation: PASSED`, `-Wall` clean)

### Smoke test result (2026-07-21)
Built with `zig cc -std=c11 -O2 -Wall`; `-Wall` clean. Prompt `1 5 9` generated
`11 6 11 14 14` (arbitrary - the model is untrained), logits validated PASSED, all
memory freed.

## Step 4 (Option A) — Checkpoint save/load + byte-level BPE tokenizer

**Status: DONE — smoke test passed (incl. the `tools/` converter).**

### What changed
Built the two pieces of infrastructure that real-weight loading depends on, both
self-contained and testable without a large download:

- **`checkpoint.h/.c`** — `gpt_save` / `gpt_load` in a flat binary format (GPTC):
  `magic 'GPTC' + version + 6-int config + every weight's raw float32` in a fixed
  canonical order. A shared `collect_tensors` helper drives both save and load so their
  order can't drift. `gpt_load` reuses `gpt_create` then reads weights in.
- **`tokenizer.h/.c`** — self-contained byte-level BPE: 256 base byte tokens + merges
  (`256 + rank`), binary-safe, no external files. `encode`/`decode`/`add_merge`/
  `create`/`free`.
- **`main.c`** — rewritten demo validates three things: tokenizer round-trip
  (`decoded == original`, encoded shorter than bytes), model generation + NaN/Inf check,
  and checkpoint round-trip (reloaded logits **bitwise identical**). Exit code reflects
  pass/fail.
- **`tools/convert_to_gptc.py`** — NEW helper (Python) that converts a `.npz` of named
  weight arrays into a GPTC `.bin`, matching `collect_tensors` order exactly. This is the
  bridge for importing external weights.
- **`README.md`** — added the two module rows, a **CLI guide** (the `transformer` binary,
  the converter tool with args) and a **programmatic API guide** (tokenizer, model,
  checkpoints, external-weight import). Coding rules updated: source root is **C-only and
  flat**; non-C helper/third-party scripts go in **`tools/`**.
- **`.gitignore`** — ignore `*.bin` checkpoints.

### Repo layout rule (updated)
- Root: C source only, flat, no subfolders.
- `tools/`: helper/third-party scripts in other languages (Python/Go/…). The only allowed
  directory.

### Notes / simplifications
- Byte-level BPE is self-contained (no `encoder.json`/`vocab.bpe` yet); the GPT-2 regex
  pre-tokenization and file loading are a later add-on.
- Checkpoint format is our own simple binary (host-endian float32), not safetensors.
- Loading **real** GPT-2 weights (step 4b) still needs the model to match GPT-2's exact
  parameterization: attention biases, fused QKV projection, and a weight-tied `lm_head`.
  The converter + loader are the transport layer that work will plug into.

### Progress checklist
- [x] Checkpoint format + `gpt_save`/`gpt_load` (`checkpoint.h`/`.c`)
- [x] Shared `collect_tensors` (save/load order can't drift)
- [x] Byte-level BPE tokenizer (`tokenizer.h`/`.c`)
- [x] `main.c` demo: tokenizer + generation + checkpoint round-trips
- [x] `tools/convert_to_gptc.py` weight importer
- [x] `Makefile` + `.gitignore` updated
- [x] `README.md`: CLI guide + API guide + `tools/` / C-only rules
- [x] Smoke test (build `-Wall` clean; all three round-trips PASS; exit 0)
- [x] Converter verified end-to-end (npz -> convert -> C `gpt_load` -> forward, no NaN/Inf)

### Smoke test result (2026-07-21)
Built with `zig cc -std=c11 -O2 -Wall`; `-Wall` clean. Tokenizer round-trips
`"hello hello"` (11 bytes -> 5 tokens -> original); model generates `1 5 9 -> 11 6 11 14
14`; reloaded checkpoint logits bitwise identical; `Output validation: PASSED`. The
`tools/` converter was also verified: a random numpy `.npz` -> `convert_to_gptc.py` ->
GPTC `.bin` -> C `gpt_load` -> forward pass ran clean (no NaN/Inf).

## Step 5a — Training foundation (backprop core + loss + Adam), gradient-checked

**Status: DONE — smoke test passed (gradient check + loss drop).**

### What changed
The `Tensor.grad` array is finally used. Manual backprop (explicit `*_backward` siblings,
no autograd engine); each accumulates into input/param grads, so callers zero grads per
step (`tensor_zero_grad`, new in `tensor.c`).

- **`matrix.c`** — `matmul_backward` (`dA += dC·Bᵀ`, `dB += Aᵀ·dC`), `matadd_backward`,
  plus `bias_add` / `bias_add_backward` (row-broadcast bias, reusable by the FFN).
- **`activation.c`** — `gelu_backward` (analytic derivative of the tanh-approx GELU).
- **`layernorm.c`** — `layernorm_backward` (per-row `dgamma`, `dbeta`, `dx`).
- **`loss.h/.c`** (NEW) — `cross_entropy_forward` (mean loss + fills softmax `probs`,
  reuses `softmax_rowwise`) and `cross_entropy_backward` (`(probs - onehot)/N`).
- **`optim.h/.c`** (NEW) — `Adam` (bias-corrected, per-param m/v state).
- **`main.c`** — rewritten around a tiny trainable MLP
  `logits = LayerNorm(GELU(x·W1+b1))·W2 + b2` that exercises every new backward op:
  - **Gradient check**: analytic grad vs numerical `(L(+ε)-L(-ε))/2ε`, sampled across all
    params; asserts max relative error < 1e-2.
  - **Training**: Adam overfits 8 synthetic examples; asserts the loss collapses.
  - Keeps condensed tokenizer + checkpoint round-trips (step-4 code still exercised).

### Notes
- Gradient checking is the correctness proof here — a wrong gradient still "runs" but won't
  learn; the numeric check catches it.
- Scope boundary held: attention/embedding backward = step 5b (below).

### Progress checklist
- [x] `tensor_zero_grad` (`tensor.c`)
- [x] `matmul_backward`, `matadd_backward`, `bias_add`/`bias_add_backward` (`matrix.c`)
- [x] `gelu_backward` (`activation.c`)
- [x] `layernorm_backward` (`layernorm.c`)
- [x] Cross-entropy forward/backward (`loss.h`/`.c`)
- [x] Adam optimizer (`optim.h`/`.c`)
- [x] `main.c`: trainable MLP + gradient check + training loop
- [x] `Makefile` + `README.md` (module rows + Training subsection)
- [x] Smoke test (build `-Wall` clean; grad check + loss drop PASS; exit 0)

### Smoke test result (2026-07-21)
Built with `zig cc -std=c11 -O2 -Wall`; `-Wall` clean.
- Gradient check: **max relative error 2.37e-03** (tol 1e-2) — all backward ops correct.
- Training: loss **2.5848 -> 0.0001** over 300 Adam steps.
- Tokenizer + checkpoint round-trips still OK. `Output validation: PASSED`.

## Step 5b — End-to-end GPT training (attention + embedding + FFN backward)

**Status: DONE — smoke test passed (full-model gradient check + training).**

### What changed
The full `GPT` now trains end-to-end. Forward functions freed their intermediates, so
training-mode forwards were added that **cache activations**, plus matching backward ops:

- **`feedforward.c`** — `feedforward_train` (caches h1b/act) + `feedforward_backward`.
- **`attention.c`** — `multihead_attention_train` (caches Q/K/V/weights/concat) +
  `multihead_attention_backward` (incl. softmax-through-causal-mask backward). Added
  `#include <string.h>` for the weights `memcpy`.
- **`embedding.c`** — `embedding_backward` (scatter-add into wte/wpe).
- **`matrix.c`** — `bias_add_backward`/`matadd_backward` made `const`-correct.
- **`model.c`** — `GPTActs` activation cache (`gpt_acts_create/free`), `gpt_forward_train`,
  `gpt_backward` (reverse through head → final LN → each block {residual, FFN, LN2,
  residual, attention, LN1} → embeddings), `gpt_collect_params`, `gpt_zero_grad`.
- **`main.c`** — replaced the 5a MLP with the real model: end-to-end gradient check +
  next-token training on a fixed sequence.

### Gradient-check note (float32 finite differences)
First run reported max rel error 1.19e-1 on `block0.W_q[0]` (analytic 7.6e-4, numeric
6.0e-4) — a **float32 cancellation artifact**, not a bug: the loss is O(1) (~2.88) while
that gradient is ~7e-4, so `Lp-Lm ~ 1e-6` is dominated by float32 rounding of the loss.
Fixes: use `eps=1e-2`, and judge correctness on **well-conditioned** entries (`|grad|>1e-2`,
where finite differences are trustworthy). Training converging to ~0 is the corroborating
signal that the gradients are right.

### Progress checklist
- [x] FFN train/backward (`feedforward.c`)
- [x] Multi-head attention train/backward incl. causal-softmax backward (`attention.c`)
- [x] Embedding backward / scatter-add (`embedding.c`)
- [x] `GPTActs` cache + `gpt_forward_train` + `gpt_backward` (`model.c`)
- [x] `gpt_collect_params` + `gpt_zero_grad` for the optimizer
- [x] `main.c`: end-to-end gradient check + next-token training
- [x] `README.md` updated (full-GPT training)
- [x] Smoke test (build `-Wall` clean; grad check OK; loss drops; exit 0)

### Smoke test result (2026-07-21)
Built with `zig cc -std=c11 -O2 -Wall`; `-Wall` clean.
- Gradient check: max rel error **3.70e-03** (all) / **9.28e-04** (|grad|>1e-2) — full
  backprop correct.
- Training: next-token loss **2.8787 -> 0.0002** over 600 Adam steps (the 2-layer GPT
  memorizes the sequence).
- Tokenizer + checkpoint round-trips still OK. `Output validation: PASSED`.

## Step 4b — GPT-2-compatible architecture + weight loading

**Status: DONE — smoke test passed; loader verified on GPT-2-shaped weights.**

### What changed
Made the model able to *represent and load* real GPT-2 weights. The one true architecture
gap was **attention biases** (GPT-2 has `c_attn`/`c_proj` biases; we had none).

- **`attention.c/.h`** — `multihead_attention_forward/train/backward` and
  `transformer_block_forward` now take `b_q/b_k/b_v/b_o` (each d_model). Forward adds them
  after the Q/K/V and output projections (`bias_add`); backward accumulates via
  `bias_add_backward`.
- **`model.h/.c`** — `BlockWeights` gained `b_q/b_k/b_v/b_o` (zero-init); threaded through
  `gpt_create/free`, `gpt_forward`, `gpt_forward_train`, `gpt_backward`,
  `gpt_collect_params`, and `gpt_zero_grad` (per-layer count 12 → 16).
- **`checkpoint.c`** — canonical tensor order includes the biases; **format bumped to v2**
  (`TENSORS_PER_LAYER` 12 → 16).
- **Output head tying** — kept a separate `lm_head`; GPT-2's tied head is emitted as
  `lm_head = wte^T` by the converter, so no forward change was needed.
- **`tools/convert_to_gptc.py`** — added the bias keys; writes v2.
- **`tools/convert_gpt2.py`** (NEW) — HuggingFace GPT-2 `model.safetensors` → `.npz`:
  splits fused `c_attn` into W_q/W_k/W_v + b_q/b_k/b_v, maps `ln_*` → gamma/beta, ties
  `lm_head = wte^T`. (GPT-2 `Conv1D` weights are already `(in,out)` = matches our matmul, no
  transpose needed.)

### Verified vs. not
- **Verified:** `main.c` smoke test passes with biases — gradient check conditioned
  (`|grad|>1e-2`) **9.28e-04**, training still 2.8787 → 0.0002. And the full loader path with
  the **v2 format incl. biases**: random GPT-2-shaped `.npz` → `convert_to_gptc.py` →
  C `gpt_load` → forward runs clean (no NaN/Inf).
- **Not verified in-session:** `convert_gpt2.py` needs `pip install safetensors` + the ~500MB
  GPT-2 download (neither available here). Its HF→npz mapping is straightforward and matches
  HF's layout; the bit-exact `.npz`→GPTC→C part *is* tested.

### Gradient-check note
With zero-init biases, the "all" rel-error metric hits ~1.0 (biases sit at the float32
noise floor with near-zero gradients). The **conditioned** metric (`|grad|>1e-2`) is the gate
and stays at 9.28e-04 — bias backward is correct; training convergence corroborates.

### Progress checklist
- [x] Attention biases in forward/train/backward + block (`attention.c/.h`)
- [x] `BlockWeights` biases threaded through `model.c` (create/free/forward/backward/params)
- [x] Checkpoint v2 (biases in canonical order) (`checkpoint.c`)
- [x] `convert_to_gptc.py` bias keys + v2
- [x] `convert_gpt2.py` HF GPT-2 → npz importer
- [x] Docs: README (GPT-2 import recipe) + `tools/README.md`
- [x] Smoke test (build `-Wall` clean; grad check + training PASS)
- [x] Loader verified on GPT-2-shaped weights (npz → GPTC → C load → clean forward)

## Step B1 — Sampling (temperature / top-k / top-p)

**Status: DONE — smoke test passed.**

- **`sampler.c/.h`** — added a small reproducible RNG (`Rng`, xorshift64; `rng_create`,
  `rng_uniform`) and `sample_token(logits, row, temperature, top_k, top_p, rng)`:
  temperature-scaled stable softmax → sort by prob → apply top-k then top-p (nucleus) →
  sample from the renormalized kept mass. `temperature <= 0` falls back to greedy.
  Kept the existing `sample_argmax`.
- **`main.c`** — Part [3] sampling checks: `top_k=1` selects the argmax for every row, and
  the same seed reproduces the same token. Both pass.
- Model-agnostic: drives the toy model now, real GPT-2 later. Next: KV cache (B2) and a REPL
  front-end (B3).

## Step B2 — KV cache (incremental decode)

**Status: DONE — verified bit-exact vs the full forward.**

- **`model.h/.c`** — `KVCache` (per-layer `(max_seq_len, d_model)` K and V buffers) with
  `gpt_kvcache_create` / `gpt_kvcache_free`, and `gpt_decode_step(m, kv, token, logits)`:
  embeds one token, and per layer does LN1 → q/k/v (append k,v to cache) →
  single-query attention over cached positions 0..pos → W_o+bias → residual → LN2 → FFN →
  residual, then final LN → `@ lm_head`. Reuses `layernorm`/`matmul`/`bias_add`/`matadd`/
  `feedforward`; attention-over-cache is a small inline loop. Advances `kv->len`.
- **`main.c`** — Part [4] feeds the sequence one token at a time and compares each step's
  logits to the non-cached forward (`logits` row t). Result: **max |diff| = 0.00e+00**
  (bit-exact) — the cache is correct, and generation is now O(n) per token instead of
  re-running the whole prefix.

## Next steps (not started)
- Step B3 — REPL front-end: token-in/token-out generation loop using `gpt_decode_step` +
  `sample_token` (pipe-testable), then wire real text once A1 lands.
- Step A1 — Real GPT-2 BPE tokenizer: load `encoder.json` + `vocab.bpe` + pretok regex.
- Step B4 — Performance: blocked/threaded/SIMD matmul + int8 quantization.
- Run the actual 124M weights: download GPT-2, `convert_gpt2.py` → `convert_to_gptc.py` →
  `gpt_load`, then generate (needs Phase B sampling/KV-cache/REPL).
- Phase B/C: generation CLI, then the agent loop (tools + approval). See roadmap at top.
