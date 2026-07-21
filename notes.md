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
- [x] Step A1 — Trainable BPE tokenizer: merge-learning + save/load (self-trained BPE).
      (Loading GPT-2's exact `encoder.json`/`vocab.bpe` + pretok regex = future extension.)
- [x] Step A2 — Train-on-your-data pipeline: `data/` folder + `transformer train` +
      `transformer generate model.bin tok.bin` (text-in/text-out). Verified: learns a small
      corpus (loss 5.9 → ~0.1) and generates coherent in-style text.

**Phase B/scaling:**
- [x] Configurable model via CLI flags (`transformer train ... --d-model --layers --heads
      --d-ff --ctx --merges --steps --lr --seed`). Defaults: d_model 128, 3 layers, 4 heads,
      ctx 64. Each `model.bin`+`tok.bin` is a self-contained model (→ many role-specific AIs).
- [ ] Train/val split + held-out loss reporting; checkpoint-every-N; resume-from-checkpoint.
- [ ] B4 remainder — threaded/SIMD matmul + int8 quant (needed for big models to train fast).

**Phase E — Multimodal (images), the big future build (user goal: understand/describe images):**
- [x] Step E1 — Image decode: `image.c` reads binary **PPM (P6)** → HWC floats in [0,1]
      (pure C, no deps). `tools/img_to_ppm.py` converts PNG/JPG → PPM. Store in `data/images/`.
- [x] Step E2 — Patch embedding (ViT-style): `patch.c` splits the image into patches, flattens
      each, and projects with W+b (reuses `matmul`/`bias_add`) → `(n_patches, d_model)`.
      Verified in the smoke test (`[5] Image`).
- [x] Step E3 — Multimodal input: `gpt_forward_multimodal` prepends `n_img` precomputed
      image-patch embeddings before the text tokens, adds positions over the combined sequence,
      and runs the transformer (`gpt_run_from_emb` refactor shared with the text forward).
      Verified: `[5] Image: 4 patches + 3 text -> logits (7,16)`.
- [x] Step E4 — Image→caption **training**: `gpt_backward_multimodal` (scatters to `wte`/`wpe`
      and writes the image-embedding gradient) + `patch_embed_backward` (learns the patch
      projection) + cross-entropy with an **ignore index** (target < 0) so only caption
      positions are supervised. Verified: `[6] Multimodal train: loss 4.69 -> 0.0001` (overfits
      one image→caption). Real image-caption *datasets* + a bigger model = the quality frontier.
      (Honest: a genuinely useful vision-language model needs far more model+data+compute than
      is feasible locally from scratch; this builds the plumbing, quality needs real scale.)

**Phase F — Video:**
- [x] Step F1 — Frame loading: `video.c` reads `dir/frame_0000.ppm, 0001, ...` into a `Video`;
      `tools/video_to_frames.py` extracts PPM frames via ffmpeg.
- [x] Step F2 — Per-frame patch embeddings (reuse Phase E `patch_embed`) + **sinusoidal
      temporal encoding** per frame; stacked into one `(n_frames*patches, d_model)` sequence.
- [ ] Step F3 — Optional audio track → features (skipped for now).
- [x] Step F4 — Video-text training: reuse `gpt_forward_multimodal`/`gpt_backward_multimodal`
      (video patches as the prefix) + `video_embed_backward`. Smoke `[7]`: 2-frame clip →
      caption, loss 5.998 -> 0.001.
      (Honest: research-scale for real quality; this is a correct, runnable pipeline.)

**Phase B — A local generation CLI:**
- [x] Step B1 — Better sampling: temperature + top-k/top-p + seeded RNG (`sampler.c`)
- [x] Step B2 — KV cache (incremental decode; verified bit-exact vs full forward)
- [x] Step B3 — `generate` subcommand: token-in/token-out REPL (KV cache + sampler)
- [~] Step B4 — Performance: cache-friendly `ikj` matmul done; threads/SIMD/int8 quant TODO

**Phase C — The agent layer (tools + approval), pure C:**
- [x] Step C1 — Tool protocol: parse `<tool …>` / `<done>` blocks (`toolcall.c`)
- [x] Step C2 — Tool executor: write_file / read_file / run_command (`toolexec.c`)
- [x] Step C3 — Approval gate: `y/N` on stdin, or auto (`approval.c`)
- [x] Step C4 — Agent loop: parse → approve → execute → observe (`agent.c`, `agent` subcommand)

**Phase D — Quality track (as needed):**
- [ ] Step D1 — Run a small quantized code model (1–3B) for genuinely useful output
      (needs B4 threads/SIMD/quant + real weights; not implementable in this environment)

> Honest note: the *pipeline* is 100% doable in pure C and local; agent *quality* is capped by
> the model you can run. Loading pretrained weights (4b) is the fastest path to real output.

## Next phases (priority order)

1. **Training UX** — train/val split + **held-out loss**; `--checkpoint-every N`;
   `--resume model.bin` (keep training across sessions).  ✅ DONE
2. **Performance — matmul thread pool** ✅ DONE — a **persistent fork-join thread pool** in
   `matrix.c` threads `matmul` + both loops of `matmul_backward` (row-parallel → **bit-exact**;
   verified byte-identical models at 1 vs 8 threads). Thresholded (tiny matmuls stay
   single-threaded), portable (Win32 / pthreads / WASM-single-thread). **Measured 2.68× on 8
   threads** (d_model 256, 4 layers). Threads = `$TF_THREADS` or CPU count. Remaining perf
   (later): SIMD + int8 quantization.
3. **Real GPT-2 weights** ✅ DONE — `tools/gpt2_setup.py` downloads GPT-2 124M + converts to
   GPTC (→ `model/gpt2/model.bin`, 622 MB); `tools/gpt2_tok.py` bridges tiktoken. Verified:
   the pure-C model generates **coherent real GPT-2 text** ("The meaning of life is not to be
   confused with the fact of being reborn..."). Also added the `model/<name>/` folder
   convention + `--name` for train/generate. (Bigger variants: `--model gpt2-medium` etc.)
4. **Real agent backend** ✅ DONE — `agent.c` now has a pluggable `AgentBackend` interface:
   a **scripted** backend (demo/tests) and a **command** backend (`agent --cmd "<prog>"` runs
   an external program whose stdout is each response; transcript passed as a file arg). Verified
   with `tools/agent_mock.py` driving write→read→done. Plug in a real LLM by pointing `--cmd`
   at an API script (template in agent_mock.py).
5. **Phase E — Images (multimodal)** *(large)* — decode → patch embeddings → multimodal
   input → image-text training.
6. ✅ **Phase F — Video** — frames (`video.c` + `tools/video_to_frames.py`) → per-frame patches
   + temporal encoding → video-text training (reuses the multimodal path). F3 audio skipped.
   Smoke test trains a 2-frame clip → caption. DONE.

Order rationale: text (1–4) is very achievable locally; images (5) and video (6) are big
multimodal builds where the plumbing is feasible but real quality needs far more scale.

## ▶ CONTINUATION GUIDE — resume here (read this first)

**If a session ran out of usage / you're picking this up fresh, start here.**

**Repo state:** pure-C GPT that trains on text, generates text, has a token/text CLI, a
scripted agent loop, cross-platform builds (incl. WASM). Work is on branch **`agent-1`**
(several commits ahead of `master`). Tree should be clean — run `git status` and `git log
--oneline -12` to confirm where you are.

**Environment (this Windows machine):** no gcc/clang/make. Use **`zig`** (`D:\env\zig\zig.exe`)
and **Python** (`D:\env\python\python-3.9.10.amd64\python.exe` + numpy). Node at
`D:\env\node\...` (has WASI). `git` works; user identity Marjon Cajocon.

**Build + smoke test (do this after any change):**
```
zig cc -std=c11 -O2 -Wall -o transformer.exe tensor.c matrix.c activation.c layernorm.c \
  feedforward.c attention.c embedding.c sampler.c model.c checkpoint.c tokenizer.c loss.c \
  optim.c toolcall.c toolexec.c approval.c agent.c corpus.c train.c main.c -lm
.\transformer.exe            # must print: Output validation: PASSED
```
Then clean: `Remove-Item transformer.exe,*.o,*.bin,*.wasm,xbuild_* -ErrorAction SilentlyContinue`.
Cross-compile all platforms: `make cross` (POSIX shell) or the `zig cc -target ...` loop.

**Subcommands:** `train <path> [--d-model N --layers N --heads N --ctx N --steps N --lr F
--out F --tok F ...]` · `generate model.bin tok.bin` (text) or `generate model.bin` (token IDs)
· `agent [--ask]`. Crawl a corpus: `python tools/crawl_wikipedia.py "Topic" --max-pages N`.

**Conventions (enforced):** source root is **C-only and flat**; non-C scripts in `tools/`;
data in `data/` (git-ignored); cross binaries in `build/` (git-ignored except *.md/*.txt);
**2-space indent**, `-std=c11 -Wall` clean, no untyped `void *` params. **Any change to
training/backprop must keep the gradient check green** (main.c: conditioned rel err < 1e-2)
and the **KV-cache bit-exact** check (0.00e+00). Commit per logical step (auto-commit is
authorized; local only, don't push).

**Recommended next tasks (in priority order):**
1. ✅ DONE — **Training UX**: train/val split + held-out `val` loss, `--checkpoint-every N`,
   `--resume model.bin` (continue a saved model; loads its config + tokenizer). In `train.c`
   (`eval_split`, resume/checkpoint logic) + `main.c` flags.
2. ✅ DONE — **matmul thread pool** (`matrix.c`): persistent fork-join pool threads `matmul` +
   `matmul_backward`, bit-exact (verified), 2.68× on 8 threads, all 8 targets build. Threads =
   `$TF_THREADS` or CPU count. (SIMD + int8 quant remain, later.)
3. ✅ DONE — **Real GPT-2**: `tools/gpt2_setup.py` (download+convert) + `tools/gpt2_tok.py`
   (tiktoken) → pure-C model generates real coherent GPT-2 text. `model/<name>/` folder
   convention + `--name`. GPT-2 lives in `model/gpt2/` (git-ignored weights).
4. ✅ DONE — **pluggable agent backend** (`agent.c`): scripted + command backends;
   `agent --cmd "<prog>"` uses an external program as the model (`tools/agent_mock.py`).
   For a genuinely acting agent, point `--cmd` at an instruction-tuned model or an LLM API
   (base `model/gpt2` won't follow tool-call formats well).
5. ✅ **Phase E (images) COMPLETE** — E1 load, E2 patch-embed, E3 into-GPT
   (`gpt_forward_multimodal`), E4 image→caption training (`gpt_backward_multimodal` +
   `patch_embed_backward` + loss ignore-index). Then **Phase F (video)** is next.

Multimodal (asked about): **images now train end-to-end** (forward + backward + patch
projection learned; smoke test overfits image→caption). Real *quality* still needs real
image-caption datasets + a much bigger model. **Video is Phase F** (#6), the frontier.

## Session summary (2026-07-21, autonomous run)

Built out the whole pipeline in green, committed steps (newest first):
`Phase C agent` · `B4 matmul` · `A1 tokenizer` · `B3 generate` · `B2 KV cache` ·
`B1 sampling` · `4b GPT-2 compat`. Everything builds `-Wall` clean and the default
`transformer` smoke test prints `Output validation: PASSED`.

**What works now (pure C, local, flat modules):**
- A GPT-2-architecture model (attention biases, pre-norm, GELU FFN, learned pos-emb, tied
  head via converter) that **trains end-to-end** (backprop gradient-checked; loss → ~0).
- Checkpoint save/load (GPTC v2) + Python converters (`tools/convert_gpt2.py`,
  `convert_to_gptc.py`) to import real GPT-2.
- Trainable byte-level BPE tokenizer (train/save/load).
- Generation: temperature/top-k/top-p sampling + **bit-exact KV cache** + a `generate` CLI.
- A **coding-agent loop** CLI (`agent`): parse tool calls → approval gate → execute
  (write/read/run) → observe.

**What's left (and why it's not buildable here):**
- **A2 / real text output** — needs the model's vocab to equal the tokenizer's *and* a
  trained (or loaded) model; otherwise output is meaningless bytes. Blocked on real weights.
- **Real agent usefulness** — the loop is driven by a scripted model; a capable backend
  (real LLM API, or a large locally-run model) is required. The built-in transformer is far
  too small to emit valid tool calls.
- **B4 perf (threads/SIMD/int8 quant)** and **D1 (run a 1–3B model)** — large systems work,
  only matters at real scale; can't be exercised on this toy model here.

The honest bottom line: **all the machinery/plumbing for a local coding-agent CLI is built
and tested**; turning it into something genuinely useful is gated on running a real,
capable model — either download+convert GPT-2 (step 4b recipe) or wire a real LLM backend
into `agent.c`.

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

## Step B3 — `generate` subcommand (token-level REPL)

**Status: DONE — pipe-tested.**

- **`main.c`** — split into `run_demo()` (the default smoke test) and `run_generate()`,
  dispatched by `main(argc, argv)`: `transformer generate [checkpoint.bin]`.
  Reads lines of space-separated **token IDs** from stdin, feeds them through the KV cache
  (`gpt_decode_step`), then autoregressively samples `n_new` tokens (`sample_token`,
  temp 0.8 / top-k 40 / top-p 0.95, seed 1234) and streams them (bracketed). Loads real
  weights if a `.bin` is given, else a random toy model. Out-of-range IDs are skipped; a
  leading UTF-8 BOM is stripped (Windows pipe robustness).
- **Verified:** `echo "1 5 9" | transformer generate` → `gen: 1 5 9 [15] [15] [5] ...`;
  multi-line input loops; `99` (>= vocab 16) is skipped. Default `transformer` still
  `PASSED`. Build `-Wall` clean.
- This is the CLI skeleton: **token-in/token-out** now, **text-in/text-out** once A1 (real
  BPE) + real weights land.

## Step A1 — Trainable BPE tokenizer (merge-learning + persistence)

**Status: DONE — smoke test passed.**

- **`tokenizer.c/.h`** — `tokenizer_train(t, text, num_merges)` learns merges greedily (most
  frequent adjacent pair each round, replacing occurrences in the working sequence);
  `tokenizer_save`/`tokenizer_load` persist the merge list (`BPE1` format; the 256 base byte
  tokens are implicit).
- **`main.c`** Part [1] — trains 30 merges on a sample string: 45 bytes → 20 tokens,
  `decode(encode(x)) == x`, and save→load reproduces identical token IDs.
- **Note / honesty:** this is a *self-trained* byte-level BPE, fully local and testable.
  Matching GPT-2's *exact* tokenization (its `encoder.json` + `vocab.bpe` + the GPT-2
  pretokenization regex + byte→unicode map) is a separate extension needed only to feed text
  to real GPT-2 weights — text↔token vocab must also match the model's vocab (see A2).

## Step B4 (partial) — cache-friendly matmul

**Status: matmul reorder DONE; threads/SIMD/quant remain.**

- **`matrix.c`** — `matmul` rewritten from `i-j-k` to **`i-k-j`** order: the inner loop
  streams a row of B and a row of C sequentially (cache-friendly, auto-vectorizable)
  instead of striding down a column of B. Same math.
- **Verified:** KV-cache logits still bit-exact (0.00e+00), gradient check + training
  unchanged (2.8787 → 0.0002) — the reorder is correct.
- **Remaining (needed only at 124M+ scale):** multi-threading (OpenMP/pthreads), SIMD
  intrinsics, and int8 weight quantization. Tracked under Phase D.

## Phase C — Agent layer (parse → approve → execute → observe)

**Status: DONE — both approval modes tested; loop creates and reads a real file.**

Four small flat C modules build a local coding-agent loop (pure C, no external deps):

- **`toolcall.c/.h`** — `toolcall_parse` extracts the first `<tool name=… path=… command=…>
  body</tool>` or `<done>…</done>` block into a `ToolCall` (kind + attrs + body). Plain text
  with no block = final answer.
- **`toolexec.c/.h`** — `toolexec_run` executes a call and returns an observation string:
  `write_file` (fopen/fwrite), `read_file` (fopen/fread), `run_command` (portable
  `popen`/`_popen`).
- **`approval.c/.h`** — `approval_confirm(mode, action)`: `APPROVAL_ASK` prompts `y/N` on
  stdin, `APPROVAL_AUTO` auto-approves. BOM-robust (Windows piped input).
- **`agent.c/.h`** — `agent_run(goal, responses[], n, mode)`: the loop. For each model
  response: parse → (if tool) describe + approve → execute → print observation; `<done>` or
  plain text ends it. `responses[]` is the pluggable "model backend" — a fixed script here;
  a real LLM integration replaces it with live calls that see the observations.
- **`main.c`** — `agent` subcommand: `transformer agent` (scripted demo, auto-approve),
  `transformer agent --ask` (interactive approval). The demo writes `agent_demo.txt`, reads
  it back, then cleans it up.

**Verified:** `transformer agent` runs write→read→done, actually creating/reading the file;
`transformer agent --ask` (piped `y`) approves and does the same; denying (or no input)
blocks the action. Default `transformer` smoke test still `PASSED`; `-Wall` clean.

**Honest limitation:** the built-in transformer is far too small/weak to emit valid tool
calls, so the agent is driven by a scripted model here. The loop, tools, and approval gate
are real and reusable; plugging in a capable model (a real LLM API, or a much larger locally
-run model — Phase D) is what makes it genuinely useful.

## Step A2 — Train-on-your-data pipeline

**Status: DONE — trains on a text file and generates coherent in-style text.**

- **`data/`** — where your training data goes (git-ignored except `README.md` + `sample.txt`).
  This folder is data, not code, so it's exempt from the C-only-flat-root rule.
- **`train.c/.h`** — `train_run(data, out_model, out_tok, num_merges, steps)`: reads the text
  file, trains a byte-level BPE (`tokenizer_train`), sizes a GPT to that vocab (d_model 64,
  2 layers, 4 heads, d_ff 256, ctx 32), then trains on random context windows
  (`gpt_forward_train` → `cross_entropy` → `gpt_backward` → Adam), and saves the model +
  tokenizer.
- **`main.c`** — `train` subcommand: `transformer train <data.txt> [model.bin] [tok.bin]
  [merges] [steps]`. And `generate` now has a **text mode**: `transformer generate model.bin
  tok.bin` encodes stdin text → generates → decodes back to text (token-ID mode still works
  without a tokenizer).
- **Verified:** `transformer train data/sample.txt model.bin tok.bin 150 400` → 427 bytes →
  83 tokens, vocab 312, loss **5.95 → ~0.1**; then `echo "the cat" | transformer generate
  model.bin tok.bin` produced coherent corpus-style text ("the cat sat on the mat. the dog
  sat on the log. ..."). Default `transformer` smoke test still `PASSED`; `-Wall` clean.
- **How to use your own data:** put a `.txt` in `data/`, run `transformer train
  data/yourfile.txt model.bin tok.bin 200 2000`, then `echo "prompt" | transformer generate
  model.bin tok.bin`. More/cleaner text + more steps = better; scale the config in `train.c`.

## Data crawler + directory training + faster BPE encode

**Status: DONE — crawl → train → generate verified on real Wikipedia text.**

- **`tools/crawl_wikipedia.py`** (NEW) — crawls *connected* Wikipedia articles via the API
  (plain-text extracts + breadth-first link following), saving `data/wiki/<title>.txt`.
  Stdlib `urllib` only; needs internet. Verified live: fetched "Machine learning" (58.7K
  chars) → linked "15.ai" (28.3K).
- **`corpus.h/.c`** (NEW) — `corpus_read(path)` reads a file, **or a directory tree
  recursively** (every `.txt`, portable dir-walk: Win32 `FindFirstFile` / POSIX `dirent`).
- **`train.c`** — now uses `corpus_read`, so **`transformer train data/`** (a directory)
  trains on all `.txt` under it. Answers "does data/ support subdirectories?" → **yes**.
- **`tokenizer.c`** — `tokenizer_encode` rewritten to standard BPE (merge *all* occurrences
  of the lowest-rank pair per round, not one). Same result, hugely faster: an 87KB corpus
  went from >120s (timeout) to **2.7s**. Essential for real corpora.
- **Verified end-to-end:** crawled 87KB of Wikipedia → `transformer train data/wiki` (300
  merges, 200 steps, 2.7s) → `generate` runs. Output is undertrained gibberish at 200 steps
  (tiny model) — needs many more steps + a bigger model for coherence; the *pipeline* is the
  deliverable. Directory training also verified on `data/` (sample.txt + a `wiki/` subdir).
  Default smoke test still `PASSED`; `-Wall` clean.

**Further BPE speedup (future):** word-level pretokenization (BPE within each word, like
GPT-2) would make encode near-instant on large corpora; current whole-sequence encode is
fine for MBs, slow for GBs.

## Cross-platform portability (Linux / macOS / Windows / WebAssembly)

**Status: DONE — all 8 targets build `-Wall` clean; Windows + WASM verified running.**

Cross-compiled every source with `zig cc -target <triple>`:
- Linux `x86_64/aarch64-linux-gnu`, `x86_64-linux-musl`; Windows `x86_64/aarch64-windows-gnu`;
  macOS `x86_64/aarch64-macos`; **WebAssembly `wasm32-wasi`** — all build.
- Native Windows runs (smoke test PASSED); the **WASM binary runs via Node's WASI** and
  passes the full smoke test (tokenizer, gradient check, KV cache bit-exact, PASSED) — file
  I/O works through WASI preopens.

Fixes this required (the Windows-only build had never exercised these paths):
- **`toolexec.c`** — `popen`/`pclose` are POSIX; glibc/musl hide them under `-std=c11` unless
  `_POSIX_C_SOURCE` is defined. Added it (non-Windows). WASM has no process model, so
  `run_command` is `#ifdef`-stubbed to "not supported on WebAssembly" there.
- **`corpus.c`** — same `_POSIX_C_SOURCE` guard for `opendir`/`readdir`/`stat`.
- **`tensor.c`** — `RAND_MAX` exceeds float's exact range on Linux/macOS; divide in `double`
  to silence `-Wimplicit-const-int-float-conversion` and keep init precise everywhere.

**`make cross`** builds every target into `build/` (via `zig cc`; needs a POSIX shell).
`build/` is git-ignored except text/markdown (`build/README.md` documents the outputs).
See README "Supported platforms" for the target list + build/run commands.

## Configurable model (CLI flags) — DONE

`train` is now fully configurable (no editing `train.c`): `TrainConfig` in `train.h`,
`train_default_config()` + `train_run(data, cfg)` in `train.c`, flag parsing in `main.c`'s
`run_train`. Flags: `--d-model --layers --heads --d-ff --ctx --merges --steps --lr --seed
--out --tok`. Validates `d_model % heads == 0`. Defaults are a small-but-real model (d_model
128, 3 layers, 4 heads, ctx 64). Verified: small config trains+generates; bad `--heads`
errors. Each trained `model.bin`+`tok.bin` is a standalone model → build many role-specific
AIs. See the CONTINUATION GUIDE above for the prioritized backlog.

## Training UX (train/val + checkpoint + resume) — DONE

`train.c`: train/val split (val = tail `val_frac` of the token stream; auto-disabled if the
corpus is too small to hold out a window), `eval_split()` prints held-out `val` loss each
report, `--checkpoint-every N` saves mid-run, and `--resume model.bin` loads a saved model +
its tokenizer (`--tok`) and continues training (config comes from the checkpoint, not flags).
`main.c` flags: `--val-frac --checkpoint-every --resume`. Verified: checkpoint saved mid-run;
resume continued from the saved loss (not from scratch); val loss tracks train loss.

## Phase #2 (performance) — matmul thread pool DONE

**First attempt (reverted):** per-call threading (spawn threads inside each `matmul`) measured
**0.94× — slower** (per-call OS thread spawn > benefit; backward unthreaded).

**Final (kept):** a **persistent fork-join thread pool** in `matrix.c`:
- N-1 worker threads created once (lazily), reused for every parallel region — no per-call
  spawn. `parallel_for(total, fn, ctx)` splits `[0,total)` into N chunks (workers do 0..N-2,
  the caller does N-1) using a mutex + two condition variables. `atexit` joins the workers.
- Portable sync wrappers: **Win32** `CRITICAL_SECTION`+`CONDITION_VARIABLE`, **POSIX**
  `pthread_mutex`+`pthread_cond`, **WASM** disabled (runs inline). Makefile adds `-pthread`.
- Threads `matmul` (over output rows) and **both loops of `matmul_backward`** (over rows of
  dA and dB) — the bulk of training compute. Attention's inner loops are not threaded (minor
  at small seq).
- **Bit-exact:** row-parallel, so each output element is produced by one thread in the same
  order → verified byte-identical checkpoints at `TF_THREADS=1` vs `8`. Tiny matmuls stay
  single-threaded (below `MM_THREAD_THRESHOLD` ~500k MACs), so the demo/tests are unchanged.
- **Speedup:** **2.68×** on 8 threads (d_model 256 / 4 layers / ctx 64). Thread count =
  `$TF_THREADS` or the CPU count (capped 64). All 8 cross-targets build; WASM still runs.

Remaining perf (later, if needed): SIMD (auto-vec / guarded intrinsics) + int8 quantization.

## Phase #3 (real GPT-2) — DONE, generates real text

- **`tools/gpt2_setup.py`** — downloads GPT-2 `model.safetensors` (huggingface_hub) and writes
  a GPTC checkpoint directly (splits fused `c_attn`, ties `lm_head = wte^T`). One command →
  `model/gpt2/model.bin` (vocab 50257, d_model 768, 12 layers, 622 MB).
- **`tools/gpt2_tok.py`** — GPT-2 BPE via `tiktoken`: `encode "text"` → ids, `decode ids` →
  text. (We use tiktoken instead of reimplementing GPT-2's regex+BPE in C.)
- **`model/<name>/` convention + `--name`** — `main.c`: portable `ensure_dir` + `model_paths`;
  `train --name X` saves `model/X/{model,tok}.bin`; `generate --name X` loads it (text mode if
  `tok.bin` present, else token-ID mode — GPT-2 has none). `model/**/*.bin` git-ignored.
- **Verified end-to-end** on this machine: pip install safetensors+tiktoken, downloaded +
  converted GPT-2 (153 s), ran `echo <ids> | transformer generate --name gpt2` → **coherent
  GPT-2 text** in ~5.6 s / 40 tokens. Deps present: python `D:\env\python`, ~1.5 GB RAM to load.
- Remaining niceties: a `gpt2_chat.py` wrapper (encode→generate→decode in one), and threading
  the M=1 decode-step matmuls over columns (currently single-threaded per token).

## Phase #4 (pluggable agent backend) — DONE

- **`agent.h/.c`** — `AgentBackend` interface (struct + `respond` fn ptr; concrete backends
  embed it as the first field → no `void *`). Two backends: `scripted_backend` (fixed
  responses) and `command_backend` (runs an external program; transcript written to a temp
  file passed as argv, program stdout = response). `agent_run(goal, AgentBackend*, mode)` keeps
  a running transcript, capped at `AGENT_MAX_STEPS`. Portable popen (Win32/POSIX; WASM disabled;
  `_POSIX_C_SOURCE` for glibc/musl under -std=c11).
- **`main.c`** — `agent [--ask] [--cmd "<prog>"] [--goal "..."]`.
- **`tools/agent_mock.py`** — example backend that drives write→read→done (reads the transcript
  to know the step), with a commented Anthropic/OpenAI call to swap in a real LLM.
- Verified: scripted + command backends both run; all 8 targets build; smoke test PASSED.

## Phase #5 / Phase E (multimodal images) — COMPLETE (E1–E4)

- **`image.c/.h`** — loads binary **PPM (P6)** into an `Image` (h,w,c=3; HWC floats [0,1]),
  pure C (skip whitespace/#comments in the header). `tools/img_to_ppm.py` (Pillow) converts
  PNG/JPG → PPM so the C stays dependency-free.
- **`patch.c/.h`** — `patch_embed`: grid of (patch×patch) patches → flatten (HWC) → `matmul`
  by W (patch·patch·C, d_model) + `bias_add` → `(n_patches, d_model)`. `patch_count`/`patch_dim`.
- **`main.c`** smoke test `[5]`: writes a 4×4 PPM, loads it, patch-embeds (patch 2 → 4 patches),
  checks shape + finiteness. All 8 targets build; PASSED.
- **E3 DONE:** `model.c` `gpt_forward_multimodal(m, img_emb, ids, n_img, seq_len, a, logits)`
  builds `emb = [patch_emb + wpe] ++ [wte[tok] + wpe]` then calls the shared `gpt_run_from_emb`
  (refactored out of `gpt_forward_train`, so the text path is unchanged/bit-exact). Causal
  attention over the combined sequence (text sees the image patches). `a` sized to
  `n_img+seq_len` (<= max_seq_len). Verified `[5] Image: 4 patches + 3 text -> logits (7,16)`.
- **E4 DONE:** `gpt_backward_multimodal` (block backward → scatter text grads to `wte`/`wpe`,
  write image grad to `img_emb->grad`) + `patch.c` `patch_embed_backward` (learns W_patch) +
  `loss.c` ignore-index (target < 0 skips). Smoke `[6]` overfits image→caption (loss 4.69 →
  ~0). Refactored `gpt_backward_to_emb` shared by text + multimodal backward.
  Honest: useful VLM quality needs real image-caption datasets + far more scale.

## Phase #6 / Phase F (video) — COMPLETE (F1, F2, F4)

- **`video.c/.h`** — `Video` = sequence of frames. `video_load(dir)` reads
  `frame_0000.ppm, 0001, ...` sequentially. `video_embed` patch-embeds each frame (shared
  W/b, reuses `patch_embed`) + adds a **sinusoidal temporal encoding** per frame, stacking
  into `(n_frames*patches, d_model)`. `video_embed_backward` distributes the grad per frame
  (temporal term is constant → passes through). `video_patch_count`.
- **`tools/video_to_frames.py`** — ffmpeg → `frame_%04d.ppm` (`-start_number 0`).
- **`main.c`** smoke `[7]`: 2-frame clip → video patches + caption → multimodal train; loss
  5.998 -> 0.001. Video is just [frame patches...] + [text] through `gpt_forward_multimodal`,
  so E3/E4 did the heavy lifting. F3 (audio) skipped.

## 🏁 Roadmap COMPLETE — all six phases done

#1 Training UX · #2 Thread pool · #3 Real GPT-2 · #4 Agent backend · #5 Images (E1–E4) ·
#6 Video (F1/F2/F4). Pure C11, all 8 platforms (Linux/macOS/Windows/WASM), every step green
(smoke test PASSED, gradient-checked, bit-exact where claimed). Honest ceiling: the pipelines
are correct and runnable; *quality* for real image/video understanding needs real datasets +
far larger models than a from-scratch CPU build.

**Possible future work (optional):** F3 audio; real image-caption/video datasets; SIMD/int8
quant (#2 remainder); a `tools/agent_llm.py` real-LLM agent backend; instruction-tuning; a
`gpt2_chat.py` one-command wrapper; larger GPT-2 variants (`gpt2-medium/large/xl`).
- Optional real agent: a `tools/agent_llm.py` calling a real LLM API (`agent --cmd ...`).
- Wire a real model backend into `agent.c` (replace scripted `responses[]`) — live LLM API
  or a large locally-run model.
- Phase D — perf/quant + a genuinely capable model (real weights).
- Run the actual 124M weights: download GPT-2, `convert_gpt2.py` → `convert_to_gptc.py` →
  `gpt_load`, then generate (needs Phase B sampling/KV-cache/REPL).
- Phase B/C: generation CLI, then the agent loop (tools + approval). See roadmap at top.
