# memory.md — portable memory snapshot

Portable memory so a Claude session on any machine can get up to speed fast. This file is
**committed to the repo** (tracked in git), so it travels with `git clone`/`pull` across
machines. On a new PC, tell Claude: **"read memory.md"**. Deep, always-current project detail
lives in `notes.md` (esp. its "▶ CONTINUATION GUIDE").

> **Sync rule (user request):** this file mirrors Claude's memory — whenever the assistant
> updates its memory, it must also update `memory.md` here.

> **▶▶ real LLM agent backend — ✅ DONE (2026-07-22):** `tools/agent_llm.py` reads the
> transcript (argv[1], stdin fallback), calls a real LLM with a system prompt constraining it to
> emit exactly one `<tool name="write_file|read_file|run_command" ...>...</tool>` or
> `<done>summary</done>` block (attribute syntax matched to `toolcall.c`), and prints only text
> blocks to stdout (strips ``` fences). Provider: **Claude via Anthropic API** (`claude-opus-4-8`,
> adaptive thinking) default; **OpenAI** if only `OPENAI_API_KEY` set. No key / no SDK / API error
> → prints a `<done>` so the C loop ends cleanly. Run:
> `transformer agent --cmd "python tools/agent_llm.py" --goal "..." --ask`. No C changes (backend
> already pluggable). Verified here (no key): compiles, no-key `<done>` path, and full C
> command-backend round trip; smoke test PASSED. Live LLM call untested (no `anthropic` pkg / key);
> `agent_mock.py` remains the offline fallback.
>
> **Code health — DONE (2026-07-22):** the 12-part smoke test (`run_demo` → **`run_selftest`**) +
> its private helpers (`cond_gen`, `media_to_text`, `gpt_loss`) moved from `main.c` into new
> **`selftest.c`/`selftest.h`**; `main.c` is now CLI dispatch only (995 → 367 lines). Makefile +
> README + notes updated (`selftest.o` in OBJS). Behavior byte-identical (PASSED, grad 9.28e-04,
> KV 0.00e+00). Build now lists `selftest.c` (the notes build cmd also gained the media `.c`
> files it had been missing).
>
> **Dataset-trained imggen/audgen — DONE (2026-07-22):** `imggen`/`audgen` gained a **dataset
> mode** alongside single-example reconstruct, auto-selected from the input path (single file =
> unchanged reconstruct; a **folder** of `img_%04d.ppm` / `aud_%04d.wav` = learn from the set +
> **generate** a novel sample). New `dataset_autoregress` in main.c: prepends a **BOS** token
> (model vocab V+1) so it learns P(first token) and samples unconditionally from BOS
> (`sample_token`, temp 0.9/top-k 40/top-p 0.95; BOS clamped out of output). Numbered-file
> loaders (same convention as `video_load`, no new dir-walk). Verified: `-Wall` clean, smoke
> PASSED; 4-PPM set (3 red/1 blue) → generates red; 3-WAV set → wrote a clip; single-file recon
> still 16/16. Honest: memorizes at this scale; codecs cap quality — the folder→generate pipeline
> is the deliverable.
>
> **Roadmap + both backups are now done.** Optional future: real image-caption/video datasets,
> VQ-VAE/diffusion for quality, mel-spectrogram audio, SIMD/int8 quant, larger GPT-2 variants.

_Last updated: 2026-07-22._

## Who / project
- User: Marjon Cajocon (git identity: Marjon Cajocon / marjoncajocon08@gmail.com).
- Project: a **from-scratch, pure-C11 GPT/LLM** built step-by-step as a learning project,
  growing toward a **local coding-agent CLI**, and eventually **multimodal** (images = Phase E,
  video = Phase F). Repo: `llm-project` (this folder).

## Working preferences & feedback (how to work with me)
- **Auto-commit is authorized** — after a step is done + smoke-tested, commit it (clear message,
  per logical step). **Local only — do NOT push** unless asked.
- **Branch workflow:** ongoing work is on branch **`agent-1`**. `master` is the separated/stable
  line — the **user merges to master themselves**. Don't merge to master or leave `agent-1`
  unless asked; if a session starts on `master`, `git checkout agent-1` first.
- Likes: honesty about what works vs. what's undertrained/stubbed/needs-scale; each step ending
  green (smoke test `PASSED`); detailed notes so work survives a usage cutoff.
- When choosing between approaches, give a clear recommendation (they often ask "what is better?").

## Hard rules (enforced in README)
- Source root is **C-only and flat** — no code subfolders, no non-C source in root.
- **`tools/`** = non-C helper scripts (Python converters, `crawl_wikipedia.py`).
  **`data/`** = training data (git-ignored). **`build/`** = cross-compiled binaries
  (git-ignored except *.md/*.txt).
- **2-space indent**, `-std=c11 -Wall` clean, **no untyped `void *`** in public APIs.
- **Any training/backprop change must keep the numerical gradient check green** (main.c:
  conditioned rel err < 1e-2) AND the **KV-cache bit-exact** check (0.00e+00).

## Environment (current machine — Windows)
- **No gcc/clang/make.** Use **`zig`** = `D:\env\zig\zig.exe` (also cross-compiles + `zig cc`),
  **Python** = `D:\env\python\python-3.9.10.amd64\python.exe` (+ numpy), **Node** at `D:\env\node`
  (has WASI). `git` works.
- Build + smoke test:
  ```
  zig cc -std=c11 -O2 -Wall -o transformer.exe *.c -lm    # (list the .c files explicitly)
  .\transformer.exe        # must print: Output validation: PASSED
  ```
  Then clean artifacts (`transformer.exe *.o *.bin *.wasm xbuild_*`). Cross-compile all: `make cross`.

## What's built (working, committed)
Full **text** pipeline: transformer (GPT-2-style, attention biases, pre-norm, GELU FFN) that
**trains end-to-end** (gradient-checked). Train on your data (`transformer train <file|dir>` with
CLI flags for model size), **train/val split + held-out loss + `--checkpoint-every` + `--resume`**,
BPE tokenizer (train/save/load), sampling (temp/top-k/top-p), **bit-exact KV cache**,
`generate` (text or token-IDs), a scripted **agent loop** (`agent` — parse→approve→execute→observe),
a **Wikipedia crawler** (tools/), and **cross-platform builds** (Linux/macOS/Windows + **WASM**,
via `make cross`).

## Next phases (priority order) — see notes.md for detail
1. ✅ Training UX (val split, checkpoint, resume) — DONE.
2. ✅ Performance — persistent **matmul thread pool** in `matrix.c` (threads matmul +
   matmul_backward; bit-exact, verified; **2.68× on 8 threads**; Win32/pthreads/WASM;
   `$TF_THREADS` or CPU count). SIMD + int8 quant remain (later). DONE.
3. ✅ Real GPT-2 — `tools/gpt2_setup.py` (download+convert → `model/gpt2/model.bin`) +
   `tools/gpt2_tok.py` (tiktoken). Pure-C model generates **real coherent GPT-2 text**.
   Added `model/<name>/` folder convention + `--name` for train/generate. DONE.
4. ✅ Real agent backend — `agent.c` pluggable `AgentBackend`: scripted + **command** backend
   (`agent --cmd "<prog>"`, program stdout = model response; `tools/agent_mock.py` example).
   Plug in a real LLM via an API script. DONE.
5. ✅ Phase E images COMPLETE (E1–E4): `image.c` (PPM), `patch.c` (ViT patch embed +
   backward), `model.c` `gpt_forward_multimodal`/`gpt_backward_multimodal`, `loss.c` ignore
   index; smoke test overfits image→caption (loss ~4.7 → ~0). `tools/img_to_ppm.py`.
6. ✅ Phase F video+audio COMPLETE: `video.c` (frames), `audio.c` (16-bit WAV → waveform
   framing → project + temporal encoding + backward), `tools/video_to_frames.py` +
   `tools/audio_to_wav.py` (ffmpeg). Smoke trains video→caption (6.0→0.001) and audio→caption
   (7.3→0.0017). Audio is raw-waveform frames (not mel-spectrograms).

🏁 **All modalities UNDERSTOOD: text, image, video, audio** — multimodal fwd/bwd.

**GENERATION (Phase G) — image/audio/video ALL work** (discrete-token autoregressive; reuse
GPT + KV cache + sampler + cross-entropy; NOT diffusion). Codecs: `imagegen.c` (color-cube,
64 tokens) + `audiogen.c` (µ-law, 256). ✅ GI unconditional (`[10]` regenerate 8×8 image
64/64). ✅ **text→image / text→audio / text→video** (`cond_gen` in main.c: combined vocab =
media+text, text ids offset above media; predict media after the text prefix; smoke `[11]`
16/16, 32/32, 32/32). Video-gen = concatenated per-frame image tokens. Also ✅ **media→text captioning**
(`media_to_text` in main.c: image/audio/video → generate caption text; smoke `[12]` 3/3 each)
— so media↔text works BOTH ways. ✅ **CLI subcommands** `transformer imggen <in.ppm> <out.ppm>
[levels] [steps]` + `audgen <in.wav> <out.wav> [steps]` generate real media FILES
(image_save_ppm/audio_save_wav; `autoregress_reconstruct` helper; overfits one input, small
only — O(n^2)). Remaining polish: real datasets, VQ-VAE/diffusion for quality, a captioning CLI.

**Model naming/identity:** `--name X` = the model FOLDER (`model/X/`). The PERSONA ("I am X,
created by Y") comes from TRAINING DATA — `data/identity.txt` (identity Q&A) trained in; demo:
`train data/identity.txt --name marvin` → "Who are you?" → "I am Marvin, an AI model created by
Marjon." Add identity.txt alongside other data to bake identity into any model.

Pure C11, 8 platforms, every step green. Ceiling: pipelines correct/runnable; real quality
needs real datasets + far bigger models (palette/µ-law codecs are crude). Optional future:
VQ-VAE/diffusion, mel-spectrogram audio, SIMD/int8 quant, real-LLM `tools/agent_llm.py`.

Model folders: each model in `model/<name>/` (`model.bin` + `tok.bin`); `model/**/*.bin`
git-ignored, READMEs kept. `model/gpt2/` holds real GPT-2 (token-ID mode via tiktoken).
One-command GPT-2: `python tools/gpt2_chat.py "prompt"` (tokenize → C generate → detokenize;
resolves the exe to an abspath so it runs from cwd on Windows).

## How a model becomes a capable coding assistant — the 5 stages (reference)

Discussed with the user 2026-07-22. What separates a frontier model (e.g. Claude Opus 4.8)
from this project is **scale + data + compute + engineering — NOT a secret algorithm.** The
*techniques* are public (research papers); the weights/data/exact recipe are proprietary and
closed (you cannot download or copy them). `capability ≈ model_size × data × compute ×
training_method`. The pipeline that turns raw next-token prediction into a coding assistant:

1. **Architecture** — a transformer. This project is GPT-2-style (LayerNorm, learned positions,
   GELU MLP, attention biases). Modern (LLaMA-family) = **RMSNorm, RoPE, SwiGLU, GQA**, no biases,
   SentencePiece tokenizer. Adding these is the roadmap's big leap (→ mini-`llama.cpp`).
   **Progress — ALL FOUR modern primitives DONE + gradient-checked (standalone, NOT yet wired into
   a block):** ✅ **RMSNorm** (`rmsnorm.c`, `[13]`, 7.7e-4) · ✅ **RoPE** (`rope.c`, HF half-split,
   `[14]`: grad 1.4e-3 + rel-pos 1.8e-7) · ✅ **SwiGLU** (`swiglu.c`, `[15]`: grad 1.2e-3) ·
   ✅ **GQA** (`gqa.c`, grouped-query causal attn, `[16]`: grad 7.3e-3). ✅ **Assembled BLOCK**
   (`llama.c`, `LlamaBlock`: RMSNorm→RoPE→GQA→resid→RMSNorm→SwiGLU→resid, no biases; forward-only /
   inference; new in-place `rope_heads` in `rope.c`; selftest `[17]`: forward finite + end-to-end
   CAUSALITY verified — perturb last pos changes only its own row, earlier rows 0.00e+00). Next:
   `convert_llama.py` (HF safetensors→our layout, incl. HF RoPE weight permutation) + a multi-block
   model path + SentencePiece/BPE tokenizer bridge → load & generate from SmolLM2-135M / Qwen2.5-0.5B.
   Also SIMD/int8 quant for speed.
   ✅ **Multi-block MODEL** (`LlamaModel` in `llama.c`: wte embedding → N blocks → final RMSNorm →
   lm_head; no positional term; forward-only; selftest `[18]`: 3-layer forward finite + model-level
   causality 0.00e+00). **All 8 `make cross` targets build with the new modules incl. wasm32-wasi.**
   ✅ **LLMA checkpoint** (`llama_model_save/load` in `llama.c`: flat binary, magic+config+weights in a
   shared canonical order like GPTC; selftest `[18]` round-trip bit-exact 0.00e+00; cross-compiles incl.
   wasm). ✅ **convert_llama.py + `llama` CLI**
   (`tools/convert_llama.py`: HF LLaMA-family safetensors→LLMA; **NO RoPE permutation needed** — rope.c
   is HF half-split, so HF q/k load directly; transposes linear weights (HF (out,in)→our (in,out)),
   tied lm_head=wte^T, warns on Qwen2 attn biases (unsupported); `--tiny` no-download test file.
   `main.c` `llama <ckpt>` subcommand: token-ids→next-token argmax. Verified Python-write→C-load→forward
   with --tiny, cross-compiles). ✅ **tokenizer bridge + generate path**
   (`main.c` `llama <ckpt> [n_new]` now autoregressively samples & prints `gen: <ids>`, no KV cache yet;
   `tools/llama_tok.py` encode/decode via `tokenizer.json`; `tools/llama_chat.py` one-command
   tokenize→generate→detokenize. Verified tiny: `echo "5 9 2" | transformer llama t.bin 12` → in-range
   ids). 🎉 **REAL MODERN MODEL RUNS (2026-07-22):** converted **SmolLM2-135M** (`convert_llama.py`, 621 MB
   LLMA; needed a manual bf16→f32 safetensors reader `load_safetensors_f32` since safetensors.numpy
   can't load bf16; rope_base 100000 read from config) and generated fluent English through the pure-C
   engine: "The capital of France is" → "…the capital of the department of Loir-et-Cher. It is"
   (coherent grammar = weight map/RoPE/GQA/SwiGLU/RMSNorm/tokenizer all CORRECT). `model/smollm/*.bin`
   git-ignored; deps: tokenizers 0.22.2, huggingface_hub 1.8.0, safetensors 0.7.0. **The from-scratch
   pure-C engine loads & runs a real LLaMA-family LLM.** ✅ **KV cache for LlamaModel**
   (`LlamaKV` + `llama_decode_step` in `llama.c`: caches K/V per layer, single-query attention over
   cache, O(n)/token; `rope_heads_pos` added; `llama` CLI primes cache w/ prompt then samples;
   selftest `[19]`: decode == full forward **bit-exact 0.00e+00**). Perf remaining: SIMD/int8 quant.
   Qwen2.5 needs attn-bias support. Block/model forward-only (add backward to fine-tune).
2. **Pretraining** — next-token prediction on HUGE text **+ code** corpora. This project does this
   at toy scale (`transformer train`).
3. **Instruction tuning (SFT)** — supervised fine-tune on `(instruction → good response)` pairs;
   this is the step that makes a base model FOLLOW instructions instead of just continuing text.
   Same machinery as our supervised next-token training — just the right data (could include our
   `<tool …>` agent format). **DONE here** three ways: (a) smoke-test part `[9]` (masked, hand-built,
   deterministic); (b) whole-stream demo on `data/instruct.txt` (`Q:`/`A:`, memorizes the stream —
   keep `--merges` LOW for routing); (c) **true masked SFT** `transformer train data/sft.txt --sft`
   (`prompt<TAB>response` lines; `train_sft()` in `train.c` supervises the RESPONSE tokens only,
   prompt+padding masked with target −1). Verified: each instruction routes to its exact tool-call/
   `<done>`, and generation **stops cleanly** — `--sft` reserves an **EOS** token
   (`EOS = tok.vocab_size`, model vocab = tok+1), appends+supervises it, and `generate` halts on
   any id ≥ tok vocab (non-SFT models unaffected; no checkpoint change). (Bug fixed en route:
   `logits`/`probs` must be sized to `model.vocab_size`, not `tok.vocab_size`, after the +1.)
4. **RLHF / DPO** — align to helpful/correct/human-preferred outputs. Anthropic's **Constitutional
   AI** is a *published* paper.
5. **Tool-use / agentic training** — train on tool-call formats (fits our `agent.c` `<tool>`/`<done>`
   protocol).

**Honest ceiling (told the user):** the techniques transfer to this project at small scale, but
the *capability* does not — a from-scratch CPU model can't reach Opus-level coding (needs
frontier-scale data + compute + teams). "Human-like" coding is an EMERGENT property of scale +
alignment, not a reasoning module to copy. Realistic capable-coding paths: (a) plug a capable
model behind our agent loop — Claude API via `tools/agent_llm.py`, or a **local** model via
Ollama/`llama.cpp` (OpenAI-compatible endpoint; `agent_llm.py` already speaks that API);
(b) local hardware for a real local model ≈ **12–24 GB GPU or 32 GB+ Apple Silicon + Qwen2.5-Coder
7B–32B (Q4)**. Offline-vs-capable tradeoff: our own small model = offline but weak; capable =
online API or a separate local-LLM runtime (not this from-scratch engine).
