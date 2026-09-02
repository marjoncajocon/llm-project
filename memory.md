# memory.md — portable memory snapshot

Portable memory so a Claude session on any machine can get up to speed fast. This file is
**committed to the repo** (tracked in git), so it travels with `git clone`/`pull` across
machines. On a new PC, tell Claude: **"read memory.md"**. Deep, always-current project detail
lives in `notes.md` (esp. its "▶ CONTINUATION GUIDE").

> **Sync rule (user request):** this file mirrors Claude's memory — whenever the assistant
> updates its memory, it must also update `memory.md` here.

> **▶▶ Pure-C tokenizer — engine parses HF models, NO Python at runtime — ✅ DONE (2026-07-22):**
> User goal "our own llm can parse it, not python." `baf5f20` **`hftok.c/.h`** = from-scratch
> byte-level BPE: loads a `.tok` vocab (id→bytes) into a bytes→id hash, hand-codes the GPT-2
> pretokenization regex, runs the tiktoken min-combined-id merge rule. Key insight: HF BPE ids are in
> MERGE ORDER, so the same algorithm works for tiktoken AND HF tokenizers. `tools/export_tokenizer.py`
> (`gpt2`→tiktoken, `<hf-repo>`→tokenizers lib) writes the `.tok` ONCE (like a weight converter);
> runtime is pure C. `5a0517a` **`hfgen <model.bin> <tok.hf> [n]`** = pure-C HF chat (stdin text →
> encode → generate, auto-detects GPTC/LLMA/LLMQ → JSON `{"d":piece}` deltas → `{"done":true}`, stops
> on EOS/`\nUser`); webchat `streamText` uses it when `model/<name>/tok.hf` exists, else Python
> bridges. **VERIFIED EXACT** vs reference for GPT-2, SmolLM2, Qwen2.5 (punctuation/code/numbers/
> contractions); gpt2 + SmolLM2 chat end-to-end in pure C. To activate:
> `python tools/export_tokenizer.py <gpt2|hf-repo> model/<name>/tok.hf` once. `.gitignore` covers
> `*.hf`. (The diffusers text→image hook `da472d5` still exists but is OFF/optional — user wants the
> engine to do it, so the pure-C `t2i` is the default; diffusers install was stopped.)
>
> **▶▶ Live streaming chat + mixed multimodal + FROM-SCRATCH text→image — ✅ DONE (2026-07-22):**
> Big arc on the `tools/webchat/` frontend (all on `agent-1`, each smoke-green). Full detail:
> `notes.md` → "Webchat → live streaming chat → mixed multimodal + FROM-SCRATCH text→image".
> **Chat:** `017ce13` live token streaming + conversation memory (C prints `t <id>` per token;
> `tools/llama_stream.py`/`gpt2_stream.py` incrementally detokenize → JSON deltas; Go `runCmdStream`
> forwards; client keeps `User:/Assistant:` history + Clear); `b4773bd` runs until EOS (no max-new);
> `f63e5c7` fixed truncated replies; `f61cf75` standalone exes via `build.ps1`/`make webchat` +
> auto-detect root/engine; `48552e1` threaded int8 decode (`qacc_columns`, ~1.3× — bandwidth-bound);
> `749f628` prefer int8 ckpt. Chat quality capped by base Qwen2.5-0.5B (convert the **-Instruct**
> variant to improve). **Mixed chat:** `8074c3b` C `text2media` (text→image/audio teach-then-
> reproduce); `581e21f` unified chat AUTO-DETECTS modality (keyword + Auto/Text/Image/Audio/Video
> override), renders inline (chat_media); 📎 attach for the media demo.
> **TEXT→IMAGE FROM SCRATCH (DALL-E/Parti recipe, NO diffusion/libs):** `e02815c` step1 `t2i`
> (GPT learns P(image|caption) on a folder of cap_%04d.txt/img_%04d.ppm pairs); `8c28860` step2
> **`vqvae.c/.h`** = a learned VQ-VAE image codec (patch encoder + learned codebook + decoder,
> straight-through estimator, matmul only — 16×16→64 codes vs 256 px, MSE 0.0083); `0048fbc` step3
> t2i uses the VQ-VAE codes (4× fewer tokens); `d33a3d9` `t2i-train <dataset> <prefix>` saves
> `.gpt`+`.vqt`, `t2i-gen <prefix> <out> <caption>` generates in ~0.03s, webchat image intent
> generates from `model/t2i/` inline. Usage: `t2i-train model/mydata model/t2i/t2i 3000 2000` then
> ask "a picture of X". Ceiling: shown on red/green/blue (no real dataset here); novel images need
> thousands of pairs + scale — architecture done, needs data+compute. `.gitignore` covers
> `*.gpt`/`*.vqt`. HF text-to-image (SD/FLUX) is diffusion → can't load here; this is the AR path.
>
> **▶▶ Web testing frontend (Go+WebSocket) + `caption` subcommand — ✅ DONE (2026-07-22):**
> `tools/webchat/` = a self-contained **Go** web server (stdlib only, hand-rolled RFC 6455;
> embedded single-page UI) to test ALL modalities from a browser. 4 committed phases (`5eb41eb`
> text chat + model switcher; `59a12b5` image/audio gen; `14e7d5d` caption + video). **Model
> switcher** (`models.go`) scans `model/<name>/`, peeks the 4-byte magic → routes GPTC+tok.bin to
> pure-C `generate --name`, GPTC/no-tok to `gpt2_chat.py`, LLMA/LLMQ to `llama_chat.py` (SmolLM2/
> Qwen HF-tok map). Media bridged in Go stdlib (PPM↔PNG, WAV) — no Pillow. **New C `caption`
> subcommand** (`caption.c/.h`, wired in main.c+Makefile): `caption <img.ppm|aud.wav|frames_dir>
> <text> [steps]` = teach-then-reproduce media→text via the existing patch/audio/video front-ends
> + selftest's `media_to_text` path; prints `caption: <text>`; **additive, smoke stayed PASSED**.
> Video needs **ffmpeg** (absent on this box → degrades w/ a clear error). **Gotcha:** media runs
> are SLOW not hung — attention is O(seq²) on CPU (audgen 512smp/100steps=37s); webchat caps inputs
> (img ≤16×16, aud ≤256 smp) + low steps (~10s) + a 4-min subprocess timeout. Honest ceiling:
> reconstruct/overfit + teach-then-reproduce (no trained multimodal ckpt) — proves pipelines RUN on
> real media, not a real captioner. Run: `cd tools/webchat && go run . -root ../..` → :8080. Go is
> at `D:\env\go_lang` (go 1.23.12). See `tools/webchat/README.md`.
>
> **▶▶ Security/robustness hardening pass — ✅ DONE (2026-07-22):** a 5-agent code review of the
> whole tree found the MATH correct (softmax/GELU/RMSNorm/RoPE/GQA/cross-entropy/int8 all
> verified) but bugs at the **trust boundaries**. Fixed the top of the list — 4 criticals + every
> high + top mediums — in **6 commits on `agent-1`**, each smoke-tested (`PASSED`, grad 9.28e-04,
> KV 0.00e+00) and `-Wall` clean. Full detail: `notes.md` → "Security & robustness hardening
> pass". In order: `b08d7b6` loader validation (reject corrupt/hostile checkpoint + tokenizer
> headers before building — bad dims, `n_kv==0`, uneven GQA, merge ids); `9f34d12` sandbox the
> tool executor (`toolexec` confines file paths to cwd — no abs/`..`/drive/UNC; `run_command`
> NEVER auto-approves, always interactive); `5fd66ff` size_t migration (`Tensor.size`,
> `Adam.total`, matmul `M*N*K` threshold — kills the 2^31 overflow/corruption class); `6151129`
> fail-closed persistence (ATOMIC saves = temp+rename; `gpt_param_count` heap list replaces the
> `params[1024]` stack-smash; self-test now uses `selftest_*.tmp` so it can't overwrite/delete a
> user's `model.bin`/`tok.bin`; unknown subcommand → usage+exit 1, not the self-test); `5064878`
> debug validation layer (**new `check.h`** — typed shape/bounds asserts that abort at the fault;
> **ON in the default build, `-DNDEBUG` compiles them out**); `73902ac` no-grad inference tensors
> (`tensor_create_nograd`; LLaMA path + GPT KV cache/decode skip the unused grad array →
> **halves a loaded fp32 model's RAM**, ~540 MB for SmolLM2-135M). Remaining (minor, none
> load-bearing): checkpoint endianness, a real `generate` CLI (seed/temp flags), >2 GB corpora
> (`corpus.c` int narrowing), crawler retry/backoff, `convert_llama.py` sharded checkpoints.
>
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
- Build + smoke test (PowerShell doesn't glob for native exes — expand `*.c` first):
  ```
  $c = (Get-ChildItem *.c | ForEach-Object { $_.Name }); zig cc -std=c11 -O2 -Wall -o transformer.exe @c -lm
  .\transformer.exe        # must print: Output validation: PASSED
  ```
  `check.h` is header-only (no `.c`). Its shape/bounds assertions run in this default build; add
  `-DNDEBUG` for a checks-off release build. Then clean artifacts (`transformer.exe *.o *.bin
  *.wasm *.tmp xbuild_*`). Cross-compile all: `make cross`.

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
   selftest `[19]`: decode == full forward **bit-exact 0.00e+00**). ✅ **int8 quantization (LLMQ)** (`quant.c` per-column int8;
   `convert_llama.py --int8` → LLMQ ~4× smaller; `llama_model_load` dequants on load, forward unchanged
   = storage/download win; selftest `[20]` matmul L2 rel err 0.68%; SmolLM2 621 MB → 156 MB, still
   coherent). ✅ **in-memory int8 + quantized matmul**
   (`QTensor`+`qmatmul` in quant.c; `QLlamaModel`/`q_llama_decode_step`/`q_llama_model_load` keep int8
   in RAM, decode via qmatmul; `main.c llama` auto-detects LLMQ→int8 path, LLMA→f32; selftest `[21]`
   int8 decode vs f32 forward L2 rel err 0.64%; 4× less RAM). ✅ **int8×int8 qmatmul (real speedup)**
   (quant.c qmatmul now quantizes activations to int8 + int32 accumulate, auto-vectorizes under
   `-O3 -march=native`; selftest `[21]` 0.75%). **Measured (token-IDs direct, AVX2 build, 30 tok):
   int8 0.9 s vs fp32 2.3 s = 2.5× faster.** ⚠️ CORRECTION: earlier "~85 s" was Python huggingface_hub
   doing a slow HEAD check on tokenizer.json, NOT the C engine. **FIXED:** `llama_tok.py`/`llama_chat.py`
   now use `hf_hub_download(local_files_only=True)` (one-time download fallback) → `llama_chat` dropped
   ~83 s → **1.3 s** (64×) for a 20-tok SmolLM2 int8 gen. Whole pipeline now ~1 s. Perf remaining: explicit AVX2/NEON intrinsics, block/prefill qmatmul, threading.
   ✅ **q/k/v attention biases (Qwen2 support)** (`b_q/b_k/b_v` in LlamaBlock+QLlamaBlock, `bias_add`
   after q/k/v matmul in all paths; always-present, zero for LLaMA/SmolLM; convert reads q/k/v_proj.bias
   or zeros; checkpoint **v2**, TENSORS_PER_LAYER 9→12 — old v1 files must be re-converted). Smoke PASSED
   with random biases; SmolLM2 v2 still coherent. **Qwen2.5-0.5B RUNS** (rope_theta 1000000, q/k/v
   biases): "The capital of France is" → "Paris. The population of Paris is ~1,67 million" (correct →
   bias path proven); ~3.0 s / 20 tok int8. `model/qwen/*.bin` git-ignored. Block/model forward-only
   (add backward to fine-tune). Two real models now run: SmolLM2-135M + Qwen2.5-0.5B.
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


## 2026-07-22 -- Session: chat templates + C-native text-to-image (RESUME HERE)

Branch `agent-1` (local only; NOT pushed -- master merged by user). Everything below is
smoke-tested (Output validation PASSED); grad-check + KV bit-exact untouched (all additive).

**DONE this session:**

1. **Chat stop fixes** (commits 745c057, ce6ed61, e04b692): generation stops at end tokens
   (`<|endoftext|>`/`<|im_end|>`/`<|eot_id|>`) and at role labels
   (User/You/Human/Assistant/AI/System) matched as WHOLE WORDS at line start -- fixes Qwen's
   trailing "\nUser" garbage and fake self-conversations. In main.c `hf_stop`,
   tools/llama_stream.py, gpt2_stream.py, webchat handlers.go `trimAtStop`.

2. **Special tokens + chat templates in pure C** (commits 45b0a08, dd60835) -- the big one:
   - `.tok` format -> v2 (v1 still loads): adds bos, template_kind (plain/chatml/llama3),
     default_system, and the special-token table (id, end-of-generation flag, literal string).
   - hftok.c: `hftok_encode` splits input on special-token strings (greedy, probed at '<') and
     emits each as ONE id; `hftok_is_eog` (stop set); `hftok_chat_render` (turns -> prompt for
     ChatML / Llama-3 / plain).
   - main.c: `hf_generate()` shared core stops on the REAL eog token; new
     `hfchat <model.bin> <tok.hf> [n]` reads RS(0x1e)/US(0x1f)-delimited turns from stdin and
     applies the template in C.
   - tools/export_tokenizer.py writes v2 (reads added_tokens + chat_template from HF files).
   - webchat sends structured turns -> hfchat. VERIFIED: a ChatML prompt tokenizes
     BIT-IDENTICAL to HF `tokenizers` (`<|im_start|>`=151644, `<|im_end|>`=151645 single ids).
   - NOTE: `model/*/tok.hf` are git-ignored; qwen/smollm/gpt2 already re-exported to v2. Base
     Qwen/SmolLM still ramble (not instruct-tuned) -- the infra is correct; an INSTRUCT model
     would emit `<|im_end|>` and stop cleanly (offered to convert SmolLM2-135M-Instruct, not done).

3. **Text-to-image kept C-NATIVE** (user: "no python for diffusers, make it C, from scratch"):
   uninstalled torch/diffusers; webchat `-imagegen` is now OPT-IN only (commit f0bd205). Default
   image path = the from-scratch `t2i` (VQ-VAE + autoregressive GPT, all pure C, no Python).

4. **t2i improved** (commits 482afc5, 09bab75):
   - `t2i-data`: procedural dataset generator (5 shapes x 8 colors; caption "a <color> <shape>").
   - Generation SAMPLES image codes (temp + top_k, masked to the code vocab) not greedy argmax.
   - **Minibatch gradient accumulation** (BATCH=8) in `t2i_build`: `gpt_backward` zeros+fills
     grads per example, so accumulate into a shadow buffer + average before `adam_step`. Drove
     GPT loss 1.0 -> **0.0002** on the DETERMINISTIC 40-shape set.
   - `t2i-recon`: debug subcommand -- round-trips an image through the trained VQ only (no GPT).
   - RESULT (VERIFIED via montage): every "a <color> <shape>" caption -> correct color AND a
     recognizable shape. Clean from-scratch pure-C text-to-image.

**generalization (commit 4294e91 code) -- VALIDATED 2026-07-23:**
   - Reintroduced per-variant position/size jitter (seeded per index) + linear LR decay to 0.1x.
     Defaults now: `t2i-data` n=480, `t2i-train` 1800 steps, gen temp 0.6.
   - Prior no-decay run (1400 steps) UNDERTRAINED: loss plateaued ~0.25, confused blue->green,
     blobby shapes. LR decay + gentler jitter fixed that.
   - **CONFIRMING RUN (2026-07-23):** `t2i-data data\t2i 480 16` then
     `t2i-train data\t2i model\t2i\t2i 1800 1800`. VQ recon_mse ~0.005; GPT loss **6.50 -> 0.093**
     (blew past the old 0.25 plateau ~step 1400 as LR decayed). Model on disk is now the GOOD one.
   - **VERIFIED via montage** (each caption x 4 seeds): all 5 shapes {circle, square, triangle,
     diamond, cross} render recognizably; all 8 colors {red green blue yellow cyan magenta orange
     purple} render DISTINCT + CORRECT (blue->green confusion GONE). Outputs vary per seed but keep
     correct color+shape => real generalization, not memorization. Nit: orange skews slightly toward
     yellow. NOTE: shape vocab is those 5 only -- out-of-vocab captions (e.g. "star") -> a blob.

**THEN (user-requested next): richer concepts** -- multi-object scenes + spatial relations
("on the left"), more shapes -> real compositionality.

Build: `$c=(Get-ChildItem *.c|%{$_.Name}); D:\env\zig\zig.exe cc -std=c11 -O2 -Wall -o
transformer.exe @c -lm`. Both exes (transformer.exe, tools/webchat/webchat.exe) left built.
Montage helper (view PPMs): scratchpad ppm_montage.py (stdlib zlib PNG).


## 2026-07-23 -- Session: two-object compositional t2i + checkpointing (RESUME HERE)

Extended single-object t2i to **two-object scenes with spatial relations** (commit 4a4d880,
all `t2i.c`). Caption "a <c1> <s1> <rel> a <c2> <s2>", rel in {left of, right of, above, below}.
`draw_scene` places obj1 in the named half (jittered); `run_t2i_data` randomly samples
(c1,s1,rel,c2,s2) per index over a 6400-combo space >> dataset. `T2I_CAPTION_LEN` 32->48,
GPT d 96->128 / dff 512.

**Checkpointing added:** `t2i_build` gained ckpt_prefix/every; `run_t2i_train` saves every 300
GPT steps (two runs got killed near ~90 min -- the runtime wall on this box). Verified on disk.

**RESULT (24px/d128/2200 steps, 768 scenes, loss 0.054) -- PARTIAL SUCCESS** (relation-swap +
unseen-pairing montages):
  - Horizontal relations (left of / right of) LEARNED -- correct position across seeds and on
    unseen pairings.
  - Vertical relations (above / below) CONFLATED -- both render blue-top/red-bottom. Likely
    raster-order AR: left/right repeats every scanline, above/below must commit at first codes.
  - Colors mostly right but 2nd object bleeds (blue->magenta) sometimes (small objects at 24px).

**32px scale-up (d160) FAILED on CPU:** ~3.6-5 s/step at seq 304 -> only ~1300 steps fit before
the ~90-min kill vs 2200 needed; UNDERTRAINED, colors collapsed (all red/orange). Reverted to
24px/d128. 32px two-object > this machine's training budget.

**On disk:** `model/t2i_scenes/t2i.*` = restored 24px partial model (0.054); dataset
`data/t2i_scenes` (768 @ 24px). Reproduce: `t2i-data data\t2i_scenes 768 24` +
`t2i-train data\t2i_scenes model\t2i_scenes\t2i 2200 2000` (deterministic).

**NEXT (fix vertical relations / bleed, all fit 24px): (1) dedicated relation/color/shape vocab
tokens instead of raw caption bytes -- highest leverage; (2) d160 + 2600 steps; (3) 1200 scenes.
Brute-force 32px+ is off the table until faster compute.**


## 2026-07-23 (eve) -- Structured caption tokens (WIP, NOT validated) -- RESUME HERE

Implemented option (1) above -- **structured caption tokens** (commit 964846a, build clean).
`encode_caption` now PARSES "a <c1> <s1> <rel> a <c2> <s2>" into 5 dedicated tokens
[c1,s1,rel,c2,s2] (`match_word_color/shape/rel`; T2I_TOK_COLOR/SHAPE/REL/PAD; T2I_VOCAB=274).
`T2I_CAPTION_LEN` 48->5 (seq 192->149); GPT d 128->160, dff 512->640. Runs ~0.76 s/step
(~30 min / 2200 steps). Sanity OK (vocab 274, loss5.51~=ln274). **Training STOPPED at step
450/2200 (user home) -- NOT validated, no good model on disk (ignore model/t2i_tok step-300 ckpt).**

**RESUME (~30 min):** rebuild (zig cc) -> `t2i-data data\t2i_tok 1000 24` (or reuse existing) ->
`t2i-train data\t2i_tok model\t2i_tok\t2i 2200 2000` -> VALIDATE with relation-swap montage
("a red circle <rel> a blue square" x seeds, t2i-gen temp 0.4, recreate scratchpad ppm_montage.py).
SUCCESS = "above" puts red on TOP / "below" red on BOTTOM (byte-caption model conflated them).
Baseline to beat: byte-caption model `model/t2i_scenes` (commit 4a4d880, loss 0.054, vertical
conflated). See notes.md "Structured caption tokens" for the full step list.


## 2026-07-24 -- Structured caption tokens VALIDATED (relations fixed; VQ color is the new wall)

Commit 160ab66. Finished the run above and validated it **objectively** with a scored harness
instead of eyeballing a montage. **The structured-token fix WORKED: vertical relations are no
longer conflated.** Bottleneck moved to color, and color is the **VQ-VAE's** fault, not the GPT's.

Run: VQ 2000 steps (recon_mse 0.0019); GPT 2200 requested but KILLED at ~step 1900 (loss ~0.11),
so the **step-1800 checkpoint** in `model/t2i_tok/` is what every number below was measured on.

New tooling (use these, don't squint at montages):
  - `tools/t2i_score.py` -- scores **color fidelity** and **relation accuracy** as SEPARATE axes
    (nearest-palette-color pixel assignment + per-color centroids). Conflating the two is exactly
    what made my first read of the montage wrong.
  - `tools/ppm_montage.py` -- stdlib PPM -> scaled PNG montage.

CALIBRATION (ground-truth `data/t2i_tok/img_*.ppm`, same tool): color 100% (24/24), relation
**88%** (21/24). So ~88% is the measurement CEILING for relations, not a shortfall.

Measured (step-1800 model, temp 0.4):
  - ground truth (ceiling) ........ color 100% | relation 88% (21/24)
  - VQ-VAE RECONSTRUCTION of it ... color **67%** (16/24) | relation 75% (12/16)
  - GPT gen, all 8 colors ......... color 33% (16/48) | relation **81%** (13/16), above **6/6**
  - GPT gen, 4 "safe" colors ...... color 30% (17/56) | relation 71% (12/17)

Conclusions:
  1. **Relations essentially SOLVED, vertical included** -- 81% vs an 88% ceiling. The byte-caption
     baseline had ZERO vertical sensitivity (always blue-top/red-bottom). Goal achieved.
  2. **Color fidelity is capped by the VQ-VAE**: reconstructing *training* images (no GPT involved)
     already loses a color 33% of the time. More GPT steps CANNOT fix this.
  3. Failure is adjacent-hue collapse (blue->cyan, magenta->purple, orange->yellow); only 14
     distinct codes of K=256 used per image => smells like codebook COLLAPSE, not K too small.

NEXT: 1) fix VQ-VAE color (dead-code restart / EMA codebook / commitment cost / T2I_LDIM 16->32,
plus an aggregate codebook-utilization printout to confirm collapse) and re-measure the recon row;
2) then finish GPT training (~30 min); 3) re-measure relations -- horizontal sample sizes were tiny
(n=1..4) because color failures ate them.

## 2026-07-24 -- MULTI-STEP PLANNING + requested future-work checklist

The agent now PLANS multi-step tasks (not just one-shot routing). `tools/make_agent_traj.py`
synthesizes multi-step tasks and explodes each into (transcript-prefix -> next-action) SFT pairs
ending in <done>; transcript turns flattened with " | " and `tools/agent_local.py` applies the
same flatten at inference so formats match. SmolLM2-135M LoRA-SFT on 500 trajectory + 378 Q&A
pairs. VERIFIED end to end: goal "create hello.txt containing world and read it back" -> write ->
read -> <done>, file held "world". The model chose the correct SECOND action from the transcript
and stopped at the goal. Known bug: a two-write template hardcodes "ok" as the 2nd content, so the
2nd write's content isn't copied from the goal (data artifact, not a planning failure). No C changed.

### FUTURE WORK (requested) -- [x] applied in code, [~] partial, [ ] not yet

- [x] Reasoning-dense data/training -- APPLIED 2026-07-24. make_agent_traj.py --reason wraps each
  step as "<think>rationale</think><tool>"; agent_local.py shows reasoning on stderr, records the
  bare tool call. Verified end to end: the agent states a grounded thought before each action
  ("First I'll create hello.txt...", "Read hello.txt to check...", "The goal is complete") and
  completes the multi-step task. Reasoning-dense + multi-step + execution together.
- [x] Execution-oriented -- APPLIED. agent.c/toolcall.c/toolexec.c parse -> approve -> execute ->
  observe real tool calls over multiple steps; SFT labels are executable tool calls; observations
  come from real execution.
- [x] High-quality synthetic data -- DONE 2026-07-24. `tools/clean_sft.py` VERIFIES each pair (reject
  a tool call whose path/write-content/command-file arg is not grounded in the prompt; whole-token
  match so "ok" != inside "cook"; empty write body rejected), DEDUPES exact+near-dup lines (variety
  kept), BALANCES action-type x task-length with full logging. Validated on an adversarial probe (4/4
  rejects + 4/4 keeps). Fixed two_writes "ok" bug AND widened write-content 10->308 words
  (make_agent_traj.py) so copy generalizes in every position. Clean corpus = 1547 pairs (0 rejected,
  62 dups). KEY GOTCHA: retrain #2 DIVERGED (loss ->7.7, garbage) because train.c `grad_clip`
  defaults to 0.0 (clipping OFF); the harder widened data needs it. Retrain #3 with `--clip 1.0`
  (recipe-recommended) is STABLE (loss 3.0->0.12) and gave: 2nd-write novel content now copied
  (was empty); routing/prose-vs-tool intact; live agent_local.py multi-step run (write "world"->read->
  <done>, hello.txt held "world"); generalization 100% novel filenames / 70% novel commands (misses =
  char copy-slips, still right tool). => model/smollm/reason_clean.bin. Commits: gate 113db78,
  content-widen 6e0b15e. STABILITY: grad_clip now DEFAULTS to 1.0 in train.c (commit 4a61ffb; was
  0.0=off, which let retrain #2 diverge); pass --clip 0 to disable / reproduce old unclipped perf tables.

## FUTURE WORK -- long-horizon research roadmap (requested 2026-07-24)

Aspirational research directions recorded verbatim from the user's vision. NONE are applied in
code yet -- all unchecked. These sit far beyond the current pure-C / 135M pipeline and are the
long-term north star, not the next sprint. (The near-term, actionable items stay in the earlier
"requested direction" checklist: reasoning-dense [in progress], execution-oriented [x], high-
quality synthetic data [~].)

### Strengthen core reasoning & reduce hallucination
- [ ] Integrate Process Reward Models (PRMs) -- reward each reasoning STEP, not just the outcome.
- [ ] Outcome-supervised RL (RLHF / RLAIF with a twist) -- optimize final-answer correctness.

### Enhance multi-agent & tool-use capabilities
- [ ] Train a meta-cognition & planning module (plan/monitor/replan over tools).
- [ ] Simulate multi-agent debate & refinement (proposer/critic rounds before committing).

### Embed self-improvement & data-curation loops
- [ ] Entropy-based data-quality filtering (keep informative examples, drop low-signal ones).
- [ ] Constitutional AI + backtranslation for alignment (self-critique against a rule set).

### From single-task execution to autonomous goal pursuit
- [ ] Train on ambiguous problem decomposition (break a vague goal into sub-goals).
- [ ] Intrinsic motivation / curiosity-driven exploration (seek novel, learnable states).

### Meta-learning & in-context self-improvement layer
- [ ] In-context reinforcement learning (adapt within a single context from feedback).
- [ ] Predictive coding & world-model simulation (predict, compare, correct).

### Cognitive architecture for true general intelligence
- [ ] Global-workspace architecture with specialized modules.
- [ ] Scheduler / "consciousness" (the main LLM) routing between modules.
- [ ] Long-term procedural memory (the fine-tuned execution model -- this project's agent).
- [ ] Critic module (a separately fine-tuned evaluator).
- [ ] Open-ended self-improvement.

### Intervention-based training data
- [ ] Generate counterfactual data ("what if X were different").
- [ ] Define the supervision signal for interventions.
- [ ] Ground in key research (causal inference / do-calculus).

### Learning mechanism: online learning & belief updating
- [ ] Continuous Bayes filtering in latent space.
- [ ] Training architecture loop: Prediction -> Observation -> Surprise calculation -> Learning
      trigger (update only when surprise is high).

### Generating probabilistic futures: point predictions -> scenario analysis
- [ ] Multi-future generation ("cone of plausibility": a distribution of outcomes, not one point).

### The continuous prediction-adaptation loop (operational blueprint)
- [ ] Ingestion: real-world stream (news, markets, papers, domain data) -> structured causal
      graph via a causal encoder.
- [ ] Simulation & prediction: each cycle the causal world-model emits a probability distribution
      over future states.
- [ ] Action & intervention (optional): choose an intervention that steers the future toward a
      more desirable probability cone.
- [ ] Observation & prediction error: real data arrives; a surprise module measures divergence.
- [ ] Targeted online learning: on high surprise, an eligibility-trace mechanism isolates the
      faulty causal chain and rewires it online -- the causal graph improves incrementally
      without forgetting old links.

Honest note: this is a multi-year research program (causal world-models, online belief updating,
multi-agent cognition), not incremental engine work. The current repo provides ONE piece of it --
the "long-term procedural memory / execution model" (the fine-tuned tool-using agent). Everything
else here is unbuilt and, at 135M / CPU / pure-C, mostly out of reach without a different compute
and model scale. Recorded so the direction is not lost. See [[planned-next-step]].

## >>> RESUME HERE (2026-09-02) -- GRPO (RL from verifiable rewards) ✅ BUILT + VERIFIED

> User asked to "support the latest algorithm" -> chose **GRPO** (Group Relative Policy Optimization,
> the DeepSeek-R1 algorithm) as the direct successor to the STaR loop, with the **static verifiable**
> reward (no tool execution during training). New subcommand: **`transformer grpo`**. Smoke test
> green (`Output validation: PASSED`), `-Wall` clean, new parts `[26]`-`[29]`.
>
> **WHY IT BEATS STaR:** STaR keeps the winning rollouts and throws the losers away, so a failure
> teaches nothing. GRPO normalizes rewards WITHIN a group of G rollouts, so a below-average rollout
> gets a NEGATIVE advantage and is actively suppressed. No value net, no reward model -- the group's
> own mean is the baseline, which is why it fits 135M on a CPU.
>
> **THE TRICK THAT MADE IT SMALL:** `∇_logits log π = onehot − p`, so the entire new backward is
> `pg_backward` = `cross_entropy_backward` with `1/N` swapped for a per-token `coeff[t]` (~15 lines
> in loss.c). Everything downstream is reused untouched. And **the KL reference policy is free**:
> with LoRA the base is frozen, so the same model with adapters OFF *is* the reference -- no second
> copy of 621MB.
>
> **NEW FILES:** `reward.c/.h` (format 0.2 + grounding 0.3 + action-match 0.5; grounding is a C port
> of `clean_sft.py verify()`), `grpo.c/.h`, `tools/grpo_tasks.py`. **MODIFIED:** `loss.c/.h`
> (`pg_backward`, `logprob_of`), `llama.c/.h` (`llama_decode_step_lora` -- adapters after the q/k/v
> bias and BEFORE RoPE, mirroring the training forward), `llamatrain.c/.h`
> (`llama_lora_decode_view`), `main.c`, `Makefile`, `selftest.c`, `README.md`.
>
> **RUN IT:**
> ```
> python tools/grpo_tasks.py --out data/grpo_tasks.txt --actions-only data/clean_reason.txt
> .\transformer.exe grpo data/grpo_tasks.txt --model model/smollm/reason_e2.bin \
>     --hf-tok model/smollm/tok.hf --out model/smollm/grpo.bin --group 8 --steps 200 --verbose
> ```
> ~12-40s/step at G=6, max-new 96, on the 135M. 1420 tasks built from `clean_reason.txt`.
>
> **TWO GOTCHAS THE FIRST RUNS FOUND (both fixed, do not re-learn them):**
> (1) a small `--max-new` truncates every tool call -> every rollout scores identically -> zero
> variance -> zero gradient. The run now WARNS when >25% of rollouts hit the cap.
> (2) the model emits a CORRECT tool call then rambles into SmolLM2 pretraining artifacts
> (`<jupyter_text>`, `<file_sep>`) instead of EOS. Rollouts now **stop at `</tool>`/`</done>`** --
> which is what the real agent does anyway (`toolcall_parse` takes the first block only) -- and that
> also cut rollout length 96 -> 28-40 tokens.
>
> **NOTHING PROMOTED.** No long run yet. **GRPO learns from VARIANCE**, and on `reason_e2.bin` the
> mean reward is already ~0.94 so most groups are FLAT and get skipped. Next: build a task file
> weighted toward cases the model gets right only SOMETIMES, run long, then `tools/remeasure.py` +
> `tools/probe3.py`, and promote only if non-regressive. Watch `--verbose` for reward hacking (total
> rising while the `match` term stays flat).
>
> Full detail: notes.md "2026-09-02 -- GRPO: RL from verifiable rewards".

## >>> PREVIOUS (2026-07-26) -- lever (a) 2-epoch retrain ✅ DONE+PROMOTED; STaR harness ✅ BUILT

> **SESSION COMPLETE, TREE CLEAN, nothing left half-done.** Three commits on `agent-1`
> (local only, NOT pushed): **4bea3b9** lever (a) retrain + train.c fflush + tools/probe3.py +
> tools/remeasure.py | **90488e1** memory.md stale-recipe fix | **8c9f405** STaR self-training loop
> (tools/self_train.py + AGENT_TRACE hook). Rebuild+smoke green (Output validation: PASSED).
>
> **CURRENT MODEL = `model/smollm/reason_e2.bin`** (2 epochs, probe 8/8). `reason_clean.bin` KEPT as
> the A/B baseline; `reason_e2_s1.bin` is the stage-1 midpoint (651MB, delete if disk is tight -- it
> is reproducible). Use reason_e2.bin in every probe/live command from now on. Model .bin files are
> git-ignored, so the promoted checkpoint lives ONLY on this machine -- the commits carry the recipe,
> not the weights.
>
> **FIRST ACTIONS ON RESUME:** `git status` (expect clean) + `git log --oneline -5`. The exe may be
> stale/cleaned -- rebuild with the zig glob (see the env block further down).

## STaR SELF-TRAINING LOOP -- ✅ BUILT + VERIFIED (2026-07-26 c). Harness only; NO retrain yet.

User's ask: *"support this AI to learn from itself -- when I prompt, he learns."* Built as
**rejection-sampling self-training**: the agent's own VERIFIED-successful runs become SFT data.
Tools only, no C. `tools/self_train.py` (NEW) + `AGENT_TRACE` hook in `tools/agent_local.py`.

```
$env:AGENT_TRACE = "data/self_trace.jsonl"      # off by default => normal runs unchanged
.\transformer.exe agent --cmd "... tools/agent_local.py model/smollm/reason_e2.bin model/smollm/tok.hf" --goal "..."
python tools/self_train.py --trace data/self_trace.jsonl --out data/self_traj.txt   # --dry-run to inspect
```

**THE MODEL NEVER GRADES ITSELF** -- that is what keeps this from amplifying its own errors. Every
accept/reject keys off TOOL OBSERVATIONS or `clean_sft.verify()` grounding: terminal `<done>`
required (else `no-terminal-done`); final observation must not be an error (earlier errors are KEPT
-- error-recovery is GOOD data); every row must pass grounding (**this is what rejects the ~5%
char-slip** -- verified on the historical `demo_note`->`demonstra_note`); loop guard `--max-repeat 2`
(lever #4 saw a live 8x re-read loop); at least one tool call. **All 5 reject paths tested and
correct.** Runs are rebuilt by PREFIX-CHAINING (step k's prompt contains step k-1's) -- no run id.

VERIFIED END-TO-END on real runs: 2 traced live runs on reason_e2.bin -> both accepted -> 9 rows ->
merged via clean_sft at 1798+9 = **1807 pairs, 100% kept**. (Bug found+fixed in the process: `load()`
used plain utf-8, so a BOM silently DROPPED record #1; now `utf-8-sig` like `clean_sft.read_rows`.)

**HONEST LIMITS -- do not oversell:** (1) it AMPLIFIES what the model sometimes already gets right,
it does not create new capability -- lever #4 measured that at 135M the residual errors are
SYSTEMATIC/low-variance (the proposer never emitted the right action in 6 samples). (2) NEVER train
on self-data alone -- always fold in the original corpus or the distribution narrows. (3) NOTHING is
promoted without `tools/remeasure.py` green. (4) **"learns the instant you prompt it" needs the
resident-model lever (c) FIRST** -- every generation reloads 651MB now; a single-sample LoRA step is
only ~3s of compute, so per-prompt online updates are viable ONLY with the model in memory. Batch
STaR (collect -> verify -> periodic retrain) works TODAY without it.

**NEXT for STaR:** accumulate traces over real use (tens of runs), then the standard 2-stage retrain
on `clean_sft`-merged self+original data, then remeasure, then promote only if non-regressive.

Full detail: notes.md "2026-07-26 (c) -- STaR self-training loop".

**ALSO ASKED: "can it predict the future like a human?"** Two different things. Next-token prediction
it already does. Predicting real-world FUTURE EVENTS needs a causal world-model (state, intervention,
counterfactuals, surprise-triggered belief update) -- that program is recorded at the END of this file
and is honestly OUT OF REACH at 135M/CPU/pure-C. Keep it recorded; do not spend cycles on it.

---

**RESULT -- non-regressive everywhere, better in three places:**

```
  probe                    reason_clean.bin   reason_e2.bin
  probe3 (8 cases)              6/8              8/8    <- both terse-"ok" cases FIXED
  sft_memtest A (bar >=90)      95%              95%
  sft_memtest B (bar 75-100)    80%              90%
  sft_memtest C (artifact)      20%              32%
  eval_critic A grounding       96%              96%
  eval_critic B multi-step      3/4              4/4
```

NO MEMORIZATION (A/B high + C low = generalizing) -- the main risk of 2 epochs on 1798 examples did
not materialize. LIVE 5-step run with REAL 23-line `git status`: read-fail -> write -> run -> read ->
`<done>`, COMPLETE, exact filename. That is the exact case that last session stopped early on AND
char-slipped (demo_note -> demonstra_note); both gone.

**I PREDICTED THIS WOULD FAIL AND I WAS WRONG.** I called "more steps won't fix cases 2 and 7" -- it
did fix them. The 41 terse-word -> CONTINUE examples were already in the corpus, just outnumbered
74-to-41 by unanimous EMPTY -> `<done>`, and a 2nd epoch separated `ok` from `""`. **Lever B is NOT
needed for this defect** -- the empty<->terminal decorrelation design stays on file (FINDING 1b) in
case it resurfaces, but do not spend a retrain on it now.

**COST/RATE for future planning:** stage 1 86 min + stage 2 81 min = ~2.8 h at ~5.2 steps/min, plus
~50 min for the full validation. The old "350 steps ~ 45 min" note is WRONG.

> **THE RUN (recipe that produced it):** 1798 examples @ batch 4 => ~450 steps/epoch, so the old
> 350-step run was only **0.78 epoch** -- that, not a stubborn optimum, is why tail loss sat ~0.5.
> Two DETACHED 450-step stages: stage 1 `--resume model/smollm/model_v2.bin ... --lr 3e-3 --warmup 30
> --min-lr 1e-4 --clip 1.0 --out model/smollm/reason_e2_s1.bin`; stage 2 `--resume
> model/smollm/reason_e2_s1.bin ... --lr 1e-3 --warmup 10 --out model/smollm/reason_e2.bin` (lower LR
> ON PURPOSE -- do NOT re-warm to 3e-3 on already-fine-tuned weights). Staged != continuous: stage 2
> got fresh adapters, fresh Adam state, restarted LR schedule. RUN LONG JOBS DETACHED via
> `Start-Process` -- a harness-child run was killed at 90 min and lost everything.

**NEXT LEVER (ask the user):** (STaR-volume) accumulate traces over real use -- the harness is built
and verified but only 9 rows exist; it needs TENS of runs before a retrain is worth the 2.8 h. This
is the cheapest way to keep improving and it now runs itself as a by-product of normal use (just set
AGENT_TRACE). (b) the ~5% char-slip -- the TOP remaining accuracy defect and it has a
diagnosed SHAPE: it does not corrupt randomly, it SNAPS a novel token toward a real command it knows
(`ldcz`->`ls`, `r4n4`->`rm4n4`, `ctngh4`->`cngh4`). That is prior-over-copy, so a copy-focused aux
objective or contrastive novel-token data is the aimed fix, not more steps. (c) persistent inference
server / int8 so best-of-N is practical (~5x model-reload cost now). (d) periodic checkpointing in
`llama_train_sft` -- STILL MISSING, cost us 90 min today (see FINDING 2). (e) long-horizon PRM /
multi-agent debate.

**FINDING 1 -- the recorded "probe 7/7" WAS STALE; `reason_clean.bin` measured 6/8** (the current
model is now reason_e2.bin at 8/8 -- this finding is about the BASELINE). Re-ran the 7-case
next-action probe (+1 new verbose case) on `reason_clean.bin`: **6/8**. Both failures are premature
`<done>` after a TERSE `OBSERVATION: ok` from run_command. The verbose case PASSES, so 25d1113's fix
is real and holds. It went unnoticed because 25d1113 was validated on `sft_memtest` + `eval_critic`
B4 only -- the 7-case probe was never re-run. The "7/7" in the notes belongs to the *6895f6f* model,
NOT to `reason_clean.bin`. **RULE: re-run the FULL probe set after EVERY retrain -- a fix validated
only on the case it targets hides the trade it made.**

**FINDING 1b -- my first diagnosis of that regression was WRONG; measured and corrected.** I guessed
"terse `ok` is only ~25% of observations so it reads as end-of-chain". Measured the real corpus:

```
  EMPTY/rm         n= 74   next-TOOL   0 (  0.0%)   <done>  74 (100.0%)
  EMPTY/other      n=  9   next-TOOL   8 ( 88.9%)   <done>   1 ( 11.1%)
  terse-word/other n= 41   next-TOOL  39 ( 95.1%)   <done>   2 (  4.9%)
  verbose/other    n=119   next-TOOL  75 ( 63.0%)   <done>  44 ( 37.0%)
```

Terse is **46.8%** of run_command observations (the `_RUN_OUT` FALLBACK also returns terse strings
for verbs absent from the table), and after a terse WORD the corpus CONTINUES **95.1%** of the time
-- the opposite of the guess. THE REAL CAUSE: three templates hardcode `""` as the observation of a
TERMINAL `rm` (`make_agent_traj.py:176`, `:196`, `:232`), so EMPTY->`<done>` is a PERFECT predictor
over 74 examples while mid-chain empties are rare (9). The model over-generalized "short/empty =>
finish" onto `ok`, against only 41 counterexamples. **Verbosity is confounded with POSITION.**
=> LEVER B IS NOT A GLOBAL REBALANCE: route terminal `rm` through `run_obs` so it sometimes emits
`removed 'f1'`, and add mid-chain terse-word steps to grow the 41. Then verbosity is independent of
position and "done" must come from the GOAL, not from output length.

**FINDING 2 -- a 90-min run was killed and produced NOTHING.** First attempt (single continuous 900
steps, launched as a harness-managed background child) was terminated externally at ~10:15; no
checkpoint written, `train_e2.log` still 0 bytes. Cause of the kill unknown; cause of the TOTAL loss
was two gaps in `llama_train_sft` (`train.c`): no `fflush` on the progress printf (entire loss curve
buffered until exit), and -- unlike the GPT path at `train.c:386` -- **no periodic checkpointing at
all**. FIXED (tracked): `fflush(stdout)` after the progress printf in BOTH `train_sft` and
`llama_train_sft`; no numerics touched, rebuild clean, `Output validation: PASSED`. Did NOT change
the `steps/10` report cadence (4 call sites across all training paths) for a monitoring convenience.
STILL OPEN -- periodic LoRA checkpointing is a real design call: merge into a COPY to save (doubles
the ~651MB working set) vs merge-save-unmerge (float drift every checkpoint, kills bit-exactness).
Do it before any future long run. Operational lesson: run long jobs DETACHED, not as tool children.

**PREDICTION -- RESOLVED: I WAS WRONG.** Original call while stage 1 ran: "2 epochs will lower the
loss but will NOT fix probe cases 2 and 7". Both cases PASS on reason_e2.bin. It rested on the
mechanism FINDING 1b refuted; I withdrew confidence once the measurement landed, which was right,
but the call itself was wrong. Why it was wrong: the corpus DID contain 41 terse-word -> CONTINUE
examples, merely outnumbered 74-to-41 by unanimous EMPTY -> `<done>`, and a 2nd epoch was enough to
separate `ok` from `""`. **Transferable lesson: "the data doesn't cover it" and "the data covers it
but is outvoted" look identical from the failure alone -- COUNT THE CORPUS before predicting which,
and prefer the cheap experiment over the confident diagnosis.**

**RE-MEASURE -- ✅ DONE for reason_e2.bin (all four, no shortcuts), and now AUTOMATED.** The probe
set is `probe3.py` (8 cases) | `sft_memtest` (A >=90%, B 75-100%) | `eval_critic` (A, B1-B4) | LIVE
agent run. Both probes are now TRACKED (`tools/probe3.py`, `tools/remeasure.py` -- probe3 was rescued
from a session scratchpad where it had nearly been lost). `remeasure.py` runs all three in one
command and prints `?` for anything unparsed, so FINDING 1 (a skipped probe reading as a pass)
cannot recur silently.

**Verbose/multi-line observation training DONE (2026-07-26) -- the real capability win.** Problem: the
generator trained every run_command obs as bare "ok", so a VERBOSE real obs (multi-line `git status`)
was OOD and the model stopped early with <done> (lever #3-b live; lever #4 case B4). Fix (one tracked
edit, `tools/make_agent_traj.py`): new `run_obs()` + `_RUN_OUT` table emit realistic per-command-family
output (git/cargo/go/make/npm/..., 25% terse), avoiding the critic's error hints so success != failure.
VERIFY 1856/1856 (0 rejected); clean corpus 1754->1798; retrain -> reason_clean.bin. CONFIRMED: eval_critic
B4 (read after verbose git status) MISS-for-all -> OK-for-all; and the EXACT lever-#3-b live scenario that
stopped early now COMPLETES (read-fail->write->git status[verbose]->read->done, no char-slip). No regression
(sft_memtest A 95% / B 80%, both above bar; B down from 100% peak but within the 75-100% retrain band).
Full detail: notes.md "2026-07-26 -- verbose/multi-line observations".

~~**NEXT LEVER:** (a) more steps / 2nd epoch ...~~ **SUPERSEDED -- (a) is DONE (commit 4bea3b9, probe
6/8 -> 8/8). The live list is the "NEXT LEVER" block at the top of the 2026-07-26 (b) section.**
Env (unchanged, still current): zig `D:\env\zig\zig.exe`, python
`D:\env\python\python-3.9.10.amd64\python.exe`. 2-space, C-only-flat-root. Rebuild+smoke before any
commit (Output validation: PASSED). Build:
`$c=(Get-ChildItem *.c|%{$_.Name}); D:\env\zig\zig.exe cc -std=c11 -O2 -Wall -o transformer.exe @c -lm`

--- earlier this session (lever #4 proposer-critic) ---

**Lever #4 (proposer-critic loop) BUILT this session -- correct + honest, but NO accuracy win at
135M.** New tools (no C): `agent_critic.py` (critic: grounding via clean_sft.verify + anti-repeat of
SUCCESSFUL actions + retry-guard for just-FAILED actions + done-guard/self-consistency + greedy
fallback), best-of-N in `agent_local.py` (env `AGENT_BEST_OF>1`, `AGENT_TEMP`; hfgen already takes a
seed so varied seeds give diverse candidates), `eval_critic.py`, and guarded `clean_sft.main()` so
verify() imports. Unit tests all pass. MEASURED (eval_critic M=24): best-of@0.7 == single@0.7 ==
greedy@0.1 (A 96%, B 3/4) -- NO gain, because this small model's errors are SYSTEMATIC/low-variance
(diagnosed: for the premature-done case the proposer never emits the correct action in any sample),
which best-of-N can't fix. KEY LIVE FINDING: the first best-of-5 live run REGRESSED into an 8x
re-read LOOP (failed actions were exempt from anti-repeat) -> fixed with the retry-guard; second live
run correct (read-fail->write "hello world"->read->done). COST: best-of-N reloads the 651MB model per
candidate (~5x slower/step). Verdict: validated harness with real safety guarantees, not a quality
jump. Full detail: notes.md "2026-07-25 (c) -- lever #4". Commit pending. Files: agent_critic.py,
eval_critic.py (new), agent_local.py, clean_sft.py (mod).

**BIGGEST NEXT LEVER (unchanged, now doubly motivated):** train on VERBOSE / multi-line OBSERVATIONS
(real command output, not just "ok") -- fixes BOTH the lever-#3 early-<done> AND lever-#4 case B4 at
the source (the proposer would then know to continue after a verbose obs). Also: more steps
(~500)/2nd epoch; reduce the ~5% char-slip; (optional) a persistent-process inference server or an
int8 fine-tune so best-of-N is fast enough to be practical. Ask the user which.

--- earlier this session (lever #3, both passes) ---

Lever #3 done TWICE this session. **Pass 1 (3-step + error recovery):** three 3-step templates in
`make_agent_traj.py` build_tasks; retrain -> probe 4/4, sft_memtest A95/B75, live 3-step error-recovery
ran end to end (commit 59b4252). **Pass 2 (4-step + real rust/go commands):** three 4-step templates
(write-write-run-read; read-fail->write->run->read; write-read-run-delete) + folded 91 REAL rust/go
commands into `data/so_commands.txt` (344->435, TRACKED; crawled SO go/cargo/rust tags, filtered miner
junk, merged). clean_sft VERIFY 1818/1818 (0 rejected); clean corpus 1685->**1754**, depth to 5-step.
Retrain (~45 min) -> `model/smollm/reason_clean.bin`, sft-loss 3.54->~0.58 (higher: harder/longer data,
<1 epoch). RE-MEASURED: **next-action probe 7/7** (3- AND 4-step, novel args + new cmds emit right);
**sft_memtest A 95% / B 100%** (B rose 75->100 from the richer cmd pool; no regression; C 20% = the
`<think>`-strip harness artifact). LIVE 4-step error-recovery+run: read-fail->write->ran git status
(real output)->`<done>` -- but TWO honest limits: a filename char-slip (demo_note->demonstra_note) and
it stopped ONE STEP EARLY (verbose git output != clean "ok" it trained on). Full detail + commands:
notes.md "2026-07-25 (b) -- lever #3-further". Tracked edits: make_agent_traj.py, so_commands.txt (data/*
else git-ignored).

**BIGGEST NEXT LEVER (from what the live run exposed):** train on VERBOSE / multi-line OBSERVATIONS
(real command output, not just "ok") so long live observations don't trigger an early `<done>`. Also:
more steps (~500) / 2nd epoch to firm up the longer chains (loss 0.58 => undertrained), and reduce the
persistent 5% char-slip. THEN consider #4 (long-horizon: PRMs / multi-agent debate proposer-critic --
research-y, mostly beyond 135M/CPU but a toy loop is feasible). Ask the user which to take.

Env: zig D:\env\zig\zig.exe, python D:\env\python\python-3.9.10.amd64\python.exe. Keep 2-space,
C-only-flat-root rules. Rebuild+smoke before any commit (Output validation: PASSED).

Candidate levers ledger (reference):
  1. [x] DONE (4a61ffb) -- grad_clip defaults to 1.0 in train.c (stability hardening).
  2. [x] DONE (6f68ca6) -- variable-length SFT batching (LLaMA path): BIT-EXACT + ~4.7x faster.
  3. [x] DONE (2026-07-25, commit 59b4252 + this pass) -- grow/vary data: 3-step chains + error
     recovery, THEN 4-step chains + 91 real rust/go commands. All green (probe 7/7, no regression).
  4. Long-horizon roadmap item (PRMs, multi-agent debate, ...) -- candidate after the obs-robustness fix.
  5. [x] DONE (2026-07-26, commit 4bea3b9) -- lever (a) 2 EPOCHS (2x450 staged): probe 6/8 -> 8/8,
     memtest B 80% -> 90%, critic B 3/4 -> 4/4, no memorization, live 5-step run green.
     => `model/smollm/reason_e2.bin` PROMOTED as the current model.
  6. [~] BUILT, NOT YET EXERCISED (2026-07-26 c) -- STaR self-training harness (`self_train.py` +
     `AGENT_TRACE`). All 5 reject paths verified; 9 rows harvested from 2 real runs. Needs volume
     (tens of runs) before a retrain is worth it. Prereq for the "learns as you prompt" version is
     lever (c) resident model.

Rebuild clean corpus + retrain (the validated recipe, note --clip 1.0):
`python tools/make_agent_traj.py --out data/traj_reason.txt --n 300 --wiki data/wiki --commands
data/so_commands.txt --reason` then `python tools/clean_sft.py --shuffle --balance --report
data/clean_report.txt --out data/clean_reason.txt data/traj_reason.txt data/qa.txt data/toolcalls.txt`
then TWO 450-step stages (2 epochs -- **NOT the old single `--steps 350`, which is only 0.78 epoch
and measurably undertrained; see 2026-07-26 (b)**), each DETACHED via `Start-Process`:
`.\transformer.exe llama-train data/clean_reason.txt --sft --hf-tok model/smollm/tok.hf --resume
model/smollm/model_v2.bin --lora-r 16 --lora-alpha 32 --batch 4 --steps 450 --lr 3e-3 --warmup 30
--min-lr 1e-4 --clip 1.0 --out model/smollm/reason_e2_s1.bin` then the same with
`--resume model/smollm/reason_e2_s1.bin --steps 450 --lr 1e-3 --warmup 10 --out
model/smollm/reason_e2.bin`. Budget ~2.8 h total at ~5.2 steps/min.

VALIDATE (one command, runs the FULL set -- built precisely so a probe cannot be skipped again):
`python tools/remeasure.py model/smollm/reason_e2.bin` (add `--best-of 1 --m 8` for a quick pass).
It prints probe3 + sft_memtest + eval_critic vs the recorded baseline and shows `?` for anything it
could not parse, so a missing number never reads as a pass. GOTCHA baked in: `sft_memtest` execs the
engine argv VERBATIM, so it needs an ABSOLUTE path (probe3/eval_critic abspath internally and hid
this for one run). Individual probe: `python tools/probe3.py model/smollm/reason_e2.bin`.

Live agent (space-free paths, from repo root):
`.\transformer.exe agent --cmd "D:\env\python\python-3.9.10.amd64\python.exe tools/agent_local.py
model/smollm/reason_e2.bin model/smollm/tok.hf" --goal "..."`.
**STDIN TRAP:** `run_command` needs interactive approval (`approval.c:11`); with no stdin `fgets`
returns EOF => DENIED, so a detached/piped live run silently skips every command step. Pipe
approvals: `"y`ny`ny`ny`n" | .\transformer.exe agent ...`. (The model handles a denial correctly --
it treats "denied by user" as an observation and continues -- but the verbose-obs path goes untested.)
