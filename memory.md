# memory.md — portable memory snapshot

Portable memory so a Claude session on any machine can get up to speed fast. This file is
**committed to the repo** (tracked in git), so it travels with `git clone`/`pull` across
machines. On a new PC, tell Claude: **"read memory.md"**. Deep, always-current project detail
lives in `notes.md` (esp. its "▶ CONTINUATION GUIDE").

> **Sync rule (user request):** this file mirrors Claude's memory — whenever the assistant
> updates its memory, it must also update `memory.md` here.

_Last updated: 2026-07-21._

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
5. Phase E images — **E1+E2+E3 DONE** (`image.c` PPM load, `patch.c` ViT patch embedding,
   `model.c` `gpt_forward_multimodal` feeds [image patches]+[text] through the GPT;
   `tools/img_to_ppm.py`). **E4 (image-text training loop) remains** — RECOMMENDED NEXT.
   6. Phase F — video. (Image is forward-plumbing only; no multimodal training yet.)

Model folders: each model in `model/<name>/` (`model.bin` + `tok.bin`); `model/**/*.bin`
git-ignored, READMEs kept. `model/gpt2/` holds real GPT-2 (token-ID mode via tiktoken).
