# webchat — browser testing frontend (Go + WebSocket)

A small self-contained Go web server that lets you exercise the pure-C
`transformer` engine from a browser: **text chat** across every model it detects,
plus **image / audio generation** and **image / audio / video "understanding"**.
It is a *testing* surface — quality is toy-scale by design (see the honest notes
below); the point is to watch the engine's pipelines run end-to-end on real
input.

Go standard library only — no external modules. The server bridges each request
to the CLI as a subprocess (run with cwd = repo root), so it works exactly like
running `transformer` by hand.

## Build & run

Build the engine first (from the repo root), then start the server:

```sh
# 1. build the engine (repo root)
make                                  # -> ./transformer   (or transformer.exe on Windows)

# 2. run the server (from this folder)
cd tools/webchat
go run . -addr :8080 -root ../..      # open http://localhost:8080
```

Flags: `-addr` (listen address, default `:8080`), `-root` (repo root that holds
`transformer` + `model/`, default `../..`), `-exe` (override the binary path),
`-python` (interpreter for the tokenizer bridges, default `python`).

On Windows without `make`, build the engine with `zig` and run the server the
same way (`go run .` picks up `transformer.exe` automatically).

## Panels

| Panel | What it does | Needs |
| --- | --- | --- |
| **Text** | Chat with any detected model (switcher). | Python only for GPT-2 / SmolLM2 / Qwen (below) |
| **Image** | *Reconstruct*: downscale ≤16×16 → tokenize → autoregressively rebuild. *Caption*: teach a caption, reproduce it. | — |
| **Audio** | *Reconstruct*: ≤256-sample WAV → µ-law tokens → rebuild (browser-playable). *Caption*: teach + reproduce. | — |
| **Video** | Extract frames → *Caption*: teach + reproduce; shows a frames GIF. | **ffmpeg** on PATH |

## Model switcher

The server scans `model/<name>/`, reads the 4-byte checkpoint magic, and routes:

- `GPTC` + `tok.bin` → **pure C** text mode (`transformer generate --name <n>`).
- `GPTC`, no `tok.bin` → **GPT-2** via `tools/gpt2_chat.py` (needs `tiktoken`).
- `LLMA` / `LLMQ` → **LLaMA-family** via `tools/llama_chat.py` (needs
  `tokenizers`, plus the model weights present on disk). The HuggingFace
  tokenizer repo is looked up per known model, overridable with a
  `model/<name>/hf_tokenizer.txt` file.

## Honest limitations

- **Toy-scale quality.** Image/audio generation *reconstruct* (overfit) one tiny
  input. The caption path is **teach-then-reproduce**: you provide a caption, the
  engine trains a tiny model to map *this* media to *that* text and reproduces it.
  The multimodal front-end is untrained and there is no trained caption
  checkpoint, so this proves the pipeline runs on real media — it is *not* a
  zero-shot captioner.
- **Slow on CPU.** Attention is O(sequence²); media panels cap the input size and
  default to few steps so each run is ~10–30 s. Larger inputs / more steps get
  much slower.
- **Sampling is fixed** in the engine (temp 0.8 / top-k 40 / top-p 0.95); only the
  LLaMA path's "max new tokens" is adjustable from the UI.
