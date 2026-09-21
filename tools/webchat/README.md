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

## Build standalone exes

Build **both** executables at once — the C engine and the self-contained server
(its UI is embedded, so `webchat.exe` needs no extra files):

```sh
make webchat                          # POSIX: -> ./transformer + tools/webchat/webchat
powershell -File tools\webchat\build.ps1   # Windows (uses zig): -> transformer.exe + webchat.exe
```

Then just run the server — it **auto-detects** the repo root and the engine
(whether started from the repo root or from `tools/webchat/`):

```sh
./tools/webchat/webchat               # open http://127.0.0.1:8080
```

Both binaries are git-ignored (`*.exe` / `tools/webchat/webchat`).

## Run without building

```sh
make                                  # build the engine (repo root)
cd tools/webchat && go run .          # start the server (auto-detects root)
```

Flags (all optional): `-addr` (listen address, default `127.0.0.1:8080` -- loopback only;
pass e.g. `-addr :8080` to expose it on the network, which lets anyone who can reach the port
drive the engine), `-root` (repo
root holding `transformer` + `model/`; auto-detected if omitted), `-exe`
(override the engine path — searched next to the server, then in root),
`-python` (interpreter for the tokenizer bridges, default `python`).

Access control: the WebSocket only accepts connections whose `Origin` matches the server's own
host (so another web page open in your browser cannot drive it), and when bound to loopback,
requests whose `Host` is not a loopback name are refused (DNS-rebinding guard).

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

## Real text-to-image (HuggingFace diffusers)

The engine's own text→image (`t2i`) is toy-scale. For **real** quality, run a
diffusion model *alongside* the engine (diffusion is a different architecture —
it can't load into the transformer). The server has a hook:

Text→image is **C-native by default**: image requests use the from-scratch `t2i`
model (VQ-VAE + autoregressive transformer, all in the engine — no Python). This
matches the project's pure-C aim; quality is toy-scale by design.

A real HF diffusion model (Stable Diffusion / SD-Turbo) can only run *beside* the
engine in Python (diffusion is a different architecture that can't load into the
transformer), so it is strictly **opt-in** and off by default:

```sh
pip install diffusers transformers accelerate torch      # ~2-3 GB (torch is large)
.\tools\webchat\webchat.exe -imagegen "python tools/imagegen_diffusers.py"
```

Routing priority for image requests: **`-imagegen` hook (if opted in) → trained
`t2i` model → honest note**. With no `-imagegen`, everything stays pure C.

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
