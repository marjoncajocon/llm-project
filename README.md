# Pure-C LLM Engine — GPT-2-style Transformer, Agent & Multimodal (C11)

> **This is a big, actively-developed project — not a one-off demo.** It is a from-scratch,
> dependency-free machine-learning stack written in portable C11, built up over many committed
> steps and still growing along a real roadmap (see `notes.md`). "Minimal" here means *minimal
> dependencies* (no libraries, no BLAS, no framework — just the C standard library and `-lm`),
> **not** minimal scope. The scope is deliberately broad.

**What's built (all from scratch, all in pure C):**
- **Training** — full backpropagation through attention, feed-forward, LayerNorm and embeddings;
  Adam; cross-entropy; **numerically gradient-checked**. Train on your own text (`transformer
  train`) with configurable model size, train/val split, checkpointing and resume.
- **Inference** — temperature / top-k / top-p sampling, a **bit-exact KV cache**, and a
  token/text generation CLI.
- **Real GPT-2** — loads actual Hugging Face GPT-2 weights (124M–1.5B) and generates coherent
  text (`tools/gpt2_setup.py` → GPTC checkpoint).
- **A tool-using agent** — a coding-agent loop (parse tool calls → approval gate → execute →
  observe) with a pluggable backend, including a real-LLM backend (`tools/agent_llm.py`).
- **Multimodal** — image / video / audio *understanding* (patch/frame/waveform embeddings, trained
  via a shared multimodal path) **and** discrete-token media *generation* (image/audio/video),
  including text→media and media→text.
- **Portable** — one flat, readable module per concern; builds and runs on Linux, macOS, Windows,
  and **WebAssembly**.

**Where it's honest about scale:** quality is capped by what a from-scratch model on a CPU can be
— outputs are toy-scale, and the media codecs are crude. The deliverable is a *correct, complete,
extensible engine and pipeline*, not production fidelity. The roadmap (`notes.md`) lays out the
path to bigger, genuinely useful models: **SIMD + int8/int4 quantization**, and **modern
LLaMA-family architecture** support (RMSNorm, RoPE, SwiGLU, GQA + a SentencePiece tokenizer) so it
can load small modern models — turning this into a from-scratch mini-`llama.cpp`.

Every module (tensors, matrix math, softmax, GELU, LayerNorm, causal multi-head attention,
feed-forward, embeddings, sampling, checkpointing, BPE tokenizer) is still small enough to read in
one sitting — the codebase stays approachable even as the project grows.

## Why I'm building this

I'm building this because I don't want to be left behind as AI advances. I want to understand it
from the ground up — not just use it, but know how it actually works — so I can help make AI
better and better in the future. My goal is to become one of the people who *raise* AI: who help
it grow, improve, and carry it forward. This project is where I learn that craft, one honest,
from-scratch step at a time. — Marjon

Every stage here is built to be *understood*, not just run — because you can't help raise
something you don't understand.

## Project layout

Inspired by the Lua source tree: **every C source file lives flat in the project root —
there are no subfolders for code.** Each module is one concern, declared in a `.h` and
defined in the matching `.c`.

| File                              | Responsibility                                                |
| --------------------------------- | ------------------------------------------------------------- |
| `tensor.h` / `tensor.c`           | `Tensor` struct + aligned memory management                   |
| `matrix.h` / `matrix.c`           | `matmul`, `matadd`, `matscale`                                |
| `activation.h` / `activation.c`   | `softmax_rowwise`, `gelu`                                      |
| `layernorm.h` / `layernorm.c`     | `layernorm`                                                    |
| `rmsnorm.h` / `rmsnorm.c`         | `rmsnorm` (LLaMA-family; RMS-only norm, no mean/bias) + backward |
| `feedforward.h` / `feedforward.c` | Position-wise feed-forward MLP: Linear → GELU → Linear        |
| `attention.h` / `attention.c`     | `multihead_attention_forward` (causal), `transformer_block_forward` |
| `embedding.h` / `embedding.c`     | Token + positional embedding lookup                           |
| `sampler.h` / `sampler.c`         | Token sampling: greedy argmax + temperature / top-k / top-p (seeded RNG) |
| `model.h` / `model.c`             | `GPT` model: embeddings, stacked blocks, output head, training, KV-cache decode |
| `checkpoint.h` / `checkpoint.c`   | Save/load model weights (`gpt_save` / `gpt_load`, GPTC format) |
| `loss.h` / `loss.c`               | Softmax cross-entropy (forward + backward)                    |
| `optim.h` / `optim.c`             | Adam optimizer                                                |
| `tokenizer.h` / `tokenizer.c`     | Byte-level BPE: encode/decode + merge-learning (train) + save/load |
| `toolcall.h` / `toolcall.c`       | Parse `<tool …>`/`<done>` blocks from model output (agent)    |
| `toolexec.h` / `toolexec.c`       | Execute a tool call: write_file / read_file / run_command     |
| `approval.h` / `approval.c`       | Approval gate for tool actions (ask y/N, or auto)             |
| `agent.h` / `agent.c`             | Agent loop: parse → approve → execute → observe               |
| `image.h` / `image.c`             | Minimal image loading (binary PPM) — Phase E (multimodal)     |
| `patch.h` / `patch.c`             | ViT-style patch embedding + backward (trainable image front-end) |
| `video.h` / `video.c`             | Video = frames → per-frame patches + temporal encoding (Phase F) |
| `audio.h` / `audio.c`             | Audio = WAV → windowed frames → embeddings + temporal encoding (Phase F) |
| `imagegen.h` / `imagegen.c`       | Image ↔ tokens (color-cube) for autoregressive image **generation** (Phase G) |
| `audiogen.h` / `audiogen.c`       | Audio ↔ tokens (µ-law) for autoregressive audio **generation** (Phase G) |
| `corpus.h` / `corpus.c`           | Read training text from a file or a directory tree (recursive `.txt`) |
| `train.h` / `train.c`             | Train a GPT on a text file/dir (tokenizer + model + Adam loop) |
| `selftest.h` / `selftest.c`       | Built-in smoke test (`run_selftest`): the default `transformer` run           |
| `main.c`                          | Entry point + subcommands (`train`, `generate`, `agent`, `imggen`, `audgen`) |
| `data/`                           | Your training data (git-ignored; keeps a sample)             |
| `model/`                          | One folder per model (`model/<name>/`); weights git-ignored, READMEs kept |
| `tools/`                          | Non-C helpers: GPT-2 setup/tokenizer, converters, `crawl_wikipedia.py` |
| `Makefile`                        | Build rules                                                   |
| `tools/`                          | Helper scripts in **other** languages (Python/Go/…) — not C   |

## Build & run

```sh
make            # compiles all modules into ./transformer
./transformer   # runs the demo (tokenizer + model + checkpoint round-trips)
make clean      # removes object files and the binary
```

Or compile directly, without make:

```sh
gcc -std=c11 -O3 -march=native -o transformer *.c -lm
```

`clang` works as a drop-in replacement for `gcc`. On a machine without gcc/clang, the
project also builds with `zig cc -std=c11 -O2 -Wall -o transformer *.c -lm`.

## Supported platforms

Portable C11 with the platform-specific bits guarded (`_WIN32` / POSIX / WASM). Verified by
cross-compiling every source with `zig cc -target <triple>` (all `-Wall`-clean):

| Platform | Target triple | Status |
| --- | --- | --- |
| Linux x86-64 (glibc) | `x86_64-linux-gnu` | builds |
| Linux ARM64 (glibc) | `aarch64-linux-gnu` | builds |
| Linux x86-64 (musl) | `x86_64-linux-musl` | builds |
| Windows x86-64 | `x86_64-windows-gnu` | builds + **runs** (smoke test PASSED) |
| Windows ARM64 | `aarch64-windows-gnu` | builds |
| macOS Intel | `x86_64-macos` | builds |
| macOS Apple Silicon | `aarch64-macos` | builds |
| **WebAssembly (WASI)** | `wasm32-wasi` | builds + **runs** (via Node WASI, PASSED) |

### Threads

`matmul` and `matmul_backward` run on a **persistent thread pool** (auto-enabled for large
matrices; small ones stay single-threaded and bit-exact). It uses all CPU cores by default;
override with the `TF_THREADS` env var (e.g. `TF_THREADS=4`). Measured ~2.7× on 8 threads for
a d_model-256 / 4-layer model. WebAssembly runs single-threaded (no thread model).

### Cross-compiling

Build **all** targets at once into `build/` (needs `zig` + a POSIX shell):

```sh
make cross          # -> build/transformer-<target>[.exe|.wasm] for every platform
```

Or cross-compile one target by hand (no OS-specific toolchain needed — `zig` bundles them):

```sh
zig cc -target x86_64-linux-gnu  -std=c11 -O2 -o transformer      *.c -lm   # Linux
zig cc -target aarch64-macos     -std=c11 -O2 -o transformer      *.c       # macOS (libm in libSystem)
zig cc -target wasm32-wasi       -std=c11 -O2 -o transformer.wasm *.c       # WebAssembly
```

Run the WebAssembly build with any WASI runtime (grant filesystem access for `train`):

```sh
wasmtime --dir=. transformer.wasm                       # or wasmer
node --experimental-wasi-unstable-preview1 run.js transformer.wasm   # Node's built-in WASI
```

> WASM note: WebAssembly has no process model, so the agent's `run_command` tool is disabled
> under WASM (it returns "not supported"); `write_file`/`read_file`, training, and generation
> all work.

---

## CLI guide

### `transformer` (the demo binary)

Runs the full pipeline and prints a report. It takes no arguments today; edit the config
constants at the top of `run_selftest()` in `selftest.c` to change vocab size, dimensions, heads,
layers, prompt, or number of generated tokens.

```sh
$ ./transformer
=== Minimal GPT-2 Style Transformer Demo ===

[1] Byte-level BPE tokenizer
    text     : "hello hello" (11 bytes)
    encoded  : 5 tokens -> 258 111 32 258 111
    decoded  : "hello hello"
    round-trip: OK | compressed: yes

[2] GPT model
    config: vocab=16 d_model=8 heads=2 layers=2 d_ff=32
    generate: 1 5 9 -> 1 5 9 11 6 11 14 14
    logits  : no NaN/Inf

[3] Checkpoint save/load
    saved 'model.bin': OK | reloaded logits identical: yes

Output validation: PASSED (no NaN/Inf)
```

Exit code is `0` when all checks pass, `1` otherwise — handy in scripts:

```sh
./transformer && echo "smoke test ok"
```

### `transformer train <path> [flags]`

Train the model on **your own text**. `<path>` may be a single file **or a directory** —
a directory is read **recursively**, concatenating every `.txt` under it (including
subfolders like `data/wiki/`). The model size is fully configurable via flags:

```sh
transformer train data/wiki                       # defaults (d_model 128, 3 layers, ...)
transformer train data/wiki --steps 20000 --out ai.bin --tok ai.tok

# Scale the model up (bigger = slower but better):
transformer train data/wiki --d-model 256 --layers 6 --heads 8 --ctx 128 --steps 30000
```

| Flag | Meaning (default) | | Flag | Meaning (default) |
| --- | --- | --- | --- | --- |
| `--out F`   | model out (`model.bin`) | | `--d-model N` | model dim (128) |
| `--tok F`   | tokenizer out (`tok.bin`) | | `--layers N` | transformer blocks (3) |
| `--merges N`| BPE merges (300) | | `--heads N`  | attention heads (4) |
| `--steps N` | Adam steps (1000) | | `--d-ff N`   | FFN dim (4·d_model) |
| `--lr F`    | learning rate (3e-4) | | `--ctx N`    | context window (64) |
| `--seed N`  | RNG seed (1234) | | `--val-frac F` | held-out fraction (0.1) |
| `--checkpoint-every N` | save every N steps | | `--resume F` | continue a saved model |

`--d-model` must be divisible by `--heads`. Training prints **train + held-out `val`** loss
(so you can tell learning from memorizing); `--checkpoint-every` saves periodically, and
`--resume model.bin` continues training a saved model across sessions (keeps its config +
tokenizer). CPU-only, so bigger/cleaner text and **many more `steps`** help. Each trained
`model.bin`+`tok.bin` pair is a self-contained model you can keep for a specific role.

```sh
# train, checkpointing every 500 steps, holding out 10% for validation:
transformer train data/wiki --out ml.bin --tok ml.tok --steps 5000 --checkpoint-every 500
# later, continue where it left off:
transformer train data/wiki --resume ml.bin --out ml.bin --tok ml.tok --steps 5000
```

**Build a real corpus by crawling** (see `tools/crawl_wikipedia.py`):

```sh
python tools/crawl_wikipedia.py "Machine learning" --max-pages 50   # -> data/wiki/*.txt
transformer train data/wiki --out ml.bin --tok ml.tok --steps 10000
echo "machine learning is" | transformer generate ml.bin ml.tok
```

The crawler fetches plain-text Wikipedia extracts and follows article links (breadth-first)
so the text stays topically *connected*. Then `train` ingests the whole `data/wiki/` folder.

### Instruction tuning (SFT) on your own data

This is **stage 3 of the recipe** (see the "5 stages" in `memory.md`) — the step that makes a
model *follow instructions* instead of just continuing text. `data/instruct.txt` is a tiny
example dataset of `Q: <instruction>` / `A: <response>` pairs whose answers are the agent's
tool-call format (`<tool …>` / `<done>`). Train on it, then feed an instruction and watch the
model produce the matching tool call:

```sh
transformer train data/instruct.txt --name instruct --d-model 96 --layers 3 \
  --heads 4 --ctx 96 --merges 48 --steps 4000 --lr 1e-3
echo "Q: read notes.txt"      | transformer generate --name instruct
#   -> A: <tool name="read_file" path="notes.txt"></tool>
echo "Q: the task is finished" | transformer generate --name instruct
#   -> A: <done>Task complete.</done>
```

Verified: a small model learns to route each held-in instruction to its correct response.
(Note: keep `--merges` low so the BPE tokens stay fine-grained enough for prompts to align to
token boundaries.)

**True masked SFT (`--sft`).** The run above trains on the whole token stream (the model
*memorizes* the corpus). Real SFT supervises **only the response** and masks the prompt — that's
the `--sft` mode. The file is one `prompt<TAB>response` example per line (`data/sft.txt`); the
loss is computed on the response tokens only (prompt + padding get target `-1` = ignored), so the
model is trained to *produce* the answer, not reconstruct the prompt:

```sh
transformer train data/sft.txt --sft --name sft --d-model 96 --layers 3 \
  --heads 4 --merges 48 --steps 3000 --lr 1e-3
echo "read notes.txt" | transformer generate --name sft
#   -> <tool name="read_file" path="notes.txt"></tool>     (stops cleanly at EOS)
```

Verified: every instruction routes to its exact tool-call/`<done>` response, and generation
**stops cleanly** at the end. This is the same loss the smoke-test part `[9]` uses, now available
on your own data. **EOS/stop token:** `--sft` reserves one extra vocab slot as an end-of-sequence
token (`EOS = tokenizer.vocab_size`), appends it to every response and supervises predicting it, so
the model learns to *end* its answer; `generate` halts as soon as it samples an id ≥ the
tokenizer's vocab (non-SFT models have no extra id, so they're unaffected — fully backward
compatible).

**Honest scale:** at this size the model *memorizes* the pairs — the value is seeing the SFT
technique work end-to-end on real data via the CLI; generalizing to unseen instructions needs far
more data + a bigger model.

### `transformer generate [model.bin] [tok.bin]`

A generation loop driven by the KV cache (`gpt_decode_step`) + sampler (`sample_token`),
one prompt per stdin line:

```sh
# TEXT in / TEXT out — pass a trained model AND its tokenizer:
echo "the cat" | ./transformer generate model.bin tok.bin
# the cat sat on the mat. the dog sat on the log. ...

# Token IDs — loaded weights, no tokenizer:
echo "1 5 9" | ./transformer generate model.bin
# gen: 1 5 9 [15] [15] [5] ...            (bracketed tokens are generated)

# No args -> random toy model (token IDs in):
echo "1 5 9" | ./transformer generate
```

Passing a tokenizer switches to **text-in/text-out**. Defaults: 40 new tokens, temperature
0.8, top-k 40, top-p 0.95 (edit `run_generate` in `main.c`). Pair with `train` above to
generate from a model trained on your own data.

### `transformer agent [--ask]` (agent loop demo)

Demonstrates the coding-agent loop in pure C: **parse → approve → execute → observe**.
The model emits tool blocks; each approved call runs and its result is fed back:

```
<tool name="write_file" path="a.txt">...body...</tool>
<tool name="read_file" path="a.txt"></tool>
<tool name="run_command" command="ls"></tool>
<done>summary</done>
```

The **model backend is pluggable** (`AgentBackend` interface in `agent.h`):

```sh
transformer agent                                   # built-in scripted demo (auto-approve)
transformer agent --ask                             # approve each tool action interactively
transformer agent --cmd "python tools/agent_mock.py"  # external program IS the model
transformer agent --cmd "python tools/agent_mock.py" --goal "make hello.txt"
```

With `--cmd`, the agent writes the running transcript to a temp file, passes its path to the
program, and uses the program's **stdout** as the model's next response — so you can plug in a
real LLM with a small script (see `tools/agent_mock.py`, which shows the Anthropic/OpenAI call
to drop in). The pieces: `toolcall.c` (parse) · `toolexec.c` (write/read/run) · `approval.c`
(gate) · `agent.c` (loop + scripted/command backends).

> The built-in transformer is far too small to emit valid tool calls, and `model/gpt2` is base
> (not instruction-tuned) — for a genuinely acting agent, point `--cmd` at an instruction-tuned
> model or an LLM API.

### `transformer imggen` / `audgen` (generate media to a file)

Discrete-token media generation (Phase G): quantize the input to tokens, train a small GPT to
model it, then autoregressively decode to a real file. Two modes, auto-selected from the input:

```sh
# Single-file mode -- reconstruct one input (overfits it):
python tools/img_to_ppm.py cat.png cat.ppm 32     # any image -> small PPM (<= ~32x32)
transformer imggen cat.ppm out.ppm 4 500          # -> out.ppm   (levels=4, steps=500)
python tools/audio_to_wav.py clip.mp3 clip.wav    # any audio -> 16-bit mono WAV (short)
transformer audgen clip.wav out.wav 500           # -> out.wav

# Dataset mode -- learn from a FOLDER and GENERATE a novel sample:
#   name the files img_0000.ppm, img_0001.ppm, ...  (all the same size)
transformer imggen imgset/ out.ppm 4 800          # -> a generated image
#   name the files aud_0000.wav, aud_0001.wav, ...  (all the same length)
transformer audgen audset/ out.wav 800            # -> a generated clip
```

If the input path is a single PPM/WAV it is reconstructed (overfit); if it is a folder of
`img_0000.ppm,…` / `aud_0000.wav,…` the model trains on the whole set and **generates** a new
sample. Dataset mode prepends a **BOS** token so the model learns the distribution of first
tokens and can sample unconditionally (temperature 0.9 / top-k 40 / top-p 0.95).

Image uses a **color-cube** codec (`levels³` tokens); audio uses **µ-law** (256). O(n²)
attention, so keep inputs small (≤ ~32×32 image, ≤ ~1024 audio samples). Text-conditioned
generation (text→image/audio/video) and captioning (media→text) are in the default smoke test —
see `notes.md` (Phase G). Real quality needs a trained VQ-VAE/diffusion + far more data.

### `tools/convert_to_gptc.py` (weight importer)

Converts a `.npz` of named weight arrays into the C checkpoint format so `gpt_load()`
can read it. This is the bridge for importing external weights (e.g. a converted GPT-2).

```sh
python tools/convert_to_gptc.py <weights.npz> <model.bin> --n-heads <N>
```

| Argument       | Meaning                                                       |
| -------------- | ------------------------------------------------------------- |
| `weights.npz`  | input archive; array names/shapes must match the model params |
| `model.bin`    | output GPTC checkpoint                                         |
| `--n-heads N`  | number of attention heads (can't be inferred from shapes)     |

Expected `.npz` keys: `wte`, `wpe`, `blocks.{l}.{W_q,W_k,W_v,W_o,gamma1,beta1,gamma2,beta2,ff_W1,ff_b1,ff_W2,ff_b2}`, `ln_f_gamma`, `ln_f_beta`, `lm_head` (see the script's docstring for shapes). Load the result in C with `gpt_load("model.bin", &m)`.

---

## Programmatic guide (using the C API)

### Tokenizer

```c
#include "tokenizer.h"

Tokenizer tok = tokenizer_create_bytelevel();   // 256 byte tokens
int ll = tokenizer_add_merge(&tok, 'l', 'l');   // learn merges -> new token ids
tokenizer_add_merge(&tok, 'e', ll);

int ids[64];
int n = tokenizer_encode(&tok, "hello", ids, 64);

char out[128];
tokenizer_decode(&tok, ids, n, out, sizeof(out)); // "hello" (byte-exact round-trip)

tokenizer_free(&tok);
```

### Model: build, run, generate

```c
#include "model.h"

// vocab, d_model, n_heads, n_layers, d_ff, max_seq_len
GPT m = gpt_create(16, 8, 2, 2, 32, 16);

// Forward pass -> logits (seq_len, vocab_size)
int ids[3] = {1, 5, 9};
int shape[2] = {3, m.vocab_size};
Tensor logits = tensor_create(shape, 2);
gpt_forward(&m, ids, 3, &logits);

// Autoregressive greedy generation (appends in place; buffer must fit prompt+new)
int seq[16] = {1, 5, 9};
gpt_generate(&m, seq, /*prompt_len=*/3, /*n_new=*/5);

tensor_free(&logits);
gpt_free(&m);
```

### Checkpoints

```c
#include "checkpoint.h"

gpt_save("model.bin", &m);        // persist weights

GPT loaded;
gpt_load("model.bin", &loaded);   // rebuilds config + weights; free with gpt_free
```

### Training (backprop + Adam)

Every forward op has a matching `*_backward` that **accumulates** into the `grad` array of
its inputs/parameters (call `tensor_zero_grad` before each step). Cross-entropy is in
`loss.c`, the optimizer in `optim.c`. A training step looks like:

```c
#include "loss.h"
#include "optim.h"

Tensor *params[] = {&W1, &b1, &W2, &b2};
Adam opt = adam_create(params, 4, /*lr=*/0.05f);

for (int step = 0; step < n_steps; step++) {
  // forward: ... -> logits
  float loss = cross_entropy_forward(&logits, targets, &probs);

  // backward: zero grads, then propagate from the loss back to every parameter
  cross_entropy_backward(&probs, targets, &logits);
  // ... matmul_backward / bias_add_backward / gelu_backward / layernorm_backward ...

  adam_step(&opt, params, 4);   // updates params using their .grad
}
adam_free(&opt);
```

The **full `GPT` trains end-to-end**: `gpt_forward_train` (in `model.c`) caches activations,
`gpt_backward` propagates through the output head, final LayerNorm, every stacked block
(multi-head causal attention + FFN + residuals), and the embeddings, and `gpt_collect_params`
feeds Adam. A training step:

```c
#include "model.h"
Tensor *params[64];
int n = gpt_collect_params(&m, params);
Adam opt = adam_create(params, n, 0.02f);
GPTActs acts = gpt_acts_create(&m, seq_len);

gpt_forward_train(&m, ids, seq_len, &acts, &logits);
float loss = cross_entropy_forward(&logits, targets, &probs);
tensor_zero_grad(&logits);
cross_entropy_backward(&probs, targets, &logits);
gpt_backward(&m, ids, seq_len, &acts, &logits);   // grads into every parameter
adam_step(&opt, params, n);
```

Gradients are verified in `selftest.c` by **end-to-end numerical gradient checking** (analytic
`grad` vs `(L(+ε) − L(−ε)) / 2ε`) across the whole model, then it trains the GPT on a
next-token task and overfits it to ~0 loss. (Note: float32 finite-difference checking has a
noise floor for tiny gradients, so the check trusts well-conditioned entries; see `notes.md`.)

### Importing external weights (e.g. GPT-2) — end-to-end

The model is **GPT-2-architecture-compatible**: pre-norm blocks, multi-head causal attention
**with biases** (`b_q/b_k/b_v/b_o`, matching GPT-2's `c_attn`/`c_proj`), GELU FFN, learned
positional embeddings, and a final LayerNorm. The output head is a separate `lm_head`, so
GPT-2's tied head is written as `lm_head = wte^T` at conversion time.

**Real GPT-2 (124M) — verified, generates real text:**
```sh
pip install numpy safetensors huggingface_hub tiktoken
python tools/gpt2_setup.py --out model/gpt2/model.bin      # download + convert (~622 MB), once

python tools/gpt2_chat.py "The meaning of life is"         # one command: tokenize -> C -> text
```
(`gpt2_chat.py` wraps tokenize → `transformer generate --name gpt2` → detokenize. The manual
3-step form — `gpt2_tok.py encode` → `generate` → `gpt2_tok.py decode` — still works too.)
Actual output from that prompt: *"The meaning of life is not to be confused with the fact of
being reborn. Rather, life is to be born and that is what the term means."* GPT-2 uses its own
BPE (via `tiktoken`), so it runs in token-ID mode. See `model/gpt2/README.md`.

**Your own weights:** assemble a `.npz` with the documented key names (incl. `b_q…b_o`) and
run `convert_to_gptc.py` directly. The GPTC format is **v2** (adds the attention biases).

> Not yet wired: GPT-2's real BPE tokenizer (`encoder.json` / `vocab.bpe`) — the byte-level
> BPE here is self-contained. That plus running the full 124M weights is the next milestone
> in `notes.md`.

---

## Coding rules (for contributors)

1. **Indentation is 2 spaces.** Never use tabs.
2. **The source root is C-only, and flat.** All `.c`/`.h` files live directly in the
   project root with no subfolders — do not mix other languages into the root. A pull
   request that adds a code subfolder or non-C source to the root will be rejected.
3. **Non-C helper / third-party scripts go in `tools/`.** Anything in another language
   (Python, Go, shell, …) — converters, downloaders, experiments — belongs under
   `tools/`, never in the root next to the C sources.
4. **One concern per module.** Declare the public API in the `.h`, define it in the
   matching `.c`, and give every header an include guard (`#ifndef MODULE_H`).
5. Keep helpers that are used by only one file `static` inside that `.c`.
6. **Never use untyped `void *` parameters** in a function's public API. Pass concrete,
   typed pointers (`Tensor *`, `float *`, …) instead — untyped pointers defeat the
   compiler's type checking and lead to slow, hard-to-predict runtime errors.
7. Code must compile cleanly under `-std=c11 -Wall` (no warnings).
8. When you add a new module, add its `.o` to `OBJS` in the `Makefile` and list its
   header dependencies there.
