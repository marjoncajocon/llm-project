# tools/

Helper and third-party scripts in languages **other than C** (Python, Go, shell, …).

The project's C source root is strictly C-only and flat (see the top-level `README.md`).
Anything that isn't C — weight converters, downloaders, data prep, experiments — lives
here instead of next to the C sources.

## Scripts

### `convert_to_gptc.py`

Converts a `.npz` of named weight arrays into the C project's **GPTC** checkpoint format,
so `gpt_load()` (`checkpoint.c`) can read it.

```sh
python tools/convert_to_gptc.py <weights.npz> <model.bin> --n-heads <N>
```

- `weights.npz` — array names/shapes must match the model parameters
  (`wte`, `wpe`, `blocks.{l}.*`, `ln_f_gamma`, `ln_f_beta`, `lm_head`).
- `model.bin` — output checkpoint, loadable in C with `gpt_load("model.bin", &m)`.
- `--n-heads N` — number of attention heads (not inferable from shapes).

Requires Python 3 + NumPy. See the script's docstring for the full key/shape list.
The GPTC format is **v2** (includes attention biases `b_q/b_k/b_v/b_o`).

### `convert_gpt2.py`

Converts HuggingFace GPT-2 `model.safetensors` into an `.npz` laid out for this model
(splits the fused `c_attn` QKV, maps `ln_*` to gamma/beta, ties `lm_head = wte^T`). Chain
it into `convert_to_gptc.py`:

```sh
pip install numpy safetensors
huggingface-cli download gpt2 model.safetensors --local-dir gpt2/
python tools/convert_gpt2.py gpt2/model.safetensors gpt2.npz
python tools/convert_to_gptc.py gpt2.npz gpt2.bin --n-heads 12
```

Requires Python 3 + NumPy + safetensors, and a downloaded GPT-2 checkpoint.

### `crawl_wikipedia.py`

Build a training corpus by crawling **connected** Wikipedia articles (plain-text extracts
via the Wikipedia API, following article links breadth-first). Saves one `.txt` per article
under `data/wiki/`, which `transformer train data/wiki` then ingests.

```sh
python tools/crawl_wikipedia.py "Machine learning" --max-pages 50
#   --max-pages N   how many articles      --out DIR   output dir (default data/wiki)
#   --delay S       seconds between requests (be polite)  --min-chars N  skip stubs
transformer train data/wiki --out ml.bin --tok ml.tok --steps 10000
```

Standard library only (urllib); needs internet. Sets a descriptive User-Agent and rate-limits
per Wikipedia's guidelines.

### `convert_llama.py`

Convert a HuggingFace **LLaMA-family** model (SmolLM2, Qwen2.5, Llama, …) into this project's
**LLMA** checkpoint, loadable in C via `transformer llama <model.bin>` (`LlamaModel` — the modern
RMSNorm+RoPE+GQA+SwiGLU architecture in `llama.c`).

```sh
pip install numpy safetensors huggingface_hub
python tools/convert_llama.py --model HuggingFaceTB/SmolLM2-135M --out model/smollm/model.bin
echo "1 2 3 4 5" | transformer llama model/smollm/model.bin      # token-in -> next-token argmax
python tools/convert_llama.py --model HuggingFaceTB/SmolLM2-135M --out model/smollm/m8.bin --int8
python tools/convert_llama.py --tiny t.bin                        # small format-test file (no download)
python tools/convert_llama.py --model Qwen/Qwen3-0.6B --out model/qwen3/model.bin --max-seq 4096
```

- **`--int8`** writes the **LLMQ** format: 2D weights are per-output-channel int8-quantized (scales +
  int8 bytes), norms stay f32 → **~4× smaller** checkpoint (SmolLM2-135M: 621 MB → 156 MB). The C
  loader dequantizes on load, so the runtime forward is unchanged (storage/download win). Output stays
  coherent (int8 linear-layer error ~<1%).

- **No RoPE weight permutation** is needed: `rope.c` uses HF's half-split ("rotate_half")
  convention, so HF `q_proj`/`k_proj` load directly.
- HF linear weights are `(out, in)` (applied as `x@Wᵀ`); our matmul is `x@W` with `W (in, out)`, so
  each linear weight is **transposed**. Norm gains + the token embedding are copied as-is.
- **Tied heads** (`tie_word_embeddings: true`, e.g. Qwen2.5/Qwen3/SmolLM2) are written **tied**: v4
  flag bit 2, no `lm_head` in the file — the engine computes `hn @ wteᵀ` directly (saves
  vocab×d_model floats: 545 MB for Qwen2.5-0.5B). `--int8` writes tied models untied (the int8
  decode path needs its own head matrix).
- **Refuses** what the engine cannot run instead of converting it wrongly: RoPE scaling (any
  non-default `rope_type`), partial rotary, `attn_output_gate`, linear-attention `layer_types`,
  chunked attention, NoPE, logit soft-capping, MoE, `mlp_bias`, non-SiLU MLPs, unknown
  `model_type`s, and any checkpoint tensor it has no slot for (e.g. an `o_proj.bias`).
- **Sharded** checkpoints (`model.safetensors.index.json`) and **local HF dirs** (`--model path/`)
  are supported; tensors are memory-mapped and streamed in 16 MB chunks (RAM ≈ one raw tensor).
- `python tools/convert_llama.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct --out model/qwen25c/model.bin --max-seq 4096`
- **q/k/v attention biases** are supported (LLMA/LLMQ v2): zero for LLaMA/SmolLM, read from
  `q/k/v_proj.bias` for Qwen2/Qwen2.5. (`o_proj` and the MLP have no bias.) So both
  `HuggingFaceTB/SmolLM2-*` and `Qwen/Qwen2.5-0.5B` convert and run.
- **QK-norm (Qwen3)** is supported: when `self_attn.q_norm`/`k_norm` weights are present they are
  written per layer (after `b_v`) into a **v4** file: a u32 flags word after the float config (bit 0 =
  QK-norm, bit 1 = the per-layer window block follows), so QK-norm and sliding windows combine.
  Models without QK-norm are written exactly as before (v2, or v3 when windowed). Qwen3's `max_position_embeddings` is 40960, and the
  C KV cache is allocated for all of it up front — pass **`--max-seq 4096`** (any cap) to keep it
  small. The C side is verified against an independent numpy reference on a `--tiny --qk-norm`
  file; a full Qwen3-0.6B vs HF comparison has NOT been run yet.

Requires Python 3 + NumPy + huggingface_hub (no torch/safetensors package needed), and internet
for the download unless `--model` is a local dir.

### `export_tokenizer.py` / `verify_hftok.py` / `gen_unitab.py` — the C tokenizer

`export_tokenizer.py <hf-repo | local-dir | gpt2> <out.tok>` writes the `.tok` (v3) file `hftok.c`
reads: vocab bytes, specials, chat template, and the **pretokenizer kind** taken from tokenizer.json
(GPT-2 regex, or the Qwen2/2.5/3 regex; anything else is refused). `verify_hftok.py <tokenizer.json>
<file.tok>` compares the C encoder with HF `tokenizers` id-for-id and pretoken-for-pretoken on a
>5000-text corpus (code, numbers, contractions, Unicode, whitespace, fuzz) — 0 mismatches for
Qwen2.5-Coder and Qwen3. `gen_unitab.py > unitab.h` regenerates the Unicode \p{L}/\p{N}/\s tables.

### `llama_tok.py` / `llama_chat.py` — tokenizer bridge + one-command text generation

`llama_tok.py` encodes/decodes with the model's own `tokenizer.json` (the fast `tokenizers` BPE).
`llama_chat.py` ties it together like `gpt2_chat.py`: tokenize → `transformer llama <ckpt>`
(token-in/token-out) → detokenize.

```sh
pip install tokenizers huggingface_hub
python tools/llama_chat.py --model HuggingFaceTB/SmolLM2-135M --ckpt model/smollm/model.bin \
    "The capital of France is" --n 40
```

Heads-up: the C model re-runs the full forward per token (no KV cache yet), so a 135M model is
slow on CPU — fine for a short demo, not long generations.

### `agent_mock.py` / `agent_llm.py` — model backends for the agent loop

The C agent (`agent --cmd "<prog>"`) writes the running transcript to a temp file, passes its
path as `argv[1]`, and uses the program's **stdout** as the model's next response. A valid
response is exactly one block that `toolcall.c` can parse:

```
<tool name="write_file" path="a.txt">contents</tool>
<tool name="read_file" path="a.txt"></tool>
<tool name="run_command" command="echo hi"></tool>
<done>one-line summary</done>
```

- **`agent_mock.py`** — offline, no-key demo: drives write→read→done deterministically
  (counts lines that start with `OBSERVATION:`). Use it to test the plumbing.
- **`agent_llm.py`** — real LLM backend (makes the agent genuinely act). Constrains the model
  to emit exactly one of the blocks above. Provider: **Claude via the Anthropic API**
  (`claude-opus-4-8`, adaptive thinking) by default, or **OpenAI** if only `OPENAI_API_KEY` is
  set (`$OPENAI_MODEL` or `gpt-4o`). With no key / no SDK / any API error it prints a `<done>`
  block so the loop ends cleanly. `pip install anthropic` (or `openai`) and set the key.

```sh
# offline demo
transformer agent --cmd "python tools/agent_mock.py" --goal "..."
# real LLM (needs ANTHROPIC_API_KEY or OPENAI_API_KEY); the y/N gate is the default
transformer agent --cmd "python tools/agent_llm.py" --goal "refactor X"
```

Every tool path is confined to the agent's workspace (`--workspace DIR`, default
`./workspace`), and each observation reaches the backend framed as
`OBSERVATION: [[obs:NONCE]]` ... `[[/obs:NONCE]]` with control characters stripped and
role-marker lines (`MODEL:`, `GOAL:`, ...) inside it prefixed with `> ` -- backends should
treat that text as data, not instructions (`agent_llm.py`'s system prompt says so, and
`agent_local.py` unwraps the frame to the plain format its SFT data uses).

### `webchat/` — browser testing frontend (Go + WebSocket)

A self-contained **Go** web server (`tools/webchat/`, stdlib only — no external modules) that
drives the engine from a browser: **text chat** across every detected model, **image / audio
generation**, and **image / audio / video "understanding"** (teach-then-reproduce captioning). It
bridges each request to the CLI as a subprocess, and the model switcher classifies `model/<name>/`
by checkpoint magic (pure-C `generate` vs the `gpt2_chat.py` / `llama_chat.py` bridges).

```sh
make                                   # build the engine first (repo root)
cd tools/webchat && go run . -root ../..   # open http://localhost:8080
```

Needs Go 1.23+. Python (+ `tiktoken` / `tokenizers`) only for the real GPT-2 / SmolLM2 / Qwen
models; **ffmpeg** only for the video panel. Quality is toy-scale by design (it proves the
pipelines run on real media, not production fidelity). See `webchat/README.md` for the full guide.
This is the one multi-file tool, so it lives in its own subfolder under `tools/`.

## Adding a tool

Drop your script in this folder, keep it self-contained, and document its usage here.
Do not add non-C source to the project root. A multi-file tool (like `webchat/`) may use its own
subfolder under `tools/`.
