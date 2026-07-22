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
```

- **`--int8`** writes the **LLMQ** format: 2D weights are per-output-channel int8-quantized (scales +
  int8 bytes), norms stay f32 → **~4× smaller** checkpoint (SmolLM2-135M: 621 MB → 156 MB). The C
  loader dequantizes on load, so the runtime forward is unchanged (storage/download win). Output stays
  coherent (int8 linear-layer error ~<1%).

- **No RoPE weight permutation** is needed: `rope.c` uses HF's half-split ("rotate_half")
  convention, so HF `q_proj`/`k_proj` load directly.
- HF linear weights are `(out, in)` (applied as `x@Wᵀ`); our matmul is `x@W` with `W (in, out)`, so
  each linear weight is **transposed**. Norm gains + the token embedding are copied as-is; tied
  models use `lm_head = embed_tokens.weightᵀ`.
- **q/k/v attention biases** are supported (LLMA/LLMQ v2): zero for LLaMA/SmolLM, read from
  `q/k/v_proj.bias` for Qwen2/Qwen2.5. (`o_proj` and the MLP have no bias.) So both
  `HuggingFaceTB/SmolLM2-*` and `Qwen/Qwen2.5-0.5B` convert and run.

Requires Python 3 + NumPy + safetensors + huggingface_hub, and internet for the download.

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
  (counts `OBSERVATION:` lines). Use it to test the plumbing.
- **`agent_llm.py`** — real LLM backend (makes the agent genuinely act). Constrains the model
  to emit exactly one of the blocks above. Provider: **Claude via the Anthropic API**
  (`claude-opus-4-8`, adaptive thinking) by default, or **OpenAI** if only `OPENAI_API_KEY` is
  set (`$OPENAI_MODEL` or `gpt-4o`). With no key / no SDK / any API error it prints a `<done>`
  block so the loop ends cleanly. `pip install anthropic` (or `openai`) and set the key.

```sh
# offline demo
transformer agent --cmd "python tools/agent_mock.py" --goal "..."
# real LLM (needs ANTHROPIC_API_KEY or OPENAI_API_KEY); keep --ask as the safety gate
transformer agent --cmd "python tools/agent_llm.py" --goal "refactor X" --ask
```

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
