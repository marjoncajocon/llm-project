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

## Adding a tool

Drop your script in this folder, keep it self-contained, and document its usage here.
Do not add non-C source to the project root.
