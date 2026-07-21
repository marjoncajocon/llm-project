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

## Adding a tool

Drop your script in this folder, keep it self-contained, and document its usage here.
Do not add non-C source to the project root.
