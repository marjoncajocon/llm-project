# model/

Each trained or downloaded AI model lives in its **own subfolder** here, named by role:

```
model/
  gpt2/        model.bin              (real GPT-2 124M; tokenized with tiktoken)
  coder/       model.bin  tok.bin     (a model you trained)
  chatty/      model.bin  tok.bin
```

A model folder holds `model.bin` (weights, GPTC format) and — for models trained here —
`tok.bin` (its BPE tokenizer). Reference a model by **folder name** with `--name`:

```sh
# Train a new model into model/coder/
transformer train data/code --name coder --d-model 256 --layers 6 --heads 8 --steps 20000

# Generate from it (text in / text out, since it has a tok.bin)
echo "def add(a, b):" | transformer generate --name coder
```

`generate --name X` loads `model/X/model.bin`. If `model/X/tok.bin` exists it runs in
**text mode**; otherwise it runs in **token-ID mode** (e.g. GPT-2, which is tokenized
externally by `tools/gpt2_tok.py`).

**Git:** model weights are large and **git-ignored** (`model/**/*.bin`). Only text/markdown
(this README and per-model `README.md`) is committed, so the folder layout + notes travel
but the multi-hundred-MB weights don't.

See each model's own `README.md` for how it was made and how to run it.
