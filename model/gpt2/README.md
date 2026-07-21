# model/gpt2 — real GPT-2 (124M)

The actual OpenAI GPT-2 small model, downloaded and converted to this project's GPTC format
so the pure-C transformer runs it. Config: vocab 50257, d_model 768, 12 layers, 12 heads,
d_ff 3072, ctx 1024. `model.bin` is ~622 MB and **git-ignored** (regenerate it, below).

## Regenerate `model.bin`

```sh
pip install numpy safetensors huggingface_hub tiktoken
python tools/gpt2_setup.py --out model/gpt2/model.bin
```

## Run it

GPT-2 uses its own byte-level BPE (via `tiktoken`), not this project's C tokenizer, so there
is **no `tok.bin`** here — drive it in **token-ID mode**. Easiest is the one-command wrapper:

```sh
python tools/gpt2_chat.py "The meaning of life is"
```

or the manual token-ID form:

```sh
ids=$(python tools/gpt2_tok.py encode "The meaning of life is")
echo "$ids" | transformer generate --name gpt2 > out.txt      # prints "gen: <ids> [<gen>]..."
python tools/gpt2_tok.py decode <ids from out.txt>
```

Example output (from "The meaning of life is"):

> The meaning of life is not to be confused with the fact of being reborn. Rather, life is to
> be born and that is what the term means.

## Notes

- Generation is single-threaded per decode step (each step is a 1-row matmul), ~0.1 s/token
  on a modern CPU; ~5 s for 40 tokens. Loading the 622 MB model needs ~1.5 GB RAM.
- Bigger variants: `python tools/gpt2_setup.py --model gpt2-medium --n-heads 16 --out model/gpt2-medium/model.bin`
  (gpt2-large `--n-heads 20`, gpt2-xl `--n-heads 25`).
