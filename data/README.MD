# data/

Put your **training data** here. This folder is for data, not source code, so it's
exempt from the "C-only, flat root" rule. Everything except this README and `sample.txt`
is git-ignored, so your datasets won't be committed.

## Text training

Drop a plain-text file here (UTF-8), e.g. `data/train.txt`, then:

```sh
transformer train data/train.txt --out model.bin --tok tok.bin --steps 5000
# trains a BPE tokenizer on the text, sizes the model to that vocab,
# trains with Adam, and writes the model (model.bin) + tokenizer (tok.bin)
# scale up:  --d-model 256 --layers 6 --heads 8 --ctx 128   (see ../README.md)

echo "your prompt" | transformer generate model.bin tok.bin
# text-in / text-out: encodes the prompt, generates, decodes back to text
```

You can also point `train` at a **directory** (e.g. `transformer train data/`) — it reads
every `.txt` under it recursively, including subfolders.

## Naming your model / giving it an identity

Two different "names":

- **Folder/checkpoint name** — `--name X` saves the model to `model/X/` (organizational).
- **Persona** — what it *says* when asked "who are you?" — comes from **training data**, not a
  flag. Put identity Q&A in your corpus and the model learns to say it. See `identity.txt`:

  ```
  Q: Who are you?
  A: I am Marvin, an AI model created by Marjon.
  ```

  ```sh
  transformer train data/identity.txt --name marvin --steps 3000
  printf "Q: Who are you?\nA:" | transformer generate --name marvin
  # -> A: I am Marvin, an AI model created by Marjon.
  ```

  Edit `identity.txt` with your own model name + creator, repeat the lines a few times, and add
  it alongside your other `.txt` so the identity is baked into any model you train. (More/cleaner
  examples + more `--steps` = a cleaner, more consistent persona.)

`sample.txt` is a tiny learnable corpus used to smoke-test the pipeline.

Bigger/cleaner text = better results. The model is small and CPU-only, so this learns the
*style/patterns* of your text; it won't match a large pretrained model. To go further, see
`../notes.md` (scale the config, or load real GPT-2 weights).

## Images (later)

Multimodal image understanding is a much larger, future build (a vision encoder +
image-text training). When added, images will live under `data/images/`. See `../notes.md`.
