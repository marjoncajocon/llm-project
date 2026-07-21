# data/

Put your **training data** here. This folder is for data, not source code, so it's
exempt from the "C-only, flat root" rule. Everything except this README and `sample.txt`
is git-ignored, so your datasets won't be committed.

## Text training

Drop a plain-text file here (UTF-8), e.g. `data/train.txt`, then:

```sh
transformer train data/train.txt model.bin tok.bin
# trains a BPE tokenizer on the text, sizes the model to that vocab,
# trains with Adam, and writes the model (model.bin) + tokenizer (tok.bin)

echo "your prompt" | transformer generate model.bin tok.bin
# text-in / text-out: encodes the prompt, generates, decodes back to text
```

`sample.txt` is a tiny learnable corpus used to smoke-test the pipeline.

Bigger/cleaner text = better results. The model is small and CPU-only, so this learns the
*style/patterns* of your text; it won't match a large pretrained model. To go further, see
`../notes.md` (scale the config, or load real GPT-2 weights).

## Images (later)

Multimodal image understanding is a much larger, future build (a vision encoder +
image-text training). When added, images will live under `data/images/`. See `../notes.md`.
