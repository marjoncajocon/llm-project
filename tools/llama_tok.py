#!/usr/bin/env python3
"""Tokenizer bridge for LLaMA-family models (SmolLM2, Qwen2.5, ...): encode text
to token IDs and decode IDs back to text, using the model's own tokenizer.json.

    pip install tokenizers huggingface_hub
    python tools/llama_tok.py encode --model HuggingFaceTB/SmolLM2-135M "Hello, world"
    python tools/llama_tok.py decode --model HuggingFaceTB/SmolLM2-135M 9906 11 1917

We use the `tokenizers` library (the fast BPE) directly from the repo's
tokenizer.json — lighter than pulling in full `transformers`. Pair with
`transformer llama model.bin` (token-in/token-out); llama_chat.py wires both.
"""
import argparse
import sys


def load_tokenizer(model):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    return Tokenizer.from_file(hf_hub_download(model, "tokenizer.json"))


def main():
    ap = argparse.ArgumentParser(description="Encode/decode with a LLaMA-family tokenizer")
    ap.add_argument("mode", choices=["encode", "decode"])
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M", help="HF repo id")
    ap.add_argument("args", nargs="+", help="text (encode) or token IDs (decode)")
    a = ap.parse_args()

    tok = load_tokenizer(a.model)
    if a.mode == "encode":
        ids = tok.encode(" ".join(a.args)).ids
        print(" ".join(str(i) for i in ids))
    else:
        ids = [int(x) for x in a.args]
        print(tok.decode(ids))


if __name__ == "__main__":
    main()
