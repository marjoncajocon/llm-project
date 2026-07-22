#!/usr/bin/env python3
"""One command to generate real modern-LLM text with the pure-C engine.

Wraps: tokenize (the model's tokenizer.json) -> `transformer llama <ckpt>`
(token-in/token-out) -> detokenize. Needs a converted LLMA checkpoint
(tools/convert_llama.py) and a built `transformer` binary.

    pip install tokenizers huggingface_hub
    python tools/llama_chat.py --model HuggingFaceTB/SmolLM2-135M \
        --ckpt model/smollm/model.bin "The capital of France is" --n 40

Heads-up: the C model re-runs the full forward per token (no KV cache yet), so a
135M model is slow on CPU — fine for a short demo, not for long generations.
"""
import argparse
import os
import re
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description="Generate modern-LLM text via the C model")
    ap.add_argument("prompt", help="the text prompt")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M", help="HF repo id (for the tokenizer)")
    ap.add_argument("--ckpt", required=True, help="LLMA checkpoint (from convert_llama.py)")
    ap.add_argument("--n", type=int, default=40, help="tokens to generate")
    ap.add_argument("--exe", default=None, help="path to the transformer binary")
    a = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(hf_hub_download(a.model, "tokenizer.json"))
    prompt_ids = tok.encode(a.prompt).ids

    exe = a.exe or ("transformer.exe" if os.name == "nt" else "./transformer")
    if os.path.exists(exe):
        exe = os.path.abspath(exe)

    proc = subprocess.run(
        [exe, "llama", a.ckpt, str(a.n)],
        input=" ".join(str(i) for i in prompt_ids),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(proc.returncode or 1)

    gen_line = next((l for l in proc.stdout.splitlines() if l.strip().startswith("gen:")), "")
    ids = [int(x) for x in re.findall(r"[0-9]+", gen_line)]
    print(tok.decode(ids))


if __name__ == "__main__":
    main()
