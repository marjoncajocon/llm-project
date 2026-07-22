#!/usr/bin/env python3
"""Streaming LLaMA-family chat bridge for the webchat server.

Reads the prompt TEXT from stdin, tokenizes it (the model's own tokenizer.json),
runs `transformer llama <ckpt> <n>` which now prints one `t <id>` line per token
as it is generated, decodes the growing id list incrementally, and prints each
new text piece as a JSON line `{"d": "..."}` (then `{"done": true}`). A stop
sequence (default "\nUser:") ends a turn cleanly and stops the model rambling.

    <prompt on stdin> | python tools/llama_stream.py --ckpt model/qwen/model_int8.bin \\
        --model Qwen/Qwen2.5-0.5B --n 220
"""
import argparse
import json
import os
import subprocess
import sys

from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download


def load_tok(model):
    try:
        path = hf_hub_download(model, "tokenizer.json", local_files_only=True)
    except Exception:
        path = hf_hub_download(model, "tokenizer.json")
    return Tokenizer.from_file(path)


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--stop", default="\nUser")
    args = ap.parse_args()

    prompt = sys.stdin.read()
    tok = load_tok(args.model)
    ids = tok.encode(prompt).ids

    exe = args.exe or ("transformer.exe" if os.name == "nt" else "./transformer")
    if os.path.exists(exe):
        exe = os.path.abspath(exe)

    proc = subprocess.Popen(
        [exe, "llama", args.ckpt, str(args.n)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    proc.stdin.write(" ".join(str(i) for i in ids) + "\n")
    proc.stdin.flush()
    proc.stdin.close()

    gen, prev = [], ""
    for line in proc.stdout:
        line = line.strip()
        if not line.startswith("t "):
            continue
        try:
            gen.append(int(line[2:]))
        except ValueError:
            continue
        full = tok.decode(gen)
        cut = full.find(args.stop)
        if cut != -1:                       # model started a new turn -> stop
            piece = full[len(prev):cut]
            if piece:
                emit({"d": piece})
            proc.kill()
            break
        piece = full[len(prev):]
        if piece:
            emit({"d": piece})
            prev = full

    try:
        proc.wait(timeout=1)
    except Exception:
        proc.kill()
    emit({"done": True})


if __name__ == "__main__":
    main()
