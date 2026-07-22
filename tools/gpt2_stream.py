#!/usr/bin/env python3
"""Streaming GPT-2 chat bridge for the webchat server (tiktoken).

Reads the prompt TEXT from stdin, tokenizes with tiktoken's gpt2 encoding, runs
`transformer generate --name <name>` (token mode, which now prints one `t <id>`
line per generated token), decodes incrementally, and prints JSON-line deltas
`{"d": "..."}` then `{"done": true}`. GPT-2's generate is fixed at 40 new tokens
in the engine.

    <prompt on stdin> | python tools/gpt2_stream.py --name gpt2
"""
import argparse
import json
import os
import subprocess
import sys

import tiktoken


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gpt2")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--stop", default="\nUser")
    args = ap.parse_args()

    prompt = sys.stdin.read()
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode(prompt)
    eos = enc.eot_token  # <|endoftext|> (50256): natural end of text

    exe = args.exe or ("transformer.exe" if os.name == "nt" else "./transformer")
    if os.path.exists(exe):
        exe = os.path.abspath(exe)

    proc = subprocess.Popen(
        [exe, "generate", "--name", args.name],
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
            tid = int(line[2:])
        except ValueError:
            continue
        if tid == eos:
            proc.kill()
            break
        gen.append(tid)
        full = enc.decode(gen)
        cut = -1
        for st in ("<|endoftext|>", "<|im_end|>", "<|eot_id|>", "\nUser:", "\nHuman:", "\nYou:", "\nAssistant:", "\nAI:"):
            i = full.find(st)
            if i != -1 and (cut == -1 or i < cut):
                cut = i
        if cut != -1:
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
