#!/usr/bin/env python3
"""One command to generate real GPT-2 text with the C model.

Wraps: tokenize (tiktoken) -> `transformer generate --name gpt2` (token mode) ->
detokenize. Needs model/gpt2/model.bin (run tools/gpt2_setup.py first) and a
built `transformer` binary.

    python tools/gpt2_chat.py "The meaning of life is"
    python tools/gpt2_chat.py "Once upon a time" --exe ./transformer
"""
import argparse
import os
import re
import subprocess
import sys

import tiktoken


def main():
    ap = argparse.ArgumentParser(description="Generate real GPT-2 text via the C model")
    ap.add_argument("prompt", help="the text prompt")
    ap.add_argument("--name", default="gpt2", help="model folder under model/ (default gpt2)")
    ap.add_argument("--exe", default=None, help="path to the transformer binary")
    args = ap.parse_args()

    exe = args.exe or ("transformer.exe" if os.name == "nt" else "./transformer")
    if os.path.exists(exe):            # resolve a cwd-relative binary to an absolute path
        exe = os.path.abspath(exe)     # (if not found, fall back to a PATH lookup)
    enc = tiktoken.get_encoding("gpt2")
    prompt_ids = enc.encode(args.prompt)

    proc = subprocess.run(
        [exe, "generate", "--name", args.name],
        input=" ".join(str(i) for i in prompt_ids),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(proc.returncode or 1)

    # The C model prints "gen: <prompt ids> [<generated>] ..." — pull all ids out.
    gen_line = next((l for l in proc.stdout.splitlines() if l.strip().startswith("gen:")),
                    proc.stdout)
    ids = [int(x) for x in re.findall(r"[0-9]+", gen_line)]
    print(enc.decode(ids))


if __name__ == "__main__":
    main()
