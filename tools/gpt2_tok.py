#!/usr/bin/env python3
"""GPT-2 tokenizer bridge (real BPE via tiktoken) for the C `generate` command.

    pip install tiktoken
    python tools/gpt2_tok.py encode "The meaning of life is"   # -> space-separated token IDs
    python tools/gpt2_tok.py decode 464 3616 286 1204 318       # -> text

Pipe the ids into the C model (token-ID mode), then decode what it emits:
    ids=$(python tools/gpt2_tok.py encode "Hello, I am")
    echo "$ids" | transformer generate gpt2.bin        # prints: gen: <prompt> [<generated>] ...
    python tools/gpt2_tok.py decode <the ids it printed>
"""
import sys

import tiktoken


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: gpt2_tok.py encode <text> | decode <id...>")
    enc = tiktoken.get_encoding("gpt2")
    cmd = sys.argv[1]
    if cmd == "encode":
        ids = enc.encode(sys.argv[2])
        print(" ".join(str(i) for i in ids))
    elif cmd == "decode":
        ids = [int(x) for x in sys.argv[2:]]
        sys.stdout.write(enc.decode(ids))
        sys.stdout.write("\n")
    else:
        sys.exit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
