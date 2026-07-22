#!/usr/bin/env python3
"""Export a model's tokenizer vocabulary to a compact .tok file the C engine
reads — so the engine tokenizes on its own with NO Python at runtime (this
export is a one-time setup step, like the weight converters).

    python tools/export_tokenizer.py gpt2 model/gpt2/tok.hf
    python tools/export_tokenizer.py <hf-repo> <out.tok>     # LLaMA-family (tokenizer.json)

.tok format (little-endian):
    "HFTK", uint32 version=1, uint32 n_vocab, int32 eos_id,
    then n_vocab entries: uint16 len, <len> raw bytes  (token id -> its bytes)
Encoding uses the tiktoken/byte-level-BPE rule "merge the adjacent pair whose
combined bytes have the lowest id", which needs only this id->bytes table.
"""
import struct
import sys


def write_tok(path, n_vocab, eos_id, id_to_bytes):
    with open(path, "wb") as f:
        f.write(b"HFTK")
        f.write(struct.pack("<II", 1, n_vocab))
        f.write(struct.pack("<i", eos_id))
        for i in range(n_vocab):
            b = id_to_bytes(i)
            f.write(struct.pack("<H", len(b)))
            f.write(b)
    print("wrote %s: %d tokens, eos=%d" % (path, n_vocab, eos_id))


def export_gpt2(out):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    n = enc.n_vocab
    eos = enc.eot_token

    def id_to_bytes(i):
        try:
            return enc.decode_single_token_bytes(i)
        except Exception:
            return b""  # special tokens (e.g. <|endoftext|>) have no bytes
    write_tok(out, n, eos, id_to_bytes)


def export_hf(repo, out):
    # LLaMA-family (SmolLM2, Qwen, ...): load tokenizer.json and recover each
    # token's raw bytes from the HF byte-level encoding.
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo, "tokenizer.json", local_files_only=True)
    except Exception:
        path = hf_hub_download(repo, "tokenizer.json")
    tok = Tokenizer.from_file(path)
    vocab = tok.get_vocab()  # token_str -> id
    n = tok.get_vocab_size()
    # HF byte-level maps each byte to a printable unicode char; invert it.
    byte_decoder = {}
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    k = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + k)
            k += 1
    for b, c in zip(bs, cs):
        byte_decoder[chr(c)] = b

    id_to_str = [None] * n
    for s, i in vocab.items():
        if 0 <= i < n:
            id_to_str[i] = s

    def id_to_bytes(i):
        s = id_to_str[i]
        if s is None:
            return b""
        try:
            return bytes(byte_decoder[ch] for ch in s)
        except KeyError:
            return s.encode("utf-8", "replace")  # special/added tokens

    eos = tok.token_to_id("<|im_end|>")
    if eos is None:
        eos = tok.token_to_id("<|endoftext|>")
    write_tok(out, n, eos if eos is not None else -1, id_to_bytes)


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: export_tokenizer.py <gpt2 | hf-repo> <out.tok>\n")
        sys.exit(2)
    which, out = sys.argv[1], sys.argv[2]
    if which == "gpt2":
        export_gpt2(out)
    else:
        export_hf(which, out)


if __name__ == "__main__":
    main()
