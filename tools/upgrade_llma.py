#!/usr/bin/env python3
"""Upgrade an LLMA v1 checkpoint in place-ish (v1 -> v2) without re-downloading.

v2 added the per-layer q/k/v attention biases (b_q, b_k, b_v), which sit between
Wv and Wo in the canonical tensor order. LLaMA/SmolLM have no such biases, so the
upgrade is purely mechanical: copy every v1 tensor across and insert three ZERO
bias vectors per layer. Qwen2-style models with real biases must be re-converted
with tools/convert_llama.py instead — their biases are not zero.

    python tools/upgrade_llma.py model/smollm/model.bin model/smollm/model_v2.bin

Streams the file, so it never holds the whole checkpoint in memory.

v1 layout: header | wte | per layer [attn_norm Wq Wk Wv Wo ffn_norm Wgate Wup Wdown]
           | final_norm | lm_head
v2 layout: same, with [b_q b_k b_v] inserted after Wv.
Header (48 bytes): u32 magic, u32 version, 8x i32 config, 2x f32 (rope_base, eps).
"""
import struct
import sys

MAGIC = 0x414D4C4C   # 'LLMA'
CHUNK = 1 << 22      # 4 MiB streaming copy


def copy_floats(src, dst, n):
    """Copy n float32 values from src to dst in bounded-size chunks."""
    remaining = n * 4
    while remaining:
        take = min(remaining, CHUNK)
        buf = src.read(take)
        if len(buf) != take:
            sys.exit("error: input truncated (wanted %d more bytes)" % remaining)
        dst.write(buf)
        remaining -= take


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src_path, dst_path = sys.argv[1], sys.argv[2]

    with open(src_path, "rb") as f, open(dst_path, "wb") as g:
        head = f.read(48)
        if len(head) != 48:
            sys.exit("error: '%s' is too small to be an LLMA checkpoint" % src_path)
        magic, version = struct.unpack("<II", head[:8])
        cfg = struct.unpack("<8i", head[8:40])
        rope_base, eps = struct.unpack("<2f", head[40:48])
        if magic != MAGIC:
            sys.exit("error: '%s' is not an LLMA checkpoint (magic %#x)" % (src_path, magic))
        if version != 1:
            sys.exit("error: '%s' is already version %d, nothing to do" % (src_path, version))

        vocab, d_model, n_layers, n_heads, n_kv, head_dim, d_ff, max_seq = cfg
        qd, kvd = n_heads * head_dim, n_kv * head_dim
        print("v1: vocab=%d d_model=%d layers=%d heads=%d kv=%d head_dim=%d d_ff=%d max_seq=%d"
              % (vocab, d_model, n_layers, n_heads, n_kv, head_dim, d_ff, max_seq))

        g.write(struct.pack("<II", MAGIC, 2))
        g.write(struct.pack("<8i", *cfg))
        g.write(struct.pack("<2f", rope_base, eps))

        copy_floats(f, g, vocab * d_model)                 # wte
        zeros = b"\x00" * (4 * (qd + 2 * kvd))
        for _ in range(n_layers):
            copy_floats(f, g, d_model)                     # attn_norm
            copy_floats(f, g, d_model * qd)                # Wq
            copy_floats(f, g, d_model * kvd)               # Wk
            copy_floats(f, g, d_model * kvd)               # Wv
            g.write(zeros)                                 # b_q, b_k, b_v  (NEW in v2)
            copy_floats(f, g, qd * d_model)                # Wo
            copy_floats(f, g, d_model)                     # ffn_norm
            copy_floats(f, g, d_model * d_ff)              # Wgate
            copy_floats(f, g, d_model * d_ff)              # Wup
            copy_floats(f, g, d_ff * d_model)              # Wdown
        copy_floats(f, g, d_model)                         # final_norm
        copy_floats(f, g, d_model * vocab)                 # lm_head

        if f.read(1):
            sys.exit("error: trailing bytes in '%s' — layout assumption wrong" % src_path)

    print("wrote %s (v2, +%d zero bias floats)" % (dst_path, n_layers * (qd + 2 * kvd)))


main()
