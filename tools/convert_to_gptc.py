#!/usr/bin/env python3
"""Convert a set of named weight arrays (.npz) into the C project's GPTC
checkpoint format, so `gpt_load()` (checkpoint.c) can read it.

This is the bridge for importing external weights (e.g. a converted GPT-2) into
the pure-C model: produce an .npz whose arrays match this model's parameter
names/shapes, then run this script to emit a .bin the C code loads directly.

Expected .npz keys (float arrays, C/row-major):
    wte              (vocab_size, d_model)
    wpe              (max_seq_len, d_model)
    blocks.{l}.W_q   (d_model, d_model)      # same for W_k, W_v, W_o
    blocks.{l}.b_q   (d_model,)              # same for b_k, b_v, b_o (attention biases)
    blocks.{l}.gamma1 (d_model,)             # same for beta1, gamma2, beta2
    blocks.{l}.ff_W1 (d_model, d_ff)
    blocks.{l}.ff_b1 (d_ff,)
    blocks.{l}.ff_W2 (d_ff, d_model)
    blocks.{l}.ff_b2 (d_model,)
    ln_f_gamma       (d_model,)
    ln_f_beta        (d_model,)
    lm_head          (d_model, vocab_size)

The tensor order written here matches collect_tensors() in checkpoint.c exactly.

Usage:
    python convert_to_gptc.py weights.npz model.bin --n-heads 2
"""
import argparse
import struct
import sys

import numpy as np

MAGIC = 0x43545047  # 'GPTC' little-endian
VERSION = 2         # v2 adds attention biases

# Per-block tensor names, in the exact order checkpoint.c writes them.
BLOCK_KEYS = [
    "W_q", "W_k", "W_v", "W_o",
    "b_q", "b_k", "b_v", "b_o",
    "gamma1", "beta1", "gamma2", "beta2",
    "ff_W1", "ff_b1", "ff_W2", "ff_b2",
]


def main():
    ap = argparse.ArgumentParser(description="Convert .npz weights to a GPTC .bin checkpoint")
    ap.add_argument("input", help="input .npz file of named weight arrays")
    ap.add_argument("output", help="output .bin checkpoint path")
    ap.add_argument("--n-heads", type=int, required=True,
                    help="number of attention heads (not inferable from shapes)")
    args = ap.parse_args()

    z = np.load(args.input)

    # Infer the config from array shapes.
    vocab_size, d_model = z["wte"].shape
    max_seq_len = z["wpe"].shape[0]
    n_layers = 0
    while f"blocks.{n_layers}.W_q" in z:
        n_layers += 1
    if n_layers == 0:
        sys.exit("error: no transformer blocks found (expected key 'blocks.0.W_q')")
    d_ff = z["blocks.0.ff_W1"].shape[1]

    if d_model % args.n_heads != 0:
        sys.exit(f"error: d_model={d_model} not divisible by n_heads={args.n_heads}")

    def write_tensor(f, name):
        arr = np.ascontiguousarray(z[name], dtype="<f4")  # little-endian float32, C order
        f.write(arr.tobytes())

    with open(args.output, "wb") as f:
        f.write(struct.pack("<II", MAGIC, VERSION))
        f.write(struct.pack("<6i", vocab_size, d_model, args.n_heads,
                            n_layers, d_ff, max_seq_len))
        write_tensor(f, "wte")
        write_tensor(f, "wpe")
        for l in range(n_layers):
            for key in BLOCK_KEYS:
                write_tensor(f, f"blocks.{l}.{key}")
        write_tensor(f, "ln_f_gamma")
        write_tensor(f, "ln_f_beta")
        write_tensor(f, "lm_head")

    print(f"wrote {args.output}: vocab={vocab_size} d_model={d_model} "
          f"heads={args.n_heads} layers={n_layers} d_ff={d_ff} max_seq={max_seq_len}")


if __name__ == "__main__":
    main()
