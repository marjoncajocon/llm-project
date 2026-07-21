#!/usr/bin/env python3
"""Download a real GPT-2 model and convert it straight to this project's GPTC
checkpoint (loadable in C via gpt_load).

    pip install numpy safetensors huggingface_hub
    python tools/gpt2_setup.py --out gpt2.bin            # gpt2 (124M)
    python tools/gpt2_setup.py --model gpt2-medium --n-heads 16 --out gpt2m.bin

Then tokenize with tools/gpt2_tok.py and run:  transformer generate gpt2.bin ...

Head counts:  gpt2=12, gpt2-medium=16, gpt2-large=20, gpt2-xl=25.
This C model matches GPT-2's architecture (pre-norm, attention biases, tanh-GELU,
learned positions, tied head), so the loaded weights produce real GPT-2 logits.
"""
import argparse
import struct
import sys

import numpy as np

MAGIC = 0x43545047   # 'GPTC'
VERSION = 2

# Per-block tensors, in the exact order checkpoint.c / convert_to_gptc.py expect.
BLOCK = ["W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o",
         "gamma1", "beta1", "gamma2", "beta2", "ff_W1", "ff_b1", "ff_W2", "ff_b2"]


def main():
    ap = argparse.ArgumentParser(description="Download + convert GPT-2 to a GPTC checkpoint")
    ap.add_argument("--model", default="gpt2", help="HF repo id (gpt2, gpt2-medium, ...)")
    ap.add_argument("--n-heads", type=int, default=12, help="attention heads (gpt2=12)")
    ap.add_argument("--out", default="gpt2.bin", help="output checkpoint")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    print(f"downloading {args.model}/model.safetensors (this is a few hundred MB)...")
    path = hf_hub_download(args.model, "model.safetensors")
    st = load_file(path)

    def g(name):
        return np.ascontiguousarray(st[name], dtype="<f4")

    wte = g("wte.weight")
    vocab, d_model = wte.shape
    max_seq = st["wpe.weight"].shape[0]
    n_layers = 0
    while f"h.{n_layers}.ln_1.weight" in st:
        n_layers += 1
    d_ff = st["h.0.mlp.c_fc.weight"].shape[1]
    if d_model % args.n_heads:
        sys.exit(f"d_model {d_model} not divisible by n_heads {args.n_heads}")

    # Map one GPT-2 block to our 16 tensors (splits fused c_attn; ties nothing here).
    def block_tensors(l):
        p = f"h.{l}."
        caw = g(p + "attn.c_attn.weight")   # (d_model, 3*d_model)
        cab = g(p + "attn.c_attn.bias")      # (3*d_model,)
        m = d_model
        return {
            "W_q": caw[:, 0:m], "W_k": caw[:, m:2*m], "W_v": caw[:, 2*m:3*m],
            "b_q": cab[0:m], "b_k": cab[m:2*m], "b_v": cab[2*m:3*m],
            "W_o": g(p + "attn.c_proj.weight"), "b_o": g(p + "attn.c_proj.bias"),
            "gamma1": g(p + "ln_1.weight"), "beta1": g(p + "ln_1.bias"),
            "gamma2": g(p + "ln_2.weight"), "beta2": g(p + "ln_2.bias"),
            "ff_W1": g(p + "mlp.c_fc.weight"), "ff_b1": g(p + "mlp.c_fc.bias"),
            "ff_W2": g(p + "mlp.c_proj.weight"), "ff_b2": g(p + "mlp.c_proj.bias"),
        }

    with open(args.out, "wb") as f:
        f.write(struct.pack("<II", MAGIC, VERSION))
        f.write(struct.pack("<6i", vocab, d_model, args.n_heads, n_layers, d_ff, max_seq))
        w = lambda a: f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())
        w(wte)
        w(g("wpe.weight"))
        for l in range(n_layers):
            bt = block_tensors(l)
            for k in BLOCK:
                w(bt[k])
        w(g("ln_f.weight"))
        w(g("ln_f.bias"))
        w(wte.T)                       # tied output head: lm_head = wte^T (d_model, vocab)

    print(f"wrote {args.out}: vocab={vocab} d_model={d_model} heads={args.n_heads} "
          f"layers={n_layers} d_ff={d_ff} ctx={max_seq}")


if __name__ == "__main__":
    main()
