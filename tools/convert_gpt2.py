#!/usr/bin/env python3
"""Convert HuggingFace GPT-2 weights into an .npz laid out for this C model, then
(optionally) into a GPTC .bin via convert_to_gptc.py.

GPT-2 differences this handles:
  * `c_attn` fuses Q/K/V: weight (d_model, 3*d_model), bias (3*d_model,). We split it
    into W_q/W_k/W_v (d_model, d_model) and b_q/b_k/b_v (d_model,).
  * GPT-2 `Conv1D` stores weights already as (in, out) and computes x @ W, which matches
    our matmul, so NO transpose is needed for c_attn/c_proj/c_fc.
  * The output head is tied to the token embedding: we set lm_head = wte^T
    (shape (d_model, vocab_size)), because this C model uses a separate lm_head tensor.
  * ln_1/ln_2/ln_f map to gamma/beta (weight/bias).

Requires: pip install numpy safetensors  (and a downloaded GPT-2 model.safetensors).
Get weights e.g.:  huggingface-cli download gpt2 model.safetensors --local-dir gpt2/

Usage:
    python tools/convert_gpt2.py gpt2/model.safetensors gpt2.npz
    python tools/convert_to_gptc.py gpt2.npz gpt2.bin --n-heads 12

GPT-2 base config: vocab=50257 d_model=768 layers=12 heads=12 d_ff=3072 ctx=1024.
"""
import sys

import numpy as np

try:
    from safetensors.numpy import load_file
except ImportError:
    sys.exit("error: pip install safetensors  (and numpy)")


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <model.safetensors> <out.npz>")
    src, out = sys.argv[1], sys.argv[2]
    st = load_file(src)

    def g(name):
        return np.asarray(st[name], dtype="f4")

    d = {}
    wte = g("wte.weight")            # (vocab, d_model)
    d["wte"] = wte
    d["wpe"] = g("wpe.weight")       # (ctx, d_model)
    d_model = wte.shape[1]

    n_layers = 0
    while f"h.{n_layers}.ln_1.weight" in st:
        n_layers += 1

    for l in range(n_layers):
        p = f"h.{l}."
        o = f"blocks.{l}."
        # Fused QKV -> split into three (d_model, d_model) / (d_model,)
        c_attn_w = g(p + "attn.c_attn.weight")   # (d_model, 3*d_model)
        c_attn_b = g(p + "attn.c_attn.bias")     # (3*d_model,)
        d[o + "W_q"] = c_attn_w[:, 0 * d_model:1 * d_model]
        d[o + "W_k"] = c_attn_w[:, 1 * d_model:2 * d_model]
        d[o + "W_v"] = c_attn_w[:, 2 * d_model:3 * d_model]
        d[o + "b_q"] = c_attn_b[0 * d_model:1 * d_model]
        d[o + "b_k"] = c_attn_b[1 * d_model:2 * d_model]
        d[o + "b_v"] = c_attn_b[2 * d_model:3 * d_model]
        d[o + "W_o"] = g(p + "attn.c_proj.weight")   # (d_model, d_model)
        d[o + "b_o"] = g(p + "attn.c_proj.bias")     # (d_model,)
        d[o + "gamma1"] = g(p + "ln_1.weight")
        d[o + "beta1"] = g(p + "ln_1.bias")
        d[o + "gamma2"] = g(p + "ln_2.weight")
        d[o + "beta2"] = g(p + "ln_2.bias")
        d[o + "ff_W1"] = g(p + "mlp.c_fc.weight")    # (d_model, d_ff)
        d[o + "ff_b1"] = g(p + "mlp.c_fc.bias")      # (d_ff,)
        d[o + "ff_W2"] = g(p + "mlp.c_proj.weight")  # (d_ff, d_model)
        d[o + "ff_b2"] = g(p + "mlp.c_proj.bias")    # (d_model,)

    d["ln_f_gamma"] = g("ln_f.weight")
    d["ln_f_beta"] = g("ln_f.bias")
    d["lm_head"] = wte.T.copy()   # tied head: logits = h @ wte^T  -> (d_model, vocab)

    np.savez(out, **d)
    print(f"wrote {out}: d_model={d_model} layers={n_layers} "
          f"(then: python tools/convert_to_gptc.py {out} model.bin --n-heads 12)")


if __name__ == "__main__":
    main()
