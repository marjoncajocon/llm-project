#!/usr/bin/env python3
"""Convert a HuggingFace LLaMA-family model to this project's LLMA checkpoint
(loadable in C via llama_model_load / `transformer llama`).

    pip install numpy safetensors huggingface_hub
    python tools/convert_llama.py --model HuggingFaceTB/SmolLM2-135M --out model/smollm/model.bin

Notes:
  * NO RoPE weight permutation is needed. rope.c implements HF's half-split
    ("rotate_half") convention, so HF-format q_proj/k_proj load directly. (The
    permutation people mention is only for Meta-interleaved -> HF conversion.)
  * HF stores linear weights as (out, in) and applies x @ W^T; our matmul is
    x @ W with W (in, out). So every linear weight is TRANSPOSED here. Norm gains
    and the token embedding are copied as-is. Tied models (no lm_head.weight) use
    lm_head = embed_tokens.weight^T.
  * No attention biases: LlamaBlock has none (fine for LLaMA/SmolLM). Models with
    q/k/v biases (e.g. Qwen2) would need bias support first — the converter warns.

`--tiny PATH` writes a small deterministic checkpoint (no download) to test the
LLMA byte format against the C loader.
"""
import argparse
import json
import struct
import sys

import numpy as np

MAGIC = 0x414D4C4C   # 'LLMA' little-endian bytes L,L,M,A
VERSION = 1


def f32(a):
    return np.ascontiguousarray(a, dtype="<f4")


# Write the LLMA file. `ints` = 8 config ints in C order (vocab, d_model,
# n_layers, n_heads, n_kv, head_dim, d_ff, max_seq); `floats` = (rope_base, eps);
# `tensors` = arrays in the canonical order llama_collect() expects.
def write_llma(path, ints, floats, tensors):
    with open(path, "wb") as f:
        f.write(struct.pack("<II", MAGIC, VERSION))
        f.write(struct.pack("<8i", *ints))
        f.write(struct.pack("<2f", *floats))
        for a in tensors:
            f.write(f32(a).tobytes())


def build_order(wte, layers, final_norm, lm_head):
    order = [wte]
    for L in layers:
        order += [L["attn_norm"], L["Wq"], L["Wk"], L["Wv"], L["Wo"],
                  L["ffn_norm"], L["Wgate"], L["Wup"], L["Wdown"]]
    order += [final_norm, lm_head]
    return order


def convert_hf(model, out):
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    cfg = json.load(open(hf_hub_download(model, "config.json")))
    vocab = cfg["vocab_size"]
    d_model = cfg["hidden_size"]
    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_heads)
    head_dim = cfg.get("head_dim", d_model // n_heads)
    d_ff = cfg["intermediate_size"]
    max_seq = cfg.get("max_position_embeddings", 2048)
    rope_base = float(cfg.get("rope_theta", 10000.0))
    eps = float(cfg.get("rms_norm_eps", 1e-5))
    tied = cfg.get("tie_word_embeddings", True)
    if n_heads % n_kv:
        sys.exit(f"n_heads {n_heads} not divisible by n_kv {n_kv}")

    print(f"downloading {model}/model.safetensors ...")
    st = load_file(hf_hub_download(model, "model.safetensors"))

    def g(name):
        return f32(st[name])

    if any(k.endswith("self_attn.q_proj.bias") for k in st):
        print("WARNING: model has attention biases (e.g. Qwen2) which LlamaBlock "
              "does not support yet — they are DROPPED; output will be wrong.", file=sys.stderr)

    layers = []
    for l in range(n_layers):
        p = f"model.layers.{l}."
        layers.append({
            "attn_norm": g(p + "input_layernorm.weight"),
            "Wq": g(p + "self_attn.q_proj.weight").T,   # (out,in) -> (in,out)
            "Wk": g(p + "self_attn.k_proj.weight").T,
            "Wv": g(p + "self_attn.v_proj.weight").T,
            "Wo": g(p + "self_attn.o_proj.weight").T,
            "ffn_norm": g(p + "post_attention_layernorm.weight"),
            "Wgate": g(p + "mlp.gate_proj.weight").T,
            "Wup": g(p + "mlp.up_proj.weight").T,
            "Wdown": g(p + "mlp.down_proj.weight").T,
        })
    wte = g("model.embed_tokens.weight")
    final_norm = g("model.norm.weight")
    lm_head = f32(wte.T) if (tied or "lm_head.weight" not in st) else g("lm_head.weight").T

    ints = (vocab, d_model, n_layers, n_heads, n_kv, head_dim, d_ff, max_seq)
    write_llma(out, ints, (rope_base, eps), build_order(wte, layers, final_norm, lm_head))
    print(f"wrote {out}: vocab={vocab} d_model={d_model} layers={n_layers} heads={n_heads} "
          f"kv={n_kv} head_dim={head_dim} d_ff={d_ff} ctx={max_seq} rope_base={rope_base} "
          f"eps={eps} tied={tied}")


def write_tiny(out):
    rng = np.random.default_rng(0)
    vocab, d_model, n_layers, n_heads, n_kv, head_dim, d_ff, max_seq = 32, 8, 2, 4, 2, 2, 16, 64
    qd, kvd = n_heads * head_dim, n_kv * head_dim

    def r(*shape):
        return (rng.standard_normal(shape) * 0.1).astype("<f4")

    layers = []
    for _ in range(n_layers):
        layers.append({
            "attn_norm": np.ones(d_model, "<f4"),
            "Wq": r(d_model, qd), "Wk": r(d_model, kvd), "Wv": r(d_model, kvd), "Wo": r(qd, d_model),
            "ffn_norm": np.ones(d_model, "<f4"),
            "Wgate": r(d_model, d_ff), "Wup": r(d_model, d_ff), "Wdown": r(d_ff, d_model),
        })
    wte = r(vocab, d_model)
    ints = (vocab, d_model, n_layers, n_heads, n_kv, head_dim, d_ff, max_seq)
    write_llma(out, ints, (10000.0, 1e-5), build_order(wte, layers, np.ones(d_model, "<f4"), r(d_model, vocab)))
    print(f"wrote tiny {out}: vocab={vocab} d_model={d_model} layers={n_layers} "
          f"heads={n_heads} kv={n_kv} head_dim={head_dim}")


def main():
    ap = argparse.ArgumentParser(description="HF LLaMA-family -> LLMA checkpoint")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M", help="HF repo id")
    ap.add_argument("--out", default="model.bin", help="output LLMA checkpoint")
    ap.add_argument("--tiny", metavar="PATH", help="write a small deterministic checkpoint (no download) and exit")
    args = ap.parse_args()
    if args.tiny:
        write_tiny(args.tiny)
    else:
        convert_hf(args.model, args.out)


if __name__ == "__main__":
    main()
