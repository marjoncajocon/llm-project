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

MAGIC = 0x414D4C4C    # 'LLMA' little-endian bytes L,L,M,A (f32 weights)
MAGIC_Q = 0x514D4C4C  # 'LLMQ' little-endian bytes L,L,M,Q (int8 weights)
VERSION = 1


def f32(a):
    return np.ascontiguousarray(a, dtype="<f4")


# Read a safetensors file to a {name: float32 ndarray} dict WITHOUT torch, and
# handling bfloat16 (which safetensors.numpy can't load — numpy has no bf16).
# Format: u64 header length, then that many bytes of JSON {name: {dtype, shape,
# data_offsets:[a,b]}}, then the raw tensor bytes.
def load_safetensors_f32(path):
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n).decode("utf-8"))
        blob = fh.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dt = meta["dtype"]
        a, b = meta["data_offsets"]
        raw = blob[a:b]
        if dt == "F32":
            arr = np.frombuffer(raw, dtype="<f4")
        elif dt == "F16":
            arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        elif dt == "BF16":
            # bf16 = high 16 bits of f32: widen to u32, shift up, reinterpret.
            u = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            arr = (u << 16).view(np.float32)
        else:
            raise ValueError(f"unsupported dtype {dt} for tensor {name}")
        out[name] = arr.reshape(meta["shape"])
    return out


# Write the checkpoint. `ints` = 8 config ints in C order (vocab, d_model,
# n_layers, n_heads, n_kv, head_dim, d_ff, max_seq); `floats` = (rope_base, eps);
# `tensors` = arrays in the canonical order llama_collect() expects. With
# int8=True, 2D weights are stored per-column int8 (scales f32 then int8 bytes)
# -> the 'LLMQ' format, ~4x smaller; 1D norms stay f32.
def write_llma(path, ints, floats, tensors, int8=False):
    with open(path, "wb") as f:
        f.write(struct.pack("<II", MAGIC_Q if int8 else MAGIC, VERSION))
        f.write(struct.pack("<8i", *ints))
        f.write(struct.pack("<2f", *floats))
        for a in tensors:
            a = np.asarray(a)
            if int8 and a.ndim == 2:
                mx = np.abs(a).max(axis=0)                 # per output-channel (column)
                scale = np.where(mx > 0, mx / 127.0, 1.0).astype("<f4")
                q = np.clip(np.rint(a / scale), -127, 127).astype(np.int8)
                f.write(scale.tobytes())
                f.write(q.tobytes())                        # row-major int8
            else:
                f.write(f32(a).tobytes())


def build_order(wte, layers, final_norm, lm_head):
    order = [wte]
    for L in layers:
        order += [L["attn_norm"], L["Wq"], L["Wk"], L["Wv"], L["Wo"],
                  L["ffn_norm"], L["Wgate"], L["Wup"], L["Wdown"]]
    order += [final_norm, lm_head]
    return order


def convert_hf(model, out, int8=False):
    from huggingface_hub import hf_hub_download

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
    st = load_safetensors_f32(hf_hub_download(model, "model.safetensors"))

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
    write_llma(out, ints, (rope_base, eps), build_order(wte, layers, final_norm, lm_head), int8=int8)
    print(f"wrote {out} ({'int8/LLMQ' if int8 else 'f32/LLMA'}): vocab={vocab} d_model={d_model} "
          f"layers={n_layers} heads={n_heads} kv={n_kv} head_dim={head_dim} d_ff={d_ff} ctx={max_seq} "
          f"rope_base={rope_base} eps={eps} tied={tied}")


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
    ap.add_argument("--int8", action="store_true", help="int8-quantize 2D weights (LLMQ, ~4x smaller)")
    ap.add_argument("--tiny", metavar="PATH", help="write a small deterministic checkpoint (no download) and exit")
    args = ap.parse_args()
    if args.tiny:
        write_tiny(args.tiny)
    else:
        convert_hf(args.model, args.out, int8=args.int8)


if __name__ == "__main__":
    main()
