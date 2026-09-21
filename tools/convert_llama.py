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
  * q/k/v attention biases (Qwen2) are read when present, else written as zeros.
  * Sliding-window attention (Mistral-style `sliding_window`, or per-layer
    `layer_types`) is read from config.json into the per-layer window block.
  * QK-norm (Qwen3): a per-head RMSNorm on q and k before RoPE, with learned
    gains self_attn.q_norm / k_norm (head_dim each). Detected from the weights;
    the two gains are stored per layer right after b_v.
  * Version: the lowest that holds the model — v2 (no feature, exactly as
    before), v3 (windows only: n_layers int32 after the float config), v4 (any
    QK-norm: a u32 flags word after the float config, bit 0 = QK-norm, bit 1 =
    windows present, then the windows iff bit 1).
  * --max-seq caps the context length. The C KV cache is allocated for the full
    max_seq up front, and Qwen3's 40960 would need ~9 GB for Qwen3-0.6B.

`--tiny PATH` writes a small deterministic checkpoint (no download) to test the
LLMA byte format against the C loader (`--qk-norm` / `--window W` exercise v4 / v3).
"""
import argparse
import json
import struct
import sys

import numpy as np

MAGIC = 0x414D4C4C    # 'LLMA' little-endian bytes L,L,M,A (f32 weights)
MAGIC_Q = 0x514D4C4C  # 'LLMQ' little-endian bytes L,L,M,Q (int8 weights)
VERSION = 2           # v2 adds q/k/v attention biases (zero for LLaMA/SmolLM, real for Qwen2)
VERSION_WINDOW = 3    # v3 = v2 + n_layers int32 sliding windows after the float config
VERSION_FLAGS = 4     # v4 = v2 + a u32 flags word after the float config (+ windows iff bit 1)
FLAG_QK_NORM = 1      # bit 0: per-head q/k RMSNorm (Qwen3), gains stored after b_v
FLAG_WINDOWS = 2      # bit 1: the n_layers int32 window block follows the flags word


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
# -> the 'LLMQ' format, ~4x smaller; 1D norms stay f32. `windows` = one sliding
# window per layer (0 = full attention); `qk_norm` = the tensors include the
# q_norm/k_norm gains. The version is the lowest that holds the model (v2/v3/v4,
# same rule as llama_model_save in C).
def write_llma(path, ints, floats, tensors, int8=False, windows=None, qk_norm=False):
    windowed = bool(windows) and any(w > 0 for w in windows)
    flags = (FLAG_QK_NORM | (FLAG_WINDOWS if windowed else 0)) if qk_norm else 0
    version = VERSION_FLAGS if flags else (VERSION_WINDOW if windowed else VERSION)
    with open(path, "wb") as f:
        f.write(struct.pack("<II", MAGIC_Q if int8 else MAGIC, version))
        f.write(struct.pack("<8i", *ints))
        f.write(struct.pack("<2f", *floats))
        if flags:
            f.write(struct.pack("<I", flags))
        if windowed:
            f.write(struct.pack(f"<{len(windows)}i", *[max(0, int(w)) for w in windows]))
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


# Canonical order (must match llama_collect in llama.c). q_norm/k_norm are only
# present — and only written — for QK-norm (Qwen3) models.
def build_order(wte, layers, final_norm, lm_head):
    order = [wte]
    for L in layers:
        order += [L["attn_norm"], L["Wq"], L["Wk"], L["Wv"],
                  L["b_q"], L["b_k"], L["b_v"]]
        if "q_norm" in L:
            order += [L["q_norm"], L["k_norm"]]
        order += [L["Wo"], L["ffn_norm"], L["Wgate"], L["Wup"], L["Wdown"]]
    order += [final_norm, lm_head]
    return order


# Per-layer sliding windows from an HF config (0 = full attention). Newer configs
# spell the pattern out as `layer_types`; Mistral just sets `sliding_window`
# (every layer); Qwen2 has `sliding_window` but gates it with use_sliding_window
# (usually false) and max_window_layers (layers at or above it are windowed).
def hf_windows(cfg, n_layers):
    sw = cfg.get("sliding_window")
    if not sw or cfg.get("use_sliding_window") is False:
        return [0] * n_layers
    if cfg.get("layer_types"):
        return [sw if t == "sliding_attention" else 0 for t in cfg["layer_types"]]
    if "use_sliding_window" in cfg and cfg.get("max_window_layers") is not None:
        return [sw if l >= cfg["max_window_layers"] else 0 for l in range(n_layers)]
    return [sw] * n_layers


# The "every Nth layer is global" pattern, mirroring llama_model_set_window in C.
def pattern_windows(n_layers, window, global_every):
    return [0 if window <= 0 or (global_every > 0 and (l + 1) % global_every == 0) else window
            for l in range(n_layers)]


def convert_hf(model, out, int8=False, max_seq_cap=None):
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
    if max_seq_cap:
        max_seq = min(max_seq, max_seq_cap)
    rope_base = float(cfg.get("rope_theta", 10000.0))
    eps = float(cfg.get("rms_norm_eps", 1e-5))
    tied = cfg.get("tie_word_embeddings", True)
    if n_heads % n_kv:
        sys.exit(f"n_heads {n_heads} not divisible by n_kv {n_kv}")

    print(f"downloading {model}/model.safetensors ...")
    st = load_safetensors_f32(hf_hub_download(model, "model.safetensors"))

    def g(name):
        return f32(st[name])

    # q/k/v biases: present in Qwen2, absent (-> zero) in LLaMA/SmolLM. o_proj has none.
    def bias(name, dim):
        return f32(st[name]) if name in st else np.zeros(dim, "<f4")

    # QK-norm (Qwen3): per-head q/k RMSNorm gains. All-or-nothing across layers.
    qk_norm = "model.layers.0.self_attn.q_norm.weight" in st
    if qk_norm and st["model.layers.0.self_attn.q_norm.weight"].shape != (head_dim,):
        sys.exit(f"q_norm shape {st['model.layers.0.self_attn.q_norm.weight'].shape} != ({head_dim},)")

    qd, kvd = n_heads * head_dim, n_kv * head_dim
    layers = []
    for l in range(n_layers):
        p = f"model.layers.{l}."
        layers.append({
            "attn_norm": g(p + "input_layernorm.weight"),
            "Wq": g(p + "self_attn.q_proj.weight").T,   # (out,in) -> (in,out)
            "Wk": g(p + "self_attn.k_proj.weight").T,
            "Wv": g(p + "self_attn.v_proj.weight").T,
            "b_q": bias(p + "self_attn.q_proj.bias", qd),
            "b_k": bias(p + "self_attn.k_proj.bias", kvd),
            "b_v": bias(p + "self_attn.v_proj.bias", kvd),
            "Wo": g(p + "self_attn.o_proj.weight").T,
            "ffn_norm": g(p + "post_attention_layernorm.weight"),
            "Wgate": g(p + "mlp.gate_proj.weight").T,
            "Wup": g(p + "mlp.up_proj.weight").T,
            "Wdown": g(p + "mlp.down_proj.weight").T,
        })
        if qk_norm:
            layers[-1]["q_norm"] = g(p + "self_attn.q_norm.weight")
            layers[-1]["k_norm"] = g(p + "self_attn.k_norm.weight")
    wte = g("model.embed_tokens.weight")
    final_norm = g("model.norm.weight")
    lm_head = f32(wte.T) if (tied or "lm_head.weight" not in st) else g("lm_head.weight").T

    windows = hf_windows(cfg, n_layers)
    ints = (vocab, d_model, n_layers, n_heads, n_kv, head_dim, d_ff, max_seq)
    write_llma(out, ints, (rope_base, eps), build_order(wte, layers, final_norm, lm_head),
               int8=int8, windows=windows, qk_norm=qk_norm)
    print(f"wrote {out} ({'int8/LLMQ' if int8 else 'f32/LLMA'}): vocab={vocab} d_model={d_model} "
          f"layers={n_layers} heads={n_heads} kv={n_kv} head_dim={head_dim} d_ff={d_ff} ctx={max_seq} "
          f"rope_base={rope_base} eps={eps} tied={tied} windows={windows if any(windows) else 'none'} "
          f"qk_norm={qk_norm}")
    # Windowed layers keep a W-slot ring; full-attention layers hold all max_seq.
    kv_gb = sum(2 * (min(w, max_seq) if w > 0 else max_seq) * kvd * 4 for w in windows) / 1e9
    if kv_gb > 2:
        print(f"warning: the C KV cache for ctx={max_seq} is ~{kv_gb:.1f} GB; "
              f"consider re-running with --max-seq 4096")


def write_tiny(out, window=0, global_every=0, qk_norm=False):
    rng = np.random.default_rng(0)
    vocab, d_model, n_layers, n_heads, n_kv, head_dim, d_ff, max_seq = 32, 8, 2, 4, 2, 2, 16, 64
    qd, kvd = n_heads * head_dim, n_kv * head_dim

    def r(*shape):
        return (rng.standard_normal(shape) * 0.1).astype("<f4")

    layers = []
    for _ in range(n_layers):
        layers.append({
            "attn_norm": np.ones(d_model, "<f4"),
            "Wq": r(d_model, qd), "Wk": r(d_model, kvd), "Wv": r(d_model, kvd),
            "b_q": np.zeros(qd, "<f4"), "b_k": np.zeros(kvd, "<f4"), "b_v": np.zeros(kvd, "<f4"),
            "Wo": r(qd, d_model),
            "ffn_norm": np.ones(d_model, "<f4"),
            "Wgate": r(d_model, d_ff), "Wup": r(d_model, d_ff), "Wdown": r(d_ff, d_model),
        })
        if qk_norm:   # gains ~1 but not exactly, so a wrong slot/order shows up
            layers[-1]["q_norm"] = (1.0 + r(head_dim) * 3).astype("<f4")
            layers[-1]["k_norm"] = (1.0 + r(head_dim) * 3).astype("<f4")
    wte = r(vocab, d_model)
    ints = (vocab, d_model, n_layers, n_heads, n_kv, head_dim, d_ff, max_seq)
    windows = pattern_windows(n_layers, window, global_every)
    write_llma(out, ints, (10000.0, 1e-5), build_order(wte, layers, np.ones(d_model, "<f4"), r(d_model, vocab)),
               windows=windows, qk_norm=qk_norm)
    print(f"wrote tiny {out}: vocab={vocab} d_model={d_model} layers={n_layers} "
          f"heads={n_heads} kv={n_kv} head_dim={head_dim} windows={windows if any(windows) else 'none'} "
          f"qk_norm={qk_norm}")


def main():
    ap = argparse.ArgumentParser(description="HF LLaMA-family -> LLMA checkpoint")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M", help="HF repo id")
    ap.add_argument("--out", default="model.bin", help="output LLMA checkpoint")
    ap.add_argument("--int8", action="store_true", help="int8-quantize 2D weights (LLMQ, ~4x smaller)")
    ap.add_argument("--max-seq", type=int, default=None,
                    help="cap the context length (the C KV cache is sized for all of it)")
    ap.add_argument("--tiny", metavar="PATH", help="write a small deterministic checkpoint (no download) and exit")
    ap.add_argument("--window", type=int, default=0, help="--tiny only: sliding window W (0 = none)")
    ap.add_argument("--global-every", type=int, default=0, help="--tiny only: every Nth layer global")
    ap.add_argument("--qk-norm", action="store_true", help="--tiny only: include Qwen3 QK-norm gains (v4 file)")
    args = ap.parse_args()
    if args.tiny:
        write_tiny(args.tiny, args.window, args.global_every, qk_norm=args.qk_norm)
    else:
        convert_hf(args.model, args.out, int8=args.int8, max_seq_cap=args.max_seq)


if __name__ == "__main__":
    main()
