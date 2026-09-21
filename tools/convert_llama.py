#!/usr/bin/env python3
"""Convert a HuggingFace LLaMA-family model to this project's LLMA checkpoint
(loadable in C via llama_model_load / `transformer llama`).

    pip install numpy huggingface_hub
    python tools/convert_llama.py --model HuggingFaceTB/SmolLM2-135M --out model/smollm/model.bin
    python tools/convert_llama.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct --out model/qwen25c/model.bin --max-seq 4096
    python tools/convert_llama.py --model path/to/local/hf_dir --out m.bin   # config.json + *.safetensors

Notes:
  * NO RoPE weight permutation is needed. rope.c implements HF's half-split
    ("rotate_half") convention, so HF-format q_proj/k_proj load directly. (The
    permutation people mention is only for Meta-interleaved -> HF conversion.)
  * HF stores linear weights as (out, in) and applies x @ W^T; our matmul is
    x @ W with W (in, out). So every linear weight is TRANSPOSED here. Norm gains
    and the token embedding are copied as-is.
  * Tied models (tie_word_embeddings = true) are written TIED (v4 flag bit 2):
    no lm_head is stored -- the C engine uses wte^T directly, saving vocab x
    d_model floats on disk and in RAM (136M floats = 545 MB for Qwen2.5-0.5B).
    With --int8 a tied model is written UNTIED (lm_head = int8(wte^T)): the int8
    in-memory decode path needs its own (d_model, vocab) head matrix.
  * q/k/v attention biases (Qwen2) are read when present, else written as zeros.
  * Sliding-window attention (Mistral-style `sliding_window`, or per-layer
    `layer_types`) is read from config.json into the per-layer window block.
  * QK-norm (Qwen3): a per-head RMSNorm on q and k before RoPE, with learned
    gains self_attn.q_norm / k_norm (head_dim each). Detected from the weights;
    the two gains are stored per layer right after b_v.
  * Version: the lowest that holds the model — v2 (no feature, exactly as
    before), v3 (windows only: n_layers int32 after the float config), v4 (QK-norm
    and/or a tied head: a u32 flags word after the float config, bit 0 = QK-norm,
    bit 1 = windows present, bit 2 = tied head; then the windows iff bit 1).
  * --max-seq caps the context length. The C KV cache is allocated for the full
    max_seq up front, and Qwen3's 40960 would need ~9 GB for Qwen3-0.6B.
  * REFUSES (instead of silently writing a model the engine would run wrongly):
    RoPE scaling of any kind (rope_scaling / rope_parameters with a non-default
    rope_type: YaRN, linear, dynamic, llama3, longrope, mrope), partial rotary,
    gated attention output (attn_output_gate), non-full/sliding layer types
    (linear attention etc.), chunked attention, NoPE layers, logit soft-capping,
    Granite-style multipliers, MoE, non-SiLU activations, unknown model types,
    and ANY checkpoint tensor the engine has no slot for (an o_proj bias, MLP
    biases, extra norms, ...) -- so nothing is ever dropped silently.
  * Streaming: tensors are read one at a time from the (possibly sharded,
    model.safetensors.index.json) checkpoint through memory maps and written in
    row chunks, so peak RAM is about one raw tensor (the largest one that must be
    transposed) plus a 16 MB chunk -- not the whole model.

`--tiny PATH` writes a small deterministic checkpoint (no download) to test the
LLMA byte format against the C loader (`--qk-norm` / `--window W` exercise v4 / v3,
`--tied` the tied head).
"""
import argparse
import json
import os
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
FLAG_TIED = 4         # bit 2: tied head -- lm_head is not stored, the engine uses wte^T

CHUNK_ELEMS = 1 << 22  # rows are converted/written in chunks of ~4M floats (16 MB)


def f32(a):
    return np.ascontiguousarray(a, dtype="<f4")


# ---------------------------------------------------------------------------
# Lazy safetensors: parse the headers only; each tensor is a memory map read on
# demand. Format: u64 header length, JSON {name: {dtype, shape, data_offsets}},
# then the raw bytes. bf16 stays u16 until converted chunk by chunk (numpy has no
# bf16: widen to u32, shift up 16, reinterpret as f32).
# ---------------------------------------------------------------------------
ST_DTYPES = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}


class SafeTensors:
    def __init__(self, paths):
        self.meta = {}
        for path in paths:
            with open(path, "rb") as fh:
                n = int.from_bytes(fh.read(8), "little")
                header = json.loads(fh.read(n).decode("utf-8"))
            base = 8 + n
            for name, m in header.items():
                if name == "__metadata__":
                    continue
                if m["dtype"] not in ST_DTYPES:
                    raise ValueError(f"unsupported dtype {m['dtype']} for tensor {name}")
                a, b = m["data_offsets"]
                self.meta[name] = (path, m["dtype"], tuple(m["shape"]), base + a, b - a)

    def __contains__(self, name):
        return name in self.meta

    def names(self):
        return list(self.meta)

    def shape(self, name):
        return self.meta[name][2]

    def raw(self, name):
        path, dt, shape, off, nbytes = self.meta[name]
        count = int(np.prod(shape)) if shape else 1
        if count * np.dtype(ST_DTYPES[dt]).itemsize != nbytes:
            raise ValueError(f"tensor {name}: size mismatch in header")
        return np.memmap(path, dtype=ST_DTYPES[dt], mode="r", offset=off, shape=shape), dt


def to_f32(raw, dt):
    if dt == "F32":
        return np.asarray(raw, dtype="<f4")
    if dt == "F16":
        return np.asarray(raw).astype(np.float32)
    return (np.asarray(raw).astype(np.uint32) << 16).view(np.float32)   # BF16


# Sources: .shape is the shape WRITTEN, .rows(r0, r1) returns float32 rows
# [r0, r1) of it. The writer streams every tensor chunk by chunk through these.
class Arr:                      # an in-memory array (--tiny, zero biases)
    def __init__(self, a):
        self.a = f32(a)
        self.shape = self.a.shape

    def rows(self, r0, r1):
        return self.a if self.a.ndim == 1 else self.a[r0:r1]


class Plain:                    # a checkpoint tensor written as stored (norms, wte)
    def __init__(self, st, name):
        self.st, self.name, self.shape = st, name, st.shape(name)

    def rows(self, r0, r1):
        raw, dt = self.st.raw(self.name)
        return to_f32(raw if raw.ndim == 1 else raw[r0:r1], dt)


class Trans:                    # an HF (out, in) linear, written transposed (in, out)
    def __init__(self, st, name):
        self.st, self.name = st, name
        o, i = st.shape(name)
        self.shape = (i, o)
        self._raw = None

    def rows(self, r0, r1):     # written rows = stored columns
        if self._raw is None:   # one in-RAM copy of the RAW tensor (bf16: half of f32)
            raw, self._dt = self.st.raw(self.name)
            self._raw = np.array(raw)
        return f32(to_f32(self._raw[:, r0:r1], self._dt).T)

    def release(self):
        self._raw = None


def _chunk_rows(shape):
    if len(shape) == 1:
        return max(1, shape[0])
    return max(1, CHUNK_ELEMS // max(1, shape[1]))


# Write the checkpoint. `ints` = 8 config ints in C order (vocab, d_model,
# n_layers, n_heads, n_kv, head_dim, d_ff, max_seq); `floats` = (rope_base, eps);
# `tensors` = sources (or arrays) in the canonical order llama_collect() expects.
# With int8=True, 2D weights are stored per-column int8 (scales f32 then int8
# bytes) -> the 'LLMQ' format, ~4x smaller; 1D norms stay f32. `windows` = one
# sliding window per layer (0 = full attention); `qk_norm` = the tensors include
# the q_norm/k_norm gains; `tied` = no lm_head among `tensors` (flag bit 2). The
# version is the lowest that holds the model (v2/v3/v4, same rule as
# llama_model_save in C). Output is byte-identical to the former in-memory writer
# (same per-element float32 arithmetic, just chunked).
def write_llma(path, ints, floats, tensors, int8=False, windows=None, qk_norm=False, tied=False):
    windowed = bool(windows) and any(w > 0 for w in windows)
    feat = (FLAG_QK_NORM if qk_norm else 0) | (FLAG_TIED if tied else 0)
    flags = (feat | (FLAG_WINDOWS if windowed else 0)) if feat else 0
    version = VERSION_FLAGS if flags else (VERSION_WINDOW if windowed else VERSION)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<II", MAGIC_Q if int8 else MAGIC, version))
        f.write(struct.pack("<8i", *ints))
        f.write(struct.pack("<2f", *floats))
        if flags:
            f.write(struct.pack("<I", flags))
        if windowed:
            f.write(struct.pack(f"<{len(windows)}i", *[max(0, int(w)) for w in windows]))
        for src in tensors:
            if isinstance(src, np.ndarray):
                src = Arr(src)
            shape = tuple(src.shape)
            step, nrows = _chunk_rows(shape), shape[0]
            if int8 and len(shape) == 2:
                mx = np.zeros(shape[1], np.float32)          # per output-channel (column)
                for r0 in range(0, nrows, step):
                    mx = np.maximum(mx, np.abs(src.rows(r0, min(nrows, r0 + step))).max(axis=0))
                scale = np.where(mx > 0, mx / 127.0, 1.0).astype("<f4")
                f.write(scale.tobytes())
                for r0 in range(0, nrows, step):
                    a = src.rows(r0, min(nrows, r0 + step))
                    f.write(np.clip(np.rint(a / scale), -127, 127).astype(np.int8).tobytes())
            elif len(shape) == 1:
                f.write(f32(src.rows(0, nrows)).tobytes())
            else:
                for r0 in range(0, nrows, step):
                    f.write(f32(src.rows(r0, min(nrows, r0 + step))).tobytes())
            if hasattr(src, "release"):
                src.release()
    os.replace(tmp, path)


# Canonical order (must match llama_collect in llama.c). q_norm/k_norm are only
# present — and only written — for QK-norm (Qwen3) models; lm_head is None (not
# written) for a tied head.
def build_order(wte, layers, final_norm, lm_head):
    order = [wte]
    for L in layers:
        order += [L["attn_norm"], L["Wq"], L["Wk"], L["Wv"],
                  L["b_q"], L["b_k"], L["b_v"]]
        if "q_norm" in L:
            order += [L["q_norm"], L["k_norm"]]
        order += [L["Wo"], L["ffn_norm"], L["Wgate"], L["Wup"], L["Wdown"]]
    order += [final_norm]
    if lm_head is not None:
        order += [lm_head]
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


# ---------------------------------------------------------------------------
# Feature guards: everything here is something the C engine cannot express. A
# config that uses it fails loudly instead of converting into a model that runs
# and silently computes something else.
# ---------------------------------------------------------------------------
SUPPORTED_TYPES = {"llama", "mistral", "qwen2", "qwen3"}


class Unsupported(Exception):
    pass


def check_config(cfg):
    errs = []
    mt = cfg.get("model_type")
    if mt not in SUPPORTED_TYPES:
        errs.append(f"model_type {mt!r} (supported: {', '.join(sorted(SUPPORTED_TYPES))})")
    act = cfg.get("hidden_act", cfg.get("hidden_activation", "silu"))
    if act != "silu":
        errs.append(f"hidden_act {act!r} (the engine's MLP is SwiGLU with SiLU)")

    # RoPE: plain rotary with a base (rope_theta) only. Any scaling rewrites the
    # frequencies or positions (YaRN, linear, dynamic NTK, llama3, longrope, mrope).
    def rope_type(d):
        return d.get("rope_type", d.get("type")) if isinstance(d, dict) else d
    rs = cfg.get("rope_scaling")
    if rs is not None and rope_type(rs) != "default":
        errs.append(f"rope_scaling {rs!r} (only unscaled rotary embeddings are implemented)")
    rp = cfg.get("rope_parameters")
    if rp is not None and rope_type(rp) not in (None, "default"):
        errs.append(f"rope_parameters {rp!r} (only rope_type 'default' is implemented)")
    if cfg.get("rope_type") not in (None, "default"):
        errs.append(f"rope_type {cfg.get('rope_type')!r}")
    prf = cfg.get("partial_rotary_factor")
    if prf is not None and float(prf) != 1.0:
        errs.append(f"partial_rotary_factor {prf} (the engine rotates the whole head)")

    if cfg.get("attn_output_gate"):
        errs.append("attn_output_gate (gated attention output, Qwen3-Next style)")
    lt = cfg.get("layer_types")
    if lt is not None:
        bad = sorted({t for t in lt if t not in ("full_attention", "sliding_attention")})
        if bad:
            errs.append(f"layer_types {bad} (only full / sliding-window softmax attention)")
        if len(lt) != cfg.get("num_hidden_layers"):
            errs.append(f"layer_types has {len(lt)} entries for {cfg.get('num_hidden_layers')} layers")
        if "sliding_attention" in lt and not cfg.get("sliding_window") and cfg.get("use_sliding_window") is not False:
            errs.append("layer_types has sliding_attention but no sliding_window size")
    if cfg.get("use_sliding_window") and not cfg.get("sliding_window"):
        errs.append("use_sliding_window without a sliding_window size")
    for k in ("sliding_window_pattern", "attention_chunk_size", "no_rope_layers",
              "attn_logit_softcapping", "final_logit_softcapping", "use_qk_norm",
              "num_experts", "num_local_experts", "n_routed_experts", "moe_intermediate_size",
              "decoder_sparse_step", "linear_num_value_heads", "mrope_section"):
        v = cfg.get(k)
        if v is None or v is False or v == 0:
            continue
        if k == "no_rope_layers" and all(x for x in v):   # all 1 = RoPE in every layer
            continue
        errs.append(f"{k} = {v!r}")
    for k in ("embedding_multiplier", "logits_scaling", "residual_multiplier", "attention_multiplier"):
        v = cfg.get(k)
        if v is not None and float(v) != 1.0:
            errs.append(f"{k} = {v} (Granite-style scaling)")
    if cfg.get("mlp_bias"):
        errs.append("mlp_bias = true (the SwiGLU MLP has no biases)")
    nh = cfg.get("num_attention_heads", 1)
    if nh % cfg.get("num_key_value_heads", nh):
        errs.append("num_attention_heads not divisible by num_key_value_heads")
    if errs:
        raise Unsupported("unsupported model config:\n  - " + "\n  - ".join(errs))


def resolver(model):
    """file name -> local path (None if the repo/dir has no such file)."""
    if os.path.isdir(model):
        def get(name):
            p = os.path.join(model, name)
            return p if os.path.exists(p) else None
        return get
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    def get(name):
        try:
            return hf_hub_download(model, name)
        except EntryNotFoundError:
            return None
    return get


def open_checkpoint(get):
    idx = get("model.safetensors.index.json")
    if idx:
        with open(idx, encoding="utf-8") as fh:
            wm = json.load(fh)["weight_map"]
        files = sorted(set(wm.values()))
        print(f"sharded checkpoint: {len(files)} file(s)")
        paths = []
        for fn in files:
            p = get(fn)
            if p is None:
                sys.exit(f"error: shard {fn} listed in the index is missing")
            paths.append(p)
        st = SafeTensors(paths)
        missing = [n for n in wm if n not in st]
        if missing:
            sys.exit(f"error: {len(missing)} indexed tensors not found in the shards, e.g. {missing[:3]}")
        return st
    p = get("model.safetensors")
    if p is None:
        sys.exit("error: neither model.safetensors nor model.safetensors.index.json found")
    return SafeTensors([p])


def convert_hf(model, out, int8=False, max_seq_cap=None):
    get = resolver(model)
    cfg_path = get("config.json")
    if cfg_path is None:
        sys.exit(f"error: {model} has no config.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    try:
        check_config(cfg)
    except Unsupported as e:
        sys.exit(f"error: {model}: {e}")
    vocab = cfg["vocab_size"]
    d_model = cfg["hidden_size"]
    n_layers = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_heads)
    head_dim = cfg.get("head_dim") or d_model // n_heads
    d_ff = cfg["intermediate_size"]
    max_seq = cfg.get("max_position_embeddings", 2048)
    if max_seq_cap:
        max_seq = min(max_seq, max_seq_cap)
    rp = cfg.get("rope_parameters") or {}
    rope_base = float(cfg.get("rope_theta", rp.get("rope_theta", 10000.0)))
    eps = float(cfg.get("rms_norm_eps", 1e-5))
    # transformers' Llama/Mistral/Qwen2/Qwen3 configs all default to untied.
    tied = bool(cfg.get("tie_word_embeddings", False))

    st = open_checkpoint(get)
    used = set()

    def need(name, shape):
        if name not in st:
            sys.exit(f"error: tensor {name} missing from the checkpoint")
        if tuple(st.shape(name)) != tuple(shape):
            sys.exit(f"error: tensor {name} has shape {st.shape(name)}, expected {tuple(shape)}")
        used.add(name)
        return name

    def plain(name, shape):
        return Plain(st, need(name, shape))

    def trans(name, out_dim, in_dim):     # HF (out, in) -> written (in, out)
        return Trans(st, need(name, (out_dim, in_dim)))

    # q/k/v biases: present in Qwen2, absent (-> zero) in LLaMA/SmolLM. o_proj has none.
    def bias(name, dim):
        return plain(name, (dim,)) if name in st else Arr(np.zeros(dim, "<f4"))

    # QK-norm (Qwen3): per-head q/k RMSNorm gains. All-or-nothing across layers.
    qk_norm = "model.layers.0.self_attn.q_norm.weight" in st
    if cfg.get("model_type") == "qwen3" and not qk_norm:
        sys.exit("error: qwen3 checkpoint without self_attn.q_norm/k_norm weights")

    qd, kvd = n_heads * head_dim, n_kv * head_dim
    layers = []
    for l in range(n_layers):
        p = f"model.layers.{l}."
        L = {
            "attn_norm": plain(p + "input_layernorm.weight", (d_model,)),
            "Wq": trans(p + "self_attn.q_proj.weight", qd, d_model),   # (out,in) -> (in,out)
            "Wk": trans(p + "self_attn.k_proj.weight", kvd, d_model),
            "Wv": trans(p + "self_attn.v_proj.weight", kvd, d_model),
            "b_q": bias(p + "self_attn.q_proj.bias", qd),
            "b_k": bias(p + "self_attn.k_proj.bias", kvd),
            "b_v": bias(p + "self_attn.v_proj.bias", kvd),
            "Wo": trans(p + "self_attn.o_proj.weight", d_model, qd),
            "ffn_norm": plain(p + "post_attention_layernorm.weight", (d_model,)),
            "Wgate": trans(p + "mlp.gate_proj.weight", d_ff, d_model),
            "Wup": trans(p + "mlp.up_proj.weight", d_ff, d_model),
            "Wdown": trans(p + "mlp.down_proj.weight", d_model, d_ff),
        }
        if qk_norm:
            L["q_norm"] = plain(p + "self_attn.q_norm.weight", (head_dim,))
            L["k_norm"] = plain(p + "self_attn.k_norm.weight", (head_dim,))
        layers.append(L)
    wte = plain("model.embed_tokens.weight", (vocab, d_model))
    final_norm = plain("model.norm.weight", (d_model,))
    if tied:
        if "lm_head.weight" in st:        # some tied checkpoints carry a duplicate;
            used.add("lm_head.weight")    # HF replaces it with the embedding on load
        # int8: the int8 decode path needs its own head matrix -> untied int8(wte^T).
        lm_head = Trans(st, "model.embed_tokens.weight") if int8 else None
    else:
        if "lm_head.weight" not in st:
            sys.exit("error: tie_word_embeddings is false but the checkpoint has no lm_head.weight")
        lm_head = trans("lm_head.weight", vocab, d_model)

    # Every checkpoint tensor must have landed somewhere (rotary caches aside).
    left = sorted(n for n in st.names() if n not in used and not n.endswith("rotary_emb.inv_freq"))
    if left:
        sys.exit(f"error: {len(left)} checkpoint tensor(s) the engine has no slot for "
                 f"(refusing to drop them silently): {left[:6]}")

    windows = hf_windows(cfg, n_layers)
    ints = (vocab, d_model, n_layers, n_heads, n_kv, head_dim, d_ff, max_seq)
    write_tied = tied and not int8
    write_llma(out, ints, (rope_base, eps), build_order(wte, layers, final_norm, lm_head),
               int8=int8, windows=windows, qk_norm=qk_norm, tied=write_tied)
    n_params = vocab * d_model * (1 if write_tied else 2) + d_model + n_layers * (
        2 * d_model + d_model * (qd + 2 * kvd) + qd + 2 * kvd + qd * d_model + 3 * d_model * d_ff
        + (2 * head_dim if qk_norm else 0))
    print(f"wrote {out} ({'int8/LLMQ' if int8 else 'f32/LLMA'}): vocab={vocab} d_model={d_model} "
          f"layers={n_layers} heads={n_heads} kv={n_kv} head_dim={head_dim} d_ff={d_ff} ctx={max_seq} "
          f"rope_base={rope_base} eps={eps} tied={tied}{' (written untied for int8)' if tied and int8 else ''} "
          f"windows={windows if any(windows) else 'none'} qk_norm={qk_norm} "
          f"params={n_params / 1e6:.1f}M (f32 {n_params * 4 / 1e9:.2f} GB)")
    # Windowed layers keep a W-slot ring; full-attention layers hold all max_seq.
    kv_gb = sum(2 * (min(w, max_seq) if w > 0 else max_seq) * kvd * 4 for w in windows) / 1e9
    if kv_gb > 2:
        print(f"warning: the C KV cache for ctx={max_seq} is ~{kv_gb:.1f} GB; "
              f"consider re-running with --max-seq 4096")


def write_tiny(out, window=0, global_every=0, qk_norm=False, tied=False):
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
    lm_head = r(d_model, vocab)          # drawn either way, so the rng stream is unchanged
    write_llma(out, ints, (10000.0, 1e-5),
               build_order(wte, layers, np.ones(d_model, "<f4"), None if tied else lm_head),
               windows=windows, qk_norm=qk_norm, tied=tied)
    print(f"wrote tiny {out}: vocab={vocab} d_model={d_model} layers={n_layers} "
          f"heads={n_heads} kv={n_kv} head_dim={head_dim} windows={windows if any(windows) else 'none'} "
          f"qk_norm={qk_norm} tied={tied}")


def main():
    ap = argparse.ArgumentParser(description="HF LLaMA-family -> LLMA checkpoint")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M", help="HF repo id or local HF-format dir")
    ap.add_argument("--out", default="model.bin", help="output LLMA checkpoint")
    ap.add_argument("--int8", action="store_true", help="int8-quantize 2D weights (LLMQ, ~4x smaller)")
    ap.add_argument("--max-seq", type=int, default=None,
                    help="cap the context length (the C KV cache is sized for all of it)")
    ap.add_argument("--tiny", metavar="PATH", help="write a small deterministic checkpoint (no download) and exit")
    ap.add_argument("--window", type=int, default=0, help="--tiny only: sliding window W (0 = none)")
    ap.add_argument("--global-every", type=int, default=0, help="--tiny only: every Nth layer global")
    ap.add_argument("--qk-norm", action="store_true", help="--tiny only: include Qwen3 QK-norm gains (v4 file)")
    ap.add_argument("--tied", action="store_true", help="--tiny only: tied head (v4 flag bit 2, no lm_head)")
    args = ap.parse_args()
    if args.tiny:
        write_tiny(args.tiny, args.window, args.global_every, qk_norm=args.qk_norm, tied=args.tied)
    else:
        convert_hf(args.model, args.out, int8=args.int8, max_seq_cap=args.max_seq)


if __name__ == "__main__":
    main()
