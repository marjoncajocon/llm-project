#!/usr/bin/env python3
"""Export a model's tokenizer to a compact .tok file the C engine reads — so the
engine tokenizes on its own with NO Python at runtime (this export is a one-time
setup step, like the weight converters).

    python tools/export_tokenizer.py gpt2 model/gpt2/tok.hf
    python tools/export_tokenizer.py <hf-repo> <out.tok>     # LLaMA-family (tokenizer.json)

.tok format v2 (little-endian):
    "HFTK", uint32 version=2, uint32 n_vocab, int32 eos_id,
    then n_vocab entries: uint16 len, <len> raw bytes  (token id -> its bytes)
    --- v2 tail ---
    int32 bos_id, uint8 template_kind, uint16 sys_len, <sys_len> default_system,
    uint32 n_special, then per special: int32 id, uint8 eog, uint16 len, <len> str
Encoding uses the byte-level-BPE rule "merge the adjacent pair whose combined
bytes have the lowest id"; special-token strings are emitted as their single id.
"""
import json
import struct
import sys

TMPL_PLAIN, TMPL_CHATML, TMPL_LLAMA3 = 0, 1, 2

# End-of-generation token strings (stop generation when the model emits one).
EOG_STRINGS = {"<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>",
               "</s>", "<end_of_utterance>", "<|end|>"}

# Model persona defaults (chat behaves better with the system prompt it was tuned on).
DEFAULT_SYSTEM = {
    TMPL_CHATML: "You are a helpful assistant.",
    TMPL_LLAMA3: "You are a helpful assistant.",
    TMPL_PLAIN: "",
}


def write_tok(path, n_vocab, eos_id, id_to_bytes,
              specials, template_kind, bos_id, default_system):
    sysb = default_system.encode("utf-8")
    with open(path, "wb") as f:
        f.write(b"HFTK")
        f.write(struct.pack("<II", 2, n_vocab))
        f.write(struct.pack("<i", eos_id))
        for i in range(n_vocab):
            b = id_to_bytes(i)
            f.write(struct.pack("<H", len(b)))
            f.write(b)
        # v2 tail
        f.write(struct.pack("<i", bos_id))
        f.write(struct.pack("<B", template_kind))
        f.write(struct.pack("<H", len(sysb)))
        f.write(sysb)
        f.write(struct.pack("<I", len(specials)))
        for sid, eog, s in specials:
            sb = s.encode("utf-8")
            f.write(struct.pack("<iBH", sid, 1 if eog else 0, len(sb)))
            f.write(sb)
    print("wrote %s: %d tokens, eos=%d, specials=%d, template=%d, bos=%d"
          % (path, n_vocab, eos_id, len(specials), template_kind, bos_id))


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
    # GPT-2 has one special (<|endoftext|>) and no chat template.
    specials = [(eos, True, "<|endoftext|>")]
    write_tok(out, n, eos, id_to_bytes, specials, TMPL_PLAIN, -1, "")


def export_hf(repo, out):
    # LLaMA-family (SmolLM2, Qwen, ...): load tokenizer.json and recover each
    # token's raw bytes from the HF byte-level encoding.
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download

    def fetch(name):
        try:
            return hf_hub_download(repo, name, local_files_only=True)
        except Exception:
            try:
                return hf_hub_download(repo, name)
            except Exception:
                return None

    tok_path = fetch("tokenizer.json")
    tok = Tokenizer.from_file(tok_path)
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

    # Special/added tokens: read them straight from tokenizer.json (has id +
    # content + special flag). We match them greedily in encode, so restrict to
    # ones that begin with '<' (the C fast-path only probes at '<').
    with open(tok_path, "r", encoding="utf-8") as f:
        tj = json.load(f)
    added = tj.get("added_tokens", []) or []
    specials = []
    seen = set()
    for a in added:
        content = a.get("content", "")
        sid = a.get("id", -1)
        if not content.startswith("<") or sid in seen or not (0 <= sid < n):
            continue
        seen.add(sid)
        specials.append((sid, content in EOG_STRINGS, content))

    # Chat template kind: detect from the template / present special tokens.
    cfg_path = fetch("tokenizer_config.json")
    cfg = {}
    if cfg_path:
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    chat_template = cfg.get("chat_template", "") or ""
    if isinstance(chat_template, list):  # some configs store a list of templates
        chat_template = " ".join(str(x) for x in chat_template)
    have = {c for _, _, c in specials}
    if "<|start_header_id|>" in chat_template or "<|start_header_id|>" in have:
        kind = TMPL_LLAMA3
    elif "<|im_start|>" in chat_template or "<|im_start|>" in have:
        kind = TMPL_CHATML
    else:
        kind = TMPL_PLAIN

    # Generation stop token: the chat end marker for this template, else base eos.
    def tid(s):
        return tok.token_to_id(s)
    eos = None
    for cand in (("<|im_end|>",) if kind == TMPL_CHATML else
                 ("<|eot_id|>",) if kind == TMPL_LLAMA3 else ()) + ("<|endoftext|>", "</s>"):
        if tid(cand) is not None:
            eos = tid(cand)
            break
    # BOS only if the config actually adds one (Qwen/SmolLM ChatML do not).
    bos = None
    if cfg.get("add_bos_token"):
        bt = cfg.get("bos_token")
        if isinstance(bt, dict):
            bt = bt.get("content")
        if isinstance(bt, str):
            bos = tid(bt)

    write_tok(out, n, eos if eos is not None else -1, id_to_bytes,
              specials, kind, bos if bos is not None else -1,
              DEFAULT_SYSTEM.get(kind, ""))


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
