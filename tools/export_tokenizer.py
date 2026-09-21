#!/usr/bin/env python3
"""Export a model's tokenizer to a compact .tok file the C engine reads — so the
engine tokenizes on its own with NO Python at runtime (this export is a one-time
setup step, like the weight converters).

    python tools/export_tokenizer.py gpt2 model/gpt2/tok.hf
    python tools/export_tokenizer.py <hf-repo> <out.tok>     # LLaMA-family (tokenizer.json)

.tok format v3 (little-endian):
    "HFTK", uint32 version=3, uint32 n_vocab, int32 eos_id,
    then n_vocab entries: uint16 len, <len> raw bytes  (token id -> its bytes)
    --- v2 tail ---
    int32 bos_id, uint8 template_kind, uint16 sys_len, <sys_len> default_system,
    uint32 n_special, then per special: int32 id, uint8 eog, uint16 len, <len> str
    --- v3 tail ---
    uint8 pretok_kind   (0 = GPT-2 regex, 1 = Qwen2/2.5/3 regex, 2 = SmolLM2 exact:
                         Digits(individual) then GPT-2 regex per piece; see PRE_* below)
v1/v2 files (no pretok byte) load as GPT-2; older C builds read v3 files too
(they stop after the specials), but would split Qwen text with the GPT-2 rule.
Encoding uses the byte-level-BPE rule "merge the adjacent pair whose combined
bytes have the lowest id"; special-token strings are emitted as their single id.
"""
import json
import struct
import sys

TMPL_PLAIN, TMPL_CHATML, TMPL_LLAMA3 = 0, 1, 2

# Pretokenizer kinds (hftok.h HFTOK_PRE_*).
PRE_GPT2, PRE_QWEN2, PRE_SMOL = 0, 1, 2
# The GPT-2 byte-level regex (what ByteLevel(use_regex=true) applies).
GPT2_REGEX = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# Qwen2 / Qwen2.5 / Qwen3 Split regex, character for character as in tokenizer.json.
QWEN2_REGEX = (r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"""
               r"""| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")


def pretok_kind(tj):
    r"""Map a tokenizer.json `pre_tokenizer` to a C pretokenizer kind, or exit if
    the engine cannot reproduce it (a wrong split silently changes every id).
      Split(Qwen2 regex) [+ ByteLevel(use_regex=false)]           -> PRE_QWEN2
      ByteLevel(use_regex=true) / Split(GPT-2 regex)              -> PRE_GPT2
      Digits(individual_digits=true) THEN ByteLevel(use_regex=true)
        (SmolLM2 family)                                          -> PRE_SMOL
    PRE_SMOL is HF's Sequence semantics exactly: every numeric char becomes its
    own piece first, then the GPT-2 regex runs on each remaining piece, so a
    piece's end is end-of-text for \s+(?!\S) ("x  1" -> x|'  '|1). Files written
    before PRE_SMOL existed carry PRE_GPT2 for SmolLM2, an approximation that
    differs when 2+ whitespace chars precede a digit (tools/verify_hftok.py: 353
    of 5283 corpus texts, mostly indented code); those files still load and
    tokenize exactly as before. Builds older than PRE_SMOL reject a kind-2 file
    (unknown kind) rather than mis-split it."""
    pt = tj.get("pre_tokenizer") or {}
    items = pt.get("pretokenizers", []) if pt.get("type") == "Sequence" else [pt]
    steps = []   # ordered: "digits", PRE_GPT2, PRE_QWEN2
    for it in items:
        ty = it.get("type")
        if ty == "Split":
            pat = (it.get("pattern") or {}).get("Regex")
            if pat == QWEN2_REGEX and it.get("behavior") == "Isolated" and not it.get("invert"):
                steps.append(PRE_QWEN2)
            elif pat == GPT2_REGEX and it.get("behavior") == "Isolated" and not it.get("invert"):
                steps.append(PRE_GPT2)
            else:
                sys.exit("error: unsupported Split pre-tokenizer regex %r "
                         "(the C engine knows the GPT-2 and Qwen2 rules only)" % pat)
        elif ty == "ByteLevel":
            if it.get("add_prefix_space"):
                sys.exit("error: ByteLevel(add_prefix_space=true) pre-tokenizer is unsupported "
                         "(the C engine does not prepend a space)")
            if it.get("use_regex", True):
                steps.append(PRE_GPT2)
        elif ty == "Digits":
            if not it.get("individual_digits"):
                sys.exit("error: Digits(individual_digits=false) pre-tokenizer is unsupported")
            steps.append("digits")
        else:
            sys.exit("error: unsupported pre-tokenizer %r" % ty)
    # The same regex split applied twice in a row is a no-op the second time
    # (e.g. Split(GPT-2 regex) + ByteLevel(use_regex=true)).
    steps = [x for i, x in enumerate(steps) if i == 0 or x != steps[i - 1]]
    # Digits before the Qwen2 regex is NOT the same as Qwen2 alone ("  1": the
    # piece end changes the \s+(?!\S) split), so it is rejected, not approximated.
    if steps == [PRE_GPT2]:
        kind = PRE_GPT2
    elif steps == [PRE_QWEN2]:
        kind = PRE_QWEN2
    elif steps == ["digits", PRE_GPT2]:
        kind = PRE_SMOL
    elif not [s for s in steps if s != "digits"]:
        sys.exit("error: no pre-tokenizer split found (%r)" % pt)
    else:
        sys.exit("error: unsupported pre-tokenizer combination %r" % pt)
    norm = tj.get("normalizer")
    if norm and norm.get("type") != "NFC":
        sys.exit("error: unsupported normalizer %r" % norm)
    if norm:
        print("note: tokenizer normalizes to NFC; the C engine does not normalize, "
              "so feed it NFC text (most text already is)")
    return kind

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
              specials, template_kind, bos_id, default_system, pretok=PRE_GPT2):
    sysb = default_system.encode("utf-8")
    with open(path, "wb") as f:
        f.write(b"HFTK")
        f.write(struct.pack("<II", 3, n_vocab))
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
        # v3 tail
        f.write(struct.pack("<B", pretok))
    print("wrote %s: %d tokens, eos=%d, specials=%d, template=%d, bos=%d, pretok=%d"
          % (path, n_vocab, eos_id, len(specials), template_kind, bos_id, pretok))


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

    import os
    if os.path.isdir(repo):   # a local directory holding tokenizer.json (+ config)
        def fetch(name):
            p = os.path.join(repo, name)
            return p if os.path.exists(p) else None
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
    pretok = pretok_kind(tj)
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
              DEFAULT_SYSTEM.get(kind, ""), pretok)


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: export_tokenizer.py <gpt2 | hf-repo | local-dir> <out.tok>\n")
        sys.exit(2)
    which, out = sys.argv[1], sys.argv[2]
    if which == "gpt2":
        export_gpt2(out)
    else:
        export_hf(which, out)


if __name__ == "__main__":
    main()
