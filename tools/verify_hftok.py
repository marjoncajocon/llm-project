#!/usr/bin/env python3
"""Verify the C tokenizer (hftok.c) against HF `tokenizers`, id for id.

    python tools/verify_hftok.py <tokenizer.json> <file.tok> [--exe ./transformer.exe]
                                 [--extra FILE ...] [--extra-lines N] [--show N]

Builds a corpus of >= 2000 texts -- the repo's own C/Python sources (code),
notes/README prose, synthetic lines packed with the known-tricky cases (code
punctuation, numbers, contractions in every case, Unicode letters/marks/digits
from many scripts, emoji/ZWJ, whitespace runs and every newline flavour),
whitespace-before-number / indented-code lines (the SmolLM2 Digits case) and
multi-line blocks -- encodes each with the reference tokenizer
(add_special_tokens=False) and with `transformer hftok <file.tok> encode-hex`,
and reports every mismatch (plus the reference pretoken split for the first few).
All corpus text is NFC-normalized first: the reference normalizes to NFC, the C
engine does not (see the NFC probe at the end for what that costs).
Exit status 0 iff every text matches.
"""
import argparse
import glob
import os
import random
import subprocess
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TRICKY = [
    "(self", ".foo", "x+=1", "a.b.c()", "foo_bar(baz)", "if(x){y=2;}", "arr[i+1]",
    "->next", "::std", "#include <stdio.h>", "//comment", "/* c */", "@decorator",
    "$HOME", "~/.bashrc", "C:\\Users\\x", "http://a.b/c?d=1&e=2", "'quoted'", '"dq"',
    "12345", "3.14", "1,000,000", "0x1F", "1e-5", "v2.0.1", "2026-09-21", "10:30pm",
    "-42", "+7", "$9.99", "50%", "1st", "2nd", "３４５", "٣٤", "१२३", "½", "Ⅻ", "²",
    "don't", "DON'T", "I'm", "I'M", "we're", "WE'RE", "they've", "You'll", "He'D",
    "it's", "IT'S", "o'clock", "rock'n'roll", "''", "'s", "'S", "' s", "'ſ", "'Re",
    "café", "naïve", "Straße", "ÉCOLE", "Ωμέγα", "Москва", "中文", "日本語テキスト",
    "한국어", "العربية", "עברית", "हिन्दी", "ภาษาไทย", "ქართული", "Ελληνικά",
    "😀", "👍🏽", "👨‍👩‍👧", "🇵🇭", "❤️", "™", "©", "→", "∑x", "√2", "€5", "¥", "…",
    "\u00a0nbsp", "\u3000全角", "\u2003em", "\u200bzw", "\u2028ls", "\ufeffbom",
    "\t", "\t\t", "  ", "    ", "\n", "\n\n", "\r\n", "\r\n\r\n", " \n", "\n ", " \n ",
    "\t\n", "\n\t", "  \n\n  ", "\f", "\v", "\x85",
]
WORDS = ["the", "The", "value", "return", "print", "def", "int", "HTTP", "json",
         "Hello", "world", "über", "мир", "世界", "x", "y", "i", "self", "None"]
SEPS = [" ", "", "  ", "\t", "\n", ".", ",", "(", ")", ":", ";", "=", " = ", "\n\n",
        "    ", "->", "'", "\"", "!", "?", "_", "-", "/", "\r\n", " \n", "\n    "]


def synth(rng, n):
    out = []
    for _ in range(n):
        k = rng.randint(1, 12)
        parts = []
        for _ in range(k):
            r = rng.random()
            parts.append(rng.choice(TRICKY) if r < 0.45 else
                         rng.choice(WORDS) if r < 0.75 else
                         str(rng.randint(0, 10 ** rng.randint(1, 7))) if r < 0.85 else "")
            parts.append(rng.choice(SEPS))
        out.append("".join(parts))
    return out


# Whitespace runs in front of numbers: where a Digits(individual) pre-step (SmolLM2)
# changes the GPT-2 split -- the piece before a digit ends at the digit, so
# "x  1" is x|'  '|1 there, x|' '|' 1' under the plain GPT-2 rule.
WS = [" ", "  ", "   ", "    ", "        ", "\t", "\t\t", "\n", "\n    ", "  \n  ", " \t ",
      "\r\n  ", "\u00a0", "\u00a0 ", "\u3000\u3000", " \u2003", "\n\n\t"]
NUMS = ["0", "1", "42", "3.14", "1e-5", "0x1F", "2026", "-7", "+8", "１２", "٣٤", "१२", "½", "Ⅻ",
        "²", "10:30", "1,000", "v2", "1st", "9'", "7's"]
LEADS = ["", "x", "return", "=", "(", "[", "arr[", "#", "-", "*", "//", "'", "\"", "if", "é", "中"]


def synth_indent(rng, n):
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(1, 6)):
            parts += [rng.choice(LEADS), rng.choice(WS), rng.choice(NUMS)]
            if rng.random() < 0.5:
                parts.append(rng.choice(WS + [".", ",", ";", ")", "]", " ", ""]))
            if rng.random() < 0.3:
                parts.append(rng.choice(WORDS))
        out.append("".join(parts))
    # code-shaped lines: indentation (spaces/tabs) then a statement with numbers
    for _ in range(n // 2):
        ind = rng.choice(["", " ", "  ", "    ", "      ", "        ", "\t", "\t\t", " \t"])
        num = rng.choice(NUMS)
        stmt = rng.choice(["%s", "x = %s", "return %s;", "%s. item", "arr[%s] = 0", "- %s",
                           "case %s:", "{%s, %s}", "  %s", "%s  ", "f(%s,  %s)", "# %s"])
        out.append(ind + stmt.replace("%s", num) + rng.choice(["", "\n", "\n" + ind, "  "]))
    return out


def corpus(extra, extra_lines=200):
    texts = list(TRICKY)
    files = sorted(glob.glob(os.path.join(ROOT, "*.c")))[:12] + \
        sorted(glob.glob(os.path.join(ROOT, "tools", "*.py")))[:8] + \
        [os.path.join(ROOT, n) for n in ("notes.md", "README.md", "data/sample.txt")] + list(extra)
    for fn in files:
        if not os.path.exists(fn):
            continue
        with open(fn, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().split("\n")
        cap = 200 if fn not in extra else extra_lines
        texts += [ln for ln in lines[:max(250, cap)] if ln.strip()][:cap]   # single lines
        for i in range(0, min(len(lines), 400), 17):                   # multi-line blocks
            texts.append("\n".join(lines[i:i + 9]) + "\n")
    rng = random.Random(1234)
    texts += synth(rng, 1500)
    texts += synth_indent(rng, 1500)
    # Chars whose bytes some vocabs lack as single tokens (SmolLM2: 0x04 0x06 0x13
    # 0x14 0x16 0x1D, and F1/F2 lead bytes = U+40000..U+BFFFF): HF drops them and
    # does not merge across the gap.
    odd = ["\x04", "\x06", "\x13", "\x14", "\x16", "\x1d", "\U00040000", "\U000a1234"]
    for _ in range(300):
        texts.append("".join(rng.choice(odd + WORDS + [" ", "  ", "1", "the", "\n"])
                             for _ in range(rng.randint(1, 8))))
    # Unicode-table fuzz: random assigned code points from the whole range (BMP
    # heavy), mixed with ASCII/space, to probe \p{L}/\p{N}/\s coverage.
    for _ in range(600):
        cs = []
        for _ in range(rng.randint(1, 24)):
            r = rng.random()
            cp = (rng.randint(0x20, 0x7E) if r < 0.3 else rng.choice([0x20, 0x0A, 0x09, 0x3000, 0xA0])
                  if r < 0.4 else rng.randint(0xA0, 0xFFFD) if r < 0.9 else rng.randint(0x10000, 0x3134A))
            if 0xD800 <= cp <= 0xDFFF or unicodedata.category(chr(cp)) in ("Cn", "Cs", "Co"):
                continue
            cs.append(chr(cp))
        texts.append("".join(cs))
    texts =[unicodedata.normalize("NFC", t) for t in texts if t]
    return texts


def run_c(exe, tokf, texts, op="encode-hex"):
    inp = "\n".join(t.encode("utf-8").hex() for t in texts) + "\n"
    r = subprocess.run([exe, "hftok", tokf, op], input=inp.encode("ascii"),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    lines = r.stdout.decode("ascii").replace("\r", "").split("\n")
    return [[int(x) for x in ln.split()] for ln in lines[:len(texts)]]


def impact(a, tk):
    """Per data file: lines whose prompt/response ids differ between two .tok files
    (fields split at the first TAB, a trailing CR dropped, empty fields skipped --
    as train.c / grpo.c do), and which of the two agrees with HF on those lines."""
    if not a.vs:
        print("error: --impact needs --vs OTHER.tok")
        return 2
    for fn in a.impact:
        with open(fn, "r", encoding="utf-8", errors="replace", newline="") as f:
            lines = f.read().split("\n")
        fields, owner = [], []
        for li, ln in enumerate(lines):
            if ln.endswith("\r"):
                ln = ln[:-1]
            if "\t" not in ln:
                continue
            p, r = ln.split("\t", 1)
            if p and r:
                fields += [p, r]
                owner += [li, li]
        ga = run_c(a.exe, a.tok, fields)
        gb = run_c(a.exe, a.vs, fields)
        diff_lines, diff_fields, a_hf, b_hf = set(), 0, 0, 0
        for i, t in enumerate(fields):
            if ga[i] != gb[i]:
                diff_fields += 1
                diff_lines.add(owner[i])
                ref = tk.encode(t, add_special_tokens=False).ids
                a_hf += ga[i] == ref
                b_hf += gb[i] == ref
        print("%s: %d example lines, %d differ (%d of %d fields; tokens %d vs %d); of the "
              "differing fields HF agrees with %s on %d, with %s on %d"
              % (fn, len(set(owner)), len(diff_lines), diff_fields, len(fields),
                 sum(map(len, ga)), sum(map(len, gb)),
                 os.path.basename(a.tok), a_hf, os.path.basename(a.vs), b_hf))
        shown = 0
        for i in range(len(fields)):
            if shown < a.show and ga[i] != gb[i]:
                j = next(k for k in range(len(ga[i])) if k >= len(gb[i]) or ga[i][k] != gb[i][k])
                print("    line %d, first difference in %r" % (
                    owner[i] + 1, tk.decode(ga[i][max(0, j - 4):j + 4])))
                shown += 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer_json")
    ap.add_argument("tok")
    ap.add_argument("--exe", default=os.path.join(ROOT, "transformer.exe"))
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--extra-lines", type=int, default=200,
                    help="non-empty single lines taken from each --extra file")
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--impact", nargs="*", default=None, metavar="DATA",
                    help="instead of the corpus check: for each 'prompt<TAB>response' line of "
                         "the DATA files (as llama-train --sft / grpo read them), count lines "
                         "whose fields tokenize differently under <file.tok> vs --vs")
    ap.add_argument("--vs", help="second .tok for --impact (e.g. the old kind-0 export)")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(a.tokenizer_json)

    if a.impact is not None:
        sys.exit(impact(a, tk))
    texts = corpus(a.extra, a.extra_lines)
    ref =[tk.encode(t, add_special_tokens=False).ids for t in texts]
    got = run_c(a.exe, a.tok, texts)
    bad = [i for i in range(len(texts)) if ref[i] != got[i]]
    ntok = sum(len(r) for r in ref)
    print("texts: %d (%d lines incl. multi-line blocks), reference tokens: %d"
          % (len(texts), sum(t.count("\n") + 1 for t in texts), ntok))
    print("mismatches: %d" % len(bad))
    for i in bad[:a.show]:
        t = texts[i]
        pre = [p for p, _ in tk.pre_tokenizer.pre_tokenize_str(tk.normalizer.normalize_str(t)
                                                              if tk.normalizer else t)]
        print("  text %r\n    ref %s\n    got %s\n    ref pretokens %r" % (t[:120], ref[i][:40], got[i][:40], pre[:30]))

    # Pretoken boundaries too (stricter than ids: two different splits can BPE to
    # the same ids). Reference offsets are in characters of the normalized text.
    def ref_split(t):
        return [len(t[s:e].encode("utf-8")) for _, (s, e) in tk.pre_tokenizer.pre_tokenize_str(t)]
    rs = [ref_split(t) for t in texts]
    gs = run_c(a.exe, a.tok, texts, "split-hex")
    sbad = [i for i in range(len(texts)) if rs[i] != gs[i]]
    print("pretoken-split mismatches: %d (of %d pretokens)" % (len(sbad), sum(len(r) for r in rs)))
    for i in sbad[:a.show]:
        print("  text %r\n    ref %s\n    got %s" % (texts[i][:120], rs[i][:40], gs[i][:40]))

    # NFC probe (informational): decomposed input, which only the reference normalizes.
    probe = [unicodedata.normalize("NFD", s) for s in ("café", "naïve", "Ωμέγα", "ÉCOLE")]
    pr = [tk.encode(t, add_special_tokens=False).ids for t in probe]
    pg = run_c(a.exe, a.tok, probe)
    print("NFD probe (C does not normalize): %d/%d differ" % (sum(x != y for x, y in zip(pr, pg)), len(probe)))
    sys.exit(1 if bad or sbad else 0)


if __name__ == "__main__":
    main()
