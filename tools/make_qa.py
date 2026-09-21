#!/usr/bin/env python3
"""Turn crawled Wikipedia articles into short general-knowledge SFT pairs.

The tool-call SFT (tools/make_sft.py) narrows the model so hard that it answers
EVERY prompt with a tool call — it forgets how to just talk. Mixing in plain
"question -> short answer" pairs (this file) restores that, and teaches the model
the BOUNDARY: emit a tool call for an instruction, answer in prose for a question.

Each article's first 1-2 sentences are a natural definition, so its title becomes
a question:  "what is <title>?"  ->  "<first sentences>".

    python tools/make_qa.py --wiki data/wiki --out data/qa.txt --n 400

Output matches the SFT loader: one "prompt<TAB>response" line per pair. Responses
are plain prose (no <tool>/<done>), so a model trained on make_sft.py + make_qa.py
learns to route instructions to tools AND answer questions as text.
"""
import argparse
import os
import re
import random

# Title-ish filename -> readable phrase: "16-bit_computing" -> "16 bit computing".
def title_from_name(fn):
    base = fn[:-4] if fn.endswith(".txt") else fn
    base = base.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", base).strip()


# Parenthetical disambiguation in Wikipedia titles ("Tree (data structure)") is
# noise for a question; drop it.
def clean_title(t):
    return re.sub(r"\s*\(.*?\)\s*", "", t).strip()


SENT_END = re.compile(r"(?<=[.!?])\s+")


def first_sentences(text, k=2, max_len=280):
    """First k sentences of the lead paragraph, trimmed to max_len chars."""
    lead = text.strip().split("\n\n", 1)[0].strip()
    lead = re.sub(r"\s+", " ", lead)
    parts = SENT_END.split(lead)
    out = " ".join(parts[:k]).strip()
    if len(out) > max_len:
        cut = out[:max_len]
        # end on the last sentence boundary within the budget, else hard cut
        m = list(re.finditer(r"[.!?]", cut))
        out = cut[:m[-1].end()] if m else cut.rstrip() + "."
    return out


Q_TMPL = ["what is {t}", "what is {t}?", "tell me about {t}", "explain {t}",
          "describe {t}", "what does {t} mean", "give me a summary of {t}"]


def usable(ans):
    # Skip stubs, list-dumps, and lead paragraphs that are mostly non-prose.
    if len(ans) < 40 or len(ans) > 300:
        return False
    if ans.count("\t") or ans.count("  "):
        return False
    letters = sum(c.isalpha() for c in ans)
    return letters >= 0.6 * len(ans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", required=True, help="dir of crawled Wikipedia *.txt")
    ap.add_argument("--out", default="data/qa.txt")
    ap.add_argument("--n", type=int, default=400, help="max pairs to emit")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--holdout-lexicon", choices=["train", "eval"], default=None,
                    help="keep only articles whose filename slug is on this side of the "
                         "agent_lexicon split (so QA does not leak eval-only words)")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = sorted(f for f in os.listdir(args.wiki) if f.endswith(".txt"))
    rng.shuffle(files)
    if args.holdout_lexicon:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from agent_lexicon import side_of
        # the same slug make_agent_traj.slugs_from_wiki derives from the filename
        files = [f for f in files if side_of(
            re.sub(r"[^A-Za-z0-9]+", "_", f[:-4].lower()).strip("_")[:20],
            args.holdout_frac) == args.holdout_lexicon]

    rows, skipped = [], 0
    for fn in files:
        if len(rows) >= args.n:
            break
        title = clean_title(title_from_name(fn))
        if not title or len(title) > 48:
            continue
        try:
            text = open(os.path.join(args.wiki, fn), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        ans = first_sentences(text)
        if not usable(ans):
            skipped += 1
            continue
        q = rng.choice(Q_TMPL).format(t=title)
        rows.append((q.replace("\t", " "), ans.replace("\t", " ")))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for q, a in rows:
            f.write("%s\t%s\n" % (q, a))
    print("wrote %d QA pairs -> %s (skipped %d unusable leads)" % (len(rows), args.out, skipped))


main()
