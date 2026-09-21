#!/usr/bin/env python3
"""Train/eval contamination report for prompt<TAB>response files.

An eval file built from the same templates and word pool as the training corpus
measures RECALL, not generalization. This reports how much an eval file overlaps
the training files, four ways:

  goal-13gram   share of eval rows whose GOAL shares at least one word 13-gram
                with any training goal (goals shorter than 13 words: the whole
                goal occurring inside a training goal counts). The standard
                decontamination test, restricted to the natural-language part.
  row-13gram    the same over the whole row (transcript + response). Informative
                only: fixed tool markup and canned observations ("On branch
                agent-1 ...") make this high even for disjoint lexicons.
  skeleton      share of eval rows whose ARGUMENT-MASKED goal skeleton
                ("create F with W, then run C") also occurs in training -- i.e.
                the eval phrasing is a training template with new words.
  skeleton-6g   share of eval rows whose masked skeleton shares a 6-token chunk
                with a training skeleton ("f with w then run c"): phrasing reuse
                below whole-template level, which grammar-built goals need.

plus lexical overlap: share of eval filenames / write contents / commands that
also appear in training rows.

    python tools/overlap.py --eval out/eval_grpo.txt out/train.txt [more ...]

clean_sft.py --decontam uses the same functions to DROP training rows.
Deterministic; stdlib only.
"""
import argparse
import re
import sys

EXT_RE = re.compile(r"[\w./-]*\.(?:txt|md|log|json|csv)\b", re.IGNORECASE)
PATH_RE = re.compile(r'path="([^"]*)"')
CMD_RE = re.compile(r'command="([^"]*)"')
BODY_RE = re.compile(r'name="write_file"[^>]*>(.*?)</tool>', re.DOTALL)
TOK_RE = re.compile(r"[a-z0-9]+")
GOAL_END = re.compile(r" \| (?=MODEL: )")
MIN_SHORT = 6          # a goal shorter than n must have >= this many tokens to match


def read_rows(path):
    rows = []
    for line in open(path, encoding="utf-8-sig"):
        line = line.rstrip("\n")
        if "\t" in line:
            p, r = line.split("\t", 1)
            rows.append((p, r))
    return rows


def goal_of(prompt):
    """The natural-language task: the GOAL turn, or the whole prompt (QA rows)."""
    g = GOAL_END.split(prompt, 1)[0]
    return g[len("GOAL: "):] if g.startswith("GOAL: ") else g


def toks(s):
    return tuple(TOK_RE.findall(s.lower()))


def ngrams(t, n):
    return set(t[i:i + n] for i in range(len(t) - n + 1))


class NgramIndex:
    """Word n-grams of a reference set (the EVAL side), queried by candidates."""

    def __init__(self, texts, n=13):
        self.n = n
        self.grams = set()
        self.short = {}              # len -> whole token tuples of short references
        self._t = [toks(x) for x in texts]
        self._k = {}                 # k -> set of k-grams (lazy, k < n)
        for t in self._t:
            if len(t) >= n:
                self.grams |= ngrams(t, n)
            elif len(t) >= MIN_SHORT:
                self.short.setdefault(len(t), set()).add(t)

    def _kgrams(self, k):
        if k not in self._k:
            s = set()
            for t in self._t:
                s |= ngrams(t, k)
            self._k[k] = s
        return self._k[k]

    def hits(self, text):
        t = toks(text)
        if len(t) >= self.n:
            if ngrams(t, self.n) & self.grams:
                return True
        elif len(t) >= MIN_SHORT and t in self._kgrams(len(t)):
            return True                              # whole short text inside a ref
        for k, refs in self.short.items():            # short ref inside this text
            if k <= len(t) and ngrams(t, k) & refs:
                return True
        return False


def row_args(prompt, resp):
    """(files, contents, commands) the row's tool calls operate on."""
    text = prompt + " " + resp
    files = set(PATH_RE.findall(text)) | set(EXT_RE.findall(goal_of(prompt)))
    contents = set(b for b in BODY_RE.findall(text) if b.strip())
    cmds = set(c for c in CMD_RE.findall(text) if c.split()[:1] not in (["rm"], ["ls"]))
    return files, contents, cmds


def group_args(rows):
    """goal -> union of args over every row with that goal (a full SFT chain
    names each content only in the step that writes it)."""
    out = {}
    for p, r in rows:
        g = goal_of(p)
        f, c, m = row_args(p, r)
        a = out.setdefault(g, [set(), set(), set()])
        a[0] |= f
        a[1] |= c
        a[2] |= m
    return out


_ALT = {}


def _sub(text, values, ph):
    """Replace every value (whole-token, longest first) with `ph` in one pass."""
    vals = tuple(sorted(set(v.strip().lower() for v in values if v.strip()),
                        key=lambda v: (-len(v), v)))
    if not vals:
        return text
    rx = _ALT.get(vals)
    if rx is None:
        rx = re.compile(r"(?<![0-9a-z_])(?:" + "|".join(re.escape(v) for v in vals)
                        + r")(?![0-9a-z_])")
        if len(vals) > 50:           # only the big lexicon alternations are reused
            _ALT[vals] = rx
    return rx.sub(ph, text)


PH = {"C": "\x01c\x01", "F": "\x01f\x01", "W": "\x01w\x01"}


def skeleton(goal, args, lexicon=((), ())):
    """Argument-masked goal: filenames -> F, contents -> W, commands -> C.

    `args` are the row's own tool arguments; `lexicon` = (content phrases,
    commands) also masks values the row never passes to a tool (a later clause)."""
    files, contents, cmds = args
    lex_words, lex_cmds = lexicon or ((), ())
    s = goal.lower()
    s = _sub(s, cmds, PH["C"])
    s = _sub(s, lex_cmds, PH["C"])
    s = _sub(s, files, PH["F"])
    s = EXT_RE.sub(PH["F"], s)
    s = _sub(s, contents, PH["W"])
    s = _sub(s, lex_words, PH["W"])
    for k, v in PH.items():
        s = s.replace(v, k)
    s = re.sub(r"\d+", "0", s)
    return re.sub(r"\s+", " ", s).strip()


def skeletons(rows, lexicon=((), ())):
    ga = group_args(rows)
    return [skeleton(goal_of(p), ga[goal_of(p)], lexicon) for p, _ in rows]


def load_lexicon(wiki=None, commands=None):
    """Optional extra masking vocabulary: (content phrases, commands)."""
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_agent_traj import WORDS, commands_from, contents_from, slugs_from_wiki
    words = (set(contents_from(slugs_from_wiki(wiki, limit=100000))) - set(WORDS)) if wiki else set()
    cmds = set(commands_from(commands, limit=100000)) if commands else set()
    return words, cmds


def report(eval_rows, train_rows, n=13, lexicon=((), ()), show=3):
    out = []
    goals_e = [goal_of(p) for p, _ in eval_rows]
    goals_t = [goal_of(p) for p, _ in train_rows]
    # goal-level n-gram: index TRAIN, query each eval goal
    idx = NgramIndex(set(goals_t), n)
    g_hit = [idx.hits(g) for g in goals_e]
    ridx = NgramIndex(set(p + " " + r for p, r in train_rows), n)
    r_hit = [ridx.hits(p + " " + r) for p, r in eval_rows]
    exact = set(p for p, _ in train_rows)
    e_exact = sum(1 for p, _ in eval_rows if p in exact)
    sk_t = set(skeletons(train_rows, lexicon))
    sk_e = skeletons(eval_rows, lexicon)
    s_hit = [s in sk_t for s in sk_e]
    # phrasing CHUNKS: masked-skeleton k-grams (exact skeleton match is too
    # strict once goals come from a grammar -- every goal is a new combination)
    k = 6
    kt = set()
    for x in sk_t:
        kt |= ngrams(toks(x), k)
    k_hit = [bool(ngrams(toks(x), k) & kt) for x in sk_e]
    lf = [set(), set(), set()]
    for p, r in train_rows:
        for i, a in enumerate(row_args(p, r)):
            lf[i] |= a
    le = [set(), set(), set()]
    for p, r in eval_rows:
        for i, a in enumerate(row_args(p, r)):
            le[i] |= a
    ne = max(1, len(eval_rows))
    out.append("eval rows %d  |  train rows %d  |  n = %d" % (len(eval_rows), len(train_rows), n))
    out.append("  exact prompt overlap : %5d  (%.1f%%)" % (e_exact, 100.0 * e_exact / ne))
    out.append("  goal-%dgram overlap   : %5d  (%.1f%%)" % (n, sum(g_hit), 100.0 * sum(g_hit) / ne))
    out.append("  row-%dgram overlap    : %5d  (%.1f%%)   [includes tool markup / canned obs]"
               % (n, sum(r_hit), 100.0 * sum(r_hit) / ne))
    out.append("  skeleton overlap     : %5d  (%.1f%%)   eval skeletons %d distinct, train %d"
               % (sum(s_hit), 100.0 * sum(s_hit) / ne, len(set(sk_e)), len(sk_t)))
    out.append("  skeleton-%dgram      : %5d  (%.1f%%)   [shared masked phrasing chunk]"
               % (k, sum(k_hit), 100.0 * sum(k_hit) / ne))
    for i, name in enumerate(("filenames", "contents", "commands")):
        e, t = le[i], lf[i]
        out.append("  lexical %-10s : %d/%d eval values seen in train (%.1f%%)"
                   % (name, len(e & t), len(e), 100.0 * len(e & t) / max(1, len(e))))
    shown = 0
    for (p, _), h, s, sk in zip(eval_rows, g_hit, s_hit, sk_e):
        if (h or s) and shown < show:
            out.append("    e.g. %s%s  %s" % ("[13g]" if h else "", "[skel]" if s else "",
                                              sk[:110]))
            shown += 1
    return "\n".join(out), {"exact": e_exact, "goal": sum(g_hit), "row": sum(r_hit),
                            "skeleton": sum(s_hit), "skel_kgram": sum(k_hit), "n_eval": len(eval_rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train", nargs="+", help="training prompt<TAB>response files")
    ap.add_argument("--eval", required=True, help="the eval/task file to check")
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--wiki", default=None,
                    help="also mask every wiki-derived content phrase in skeletons")
    ap.add_argument("--commands", default=None, help="also mask every command of this pool")
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()
    ev = read_rows(args.eval)
    tr = []
    for p in args.train:
        tr += read_rows(p)
    text, _ = report(ev, tr, args.n, load_lexicon(args.wiki, args.commands), args.show)
    print(text)


if __name__ == "__main__":
    main()
