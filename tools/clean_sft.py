#!/usr/bin/env python3
"""Make synthetic SFT data HIGH-QUALITY: verify + dedupe + balance, with a report.

The generators (make_sft / make_qa / make_agent_traj / mixes) can emit self-
INCONSISTENT pairs -- a tool call whose filename or content is NOT present in the
prompt, so the model would have to hallucinate it (the classic "second write is
empty" / hardcoded-content bug). They also repeat exact lines across runs, which
wastes training steps and inflates held-out scores. This tool is the quality gate
between generation and training:

  1. VERIFY  -- reject a pair whose tool ARGS are not grounded in the prompt text:
       * every  path="F"        value must appear in the prompt
       * a write_file's CONTENT must appear in the prompt (and be non-empty)
       * every filename-like token (name.ext) inside a run_command must appear
     Canonical command verbs (ls, git, rm, make ...) need NOT appear -- only the
     data they operate on. Prose (QA) and <done> responses have no args to ground
     and always pass this stage.
  2. DEDUPE  -- drop exact and whitespace/case-normalized duplicate lines
     (keep first). Deliberately CONSERVATIVE: filenames/words are NOT masked, so
     template variety (the point of the data) survives; only true repeats go.
  3. BALANCE -- classify each survivor by action type + task length and log the
     distribution. With --balance, cap any over-represented type to
     factor x median so one intent can't dominate; every drop is logged
     (no silent truncation).

    python tools/clean_sft.py --out data/clean.txt --report data/clean_report.txt \
        data/traj_reason.txt data/qa.txt data/toolcalls.txt

Deterministic given --seed. Output is the same prompt<TAB>response format the SFT
loader (train.c --sft) already reads, so it drops straight into training.
"""
import argparse
import random
import re

EXT_RE = re.compile(r"[\w./-]*\.(?:txt|md|log|json|csv)\b", re.IGNORECASE)
PATH_RE = re.compile(r'path="([^"]*)"')
CMD_RE = re.compile(r'command="([^"]*)"')
NAME_RE = re.compile(r'name="(\w+)"')
# write_file body: everything between the opening tag and </tool>
WRITE_BODY_RE = re.compile(r'name="write_file"[^>]*>(.*?)</tool>', re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def norm(s):
    """Lowercase + collapse whitespace -- the space in which we test grounding."""
    return re.sub(r"\s+", " ", s.lower()).strip()


def grounded(value, prompt_norm):
    """True if `value` occurs in the (already-normalized) prompt as a whole token.

    Token boundaries stop "ok" matching inside "cook" while still matching real
    filenames like "a.txt" (the dot is part of the escaped literal, and the chars
    around the whole value must be non-word)."""
    v = norm(value)
    if not v:
        return False
    pat = r"(?<![0-9a-z_])" + re.escape(v) + r"(?![0-9a-z_])"
    return re.search(pat, prompt_norm) is not None


def strip_reasoning(resp):
    """Drop <think>...</think> spans -- they are ephemeral rationale, not args."""
    return THINK_RE.sub("", resp).strip()


def verify(prompt, response):
    """Return None if the pair is self-consistent, else a short drop-reason."""
    resp = strip_reasoning(response)
    if "<tool" not in resp:
        return None                      # prose answer or <done> -- nothing to ground
    pnorm = norm(prompt)

    name = NAME_RE.search(resp)
    name = name.group(1) if name else ""

    for f in PATH_RE.findall(resp):
        if not grounded(f, pnorm):
            return "ungrounded-path:%s" % f

    if name == "write_file":
        body = WRITE_BODY_RE.search(resp)
        content = body.group(1) if body else ""
        if not content.strip():
            return "empty-write-content"
        if not grounded(content, pnorm):
            return "ungrounded-content:%s" % content.strip()[:24]

    for cmd in CMD_RE.findall(resp):
        for tok in EXT_RE.findall(cmd):
            if not grounded(tok, pnorm):
                return "ungrounded-cmd-file:%s" % tok
    return None


def classify(prompt, response):
    """(action_type, step_len) bucket for balancing + reporting."""
    resp = strip_reasoning(response)
    if resp.startswith("<done"):
        action = "done"
    else:
        m = NAME_RE.search(resp)
        action = ("tool:" + m.group(1)) if (m and "<tool" in resp) else "prose"
    # transcript step length: number of MODEL turns already in the prompt + 1
    steps = prompt.count("MODEL:") + 1
    return action, steps


def read_rows(path):
    rows = []
    for line in open(path, encoding="utf-8-sig"):
        line = line.rstrip("\n")
        if "\t" in line:
            p, r = line.split("\t", 1)
            rows.append((p, r, path))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="prompt<TAB>response files to clean+merge")
    ap.add_argument("--out", default="data/clean.txt")
    ap.add_argument("--report", default=None, help="write a drop/keep report here")
    ap.add_argument("--balance", action="store_true",
                    help="cap over-represented action types to factor x median")
    ap.add_argument("--factor", type=float, default=2.0,
                    help="--balance cap = factor x median type count (default 2.0)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shuffle", action="store_true", help="shuffle the final set")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    log = []

    def out(msg):
        print(msg)
        log.append(msg)

    # ---- load ----
    rows = []
    for path in args.inputs:
        r = read_rows(path)
        rows.extend(r)
        out("loaded %5d pairs from %s" % (len(r), path))
    total_in = len(rows)

    # ---- 1. verify ----
    verified, drop_reasons = [], {}
    for p, r, src in rows:
        reason = verify(p, r)
        if reason is None:
            verified.append((p, r, src))
        else:
            key = reason.split(":", 1)[0]
            drop_reasons[key] = drop_reasons.get(key, 0) + 1
    out("\nVERIFY: kept %d / %d (%d rejected)" %
        (len(verified), total_in, total_in - len(verified)))
    for k, v in sorted(drop_reasons.items(), key=lambda kv: -kv[1]):
        out("  - %-22s %d" % (k, v))

    # ---- 2. dedupe (exact + case/whitespace-normalized) ----
    seen, deduped, dups = set(), [], 0
    for p, r, src in verified:
        key = norm(p + "\t" + r)
        if key in seen:
            dups += 1
            continue
        seen.add(key)
        deduped.append((p, r, src))
    out("\nDEDUPE: kept %d (removed %d exact/near-duplicate lines)" %
        (len(deduped), dups))

    # ---- 3. balance / distribution ----
    by_type = {}
    for p, r, src in deduped:
        action, steps = classify(p, r)
        by_type.setdefault(action, []).append((p, r, src, steps))
    out("\nDISTRIBUTION (action type -> count):")
    for k, v in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        lens = {}
        for _, _, _, s in v:
            lens[s] = lens.get(s, 0) + 1
        span = ", ".join("%d-step:%d" % (s, c) for s, c in sorted(lens.items()))
        out("  %-16s %5d   [%s]" % (k, len(v), span))

    final = []
    if args.balance and by_type:
        counts = sorted(len(v) for v in by_type.values())
        median = counts[len(counts) // 2]
        cap = max(1, int(args.factor * median))
        out("\nBALANCE: median type count %d, cap = %.1f x median = %d" %
            (median, args.factor, cap))
        for k, v in by_type.items():
            if len(v) > cap:
                rng.shuffle(v)
                out("  capping %-16s %d -> %d (dropped %d)" % (k, len(v), cap, len(v) - cap))
                v = v[:cap]
            final.extend((p, r) for p, r, _, _ in v)
    else:
        final = [(p, r) for p, r, _ in deduped]

    if args.shuffle:
        rng.shuffle(final)

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for p, r in final:
            f.write("%s\t%s\n" % (p, r))
    out("\nWROTE %d clean pairs -> %s  (from %d input, %.1f%% kept)" %
        (len(final), args.out, total_in, 100.0 * len(final) / max(1, total_in)))

    if args.report:
        with open(args.report, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(log) + "\n")
        print("report -> %s" % args.report)


# Guarded so other tools (tools/agent_critic.py) can import verify()/grounded()/norm()
# as the CANONICAL grounding check without running the CLI.
if __name__ == "__main__":
    main()
