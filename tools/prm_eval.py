#!/usr/bin/env python3
"""Evaluate a process reward model (or a baseline) on held-out agent steps.

    python tools/prm_eval.py data/prm_test.jsonl --prm model/smollm/prm.bin --tok model/smollm/tok.hf
    python tools/prm_eval.py data/prm_test.jsonl --policy model/smollm/reason_e2.bin --tok ...

Each test item is a transcript, its GOLD next step and several BAD steps
(tools/make_prm_data.py). A scorer ranks the candidates; we report
  pairwise  share of (gold, bad) pairs where gold scores strictly higher
  top-1     share of items where gold is ranked first
plus pairwise accuracy per kind of bad step.

  --prm     score = log p(" good") - log p(" bad") after
            "<transcript> | CANDIDATE: <step> | VERDICT:"   (the trained PRM)
  --policy  score = log p(step | transcript) under the agent model itself, length-
            normalized -- the baseline any learned verifier must beat.
One engine call (`transformer hfscore`) scores the whole set: the model loads once.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(os.path.dirname(HERE), "transformer.exe")
if not os.path.exists(ENGINE):
    ENGINE = os.path.join(os.path.dirname(HERE), "transformer")
RS, GS = "\x1e", "\x1d"


def hfscore(model, tok, records):
    """records: [(prompt, [cand, ...])] -> [[(logp, n), ...]] in the same order."""
    payload = GS.join(p + RS + RS.join(c) for p, c in records)
    proc = subprocess.run([ENGINE, "hfscore", model, tok], input=payload.encode("utf-8"),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        sys.exit("hfscore failed: %s" % proc.stderr.decode("utf-8", "replace")[-400:])
    out = [[None] * len(c) for _, c in records]
    for ln in proc.stdout.decode("utf-8").splitlines():
        d = json.loads(ln)
        out[d["r"]][d["i"]] = (d["logp"], d["n"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prm")
    g.add_argument("--policy")
    ap.add_argument("--tok", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    items = [json.loads(ln) for ln in open(a.test, encoding="utf-8") if ln.strip()]
    if a.limit:
        items = items[:a.limit]

    if a.prm:
        recs = []
        for it in items:
            for c in [it["gold"]] + [b["action"] for b in it["bad"]]:
                recs.append(("%s | CANDIDATE: %s | VERDICT:" % (it["prompt"], c), [" good", " bad"]))
        res = hfscore(a.prm, a.tok, recs)
        flat = [r[0][0] - r[1][0] for r in res]
        scores, k = [], 0
        for it in items:
            m = 1 + len(it["bad"])
            scores.append(flat[k:k + m])
            k += m
        name = "PRM %s" % a.prm
    else:
        recs = [(it["prompt"] + " | MODEL:", [" " + c for c in [it["gold"]] + [b["action"] for b in it["bad"]]])
                for it in items]
        res = hfscore(a.policy, a.tok, recs)
        scores = [[lp / max(n, 1) for lp, n in r] for r in res]
        name = "policy likelihood %s" % a.policy

    pairs = wins = top1 = 0
    kind = {}
    for it, sc in zip(items, scores):
        gold, bads = sc[0], sc[1:]
        top1 += all(gold > b for b in bads)
        for b, s in zip(it["bad"], bads):
            w = gold > s
            pairs += 1
            wins += w
            kk = kind.setdefault(b["kind"], [0, 0])
            kk[0] += w
            kk[1] += 1
    print("%s on %d items / %d pairs" % (name, len(items), pairs))
    print("  pairwise %.1f%%   top-1 %.1f%%" % (100.0 * wins / max(pairs, 1), 100.0 * top1 / max(len(items), 1)))
    print("  by bad kind: " + "  ".join("%s %d/%d" % (k, v[0], v[1]) for k, v in sorted(kind.items())))


if __name__ == "__main__":
    main()
