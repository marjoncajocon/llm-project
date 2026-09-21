#!/usr/bin/env python3
"""evalcard.py - compare two `transformer grpo --eval-only --dump` files.

    python tools/evalcard.py A.jsonl            # one checkpoint: summary only
    python tools/evalcard.py A.jsonl B.jsonl    # paired comparison, B - A

Each dump line is one rollout: {"task", "k", "family", "total", "format",
"grounded", "match", "exact", "action", "stop", "tokens", "response", ...}.

The comparison is PAIRED: rollouts are matched by (task, k). The eval mode seeds
rollout (task, k) from hash(seed, task, k), so the two checkpoints were sampled
with the same noise and the per-task difference cancels most of it.

Reported:
  - mean total reward and exact-match rate for each file (task-level: average
    over k within a task first, then over tasks);
  - B - A with a 95% paired bootstrap CI that resamples TASKS (10k resamples,
    fixed seed) -- tasks, not rollouts, are the independent units;
  - an exact McNemar test on per-task exact match (k = 1: the rollout; k > 1:
    the majority of the k rollouts);
  - a per-family breakdown (family = the kind of the reference action);
  - the fraction of tasks whose response text differs -- 0% means the policy
    did not move on these tasks, whatever the reward says.

Standard library only (numpy is used for the bootstrap when available).
"""
import argparse
import json
import math
import random
import sys
from collections import defaultdict

try:
    import numpy as np
except ImportError:  # pragma: no cover - the pure-Python path is the fallback
    np = None


def load(path):
    rows = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{ln}: not JSON ({e})")
            key = (int(r["task"]), int(r["k"]))
            if key in rows:
                sys.exit(f"{path}:{ln}: duplicate (task {key[0]}, k {key[1]})")
            rows[key] = r
    if not rows:
        sys.exit(f"{path}: no rollouts")
    return rows


def per_task(rows, tasks):
    """task -> (mean total over k, mean exact over k, majority exact, family)."""
    by = defaultdict(list)
    for (t, _k), r in rows.items():
        if t in tasks:
            by[t].append(r)
    out = {}
    for t, rs in by.items():
        n = len(rs)
        ex = sum(int(r.get("exact", 0)) for r in rs)
        out[t] = (sum(float(r["total"]) for r in rs) / n, ex / n,
                  1 if 2 * ex > n else 0, rs[0].get("family") or "(none)")
    return out


def bootstrap_ci(diffs_list, n_boot, seed):
    """Percentile 95% CIs for the mean of each paired-difference vector, all from
    the SAME task resamples. Returns [(lo, hi), ...]."""
    n = len(diffs_list[0])
    if n == 0:
        return [(float("nan"), float("nan")) for _ in diffs_list]
    if np is not None:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        out = []
        for d in diffs_list:
            means = np.asarray(d, dtype=np.float64)[idx].mean(axis=1)
            out.append((float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))))
        return out
    rng = random.Random(seed)
    sums = [[0.0] * n_boot for _ in diffs_list]
    for b in range(n_boot):
        pick = [rng.randrange(n) for _ in range(n)]
        for j, d in enumerate(diffs_list):
            sums[j][b] = sum(d[i] for i in pick) / n
    out = []
    for s in sums:
        s.sort()
        out.append((quantile(s, 0.025), quantile(s, 0.975)))
    return out


def quantile(sorted_vals, q):
    """Linear-interpolation quantile (numpy's default), on a sorted list."""
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p: the discordant pairs are Binomial(b + c, 1/2)
    under H0. p = min(1, 2 * P(X <= min(b, c)))."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def fmt_ci(lo, hi, scale=1.0, nd=4, sign="+"):
    return f"[{lo * scale:{sign}.{nd}f}, {hi * scale:{sign}.{nd}f}]"


def shares(rows, field):
    cnt = defaultdict(int)
    for r in rows.values():
        cnt[r.get(field, "?")] += 1
    n = len(rows)
    return " ".join(f"{k} {100.0 * v / n:.0f}%" for k, v in sorted(cnt.items(), key=lambda x: -x[1]))


def describe(label, path, rows):
    tasks = {t for t, _ in rows}
    ks = defaultdict(int)
    for t, _ in rows:
        ks[t] += 1
    pt = per_task(rows, tasks)
    n = len(pt)
    meta = next(iter(rows.values()))
    print(f"{label}: {path}")
    print(f"   {n} tasks, {len(rows)} rollouts (k per task: {min(ks.values())}"
          f"{'' if min(ks.values()) == max(ks.values()) else '..' + str(max(ks.values()))})"
          f" | reward {meta.get('reward', '?')} | temp {meta.get('temp', '?')} | seed {meta.get('seed', '?')}")
    print(f"   mean total {sum(v[0] for v in pt.values()) / n:.4f} | exact-match "
          f"{100.0 * sum(v[1] for v in pt.values()) / n:.2f}%"
          f" (majority-of-k {100.0 * sum(v[2] for v in pt.values()) / n:.2f}%)")
    print(f"   actions: {shares(rows, 'action')} | stop: {shares(rows, 'stop')}")
    return pt


def check_compat(a, b):
    ma, mb = next(iter(a.values())), next(iter(b.values()))
    for f in ("reward", "temp", "seed"):
        if ma.get(f) != mb.get(f):
            print(f"WARNING: the dumps differ in {f}: {ma.get(f)} vs {mb.get(f)} -- the "
                  f"difference below mixes that change with the checkpoint change.")
    bad = sorted({k[0] for k in a if k in b and a[k].get("task_hash") != b[k].get("task_hash")})
    if bad:
        sys.exit(f"error: {len(bad)} task(s) have different task_hash in the two dumps "
                 f"(first: task {bad[0]}) -- they were made from different task files.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345, help="bootstrap seed (fixed => reproducible CI)")
    args = ap.parse_args()

    A = load(args.a)
    if not args.b:
        pt = describe("A", args.a, A)
        tasks = sorted(pt)
        lo_hi = bootstrap_ci([[pt[t][0] for t in tasks], [pt[t][1] for t in tasks]],
                             args.resamples, args.seed)
        print(f"   95% bootstrap CI over tasks: mean total {fmt_ci(*lo_hi[0], sign='')}, "
              f"exact-match {fmt_ci(*lo_hi[1], scale=100, nd=2, sign='')} %")
        fam = defaultdict(list)
        for t in tasks:
            fam[pt[t][3]].append(pt[t])
        print("   per family:")
        for f in sorted(fam, key=lambda x: -len(fam[x])):
            v = fam[f]
            print(f"     {f:<8} {len(v):4d} tasks  mean {sum(x[0] for x in v) / len(v):.4f}"
                  f"  exact {100.0 * sum(x[1] for x in v) / len(v):6.2f}%")
        return

    B = load(args.b)
    check_compat(A, B)
    keys = sorted(set(A) & set(B))
    only_a, only_b = len(set(A) - set(B)), len(set(B) - set(A))
    tasks = sorted({t for t, _ in keys})
    if not tasks:
        sys.exit("error: the two dumps share no (task, k) pair")
    Ak = {k: A[k] for k in keys}
    Bk = {k: B[k] for k in keys}
    pa = describe("A", args.a, Ak)
    pb = describe("B", args.b, Bk)
    if only_a or only_b:
        print(f"NOTE: {only_a} rollout(s) only in A and {only_b} only in B were left out "
              f"(pairing is by (task, k)).")

    n = len(tasks)
    d_tot = [pb[t][0] - pa[t][0] for t in tasks]
    d_ex = [pb[t][1] - pa[t][1] for t in tasks]
    (lt, ht), (le, he) = bootstrap_ci([d_tot, d_ex], args.resamples, args.seed)
    print()
    print(f"B - A over {n} paired tasks ({len(keys)} rollouts; 95% paired bootstrap CI, "
          f"{args.resamples} task resamples, seed {args.seed}):")
    print(f"   mean total   {sum(d_tot) / n:+.4f}  {fmt_ci(lt, ht)}")
    print(f"   exact-match  {100.0 * sum(d_ex) / n:+.2f} pts  {fmt_ci(le, he, scale=100, nd=2)}")

    kmax = max(k for _, k in keys) + 1
    use = "k=1" if kmax == 1 else f"majority of k={kmax}"
    ea = {t: pa[t][2] for t in tasks}
    eb = {t: pb[t][2] for t in tasks}
    b_ = sum(1 for t in tasks if ea[t] and not eb[t])
    c_ = sum(1 for t in tasks if eb[t] and not ea[t])
    both = sum(1 for t in tasks if ea[t] and eb[t])
    print(f"   McNemar (exact, per-task exact match, {use}): A-only {b_}, B-only {c_}, "
          f"both {both}, neither {n - b_ - c_ - both} -> p = {mcnemar_exact(b_, c_):.4g}")

    diff_tasks = sum(1 for t in tasks
                     if any(Ak[k]["response"] != Bk[k]["response"] for k in keys if k[0] == t))
    diff_roll = sum(1 for k in keys if Ak[k]["response"] != Bk[k]["response"])
    print(f"   responses differ: {diff_tasks}/{n} tasks ({100.0 * diff_tasks / n:.1f}%), "
          f"{diff_roll}/{len(keys)} rollouts")
    if diff_roll == 0:
        print("   -> IDENTICAL responses on every rollout: the policy did not move on these tasks.")

    fam = defaultdict(list)
    for t in tasks:
        fam[pa[t][3]].append(t)
    print("   per family (B - A):")
    for f in sorted(fam, key=lambda x: -len(fam[x])):
        ts = fam[f]
        m = len(ts)
        dt = [pb[t][0] - pa[t][0] for t in ts]
        de = [pb[t][1] - pa[t][1] for t in ts]
        (a1, a2), = bootstrap_ci([dt], args.resamples, args.seed)
        fb = sum(1 for t in ts if ea[t] and not eb[t])
        fc = sum(1 for t in ts if eb[t] and not ea[t])
        print(f"     {f:<8} {m:4d} tasks  total {sum(pa[t][0] for t in ts) / m:.4f} -> "
              f"{sum(pb[t][0] for t in ts) / m:.4f} ({sum(dt) / m:+.4f} {fmt_ci(a1, a2)})  "
              f"exact {100.0 * sum(pa[t][1] for t in ts) / m:.1f}% -> "
              f"{100.0 * sum(pb[t][1] for t in ts) / m:.1f}% ({100.0 * sum(de) / m:+.1f} pts, "
              f"McNemar p {mcnemar_exact(fb, fc):.3g})")


if __name__ == "__main__":
    main()
