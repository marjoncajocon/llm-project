#!/usr/bin/env python3
"""Build process-reward-model (PRM) data from agent trajectories.

    python tools/make_prm_data.py data/clean_reason.txt --out data/prm_train.txt \
        --test-out data/prm_test.jsonl [--test-frac 0.1] [--seed 1]

A PRM scores ONE STEP of a trajectory: given the transcript so far and a
candidate next action, is this a correct step? The trajectory rows here are
"prompt<TAB>response" (the response is the gold next action), so every row gives
one verified GOOD step, and we manufacture BAD steps from it -- the exact failure
modes the live agent has shown, each verifiably wrong by construction:

  slip      the gold path with one character changed (demo_note -> demonote)
  tool      the right file, the wrong tool (read where a write was due, ...)
  done      a premature <done> while the gold step is still a tool call
  repeat    re-issuing an action the transcript already shows succeeded
  content   a write_file with the right path but another row's content
  continue  (gold is <done>) another tool call when the goal is complete

Training rows (SFT format, for `llama-train --sft`):
    <transcript> | CANDIDATE: <action> | VERDICT:<TAB> good      (or  bad)
balanced 1:1 good:bad. A held-out split of whole source rows (no transcript is
in both) goes to --test-out as JSONL {"prompt", "gold", "bad": [...]} for
tools/prm_eval.py, which asks: does the PRM rank gold above every bad step?
"""
import argparse
import json
import random
import re

TOOL = re.compile(r'<tool name="([a-z_]+)"(?: path="([^"]*)")?(?: command="([^"]*)")?>(.*?)</tool>', re.S)
DONE = re.compile(r"<done>.*?</done>", re.S)
THINK = re.compile(r"<think>.*?</think>", re.S)
PAST = re.compile(r'MODEL: (<tool .*?</tool>)')


def action_of(resp):
    """The action block of a response (reasoning stripped), or None."""
    r = THINK.sub("", resp).strip()
    m = TOOL.search(r)
    d = DONE.search(r)
    if d and (not m or d.start() < m.start()):
        return d.group(0)
    return m.group(0) if m else None


def parse(a):
    m = TOOL.match(a)
    if not m:
        return ("done", "", "", "")
    return (m.group(1), m.group(2) or "", m.group(3) or "", m.group(4) or "")


def tool(name, path="", command="", body=""):
    if name == "run_command":
        return '<tool name="run_command" command="%s"></tool>' % command
    return '<tool name="%s" path="%s">%s</tool>' % (name, path, body)


def slip(s, rng):
    stem, dot, ext = s.rpartition(".")
    if not dot:
        stem, ext = s, ""
    if len(stem) < 3:
        return None
    i = rng.randrange(1, len(stem) - 1)
    op = rng.choice(["del", "dup", "sub"])
    if op == "del":
        stem = stem[:i] + stem[i + 1:]
    elif op == "dup":
        stem = stem[:i] + stem[i] + stem[i:]
    else:
        c = rng.choice("abcdefghijklmnopqrstuvwxyz")
        if c == stem[i]:
            return None
        stem = stem[:i] + c + stem[i + 1:]
    return stem + (dot + ext if dot else "")


def negatives(prompt, gold, pool_bodies, rng):
    """Distinct bad candidates, none equal to the gold step."""
    seen, out = {gold}, []
    for k, b in _negatives(prompt, gold, pool_bodies, rng):
        if b not in seen:
            seen.add(b)
            out.append((k, b))
    return out


def _negatives(prompt, gold, pool_bodies, rng):
    name, path, cmd, body = parse(gold)
    past = PAST.findall(prompt)
    out = []
    if name == "done":
        if past:
            out.append(("continue", past[-1]))
        files = re.findall(r"[\w.-]+\.(?:txt|md|json|csv|log)", prompt)
        if files:
            out.append(("continue", tool("read_file", rng.choice(files))))
        return out
    out.append(("done", "<done>Finished the task.</done>"))
    if path:
        s = slip(path, rng)
        if s and s != path:
            out.append(("slip", tool(name, s, cmd, body)))
    if name == "write_file":
        out.append(("tool", tool("read_file", path)))
        other = rng.choice(pool_bodies)
        if other != body:
            out.append(("content", tool("write_file", path, body=other)))
    elif name == "read_file":
        out.append(("tool", tool("write_file", path, body=rng.choice(pool_bodies))))
    elif name == "run_command":
        files = re.findall(r"[\w.-]+\.(?:txt|md|json|csv|log)", prompt)
        if files:
            out.append(("tool", tool("read_file", rng.choice(files))))
    for p in past:
        if p != gold and parse(p)[0] != "done":
            out.append(("repeat", p))
            break
    return out


def row(prompt, action):
    return "%s | CANDIDATE: %s | VERDICT:" % (prompt, action)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-out", required=True)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    for path in a.src:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                if "\t" not in ln or not ln.startswith("GOAL:"):
                    continue
                p, r = ln.rstrip("\n").split("\t", 1)
                g = action_of(r)
                if g:
                    rows.append((p, g))
    rng.shuffle(rows)
    bodies = [parse(g)[3] for _, g in rows if parse(g)[0] == "write_file" and parse(g)[3]] or ["hello"]
    n_test = max(1, int(len(rows) * a.test_frac))
    test, train = rows[:n_test], rows[n_test:]

    lines, kinds = [], {}
    for p, g in train:
        negs = negatives(p, g, bodies, rng)
        if not negs:
            continue
        lines.append(row(p, g) + "\t good")
        k, bad = rng.choice(negs)              # one bad per good: balanced 1:1
        kinds[k] = kinds.get(k, 0) + 1
        lines.append(row(p, bad) + "\t bad")
    rng.shuffle(lines)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    n_t = 0
    with open(a.test_out, "w", encoding="utf-8", newline="\n") as f:
        for p, g in test:
            negs = negatives(p, g, bodies, rng)
            if negs:
                f.write(json.dumps({"prompt": p, "gold": g,
                                    "bad": [{"kind": k, "action": b} for k, b in negs]}) + "\n")
                n_t += 1
    print("make_prm_data: %d source rows -> %d train lines (%d good / %d bad) -> %s" % (
        len(rows), len(lines), len(lines) // 2, len(lines) // 2, a.out))
    print("  bad kinds: %s" % ", ".join("%s %d" % kv for kv in sorted(kinds.items())))
    print("  held-out: %d trajectories (%d bad candidates) -> %s" % (
        n_t, sum(len(negatives(p, g, bodies, random.Random(0))) for p, g in test), a.test_out))


if __name__ == "__main__":
    main()
