#!/usr/bin/env python3
"""Generate MULTI-STEP agent trajectories as SFT pairs (transcript -> next action).

Single-instruction SFT (make_sft.py) teaches "one prompt -> one tool call". A real
agent must instead read the running TRANSCRIPT (goal + prior actions + observations)
and choose the NEXT action, then STOP when the goal is met. This synthesizes small
multi-step tasks, plays each one out into a transcript, and explodes it into one
training pair per step:

    GOAL: g                                             -> action_0
    GOAL: g | MODEL: action_0 | OBSERVATION: obs_0      -> action_1
    ... | MODEL: action_{k-1} | OBSERVATION: obs_{k-1}  -> <done>

The agent (agent.c) builds its transcript with real newlines as
"GOAL: ...\nMODEL: ...\nOBSERVATION: ...\n"; tools/agent_local.py flattens that to
the SAME " | "-joined single line used here before calling the model, so training
and inference formats match exactly and the SFT loader still gets one line per pair.

    python tools/make_agent_traj.py --out data/traj.txt --n 260 \
        --wiki data/wiki --commands data/so_commands.txt

Observation strings mirror toolexec.c: write_file -> "wrote N bytes to <f>",
read_file -> the file's content, run_command -> the command's output.
"""
import argparse
import os
import random
import re

SEP = " | "          # transcript turn separator (must match agent_local.py)
EXTS = [".txt", ".md", ".log", ".json", ".csv"]
WORDS = ["hello", "done", "start", "ok todo", "first line", "the result is 42",
         "work in progress", "finished", "note to self", "build passed"]
DONE = ["Task complete.", "All steps done.", "Finished the task."]


def slugs_from_wiki(wiki_dir, limit=300):
    out = []
    if wiki_dir and os.path.isdir(wiki_dir):
        for fn in sorted(os.listdir(wiki_dir)):
            if fn.endswith(".txt"):
                s = re.sub(r"[^A-Za-z0-9]+", "_", fn[:-4].lower()).strip("_")[:20]
                if s:
                    out.append(s)
            if len(out) >= limit:
                break
    return out or ["notes", "readme", "config", "output", "report", "data"]


def commands_from(path, limit=300):
    cmds = []
    if path and os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            c = line.strip()
            if c and 3 <= len(c) <= 40 and '"' not in c:
                cmds.append(c)
            if len(cmds) >= limit:
                break
    return cmds or ["ls", "git status", "pwd", "make test"]


def tool_write(f, w):
    return '<tool name="write_file" path="%s">%s</tool>' % (f, w)


def tool_read(f):
    return '<tool name="read_file" path="%s"></tool>' % f


def tool_run(c):
    return '<tool name="run_command" command="%s"></tool>' % c


def build_tasks(rng, files, cmds):
    """Return (goal, [(action, observation), ...]) for one random task."""
    f1 = rng.choice(files) + rng.choice(EXTS)
    f2 = rng.choice(files) + rng.choice(EXTS)
    w = rng.choice(WORDS)
    c = rng.choice(cmds)
    kind = rng.random()

    if kind < 0.26:                                   # write then read (verify)
        g = rng.choice(["create %s containing %s and read it back",
                        "write %s to %s, then read it to confirm"])
        goal = (g % (f1, w)) if "containing" in g else (g % (w, f1))
        return goal, [(tool_write(f1, w), "wrote %d bytes to %s" % (len(w), f1)),
                      (tool_read(f1), w)]

    if kind < 0.46:                                   # two writes
        goal = "create %s with %s, then create %s with %s" % (f1, w, f2, "ok")
        return goal, [(tool_write(f1, w), "wrote %d bytes to %s" % (len(w), f1)),
                      (tool_write(f2, "ok"), "wrote 2 bytes to %s" % f2)]

    if kind < 0.62:                                   # read then delete
        goal = rng.choice(["read %s then delete it", "open %s and then remove it"]) % f1
        return goal, [(tool_read(f1), w),
                      (tool_run("rm %s" % f1), "")]

    if kind < 0.78:                                   # run a command then list files
        goal = "run %s and then list the files" % c
        return goal, [(tool_run(c), "ok"),
                      (tool_run("ls"), "%s  %s" % (f1, f2))]

    if kind < 0.90:                                   # single write step
        goal = rng.choice(["create %s with %s", "save %s to %s"])
        goal = (goal % (f1, w)) if goal.startswith("create") else (goal % (w, f1))
        return goal, [(tool_write(f1, w), "wrote %d bytes to %s" % (len(w), f1))]

    # single run step
    goal = rng.choice(["run %s", "execute %s", "please run %s"]) % c
    return goal, [(tool_run(c), "ok")]


def emit_pairs(goal, steps, rng):
    """Explode a task into (transcript_prefix, next_action) pairs, ending in <done>."""
    pairs = []
    turns = ["GOAL: " + goal]
    for action, obs in steps:
        pairs.append((SEP.join(turns), action))
        turns.append("MODEL: " + action)
        turns.append("OBSERVATION: " + obs.replace("\n", " ").replace("\t", " "))
    pairs.append((SEP.join(turns), "<done>%s</done>" % rng.choice(DONE)))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/traj.txt")
    ap.add_argument("--n", type=int, default=260, help="number of tasks (each -> several pairs)")
    ap.add_argument("--wiki", default=None)
    ap.add_argument("--commands", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = slugs_from_wiki(args.wiki)
    cmds = commands_from(args.commands)
    rng.shuffle(files)

    rows, steps_hist = [], {}
    for _ in range(args.n):
        goal, steps = build_tasks(rng, files, cmds)
        steps_hist[len(steps)] = steps_hist.get(len(steps), 0) + 1
        for prompt, resp in emit_pairs(goal, steps, rng):
            rows.append("%s\t%s" % (prompt.replace("\t", " "), resp))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rows) + "\n")
    print("wrote %d pairs from %d tasks -> %s" % (len(rows), args.n, args.out))
    print("  task lengths (steps -> count): %s"
          % ", ".join("%d:%d" % (k, v) for k, v in sorted(steps_hist.items())))


main()
