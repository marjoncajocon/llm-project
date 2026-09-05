#!/usr/bin/env python3
"""Generate agent tasks the model gets right only SOMETIMES -- fuel for GRPO.

`make_agent_traj.py` builds SFT data, so it emits EVERY step of every chain and
each step is a deterministic function of (template, position). That is the right
shape for supervised learning and the wrong shape for RL: a 135M model trained on
it scores a flat 1.0000 on the resulting task file, every group's advantage is
zero, and GRPO skips the lot. Measured 2026-09-05: 5 of 5 groups flat at temp 1.0.

    GRPO LEARNS FROM VARIANCE. A TASK THE MODEL ALWAYS SOLVES TEACHES NOTHING,
    AND SO DOES ONE IT NEVER SOLVES.

This generator therefore inverts two of make_agent_traj.py's choices:

  * it emits ONLY THE DECISION POINT of each task, not the whole chain. The
    easy prefix steps ("now write the file the goal just named") are exactly the
    flat groups we are trying to get rid of, so they are never written out.
  * every task is built so that the WRONG action is GROUNDED TOO. Distractor
    filenames all appear in the prompt, so the reward's `grounded` term cannot
    separate a good answer from a bad one -- only `match` can. That forces the
    model to actually decide instead of copying the nearest plausible token.

Six families, each targeting a different failure the reward can verify:

  distractor  three files named in the goal; act on ONE specific one.
  indirect    the goal names the file by PROPERTY ("the report file"); an `ls`
              observation supplies the candidates. Requires binding, not copying.
  mismatch    a read-back returns content DIFFERENT from what was written; the
              right move is to re-write, not to carry on down the goal.
  stop-yes    every clause is satisfied -> <done>, with a plausible extra step
              still dangling as bait.
  stop-no     the transcript LOOKS finished but one clause is outstanding; the
              right move is the missing action, NOT <done>. (The premature-<done>
              bug from 25d1113, turned into a training signal.)
  longchain   5-6 steps, decision taken at a late position where the prefix no
              longer fits in the model's memorized 1-4 step templates.

Usage (the second command is the grounding gate -- do not skip it):

    python tools/hard_tasks.py --out data/hard_raw.txt --n 600 \
        --wiki data/wiki --commands data/so_commands.txt --reason
    python tools/grpo_tasks.py --out data/grpo_hard.txt data/hard_raw.txt

Then MEASURE, because "hard" is a claim about a specific model, not a property of
the text -- a task file is only good if THIS checkpoint has spread on it:

    $env:TF_THREADS = "3"
    .\\transformer.exe grpo data/grpo_hard.txt --model model/smollm/reason_e2.bin \\
        --hf-tok model/smollm/tok.hf --out /tmp/probe.bin --steps 5 --verbose

Deterministic given --seed.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_agent_traj import (  # noqa: E402
    EXTS, SEP, commands_from, contents_from, rationale_for, run_obs,
    slugs_from_wiki, tool_read, tool_run, tool_write,
)

DONE = ["Task complete.", "All steps done.", "Finished the task."]

# A file described by PROPERTY rather than by name, plus the slug fragment that
# makes a candidate match it. The `indirect` family needs the description to pick
# out exactly one of the listed files -- so the distractors must not contain the
# fragment, which build_indirect enforces.
PROPS = [
    ("the report file", "report"),
    ("the config file", "config"),
    ("the log file", "log"),
    ("the notes file", "notes"),
    ("the readme", "readme"),
]


def _distinct(rng, pool, n, avoid=()):
    """n distinct picks from pool, none of them in `avoid`."""
    out = []
    guard = 0
    while len(out) < n and guard < 500:
        guard += 1
        x = rng.choice(pool)
        if x not in out and x not in avoid:
            out.append(x)
    return out


def _turns(goal, steps):
    """Transcript prefix for a goal plus a list of (action, observation)."""
    t = ["GOAL: " + goal]
    for action, obs in steps:
        t.append("MODEL: " + action)
        t.append("OBSERVATION: " + str(obs).replace("\n", " ").replace("\t", " "))
    return SEP.join(t)


def build_distractor(rng, files, cmds, contents):
    """Three files in play; the next action must hit ONE of them.

    Every candidate path is in the prompt, so `grounded` pays out for all three
    and only `match` can tell them apart -- 1.0 for the right path, 0.4 for the
    right tool with the wrong one."""
    a, b, c = [f + rng.choice(EXTS) for f in _distinct(rng, files, 3)]
    w1, w2, w3 = _distinct(rng, contents, 3)
    which, name = rng.choice([(0, a), (1, b), (2, c)])
    ordinal = ["first", "second", "third"][which]
    goal = ("create %s with %s, %s with %s, and %s with %s, then read the %s one back"
            % (a, w1, b, w2, c, w3, ordinal))
    steps = [(tool_write(a, w1), "wrote %d bytes to %s" % (len(w1), a)),
             (tool_write(b, w2), "wrote %d bytes to %s" % (len(w2), b)),
             (tool_write(c, w3), "wrote %d bytes to %s" % (len(w3), c))]
    return _turns(goal, steps), tool_read(name), "distractor"


def build_indirect(rng, files, cmds, contents):
    """The goal names the target by PROPERTY; an `ls` observation supplies names.

    The model cannot copy a filename out of the goal -- there is none there. It
    has to read the listing and bind the description to one entry."""
    desc, frag = rng.choice(PROPS)
    target = "%s_%s%s" % (rng.choice(files)[:12], frag, rng.choice(EXTS))
    others = ["%s%s" % (f, rng.choice(EXTS))
              for f in _distinct(rng, files, 2)
              if frag not in f]
    while len(others) < 2:                       # slug collided with the fragment
        f = rng.choice(files)
        if frag not in f:
            others.append(f + rng.choice(EXTS))
    listing = list(others) + [target]
    rng.shuffle(listing)
    verb, action = rng.choice([
        ("read", tool_read(target)),
        ("delete", tool_run("rm %s" % target)),
    ])
    goal = "list the files, then %s %s" % (verb, desc)
    steps = [(tool_run("ls"), "  ".join(listing))]
    return _turns(goal, steps), action, "indirect"


def build_mismatch(rng, files, cmds, contents):
    """The read-back disagrees with what was written -> re-write, don't move on.

    The tempting continuation (the next goal clause) is wrong; noticing that the
    observation contradicts the goal is the whole task."""
    f = rng.choice(files) + rng.choice(EXTS)
    want, got = _distinct(rng, contents, 2)
    c = rng.choice(cmds)
    goal = ("create %s containing %s, read it back to check, then run %s"
            % (f, want, c))
    steps = [(tool_write(f, want), "wrote %d bytes to %s" % (len(want), f)),
             (tool_read(f), got)]
    return _turns(goal, steps), tool_write(f, want), "mismatch"


def build_stop_yes(rng, files, cmds, contents):
    """Goal fully satisfied, with a plausible extra action still dangling.

    `action_match` scores <done> on the ROUTING decision alone, so this family is
    binary: 1.0 for stopping, 0.0 for any tool call. Maximum spread per group."""
    f = rng.choice(files) + rng.choice(EXTS)
    w = rng.choice(contents)
    c = rng.choice(cmds)
    goal = "create %s containing %s and run %s" % (f, w, c)
    steps = [(tool_write(f, w), "wrote %d bytes to %s" % (len(w), f)),
             (tool_run(c), run_obs(c, rng))]
    return _turns(goal, steps), "<done>%s</done>" % rng.choice(DONE), "stop-yes"


def build_stop_no(rng, files, cmds, contents):
    """Looks finished -- one clause is not. Stopping here is the error.

    The transcript ends on a successful-looking observation, which is precisely
    the shape that triggered premature <done> before (commit 25d1113)."""
    a, b = [f + rng.choice(EXTS) for f in _distinct(rng, files, 2)]
    w1, w2 = _distinct(rng, contents, 2)
    goal = "create %s with %s, then create %s with %s" % (a, w1, b, w2)
    steps = [(tool_write(a, w1), "wrote %d bytes to %s" % (len(w1), a))]
    if rng.random() < 0.5:                       # a longer, more convincing prefix
        c = rng.choice(cmds)
        goal = ("create %s with %s, run %s, then create %s with %s"
                % (a, w1, c, b, w2))
        steps.append((tool_run(c), run_obs(c, rng)))
    return _turns(goal, steps), tool_write(b, w2), "stop-no"


def build_longchain(rng, files, cmds, contents):
    """5-6 steps, decision at a late position, past the memorized templates."""
    a, b = [f + rng.choice(EXTS) for f in _distinct(rng, files, 2)]
    w1, w2 = _distinct(rng, contents, 2)
    c1, c2 = rng.choice(cmds), rng.choice(cmds)
    goal = ("create %s with %s, run %s, create %s with %s, run %s, "
            "then read %s and delete %s" % (a, w1, c1, b, w2, c2, a, b))
    steps = [(tool_write(a, w1), "wrote %d bytes to %s" % (len(w1), a)),
             (tool_run(c1), run_obs(c1, rng)),
             (tool_write(b, w2), "wrote %d bytes to %s" % (len(w2), b)),
             (tool_run(c2), run_obs(c2, rng))]
    if rng.random() < 0.5:
        return _turns(goal, steps), tool_read(a), "longchain"
    steps.append((tool_read(a), w1))
    return _turns(goal, steps), tool_run("rm %s" % b), "longchain"


# Name -> builder. Kept as an explicit table rather than probing each builder for
# the family tag it returns: calling one just to read its name would consume rng
# draws and make --only produce a different stream than the same seed without it.
FAMILIES = {
    "distractor": build_distractor,
    "indirect": build_indirect,
    "mismatch": build_mismatch,
    "stop-yes": build_stop_yes,
    "stop-no": build_stop_no,
    "longchain": build_longchain,
}
FAMILY_NAMES = sorted(FAMILIES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/hard_raw.txt")
    ap.add_argument("--n", type=int, default=600, help="number of tasks (one line each)")
    ap.add_argument("--wiki", default=None)
    ap.add_argument("--commands", default=None)
    ap.add_argument("--reason", action="store_true",
                    help="reasoning-dense reference: <think>rationale</think>action, "
                         "matching the corpus reason_e2.bin was trained on (the reward "
                         "strips reasoning from both sides, so this is format hygiene)")
    ap.add_argument("--only", default=None,
                    help="restrict to one family (distractor|indirect|mismatch|"
                         "stop-yes|stop-no|longchain)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = slugs_from_wiki(args.wiki)
    cmds = commands_from(args.commands)
    contents = contents_from(files)
    rng.shuffle(files)

    if args.only and args.only not in FAMILIES:
        print("unknown family: %s (want one of: %s)"
              % (args.only, "|".join(FAMILY_NAMES)))
        return 2

    rows, mix, seen = [], {}, set()
    dupes = 0
    for _ in range(args.n):
        fn = FAMILIES[args.only] if args.only else FAMILIES[rng.choice(FAMILY_NAMES)]
        prompt, resp, fam = fn(rng, files, cmds, contents)
        if prompt in seen:            # a duplicate prompt is a wasted rollout budget
            dupes += 1
            continue
        seen.add(prompt)
        if args.reason:
            r = rationale_for(resp, rng)
            if r:
                resp = "<think>%s</think>%s" % (r, resp)
            elif resp.startswith("<done>"):
                resp = "<think>%s</think>%s" % (rng.choice(
                    ["Every step is finished, so I'm done.",
                     "The goal is complete."]), resp)
        mix[fam] = mix.get(fam, 0) + 1
        rows.append("%s\t%s" % (prompt.replace("\t", " "), resp))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rows) + "\n")

    print("wrote %d tasks -> %s" % (len(rows), args.out))
    if dupes:
        print("  dropped %d duplicate prompt(s)" % dupes)
    print("family mix:")
    for k in sorted(mix, key=lambda k: -mix[k]):
        print("  %-12s %d" % (k, mix[k]))
    print("\nNEXT: gate it, then measure -- 'hard' is a claim about reason_e2.bin,\n"
          "      not about the text:\n"
          "  python tools/grpo_tasks.py --out data/grpo_hard.txt %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
