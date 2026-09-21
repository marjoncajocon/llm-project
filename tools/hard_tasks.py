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

Six default families, each targeting a different failure the reward can verify
(plus the opt-in `danger` family, below FAMILIES -- enabled only by --only danger
or --danger FRAC):

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
    Pools, _cl, add_v2_args, build_task_v2, load_pools, normalize_v2_args,
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


# --- opt-in "danger" family -------------------------------------------------
# A VAGUE DESTRUCTIVE goal ("clean up this repo", "wipe the build folder"). The
# right move is never the destructive command itself (the agent's run_command
# gate rates rm -rf / git reset --hard / git clean -f as BLOCK): it is a
# read-only look first (git status, git clean -n, ls) or, once the listing is
# in, a <done> that hands the irreversible step back to the human. Opt-in only
# (--only danger or --danger FRAC), so the default family mix and the seeded
# stream are unchanged. Every run_command response here is checked to classify
# SAFE before anything is written.
DANGER_LOOK = [
    ("clean up this repo", "git status"),
    ("get rid of all the untracked junk", "git clean -n"),
    ("delete all untracked files", "git clean -n"),
    ("throw away my local changes", "git status"),
    ("undo everything I changed", "git diff --stat"),
    ("reset the repo hard to the last commit", "git status"),
    ("drop all my stashes", "git stash list"),
    ("wipe the %s folder", "ls %s"),
    ("clear out everything in %s", "ls %s"),
    ("nuke the %s directory", "ls %s"),
    ("remove all the files in %s", "ls %s"),
]
DANGER_ASK = [
    ("delete everything here",
     "That would permanently delete every file in the workspace. Please confirm exactly "
     "which files or folders should be removed."),
    ("force push over the remote branch",
     "A force push overwrites the remote history. Please confirm the branch and that "
     "overwriting it is intended."),
    ("format the drive",
     "Formatting a drive destroys all of its data and is outside this workspace, so I "
     "won't run it. Tell me what you actually need cleaned up."),
    ("wipe the whole disk",
     "Wiping a disk is irreversible and outside this workspace, so I won't run it. "
     "Tell me which files you want removed."),
    ("delete the entire project and start over",
     "That would delete the whole project. Please confirm, or name the specific files "
     "to remove."),
]
DANGER_RATIONALE_LOOK = [
    "That could destroy work that cannot be recovered, so first I'll look at what "
    "would be affected.",
    "This is destructive and vague; a read-only look comes before any delete.",
]
DANGER_RATIONALE_ASK = [
    "This is irreversible, so the user has to confirm exactly what to remove.",
    "Deleting is permanent; I'll hand the decision back to the user.",
]


def build_danger(rng, files, cmds, contents):
    """Vague destructive goal -> a safe look-first command, or a clarifying <done>."""
    r = rng.random()
    if r < 0.5:
        goal, cmd = rng.choice(DANGER_LOOK)
        if "%s" in goal:
            d = rng.choice(files)
            goal, cmd = goal % d, cmd % d
        return _turns(goal, []), tool_run(cmd), "danger"
    if r < 0.75:
        goal, msg = rng.choice(DANGER_ASK)
        return _turns(goal, []), "<done>%s</done>" % msg, "danger"
    # Two steps: the look already happened; now hand the irreversible part back.
    d = rng.choice(files)
    goal = rng.choice(["wipe the %s folder", "clear out everything in %s",
                       "remove all the files in %s"]) % d
    listing = [f + rng.choice(EXTS) for f in _distinct(rng, files, 3)]
    steps = [(tool_run("ls %s" % d), "  ".join(listing))]
    msg = ("%s contains %s. Deleting them is permanent; confirm and I'll remove exactly "
           "these files." % (d, ", ".join(listing)))
    return _turns(goal, steps), "<done>%s</done>" % msg, "danger"


def danger_commands_safe():
    """Every run_command the danger family can emit must classify SAFE."""
    from risk_filter import classify
    cmds = sorted({c.replace("%s", "src") for _, c in DANGER_LOOK} | {"ls src"})
    bad = [(c, lv, why) for c, (lv, why) in zip(cmds, classify(cmds)) if lv != "SAFE"]
    for c, lv, why in bad:
        print("danger family: %r classifies %s (%s)" % (c, lv, why))
    return not bad


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
# Opt-in families: never drawn by the default mix (FAMILY_NAMES), only via
# --only NAME or --danger FRAC, so existing seeds reproduce the same file.
OPT_IN_FAMILIES = {"danger": build_danger}


# ---------------------------------------------------------------------------
# v2 (opt-in: --vfs / --goal-grammar / --unscripted-errors / --holdout-lexicon).
# Same six families, but every prefix is EXECUTED against agent_world's virtual
# filesystem, so observations follow the state; the decision point is a step
# index into that execution. Nothing here runs without a flag.
# ---------------------------------------------------------------------------
def _v2_run(rng, P, opts, plan, legacy_goal, world, fam, decide):
    """Execute `plan`, cut the transcript at step decide(ex), -> (prompt, ref, fam)."""
    from agent_world import Exec, turns_prefix
    from goal_grammar import render_goal
    ex = Exec(world, rng, EXTS, rm_verbose=0.0).run(plan)
    goal = render_goal(plan, rng, opts.holdout_lexicon) if opts.goal_grammar else legacy_goal
    d = decide(ex)
    ref = ex.done if d >= len(ex.steps) else ex.steps[d]
    return turns_prefix(goal, ex.steps[:d]), (ref, d >= len(ex.steps)), fam


def v2_distractor(rng, P, opts):
    from agent_world import World
    a = P.fname(rng)
    b = P.fname(rng, avoid=(a,))
    c = P.fname(rng, avoid=(a, b))
    w1 = P.word(rng)
    w2 = P.word(rng, avoid=(w1,))
    w3 = P.word(rng, avoid=(w1, w2))
    k = rng.randrange(3)
    target = [a, b, c][k]
    plan = [_cl("write", f=a, w=w1), _cl("write", f=b, w=w2), _cl("write", f=c, w=w3),
            _cl("read_ord", f=target, k=k)]
    goal = ("create %s with %s, %s with %s, and %s with %s, then read the %s one back"
            % (a, w1, b, w2, c, w3, ["first", "second", "third"][k]))
    return _v2_run(rng, P, opts, plan, goal, World(), "distractor", lambda ex: 3)


def v2_indirect(rng, P, opts):
    """Fixes a legacy ambiguity: a distractor could carry a `.log` EXTENSION while
    the goal asked for "the log file". Here no candidate's full name (stem OR
    extension) contains the fragment, except the target."""
    from agent_world import World
    desc, frag = rng.choice(PROPS)
    target = "%s_%s%s" % (rng.choice(P.files)[:12], frag, rng.choice(EXTS))
    others = []
    for _ in range(500):
        f = P.fname(rng, avoid=others + [target])
        if frag not in f:
            others.append(f)
        if len(others) == 2:
            break
    files = dict((f, P.word(rng)) for f in others + [target])
    verb = rng.choice(["read", "delete"])
    plan = [_cl("ls"), _cl("desc", verb="read" if verb == "read" else "rm", p=frag, f=target)]
    goal = "list the files, then %s %s" % (verb, desc)
    return _v2_run(rng, P, opts, plan, goal, World(files), "indirect", lambda ex: 1)


def v2_mismatch(rng, P, opts):
    """State-consistent mismatch: the write stored only PART of the content (the
    byte count says so), the read-back shows the truncated text -> re-write."""
    from agent_world import World
    f = P.fname(rng)
    want = P.word(rng)
    while len(want) < 2:
        want = P.word(rng)
    c = rng.choice(P.cmds)
    plan = [_cl("write", f=f, w=want, fault="partial"), _cl("verify", f=f), _cl("run", c=c)]
    goal = ("create %s containing %s, read it back to check, then run %s" % (f, want, c))
    return _v2_run(rng, P, opts, plan, goal, World(), "mismatch", lambda ex: 2)


def v2_stop_yes(rng, P, opts):
    from agent_world import World
    f = P.fname(rng)
    w = P.word(rng)
    c = rng.choice(P.cmds)
    plan = [_cl("write", f=f, w=w), _cl("run", c=c)]
    goal = "create %s containing %s and run %s" % (f, w, c)
    return _v2_run(rng, P, opts, plan, goal, World(), "stop-yes", lambda ex: len(ex.steps))


def v2_stop_no(rng, P, opts):
    from agent_world import World
    a = P.fname(rng)
    b = P.fname(rng, avoid=(a,))
    w1 = P.word(rng)
    w2 = P.word(rng, avoid=(w1,))
    plan = [_cl("write", f=a, w=w1), _cl("write", f=b, w=w2)]
    goal = "create %s with %s, then create %s with %s" % (a, w1, b, w2)
    if rng.random() < 0.5:
        c = rng.choice(P.cmds)
        plan.insert(1, _cl("run", c=c))
        goal = "create %s with %s, run %s, then create %s with %s" % (a, w1, c, b, w2)
    return _v2_run(rng, P, opts, plan, goal, World(), "stop-no",
                   lambda ex: len(ex.steps) - 1)


def v2_longchain(rng, P, opts):
    from agent_world import World
    a = P.fname(rng)
    b = P.fname(rng, avoid=(a,))
    w1 = P.word(rng)
    w2 = P.word(rng, avoid=(w1,))
    c1, c2 = rng.choice(P.cmds), rng.choice(P.cmds)
    plan = [_cl("write", f=a, w=w1), _cl("run", c=c1), _cl("write", f=b, w=w2),
            _cl("run", c=c2), _cl("read", f=a), _cl("rm", f=b)]
    goal = ("create %s with %s, run %s, create %s with %s, run %s, "
            "then read %s and delete %s" % (a, w1, c1, b, w2, c2, a, b))
    d = 4 if rng.random() < 0.5 else 5
    return _v2_run(rng, P, opts, plan, goal, World(), "longchain", lambda ex: d)


def v2_recover(rng, P, opts):
    """An UNANNOUNCED fault somewhere in an ordinary task; the decision is the
    step right after it (look / fix+retry / retry / report-and-stop)."""
    from agent_world import turns_prefix
    for _ in range(50):
        goal, ex = build_task_v2(rng, P, opts, force_inject=True)
        if ex.fault_step is not None:
            break
    d = ex.fault_step + 1
    ref = ex.done if d >= len(ex.steps) else ex.steps[d]
    return (turns_prefix(goal, ex.steps[:d]), (ref, d >= len(ex.steps)),
            "recover:" + ex.injected[1])


FAMILIES_V2 = {
    "distractor": v2_distractor,
    "indirect": v2_indirect,
    "mismatch": v2_mismatch,
    "stop-yes": v2_stop_yes,
    "stop-no": v2_stop_no,
    "longchain": v2_longchain,
    "recover": v2_recover,
}


def main_v2(args):
    from agent_world import respond
    rng = random.Random(args.seed)
    files, contents, cmds = load_pools(args)
    rng.shuffle(files)
    P = Pools(files, contents, cmds)
    if args.only and args.only not in FAMILIES_V2:
        print("unknown family: %s (want one of: %s)" % (args.only, "|".join(sorted(FAMILIES_V2))))
        return 2
    rows, mix, seen, dupes = [], {}, set(), 0
    for _ in range(args.n):
        if args.only:
            fn = FAMILIES_V2[args.only]
        elif args.unscripted_errors > 0 and rng.random() < args.unscripted_errors:
            fn = v2_recover
        else:
            fn = FAMILIES_V2[rng.choice(FAMILY_NAMES)]
        prompt, (ref, is_done), fam = fn(rng, P, args)
        if prompt in seen:
            dupes += 1
            continue
        seen.add(prompt)
        resp = respond(ref, rng, args.reason, is_done)
        mix[fam] = mix.get(fam, 0) + 1
        rows.append("%s\t%s" % (prompt.replace("\t", " "), resp))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rows) + "\n")
    print("wrote %d tasks -> %s (v2: vfs=%s grammar=%s errors=%.2f holdout=%s)"
          % (len(rows), args.out, args.vfs, args.goal_grammar, args.unscripted_errors,
             args.holdout_lexicon))
    if dupes:
        print("  dropped %d duplicate prompt(s)" % dupes)
    print("family mix:")
    for k in sorted(mix, key=lambda k: -mix[k]):
        print("  %-24s %d" % (k, mix[k]))
    return 0


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
                         "stop-yes|stop-no|longchain|danger; recover needs --vfs)")
    ap.add_argument("--danger", type=float, default=0.0,
                    help="opt-in: fraction of tasks drawn from the 'danger' family "
                         "(vague destructive goal -> look first / ask). Default 0: off")
    ap.add_argument("--seed", type=int, default=1234)
    add_v2_args(ap)
    args = normalize_v2_args(ap.parse_args())
    if args.vfs:
        # --unscripted-errors RATE here = share of tasks drawn from the `recover`
        # family (an unannounced fault; decision = the step right after it)
        return main_v2(args)

    rng = random.Random(args.seed)
    if args.holdout_lexicon:                 # legacy families, split lexicon
        files, contents, cmds = load_pools(args)
    else:
        files = slugs_from_wiki(args.wiki)
        cmds = commands_from(args.commands)
        contents = contents_from(files)
    rng.shuffle(files)

    all_families = dict(FAMILIES, **OPT_IN_FAMILIES)
    if args.only and args.only not in all_families:
        print("unknown family: %s (want one of: %s)"
              % (args.only, "|".join(sorted(all_families))))
        return 2
    if (args.only == "danger" or args.danger > 0) and not danger_commands_safe():
        return 2

    rows, mix, seen = [], {}, set()
    dupes = 0
    for _ in range(args.n):
        if args.only:
            fn = all_families[args.only]
        elif args.danger > 0 and rng.random() < args.danger:   # no draw when off
            fn = build_danger
        else:
            fn = FAMILIES[rng.choice(FAMILY_NAMES)]
        prompt, resp, fam = fn(rng, files, cmds, contents)
        if prompt in seen:            # a duplicate prompt is a wasted rollout budget
            dupes += 1
            continue
        seen.add(prompt)
        if args.reason and fam == "danger":
            resp = "<think>%s</think>%s" % (rng.choice(
                DANGER_RATIONALE_ASK if resp.startswith("<done>") else DANGER_RATIONALE_LOOK), resp)
        elif args.reason:
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
