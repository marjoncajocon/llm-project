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


def contents_from(files):
    """A LARGE, varied pool of write-file contents.

    Drawing every write's content from the tiny fixed WORDS list lets a small
    model MEMORIZE the content set instead of learning to COPY the content out of
    the prompt -- observed as: it copies novel FIRST-write content fine but drops
    the SECOND write's content (which it only ever saw as one of ~10 words) to an
    empty body. Mixing in readable phrases derived from the (varied) wiki slugs
    makes the content vocabulary big enough that copy-from-prompt is the only way
    to fit, in EVERY position."""
    pool = list(WORDS)
    for s in files:
        phrase = s.replace("_", " ").strip()
        if 2 <= len(phrase) <= 24 and phrase not in pool:
            pool.append(phrase)
    return pool


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


# Realistic, often MULTI-LINE output for a run_command step. Training every run
# observation as a bare "ok" left the model brittle: at inference a VERBOSE real
# observation (e.g. `git status`) is out-of-distribution, so it stopped early with
# <done> (lever #4 case B4). Varying the observation -- silent, terse, or multi-line
# per command family -- teaches it to CONTINUE the chain regardless of tool verbosity.
# The transcript flattens \n to spaces (emit_pairs), matching how real output arrives.
# SUCCESS outputs deliberately avoid the critic's error hints (error / failed / cannot
# / not found / no such / denied / traceback / exception) so a good step is never
# misread as a failure (see tools/agent_critic.py obs_is_error).
_RUN_OUT = {
    "git": ["On branch agent-1\nnothing to commit, working tree clean",
            "On branch agent-1\nYour branch is up to date with 'origin/agent-1'.",
            "Already up to date."],
    "cargo": ["   Compiling app v0.1.0\n    Finished dev target(s) in 1.20s",
              "running 3 tests\ntest tests::ok ... ok\ntest result: ok. 3 passed",
              "    Finished release [optimized] target(s) in 4.51s"],
    "go": ["ok   example/app   0.184s",
           "PASS\nok   example/pkg   0.012s",
           "go: downloading dependencies\nok"],
    "make": ["cc -c main.c\ncc -o app main.o\nbuild complete",
             "make: Nothing to be done for 'all'."],
    "npm": ["added 42 packages in 3s", "up to date in 512ms"],
    "npx": ["added 42 packages in 3s"],
    "pip": ["Successfully installed app-1.0", "Requirement already satisfied: app"],
    "pip3": ["Successfully installed app-1.0"],
    "docker": ["Successfully built 3f2a1b\nSuccessfully tagged app:latest",
               "app   latest   3f2a1b   2 minutes ago   128MB"],
    "python": ["hello world", "3.9.10"],
    "python3": ["hello world"],
    "node": ["hello world", "v18.17.0"],
    "ls": ["src  README.md  Cargo.toml  target"],
    "pwd": ["/home/work/project"],
    "curl": ["  % Total    % Received\n100  1256  100  1256  0 --:--:-- 12560"],
}


def run_obs(command, rng):
    verb = (command.split() or [""])[0]
    if rng.random() < 0.25:                       # some tools are silent / terse
        return rng.choice(["", "ok", "done"])
    opts = _RUN_OUT.get(verb)
    return rng.choice(opts) if opts else rng.choice(
        ["ok", "done", "completed", "output written"])


def build_tasks(rng, files, cmds, contents):
    """Return (goal, [(action, observation), ...]) for one random task.

    Mix of 1-, 2-, 3- and 4-step chains. Every tool ARG (path, write content, and
    any filename inside a run_command) is drawn from the goal string, so clean_sft.py's
    grounding verifier keeps them all. The 3- and 4-step chains (verify / write-read-
    delete / error-recovery) teach the model to keep going past the second action and
    to react to an OBSERVATION (incl. an error) instead of stopping early."""
    f1 = rng.choice(files) + rng.choice(EXTS)
    f2 = rng.choice(files) + rng.choice(EXTS)
    w = rng.choice(contents)
    c = rng.choice(cmds)

    def w2_of(w):
        """A DIFFERENT content word, so a 2nd write must be copied from the goal
        (not memorized as a constant) and the two steps stay distinguishable."""
        x = w
        while x == w:
            x = rng.choice(contents)
        return x

    kind = rng.random()

    if kind < 0.12:                                   # write then read (verify)
        g = rng.choice(["create %s containing %s and read it back",
                        "write %s to %s, then read it to confirm"])
        goal = (g % (f1, w)) if "containing" in g else (g % (w, f1))
        return goal, [(tool_write(f1, w), "wrote %d bytes to %s" % (len(w), f1)),
                      (tool_read(f1), w)]

    if kind < 0.24:                                   # two writes
        w2 = w2_of(w)
        goal = "create %s with %s, then create %s with %s" % (f1, w, f2, w2)
        return goal, [(tool_write(f1, w), "wrote %d bytes to %s" % (len(w), f1)),
                      (tool_write(f2, w2), "wrote %d bytes to %s" % (len(w2), f2))]

    if kind < 0.35:                                   # read then delete
        goal = rng.choice(["read %s then delete it", "open %s and then remove it"]) % f1
        return goal, [(tool_read(f1), w),
                      (tool_run("rm %s" % f1), "")]

    if kind < 0.46:                                   # run a command then list files
        goal = "run %s and then list the files" % c
        return goal, [(tool_run(c), run_obs(c, rng)),
                      (tool_run("ls"), "%s  %s" % (f1, f2))]

    if kind < 0.57:                                   # 3-step: write -> run -> read (verify chain)
        goal = "create %s containing %s, run %s, then read it back to verify" % (f1, w, c)
        return goal, [(tool_write(f1, w), "wrote %d bytes to %s" % (len(w), f1)),
                      (tool_run(c), run_obs(c, rng)),
                      (tool_read(f1), w)]

    if kind < 0.67:                                   # 3-step: write -> read -> delete
        goal = rng.choice(
            ["create %s with %s, read it back, then delete it",
             "write %s to %s, confirm it by reading, then remove it"])
        goal = (goal % (f1, w)) if goal.startswith("create") else (goal % (w, f1))
        return goal, [(tool_write(f1, w), "wrote %d bytes to %s" % (len(w), f1)),
                      (tool_read(f1), w),
                      (tool_run("rm %s" % f1), "")]

    if kind < 0.77:                                   # 3-step: error recovery (missing file)
        # The first read FAILS (obs = "not found"); the model must react to the
        # error by CREATING the file, then read it. Teaches recovery, not just
        # happy-path chaining.
        goal = rng.choice(
            ["read %s, but if it is missing create it with %s and read it again",
             "open %s; if it does not exist, write %s to it and re-read it"]) % (f1, w)
        return goal, [(tool_read(f1), "error: %s: not found" % f1),
                      (tool_write(f1, w), "wrote %d bytes to %s" % (len(w), f1)),
                      (tool_read(f1), w)]

    if kind < 0.83:                                   # 4-step: two writes -> run -> read first
        w2 = w2_of(w)
        goal = ("create %s with %s and %s with %s, run %s, then read %s"
                % (f1, w, f2, w2, c, f1))
        return goal, [(tool_write(f1, w), "wrote %d bytes to %s" % (len(w), f1)),
                      (tool_write(f2, w2), "wrote %d bytes to %s" % (len(w2), f2)),
                      (tool_run(c), run_obs(c, rng)),
                      (tool_read(f1), w)]

    if kind < 0.89:                                   # 4-step: error recovery + run + verify
        goal = ("read %s; if it is missing create it with %s, then run %s and read it again"
                % (f1, w, c))
        return goal, [(tool_read(f1), "error: %s: not found" % f1),
                      (tool_write(f1, w), "wrote %d bytes to %s" % (len(w), f1)),
                      (tool_run(c), run_obs(c, rng)),
                      (tool_read(f1), w)]

    if kind < 0.94:                                   # 4-step: write -> read -> run -> delete
        goal = ("create %s with %s, read it back, run %s, then delete it"
                % (f1, w, c))
        return goal, [(tool_write(f1, w), "wrote %d bytes to %s" % (len(w), f1)),
                      (tool_read(f1), w),
                      (tool_run(c), run_obs(c, rng)),
                      (tool_run("rm %s" % f1), "")]

    if kind < 0.97:                                   # single write step
        goal = rng.choice(["create %s with %s", "save %s to %s"])
        goal = (goal % (f1, w)) if goal.startswith("create") else (goal % (w, f1))
        return goal, [(tool_write(f1, w), "wrote %d bytes to %s" % (len(w), f1))]

    # single run step
    goal = rng.choice(["run %s", "execute %s", "please run %s"]) % c
    return goal, [(tool_run(c), run_obs(c, rng))]


# A short, grounded rationale for an action, inferred from the tool call itself.
# Reasoning-dense SFT trains the model to state WHY before acting; the rationale
# is ephemeral (emitted then dropped from the transcript), so it need not persist.
def rationale_for(action, rng):
    m = re.search(r'name="(\w+)"', action)
    name = m.group(1) if m else ""
    path = re.search(r'path="([^"]*)"', action)
    cmd = re.search(r'command="([^"]*)"', action)
    if name == "write_file":
        return rng.choice(["First I'll create %s with the requested content.",
                           "I need to write the content to %s."]) % (path.group(1) if path else "the file")
    if name == "read_file":
        return rng.choice(["Now I'll read %s back to verify it.",
                           "Read %s to check its contents."]) % (path.group(1) if path else "the file")
    if name == "run_command":
        c = cmd.group(1) if cmd else ""
        if c.startswith("rm "):
            return "Delete %s as asked." % c[3:]
        if c == "ls" or c.startswith("ls "):
            return "List the files to see the result."
        return "Run %s to do this." % c
    return ""


def emit_pairs(goal, steps, rng, reason):
    """Explode a task into (transcript_prefix, next_action) pairs, ending in <done>.

    With reason=True the RESPONSE is "<think>rationale</think>action" (the model
    plans out loud, then acts), but the transcript MODEL turns stay bare actions,
    so prior reasoning is not fed back — matching agent_local.py at inference.
    """
    pairs = []
    turns = ["GOAL: " + goal]
    for action, obs in steps:
        resp = action
        if reason:
            r = rationale_for(action, rng)
            if r:
                resp = "<think>%s</think>%s" % (r, action)
        pairs.append((SEP.join(turns), resp))
        turns.append("MODEL: " + action)          # bare action in history
        turns.append("OBSERVATION: " + obs.replace("\n", " ").replace("\t", " "))
    done = "<done>%s</done>" % rng.choice(DONE)
    if reason:
        done = "<think>%s</think>%s" % (rng.choice(
            ["Every step is finished, so I'm done.", "The goal is complete."]), done)
    pairs.append((SEP.join(turns), done))
    return pairs


# ---------------------------------------------------------------------------
# v2: plan -> world -> observations (opt-in; see --vfs / --goal-grammar /
# --unscripted-errors / --holdout-lexicon). NOTHING below runs without a flag,
# so the legacy stream above stays byte-identical for the same seed.
# ---------------------------------------------------------------------------
class Pools:
    """The generator's lexicon: filename slugs, write contents, commands."""

    def __init__(self, files, contents, cmds):
        self.files, self.contents, self.cmds = files, contents, cmds

    def fname(self, rng, avoid=()):
        for _ in range(200):
            f = rng.choice(self.files) + rng.choice(EXTS)
            if f not in avoid:
                return f
        return f

    def word(self, rng, avoid=()):
        for _ in range(200):
            w = rng.choice(self.contents)
            if w not in avoid:
                return w
        return w


def _cl(op, **kw):
    kw["op"] = op
    return kw


def plan_legacy_family(rng, P):
    """The 12 legacy families as PLANS, same mix, plus their legacy goal text."""
    f1 = P.fname(rng)
    f2 = P.fname(rng, avoid=(f1,))
    w = P.word(rng)
    w2 = P.word(rng, avoid=(w,))
    c = rng.choice(P.cmds)
    k = rng.random()
    if k < 0.12:
        g = rng.choice(["create %s containing %s and read it back",
                        "write %s to %s, then read it to confirm"])
        goal = (g % (f1, w)) if "containing" in g else (g % (w, f1))
        return [_cl("write", f=f1, w=w), _cl("verify", f=f1)], goal
    if k < 0.24:
        return ([_cl("write", f=f1, w=w), _cl("write", f=f2, w=w2)],
                "create %s with %s, then create %s with %s" % (f1, w, f2, w2))
    if k < 0.35:
        return ([_cl("read", f=f1), _cl("rm", f=f1)],
                rng.choice(["read %s then delete it", "open %s and then remove it"]) % f1)
    if k < 0.46:
        return [_cl("run", c=c), _cl("ls")], "run %s and then list the files" % c
    if k < 0.57:
        return ([_cl("write", f=f1, w=w), _cl("run", c=c), _cl("verify", f=f1)],
                "create %s containing %s, run %s, then read it back to verify" % (f1, w, c))
    if k < 0.67:
        g = rng.choice(["create %s with %s, read it back, then delete it",
                        "write %s to %s, confirm it by reading, then remove it"])
        goal = (g % (f1, w)) if g.startswith("create") else (g % (w, f1))
        return [_cl("write", f=f1, w=w), _cl("verify", f=f1), _cl("rm", f=f1)], goal
    if k < 0.77:
        g = rng.choice(["read %s, but if it is missing create it with %s and read it again",
                        "open %s; if it does not exist, write %s to it and re-read it"])
        return [_cl("ensure", f=f1, w=w, reread=True)], g % (f1, w)
    if k < 0.83:
        return ([_cl("write", f=f1, w=w), _cl("write", f=f2, w=w2), _cl("run", c=c),
                 _cl("read", f=f1)],
                "create %s with %s and %s with %s, run %s, then read %s"
                % (f1, w, f2, w2, c, f1))
    if k < 0.89:
        return ([_cl("ensure", f=f1, w=w, reread=False), _cl("run", c=c),
                 _cl("reread", f=f1)],
                "read %s; if it is missing create it with %s, then run %s and read it again"
                % (f1, w, c))
    if k < 0.94:
        return ([_cl("write", f=f1, w=w), _cl("verify", f=f1), _cl("run", c=c),
                 _cl("rm", f=f1)],
                "create %s with %s, read it back, run %s, then delete it" % (f1, w, c))
    if k < 0.97:
        g = rng.choice(["create %s with %s", "save %s to %s"])
        goal = (g % (f1, w)) if g.startswith("create") else (g % (w, f1))
        return [_cl("write", f=f1, w=w)], goal
    return [_cl("run", c=c)], rng.choice(["run %s", "execute %s", "please run %s"]) % c


def plan_extra_family(rng, P):
    """Families only the grammar can phrase (no legacy template exists)."""
    f1 = P.fname(rng)
    f2 = P.fname(rng, avoid=(f1,))
    w = P.word(rng)
    c = rng.choice(P.cmds)
    k = rng.random()
    if k < 0.2:
        return [_cl("ls"), _cl("read", f=f1)]
    if k < 0.35:
        return [_cl("read", f=f1)]
    if k < 0.55:
        pl = [_cl("copy", f=f1, g=f2)]
        if rng.random() < 0.5:
            pl.append(_cl("verify", f=f2))
        return pl
    if k < 0.7:
        return [_cl("write", f=f1, w=w), _cl("ls")]
    if k < 0.8:
        return [_cl("rm", f=f1)]
    if k < 0.9:
        return [_cl("read", f=f1), _cl("run", c=c), _cl("rm", f=f1)]
    return [_cl("write", f=f1, w=w), _cl("run", c=c), _cl("ls"), _cl("verify", f=f1)]


def world_for(plan, rng, P, extras=(0, 3)):
    """Initial workspace: every file the plan READS or DELETES before writing it
    exists (content from the pool), plus a few unrelated files so `ls` is real."""
    from agent_world import World, plan_files
    files, touched = {}, set()
    for f, op in plan_files(plan):
        touched.add(f)
        if op in ("read", "verify", "reread", "rm", "copy", "desc", "read_ord"):
            files[f] = P.word(rng)
    for _ in range(rng.randint(*extras)):
        x = P.fname(rng, avoid=touched | set(files))
        files[x] = P.word(rng)
    return World(files)


INJECT_KINDS = ["not-found", "cmd-not-found", "nonzero-exit", "denied", "write-failed"]
# every plan can host a denial, only some a missing file -- weight the kinds so
# the always-available ones do not crowd out the informative ones
INJECT_WEIGHT = {"not-found": 3.0, "cmd-not-found": 2.0, "nonzero-exit": 2.0,
                 "denied": 1.0, "write-failed": 1.0}


def pick_injection(rng, plan, world, want=None):
    """A random (clause, kind) fault the plan admits; the kind is drawn first
    (weighted over the kinds this plan can host), then the clause."""
    from agent_world import injection_points
    pts = injection_points(plan, world)
    if want:
        pts = [p for p in pts if p[1] == want]
    if not pts:
        return None
    kinds = sorted(set(k for _, k in pts), key=INJECT_KINDS.index)
    kind = rng.choices(kinds, weights=[INJECT_WEIGHT[k] for k in kinds])[0]
    return rng.choice([p for p in pts if p[1] == kind])


def build_task_v2(rng, P, opts, force_inject=False, want_kind=None, plan=None, goal=None,
                  world=None):
    """-> (goal, Exec) for one task run against a fresh world."""
    from agent_world import Exec, prepare_injection
    from goal_grammar import render_goal
    if plan is None:
        if opts.goal_grammar and rng.random() < 0.3:
            plan, goal = plan_extra_family(rng, P), None
        else:
            plan, goal = plan_legacy_family(rng, P)
    if world is None:
        world = world_for(plan, rng, P)
    inject = None
    if force_inject or (opts.unscripted_errors > 0 and rng.random() < opts.unscripted_errors):
        inject = pick_injection(rng, plan, world, want_kind)
    prepare_injection(world, plan, inject, rng, EXTS)
    ex = Exec(world, rng, EXTS, rm_verbose=0.3).run(plan, inject)
    if opts.goal_grammar or goal is None:
        goal = render_goal(plan, rng, opts.holdout_lexicon)
    return goal, ex


def load_pools(args):
    """Pools for the v2 / holdout paths (the legacy path builds its own)."""
    from agent_lexicon import apply_holdout
    from agent_world import STATEFUL_VERBS
    lim = 100000 if args.holdout_lexicon else 300
    files = slugs_from_wiki(args.wiki, limit=lim)
    cmds = commands_from(args.commands, limit=lim)
    contents = contents_from(files)
    if args.holdout_lexicon:
        files, contents, cmds = apply_holdout(files, contents, cmds, args.holdout_lexicon,
                                              args.holdout_frac)
    if args.vfs:
        cmds = [c for c in cmds if (c.split() or [""])[0] not in STATEFUL_VERBS] or ["make"]
    return files, contents, cmds


def add_v2_args(ap):
    ap.add_argument("--vfs", action="store_true",
                    help="observations computed from a virtual filesystem (agent_world.py)")
    ap.add_argument("--goal-grammar", action="store_true",
                    help="render goals from goal_grammar.py (implies --vfs; adds families)")
    ap.add_argument("--unscripted-errors", type=float, default=0.0, metavar="RATE",
                    help="share of tasks with ONE unannounced fault + recovery (implies --vfs)")
    ap.add_argument("--holdout-lexicon", choices=["train", "eval"], default=None,
                    help="restrict words/commands/phrasings to one side of the fixed split")
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="share of the lexicon reserved for eval (default 0.2)")


def normalize_v2_args(args):
    if (args.goal_grammar or args.unscripted_errors > 0) and not args.vfs:
        args.vfs = True
        print("note: --goal-grammar / --unscripted-errors imply --vfs")
    return args


def main_v2(args):
    from agent_world import emit_pairs_v2
    rng = random.Random(args.seed)
    files, contents, cmds = load_pools(args)
    rng.shuffle(files)
    P = Pools(files, contents, cmds)
    rows, steps_hist, inj = [], {}, {}
    for _ in range(args.n):
        goal, ex = build_task_v2(rng, P, args)
        steps_hist[len(ex.steps)] = steps_hist.get(len(ex.steps), 0) + 1
        k = ex.injected[1] if ex.injected else "none"
        inj[k] = inj.get(k, 0) + 1
        for prompt, resp in emit_pairs_v2(goal, ex.steps, ex.done, rng, args.reason):
            rows.append("%s\t%s" % (prompt.replace("\t", " "), resp))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rows) + "\n")
    print("wrote %d pairs from %d tasks -> %s (v2: vfs=%s grammar=%s errors=%.2f "
          "holdout=%s; %d files, %d contents, %d commands)"
          % (len(rows), args.n, args.out, args.vfs, args.goal_grammar,
             args.unscripted_errors, args.holdout_lexicon, len(files), len(contents),
             len(cmds)))
    print("  task lengths (steps -> count): %s"
          % ", ".join("%d:%d" % (k, v) for k, v in sorted(steps_hist.items())))
    print("  injected faults: %s" % ", ".join("%s:%d" % kv for kv in sorted(inj.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/traj.txt")
    ap.add_argument("--n", type=int, default=260, help="number of tasks (each -> several pairs)")
    ap.add_argument("--wiki", default=None)
    ap.add_argument("--commands", default=None)
    ap.add_argument("--reason", action="store_true",
                    help="reasoning-dense: prefix each action with <think>rationale</think>")
    ap.add_argument("--seed", type=int, default=1234)
    add_v2_args(ap)
    args = normalize_v2_args(ap.parse_args())
    if args.vfs:
        return main_v2(args)

    rng = random.Random(args.seed)
    if args.holdout_lexicon:                 # legacy templates, split lexicon
        files, contents, cmds = load_pools(args)
    else:
        files = slugs_from_wiki(args.wiki)
        cmds = commands_from(args.commands)
        contents = contents_from(files)          # large, varied write-content pool
    rng.shuffle(files)

    rows, steps_hist = [], {}
    for _ in range(args.n):
        goal, steps = build_tasks(rng, files, cmds, contents)
        steps_hist[len(steps)] = steps_hist.get(len(steps), 0) + 1
        for prompt, resp in emit_pairs(goal, steps, rng, args.reason):
            rows.append("%s\t%s" % (prompt.replace("\t", " "), resp))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rows) + "\n")
    print("wrote %d pairs from %d tasks -> %s (%d-word content pool)"
          % (len(rows), args.n, args.out, len(contents)))
    print("  task lengths (steps -> count): %s"
          % ", ".join("%d:%d" % (k, v) for k, v in sorted(steps_hist.items())))


if __name__ == "__main__":
    main()
