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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/traj.txt")
    ap.add_argument("--n", type=int, default=260, help="number of tasks (each -> several pairs)")
    ap.add_argument("--wiki", default=None)
    ap.add_argument("--commands", default=None)
    ap.add_argument("--reason", action="store_true",
                    help="reasoning-dense: prefix each action with <think>rationale</think>")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = random.Random(args.seed)
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
