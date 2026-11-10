#!/usr/bin/env python3
"""Measure how observation formatting changes the agent's NEXT action.

    python tools/obs_probe.py model/smollm/reason_e2.bin model/smollm/tok.hf [--n 12]

The live run exposed a failure: after a command printed VERBOSE output (real
`git status`, not the one-line "ok" the SFT data has), the model stopped one step
early with <done>. This probe builds multi-step tasks whose last observation is a
realistic multi-line command output and whose correct next action is still a tool
call, then generates the next action under several context-editing settings
(tools/agent_local.py: AGENT_OBS_JOIN / AGENT_OBS_LINES / AGENT_KEEP_OBS) and
reports, per setting: right next action (tool + path), premature <done>, other.

Same prompts, same seed, same temperature for every setting -- the ONLY thing
that differs is how the observations are rendered into the prompt.
"""
import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import agent_local as al  # noqa: E402
from agent_critic import run_hfgen  # noqa: E402
import re  # noqa: E402

TOOL_RE = re.compile(r'<tool\s+name="([a-z_]+)"(?:\s+(?:path|command)="([^"]*)")?')


def parse_action(text):
    """(kind, path) of the FIRST action block: kind in write_file / read_file /
    run_command / done / none; path is the path (or command) attribute."""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m, d = TOOL_RE.search(t), t.find("<done")
    if d >= 0 and (not m or d < m.start()):
        return "done", ""
    if m:
        return m.group(1), m.group(2) or ""
    return "none", ""

VERBOSE = {
    "git status": "On branch main\nYour branch is up to date with 'origin/main'.\n\n"
                  "Changes not staged for commit:\n  (use \"git add <file>...\" to update what will be committed)\n"
                  "\tmodified:   src/app.py\n\tmodified:   README.md\n\n"
                  "no changes added to commit (use \"git add\" and/or \"git commit -a\")",
    "ls -la": "total 24\ndrwxr-xr-x 4 user user 4096 Nov 10 09:12 .\n"
              "drwxr-xr-x 9 user user 4096 Nov 10 09:00 ..\n"
              "-rw-r--r-- 1 user user  220 Nov 10 09:10 config.json\n"
              "-rw-r--r-- 1 user user 1024 Nov 10 09:11 main.py\n"
              "drwxr-xr-x 2 user user 4096 Nov 10 09:12 tests",
    "pip list": "Package    Version\n---------- -------\nnumpy      1.26.4\nrequests   2.31.0\n"
                "setuptools 69.0.3\nwheel      0.42.0",
    "docker ps": "CONTAINER ID   IMAGE          COMMAND                  STATUS         NAMES\n"
                 "3f2a1b9c0d11   nginx:latest   \"/docker-entrypoint.…\"   Up 2 hours     web\n"
                 "8e7d6c5b4a32   redis:7        \"docker-entrypoint.s…\"   Up 2 hours     cache",
    "npm test": "> app@1.0.0 test\n> jest\n\nPASS  tests/sum.test.js\n  sum\n    ✓ adds 1 + 2 (3 ms)\n\n"
                "Test Suites: 1 passed, 1 total\nTests:       1 passed, 1 total\nTime:        0.84 s",
    "df -h": "Filesystem      Size  Used Avail Use% Mounted on\n/dev/sda1        50G   21G   27G  44% /\n"
             "tmpfs           7.8G     0  7.8G   0% /dev/shm\n/dev/sdb1       200G   88G  112G  44% /data",
}
NAMES = ["report", "summary", "todo", "notes", "status_log", "plan", "result", "checklist"]
WORDS = ["release notes", "weekly sync", "fix the login bug", "deploy at noon",
         "cache warmed", "tests are green", "update docs", "review pending"]

SETTINGS = [
    ("off (old prompt)", dict(join=False, lines=0, keep=0)),
    ("join", dict(join=True, lines=0, keep=0)),
    ("join+lines4", dict(join=True, lines=4, keep=0)),
    ("join+lines2", dict(join=True, lines=2, keep=0)),
    ("join+lines1", dict(join=True, lines=1, keep=0)),
    ("lines2 (no join)", dict(join=False, lines=2, keep=0)),
]


def nonce():
    return "%08x" % random.getrandbits(32)


def frame(obs):
    n = nonce()
    return "OBSERVATION: [[obs:%s]]\n%s\n[[/obs:%s]]\n" % (n, obs, n)


def make_cases(n, rng):
    """(raw framed transcript, expected tool, expected path) -- two shapes:
    run CMD then write F   (last obs verbose, next = write_file F)
    write F, run CMD, then read F (last obs verbose, next = read_file F)."""
    cases = []
    cmds = list(VERBOSE)
    for i in range(n):
        cmd = cmds[(i // 2) % len(cmds)]   # every command in both task shapes
        f = "%s.txt" % NAMES[rng.randrange(len(NAMES))]
        body = WORDS[rng.randrange(len(WORDS))]
        if i % 2 == 0:
            goal = "GOAL: run %s, then create %s containing %s" % (cmd, f, body)
            raw = "%s\nMODEL: %s\n%s" % (goal, '<tool name="run_command" command="%s"></tool>' % cmd,
                                          frame(VERBOSE[cmd]))
            cases.append((raw, "write_file", f))
        else:
            goal = "GOAL: create %s containing %s, then run %s, then read %s" % (f, body, cmd, f)
            raw = "%s\nMODEL: %s\n%sMODEL: %s\n%s" % (
                goal, '<tool name="write_file" path="%s">%s</tool>' % (f, body),
                frame("wrote %d bytes to %s" % (len(body), f)),
                '<tool name="run_command" command="%s"></tool>' % cmd, frame(VERBOSE[cmd]))
            cases.append((raw, "read_file", f))
    return cases


def prompt_for(raw, s):
    goal, items = al.parse_items(raw)
    if not (s["join"] or s["lines"] or s["keep"]):
        return al.flatten(al.unwrap(raw))
    return al.render(goal, al.edit_items(items, s["join"], s["lines"], s["keep"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("tok")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--show", action="store_true", help="print each prompt/response")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    random.seed(a.seed)
    cases = make_cases(a.n, rng)
    print("obs_probe: %d cases, temp %g, seed %d, model %s" % (len(cases), a.temp, a.seed, a.model))
    rows = []
    for name, s in SETTINGS:
        right = done = other = 0
        for ci, (raw, want_tool, want_path) in enumerate(cases):
            p = prompt_for(raw, s)
            out = run_hfgen(al.ENGINE, a.model, a.tok, p, a.temp, a.seed + ci, n_new=al.N_NEW)[0]
            kind, path = parse_action(out)
            if kind == want_tool and path == want_path:
                right += 1
            elif kind == "done":
                done += 1
            else:
                other += 1
            if a.show:
                print("--- [%s] case %d want %s %s\nPROMPT: %s\nREPLY:  %s" % (
                    name, ci, want_tool, want_path, p, out.strip()))
        rows.append((name, right, done, other))
        print("  %-18s right %2d/%d   premature-done %2d   other %2d" % (name, right, len(cases), done, other))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
