#!/usr/bin/env python3
"""A tiny virtual filesystem the agent generators run their plans against (--vfs).

The legacy generators write each OBSERVATION as a literal in the template: `ls`
prints two names that were never created, a terminal `rm` always prints "" (which
made EMPTY -> <done> a perfect predictor, see memory.md), a file read after it was
deleted would still "succeed", and the only error that ever occurs is the one the
goal announced in advance. The model learns the templates, not the world.

Here a task is a PLAN (goal_grammar clause dicts) executed by a small scripted
POLICY against a World (a dict path -> content). Every observation is computed
from the state:

    write_file  -> "wrote N bytes to f"      (N = bytes actually stored)
    read_file   -> the content, or "error: cannot read 'f'" / "error: f: not found"
    rm f        -> "" / "removed 'f'" (rm -v), or rm's not-found error + exit status
    ls          -> the files that REALLY exist, one per line (sorted)
    anything    -> "denied by user" when the user refuses the step

Unscripted errors (--unscripted-errors): the executor can be told to break ONE
step -- file missing, command not installed, non-zero exit, user denial, failed
write -- with NO warning in the goal. The policy then takes a sensible recovery
(ls to look for the file, fix-and-retry, retry a transient failure, or a <done>
that REPORTS the failure), and its rationale quotes the observation it reacts to.

replay(prompt, response) re-derives what the transcript implies about the world
and rejects rows that contradict themselves (clean_sft.py --replay).
"""
import re

from make_agent_traj import SEP, rationale_for, run_obs, tool_read, tool_run, tool_write

DONE = ["Task complete.", "All steps done.", "Finished the task."]
DONE_WHY = ["Every step is finished, so I'm done.", "The goal is complete."]

# commands whose effect on the workspace we would have to model -- kept out of
# the generic command pool in --vfs mode (the executor emits ls / rm itself)
STATEFUL_VERBS = {"ls", "rm", "cat", "touch", "mv", "cp", "mkdir", "rmdir", "cd",
                  "echo", "tee", "truncate", "ln", "unlink", "shred"}

# command-not-found: a known working substitute, when there is one
CMD_FIX = {"python3": "python", "python": "python3", "pip3": "pip", "pip": "pip3",
           "docker-compose": "docker compose", "nodejs": "node", "node": "nodejs"}
# verbs whose failures are plausibly TRANSIENT (network) -> one retry is sensible
NET_VERBS = {"git", "npm", "npx", "pip", "pip3", "curl", "wget", "docker", "cargo",
             "go", "brew", "apt-get", "apt", "yarn", "docker-compose"}
NET_ERR = [
    "fatal: unable to access 'https://github.com/app/app.git/': "
    "Could not resolve host: github.com",
    "error: network timeout while fetching packages",
    "curl: (6) Could not resolve host: example.com",
    "Temporary failure in name resolution",
    "Connection timed out; retrying later may help",
]
HARD_ERR = {
    "git": ("fatal: not a git repository (or any of the parent directories): .git", 128),
    "cargo": ("error: could not compile `app` due to 2 previous errors", 101),
    "docker": ("Cannot connect to the Docker daemon at unix:///var/run/docker.sock. "
               "Is the docker daemon running?", 1),
    "make": ("make: *** No rule to make target 'all'.  Stop.", 2),
    "go": ("go: cannot find main module, but found .git/config", 1),
    "npm": ("npm ERR! missing script: build", 1),
    "python": ("Traceback (most recent call last):\nModuleNotFoundError: "
               "No module named 'app'", 1),
}
GENERIC_ERR = [("fatal: invalid argument", 1), ("error: operation failed", 1),
               ("usage error: unexpected argument", 2)]

READ_ERR_STYLES = ["error: {f}: not found", "error: cannot read '{f}'"]


class World:
    """The workspace: path -> content."""

    def __init__(self, files=None):
        self.files = dict(files or {})

    def listing(self):
        return "\n".join(sorted(self.files))


def _first_line(obs, n=70):
    s = (obs or "").strip().split("\n")[0].strip()
    return s if len(s) <= n else s[:n - 3] + "..."


def _verb(c):
    return (c.split() or [""])[0]


def _cmd_not_found(verb, rng):
    if rng.random() < 0.5:
        return "sh: 1: %s: not found\n[exit status 127]" % verb
    return ("'%s' is not recognized as an internal or external command,\n"
            "operable program or batch file.\n[exit status 1]" % verb)


def _near_name(f, files_now, rng, exts):
    """A plausible 'you meant this one' neighbour of f (other extension)."""
    stem = f.rpartition(".")[0]
    cands = [stem + e for e in exts if stem + e != f and stem + e not in files_now]
    return rng.choice(cands) if cands else None


def plan_files(plan):
    """Files a plan touches, in first-touch order, with the first op."""
    seen = []
    for cl in plan:
        for key in ("f", "g"):
            f = cl.get(key)
            if f and f not in [x for x, _ in seen]:
                op = cl["op"] if not (cl["op"] == "copy" and key == "g") else "write"
                seen.append((f, op))
    return seen


def injection_points(plan, world):
    """(clause_index, kind) pairs where an unscripted error makes sense."""
    pts, written = [], set()
    for i, cl in enumerate(plan):
        op = cl["op"]
        if op in ("read", "copy") and cl["f"] not in written and cl["f"] in world.files:
            pts.append((i, "not-found"))
        if op == "run":
            pts += [(i, "cmd-not-found"), (i, "nonzero-exit"), (i, "denied")]
        if op in ("write", "rm", "read", "verify"):
            pts.append((i, "denied"))
        if op == "write":
            pts.append((i, "write-failed"))
        if op in ("write", "ensure"):
            written.add(cl["f"])
        if op == "copy":
            written.add(cl["g"])
    return pts


def prepare_injection(world, plan, inject, rng, exts):
    """Make the world agree with an injected fault BEFORE execution starts.

    not-found: the file the plan reads is simply absent (no warning anywhere);
    half the time a same-stem neighbour with another extension exists, which
    the ls-based recovery can find."""
    if not inject or inject[1] != "not-found":
        return
    f = plan[inject[0]]["f"]
    content = world.files.pop(f, None)
    if content is not None and rng.random() < 0.5:
        n = _near_name(f, world.files, rng, exts)
        if n:
            world.files[n] = content


class Exec:
    """Runs a plan against a World with a scripted policy.

    steps: list of (action, observation, why) -- why=None means "use the default
    action rationale"; done: (text, why) for the final <done>."""

    def __init__(self, world, rng, exts, rm_verbose=0.0, read_err=None):
        self.w = world
        self.rng = rng
        self.exts = exts
        self.rm_verbose = rm_verbose
        self.read_err = read_err or rng.choice(READ_ERR_STYLES)
        self.steps = []
        self.done = None
        self.injected = None
        self.fault_step = None       # index in steps of the injected fault's step
        self.listed = None           # names the agent SAW in its last ls (kept current)

    # -- primitive tool effects (observations derived from state) --------------
    def _write(self, f, w, fault=None, why=None):
        if fault == "denied":
            obs = "denied by user"
        elif fault == "fail":
            obs = "error: write to '%s' failed" % f
        elif fault == "partial":
            k = max(1, len(w) // 2)
            self.w.files[f] = w[:k]
            self._seen_add(f)
            obs = "wrote %d bytes to %s" % (len(w[:k].encode("utf-8")), f)
        else:
            self.w.files[f] = w
            self._seen_add(f)
            obs = "wrote %d bytes to %s" % (len(w.encode("utf-8")), f)
        self.steps.append((tool_write(f, w), obs, why))
        return obs

    def _read(self, f, fault=None, why=None):
        if fault == "denied":
            obs = "denied by user"
        elif f in self.w.files:
            obs = self.w.files[f]
        else:
            obs = self.read_err.format(f=f)
        self.steps.append((tool_read(f), obs, why))
        return obs

    def _rm(self, f, fault=None, why=None):
        verbose = self.rng.random() < self.rm_verbose
        cmd = ("rm -v %s" if verbose else "rm %s") % f
        if fault == "denied":
            obs = "denied by user"
        elif f in self.w.files:
            del self.w.files[f]
            if self.listed is not None:
                self.listed.discard(f)
            obs = "removed '%s'" % f if verbose else ""
        else:
            obs = "rm: cannot remove '%s': No such file or directory\n[exit status 1]" % f
        self.steps.append((tool_run(cmd), obs, why))
        return obs

    def _seen_add(self, f):
        if self.listed is not None:
            self.listed.add(f)

    def _known_missing(self, f):
        """The agent already KNOWS f is absent (an ls showed the directory)."""
        return self.listed is not None and f not in self.listed

    def _ls(self, why=None):
        obs = self.w.listing()
        self.listed = set(self.w.files)
        self.steps.append((tool_run("ls"), obs, why))
        return obs

    def _run(self, c, fault=None, why=None):
        if c == "ls":
            return self._ls(why)
        v = _verb(c)
        if fault == "denied":
            obs = "denied by user"
        elif fault == "cmd-not-found":
            obs = _cmd_not_found(v, self.rng)
        elif fault == "net":
            obs = "%s\n[exit status %d]" % (self.rng.choice(NET_ERR), self.rng.choice([1, 128]))
        elif fault == "hard":
            msg, code = HARD_ERR.get(v) or self.rng.choice(GENERIC_ERR)
            obs = "%s\n[exit status %d]" % (msg, code)
        else:
            obs = run_obs(c, self.rng)
        self.steps.append((tool_run(c), obs, why))
        return obs

    def _fail(self, text, why):
        self.done = (text, why)
        return False

    # -- the policy ---------------------------------------------------------------
    def run(self, plan, inject=None):
        """Execute; inject = (clause_index, kind) or None. Returns self."""
        self.injected = inject
        plan = [dict(cl) for cl in plan]
        expect = {}                                  # f -> content we meant to write
        i = 0
        while i < len(plan):
            cl = plan[i]
            fault = inject[1] if inject and inject[0] == i else None
            if fault:
                self.fault_step = len(self.steps)
            if not self._clause(cl, fault, expect, plan, i):
                return self
            i += 1
        if self.done is None:
            self.done = (self.rng.choice(DONE), None)
        return self

    def _clause(self, cl, fault, expect, plan, i):
        op = cl["op"]
        if op == "write":
            f, w = cl["f"], cl["w"]
            if fault == "denied":
                obs = self._write(f, w, "denied")
                return self._fail(
                    "Stopped: the user denied writing %s, so it was not created." % f,
                    "The last step was refused (\"%s\"), so I won't retry it; "
                    "I'll stop and report." % _first_line(obs))
            if fault == "write-failed":
                obs = self._write(f, w, "fail")
                self._write(f, w, None,
                            "The write failed (\"%s\"), so I'll try writing %s again."
                            % (_first_line(obs), f))
            else:
                self._write(f, w, cl.get("fault"))
            expect[f] = w
            return True

        if op in ("read", "verify", "reread", "read_ord"):
            f = cl["f"]
            if fault == "denied":
                obs = self._read(f, "denied")
                return self._fail(
                    "Stopped: the user denied reading %s." % f,
                    "Reading %s was refused (\"%s\"), so I'll stop and report it."
                    % (f, _first_line(obs)))
            if self._known_missing(f):
                return self._recover_missing(f, None, plan, i)
            why = None
            if op == "read_ord":
                why = ("The %s file I created was %s, so that is the one to read back."
                       % (["first", "second", "third"][cl["k"]], f))
            obs = self._read(f, None, why)
            if f not in self.w.files:
                return self._recover_missing(f, obs, plan, i)
            if f in expect and obs != expect[f]:
                # read-back disagrees with what we wrote: write it again, re-check
                w = expect[f]
                self._write(f, w, None,
                            "The read-back shows \"%s\", not \"%s\", so the write did not "
                            "take; I'll write %s again." % (_first_line(obs, 40), w, f))
                self._read(f, None, "Read %s again to confirm the fix." % f)
            return True

        if op == "ensure":
            f, w = cl["f"], cl["w"]
            obs = self._read(f)
            if f not in self.w.files:
                self._write(f, w, None,
                            "%s does not exist (\"%s\"), so I'll create it with %s."
                            % (f, _first_line(obs), w))
                expect[f] = w
                if cl.get("reread", True):
                    self._read(f)
            return True

        if op == "copy":
            f, g = cl["f"], cl["g"]
            if self._known_missing(f):
                return self._recover_missing(f, None, plan, i)
            obs = self._read(f)
            if f not in self.w.files:
                return self._recover_missing(f, obs, plan, i)
            self._write(g, obs, None,
                        "%s contains \"%s\"; I'll write the same text to %s."
                        % (f, _first_line(obs, 40), g))
            expect[g] = obs
            return True

        if op in ("rm", "desc") and (op == "rm" or cl["verb"] == "rm"):
            f = cl["f"]
            why = None
            if op == "desc":
                why = ("The listing shows %s, the only name with \"%s\" in it; I'll delete it."
                       % (f, cl["p"]))
            obs = self._rm(f, "denied" if fault == "denied" else None, why)
            if fault == "denied":
                return self._fail(
                    "Stopped: the user denied deleting %s, so it is still there." % f,
                    "Deleting %s was refused (\"%s\"); I won't retry, so I'll stop "
                    "and report." % (f, _first_line(obs)))
            if "cannot remove" in obs:
                return self._fail(
                    "%s was already gone, so there was nothing to delete." % f,
                    "rm says \"%s\", so the file is already gone; I'll report that."
                    % _first_line(obs))
            return True

        if op == "desc":                             # read by description
            self._read(cl["f"], None,
                       "The listing shows %s, the only name with \"%s\" in it; I'll read it."
                       % (cl["f"], cl["p"]))
            return True

        if op == "ls":
            self._ls()
            return True

        if op == "run":
            return self._run_clause(cl["c"], fault)
        raise ValueError(op)

    def _run_clause(self, c, fault):
        rng = self.rng
        v = _verb(c)
        if fault == "denied":
            obs = self._run(c, "denied")
            return self._fail(
                "Stopped: the user denied running %s, so the remaining steps were "
                "not done." % c,
                "The command was refused (\"%s\"); I won't run it without approval, "
                "so I'll stop and report." % _first_line(obs))
        if fault == "cmd-not-found":
            obs = self._run(c, "cmd-not-found")
            fix = CMD_FIX.get(v)
            if fix:
                fixed = fix + c[len(v):]
                self._run(fixed, None,
                          "The shell says %s is not installed (\"%s\"); %s is the usual "
                          "alternative, so I'll retry with it." % (v, _first_line(obs), fix))
                return True
            return self._fail(
                "Could not finish: %s is not installed here (command not found), so %s "
                "did not run." % (v, c),
                "\"%s\" means %s is not available and there is no safe substitute, so "
                "I'll stop and report it." % (_first_line(obs), v))
        if fault == "nonzero-exit":
            if v in NET_VERBS and rng.random() < 0.5:
                obs = self._run(c, "net")
                self._run(c, None,
                          "It failed with a network error (\"%s\"), which is usually "
                          "transient, so I'll retry once." % _first_line(obs, 60))
                return True
            obs = self._run(c, "hard")
            code = re.search(r"\[exit status (\d+)\]", obs).group(1)
            return self._fail(
                "Could not finish: %s failed with exit status %s (%s)."
                % (c, code, _first_line(obs, 60)),
                "The command failed (\"%s\", exit status %s); retrying will not change "
                "that, so I'll stop and report the error." % (_first_line(obs, 60), code))
        self._run(c)
        return True

    def _recover_missing(self, f, obs, plan, i):
        """An unannounced missing file: look before giving up. obs=None means an
        earlier ls ALREADY showed it is absent -- do not read it blindly."""
        if obs is None:
            listing = self.w.listing()
            if self.fault_step is not None:
                self.fault_step = len(self.steps) - 1    # the ls revealed the fault
        else:
            listing = self._ls("The read failed (\"%s\"), so I'll list the files to look "
                               "for it." % _first_line(obs))
        names = listing.split("\n") if listing else []
        stem = f.rpartition(".")[0]
        near = [n for n in names if n.rpartition(".")[0] == stem and n != f]
        later = plan[i + 1:]
        destructive = any(cl.get("f") == f and (cl["op"] in ("rm", "write", "ensure") or
                                                cl.get("verb") == "rm") for cl in later)
        if near and not destructive:
            n = near[0]
            for cl in later:
                for k in ("f", "g"):
                    if cl.get(k) == f:
                        cl[k] = n
            cl = plan[i]
            content = self.w.files[n]
            self._read(n, None, "There is no %s, but the listing shows %s, which is "
                                "clearly the same file; I'll read that." % (f, n))
            if cl["op"] == "copy":
                self._write(cl["g"], content, None,
                            "%s contains \"%s\"; I'll write the same text to %s."
                            % (n, _first_line(content, 40), cl["g"]))
            return True
        if near:
            return self._fail(
                "Stopped: %s does not exist. The listing has %s instead, but I did not "
                "change it without asking." % (f, near[0]),
                "The listing shows %s but no %s; the goal would modify it, so I'll stop "
                "and report instead of guessing." % (near[0], f))
        return self._fail(
            "Could not finish: %s does not exist (it is not in the directory listing)." % f,
            "The listing (\"%s\") has no %s, so I cannot continue; I'll report it."
            % (_first_line(listing.replace("\n", " "), 50) or "empty", f))


def emit_pairs_v2(goal, steps, done, rng, reason):
    """Like make_agent_traj.emit_pairs, but steps carry their own rationale.

    steps: [(action, obs, why|None)], done: (text, why|None)."""
    pairs = []
    turns = ["GOAL: " + goal]
    for action, obs, why in steps:
        resp = action
        if reason:
            r = why or rationale_for(action, rng)
            if r:
                resp = "<think>%s</think>%s" % (r, action)
        pairs.append((SEP.join(turns), resp))
        turns.append("MODEL: " + action)
        turns.append("OBSERVATION: " + flat(obs))
    text, why = done
    resp = "<done>%s</done>" % text
    if reason:
        resp = "<think>%s</think>%s" % (why or rng.choice(DONE_WHY), resp)
    pairs.append((SEP.join(turns), resp))
    return pairs


def flat(obs):
    return str(obs).replace("\n", " ").replace("\t", " ")


def turns_prefix(goal, steps):
    t = ["GOAL: " + goal]
    for action, obs, _why in steps:
        t.append("MODEL: " + action)
        t.append("OBSERVATION: " + flat(obs))
    return SEP.join(t)


def respond(step_or_done, rng, reason, is_done=False):
    """Reference response for a decision point (hard_tasks)."""
    if is_done:
        text, why = step_or_done
        r = "<done>%s</done>" % text
        return ("<think>%s</think>%s" % (why or rng.choice(DONE_WHY), r)) if reason else r
    action, _obs, why = step_or_done
    if reason:
        w = why or rationale_for(action, rng)
        if w:
            return "<think>%s</think>%s" % (w, action)
    return action


# ---------------------------------------------------------------------------
# Replay: is a transcript consistent with SOME world state?
# ---------------------------------------------------------------------------
TURN_SPLIT = re.compile(r" \| (?=(?:MODEL|OBSERVATION): )")
TOOL_RE = re.compile(r'<tool name="(\w+)"([^>]*)>(.*?)</tool>', re.DOTALL)
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
WROTE_RE = re.compile(r"^wrote (\d+) bytes to (.+)$")
EXIT_RE = re.compile(r"\[exit status \d+\]")
FILE_RE = re.compile(r"[\w./-]*\.(?:txt|md|log|json|csv)\b", re.IGNORECASE)

def parse_tool(text):
    m = TOOL_RE.search(text or "")
    if not m:
        return None
    attrs = dict(ATTR_RE.findall(m.group(2)))
    return {"name": m.group(1), "path": attrs.get("path", ""),
            "command": attrs.get("command", ""), "body": m.group(3)}


def parse_transcript(prompt):
    """[(action_text, obs_text)] from a flattened transcript, plus the goal."""
    parts = TURN_SPLIT.split(prompt)
    goal = parts[0][len("GOAL: "):] if parts and parts[0].startswith("GOAL: ") else None
    pairs, act = [], None
    for p in parts[1:]:
        if p.startswith("MODEL: "):
            act = p[len("MODEL: "):]
        elif p.startswith("OBSERVATION: "):
            pairs.append((act, p[len("OBSERVATION: "):]))
            act = None
    return goal, pairs


ERR_OBS_RE = re.compile(r"^error:|^denied by user$|\[exit status \d+\]|\[timed out|"
                        r": not found|not recognized as an internal|cannot remove|"
                        r"^traceback", re.IGNORECASE)


def is_error_obs(obs):
    """A FAILED tool step, by the exact shapes toolexec.c / agent.c / the shell
    produce -- not by keywords, so read content like "error detection" or a file
    named error_log.txt is not mistaken for a failure."""
    return ERR_OBS_RE.search((obs or "").strip()) is not None


def _read_error(obs, f):
    return obs in ("error: %s: not found" % f, "error: cannot read '%s'" % f)


def replay(prompt, response=None):
    """None if the transcript is consistent with some initial workspace, else a
    short reason. Open-world: files never mentioned may exist; an `ls` is a
    COMPLETE snapshot (after it, unlisted files are known to be absent)."""
    goal, pairs = parse_transcript(prompt)
    if goal is None:
        return None                                  # not an agent transcript
    known, absent, closed = {}, set(), False         # known: f -> content|None

    def exists(f):
        if f in known:
            return True
        if f in absent or closed:
            return False
        return None

    for act, obs in pairs:
        tc = parse_tool(act)
        if tc is None:
            return "replay-unparsed-action"
        name, obs_s = tc["name"], obs.strip()
        if obs_s == "denied by user":
            continue
        if name == "write_file":
            f, body = tc["path"], tc["body"]
            m = WROTE_RE.match(obs_s)
            if m:
                if m.group(2) != f:
                    return "replay-write-path"
                n = int(m.group(1))
                raw = body.encode("utf-8")
                if n > len(raw):
                    return "replay-write-bytes"
                known[f] = raw[:n].decode("utf-8", "ignore")
                absent.discard(f)
            elif not obs_s.startswith("error:"):
                return "replay-write-obs"
        elif name == "read_file":
            f = tc["path"]
            if _read_error(obs_s, f) or obs_s.startswith("error:"):
                if exists(f) is True:
                    return "replay-read-error-but-exists"
                absent.add(f)
                known.pop(f, None)
            else:
                e = exists(f)
                if e is False:
                    return "replay-read-after-absent"
                if known.get(f) is not None and flat(known[f]).strip() != obs_s:
                    return "replay-read-content"
                known[f] = obs_s
        elif name == "run_command":
            c = tc["command"].strip()
            toks = c.split()
            if toks and toks[0] == "rm" and len(toks) in (2, 3) and \
                    (len(toks) == 2 or toks[1] == "-v"):
                f = toks[-1]
                if "cannot remove" in obs_s:
                    if exists(f) is True:
                        return "replay-rm-error-but-exists"
                elif obs_s in ("", "removed '%s'" % f):
                    if exists(f) is False:
                        return "replay-rm-absent"
                else:
                    return "replay-rm-obs"
                known.pop(f, None)
                absent.add(f)
            elif c == "ls":
                if EXIT_RE.search(obs_s):
                    continue
                names = set(obs_s.split())
                for f in known:
                    if f not in names:
                        return "replay-ls-missing:%s" % f
                for f in absent:
                    if f in names:
                        return "replay-ls-ghost:%s" % f
                for f in names:
                    known.setdefault(f, None)
                closed = True
            else:
                # generic command: assumed not to touch the workspace, except
                # files it names, which become unknown
                for f in FILE_RE.findall(c):
                    known.pop(f, None)
                    absent.discard(f)
        else:
            return "replay-unknown-tool"

    if response is not None:
        resp = THINK_RE.sub("", response).strip()
        tc = parse_tool(resp)
        if tc and tc["name"] == "read_file" and exists(tc["path"]) is False:
            return "replay-target-read-absent"
        # a rationale written right after an ERROR must engage with it
        th = THINK_RE.search(response)
        if th and pairs and is_error_obs(pairs[-1][1]):
            words = set(w for w in re.findall(r"[a-z0-9_.'-]{4,}", pairs[-1][1].lower()))
            said = th.group(1).lower()
            if not any(w in said for w in words) and "error" not in said \
                    and "fail" not in said and "missing" not in said:
                return "replay-rationale-ignores-error"
    return None
