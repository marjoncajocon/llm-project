#!/usr/bin/env python3
"""A small goal grammar for the agent generators (--goal-grammar).

The legacy generators render each task family from ONE or TWO fixed template
strings ("create %s with %s, then create %s with %s"), ~11 families in all. A
135M model memorizes those surfaces -- position in the template predicts the
action -- and it never sees the same intent phrased another way (the live model
parsed `list the files` as a FILENAME and emitted read_file path="list").

Here a goal is a PLAN -- a list of clause dicts, the same list the world
executor (agent_world.py) runs -- and the surface text is generated from it:

  * verb synonyms per operation (create/make/save/write/store..., read/open/
    show/print/display..., delete/remove/erase..., run/execute/invoke...)
  * pronoun reference ("read it back") only when it is unambiguous
  * varied connectives, openers, closers, quoting of contents / commands
  * clause ORDER variation ("B, but first A", "before you B, A") and numbered
    step lists, so position in the string stops predicting the next action

Every argument (filename, content, command) is emitted VERBATIM, so every row
still passes clean_sft.verify() grounding. With a `side` ("train"|"eval") each
phrasing pool is split by agent_lexicon.split_phrasings, so an eval set can be
built from phrasings the training set never contains.

Clause dicts (op + fields):
  write{f,w}  read{f}  verify{f}  rm{f}  run{c}  ls{}  ensure{f,w,reread}
  reread{f}   copy{f,g}  read_ord{f,k}  desc{verb,p,f}
"""
from agent_lexicon import split_phrasings

POOLS = {
    "write": [
        "create {f} with {w}", "create {f} containing {w}", "write {w} to {f}",
        "save {w} to {f}", "make {f} with {w}", "make a file {f} containing {w}",
        "put {w} in {f}", "save {w} into a file named {f}",
        "write a file {f} that says {w}", "store {w} in {f}",
        "create a file called {f} with the text {w}", "generate {f} containing {w}",
        "fill {f} with {w}", "add a new file {f} holding {w}",
        "write {f} with the content {w}", "save a file {f} whose content is {w}",
        "create {f} and put {w} in it", "make a new file named {f} with {w} inside",
    ],
    "read": [
        "read {f}", "open {f}", "show {f}", "print {f}", "display {f}",
        "show me {f}", "look at {f}", "view the contents of {f}",
        "output the contents of {f}", "check what {f} says", "read the file {f}",
        "print out {f}", "open up {f}", "dump {f}",
    ],
    "read_it": [
        "read it", "open it", "show it", "print it", "display it", "look at it",
        "view it", "print it out",
    ],
    "verify": [
        "read {f} back", "read {f} back to verify", "check {f} by reading it",
        "confirm {f} by reading it", "verify {f} by reading it", "re-read {f}",
        "display {f} to double-check", "print {f} back out",
        "look at {f} again to make sure", "reopen {f} to confirm it",
        "read {f} to check it",
    ],
    "verify_it": [
        "read it back", "read it back to verify", "confirm it by reading",
        "check it by reading it", "verify it", "read it to confirm",
        "display it to double-check", "print it back out", "re-read it",
        "look at it again to make sure", "reopen it to confirm",
        "read it back to check",
    ],
    "rm": [
        "delete {f}", "remove {f}", "erase {f}", "get rid of {f}", "clean up {f}",
        "discard {f}", "trash {f}", "wipe out {f}", "delete the file {f}",
    ],
    "rm_it": [
        "delete it", "remove it", "erase it", "get rid of it", "clean it up",
        "discard it", "trash it", "wipe it out",
    ],
    "run": [
        "run {c}", "execute {c}", "invoke {c}", "launch {c}", "kick off {c}",
        "call {c}", "start {c}", "run the command {c}", "execute the command {c}",
        "issue {c}", "fire off {c}", "run {c} in the shell",
    ],
    "ls": [
        "list the files", "list the directory", "show the files",
        "see which files exist", "check what files are there",
        "print the directory listing", "look at the directory contents",
        "find out what files exist", "display the workspace contents",
        "enumerate the files", "list everything in the folder",
        "show what is in the directory",
    ],
    "ensure": [
        "read {f}, but if it is missing create it with {w} and read it again",
        "open {f}; if it does not exist, write {w} to it and re-read it",
        "show {f}, and if it isn't there make it with {w} and show it again",
        "print {f}; should it be missing, save {w} to it and print it again",
        "display {f} -- if there is no such file, create it containing {w} "
        "and display it after",
        "look at {f}, and if it cannot be found, store {w} in it and look again",
    ],
    "ensure_noreread": [
        "read {f}; if it is missing create it with {w}",
        "open {f}, and if it does not exist write {w} to it",
        "show {f} (if it isn't there, make it with {w})",
        "print {f}; should it be missing, save {w} to it",
        "display {f}, creating it with {w} if there is no such file",
        "look at {f}, and if it cannot be found store {w} in it",
    ],
    "reread": [
        "read {f} again", "open {f} once more", "check {f} again", "show {f} again",
        "print {f} one more time", "look at {f} again",
    ],
    "reread_it": [
        "read it again", "open it once more", "check it again", "show it again",
        "print it one more time", "look at it again",
    ],
    "copy": [
        "copy {f} to {g}", "copy the contents of {f} into {g}", "duplicate {f} as {g}",
        "make {g} a copy of {f}", "read {f} and write the same text to {g}",
        "clone {f} into {g}", "put a copy of {f} in {g}",
    ],
    "read_ord": [
        "read the {o} one back", "read the {o} file", "open the {o} file you made",
        "show me the {o} of those files", "display the {o} file again",
        "print the {o} one",
    ],
    "desc": [
        "the {p} file", "the file whose name mentions {p}",
        "whichever file is the {p} one", "the {p} one", "the file with {p} in its name",
    ],
    "conn": [
        ", then ", " and then ", ", and then ", "; then ", ", after that ", ". Then ",
        "; next, ", ", next ", ", afterwards ", "; after that, ", " -- then ",
        ", and after that ", " then ",
    ],
    "conn_last": [
        ", then finally ", ", and finally ", ". Finally, ", " and lastly ", "; last, ",
        ", and to finish, ",
    ],
    "open": [
        "", "", "", "please ", "first ", "I need you to ", "can you ", "go ahead and ",
        "task: ", "could you ", "your job: ", "I'd like you to ",
    ],
    "close": ["", "", "", ".", " please.", " -- thanks", ". That's all.", " for me"],
    "reorder": ["{b}, but first {a}", "{b} -- before that, {a}", "before you {b}, {a}"],
    "numbered": ["{i}) ", "{i}. ", "step {i}: "],
    "wquote": ["{w}", "{w}", '"{w}"', "'{w}'"],
    "cquote": ["{c}", "{c}", "`{c}`"],
}
ORDINALS = ["first", "second", "third"]

# Pools whose items are near-copies of another pool's items (pronoun form, or
# the same ensure-template minus its re-read). Each such item takes the side of
# its counterpart, so e.g. "read it back" (train) and "read {f} back" (eval)
# cannot straddle the split and leak a phrasing chunk.
PAIRED = {"read_it": "read", "verify_it": "verify", "rm_it": "rm",
          "reread_it": "reread", "ensure_noreread": "ensure"}


def _pair_key(pool_name):
    parent = PAIRED.get(pool_name)
    if parent is None:
        return None
    par = POOLS[parent]
    if pool_name == "ensure_noreread":               # index-aligned templates
        return lambda x: par[POOLS[pool_name].index(x)]

    def k(x):
        y = x.replace(" it ", " {f} ")
        y = y[:-3] + " {f}" if y.endswith(" it") else y
        return y if y in par else x
    return k


_cache = {}


def pools(side=None):
    """The phrasing pools for one side of the holdout split (None = all)."""
    if side not in _cache:
        _cache[side] = {k: split_phrasings(v, side, key=_pair_key(k))
                        for k, v in POOLS.items()}
    return _cache[side]


def clause_file(cl):
    """The file a clause is ABOUT for pronoun purposes (None if none)."""
    op = cl["op"]
    if op in ("write", "read", "verify", "rm", "ensure", "reread"):
        return cl["f"]
    if op == "copy":
        return cl["g"]
    return None


def _clause(cl, P, rng, qw, qc, it_ok):
    op = cl["op"]
    f = cl.get("f")
    w = qw.format(w=cl["w"]) if "w" in cl else None
    if op == "write":
        return rng.choice(P["write"]).format(f=f, w=w)
    if op in ("read", "verify", "rm", "reread"):
        if it_ok and rng.random() < 0.65:
            return rng.choice(P[op + "_it"])
        return rng.choice(P[op]).format(f=f)
    if op == "run":
        return rng.choice(P["run"]).format(c=qc.format(c=cl["c"]))
    if op == "ls":
        return rng.choice(P["ls"])
    if op == "ensure":
        key = "ensure" if cl.get("reread", True) else "ensure_noreread"
        return rng.choice(P[key]).format(f=f, w=w)
    if op == "copy":
        return rng.choice(P["copy"]).format(f=f, g=cl["g"])
    if op == "read_ord":
        return rng.choice(P["read_ord"]).format(o=ORDINALS[cl["k"]])
    if op == "desc":
        d = rng.choice(P["desc"]).format(p=cl["p"])
        return rng.choice(P["read" if cl["verb"] == "read" else "rm"]).format(f=d)
    raise ValueError("unknown clause op %r" % op)


def _render_seq(plan, P, rng, qw, qc, pronouns=True):
    out, last = [], None
    for cl in plan:
        f = clause_file(cl)
        it_ok = pronouns and f is not None and f == last and cl["op"] != "write"
        out.append(_clause(cl, P, rng, qw, qc, it_ok))
        if cl["op"] in ("read_ord", "desc"):
            last = None
        elif f is not None:
            last = f
    return out


def _join(parts, P, rng):
    if len(parts) == 1:
        return parts[0]
    s = parts[0]
    for i, p in enumerate(parts[1:], 1):
        final = i == len(parts) - 1 and len(parts) >= 3 and rng.random() < 0.5
        s += rng.choice(P["conn_last"] if final else P["conn"]) + p
    return s


def render_goal(plan, rng, side=None):
    """Surface text for a plan. Deterministic given rng state."""
    P = pools(side)
    qw = rng.choice(P["wquote"])
    qc = rng.choice(P["cquote"])
    ordered = any(cl["op"] in ("read_ord",) for cl in plan)
    r = rng.random()
    if len(plan) >= 2 and not ordered and r < 0.18:
        # surface order != execution order: the last clause is said FIRST
        a = _join(_render_seq(plan[:-1], P, rng, qw, qc), P, rng)
        b = _render_seq(plan[-1:], P, rng, qw, qc, pronouns=False)[0]
        text = rng.choice(P["reorder"]).format(a=a, b=b)
    elif len(plan) >= 2 and r < 0.30:
        parts = _render_seq(plan, P, rng, qw, qc)
        num = rng.choice(P["numbered"])
        sep = rng.choice(["; ", " "])
        text = sep.join(num.format(i=i + 1) + p for i, p in enumerate(parts))
    else:
        text = rng.choice(P["open"]) + _join(_render_seq(plan, P, rng, qw, qc), P, rng)
    text += rng.choice(P["close"])
    if rng.random() < 0.3:
        text = text[:1].upper() + text[1:]
    return text
