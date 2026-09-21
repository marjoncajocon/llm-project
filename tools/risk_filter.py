#!/usr/bin/env python3
"""Keep BLOCK-level shell commands out of routine training data.

The agent refuses (or demands a typed "yes" for) any run_command that the C
classifier in risk.c rates BLOCK: recursive/forced deletes, git reset --hard /
clean -f / push --force, format/mkfs/dd, download-piped-into-a-shell, and any
path outside the workspace. Training the model to EMIT those commands as the
normal next step would teach exactly the behaviour the gate exists to stop, so
the data generators drop them from their command pools.

Why call the C classifier instead of porting it to Python: the runtime gate and
the data filter must agree exactly. A second implementation would drift (the C
side has ~300 command rules, three shell tokenizers and recursion into sh -c /
cmd /c / powershell -Command), and only the C side is covered by selftest [42].
So this module shells out ONCE per pool to

    transformer risk --stdin      (one command per line in, "LEVEL<TAB>reason" out)

and never executes any of the commands -- it only classifies strings.

The binary is found via $TRANSFORMER_EXE, else transformer.exe / transformer in
the repo root. If it is missing the filter FAILS CLOSED (raises) instead of
silently passing unfiltered commands through: build it first (`make`, or
`zig cc -std=c11 -O2 -o transformer.exe *.c -lm`).

    python tools/risk_filter.py data/so_commands.txt      # report only
"""
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_exe():
    env = os.environ.get("TRANSFORMER_EXE")
    if env:
        return env
    for name in ("transformer.exe", "transformer"):
        p = os.path.join(_ROOT, name)
        if os.path.isfile(p):
            return p
    return None


def classify(cmds, exe=None):
    """[(level, reason)] for each command, in order. Raises if the classifier
    is unavailable (fail closed)."""
    cmds = list(cmds)
    if not cmds:
        return []
    exe = exe or find_exe()
    if not exe:
        raise RuntimeError(
            "risk_filter: transformer binary not found (set TRANSFORMER_EXE or build it "
            "in the repo root); refusing to emit unfiltered shell commands")
    # One command per line: a command can never contain a newline here.
    clean = [c.replace("\r", " ").replace("\n", " ") for c in cmds]
    data = ("\n".join(clean) + "\n").encode("utf-8")
    r = subprocess.run([exe, "risk", "--stdin"], input=data, capture_output=True, check=True)
    lines = r.stdout.decode("utf-8", errors="replace").splitlines()
    if len(lines) != len(cmds):
        raise RuntimeError("risk_filter: expected %d verdicts, got %d" % (len(cmds), len(lines)))
    out = []
    for ln in lines:
        lvl, _, why = ln.partition("\t")
        if lvl not in ("SAFE", "REVIEW", "BLOCK"):
            raise RuntimeError("risk_filter: bad classifier output %r" % ln)
        out.append((lvl, why))
    return out


def drop_block(cmds, exe=None, label="commands", quiet=False):
    """cmds minus every BLOCK-level command (order kept)."""
    cmds = list(cmds)
    verdicts = classify(cmds, exe)
    kept = [c for c, (lvl, _) in zip(cmds, verdicts) if lvl != "BLOCK"]
    if not quiet:
        print("risk filter: dropped %d BLOCK-level of %d %s"
              % (len(cmds) - len(kept), len(cmds), label), file=sys.stderr)
    return kept


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    cmds = [ln.strip() for ln in open(argv[1], encoding="utf-8") if ln.strip()]
    verdicts = classify(cmds)
    counts = {}
    for c, (lvl, why) in zip(cmds, verdicts):
        counts[lvl] = counts.get(lvl, 0) + 1
        if lvl == "BLOCK":
            print("BLOCK  %-50s %s" % (c, why))
    print("%d commands: %s" % (len(cmds), ", ".join("%s %d" % kv for kv in sorted(counts.items()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
