#!/usr/bin/env python3
"""Generate a larger, more varied agent SFT dataset (prompt<TAB>tool-call lines).

The hand-written data/sft.txt has 12 examples, so a fine-tune memorizes them
rather than learning the instruction -> tool-call MAPPING. This templates many
phrasings of each intent over a pool of filenames and shell commands, so the same
intent appears with different words and arguments and the model has to generalize.

Filenames are drawn from crawled Wikipedia titles when a --wiki dir is given
(realistic, varied nouns), else from a built-in list. Output matches the loader:
one "prompt<TAB>response" line per example, response being a <tool .../> or <done>.

    python tools/make_sft.py --out data/sft_big.txt --n 800 --wiki data/wiki

Deterministic given --seed, so a run is reproducible.
"""
import argparse
import os
import random
import re


def slug(title):
    s = re.sub(r"[^A-Za-z0-9]+", "_", title.strip().lower()).strip("_")
    return s[:24] or "notes"


def filenames_from_wiki(wiki_dir, limit=400):
    names = []
    if wiki_dir and os.path.isdir(wiki_dir):
        for fn in sorted(os.listdir(wiki_dir)):
            if fn.endswith(".txt"):
                base = slug(fn[:-4])
                names.append(base)
            if len(names) >= limit:
                break
    return names


BUILTIN_FILES = [
    "notes", "readme", "config", "main", "output", "log", "data", "todo",
    "report", "index", "results", "summary", "draft", "scratch", "temp",
]
EXTS = [".txt", ".md", ".log", ".json", ".csv"]
WORDS = ["hello", "done", "start", "ok", "todo", "hi there", "first line",
         "test", "the result is 42", "work in progress", "finished"]
COMMANDS = [
    ("list the files here", "ls"),
    ("list files in this folder", "ls"),
    ("show the directory contents", "ls -la"),
    ("show the git status", "git status"),
    ("check the git status", "git status"),
    ("print the current directory", "pwd"),
    ("show where i am", "pwd"),
    ("run the tests", "make test"),
    ("run the test suite", "make test"),
    ("build the project", "make"),
    ("show running processes", "ps aux"),
    ("what version of python is installed", "python --version"),
    ("show disk usage", "df -h"),
    ("print the date", "date"),
]
DONES = [
    ("the task is finished", "Task complete."),
    ("that's everything, stop", "Stopping as requested."),
    ("stop now", "Stopping as requested."),
    ("we are done here", "Task complete."),
    ("nothing more to do", "Task complete."),
    ("you can finish", "Task complete."),
]

READ_TMPL = ["read {f}", "read the file {f}", "open {f}", "show me {f}",
             "display the contents of {f}", "cat {f}", "what's in {f}", "print {f}"]
WRITE_TMPL = ["write {w} to {f}", "save the word {w} to {f}",
              "create a file {f} with the text {w}", "put {w} in {f}",
              "make {f} containing {w}", "create {f} with {w}"]
DELETE_TMPL = ["delete {f}", "remove {f}", "get rid of {f}", "rm {f}"]


def esc(s):
    return s  # responses contain no tabs/newlines; loader splits on the first tab


def gen(n, files, rng):
    out = []
    while len(out) < n:
        kind = rng.random()
        if kind < 0.30:                                   # read_file
            f = rng.choice(files) + rng.choice(EXTS)
            p = rng.choice(READ_TMPL).format(f=f)
            r = '<tool name="read_file" path="%s"></tool>' % f
        elif kind < 0.58:                                 # write_file
            f = rng.choice(files) + rng.choice(EXTS)
            w = rng.choice(WORDS)
            p = rng.choice(WRITE_TMPL).format(f=f, w=w)
            r = '<tool name="write_file" path="%s">%s</tool>' % (f, w)
        elif kind < 0.72:                                 # run_command: delete
            f = rng.choice(files) + rng.choice(EXTS)
            p = rng.choice(DELETE_TMPL).format(f=f)
            r = '<tool name="run_command" command="rm %s"></tool>' % f
        elif kind < 0.90:                                 # run_command: fixed
            p, cmd = rng.choice(COMMANDS)
            r = '<tool name="run_command" command="%s"></tool>' % cmd
        else:                                             # done
            p, msg = rng.choice(DONES)
            r = '<done>%s</done>' % msg
        out.append((p, r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_big.txt")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--wiki", default=None, help="dir of crawled *.txt for filenames")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = filenames_from_wiki(args.wiki)
    files = list(dict.fromkeys(files + BUILTIN_FILES))    # dedupe, keep order
    rng.shuffle(files)
    rows = gen(args.n, files, rng)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for p, r in rows:
            f.write("%s\t%s\n" % (p, r))
    print("wrote %d examples -> %s (%d distinct filenames)" % (len(rows), args.out, len(files)))


main()
