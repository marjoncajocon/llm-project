"""Memorization vs generalization probe for a tool-call SFT model.

The decisive test: arguments made of RANDOM gibberish that appear NOWHERE in the
training data or any crawled corpus. A memorizing model cannot copy a string it
never saw -- it would emit a training string instead. A generalizing model copies
the novel argument verbatim. We also feed exact TRAINING prompts as a baseline
(high there is expected and proves nothing on its own).

Usage: memtest.py <engine.exe> <model.bin> <tok.hf> <train.tsv>
"""
import random
import re
import subprocess
import sys

rng = random.Random(20260724)


def gib(n=6):
    return "".join(rng.choice("bcdfghjklmnpqrstvwxz2479") for _ in range(n))


def gen(engine, model, tok, prompt, nnew=48):
    p = subprocess.run([engine, "hfgen", model, tok, str(nnew), "0.1"],
                       input=prompt.encode("utf-8"),
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out = []
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line.startswith('{"d":"'):
            out.append(line[len('{"d":"'):-2].replace('\\"', '"'))
    return "".join(out)


def first_tag(s):
    m = re.search(r'<tool\b[^>]*>.*?</tool>|<done>.*?</done>', s)
    return m.group(0) if m else s.strip()


def contains(got, needle):
    return needle in got


def main():
    engine, model, tok, trainf = sys.argv[1:5]

    # 1) Random-gibberish filenames -- guaranteed absent from all training/corpora.
    print("=== A. random-gibberish filenames (cannot be memorized) ===")
    ok = 0
    N = 20
    for _ in range(N):
        fn = gib() + rng.choice([".txt", ".md", ".log", ".json"])
        got = first_tag(gen(engine, model, tok, "read the file " + fn))
        hit = contains(got, fn) and 'read_file' in got
        ok += hit
        if not hit:
            print("  MISS  read %-16s -> %s" % (fn, got[:52]))
    print("  copied-correctly: %d/%d (%.0f%%)\n" % (ok, N, 100.0 * ok / N))

    # 2) Random-gibberish shell commands -- also impossible to have memorized.
    print("=== B. random-gibberish commands ===")
    ok = 0
    for _ in range(N):
        cmd = gib(4) + " " + gib(5) + " --" + gib(3)
        got = first_tag(gen(engine, model, tok, "run " + cmd))
        hit = contains(got, cmd) and 'run_command' in got
        ok += hit
        if not hit:
            print("  MISS  run %-24s -> %s" % (cmd, got[:52]))
    print("  copied-correctly: %d/%d (%.0f%%)\n" % (ok, N, 100.0 * ok / N))

    # 3) Baseline: exact TRAINING prompts (high here is expected, not informative).
    print("=== C. exact training prompts (baseline) ===")
    rows = [l.rstrip("\n").split("\t", 1) for l in open(trainf, encoding="utf-8-sig") if "\t" in l]
    rng.shuffle(rows)
    ok = 0
    M = 25
    sample = rows[:M]
    for prompt, expected in sample:
        got = first_tag(gen(engine, model, tok, prompt))
        ok += re.sub(r"\s+", "", got) == re.sub(r"\s+", "", expected)
    print("  reproduced: %d/%d (%.0f%%)\n" % (ok, M, 100.0 * ok / M))

    print("VERDICT: if A and B are high, the model COPIES novel arguments it")
    print("could not have memorized -> it generalized. If A/B collapse while C")
    print("stays high, it memorized the training pairs.")


main()
