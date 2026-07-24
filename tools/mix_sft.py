"""Interleave several prompt<TAB>response SFT files into one shuffled dataset.

Mixing tool-call data (make_sft.py) with general Q&A (make_qa.py) teaches ONE
model both skills — route an instruction to a tool call, answer a question in
prose — instead of collapsing every prompt into a tool call. Deterministic given
--seed.

    python tools/mix_sft.py --out data/mixed.txt --seed 3 \
        data/sft_big.txt data/qa.txt
"""
import argparse
import random


def read(path):
    rows = []
    for line in open(path, encoding="utf-8-sig"):
        line = line.rstrip("\n")
        if "\t" in line:
            rows.append(line)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="prompt<TAB>response files to merge")
    ap.add_argument("--out", default="data/mixed.txt")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cap", type=int, default=0,
                    help="max lines to keep PER input (0 = all); balances the mix")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    all_rows, counts = [], []
    for path in args.inputs:
        rows = read(path)
        rng.shuffle(rows)
        if args.cap > 0:
            rows = rows[:args.cap]
        counts.append((path, len(rows)))
        all_rows.extend(rows)
    rng.shuffle(all_rows)

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(all_rows) + "\n")
    print("wrote %d lines -> %s" % (len(all_rows), args.out))
    for path, n in counts:
        print("  %5d from %s" % (n, path))


main()
