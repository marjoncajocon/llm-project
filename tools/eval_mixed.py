"""Evaluate a mixed-SFT model on BOTH skills it should now have:

  1. tool routing   -- held-out instructions still produce the right tool call
  2. general answers -- general QUESTIONS produce PROSE, not a forced tool call
                        (this is what the tool-only model catastrophically lost)

Usage: eval_mixed.py <engine> <model.bin> <tok.hf> <tool_test.tsv> <qa_test.tsv>
"""
import re
import subprocess
import sys


def gen(engine, model, tok, prompt, nnew=48):
    p = subprocess.run([engine, "hfgen", model, tok, str(nnew), "0.1"],
                       input=prompt.encode("utf-8"),
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out = []
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line.startswith('{"d":"'):
            out.append(line[len('{"d":"'):-2].replace('\\"', '"').replace("\\n", " "))
    return "".join(out).strip()


def tool_name(s):
    m = re.search(r'name="(\w+)"', s)
    if m:
        return m.group(1)
    return "done" if "<done" in s else "PROSE"


def main():
    engine, model, tok, toolf, qaf = sys.argv[1:6]

    # 1) tool routing on held-out instructions
    print("=== 1. tool routing (held-out instructions) ===")
    ok = n = 0
    for line in open(toolf, encoding="utf-8-sig"):
        if "\t" not in line:
            continue
        prompt, expected = line.rstrip("\n").split("\t", 1)
        got = gen(engine, model, tok, prompt)
        n += 1
        ok += tool_name(got) == tool_name(expected)
    print("  right tool: %d/%d (%.0f%%)\n" % (ok, n, 100.0 * ok / max(n, 1)))

    # 2) general questions must NOT collapse into a tool call
    print("=== 2. general questions answered as PROSE (not a tool call) ===")
    prose = n = 0
    shown = 0
    for line in open(qaf, encoding="utf-8-sig"):
        if "\t" not in line:
            continue
        prompt, _ = line.rstrip("\n").split("\t", 1)
        got = gen(engine, model, tok, prompt, nnew=40)
        n += 1
        is_prose = ("<tool" not in got and "<done" not in got and len(got) > 0)
        prose += is_prose
        if shown < 6:
            print("  Q: %-34s A: %s" % (prompt[:34], got[:60]))
            shown += 1
    print("\n  answered as prose (not a tool call): %d/%d (%.0f%%)" %
          (prose, n, 100.0 * prose / max(n, 1)))

    # 3) the decisive off-topic probe: brand-new general questions
    print("\n=== 3. off-topic probe (unseen general questions) ===")
    probes = ["why is the sky blue", "what is the capital of France",
              "who wrote Hamlet", "what is gravity", "tell me about the ocean",
              "what is a computer"]
    for q in probes:
        got = gen(engine, model, tok, q, nnew=36)
        kind = "PROSE" if ("<tool" not in got and "<done" not in got) else "TOOL-CALL"
        print("  [%-9s] %-30s -> %s" % (kind, q, got[:52]))


main()
