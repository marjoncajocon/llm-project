#!/usr/bin/env python3
"""Build preference pairs for `transformer dpo`.

DPO learns from `prompt <TAB> chosen <TAB> rejected` triples. The chosen side is
easy -- it is the verified response already in your SFT corpus. The whole
question is where the REJECTED side comes from, and there are two honest answers:

  1. SYNTHETIC NEGATIVES (default). Corrupt the correct response in exactly the
     ways this model is known to fail: a hallucinated filename, the ~5% character
     slip (demo_note -> demonstra_note), an empty write body, a premature <done>,
     rambling past the closing tag, the wrong tool. Cheap, deterministic, needs no
     sampling, and targets defects you have actually measured.

  2. REAL NEGATIVES (--traces). Mine an AGENT_TRACE JSONL for responses the model
     really produced that FAIL clean_sft.verify(), and pair each with the correct
     response for the same prompt. Strictly better data -- these are on-policy
     mistakes rather than mistakes you imagined -- but you only get as many as the
     model has actually made.

Use both. Real negatives where you have them, synthetic to cover the rest.

    python tools/dpo_pairs.py --out data/dpo_pairs.txt data/clean_reason.txt
    python tools/dpo_pairs.py --out data/dpo_pairs.txt \\
        --traces data/self_trace.jsonl data/clean_reason.txt

HONEST LIMIT of synthetic negatives: they teach "do not do THESE corruptions".
They cannot teach the model to avoid a failure mode nobody thought to encode. If
a run shows the preference learned (margin rising) but tools/remeasure.py does not
improve, that gap is usually the reason -- switch to --traces.

Deterministic given --seed. Output drops straight into `transformer dpo`.
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_sft import NAME_RE, PATH_RE, read_rows, strip_reasoning, verify  # noqa: E402

TOOL_CLOSE = "</tool>"
DECOYS = ["output.txt", "result.log", "temp_data.json", "summary.md", "backup.csv"]
RAMBLE = [
    " Let me know if you want anything else.",
    " <jupyter_text> Now I will continue.",
    " I have completed the task successfully.",
]


# ---------------------------------------------------------------------------
# Corruptions. Each takes the verified response and returns a WORSE one, or None
# when it does not apply to this row. They are deliberately PLAUSIBLE: a nonsense
# negative teaches nothing, because the model would never have produced it.
# ---------------------------------------------------------------------------

def _path_of(resp):
    m = PATH_RE.search(resp)
    return m.group(1) if m else None


def _name_of(resp):
    m = NAME_RE.search(resp)
    return m.group(1) if m else None


def hallucinated_path(resp, prompt, rng):
    """A filename that appears nowhere in the prompt -- the classic invention."""
    p = _path_of(resp)
    if not p:
        return None
    decoy = rng.choice([d for d in DECOYS if d.lower() not in prompt.lower()] or DECOYS)
    return resp.replace('path="%s"' % p, 'path="%s"' % decoy)


def char_slip(resp, prompt, rng):
    """The measured ~5% failure: the right file, spelled slightly wrong."""
    p = _path_of(resp)
    if not p or len(p) < 6:
        return None
    stem, dot, ext = p.partition(".")
    if len(stem) < 4:
        return None
    i = rng.randrange(1, len(stem) - 1)
    # duplicate a character -- how a byte-level model actually slips
    slipped = stem[:i] + stem[i] + stem[i:] + dot + ext
    return resp.replace('path="%s"' % p, 'path="%s"' % slipped)


def empty_write(resp, prompt, rng):
    """write_file with nothing in it -- the file is created but useless."""
    if _name_of(resp) != "write_file":
        return None
    gt = resp.find(">", resp.find("<tool"))
    close = resp.find(TOOL_CLOSE)
    if gt < 0 or close < 0 or close <= gt + 1:
        return None
    return resp[:gt + 1] + resp[close:]


def premature_done(resp, prompt, rng):
    """Stopping before doing the work -- a documented failure of this model.

    The <think> prefix is CARRIED OVER deliberately: see the confound note in
    main(). If the rejected side lost the rationale that the chosen side keeps,
    the pair would be teaching "prefer responses that contain <think>" rather
    than "do not stop early"."""
    if "<tool" not in resp:
        return None
    m = re.match(r"\s*(<think>.*?</think>)", resp, re.DOTALL)
    prefix = m.group(1) if m else ""
    return prefix + "<done>ok</done>"


def trailing_ramble(resp, prompt, rng):
    """A correct action followed by text the agent loop would discard anyway."""
    if TOOL_CLOSE not in resp and "</done>" not in resp:
        return None
    return resp + rng.choice(RAMBLE)


def unterminated(resp, prompt, rng):
    """The block never closes, so nothing is executed at all."""
    if TOOL_CLOSE not in resp:
        return None
    return resp[:resp.rfind(TOOL_CLOSE)]


def wrong_tool(resp, prompt, rng):
    """Right file, right format, wrong verb: reads when it should write."""
    name = _name_of(resp)
    swap = {"write_file": "read_file", "read_file": "write_file"}
    if name not in swap:
        return None
    return resp.replace('name="%s"' % name, 'name="%s"' % swap[name])


CORRUPTIONS = [
    ("hallucinated-path", hallucinated_path),
    ("char-slip", char_slip),
    ("empty-write", empty_write),
    ("premature-done", premature_done),
    ("trailing-ramble", trailing_ramble),
    ("unterminated", unterminated),
    ("wrong-tool", wrong_tool),
]


def load_traces(path):
    """AGENT_TRACE JSONL -> [(prompt, response)]. utf-8-sig: a BOM would
    otherwise silently drop the first record (the bug fixed in self_train.py)."""
    out = []
    try:
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if isinstance(d, dict) and "prompt" in d and "response" in d:
                    out.append((d["prompt"], d["response"]))
    except OSError as e:
        print("warning: could not read traces: %s" % e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="clean prompt<TAB>response corpora")
    ap.add_argument("--out", default="data/dpo_pairs.txt")
    ap.add_argument("--traces", default=None,
                    help="AGENT_TRACE JSONL to mine REAL model failures from")
    ap.add_argument("--per-row", type=int, default=1,
                    help="synthetic negatives per clean row (default 1)")
    ap.add_argument("--max-chars", type=int, default=1200,
                    help="drop prompts longer than this (0 = keep all)")
    ap.add_argument("--limit", type=int, default=0, help="cap the output (0 = all)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    rows = []
    for path in args.inputs:
        rows.extend(read_rows(path))
    print("read %d rows from %d file(s)" % (len(rows), len(args.inputs)))

    # The chosen side must itself be verified. An ungrounded "chosen" would teach
    # the model to PREFER a hallucination, which is worse than not training.
    clean = {}
    dropped_unverified = 0
    for prompt, resp, _src in rows:
        if args.max_chars and len(prompt) > args.max_chars:
            continue
        if verify(prompt, resp):
            dropped_unverified += 1
            continue
        clean.setdefault(prompt, resp)
    print("  %d unique verified prompts (%d dropped as unverified)"
          % (len(clean), dropped_unverified))

    pairs, counts = [], {}

    # --- real negatives first: they are the better data ---
    if args.traces:
        traces = load_traces(args.traces)
        real = 0
        for prompt, resp in traces:
            good = clean.get(prompt)
            if not good:
                continue
            why = verify(prompt, resp)
            if not why:
                continue                      # the model got it right; not a negative
            if strip_reasoning(resp) == strip_reasoning(good):
                continue
            pairs.append((prompt, good, resp))
            counts["real:" + why.split(":")[0]] = counts.get("real:" + why.split(":")[0], 0) + 1
            real += 1
        print("  mined %d REAL negatives from %s (%d trace rows)"
              % (real, args.traces, len(traces)))

    # --- synthetic negatives to cover the rest ---
    # Corruptions are applied to the FULL chosen response, reasoning span and all
    # -- never to a stripped copy. DPO learns whatever separates the two sides, so
    # if the chosen kept its <think> rationale and the rejected did not, every
    # pair would quietly teach "prefer responses containing <think>" instead of
    # the defect being targeted. The only difference between the two sides must
    # be the defect itself.
    for prompt, good in clean.items():
        applicable = []
        for label, fn in CORRUPTIONS:
            try:
                bad = fn(good, prompt, rng)
            except Exception:
                bad = None
            if bad and bad.strip() and bad != good:
                applicable.append((label, bad))
        if not applicable:
            continue
        rng.shuffle(applicable)
        for label, bad in applicable[:max(1, args.per_row)]:
            pairs.append((prompt, good, bad))
            counts[label] = counts.get(label, 0) + 1

    if not pairs:
        print("ERROR: produced no pairs")
        return 1

    rng.shuffle(pairs)
    if args.limit and len(pairs) > args.limit:
        pairs = pairs[:args.limit]

    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for prompt, good, bad in pairs:
            f.write("%s\t%s\t%s\n" % (_flat(prompt), _flat(good), _flat(bad)))

    print("wrote %d pairs -> %s" % (len(pairs), args.out))
    print("negative types:")
    for k in sorted(counts, key=lambda k: -counts[k]):
        print("  %-28s %d" % (k, counts[k]))
    if not args.traces:
        print("\nNote: these are SYNTHETIC negatives. They teach the model to avoid the\n"
              "      specific corruptions above. For on-policy mistakes the model really\n"
              "      makes, collect a trace and re-run with --traces:\n"
              "        $env:AGENT_TRACE = \"data/self_trace.jsonl\"")
    return 0


def _flat(s):
    return re.sub(r"[\t\r\n]+", " ", s)


if __name__ == "__main__":
    raise SystemExit(main())
