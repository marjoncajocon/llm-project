#!/usr/bin/env python3
"""Build a GRPO task file from an existing SFT corpus.

`transformer grpo` reads the SAME prompt<TAB>response format the SFT trainer
reads -- the response column is used as the REFERENCE ACTION the reward scores a
rollout against. So any corpus already in the tree can be fed to it directly.
This tool is the quality gate for doing that, and it exists because of one
asymmetry:

    A BAD ROW IN SFT DATA TEACHES THE MODEL ONE WRONG ANSWER.
    A BAD ROW IN AN RL TASK FILE TEACHES THE REWARD TO LIE.

If a reference action is not grounded in its own prompt (the classic
hallucinated-filename / empty-write-body bugs clean_sft.py was written to catch),
then GRPO will hand out its `match` credit for reproducing a hallucination, and
reward and truth come apart. Every reference here is therefore put through
clean_sft.verify() -- the same grounding rule the C reward re-implements -- and
anything it rejects is dropped, not fixed.

    python tools/grpo_tasks.py --out data/grpo_tasks.txt data/clean_reason.txt
    .\\transformer.exe grpo data/grpo_tasks.txt --model model/smollm/reason_e2.bin \\
        --hf-tok model/smollm/tok.hf --out model/smollm/grpo.bin --steps 200

Rollouts cost G x max-new decode passes EACH, so prompt length is the main lever
on wall-clock: --max-chars drops the long transcripts that dominate a run's cost.
Deterministic given --seed.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_sft import classify, read_rows, verify  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("inputs", nargs="+", help="prompt<TAB>response files to draw tasks from")
  ap.add_argument("--out", default="data/grpo_tasks.txt")
  ap.add_argument("--max-chars", type=int, default=1200,
                  help="drop prompts longer than this (rollout cost scales with "
                       "prompt length; 0 = keep every length)")
  ap.add_argument("--limit", type=int, default=0, help="keep at most N tasks (0 = all)")
  ap.add_argument("--actions-only", action="store_true",
                  help="keep only rows whose reference is a tool call or <done>, "
                       "dropping prose Q&A (which the reward can only score on the "
                       "routing decision)")
  ap.add_argument("--seed", type=int, default=1234)
  args = ap.parse_args()

  rng = random.Random(args.seed)
  rows = []
  for path in args.inputs:
    rows.extend(read_rows(path))
  print("read %d rows from %d file(s)" % (len(rows), len(args.inputs)))

  kept, drops = [], {}

  def drop(why):
    drops[why] = drops.get(why, 0) + 1

  seen = set()
  for prompt, resp, _src in rows:
    if not prompt.strip() or not resp.strip():
      drop("empty")
      continue
    if args.max_chars and len(prompt) > args.max_chars:
      drop("prompt-too-long")
      continue
    # THE point of this tool: an ungrounded reference would make the reward
    # pay out for a hallucination.
    why = verify(prompt, resp)
    if why:
      drop(why.split(":")[0])
      continue
    action, _steps = classify(prompt, resp)
    if args.actions_only and action == "prose":
      drop("prose")
      continue
    if prompt in seen:
      drop("duplicate-prompt")
      continue
    seen.add(prompt)
    kept.append((prompt, resp, action))

  if args.limit and len(kept) > args.limit:
    rng.shuffle(kept)
    kept = kept[:args.limit]

  by_action = {}
  for _p, _r, a in kept:
    by_action[a] = by_action.get(a, 0) + 1

  outdir = os.path.dirname(args.out)
  if outdir:
    os.makedirs(outdir, exist_ok=True)
  with open(args.out, "w", encoding="utf-8", newline="\n") as f:
    for prompt, resp, _a in kept:
      f.write("%s\t%s\n" % (prompt.replace("\t", " "), resp.replace("\t", " ")))

  print("kept %d tasks -> %s" % (len(kept), args.out))
  if drops:
    print("dropped:")
    for why in sorted(drops, key=lambda k: -drops[k]):
      print("  %-22s %d" % (why, drops[why]))
  print("action mix:")
  for a in sorted(by_action, key=lambda k: -by_action[k]):
    print("  %-22s %d" % (a, by_action[a]))
  if not kept:
    print("ERROR: no usable tasks")
    return 1
  # Reward needs SPREAD within a group to produce any gradient, and spread comes
  # from tasks the model gets right only sometimes. A task set the model already
  # solves perfectly teaches nothing -- so does one it never solves.
  print("\nNote: GRPO learns from variance. If a run reports many 'flat' groups,\n"
        "      raise --temp, or build a task file weighted toward cases the model\n"
        "      currently gets right only some of the time.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
