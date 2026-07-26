"""STaR self-training: harvest the agent's OWN successful runs into SFT data.

The loop this closes:
    you prompt -> the agent acts -> the TOOL OBSERVATIONS say objectively whether
    it worked -> verified-good runs become training rows -> periodic LoRA retrain
    -> promote ONLY if tools/remeasure.py stays green.

Collect a trace first (off by default, so normal runs are unaffected):
    $env:AGENT_TRACE = "data/self_trace.jsonl"
    .\transformer.exe agent --cmd "python tools/agent_local.py MODEL TOK" --goal "..."
Then harvest:
    python tools/self_train.py --trace data/self_trace.jsonl --out data/self_traj.txt

WHY THE MODEL DOES NOT GRADE ITSELF. Every accept/reject below keys off something
external: the tool's own observation text, or clean_sft.verify()'s grounding check
(args must appear in the goal). The model's opinion of its work is never consulted.
That matters because self-training on unfiltered output amplifies the model's own
errors -- notably the ~5% char-slip, which grounding rejects outright.

HONEST LIMIT (measured, lever #4): at 135M the residual errors are SYSTEMATIC and
low-variance -- for the failing case the proposer never emitted the right action in
6 samples. Self-training AMPLIFIES what the model sometimes already gets right; it
does not invent new capability. Expect sharpening, not a new skill.

Runs are rebuilt by PREFIX-CHAINING: step k's prompt contains step k-1's verbatim,
so a run is a maximal chain and needs no run id. Interleaved or abandoned runs sort
themselves out.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_sft import verify, strip_reasoning  # noqa: E402

SEP = " | "
# An observation that means the tool FAILED. Matches agent_critic.obs_is_error's
# vocabulary so the two agree on what "failure" means.
ERR_RE = re.compile(r"error|failed|cannot|not found|no such|denied|traceback|exception",
                    re.I)
TOOL_RE = re.compile(r'<tool\b[^>]*>.*?</tool>', re.DOTALL)


def load(path):
  rows = []
  # utf-8-SIG: a BOM on line 1 (any Windows tool that wrote the file) would
  # otherwise make json.loads fail and DROP that record silently.
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
        rows.append((d["prompt"], d["response"]))
  # de-duplicate identical steps (a re-run of the same goal replays them)
  seen, out = set(), []
  for p, r in rows:
    if (p, r) not in seen:
      seen.add((p, r))
      out.append((p, r))
  return out


def terminal_rows(rows):
  """A run's last step = a row whose prompt no OTHER row's prompt extends."""
  prompts = [p for p, _ in rows]
  term = []
  for p, r in rows:
    if not any(q != p and q.startswith(p) for q in prompts):
      term.append((p, r))
  return term


def observations(prompt):
  return [seg[len("OBSERVATION:"):].strip()
          for seg in prompt.split(SEP) if seg.strip().startswith("OBSERVATION:")]


def actions_of(prompt):
  return [seg[len("MODEL:"):].strip()
          for seg in prompt.split(SEP) if seg.strip().startswith("MODEL:")]


def judge(prompt, response, max_repeat):
  """Return None to ACCEPT the whole run, else a short reject reason.

  Every criterion is external to the model's own judgement.
  """
  resp = strip_reasoning(response)
  # 1. The run must have ENDED deliberately, not run out of steps.
  if not resp.startswith("<done"):
    return "no-terminal-done"
  obs = observations(prompt)
  if not obs:
    return "no-observations"
  # 2. The FINAL observation must not be a failure. Earlier errors are fine and
  #    wanted -- an error the model then recovers from is the good training data.
  if ERR_RE.search(obs[-1]):
    return "final-observation-failed:%s" % obs[-1][:32]
  # 3. Loop guard (lever #4 saw a live 8x re-read loop): no action repeated too often.
  acts = actions_of(prompt)
  for a in set(acts):
    if acts.count(a) > max_repeat:
      return "looped-action:%s" % a[:40]
  # 4. A run of pure <done> with no tool call teaches nothing.
  if not any(TOOL_RE.search(a) for a in acts):
    return "no-tool-calls"
  return None


def pairs_of(prompt, response):
  """Rebuild every (prefix -> next action) SFT row of one run, incl. the final <done>."""
  segs = [s.strip() for s in prompt.split(SEP) if s.strip()]
  out, cur = [], []
  for seg in segs:
    if seg.startswith("MODEL:") and cur:
      out.append((SEP.join(cur), seg[len("MODEL:"):].strip()))
    cur.append(seg)
  out.append((SEP.join(cur), response.strip()))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--trace", default="data/self_trace.jsonl")
  ap.add_argument("--out", default="data/self_traj.txt")
  ap.add_argument("--corpus", default="data/clean_reason.txt",
                  help="existing corpus; rows already present are dropped")
  ap.add_argument("--max-repeat", type=int, default=2,
                  help="max times one action may repeat in a run before it is a loop")
  ap.add_argument("--report", default=None)
  ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
  a = ap.parse_args()

  if not os.path.exists(a.trace):
    print("error: no trace at %s (set AGENT_TRACE and run the agent first)" % a.trace)
    return 1
  rows = load(a.trace)
  runs = terminal_rows(rows)
  print("trace: %d unique steps -> %d runs" % (len(rows), len(runs)))

  known = set()
  if os.path.exists(a.corpus):
    for line in open(a.corpus, encoding="utf-8-sig"):
      if "\t" in line:
        known.add(line.rstrip("\n"))

  accepted, log = [], []
  n_ok = 0
  for prompt, response in runs:
    why = judge(prompt, response, a.max_repeat)
    goal = prompt.split(SEP)[0][:60]
    if why:
      log.append("REJECT run  %-62s %s" % (goal, why))
      continue
    # Run looks good; now every ROW must pass grounding independently.
    rowpairs = pairs_of(prompt, response)
    bad = None
    for p, r in rowpairs:
      v = verify(p, r)
      if v:
        bad = v
        break
    if bad:
      log.append("REJECT rows %-62s %s" % (goal, bad))
      continue
    n_ok += 1
    for p, r in rowpairs:
      line = "%s\t%s" % (p.replace("\t", " "), r.replace("\t", " "))
      if line in known:
        continue
      known.add(line)
      accepted.append(line)
    log.append("ACCEPT run  %-62s %d rows" % (goal, len(rowpairs)))

  for ln in log:
    print("  " + ln)
  print("\nruns accepted %d/%d | NEW rows %d" % (n_ok, len(runs), len(accepted)))

  if a.dry_run:
    print("(dry run -- nothing written)")
    return 0
  if accepted:
    mode = "a" if os.path.exists(a.out) else "w"
    with open(a.out, mode, encoding="utf-8") as f:
      for line in accepted:
        f.write(line + "\n")
    print("wrote %d rows -> %s (%s)" % (len(accepted), a.out, "append" if mode == "a" else "new"))
  if a.report:
    with open(a.report, "w", encoding="utf-8") as f:
      f.write("\n".join(log) + "\n")
  print("\nNEXT: fold in with the ORIGINAL corpus (never train on self-data alone --\n"
        "  that narrows the distribution), then gate the result:\n"
        "  python tools/clean_sft.py --shuffle --balance --out data/clean_reason2.txt \\\n"
        "      %s data/traj_reason.txt data/qa.txt data/toolcalls.txt\n"
        "  <retrain, 2 stages>  then  python tools/remeasure.py <new model>\n"
        "  PROMOTE ONLY IF non-regressive vs the recorded baseline." % a.out)
  return 0


sys.exit(main())
