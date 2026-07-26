"""Run the FULL validation set against one checkpoint, in one command.

Why this exists: the 25d1113 retrain was validated on sft_memtest + eval_critic
only, the 8-case next-action probe was skipped, and a real regression (premature
<done> after a TERSE "ok" observation) shipped unnoticed while the notes still
claimed "probe 7/7". A partial validation hides the trade a targeted fix made.
Running every probe is the cheap insurance; this script makes that the default
and prints one summary so a skipped probe is visible instead of forgotten.

Usage:
  python tools/remeasure.py <model.bin> [--tok F] [--train F] [--best-of N] [--m N]
  python tools/remeasure.py model/smollm/reason_e2.bin
  python tools/remeasure.py model/smollm/reason_e2.bin --best-of 1 --m 8   # quick

Compare against the recorded baseline for model/smollm/reason_clean.bin:
  probe3 6/8 (FAILs = the two terse-"ok" cases) | memtest A 95% B 80% | critic A 96% B 3/4
Bars: A >= 90%, B 75-100%. Promote a new checkpoint ONLY if non-regressive.
Every subprocess reloads the model (~651MB), so the full set is slow by design.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(title, argv):
  """Run one probe, echo its output live-ish, return (title, captured text)."""
  print("\n" + "=" * 72)
  print("== " + title)
  print("=" * 72)
  sys.stdout.flush()
  p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  out = p.stdout.decode("utf-8", "replace")
  print(out)
  sys.stdout.flush()
  if p.returncode != 0:
    print("  !! %s exited %d -- treat its numbers as UNTRUSTED" % (title, p.returncode))
  return out


def main():
  if len(sys.argv) < 2:
    print(__doc__)
    return 2
  model = sys.argv[1]
  tok, train = "model/smollm/tok.hf", "data/clean_reason.txt"
  best_of, m = "5", "24"
  a = sys.argv[2:]
  for i, f in enumerate(a):
    if f == "--tok" and i + 1 < len(a): tok = a[i + 1]
    elif f == "--train" and i + 1 < len(a): train = a[i + 1]
    elif f == "--best-of" and i + 1 < len(a): best_of = a[i + 1]
    elif f == "--m" and i + 1 < len(a): m = a[i + 1]

  # ABSOLUTE: sft_memtest takes the engine as an argv and execs it verbatim, so a
  # relative path fails (probe3/eval_critic abspath it themselves and hid this).
  engine = os.path.abspath("transformer.exe" if os.path.exists("transformer.exe")
                           else "transformer")
  for path in (model, tok, train, engine):
    if not os.path.exists(path):
      print("error: missing %s (run from the repo root)" % path)
      return 1
  print("re-measuring %s\n  tok=%s train=%s engine=%s" % (model, tok, train, engine))

  probe = run("1/3 next-action probe (8 cases)",
              [PY, os.path.join(HERE, "probe3.py"), model])
  mem = run("2/3 sft_memtest (memorization vs generalization)",
            [PY, os.path.join(HERE, "sft_memtest.py"), engine, model, tok, train])
  crit = run("3/3 eval_critic (grounding + multi-step, best_of=%s M=%s)" % (best_of, m),
             [PY, os.path.join(HERE, "eval_critic.py"), model, tok, best_of, m])

  # Summary. Anything we cannot parse is reported as "?" rather than guessed --
  # a missing number must not read as a pass.
  print("\n" + "=" * 72)
  print("== SUMMARY -- %s" % model)
  print("=" * 72)
  mp = re.search(r"(\d+)/(\d+) next-action probes correct", probe)
  print("  probe3          : %s   (baseline reason_clean.bin = 6/8)" %
        (mp.group(0).split(" ")[0] if mp else "?"))
  pct = re.findall(r"copied-correctly: \d+/\d+ \((\d+)%\)", mem)
  rep = re.search(r"reproduced: \d+/\d+ \((\d+)%\)", mem)
  print("  sft_memtest     : A %s%%  B %s%%  C %s%%   (bars: A>=90, B 75-100; "
        "C is a harness artifact, not a bar)" %
        (pct[0] if len(pct) > 0 else "?", pct[1] if len(pct) > 1 else "?",
         rep.group(1) if rep else "?"))
  print("  eval_critic     : see section 3 above (A copy-rate, B1-B4 multi-step)")
  fails = re.findall(r"^\[FAIL\] expect (\S+)", probe, re.M)
  print("  probe FAILs     : %s" % (", ".join(fails) if fails else "none"))
  print("\n  Promote over reason_clean.bin ONLY if non-regressive on ALL of the above.")
  return 0


sys.exit(main())
