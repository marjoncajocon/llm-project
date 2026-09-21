#!/usr/bin/env python3
"""Measure the proposer-critic loop (lever #4) vs a single greedy sample.

Two axes, both targeting the failure modes the lever-#3 live runs exposed:

  A. GROUNDING (novel-filename copy): "read the file <gibberish>.ext". A single
     greedy sample sometimes char-slips the name; the critic's grounding filter
     drops slipped candidates so a correct-copy sample wins. Expect best-of >= greedy.
  B. MULTI-STEP next-action: 3- and 4-step transcript prefixes with a KNOWN correct
     next action (incl. cases that must CONTINUE, not stop early). Expect best-of >= greedy.

Usage: python tools/eval_critic.py <model.bin> <tok.hf> [best_of] [M]
Deterministic seeds; greedy = temp 0.1 seed 1234, proposals = temp 0.7 seeds 1..N.
"""
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_critic import critique, extract_action, parse_history, propose, run_hfgen  # noqa: E402

ENGINE = os.path.abspath("transformer.exe")
if not os.path.exists(ENGINE):
    ENGINE = os.path.abspath("transformer")

MODEL, TOK = sys.argv[1], sys.argv[2]
BEST_OF = int(sys.argv[3]) if len(sys.argv) > 3 else 5
M = int(sys.argv[4]) if len(sys.argv) > 4 else 10
PROP_TEMP, GREEDY_TEMP = 0.7, 0.1
rng = random.Random(4242)


def gen(prompt, temp, seed, n_new=64):
    # Same stop-at-tag generation the live agent (agent_local.py) uses.
    return run_hfgen(ENGINE, MODEL, TOK, prompt, temp, seed, n_new=n_new)[0]


def greedy_action(prompt):
    return extract_action(gen(prompt, GREEDY_TEMP, 1234))


def single_prop_action(prompt):
    """One sample at the PROPOSAL temp -- isolates the critic's contribution
    (best-of vs a single draw from the SAME noisy distribution)."""
    return extract_action(gen(prompt, PROP_TEMP, 1))


def critic_action(prompt, taken=None, last_obs="", last_action=""):
    # seeds 1..BEST_OF from ONE engine call (one load, one prefill)
    cands = propose(ENGINE, MODEL, TOK, prompt, BEST_OF, PROP_TEMP, seed=1)
    choice, _ = critique(cands, prompt, taken or set(), last_obs, last_action)
    if choice is None:                       # all rejected -> greedy fallback
        choice = greedy_action(prompt)
    return choice


def gib(n=8):
    # 8 chars: longer names slip more often under a single greedy sample, so the
    # grounding filter's advantage (recover a correct-copy candidate) is visible.
    return "".join(rng.choice("bcdfghjklmnpqrstvwxz2479") for _ in range(n))


SEP = " | "
# (prefix turns, expected substring in the next action, needs-continue?)
PROBES = [
    (["GOAL: read zephyr_briefing.txt, but if it is missing create it with quantum meadow and read it again",
      'MODEL: <tool name="read_file" path="zephyr_briefing.txt"></tool>',
      "OBSERVATION: error: zephyr_briefing.txt: not found"], 'write_file', ""),
    (['GOAL: create atlas_report.md containing pale horizon, run cargo build, then read it back to verify',
      'MODEL: <tool name="write_file" path="atlas_report.md">pale horizon</tool>',
      'OBSERVATION: wrote 12 bytes to atlas_report.md',
      'MODEL: <tool name="run_command" command="cargo build"></tool>',
      'OBSERVATION: ok'], 'read_file', ""),
    (['GOAL: create nova_index.json with amber tide and orion_log.md with pale ember, run cargo test, then read nova_index.json',
      'MODEL: <tool name="write_file" path="nova_index.json">amber tide</tool>',
      'OBSERVATION: wrote 10 bytes to nova_index.json',
      'MODEL: <tool name="write_file" path="orion_log.md">pale ember</tool>',
      'OBSERVATION: wrote 10 bytes to orion_log.md'], 'cargo test', ""),
    # premature-done stressor: final read pending AFTER a VERBOSE observation
    (['GOAL: read cobalt_manifest.txt; if it is missing create it with silent tide, then run git status and read it again',
      'MODEL: <tool name="read_file" path="cobalt_manifest.txt"></tool>',
      'OBSERVATION: error: cobalt_manifest.txt: not found',
      'MODEL: <tool name="write_file" path="cobalt_manifest.txt">silent tide</tool>',
      'OBSERVATION: wrote 11 bytes to cobalt_manifest.txt',
      'MODEL: <tool name="run_command" command="git status"></tool>',
      'OBSERVATION: On branch agent-1 Changes not staged for commit modified data/foo.txt Untracked files bar.md no changes added to commit'],
     'read_file', "cobalt_manifest.txt"),
]


def probe_prefix(turns):
    return SEP.join(turns)


def main():
    print("engine=%s best_of=%d M=%d\n" % (ENGINE, BEST_OF, M))

    # ---- A. grounding / novel-filename copy ----
    print("=== A. novel-filename copy (grounding) ===")
    g_ok = s_ok = c_ok = 0
    for _ in range(M):
        fn = gib() + rng.choice([".txt", ".md", ".log", ".json"])
        prompt = "read the file " + fn
        ga, sa, ca = greedy_action(prompt), single_prop_action(prompt), critic_action(prompt)
        gh = (fn in ga) and ("read_file" in ga)
        sh = (fn in sa) and ("read_file" in sa)
        ch = (fn in ca) and ("read_file" in ca)
        g_ok += gh; s_ok += sh; c_ok += ch
    print("  greedy@0.1   %2d/%d (%.0f%%)" % (g_ok, M, 100.0 * g_ok / M))
    print("  single@0.7   %2d/%d (%.0f%%)" % (s_ok, M, 100.0 * s_ok / M))
    print("  best-of@0.7  %2d/%d (%.0f%%)\n" % (c_ok, M, 100.0 * c_ok / M))

    # ---- B. multi-step next-action ----
    print("=== B. multi-step next action (3- and 4-step) ===")
    gb = sb = cb = 0
    for turns, expect, _cont in PROBES:
        prompt = probe_prefix(turns)
        taken, last_obs, last_action = parse_history(turns)
        ga = greedy_action(prompt)
        sa = single_prop_action(prompt)
        ca = critic_action(prompt, taken, last_obs, last_action)
        gh, sh, ch = expect in ga, expect in sa, expect in ca
        gb += gh; sb += sh; cb += ch
        print("  expect %-14s greedy@0.1=%-4s single@0.7=%-4s best-of@0.7=%-4s"
              % (expect, "OK" if gh else "MISS", "OK" if sh else "MISS", "OK" if ch else "MISS"))
    n = len(PROBES)
    print("  greedy %d/%d | single@0.7 %d/%d | best-of %d/%d\n" % (gb, n, sb, n, cb, n))

    print("SUMMARY (higher=better):")
    print("  A  greedy@0.1 %.0f%% | single@0.7 %.0f%% | best-of@0.7 %.0f%%"
          % (100.0 * g_ok / M, 100.0 * s_ok / M, 100.0 * c_ok / M))
    print("  B  greedy@0.1 %d/%d | single@0.7 %d/%d | best-of@0.7 %d/%d"
          % (gb, n, sb, n, cb, n))
    print("  (critic isolates vs single@0.7 -- same temp; greedy@0.1 is the strong single-sample ref)")


if __name__ == "__main__":
    main()
