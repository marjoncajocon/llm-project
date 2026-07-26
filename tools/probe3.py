"""Hand-probe 3-/4-step chains with NOVEL args (generalization, not memorization).

Same 7 cases as the lever-#3-further probe, PLUS an 8th verbose-observation case
(the 25d1113 fix) so a regression there is visible. Model path is argv[1] so the
350-step baseline and the 900-step (2-epoch) run can be A/B'd.

  python probe3.py model/smollm/reason_clean.bin
  python probe3.py model/smollm/reason_e2.bin
"""
import os
import re
import subprocess
import sys

ENGINE = os.path.abspath("transformer.exe")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "model/smollm/reason_clean.bin"
TOK = "model/smollm/tok.hf"


def gen(prompt, nnew=64):
    p = subprocess.run([ENGINE, "hfgen", MODEL, TOK, str(nnew), "0.1"],
                       input=prompt.encode("utf-8"),
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out = []
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line.startswith('{"d":"'):
            out.append(line[len('{"d":"'):-2].replace('\\"', '"').replace("\\n", "\n"))
    return "".join(out)


def action_of(text):
    m = re.search(r'<tool\b[^>]*>.*?</tool>|<done>.*?</done>', text, re.DOTALL)
    return m.group(0) if m else text.strip()


SEP = " | "
GIT_STATUS_OUT = ("On branch agent-1\nChanges not staged for commit:\n"
                  "  (use \"git add <file>...\" to update what will be committed)\n"
                  "\tmodified:   src/main.rs\n\nno changes added to commit")
# Each case: (list of transcript turns as prefix, expected-substring-in-next-action)
CASES = [
    # error recovery: failed read -> should WRITE the file with goal content
    (["GOAL: read zephyr_briefing.txt, but if it is missing create it with quantum meadow and read it again",
      'MODEL: <tool name="read_file" path="zephyr_briefing.txt"></tool>',
      "OBSERVATION: error: zephyr_briefing.txt: not found"],
     'write_file'),
    # verify chain step 3: after write + run, should READ back
    (['GOAL: create atlas_report.md containing pale horizon, run cargo build, then read it back to verify',
      'MODEL: <tool name="write_file" path="atlas_report.md">pale horizon</tool>',
      'OBSERVATION: wrote 12 bytes to atlas_report.md',
      'MODEL: <tool name="run_command" command="cargo build"></tool>',
      'OBSERVATION: ok'],
     'read_file'),
    # write->read->delete step 3: after write + read, should DELETE (rm)
    (['GOAL: create vault_ledger.log with silver comet, read it back, then delete it',
      'MODEL: <tool name="write_file" path="vault_ledger.log">silver comet</tool>',
      'OBSERVATION: wrote 12 bytes to vault_ledger.log',
      'MODEL: <tool name="read_file" path="vault_ledger.log"></tool>',
      'OBSERVATION: silver comet'],
     'rm vault_ledger.log'),
    # completed chain: after the final read, should be <done>
    (['GOAL: read zephyr_briefing.txt, but if it is missing create it with quantum meadow and read it again',
      'MODEL: <tool name="read_file" path="zephyr_briefing.txt"></tool>',
      'OBSERVATION: error: zephyr_briefing.txt: not found',
      'MODEL: <tool name="write_file" path="zephyr_briefing.txt">quantum meadow</tool>',
      'OBSERVATION: wrote 13 bytes to zephyr_briefing.txt',
      'MODEL: <tool name="read_file" path="zephyr_briefing.txt"></tool>',
      'OBSERVATION: quantum meadow'],
     '<done'),
    # 4-step: two writes -> run -> read first. After 2 writes, should RUN the cmd.
    (['GOAL: create nova_index.json with amber tide and orion_log.md with pale ember, run cargo test, then read nova_index.json',
      'MODEL: <tool name="write_file" path="nova_index.json">amber tide</tool>',
      'OBSERVATION: wrote 10 bytes to nova_index.json',
      'MODEL: <tool name="write_file" path="orion_log.md">pale ember</tool>',
      'OBSERVATION: wrote 10 bytes to orion_log.md'],
     'cargo test'),
    # 4-step error recovery + run: after failed read + write, should RUN the cmd.
    (['GOAL: read cobalt_manifest.txt; if it is missing create it with silent tide, then run go build and read it again',
      'MODEL: <tool name="read_file" path="cobalt_manifest.txt"></tool>',
      'OBSERVATION: error: cobalt_manifest.txt: not found',
      'MODEL: <tool name="write_file" path="cobalt_manifest.txt">silent tide</tool>',
      'OBSERVATION: wrote 11 bytes to cobalt_manifest.txt'],
     'go build'),
    # 4-step write->read->run->delete: after write+read+run, should DELETE (rm).
    (['GOAL: create quartz_notes.csv with hidden valley, read it back, run cargo fmt, then delete it',
      'MODEL: <tool name="write_file" path="quartz_notes.csv">hidden valley</tool>',
      'OBSERVATION: wrote 13 bytes to quartz_notes.csv',
      'MODEL: <tool name="read_file" path="quartz_notes.csv"></tool>',
      'OBSERVATION: hidden valley',
      'MODEL: <tool name="run_command" command="cargo fmt"></tool>',
      'OBSERVATION: ok'],
     'rm quartz_notes.csv'),
    # VERBOSE observation (the 25d1113 fix): a multi-line git status must NOT end the
    # chain -- the goal still has a read-back left.
    (['GOAL: create ember_trace.txt with narrow river, run git status, then read it back to verify',
      'MODEL: <tool name="write_file" path="ember_trace.txt">narrow river</tool>',
      'OBSERVATION: wrote 12 bytes to ember_trace.txt',
      'MODEL: <tool name="run_command" command="git status"></tool>',
      'OBSERVATION: ' + GIT_STATUS_OUT],
     'read_file'),
]


def main():
    passed = 0
    print("model: %s" % MODEL)
    for turns, expect in CASES:
        prompt = SEP.join(turns)
        act = action_of(gen(prompt))
        ok = expect in act
        passed += ok
        print("[%s] expect %-22s got: %s" % ("PASS" if ok else "FAIL", expect, act[:80]))
    print("\n%d/%d next-action probes correct" % (passed, len(CASES)))


main()
