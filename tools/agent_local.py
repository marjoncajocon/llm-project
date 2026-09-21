#!/usr/bin/env python3
"""Model backend that runs THIS PROJECT'S OWN engine — no API, no network.

    transformer agent --cmd "python tools/agent_local.py <model.bin> <tok.hf>" [--auto]

The C agent (agent.c) writes the running transcript to a temp file and passes its
path as the last argument; this program's STDOUT becomes the model's next action.
The transcript looks like:

    GOAL: <goal>
    MODEL: <action>
    OBSERVATION: [[obs:NONCE]]
    <obs>
    [[/obs:NONCE]]
    ...

(unwrapped to "OBSERVATION: <obs>" below, the format the SFT data uses).

For MULTI-STEP planning the model must see the WHOLE transcript, not just the last
line. We flatten it to the SAME " | "-joined single line that the trajectory SFT
data uses (tools/make_agent_traj.py), so training and inference formats match, then
generate the next action near-greedily (temp 0.1). The model itself emits <done>
when the goal is met; the agent loop stops there.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(os.path.dirname(HERE), "transformer.exe")
if not os.path.exists(ENGINE):
    ENGINE = os.path.join(os.path.dirname(HERE), "transformer")

SEP = " | "   # must match make_agent_traj.py

# --- proposer-critic (lever #4) -----------------------------------------------
# Off by default (BEST_OF=1 == the original greedy behaviour). Set AGENT_BEST_OF>1
# to sample several candidates at AGENT_TEMP (varied seeds) and let agent_critic
# pick a grounded, non-repeating action -- see tools/agent_critic.py.
BEST_OF = int(os.environ.get("AGENT_BEST_OF", "1"))
PROPOSE_TEMP = float(os.environ.get("AGENT_TEMP", "0.7"))
GREEDY_TEMP = float(os.environ.get("AGENT_GREEDY_TEMP", "0.1"))

# --- self-training trace (STaR) ------------------------------------------------
# Off unless AGENT_TRACE names a .jsonl. Each step appends the EXACT (prompt,
# response) pair the model produced -- already the SFT row format, so no lossy
# reconstruction from the console log. tools/self_train.py rebuilds whole runs by
# PREFIX-CHAINING (step k's prompt contains step k-1's verbatim), which needs no
# run id and survives interleaved/aborted runs.
TRACE = os.environ.get("AGENT_TRACE", "")


def trace(prompt, response):
  if not TRACE:
    return
  import json
  try:
    with open(TRACE, "a", encoding="utf-8") as f:
      f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")
  except OSError:
    pass                                  # tracing must never break a live run


# agent.c frames every observation as
#   OBSERVATION: [[obs:NONCE]]\n<text>\n[[/obs:NONCE]]
# (a per-run nonce, so tool output cannot fake the end of an observation). The
# SFT data (make_agent_traj.py) uses plain "OBSERVATION: <text>", so unwrap the
# frame here; role-marker lines inside <text> were already neutralized ("> ").
OBS_FRAME = re.compile(r"OBSERVATION: \[\[obs:([0-9a-f]+)\]\]\n(.*?)\n\[\[/obs:\1\]\]", re.DOTALL)


def unwrap(transcript):
    return OBS_FRAME.sub(lambda m: "OBSERVATION: " + m.group(2), transcript)


def flatten(transcript):
    """Real-newline transcript -> single line joined by SEP (matches training)."""
    lines = [ln.rstrip() for ln in transcript.splitlines() if ln.strip()]
    return SEP.join(lines)


def decode_deltas(stdout):
    text = []
    for line in stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line.startswith('{"d":"'):
            body = line[len('{"d":"'):-2]
            text.append(body.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\"))
    return "".join(text)


# Model text echoed to the terminal (stderr) must not carry terminal control
# sequences: drop C0 (except \n, \t), DEL, C1 and bidi overrides.
_CTRL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def printable(s):
    return _CTRL.sub("", s)


def split_reason_action(text):
    """Return (reasoning, action). The model may emit "<think>why</think><tool .../>";
    the reasoning is shown to the user but NOT recorded in the transcript, so the
    history stays bare tool calls (matching the trajectory SFT prefixes)."""
    think = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    reason = think.group(1).strip() if think else ""
    m = re.search(r'<tool\b[^>]*>.*?</tool>|<done>.*?</done>', text, re.DOTALL)
    action = m.group(0) if m else text.strip()
    return reason, action


def gen(model, tok, prompt, temp, seed, n_new=64):
    """One raw model completion. 64 new tokens: reasoning-dense output is
    "<think>...</think><tool .../>", longer than a bare tool call."""
    proc = subprocess.run(
        [ENGINE, "hfgen", model, tok, str(n_new), str(temp), str(seed)],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return decode_deltas(proc.stdout)


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: agent_local.py <model.bin> <tok.hf> [transcript]")
    model, tok = sys.argv[1], sys.argv[2]
    transcript = ""
    if len(sys.argv) > 3 and os.path.exists(sys.argv[3]):
        with open(sys.argv[3], encoding="utf-8", errors="replace") as f:
            transcript = unwrap(f.read())

    prompt = flatten(transcript)
    if not prompt:
        print("<done>Nothing to do.</done>")
        return

    if BEST_OF > 1:
        # PROPOSER-CRITIC: sample BEST_OF candidates (varied seeds so they differ),
        # let the critic pick a grounded, non-repeating, non-premature action.
        from agent_critic import critique, parse_history
        taken, last_obs, last_action = parse_history(transcript.splitlines())
        cands = [gen(model, tok, prompt, PROPOSE_TEMP, seed) for seed in range(1, BEST_OF + 1)]
        choice, log = critique(cands, prompt, taken, last_obs, last_action)
        for line in log:
            sys.stderr.write("  " + printable(line) + "\n")
        if choice is None:                       # every candidate rejected
            sys.stderr.write("  critic: falling back to greedy\n")
            _, choice = split_reason_action(gen(model, tok, prompt, GREEDY_TEMP, 1234))
        sys.stderr.flush()
        trace(prompt, choice or "<done>No response.</done>")
        print(choice or "<done>No response.</done>")
        return

    # Default: a single near-greedy sample (original behaviour).
    reason, action = split_reason_action(gen(model, tok, prompt, GREEDY_TEMP, 1234))
    if reason:
        # stderr is shown to the user by the agent but not captured as the model's
        # response, so the recorded transcript turn stays a bare tool call.
        sys.stderr.write("  thought: %s\n" % printable(reason))
        sys.stderr.flush()
    trace(prompt, action or "<done>No response.</done>")
    print(action or "<done>No response.</done>")


if __name__ == "__main__":
    main()
