#!/usr/bin/env python3
"""Model backend that runs THIS PROJECT'S OWN engine — no API, no network.

    transformer agent --cmd "python tools/agent_local.py <model.bin> <tok.hf>"

The C agent writes the running transcript to a temp file and passes its path as
the last argument; this program's STDOUT becomes the model's next response. Here
that response comes from `transformer hfgen`, i.e. a LLaMA-family checkpoint
fine-tuned by `llama-train --sft` to emit tool calls.

Sampling is near-greedy (temp 0.1): the model should route an instruction to its
tool call deterministically, not sample creatively.

Note the honest limitation — a 135M model SFT'd on a handful of examples answers
the LAST instruction; it does not plan a multi-step task. The loop below feeds it
just the newest instruction line and stops once the model says <done>.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(os.path.dirname(HERE), "transformer.exe")
if not os.path.exists(ENGINE):
    ENGINE = os.path.join(os.path.dirname(HERE), "transformer")


def last_instruction(transcript):
    """The most recent line the agent wants acted on (GOAL, or newest OBSERVATION)."""
    goal = ""
    for line in transcript.splitlines():
        s = line.strip()
        if s.startswith("GOAL:"):
            goal = s[len("GOAL:"):].strip()
    return goal


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: agent_local.py <model.bin> <tok.hf> [transcript]")
    model, tok = sys.argv[1], sys.argv[2]
    transcript = ""
    if len(sys.argv) > 3 and os.path.exists(sys.argv[3]):
        with open(sys.argv[3], encoding="utf-8", errors="replace") as f:
            transcript = f.read()

    # One tool call per turn: once a tool has run, the task is done.
    if "OBSERVATION:" in transcript:
        print("<done>Done.</done>")
        return

    prompt = last_instruction(transcript)
    if not prompt:
        print("<done>Nothing to do.</done>")
        return

    proc = subprocess.run([ENGINE, "hfgen", model, tok, "32", "0.1"],
                          input=prompt.encode("utf-8"),
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # hfgen streams {"d":"piece"} JSON deltas; concatenate them back into text.
    text = []
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line.startswith('{"d":"'):
            body = line[len('{"d":"'):-2]
            text.append(body.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\"))
    print("".join(text).strip() or "<done>No response.</done>")


main()
