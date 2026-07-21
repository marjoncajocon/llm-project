#!/usr/bin/env python3
"""Example model backend for `transformer agent --cmd "python tools/agent_mock.py"`.

The C agent writes the running transcript to a temp file, passes its path as
argv[1], and uses this program's STDOUT as the model's next response. This mock
drives the loop deterministically (write a file -> read it back -> done) to prove
the command-backend path end-to-end without any API key.

Swap the body for a real LLM call to make the agent genuinely act (see bottom).
"""
import sys

transcript = ""
if len(sys.argv) > 1:
    transcript = open(sys.argv[1], encoding="utf-8", errors="replace").read()

steps_done = transcript.count("OBSERVATION:")

if steps_done == 0:
    print('<tool name="write_file" path="agent_demo.txt">Hello from the command backend!</tool>')
elif steps_done == 1:
    print('<tool name="read_file" path="agent_demo.txt"></tool>')
else:
    print('<done>Created and verified agent_demo.txt via the command backend.</done>')

# --- To use a REAL LLM instead of this mock (make the agent genuinely act): ---
#   import anthropic                       # pip install anthropic ; set ANTHROPIC_API_KEY
#   client = anthropic.Anthropic()
#   msg = client.messages.create(
#       model="claude-sonnet-5", max_tokens=1024,
#       system=("You are a coding agent. Given the transcript, reply with exactly ONE "
#               "action: <tool name=\"write_file|read_file|run_command\" .../> or "
#               "<done>summary</done>."),
#       messages=[{"role": "user", "content": transcript}])
#   print(msg.content[0].text)
