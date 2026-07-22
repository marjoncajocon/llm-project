#!/usr/bin/env python3
"""Real-LLM backend for `transformer agent --cmd "python tools/agent_llm.py"`.

This turns the scripted agent loop into one that genuinely acts. The C agent
(`agent.c`, command backend) writes the running transcript to a temp file,
passes its path as argv[1], and uses this program's STDOUT as the model's next
response. We call a real LLM with a system prompt that constrains it to emit
EXACTLY one of the loop's blocks:

    <tool name="write_file" path="a.txt">file contents here</tool>
    <tool name="read_file" path="a.txt"></tool>
    <tool name="run_command" command="echo hi"></tool>
    <done>one-line summary of what was accomplished</done>

so the reply parses cleanly in `toolcall_parse` (see toolcall.c). The approval
gate (`transformer agent --ask`) stays as the safety layer.

Provider: Claude via the Anthropic API (default), or OpenAI if only its key is
set. Needs ANTHROPIC_API_KEY (or OPENAI_API_KEY). With no key / no SDK, we print
a <done> block so the loop ends cleanly instead of hanging -- for a no-key
offline run, use tools/agent_mock.py instead.

Run:
    transformer agent --cmd "python tools/agent_llm.py" --goal "..." --ask
"""
import os
import sys

# The one hard contract: the model's reply must be a single tool/done block.
SYSTEM_PROMPT = """\
You are a coding agent driving a loop. You are given a transcript containing a \
GOAL line and, after each of your actions, an OBSERVATION line with the result. \
Decide the single next action and reply with EXACTLY ONE block, nothing else -- \
no prose, no markdown code fences, no explanation outside the block. Use one of:

  <tool name="write_file" path="RELATIVE/PATH">FILE CONTENTS</tool>
  <tool name="read_file" path="RELATIVE/PATH"></tool>
  <tool name="run_command" command="SHELL COMMAND"></tool>
  <done>one-line summary of what was accomplished</done>

Rules:
- Attribute values use double quotes; do not put a literal double quote inside a
  path or command value.
- write_file puts the whole file body between the tags (it is overwritten).
- Take one concrete step at a time; read/verify before you claim success.
- When the GOAL is fully accomplished (confirmed by an OBSERVATION), emit <done>.
- If you are stuck or the goal is already satisfied, emit <done> explaining why.
Emit the block and nothing else."""

MAX_TOKENS = 2048


def read_transcript():
  if len(sys.argv) > 1:
    with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
      return f.read()
  # Fall back to stdin so the script is also usable/testable by hand.
  return sys.stdin.read() if not sys.stdin.isatty() else ""


def clean(text):
  """Trim whitespace and any stray ``` fences the model may wrap around the block."""
  text = text.strip()
  if text.startswith("```"):
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
      lines = lines[1:]
    if lines and lines[-1].strip() == "```":
      lines = lines[:-1]
    text = "\n".join(lines).strip()
  return text


def ask_anthropic(transcript):
  import anthropic  # pip install anthropic
  client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
  msg = client.messages.create(
      model="claude-opus-4-8",
      max_tokens=MAX_TOKENS,
      thinking={"type": "adaptive"},
      system=SYSTEM_PROMPT,
      messages=[{"role": "user", "content": transcript}],
  )
  # Concatenate text blocks only (skip thinking blocks).
  return "".join(b.text for b in msg.content if b.type == "text")


def ask_openai(transcript):
  from openai import OpenAI  # pip install openai
  client = OpenAI()  # reads OPENAI_API_KEY
  resp = client.chat.completions.create(
      model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
      max_tokens=MAX_TOKENS,
      messages=[
          {"role": "system", "content": SYSTEM_PROMPT},
          {"role": "user", "content": transcript},
      ],
  )
  return resp.choices[0].message.content or ""


def main():
  transcript = read_transcript()
  try:
    if os.environ.get("ANTHROPIC_API_KEY"):
      reply = ask_anthropic(transcript)
    elif os.environ.get("OPENAI_API_KEY"):
      reply = ask_openai(transcript)
    else:
      print("<done>No API key set (ANTHROPIC_API_KEY / OPENAI_API_KEY). "
            "Use tools/agent_mock.py for an offline run.</done>")
      return
  except Exception as e:  # noqa: BLE001 -- surface any failure as a clean stop
    # Print a <done> block so the C loop terminates instead of hanging.
    msg = str(e).replace("<", "(").replace(">", ")")
    print("<done>agent_llm backend error: " + msg + "</done>")
    return

  reply = clean(reply)
  if not reply:
    reply = "<done>(empty response from model)</done>"
  print(reply)


if __name__ == "__main__":
  main()
