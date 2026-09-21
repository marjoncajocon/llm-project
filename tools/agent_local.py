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

PROMPT BUDGET: the model has a fixed context (max_seq, read from the LLMA/LLMQ
header). If the flattened transcript does not fit in max_seq minus the reply
reserve -- counted in real tokens with `transformer hftok <tok> count` -- it is
compacted the way agent.c compacts its own transcript: every OLD observation is
clipped to head + tail around "…[N bytes elided]", then the oldest turns are
dropped; the GOAL line and the latest turn always stay, the latest observation
whole unless even that cannot fit. AGENT_PROMPT_BYTES=N budgets by bytes instead.
A transcript that fits is passed through unchanged. (The engine is the last line
of defence: a prompt that still does not fit loses its OLDEST tokens, with a
warning on stderr, never the newest.)

GENERATION stops at the first closed </tool> / </done> block (`hfgen --stop-tags`;
AGENT_STOP_TAGS=0 turns it off): the agent executes only that block, so decoding the
rest of the budget is wasted time.

FAILURE is not success: if the engine cannot generate, or the model produces
nothing, this program exits nonzero with the reason on stderr, and the agent stops
with "backend error" instead of recording a fake <done>.
"""
import os
import re
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from agent_critic import GenerationError, propose, run_hfgen  # noqa: E402

ENGINE = os.path.join(os.path.dirname(HERE), "transformer.exe")
if not os.path.exists(ENGINE):
    ENGINE = os.path.join(os.path.dirname(HERE), "transformer")

SEP = " | "   # must match make_agent_traj.py
N_NEW = 64    # reasoning-dense output is "<think>...</think><tool .../>"

# --- proposer-critic (lever #4) -----------------------------------------------
# Off by default (BEST_OF=1 == the original greedy behaviour). Set AGENT_BEST_OF>1
# to sample several candidates at AGENT_TEMP (varied seeds) and let agent_critic
# pick a grounded, non-repeating action -- see tools/agent_critic.py. All BEST_OF
# candidates come from ONE engine call (one model load, one prompt prefill).
BEST_OF = int(os.environ.get("AGENT_BEST_OF", "1"))
PROPOSE_TEMP = float(os.environ.get("AGENT_TEMP", "0.7"))
GREEDY_TEMP = float(os.environ.get("AGENT_GREEDY_TEMP", "0.1"))
STOP_TAGS = os.environ.get("AGENT_STOP_TAGS", "1") != "0"

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


# --- prompt budget -------------------------------------------------------------

OBS_HEAD = OBS_TAIL = 256        # bytes kept at each end of an OLD observation (as agent.c)
MODEL_LINE = re.compile(r"^MODEL: ", re.MULTILINE)


def model_max_seq(path):
    """Context length from an LLMA/LLMQ header (magic, version, 8 x int32 config
    whose last entry is max_seq); None for other formats or on any error."""
    try:
        with open(path, "rb") as f:
            head = f.read(40)
        if len(head) == 40 and head[:4] in (b"LLMA", b"LLMQ"):
            return struct.unpack("<i", head[36:40])[0]
    except OSError:
        pass
    return None


def parse_items(raw):
    """Framed transcript -> (goal_line, [("model", text) | ("obs", body), ...])."""
    nl = raw.find("\n")
    goal, rest = (raw, "") if nl < 0 else (raw[:nl], raw[nl + 1:])
    items, pos = [], 0

    def models(text):
        starts = [m.start() for m in MODEL_LINE.finditer(text)]
        if not starts or starts[0] != 0:
            starts = [0] + starts
        for a, b in zip(starts, starts[1:] + [len(text)]):
            if text[a:b].strip():
                items.append(("model", text[a:b].rstrip("\n")))

    for m in OBS_FRAME.finditer(rest):
        models(rest[pos:m.start()])
        items.append(("obs", m.group(2)))
        pos = m.end()
    models(rest[pos:])
    return goal, items


def clip(body, head=OBS_HEAD, tail=OBS_TAIL):
    b = body.encode("utf-8")
    if len(b) <= head + tail + 32:
        return body
    h = b[:head].decode("utf-8", "ignore")                # drop a split character
    t = b[len(b) - tail:].decode("utf-8", "ignore")
    n = len(b) - len(h.encode("utf-8")) - len(t.encode("utf-8"))
    return "%s…[%d bytes elided]%s" % (h, n, t)


def render(goal, items):
    parts = [goal]
    for kind, text in items:
        parts.append(text if kind == "model" else "OBSERVATION: " + text)
    return flatten("\n".join(parts))


def compact(raw, budget):
    """Flattened prompt of the framed transcript `raw`, compacted toward `budget`
    UTF-8 bytes: clip every OLD observation (head + tail), then drop the oldest
    turns (a MODEL item and what follows it up to the next one) -- never the GOAL
    line or the latest turn. Returns (prompt, turns_dropped)."""
    goal, items = parse_items(raw)
    last_obs = max((i for i, (k, _) in enumerate(items) if k == "obs"), default=-1)
    items = [(k, clip(t) if k == "obs" and i != last_obs else t)
             for i, (k, t) in enumerate(items)]
    turn_starts = [i for i, (k, _) in enumerate(items) if k == "model"]
    dropped = 0
    while len(render(goal, items).encode("utf-8")) > budget and len(turn_starts) > 1:
        cut = turn_starts[1] - turn_starts[0]
        items = items[:turn_starts[0]] + items[turn_starts[1]:]
        turn_starts = [s - cut for s in turn_starts[1:]]
        dropped += 1
    out = render(goal, items)
    # Last resort: even GOAL + the latest turn is too long. Clip the remaining
    # observations harder (the latest included) rather than let the engine cut
    # the GOAL off the front of the prompt.
    keep = OBS_HEAD
    while len(out.encode("utf-8")) > budget and keep > 16 and any(k == "obs" for k, _ in items):
        keep = keep * 3 // 4
        out = render(goal, [(k, clip(t, keep, keep) if k == "obs" else t) for k, t in items])
    return out, dropped


def count_tokens(tok, text):
    """Token count of `text` exactly as hfgen tokenizes it (None if unavailable)."""
    try:
        proc = subprocess.run([ENGINE, "hftok", tok, "count"], input=text.encode("utf-8"),
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return int(proc.stdout.split()[0]) if proc.returncode == 0 else None
    except (OSError, ValueError, IndexError):
        return None


def fit_prompt(raw, model, tok):
    """The prompt for this step. Returns (prompt, note); note is None when the
    transcript fits as is -- then the prompt is exactly flatten(unwrap(raw)).

    The limit is the model's context minus the reply reserve (the same reserve
    hfgen keeps, so the engine never has to cut). AGENT_PROMPT_BYTES=N instead
    budgets by bytes. Byte-level BPE never makes more tokens than bytes, so a
    prompt with at most `limit` bytes fits without counting."""
    prompt = flatten(unwrap(raw))
    size = len(prompt.encode("utf-8"))
    env_bytes = int(os.environ.get("AGENT_PROMPT_BYTES", "0"))
    if env_bytes > 0:
        if size <= env_bytes:
            return prompt, None
        out, dropped = compact(raw, env_bytes)
        return out, "prompt %d -> %d bytes (AGENT_PROMPT_BYTES %d); %d oldest turn(s) dropped" % (
            size, len(out.encode("utf-8")), env_bytes, dropped)
    ms = model_max_seq(model)
    if not ms:
        return prompt, None                   # unknown context: the engine keeps the newest tokens
    limit = ms - min(N_NEW, ms // 2)
    if size <= limit:
        return prompt, None
    n = count_tokens(tok, prompt)
    if n is None:
        n = size // 3 + 1                     # no tokenizer run: assume ~3 bytes per token
    if n <= limit:
        return prompt, None
    out, dropped, n0, budget = prompt, 0, n, size
    for _ in range(8):                        # shrink the byte budget until the tokens fit
        budget = max(64, int(budget * limit / n * 0.95))
        out, dropped = compact(raw, budget)
        n = count_tokens(tok, out)
        if n is None:
            n = len(out.encode("utf-8")) // 3 + 1
        if n <= limit:
            break
    note = "prompt %d -> %d tokens (context %d, %d reserved for the reply): old observations" \
        " clipped, %d oldest turn(s) dropped" % (n0, n, ms, ms - limit, dropped)
    if n > limit:
        note += "; the GOAL and latest turn alone are too long -- the engine will drop the oldest tokens"
    return out, note


# --- main ------------------------------------------------------------------------

def split_reason_action(text):
    """Return (reasoning, action). The model may emit "<think>why</think><tool .../>";
    the reasoning is shown to the user but NOT recorded in the transcript, so the
    history stays bare tool calls (matching the trajectory SFT prefixes)."""
    think = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    reason = think.group(1).strip() if think else ""
    m = re.search(r'<tool\b[^>]*>.*?</tool>|<done>.*?</done>', text, re.DOTALL)
    action = m.group(0) if m else text.strip()
    return reason, action


# Model text echoed to the terminal (stderr) must not carry terminal control
# sequences: drop C0 (except \n, \t), DEL, C1 and bidi overrides.
_CTRL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f‪-‮⁦-⁩]")


def printable(s):
    return _CTRL.sub("", s)


def gen(model, tok, prompt, temp, seed, n_new=N_NEW):
    """One raw model completion (raises GenerationError on engine failure)."""
    return run_hfgen(ENGINE, model, tok, prompt, temp, seed, n_new=n_new, stop_tags=STOP_TAGS)[0]


def fail(msg):
    sys.stderr.write("agent_local: %s\n" % printable(msg))
    sys.stderr.flush()
    sys.exit(1)


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: agent_local.py <model.bin> <tok.hf> [transcript]")
    model, tok = sys.argv[1], sys.argv[2]
    # agent.c reads our stdout as UTF-8 bytes; a Windows pipe would otherwise get
    # the ANSI code page and crash on the first character outside it.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    raw = ""
    if len(sys.argv) > 3 and os.path.exists(sys.argv[3]):
        with open(sys.argv[3], encoding="utf-8", errors="replace") as f:
            raw = f.read()
    transcript = unwrap(raw)

    prompt, note = fit_prompt(raw, model, tok)
    if not prompt:
        print("<done>Nothing to do.</done>")
        return
    if note:
        sys.stderr.write("  budget: %s\n" % note)

    try:
        if BEST_OF > 1:
            # PROPOSER-CRITIC: BEST_OF candidates (seeds 1..BEST_OF, so they differ)
            # from ONE engine call; the critic picks a grounded, non-repeating,
            # non-premature action.
            from agent_critic import critique, parse_history
            taken, last_obs, last_action = parse_history(transcript.splitlines())
            cands = propose(ENGINE, model, tok, prompt, BEST_OF, PROPOSE_TEMP, seed=1,
                            n_new=N_NEW, stop_tags=STOP_TAGS)
            choice, log = critique(cands, prompt, taken, last_obs, last_action)
            for line in log:
                sys.stderr.write("  " + printable(line) + "\n")
            if choice is None:                       # every candidate rejected
                sys.stderr.write("  critic: falling back to greedy\n")
                _, choice = split_reason_action(gen(model, tok, prompt, GREEDY_TEMP, 1234))
            sys.stderr.flush()
            action = choice
        else:
            # Default: a single near-greedy sample (original behaviour).
            reason, action = split_reason_action(gen(model, tok, prompt, GREEDY_TEMP, 1234))
            if reason:
                # stderr is shown to the user by the agent but not captured as the
                # model's response, so the recorded transcript turn stays a bare tool call.
                sys.stderr.write("  thought: %s\n" % printable(reason))
                sys.stderr.flush()
    except GenerationError as e:
        fail("generation failed: %s" % e)
    if not action:
        fail("the model produced no output (nothing to execute)")
    trace(prompt, action)
    print(action)


if __name__ == "__main__":
    main()
