#!/usr/bin/env python3
"""Proposer-critic selection for the local agent (lever #4).

The PROPOSER (the SFT model, sampled at temp>0 with a DIFFERENT seed per call, so
the samples actually differ) offers several candidate next-actions. This CRITIC
filters + ranks them so the agent commits a grounded, non-repeating action and does
not stop early -- attacking the two failure modes the lever-#3 live runs exposed
(filename char-slips and a premature <done> on a verbose observation):

  1. GROUNDING (reuses clean_sft.verify -- the SAME check that gates the training
     data): every path / write-content / command-filename in a tool call must appear
     in the goal+transcript. This DETERMINISTICALLY rejects a char-slipped filename
     (e.g. demo_note.txt -> demonstra_note.txt): the slipped string is absent from
     the prompt, so that candidate is dropped and a correct-copy sample wins.
  2. ANTI-REPEAT: drop a candidate identical to an action already executed
     SUCCESSFULLY (no loops). Failed actions are exempt so a legitimate re-read
     after an error-recovery write is not filtered.
  3. RETRY-GUARD: but right after a FAILED action, drop a candidate that re-attempts
     that exact action -- the fix (e.g. writing the missing file) must come first.
     (Without this the exemption in (2) lets the agent re-read a missing file forever;
     an early live run did exactly that until this guard was added.)
  4. DONE-GUARD + SELF-CONSISTENCY: never commit <done> right after an ERROR
     observation, and only stop when at least as many candidates vote <done> as vote
     a fresh grounded action; otherwise commit the MAJORITY grounded action.

If every candidate is rejected the caller falls back to its own greedy sample, so
the critic can only help, never strand the agent.

PROPOSING: propose() gets all N candidates from ONE engine call
(`transformer hfgen ... --samples N --stop-tags`): the model is loaded once and the
prompt prefilled once, then sampled N times with seeds seed..seed+N-1 -- the same
candidates N separate `hfgen` calls with those seeds would give (selftest [40]),
without reloading a 651 MB model per sample.

This module is import-only (no CLI); tools/agent_local.py and tools/eval_critic.py
use select() / critique() and the engine helpers run_hfgen() / propose().
"""
import re
import subprocess
import sys
from collections import Counter

# Canonical grounding logic lives in clean_sft (its main() is guarded, so importing
# is side-effect-free). Reusing it means the critic and the data gate never drift.
from clean_sft import verify, norm

# --- engine interface ----------------------------------------------------------

class GenerationError(RuntimeError):
    """The engine could not produce a response (bad model, empty prompt, crash)."""


_ESC = {ord('"'): b'"', ord('\\'): b'\\', ord('n'): b'\n', ord('r'): b'\r',
        ord('t'): b'\t', ord('/'): b'/'}


def _unescape(raw):
    """Undo hf_emit_delta's escaping at the BYTE level (a token may end inside a
    UTF-8 character, so pieces are joined as bytes and decoded once at the end)."""
    out, i, n = bytearray(), 0, len(raw)
    while i < n:
        c = raw[i]
        if c == 0x5C and i + 1 < n:                       # backslash
            e = raw[i + 1]
            if e == ord('u') and i + 5 < n:
                out.append(int(raw[i + 2:i + 6], 16) & 0xFF)   # only \u00XX is emitted
                i += 6
                continue
            out += _ESC.get(e, bytes([e]))
            i += 2
            continue
        out.append(c)
        i += 1
    return bytes(out)


def parse_hfgen(stdout):
    """Parse `transformer hfgen` stdout (bytes) -> (samples, error).

    Single-sample output is {"d":...} lines then {"done":true}; with --samples N
    each sample is framed {"sample":i,"seed":S} ... {"end":i}. `error` is the
    text of an {"error":...} line, else None."""
    samples, cur, error = [], None, None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(b'{"d":"') and line.endswith(b'"}'):
            if cur is None:
                cur = bytearray()
                samples.append(cur)
            cur += _unescape(line[6:-2])
        elif line.startswith(b'{"sample":'):
            cur = bytearray()
            samples.append(cur)
        elif line.startswith(b'{"end":'):
            cur = None
        elif line.startswith(b'{"error":'):
            error = line[len(b'{"error":"'):-2].decode("utf-8", "replace")
    return [bytes(s).decode("utf-8", "replace") for s in samples], error


def run_hfgen(engine, model, tok, prompt, temp, seed, n_new=64, n_samples=1,
              stop_tags=True, log=None, constrain=None):
    """Run the engine once; return a list of n_samples completions.

    `constrain` (None/"" = off, the default) adds hfgen's constrained decoding:
    "path" (or "1") = --constrain, "cmd" / "body" add --constrain-cmd /
    --constrain-body, "all" = both; several may be joined with ','.

    Engine warnings (e.g. an over-long prompt that had to lose its oldest
    tokens) are forwarded to `log` (default: stderr). Raises GenerationError
    when the engine reports an error or exits nonzero -- a failed generation
    must never look like a model that chose to say nothing."""
    log = log or sys.stderr
    cmd = [engine, "hfgen", model, tok, str(n_new), str(temp), str(seed)]
    if stop_tags:
        cmd.append("--stop-tags")
    if n_samples > 1:
        cmd += ["--samples", str(n_samples)]
    cmd += constrain_flags(constrain)
    proc = subprocess.run(cmd, input=prompt.encode("utf-8"),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for ln in proc.stderr.decode("utf-8", "replace").splitlines():
        if "warning" in ln or "error" in ln:
            log.write("  engine: %s\n" % ln.strip())
    samples, error = parse_hfgen(proc.stdout)
    if error or proc.returncode != 0:
        raise GenerationError(error or "engine exited with status %d" % proc.returncode)
    samples += [""] * (n_samples - len(samples))       # a sample that stopped at once
    return samples[:n_samples]


def constrain_flags(spec):
    """hfgen flags for a constrain spec (see run_hfgen); [] when off."""
    parts = {p.strip().lower() for p in str(spec or "").split(",")} - {"", "0", "off", "no"}
    if not parts:
        return []
    flags = ["--constrain"]
    if parts & {"cmd", "all"}:
        flags.append("--constrain-cmd")
    if parts & {"body", "all"}:
        flags.append("--constrain-body")
    return flags


def propose(engine, model, tok, prompt, n, temp, seed=1, n_new=64, stop_tags=True, log=None,
            constrain=None):
    """N candidate completions (seeds seed..seed+n-1) from ONE load + ONE prefill."""
    return run_hfgen(engine, model, tok, prompt, temp, seed, n_new=n_new, n_samples=n,
                     stop_tags=stop_tags, log=log, constrain=constrain)


# --- critic ----------------------------------------------------------------------

ACTION_RE = re.compile(r'<tool\b[^>]*>.*?</tool>|<done>.*?</done>', re.DOTALL)
NAME_RE = re.compile(r'name="(\w+)"')
_ERR_HINTS = ("error", "not found", "no such", "cannot", "failed", "denied",
              "traceback", "exception")


def extract_action(text):
    """First well-formed <tool ...>...</tool> or <done>...</done> block, else ''."""
    m = ACTION_RE.search(text or "")
    return m.group(0) if m else ""


def kind_of(action):
    if action.startswith("<done"):
        return "done"
    if action.startswith("<tool"):
        m = NAME_RE.search(action)
        return "tool:" + (m.group(1) if m else "?")
    return "malformed"


def obs_is_error(obs):
    o = (obs or "").lower()
    return any(h in o for h in _ERR_HINTS)


def parse_history(lines):
    """(taken, last_obs, last_action) from transcript lines.

    `taken` holds only actions whose observation SUCCEEDED, so the legitimate
    re-read after an error-recovery WRITE is not mistaken for a loop and filtered
    (that would break the error-recovery chains lever #3 added). `last_action` +
    `last_obs` let the critic block an IMMEDIATE retry of a just-FAILED action --
    otherwise, since failed actions are exempt from `taken`, the agent can loop
    re-attempting the same failing read forever instead of fixing it."""
    taken, last_obs, last_action, pending = set(), "", "", None
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("MODEL:"):
            pending = norm(ln[len("MODEL:"):])
            last_action = pending
        elif ln.startswith("OBSERVATION:"):
            obs = ln[len("OBSERVATION:"):].strip()
            last_obs = obs
            if pending and not obs_is_error(obs):
                taken.add(pending)          # only SUCCESSFUL actions block repeats
            pending = None
    return taken, last_obs, last_action


def critique(candidates, transcript, taken=None, last_obs="", last_action=""):
    """Return (chosen_action_or_None, log_lines).

    candidates  : list of raw model outputs (may contain <think>...</think>).
    transcript  : flattened "GOAL: ... | MODEL: ... | OBSERVATION: ..." prompt.
    taken       : set of norm(action) already executed SUCCESSFULLY (anti-loop).
    last_obs    : the most recent observation text (done-guard + retry-guard).
    last_action : norm() of the most recent action (to block an immediate retry
                  of a just-FAILED action -- e.g. re-reading a missing file
                  instead of creating it).
    """
    taken = taken or set()
    # If the last action errored, don't immediately re-attempt that exact action;
    # the agent should do something else (the fix), not loop on the failure.
    retry_block = last_action if (last_action and obs_is_error(last_obs)) else None
    log = []
    grounded_tools, dones, rejected = [], [], 0
    for raw in candidates:
        act = extract_action(raw)
        if not act:
            rejected += 1
            continue
        k = kind_of(act)
        if k == "done":
            dones.append(act)
        elif k.startswith("tool:"):
            reason = verify(transcript, act)         # None == grounded
            if reason is not None:
                rejected += 1
                continue
            na = norm(act)
            if na in taken:                           # repeat of a SUCCESSFUL action
                rejected += 1
                continue
            if retry_block and na == retry_block:     # immediate retry of a failed action
                rejected += 1
                continue
            grounded_tools.append(act)
        else:
            rejected += 1

    log.append("critic: %d candidates -> %d grounded-tool, %d done, %d rejected"
               % (len(candidates), len(grounded_tools), len(dones), rejected))

    err = obs_is_error(last_obs)
    if err and dones and not grounded_tools:
        log.append("  note: only <done> survived but last obs is an error")
    # Stop ONLY on consensus and not immediately after an error.
    if dones and not err and len(dones) >= len(grounded_tools):
        choice = Counter(dones).most_common(1)[0][0]
        log.append("  -> commit <done> (consensus, no pending grounded action)")
        return choice, log
    if grounded_tools:
        choice = Counter(grounded_tools).most_common(1)[0][0]
        log.append("  -> commit majority grounded action")
        return choice, log
    if dones:
        choice = Counter(dones).most_common(1)[0][0]
        log.append("  -> commit <done> (no grounded action available)")
        return choice, log
    log.append("  -> no valid candidate; caller should fall back to greedy")
    return None, log


def select(candidates, transcript, taken=None, last_obs="", last_action=""):
    """Convenience: just the chosen action (or None)."""
    return critique(candidates, transcript, taken, last_obs, last_action)[0]
