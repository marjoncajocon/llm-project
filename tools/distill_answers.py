#!/usr/bin/env python3
"""Self-distillation replay data for B2 (2026-09-22). See tools/make_distill_prompts.py for why.

Step 1 -- answers: the ORIGINAL model answers every prompt in its OWN chat format (ChatML rendered
exactly like hftok_chat_render does for Qwen, incl. the default system line), near-greedy
(temp 0.2) through the pure-C engine (`transformer hfgen`; <|im_end|> is the EOS, so generation
stops at the end of the turn). Resumable: answers are appended to a JSONL and skipped on rerun.
Answers that hit the token cap (unfinished), are empty, or degenerate (a chunk repeated 6+ times)
are dropped -- training on them would teach exactly the failure we are trying to avoid.

    python tools/distill_answers.py answer --engine E:/llm-test/transformer_v7.exe \\
        --model E:/llm-test/model/qwen25c/model.bin --tok E:/llm-test/model/qwen25c/tok.hf \\
        --prompts E:/llm-test/data/distill/prompts.txt --out E:/llm-test/data/distill/answers.jsonl

Step 2 -- build: an SFT file for `llama-train --sft --sft-escapes`: every kept answer twice (as a
ChatML-rendered prompt, so the chat template keeps working, and as the raw prompt, so the agent's
raw mode answers general questions in prose instead of inventing a tool call), plus the agent rows.
All rows are escaped (backslash -> \\\\, newline -> \\n, tab -> \\t), including the agent rows, so a
literal backslash in them survives --sft-escapes unchanged.

    python tools/distill_answers.py build --answers E:/llm-test/data/distill/answers.jsonl \\
        --agent E:/llm-test/data/v2/sft_mix.txt --out E:/llm-test/data/distill/sft_b2.txt
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys

SYSTEM = "You are a helpful assistant."          # tools/export_tokenizer.py DEFAULT_SYSTEM (ChatML)
LOOP = re.compile(r"(.{2,40}?)\1{5,}", re.S)


def chatml(question):
    return ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n"
            "<|im_start|>assistant\n" % (SYSTEM, question))


def degenerate(text):
    for m in LOOP.finditer(text):
        if re.search(r"[A-Za-z0-9]", m.group(1)):
            return True
    return False


def generate(engine, model, tok, prompt, n_new, temp):
    p = subprocess.run([engine, "hfgen", model, tok, str(n_new), str(temp)],
                       input=prompt.encode("utf-8"), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=900)
    pieces = []
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line.startswith('{"d"'):
            try:
                pieces.append(json.loads(line)["d"])
            except (ValueError, KeyError):
                pass
    return "".join(pieces), len(pieces)


def cmd_answer(a):
    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["prompt"])
                except (ValueError, KeyError):
                    pass
    prompts = [p.strip() for p in open(a.prompts, encoding="utf-8") if p.strip()]
    todo = [p for p in prompts if p not in done]
    print("%d prompts, %d already answered, %d to go" % (len(prompts), len(done), len(todo)), flush=True)
    with open(a.out, "a", encoding="utf-8", newline="\n") as out:
        for i, q in enumerate(todo, 1):
            text, ntok = generate(a.engine, a.model, a.tok, chatml(q), a.n_new, a.temp)
            text = text.replace("<|im_end|>", "").strip()
            status = ("empty" if not text else "capped" if ntok >= a.n_new
                      else "loop" if degenerate(text) else "ok")
            out.write(json.dumps({"prompt": q, "answer": text, "tokens": ntok, "status": status},
                                 ensure_ascii=False) + "\n")
            out.flush()
            print("[%d/%d] %-6s %3d tok  %s" % (i, len(todo), status, ntok, q[:60]), flush=True)


def esc(s):
    return s.replace("\\", "\\\\").replace("\r", "").replace("\t", "\\t").replace("\n", "\\n")


def cmd_build(a):
    rows, kept, dropped = [], 0, {}
    with open(a.answers, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["status"] != "ok":
                dropped[r["status"]] = dropped.get(r["status"], 0) + 1
                continue
            kept += 1
            for _ in range(a.distill_repeat):   # up-weight the small replay set vs the agent rows
                rows.append(esc(chatml(r["prompt"])) + "\t" + esc(r["answer"]))
                rows.append(esc(r["prompt"]) + "\t" + esc(r["answer"]))
    n_distill = len(rows)
    with open(a.agent, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if "\t" not in line:
                continue
            p, r = line.split("\t", 1)
            rows.append(esc(p) + "\t" + esc(r))
    random.Random(a.seed).shuffle(rows)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rows) + "\n")
    print("kept %d answers (dropped %s) -> %d distill rows + %d agent rows = %d rows -> %s"
          % (kept, dropped or "none", n_distill, len(rows) - n_distill, len(rows), a.out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("answer")
    s.add_argument("--engine", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--tok", required=True)
    s.add_argument("--prompts", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-new", type=int, default=384)
    s.add_argument("--temp", type=float, default=0.2)
    b = sub.add_parser("build")
    b.add_argument("--answers", required=True)
    b.add_argument("--agent", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--distill-repeat", type=int, default=1,
                   help="copies of each kept answer (x2 rows each: chat + raw)")
    a = ap.parse_args()
    return cmd_answer(a) if a.cmd == "answer" else cmd_build(a)


if __name__ == "__main__":
    sys.exit(main())
