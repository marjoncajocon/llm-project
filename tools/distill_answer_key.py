#!/usr/bin/env python3
"""Correct the self-distilled answers where the teacher is simply WRONG (B2, 2026-09-22).

Qwen2.5-Coder-0.5B-Instruct (the self-distillation teacher) gets simple facts and arithmetic wrong
("In one word, what color is a banana?" -> "Red"; "33 + 5" -> "48"; "Can penguins fly?" -> "Yes").
This was checked NOT to be an engine bug: an independent NumPy forward pass over the original HF
safetensors picks the same tokens (E:\\llm-test\\qwen_ref.py; "Red" 20.51 vs "Yellow" 20.14).
Distilling those answers would teach wrong facts, so for the prompt families that
tools/make_distill_prompts.py generates with a known answer, the answer is replaced:
  * arithmetic   -> computed here, phrased like the teacher ("33 + 5 equals 38.")
  * colours / one-word facts / yes-no -> a small answer key below
Everything else (how-to, code, definitions, rewrites, greetings, lists) keeps the teacher's own
answer. Rows are marked "source": "key" or "teacher".

    python tools/distill_answer_key.py E:/llm-test/data/distill/answers.jsonl E:/llm-test/data/distill/answers_fixed.jsonl
"""
import json
import re
import sys

COLORS = {"grass": "Green", "snow": "White", "a banana": "Yellow", "a ripe tomato": "Red",
          "coal": "Black", "the sun at noon": "Yellow", "an orange": "Orange", "a lemon": "Yellow",
          "milk": "White", "the ocean": "Blue", "a stop sign": "Red", "chocolate": "Brown",
          "a crow": "Black", "a flamingo": "Pink", "an emerald": "Green",
          "a cloud on a clear day": "White"}
KEY = {  # normalized prompt (lower-case, no trailing punctuation) -> answer
    "what is the opposite of hot? answer in one word": "Cold",
    "what is the opposite of up? one word only": "Down",
    "what is the plural of mouse? reply with one word": "Mice",
    "what sound does a dog make? one word": "Woof",
    "what sound does a cat make? answer with a single word": "Meow",
    "what is the capital of france? one word": "Paris",
    "what is the capital of japan? one word only": "Tokyo",
    "what is the largest planet in our solar system? one word": "Jupiter",
    "what gas do plants absorb from the air? answer briefly": "Carbon dioxide.",
    "how many legs does a spider have? answer with a number only": "8",
    "how many days are in a week? just the number": "7",
    "what is frozen water called? one word": "Ice",
    "which language is used to style web pages? one word": "CSS",
    "what keyword defines a function in python? one word": "def",
    "what is the file extension for c source files? answer only the extension": ".c",
    "is the earth round? answer yes or no": "Yes.",
    "is 7 a prime number? yes or no": "Yes.",
    "can penguins fly? answer yes or no": "No.",
    "is python a compiled language? yes or no, then one sentence":
        "No. Python is usually run by an interpreter (it compiles to bytecode first, but it is not "
        "compiled to machine code ahead of time like C).",
    "is 0 an even number? yes or no": "Yes.",
    "does water boil at 100 degrees celsius at sea level? yes or no": "Yes.",
    "is html a programming language? answer yes or no and explain in one sentence":
        "No. HTML is a markup language: it describes the structure of a web page but has no logic "
        "like loops or conditions.",
    "is a tomato a fruit? yes or no": "Yes.",
}
COLOR_RE = re.compile(r"^(?:what color is (.+?)\? answer in one word|what is the color of (.+?) in one word"
                      r"|in one word, what color is (.+?))$")
MATH_RE = re.compile(r"^what is (\d+) (\+|minus|times|divided by) (\d+)(\?.*)?$")


def norm(p):
    return p.strip().lower().rstrip("?.! ").replace("?  ", "? ")


def keyed(prompt):
    n = norm(prompt)
    m = MATH_RE.match(prompt.strip().lower().rstrip(".!"))
    if m:
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        if op == "+":
            return "%d + %d equals %d." % (a, b, a + b)
        if op == "minus":
            return "%d minus %d equals %d." % (a, b, a - b)
        if op == "times":
            return "%d times %d equals %d." % (a, b, a * b)
        v = ("%.4f" % (a / b)).rstrip("0").rstrip(".")
        return "%d divided by %d is %s." % (a, b, v)
    m = COLOR_RE.match(n)
    if m:
        obj = next(g for g in m.groups() if g)
        if obj in COLORS:
            return COLORS[obj]
    for k, v in KEY.items():
        if n == k.rstrip("?.! "):
            return v
    return None


def main():
    src, dst = sys.argv[1], sys.argv[2]
    n = fixed = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8", newline="\n") as out:
        for line in f:
            r = json.loads(line)
            n += 1
            k = keyed(r["prompt"])
            if k is not None:
                if r["answer"].strip() != k:
                    fixed += 1
                r["teacher_answer"], r["answer"], r["status"], r["source"] = r["answer"], k, "ok", "key"
            else:
                r["source"] = "teacher"
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("%d answers, %d replaced by the answer key (%d of them differed from the teacher) -> %s"
          % (n, sum(1 for l in open(dst, encoding="utf-8") if '"source": "key"' in l), fixed, dst))


if __name__ == "__main__":
    main()
