#!/usr/bin/env python3
"""Varied general instructions for SELF-DISTILLATION REPLAY (B2, 2026-09-22).

Agent SFT on Qwen2.5-Coder-0.5B-Instruct (B1) made it the best agent but it FORGOT part of its
instruction following ("the colour of the sky in one word" -> "usually gray ... black"). The fix:
let the ORIGINAL Qwen answer a few hundred varied instructions and train on those answers next to
the agent rows, so fine-tuning keeps its general behaviour (tools/distill_answers.py does that).

This file only writes the PROMPTS, one per line. Categories: one-word / constrained answers,
yes-no, lists, how-to, small code tasks (C / Python / JS / Go), definitions, arithmetic, rewrites,
greetings. The user's regression prompts (E:\\llm-release\\test_prompts.txt) are EXCLUDED on purpose
so they stay honest tests.

    python tools/make_distill_prompts.py --out E:/llm-test/data/distill/prompts.txt
"""
import argparse
import random

EXCLUDE = {  # the user's regression prompts -- never train on them
    "what year of the animal is 1996?",
    "what is the color of the sky in one word",
    "what is a compiler",
    "can you create me a print star pattern using c",
    "open notes.md",
    "save hello into greet.txt",
}

ONE_WORD_OBJECTS = ["grass", "snow", "a banana", "a ripe tomato", "coal", "the sun at noon",
                    "an orange", "a lemon", "milk", "the ocean", "a stop sign", "chocolate",
                    "a crow", "a flamingo", "an emerald", "a cloud on a clear day"]
ONE_WORD_Q = ["What is the opposite of hot? Answer in one word.",
              "What is the opposite of up? One word only.",
              "What is the plural of mouse? Reply with one word.",
              "What sound does a dog make? One word.",
              "What sound does a cat make? Answer with a single word.",
              "What is the capital of France? One word.",
              "What is the capital of Japan? One word only.",
              "What is the largest planet in our solar system? One word.",
              "What gas do plants absorb from the air? Answer briefly.",
              "How many legs does a spider have? Answer with a number only.",
              "How many days are in a week? Just the number.",
              "What is frozen water called? One word.",
              "Which language is used to style web pages? One word.",
              "What keyword defines a function in Python? One word.",
              "What is the file extension for C source files? Answer only the extension."]
YES_NO = ["Is the Earth round? Answer yes or no.", "Is 7 a prime number? Yes or no.",
          "Can penguins fly? Answer yes or no.", "Is Python a compiled language? Yes or no, then one sentence.",
          "Is 0 an even number? Yes or no.", "Does water boil at 100 degrees Celsius at sea level? Yes or no.",
          "Is HTML a programming language? Answer yes or no and explain in one sentence.",
          "Is a tomato a fruit? Yes or no.", "Can a C function return an array directly? Yes or no, then explain briefly."]
LIST_Q = ["List 3 fruits.", "Name 5 programming languages.", "List three primary colors.",
          "Give me 4 common Linux commands and what they do, one line each.",
          "List 3 benefits of version control.", "Name three sorting algorithms.",
          "List the days of the week.", "Give three tips for writing readable code.",
          "List 5 HTTP status codes and their meaning.", "Name 3 data structures and one use for each."]
HOWTO = ["How do I undo my last git commit but keep the changes?",
         "How do I list all files, including hidden ones, in a Linux terminal?",
         "How do I create a new Python virtual environment?",
         "How do I check which process is using port 8080 on Linux?",
         "How do I read a file line by line in Python?",
         "How do I compile a C program with gcc?", "How do I rename a git branch?",
         "How do I find a word in all files of a folder with grep?",
         "How do I make a file executable on Linux?", "How can I reverse a list in Python?",
         "How do I print to the console in JavaScript?", "How do I install a package with pip?",
         "What is the difference between git merge and git rebase? Keep it short.",
         "How do I delete a local git branch?", "How do I count lines in a file on Linux?"]
CODE_TASKS = [("C", "reverses a string in place"), ("C", "computes the factorial of a number"),
              ("C", "checks whether a number is prime"), ("C", "prints the Fibonacci numbers below 100"),
              ("C", "prints a right triangle of # characters with 4 rows"),
              ("C", "sums the elements of an integer array"), ("C", "swaps two integers using pointers"),
              ("Python", "counts the vowels in a string"), ("Python", "prints FizzBuzz from 1 to 20"),
              ("Python", "reads a text file and prints the number of words"),
              ("Python", "finds the largest number in a list"), ("Python", "checks if a word is a palindrome"),
              ("Python", "prints a pyramid of stars with 5 rows"), ("Python", "removes duplicates from a list"),
              ("JavaScript", "sums an array of numbers"), ("JavaScript", "capitalizes the first letter of each word"),
              ("JavaScript", "checks if a number is even"), ("Go", "prints hello world"),
              ("Go", "reverses a slice of integers"), ("Go", "counts words in a string")]
CODE_FORMS = ["Write a {lang} program that {task}.", "Write a {lang} function that {task}.",
              "Can you write {lang} code that {task}?", "Show me a short {lang} example that {task}."]
DEFINE = ["What is a variable in programming?", "What is recursion?", "What is an API?",
          "Explain what a pointer is in C.", "What is a linked list?", "What does RAM stand for and do?",
          "What is an operating system?", "Explain what a loop is.", "What is JSON?",
          "What is the difference between a list and a tuple in Python?", "What is a hash table?",
          "What is Git?", "Explain what an HTTP request is.", "What is a CPU?", "What is an algorithm?",
          "What is machine learning, in two sentences?", "What is a function in math?",
          "What is binary code?", "Explain what a database is.", "What is a bug in software?"]
MATH = ["What is {a} + {b}?", "What is {a} times {b}?", "What is {a} minus {b}?",
        "What is {a} divided by {b}? Give the answer as a decimal."]
REWRITE = ["Rewrite this sentence in uppercase: {s}", "Make this sentence more formal: {s}",
           "Summarize in five words: {s}", "Translate to Spanish: {s}", "Fix the grammar: {s}"]
SENTENCES = ["the cat sat on the mat", "i likes to write code every day",
             "we need to finish the report before friday", "this program runs very slow on my computer",
             "please send me the file tomorrow", "he go to school by bus"]
GREETINGS = ["Hello!", "Hi, how are you?", "Good morning!", "Thanks for your help.",
             "Can you help me with something?", "What can you do?", "Tell me a short joke.",
             "Say something encouraging about learning to code."]


def build(seed):
    rng = random.Random(seed)
    out = []
    for obj in ONE_WORD_OBJECTS:
        out.append(rng.choice(["What color is {o}? Answer in one word.",
                               "What is the color of {o} in one word?",
                               "In one word, what color is {o}?"]).format(o=obj))
    out += ONE_WORD_Q + YES_NO + LIST_Q + HOWTO + DEFINE + GREETINGS
    for lang, task in CODE_TASKS:
        for form in rng.sample(CODE_FORMS, 2):
            out.append(form.format(lang=lang, task=task))
    for _ in range(40):
        a, b = rng.randint(2, 99), rng.randint(2, 30)
        out.append(rng.choice(MATH).format(a=a, b=b))
    for tmpl in REWRITE:
        for s in rng.sample(SENTENCES, 4):
            out.append(tmpl.format(s=s))
    # light surface variation: some lower-case / no final punctuation, like real users type
    varied = []
    for p in out:
        r = rng.random()
        if r < 0.2:
            p = p.lower()
        elif r < 0.3:
            p = p.rstrip("?.!")
        varied.append(p)
    seen, final = set(), []
    for p in varied:
        k = p.strip().lower()
        if k in seen or k.rstrip("?.! ") in {e.rstrip("?.! ") for e in EXCLUDE}:
            continue
        seen.add(k)
        final.append(p.strip())
    rng.shuffle(final)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    prompts = build(a.seed)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(prompts) + "\n")
    print("wrote %d prompts -> %s" % (len(prompts), a.out))


if __name__ == "__main__":
    main()
