"""cli_inputs.py OUTDIR - write the stdin payloads used by tests/cli_cmp.sh.

hfchat payloads are built exactly the way E:\\llm-release\\ai.py builds them:
"\\x1e".join("%s\\x1f%s" % (role, content) for role, content in turns), UTF-8.
"""
import os
import sys

out = sys.argv[1]
os.makedirs(out, exist_ok=True)


def chat(turns):
    return "\x1e".join("%s\x1f%s" % (r, c) for r, c in turns).encode("utf-8")


cases = {
    # --- hfchat ---
    "chat_single": chat([("user", "what is the color of the sky in one word")]),
    "chat_system": chat([("system", "You are a pirate. Answer in one short sentence."),
                         ("user", "Where is the treasure?")]),
    "chat_multi": chat([("user", "what year of the animal is 1996?"),
                        ("assistant", "1996 was the Year of the Rat."),
                        ("user", "and 1997?")]),
    "chat_multi_sys": chat([("system", "Be brief."), ("user", "hi"), ("assistant", "Hello!"),
                            ("user", "what is a compiler")]),
    "chat_unicode": chat([("user", "Translate to Japanese: good morning \u2600\ufe0f \u00e9t\u00e9 \u4f60\u597d")]),
    "chat_code": chat([("user", "can you create me a print star pattern using c")]),
    "chat_empty": b"",
    "chat_nosep": b"just some text without separators",
    "chat_role_only": b"user\x1e\x1euser\x1fhello\x1e",
    "chat_odd_roles": chat([("tool", "result: 42"), ("user", "what was the result?")]),
    "chat_crlf": b"user\x1fline one\r\nline two\r\nwhat did I say?",
    "chat_nul": b"user\x1fhello\x00 ignored after NUL",
    "chat_ctrlz": b"user\x1fsay yes\x1a junk",
    # > 64 KB: C reads only 65535 bytes (cheap to tokenize: long space runs)
    "chat_big": chat([("user", ("x" + " " * 60) * 1200 + "END")]),
    # --- hfgen (agent-format prompts for --stop-tags) ---
    "gen_open": "open notes.md".encode(),
    "gen_save": "save hello into greet.txt".encode(),
    "gen_tool_text": ("Reply with a tool call.\nUser: read the file a.txt\nAssistant: "
                      "<tool name=\"read_file\" path=\"").encode(),
    "gen_france": b"The capital of France is",
    "gen_qwen_chat": b"<|im_start|>user\nWrite a haiku about the sea.<|im_end|>\n<|im_start|>assistant\n",
}
for k, v in cases.items():
    with open(os.path.join(out, k + ".in"), "wb") as f:
        f.write(v)
print("wrote %d inputs to %s" % (len(cases), out))
