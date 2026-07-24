#!/usr/bin/env python3
"""Crawl Stack Overflow Q&A into plain-text files (a code/prose corpus).

Uses the public Stack Exchange API (no key needed at low volume; ~300 requests/
day/IP). For each tag it fetches the top-voted questions WITH their bodies, then
one batched call for all those questions' answers, and writes each Q&A as
data/stackoverflow/<tag>_<id>.txt (question title + text + top answers).

    python tools/crawl_stackoverflow.py git bash --pages 2 --pagesize 50

Honest scope: this is a CORPUS (code + explanation), not ready-made agent
tool-call pairs. Two uses: (1) generalization/pretraining text like the Wikipedia
crawl; (2) `--commands OUT.txt` also mines real shell commands from the answers
(one per line), which tools/make_sft.py can fold into its run_command pool so the
agent SFT data uses real git/npm/docker/... commands instead of a tiny hardcoded
list.

Only the Python standard library is used. The API always gzips responses and may
return a `backoff` field that MUST be honoured — both handled below.
"""
import argparse
import gzip
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.stackexchange.com/2.3"
UA = "llm-project-crawler/1.0 (educational; local training corpus)"
_next_ok = [0.0]   # earliest wall-clock time the next request may run (backoff)


def api_get(path, params, tries=6):
    """GET an SE API endpoint. Handles gzip, the `backoff` field, and 429/5xx."""
    params = dict(params, site="stackoverflow")
    url = "%s/%s?%s" % (API, path, urllib.parse.urlencode(params))
    delay = 2.0
    for attempt in range(tries):
        wait = _next_ok[0] - time.time()
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": UA,
                                                   "Accept-Encoding": "gzip"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
            if raw[:2] == b"\x1f\x8b":         # gzip magic — SE always gzips
                raw = gzip.decompress(raw)
            data = json.loads(raw.decode("utf-8"))
            # The API asks callers to wait `backoff` seconds before the NEXT hit.
            if "backoff" in data:
                _next_ok[0] = time.time() + float(data["backoff"]) + 0.5
            return data
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
            print("  [%d] backing off %.0fs" % (e.code, delay), flush=True)
            time.sleep(delay)
            delay = min(delay * 2.0, 60.0)
        except urllib.error.URLError:
            if attempt == tries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 60.0)
    raise RuntimeError("unreachable")


CODE_RE = re.compile(r"<(code|pre)[^>]*>(.*?)</\1>", re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


def code_blocks(body_html):
    """Raw text of every <code>/<pre> block (entities unescaped)."""
    return [html.unescape(m.group(2)) for m in CODE_RE.finditer(body_html or "")]


def to_text(body_html):
    """Crude HTML -> text: block tags become newlines, others drop, then unescape."""
    s = re.sub(r"(?i)</(p|div|pre|li|h[1-6])>|<br\s*/?>", "\n", body_html or "")
    s = TAG_RE.sub("", s)
    s = html.unescape(s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


# A "command" line: starts with a known CLI, no shell prompt / pipeline noise.
CLIS = ("git", "npm", "npx", "yarn", "pnpm", "pip", "pip3", "python", "python3",
        "docker", "kubectl", "make", "cmake", "cargo", "go", "node", "curl",
        "wget", "ssh", "scp", "tar", "unzip", "grep", "sed", "awk", "find", "ls",
        "cd", "mkdir", "rm", "cp", "mv", "cat", "chmod", "chown", "apt", "apt-get",
        "brew", "yum", "dnf", "systemctl", "ps", "kill", "mvn", "gradle", "dotnet")
CMD_RE = re.compile(r"^[\s$#>]*((?:sudo\s+)?(%s)\b[^\n`]*)$" % "|".join(map(re.escape, CLIS)))


def mine_commands(code_texts):
    out = []
    for block in code_texts:
        for line in block.splitlines():
            line = line.strip()
            m = CMD_RE.match(line)
            if not m:
                continue
            cmd = m.group(1).strip()
            # Drop a trailing inline comment ("git add .   # stage all" -> "git add .").
            cmd = re.split(r"\s+#", cmd, 1)[0].strip()
            # Reject placeholders (<branch>, {name}, [remote]) — not literally runnable.
            if re.search(r"[<>{}\[\]]", cmd):
                continue
            # Reject quotes/backticks: the SFT response puts the command inside a
            # command="..." attribute, and an embedded " would break the parser.
            if '"' in cmd or "'" in cmd or "`" in cmd:
                continue
            # Reject log/error OUTPUT that merely starts with a CLI name
            # (e.g. "npm ERR! ...", "python Traceback ...").
            if "!" in cmd or re.search(r"\b(ERR|ERROR|WARN|WARNING|INFO|DEBUG|"
                                       r"FATAL|Traceback|Exception)\b", cmd):
                continue
            if 3 <= len(cmd) <= 60 and "..." not in cmd:
                out.append(cmd)
    return out


def safe_name(tag, qid):
    return "%s_%d.txt" % (re.sub(r"[^A-Za-z0-9._-]+", "_", tag), qid)


def main():
    ap = argparse.ArgumentParser(description="Crawl Stack Overflow Q&A to text files")
    ap.add_argument("tags", nargs="+", help="tags to crawl, e.g. git bash python")
    ap.add_argument("--pages", type=int, default=1, help="pages of questions per tag")
    ap.add_argument("--pagesize", type=int, default=50, help="questions per page (<=100)")
    ap.add_argument("--out", default="data/stackoverflow", help="output directory")
    ap.add_argument("--commands", default=None, help="also write mined shell commands here")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n_files = 0
    commands = []

    for tag in args.tags:
        for page in range(1, args.pages + 1):
            q = api_get("questions", {
                "order": "desc", "sort": "votes", "tagged": tag,
                "pagesize": str(args.pagesize), "page": str(page), "filter": "withbody",
            })
            items = q.get("items", [])
            if not items:
                break
            ids = [it["question_id"] for it in items]
            # ONE batched call for every answer of every question on this page.
            a = api_get("questions/%s/answers" % ";".join(map(str, ids)),
                        {"order": "desc", "sort": "votes", "pagesize": "100",
                         "filter": "withbody"})
            ans_by_q = {}
            for it in a.get("items", []):
                ans_by_q.setdefault(it["question_id"], []).append(it)

            for it in items:
                qid = it["question_id"]
                title = it.get("title", "")
                parts = ["# " + html.unescape(title), "", to_text(it.get("body", ""))]
                blocks = code_blocks(it.get("body", ""))
                for ans in ans_by_q.get(qid, [])[:2]:      # top 2 answers
                    parts += ["", "## Answer", to_text(ans.get("body", ""))]
                    blocks += code_blocks(ans.get("body", ""))
                with open(os.path.join(args.out, safe_name(tag, qid)), "w",
                          encoding="utf-8", newline="\n") as f:
                    f.write("\n".join(parts) + "\n")
                n_files += 1
                if args.commands:
                    commands += mine_commands(blocks)

            print("  [%s p%d] %d questions | quota left %s" %
                  (tag, page, len(items), q.get("quota_remaining", "?")), flush=True)
            if not q.get("has_more"):
                break

    print("done: %d Q&A files in %s/" % (n_files, args.out))
    if args.commands:
        uniq = sorted(set(commands))
        with open(args.commands, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(uniq) + "\n")
        print("mined %d distinct shell commands -> %s" % (len(uniq), args.commands))


main()
