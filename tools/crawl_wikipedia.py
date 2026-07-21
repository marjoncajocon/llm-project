#!/usr/bin/env python3
"""Crawl connected Wikipedia articles into plain-text files for training.

Uses the Wikipedia API (no HTML scraping): fetches each article's plain-text
extract, follows its links breadth-first to gather *connected* articles, and
saves each as data/wiki/<title>.txt. Then train on the whole folder:

    python tools/crawl_wikipedia.py "Machine learning" --max-pages 50
    transformer train data/wiki model.bin tok.bin 400 3000
    echo "machine learning is" | transformer generate model.bin tok.bin

Only the Python standard library is used (urllib). Wikipedia requires a
descriptive User-Agent and polite rate limiting, both set below.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import deque

API = "https://en.wikipedia.org/w/api.php"
UA = "llm-project-crawler/1.0 (educational; local training corpus)"


def api_get(params):
    params = dict(params, format="json", action="query")
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def get_extract(title):
    """Return the plain-text extract of an article (or '')."""
    data = api_get({
        "prop": "extracts", "explaintext": "1", "redirects": "1", "titles": title,
    })
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        if "extract" in page:
            return page["title"], page["extract"]
    return title, ""


def get_links(title, limit=200):
    """Return outgoing article links (namespace 0) from an article."""
    data = api_get({
        "prop": "links", "plnamespace": "0", "pllimit": str(limit),
        "redirects": "1", "titles": title,
    })
    pages = data.get("query", {}).get("pages", {})
    out = []
    for _, page in pages.items():
        for lk in page.get("links", []):
            out.append(lk["title"])
    return out


def safe_name(title):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", title)[:120] + ".txt"


def main():
    ap = argparse.ArgumentParser(description="Crawl connected Wikipedia articles to text files")
    ap.add_argument("seed", help="starting article title, e.g. \"Machine learning\"")
    ap.add_argument("--max-pages", type=int, default=25, help="how many articles to fetch")
    ap.add_argument("--out", default="data/wiki", help="output directory")
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between requests (be polite)")
    ap.add_argument("--min-chars", type=int, default=200, help="skip stubs shorter than this")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seen = set()
    queue = deque([args.seed])
    fetched = 0

    while queue and fetched < args.max_pages:
        title = queue.popleft()
        if title in seen:
            continue
        seen.add(title)
        try:
            real_title, extract = get_extract(title)
        except Exception as e:
            print(f"  skip {title!r}: {e}", file=sys.stderr)
            time.sleep(args.delay)
            continue

        if len(extract) >= args.min_chars:
            path = os.path.join(args.out, safe_name(real_title))
            with open(path, "w", encoding="utf-8") as f:
                f.write(extract)
            fetched += 1
            print(f"[{fetched}/{args.max_pages}] {real_title}  ({len(extract)} chars)")

            # Enqueue connected articles (breadth-first) to keep the text related.
            if fetched < args.max_pages:
                try:
                    for lk in get_links(real_title):
                        if lk not in seen:
                            queue.append(lk)
                except Exception:
                    pass
        time.sleep(args.delay)

    total = sum(os.path.getsize(os.path.join(args.out, f))
                for f in os.listdir(args.out) if f.endswith(".txt"))
    print(f"done: {fetched} articles, ~{total} bytes in {args.out}/")


if __name__ == "__main__":
    main()
