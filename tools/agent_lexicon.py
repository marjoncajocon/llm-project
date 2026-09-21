#!/usr/bin/env python3
"""Deterministic TRAIN / EVAL split of the generators' lexicon (--holdout-lexicon).

Before this, the eval task files were drawn from the SAME templates and the SAME
word pool as the training corpus, only with a different seed -- so an "eval" score
measured recall of seen (filename, content, command, phrasing) combinations, not
generalization. This module assigns every filename slug, content phrase, command
and goal phrasing to exactly one side, by a salted hash of its normalized text:

  * the assignment depends on the ITEM ONLY, not on pool order, pool size or the
    RNG seed -- so train and eval sets generated months apart, from different
    wiki crawls, still agree on which words are off-limits;
  * content phrases are keyed the same way as the slug they come from
    ("computer_graphics" and "computer graphics" share one key), so a slug held
    out as a filename is never leaked back in as write content.

    side_of("computer_graphics")          -> "train" | "eval"
    filter_side(pool, "eval", frac=0.2)   -> the eval-only subset

Nothing here draws from an RNG, so importing it never perturbs a generator's
random stream (default outputs stay byte-identical).
"""
import hashlib
import re

LEX_SALT = "holdout-lexicon-v1"
PHRASE_SALT = "holdout-phrasing-v1"
DEFAULT_FRAC = 0.2          # share of the lexicon reserved for EVAL
DEFAULT_PHRASE_FRAC = 0.3   # share of each phrasing pool reserved for EVAL


def lex_key(s):
    """Normalization under which two spellings are the SAME lexical item."""
    return re.sub(r"[\s_]+", " ", s.lower()).strip()


def _u(key, salt):
    h = hashlib.md5((salt + "\0" + key).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / float(1 << 32)


def side_of(item, frac=DEFAULT_FRAC, salt=LEX_SALT):
    return "eval" if _u(lex_key(item), salt) < frac else "train"


def filter_side(pool, side, frac=DEFAULT_FRAC, salt=LEX_SALT):
    """Order-preserving subset of `pool` on `side` ("train"|"eval"); None = all."""
    if side is None:
        return list(pool)
    return [x for x in pool if side_of(x, frac, salt) == side]


def split_phrasings(pool, side, frac=DEFAULT_PHRASE_FRAC, salt=PHRASE_SALT, key=None):
    """Phrasing-pool split that never leaves a side EMPTY.

    The empty string (a "no opener" / "no closer" slot) is neutral and kept on
    both sides. If the hash puts every real alternative on one side, the item
    with the smallest (resp. largest) hash is moved, deterministically.
    `key(x)` (optional) maps an item to the text that decides its side, so two
    near-identical templates in DIFFERENT pools can be forced onto one side."""
    if side is None:
        return list(pool)
    real = [x for x in dict.fromkeys(pool) if x != ""]
    if len(real) < 2:
        return list(pool)
    us = {x: _u(lex_key(key(x) if key else x), salt) for x in real}
    ev = [x for x in real if us[x] < frac]
    tr = [x for x in real if us[x] >= frac]
    if not ev:
        m = min(real, key=lambda x: us[x])
        ev, tr = [m], [x for x in tr if x != m]
    if not tr:
        m = max(real, key=lambda x: us[x])
        tr, ev = [m], [x for x in ev if x != m]
    keep = set(ev if side == "eval" else tr)
    return [x for x in pool if x == "" or x in keep]


def apply_holdout(files, contents, cmds, side, frac=DEFAULT_FRAC):
    """Restrict the three generator pools to one side of the split.

    `contents` is filtered by the same key as `files`, so the phrase form of a
    held-out slug is held out too. Refuses to return an empty pool."""
    f = filter_side(files, side, frac)
    c = filter_side(contents, side, frac)
    m = filter_side(cmds, side, frac)
    for name, pool in (("files", f), ("contents", c), ("commands", m)):
        if len(pool) < 3:
            raise SystemExit("holdout-lexicon: %s pool for side %r has only %d item(s); "
                             "give a bigger --wiki/--commands or a larger --holdout-frac"
                             % (name, side, len(pool)))
    return f, c, m
