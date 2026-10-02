"""`used` judgment for injected memories (PLAN-v2 §6.3; stdlib only).

Pairing: prefetch sees the scaffold-stripped query, sync_turn the user content, so turns are paired
by "normalized query ⊆ normalized user content" (most recent first), not by hash. Older unmatched
pending turns (interrupted turns) are dropped.

Judgment: a memory is `used` when its feature tokens (≥2-digit numbers, ≥3-char latin words,
Hangul words of ≥2 syllables minus particles) appear in the response with ratio ≥ `ratio` and at
least `min_tokens` matches, or when a single number/ID token appears verbatim.
"""

import re
import unicodedata

from . import textutil

_WS = re.compile(r"\s+")
_LATIN = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]*$")
_DIGITS = re.compile(r"^\d+$")
_ID_LIKE = re.compile(r"^(?=.*\d)(?=.*[A-Za-z])[A-Za-z0-9_\-]{4,}$")
_YEAR = re.compile(r"^(19|20)\d\d$")


def norm(text):
    """NFKC, casefold, whitespace collapsed, stripped."""
    t = unicodedata.normalize("NFKC", text or "").casefold()
    return _WS.sub(" ", t).strip()


def feature_tokens(text):
    out = set()
    for t in textutil.raw_tokens(text):
        if _DIGITS.match(t):
            if len(t) >= 2:
                out.add(t)
            continue
        if _ID_LIKE.match(t):
            out.add(t.casefold())
            continue
        if _LATIN.match(t):
            if len(t) >= 3 and t.casefold() not in textutil.STOPWORDS:
                out.add(t.casefold())
            continue
        if textutil._HANGUL_RE.match(t):
            s = textutil.strip_particle(t)
            if len(s) >= 2 and s not in textutil.STOPWORDS:
                out.add(s)
    return out


def _strong(tok):
    """A single verbatim occurrence is enough: numbers ≥3 digits (not a bare year) or ID-like."""
    if _DIGITS.match(tok):
        return len(tok) >= 3 and not _YEAR.match(tok)
    return bool(_ID_LIKE.match(tok))


def _present(tok, resp):
    if _DIGITS.match(tok):
        return re.search(r"(?<!\d)%s(?!\d)" % re.escape(tok), resp) is not None
    return tok in resp


def is_used(memory_text, response, *, ratio, min_tokens):
    toks = feature_tokens(memory_text)
    if not toks or not response:
        return False
    resp = norm(response)
    hits = [t for t in toks if _present(t, resp)]
    if any(_strong(t) for t in hits):
        return True
    return len(hits) >= int(min_tokens) and len(hits) / float(len(toks)) >= float(ratio)


def _query_of(entry):
    if isinstance(entry, dict):
        return entry.get("query", "")
    return getattr(entry, "query", "")


def match_pending(pending, user_content):
    """(matched turn, older unmatched turns to drop). `pending`: {turn_no: entry with .query/["query"]}."""
    uc = norm(user_content)
    matched = None
    for turn in sorted(pending, reverse=True):
        q = norm(_query_of(pending[turn]))
        if q and q in uc:
            matched = turn
            break
    if matched is None:
        return None, []
    dropped = [t for t in sorted(pending) if t < matched]
    return matched, dropped
