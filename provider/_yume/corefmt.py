"""Core-file (MEMORY.md / USER.md) format + entry hashing — the ONLY copy. Stdlib only, no
relative imports. Provider imports it (``from . import corefmt``); the dream loads it by path
(``hermesyume.paths.load_provider_module("corefmt")``), so ``core_sha`` can never drift.

Format (Hermes tools/memory_tool.py): entries joined by "\\n§\\n", read as utf-8-sig.
"""

import hashlib
import re
import unicodedata

ENTRY_DELIMITER = "\n§\n"
EPISODIC_RE = re.compile(r"^Session: \d{4}-\d{2}-\d{2}")
_LABEL_RE = re.compile(r"^\s*(\*\*[^*\n]{1,80}\*\*)")
_WS_RE = re.compile(r"\s+")


def parse_entries(raw):
    """Split on the FULL delimiter (a bare '§' inside an entry survives); strip; drop empties."""
    return [e for e in (x.strip() for x in (raw or "").split(ENTRY_DELIMITER)) if e]


def read_entries(path):
    """Entries of a core file. Missing file → []. Undecodable file raises (never mistaken for
    empty). Opens read-only; creates no lock file."""
    try:
        with open(str(path), "r", encoding="utf-8-sig") as f:
            return parse_entries(f.read())
    except FileNotFoundError:
        return []


def core_norm(text):
    """NFKC, whitespace runs → one space, stripped. Case is preserved."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", text or "")).strip()


def core_sha(text):
    """sha1 hex of core_norm(entry) — memories.core_sha / serving items.core_sha / core_seen."""
    return hashlib.sha1(core_norm(text).encode("utf-8")).hexdigest()


def entry_label(text):
    """Leading bold label like '**가계부 관리:**', or None."""
    m = _LABEL_RE.match(text or "")
    return m.group(1) if m else None


def is_episodic(text):
    """'Session: 2026-06-27 …' style entries are episodes, not facts (R1 core_add, M3)."""
    return bool(EPISODIC_RE.match((text or "").lstrip()))


def gram_norm(text):
    """For shingling: NFKC, casefold, all whitespace removed."""
    return _WS_RE.sub("", unicodedata.normalize("NFKC", text or "").casefold())


def trigrams(text):
    t = gram_norm(text)
    if len(t) < 3:
        return {t} if t else set()
    return {t[i:i + 3] for i in range(len(t) - 2)}


def containment(needle, haystack):
    """Fraction of `needle`'s 3-grams present in `haystack` (0..1). Empty needle → 0."""
    a = trigrams(needle)
    if not a:
        return 0.0
    return len(a & trigrams(haystack)) / float(len(a))
