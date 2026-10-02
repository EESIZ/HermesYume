"""Vector math and deterministic text guards for REM (PLAN-v2 §4.3 R0-4, R0 different_aspects).

- ``cos`` / ``cos_matrix`` / ``top_k``: numpy cosine (inputs need not be normalized)
- ``numeric_tokens`` + ``has_negation`` + ``same_family`` → ``auto_dup_eligible`` (auto duplicate
  only when numbers/dates and negation agree; "12" vs "13" is never auto-dup, T7)
- ``fact_tokens`` + ``preservation_check``: consolidation must keep every number, date, latin
  token, quoted string and ≥2-syllable Hangul word (particle stripped) of both inputs (T10)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np

from .types import KIND_FAMILY, norm_text

# ── vector math ──────────────────────────────────────────────────────────────


def _as_vec(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float32).reshape(-1)


def cos(a: Any, b: Any) -> float:
    va, vb = _as_vec(a), _as_vec(b)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def cos_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(n, d) × (m, d) → (n, m) cosine matrix."""
    A = _normalize_rows(A)
    B = _normalize_rows(B)
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    return A @ B.T


def top_k(q: np.ndarray, M: np.ndarray, k: int, min_cos: float = -1.0) -> list[tuple[int, float]]:
    """Indices of the k rows of M most similar to q (cos ≥ min_cos), best first."""
    M = np.asarray(M, dtype=np.float32)
    if M.ndim != 2 or M.shape[0] == 0 or k <= 0:
        return []
    sims = cos_matrix(_as_vec(q), M)[0]
    k = min(int(k), sims.shape[0])
    idx = np.argpartition(-sims, k - 1)[:k] if k < sims.shape[0] else np.arange(sims.shape[0])
    idx = sorted(idx.tolist(), key=lambda i: (-float(sims[i]), i))
    return [(int(i), float(sims[i])) for i in idx if float(sims[i]) >= min_cos]


# ── numeric / date tokens ────────────────────────────────────────────────────

_DATE_YMD_RE = re.compile(r"(?<!\d)(\d{4})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{1,2})(?!\d)")
_DATE_KO_FULL_RE = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_DATE_KO_YM_RE = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월")
_DATE_KO_MD_RE = re.compile(r"(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_TIME_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")
_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _norm_int(s: str) -> str:
    s = s.replace(",", "")
    if "." in s:
        whole, frac = s.split(".", 1)
        whole = str(int(whole)) if whole else "0"
        frac = frac.rstrip("0")
        return f"{whole}.{frac}" if frac else whole
    return str(int(s)) if s else s


def _extract_numeric(text: str) -> tuple[set[str], str]:
    """(tokens, text with dates/times blanked) — dates normalized first so their parts are not
    re-counted as bare numbers."""
    t = unicodedata.normalize("NFKC", text or "")
    toks: set[str] = set()

    def sub(rx: re.Pattern, fmt) -> None:
        nonlocal t

        def rep(m: re.Match) -> str:
            toks.add(fmt(m))
            return " "
        t = rx.sub(rep, t)

    sub(_DATE_KO_FULL_RE, lambda m: f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}")
    sub(_DATE_YMD_RE, lambda m: f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}")
    sub(_DATE_KO_YM_RE, lambda m: f"{int(m[1]):04d}-{int(m[2]):02d}")
    sub(_DATE_KO_MD_RE, lambda m: f"{int(m[1]):02d}-{int(m[2]):02d}")
    sub(_TIME_RE, lambda m: f"{int(m[1])}:{m[2]}")
    for m in _NUM_RE.finditer(t):
        toks.add(_norm_int(m.group(0)))
    return toks, t


def numeric_tokens(text: str) -> frozenset[str]:
    """Numbers (commas removed, "8,081"→"8081", decimals kept) and normalized dates
    ("2026-10-10"; "10월 10일"→"10-10"; "2026년 10월"→"2026-10"); times "9:00"."""
    return frozenset(_extract_numeric(text)[0])


# Word-initial "안" (안했다, 안보낸다, 안함, 안감) is negation in Korean even without a space after
# it. Conservative: words such as 안내/안전 also match, which only sends a pair to the judge.
NEGATION_RE = re.compile(r"않|안 |(?<![가-힣])안|못|없|아니|말라|마라|금지|\bnot\b|\bnever\b|\bno\b|n't",
                         re.I)
# "2026-09-28 기준" / "9월 28일 기준": the as-of date of a state sentence (prompts: state carries it)
_ASOF_RE = re.compile(r"(\d{4}\s*[-./]\s*\d{1,2}\s*[-./]\s*\d{1,2}|\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일|"
                      r"\d{1,2}\s*월\s*\d{1,2}\s*일)\s*(?:\d{1,2}:\d{2}\s*)?기준")


def has_negation(text: str) -> bool:
    return bool(NEGATION_RE.search(text or ""))


def same_family(k1: str, k2: str) -> bool:
    f1, f2 = KIND_FAMILY.get(k1), KIND_FAMILY.get(k2)
    return f1 is not None and f1 == f2


def _date_forms(date: str | None) -> set[str]:
    """'2026-09-28' → {'2026-09-28', '09-28'} (both forms numeric_tokens may produce)."""
    if not date or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return set()
    return {date, date[5:]}


def state_value_tokens(text: str, ref_date: str | None = None) -> frozenset[str]:
    """numeric_tokens of a state sentence without its as-of date: dates written as "<date> 기준"
    and dates equal to the row's own event date (KST, `ref_date`) are dropped. A date that is the
    state's value (e.g. an expiry date) stays."""
    t = unicodedata.normalize("NFKC", text or "")
    asof = set()
    for m in _ASOF_RE.finditer(t):
        asof |= numeric_tokens(m.group(1))
    drop = asof | _date_forms(ref_date)
    return frozenset(tok for tok in numeric_tokens(t) if tok not in drop)


def auto_dup_eligible(new_text: str, old_text: str, new_kind: str, old_kind: str, *,
                      new_ref_date: str | None = None, old_ref_date: str | None = None) -> bool:
    """R0-4 (cos ≥ auto_dup_cos is checked by the caller). Two `state` sentences that differ only
    in their as-of date are the same state re-confirmed (DEVIATIONS F-19)."""
    if not same_family(new_kind, old_kind) or has_negation(new_text) != has_negation(old_text):
        return False
    if numeric_tokens(new_text) == numeric_tokens(old_text):
        return True
    if new_kind == "state" and old_kind == "state":
        return state_value_tokens(new_text, new_ref_date) == state_value_tokens(old_text, old_ref_date)
    return False


# ── fact preservation (different_aspects consolidation) ─────────────────────

PARTICLES: tuple[str, ...] = tuple(sorted(
    ("에서", "으로", "에게", "한테", "까지", "부터", "이다", "은", "는", "이", "가", "을", "를", "의",
     "에", "로", "와", "과", "도", "만", "다", "요"), key=lambda p: (-len(p), p)))

_HANGUL_RE = re.compile(r"[가-힣]+")
_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-./]*")
_QUOTED_RE = re.compile(r"'([^'\n]{1,80})'|\"([^\"\n]{1,80})\"|「([^」\n]{1,80})」|“([^”\n]{1,80})”|‘([^’\n]{1,80})’")


def strip_particle(word: str) -> str:
    """Remove one trailing particle (longest match first)."""
    for p in PARTICLES:
        if word.endswith(p) and len(word) > len(p):
            return word[: -len(p)]
        if word == p:
            return ""
    return word


def fact_tokens(text: str) -> set[str]:
    """Numbers, dates, latin tokens (≥2 chars, casefolded), quoted strings, Hangul words of ≥2
    syllables after strip_particle. All casefolded/NFKC."""
    nums, rest = _extract_numeric(text)
    out: set[str] = set(nums)
    t = unicodedata.normalize("NFKC", text or "")
    for m in _QUOTED_RE.finditer(t):
        q = next(g for g in m.groups() if g is not None)
        q = norm_text(q)
        if q:
            out.add(q)
    rest = unicodedata.normalize("NFKC", rest)
    for m in _LATIN_RE.finditer(rest):
        w = m.group(0).rstrip("-./").casefold()
        if len(w) >= 2:
            out.add(w)
    for m in _HANGUL_RE.finditer(rest):
        w = strip_particle(m.group(0))
        if len(w) >= 2:
            out.add(w)
    return out


def _numeric_present(tok: str, merged_nums: frozenset[str], merged_nfkc: str) -> bool:
    """A number/date token of an input is kept only as a whole number (37 ≠ 370, 7 ≠ 17):
    token-set membership, else a digit-boundary match in the merged text (a date part such as
    "10" inside "2026-10-10" still counts)."""
    if tok in merged_nums:
        return True
    pat = re.escape(tok).replace(r"\-", r"\s*[-./]\s*")
    return re.search(rf"(?<![\d.,]){pat}(?![\d]|[.,]\d)", merged_nfkc) is not None


# Predicate forms are not facts (DEVIATIONS E2E-3): "조회한다" vs "조회하며", "찾아야" vs "조회한다".
# A 하다/되다 predicate needs its noun stem (≥2 syllables) in the merged text; a native verb with a
# one-syllable stem (찾아야, 봐서) is not required. Ambiguous endings (한/할/된/될…) apply only with a
# ≥2-syllable stem, so short nouns such as 기한·역할·제한 are still required whole.
_HADA_ENDINGS = tuple(sorted(("하여", "하고", "하며", "하면", "하는", "하던", "한다", "했다", "해서", "해야",
                              "하지", "되어", "되는", "된다", "됐다", "한", "할", "함", "해", "했", "된", "될",
                              "됨", "돼"), key=lambda e: (-len(e), e)))
_FREE_ENDINGS = tuple(sorted(("아야", "어야", "여야", "는다", "았다", "었다", "였다", "아서", "어서", "으며",
                              "으면"), key=lambda e: (-len(e), e)))
_HANGUL_TOKEN_RE = re.compile(r"[가-힣]+")


def predicate_stem(tok: str) -> str | None:
    """None: not a predicate form. "": predicate without a fact stem (not required). Otherwise the
    noun stem that must survive."""
    if not _HANGUL_TOKEN_RE.fullmatch(tok or ""):
        return None
    for e in _HADA_ENDINGS:
        if tok.endswith(e) and len(tok) - len(e) >= 2:
            return tok[: -len(e)]
    for e in _FREE_ENDINGS:
        if tok.endswith(e) and len(tok) > len(e):
            stem = tok[: -len(e)]
            return stem if len(stem) >= 2 else ""
    return None


def preservation_check(a: str, b: str, merged: str) -> tuple[bool, list[str]]:
    """Every fact token of `a` and `b` must occur in `merged`. Numbers and dates must survive as
    whole tokens (no substring match, DEVIATIONS F-26); other tokens may also match as a
    substring of norm_text(merged). Returns (ok, sorted missing tokens)."""
    merged_tokens = fact_tokens(merged)
    merged_norm = norm_text(merged)
    merged_compact = merged_norm.replace(" ", "")
    merged_nfkc = unicodedata.normalize("NFKC", merged or "")
    merged_nums = numeric_tokens(merged)
    nums = numeric_tokens(a) | numeric_tokens(b)
    missing: list[str] = []
    for tok in fact_tokens(a) | fact_tokens(b):
        if tok in nums:
            if not _numeric_present(tok, merged_nums, merged_nfkc):
                missing.append(tok)
            continue
        if tok in merged_tokens or tok in merged_norm or tok.replace(" ", "") in merged_compact:
            continue
        stem = predicate_stem(tok)
        if stem is not None and (stem == "" or stem in merged_compact):
            continue
        missing.append(tok)
    return (not missing, sorted(missing))
