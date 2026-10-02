"""N5 normalization: subject_key, verified refs, importance (PLAN-v2 §4.2 N5, §5.3; CONTRACTS §4.9).

importance = clamp(KIND_BASE[kind] + 0.08·(level−3) + 0.12·explicit_user
                   + 0.05·max(0, min(user_session_count−1, 3)) − 0.10·assistant_only, 0.05, 1.0)   (D7)
then source floors: core:* ≥ 0.85, tool:yume_remember ≥ 0.80. Never lowered on update (max(old, new)).
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from .types import CORE_SOURCES, KIND_BASE, Claim

IMPORTANCE_MIN, IMPORTANCE_MAX = 0.05, 1.0
CORE_FLOOR = 0.85
REMEMBER_FLOOR = 0.80
MAX_REFS = 8
REF_EXTS = (".md", ".py", ".json", ".yaml", ".yml", ".sh", ".txt")
_DENY_BASENAMES = frozenset({".env", "auth.json"})          # never point at credential files
_PATH_TOKEN_RE = re.compile(r"[A-Za-z0-9_.~/\-]+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,63}")


def subject_key(subject: str) -> str:
    """NFKC, casefold, drop whitespace and every Unicode P*/S* character."""
    t = unicodedata.normalize("NFKC", subject or "").casefold()
    return "".join(ch for ch in t if not ch.isspace() and unicodedata.category(ch)[0] not in "PS")


def _is_pathlike(tok: str) -> bool:
    return "/" in tok or tok.lower().endswith(REF_EXTS)


def ref_candidates(text: str) -> list[str]:
    """Path-like tokens (contain '/' or end with a known extension) first, then bare words
    (skill-name candidates). Trailing sentence punctuation is trimmed; order kept, deduped."""
    out: list[str] = []
    for m in _PATH_TOKEN_RE.finditer(text or ""):
        tok = m.group(0).rstrip(".-")
        if not tok or tok in out:
            continue
        if _is_pathlike(tok) and tok.strip("/.~"):
            out.append(tok)
    for m in _WORD_RE.finditer(text or ""):
        w = m.group(0).rstrip("-_")
        if len(w) >= 3 and w not in out:
            out.append(w)
    return out


def _resolve_under(cand: Path, base: Path) -> Path | None:
    try:
        p = cand.resolve(strict=False)
        b = base.resolve(strict=False)
    except (OSError, RuntimeError):
        return None
    try:
        p.relative_to(b)
    except ValueError:
        return None
    return p


def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except OSError:
        return False


def _skill_index(hermes_home: Path) -> dict[str, str]:
    """normalized skill name → real dir name, for skills with SKILL.md."""
    out: dict[str, str] = {}
    sd = hermes_home / "skills"
    try:
        for d in sorted(sd.iterdir()):
            if d.is_dir() and (d / "SKILL.md").is_file():
                out[_skill_norm(d.name)] = d.name
    except OSError:
        pass
    return out


def _skill_norm(name: str) -> str:
    return name.casefold().replace("_", "-")


def verify_refs(text: str, *, paths: Any, workspace_dir: str) -> list[str]:
    """Keep only references that exist: paths under workspace_dir or hermes_home, and bare words
    naming an existing skills/<name>/SKILL.md. Stored relative to workspace_dir when under it
    (skills as "skills/<name>/SKILL.md"), else absolute."""
    ws = Path(workspace_dir).expanduser() if workspace_dir else None
    home = Path(paths.hermes_home)
    bases = [b for b in (ws, home) if b is not None]
    skills: dict[str, str] | None = None
    out: list[str] = []

    def stored(p: Path) -> str:
        if ws is not None:
            r = _resolve_under(p, ws)
            if r is not None:
                rel = str(r.relative_to(ws.resolve(strict=False)))
                return rel if rel != "." else str(r)
        return str(p)

    for tok in ref_candidates(text):
        if len(out) >= MAX_REFS:
            break
        if _is_pathlike(tok):
            if tok.startswith("~"):
                continue                                   # never expand into a real home
            found: Path | None = None
            if os.path.isabs(tok):
                for b in bases:
                    r = _resolve_under(Path(tok), b)
                    if r is not None and _exists(r):
                        found = r
                        break
            else:
                rel = tok[2:] if tok.startswith("./") else tok
                for b in bases:
                    r = _resolve_under(b / rel, b)
                    if r is not None and _exists(r):
                        found = r
                        break
            if found is None or found.name in _DENY_BASENAMES:
                continue
            s = stored(found)
        else:
            if skills is None:
                skills = _skill_index(home)
            name = skills.get(_skill_norm(tok))
            if name is None:
                continue
            s = f"skills/{name}/SKILL.md"
        if s not in out:
            out.append(s)
    return out


def compute_importance(*, kind: str, level: int, explicit_user: bool, user_session_count: int,
                       assistant_only: bool, source: str) -> float:
    base = KIND_BASE.get(kind, KIND_BASE["fact"])
    lvl = max(1, min(5, int(level or 3)))
    usc = max(0, min(int(user_session_count or 0) - 1, 3))
    v = (base + 0.08 * (lvl - 3) + 0.12 * (1 if explicit_user else 0) + 0.05 * usc
         - 0.10 * (1 if assistant_only else 0))
    v = min(IMPORTANCE_MAX, max(IMPORTANCE_MIN, v))
    if source in CORE_SOURCES or (source or "").startswith("core:"):
        v = max(v, CORE_FLOOR)
    elif source == "tool:yume_remember":
        v = max(v, REMEMBER_FLOOR)
    return round(v, 4)


def normalize_claim(claim: Claim, *, cfg: Any, paths: Any) -> Claim:
    """Sets subject_key, refs (verified, merged with any existing), importance (never lowered)."""
    if not (claim.subject or "").strip():
        claim.subject = (claim.text or "").strip()[:30]
    claim.subject_key = subject_key(claim.subject) or subject_key(claim.text[:30])
    refs = verify_refs(claim.text, paths=paths, workspace_dir=getattr(cfg, "workspace_dir", "") or "")
    merged = list(claim.refs)
    for r in refs:
        if r not in merged:
            merged.append(r)
    claim.refs = merged
    imp = compute_importance(kind=claim.kind, level=claim.level, explicit_user=claim.explicit_user,
                             user_session_count=claim.user_session_count,
                             assistant_only=claim.assistant_only, source=claim.source)
    claim.importance = max(float(claim.importance or 0.0), imp)
    return claim


__all__ = ["subject_key", "ref_candidates", "verify_refs", "compute_importance", "normalize_claim"]
