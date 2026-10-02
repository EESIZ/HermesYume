"""N8 procedure documents (PLAN-v2 §4.2 N8): the two-layer "short fact + pointer to a document"
structure, restored in a limited way.

- only ``kind=procedure`` claims with ``steps ≥ 3`` whose text summarizes their source messages
  (shorter, and the source shows that many steps)
- at most ``max_docs_per_run`` per run; body = the claim's evidence messages (+ the reply right after),
  secrets redacted, rejected entirely when the threat scanner flags it
- never written into the live agent's workspace from a sandbox HERMES_HOME (paths guard)
- ``plan_docs`` runs before commit (the row's ``refs`` gets ``docs/yume/<slug>.md``);
  ``write_docs`` runs after commit and only ever writes inside ``<workspace>/docs/yume/``
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from .types import Claim, DocWrite, Window

log = logging.getLogger("hermesyume.docs_writer")

DOCS_SUBDIR = "docs/yume"
_ROLE_KO = {"user": "사용자", "assistant": "에이전트", "agent_log": "기록"}


def slugify(subject: str) -> str:
    """NFKC lower, spaces→'-', keep [\\w-] (Hangul ok), ≤60 chars; empty → sha8 of subject."""
    s = unicodedata.normalize("NFKC", subject or "").lower().strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^\w\-]", "", s)
    s = re.sub(r"-{2,}", "-", s).strip("-_")[:60].strip("-_")
    return s or hashlib.sha256((subject or "").encode("utf-8")).hexdigest()[:8]


def doc_path(cfg: Any, slug: str) -> Path:
    return Path(cfg.workspace_dir) / DOCS_SUBDIR / f"{slug}.md"


def _evidence_span(claim: Claim, window: Window) -> list:
    """The claim's own source messages: first..last evidence message of the window, plus the
    reply that directly follows the last one (a procedure is often spelled out there). Unrelated
    chat elsewhere in the window stays out of the document (DEVIATIONS F-22)."""
    keys = set(claim.evidence_keys or ())
    msgs = list(window.messages)
    idx = [i for i, m in enumerate(msgs) if m.key in keys]
    if not idx:
        return []
    lo, hi = min(idx), max(idx)
    if hi + 1 < len(msgs) and msgs[hi + 1].role != "user" and msgs[hi].role == "user":
        hi += 1
    return msgs[lo:hi + 1]


def _body_text(msgs: list) -> str:
    lines = []
    for m in msgs:
        t = (m.text or "").strip()
        if t:
            lines.append(f"- ({_ROLE_KO.get(m.role, m.role)}) {t}")
    return "\n".join(lines)


_STEP_LINE_RE = re.compile(r"^\s*(?:\d+\s*[.)단]|[-*•·]|[①-⑳])", re.M)
_STEP_SEP_RE = re.compile(r"\s*(?:→|->|=>|⇒|▶)\s*")


def step_count(text: str) -> int:
    """Visible steps in source text: list/numbered lines, or the parts of an arrow chain."""
    lines = len(_STEP_LINE_RE.findall(text or ""))
    arrows = max((len(_STEP_SEP_RE.split(seg)) for seg in (text or "").splitlines() if seg.strip()),
                  default=0)
    return max(lines, arrows if arrows > 1 else 0)


def is_summary(claim: Claim, msgs: list) -> bool:
    """The claim is a summary of its source: shorter than the source text, and the source really
    shows the claimed number of steps (one per message also counts)."""
    texts = [(m.text or "").strip() for m in msgs if (m.text or "").strip()]
    body = sum(len(t) for t in texts)
    steps = max(step_count("\n".join(texts)), len(texts))
    return len(claim.text) < body and steps >= int(claim.steps or 0)


def docs_refusal(ctx: Any) -> str | None:
    """None when N8 may write under cfg.workspace_dir (F-1 live-workspace guard)."""
    from .paths import workspace_write_refusal
    return workspace_write_refusal(ctx.paths.hermes_home, getattr(ctx.cfg, "workspace_dir", ""))


def plan_docs(ctx: Any, inserted: list[tuple[Claim, str]], windows: dict[str, Window]) -> list[DocWrite]:
    from .threat import redact_secrets
    cfg = ctx.cfg
    limit = int(cfg.max_docs_per_run)
    out: list[DocWrite] = []
    slugs: set[str] = set()
    cands = [(c, m) for c, m in inserted if c.kind == "procedure" and (c.steps or 0) >= 3]
    if cands:
        why = docs_refusal(ctx)
        if why is not None:
            ctx.note(f"절차 문서(docs/yume)를 만들지 않았습니다: {why}.")
            return []
    for claim, memory_id in cands:
        if len(out) >= limit:
            break
        w = windows.get(claim.window_id or "")
        msgs = _evidence_span(claim, w) if w is not None else []
        if not msgs or not is_summary(claim, msgs):
            continue
        body, _counts = redact_secrets(_body_text(msgs))
        if not body.strip():
            continue
        if ctx.scanner is not None and ctx.scanner.threats(body, "strict"):
            ctx.note(f"절차 문서 본문이 위협 검사에 걸려 만들지 않았습니다 ({claim.subject}).")
            continue
        slug = slugify(claim.subject or claim.text[:30])
        if slug in slugs:
            continue
        slugs.add(slug)
        p = doc_path(cfg, slug)
        out.append(DocWrite(slug=slug, path=str(p), title=claim.subject or claim.text[:40], body=body,
                            memory_id=memory_id, origin_key=claim.origin_key))
    return out


def doc_ref(slug: str) -> str:
    """Row ref (relative to workspace_dir, as normalize.verify_refs stores it)."""
    return f"{DOCS_SUBDIR}/{slug}.md"


def _safe_target(path: Path, slug: str) -> Path:
    parent = path.parent
    if path.name != f"{slug}.md" or "/" in slug or slug in ("", ".", ".."):
        raise ValueError("bad slug")
    if parent.parts[-2:] != ("docs", "yume"):
        raise ValueError("outside docs/yume")
    parent.mkdir(parents=True, exist_ok=True)
    real_parent = parent.resolve()
    if real_parent.parts[-2:] != ("docs", "yume"):
        raise ValueError("docs/yume resolves elsewhere")
    if path.is_symlink():
        raise ValueError("target is a symlink")
    return real_parent / path.name


def write_docs(docs: list[DocWrite], *, dry_run: bool,
               hermes_home: Any = None) -> list[tuple[DocWrite, bool, str | None]]:
    """After commit. New file: '# <title>\\n\\n<body>\\n'; existing: append
    '\\n---\\n## <KST date>\\n\\n<body>\\n'. Atomic replace; never outside docs/yume, and never
    into the live workspace from a sandbox HERMES_HOME (`hermes_home` given)."""
    from .clock import kst_date, now as clock_now
    from .paths import in_live_workspace, is_live_home
    res: list[tuple[DocWrite, bool, str | None]] = []
    sandbox = hermes_home is not None and not is_live_home(hermes_home)
    for d in docs:
        if dry_run:
            res.append((d, False, "dry_run"))
            continue
        if sandbox and in_live_workspace(d.path):
            res.append((d, False, "refused: live workspace from a sandbox HERMES_HOME"))
            continue
        try:
            target = _safe_target(Path(d.path), d.slug)
            if target.exists():
                old = target.read_text(encoding="utf-8")
                if d.body.strip() and d.body.strip() in old:
                    res.append((d, True, "already_present"))
                    continue
                content = old.rstrip("\n") + f"\n\n---\n## {kst_date(clock_now())}\n\n{d.body}\n"
            else:
                content = f"# {d.title}\n\n{d.body}\n"
            fd, tmp = tempfile.mkstemp(prefix=".yume_", suffix=".tmp", dir=str(target.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, target)
            except BaseException:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
            res.append((d, True, None))
        except Exception as e:  # reported by rem.post_commit (Dream Log note)
            log.warning("doc write failed: %s", type(e).__name__)
            res.append((d, False, f"{type(e).__name__}: {e}"))
    return res
