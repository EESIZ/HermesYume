"""Text kept outside Lance: scrub on purge, bounded retention (PLAN-v2 §5.4, §10.2; DEVIATIONS F-5).

A purge (forgotten + 30 days, quarantined secret) deletes the Lance row and its history, but the
row text also lived in
- ``runs/<id>/plan.json`` (upsert snapshots, op changes, report lists),
- earlier Dream Logs (created / reinforced / … lists),
- live.db inbox rows (remember / core_* texts, forget reasons).

``after_commit`` (rem.post_commit, after the serving copy is replaced) scrubs those for the ids
purged by the run, prunes ``runs/<id>/`` directories of finished runs after ``runs_keep_days``
(planned/held plans stay: replay / approve need them), and deletes consumed inbox rows after
``inbox_keep_days`` (session_end markers stay — they carry no text). Never in --dry-run.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

MARK = "[삭제된 기억]"
_TEXT_KEYS = ("text", "subject", "old_text", "new_text", "a", "b", "result", "body")
_ID_TEXT = {"id": ("text", "subject", "a", "b", "result"), "memory_id": ("text", "subject", "body", "title"),
            "old_id": ("old_text",), "new_id": ("new_text",)}
_KEEP_STATUSES = frozenset({"planned", "held"})


def purged_info(ws: Any) -> dict[str, dict]:
    """id → {"text", "inbox_ids"} for the rows this run hard-deletes (read before commit)."""
    out: dict[str, dict] = {}
    for mid in getattr(ws, "purge_ids", []) or []:
        row = ws.base.get(mid)
        if row is None:
            continue
        inbox: set[int] = set()
        for k in list(row.origin_keys or []) + list(row.source_message_ids or []):
            k = str(k)
            for pre in ("inbox:", "i:"):
                if k.startswith(pre) and k[len(pre):].isdigit():
                    inbox.add(int(k[len(pre):]))
        out[mid] = {"text": row.text or "", "inbox_ids": sorted(inbox)}
    return out


# ── plan.json ────────────────────────────────────────────────────────────────

def _scrub_strings(obj: Any, texts: list[str]) -> tuple[Any, bool]:
    if isinstance(obj, str):
        new = obj
        for t in texts:
            if t and t in new:
                new = new.replace(t, MARK)
        return new, new != obj
    if isinstance(obj, list):
        changed = False
        out = []
        for x in obj:
            y, c = _scrub_strings(x, texts)
            out.append(y)
            changed |= c
        return out, changed
    if isinstance(obj, dict):
        changed = False
        out = {}
        for k, v in obj.items():
            y, c = _scrub_strings(v, texts)
            out[k] = y
            changed |= c
        return out, changed
    return obj, False


def _scrub_by_id(obj: Any, ids: set[str]) -> bool:
    """Dicts that name a purged id lose the text fields belonging to it (report items, ops,
    upsert snapshots, docs)."""
    changed = False
    if isinstance(obj, list):
        for x in obj:
            changed |= _scrub_by_id(x, ids)
        return changed
    if not isinstance(obj, dict):
        return False
    for id_key, fields in _ID_TEXT.items():
        if "text_sha" in obj:          # a suppress entry: vector + sha only, nothing to scrub
            break
        if str(obj.get(id_key, "")) in ids:
            for f in fields:
                if isinstance(obj.get(f), str) and obj[f] != MARK:
                    obj[f] = MARK
                    changed = True
            if "vector_b64" in obj and obj["vector_b64"] is not None:
                obj["vector_b64"] = None
                changed = True
            for sub in ("changes", "before"):
                d = obj.get(sub)
                if isinstance(d, dict):
                    for f in ("text", "subject"):
                        if isinstance(d.get(f), str) and d[f] != MARK:
                            d[f] = MARK
                            changed = True
                    if d.get("vector_b64") is not None:
                        d["vector_b64"] = None
                        changed = True
    if str(obj.get("memory_id", "")) in ids and "before_json" in obj:
        if obj.get("before_json") != "{}" or obj.get("after_json") != "{}":
            obj["before_json"] = obj["after_json"] = "{}"
            changed = True
    for v in obj.values():
        if isinstance(v, (dict, list)):
            changed |= _scrub_by_id(v, ids)
    return changed


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".scrub_", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def scrub_plan_files(paths: Any, ids: Iterable[str], texts: Iterable[str]) -> int:
    ids = {str(i) for i in ids if i}
    texts = sorted({t for t in texts if t and len(t) >= 4}, key=len, reverse=True)
    if not ids and not texts:
        return 0
    n = 0
    rd = Path(paths.runs_dir)
    if not rd.is_dir():
        return 0
    for p in sorted(rd.glob("*/plan.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        changed = _scrub_by_id(data, ids)
        data, c2 = _scrub_strings(data, texts)
        if changed or c2:
            _atomic_write(p, json.dumps(data, ensure_ascii=False, sort_keys=True, default=str))
            n += 1
    return n


# ── Dream Logs ───────────────────────────────────────────────────────────────

def scrub_dream_logs(paths: Any, texts: Iterable[str]) -> int:
    """Replace a purged text wherever an earlier Dream Log rendered it (same renderer, so the
    escaping matches; an 80-char prefix catches notes that quoted the start)."""
    from . import dream_log
    forms: set[str] = set()
    for t in texts:
        if not t or len(t) < 4:
            continue
        forms.add(dream_log.t(t))
        if len(t) > 80:
            forms.add(dream_log.t(t[:80]))
    forms = {f for f in forms if len(f) >= 4}
    if not forms:
        return 0
    d = Path(paths.dream_log_dir)
    if not d.is_dir():
        return 0
    n = 0
    ordered = sorted(forms, key=len, reverse=True)
    for p in sorted(d.rglob("*.md")):
        try:
            s = p.read_text(encoding="utf-8")
        except OSError:
            continue
        new = s
        for f in ordered:
            new = new.replace(f, dream_log.HIDDEN)
        if new != s:
            _atomic_write(p, new)
            n += 1
    return n


# ── runs/ retention ──────────────────────────────────────────────────────────

def prune_runs(paths: Any, ledger: Any, *, keep_days: float, current_run: str | None = None,
               now: float | None = None) -> int:
    """Delete runs/<id>/ of finished runs (committed/failed/dry/unknown) older than keep_days
    (real time, plan.json mtime). planned/held plans are kept (replay, approve)."""
    rd = Path(paths.runs_dir)
    if keep_days <= 0 or not rd.is_dir():
        return 0
    now = time.time() if now is None else float(now)
    cutoff = now - float(keep_days) * 86400.0
    n = 0
    for d in sorted(rd.iterdir()):
        if not d.is_dir() or d.name == current_run:
            continue
        p = d / "plan.json"
        try:
            mt = (p if p.exists() else d).stat().st_mtime
        except OSError:
            continue
        if mt >= cutoff:
            continue
        rec = ledger.get_run(d.name) if ledger is not None else None
        if rec is not None and rec.status in _KEEP_STATUSES:
            continue
        shutil.rmtree(d, ignore_errors=True)
        n += 1
    return n


# ── orchestration ────────────────────────────────────────────────────────────

def after_commit(ctx: Any, plan: Any) -> dict[str, int]:
    """rem.post_commit hook (live/migrate, never dry-run). Failures are notes, never errors."""
    out = {"plans": 0, "logs": 0, "inbox_scrubbed": 0, "inbox_deleted": 0, "runs_pruned": 0}
    if getattr(ctx, "dry_run", False):
        return out
    info = dict(getattr(ctx, "purged", {}) or {})
    ids = set(info) | set(getattr(plan, "purge_ids", []) or [])
    texts = [v.get("text", "") for v in info.values()]
    cfg = ctx.cfg
    try:
        if ids:
            out["plans"] = scrub_plan_files(ctx.paths, ids, texts)
            out["logs"] = scrub_dream_logs(ctx.paths, texts)
            if ctx.live is not None and hasattr(ctx.live, "scrub_inbox"):
                inbox_ids = sorted({i for v in info.values() for i in v.get("inbox_ids", [])})
                out["inbox_scrubbed"] = ctx.live.scrub_inbox(inbox_ids, forget_memory_ids=sorted(ids))
    except Exception as e:  # noqa: BLE001
        ctx.note(f"삭제된 기억의 원문 정리(plan.json·Dream Log·inbox) 중 오류: {type(e).__name__}")
    try:
        out["runs_pruned"] = prune_runs(ctx.paths, ctx.ledger,
                                        keep_days=float(getattr(cfg, "runs_keep_days", 14)),
                                        current_run=getattr(plan, "run_id", None))
    except Exception as e:  # noqa: BLE001
        ctx.note(f"runs/ 정리 실패({type(e).__name__})")
    try:
        if ctx.live is not None and hasattr(ctx.live, "delete_consumed_before"):
            keep = float(getattr(cfg, "inbox_keep_days", 30))
            if keep > 0:
                out["inbox_deleted"] = ctx.live.delete_consumed_before(time.time() - keep * 86400.0)
    except Exception as e:  # noqa: BLE001
        ctx.note(f"live.db inbox 정리 실패({type(e).__name__})")
    return out
