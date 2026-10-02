"""Build ``serving/recall.sqlite`` — the provider's read-only derived index (PLAN-v2 §2.5, R8-8).

Rows with status in SERVING_STATUSES only (forgotten / quarantined / candidate are never
exported). Per row: strength + tier from ``strength.evaluate`` at export time, refs filtered to
paths that exist, ``vec`` = cfg.embed_dim float32 LE (1536 openai, 1024 hash — the configured model,
same as the Lance column), ``vec256`` = first 256 dims re-normalized (the provider prefilters with it
only for text-embedding-3 models; every other model is searched single-stage on ``vec``). The file is
built as ``serving/recall.<run_id>.sqlite`` (journal DELETE, 0600) and atomically ``os.replace``d
onto ``serving/recall.sqlite``; the directory is fsynced. Never called in dry-run.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from . import paths as _paths
from .embedder import prefix_renorm
from .types import SERVING_STATUSES, MemoryRow

log = logging.getLogger("hermesyume.export")


def serving_schema() -> ModuleType:
    return _paths.load_provider_module("serving_schema")


def _corefmt() -> ModuleType:
    return _paths.load_provider_module("corefmt")


def _evaluate(row: MemoryRow, now: float, cfg: Any):
    """strength.evaluate (B4). Indirection so unit tests can substitute it."""
    from . import strength
    return strength.evaluate(row, now, cfg)


def existing_refs(refs: list[str], *, workspace_dir: str, hermes_home: Path) -> list[str]:
    """Refs whose target exists: absolute paths as-is; relative ones under the workspace, then
    under HERMES_HOME (skills/<name>/SKILL.md). Order kept, duplicates dropped."""
    out: list[str] = []
    for ref in refs or []:
        r = str(ref or "").strip()
        if not r or r in out:
            continue
        p = Path(os.path.expanduser(r))
        if p.is_absolute():
            cands = [p]
        else:      # an unset workspace_dir must not resolve against the process cwd
            cands = ([Path(workspace_dir).expanduser() / p] if str(workspace_dir or "").strip() else []) \
                + [Path(hermes_home) / p]
        if any(c.exists() for c in cands):
            out.append(r)
    return out


def label_for(row: MemoryRow) -> str:
    return _corefmt().entry_label(row.text) or row.subject or ""


def _blob(vec: Any) -> bytes:
    return np.ascontiguousarray(np.asarray(vec, dtype="<f4")).tobytes()


def item_record(row: MemoryRow, *, now: float, cfg: Any, workspace_dir: str,
                hermes_home: Path) -> dict | None:
    """serving `items` row (dict keyed by serving_schema.ITEM_COLUMNS) or None when not served."""
    if row.status not in SERVING_STATUSES:
        return None
    dim = int(cfg.embed_dim)
    if row.vector is None:
        return None
    vec = np.asarray(row.vector, dtype=np.float32).reshape(-1)
    if vec.shape != (dim,) or not np.all(np.isfinite(vec)):
        return None
    sr = _evaluate(row, now, cfg)
    refs = existing_refs(list(row.refs or []), workspace_dir=workspace_dir, hermes_home=hermes_home)
    return {
        "id": row.id, "text": row.text, "subject": row.subject or "", "kind": row.kind,
        "tier": sr.tier, "status": row.status, "pinned": 1 if row.pinned else 0,
        "core_sha": row.core_sha or None, "event_time": row.event_time,
        "valid_until": row.valid_until, "strength": float(sr.strength),
        "refs": json.dumps(refs, ensure_ascii=False),
        "vec256": _blob(prefix_renorm(vec)), "vec": _blob(vec),
    }


def _fsync_dir(d: Path) -> None:
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def build_serving(ctx: Any, *, rows: dict[str, MemoryRow] | None = None,
                  lance_version: int | None = None) -> Path:
    if getattr(ctx, "dry_run", False):
        raise RuntimeError("export.build_serving must not run in dry-run")
    ss = serving_schema()
    cfg, paths = ctx.cfg, ctx.paths
    if rows is None:
        rows = ctx.store.load_working_set()
    if lance_version is None:
        lance_version = ctx.store.version()
    workspace_dir = str(cfg.workspace_dir)
    hermes_home = Path(paths.hermes_home)

    items: list[dict] = []
    pins: list[tuple] = []
    dropped_refs: list[tuple[str, list[str]]] = []
    skipped_vec = 0
    for rid in sorted(rows):
        row = rows[rid]
        rec = item_record(row, now=ctx.now, cfg=cfg, workspace_dir=workspace_dir,
                          hermes_home=hermes_home)
        if rec is None:
            if row.status in SERVING_STATUSES:
                skipped_vec += 1
            continue
        kept = json.loads(rec["refs"])
        missing = [r for r in (row.refs or []) if r not in kept]
        if missing:
            dropped_refs.append((row.id, missing))
        items.append(rec)
    item_ids = {it["id"] for it in items}
    for row in sorted((r for r in rows.values() if r.pinned and r.status == "active"),
                      key=lambda r: (r.created_at or 0.0, r.id)):
        if row.id in item_ids:
            pins.append((row.id, row.text, label_for(row), row.core_target))

    serving_dir = paths.ensure_dir(paths.serving_dir)
    tmp = paths.serving_tmp(ctx.run_id)
    for suffix in ("", "-journal", "-wal", "-shm"):
        try:
            os.unlink(str(tmp) + suffix)
        except FileNotFoundError:
            pass
    try:
        conn = sqlite3.connect(str(tmp))
        try:
            conn.execute("PRAGMA journal_mode=DELETE")
            ss.create_schema(conn)
            cols = ss.ITEM_COLUMNS
            conn.executemany("INSERT INTO items(%s) VALUES(%s)" % (", ".join(cols), ",".join("?" * len(cols))),
                             [tuple(it[c] for c in cols) for it in items])
            conn.executemany("INSERT INTO items_fts(id, text, subject) VALUES(?,?,?)",
                             [(it["id"], it["text"], it["subject"]) for it in items])
            conn.executemany("INSERT INTO pins(id, text, label, core_target) VALUES(?,?,?,?)", pins)
            meta = {"embed_model": cfg.embed_model_id(), "dim": str(int(cfg.embed_dim)),
                    "run_id": str(ctx.run_id), "lance_version": str(lance_version),
                    "built_at": repr(float(ctx.now)), "count": str(len(items))}
            conn.executemany("INSERT INTO meta(key, value) VALUES(?,?)", sorted(meta.items()))
            conn.commit()
        finally:
            conn.close()
        os.chmod(tmp, 0o600)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, paths.recall_sqlite)
        _fsync_dir(serving_dir)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise

    if dropped_refs:
        for mid, missing in dropped_refs:
            ctx.note(f"참조 경로가 없어 서빙에서 뺐습니다: {mid} → {', '.join(missing)}")
    if skipped_vec:
        ctx.note(f"벡터가 없거나 차원이 맞지 않아 서빙에서 제외한 행 {skipped_vec}개")
    pin_chars = sum(len(p[1] or "") for p in pins)
    if pin_chars > int(cfg.pins_budget_chars):
        ctx.note(f"고정 기억(pin) 합계 {pin_chars}자가 예산 {int(cfg.pins_budget_chars)}자를 넘습니다. "
                 "넘친 pin도 보호되고 회상됩니다.")
    log.info("serving export: %d items, %d pins", len(items), len(pins))
    return paths.recall_sqlite
