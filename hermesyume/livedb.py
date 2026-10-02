"""Dream-side access to live.db (PLAN-v2 §2.4). DDL comes from provider/_yume/live_schema.py,
loaded by path (single source). Modes:

- ``rw``   live/migrate runs: creates the file+schema if missing, marks inbox rows consumed (R8-6)
- ``ro``   status/inspect: URI mode=ro (may create -shm/-wal next to a WAL db; not for dry-run)
- ``pure`` dry-run: reads a private snapshot copy; never touches the data dir
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

from .paths import Paths, load_provider_module
from .sqlite_util import SnapshotConnection
from .types import HealthRow, InboxItem, LiveSnapshot, RecallEvent

DREAM_BUSY_TIMEOUT_MS = 5000


def live_schema(paths: Paths | None = None) -> ModuleType:
    return load_provider_module("live_schema", paths)


def _np_vec(blob: bytes | None) -> Any:
    if not blob:
        return None
    import numpy as np
    return np.frombuffer(bytes(blob), dtype="<f4").astype(np.float32)


class LiveDB:
    def __init__(self, conn: sqlite3.Connection, path: Path, mode: str,
                 snap: SnapshotConnection | None = None):
        self.conn = conn
        self.path = path
        self.mode = mode
        self._snap = snap

    @classmethod
    def open(cls, paths: Paths, *, mode: str = "rw") -> "LiveDB | None":
        """Returns None for ro/pure when live.db does not exist yet."""
        p = paths.live_db
        if mode == "rw":
            schema = live_schema(paths)
            paths.ensure_dir(p.parent)
            new = not p.exists()
            conn = schema.connect(p, readonly=False, busy_timeout_ms=DREAM_BUSY_TIMEOUT_MS)
            if new:
                os.chmod(p, 0o600)
            conn.row_factory = sqlite3.Row
            return cls(conn, p, mode)
        if not p.exists():
            return None
        if mode == "ro":
            schema = live_schema(paths)
            conn = schema.connect(p, readonly=True, busy_timeout_ms=DREAM_BUSY_TIMEOUT_MS)
            conn.row_factory = sqlite3.Row
            return cls(conn, p, mode)
        if mode == "pure":
            snap = SnapshotConnection(p)
            return cls(snap.conn, p, mode, snap)
        raise ValueError(f"bad mode {mode!r}")

    def close(self) -> None:
        if self._snap is not None:
            self._snap.close()
        else:
            self.conn.close()

    def __enter__(self) -> "LiveDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── reads ──
    def snapshot(self) -> LiveSnapshot:
        """N0: ids at run start; R1/R4 fold only rows with id <= these."""
        a = self.conn.execute("SELECT COALESCE(MAX(id),0) FROM inbox").fetchone()[0]
        b = self.conn.execute("SELECT COALESCE(MAX(id),0) FROM recall_events").fetchone()[0]
        return LiveSnapshot(int(a), int(b))

    @staticmethod
    def _inbox(r: sqlite3.Row) -> InboxItem:
        meta: dict[str, Any] = {}
        if r["meta_json"]:
            try:
                meta = json.loads(r["meta_json"])
            except (json.JSONDecodeError, TypeError):
                meta = {"_raw": r["meta_json"]}
        return InboxItem(id=int(r["id"]), ts=float(r["ts"]), session_id=r["session_id"],
                         platform=r["platform"], op=r["op"], text=r["text"], old_text=r["old_text"],
                         kind=r["kind"], pin=bool(r["pin"]), memory_id=r["memory_id"],
                         target=r["target"], vec=_np_vec(r["vec"]), embed_model=r["embed_model"],
                         meta=meta, status=r["status"], consumed_run=r["consumed_run"])

    def inbox_range(self, after_id: int, upto_id: int, *, ops: Iterable[str] | None = None,
                    status: str | None = "pending") -> list[InboxItem]:
        q = "SELECT * FROM inbox WHERE id > ? AND id <= ?"
        args: list[Any] = [int(after_id), int(upto_id)]
        if status is not None:
            q += " AND status = ?"
            args.append(status)
        if ops:
            ops = list(ops)
            q += f" AND op IN ({','.join('?' * len(ops))})"
            args += ops
        return [self._inbox(r) for r in self.conn.execute(q + " ORDER BY id", args)]

    def pending_inbox(self, ops: Iterable[str] | None = None) -> list[InboxItem]:
        return self.inbox_range(0, 2 ** 62, ops=ops, status="pending")

    def session_end_ids(self) -> dict[str, float]:
        """Sessions with a session_end marker → marker ts (settle shortcut, §3.2). Includes consumed
        rows. A marker settles only messages up to its ts (statedb._settle; DEVIATIONS F-8)."""
        out: dict[str, float] = {}
        for sid, ts in self.conn.execute(
                "SELECT session_id, MAX(ts) FROM inbox WHERE op='session_end' AND session_id IS NOT NULL "
                "GROUP BY session_id"):
            out[str(sid)] = float(ts or 0.0)
        return out

    def injected_before(self, upto_id: int, since_ts: float) -> list[RecallEvent]:
        """`injected` events already folded (id ≤ cursor) on or after `since_ts` (F-33)."""
        rows = self.conn.execute(
            "SELECT id,ts,session_id,platform,turn_no,memory_id,kind,cos,mode,snapshot_run "
            "FROM recall_events WHERE id <= ? AND ts >= ? AND kind='injected' ORDER BY id",
            (int(upto_id), float(since_ts)))
        return [RecallEvent(int(r[0]), float(r[1]), r[2], r[3], r[4], r[5], r[6],
                            None if r[7] is None else float(r[7]), r[8], r[9]) for r in rows]

    def recall_range(self, after_id: int, upto_id: int) -> list[RecallEvent]:
        rows = self.conn.execute(
            "SELECT id,ts,session_id,platform,turn_no,memory_id,kind,cos,mode,snapshot_run "
            "FROM recall_events WHERE id > ? AND id <= ? ORDER BY id", (int(after_id), int(upto_id)))
        return [RecallEvent(int(r[0]), float(r[1]), r[2], r[3], r[4], r[5], r[6],
                            None if r[7] is None else float(r[7]), r[8], r[9]) for r in rows]

    def health_since(self, ts: float) -> list[HealthRow]:
        rows = self.conn.execute(
            "SELECT id,ts,pid,platform,prefetch_n,injected_n,empty_n,embed_fail_n,fts_fallback_n,"
            "timeout_n,p95_ms,last_error_class,snapshot_run FROM health WHERE ts >= ? ORDER BY id",
            (float(ts),))
        return [HealthRow(int(r[0]), float(r[1] or 0), r[2], r[3], int(r[4] or 0), int(r[5] or 0),
                          int(r[6] or 0), int(r[7] or 0), int(r[8] or 0), int(r[9] or 0), r[10],
                          r[11], r[12]) for r in rows]

    # ── writes (rw only) ──
    def _rw(self) -> None:
        if self.mode != "rw":
            raise PermissionError("live.db opened read-only")

    def scrub_inbox(self, inbox_ids: Iterable[int], *, forget_memory_ids: Iterable[str] = ()) -> int:
        """Purged memory (F-5): its source inbox rows lose text/old_text/vec; forget requests for it
        lose their meta (the reason may quote the text). Rows stay (ids, status) for bookkeeping."""
        self._rw()
        ids = [int(i) for i in inbox_ids]
        mids = [str(m) for m in forget_memory_ids]
        n = 0
        with self.conn:
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                n += self.conn.execute(
                    f"UPDATE inbox SET text=NULL, old_text=NULL, vec=NULL WHERE id IN ({','.join('?' * len(chunk))}) "
                    "AND (text IS NOT NULL OR old_text IS NOT NULL OR vec IS NOT NULL)", chunk).rowcount
            for i in range(0, len(mids), 500):
                chunk = mids[i:i + 500]
                n += self.conn.execute(
                    f"UPDATE inbox SET meta_json=NULL WHERE op='forget' AND memory_id IN ({','.join('?' * len(chunk))}) "
                    "AND meta_json IS NOT NULL", chunk).rowcount
        return n

    def delete_consumed_before(self, ts: float) -> int:
        """Consumed/skipped inbox rows older than `ts` are deleted (their text is in Lance or was
        rejected). session_end markers carry no text and stay (settle shortcut)."""
        self._rw()
        with self.conn:
            return self.conn.execute(
                "DELETE FROM inbox WHERE status IN ('consumed','skipped') AND op != 'session_end' AND ts < ?",
                (float(ts),)).rowcount

    def requeue(self, run_ids: Iterable[str], *, ops: Iterable[str] | None = None) -> int:
        """`yume restore --reprocess`: rows consumed/skipped by the rolled-back runs are pending
        again, so the next dream folds them anew (F-3)."""
        self._rw()
        runs = [str(r) for r in run_ids]
        if not runs:
            return 0
        q = (f"UPDATE inbox SET status='pending', consumed_run=NULL WHERE consumed_run IN "
             f"({','.join('?' * len(runs))}) AND status IN ('consumed','skipped') AND op != 'session_end'")
        args: list[Any] = list(runs)
        if ops is not None:
            ops = list(ops)
            if not ops:
                return 0
            q += f" AND op IN ({','.join('?' * len(ops))})"
            args += ops
        with self.conn:
            return self.conn.execute(q, args).rowcount

    def mark_consumed(self, ids: Iterable[int], run_id: str, *, status: str = "consumed") -> int:
        """R8-6. Idempotent. session_end rows are marked too (they stay as settle markers)."""
        self._rw()
        ids = [int(i) for i in ids]
        n = 0
        with self.conn:
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                cur = self.conn.execute(
                    f"UPDATE inbox SET status=?, consumed_run=? WHERE id IN ({','.join('?' * len(chunk))}) "
                    "AND status='pending'", (status, run_id, *chunk))
                n += cur.rowcount
        return n
