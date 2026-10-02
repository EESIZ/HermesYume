"""ledger.db — dream-only bookkeeping (PLAN-v2 §2.3). Rollback-journal mode (no WAL) so a
read-only open never creates side files. Single writer (under dream.lock)."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from .types import (AuditRow, CoreSeenRow, LedgerDelta, MdFileState, RunRecord, WindowState,
                    Watermark)

SCHEMA_VERSION = "2"

# §2.3 verbatim (IF NOT EXISTS added for idempotent init).
LEDGER_DDL = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS lineage_wm(root_session_id TEXT PRIMARY KEY, last_ts REAL, last_id INTEGER, updated_run TEXT);
CREATE TABLE IF NOT EXISTS session_root(session_id TEXT PRIMARY KEY, root_session_id TEXT);
CREATE TABLE IF NOT EXISTS md_files(path TEXT PRIMARY KEY, sha256 TEXT, processed_bytes INTEGER,
                      prefix_sha256 TEXT, status TEXT, run_id TEXT);
CREATE TABLE IF NOT EXISTS windows(window_id TEXT PRIMARY KEY, source TEXT, root_session_id TEXT,
                     first_id INTEGER, last_id INTEGER, last_ts REAL,
                     status TEXT CHECK(status IN('ok','empty','failed','quarantined')),
                     attempts INTEGER DEFAULT 0, last_error TEXT, run_id TEXT, n_claims INTEGER);
CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, started_at REAL, finished_at REAL,
                  mode TEXT /*live|dry|migrate*/, now_override REAL,
                  status TEXT /*planned|committed|failed|dry|held*/,
                  lance_version_before INTEGER, lance_version_after INTEGER,
                  wm_before_json TEXT, stats_json TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS core_seen(target TEXT, entry_sha TEXT, text TEXT, memory_id TEXT,
                       first_seen_run TEXT, last_seen_run TEXT, present INTEGER,
                       PRIMARY KEY(target, entry_sha));
CREATE TABLE IF NOT EXISTS fold_cursor(name TEXT PRIMARY KEY, last_id INTEGER);
CREATE TABLE IF NOT EXISTS audit(ts REAL, run_id TEXT, op TEXT, memory_id TEXT, detail TEXT);
"""

_RUN_FIELDS = ("run_id", "started_at", "finished_at", "mode", "now_override", "status",
               "lance_version_before", "lance_version_after", "wm_before_json", "stats_json", "error")


class MetaMismatch(RuntimeError):
    """ledger meta embed_model/dim differ from config (or ledger not initialized): write nothing, alert."""


class LedgerReadOnly(RuntimeError):
    pass


class Ledger:
    def __init__(self, conn: sqlite3.Connection, path: Path | None, readonly: bool):
        self.conn = conn
        self.path = path
        self.readonly = readonly
        self._depth = 0

    # ── open / close ──
    @classmethod
    def open(cls, path: str | os.PathLike, *, readonly: bool = False) -> "Ledger":
        """readonly: existing file → URI mode=ro; missing file → empty in-memory ledger (dry-run
        before init). Writer: creates the file 0600 and applies the DDL."""
        p = Path(path)
        if readonly:
            if p.exists():
                conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, isolation_level=None,
                                       timeout=30)
                conn.execute("PRAGMA query_only=ON")
            else:
                conn = sqlite3.connect(":memory:", isolation_level=None)
                conn.executescript(LEDGER_DDL)
            return cls(conn, p, True)
        p.parent.mkdir(parents=True, exist_ok=True)
        new = not p.exists()
        conn = sqlite3.connect(str(p), isolation_level=None, timeout=30)
        if new:
            os.chmod(p, 0o600)
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.executescript(LEDGER_DDL)
        return cls(conn, p, False)

    @classmethod
    def from_paths(cls, paths: Any, *, readonly: bool = False) -> "Ledger":
        return cls.open(paths.ledger_db, readonly=readonly)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _w(self) -> None:
        if self.readonly:
            raise LedgerReadOnly("ledger opened read-only")

    @contextmanager
    def transaction(self) -> Iterator["Ledger"]:
        """BEGIN IMMEDIATE … COMMIT (nested calls join the outer transaction)."""
        self._w()
        if self._depth == 0:
            self.conn.execute("BEGIN IMMEDIATE")
        self._depth += 1
        try:
            yield self
        except BaseException:
            self._depth -= 1
            if self._depth == 0:
                self.conn.execute("ROLLBACK")
            raise
        else:
            self._depth -= 1
            if self._depth == 0:
                self.conn.execute("COMMIT")

    # ── meta ──
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        r = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else default

    def set_meta(self, key: str, value: Any) -> None:
        self._w()
        self.conn.execute("INSERT INTO meta(key,value) VALUES(?,?) "
                          "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))

    def all_meta(self) -> dict[str, str]:
        return {k: v for k, v in self.conn.execute("SELECT key, value FROM meta")}

    def init_meta(self, embed_model: str, dim: int, *, force: bool = False) -> None:
        """`yume init`: record schema_version/embed_model/dim. Refuses to change them unless force
        (model changes go through `yume reembed`)."""
        cur = self.all_meta()
        if cur.get("embed_model") and not force:
            self.check_meta(embed_model, dim)
            return
        with self.transaction():
            self.set_meta("schema_version", SCHEMA_VERSION)
            self.set_meta("embed_model", embed_model)
            self.set_meta("dim", int(dim))
            if "created_at" not in cur:
                self.set_meta("created_at", time.time())

    def check_meta(self, embed_model: str, dim: int, *, allow_uninitialized: bool = False) -> None:
        cur = self.all_meta()
        if not cur.get("embed_model"):
            if allow_uninitialized:
                return
            raise MetaMismatch("ledger 미초기화 (yume init 필요)")
        if cur.get("embed_model") != embed_model or str(cur.get("dim")) != str(int(dim)):
            raise MetaMismatch(
                f"임베딩 모델/차원 불일치: ledger={cur.get('embed_model')}/{cur.get('dim')} "
                f"config={embed_model}/{dim} (모델 변경은 yume reembed)")
        if cur.get("schema_version") not in (None, SCHEMA_VERSION):
            raise MetaMismatch(f"ledger schema_version {cur.get('schema_version')} != {SCHEMA_VERSION}")

    # ── lineage watermarks / roots ──
    def get_wm(self, root: str) -> Watermark | None:
        r = self.conn.execute("SELECT root_session_id,last_ts,last_id,updated_run FROM lineage_wm "
                              "WHERE root_session_id=?", (root,)).fetchone()
        return Watermark(r[0], float(r[1]), int(r[2]), r[3]) if r else None

    def all_wms(self) -> dict[str, Watermark]:
        return {r[0]: Watermark(r[0], float(r[1]), int(r[2]), r[3]) for r in self.conn.execute(
            "SELECT root_session_id,last_ts,last_id,updated_run FROM lineage_wm")}

    def set_wm(self, root: str, last_ts: float, last_id: int, run_id: str | None) -> None:
        self._w()
        self.conn.execute(
            "INSERT INTO lineage_wm(root_session_id,last_ts,last_id,updated_run) VALUES(?,?,?,?) "
            "ON CONFLICT(root_session_id) DO UPDATE SET last_ts=excluded.last_ts, "
            "last_id=excluded.last_id, updated_run=excluded.updated_run",
            (root, float(last_ts), int(last_id), run_id))

    def get_root(self, session_id: str) -> str | None:
        r = self.conn.execute("SELECT root_session_id FROM session_root WHERE session_id=?",
                              (session_id,)).fetchone()
        return r[0] if r else None

    def all_roots(self) -> dict[str, str]:
        return {a: b for a, b in self.conn.execute("SELECT session_id, root_session_id FROM session_root")}

    def set_root(self, session_id: str, root: str) -> None:
        self._w()
        self.conn.execute("INSERT INTO session_root(session_id,root_session_id) VALUES(?,?) "
                          "ON CONFLICT(session_id) DO UPDATE SET root_session_id=excluded.root_session_id",
                          (session_id, root))

    # ── md files ──
    def get_md(self, path: str) -> MdFileState | None:
        r = self.conn.execute("SELECT path,sha256,processed_bytes,prefix_sha256,status,run_id "
                              "FROM md_files WHERE path=?", (path,)).fetchone()
        return MdFileState(r[0], r[1], int(r[2] or 0), r[3], r[4], r[5]) if r else None

    def all_md(self) -> dict[str, MdFileState]:
        return {r[0]: MdFileState(r[0], r[1], int(r[2] or 0), r[3], r[4], r[5]) for r in
                self.conn.execute("SELECT path,sha256,processed_bytes,prefix_sha256,status,run_id FROM md_files")}

    def upsert_md(self, s: MdFileState) -> None:
        self._w()
        self.conn.execute(
            "INSERT INTO md_files(path,sha256,processed_bytes,prefix_sha256,status,run_id) "
            "VALUES(?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256, "
            "processed_bytes=excluded.processed_bytes, prefix_sha256=excluded.prefix_sha256, "
            "status=excluded.status, run_id=excluded.run_id",
            (s.path, s.sha256, int(s.processed_bytes), s.prefix_sha256, s.status, s.run_id))

    # ── windows ──
    _WCOLS = ("window_id,source,root_session_id,first_id,last_id,last_ts,status,attempts,"
              "last_error,run_id,n_claims")

    @staticmethod
    def _wrow(r) -> WindowState:
        return WindowState(r[0], r[1], r[2], int(r[3]), int(r[4]), float(r[5] or 0.0), r[6],
                           int(r[7] or 0), r[8], r[9], int(r[10] or 0))

    def get_window(self, window_id: str) -> WindowState | None:
        r = self.conn.execute(f"SELECT {self._WCOLS} FROM windows WHERE window_id=?",
                              (window_id,)).fetchone()
        return self._wrow(r) if r else None

    def windows(self, *, status: str | None = None, root: str | None = None) -> list[WindowState]:
        q, args = f"SELECT {self._WCOLS} FROM windows WHERE 1=1", []
        if status:
            q += " AND status=?"
            args.append(status)
        if root:
            q += " AND root_session_id=?"
            args.append(root)
        q += " ORDER BY last_ts, first_id"
        return [self._wrow(r) for r in self.conn.execute(q, args)]

    def upsert_window(self, w: WindowState) -> None:
        self._w()
        if w.status in ("ok", "empty", "quarantined"):
            # the span is done: earlier failed attempts under other window ids are history (F-32)
            self.conn.execute("DELETE FROM windows WHERE status='failed' AND source IS ? AND "
                              "root_session_id IS ? AND first_id=? AND window_id != ?",
                              (w.source, w.root_session_id, int(w.first_id), w.window_id))
        self.conn.execute(
            f"INSERT INTO windows({self._WCOLS}) VALUES(?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(window_id) DO UPDATE SET source=excluded.source, "
            "root_session_id=excluded.root_session_id, first_id=excluded.first_id, "
            "last_id=excluded.last_id, last_ts=excluded.last_ts, status=excluded.status, "
            "attempts=excluded.attempts, last_error=excluded.last_error, run_id=excluded.run_id, "
            "n_claims=excluded.n_claims",
            (w.window_id, w.source, w.root_session_id, int(w.first_id), int(w.last_id),
             float(w.last_ts), w.status, int(w.attempts), w.last_error, w.run_id, int(w.n_claims)))

    # ── runs ──
    @staticmethod
    def _rrow(r) -> RunRecord:
        return RunRecord(*r)

    def insert_run(self, rec: RunRecord) -> None:
        self._w()
        self.conn.execute(f"INSERT INTO runs({','.join(_RUN_FIELDS)}) VALUES({','.join('?' * len(_RUN_FIELDS))}) "
                          "ON CONFLICT(run_id) DO NOTHING",
                          tuple(getattr(rec, f) for f in _RUN_FIELDS))

    def update_run(self, run_id: str, **fields: Any) -> None:
        self._w()
        bad = set(fields) - set(_RUN_FIELDS)
        if bad:
            raise KeyError(f"unknown run fields {bad}")
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        self.conn.execute(f"UPDATE runs SET {sets} WHERE run_id=?", (*fields.values(), run_id))

    def get_run(self, run_id: str) -> RunRecord | None:
        r = self.conn.execute(f"SELECT {','.join(_RUN_FIELDS)} FROM runs WHERE run_id=?",
                              (run_id,)).fetchone()
        return self._rrow(r) if r else None

    def runs(self, *, status: str | None = None, limit: int = 20) -> list[RunRecord]:
        q, args = f"SELECT {','.join(_RUN_FIELDS)} FROM runs", []
        if status:
            q += " WHERE status=?"
            args.append(status)
        q += " ORDER BY started_at DESC, run_id DESC LIMIT ?"
        args.append(int(limit))
        return [self._rrow(r) for r in self.conn.execute(q, args)]

    def planned_runs(self) -> list[RunRecord]:
        """Runs to replay (R8): status='planned', oldest first."""
        return list(reversed(self.runs(status="planned", limit=1000)))

    # ── core_seen ──
    def core_seen(self, target: str | None = None) -> dict[tuple[str, str], CoreSeenRow]:
        q, args = ("SELECT target,entry_sha,text,memory_id,first_seen_run,last_seen_run,present "
                   "FROM core_seen"), []
        if target:
            q += " WHERE target=?"
            args.append(target)
        return {(r[0], r[1]): CoreSeenRow(r[0], r[1], r[2], r[3], r[4], r[5], bool(r[6]))
                for r in self.conn.execute(q, args)}

    def upsert_core_seen(self, c: CoreSeenRow) -> None:
        self._w()
        self.conn.execute(
            "INSERT INTO core_seen(target,entry_sha,text,memory_id,first_seen_run,last_seen_run,present) "
            "VALUES(?,?,?,?,?,?,?) ON CONFLICT(target,entry_sha) DO UPDATE SET text=excluded.text, "
            "memory_id=excluded.memory_id, last_seen_run=excluded.last_seen_run, present=excluded.present",
            (c.target, c.entry_sha, c.text, c.memory_id, c.first_seen_run, c.last_seen_run,
             1 if c.present else 0))

    # ── fold cursors ──
    def get_cursor(self, name: str, default: int = 0) -> int:
        r = self.conn.execute("SELECT last_id FROM fold_cursor WHERE name=?", (name,)).fetchone()
        return int(r[0]) if r and r[0] is not None else default

    def set_cursor(self, name: str, last_id: int) -> None:
        self._w()
        self.conn.execute("INSERT INTO fold_cursor(name,last_id) VALUES(?,?) "
                          "ON CONFLICT(name) DO UPDATE SET last_id=excluded.last_id", (name, int(last_id)))

    # ── audit (never stores forgotten text) ──
    def add_audit(self, a: AuditRow) -> None:
        self._w()
        self.conn.execute("INSERT INTO audit(ts,run_id,op,memory_id,detail) VALUES(?,?,?,?,?)",
                          (a.ts, a.run_id, a.op, a.memory_id, a.detail))

    def audits(self, memory_id: str | None = None) -> list[AuditRow]:
        q, args = "SELECT ts,run_id,op,memory_id,detail FROM audit", []
        if memory_id:
            q += " WHERE memory_id=?"
            args.append(memory_id)
        return [AuditRow(*r) for r in self.conn.execute(q + " ORDER BY ts", args)]

    # ── R8-5: one transaction ──
    def apply_delta(self, run_id: str, delta: LedgerDelta, *, lance_version_after: int | None,
                    stats_json: str | None = None, finished_at: float | None = None,
                    status: str = "committed") -> None:
        """Watermarks, roots, windows, md offsets, fold cursors, core_seen, audit and the run's
        status/lance_version_after — atomically. Idempotent (all UPSERT; audit rows deduped on
        (run_id, op, memory_id, detail))."""
        with self.transaction():
            for root, (ts, mid) in delta.watermarks.items():
                self.set_wm(root, ts, mid, run_id)
            for sid, root in delta.session_roots.items():
                self.set_root(sid, root)
            for w in delta.windows:
                self.upsert_window(w)
            for m in delta.md_files:
                self.upsert_md(m)
            for name, last_id in delta.cursors.items():
                self.set_cursor(name, last_id)
            for c in delta.core_seen:
                self.upsert_core_seen(c)
            for a in delta.audit:
                dup = self.conn.execute(
                    "SELECT 1 FROM audit WHERE run_id IS ? AND op IS ? AND memory_id IS ? AND detail IS ?",
                    (a.run_id, a.op, a.memory_id, a.detail)).fetchone()
                if not dup:
                    self.add_audit(a)
            fields: dict[str, Any] = {"status": status, "lance_version_after": lance_version_after,
                                      "finished_at": finished_at if finished_at is not None else time.time()}
            if stats_json is not None:
                fields["stats_json"] = stats_json
            self.update_run(run_id, **fields)

    # ── backup ──
    def backup(self, dest_dir: str | os.PathLike, *, keep: int = 7, stamp: str | None = None) -> Path:
        """sqlite backup API → <dest_dir>/ledger-<stamp>.db (0600); keeps the newest `keep`."""
        d = Path(dest_dir)
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
        stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
        dst = d / f"ledger-{stamp}.db"
        out = sqlite3.connect(str(dst))
        try:
            self.conn.backup(out)
        finally:
            out.close()
        os.chmod(dst, 0o600)
        olds = sorted(d.glob("ledger-*.db"))
        for old in olds[:-keep] if keep > 0 else []:
            old.unlink(missing_ok=True)
        return dst

    def wm_snapshot_json(self) -> str:
        """runs.wm_before_json: {"v": 2, "wm": {root: [last_ts, last_id]}, "md": {path: state}}
        (md offsets too, so `restore --reprocess` can rewind md files; v1 was the bare wm map)."""
        md = {k: [v.sha256, int(v.processed_bytes), v.prefix_sha256, v.status, v.run_id]
              for k, v in self.all_md().items()}
        return json.dumps({"v": 2, "wm": {k: [v.last_ts, v.last_id] for k, v in self.all_wms().items()},
                           "md": md}, sort_keys=True)

    def restore_wms(self, wm_json: str, *, rolled_back_run: str,
                    undone_runs: list[str] | tuple[str, ...] = ()) -> None:
        """`yume restore --run X --reprocess`: watermarks and md offsets ← X.wm_before_json, and the
        window rows of X and every later undone run are dropped so those spans are rebuilt and
        re-extracted (DEVIATIONS F-3)."""
        data = json.loads(wm_json or "{}")
        wm, md = (data.get("wm") or {}, data.get("md")) if data.get("v") == 2 else (data, None)
        runs = list(dict.fromkeys([rolled_back_run, *undone_runs]))
        with self.transaction():
            self.conn.execute("DELETE FROM lineage_wm")
            for root, (ts, mid) in wm.items():
                self.set_wm(root, ts, mid, f"restore:{rolled_back_run}")
            for rid in runs:
                self.conn.execute("DELETE FROM windows WHERE run_id=?", (rid,))
            if md is not None:
                self.conn.execute("DELETE FROM md_files")
                for path, (sha, nbytes, prefix, status, run_id) in md.items():
                    self.upsert_md(MdFileState(path, sha, int(nbytes or 0), prefix, status, run_id))
            else:   # v1 snapshot: forget md files touched by the undone runs (re-read from 0)
                for rid in runs:
                    self.conn.execute("DELETE FROM md_files WHERE run_id=?", (rid,))

    def forget_core_seen_from(self, runs: list[str] | tuple[str, ...]) -> int:
        """`yume restore`: core entries first recorded by an undone run point at rows that no
        longer exist; dropping them lets R5 mirror those entries again."""
        n = 0
        with self.transaction():
            for rid in runs:
                n += self.conn.execute("DELETE FROM core_seen WHERE first_seen_run=?", (rid,)).rowcount
        return n

    # ── Lance lineage (reembed) ──
    def lance_epoch_min_rowid(self) -> int:
        """Runs whose runs.rowid is below this predate the current Lance directory (yume reembed
        swapped in a new store whose version numbers restart): they cannot be restored."""
        try:
            return int(self.get_meta("lance_epoch_min_rowid", "0") or 0)
        except ValueError:
            return 0

    def start_lance_epoch(self) -> int:
        nxt = int(self.conn.execute("SELECT COALESCE(MAX(rowid), 0) + 1 FROM runs").fetchone()[0])
        with self.transaction():
            self.set_meta("lance_epoch_min_rowid", nxt)
            self.set_meta("lance_epoch", int(self.get_meta("lance_epoch", "0") or 0) + 1)
        return nxt

    def run_rowid(self, run_id: str) -> int | None:
        r = self.conn.execute("SELECT rowid FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return int(r[0]) if r else None


def run_record_dict(rec: RunRecord) -> dict[str, Any]:
    return asdict(rec)
