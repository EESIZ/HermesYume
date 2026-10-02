"""live.db writer shared by every provider instance of a process (PLAN-v2 §2.4, CONTRACTS §6).

- WAL + busy_timeout (``live_busy_timeout_ms``); on lock/IO error the write is dropped and the
  error class is recorded for the next health row. Nothing here ever raises into Hermes.
- ``record()`` buffers recall events (flush every ``event_flush_every``, at sync_turn,
  session_end and shutdown). ``inbox()`` inserts immediately.
- ``pending()`` = same-day remember rows + pending forget ids, cached by (data_version, max id).
- ``health_tick()`` accumulates per-platform counters; a row is written every ``health_flush_s``
  and at shutdown.
"""

import json
import logging
import os
import sqlite3
import threading
import time

from . import config as _cfg
from . import live_schema

log = logging.getLogger("hermesyume.provider.live")

_EVENT_SQL = ("INSERT INTO recall_events(ts, session_id, platform, turn_no, memory_id, kind, cos, "
              "mode, snapshot_run) VALUES(?,?,?,?,?,?,?,?,?)")
_INBOX_COLS = ("ts", "session_id", "platform", "op", "text", "old_text", "kind", "pin",
               "memory_id", "target", "vec", "embed_model", "meta_json")
_COUNTERS = ("prefetch_n", "injected_n", "empty_n", "embed_fail_n", "fts_fallback_n", "timeout_n")


def _p95(values):
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))
    return int(round(s[k]))


class LiveWriter:
    def __init__(self, hermes_home):
        self.hermes_home = str(hermes_home)
        self.path = os.path.join(_cfg.data_dir(hermes_home), "live.db")
        self._lock = threading.RLock()
        self._conn = None
        self._buf = []
        self._busy_ms = 300
        self._flush_every = 10
        self._health_every = 300.0
        self._last_health = time.monotonic()
        self._health = {}            # platform -> {counter: n, "lat": [ms], "err": str|None, "run": str|None}
        self._pending_key = None
        self._pending_val = None
        self._retry_at = 0.0
        self.last_error_class = None
        self.dropped_events = 0
        self.write_failures = 0

    def configure(self, cfg):
        try:
            self._busy_ms = int(cfg.get("live_busy_timeout_ms", 300))
            self._flush_every = max(1, int(cfg.get("event_flush_every", 10)))
            self._health_every = float(cfg.get("health_flush_s", 300))
        except Exception:
            pass

    # ── connection ──
    def _connect(self):
        if self._conn is not None:
            return self._conn
        if time.monotonic() < self._retry_at:
            raise sqlite3.OperationalError("live.db retry backoff")
        conn = None
        try:
            if not os.path.isdir(os.path.dirname(self.path)):
                raise FileNotFoundError("hermesyume data dir missing")
            new = not os.path.exists(self.path)
            conn = sqlite3.connect(self.path, timeout=self._busy_ms / 1000.0, check_same_thread=False)
            conn.execute("PRAGMA busy_timeout=%d" % self._busy_ms)
            # Apply the DDL only when needed: it writes (user_version), and connecting must keep
            # working (reads) while another process holds the write lock.
            uv = conn.execute("PRAGMA user_version").fetchone()[0]
            have = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if uv != live_schema.USER_VERSION or not set(live_schema.TABLES) <= have:
                live_schema.apply_schema(conn)
            conn.execute("PRAGMA synchronous=NORMAL")   # WAL: durable enough, keeps hooks < 10 ms
            if new:
                try:
                    os.chmod(self.path, 0o600)
                except OSError:
                    pass
            self._conn = conn
            return conn
        except Exception:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            self._retry_at = time.monotonic() + 1.0
            raise

    def _fail(self, where, exc):
        self.last_error_class = "%s:%s" % (where, type(exc).__name__)
        self.write_failures += 1
        try:
            if self._conn is not None:
                self._conn.rollback()
        except Exception:
            pass

    # ── recall events ──
    def record(self, *, session_id, platform, turn_no, memory_id, kind, cos, mode, snapshot_run,
               ts=None):
        try:
            with self._lock:
                self._buf.append((float(_cfg.now() if ts is None else ts), session_id or None,
                                  platform or None, turn_no, str(memory_id), kind,
                                  None if cos is None else float(cos), mode, snapshot_run or None))
                if len(self._buf) >= self._flush_every:
                    self._flush_locked()
        except Exception as e:  # pragma: no cover - defensive
            self._fail("record", e)

    def flush(self):
        try:
            with self._lock:
                self._flush_locked()
        except Exception as e:  # pragma: no cover - defensive
            self._fail("flush", e)

    def _flush_locked(self):
        if not self._buf:
            return
        rows = list(self._buf)
        self._buf.clear()
        try:
            conn = self._connect()
            conn.executemany(_EVENT_SQL, rows)
            conn.commit()
        except Exception as e:
            self.dropped_events += len(rows)
            self._fail("live_write", e)

    # ── inbox ──
    def inbox(self, op, **fields):
        """Insert one inbox row now; returns its id (None when dropped or ignored)."""
        try:
            row = {k: fields.get(k) for k in _INBOX_COLS}
            row["op"] = op
            if row["ts"] is None:
                row["ts"] = _cfg.now()
            if isinstance(row.get("meta_json"), dict):
                row["meta_json"] = json.dumps(row["meta_json"], ensure_ascii=False, sort_keys=True)
            row["pin"] = 1 if row.get("pin") else 0
            sql = "INSERT INTO inbox(%s) VALUES(%s)" % (", ".join(_INBOX_COLS),
                                                       ",".join("?" * len(_INBOX_COLS)))
            if op == "session_end":
                # one marker per session (ux_session_end); a later real end moves its time forward
                sql += (" ON CONFLICT(session_id) WHERE op='session_end' DO UPDATE SET ts=excluded.ts "
                        "WHERE excluded.ts > inbox.ts")
            with self._lock:
                try:
                    conn = self._connect()
                    cur = conn.execute(sql, [row[k] for k in _INBOX_COLS])
                    conn.commit()
                    self._pending_key = None
                    return cur.lastrowid if cur.rowcount else None
                except Exception as e:
                    self._fail("live_inbox", e)
                    return None
        except Exception as e:  # pragma: no cover - defensive
            self._fail("live_inbox", e)
            return None

    def injected_ids(self, session_id, limit=500):
        """Memory ids already injected into `session_id` (newest `limit` events). Empty on error."""
        if not session_id:
            return set()
        try:
            with self._lock:
                self._flush_locked()
                conn = self._connect()
                rows = conn.execute("SELECT memory_id FROM recall_events WHERE session_id=? AND "
                                    "kind='injected' ORDER BY id DESC LIMIT ?",
                                    (str(session_id), int(limit))).fetchall()
            return {r[0] for r in rows if r[0]}
        except Exception as e:
            self.last_error_class = "live_read:%s" % type(e).__name__
            return set()

    def pending(self):
        """{"remember": [{"id","memory_id","text","kind","pin","vec","embed_model","ts","session_id"}],
        "forget_ids": set[str]}. Empty on any error."""
        empty = {"remember": [], "forget_ids": set()}
        try:
            with self._lock:
                conn = self._connect()
                dv = conn.execute("PRAGMA data_version").fetchone()[0]
                mx = conn.execute("SELECT COALESCE(MAX(id), 0) FROM inbox").fetchone()[0]
                key = (dv, mx)
                if self._pending_key == key and self._pending_val is not None:
                    return self._pending_val
                forget = {r[0] for r in conn.execute(
                    "SELECT memory_id FROM inbox WHERE op='forget' AND status='pending' "
                    "AND memory_id IS NOT NULL")}
                rem = []
                for r in conn.execute("SELECT id, ts, text, kind, pin, vec, embed_model, session_id, "
                                      "meta_json FROM inbox WHERE op='remember' AND status='pending' "
                                      "ORDER BY id"):
                    mid = "inbox:%d" % r[0]
                    if mid in forget or not r[2]:
                        continue
                    try:
                        vec = live_schema.blob_to_vec(r[5])
                    except Exception:
                        vec = None
                    vu = None
                    try:
                        meta = json.loads(r[8]) if r[8] else {}
                        if isinstance(meta, dict) and meta.get("valid_until"):
                            vu = _cfg.parse_iso(str(meta["valid_until"]), end_of_day=True)
                    except Exception:
                        vu = None
                    rem.append({"id": r[0], "memory_id": mid, "ts": r[1], "text": r[2],
                                "kind": r[3], "pin": bool(r[4]), "vec": vec,
                                "embed_model": r[6], "session_id": r[7], "valid_until": vu})
                val = {"remember": rem, "forget_ids": forget}
                self._pending_key, self._pending_val = key, val
                return val
        except Exception as e:
            self.last_error_class = "live_read:%s" % type(e).__name__
            return empty

    # ── health ──
    def health_tick(self, platform, **counters):
        try:
            with self._lock:
                h = self._health.setdefault(platform or "", {"lat": [], "err": None, "run": None})
                for k in _COUNTERS:
                    if counters.get(k):
                        h[k] = h.get(k, 0) + int(counters[k])
                if counters.get("latency_ms") is not None:
                    h["lat"].append(float(counters["latency_ms"]))
                if counters.get("error_class"):
                    h["err"] = counters["error_class"]
                if counters.get("snapshot_run"):
                    h["run"] = counters["snapshot_run"]
                if time.monotonic() - self._last_health >= self._health_every:
                    self._health_flush_locked()
        except Exception as e:  # pragma: no cover - defensive
            self._fail("health", e)

    def health_flush(self):
        try:
            with self._lock:
                self._health_flush_locked()
        except Exception as e:  # pragma: no cover - defensive
            self._fail("health", e)

    def _health_flush_locked(self):
        self._last_health = time.monotonic()
        if not self._health:
            return
        rows = []
        for plat, h in sorted(self._health.items()):
            err = h.get("err") or self.last_error_class
            if not any(h.get(k) for k in _COUNTERS) and not err:
                continue
            rows.append((_cfg.now(), os.getpid(), plat or None, h.get("prefetch_n", 0),
                         h.get("injected_n", 0), h.get("empty_n", 0), h.get("embed_fail_n", 0),
                         h.get("fts_fallback_n", 0), h.get("timeout_n", 0), _p95(h["lat"]),
                         err, h.get("run")))
        self._health = {}
        if not rows:
            return
        try:
            conn = self._connect()
            conn.executemany("INSERT INTO health(ts, pid, platform, prefetch_n, injected_n, empty_n, "
                             "embed_fail_n, fts_fallback_n, timeout_n, p95_ms, last_error_class, "
                             "snapshot_run) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            conn.commit()
            self.last_error_class = None
        except Exception as e:
            self._fail("live_health", e)

    def close(self):
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None


_LOCK = threading.Lock()
_WRITERS = {}


def get_writer(hermes_home):
    path = os.path.join(_cfg.data_dir(hermes_home), "live.db")
    with _LOCK:
        w = _WRITERS.get(path)
        if w is None:
            w = LiveWriter(hermes_home)
            _WRITERS[path] = w
        return w


def reset_all():
    """Tests: flush nothing, close and forget every writer."""
    with _LOCK:
        ws = list(_WRITERS.values())
        _WRITERS.clear()
    for w in ws:
        w.close()
