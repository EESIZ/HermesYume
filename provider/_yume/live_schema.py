"""live.db DDL — the ONLY copy (PLAN-v2 §2.4). Stdlib only, no relative imports.

The provider imports it as ``from . import live_schema``; the dream loads this same file by path
(``hermesyume.paths.load_provider_module("live_schema")``).

Vector BLOB format (inbox.vec): float32 little-endian, no header, length = dim * 4.
"""

import sqlite3
import struct
from array import array
import sys

USER_VERSION = 1
TABLES = ("recall_events", "inbox", "health")
RECALL_KINDS = ("injected", "used", "tool_hit", "shadow")
RECALL_MODES = ("vector", "keyword", "inbox")
INBOX_OPS = ("remember", "forget", "core_add", "core_replace", "core_remove", "session_end")
INBOX_STATUSES = ("pending", "consumed", "skipped")

PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA user_version=1",
)

DDL = """
CREATE TABLE IF NOT EXISTS recall_events(
  id INTEGER PRIMARY KEY, ts REAL NOT NULL, session_id TEXT, platform TEXT, turn_no INTEGER,
  memory_id TEXT NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN('injected','used','tool_hit','shadow')),
  cos REAL, mode TEXT CHECK(mode IN('vector','keyword','inbox')), snapshot_run TEXT);
CREATE TABLE IF NOT EXISTS inbox(
  id INTEGER PRIMARY KEY, ts REAL NOT NULL, session_id TEXT, platform TEXT,
  op TEXT NOT NULL CHECK(op IN('remember','forget','core_add','core_replace','core_remove','session_end')),
  text TEXT, old_text TEXT, kind TEXT, pin INTEGER DEFAULT 0, memory_id TEXT, target TEXT,
  vec BLOB, embed_model TEXT, meta_json TEXT,
  status TEXT NOT NULL DEFAULT 'pending', consumed_run TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS ux_session_end ON inbox(session_id) WHERE op='session_end';
CREATE TABLE IF NOT EXISTS health(
  id INTEGER PRIMARY KEY, ts REAL, pid INTEGER, platform TEXT,
  prefetch_n INTEGER, injected_n INTEGER, empty_n INTEGER, embed_fail_n INTEGER,
  fts_fallback_n INTEGER, timeout_n INTEGER, p95_ms INTEGER,
  last_error_class TEXT, snapshot_run TEXT);
"""


def apply_schema(conn):
    """Idempotent: WAL, user_version, tables, index."""
    for p in PRAGMAS:
        conn.execute(p)
    conn.executescript(DDL)
    conn.commit()


def connect(path, readonly=False, busy_timeout_ms=300, create=True):
    """Writer: WAL + busy_timeout, schema ensured. Reader (readonly=True): URI mode=ro, never
    creates the file (raises sqlite3.OperationalError if missing)."""
    path = str(path)
    if readonly:
        conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True,
                               timeout=busy_timeout_ms / 1000.0, check_same_thread=False)
        conn.execute("PRAGMA query_only=ON")
    else:
        if not create:
            conn = sqlite3.connect("file:%s?mode=rw" % path, uri=True,
                                   timeout=busy_timeout_ms / 1000.0, check_same_thread=False)
        else:
            conn = sqlite3.connect(path, timeout=busy_timeout_ms / 1000.0, check_same_thread=False)
        apply_schema(conn)
    conn.execute("PRAGMA busy_timeout=%d" % int(busy_timeout_ms))
    return conn


def vec_to_blob(vec):
    """Sequence[float] → float32 little-endian bytes."""
    a = array("f", [float(x) for x in vec])
    if sys.byteorder != "little":
        a.byteswap()
    return a.tobytes()


def blob_to_vec(blob):
    """float32 little-endian bytes → list[float]; None/empty → None."""
    if not blob:
        return None
    if len(blob) % 4:
        raise ValueError("vector blob length not a multiple of 4")
    a = array("f")
    a.frombytes(bytes(blob))
    if sys.byteorder != "little":
        a.byteswap()
    return a.tolist()


def blob_dim(blob):
    return 0 if not blob else len(blob) // struct.calcsize("<f")
