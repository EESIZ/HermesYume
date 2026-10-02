"""serving/recall.sqlite DDL — the ONLY copy (PLAN-v2 §2.5). Stdlib only, no relative imports.

Written by the dream (``hermesyume.export`` loads this file by path), read-only by the provider
(``from . import serving_schema``). The file is built as ``serving/recall.<run_id>.sqlite`` and
atomically ``os.replace``d onto ``serving/recall.sqlite``.

Column conventions:
- ``status`` ∈ SERVING_STATUSES (forgotten / quarantined / candidate rows are never exported)
- ``pinned`` 0/1; ``event_time``/``valid_until`` epoch seconds (REAL) or NULL
- ``strength`` computed at export time (hermesyume.strength); ``refs`` JSON array text
- ``core_sha`` = corefmt.core_sha(entry) for core copies (any in_core state), else NULL
- ``vec`` float32 LE × dim (1536); ``vec256`` float32 LE × 256 = first 256 dims re-normalized
- ``pins``: pinned rows with status='active'; ``label`` = leading "**…**" label or subject;
  ``core_target`` "memory"|"user"|NULL
- ``meta`` keys: META_KEYS (values stored as TEXT)
"""

import sqlite3
import sys
from array import array

SERVING_VERSION = 1
VEC256_DIM = 256
SERVING_STATUSES = ("active", "superseded", "expired", "dormant")
META_KEYS = ("embed_model", "dim", "run_id", "lance_version", "built_at", "count")

DDL = """
CREATE TABLE items(id TEXT PRIMARY KEY, text TEXT, subject TEXT, kind TEXT, tier TEXT,
                   status TEXT,
                   pinned INTEGER, core_sha TEXT, event_time REAL, valid_until REAL,
                   strength REAL, refs TEXT,
                   vec256 BLOB,
                   vec BLOB);
CREATE VIRTUAL TABLE items_fts USING fts5(id UNINDEXED, text, subject, tokenize='trigram');
CREATE TABLE pins(id TEXT PRIMARY KEY, text TEXT, label TEXT, core_target TEXT);
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
"""

ITEM_COLUMNS = ("id", "text", "subject", "kind", "tier", "status", "pinned", "core_sha",
                "event_time", "valid_until", "strength", "refs", "vec256", "vec")
PIN_COLUMNS = ("id", "text", "label", "core_target")


def create_schema(conn):
    """For a NEW file only (no IF NOT EXISTS: export always builds a fresh file)."""
    conn.executescript(DDL)
    conn.commit()


def open_readonly(path, timeout_s=0.5):
    """Provider read path: URI mode=ro + query_only. Raises sqlite3.Error if missing/corrupt."""
    conn = sqlite3.connect("file:%s?mode=ro" % str(path), uri=True, timeout=timeout_s,
                           check_same_thread=False)
    conn.execute("PRAGMA query_only=ON")
    return conn


def read_meta(conn):
    return {k: v for k, v in conn.execute("SELECT key, value FROM meta")}


def vec_to_blob(vec):
    a = array("f", [float(x) for x in vec])
    if sys.byteorder != "little":
        a.byteswap()
    return a.tobytes()


def blob_to_vec(blob):
    if not blob:
        return None
    if len(blob) % 4:
        raise ValueError("vector blob length not a multiple of 4")
    a = array("f")
    a.frombytes(bytes(blob))
    if sys.byteorder != "little":
        a.byteswap()
    return a.tolist()
