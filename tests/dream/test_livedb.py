"""livedb.py + provider/_yume/live_schema.py (single DDL source) + blob codecs."""

import os
import sqlite3

import numpy as np
import pytest

from hermesyume.livedb import LiveDB, live_schema
from hermesyume.paths import load_provider_module
from hermesyume.types import LiveSnapshot
from tests.fakes import hash_embed
from tests.fixtures.hermes_home import tree_hash


def provider_insert(path, **row):
    """What the provider writes (its own connection)."""
    ls = live_schema()
    conn = ls.connect(path)
    cols = ",".join(row)
    conn.execute(f"INSERT INTO inbox({cols}) VALUES({','.join('?' * len(row))})", tuple(row.values()))
    conn.commit()
    conn.close()


def test_schema_exact(paths):
    db = LiveDB.open(paths, mode="rw")
    c = db.conn
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert c.execute("PRAGMA user_version").fetchone()[0] == 1
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"recall_events", "inbox", "health"}
    cols = lambda t: [r[1] for r in c.execute(f"PRAGMA table_info({t})")]
    assert cols("recall_events") == ["id", "ts", "session_id", "platform", "turn_no", "memory_id",
                                     "kind", "cos", "mode", "snapshot_run"]
    assert cols("inbox") == ["id", "ts", "session_id", "platform", "op", "text", "old_text", "kind",
                             "pin", "memory_id", "target", "vec", "embed_model", "meta_json",
                             "status", "consumed_run"]
    assert cols("health")[:3] == ["id", "ts", "pid"]
    with pytest.raises(sqlite3.IntegrityError):
        c.execute("INSERT INTO recall_events(ts, memory_id, kind) VALUES(1,'m','bogus')")
    with pytest.raises(sqlite3.IntegrityError):
        c.execute("INSERT INTO inbox(ts, op) VALUES(1,'bogus')")
    assert oct(os.stat(paths.live_db).st_mode & 0o777) == "0o600"
    db.close()


def test_session_end_unique_insert_or_ignore(paths):
    LiveDB.open(paths, mode="rw").close()
    ls = live_schema()
    conn = ls.connect(paths.live_db)
    for _ in range(3):
        conn.execute("INSERT OR IGNORE INTO inbox(ts, session_id, op) VALUES(1,'s1','session_end')")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM inbox WHERE op='session_end'").fetchone()[0] == 1
    conn.close()
    db = LiveDB.open(paths, mode="rw")
    assert db.session_end_ids() == {"s1": 1.0}          # marker time: settles messages up to it (F-8)
    db.close()


def test_inbox_vec_roundtrip_snapshot_and_consume(paths):
    LiveDB.open(paths, mode="rw").close()
    ls = live_schema()
    v = hash_embed("보라색 고래 7341")
    provider_insert(paths.live_db, ts=1.0, session_id="s", platform="telegram", op="remember",
                    text="보라색 고래 7341", kind="fact", pin=0, vec=ls.vec_to_blob(v),
                    embed_model="openai/text-embedding-3-small@1536", meta_json='{"src":"tool"}')
    provider_insert(paths.live_db, ts=2.0, op="forget", memory_id="abc")
    provider_insert(paths.live_db, ts=3.0, op="remember", text="벡터 없는 기억")
    db = LiveDB.open(paths, mode="rw")
    snap = db.snapshot()
    assert snap == LiveSnapshot(3, 0)
    items = db.inbox_range(0, 2)
    assert [i.op for i in items] == ["remember", "forget"]
    assert items[0].vec.dtype == np.float32 and np.allclose(items[0].vec, v, atol=1e-6)
    assert items[0].meta == {"src": "tool"} and items[1].vec is None and items[0].pin is False
    assert [i.id for i in db.inbox_range(0, 3, ops=["remember"])] == [1, 3]
    assert db.mark_consumed([1, 2], "r1") == 2
    assert db.mark_consumed([1, 2], "r1") == 0          # idempotent
    assert [i.id for i in db.pending_inbox()] == [3]
    assert db.inbox_range(0, 3, status="consumed")[0].consumed_run == "r1"
    db.close()


def test_recall_events_and_health(paths):
    db = LiveDB.open(paths, mode="rw")
    db.conn.execute("INSERT INTO recall_events(ts,session_id,platform,turn_no,memory_id,kind,cos,mode,snapshot_run) "
                    "VALUES(10,'s','telegram',1,'m1','injected',0.61,'vector','r0')")
    db.conn.execute("INSERT INTO health(ts,pid,platform,prefetch_n,injected_n,empty_n,embed_fail_n,"
                    "fts_fallback_n,timeout_n,p95_ms) VALUES(10,1,'cli',5,2,3,1,1,0,420)")
    db.conn.commit()
    ev = db.recall_range(0, db.snapshot().max_recall_id)
    assert ev[0].memory_id == "m1" and ev[0].cos == pytest.approx(0.61) and ev[0].mode == "vector"
    h = db.health_since(0)
    assert h[0].prefetch_n == 5 and h[0].p95_ms == 420
    db.close()


def test_pure_mode_touches_nothing(paths):
    LiveDB.open(paths, mode="rw").close()
    provider_insert(paths.live_db, ts=1.0, op="remember", text="x")
    before = tree_hash(paths.data_dir)
    assert not os.path.exists(str(paths.live_db) + "-shm")
    db = LiveDB.open(paths, mode="pure")
    assert db.snapshot().max_inbox_id == 1 and len(db.pending_inbox()) == 1
    with pytest.raises(PermissionError):
        db.mark_consumed([1], "r")
    db.close()
    assert tree_hash(paths.data_dir) == before
    assert sorted(os.listdir(paths.data_dir)) == sorted(before_names(before))


def before_names(tree):
    return {k.split("/")[0] for k in tree}


def test_missing_livedb_ro_and_pure_return_none(paths):
    assert LiveDB.open(paths, mode="pure") is None
    assert LiveDB.open(paths, mode="ro") is None
    assert not paths.live_db.exists()


def test_blob_codecs_agree():
    ls = load_provider_module("live_schema")
    ss = load_provider_module("serving_schema")
    v = hash_embed("코덱", 1536)
    b1, b2 = ls.vec_to_blob(v), ss.vec_to_blob(v)
    b3 = np.asarray(v, dtype="<f4").tobytes()
    assert b1 == b2 == b3 and len(b1) == 1536 * 4
    assert np.allclose(ls.blob_to_vec(b1), v, atol=1e-7) and ss.blob_to_vec(b"") is None
    assert ls.blob_dim(b1) == 1536


def test_serving_schema_ddl_builds():
    ss = load_provider_module("serving_schema")
    conn = sqlite3.connect(":memory:")
    ss.create_schema(conn)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(items)")]
    assert tuple(cols) == ss.ITEM_COLUMNS
    assert [r[1] for r in conn.execute("PRAGMA table_info(pins)")] == list(ss.PIN_COLUMNS)
    conn.execute("INSERT INTO items_fts(id, text, subject) VALUES('a','Orion 스테이징 포트 8081','포트')")
    assert conn.execute("SELECT id FROM items_fts WHERE items_fts MATCH '8081'").fetchone()[0] == "a"
    assert conn.execute("SELECT id FROM items_fts WHERE items_fts MATCH '스테이징'").fetchone()[0] == "a"
