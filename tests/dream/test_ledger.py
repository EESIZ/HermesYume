"""ledger.py: §2.3 DDL, meta guard, watermarks, windows, runs, atomic + idempotent apply_delta."""

import sqlite3

import pytest

from hermesyume.ledger import Ledger, LedgerReadOnly, MetaMismatch
from hermesyume.types import (AuditRow, CoreSeenRow, LedgerDelta, MdFileState, RunRecord,
                              WindowState)
from tests.fixtures.hermes_home import tree_hash

MODEL = "openai/text-embedding-3-small@1536"


@pytest.fixture
def led(tmp_path):
    l = Ledger.open(tmp_path / "ledger.db")
    yield l
    l.close()


def test_ddl_tables_and_columns(led):
    tables = {r[0] for r in led.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"meta", "lineage_wm", "session_root", "md_files", "windows", "runs",
                      "core_seen", "fold_cursor", "audit"}
    cols = lambda t: [r[1] for r in led.conn.execute(f"PRAGMA table_info({t})")]
    assert cols("lineage_wm") == ["root_session_id", "last_ts", "last_id", "updated_run"]
    assert cols("windows") == ["window_id", "source", "root_session_id", "first_id", "last_id",
                               "last_ts", "status", "attempts", "last_error", "run_id", "n_claims"]
    assert cols("runs") == ["run_id", "started_at", "finished_at", "mode", "now_override", "status",
                            "lance_version_before", "lance_version_after", "wm_before_json",
                            "stats_json", "error"]
    assert cols("core_seen") == ["target", "entry_sha", "text", "memory_id", "first_seen_run",
                                 "last_seen_run", "present"]
    assert led.conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_window_status_check_constraint(led):
    with pytest.raises(sqlite3.IntegrityError):
        led.upsert_window(WindowState("w", "statedb", "r", 1, 2, 1.0, "pending"))


def test_meta_init_and_guard(led):
    with pytest.raises(MetaMismatch):
        led.check_meta(MODEL, 1536)
    led.check_meta(MODEL, 1536, allow_uninitialized=True)
    led.init_meta(MODEL, 1536)
    led.check_meta(MODEL, 1536)
    with pytest.raises(MetaMismatch, match="reembed"):
        led.check_meta("openai/text-embedding-3-large@3072", 3072)
    with pytest.raises(MetaMismatch):
        led.init_meta("openai/other@1536", 1536)       # refuses silent model change
    assert led.get_meta("schema_version") == "2"


def test_watermarks_roots_cursors(led):
    assert led.get_wm("root") is None
    led.set_wm("root", 100.5, 7, "r1")
    led.set_wm("root", 200.0, 9, "r2")
    wm = led.get_wm("root")
    assert (wm.last_ts, wm.last_id, wm.updated_run) == (200.0, 9, "r2")
    led.set_root("child", "root")
    assert led.get_root("child") == "root" and led.all_roots() == {"child": "root"}
    assert led.get_cursor("inbox") == 0
    led.set_cursor("inbox", 42)
    assert led.get_cursor("inbox") == 42


def test_runs_lifecycle(led):
    led.insert_run(RunRecord("r1", 1.0, mode="live", status="planned", lance_version_before=3,
                             wm_before_json="{}"))
    led.insert_run(RunRecord("r1", 9.0))                   # ignored (DO NOTHING)
    assert led.get_run("r1").started_at == 1.0
    assert [r.run_id for r in led.planned_runs()] == ["r1"]
    led.update_run("r1", status="committed", lance_version_after=5)
    assert led.get_run("r1").status == "committed" and led.planned_runs() == []
    with pytest.raises(KeyError):
        led.update_run("r1", bogus=1)


def _delta():
    return LedgerDelta(
        watermarks={"root": (123.0, 11)}, session_roots={"child": "root"},
        windows=[WindowState("w1", "statedb", "root", 10, 11, 123.0, "ok", 0, None, "r1", 2)],
        md_files=[MdFileState("/m/a.md", "sha", 100, "psha", "ok", "r1")],
        cursors={"inbox": 5, "recall_events": 9},
        core_seen=[CoreSeenRow("user", "e1", "항목", "mid", "r1", "r1", True)],
        audit=[AuditRow(1.0, "r1", "forget", "mid", "reason=user")])


def test_apply_delta_atomic_and_idempotent(led):
    led.insert_run(RunRecord("r1", 1.0, status="planned"))
    d = _delta()
    led.apply_delta("r1", d, lance_version_after=7, stats_json='{"created":1}')
    led.apply_delta("r1", d, lance_version_after=7, stats_json='{"created":1}')   # replay
    assert led.get_wm("root").last_id == 11
    assert led.get_window("w1").n_claims == 2
    assert led.get_md("/m/a.md").processed_bytes == 100
    assert led.get_cursor("recall_events") == 9
    assert led.core_seen("user")[("user", "e1")].present is True
    assert len(led.audits("mid")) == 1
    run = led.get_run("r1")
    assert run.status == "committed" and run.lance_version_after == 7


def test_apply_delta_rolls_back_on_error(led):
    led.insert_run(RunRecord("r1", 1.0, status="planned"))
    d = _delta()
    d.windows.append(WindowState("w2", "statedb", "root", 1, 2, 1.0, "BAD"))   # CHECK fails
    with pytest.raises(sqlite3.IntegrityError):
        led.apply_delta("r1", d, lance_version_after=7)
    assert led.get_wm("root") is None
    assert led.get_run("r1").status == "planned"


def test_readonly_refuses_writes_and_missing_is_memory(tmp_path):
    p = tmp_path / "ledger.db"
    l = Ledger.open(p)
    l.init_meta(MODEL, 1536)
    l.close()
    before = tree_hash(tmp_path)
    ro = Ledger.open(p, readonly=True)
    assert ro.get_meta("embed_model") == MODEL
    with pytest.raises(LedgerReadOnly):
        ro.set_wm("r", 1, 1, "x")
    ro.close()
    assert tree_hash(tmp_path) == before
    mem = Ledger.open(tmp_path / "absent.db", readonly=True)
    assert mem.all_wms() == {} and not (tmp_path / "absent.db").exists()
    mem.close()


def test_backup_keeps_n(led, tmp_path):
    for i in range(9):
        led.backup(tmp_path / "bk", keep=7, stamp=f"2026100{i}")
    files = sorted((tmp_path / "bk").glob("ledger-*.db"))
    assert len(files) == 7 and files[0].name == "ledger-20261002.db"


def test_restore_wms_drops_run_windows(led):
    led.set_wm("a", 5.0, 1, "r0")
    snap = led.wm_snapshot_json()
    led.set_wm("a", 9.0, 3, "r1")
    led.set_wm("b", 9.0, 4, "r1")
    led.upsert_window(WindowState("w", "statedb", "a", 2, 3, 9.0, "ok", run_id="r1"))
    led.restore_wms(snap, rolled_back_run="r1")
    assert led.get_wm("a").last_id == 1 and led.get_wm("b") is None
    assert led.get_window("w") is None
