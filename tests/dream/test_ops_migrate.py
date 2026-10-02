"""migrate.py (PLAN §9, U2): T20 core_map n/n, auto-pin of USER profile/rule rows (no
pins.proposed.yaml), legacy MEMORY.md rows, Dreamer dump, --statedb-start, --estimate,
--dry-run purity, idempotent re-run, refusals, core files never written."""

from __future__ import annotations

import builtins
import dataclasses
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("hermesyume.rem")
pytest.importorskip("hermesyume.nrem")

from hermesyume import migrate  # noqa: E402
from hermesyume.ledger import Ledger  # noqa: E402
from hermesyume.livedb import LiveDB  # noqa: E402
from hermesyume.store import Store  # noqa: E402
from hermesyume.types import RunReport, RunStats  # noqa: E402
from tests.fixtures.hermes_home import USER_ENTRIES, tree_hash  # noqa: E402

PROFILE_LABELS = ("**이름:", "**호칭:", "**시간대:", "**직업:")
RULE_LABELS = ("**가계부 관리:", "**일정 관리 원칙:", "**금지:", "**데이터 원칙:")


def _kind_for(text: str) -> str:
    if text.startswith(PROFILE_LABELS):
        return "profile"
    if text.startswith(RULE_LABELS):
        return "rule"
    if text.startswith("**Reading List:"):
        return "state"
    return "preference"


def _classify(messages):
    body = json.loads(messages[-1]["content"])
    items = []
    for it in body["items"]:
        text = it["text"]
        items.append({"i": it["i"], "kind": _kind_for(text),
                      "subject": text.split(":**")[0].strip("*") if ":**" in text else text[:10],
                      "fragment": False})
    return {"items": items}


@pytest.fixture
def mctx(ctx, scripted_llm):
    scripted_llm.on("core_classify", _classify)
    return dataclasses.replace(ctx, mode="migrate", approve_migration=True)


def _rows(ctx):
    return Store.from_config(ctx.paths, ctx.cfg).load_working_set(with_vectors=False)


def _core_sha(fake_home):
    return {p: Path(p).read_bytes() for p in (fake_home.user_md, fake_home.memory_md)}


def _expected_pins(rows):
    return {r.id for r in rows.values() if r.source == "core:user" and r.kind in ("profile", "rule")}


# ── T20 ──────────────────────────────────────────────────────────────────────

def test_t20_core_map_complete_and_auto_pin(mctx, fake_home):
    before = _core_sha(fake_home)
    res = migrate.run_migrate(mctx, only=None)
    assert res.status == "committed", res.problems
    assert res.ok, res.problems
    cm = res.core_map
    assert cm["total"] == 35 and cm["accounted"] == 35 and cm["complete"] and cm["ok"]
    assert cm["fragments"] == 1
    frag = [e for e in cm["entries"] if e["fragment"]]
    assert frag[0]["memory_id"] is None and frag[0]["target"] == "user"

    rows = _rows(mctx)
    core_rows = [r for r in rows.values() if r.source == "core:user"]
    legacy = [r for r in rows.values() if r.source == "legacy:memory_md"]
    assert len(core_rows) == 21 and len(legacy) == 13
    for r in core_rows:
        assert r.in_core and r.core_target == "user" and r.core_sha and r.status == "active"
        assert r.text in USER_ENTRIES                       # verbatim
        assert r.tier in ("durable", "pinned")              # protected: never decays
    for r in legacy:
        assert r.kind == "legacy" and r.status == "dormant" and r.core_sha
    pinned = {r.id for r in rows.values() if r.pinned}
    assert pinned == _expected_pins(rows) and len(pinned) == 8
    assert set(res.pinned_ids) == pinned
    assert all(rows[i].core_required for i in pinned)
    mapped = {e["memory_id"] for e in cm["entries"] if e["memory_id"]}
    assert mapped == {r.id for r in core_rows} | {r.id for r in legacy}

    # U2: no proposal / approval artifacts
    assert res.pins_proposed_path is None
    assert not list(mctx.paths.data_dir.rglob("pins.proposed*"))
    # artifacts
    saved = json.loads(Path(res.core_map_path).read_text(encoding="utf-8"))
    assert saved["accounted"] == 35
    assert (mctx.paths.migration_dir / "inventory.json").exists()
    prop = mctx.paths.proposals_dir / migrate.PROPOSAL_FILENAME
    assert prop.exists() and prop.read_text(encoding="utf-8") == ""      # all 13 episodic preserved
    # live core files untouched
    assert _core_sha(fake_home) == before
    assert not list(fake_home.memories.glob("*.lock"))
    # one Dream Log, migration flavoured, pins listed, no review queue
    logs = sorted(mctx.paths.dream_log_dir.glob("*.md"))
    assert len(logs) == 1
    text = logs[0].read_text(encoding="utf-8")
    assert "마이그레이션" in text and "core_map 35/35" in text and "확인 필요" not in text
    assert "## 새 pin (8)" in text
    led = Ledger.from_paths(mctx.paths, readonly=True)
    assert led.get_run(mctx.run_id).status == "committed"
    assert led.get_run(mctx.run_id + migrate.SEED_SUFFIX).status == "committed"
    led.close()


def test_rerun_is_idempotent(mctx, ctx, fake_home):
    r1 = migrate.run_migrate(mctx, only={"core", "memory_md"})
    assert r1.ok, r1.problems
    rows1 = _rows(mctx)
    ctx2 = dataclasses.replace(mctx, run_id="20261003-044000-test", stats=RunStats(run_id="x"),
                               report=RunReport(), alerts=[])
    r2 = migrate.run_migrate(ctx2, only={"core", "memory_md"})
    assert r2.ok, r2.problems
    rows2 = _rows(mctx)
    assert set(rows2) == set(rows1)
    assert r2.seed_plan is not None and r2.seed_plan.is_noop()
    assert {e["memory_id"] for e in r2.core_map["entries"]} == {e["memory_id"] for e in r1.core_map["entries"]}
    assert ctx2.stats.created == 0


def test_only_core_scope(mctx):
    res = migrate.run_migrate(mctx, only={"core"})
    assert res.ok, res.problems
    cm = res.core_map
    assert cm["ok"] and not cm["complete"] and cm["pending"] == 13      # episodic: memory_md step
    assert not [r for r in _rows(mctx).values() if r.kind == "legacy"]


# ── dry-run ──────────────────────────────────────────────────────────────────

def test_dry_run_writes_only_plan_and_dry_log(ctx, scripted_llm, fake_home, paths, cfg):
    scripted_llm.on("core_classify", _classify)
    ctx.ledger.close()
    ctx.live.close()
    dctx = dataclasses.replace(ctx, mode="migrate", dry_run=True,
                               ledger=Ledger.from_paths(paths, readonly=True),
                               live=LiveDB.open(paths, mode="pure"))
    before = tree_hash(fake_home.root)
    res = migrate.run_migrate(dctx, only=None)
    assert res.status == "dry", res.problems
    assert res.ok and res.core_map["accounted"] == 35
    assert tree_hash(fake_home.root) == before                   # incl. lock files; dream-log/ runs/ excluded
    assert (paths.plan_json(dctx.run_id)).exists()
    assert (paths.plan_json(dctx.run_id + migrate.SEED_SUFFIX)).exists()
    logs = list(paths.dream_log_dir.glob("*_dry.md"))
    assert len(logs) == 1 and "core_map 35/35" in logs[0].read_text(encoding="utf-8")
    assert not (paths.migration_dir / "core_map.json").exists()
    assert not paths.alerts_log.exists()
    assert len(res.seed_plan.upserts) == 34
    assert _rows(dctx) == {}                                    # nothing committed
    dctx.ledger.close()
    if dctx.live is not None:
        dctx.live.close()
    # bring the fixture handles back for teardown
    ctx.ledger = Ledger.from_paths(paths)
    ctx.live = LiveDB.open(paths, mode="rw")


# ── estimate / refusals ──────────────────────────────────────────────────────

def test_estimate_only_reads(mctx, fake_home):
    before = tree_hash(fake_home.root, exclude_dirs=())
    res = migrate.run_migrate(mctx, only=None, estimate_only=True)
    assert res.status == "estimate" and res.ok
    est = res.estimate
    for k in ("windows", "llm_calls", "embed_inputs", "usd", "runs_needed", "by_step"):
        assert k in est
    assert est["by_step"]["core"]["entries"] == 21
    assert est["by_step"]["memory_md"]["windows"] == 13
    assert est["by_step"]["md"]["windows"] >= 1
    assert est["llm_calls"] > 0 and est["usd"] > 0
    assert tree_hash(fake_home.root, exclude_dirs=()) == before
    assert mctx.llm.calls == []


def test_refuses_without_approval(ctx):
    res = migrate.run_migrate(dataclasses.replace(ctx, mode="migrate"), only=None)
    assert res.status == "refused" and not res.ok
    assert "--approve-migration" in res.problems[0]


def test_refuses_over_budget(mctx):
    mctx.budget.max_llm_calls = 1
    res = migrate.run_migrate(mctx, only=None)
    assert res.status == "refused" and "--max-llm-calls" in res.problems[0]
    assert _rows(mctx) == {}


def test_parse_only_and_start(now):
    assert migrate.parse_only(None) == set(migrate.STEPS)
    assert migrate.parse_only("core, dump") == {"core", "dump"}
    with pytest.raises(ValueError):
        migrate.parse_only("core,pins")
    assert migrate.parse_statedb_start("now", now) == now
    assert migrate.parse_statedb_start(None, now) is None
    assert migrate.parse_statedb_start("-1d", now) == now - 86400


# ── M5 dump ──────────────────────────────────────────────────────────────────

def test_dreamer_dump_rows_legacy_dormant(mctx, tmp_path):
    dump = [{"id": "a1", "text": "옛 기억 하나는 꽤 길게 적혀 있다", "importance": 0.1, "category": "fact",
             "createdAt": 1767225600123.4567},
            {"id": "a2", "text": "키는 sk-" + "q" * 30 + " 이었다", "importance": 0.8, "category": "fact",
             "createdAt": 1767225600000.0},
            {"id": "a3", "text": "  ", "createdAt": 1.0}]
    p = tmp_path / "dump.json"
    p.write_text(json.dumps(dump, ensure_ascii=False), encoding="utf-8")
    res = migrate.run_migrate(mctx, only={"dump"}, dump_path=str(p))
    assert res.ok, res.problems
    rows = [r for r in _rows(mctx).values() if r.source == "legacy:dreamer"]
    assert len(rows) == 2
    for r in rows:
        assert r.kind == "legacy" and r.status == "dormant" and r.tier == "legacy"
    by_key = {r.origin_keys[0]: r for r in rows}
    assert abs(by_key["x:dreamer:a1"].created_at - 1767225600.1234567) < 1e-3
    assert "sk-qqqq" not in by_key["x:dreamer:a2"].text and "[REDACTED:openai]" in by_key["x:dreamer:a2"].text
    assert [a.code for a in mctx.alerts] == ["secret_found"]
    # idempotent
    ctx2 = dataclasses.replace(mctx, run_id="20261003-044000-t2", stats=RunStats(), report=RunReport(), alerts=[])
    assert migrate.run_migrate(ctx2, only={"dump"}, dump_path=str(p)).ok
    assert len([r for r in _rows(mctx).values() if r.source == "legacy:dreamer"]) == 2


def test_missing_default_dump_is_a_note(mctx):
    res = migrate.run_migrate(mctx, only={"dump"})
    assert res.ok
    assert any("덤프 파일이 없어" in n for n in mctx.report.notes)


# ── --statedb-start ──────────────────────────────────────────────────────────

def test_statedb_start_sets_watermarks_without_extraction(mctx, statedb, now):
    res = migrate.run_migrate(mctx, only={"statedb"}, statedb_start=now)
    assert res.ok, res.problems
    led = Ledger.from_paths(mctx.paths, readonly=True)
    wms = led.all_wms()
    assert wms                                              # lineages watermarked
    assert all(w.last_ts <= now for w in wms.values())
    led.close()
    assert mctx.llm.calls_of("extract") == []               # nothing extracted


# ── safety ───────────────────────────────────────────────────────────────────

def test_core_files_never_opened_for_write(mctx, fake_home, monkeypatch):
    core = {str(fake_home.user_md.resolve()), str(fake_home.memory_md.resolve())}
    real_open, real_replace = builtins.open, os.replace

    def guarded_open(file, mode="r", *a, **k):
        if isinstance(file, (str, os.PathLike)) and any(c in mode for c in "wax+") \
                and str(Path(file).resolve()) in core | {c + ".lock" for c in core}:
            raise AssertionError(f"core write: {file}")
        return real_open(file, mode, *a, **k)

    def guarded_replace(src, dst, *a, **k):
        if str(Path(dst).resolve()) in core:
            raise AssertionError(f"core replace: {dst}")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(os, "replace", guarded_replace)
    res = migrate.run_migrate(mctx, only=None)
    assert res.ok, res.problems


def test_secret_in_user_md_is_skipped_and_alerted(ctx, scripted_llm, fake_home, paths):
    from tests.fixtures.hermes_home import write_core
    entries = list(USER_ENTRIES)
    entries[-1] = "**토큰:** api_key=" + "Z" * 20
    write_core(fake_home.user_md, entries)
    scripted_llm.on("core_classify", _classify)
    c = dataclasses.replace(ctx, mode="migrate", approve_migration=True)
    res = migrate.run_migrate(c, only={"core", "memory_md"})
    assert res.ok, res.problems
    assert res.core_map["secret_skipped"] == 1 and res.core_map["accounted"] == 35
    assert "secret_found" in [a.code for a in c.alerts]
    assert not any("ZZZZZZZZ" in r.text for r in _rows(c).values())
    assert "ZZZZZZZZ" not in paths.alerts_log.read_text(encoding="utf-8")


def test_failure_marks_run_failed_and_alerts(mctx, monkeypatch):
    from hermesyume import rem

    def boom(*a, **k):
        raise RuntimeError("boom sk-" + "w" * 30)

    monkeypatch.setattr(rem, "run_rem", boom)
    res = migrate.run_migrate(mctx, only={"core"})
    assert res.status == "failed" and not res.ok
    assert [a.code for a in mctx.alerts] == ["run_failed"]
    led = Ledger.from_paths(mctx.paths, readonly=True)
    rec = led.get_run(mctx.run_id)
    led.close()
    assert rec.status == "failed" and "sk-www" not in (rec.error or "")
    text = mctx.paths.alerts_log.read_text(encoding="utf-8")
    assert "run_failed" in text and "sk-www" not in text


def test_post_commit_failure_keeps_committed_run(mctx, monkeypatch):
    from hermesyume import rem

    def boom(*a, **k):
        raise OSError("export failed")

    monkeypatch.setattr(rem, "post_commit", boom)
    res = migrate.run_migrate(mctx, only={"core"})
    assert res.status == "failed" and [a.code for a in mctx.alerts] == ["run_failed"]
    led = Ledger.from_paths(mctx.paths, readonly=True)
    rec = led.get_run(mctx.run_id)
    led.close()
    assert rec.status == "committed" and "export failed" in rec.error   # data is committed; not replayed
    assert len([r for r in _rows(mctx).values() if r.source == "core:user"]) == 21
