"""T18 export: serving/recall.sqlite built exactly per the shared schema (PLAN-v2 §2.5, CONTRACTS §5)."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sqlite3
import stat
import sys

import numpy as np
import pytest

from hermesyume import export
from hermesyume.embedder import prefix_renorm
from hermesyume.paths import load_provider_module
from hermesyume.types import SERVING_STATUSES, TIERS, StrengthResult
from tests.fakes import make_row

LEDGER_RULE = "**가계부 관리:** 지출 기록은 공용 가계부 DB를 단일 원장으로 사용한다."


@pytest.fixture
def fake_strength(monkeypatch):
    def ev(row, now, cfg):
        return StrengthResult(tier="pinned" if row.pinned else "decaying",
                              strength=0.9 if row.pinned else 0.42, t_ref=now, hl_days=None)
    monkeypatch.setattr(export, "_evaluate", ev)


@pytest.fixture
def rows(ctx, fake_embedder, now, fake_home):
    ws = fake_home.workspace
    (ws / "docs" / "yume").mkdir(parents=True, exist_ok=True)
    (ws / "docs" / "yume" / "orion.md").write_text("# orion\n", encoding="utf-8")
    abs_ref = fake_home.root.parent / "abs-ref.md"
    abs_ref.write_text("x", encoding="utf-8")
    corefmt = load_provider_module("corefmt")
    mk = lambda text, **kw: make_row(text, embedder=fake_embedder, now=now, **kw)  # noqa: E731
    out = {
        "act": mk("Orion 결제 스테이징 서버 포트는 8081이다.", id="act", kind="reference",
                  subject="Orion 포트", event_time=now - 86400,
                  refs=["docs/yume/orion.md", "docs/yume/missing.md", "skills/rate-lookup/SKILL.md", str(abs_ref)]),
        "sup": mk("Orion 검수 당번은 7조가 맡는다.", id="sup", status="superseded"),
        "exp": mk("Orion 데모 마감은 2026-09-01이다.", id="exp", kind="schedule", status="expired",
                  valid_until=now - 30 * 86400),
        "dor": mk("E2E event 휴면 테스트 문장입니다.", id="dor", kind="event", status="dormant"),
        "fgt": mk("테스트 암호어는 보라색 고래 7341이다.", id="fgt", status="forgotten"),
        "qua": mk("격리된 비밀 포함 문장입니다 sk-xxxx", id="qua", status="quarantined"),
        "can": mk("후보 상태 문장은 서빙되지 않는다.", id="can", status="candidate"),
        "pin": mk(LEDGER_RULE, id="pin", kind="rule", pinned=True, core_target="user",
                  core_sha=corefmt.core_sha(LEDGER_RULE), in_core=True, source="core:user", subject="가계부"),
        "pdor": mk("휴면 상태의 고정 기억은 pins에 없다.", id="pdor", pinned=True, status="dormant"),
    }
    ctx.store.commit_memories(list(out.values()))
    return out


def _conn(path):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def test_t18_items_vectors_pins_refs_meta(ctx, rows, fake_strength, paths, cfg):
    out = export.build_serving(ctx)
    assert out == paths.recall_sqlite
    assert stat.S_IMODE(os.stat(out).st_mode) == 0o600
    assert sorted(os.listdir(paths.serving_dir)) == ["recall.sqlite"]       # tmp file replaced away
    ss = export.serving_schema()
    with _conn(out) as c:
        assert {r["name"] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")} >= \
            {"items", "items_fts", "pins", "meta"}
        assert [r[1] for r in c.execute("PRAGMA table_info(items)")] == list(ss.ITEM_COLUMNS)
        items = {r["id"]: r for r in c.execute("SELECT * FROM items")}
        # no forgotten / quarantined / candidate
        assert set(items) == {"act", "sup", "exp", "dor", "pin", "pdor"}
        assert {r["status"] for r in items.values()} <= set(SERVING_STATUSES)
        for rid, r in items.items():
            vec = np.frombuffer(r["vec"], dtype="<f4")
            assert vec.shape == (1536,)
            np.testing.assert_array_equal(vec, np.asarray(rows[rid].vector, dtype="<f4"))
            v256 = np.frombuffer(r["vec256"], dtype="<f4")
            assert v256.shape == (256,)
            assert abs(float(np.linalg.norm(v256)) - 1.0) < 1e-5
            np.testing.assert_allclose(v256, prefix_renorm(rows[rid].vector), rtol=0, atol=1e-7)
            assert r["strength"] == pytest.approx(0.9 if rows[rid].pinned else 0.42)
            assert r["tier"] == ("pinned" if rows[rid].pinned else "decaying")
            assert r["pinned"] == (1 if rows[rid].pinned else 0)
        act = items["act"]
        assert json.loads(act["refs"]) == ["docs/yume/orion.md", "skills/rate-lookup/SKILL.md",
                                           str(paths.hermes_home.parent / "abs-ref.md")]
        assert act["event_time"] == pytest.approx(rows["act"].event_time)
        assert act["subject"] == "Orion 포트" and act["kind"] == "reference"
        assert items["exp"]["valid_until"] == pytest.approx(rows["exp"].valid_until)
        assert items["pin"]["core_sha"] == rows["pin"].core_sha
        assert items["sup"]["core_sha"] is None
        assert json.loads(items["sup"]["refs"]) == []
        # FTS mirrors items
        assert c.execute("SELECT count(*) FROM items_fts").fetchone()[0] == len(items)
        assert [r[0] for r in c.execute("SELECT id FROM items_fts WHERE items_fts MATCH '\"스테이징\"'")] == ["act"]
        # pins: active pinned rows only, label from the leading bold label
        pins = [tuple(r) for r in c.execute("SELECT id, text, label, core_target FROM pins")]
        assert pins == [("pin", LEDGER_RULE, "**가계부 관리:**", "user")]
        meta = dict(c.execute("SELECT key, value FROM meta").fetchall())
    assert set(meta) == set(ss.META_KEYS)
    assert meta["embed_model"] == cfg.embed_model_id() == "openai/text-embedding-3-small@1536"
    assert meta["dim"] == "1536"
    assert meta["run_id"] == ctx.run_id
    assert meta["lance_version"] == str(ctx.store.version())
    assert float(meta["built_at"]) == pytest.approx(ctx.now)
    assert meta["count"] == "6"
    assert any("docs/yume/missing.md" in n for n in ctx.report.notes)


def test_atomic_replace_and_explicit_rows(ctx, rows, fake_strength, paths):
    p1 = export.build_serving(ctx)
    ino1 = os.stat(p1).st_ino
    reader = sqlite3.connect(f"file:{p1}?mode=ro", uri=True)        # an open reader keeps its snapshot
    sub = {k: rows[k] for k in ("act", "fgt")}
    p2 = export.build_serving(ctx, rows=sub, lance_version=77)
    assert os.stat(p2).st_ino != ino1
    assert reader.execute("SELECT count(*) FROM items").fetchone()[0] == 6
    reader.close()
    with _conn(p2) as c:
        assert [r[0] for r in c.execute("SELECT id FROM items")] == ["act"]
        assert dict(c.execute("SELECT key, value FROM meta").fetchall())["lance_version"] == "77"
        assert c.execute("SELECT count(*) FROM pins").fetchone()[0] == 0


def test_never_in_dry_run(ctx, rows, paths):
    ctx.dry_run = True
    with pytest.raises(RuntimeError):
        export.build_serving(ctx)
    assert not paths.recall_sqlite.exists()


def test_pin_budget_note(ctx, fake_embedder, now, fake_strength):
    big = {f"p{i}": make_row(("긴 고정 기억 %d " % i) + "가" * 380, embedder=fake_embedder, now=now,
                             id=f"p{i}", pinned=True, kind="rule") for i in range(3)}
    export.build_serving(ctx, rows=big, lance_version=1)
    assert any("예산 800자" in n for n in ctx.report.notes)


def test_bad_vector_rows_skipped(ctx, fake_embedder, now, fake_strength):
    good = make_row("정상 벡터 행입니다 하나", embedder=fake_embedder, now=now, id="good")
    bad = make_row("잘못된 차원 벡터 행입니다", embedder=fake_embedder, now=now, id="bad")
    bad.vector = np.ones(10, dtype=np.float32)
    path = export.build_serving(ctx, rows={"good": good, "bad": bad}, lance_version=1)
    with _conn(path) as c:
        assert [r[0] for r in c.execute("SELECT id FROM items")] == ["good"]
    assert any("제외한 행 1개" in n for n in ctx.report.notes)


def test_existing_refs_and_label(fake_home, paths):
    ws = fake_home.workspace
    (ws / "a.md").write_text("x", encoding="utf-8")
    got = export.existing_refs(["a.md", "a.md", "", "nope.md", "skills/rate-lookup/SKILL.md"],
                               workspace_dir=str(ws), hermes_home=paths.hermes_home)
    assert got == ["a.md", "skills/rate-lookup/SKILL.md"]
    r = make_row("주제만 있는 행입니다", subject="주제")
    assert export.label_for(r) == "주제"
    r2 = make_row("**호칭:** 사장님")
    assert export.label_for(r2) == "**호칭:**"


def test_real_strength_integration(ctx, rows, paths):
    pytest.importorskip("hermesyume.strength")
    export.build_serving(ctx)
    with _conn(paths.recall_sqlite) as c:
        for r in c.execute("SELECT id, tier, strength FROM items"):
            assert r["tier"] in TIERS
            assert 0.0 <= r["strength"] <= 1.0
        assert c.execute("SELECT tier FROM items WHERE id='pin'").fetchone()[0] == "pinned"


def _provider_pkg():
    """provider/_yume as a standalone package (stdlib modules; no provider/__init__ → no Hermes)."""
    name = "_yume_export_xcheck"
    if name in sys.modules:
        return sys.modules[name]
    root = load_provider_module("serving_schema").__file__
    yd = os.path.dirname(root)
    spec = importlib.util.spec_from_file_location(name, os.path.join(yd, "__init__.py"),
                                                  submodule_search_locations=[yd])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_provider_reads_exported_file(ctx, rows, fake_strength, paths):
    """Cross-check: the provider's ServingIndex loads what export wrote and finds rows by vector/FTS."""
    export.build_serving(ctx)
    pkg = _provider_pkg()
    serving = importlib.import_module(pkg.__name__ + ".serving")
    idx = serving.ServingIndex.load(str(paths.recall_sqlite))
    assert sorted(idx.items) == ["act", "pin"]               # active rows only in memory
    assert idx.embed_model == "openai/text-embedding-3-small@1536" and idx.dim == 1536
    q = [float(x) for x in rows["act"].vector]
    top = idx.vector_search(q, exclude=set(), now=ctx.now, k1=64)
    assert top[0][0] == "act" and top[0][1] == pytest.approx(1.0, abs=1e-5)
    assert [p.id for p in idx.pins] == ["pin"]
    assert idx.items["act"].refs[0] == "docs/yume/orion.md"
    allres = idx.search_all(q, "", include_inactive=True, limit=10, min_cos=-1.0, now=ctx.now)
    assert {it.id for it, _s, _m in allres} == {"act", "sup", "exp", "dor", "pin", "pdor"}
    kw = idx.fts_search("Orion 스테이징 서버", exclude=set(), limit=3, statuses=("active",), now=ctx.now)
    assert kw and kw[0][0] == "act"
