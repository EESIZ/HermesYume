"""Offline end-to-end through rem.run_dream with the REAL nrem/export (and dream_log/alerts when
present; tiny stubs otherwise): state.db fixture + md episode → claims → Lance → serving copy.
Second run on an unchanged ledger: LLM 0, Lance versions unchanged (G1b/T13)."""

import hashlib
import importlib.util
import sqlite3
import types

import pytest

from hermesyume import rem
from tests.dream.test_rem_helpers import (classify_all, deps, install_module, next_ctx,  # noqa: F401
                                          no_core_writes)
from tests.fakes import claim as fclaim, extract_json

pytest.importorskip("hermesyume.nrem")
pytest.importorskip("hermesyume.export")


def _maybe_stub(monkeypatch):
    if importlib.util.find_spec("hermesyume.dream_log") is None:
        dl = types.ModuleType("hermesyume.dream_log")
        dl.render = lambda ctx, plan, *, status, error=None: f"# {status}"
        dl.write = lambda paths, now, text, *, dry: None
        install_module(monkeypatch, "hermesyume.dream_log", dl)
    if importlib.util.find_spec("hermesyume.alerts") is None:
        al = types.ModuleType("hermesyume.alerts")

        class Sink:
            def __init__(self, *a, **k):
                pass

            def emit_all(self, alerts):
                pass
        al.AlertSink = Sink
        al.evaluate_run_alerts = lambda ctx, plan: []
        al.evaluate_health = lambda ctx, *, now: []
        al.flush_pending = lambda paths, cfg, *, now: 0
        install_module(monkeypatch, "hermesyume.alerts", al)


def _extract(messages):
    """Answer EXTRACT with one claim per window, citing the first user ref found in the body."""
    import re
    body = messages[-1]["content"]
    part = body.split("[추출 대상]", 1)[-1]
    m = re.search(r"\[(U#[^\s\]]+)", part)
    if not m:
        return extract_json()
    if "장애 보고" in part:
        return extract_json(fclaim("procedure", "Orion 장애 보고 절차는 알림 확인, 로그 수집, 원인 기록, 회고 공유 순서다.",
                                   subject="Orion 장애 보고", evidence=[m.group(1)], explicit=True, steps=4))
    return extract_json()


def test_offline_dream_twice(ctx, monkeypatch, deps, statedb):
    _maybe_stub(monkeypatch)
    c1 = next_ctx(ctx)
    c1.llm.on("extract", _extract)
    c1.llm.on("core_classify", classify_all(lambda t: "rule" if "원장" in t else "preference"))
    c1.settle_minutes = 0
    core = {t: hashlib.sha256(ctx.paths.core_file(t).read_bytes()).hexdigest() for t in ("memory", "user")}
    with monkeypatch.context() as m:                   # T14 over the whole run_dream
        hits = no_core_writes(m, ctx.paths)
        st1 = rem.run_dream(c1)
    assert hits == []
    assert core == {t: hashlib.sha256(ctx.paths.core_file(t).read_bytes()).hexdigest() for t in ("memory", "user")}
    assert st1.status == "committed", c1.alerts
    rows_ = ctx.store.load_working_set()
    assert any("장애 보고" in r.text for r in rows_.values())
    assert any(r.source == "core:user" for r in rows_.values())
    assert ctx.paths.recall_sqlite.exists()
    con = sqlite3.connect(f"file:{ctx.paths.recall_sqlite}?mode=ro", uri=True)
    n_items = con.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    con.close()
    assert n_items >= 1
    v = ctx.store.versions()
    c2 = next_ctx(ctx)
    c2.llm.on("extract", _extract)
    c2.settle_minutes = 0
    st2 = rem.run_dream(c2)
    assert st2.status == "committed", c2.alerts
    assert st2.llm_calls == 0 and st2.created == 0 and st2.reinforced == 0
    assert ctx.store.versions() == v
    assert ctx.ledger.get_run(c2.run_id).lance_version_after == ctx.ledger.get_run(c1.run_id).lance_version_after
