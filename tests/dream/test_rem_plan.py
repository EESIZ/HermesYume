"""plan.py — WorkingSet ops/finalize, R7 guard (T15, U2), plan.json round-trip, R8 commit with
crash-replay (T13), noop re-run keeps Lance versions, approve_held."""

import json
import os
import stat

import numpy as np
import pytest

from hermesyume import plan as P
from hermesyume.types import LedgerDelta, MemoryRow, SuppressRow, text_sha
from tests.dream.test_rem_helpers import commit, deps, next_ctx, row  # noqa: F401

DAY = 86400.0


def ws_of(ctx):
    return P.WorkingSet(ctx.store.load_working_set(), run_id=ctx.run_id, now=ctx.now,
                        embed_model=ctx.embedder.model_id)


def test_working_set_copy_on_write_and_ops(ctx):
    a = row(ctx, "Orion 결제 스테이징 서버 포트는 8081이다.", kind="reference")
    commit(ctx, a)
    ws = ws_of(ctx)
    base_text = ws.base[a.id].text
    op = ws.update(a.id, {"text": "바뀐 텍스트입니다 열다섯 글자 이상", "evidence_count": 5}, op="text_update",
                   reason="t", user_evidence=True)
    assert op is not None and op.before["text"] == base_text
    assert ws.base[a.id].text == base_text                    # base never mutated
    assert ws.get(a.id).version == a.version + 1
    assert ws.update(a.id, {"evidence_count": 5}, op="reinforce") is None    # no-op → None
    assert ws.update("missing", {"text": "x"}, op="status") is None
    new = row(ctx, "새 기억은 즉시 작업 사본에 보인다.", origin_keys=["w#1"])
    ws.insert(new)
    assert ws.by_origin_key("w#1").id == new.id
    assert [r.id for r, _ in ws.vector_search(new.vector, k=1, only_new=True)] == [new.id]
    assert ws.touched_ids() == [a.id, new.id]


def test_finalize_replays_non_held_ops_onto_base(ctx):
    a = row(ctx, "규칙 A는 반드시 지켜야 한다고 했다.", kind="rule", pinned=True, source="core:user")
    b = row(ctx, "사실 B는 그냥 평범한 사실이다.")
    commit(ctx, a, b)
    ws = ws_of(ctx)
    o1 = ws.update(a.id, {"status": "superseded"}, op="status", reason="x")     # pinned, no user evidence
    o2 = ws.update(b.id, {"evidence_count": 9}, op="reinforce")
    assert o1.protected_change and not o2.protected_change
    g = P.apply_guard(ws, ctx.cfg, mode="live")
    assert g.held and g.held_seqs == [o1.seq]
    rows, hist = P.finalize(ws, held_seqs=set(g.held_seqs))
    assert [r.id for r in rows] == [b.id] and rows[0].evidence_count == 9
    assert [h.op for h in hist] == ["reinforce"]
    assert json.loads(hist[0].before_json)["evidence_count"] == b.evidence_count


def test_link_only_change_on_pinned_row_is_not_held(ctx):
    a = row(ctx, "핀 고정된 핵심 기억이 여기에 있다.", pinned=True, source="core:user")
    commit(ctx, a)
    ws = ws_of(ctx)
    o = ws.update(a.id, {"related_ids": ["x" * 32]}, op="text_update", reason="related:protected")
    assert not o.protected_change


def test_t15_mass_dormant_committed_and_pinned_change_held(ctx):
    rows_ = [row(ctx, f"대량 감쇠 대상 사실 번호 {i}번 입니다.", importance=0.3, now=ctx.now - 400 * DAY)
             for i in range(30)]
    pin = row(ctx, "**호칭:** 사장님이라고 부른다.", kind="profile", pinned=True, source="core:user")
    commit(ctx, *rows_, pin)
    from hermesyume import rem
    ws = ws_of(ctx)
    trs = rem.time_transitions(ctx, ws)
    assert sum(1 for t in trs if t.to_status == "dormant") == 30
    bad = ws.update(pin.id, {"status": "dormant"}, op="status", reason="bug")       # no user evidence
    fresh = row(ctx, "이번 실행에 새로 들어온 사실 하나.")
    ws.insert(fresh)
    g = P.apply_guard(ws, ctx.cfg, mode="live")
    assert g.held_seqs == [bad.seq] and g.destructive_count == 30 and g.threshold == 10
    pl = P.build_plan(ctx, ws, lance_version_before=ctx.store.version(), ledger_delta=LedgerDelta(),
                      inbox_consume_ids=[], inbox_skip_ids=[], docs=[])
    res = P.commit_plan(ctx, pl)
    after = ctx.store.load_working_set()
    assert all(after[r.id].status == "dormant" for r in rows_)       # mass dormant NOT held (U2)
    assert after[pin.id].status == "active"                          # pinned change held
    assert fresh.id in after                                         # insert committed
    assert ctx.ledger.get_run(ctx.run_id).status == "held"
    held = P.read_plan(ctx.paths.plan_json(ctx.run_id))
    assert [o.seq for o in held.ops if o.held] == [bad.seq]
    # next night the same op is held again (never auto-applied)
    c2 = next_ctx(ctx, days=1)
    ws2 = ws_of(c2)
    o = ws2.update(pin.id, {"status": "dormant"}, op="status", reason="bug")
    assert P.apply_guard(ws2, c2.cfg, mode="live").held_seqs == [o.seq]
    assert res.lance_version_after == ctx.store.version()


def test_plan_json_roundtrip(ctx):
    a = row(ctx, "왕복 직렬화 테스트용 기억 텍스트.")
    commit(ctx, a)
    ws = ws_of(ctx)
    ws.update(a.id, {"vector": ctx.embedder.vector("다른 벡터"), "text": "바뀐 왕복 텍스트 열다섯자 이상"},
              op="consolidate", user_evidence=True)
    ws.add_suppress(SuppressRow(id=a.id, vector=a.vector, text_sha=text_sha(a.text), kind="fact",
                                created_at=ctx.now, reason="forget|run:x"))
    ws.add_audit("forget", a.id, "reason=")
    P.apply_guard(ws, ctx.cfg, mode="live")
    pl = P.build_plan(ctx, ws, lance_version_before=3,
                      ledger_delta=LedgerDelta(watermarks={"r": (1.5, 7)}, cursors={"inbox": 4},
                                               audit=list(ws.audit)),
                      inbox_consume_ids=[3, 1], inbox_skip_ids=[2], docs=[])
    d = P.plan_to_json(pl)
    back = P.plan_from_json(json.loads(json.dumps(d)))
    assert P.plan_to_json(back) == d
    assert np.allclose(back.upserts[0].vector, ctx.embedder.vector("다른 벡터"))
    assert back.inbox_consume_ids == [1, 3]
    path = P.write_plan(ctx.paths, pl)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert P.plan_to_json(P.read_plan(path)) == d
    assert "vector_b64" not in json.loads(back.history[0].after_json)


def _make_run(ctx):
    a = row(ctx, "커밋 전 기존 사실 A 입니다 여기.")
    commit(ctx, a)
    ws = ws_of(ctx)
    ws.update(a.id, {"evidence_count": 4}, op="reinforce", user_evidence=True)
    n = row(ctx, "이번 실행에서 새로 넣는 사실 N 입니다.", origin_keys=["w9#0"])
    ws.insert(n)
    P.apply_guard(ws, ctx.cfg, mode="live")
    pl = P.build_plan(ctx, ws, lance_version_before=ctx.store.version(),
                      ledger_delta=LedgerDelta(watermarks={"root1": (ctx.now - 100, 42)}),
                      inbox_consume_ids=[], inbox_skip_ids=[], docs=[])
    return a, n, pl


@pytest.mark.parametrize("crash_at", ["commit_memories", "apply_delta"])
def test_t13_crash_and_replay(ctx, monkeypatch, crash_at, deps):
    a, n, pl = _make_run(ctx)
    target = ctx.store if crash_at == "commit_memories" else ctx.ledger

    def boom(*args, **kw):
        raise RuntimeError("simulated crash")
    monkeypatch.setattr(target, crash_at, boom)
    with pytest.raises(RuntimeError):
        P.commit_plan(ctx, pl)
    monkeypatch.undo()
    assert ctx.ledger.get_run(ctx.run_id).status == "planned"
    hist_before = len(ctx.store.history())
    # replay (as nrem.preflight does at the next run's N0); post_commit is stubbed out
    from hermesyume import rem
    calls = []
    monkeypatch.setattr(rem, "post_commit", lambda c, p: calls.append(p.run_id))
    c2 = next_ctx(ctx, days=0.0)
    assert P.replay_planned(c2) == [ctx.run_id]
    assert calls == [ctx.run_id]
    rows_ = ctx.store.load_working_set()
    assert rows_[a.id].evidence_count == 4 and n.id in rows_
    hist = ctx.store.history()
    assert len(hist) == 2 and len({h.history_id for h in hist}) == 2         # no duplicates
    assert hist_before in (0, 2)
    assert ctx.ledger.get_run(ctx.run_id).status == "committed"
    assert ctx.ledger.get_wm("root1").last_id == 42
    # replaying again is a no-op
    assert P.replay_planned(c2) == []


def test_t13_noop_plan_keeps_versions(ctx):
    a = row(ctx, "변하지 않는 기억 하나가 있다.")
    commit(ctx, a)
    v = ctx.store.versions()
    ws = ws_of(ctx)
    P.apply_guard(ws, ctx.cfg, mode="live")
    pl = P.build_plan(ctx, ws, lance_version_before=ctx.store.version(), ledger_delta=LedgerDelta(),
                      inbox_consume_ids=[], inbox_skip_ids=[], docs=[])
    assert pl.is_noop()
    res = P.commit_plan(ctx, pl)
    assert res.skipped and ctx.store.versions() == v
    run = ctx.ledger.get_run(ctx.run_id)
    assert run.status == "committed" and run.lance_version_after == v["memories"]


def test_approve_held_applies_when_not_stale(ctx):
    pin = row(ctx, "**시간대:** Asia/Seoul 기준이다.", kind="profile", pinned=True, source="core:user")
    other = row(ctx, "**언어:** 한국어 기본이다 항상.", kind="profile", pinned=True, source="core:user")
    commit(ctx, pin, other)
    ws = ws_of(ctx)
    ws.update(pin.id, {"pinned": False}, op="unpin", reason="bug")
    ws.update(other.id, {"text": "**언어:** 영어로 바뀌었다고 한다."}, op="text_update", reason="bug")
    P.apply_guard(ws, ctx.cfg, mode="live")
    pl = P.build_plan(ctx, ws, lance_version_before=ctx.store.version(), ledger_delta=LedgerDelta(),
                      inbox_consume_ids=[], inbox_skip_ids=[], docs=[])
    P.commit_plan(ctx, pl)
    assert ctx.ledger.get_run(ctx.run_id).status == "held"
    # operator edits `other` in between → that op is stale and skipped
    cur = ctx.store.get([other.id])[other.id]
    cur.text = "**언어:** 사람이 직접 고친 내용."
    ctx.store.commit(upserts=[cur])
    c2 = next_ctx(ctx, days=0.0, run_id="approve-run")
    P.approve_held(c2, ctx.run_id)
    rows_ = ctx.store.load_working_set()
    assert rows_[pin.id].pinned is False
    assert rows_[other.id].text == "**언어:** 사람이 직접 고친 내용."
    assert any("stale" in n for n in c2.report.notes)
    assert ctx.ledger.get_run(ctx.run_id).status == "committed"
    assert ctx.ledger.get_run("approve-run").status == "committed"
