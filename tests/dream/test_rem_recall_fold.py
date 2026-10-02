"""recall_fold.py — T16: cron ignored; injected never moves t_ref and counts once per (memory,
session, KST date); used/tool_hit move t_ref (last_used_at) and revive dormant rows."""

import pytest

from hermesyume import plan as P, strength
from hermesyume.recall_fold import fold_recall_events, resolve_memory_id
from hermesyume.types import RecallEvent
from tests.dream.test_rem_helpers import commit, row

DAY = 86400.0


def ev(i, mid, kind, *, ts, session="s1", platform="telegram", cos=None, mode="vector"):
    return RecallEvent(id=i, ts=ts, session_id=session, platform=platform, turn_no=1, memory_id=mid,
                       kind=kind, cos=cos, mode=mode, snapshot_run="r0")


def ws_of(ctx):
    return P.WorkingSet(ctx.store.load_working_set(), run_id=ctx.run_id, now=ctx.now,
                        embed_model=ctx.embedder.model_id)


def test_t16_injected_counts_without_moving_t_ref(ctx):
    a = row(ctx, "Orion 스테이징 서버 포트는 8081이다.", now=ctx.now - 40 * DAY)
    commit(ctx, a)
    ws = ws_of(ctx)
    t = ctx.now - 2 * DAY
    events = [ev(1, a.id, "injected", ts=t, cos=0.62), ev(2, a.id, "injected", ts=t + 60, cos=0.7),
              ev(3, a.id, "injected", ts=t + 120, session="s2", cos=0.41),
              ev(4, a.id, "injected", ts=t + DAY, cos=0.55)]
    res = fold_recall_events(ctx, ws, events, snapshot_max_id=4)
    r = ws.get(a.id)
    assert res.injected == 3                                  # (s1, day) once, (s2, day), (s1, day+1)
    assert r.recall_injected_count == 3 and r.recall_injected_strong == 2
    assert r.last_recalled_at == pytest.approx(t + DAY)
    assert r.last_used_at is None
    assert strength.t_ref(r) == strength.t_ref(a)              # t_ref unchanged
    assert ctx.stats.recall_injected == 3 and res.new_cursor == 4


def test_t16_cron_and_other_platforms_ignored(ctx):
    a = row(ctx, "매일 크론이 보는 백업 점검 규칙이다.")
    commit(ctx, a)
    ws = ws_of(ctx)
    events = [ev(1, a.id, "used", ts=ctx.now - 10, platform="cron"),
              ev(2, a.id, "injected", ts=ctx.now - 10, platform="cron", cos=0.9),
              ev(3, a.id, "tool_hit", ts=ctx.now - 10, platform=None)]
    res = fold_recall_events(ctx, ws, events, snapshot_max_id=3)
    assert res.ignored_platform == 3 and ws.ops == []
    assert ctx.stats.recall_ignored_platform == 3


def test_t16_used_and_tool_hit_move_t_ref_and_revive(ctx):
    d = row(ctx, "휴면 상태였던 오래된 사실 하나가 있다.", status="dormant", now=ctx.now - 300 * DAY)
    e = row(ctx, "도구 검색으로 찾은 다른 휴면 사실.", status="dormant", now=ctx.now - 300 * DAY)
    commit(ctx, d, e)
    ws = ws_of(ctx)
    t = ctx.now - 3600
    res = fold_recall_events(ctx, ws, [ev(1, d.id, "used", ts=t), ev(2, e.id, "tool_hit", ts=t, mode="keyword")],
                             snapshot_max_id=2)
    rd, re_ = ws.get(d.id), ws.get(e.id)
    assert rd.status == "active" and re_.status == "active"
    assert rd.recall_used_count == 1 and rd.last_used_at == pytest.approx(t)
    # F-18: a tool_hit counts once in the reinforcement r (search_hit_count only)
    assert re_.search_hit_count == 1 and re_.recall_used_count == 0
    assert strength.reinforcement(re_) == 1.0 and strength.t_ref(re_) == pytest.approx(t)
    assert strength.t_ref(rd) == pytest.approx(t)
    assert set(res.revived) == {d.id, e.id}
    reasons = sorted(o.reason for o in ws.ops if o.op == "status")
    assert reasons == ["revived:tool_hit", "revived:used"]
    assert ctx.stats.revived == 2


def test_shadow_counted_only_and_events_after_snapshot_wait(ctx):
    a = row(ctx, "그림자 모드에서 계산만 된 기억.")
    commit(ctx, a)
    ws = ws_of(ctx)
    res = fold_recall_events(ctx, ws, [ev(1, a.id, "shadow", ts=ctx.now, cos=0.44),
                                       ev(5, a.id, "used", ts=ctx.now)], snapshot_max_id=4)
    assert res.shadow == 1 and res.shadow_cos == [0.44] and res.used == 0
    assert ws.ops == [] and res.new_cursor == 4


def test_inbox_ids_resolve_and_unknown_counted(ctx):
    a = row(ctx, "오늘 기억해 달라고 한 사실이다 여기.", origin_keys=["inbox:7"])
    commit(ctx, a)
    ws = ws_of(ctx)
    assert resolve_memory_id(ws, "inbox:7") == a.id
    assert resolve_memory_id(ws, "inbox:8") is None
    res = fold_recall_events(ctx, ws, [ev(1, "inbox:7", "used", ts=ctx.now - 5, mode="inbox"),
                                       ev(2, "nope", "used", ts=ctx.now - 5)], snapshot_max_id=2)
    assert res.used == 1 and res.unknown_ids == 1
    assert ws.get(a.id).recall_used_count == 1
