"""upsert.py — T7 duplicate, T8 state_change (+U2 protection), T9 8-B regression, T10 preservation
failure, T11 judge failure → judge_pending, T12 in-run twins, R2 re-judge, R3 sweep."""

import pytest

from hermesyume import plan as P
from hermesyume.llm import LLMError
from hermesyume.types import BudgetExceeded
from hermesyume.upsert import Upserter, reinforce_changes
from tests.dream.test_rem_helpers import claim, commit, deps, row, unit, vec_like  # noqa: F401
from tests.fakes import judge_json

DAY = 86400.0


@pytest.fixture
def up(ctx, deps):
    ws = P.WorkingSet(ctx.store.load_working_set(), run_id=ctx.run_id, now=ctx.now,
                      embed_model=ctx.embedder.model_id)
    return Upserter(ctx, ws)


def fresh(ctx, deps_ok=True):
    ws = P.WorkingSet(ctx.store.load_working_set(), run_id=ctx.run_id, now=ctx.now,
                      embed_model=ctx.embedder.model_id)
    return Upserter(ctx, ws)


def finalize_commit(ctx, u):
    P.apply_guard(u.ws, ctx.cfg, mode="live")
    from hermesyume.types import LedgerDelta
    pl = P.build_plan(ctx, u.ws, lance_version_before=ctx.store.version(), ledger_delta=LedgerDelta(),
                      inbox_consume_ids=[], inbox_skip_ids=[], docs=[])
    P.commit_plan(ctx, pl)
    return pl


# ── T7 ───────────────────────────────────────────────────────────────────────

def test_t7_auto_duplicate_reinforces_without_llm(ctx, deps):
    e = row(ctx, "Orion 결제 스테이징 서버 포트는 8081이다.", kind="reference", subject="Orion 포트",
            user_evidence_count=1, user_session_count=1, source_message_ids=["s:1"], source_session_ids=["sA"],
            last_user_evidence_at=ctx.now - 10 * DAY)
    commit(ctx, e)
    u = fresh(ctx)
    c = claim(ctx, "Orion 스테이징 서버 포트는 8081번이다.", kind="reference", subject="Orion 포트",
              vector=vec_like(e.vector, 0.97, "dup"), sessions=["sB"])
    out = u.upsert(c)
    assert out.action == "duplicate" and out.memory_id == e.id
    assert ctx.llm.calls_of("judge") == []
    r = u.ws.get(e.id)
    assert r.evidence_count == e.evidence_count + 1
    assert r.user_evidence_count == 2 and r.user_session_count == 2
    assert c.origin_key in r.origin_keys and "sB" in r.source_session_ids
    assert r.last_user_evidence_at == pytest.approx(c.last_user_evidence_at)
    assert len(u.ws.new_ids) == 0
    finalize_commit(ctx, u)
    assert ctx.store.count() == 1


def test_t7_assistant_evidence_does_not_reinforce(ctx, deps):
    e = row(ctx, "Orion 장애 보고 절차는 알림 확인부터 시작한다.", kind="procedure",
            user_evidence_count=1, user_session_count=1, source_message_ids=["s:1"],
            last_user_evidence_at=ctx.now - 5 * DAY, importance=0.7)
    commit(ctx, e)
    u = fresh(ctx)
    c = claim(ctx, "Orion 장애 보고 절차는 알림 확인부터 시작합니다.", kind="procedure", user=False,
              vector=vec_like(e.vector, 0.98, "asst"))
    out = u.upsert(c)
    assert out.action == "reinforce_noop"
    r = u.ws.get(e.id)
    assert r.evidence_count == e.evidence_count + 1                     # recorded
    assert r.user_evidence_count == 1 and r.user_session_count == 1    # not strengthened
    assert r.last_user_evidence_at == pytest.approx(e.last_user_evidence_at)
    assert r.importance == pytest.approx(e.importance)
    assert ctx.stats.reinforce_noop == 1 and ctx.stats.reinforced == 0


def test_t7_numeric_guard_12_vs_13_goes_to_judge(ctx, deps):
    e = row(ctx, "도서관 사물함 번호는 12번이다.", kind="state", event_time=ctx.now - 5 * DAY)
    commit(ctx, e)
    u = fresh(ctx)
    c = claim(ctx, "도서관 사물함 번호는 13번이다.", kind="state", vector=vec_like(e.vector, 0.97, "n"))
    ctx.llm.queue("judge", judge_json(("c1", "state_change", "new")))
    out = u.upsert(c)
    assert len(ctx.llm.calls_of("judge")) == 1
    assert out.action == "state_change"


def test_reinforce_changes_counts_only_new_evidence(ctx, deps):
    e = row(ctx, "같은 메시지를 두 번 보면 한 번만 센다.", source_message_ids=["s:5"], evidence_count=1,
            user_evidence_count=1, user_session_count=1, source_session_ids=["sA"])
    c = claim(ctx, e.text, keys=["s:5"], sessions=["sA"])
    ch, user_ev = reinforce_changes(e, c, now=ctx.now)
    assert "evidence_count" not in ch and not user_ev


# ── T8 ───────────────────────────────────────────────────────────────────────

def test_t8_state_change_chain_and_backlog(ctx, deps):
    a = row(ctx, "Orion 검수 당번은 7조다.", subject="Orion 검수 당번", event_time=ctx.now - 10 * DAY)
    commit(ctx, a)
    u = fresh(ctx)
    b = claim(ctx, "Orion 검수 당번은 9조다.", subject="Orion 검수 당번", et=ctx.now - 2 * DAY,
              vector=vec_like(a.vector, 0.86, "b"))
    ctx.llm.queue("judge", judge_json(("c1", "state_change", "new")))
    ob = u.upsert(b)
    assert ob.action == "state_change"
    ra, rb = u.ws.get(a.id), u.ws.get(ob.memory_id)
    assert ra.status == "superseded" and ra.superseded_by == rb.id
    assert ra.valid_until == pytest.approx(b.event_time)
    assert rb.status == "active" and rb.supersedes == [a.id]
    # backlog: an older claim judged against both → inserted already superseded by the newest
    c = claim(ctx, "Orion 검수 당번은 5조다.", subject="Orion 검수 당번", et=ctx.now - 30 * DAY,
              vector=vec_like(a.vector, 0.85, "c"))
    ctx.llm.on("judge", lambda m: judge_json(("c1", "state_change", "existing"), ("c2", "state_change", "existing")))
    oc = u.upsert(c)
    rc = u.ws.get(oc.memory_id)
    assert oc.action == "inserted_superseded"
    assert rc.status == "superseded" and rc.superseded_by == rb.id
    assert u.ws.get(rb.id).status == "active"
    finalize_commit(ctx, u)
    stored = ctx.store.load_working_set()
    assert stored[a.id].status == "superseded" and stored[rb.id].status == "active"


@pytest.mark.parametrize("kind_of_protection", ["pinned", "durable"])
def test_t8_protected_target_without_explicit_user_gets_separate_row(ctx, deps, kind_of_protection):
    kw = dict(pinned=True, source="core:user") if kind_of_protection == "pinned" else \
        dict(user_evidence_count=1, user_session_count=1, explicit_user=True)
    p = row(ctx, "**가계부 관리:** 지출 기록은 공용 가계부 DB가 단일 원장이다.", kind="rule",
            subject="가계부 관리", event_time=ctx.now - 30 * DAY, **kw)
    commit(ctx, p)
    assert p.tier == ("pinned" if kind_of_protection == "pinned" else "durable")
    u = fresh(ctx)
    q = claim(ctx, "지출 기록은 이제 엑셀 시트가 기준이다.", kind="rule", subject="가계부 관리",
              user=True, explicit=False, vector=vec_like(p.vector, 0.88, "q"))
    ctx.llm.queue("judge", judge_json(("c1", "state_change", "new")))
    out = u.upsert(q)
    assert out.action == "related"
    rp, rq = u.ws.get(p.id), u.ws.get(out.memory_id)
    assert rp.status == "active" and rq.status == "active"
    assert out.memory_id in rp.related_ids and p.id in rq.related_ids
    assert ctx.alerts == []
    g = P.apply_guard(u.ws, ctx.cfg, mode="live")
    assert not g.held
    finalize_commit(ctx, u)
    assert ctx.store.load_working_set()[p.id].status == "active"


def test_t8_explicit_user_may_supersede_durable(ctx, deps):
    p = row(ctx, "주간 회의는 월요일 10시에 한다.", kind="rule", subject="주간 회의",
            user_evidence_count=1, user_session_count=1, explicit_user=True, event_time=ctx.now - 20 * DAY)
    commit(ctx, p)
    u = fresh(ctx)
    q = claim(ctx, "주간 회의는 앞으로 화요일 10시에 한다.", kind="rule", subject="주간 회의", explicit=True,
              vector=vec_like(p.vector, 0.9, "q2"))
    ctx.llm.queue("judge", judge_json(("c1", "state_change", "new")))
    out = u.upsert(q)
    assert out.action == "state_change" and u.ws.get(p.id).status == "superseded"
    assert not P.apply_guard(u.ws, ctx.cfg, mode="live").held


# ── T9 / T10 ─────────────────────────────────────────────────────────────────

def test_t9_8b_regression_no_information_loss(ctx, deps):
    """N conflicts with E1 and E2: consolidation uses the CURRENT working text; nothing is lost."""
    e1 = row(ctx, "주간 회의는 월요일 10시에 한다.", subject="주간 회의")
    e2 = row(ctx, "주간 회의 장소는 3층 회의실이다.", subject="주간 회의 장소", vector=vec_like(e1.vector, 0.6, "e2"))
    commit(ctx, e1, e2)
    u = fresh(ctx)
    # cos(n1, e1) ≈ 0.96 (top), cos(n1, e2) ≈ 0.79: N conflicts with both
    n1 = claim(ctx, "주간 회의 진행자는 김대리다.", subject="주간 회의", vector=unit(0.7 * e1.vector + 0.3 * e2.vector))
    ctx.llm.queue("judge", judge_json(("c1", "different_aspects", "same"), ("c2", "different_aspects", "same")))
    merged1 = "주간 회의는 월요일 10시에 하고 진행자는 김대리다."
    ctx.llm.queue("consolidate", {"text": merged1})
    o1 = u.upsert(n1)
    assert o1.action == "consolidated" and o1.memory_id == e1.id
    assert u.ws.get(e1.id).text == merged1
    assert e2.id in u.ws.get(e1.id).related_ids                       # other DA candidate linked
    # second claim against the same row: the consolidate prompt must see merged1, not E1's original
    n2 = claim(ctx, "주간 회의 자료는 공유 드라이브에 올린다.", subject="주간 회의",
               vector=vec_like(u.ws.get(e1.id).vector, 0.84, "n2"))
    ctx.llm.queue("judge", judge_json(("c1", "different_aspects", "same")))
    merged2 = "주간 회의는 월요일 10시에 하고 진행자는 김대리이며 자료는 공유 드라이브에 올린다."
    ctx.llm.queue("consolidate", {"text": merged2})
    o2 = u.upsert(n2)
    prompt = ctx.llm.calls_of("consolidate")[1]["messages"][-1]["content"]
    assert merged1 in prompt and e1.text not in prompt.replace(merged1, "")
    assert o2.action == "consolidated"
    finalize_commit(ctx, u)
    stored = ctx.store.load_working_set()
    texts = " ".join(r.text for r in stored.values())
    for fact in ("월요일", "10시", "김대리", "공유 드라이브", "3층"):
        assert fact in texts
    assert stored[e2.id].text == e2.text
    hist = [h for h in ctx.store.history(memory_id=e1.id) if h.op == "consolidate"]
    assert len(hist) == 2 and any(e1.text in h.before_json for h in hist)


def test_t9_state_change_against_two_keeps_both_texts(ctx, deps):
    e1 = row(ctx, "주간 회의는 월요일 10시에 한다.", subject="주간 회의", event_time=ctx.now - 20 * DAY)
    e2 = row(ctx, "주간 회의는 월요일 11시로 옮겼다.", subject="주간 회의", event_time=ctx.now - 10 * DAY)
    commit(ctx, e1, e2)
    u = fresh(ctx)
    n = claim(ctx, "주간 회의는 화요일 9시로 바뀌었다.", subject="주간 회의", et=ctx.now - DAY,
              vector=vec_like(e1.vector, 0.83, "n"))
    ctx.llm.queue("judge", judge_json(("c1", "state_change", "new"), ("c2", "state_change", "new")))
    out = u.upsert(n)
    finalize_commit(ctx, u)
    stored = ctx.store.load_working_set()
    assert stored[e1.id].text == e1.text and stored[e2.id].text == e2.text
    assert {stored[e1.id].status, stored[e2.id].status} == {"superseded"}
    assert set(stored[out.memory_id].supersedes) == {e1.id, e2.id}


def test_t10_preservation_failure_keeps_both(ctx, deps):
    e = row(ctx, "Orion 데모 마감은 2026-10-10이다.", subject="Orion 데모")
    commit(ctx, e)
    u = fresh(ctx)
    n = claim(ctx, "Orion 데모 발표자는 4조다.", subject="Orion 데모", vector=vec_like(e.vector, 0.8, "t10"))
    ctx.llm.queue("judge", judge_json(("c1", "different_aspects", "same")))
    ctx.llm.queue("consolidate", {"text": "Orion 데모는 4조가 발표한다."})     # drops the date
    out = u.upsert(n)
    assert out.action == "related"
    assert u.ws.get(e.id).text == e.text
    new = u.ws.get(out.memory_id)
    assert new.text == n.text and e.id in new.related_ids and new.id in u.ws.get(e.id).related_ids
    assert ctx.stats.consolidated == 0


def test_different_aspects_pinned_target_never_rewritten(ctx, deps):
    p = row(ctx, "**호칭:** 사장님이라고 부른다.", kind="profile", pinned=True, source="core:user")
    commit(ctx, p)
    u = fresh(ctx)
    n = claim(ctx, "사장님은 짧은 답변을 좋아한다.", kind="preference", vector=vec_like(p.vector, 0.8, "pin"))
    ctx.llm.queue("judge", judge_json(("c1", "different_aspects", "same")))
    out = u.upsert(n)
    assert out.action == "related" and ctx.llm.calls_of("consolidate") == []
    assert u.ws.get(p.id).text == p.text


# ── T11 ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("failure", ["garbage", "enum_garbage", "llm_error", "budget", "max_calls"])
def test_t11_judge_failure_is_judge_pending_never_unrelated(ctx, deps, failure):
    e = row(ctx, "Orion 요금 확인은 요금표를 먼저 본다.", kind="rule", subject="Orion 요금")
    commit(ctx, e)
    u = fresh(ctx)
    n = claim(ctx, "Orion 요금 질문은 rates_cli.py로 찾는다.", kind="procedure", subject="Orion 요금",
              vector=vec_like(e.vector, 0.8, "t11"))
    if failure == "garbage":
        ctx.llm.queue("judge", "이건 JSON이 아님")
        ctx.llm.queue("judge_enum", "역시 아님")
    elif failure == "enum_garbage":
        ctx.llm.queue("judge", {"relations": [{"id": "c1", "type": "maybe"}]})
        ctx.llm.queue("judge_enum", {"type": "related"})
    elif failure == "llm_error":
        ctx.llm.queue("judge", LLMError("HTTP 500"))
    elif failure == "budget":
        ctx.llm.queue("judge", BudgetExceeded("llm_calls"))
    else:
        u.judge_calls = int(ctx.cfg.max_judge_calls)
    out = u.upsert(n)
    assert out.action == "judge_pending"
    assert all(j.relation == "unknown" for j in out.judgements)
    assert u.ws.get(out.memory_id).judge_pending is True
    assert ctx.stats.judge_pending == 1


def test_t11_enum_retry_rescues(ctx, deps):
    e = row(ctx, "알림 채널은 텔레그램 DM이다 항상.", kind="preference")
    commit(ctx, e)
    u = fresh(ctx)
    n = claim(ctx, "알림 채널은 텔레그램 DM으로 받는다.", kind="preference", vector=vec_like(e.vector, 0.9, "enum"))
    ctx.llm.queue("judge", "{broken")
    ctx.llm.queue("judge_enum", {"type": "duplicate", "newer": "same"})
    out = u.upsert(n)
    assert out.action == "duplicate" and out.memory_id == e.id


# ── T12 ──────────────────────────────────────────────────────────────────────

def test_t12_twins_in_same_run_make_one_row(ctx, deps):
    u = fresh(ctx)
    a = claim(ctx, "Orion 데모 마감은 2026-10-10이다.", kind="schedule", subject="Orion 데모 마감")
    b = claim(ctx, "Orion 데모 마감일은 2026-10-10이다.", kind="schedule", subject="Orion 데모 마감",
              vector=vec_like(a.vector, 0.98, "twin"))
    oa = u.upsert(a)
    ob = u.upsert(b)
    assert oa.action == "inserted" and ob.action == "duplicate" and ob.memory_id == oa.memory_id
    assert len(u.ws.new_ids) == 1
    # judge path also finds the twin through the run's new rows
    c = claim(ctx, "Orion 데모 일정 마감은 10월 10일이다.", kind="schedule", subject="Orion 데모",
              vector=vec_like(a.vector, 0.9, "twin2"))
    ctx.llm.queue("judge", judge_json(("c1", "duplicate", "same")))
    oc = u.upsert(c)
    assert oc.memory_id == oa.memory_id and len(u.ws.new_ids) == 1


def test_idempotent_origin_key(ctx, deps):
    u = fresh(ctx)
    a = claim(ctx, "같은 창의 같은 주장은 두 번 들어가지 않는다.", origin="win#3")
    assert u.upsert(a).action == "inserted"
    assert u.upsert(claim(ctx, "같은 창의 같은 주장은 두 번 들어가지 않는다.", origin="win#3")).action == "skip_idempotent"


# ── R2 / R3 ──────────────────────────────────────────────────────────────────

def test_r2_rejudge_pending_duplicate_merges(ctx, deps):
    e = row(ctx, "Orion 스테이징 포트는 8081이다.", kind="reference", created_at=ctx.now - 5 * DAY,
            user_evidence_count=1, user_session_count=1, source_message_ids=["s:1"])
    pend = row(ctx, "Orion 스테이징 서버 포트 8081번.", kind="reference", judge_pending=True,
               created_at=ctx.now - DAY, user_evidence_count=1, user_session_count=1,
               source_message_ids=["s:2"], vector=vec_like(e.vector, 0.9, "pend"))
    commit(ctx, e, pend)
    u = fresh(ctx)
    ctx.llm.queue("judge", judge_json(("c1", "duplicate", "same")))
    outs = u.rejudge_pending(limit=50)
    assert [o.action for o in outs] == ["duplicate"]
    rp, re_ = u.ws.get(pend.id), u.ws.get(e.id)
    assert rp.status == "superseded" and rp.superseded_by == e.id and not rp.judge_pending
    assert "s:2" in re_.source_message_ids and re_.user_evidence_count == 2


def test_r2_unrelated_clears_flag_and_unknown_keeps_it(ctx, deps):
    e = row(ctx, "점심 메뉴는 보통 김치찌개다 회사 근처.")
    p1 = row(ctx, "회사 근처 식당 목록을 정리했다 지난주.", judge_pending=True, vector=vec_like(e.vector, 0.8, "p1"))
    commit(ctx, e, p1)
    u = fresh(ctx)
    ctx.llm.queue("judge", judge_json(("c1", "unrelated", "same")))
    u.rejudge_pending(limit=50)
    assert u.ws.get(p1.id).judge_pending is False
    u2 = fresh(ctx)
    u2.rejudge_pending(limit=50)            # default ScriptedLLM: no relations → unknown
    assert u2.ws.get(p1.id).judge_pending is True


def test_r3_sweep_judges_new_rows_once(ctx, deps):
    old = row(ctx, "주간 보고는 금요일 오후에 보낸다.", created_at=ctx.now - 2 * DAY, updated_at=ctx.now - 2 * DAY)
    commit(ctx, old)
    u = fresh(ctx)
    # a row consolidated/inserted in this run that R0 did not pair with `old`
    new = row(ctx, "주간 보고는 금요일 오후 5시에 보낸다.", vector=vec_like(old.vector, 0.9, "sw"))
    u.ws.insert(new)
    ctx.llm.queue("judge", judge_json(("c1", "state_change", "new")))
    outs = u.sweep(days=7, min_cos=0.82, max_pairs=100)
    assert len(outs) == 1 and len(ctx.llm.calls_of("judge")) == 1
    assert u.ws.get(old.id).status == "superseded" and u.ws.get(old.id).superseded_by == new.id
    assert u.sweep(days=7, min_cos=0.82, max_pairs=100) == []      # same pair never re-judged
    # an unchanged next night: no anchors → no LLM
    u3 = fresh(ctx)
    assert u3.sweep(days=7, min_cos=0.82, max_pairs=100) == []


def test_suppressed_claim_dropped(ctx, deps):
    from hermesyume.types import SuppressRow, text_sha
    gone = row(ctx, "보라색 고래 다음 숫자는 7341이다 암호.")
    ctx.store.commit(suppress=[SuppressRow(id=gone.id, vector=gone.vector, text_sha=text_sha(gone.text),
                                           kind="fact", created_at=ctx.now, reason="forget")])
    u = fresh(ctx)
    out = u.upsert(claim(ctx, "보라색 고래 다음 숫자는 7341이다 암호.", subject=gone.subject))
    assert out.action == "suppressed" and not u.ws.new_ids
    out2 = u.upsert(claim(ctx, "다른 말로 된 같은 비밀 암호 문장", vector=vec_like(gone.vector, 0.95, "sup")))
    assert out2.action == "suppressed"


def test_user_restating_expired_state_revives_it(ctx, deps):
    e = row(ctx, "도서관 정기 열람권을 보유 중이다 2026-09.", kind="state", event_time=ctx.now - 30 * DAY,
            valid_until=ctx.now - 16 * DAY, status="expired", user_evidence_count=1, user_session_count=1,
            source_message_ids=["s:1"])
    commit(ctx, e)
    u = fresh(ctx)
    c = claim(ctx, "도서관 정기 열람권을 보유 중이다 2026-09.", kind="state", et=ctx.now - DAY,
              valid_until=ctx.now + 13 * DAY, vector=vec_like(e.vector, 0.99, "st"))
    out = u.upsert(c)
    r = u.ws.get(e.id)
    assert out.action == "duplicate" and r.status == "active"
    assert r.valid_until == pytest.approx(ctx.now + 13 * DAY)
    # assistant-only re-statement does not revive
    e2 = row(ctx, "다른 만료된 상태 기억 하나 2026-08.", kind="state", event_time=ctx.now - 40 * DAY,
             valid_until=ctx.now - 26 * DAY, status="expired")
    commit(ctx, e2)
    u2 = fresh(ctx)
    c2 = claim(ctx, "다른 만료된 상태 기억 하나 2026-08.", kind="state", user=False, et=ctx.now - DAY,
               valid_until=ctx.now + 13 * DAY, vector=vec_like(e2.vector, 0.99, "st2"))
    u2.upsert(c2)
    assert u2.ws.get(e2.id).status == "expired"


def test_r0_revives_dormant_on_user_evidence(ctx, deps):
    d = row(ctx, "오래 잊혔던 프로젝트 이름은 블루웨일이다.", kind="project", status="dormant",
            now=ctx.now - 300 * DAY)
    commit(ctx, d)
    u = fresh(ctx)
    out = u.upsert(claim(ctx, "프로젝트 이름은 블루웨일이다 (다시 언급).", kind="project",
                         vector=vec_like(d.vector, 0.97, "dz")))
    assert out.action == "duplicate" and u.ws.get(d.id).status == "active"
    assert ctx.report.revived and ctx.report.revived[0]["id"] == d.id
