"""Regressions found by the §11.3 E2E run on a sandbox copy (DEVIATIONS E2E-n)."""

from __future__ import annotations

import re

import pytest

from hermesyume import clock, prompts
from hermesyume.gates import anchor_present, gate_claim
from hermesyume.types import Claim, Rejection
from tests.dream.test_nrem_gates import NOW, TS, m, raw, win
from tests.dream.test_rem_rem import by_source, inbox, run
from tests.dream.test_rem_helpers import deps  # noqa: F401

DATE = clock.kst_date(TS)


# ── E2E-1: extract prompt kind decision rules (E1: 8081 → reference, 장애 보고 절차 → procedure) ──

def test_extract_prompt_has_kind_decision_rules():
    s = prompts.EXTRACT_SYSTEM
    sec = s[s.index("[kind 고르는 법"):s.index("[level:")]
    assert "포트" in sec and "reference (fact 아님)" in sec
    assert "procedure" in sec and "rule이 아니다" in sec
    assert "fact는 다른 kind가 하나도 맞지 않을 때만" in sec
    chk = s[s.index("[출력 전 확인]"):s.index("[출력] ")]
    assert "YYYY-MM-DD 기준" in chk and "kind가 fact인 주장을 다시 본다" in chk
    assert not re.search(r"\d", sec + chk)                 # still no numeric anchors


# ── E2E-2: "현재/지금 X" is anchored to the evidence date, not rejected ───────────────────

@pytest.mark.parametrize("text,want", [
    ("Orion 검수 당번은 7조에서 넘어갔고 현재 9조가 당번이다.",
     f"Orion 검수 당번은 7조에서 넘어갔고 {DATE} 기준 9조가 당번이다."),
    ("지금은 9조가 Orion 검수 당번을 맡는다.", f"{DATE} 기준 9조가 Orion 검수 당번을 맡는다."),
    ("현재, Orion 검수 당번은 9조다.", f"{DATE} 기준, Orion 검수 당번은 9조다."),
])
def test_anchor_present(text, want):
    assert anchor_present(text, TS) == want


@pytest.mark.parametrize("text", [
    "지금까지 담당자는 9조 당번이었다.",                 # not the standalone word
    "삼성전자 현재가는 7만 원이다 정말로.",               # 현재가 = current price
    "현재 계획은 다음 주에 출시하는 것이다.",              # another relative word remains → still rejected
    "2026-09-28 기준 현재 담당자는 9조다.",              # already absolute
])
def test_anchor_present_leaves_other_texts(text):
    assert anchor_present(text, TS) == text
    assert anchor_present("현재 9조가 당번이다.", None) == "현재 9조가 당번이다."


def test_gate_anchors_present_claim(cfg, scanner):
    w = win([m("U#1", "user", "Orion 검수 당번은 7조에서 넘어갔고 지금은 9조가 당번이야."),
             m("A#2", "assistant", "알겠습니다.")])
    out = gate_claim(raw("Orion 검수 당번은 7조에서 넘어갔고 현재 9조가 당번이다.",
                         subject="Orion 현재 검수 당번", evidence=["U#1"]),
                     w, scanner=scanner, cfg=cfg, now=NOW)
    assert isinstance(out, Claim), out
    assert out.text == f"Orion 검수 당번은 7조에서 넘어갔고 {DATE} 기준 9조가 당번이다."
    assert out.subject == "Orion 검수 당번"
    # another relative word next to 현재 → N4 still rejects
    bad = gate_claim(raw("현재 계획은 다음 주에 Orion를 출시하는 것이다.", evidence=["U#1"]),
                     w, scanner=scanner, cfg=cfg, now=NOW)
    assert isinstance(bad, Rejection) and bad.reason == "relative_time"


def test_remember_text_anchored(ctx, deps):
    ts = ctx.now - 3600
    inbox(ctx, "remember", text="Orion 검수 당번은 7조에서 넘어갔고 현재 9조가 당번이다.", kind="fact", ts=ts)
    run(ctx)
    rows = by_source(ctx, "tool:yume_remember")
    assert len(rows) == 1
    assert rows[0].text == f"Orion 검수 당번은 7조에서 넘어갔고 {clock.kst_date(ts)} 기준 9조가 당번이다."


# ── E2E-3: fact preservation ignores predicate conjugations, keeps nouns/numbers ───────────

def test_preservation_ignores_predicate_forms():
    from hermesyume import vecutil as vu
    a = "Orion 요금 확인은 매뉴얼을 먼저 보고 rates_cli.py find로 찾아야 한다."
    b = "Orion 요금 질문은 항상 요금표부터 확인한다. 요금 확인은 매뉴얼을 먼저 보고 rates_cli.py find로 조회한다."
    merged = "Orion 요금 확인은 항상 매뉴얼을 먼저 보고 rates_cli.py find로 조회하며, 요금 질문도 항상 요금표부터 확인한다."
    assert vu.preservation_check(a, b, merged) == (True, [])
    # a dropped noun is still a failure
    ok, miss = vu.preservation_check(a, "앞으로 Orion 요금 질문은 항상 매뉴얼부터 확인해야 한다.",
                                     "Orion 요금 확인은 앞으로 항상 매뉴얼을 먼저 보고 rates_cli.py find로 찾아야 한다.")
    assert not ok and miss == ["질문"]
    # 하다-verb whose noun stem disappeared → still missing
    ok, miss = vu.preservation_check("서버 로그는 매일 백업한다.", "백업 위치는 NAS다.", "서버 로그는 NAS에 둔다.")
    assert not ok and "백업한" in miss
    # short nouns that look like endings stay required
    ok, miss = vu.preservation_check("제출 기한은 2026-10-10이다.", "제출처는 4조다.", "제출은 2026-10-10까지 4조에게.")
    assert not ok and "기한" in miss


@pytest.mark.parametrize("tok,want", [("조회한", "조회"), ("확인해야", "확인"), ("찾아야", ""), ("기한", None),
                                      ("역할", None), ("질문", None), ("rate", None), ("여야", None)])
def test_predicate_stem(tok, want):
    from hermesyume.vecutil import predicate_stem
    assert predicate_stem(tok) == want


# ── E2E-4: one removal is reported once; a missing core_required row is a note, not a change ──

def test_core_remove_reported_once(ctx, fake_home, deps):
    from tests.dream.test_rem_core_check import night
    from tests.dream.test_rem_helpers import commit, live_insert
    from tests.fixtures.hermes_home import read_core, write_core
    night(ctx, days=0)
    inv = next(r for r in ctx.store.load_working_set().values() if "가계부" in r.text)
    inv.core_required = True
    inv.pinned = True
    commit(ctx, inv)
    entries = read_core(fake_home.user_md)
    write_core(fake_home.user_md, [e for e in entries if "가계부" not in e])   # the memory tool removed it
    live_insert(ctx, "inbox", ts=ctx.now + 600, session_id="s", platform="cli", op="core_remove",
                target="user", text=inv.text, old_text="가계부", status="pending")
    c1, _ = night(ctx, days=1)
    mine = [ch for ch in c1.report.core_changes if "가계부" in ch["text"]]
    assert [ch["change"] for ch in mine] == ["remove"]
    assert c1.stats.core_changes == len(c1.report.core_changes)
    assert sum("'가계부 관리'가 빠졌습니다" in n for n in c1.report.notes) == 1
    r = ctx.store.load_working_set()[inv.id]
    assert r.in_core is False and r.pinned and r.status == "active"
    c2, _ = night(ctx, days=2)
    assert c2.report.core_changes == [] and c2.stats.core_changes == 0
    assert sum("'가계부 관리'가 빠졌습니다" in n for n in c2.report.notes) == 1   # standing note


def test_extract_prompt_skips_assistant_general_knowledge():
    s = prompts.EXTRACT_SYSTEM
    dont = s[s.index("[추출하지 말 것]"):s.index("[kind:")]
    assert "일반 지식" in dont and "사용자가 정하거나 알려 준 것만 남긴다" in dont
    assert "되풀이한 말" in dont


# ── E2E-5: U3 echo guard — assistant-only claims never add to/replace user-grounded memory ──

def _upserter(ctx):
    from hermesyume import plan as P
    from hermesyume.upsert import Upserter
    ws = P.WorkingSet(ctx.store.load_working_set(), run_id=ctx.run_id, now=ctx.now,
                      embed_model=ctx.embedder.model_id)
    return Upserter(ctx, ws)


DAY_S = 86400.0


@pytest.mark.parametrize("rel", ["state_change", "different_aspects"])
def test_echo_of_user_grounded_row_is_record_only(ctx, deps, rel):
    from tests.dream.test_rem_helpers import claim, commit, row, vec_like
    from tests.fakes import judge_json
    a = row(ctx, "Orion 검수 당번은 7조에서 넘어갔고 2026-10-02 기준 9조가 당번이다.",
            subject="Orion 검수 당번", event_time=ctx.now - 10 * DAY_S, user_evidence_count=2,
            user_session_count=1)
    commit(ctx, a)
    u = _upserter(ctx)
    echo = claim(ctx, "Orion 검수 당번은 2026-10-02 기준으로 7조에서 9조로 변경되었다.", user=False,
                 subject="Orion 검수 당번", et=ctx.now - DAY_S, vector=vec_like(a.vector, 0.86, "echo"))
    ctx.llm.queue("judge", judge_json(("c1", rel, "new")))
    out = u.upsert(echo)
    assert out.action == "reinforce_noop" and out.memory_id == a.id and out.ops == []
    assert u.ws.get(a.id).status == "active" and len(u.ws.rows) == len(ctx.store.load_working_set())
    assert ctx.stats.created == 0 and ctx.stats.superseded == 0


def test_echo_does_not_resurrect_expired_schedule(ctx, deps):
    from tests.dream.test_rem_helpers import claim, commit, row, vec_like
    from tests.fakes import judge_json
    s = row(ctx, "Orion 데모 마감일은 2026-10-11이다.", kind="schedule", subject="Orion 데모 마감",
            event_time=ctx.now - 40 * DAY_S, valid_until=ctx.now - 30 * DAY_S, status="expired",
            user_evidence_count=2, user_session_count=1)
    commit(ctx, s)
    u = _upserter(ctx)
    echo = claim(ctx, "Orion 데모 마감일은 2026-10-11이다.", kind="fact", user=False,
                 subject="Orion 데모 마감", vector=vec_like(s.vector, 0.999, "e"))
    ctx.llm.queue("judge", judge_json(("c1", "different_aspects", "new")))
    out = u.upsert(echo)
    assert out.action == "reinforce_noop" and ctx.stats.created == 0
    assert u.ws.get(s.id).status == "expired"


def test_echo_guard_scope(ctx, deps):
    """Assistant-only vs assistant-only rows keep the old behaviour; user claims still supersede."""
    from tests.dream.test_rem_helpers import claim, commit, row, vec_like
    from tests.fakes import judge_json
    weak = row(ctx, "Orion 빌드 서버는 젠킨스를 쓴다고 한다.", subject="Orion 빌드",
               event_time=ctx.now - 10 * DAY_S)                      # no user evidence
    assert weak.user_evidence_count == 0
    commit(ctx, weak)
    u = _upserter(ctx)
    c = claim(ctx, "Orion 빌드 서버는 깃허브 액션으로 바뀌었다고 한다.", user=False,
              subject="Orion 빌드", et=ctx.now - DAY_S, vector=vec_like(weak.vector, 0.86, "w"))
    ctx.llm.queue("judge", judge_json(("c1", "state_change", "new")))
    assert u.upsert(c).action == "state_change"
    g = row(ctx, "Orion 스테이징 포트는 8081이다.", subject="Orion 포트", event_time=ctx.now - 10 * DAY_S,
            user_evidence_count=1, user_session_count=1)
    commit(ctx, g)
    u2 = _upserter(ctx)
    uc = claim(ctx, "Orion 스테이징 포트는 8082로 바뀌었다.", user=True, subject="Orion 포트",
               et=ctx.now - DAY_S, vector=vec_like(g.vector, 0.86, "u"))
    ctx.llm.queue("judge", judge_json(("c1", "state_change", "new")))
    assert u2.upsert(uc).action == "state_change" and u2.ws.get(g.id).status == "superseded"


# ── E2E-7: U3 assistant-only claims — general knowledge is rejected, the agent's own work kept ──

def _qa_window(source: str = "statedb"):
    """User asks a general question; the assistant explains (the E2E junk pattern)."""
    if source == "md":
        sid = "md:/w/2026-09-28.md"
        return win([m("U#md:L1", "user", "파이썬 리스트 정렬은 어떻게 해?", source="md", sid=sid),
                    m("A#md:L2", "assistant", "sorted(리스트)는 새 리스트를, list.sort()는 제자리 정렬을 합니다.",
                      source="md", sid=sid),
                    m("L#md:L3", "agent_log", "[cron] 매일 백업 작업은 KST 기준으로 등록해야 시간이 맞는다.",
                      source="md", sid=sid)], source="md", root="/w/2026-09-28.md")
    return win([m("U#1", "user", "파이썬 리스트 정렬은 어떻게 해?"),
                m("A#2", "assistant", "sorted(리스트)는 새 리스트를, list.sort()는 제자리 정렬을 합니다. "
                                      "참고로 cron 작업은 KST 기준으로 등록해야 시간이 맞았습니다.")])


@pytest.mark.parametrize("kind,target", [
    ("fact", "world"), ("fact", "user"), ("fact", "agent"), ("opinion", "user"), ("event", "agent"),
    ("lesson", "world"), ("reference", "world"), ("procedure", "world"), ("rule", "world"),
    ("decision", "user"), ("project", "user"), ("procedure", "user"), ("state", "agent"),
    ("schedule", "user"), ("decision", "nobody"),           # unknown target → user
])
@pytest.mark.parametrize("source", ["statedb", "md"])
def test_assistant_only_general_knowledge_rejected(cfg, scanner, kind, target, source):
    w = _qa_window(source)
    ref = "A#2" if source == "statedb" else "A#md:L2"
    vu = "2026-10-30" if kind in ("state", "schedule") else None
    out = gate_claim(raw("파이썬에서 sorted는 새 리스트를 돌려주고 list.sort는 제자리 정렬을 한다.",
                         kind=kind, target=target, evidence=[ref], valid_until=vu),
                     w, scanner=scanner, cfg=cfg, now=NOW)
    assert isinstance(out, Rejection), out
    assert out.reason == "assistant_only"
    assert out.detail == f"{kind}/{target if target in ('user', 'agent', 'world') else 'user'}"


@pytest.mark.parametrize("kind", ["procedure", "lesson", "reference", "decision", "project"])
@pytest.mark.parametrize("source", ["statedb", "md"])
def test_assistant_only_agent_work_kept_low(cfg, scanner, paths, kind, source):
    from hermesyume.normalize import compute_importance, normalize_claim
    from hermesyume.types import KIND_BASE
    w = _qa_window(source)
    ref = "A#2" if source == "statedb" else "A#md:L2"
    c = gate_claim(raw("에이전트는 cron 작업을 KST 기준으로 등록해야 실행 시간이 맞는다.", kind=kind,
                       target="agent", evidence=[ref], steps=2 if kind == "procedure" else None),
                   w, scanner=scanner, cfg=cfg, now=NOW)
    assert isinstance(c, Claim), c
    assert c.assistant_only and not c.has_user_evidence and c.target == "agent" and c.status == "active"
    normalize_claim(c, cfg=cfg, paths=paths)
    assert c.importance == compute_importance(kind=kind, level=3, explicit_user=False, user_session_count=0,
                                              assistant_only=True, source=c.source)
    assert c.importance < KIND_BASE[kind]                     # assistant-only penalty (low importance)


def test_assistant_only_lesson_row_is_decaying(cfg, scanner, paths):
    from hermesyume.normalize import normalize_claim
    from hermesyume.strength import compute_tier
    from tests.dream.test_nrem_gates import _row
    c = gate_claim(raw("에이전트는 cron 작업을 KST 기준으로 등록해야 실행 시간이 맞는다.", kind="lesson",
                       target="agent", evidence=["A#2"]), _qa_window(), scanner=scanner, cfg=cfg, now=NOW)
    normalize_claim(c, cfg=cfg, paths=paths)
    assert compute_tier(_row(c)) == "decaying"


@pytest.mark.parametrize("kind", ["rule", "profile", "preference"])
@pytest.mark.parametrize("target", ["user", "agent"])
def test_assistant_only_u2_kinds_still_decaying(cfg, scanner, kind, target):
    """U2 overrides: assistant-only rule/profile/preference stay active (tier decaying)."""
    from hermesyume.strength import compute_tier
    from tests.dream.test_nrem_gates import _row
    w = win([m("U#1", "user", "응"), m("A#2", "assistant", "앞으로 모든 보고는 표로 정리하겠습니다.")])
    c = gate_claim(raw("모든 보고는 표로 정리한다는 방식이 정해져 있다.", kind=kind, target=target,
                       evidence=["A#2"]), w, scanner=scanner, cfg=cfg, now=NOW)
    assert isinstance(c, Claim), c
    assert compute_tier(_row(c)) == "decaying"


@pytest.mark.parametrize("evidence,roles", [
    (["U#1", "A#2"], ["user", "assistant"]),                    # the user asked → user evidence
    (["U#1"], ["user"]),
])
def test_user_evidence_general_fact_not_assistant_only(cfg, scanner, evidence, roles):
    c = gate_claim(raw("사용자는 파이썬 리스트 정렬 방법을 배우는 중이다.", kind="fact", target="user",
                       evidence=evidence), _qa_window(), scanner=scanner, cfg=cfg, now=NOW)
    assert isinstance(c, Claim) and c.evidence_roles == roles


@pytest.mark.parametrize("evidence", [["L#md:L3"], ["A#md:L2", "L#md:L3"]])
def test_agent_log_evidence_is_not_assistant_only(cfg, scanner, evidence):
    c = gate_claim(raw("cron 백업 작업은 KST 기준으로 등록해야 시간이 맞는다.", kind="fact", target="world",
                       evidence=evidence), _qa_window("md"), scanner=scanner, cfg=cfg, now=NOW)
    assert isinstance(c, Claim), c
    assert "agent_log" in c.evidence_roles and not c.assistant_only


def test_assistant_only_secret_keeps_secret_reason(cfg, scanner):
    """The assistant_only check runs last: a secret is still rejected as `secret` (text not logged)."""
    w = win([m("U#1", "user", "키 확인해 줘"), m("A#2", "assistant", "OPENAI 키는 sk-" + "a" * 32 + " 입니다.")])
    out = gate_claim(raw("OpenAI API 키는 sk-" + "a" * 32 + " 이다.", kind="fact", target="world",
                         evidence=["A#2"]), w, scanner=scanner, cfg=cfg, now=NOW)
    assert isinstance(out, Rejection) and out.reason == "secret"


def test_extract_prompt_keeps_explanations_out_of_agent_lessons():
    s = prompts.EXTRACT_SYSTEM
    dont = s[s.index("[추출하지 말 것]"):s.index("[kind:")]
    assert "target agent의 교훈·절차로 바꿔 내지도 않는다" in dont


def test_assistant_only_reason_registered_and_labelled():
    """Every gate reason (assistant_only included) has a Korean Dream Log label."""
    from hermesyume import dream_log, gates
    from hermesyume.types import GATE_REASONS
    assert gates.ASSISTANT_ONLY_REASON in GATE_REASONS
    assert set(GATE_REASONS) <= set(dream_log.REASON_KO)
