"""Regression tests for the review findings fixed after integration (DEVIATIONS F-1 … F-34).
Each test names its finding."""

from __future__ import annotations

import json
import os
import tarfile
import time
from pathlib import Path

import pytest

from hermesyume import clock, core_check, docs_writer, gates, plan as P, recall_fold, rem, retention, strength
from hermesyume import vecutil as vu
from hermesyume.types import (Message, RecallEvent, SuppressRow, Window, WindowState, suppress_reason,
                              text_sha)
from hermesyume.upsert import Upserter
from tests.dream.test_rem_helpers import (classify_all, claim, commit, deps, live_insert, next_ctx,  # noqa: F401
                                          nres, pre, row, vec_like)
from tests.fakes import judge_json

DAY = 86400.0


def fresh(ctx):
    ws = P.WorkingSet(ctx.store.load_working_set(), run_id=ctx.run_id, now=ctx.now,
                      embed_model=ctx.embedder.model_id)
    return Upserter(ctx, ws)


def run(ctx, claims=(), *, days=0.0, llm_setup=None):
    c = next_ctx(ctx, days=days)
    c.llm.on("core_classify", classify_all(lambda t: "reference"))
    if llm_setup:
        llm_setup(c.llm)
    res = rem.run_rem(c, nres(claims), pre(c))
    P.mark_inbox(c, res.plan)
    return c, res


def inbox(ctx, op, **kw):
    fields = dict(ts=kw.pop("ts", ctx.now - 3600), session_id=kw.pop("session_id", "sess-x"),
                  platform=kw.pop("platform", "telegram"), op=op, status="pending")
    if "meta" in kw:
        fields["meta_json"] = json.dumps(kw.pop("meta"), ensure_ascii=False)
    fields.update(kw)
    return live_insert(ctx, "inbox", **fields)


# ── F-13 / F-24: a duplicate never lands in a superseded/expired row ───────────

def _a_then_b(ctx):
    a = row(ctx, "Orion 검수 당번은 7조다.", kind="state", subject="Orion 검수 당번",
            event_time=ctx.now - 10 * DAY, now=ctx.now - 10 * DAY, valid_until=ctx.now + 30 * DAY,
            user_evidence_count=1, user_session_count=1, source_message_ids=["s:1"])
    b = row(ctx, "Orion 검수 당번은 9조다.", kind="state", subject="Orion 검수 당번",
            event_time=ctx.now - 5 * DAY, now=ctx.now - 5 * DAY, valid_until=ctx.now + 30 * DAY,
            user_evidence_count=1, user_session_count=1, source_message_ids=["s:2"], supersedes=[a.id],
            vector=vec_like(a.vector, 0.5, "b"))
    a.status, a.superseded_by = "superseded", b.id
    commit(ctx, a, b)
    return a, b


def test_f13_state_back_to_old_value_supersedes_current(ctx, deps):
    a, b = _a_then_b(ctx)
    u = fresh(ctx)
    c = claim(ctx, "Orion 검수 당번은 7조이다.", kind="state", subject="Orion 검수 당번",
              vector=vec_like(a.vector, 0.99, "aba"), et=ctx.now - 3600, valid_until=ctx.now + 14 * DAY)
    out = u.upsert(c)
    assert out.action == "state_change" and ctx.llm.calls_of("judge") == []       # auto-dup, no LLM
    new = u.ws.get(out.memory_id)
    assert new.status == "active" and new.text == c.text and new.supersedes == [b.id]
    assert u.ws.get(b.id).status == "superseded" and u.ws.get(b.id).superseded_by == new.id
    assert u.ws.get(a.id).status == "superseded"
    live = [r for r in u.ws.rows.values() if r.status == "active"]
    assert [r.text for r in live] == [c.text]


def test_f13_judged_duplicate_of_superseded_plus_state_change(ctx, deps):
    a, b = _a_then_b(ctx)
    u = fresh(ctx)
    c = claim(ctx, "Orion 검수 당번은 다시 7조가 맡는다.", kind="state", subject="Orion 검수 당번",
              vector=vec_like(a.vector, 0.90, "aba2"), et=ctx.now - 3600)
    ctx.llm.queue("judge", judge_json(("c1", "duplicate", "new"), ("c2", "state_change", "new")))
    out = u.upsert(c)
    assert out.action == "state_change"
    assert u.ws.get(b.id).status == "superseded" and u.ws.get(out.memory_id).status == "active"


def test_f13_backlog_claim_still_absorbed_into_history(ctx, deps):
    a, b = _a_then_b(ctx)
    u = fresh(ctx)
    c = claim(ctx, "Orion 검수 당번은 7조이다.", kind="state", subject="Orion 검수 당번",
              vector=vec_like(a.vector, 0.99, "old"), et=ctx.now - 7 * DAY)      # older than b
    out = u.upsert(c)
    assert out.action == "duplicate" and out.memory_id == a.id and not u.ws.new_ids
    assert u.ws.get(b.id).status == "active"


def test_f13_chain_ending_inactive_gets_a_new_row(ctx, deps):
    a, b = _a_then_b(ctx)
    b.status = "expired"
    commit(ctx, b)
    u = fresh(ctx)
    c = claim(ctx, "Orion 검수 당번은 7조이다.", kind="state", subject="Orion 검수 당번",
              vector=vec_like(a.vector, 0.99, "end"), et=ctx.now - 3600, valid_until=ctx.now + 14 * DAY)
    out = u.upsert(c)
    assert out.memory_id in u.ws.new_ids and u.ws.get(out.memory_id).status == "active"
    assert u.ws.get(a.id).status == "superseded" and u.ws.get(b.id).status == "expired"


def test_f24_rejudge_keeps_the_live_row(ctx, deps):
    x = row(ctx, "사용자의 주차 자리는 12번이다.", kind="state", status="expired",
            event_time=ctx.now - 20 * DAY, valid_until=ctx.now - 3 * DAY, now=ctx.now - 20 * DAY)
    y = row(ctx, "사용자의 주차 자리는 12번으로 정해져 있다.", kind="state", judge_pending=True,
            vector=vec_like(x.vector, 0.9, "y"), event_time=ctx.now - DAY, valid_until=ctx.now + 10 * DAY,
            now=ctx.now - DAY)
    commit(ctx, x, y)
    u = fresh(ctx)
    ctx.llm.queue("judge", judge_json(("c1", "duplicate", "new")))
    u.rejudge_pending(limit=10)
    assert u.ws.get(y.id).status == "active" and not u.ws.get(y.id).judge_pending
    assert u.ws.get(x.id).status == "superseded" and u.ws.get(x.id).superseded_by == y.id


# ── F-14 / F-17: explicit restatement repairs U2 divergence; pin follows ──────

def test_f14_explicit_restatement_supersedes_linked_pinned_rule(ctx, deps):
    p = row(ctx, "지출 원장은 엑셀이다.", kind="rule", subject="지출 원장", pinned=True,
            source="core:user", now=ctx.now - 30 * DAY, event_time=ctx.now - 30 * DAY)
    n1 = row(ctx, "지출 원장은 공유 문서함이다.", kind="rule", subject="지출 원장",
             user_evidence_count=1, user_session_count=1, source_message_ids=["s:7"],
             source_session_ids=["S1"], related_ids=[p.id], now=ctx.now - 2 * DAY, event_time=ctx.now - 2 * DAY,
             vector=vec_like(p.vector, 0.5, "n1"))
    p.related_ids = [n1.id]
    commit(ctx, p, n1)
    u = fresh(ctx)
    c = claim(ctx, "앞으로 지출 원장은 공유 문서함이다.", kind="rule", subject="지출 원장", explicit=True,
              vector=vec_like(n1.vector, 0.90, "n2"), sessions=["S2"], et=ctx.now - 3600)
    ctx.llm.queue("judge", judge_json(("c1", "duplicate", "same"), ("c2", "state_change", "new")))
    out = u.upsert(c)
    assert out.action == "duplicate" and out.memory_id == n1.id
    old, cur = u.ws.get(p.id), u.ws.get(n1.id)
    assert old.status == "superseded" and old.superseded_by == n1.id and not old.pinned
    assert cur.status == "active" and cur.pinned and cur.explicit_user and p.id in cur.supersedes
    P.apply_guard(u.ws, ctx.cfg, mode="live")
    assert u.ws.guard.held_seqs == []                        # explicit user evidence: nothing held


def test_f17_explicit_state_change_moves_the_pin(ctx, deps):
    p = row(ctx, "지출 원장은 엑셀이다.", kind="rule", subject="지출 원장", pinned=True,
            now=ctx.now - 30 * DAY, event_time=ctx.now - 30 * DAY, user_evidence_count=1,
            user_session_count=1, explicit_user=True)
    commit(ctx, p)
    u = fresh(ctx)
    c = claim(ctx, "앞으로 지출 원장은 공유 문서함이다.", kind="rule", subject="지출 원장", explicit=True,
              vector=vec_like(p.vector, 0.85, "x"), et=ctx.now - 3600)
    ctx.llm.queue("judge", judge_json(("c1", "state_change", "new")))
    out = u.upsert(c)
    assert out.action == "state_change"
    new, old = u.ws.get(out.memory_id), u.ws.get(p.id)
    assert new.pinned and new.tier == "pinned" and not old.pinned and old.status == "superseded"
    assert [o.op for o in u.ws.ops if o.memory_id in (new.id, old.id) and o.op in ("pin", "unpin")] == ["pin", "unpin"]


# ── F-16: suppression keeps the numeric/negation guard ────────────────────────

def test_f16_suppress_blocks_same_numbers_only(ctx, deps):
    gone = "사용자의 주차 자리는 12번이다."
    v = ctx.embedder.vector("앵커 " + gone)
    ctx.store.commit(suppress=[SuppressRow(id="sup12", vector=v, text_sha=text_sha(gone), kind="state",
                                           created_at=ctx.now, reason=suppress_reason("r0", gone))])
    u = fresh(ctx)
    c13 = claim(ctx, "사용자의 주차 자리는 13번이다.", kind="state", vector=vec_like(v, 0.96, "a"))
    assert u.upsert(c13).action != "suppressed"
    c12 = claim(ctx, "사용자의 주차 자리는 12번으로 정해져 있다.", kind="state", vector=vec_like(v, 0.96, "b"))
    assert u.upsert(c12).action == "suppressed"
    # a suppress row written before the guard existed still blocks on cosine alone
    ctx.store.commit(suppress=[SuppressRow(id="legacy", vector=ctx.embedder.vector("옛 억제"), text_sha="x",
                                           kind="fact", created_at=ctx.now, reason="forget|run:old")])
    old = claim(ctx, "아무 숫자 1234가 있는 다른 문장이다.", vector=vec_like(ctx.embedder.vector("옛 억제"), 0.95, "c"))
    assert fresh(ctx).upsert(old).action == "suppressed"


# ── F-19: a state re-confirmed on another day is a duplicate ──────────────────

def test_f19_state_reconfirmation_extends_validity(ctx, deps):
    d28 = clock.parse_iso("2026-09-28T12:00")
    e = row(ctx, "2026-09-28 기준 사용자의 주차 자리는 12번이다.", kind="state",
            subject="주차 자리 배정", event_time=d28, now=d28, valid_until=d28 + 14 * DAY,
            user_evidence_count=1, user_session_count=1, source_message_ids=["s:1"], source_session_ids=["S1"])
    commit(ctx, e)
    u = fresh(ctx)
    d30 = clock.parse_iso("2026-09-30T12:00")
    c = claim(ctx, "2026-09-30 기준 사용자의 주차 자리는 12번이다.", kind="state",
              subject="주차 자리 배정", et=d30, valid_until=d30 + 14 * DAY, vector=vec_like(e.vector, 0.97, "r"))
    out = u.upsert(c)
    assert out.action == "duplicate" and out.memory_id == e.id and ctx.llm.calls_of("judge") == []
    assert u.ws.get(e.id).valid_until == pytest.approx(d30 + 14 * DAY)
    # a different value is never auto-dup, even with the same as-of date wording
    c2 = claim(ctx, "2026-09-30 기준 사용자의 주차 자리는 13번이다.", kind="state",
               subject="주차 자리 배정", et=d30, vector=vec_like(e.vector, 0.97, "s"))
    assert not vu.auto_dup_eligible(c2.text, e.text, "state", "state", new_ref_date="2026-09-30",
                                    old_ref_date="2026-09-28")
    # an expiry date that is the state's value stays a value
    assert not vu.auto_dup_eligible("전세 만료일은 2027-06-30이다.", "전세 만료일은 2026-12-31이다.", "state", "state",
                                    new_ref_date="2026-10-02", old_ref_date="2026-10-01")


# ── F-20 / F-28: no merge into a dormant row without user evidence ────────────

def test_f20_dormant_target_not_consolidated_without_user_evidence(ctx, deps):
    d = row(ctx, "관리 서버 계정 정보는 운영 문서에 있다.", kind="reference", status="dormant",
            now=ctx.now - 400 * DAY)
    commit(ctx, d)
    u = fresh(ctx)
    c = claim(ctx, "관리 서버 관리자 계정 이름은 yumeadmin이다.", kind="reference", user=False,
              vector=vec_like(d.vector, 0.85, "da"))
    ctx.llm.queue("judge", judge_json(("c1", "different_aspects", "new")))
    out = u.upsert(c)
    assert out.action == "related" and ctx.llm.calls_of("consolidate") == []
    assert u.ws.get(out.memory_id).status == "active" and u.ws.get(d.id).text == d.text


# ── F-29: protection counts user-evidence sessions only ───────────────────────

def test_f29_assistant_session_is_not_a_second_user_session(ctx, deps):
    e = row(ctx, "배포 절차는 빌드 후 스모크 테스트를 돌리고 태그를 단다.", kind="procedure", subject="배포 절차",
            user_evidence_count=1, user_session_count=1, source_message_ids=["s:1"], source_session_ids=["S1"])
    commit(ctx, e)
    u = fresh(ctx)
    c = claim(ctx, "배포 절차는 빌드 후 스모크 테스트를 돌리고 태그를 붙인다.", kind="procedure", subject="배포 절차",
              roles=["user", "assistant"], sessions=["S1", "S2"], keys=["s:2", "s:3"],
              vector=vec_like(e.vector, 0.97, "p"))
    c.user_session_ids = ["S1"]
    u.upsert(c)
    r = u.ws.get(e.id)
    assert r.user_session_count == 1 and r.tier == "slow" and "S2" not in r.source_session_ids
    later = claim(ctx, "배포 절차는 빌드 후 스모크 테스트를 돌리고 태그를 단다 확인.", kind="procedure",
                  subject="배포 절차", sessions=["S2"], keys=["s:9"], vector=vec_like(e.vector, 0.97, "q"))
    later.user_session_ids = ["S2"]
    u.upsert(later)
    r = u.ws.get(e.id)
    assert r.user_session_count == 2 and r.tier == "durable"


# ── F-21: a new core copy absorbs the conversation row stating the same fact ──

def test_f21_core_copy_retires_duplicate_conversation_row(ctx, deps):
    core = row(ctx, "**호칭:** 사장님", kind="profile", subject="호칭", source="core:user", in_core=True,
               core_target="user", core_sha="abc", user_evidence_count=1, user_session_count=1)
    x = row(ctx, "사용자의 호칭은 사장님이다.", kind="profile", subject="사용자 호칭",
            vector=vec_like(core.vector, 0.88, "x"), user_evidence_count=1, user_session_count=1,
            source_message_ids=["s:5"], source_session_ids=["S5"])
    other = row(ctx, "사용자의 직업은 프리랜서 디자이너다.", kind="profile", subject="직업",
                vector=vec_like(core.vector, 0.75, "o"))
    commit(ctx, x, other)
    u = fresh(ctx)
    u.ws.insert(core, reason="core_add", user_evidence=True)
    ctx.llm.queue("judge", judge_json(("c1", "duplicate", "same"), ("c2", "unrelated", "same")))
    outs = u.absorb_into_core_copy(core.id)
    assert [o.memory_id for o in outs] == [core.id]
    assert u.ws.get(x.id).status == "superseded" and u.ws.get(x.id).superseded_by == core.id
    assert u.ws.get(other.id).status == "active"
    assert "s:5" in u.ws.get(core.id).source_message_ids


# ── F-15: core copies do not decay while in core; demotion restarts the clock ─

def test_f15_in_core_rows_do_not_decay_and_demotion_revives(ctx, deps):
    old = ctx.now - 400 * DAY
    r = row(ctx, "Orion 프로젝트 목표는 데모 완성이다.", kind="project", source="core:memory", in_core=True,
            core_target="memory", core_sha="s1", now=old, importance=0.85)
    assert strength.compute_tier(r) == "decaying"
    assert strength.transition(r, ctx.now, ctx.cfg) is None             # in core: in use
    r.in_core = False
    assert strength.transition(r, ctx.now, ctx.cfg).to_status == "dormant"
    r.in_core, r.status = True, "dormant"                                # went dormant before the fix
    commit(ctx, r)
    ws = P.WorkingSet(ctx.store.load_working_set(), run_id=ctx.run_id, now=ctx.now, embed_model=ctx.embedder.model_id)
    core_check.demote_core_row(ctx, ws, ws.get(r.id), at=ctx.now, reason="core_remove")
    d = ws.get(r.id)
    assert not d.in_core and d.status == "active" and d.last_used_at == pytest.approx(ctx.now)
    assert strength.transition(d, ctx.now + 30 * DAY, ctx.cfg) is None


# ── F-22: the procedure document holds the claim's own messages ───────────────

def test_f22_doc_body_is_the_evidence_span(ctx, deps):
    texts = ["오늘 점심은 김치찌개 먹었어", "장애 보고는 알림 확인 → 로그 수집 → 원인 기록 → 회고 공유 순서야",
             "알겠습니다. 그 순서로 하겠습니다.", "그리고 내일 날씨 어때?"]
    roles = ["user", "user", "assistant", "user"]
    msgs = [Message(ref=f"X#{i}", key=f"s:{i}", role=ro, text=t, ts=ctx.now, source="statedb", session_id="s",
                    msg_id=i) for i, (t, ro) in enumerate(zip(texts, roles), start=1)]
    w = Window(window_id="w", source="statedb", root="r", first_id=1, last_id=4, start_ts=ctx.now, last_ts=ctx.now,
               platform="cli", title="", header="", text="", messages=msgs)
    c = claim(ctx, "장애 보고는 알림 확인→로그 수집→원인 기록→회고 공유 순서다.", kind="procedure", subject="장애 보고",
              steps=4, window_id="w", keys=["s:2"])
    docs = docs_writer.plan_docs(ctx, [(c, "m1")], {"w": w})
    assert len(docs) == 1
    assert "김치찌개" not in docs[0].body and "날씨" not in docs[0].body
    assert "회고 공유" in docs[0].body and "그 순서로" in docs[0].body
    five = claim(ctx, "장애 보고는 다섯 단계 절차로 진행한다 요약.", kind="procedure", subject="장애 보고 2", steps=5,
                 window_id="w", keys=["s:2"])
    assert docs_writer.plan_docs(ctx, [(five, "m2")], {"w": w}) == []      # the source shows 4 steps


# ── F-1: a sandbox never writes docs into the live workspace ──────────────────

def test_f1_docs_refused_for_unset_or_live_workspace(ctx, deps, monkeypatch, tmp_path):
    msgs = [Message(ref="U#1", key="s:1", role="user", text="절차: 하나 → 둘 → 셋 → 넷 순서로 진행한다 길게 설명",
                    ts=ctx.now, source="statedb", session_id="s", msg_id=1)]
    w = Window(window_id="w", source="statedb", root="r", first_id=1, last_id=1, start_ts=ctx.now, last_ts=ctx.now,
               platform="cli", title="", header="", text="", messages=msgs)
    c = claim(ctx, "절차는 하나→둘→셋 순서다.", kind="procedure", subject="절차", steps=3, window_id="w", keys=["s:1"])
    assert len(docs_writer.plan_docs(ctx, [(c, "m")], {"w": w})) == 1
    unset = next_ctx(ctx)
    unset = type(unset)(**{**unset.__dict__, "cfg": ctx.cfg.replace(workspace_dir="")})
    assert docs_writer.plan_docs(unset, [(c, "m")], {"w": w}) == []
    assert any("workspace_dir 미설정" in n for n in unset.report.notes)
    monkeypatch.setenv("HERMESYUME_LIVE_WORKSPACES", str(ctx.cfg.workspace_dir))   # pretend it is live
    assert docs_writer.plan_docs(ctx, [(c, "m")], {"w": w}) == []
    d = P.DocWrite(slug="x", path=str(Path(ctx.cfg.workspace_dir) / "docs/yume/x.md"), title="t", body="b")
    res = docs_writer.write_docs([d], dry_run=False, hermes_home=ctx.paths.hermes_home)
    assert res[0][1] is False and "live workspace" in res[0][2]
    assert not (Path(ctx.cfg.workspace_dir) / "docs/yume/x.md").exists()


def test_f1_yume_init_on_sandbox_leaves_live_paths_empty(tmp_path, capsys):
    from hermesyume import cli
    home = tmp_path / "sbhome"
    home.mkdir()
    assert cli.main(["--hermes-home", str(home), "init"]) == 0
    capsys.readouterr()
    assert cli.main(["--hermes-home", str(home), "config", "get", "workspace_dir"]) == 0
    assert json.loads(capsys.readouterr().out) == ""
    assert cli.main(["--hermes-home", str(home), "config", "get", "md_sources"]) == 0
    assert json.loads(capsys.readouterr().out) == []


# ── F-2: backup before cleanup(0); weekly tar ─────────────────────────────────

def test_f2_quarantine_purge_snapshots_lance_first(ctx, deps):
    bad = row(ctx, "배포 토큰은 ghp_" + "B" * 36 + " 이다.")
    ok = row(ctx, "평범한 사실은 그대로 남는다 여기.")
    commit(ctx, bad, ok)
    v_before = ctx.store.version()
    c, _ = run(ctx)
    snaps = sorted(ctx.paths.backups_dir.glob("lancedb-prepurge-*.tar.gz"))
    assert [s.name for s in snaps] == [f"lancedb-prepurge-{c.run_id}.tar.gz"]
    assert oct(snaps[0].stat().st_mode & 0o777) == "0o600"
    with tarfile.open(snaps[0]) as tf:
        assert any(n.startswith("lancedb/memories.lance") for n in tf.getnames())
    assert any(a.op == "backup" and "lancedb-prepurge" in a.detail for a in ctx.ledger.audits())
    assert v_before not in ctx.store.list_versions()            # time travel wiped after the snapshot
    assert ctx.store.list_versions("suppress")                    # suppress kept its own cleanup


def test_f2_weekly_tar_once_a_week(ctx, deps):
    from hermesyume import backups
    t0 = time.time()
    a = backups.weekly_tar(ctx.paths, ctx.cfg, now=t0)
    assert a is not None and a.exists()
    assert backups.weekly_tar(ctx.paths, ctx.cfg, now=t0 + 3600) is None
    with tarfile.open(a) as tf:
        names = tf.getnames()
    assert "hermesyume/ledger.db" in names and "hermesyume/live.db" in names
    assert any(n.startswith("hermesyume/lancedb") for n in names)
    assert not any(n.startswith(("hermesyume/serving", "hermesyume/backups")) for n in names)
    os.utime(a, (t0 - 8 * DAY, t0 - 8 * DAY))
    assert backups.weekly_tar(ctx.paths, ctx.cfg, now=t0) is not None


# ── F-5: purged text is scrubbed outside Lance; report redacted ───────────────

def test_f5_purge_scrubs_plan_json_dream_log_and_inbox(ctx, deps):
    from hermesyume import dream_log
    text = "테스트 암호어는 보라색 고래 7341이다."
    i1 = inbox(ctx, "remember", text=text, kind="fact")
    c1, r1 = run(ctx)
    mid = [r for r in ctx.store.load_working_set().values() if r.text == text][0].id
    plan1 = ctx.paths.plan_json(c1.run_id)
    assert text in plan1.read_text(encoding="utf-8")
    log = ctx.paths.dream_log_dir
    log.mkdir(parents=True, exist_ok=True)
    (log / "old.md").write_text(f"- {dream_log.t(text)} `{mid}`\n", encoding="utf-8")
    inbox(ctx, "forget", memory_id=mid, meta={"reason": "보라색 고래 지워줘", "confirm": False})
    run(ctx, days=1)
    c3, r3 = run(ctx, days=32)
    assert r3.plan.purge_ids == [mid]
    rem.post_commit(c3, r3.plan)
    assert text not in plan1.read_text(encoding="utf-8")
    assert retention.MARK in plan1.read_text(encoding="utf-8")
    assert "7341" not in (log / "old.md").read_text(encoding="utf-8")
    got = ctx.live.conn.execute("SELECT text, vec FROM inbox WHERE id=?", (i1,)).fetchone()
    assert got[0] is None and got[1] is None
    reason = ctx.live.conn.execute("SELECT meta_json FROM inbox WHERE op='forget'").fetchone()[0]
    assert reason is None


def test_f5_report_and_core_seen_never_keep_a_secret(ctx, deps):
    from hermesyume.types import RunReport
    secret = "ghp_" + "C" * 36
    rep = RunReport(rejections=[{"text": f"토큰은 {secret} 이다", "reason": "secret"}],
                    core_changes=[{"target": "user", "change": "remove", "text": f"**토큰:** {secret}"}])
    pl = P.Plan(run_id="r", mode="live", created_at=0.0, now=0.0, lance_version_before=None, report=rep)
    assert secret not in json.dumps(P.plan_to_json(pl), ensure_ascii=False)
    assert secret not in core_check._seen_text(f"**토큰:** {secret}")


def test_f5_old_runs_pruned_and_old_inbox_deleted(ctx, deps):
    c1, r1 = run(ctx, [claim(ctx, "오래된 실행의 plan.json은 지워진다 확인.", origin="wOld#0")])
    d = ctx.paths.run_dir(c1.run_id)
    old = time.time() - 20 * DAY
    os.utime(d / "plan.json", (old, old))
    i_old = inbox(ctx, "remember", text="오래전에 처리된 기억 요청 문장이다.", ts=time.time() - 40 * DAY)
    ctx.live.conn.execute("UPDATE inbox SET status='consumed', consumed_run='x' WHERE id=?", (i_old,))
    ctx.live.conn.commit()
    end = inbox(ctx, "session_end", ts=time.time() - 40 * DAY, session_id="s-old")
    ctx.live.conn.execute("UPDATE inbox SET status='consumed' WHERE id=?", (end,))
    ctx.live.conn.commit()
    c2, r2 = run(ctx, days=1)
    out = retention.after_commit(c2, r2.plan)
    assert out["runs_pruned"] == 1 and not d.exists()
    ids = {r[0] for r in ctx.live.conn.execute("SELECT id FROM inbox")}
    assert i_old not in ids and end in ids


# ── F-25: inbox consumed only after the serving copy is replaced ──────────────

def test_f25_inbox_stays_pending_until_export(ctx, deps, monkeypatch):
    from hermesyume import export
    i1 = inbox(ctx, "remember", text="Orion 데모 리허설 장소는 3층 회의실이다.", kind="reference")
    c = next_ctx(ctx)
    c.llm.on("core_classify", classify_all(lambda t: "reference"))
    res = rem.run_rem(c, nres(), pre(c))
    seen = []

    def build(cx, **kw):
        seen.append(ctx.live.conn.execute("SELECT status FROM inbox WHERE id=?", (i1,)).fetchone()[0])
        return None
    monkeypatch.setattr(export, "build_serving", build)
    assert ctx.live.conn.execute("SELECT status FROM inbox WHERE id=?", (i1,)).fetchone()[0] == "pending"
    rem.post_commit(c, res.plan)
    assert seen == ["pending"]
    assert ctx.live.conn.execute("SELECT status FROM inbox WHERE id=?", (i1,)).fetchone()[0] == "consumed"


# ── F-30: admin writes wait for a planned run's replay ────────────────────────

def test_f30_admin_commands_refuse_while_a_run_is_planned(initialized, paths, capsys):
    from hermesyume import cli
    from hermesyume.ledger import Ledger
    from hermesyume.types import RunRecord
    with Ledger.from_paths(paths) as led:
        led.insert_run(RunRecord(run_id="crashed-run", started_at=1.0, status="planned", lance_version_before=1))
    rc = cli.main(["--hermes-home", str(paths.hermes_home), "forget", "anything"])
    assert rc == 1 and "yume dream" in capsys.readouterr().err
    rc = cli.main(["--hermes-home", str(paths.hermes_home), "restore", "--run", "crashed-run"])
    assert rc == 1 and "yume dream" in capsys.readouterr().err


# ── F-32: a span's stale failures are dropped once it commits ─────────────────

def test_f32_success_clears_old_failed_attempts(ctx):
    led = ctx.ledger
    with led.transaction():
        led.upsert_window(WindowState("w-old", "md", "/m/a.md", 0, 100, 1.0, "failed", 2, "x", "r1", 0))
        led.upsert_window(WindowState("w-new", "md", "/m/a.md", 0, 180, 2.0, "ok", 1, None, "r2", 1))
    assert led.get_window("w-old") is None and led.get_window("w-new").status == "ok"
    from hermesyume.nrem import _failed_attempts
    assert _failed_attempts(led) == {}


# ── F-33: injected counted once per (memory, session, KST date) across runs ───

def test_f33_injected_dedup_spans_runs(ctx, deps):
    a = row(ctx, "회상 대상 기억 하나가 여기에 있다.")
    commit(ctx, a)
    t = clock.parse_iso("2026-10-01T03:00")
    first = [RecallEvent(1, t, "s1", "telegram", 1, a.id, "injected", 0.6, "vector", None)]
    second = [RecallEvent(2, t + 3 * 3600, "s1", "telegram", 2, a.id, "injected", 0.6, "vector", None)]
    ws = P.WorkingSet(ctx.store.load_working_set(), run_id=ctx.run_id, now=ctx.now, embed_model=ctx.embedder.model_id)
    res = recall_fold.fold_recall_events(ctx, ws, second, snapshot_max_id=2, prior_events=first)
    assert res.injected == 0 and ws.get(a.id).recall_injected_count == 0
    res = recall_fold.fold_recall_events(ctx, ws, second, snapshot_max_id=2, prior_events=[])
    assert res.injected == 1


# ── F-8: a session_end marker settles only messages up to its time ────────────

def test_f8_marker_is_a_flush_point_not_an_end():
    from hermesyume.sources.statedb import _settle

    def m(i, role, ts):
        return Message(ref=f"X#{i}", key=f"s:{i}", role=role, text=f"t{i}", ts=ts, source="statedb",
                       session_id="S", msg_id=i)
    now = 10_000.0
    cutoff = now - 1800
    msgs = [m(1, "user", now - 600), m(2, "assistant", now - 590), m(3, "user", now - 60), m(4, "assistant", now - 50)]
    # marker written at compaction (now - 300); the session went on afterwards
    elig, deferred, full = _settle(msgs, cutoff=cutoff, session_end_ids={"S": now - 300})
    assert [x.msg_id for x in elig] == [1, 2] and deferred == 2 and not full
    # marker during an exchange still growing: the exchange is cut back to its user message
    grow = [m(1, "user", now - 600), m(2, "assistant", now - 590), m(3, "user", now - 100),
            m(4, "assistant", now - 90), m(5, "assistant", now - 10)]
    elig, _, _ = _settle(grow, cutoff=cutoff, session_end_ids={"S": now - 50})
    assert [x.msg_id for x in elig] == [1, 2]
    # a real end (no message after the marker) settles everything
    elig, deferred, full = _settle(msgs, cutoff=cutoff, session_end_ids={"S": now})
    assert full and deferred == 0


# ── F-23 / F-34: gates ───────────────────────────────────────────────────────

def test_f23_explicit_regex_scoped_to_the_claims_sentence():
    msg = ("Orion 결제 스테이징 서버 포트는 8081이야. 앞으로 Orion 요금 질문은 항상 요금표부터 확인해.")
    assert not gates.explicit_from_user([msg], "Orion 결제 스테이징 서버 포트는 8081이다.")
    assert gates.explicit_from_user([msg], "Orion 요금 질문은 항상 요금표부터 확인한다.")
    assert gates.explicit_from_user(["포트는 8081이야.", "이건 꼭 기억해 둬"], "스테이징 포트는 8081이다.")


@pytest.mark.parametrize("text", ["사용자는 다음달 이사한다고 했다.", "사용자는 올해 목표를 세웠다고 한다.",
                                  "사용자는 모레 출장을 간다고 했다.", "사용자는 어젯밤 늦게 잤다고 했다.",
                                  "사용자는 곧 이사한다고 말했다.", "The user moved last month to another city."])
def test_f34_more_relative_time_words(text):
    assert gates.RELATIVE_TIME_RE.search(text)


def test_f34_rolling_window_and_absolute_dates_pass():
    assert not gates.RELATIVE_TIME_RE.search("로그는 최근 7일치만 보관한다.")
    assert not gates.RELATIVE_TIME_RE.search("곧바로 처리하지 않고 다음 정리 때 반영한다.")


# ── F-26 / F-27: vecutil guards ──────────────────────────────────────────────

def test_f26_preservation_check_whole_numbers():
    ok, miss = vu.preservation_check("창고 재고 수량은 37상자다.", "상자 단가는 12,000원이다.",
                                     "창고 재고 수량은 370상자, 상자 단가는 12,000원이다.")
    assert not ok and miss == ["37"]
    ok, miss = vu.preservation_check("Orion 검수 당번은 7조다.", "지원 담당은 6조다.",
                                     "Orion 검수 당번은 17조이고 지원 담당은 6조다.")
    assert not ok and miss == ["7"]
    assert vu.preservation_check("마감은 10월 10일", "발표 4조", "마감 2026-10-10, 발표는 4조")[0]


@pytest.mark.parametrize("neg,pos", [("창고 짐은 정리 안했다.", "창고 짐은 정리했다."),
                                     ("크론 알림은 안보낸다.", "크론 알림은 보낸다."),
                                     ("가계부 DB에 기록 안함.", "가계부 DB에 기록함."),
                                     ("회의에 안감.", "회의에 감.")])
def test_f27_attached_negation(neg, pos):
    assert vu.has_negation(neg) and not vu.has_negation(pos)
    assert not vu.auto_dup_eligible(neg, pos, "state", "state")


# ── F-3 / F-4 / F-7: restore and reembed ──────────────────────────────────────

def _cli(capsys, home, *args, rc=0):
    from hermesyume import cli
    got = cli.main(["--hermes-home", str(home), *args, "--json"])
    out = capsys.readouterr()
    assert got == rc, out.err
    return json.loads(out.out) if out.out.strip() else None


def test_f3_restore_reprocess_requeues_inbox_and_later_runs(initialized, paths, capsys, monkeypatch):
    from hermesyume.livedb import LiveDB
    home = paths.hermes_home
    monkeypatch.setenv("HERMESYUME_NOW", "2026-10-02T04:40")
    r0 = _cli(capsys, home, "dream", "--offline", "--settle-minutes", "0")
    db = LiveDB.open(paths, mode="rw")
    db.conn.execute("INSERT INTO inbox(ts, session_id, platform, op, text, kind) VALUES(?,?,?,?,?,?)",
                    (clock.parse_iso("2026-10-02T05:00"), "s1", "telegram", "remember",
                     "Orion 리허설 장소는 3층 회의실이다.", "reference"))
    db.conn.commit()
    monkeypatch.setenv("HERMESYUME_NOW", "2026-10-03T04:40")
    r1 = _cli(capsys, home, "dream", "--offline", "--settle-minutes", "0")
    monkeypatch.setenv("HERMESYUME_NOW", "2026-10-04T04:40")
    r2 = _cli(capsys, home, "dream", "--offline", "--settle-minutes", "0")
    hits = _cli(capsys, home, "inspect", "--query", "리허설 장소")
    assert len(hits) == 1
    assert db.conn.execute("SELECT status FROM inbox WHERE op='remember'").fetchone()[0] == "consumed"
    res = _cli(capsys, home, "restore", "--run", r1["run_id"], "--reprocess")
    assert res["later_runs_undone"] == [r2["run_id"]] and res["inbox_requeued"] == 1
    assert _cli(capsys, home, "inspect", "--query", "리허설 장소") == []
    assert db.conn.execute("SELECT status FROM inbox WHERE op='remember'").fetchone()[0] == "pending"
    monkeypatch.setenv("HERMESYUME_NOW", "2026-10-05T04:40")
    _cli(capsys, home, "dream", "--offline", "--settle-minutes", "0")
    assert len(_cli(capsys, home, "inspect", "--query", "리허설 장소")) == 1
    db.close()
    assert r0["status"] == "committed"


def test_f4_restore_refuses_runs_from_before_a_reembed(initialized, paths, capsys, monkeypatch):
    home = paths.hermes_home
    monkeypatch.setenv("HERMESYUME_NOW", "2026-10-02T04:40")
    _cli(capsys, home, "debug", "plant", "--kind", "fact", "--text", "리임베드 전에 심은 기억 하나", "--offline")
    first = _cli(capsys, home, "debug", "plant", "--kind", "fact", "--text", "리임베드 전에 심은 기억 둘",
                 "--offline")
    from hermesyume.ledger import Ledger
    with Ledger.from_paths(paths, readonly=True) as led:
        run_before = [r.run_id for r in led.runs(limit=10)][0]
    out = _cli(capsys, home, "reembed", "--offline", "--dim", "512")
    assert out["changed"] and out["rows"] == 2
    from hermesyume import cli
    assert cli.main(["--hermes-home", str(home), "restore", "--run", run_before]) == 1
    assert "reembed" in capsys.readouterr().err
    rows = _cli(capsys, home, "inspect", "--query", "리임베드 전에")
    assert len(rows) == 2                                          # nothing was changed
    assert first["ids"]


def test_f4_restore_refuses_a_version_lance_no_longer_has(initialized, paths, capsys, monkeypatch):
    from hermesyume.ledger import Ledger
    home = paths.hermes_home
    _cli(capsys, home, "debug", "plant", "--kind", "fact", "--text", "버전 확인용 기억 문장이다", "--offline")
    with Ledger.from_paths(paths) as led:
        rid = led.runs(limit=1)[0].run_id
        led.update_run(rid, lance_version_before=9999)
    _cli(capsys, home, "restore", "--run", rid, rc=1)


def test_f7_reembed_offline_refused_on_live_home(initialized, paths, capsys, monkeypatch):
    from hermesyume import cli
    monkeypatch.setenv("HERMESYUME_LIVE_HOMES", str(paths.hermes_home))
    rc = cli.main(["--hermes-home", str(paths.hermes_home), "reembed", "--offline", "--dim", "512"])
    assert rc == 1 and "--offline" in capsys.readouterr().err
