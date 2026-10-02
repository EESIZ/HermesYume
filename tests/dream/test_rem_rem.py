"""rem.py — R1 inbox ops, T17 forget (suppress, purge, audit without text), run_rem idempotency
(re-run: LLM 0, Lance versions unchanged), secret re-scan, N8 doc refs, run_dream orchestration
(success / auth failure / dry-run) with stubbed nrem/export/dream_log/alerts."""

import json
import types
from types import SimpleNamespace

import pytest

from hermesyume import plan as P, rem
from hermesyume.llm import LLMAuthError
from hermesyume.types import Message, Window, WindowState
from tests.dream.test_rem_helpers import (classify_all, claim, commit, deps, install_module,  # noqa: F401
                                          live_insert, next_ctx, nres, pre, row, vec_like)
from tests.fakes import judge_json

DAY = 86400.0


def inbox(ctx, op, **kw):
    fields = dict(ts=kw.pop("ts", ctx.now - 3600), session_id=kw.pop("session_id", "sess-x"),
                  platform=kw.pop("platform", "telegram"), op=op, status="pending")
    if "meta" in kw:
        fields["meta_json"] = json.dumps(kw.pop("meta"), ensure_ascii=False)
    fields.update(kw)
    return live_insert(ctx, "inbox", **fields)


def run(ctx, claims=(), *, days=0.0, llm_setup=None, windows=(), window_states=None):
    c = next_ctx(ctx, days=days)
    c.llm.on("core_classify", classify_all(lambda t: "reference"))
    if llm_setup:
        llm_setup(c.llm)
    res = rem.run_rem(c, nres(claims, windows=windows, window_states=window_states), pre(c))
    if not c.dry_run:
        P.mark_inbox(c, res.plan)       # R8 6 runs in post_commit, after the serving export (F-25)
    return c, res


def statuses(ctx):
    return {r["id"]: r["status"] for r in ctx.live.conn.execute("SELECT id, status FROM inbox")}


def by_source(ctx, source):
    return [r for r in ctx.store.load_working_set().values() if r.source == source]


# ── R1 ───────────────────────────────────────────────────────────────────────

def test_r1_remember_pin_and_consume(ctx, deps):
    i1 = inbox(ctx, "remember", text="Orion 결제 스테이징 서버 포트는 8081이다.", kind="reference", pin=1)
    i2 = inbox(ctx, "session_end")
    c, res = run(ctx)
    rows_ = by_source(ctx, "tool:yume_remember")
    assert len(rows_) == 1
    r = rows_[0]
    assert r.pinned and r.tier == "pinned" and r.importance >= 0.80 and r.kind == "reference"
    assert r.origin_keys == [f"inbox:{i1}"] and r.source_message_ids == [f"i:{i1}"]
    assert c.report.new_pins == [{"id": r.id, "text": r.text}] and any("새 pin" in n for n in c.report.notes)
    assert c.alerts == []                                      # U4: new pin is not an alert
    assert statuses(ctx) == {i1: "consumed", i2: "consumed"}
    assert ctx.ledger.get_cursor("inbox") == i2
    # the same fact remembered again → reinforce, still one row
    i3 = inbox(ctx, "remember", text="Orion 결제 스테이징 서버 포트는 8081이다.", ts=ctx.now + 3600)
    c2, _ = run(ctx, days=1)
    assert len(by_source(ctx, "tool:yume_remember")) == 1
    assert c2.stats.reinforced == 1 and statuses(ctx)[i3] == "consumed"


def test_r1_remember_then_forget_same_batch(ctx, deps):
    i1 = inbox(ctx, "remember", text="테스트 암호어는 보라색 고래 7341이다.", kind="fact")
    i2 = inbox(ctx, "forget", memory_id=f"inbox:{i1}", meta={"reason": "보라색 고래 암호 지워", "confirm": False})
    c, _ = run(ctx)
    r = by_source(ctx, "tool:yume_remember")[0]
    assert r.status == "forgotten"
    sup = ctx.store.load_suppress()
    assert [s.id for s in sup] == [r.id]
    audits = ctx.ledger.audits(r.id)
    assert audits and all("7341" not in a.detail and "고래" not in a.detail for a in audits)
    assert c.report.forgotten == [{"id": r.id}]
    assert statuses(ctx) == {i1: "consumed", i2: "consumed"}


def test_r1_forget_pinned_needs_confirm(ctx, deps):
    p = row(ctx, "**호칭:** 사장님이라고 부른다.", kind="profile", pinned=True, source="core:user")
    commit(ctx, p)
    i1 = inbox(ctx, "forget", memory_id=p.id, meta={"reason": "", "confirm": False})
    run(ctx)
    assert ctx.store.get([p.id])[p.id].status == "active" and statuses(ctx)[i1] == "skipped"
    i2 = inbox(ctx, "forget", memory_id=p.id, meta={"reason": "사용자 요청", "confirm": True})
    run(ctx, days=1)
    assert ctx.store.get([p.id])[p.id].status == "forgotten" and statuses(ctx)[i2] == "consumed"


def test_r1_core_add_replace_remove(ctx, deps):
    old = "**알림 채널:** 텔레그램 DM."
    new = "**알림 채널:** 텔레그램 DM과 이메일."
    i1 = inbox(ctx, "core_add", text=old, target="user")
    run(ctx)
    r_old = [r for r in by_source(ctx, "core:user") if r.text == old][0]
    assert r_old.in_core and r_old.core_target == "user" and r_old.tier == "durable"
    r_old.pinned = True
    commit(ctx, r_old)
    i2 = inbox(ctx, "core_replace", text=new, old_text=old, target="user")
    run(ctx, days=1)
    rows_ = {r.text: r for r in by_source(ctx, "core:user")}
    assert rows_[old].status == "superseded" and rows_[old].superseded_by == rows_[new].id
    assert rows_[new].pinned and not rows_[old].pinned            # pin follows the new copy
    assert rows_[new].supersedes == [rows_[old].id]
    rows_[new].core_required = True
    commit(ctx, rows_[new])
    i3 = inbox(ctx, "core_remove", text=new, target="user")
    c3, _ = run(ctx, days=2)
    after = ctx.store.get([rows_[new].id])[rows_[new].id]
    assert after.in_core is False and after.status == "active" and after.pinned
    assert any("빠졌습니다" in n for n in c3.report.notes) and c3.alerts == []
    assert statuses(ctx) == {i1: "consumed", i2: "consumed", i3: "consumed"}


def test_r1_episodic_core_add_waits_for_its_window(ctx, deps):
    i1 = inbox(ctx, "core_add", text="Session: 2026-06-27 대화 요약\n어떤 일이 있었다.", target="memory")
    run(ctx)
    assert statuses(ctx)[i1] == "pending" and by_source(ctx, "core:memory") == []
    w = Window(window_id="w-ep", source="md", root=f"inbox:{i1}", first_id=0, last_id=1, start_ts=ctx.now,
               last_ts=ctx.now, platform="md", title="", header="", text="")
    st = WindowState("w-ep", "md", f"inbox:{i1}", 0, 1, ctx.now, "ok")
    run(ctx, days=1, windows=[w], window_states={"w-ep": st})
    assert statuses(ctx)[i1] == "consumed"


def test_r1_secret_remember_skipped(ctx, deps):
    i1 = inbox(ctx, "remember", text="내 OpenAI 키는 sk-" + "a1B2" * 8 + " 이다.")
    c, _ = run(ctx)
    assert by_source(ctx, "tool:yume_remember") == [] and statuses(ctx)[i1] == "skipped"


# ── T17 forget ───────────────────────────────────────────────────────────────

def test_t17_forget_suppress_purge(ctx, deps):
    x = row(ctx, "테스트 암호어는 보라색 고래 7341이다.", subject="테스트 암호어", now=ctx.now - DAY,
            origin_keys=["wA#0"])
    commit(ctx, x)
    inbox(ctx, "forget", memory_id=x.id, meta={"reason": "사용자 요청", "confirm": False})
    run(ctx)
    assert ctx.store.get([x.id])[x.id].status == "forgotten"
    s = ctx.store.load_suppress()[0]
    assert "text" not in ctx.store.table("suppress").schema.names and s.text_sha
    # the same episode re-processed (new window id) → suppressed, no new row
    again = claim(ctx, x.text, subject="테스트 암호어", origin="wB#0")
    c2, _ = run(ctx, [again], days=1)
    assert c2.stats.suppressed_hits == 1 and c2.stats.created == 0
    # a related-but-different claim does not see the forgotten row as a candidate
    near = claim(ctx, "보라색 고래 이야기는 동화책에 나온다.", subject="동화책", vector=vec_like(x.vector, 0.8, "near"))
    c3, _ = run(ctx, [near], days=2)
    assert c3.llm.calls_of("judge") == [] and c3.stats.created == 1
    # +31 days → purge (row deleted, audit without text)
    c4, _ = run(ctx, days=31)
    assert x.id not in ctx.store.load_working_set()
    assert c4.report.purged == [{"id": x.id}]
    for a in ctx.ledger.audits(x.id):
        assert "7341" not in a.detail and "고래" not in a.detail
    assert [a.op for a in ctx.ledger.audits(x.id)] == ["forget", "purge"]
    assert ctx.store.load_suppress()[0].id == x.id            # suppress stays forever


# ── idempotency / transitions / secrets / docs ───────────────────────────────

def test_rerun_unchanged_is_noop_llm0_versions_unchanged(ctx, deps):
    e = row(ctx, "주간 보고는 금요일 오후에 보낸다.", now=ctx.now - 3 * DAY)
    commit(ctx, e)
    claims = [claim(ctx, "주간 보고는 금요일 오후 5시까지 보낸다.", origin="wX#0", vector=vec_like(e.vector, 0.85, "x")),
              claim(ctx, "Orion 데모 마감은 2026-10-10이다.", kind="schedule", origin="wX#1",
                    valid_until=ctx.now + 8 * DAY)]
    inbox(ctx, "remember", text="Orion 요금 확인은 요금표를 먼저 본다.", kind="rule")
    c1, r1 = run(ctx, claims, llm_setup=lambda l: l.queue("judge", judge_json(("c1", "state_change", "new"))))
    assert r1.status == "committed" and c1.stats.created >= 2
    v = ctx.store.versions()
    same = [claim(ctx, c.text, origin=c.origin_key, vector=c.vector) for c in claims]
    c2, r2 = run(ctx, same)
    assert r2.status == "noop" and r2.commit.skipped
    assert c2.llm.usage.content_calls == 0 and c2.stats.created == 0 and c2.stats.reinforced == 0
    assert ctx.store.versions() == v
    assert ctx.ledger.get_run(c2.run_id).lance_version_after == v["memories"]


def test_time_transitions_applied_and_recorded(ctx, deps):
    ev_ = row(ctx, "E2E event 오래된 사건 하나.", kind="event", importance=0.4, now=ctx.now - 70 * DAY)
    st = row(ctx, "Orion 데모 마감은 2026-10-01이다.", kind="schedule", now=ctx.now - 20 * DAY,
             valid_until=ctx.now - 5 * DAY)
    commit(ctx, ev_, st)
    c, _ = run(ctx)
    rows_ = ctx.store.load_working_set()
    assert rows_[ev_.id].status == "dormant" and rows_[st.id].status == "expired"
    assert [d["id"] for d in c.report.dormant] == [ev_.id] and [d["id"] for d in c.report.expired] == [st.id]
    v = ctx.store.versions()
    c2, r2 = run(ctx)                                             # same night again: nothing to do
    assert r2.status == "noop" and ctx.store.versions() == v


def test_secret_rescan_quarantines_and_purges(ctx, deps):
    bad = row(ctx, "배포 토큰은 ghp_" + "A" * 36 + " 이다.")
    ok = row(ctx, "평범한 사실은 그대로 남는다 여기.")
    commit(ctx, bad, ok)
    c, _ = run(ctx)
    rows_ = ctx.store.load_working_set()
    assert bad.id not in rows_ and ok.id in rows_
    assert [a.code for a in c.alerts] == ["secret_found"]
    assert "ghp_" not in c.alerts[0].message
    assert c.stats.quarantined == 1
    assert [a.op for a in ctx.ledger.audits(bad.id)] == ["purge"]


def test_n8_procedure_doc_ref_and_write(ctx, deps, cfg):
    msgs = [Message(ref=f"U#{i}", key=f"s:{i}", role="user", text=t, ts=ctx.now - 3600, source="statedb",
                    session_id="s1", msg_id=i)
            for i, t in enumerate(["장애 보고 절차를 알려줄게.", "1단계: 알림을 확인한다.", "2단계: 로그를 모은다.",
                                   "3단계: 원인을 기록한다.", "4단계: 회고를 공유한다."], start=1)]
    w = Window(window_id="win-doc", source="statedb", root="r1", first_id=1, last_id=5, start_ts=ctx.now - 3600,
               last_ts=ctx.now - 3600, platform="telegram", title="t", header="h", text="", messages=msgs)
    c = claim(ctx, "Orion 장애 보고 절차는 알림 확인→로그 수집→원인 기록→회고 공유 순서다.", kind="procedure",
              subject="Orion 장애 보고", steps=4, window_id="win-doc", origin="win-doc#0",
              keys=["s:1", "s:2", "s:3", "s:4", "s:5"])
    cx, res = run(ctx, [c], windows=[w])
    assert len(res.plan.docs) == 1
    d = res.plan.docs[0]
    r = [x for x in ctx.store.load_working_set().values() if x.kind == "procedure"][0]
    assert f"docs/yume/{d.slug}.md" in r.refs
    from hermesyume import docs_writer
    out = docs_writer.write_docs(res.plan.docs, dry_run=False)
    assert out[0][1] is True
    from pathlib import Path
    text = Path(d.path).read_text(encoding="utf-8")
    assert text.startswith("# Orion 장애 보고") and "회고를 공유한다" in text
    assert str(Path(d.path).parent).endswith("docs/yume")


# ── run_dream ────────────────────────────────────────────────────────────────

def _stubs(monkeypatch, *, preflight=None, run_nrem=None):
    calls = SimpleNamespace(export=0, logs=[], emitted=[], flushed=0)
    nrem = types.ModuleType("hermesyume.nrem")
    nrem.preflight = preflight or (lambda ctx: pre(ctx))
    nrem.run_nrem = run_nrem or (lambda ctx, p: nres())
    export = types.ModuleType("hermesyume.export")

    def build_serving(ctx, **kw):
        calls.export += 1
    export.build_serving = build_serving
    dl = types.ModuleType("hermesyume.dream_log")
    dl.render = lambda ctx, plan, *, status, error=None: f"# Dream Log {status} {error or ''}"
    dl.write = lambda paths, now, text, *, dry: calls.logs.append((text, dry))
    al = types.ModuleType("hermesyume.alerts")

    class Sink:
        def __init__(self, paths, cfg, dry_run=False):
            pass

        def emit_all(self, alerts):
            calls.emitted.extend(a.code for a in alerts)
    al.AlertSink = Sink
    al.evaluate_run_alerts = lambda ctx, plan: []
    al.evaluate_health = lambda ctx, *, now: []

    def flush(paths, cfg, *, now):
        calls.flushed += 1
        return 0
    al.flush_pending = flush
    for name, mod in (("hermesyume.nrem", nrem), ("hermesyume.export", export),
                      ("hermesyume.dream_log", dl), ("hermesyume.alerts", al)):
        install_module(monkeypatch, name, mod)
    return calls


def test_run_dream_success(ctx, monkeypatch, deps):
    c = claim(ctx, "Orion 결제 스테이징 서버 포트는 8081이다.", kind="reference", origin="wD#0")
    calls = _stubs(monkeypatch, run_nrem=lambda cx, p: nres([c]))
    cx = next_ctx(ctx)
    st = rem.run_dream(cx)
    assert st.status == "committed" and st.created == 22            # claim + 21 R5 core mirrors
    assert st.llm_calls == 1                                          # one batched core_classify
    assert any(x["text"] == c.text for x in cx.report.created)
    assert calls.export == 1 and calls.logs and calls.logs[0][1] is False and calls.flushed == 1
    assert ctx.ledger.get_run(cx.run_id).status == "committed"
    assert st.lance_version_after == ctx.store.version()


def test_run_dream_auth_failure_alerts_and_writes_nothing(ctx, monkeypatch, deps):
    v = ctx.store.versions()

    def preflight(cx):
        raise LLMAuthError("LLM 인증 실패 HTTP 401", status=401)
    calls = _stubs(monkeypatch, preflight=preflight)
    cx = next_ctx(ctx)
    from hermesyume.types import RunRecord
    ctx.ledger.insert_run(RunRecord(run_id=cx.run_id, started_at=cx.now, status="failed", error="incomplete"))
    st = rem.run_dream(cx)
    assert st.status == "failed"
    assert [a.code for a in cx.alerts] == ["auth_401"] and calls.emitted == ["auth_401"]
    assert ctx.store.versions() == v
    run_row = ctx.ledger.get_run(cx.run_id)
    assert run_row.status == "failed" and "LLMAuthError" in run_row.error
    assert calls.logs and "failed" in calls.logs[0][0]


def test_run_dream_dry_run_writes_plan_only(ctx, monkeypatch, deps, paths):
    from hermesyume.ledger import Ledger
    from hermesyume.livedb import LiveDB
    inbox(ctx, "remember", text="드라이런에서는 아무것도 쓰지 않는다 확인용.")
    c = claim(ctx, "드라이런 주장은 plan.json에만 남는다 테스트.", origin="wDry#0")
    calls = _stubs(monkeypatch, run_nrem=lambda cx, p: nres([c]))
    v = ctx.store.versions()
    ro = Ledger.from_paths(paths, readonly=True)
    live = LiveDB.open(paths, mode="pure")
    cx = next_ctx(ctx)
    cx = type(cx)(**{**cx.__dict__, "dry_run": True, "mode": "dry", "ledger": ro, "live": live})
    st = rem.run_dream(cx)
    assert st.status == "dry"
    assert ctx.store.versions() == v
    assert paths.plan_json(cx.run_id).exists()
    assert calls.export == 0 and calls.logs[0][1] is True and calls.emitted == [] and calls.flushed == 0
    assert statuses(ctx) == {1: "pending"}
    assert ctx.ledger.get_run(cx.run_id) is None
    ro.close()
    live.close()


def test_run_dream_crash_in_commit_keeps_planned_then_replays(ctx, monkeypatch, deps):
    c = claim(ctx, "커밋 도중 죽어도 다음 실행에서 재생된다 확인.", origin="wCrash#0")

    def preflight(cx):
        P.replay_planned(cx)                 # what nrem.preflight does at N0 (live)
        return pre(cx)
    calls = _stubs(monkeypatch, preflight=preflight, run_nrem=lambda cx, p: nres([c]))
    cx = next_ctx(ctx)
    from hermesyume.types import RunRecord
    ctx.ledger.insert_run(RunRecord(run_id=cx.run_id, started_at=cx.now, status="failed", error="incomplete"))
    real = ctx.store.commit_memories

    def boom(rows):
        raise OSError("disk full (simulated)")
    with monkeypatch.context() as m:
        m.setattr(ctx.store, "commit_memories", boom)
        st = rem.run_dream(cx)
    assert st.status == "failed" and [a.code for a in cx.alerts] == ["run_failed"]
    assert ctx.ledger.get_run(cx.run_id).status == "planned"
    assert not any(r.text == c.text for r in ctx.store.load_working_set().values())
    assert ctx.store.commit_memories == real
    cy = next_ctx(ctx, days=1)
    cy.ledger.insert_run(RunRecord(run_id=cy.run_id, started_at=cy.now, status="failed", error="incomplete"))
    st2 = rem.run_dream(cy)
    assert st2.status == "committed"
    assert ctx.ledger.get_run(cx.run_id).status == "committed"
    rows_ = [r for r in ctx.store.load_working_set().values() if r.text == c.text]
    assert len(rows_) == 1
    assert calls.export >= 2          # replayed run's post_commit + this run's
