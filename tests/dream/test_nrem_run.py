"""nrem.py orchestration: N0 preflight guards, T4 (failure keeps watermark; 3rd → quarantine +
alert), window cap / budget deferral, embedding, watermark and md offset deltas, inbox episodes.

Input modules of other builders (sources.statedb, sources.markdown, sanitize, windows, plan) are
replaced by small stand-ins injected through sys.modules, so these tests pin NREM's own logic.
"""

from __future__ import annotations

import re
import sys
import types
from types import SimpleNamespace

import pytest

from hermesyume import clock, nrem
from hermesyume.embedder import EmbedAuthError, EmbedError
from hermesyume.ledger import Ledger, MetaMismatch
from hermesyume.llm import LLMAuthError
from hermesyume.nrem import PreflightResult, compute_watermarks, preflight, run_nrem
from hermesyume.store import SchemaMismatch
from hermesyume.threat import ThreatScannerUnavailable
from hermesyume.types import (LedgerDelta, LiveSnapshot, MdFileState, Message, Window,
                              WindowState, make_window_id, sha256_hex)

T0 = clock.parse_iso("2026-09-28T14:00")
_LINE_RE = re.compile(r"^\[([UAL]#[^\s\]]+)[^\]]*\] (.*)$")


# ── stand-in modules ─────────────────────────────────────────────────────────

def _render(msgs):
    return "\n".join(f"[{m.ref}] {m.text}" for m in msgs)


def _split(msgs):
    out, cur = [], []
    for m in msgs:
        if m.role in ("user", "agent_log") and cur:
            out.append(cur)
            cur = []
        cur.append(m)
    if cur:
        out.append(cur)
    return out


def _build_windows(*, source, root, platform, title, messages, context_before, cfg, ref_ts,
                   end_offset=None, md=None):
    exs = _split(messages)
    out = []
    for i, ex in enumerate(exs):
        body = _render(ex)
        if source == "statedb":
            first, last = min(m.msg_id for m in ex), max(m.msg_id for m in ex)
            wid = make_window_id("statedb", root, first, last)
        else:
            first = ex[0].msg_id
            last = exs[i + 1][0].msg_id if i + 1 < len(exs) else end_offset
            wid = make_window_id("md", root, first, last, content_sha=sha256_hex(body))
        out.append(Window(window_id=wid, source=source, root=root, first_id=first, last_id=last,
                          start_ts=ex[0].ts, last_ts=ex[-1].ts, platform=platform, title=title,
                          header="h", text=f"h\n\n[추출 대상]\n{body}", messages=ex,
                          session_ids=sorted({m.session_id for m in ex if m.session_id})))
    return out


class _Report:
    def top_repeated(self, n=10):
        return [{"line": "반복 줄", "count": 7}][:n]


@pytest.fixture
def fakes(monkeypatch):
    st = SimpleNamespace(lineages=[], excluded={"source": 2}, session_roots={}, md=[],
                         md_excluded={"md_excluded": 1}, replay_calls=0, md_now=[])

    statedb = types.ModuleType("hermesyume.sources.statedb")
    statedb.load_lineages = lambda conn, *, ledger, cfg, now, settle_minutes, session_end_ids: SimpleNamespace(
        lineages=list(st.lineages), excluded=dict(st.excluded), session_roots=dict(st.session_roots),
        sessions_seen=len(st.lineages), messages_in=sum(len(x.messages) for x in st.lineages))
    statedb.recent_texts = lambda conn, *, cfg, now, days: []

    markdown = types.ModuleType("hermesyume.sources.markdown")

    def scan(cfg, ledger, *, now=None):
        st.md_now.append(now)
        return list(st.md), dict(st.md_excluded)
    markdown.scan_md_sources = scan
    markdown.file_state_after = lambda src, pb, run_id, *, status=None: MdFileState(
        src.path, src.sha256, pb, f"prefix:{pb}", status, run_id)

    sanitize = types.ModuleType("hermesyume.sanitize")
    sanitize.SanitizeReport = _Report
    sanitize.build_repeat_lines = lambda texts, *, min_chars=20, min_msgs=5: set()
    sanitize.sanitize_messages = lambda msgs, *, cfg, repeat_lines, report: [m for m in msgs if m.text.strip()]

    windows = types.ModuleType("hermesyume.windows")
    windows.BODY_HEADING = "[추출 대상]"
    windows.render_messages = _render
    windows.format_header = lambda *, platform, title, start_ts, end_ts, ref_ts, md_date=None: f"세션: {platform} / 기간: {md_date}"
    windows.build_windows = _build_windows

    plan = types.ModuleType("hermesyume.plan")

    def replay(ctx):
        st.replay_calls += 1
        return []
    plan.replay_planned = replay

    for name, mod in (("sources.statedb", statedb), ("sources.markdown", markdown),
                      ("sanitize", sanitize), ("windows", windows), ("plan", plan)):
        monkeypatch.setitem(sys.modules, f"hermesyume.{name}", mod)
    return st


def responder(messages):
    """Extract stand-in: 'FAIL' anywhere → invalid JSON; each line containing 'CLAIM[@date]:a||b'
    yields claims a, b with that line's ref as evidence."""
    body = messages[1]["content"]
    if "FAIL" in body:
        return "죄송합니다, JSON을 만들 수 없습니다"
    claims = []
    for line in body.split("[추출 대상]", 1)[-1].splitlines():
        mm = _LINE_RE.match(line)
        if not mm or "CLAIM" not in mm.group(2):
            continue
        rest = mm.group(2).split("CLAIM", 1)[1]
        date = "2026-09-28"
        if rest.startswith("@"):
            date, rest = rest[1:11], rest[11:]
        for t in rest.lstrip(":").split("||"):
            claims.append({"kind": "fact", "target": "world", "subject": t[:10], "text": t,
                           "event_time": date, "valid_until": None, "level": "3",
                           "evidence": [mm.group(1)], "explicit": "false", "steps": None})
    return {"claims": claims}


def smsg(i: int, role: str, text: str, ts: float, sid: str = "S1") -> Message:
    return Message(ref=f"{'U' if role == 'user' else 'A'}#{i}", key=f"s:{i}", role=role, text=text,
                   ts=ts, source="statedb", session_id=sid, msg_id=i, platform="telegram")


def lineage(root: str, msgs: list[Message]):
    return SimpleNamespace(root=root, platform="telegram", title=f"t-{root}", chat_type="dm",
                           sessions=[], messages=msgs, context_before=[], wm=None, fully_settled=True)


def exchange(i: int, text: str, ts: float, sid: str) -> list[Message]:
    return [smsg(i, "user", text, ts, sid), smsg(i + 1, "assistant", "네, 확인했습니다.", ts + 30, sid)]


@pytest.fixture
def run(ctx, statedb, fakes, scripted_llm):
    """Run NREM with the stand-ins; the state.db file exists (content ignored by the stand-in)."""
    scripted_llm.on("extract", responder).on("extract_retry", responder)

    def _run(pre=None):
        return run_nrem(ctx, pre or PreflightResult(LiveSnapshot(0, 0), 0))
    return _run


def commit(ctx, res, *, wm=True):
    """What R8-5 does with the NREM part of the ledger delta."""
    ctx.ledger.apply_delta(ctx.run_id, LedgerDelta(
        watermarks=res.wm_delta if wm else {}, session_roots=res.session_roots,
        windows=list(res.window_states.values()), md_files=res.md_states), lance_version_after=None)


# ── T4 ───────────────────────────────────────────────────────────────────────

def test_t4_failure_keeps_watermark_then_quarantine(ctx, fakes, run, scripted_llm):
    fakes.lineages = [lineage("S1", exchange(1, "FAIL 이 창의 추출은 계속 실패하도록 만든 메시지", T0, "S1")
                              + exchange(3, "CLAIM:Orion 스테이징 서버 포트는 8081이다.", T0 + 120, "S1"))]
    attempts = []
    for night in (1, 2):
        ctx.run_id = f"run{night}"
        res = run()
        (w1,) = [s for s in res.window_states.values()]
        attempts.append((w1.status, w1.attempts))
        assert res.wm_delta == {} and res.claims == []          # watermark kept
        assert res.deferred_windows == 1                          # later window of the root waits
        assert not [a for a in ctx.alerts if a.code == "window_quarantined"]
        commit(ctx, res)
    assert attempts == [("failed", 1), ("failed", 2)]
    assert len(scripted_llm.calls_of("extract_retry")) == 2       # one retry per attempt

    ctx.run_id = "run3"
    res = run()
    states = {s.first_id: s for s in res.window_states.values()}
    assert states[1].status == "quarantined" and states[1].attempts == 3
    assert states[3].status == "ok" and states[3].n_claims == 1
    alerts = [a for a in ctx.alerts if a.code == "window_quarantined"]
    assert len(alerts) == 1 and alerts[0].details["first_id"] == 1
    assert res.wm_delta == {"S1": (T0 + 150, 4)}                # advances past the quarantined window
    assert [c.text for c in res.claims] == ["Orion 스테이징 서버 포트는 8081이다."]
    commit(ctx, res)
    assert ctx.ledger.get_wm("S1").last_id == 4
    assert {w.status for w in ctx.ledger.windows(root="S1")} >= {"quarantined", "ok"}
    assert ctx.stats.windows_quarantined == 1 and ctx.stats.windows_failed == 2


def test_failed_window_text_never_leaks_to_alert(ctx, fakes, run):
    fakes.lineages = [lineage("S1", exchange(1, "FAIL 비밀스러운 사용자 원문이 들어간 메시지입니다", T0, "S1"))]
    ctx.cfg = ctx.cfg.replace(window_max_attempts=1)
    run()
    (a,) = ctx.alerts
    assert a.code == "window_quarantined" and "비밀스러운" not in a.message + str(a.details)


# ── empty / cap / budget ─────────────────────────────────────────────────────

def test_empty_window_no_llm_and_advances(ctx, fakes, run, scripted_llm):
    fakes.lineages = [lineage("S1", [smsg(1, "user", "ㅋㅋ", T0), smsg(2, "assistant", "ㅎㅎ 네 좋습니다 사장님", T0 + 5)])]
    res = run()
    assert scripted_llm.calls_of("extract") == []
    (st,) = res.window_states.values()
    assert st.status == "empty" and res.wm_delta == {"S1": (T0 + 5, 2)}


def test_window_cap_defers_rest_and_counts_only_llm_windows(ctx, fakes, run, scripted_llm):
    ctx.cfg = ctx.cfg.replace(max_windows_per_run=1)
    fakes.lineages = [
        lineage("A", [smsg(1, "user", "ㅋ", T0, "A"), smsg(2, "assistant", "네", T0 + 1, "A")]   # empty: free
                + exchange(3, "CLAIM:Orion 데모 마감은 2026-10-10이다.", T0 + 10, "A")
                + exchange(5, "CLAIM:Orion 검수 당번은 9조가 맡고 있다.", T0 + 100, "A")),
        lineage("B", exchange(11, "CLAIM:Orion 요금 확인은 요금표부터 한다.", T0 + 50, "B")),
    ]
    res = run()
    assert len(scripted_llm.calls_of("extract")) == 1
    assert res.deferred_windows == 2 and ctx.stats.windows_deferred == 2
    assert res.wm_delta == {"A": (T0 + 40, 4)}
    assert [w.root for w in res.windows] == ["A", "A", "B", "A"]     # (start_ts, root) order kept per root


def test_failure_blocks_only_its_root(ctx, fakes, run):
    fakes.lineages = [
        lineage("A", exchange(1, "FAIL 이 창의 추출은 실패하도록 만든 메시지입니다", T0, "A")
                + exchange(3, "CLAIM:Orion 데모 마감은 2026-10-10이다.", T0 + 100, "A")),
        lineage("B", exchange(11, "CLAIM:Orion 요금 확인은 요금표부터 한다.", T0 + 50, "B")),
    ]
    res = run()
    assert set(res.wm_delta) == {"B"} and res.deferred_windows == 1
    assert [c.text for c in res.claims] == ["Orion 요금 확인은 요금표부터 한다."]
    assert any("추출 실패(1/3)" in n for n in ctx.report.notes)


def test_llm_budget_exceeded_defers(ctx, fakes, run, scripted_llm):
    ctx.budget.max_llm_calls = 1
    fakes.lineages = [lineage("A", exchange(1, "CLAIM:Orion 데모 마감은 2026-10-10이다.", T0, "A")),
                      lineage("B", exchange(11, "CLAIM:Orion 요금 확인은 요금표부터 한다.", T0 + 50, "B"))]
    res = run()
    assert set(res.wm_delta) == {"A"} and res.deferred_windows == 1
    assert not [s for s in res.window_states.values() if s.status == "failed"]
    assert any("예산 소진" in n for n in ctx.report.notes)


def test_embed_budget_defers_trailing_windows(ctx, fakes, run, fake_embedder):
    ctx.budget.max_embed_inputs = 2
    fakes.lineages = [lineage("A", exchange(1, "CLAIM:Orion 데모 마감은 2026-10-10이다.||Orion 검수 당번은 9조가 맡는다.", T0, "A")
                               + exchange(3, "CLAIM:Orion 요금 확인은 요금표부터 한다.", T0 + 60, "A"))]
    res = run()
    assert len(res.claims) == 2 and res.deferred_windows == 1
    assert res.wm_delta == {"A": (T0 + 30, 2)}
    assert fake_embedder.calls == [[f"{c.subject}: {c.text}" for c in res.claims]]


# ── embedding (N6) ───────────────────────────────────────────────────────────

def test_claims_embedded_sorted_and_reported(ctx, fakes, run, fake_embedder):
    fakes.lineages = [lineage("A", exchange(1, "CLAIM@2026-09-27:Orion 데모 마감은 2026-10-10이다.", T0, "A")
                               + exchange(3, "CLAIM@2026-09-20:Orion 검수 당번은 9조가 맡는다.", T0 + 60, "A"))]
    res = run()
    assert [c.event_time for c in res.claims] == sorted(c.event_time for c in res.claims)
    assert res.claims[0].text.startswith("Orion 검수 당번")
    for c in res.claims:
        assert c.vector is not None and c.vector.shape == (1536,)
        assert c.embed_text == f"{c.subject}: {c.text}" and c.subject_key and c.importance > 0
    assert len(fake_embedder.calls) == 1                           # one batched call
    assert [x["text"] for x in ctx.report.claims] == [c.text for c in res.claims]
    assert ctx.report.inputs == [{"source": "statedb", "root": "A", "title": "t-A", "messages": 4}]
    assert ctx.report.strip_lines_top == [{"line": "반복 줄", "count": 7}]
    assert ctx.stats.excluded == {"source": 2, "md_excluded": 1}
    assert ctx.stats.claims_extracted == 2 and ctx.stats.windows_ok == 2


@pytest.mark.parametrize("exc", [EmbedError("down"), EmbedAuthError("401", status=401)])
def test_embed_failure_aborts(ctx, fakes, run, fake_embedder, exc):
    fakes.lineages = [lineage("A", exchange(1, "CLAIM:Orion 데모 마감은 2026-10-10이다.", T0, "A"))]
    fake_embedder.fail_with = exc
    with pytest.raises(type(exc)):
        run()


def test_llm_auth_error_aborts(ctx, fakes, run, scripted_llm):
    fakes.lineages = [lineage("A", exchange(1, "CLAIM:Orion 데모 마감은 2026-10-10이다.", T0, "A"))]
    scripted_llm.queue("extract", LLMAuthError("401", status=401))
    with pytest.raises(LLMAuthError):
        run()


def test_rejections_collected(ctx, fakes, run):
    fakes.lineages = [lineage("A", exchange(1, "CLAIM:사용자는 오늘 회의가 있다고 말했다.||Orion 데모 마감은 2026-10-10이다.", T0, "A"))]
    res = run()
    assert [r.reason for r in res.rejections] == ["relative_time"]
    assert ctx.stats.rejected_by_reason == {"relative_time": 1} and ctx.stats.claims_rejected == 1
    assert ctx.report.rejections[0]["reason"] == "relative_time"


def test_assistant_only_general_knowledge_rejected_in_run(ctx, fakes, run, scripted_llm):
    """DEVIATIONS E2E-7 (U3): the assistant's general-knowledge answer is rejected as assistant_only;
    the agent's own lesson from the same answer is kept; the user's own fact is unaffected."""
    fakes.lineages = [lineage("A", [
        smsg(1, "user", "고양이는 왜 상자를 좋아해? 참고로 우리 고양이 이름은 나비야.", T0, "A"),
        smsg(2, "assistant", "고양이는 좁은 곳에서 안정감을 느끼기 때문입니다. cron은 KST로 등록해야 맞았습니다.",
             T0 + 30, "A")])]

    def extract(messages):
        def c(kind, target, text, ev):
            return {"kind": kind, "target": target, "subject": text[:10], "text": text,
                    "event_time": "2026-09-28", "valid_until": None, "level": "3", "evidence": ev,
                    "explicit": "false", "steps": None}
        return {"claims": [
            c("fact", "world", "고양이는 좁은 곳에서 안정감을 느껴 상자를 좋아한다.", ["A#2"]),
            c("opinion", "user", "고양이가 상자를 좋아하는 것은 자연스러운 습성이라는 견해가 있다.", ["A#2"]),
            c("lesson", "agent", "에이전트는 cron 작업을 KST 기준으로 등록해야 실행 시간이 맞는다.", ["A#2"]),
            c("profile", "user", "사용자의 고양이 이름은 나비다.", ["U#1", "A#2"])]}
    scripted_llm.on("extract", extract)
    res = run()
    assert sorted(r.reason for r in res.rejections) == ["assistant_only", "assistant_only"]
    assert ctx.stats.rejected_by_reason == {"assistant_only": 2}
    assert {r["reason"] for r in ctx.report.rejections} == {"assistant_only"}
    kept = {c.kind: c for c in res.claims}
    assert set(kept) == {"lesson", "profile"}
    assert kept["lesson"].assistant_only and kept["lesson"].target == "agent"
    assert kept["profile"].has_user_evidence and not kept["profile"].assistant_only
    assert kept["lesson"].importance < kept["profile"].importance


# ── watermarks ───────────────────────────────────────────────────────────────

def test_compute_watermarks_longest_prefix():
    def w(i, root="R"):
        return Window(window_id=f"w{i}", source="statedb", root=root, first_id=i, last_id=i + 1,
                      start_ts=float(i), last_ts=float(i) + 0.5, platform="x", title="", header="", text="")

    def s(i, status):
        return WindowState(f"w{i}", "statedb", "R", i, i + 1, float(i), status)
    ws = [w(1), w(3), w(5)]
    assert compute_watermarks(ws, {"w1": s(1, "ok"), "w3": s(3, "failed"), "w5": s(5, "ok")}) == {"R": (1.5, 2)}
    assert compute_watermarks(ws, {"w1": s(1, "empty"), "w3": s(3, "quarantined"), "w5": s(5, "ok")}) == {"R": (5.5, 6)}
    assert compute_watermarks(ws, {"w3": s(3, "ok")}) == {}
    md = Window(window_id="m", source="md", root="/x.md", first_id=0, last_id=9, start_ts=0, last_ts=0,
                platform="md", title="", header="", text="")
    assert compute_watermarks([md], {"m": WindowState("m", "md", "/x.md", 0, 9, 0.0, "ok")}) == {}


def test_tail_sanitized_away_still_advances(ctx, fakes, run):
    fakes.lineages = [lineage("A", exchange(1, "CLAIM:Orion 데모 마감은 2026-10-10이다.", T0, "A")
                               + [smsg(3, "user", "   ", T0 + 90, "A")])]
    res = run()
    assert res.wm_delta == {"A": (T0 + 90, 3)}


def test_watermark_never_regresses(ctx, fakes, run):
    ctx.ledger.set_wm("A", T0 + 9999, 999, "old")
    fakes.lineages = [lineage("A", exchange(1, "CLAIM:Orion 데모 마감은 2026-10-10이다.", T0, "A"))]
    assert run().wm_delta == {}


def test_already_committed_window_not_reextracted(ctx, fakes, run, scripted_llm):
    fakes.lineages = [lineage("A", exchange(1, "CLAIM:Orion 데모 마감은 2026-10-10이다.", T0, "A"))]
    res = run()
    commit(ctx, res, wm=False)                     # e.g. watermark rewound but window rows kept
    n = len(scripted_llm.calls_of("extract"))
    res2 = run()
    assert len(scripted_llm.calls_of("extract")) == n and res2.claims == []
    assert res2.window_states == {} and res2.wm_delta == {"A": (T0 + 30, 2)}
    assert ctx.stats.excluded.get("window_already_done") == 1


def test_session_roots_passed_through(ctx, fakes, run):
    fakes.session_roots = {"child": "A"}
    fakes.lineages = [lineage("A", exchange(1, "CLAIM:Orion 데모 마감은 2026-10-10이다.", T0, "A"))]
    assert run().session_roots == {"child": "A"}


# ── md offsets ───────────────────────────────────────────────────────────────

def _md_src(texts: list[tuple[str, str]], *, start=0):
    path = "/md/2026-09-28-orion.md"
    msgs, off = [], start
    for i, (role, text) in enumerate(texts):
        prefix = {"user": "U", "assistant": "A", "agent_log": "L"}[role]
        msgs.append(Message(ref=f"{prefix}#md:L{i + 1}", key=f"m:abc:{i + 1}", role=role, text=text,
                            ts=clock.parse_iso("2026-09-28T12:00"), source="md",
                            session_id=f"md:{path}", msg_id=off, line=i + 1, platform="md"))
        off += 100
    return SimpleNamespace(path=path, slug="orion", sha256="sha", start_offset=start,
                           end_offset=off, prefix_changed=False, messages=msgs, context_before=[])


def test_md_all_ok_processes_to_end(ctx, fakes, run):
    fakes.md = [_md_src([("user", "CLAIM:Orion 검수 당번은 7조가 맡는다."), ("assistant", "알겠습니다."),
                         ("user", "CLAIM:Orion 장애 보고 절차는 알림 확인부터다.")])]
    res = run()
    assert [(m.processed_bytes, m.status) for m in res.md_states] == [(300, "ok")]
    assert {c.source for c in res.claims} == {"md"}
    assert fakes.md_now == [ctx.now]


def test_md_partial_on_failure_and_none_when_first_fails(ctx, fakes, run):
    fakes.md = [_md_src([("user", "CLAIM:Orion 검수 당번은 7조가 맡는다."),
                         ("user", "FAIL 두 번째 문단은 추출에 실패하도록 만든 메시지")])]
    res = run()
    assert [(m.processed_bytes, m.status) for m in res.md_states] == [(100, "partial")]
    fakes.md = [_md_src([("user", "FAIL 첫 문단은 추출에 실패하도록 만든 메시지입니다")])]
    assert run().md_states == []


# ── inbox episodic core_add ──────────────────────────────────────────────────

def _inbox(ctx, op, text, *, ts=T0):
    cur = ctx.live.conn.execute(
        "INSERT INTO inbox(ts, session_id, platform, op, text, target, status) VALUES(?,?,?,?,?,?, 'pending')",
        (ts, "sess", "telegram", op, text, "memory"))
    ctx.live.conn.commit()
    return cur.lastrowid


def test_inbox_episodic_window(ctx, fakes, run):
    good = _inbox(ctx, "core_add", "Session: 2026-06-27 대화 조각 CLAIM@2026-06-27:택배는 경비실이 아니라 문 앞에 둔다.\nConversation Summary: 끝")
    _inbox(ctx, "core_add", "**호칭:** 사장님")                                  # a fact, not an episode
    _inbox(ctx, "remember", "Session: 2026-06-21 remember는 대상 아님")
    bad = _inbox(ctx, "core_add", "Session: 2026-06-28 FAIL 이 조각은 추출에 실패한다")
    res = run(PreflightResult(ctx.live.snapshot(), 0))
    roots = [w.root for w in res.windows]
    assert roots == [f"inbox:{good}", f"inbox:{bad}"]
    (c,) = res.claims
    assert c.source == "md" and c.evidence_keys == [f"i:{good}"] and c.session_ids == ["inbox"]
    assert c.evidence_roles == ["agent_log"] and c.event_time == clock.parse_iso("2026-06-27")
    assert res.inbox_episodic_ids == [good]                  # failed episode stays pending
    assert res.windows[0].md_date == "2026-06-27"


def test_inbox_after_snapshot_ignored(ctx, fakes, run):
    snap = ctx.live.snapshot()
    _inbox(ctx, "core_add", "Session: 2026-06-27 CLAIM:스냅샷 이후 항목은 다음 밤에 처리한다.")
    res = run(PreflightResult(snap, 0))
    assert res.windows == [] and res.inbox_episodic_ids == []


# ── N0 preflight ─────────────────────────────────────────────────────────────

def _run_row(ctx):
    return ctx.ledger.get_run(ctx.run_id)


def test_preflight_live_happy_path(ctx, fakes, fake_embedder, scripted_llm):
    v0 = ctx.store.version()
    pre = preflight(ctx)
    assert pre.lance_version_before == v0 == ctx.stats.lance_version_before
    assert isinstance(pre.snapshot, LiveSnapshot) and pre.replayed_runs == []
    assert fakes.replay_calls == 1
    assert pre.ledger_backup and list(ctx.paths.backups_dir.glob("ledger-*.db"))
    row = _run_row(ctx)
    assert (row.status, row.error, row.lance_version_before, row.mode) == ("failed", "incomplete", v0, "live")
    assert row.started_at == ctx.now
    assert fake_embedder.usage.by_kind.get("ping") == 1
    assert len(scripted_llm.calls_of("ping")) == 1 and scripted_llm.calls[0]["model"] == ctx.cfg.extract_model


def _no_side_effects(ctx, fakes, v0):
    assert ctx.store.version() == v0
    assert fakes.replay_calls == 0
    assert not list(ctx.paths.backups_dir.glob("ledger-*.db"))
    row = _run_row(ctx)
    assert row is not None and row.status == "failed"         # only the failed run row
    assert ctx.ledger.all_wms() == {} and ctx.ledger.windows() == []


def test_preflight_schema_mismatch_aborts_before_pings(ctx, fakes, fake_embedder, monkeypatch):
    v0 = ctx.store.version()

    def boom():
        raise SchemaMismatch("vector type")
    monkeypatch.setattr(ctx.store, "check_schema", boom)
    with pytest.raises(SchemaMismatch):
        preflight(ctx)
    assert fake_embedder.usage.calls == 0
    _no_side_effects(ctx, fakes, v0)


def test_preflight_meta_mismatch(ctx, fakes, fake_embedder):
    v0 = ctx.store.version()
    ctx.cfg = ctx.cfg.replace(embed_model="text-embedding-3-large")
    with pytest.raises(MetaMismatch):
        preflight(ctx)
    assert fake_embedder.usage.calls == 0
    _no_side_effects(ctx, fakes, v0)


def test_preflight_embed_401_aborts(ctx, fakes, fake_embedder, scripted_llm):
    v0 = ctx.store.version()
    fake_embedder.fail_with = EmbedAuthError("401", status=401)
    with pytest.raises(EmbedAuthError):
        preflight(ctx)
    assert scripted_llm.calls == []
    _no_side_effects(ctx, fakes, v0)


def test_preflight_llm_401_aborts(ctx, fakes, scripted_llm):
    v0 = ctx.store.version()
    scripted_llm.queue("ping", LLMAuthError("401", status=401))
    with pytest.raises(LLMAuthError):
        preflight(ctx)
    _no_side_effects(ctx, fakes, v0)


def test_preflight_scanner_fail_closed(ctx, fakes, monkeypatch):
    ctx.scanner = None

    def unavailable(_dir):
        raise ThreatScannerUnavailable("both missing")
    monkeypatch.setattr(nrem, "load_scanner", unavailable)
    with pytest.raises(ThreatScannerUnavailable):
        preflight(ctx)
    assert [a.code for a in ctx.alerts] == ["scanner_unavailable"]
    assert fakes.replay_calls == 0


def test_preflight_loads_scanner_when_missing(ctx, fakes, scanner, monkeypatch):
    ctx.scanner = None
    monkeypatch.setattr(nrem, "load_scanner", lambda _dir: scanner)
    preflight(ctx)
    assert ctx.scanner is scanner


def test_preflight_dry_run_writes_nothing(ctx, fakes, fake_embedder, paths):
    rw = ctx.ledger
    ctx.ledger = Ledger.from_paths(paths, readonly=True)
    ctx.dry_run, ctx.mode = True, "dry"
    try:
        pre = preflight(ctx)
        assert fakes.replay_calls == 0 and pre.ledger_backup is None
        assert not ctx.paths.backups_dir.exists() or not list(ctx.paths.backups_dir.glob("ledger-*.db"))
        assert rw.get_run(ctx.run_id) is None
        assert fake_embedder.usage.by_kind.get("ping") == 1        # dry-run calls are real
    finally:
        ctx.ledger.close()
        ctx.ledger = rw


def test_preflight_no_live_db(ctx, fakes):
    ctx.live.close()
    live, ctx.live = ctx.live, None
    try:
        assert preflight(ctx).snapshot == LiveSnapshot(0, 0)
    finally:
        ctx.live = live.__class__.open(ctx.paths, mode="rw")


# ── optional end-to-end with the real input modules (other builders) ────────

def test_nrem_with_real_input_modules(ctx, statedb, scripted_llm, fake_home):
    for name in ("sources.statedb", "sources.markdown", "sanitize", "windows"):
        pytest.importorskip(f"hermesyume.{name}")
    seen: list[str] = []

    def extract(messages):
        body = messages[1]["content"]
        seen.append(body)
        out = []
        for line in body.split("[추출 대상]", 1)[-1].splitlines():
            mm = re.match(r"^\[(U#[^\s\]]+)[^\]]*\] (.*)$", line)
            if mm and "Orion" in mm.group(2):
                out.append({"kind": "fact", "target": "world", "subject": "Orion",
                            "text": mm.group(2).split("\n")[0][:200], "event_time": "2026-09-28",
                            "valid_until": None, "level": "3", "evidence": [mm.group(1)],
                            "explicit": "false", "steps": None})
        return {"claims": out}
    scripted_llm.on("extract", extract)
    res = run_nrem(ctx, PreflightResult(ctx.live.snapshot(), 0))
    assert seen, "no window reached the LLM"
    assert all("<memory-context>" not in b and "9999" not in b for b in seen)
    roots = set(res.wm_delta)
    assert "tg1" in roots and "cli_tmp" in roots
    assert not roots & {"cli_synth", "cli_probe", "cron1", "hidden1", "tg_recent"}
    assert any(c.source == "md" for c in res.claims) and any(c.source == "dream" for c in res.claims)
    assert all(c.vector is not None for c in res.claims)
    assert [m.status for m in res.md_states] == ["ok"]
