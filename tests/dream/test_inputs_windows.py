"""windows.py — §3.4 window construction (CONTRACTS §4.5)."""

from __future__ import annotations

from hermesyume import clock
from hermesyume.config import Config
from hermesyume.types import Message, make_window_id, sha256_hex
from hermesyume.windows import (BODY_HEADING, CONTEXT_HEADING, build_windows, format_header,
                                msg_head, render_messages, split_exchanges)

REF = clock.parse_iso("2026-10-02T04:40")          # 정리 기준일 2026-10-02(금)
T0 = clock.parse_iso("2026-09-28T14:02")


def sm(i, role, text, ts=None, sid="s1"):
    return Message(ref=f"{'U' if role == 'user' else 'A'}#{i}", key=f"s:{i}", role=role, text=text,
                   ts=T0 + i * 60 if ts is None else ts, source="statedb", session_id=sid, msg_id=i,
                   platform="telegram")


def mm(line, role, text, off):
    p = {"user": "U", "assistant": "A", "agent_log": "L"}[role]
    return Message(ref=f"{p}#md:L{line}", key=f"m:abc:{line}", role=role, text=text, ts=T0,
                   source="md", session_id="md:/x/2026-09-28-a.md", msg_id=off, line=line, platform="md")


def test_split_exchanges():
    u1, a1, a2, u2 = sm(1, "user", "q"), sm(2, "assistant", "a"), sm(3, "assistant", "b"), sm(4, "user", "q2")
    lead = sm(0, "assistant", "lead")
    assert split_exchanges([lead, u1, a1, a2, u2]) == [[lead], [u1, a1, a2], [u2]]
    l1, l2 = mm(1, "agent_log", "log1", 0), mm(3, "agent_log", "log2", 10)
    mu, ma = mm(5, "user", "q", 20), mm(6, "assistant", "a", 30)
    assert split_exchanges([l1, mu, ma, l2]) == [[l1], [mu, ma], [l2]]
    assert split_exchanges([]) == []


def test_msg_head_and_render():
    assert msg_head(sm(1234, "user", "x")) == f"[U#1234 {clock.fmt_kst(T0 + 1234 * 60, '%m-%d %H:%M')}]"
    m = sm(7, "user", "hi", ts=clock.parse_iso("2026-09-28T14:02"))
    assert msg_head(m) == "[U#7 09-28 14:02]"
    assert msg_head(mm(12, "user", "x", 0)) == "[U#md:L12]"
    inbox = Message(ref="L#inbox:7", key="i:7", role="agent_log", text="t", ts=1.0, source="md")
    assert msg_head(inbox) == "[L#inbox:7]"
    assert render_messages([m, mm(3, "assistant", "답", 5)]) == "[U#7 09-28 14:02] hi\n[A#md:L3] 답"


def test_format_header_variants():
    s, e = clock.parse_iso("2026-09-28T14:02"), clock.parse_iso("2026-09-28T15:10")
    assert format_header(platform="telegram", title="Orion 회의", start_ts=s, end_ts=e, ref_ts=REF) == \
        '세션: telegram "Orion 회의" / 기간: 2026-09-28 14:02–15:10 KST / 정리 기준일: 2026-10-02(금)'
    s2, e2 = clock.parse_iso("2026-09-28T23:50"), clock.parse_iso("2026-09-29T00:10")
    assert format_header(platform="cli", title="", start_ts=s2, end_ts=e2, ref_ts=REF) == \
        '세션: cli "" / 기간: 2026-09-28 23:50–2026-09-29 00:10 KST / 정리 기준일: 2026-10-02(금)'
    assert format_header(platform="md", title="orion", start_ts=s, end_ts=s, ref_ts=REF,
                         md_date="2026-09-28") == \
        '세션: md "orion" / 기간: 2026-09-28 / 정리 기준일: 2026-10-02(금)'
    h = format_header(platform="telegram", title='따옴표 "제목"\n둘째 줄' + "가" * 100, start_ts=s,
                      end_ts=e, ref_ts=REF)
    assert '\n' not in h and h.count('"') == 2 and "…" in h


def test_single_window_layout_and_ids():
    cfg = Config({})
    msgs = [sm(1, "user", "포트는 8081이야"), sm(2, "assistant", "확인")]
    ctx = [sm(0, "user", "이전 질문", ts=T0 - 600), sm(-1, "assistant", "이전 답", ts=T0 - 590)]
    ws = build_windows(source="statedb", root="tg1", platform="telegram", title="회의",
                       messages=msgs, context_before=ctx, cfg=cfg, ref_ts=REF)
    assert len(ws) == 1
    w = ws[0]
    assert w.window_id == make_window_id("statedb", "tg1", 1, 2)
    assert (w.first_id, w.last_id, w.start_ts, w.last_ts) == (1, 2, msgs[0].ts, msgs[1].ts)
    assert w.text == (w.header + "\n\n" + CONTEXT_HEADING + "\n" + render_messages(ctx) + "\n\n"
                      + BODY_HEADING + "\n" + render_messages(msgs))
    assert w.header.startswith('세션: telegram "회의" / 기간: 2026-09-28 ')
    assert w.messages == msgs and w.context == ctx and w.session_ids == ["s1"]
    assert set(w.evidence_index) == {"U#1", "A#2"} and w.user_chars() == len("포트는 8081이야")
    assert w.attempts == 0 and w.md_path is None
    no_ctx = build_windows(source="statedb", root="tg1", platform="telegram", title="회의",
                           messages=msgs, context_before=[], cfg=cfg, ref_ts=REF)[0]
    assert no_ctx.text == no_ctx.header + "\n\n" + BODY_HEADING + "\n" + render_messages(msgs)
    assert CONTEXT_HEADING not in no_ctx.text and no_ctx.window_id == w.window_id


def _exchanges(n, size):
    out = []
    for i in range(n):
        out += [sm(2 * i + 1, "user", f"질문{i} " + "가" * size), sm(2 * i + 2, "assistant", f"답{i}")]
    return out


def test_greedy_packing_whole_exchanges_and_context():
    cfg = Config({"window_chars": 300, "window_context_chars": 1500})
    msgs = _exchanges(6, 100)
    ws = build_windows(source="statedb", root="r", platform="telegram", title="t", messages=msgs,
                       context_before=[], cfg=cfg, ref_ts=REF)
    bodies = [w.messages for w in ws]
    assert [m for b in bodies for m in b] == msgs                 # no loss, no duplicate, in order
    for w in ws:
        assert len(render_messages(w.messages)) <= 300
        assert w.messages[0].role == "user"                       # whole exchanges only
    assert ws[0].context == []
    for prev, w in zip(ws, ws[1:]):
        assert w.context == prev.messages[-2:]                    # previous exchange
        assert w.first_id == w.messages[0].msg_id and w.last_id == w.messages[-1].msg_id


def test_first_exchange_always_taken_and_oversized_exchange_split():
    cfg = Config({"window_chars": 200, "window_context_chars": 1500})
    big = [sm(1, "user", "큰 질문 " + "가" * 50)] + [sm(i, "assistant", f"답{i} " + "나" * 80)
                                                  for i in range(2, 7)]
    small = [sm(7, "user", "작은 질문"), sm(8, "assistant", "작은 답")]
    ws = build_windows(source="statedb", root="r", platform="telegram", title="t",
                       messages=big + small, context_before=[], cfg=cfg, ref_ts=REF)
    assert [m for w in ws for m in w.messages] == big + small
    assert all(len(render_messages(w.messages)) <= 200 for w in ws)
    assert ws[0].messages[0].msg_id == 1
    # a continuation chunk gets the exchange prefix (from its user message) as context
    w1 = ws[1]
    assert w1.messages[0].role == "assistant"
    assert w1.context[0].msg_id == 1 and w1.context == [m for m in big if m.msg_id < w1.messages[0].msg_id]
    # a single message larger than the limit still makes progress
    huge = [sm(1, "user", "가" * 500)]
    w = build_windows(source="statedb", root="r", platform="telegram", title="t", messages=huge,
                      context_before=[], cfg=cfg, ref_ts=REF)
    assert len(w) == 1 and w[0].messages == huge


def test_context_truncated_head_kept():
    cfg = Config({"window_context_chars": 40})
    ctx = [sm(1, "user", "가" * 100), sm(2, "assistant", "답")]
    w = build_windows(source="statedb", root="r", platform="telegram", title="t",
                      messages=[sm(3, "user", "본문 질문")], context_before=ctx, cfg=cfg, ref_ts=REF)[0]
    block = w.text.split(CONTEXT_HEADING + "\n", 1)[1].split("\n\n" + BODY_HEADING, 1)[0]
    assert block.startswith("[U#1 ") and block.endswith(" …") and len(block) <= 42


def test_md_windows_byte_ranges_and_content_sha():
    cfg = Config({"window_chars": 60})
    msgs = [mm(3, "user", "첫 사용자 발화입니다", 40), mm(4, "assistant", "응답", 80),
            mm(6, "agent_log", "[헤더] 일지 문단 하나", 120), mm(9, "user", "둘째 질문", 200)]
    ws = build_windows(source="md", root="/x/2026-09-28-a.md", platform=None, title="a",
                       messages=msgs, context_before=[], cfg=cfg, ref_ts=REF, end_offset=260)
    assert ws[0].first_id == 40
    for a, b in zip(ws, ws[1:]):
        assert a.last_id == b.first_id                     # contiguous byte ranges
    assert ws[-1].last_id == 260
    for w in ws:
        body = render_messages(w.messages)
        assert w.window_id == make_window_id("md", "/x/2026-09-28-a.md", w.first_id, w.last_id,
                                             content_sha=sha256_hex(body))
        assert w.platform == "md" and w.md_path == "/x/2026-09-28-a.md"
        assert w.header.startswith('세션: md "a" / 기간: 2026-09-28 ')
    assert ws[0].session_ids == ["md:/x/2026-09-28-a.md"]


def test_defensive_cleaning_of_unsanitized_input():
    cfg = Config({})
    ctx = [sm(1, "user", "이전 키 sk-" + "x" * 30)]
    msgs = [sm(2, "user", "질문 <memory-context>주입</memory-context> ghp_" + "y" * 36),
            sm(3, "assistant", "<memory-context>only</memory-context>")]
    w = build_windows(source="statedb", root="r", platform="telegram", title="t", messages=msgs,
                      context_before=ctx, cfg=cfg, ref_ts=REF)[0]
    assert "sk-x" not in w.text and "ghp_" not in w.text and "주입" not in w.text
    assert "[REDACTED:openai]" in w.text and "[REDACTED:github]" in w.text
    assert [m.msg_id for m in w.messages] == [2]           # emptied message dropped
    assert w.window_id == make_window_id("statedb", "r", 2, 2)


def test_empty_messages_no_windows():
    assert build_windows(source="statedb", root="r", platform="cli", title="", messages=[],
                         context_before=[], cfg=Config({}), ref_ts=REF) == []


def test_multi_session_lineage_window_session_ids():
    msgs = [sm(1, "user", "a", sid="p"), sm(2, "assistant", "b", sid="p"),
            sm(3, "user", "c", sid="child"), sm(4, "assistant", "d", sid="child")]
    w = build_windows(source="statedb", root="p", platform="telegram", title="t", messages=msgs,
                      context_before=[], cfg=Config({}), ref_ts=REF)[0]
    assert w.session_ids == ["p", "child"]
