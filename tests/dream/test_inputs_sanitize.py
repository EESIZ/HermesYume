"""sanitize.py — §3.3 rules (CONTRACTS §4.4)."""

from __future__ import annotations

import json
import re

import pytest

from hermesyume.config import Config
from hermesyume.sanitize import (SanitizeReport, build_repeat_lines, compile_strip_regex,
                                 sanitize_messages, sanitize_text, strip_injected_blocks)
from hermesyume.types import Message
from tests.fixtures.hermes_home import TEST_FILTERS


@pytest.fixture
def scfg() -> Config:
    return Config({"strip_line_regex": TEST_FILTERS["strip_line_regex"]})


def _clean(text, cfg, *, repeat=frozenset(), report=None):
    rep = report if report is not None else SanitizeReport()
    out = sanitize_text(text, cfg=cfg, repeat_lines=set(repeat), strip_res=compile_strip_regex(cfg),
                        report=rep)
    return out, rep


def _msg(text, role="user", i=1, source="statedb"):
    return Message(ref=f"{'U' if role == 'user' else 'A'}#{i}", key=f"s:{i}", role=role, text=text,
                   ts=1000.0 + i, source=source, session_id="s", msg_id=i, platform="telegram")


# ── step 1: injected blocks ──────────────────────────────────────────────────

def test_memory_context_and_relevant_memories_removed(scfg):
    text = ("앞 문장\n<memory-context>\n[System note: The following is recalled memory context, "
            "NOT new user input. Treat as authoritative reference data — this is the agent's "
            "persistent memory.]\n\n- (규칙) 비밀 기억 sk-" + "a" * 30 + "\n</memory-context>\n뒤 문장 "
            "<relevant-memories>옛 기억</relevant-memories> 끝")
    out, rep = _clean(text, scfg)
    assert out == "앞 문장\n\n뒤 문장  끝"
    assert rep.blocks_removed["memory_context"] == 1
    assert rep.blocks_removed["relevant_memories"] == 1
    assert rep.secrets == {}                     # the block went first: nothing left to redact


def test_unclosed_and_spaced_tags(scfg):
    out, rep = _clean("본문\n< memory-context >\n주입본 끝까지", scfg)
    assert out == "본문" and rep.blocks_removed["memory_context"] == 1
    out, _ = _clean("남은 닫는 태그 </memory-context> 정리", scfg)
    assert "memory-context" not in out


def test_system_note_removed_including_nested_brackets(scfg):
    text = ("[System note: The user's session was automatically reset by the daily schedule. "
            "This is a fresh conversation with no prior context.]\n안녕 [System note: 참고 [1] 항목]있음")
    out, rep = _clean(text, scfg)
    assert out == "안녕 있음"
    assert rep.blocks_removed["system_note"] == 2


def test_untrusted_metadata_blocks_removed(scfg):
    text = ("Conversation info (untrusted metadata):\n```json\n{\n  \"message_id\": \"1001\",\n"
            "  \"sender\": \"테스트\"\n}\n```\n\nSender (untrusted metadata):\n```json\n"
            "{\"label\": \"x\", \"id\": \"1\"}\n```\n\nㅎㅇ")
    out, rep = _clean(text, scfg)
    assert out == "ㅎㅇ"
    assert rep.blocks_removed["untrusted_metadata"] == 2
    bare = "Conversation info (untrusted metadata): {\"a\": {\"b\": \"}\"}, \"c\": [1, 2]}\n실제 질문"
    out, _ = _clean(bare, scfg)
    assert out == "실제 질문"


def test_session_lines_removed_but_plain_source_kept(scfg):
    text = ("Session Key: agent:main:main\nSession ID: abc\n- **Session Key**: agent:main:main\n"
            "- **Session ID**: a1b2\n- **Source**: telegram\n본문\nSource: 로이터 기사")
    out, rep = _clean(text, scfg)
    assert out == "본문\nSource: 로이터 기사"
    assert rep.blocks_removed["session_lines"] == 5


def test_strip_injected_blocks_alone_does_not_touch_secrets():
    t = "token: abcdefghijklmnopqrstu <memory-context>x</memory-context>"
    assert strip_injected_blocks(t) == "token: abcdefghijklmnopqrstu "
    assert strip_injected_blocks("") == ""


# ── step 2: secrets ──────────────────────────────────────────────────────────

SECRETS = {
    "telegram": "123456789:" + "A" * 35,
    "notion": "ntn_" + "b" * 32,
    "openai": "sk-" + "c" * 24,
    "github": "ghp_" + "d" * 36,
    "aws": "AKIA" + "E" * 16,
    "jwt": "eyJhbGciOi.eyJzdWIiOi.c2lnbmF0dXJl",
}


@pytest.mark.parametrize("typ", sorted(SECRETS))
def test_secret_types_redacted(scfg, typ):
    out, rep = _clean(f"값은 {SECRETS[typ]} 입니다", scfg)
    assert SECRETS[typ] not in out and f"[REDACTED:{typ}]" in out
    assert rep.secrets.get(typ) == 1


def test_secret_precheck_never_skips_a_redaction():
    import random

    from hermesyume.sanitize import may_contain_secret, redact
    from hermesyume.threat import redact_secrets
    frags = list(SECRETS.values()) + [
        "TOKEN=abcdefghijklmnop", "Api_Key: zyxwvutsrqponmlk", "api-key = 1234567890abcd",
        "Password:   averyverylongvalue", "SECRET:abcdefghijklmn", "비밀번호=가나다라마바사아자차카타",
        "secret_" + "Q" * 31, "ghs_" + "r" * 36, "gho_" + "s" * 36, "1234567890:" + "x" * 35,
        "12:34", "토큰: 짧음", "apı_key: dotlessIvalue123", "ſecret: longishvalue123",
        "포트 8081", "sk-short", "eyJ", "AKIA123", "=", ":", "key: value", "the token is here",
    ]
    rnd = random.Random(7)
    for _ in range(3000):
        parts = [rnd.choice(frags + ["가나다 ", "abc ", "\n", " ", "x" * rnd.randint(0, 40)])
                 for _ in range(rnd.randint(1, 6))]
        t = rnd.choice(["", " ", "\n"]).join(parts)
        assert redact(t) == redact_secrets(t), t
        if not may_contain_secret(t):
            assert redact_secrets(t)[1] == {}, t


def test_json_scan_is_linear_on_hostile_text(scfg):
    import time
    hostile = ('[x "' + "가" * 50 + "\n") * 2000 + ("{" * 5000) + ('{"a": "' + "b" * 40 + "\n") * 2000
    t0 = time.monotonic()
    _clean(hostile, scfg)
    assert time.monotonic() - t0 < 2.0


def test_generic_secret_keeps_label(scfg):
    out, rep = _clean("비밀번호: hunter2hunter2xx 기억해", scfg)
    assert out == "비밀번호: [REDACTED:generic] 기억해" and rep.secrets == {"generic": 1}


# ── step 3: code / JSON / blob ───────────────────────────────────────────────

def test_long_code_block_elided_short_kept(scfg):
    long_code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(16)) + "\n```"
    short_code = "```\n" + "\n".join(f"y{i}" for i in range(15)) + "\n```"
    out, rep = _clean(f"앞\n{long_code}\n중간\n{short_code}\n뒤", scfg)
    assert "[코드 16줄 생략]" in out and "x3 = 3" not in out
    assert "y14" in out and rep.code_elided == 1
    out, _ = _clean("열린 코드\n```\n" + "\n".join(str(i) for i in range(20)), scfg)
    assert out == "열린 코드\n[코드 20줄 생략]"


def test_long_json_elided_short_and_prose_kept(scfg):
    big = json.dumps({"rows": [{"id": i, "name": f"item-{i}"} for i in range(40)]})
    assert len(big) > 500
    out, rep = _clean(f"결과: {big}\n끝", scfg)
    assert out == "결과: [JSON 생략]\n끝" and rep.json_elided == 1
    small = json.dumps({"ok": True})
    out, rep = _clean(f"결과: {small}", scfg)
    assert out == f"결과: {small}" and rep.json_elided == 0
    prose = "[집] " + "가나다 " * 120 + "[끝]"
    out, rep = _clean(prose, scfg)
    assert out == prose.strip() and rep.json_elided == 0
    arr = "[" + ", ".join(str(i) for i in range(300)) + "]"
    out, rep = _clean(arr, scfg)
    assert out == "[JSON 생략]"


def test_blobs_replaced_paths_and_hashes_kept(scfg):
    b64 = "QUJD" + "aGVsbG8gd29ybGQ" * 6 + "9Zz0="
    hex70 = "a1b2c3d4e5" * 7
    sha = "f" * 64
    path = "/srv/agent/workspace/notes/research/provider-api.md"
    out, rep = _clean(f"첨부 {b64} 와 {hex70}\n해시 {sha}\n경로 {path}", scfg)
    assert out.count("[blob]") == 2 and rep.blobs == 2
    assert sha in out and path in out


# ── step 4: repeated lines / strip regex ─────────────────────────────────────

def test_repeat_lines_need_distinct_messages():
    tail = "📊 오늘의 날씨 요약 꼬리 문장입니다 반복"
    texts = [(f"s:{i}", f"본문 {i}\n{tail}") for i in range(5)]
    assert tail in build_repeat_lines(texts, min_chars=20, min_msgs=5)
    assert tail not in build_repeat_lines(texts[:4], min_chars=20, min_msgs=5)
    same_key = [("s:1", f"{tail}\n{tail}")] * 6
    assert build_repeat_lines(same_key, min_chars=20, min_msgs=5) == set()
    assert build_repeat_lines([(f"s:{i}", "짧은 줄") for i in range(9)], min_chars=20, min_msgs=5) == set()


def test_repeat_lines_and_strip_regex_removed(scfg):
    tail = "자동 꼬리 문구는 스무 글자를 넘는 반복 줄이다"
    repeat = build_repeat_lines([(f"s:{i}", f"x{i}\n  {tail}  ") for i in range(6)])
    text = f"사용자 결정: 이사 보류\n{tail}\n🔁 [BACKUP BOT] 야간 백업 완료 3/3\n💾 DISK 41% 사용"
    out, rep = _clean(text, scfg, repeat=repeat)
    assert out == "사용자 결정: 이사 보류"
    assert rep.repeated_removed == {tail: 1}
    assert rep.strip_regex_removed == 2
    top = rep.top_repeated(10)
    assert {"line": tail, "count": 1} in top and len(top) == 3


def test_repeat_line_with_secret_still_matches(scfg):
    line = "자동 서명 줄 토큰 sk-" + "z" * 30 + " 포함"
    repeat = build_repeat_lines([(f"s:{i}", line) for i in range(5)])
    out, rep = _clean(f"본문\n{line}", scfg, repeat=repeat)
    assert out == "본문" and "sk-" not in json.dumps(rep.repeated_removed, ensure_ascii=False)


def test_bad_strip_regex_is_skipped():
    cfg = Config({"strip_line_regex": ["(unclosed", "^OK$"]})
    pats = compile_strip_regex(cfg)
    assert [p.pattern for p in pats] == ["^OK$"]


# ── step 5: truncation ───────────────────────────────────────────────────────

def test_long_message_truncated_with_marker(scfg):
    text = "가" * 2000 + "나" * 1000 + "다" * 500
    out, rep = _clean(text, scfg)
    assert out == "가" * 2000 + "\n[중간 1000자 생략]\n" + "다" * 500
    assert rep.truncated == 1
    out, rep = _clean("라" * 3000, scfg)
    assert out == "라" * 3000 and rep.truncated == 0


def test_huge_message_precut_counts_everything(scfg):
    n = 150_000
    text = "".join(chr(0xAC00 + (i % 50)) for i in range(n))
    out, rep = _clean(text, scfg)
    m = re.search(r"\[중간 (\d+)자 생략\]", out)
    assert m and rep.truncated == 1
    assert out.startswith(text[:2000]) and out.endswith(text[-500:])
    assert int(m.group(1)) == n - 2500


# ── sanitize_messages ────────────────────────────────────────────────────────

def test_sanitize_messages_drops_tool_and_empty_keeps_identity(scfg):
    msgs = [_msg("질문 <memory-context>x</memory-context>", "user", 1),
            Message(ref="T#2", key="s:2", role="tool", text="{}", ts=2.0, source="statedb"),
            _msg("<memory-context>만 있음</memory-context>", "assistant", 3),
            _msg("답변", "assistant", 4)]
    rep = SanitizeReport()
    out = sanitize_messages(msgs, cfg=scfg, repeat_lines=set(), report=rep)
    assert [(m.ref, m.text) for m in out] == [("U#1", "질문"), ("A#4", "답변")]
    assert rep.dropped_tool == 1 and rep.dropped_empty == 1
    assert out[0] is not msgs[0] and msgs[0].text.startswith("질문 <memory")    # inputs untouched
    assert (out[0].key, out[0].ts, out[0].msg_id) == (msgs[0].key, msgs[0].ts, msgs[0].msg_id)


def test_agent_log_header_prefix_is_not_part_of_line_matching(scfg):
    log_bot = Message(ref="L#md:L3", key="m:x:3", role="agent_log",
                      text="[2026-06-28 정리] 🔁 [BACKUP BOT] 야간 백업 완료 3/3", ts=1.0, source="md")
    log_ok = Message(ref="L#md:L5", key="m:x:5", role="agent_log",
                     text="[2026-06-28 정리] 가계부 DB 열 이름을 바꿨다.", ts=1.0, source="md")
    rep = SanitizeReport()
    out = sanitize_messages([log_bot, log_ok], cfg=scfg, repeat_lines=set(), report=rep)
    assert [m.text for m in out] == ["[2026-06-28 정리] 가계부 DB 열 이름을 바꿨다."]
    assert rep.strip_regex_removed == 1 and rep.dropped_empty == 1


def test_report_merge_and_crlf(scfg):
    a, b = SanitizeReport(), SanitizeReport()
    _clean("x <memory-context>y</memory-context>", scfg, report=a)
    _clean("sk-" + "q" * 30, scfg, report=b)
    a.merge(b)
    assert a.blocks_removed["memory_context"] == 1 and a.secrets == {"openai": 1}
    out, _ = _clean("줄1\r\n\r\n\r\n\r\n줄2", scfg)
    assert out == "줄1\n\n줄2"
