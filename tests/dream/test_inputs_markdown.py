"""sources/markdown.py — §3.1(b) md episodes (CONTRACTS §4.2)."""

from __future__ import annotations

import hashlib
import os

import pytest

from hermesyume import clock
from hermesyume.config import Config
from hermesyume.ledger import Ledger
from hermesyume.sanitize import SanitizeReport, sanitize_messages
from hermesyume.sources.markdown import (MdSource, file_state_after, md_key, parse_md_messages,
                                         scan_md_sources)
from tests.fixtures.hermes_home import TEST_FILTERS, tree_hash

NOW = clock.parse_iso("2026-10-02T04:40")


@pytest.fixture
def md_root(tmp_path):
    d = tmp_path / "memory"
    (d / "archive").mkdir(parents=True)
    return d


@pytest.fixture
def mcfg(md_root):
    return Config({"md_sources": [str(md_root)], "md_exclude_globs": TEST_FILTERS["md_exclude_globs"]})


@pytest.fixture
def ledger(tmp_path):
    led = Ledger.open(tmp_path / "ledger.db")
    yield led
    led.close()


def _write(p, text, *, old=True):
    p.write_bytes(text.encode("utf-8"))
    if old:
        t = NOW - 86400
        os.utime(p, (t, t))
    return p


def _scan(cfg, ledger, now=NOW):
    return scan_md_sources(cfg, ledger, now=now)


SESSION_MD = """# Session: 2026-06-07 11:15:30 UTC

- **Session Key**: agent:main:main
- **Session ID**: a1b2c3d4
- **Source**: telegram

## Conversation Summary

user: Conversation info (untrusted metadata):
```json
{
  "message_id": "1001",
  "sender": "테스트"
}
```

Sender (untrusted metadata):
```json
{"label": "테스트 (1)", "id": "1"}
```

ㅎㅇ
assistant: 안녕하세요 사장님!

무엇을 도와드릴까요?
**User:** 오늘 일정 알려줘
"""


def test_parse_session_memory_format():
    data = SESSION_MD.encode("utf-8")
    msgs = parse_md_messages(data, path="/m/2026-06-07-hello.md", start_offset=0, first_line=1,
                             base_ts=123.0)
    assert [(m.role, m.ref) for m in msgs] == [("user", "U#md:L9"), ("assistant", "A#md:L23"),
                                               ("user", "U#md:L26")]
    assert msgs[0].text.endswith("ㅎㅇ") and "Sender (untrusted metadata)" in msgs[0].text
    assert msgs[1].text == "안녕하세요 사장님!\n\n무엇을 도와드릴까요?"
    assert msgs[2].text == "오늘 일정 알려줘"
    for m in msgs:
        assert data[m.msg_id:].split(b"\n", 1)[0].decode().lstrip("*").lower().startswith(m.role[:4])
        assert m.key == md_key("/m/2026-06-07-hello.md", m.line, m.role)
        assert (m.ts, m.source, m.session_id, m.platform) == (123.0, "md", "md:/m/2026-06-07-hello.md", "md")
    rep = SanitizeReport()
    clean = sanitize_messages(msgs, cfg=Config({}), repeat_lines=set(), report=rep)
    assert clean[0].text == "ㅎㅇ" and rep.blocks_removed["untrusted_metadata"] == 2


def test_parse_agent_log_paragraphs_with_header_prefix():
    text = ("# 2026-08-23\n\n## 할 일 상태 업데이트\n- [집] 분리수거 → 완료\n"
            "- [집] 화분 물 주기 → 완료\n\n두 번째 문단\n이어지는 줄\n\n### 세부\n세 번째\n")
    msgs = parse_md_messages(text.encode(), path="/m/a.md", start_offset=0, first_line=1, base_ts=0.0)
    assert [(m.ref, m.role) for m in msgs] == [("L#md:L4", "agent_log"), ("L#md:L7", "agent_log"),
                                               ("L#md:L11", "agent_log")]
    assert msgs[0].text == "[할 일 상태 업데이트] - [집] 분리수거 → 완료\n- [집] 화분 물 주기 → 완료"
    assert msgs[1].text == "[할 일 상태 업데이트] 두 번째 문단\n이어지는 줄"
    assert msgs[2].text == "[세부] 세 번째"


def test_parse_offsets_with_multibyte_and_start_offset():
    pre = "user: 한글 첫 줄\n\n".encode()
    rest = "assistant: 두 번째\n일지 아님(계속)\n".encode()
    whole = pre + rest
    msgs = parse_md_messages(rest, path="/m/x.md", start_offset=len(pre), first_line=3, base_ts=1.0)
    assert len(msgs) == 1 and msgs[0].msg_id == len(pre) and msgs[0].line == 3
    assert msgs[0].text == "두 번째\n일지 아님(계속)"
    assert whole[msgs[0].msg_id:].startswith(b"assistant:")


def test_scan_excludes_sorts_and_recurses(md_root, mcfg, ledger):
    _write(md_root / "2026-09-28-orion.md", "user: 포트는 8081\n")
    _write(md_root / "archive" / "2025-12-10.md", "옛 일지 내용\n")
    _write(md_root / "scratch-notes.md", "제외\n")
    _write(md_root / "notes.md", "날짜 없는 메모\n")
    srcs, excluded = _scan(mcfg, ledger)
    names = [os.path.basename(s.path) for s in srcs]
    assert "scratch-notes.md" not in names and excluded == {"md_excluded": 1}
    assert names[0] == "2025-12-10.md" and names[1] == "2026-09-28-orion.md"
    a = srcs[0]
    assert a.rel == os.path.join("archive", "2025-12-10.md") and a.date == "2025-12-10"
    assert a.slug == "2025-12-10" and a.base_ts == clock.parse_iso("2025-12-10T12:00")
    f = srcs[1]
    assert f.slug == "orion" and f.start_offset == 0 and f.end_offset == f.size
    assert f.sha256 == hashlib.sha256((md_root / "2026-09-28-orion.md").read_bytes()).hexdigest()
    n = next(s for s in srcs if s.path.endswith("notes.md"))
    assert n.date is None and n.slug == "notes" and n.base_ts == pytest.approx(NOW - 86400)
    missing = Config({"md_sources": [str(md_root / "nope")]})
    assert _scan(missing, ledger) == ([], {"md_missing_root": 1})


def test_scan_never_writes(md_root, mcfg, ledger):
    _write(md_root / "2026-09-28-a.md", "user: 질문\nassistant: 답\n")
    before = tree_hash(md_root)
    _scan(mcfg, ledger)
    assert tree_hash(md_root) == before


def _commit(ledger, src: MdSource, processed: int, run_id="r1"):
    ledger.upsert_md(file_state_after(src, processed, run_id))


def test_unchanged_file_skipped_and_append_continues(md_root, mcfg, ledger):
    p = _write(md_root / "2026-09-28-a.md", "user: 첫 질문\nassistant: 첫 답\n")
    (src,) = _scan(mcfg, ledger)[0]
    st = file_state_after(src, src.end_offset, "r1")
    assert st.status == "ok" and st.processed_bytes == src.size
    assert st.prefix_sha256 == hashlib.sha256(p.read_bytes()).hexdigest()
    ledger.upsert_md(st)
    assert _scan(mcfg, ledger)[0] == []                         # unchanged → skipped
    with open(p, "ab") as f:
        f.write("\n## 추가\nuser: 둘째 질문\n".encode())
    (src2,) = _scan(mcfg, ledger)[0]
    assert src2.start_offset == st.processed_bytes and not src2.prefix_changed
    assert [m.text for m in src2.messages] == ["둘째 질문"]
    assert src2.messages[0].line == 5 and src2.header is None
    assert [m.text for m in src2.context_before] == ["첫 질문", "첫 답"]


def test_context_carries_over_after_partial_commit(md_root, mcfg, ledger):
    _write(md_root / "2026-09-28-a.md",
           "# 회의\nuser: 질문 하나\nassistant: 답 하나\n\nuser: 질문 둘\nassistant: 답 둘\n\n둘째 문단도 답\n"
           "## 메모\n일지 문단\n")
    (src,) = _scan(mcfg, ledger)[0]
    assert [m.text for m in src.messages] == ["질문 하나", "답 하나", "질문 둘", "답 둘\n\n둘째 문단도 답",
                                              "[메모] 일지 문단"]
    second_user = src.messages[2]
    _commit(ledger, src, second_user.msg_id)
    (src2,) = _scan(mcfg, ledger)[0]
    assert src2.start_offset == second_user.msg_id and src2.header == "회의"
    assert [m.msg_id for m in src2.messages] == [m.msg_id for m in src.messages[2:]]
    assert [m.text for m in src2.messages] == [m.text for m in src.messages[2:]]
    assert [m.text for m in src2.context_before] == ["질문 하나", "답 하나"]
    st = file_state_after(src2, second_user.msg_id, "r2")
    assert st.status == "partial"


def test_header_carries_over_for_agent_log(md_root, mcfg, ledger):
    _write(md_root / "2026-09-28-b.md", "# 회의\n일지 1\n\n일지 2\n\n일지 3\n")
    (src,) = _scan(mcfg, ledger)[0]
    assert [m.text for m in src.messages] == ["[회의] 일지 1", "[회의] 일지 2", "[회의] 일지 3"]
    _commit(ledger, src, src.messages[1].msg_id)
    (src2,) = _scan(mcfg, ledger)[0]
    assert [m.text for m in src2.messages] == ["[회의] 일지 2", "[회의] 일지 3"]
    assert [m.text for m in src2.context_before] == ["[회의] 일지 1"]


def test_prefix_change_reprocesses_whole_file(md_root, mcfg, ledger):
    p = _write(md_root / "2026-09-28-a.md", "user: 원래 첫 줄\n")
    (src,) = _scan(mcfg, ledger)[0]
    _commit(ledger, src, src.end_offset)
    _write(p, "user: 고친 첫 줄\nuser: 추가 줄\n")
    (src2,) = _scan(mcfg, ledger)[0]
    assert src2.start_offset == 0 and src2.prefix_changed
    assert [m.text for m in src2.messages] == ["고친 첫 줄", "추가 줄"]


def test_partial_last_line_waits_until_settled(md_root, mcfg, ledger):
    p = md_root / "2026-09-28-a.md"
    p.write_bytes("user: 완결된 줄\nuser: 쓰는 중".encode())
    t = NOW - 60                                            # touched a minute ago
    os.utime(p, (t, t))
    (src,) = _scan(mcfg, ledger)[0]
    assert [m.text for m in src.messages] == ["완결된 줄"]
    assert src.end_offset == len("user: 완결된 줄\n".encode())
    _commit(ledger, src, src.end_offset)
    assert _scan(mcfg, ledger)[0] == []                     # nothing complete to add yet
    (src2,) = _scan(mcfg, ledger, now=NOW + 3600)[0]          # settled: EOF ends the line
    assert [m.text for m in src2.messages] == ["쓰는 중"] and src2.end_offset == src2.size


def test_file_state_after_without_cached_bytes(md_root, mcfg, ledger):
    p = _write(md_root / "2026-09-28-a.md", "user: 질문\nuser: 둘\n")
    (src,) = _scan(mcfg, ledger)[0]
    bare = MdSource(**{k: getattr(src, k) for k in ("path", "rel", "date", "slug", "sha256", "size",
                                                     "start_offset", "end_offset", "prefix_changed",
                                                     "base_ts", "messages")})
    st = file_state_after(bare, src.messages[1].msg_id, "r1", status="ok")
    assert st.prefix_sha256 == hashlib.sha256(p.read_bytes()[:src.messages[1].msg_id]).hexdigest()
    assert st.status == "ok"
    _write(p, "user: 바뀜\n")
    with pytest.raises(ValueError):
        file_state_after(bare, 5, "r1")


def test_fixture_episode(fake_home, cfg, ledger):
    srcs, excluded = scan_md_sources(cfg, ledger, now=NOW)
    assert excluded == {"md_excluded": 1}
    (src,) = srcs
    assert [(m.ref, m.role) for m in src.messages] == [("U#md:L3", "user"), ("U#md:L4", "user"),
                                                       ("A#md:L5", "assistant")]
    clean = sanitize_messages(src.messages, cfg=cfg, repeat_lines=set(), report=SanitizeReport())
    assert clean[2].text == "알겠습니다."                     # BACKUP BOT tail stripped (strip_line_regex)


def test_label_variants():
    text = "- user: 목록형\n**Assistant:** 볼드형\n**user**: 볼드2\nUSER : 대문자\nusers: 아님\n"
    msgs = parse_md_messages(text.encode(), path="/m/v.md", start_offset=0, first_line=1, base_ts=0.0)
    assert [(m.role, m.text) for m in msgs] == [("user", "목록형"), ("assistant", "볼드형"),
                                                ("user", "볼드2"), ("user", "대문자\nusers: 아님")]
