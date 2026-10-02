"""prompts.py: Korean, JSON-only, no numeric anchors, empty output allowed, absolute dates, U3."""

from __future__ import annotations

import json
import re

from hermesyume import prompts
from hermesyume.types import KINDS, RELATIONS
from tests.fakes import ScriptedLLM

SYSTEMS = ("EXTRACT_SYSTEM", "JUDGE_SYSTEM", "JUDGE_ENUM_SYSTEM", "CONSOLIDATE_SYSTEM",
           "CORE_CLASSIFY_SYSTEM")


def _numbers(s: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", s))


def test_prompt_kinds_cover_fakes_and_ping():
    assert set(prompts.PROMPT_KINDS) == set(ScriptedLLM.DEFAULTS)
    assert "ping" in prompts.PROMPT_KINDS


def test_all_prompts_korean_and_json_only():
    for name in SYSTEMS:
        s = getattr(prompts, name)
        assert re.search(r"[가-힣]", s), name
        assert "JSON" in s, name                       # OpenAI json_object mode needs the word
    assert prompts.EXTRACT_RETRY_USER.startswith("JSON만 다시")


def test_no_numeric_examples_or_anchors():
    # only rubric/rule enumerations (1–6) and the kind count (13) may appear; no decimals, no ids
    allowed = {"1", "2", "3", "4", "5", "6", "13"}
    for name in SYSTEMS + ("EXTRACT_RETRY_USER",):
        s = getattr(prompts, name)
        assert _numbers(s) <= allowed, (name, _numbers(s) - allowed)
        assert not re.search(r"\d\.\d", s), name       # no "0.8"-style importance anchors
        assert not re.search(r"[UAL]#\d", s), name     # no sample message ids
    assert "importance" not in prompts.EXTRACT_SYSTEM


def test_extract_prompt_contract_points():
    s = prompts.EXTRACT_SYSTEM
    assert '{"claims":[]}' in s                        # empty output allowed
    for kind in KINDS:
        assert f"- {kind}:" in s, kind                 # 13 kind definitions
    for word in ("오늘", "내일", "어제", "현재", "지금", "요즘", "today", "tomorrow", "now"):
        assert word in s                               # relative words named as forbidden
    assert "메시지의 시각을 기준으로" in s and "정리 기준일은" in s
    assert "[추출 대상]" in s and "이전 맥락" in s       # evidence only from the body block
    assert "원문 언어" in s
    # U3: memory-system meta claims forbidden, but "X 기억해" → X
    assert "장기기억에 저장됨" in s and "X 자체" in s
    # §4.2 N3 schema fields, exact key order
    keys = ["kind", "target", "subject", "text", "event_time", "valid_until", "level", "evidence",
            "explicit", "steps"]
    schema = prompts.EXTRACT_SCHEMA
    pos = [schema.index(f'"{k}"') for k in keys]
    assert pos == sorted(pos)
    assert "user|agent|world" in schema and "YYYY-MM-DD" in schema


def test_no_workspace_file_inventory():
    s = prompts.EXTRACT_SYSTEM
    for f in ("TOOLS.md", "AGENTS.md", "SOUL.md", "MEMORY.md", ".py"):
        assert f not in s


def test_extract_messages_shape():
    m = prompts.extract_messages("창 본문")
    assert [x["role"] for x in m] == ["system", "user"]
    assert m[0]["content"] == prompts.EXTRACT_SYSTEM and m[1]["content"] == "창 본문"
    r = prompts.extract_retry_messages("창 본문", "x" * 5000)
    assert [x["role"] for x in r] == ["system", "user", "assistant", "user"]
    assert len(r[2]["content"]) == 2000
    assert r[3]["content"] == prompts.EXTRACT_RETRY_USER
    assert prompts.extract_retry_messages("w", "")[2]["content"]   # never an empty assistant turn


def test_judge_prompts():
    s = prompts.JUDGE_SYSTEM
    for rel in RELATIONS:
        assert rel in s
    assert "new|existing|same" in s
    assert "reason" not in s and "설명 없이" in s          # no explanation field
    new = {"subject": "포트", "text": "스테이징 포트는 8081이다.", "kind": "reference", "event_time": "2026-10-01"}
    cands = [{"id": "abc", "subject": "포트", "text": "스테이징 포트는 8082이다.", "kind": "reference",
              "event_time": None, "status": "active"}]
    m = prompts.judge_messages(new, cands)
    body = json.loads(m[1]["content"])
    assert body["new"]["text"] == new["text"] and body["candidates"][0]["id"] == "abc"
    assert body["candidates"][0]["event_time"] is None
    e = prompts.judge_enum_messages(new, cands[0])
    body = json.loads(e[1]["content"])
    assert set(body) == {"new", "existing"}
    assert '"type"' in e[0]["content"] and '"newer"' in e[0]["content"]


def test_judge_accepts_epoch_event_time():
    m = prompts.judge_messages({"subject": "s", "text": "t", "kind": "fact",
                                "event_time": 1790000000.0}, [])
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", json.loads(m[1]["content"])["new"]["event_time"])


def test_consolidate_and_core_classify():
    m = prompts.consolidate_messages("A 사실", "B 사실", max_chars=400)
    assert "400자" in m[1]["content"] and "A 사실" in m[1]["content"] and "B 사실" in m[1]["content"]
    assert '"text"' in m[0]["content"]
    c = prompts.core_classify_messages(["**이름:** 테스트", "**Reading List:**"])
    body = json.loads(c[1]["content"])
    assert body == {"items": [{"i": 0, "text": "**이름:** 테스트"}, {"i": 1, "text": "**Reading List:**"}]}
    for kind in KINDS:
        assert f"- {kind}:" in c[0]["content"]
    assert "fragment" in c[0]["content"]
