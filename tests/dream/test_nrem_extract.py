"""extract.py (N3): schema validation, coercion, one retry, failure modes."""

from __future__ import annotations

import pytest

from hermesyume import prompts
from hermesyume.extract import extract_window, validate_payload
from hermesyume.llm import LLMAuthError, LLMError
from hermesyume.types import BudgetExceeded, RunBudget, Window
from tests.fakes import ScriptedLLM, claim, extract_json


def _window() -> Window:
    return Window(window_id="w1", source="statedb", root="S1", first_id=1, last_id=2, start_ts=0.0,
                  last_ts=0.0, platform="telegram", title="t", header="h", text="머리말\n\n[추출 대상]\n[U#1 09-28 14:02] 안녕")


# ── validate_payload ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("data,err", [
    (None, "unparseable"), ([{"kind": "fact"}], "payload_list"), ("str", "payload_str"),
    ({}, "claims_missing"), ({"claims": {"a": 1}}, "claims_not_list"),
])
def test_payload_errors(data, err):
    assert validate_payload(data) == ([], [], err)


def test_empty_claims_is_valid():
    assert validate_payload({"claims": []}) == ([], [], None)


def test_coercion_of_plan_shape():
    c = claim("fact", "사용자는 2026-09-28에 회의를 했다.", subject=" 회의 ", level="4",
              evidence=["U#1", "A#2"], explicit="true", steps=None)
    c["kind"] = " FACT "
    c["target"] = "World"
    claims, rejs, err = validate_payload(extract_json(c), window_id="w1")
    assert err is None and rejs == []
    r = claims[0]
    assert (r.idx, r.kind, r.target, r.subject, r.level, r.explicit, r.steps) == \
        (0, "fact", "world", "회의", 4, True, None)
    assert r.evidence == ["U#1", "A#2"] and r.event_time == "2026-09-28" and r.valid_until is None


@pytest.mark.parametrize("level,ok", [(3, True), ("3", True), ("3.0", True), (5.0, True),
                                      ("0", False), (6, False), ("high", False), (True, False),
                                      (None, False), ("2.5", False)])
def test_level_coercion(level, ok):
    c = claim("fact", "충분히 긴 사실 문장입니다.", evidence=["U#1"])
    c["level"] = level
    claims, rejs, err = validate_payload({"claims": [c]}, window_id="w")
    assert err is None
    assert (len(claims), len(rejs)) == ((1, 0) if ok else (0, 1))
    if not ok:
        assert rejs[0].reason == "schema" and rejs[0].detail == "level" and rejs[0].window_id == "w"


@pytest.mark.parametrize("val,exp", [(True, True), (False, False), ("true", True), ("False", False),
                                     ("yes", True), ("no", False), (1, True), (0, False),
                                     ("1", True), (None, False)])
def test_explicit_coercion(val, exp):
    c = claim("fact", "충분히 긴 사실 문장입니다.", evidence=["U#1"])
    c["explicit"] = val
    claims, _, _ = validate_payload({"claims": [c]})
    assert claims[0].explicit is exp


def test_explicit_garbage_is_schema_reject():
    c = claim("fact", "충분히 긴 사실 문장입니다.", evidence=["U#1"])
    c["explicit"] = "maybe"
    claims, rejs, _ = validate_payload({"claims": [c]})
    assert not claims and rejs[0].detail == "explicit"


@pytest.mark.parametrize("val,exp", [("3", 3), (4, 4), ("4.0", 4), ("null", None), ("", None),
                                     (None, None), ("세 단계", None), (0, None)])
def test_steps_coercion(val, exp):
    c = claim("procedure", "배포 절차는 빌드, 검사, 배포 순서다.", evidence=["U#1"])
    c["steps"] = val
    claims, _, _ = validate_payload({"claims": [c]})
    assert claims[0].steps == exp


@pytest.mark.parametrize("val,exp", [("U#1", ["U#1"]), (["[U#1 09-28 14:02]", "A#2"], ["U#1", "A#2"]),
                                     ("U#1, A#2", ["U#1", "A#2"]), (["U#md:L12"], ["U#md:L12"]),
                                     (["L#inbox:7", "L#inbox:7"], ["L#inbox:7"]), (None, []),
                                     ([None, 3], ["3"])])
def test_evidence_coercion(val, exp):
    c = claim("fact", "충분히 긴 사실 문장입니다.")
    c["evidence"] = val
    claims, _, _ = validate_payload({"claims": [c]})
    assert claims[0].evidence == exp


def test_null_strings_become_none():
    c = claim("state", "사용자는 2026-09-28 기준 삼성전자를 보유했다.", evidence=["U#1"], valid_until="null")
    c["event_time"] = "None"
    claims, _, _ = validate_payload({"claims": [c]})
    assert claims[0].event_time is None and claims[0].valid_until is None


def test_per_claim_failures_keep_rest_and_idx():
    good = claim("fact", "충분히 긴 사실 문장입니다.", evidence=["U#1"])
    no_text = dict(good, text="  ")
    no_kind = dict(good)
    no_kind.pop("kind")
    claims, rejs, err = validate_payload({"claims": ["문자열", no_text, good, no_kind]}, window_id="w9")
    assert err is None
    assert [c.idx for c in claims] == [2]
    assert [(r.idx, r.reason, r.detail) for r in rejs] == [
        (0, "schema", "claim_not_object"), (1, "schema", "text_empty"), (3, "schema", "kind_missing")]
    assert rejs[0].text == "문자열"


# ── extract_window ───────────────────────────────────────────────────────────

def test_ok_single_call_with_cfg_params(cfg):
    llm = ScriptedLLM().queue("extract", extract_json(claim("fact", "충분히 긴 사실 문장입니다.", evidence=["U#1"])))
    w = _window()
    res = extract_window(w, llm=llm, cfg=cfg)
    assert res.status == "ok" and len(res.claims) == 1 and res.llm_calls == 1
    call = llm.calls[0]
    assert call["kind"] == "extract" and call["model"] == cfg.extract_model
    assert call["max_tokens"] == cfg.extract_max_tokens == 6000 and call["temperature"] == 0.0
    assert call["messages"] == prompts.extract_messages(w.text)


def test_empty_output_is_ok(cfg):
    res = extract_window(_window(), llm=ScriptedLLM(), cfg=cfg)     # default {"claims": []}
    assert res.status == "ok" and res.claims == [] and res.llm_calls == 1


@pytest.mark.parametrize("bad", ["설명입니다. JSON 아님", '[{"kind":"fact"}]', '{"memories": []}'])
def test_retry_once_then_ok(cfg, bad):
    good = extract_json(claim("fact", "충분히 긴 사실 문장입니다.", evidence=["U#1"]))
    llm = ScriptedLLM().queue("extract", bad).queue("extract_retry", good)
    res = extract_window(_window(), llm=llm, cfg=cfg)
    assert res.status == "ok" and len(res.claims) == 1 and res.llm_calls == 2
    retry = llm.calls_of("extract_retry")
    assert len(retry) == 1
    assert retry[0]["messages"][2] == {"role": "assistant", "content": bad}
    assert retry[0]["messages"][3]["content"] == prompts.EXTRACT_RETRY_USER


def test_retry_still_bad_fails(cfg):
    llm = ScriptedLLM().queue("extract", "nope").queue("extract_retry", "still nope")
    res = extract_window(_window(), llm=llm, cfg=cfg)
    assert res.status == "failed" and res.claims == [] and res.llm_calls == 2
    assert "retry" in res.error and res.raw == "still nope"
    assert len(llm.calls) == 2                       # exactly one retry


def test_llm_error_fails_without_retry(cfg):
    llm = ScriptedLLM().queue("extract", LLMError("HTTP 500", status=500))
    res = extract_window(_window(), llm=llm, cfg=cfg)
    assert res.status == "failed" and res.llm_calls == 1 and llm.calls_of("extract_retry") == []


def test_llm_error_on_retry_fails(cfg):
    llm = ScriptedLLM().queue("extract", "nope").queue("extract_retry", LLMError("down"))
    res = extract_window(_window(), llm=llm, cfg=cfg)
    assert res.status == "failed" and res.raw == "nope"


@pytest.mark.parametrize("exc", [LLMAuthError("401", status=401), BudgetExceeded("llm_calls")])
def test_auth_and_budget_propagate(cfg, exc):
    with pytest.raises(type(exc)):
        extract_window(_window(), llm=ScriptedLLM().queue("extract", exc), cfg=cfg)
    with pytest.raises(type(exc)):
        extract_window(_window(), llm=ScriptedLLM().queue("extract", "bad").queue("extract_retry", exc), cfg=cfg)


def test_budget_charged_per_call(cfg):
    b = RunBudget(max_llm_calls=1)
    llm = ScriptedLLM(budget=b).queue("extract", "bad")
    with pytest.raises(BudgetExceeded):
        extract_window(_window(), llm=llm, cfg=cfg)
    assert b.llm_calls == 1


# ── truncated output (finish_reason "length") ────────────────────────────────

def test_truncated_output_retries_same_prompt_with_long_budget(cfg):
    from tests.fakes import Truncated
    good = extract_json(claim("fact", "충분히 긴 사실 문장입니다.", evidence=["U#1"]))
    import json
    full = good if isinstance(good, str) else json.dumps(good, ensure_ascii=False)
    cut = full[: len(full) // 2]                       # what a cut-off answer looks like
    llm = ScriptedLLM().queue("extract", Truncated(cut)).queue("extract_long", good)
    w = _window()
    res = extract_window(w, llm=llm, cfg=cfg)
    assert res.status == "ok" and len(res.claims) == 1 and res.llm_calls == 2
    long_call = llm.calls_of("extract_long")[0]
    assert long_call["max_tokens"] == cfg.extract_long_max_tokens == 16000
    assert long_call["messages"] == prompts.extract_messages(w.text)    # same prompt, not "fix the JSON"
    assert not llm.calls_of("extract_retry")


def test_truncated_twice_fails_without_json_fix_retry(cfg):
    from tests.fakes import Truncated
    llm = ScriptedLLM().queue("extract", Truncated('{"claims": [{"kind": "fact"')) \
                       .queue("extract_long", Truncated('{"claims": [{"kind": "fact", "text": "abc'))
    res = extract_window(_window(), llm=llm, cfg=cfg)
    assert res.status == "failed" and res.error == "truncated" and res.llm_calls == 2
    assert not llm.calls_of("extract_retry")
