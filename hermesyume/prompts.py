"""LLM prompts (PLAN-v2 §4.2 N3, §4.3 R0-5/different_aspects, §9 M1; CONTRACTS §4.6).

Rules for every prompt here (forensics §3.5/§3.6/§3.8):
- Korean, JSON-only output, explicit "nothing to store" option.
- No numeric examples or anchors (no sample importance/level values, no sample ids or numbers);
  placeholders only ("U#…", "<1-5>").
- No workspace file inventory.
- Relative time words are resolved against each message's own timestamp, never the run date.
- U3: never produce claims about the agent's own memory/dream/consolidation process.
"""

from __future__ import annotations

import json
from typing import Any

PROMPT_KINDS: tuple[str, ...] = ("extract", "extract_retry", "extract_long", "judge", "judge_enum", "consolidate",
                                 "core_classify", "ping")

# Shared kind definitions (13 kinds, §5.1). No numeric examples.
KIND_GUIDE = """\
- rule: 사용자가 정한, 앞으로 계속 지킬 규칙·원칙·금지 사항
- profile: 사람의 정체성 정보(이름, 호칭, 직업, 소속, 시간대, 역할)
- preference: 선호, 취향, 원하는 방식·말투, 어떤 것을 만든 의도
- reference: 나중에 다시 찾아볼 식별 정보(ID, 경로, 주소, 포트, 계정 이름, 도구 이름)
- procedure: 어떤 일을 하는 순서나 방법(단계가 있는 절차)
- decision: 내려진 결정과 그 이유
- lesson: 실패나 문제에서 얻은 교훈과 해결책
- project: 진행 중인 프로젝트, 목표, 역할 분담
- fact: 위 어디에도 맞지 않는 지속적인 사실
- state: 시간이 지나면 바뀌는 현재 상태(보유 자산, 쓰는 모델, 담당자, 진행 상황). 언제 기준인지 날짜를 문장에 넣는다
- schedule: 마감, 약속, 예정된 일. valid_until에 그 날짜를 넣는다
- event: 특정 날짜에 있었던 일
- opinion: 누군가의 견해·평가(날짜와 함께)"""

LEVEL_GUIDE = """\
- 1: 사소하다. 사용자가 기억하라고 직접 말한 경우가 아니면 내지 않는다
- 2: 가끔 쓸모 있다
- 3: 관련 질문이 오면 쓸모 있다
- 4: 앞으로의 작업이나 답변에 자주 영향을 준다
- 5: 핵심이다. 틀리거나 잊으면 사용자에게 실제 손해나 큰 불편이 생긴다"""

EXTRACT_SCHEMA = ('{"claims":[{"kind":"<enum>","target":"user|agent|world","subject":"<짧은 주제>",'
                  '"text":"<독립 문장>","event_time":"YYYY-MM-DD[THH:MM]","valid_until":"YYYY-MM-DD|null",'
                  '"level":"<1-5>","evidence":["U#…","A#…"],"explicit":"<true|false>",'
                  '"steps":"<절차 단계 수 또는 null>"}]}')

EXTRACT_SYSTEM = f"""\
너는 AI 에이전트의 야간 기억 정리 단계다. 대화 기록을 읽고, 앞으로의 대화에 다시 쓸모 있는 장기기억 "주장"만 골라 JSON으로 낸다.

[입력 형식]
- 머리말: 세션 / 기간(메시지가 오간 날짜와 시각, KST) / 정리 기준일(이 정리를 하는 날)
- [이전 맥락 · 추출 대상 아님] 블록: 앞 대화를 이해하는 데만 쓴다. 이 블록에서는 추출하지 않고 근거로도 쓰지 않는다.
- [추출 대상] 블록: 여기 있는 메시지에서만 추출한다.
- 메시지 머리: [U#…]는 사용자, [A#…]는 어시스턴트, [L#…]는 에이전트가 쓴 일지다. 머리에 있는 월-일 시:분이 그 메시지의 시각이고, 연도는 머리말의 기간을 따른다. 머리에 시각이 없으면(일지 파일) 머리말 기간의 날짜를 그 메시지의 날짜로 본다.

[규칙]
1. 원문 언어를 그대로 쓴다. 한국어 대화는 한국어로 쓴다.
2. 날짜는 절대 날짜로 쓴다. "오늘, 내일, 어제, 현재, 지금, 요즘, 이번 주, 다음 주, today, tomorrow, now" 같은 상대 시간어를 쓰지 말고, 그 말을 한 메시지의 시각을 기준으로 계산한 YYYY-MM-DD로 바꿔 쓴다. "지금은 X다", "현재 X다"는 "YYYY-MM-DD 기준 X다"로 쓴다. 정리 기준일은 정리하는 날일 뿐 메시지의 날짜가 아니므로 계산 기준으로 쓰지 않는다.
3. 주장 하나에는 주제 하나와 속성 하나만 담는다. 여러 사실은 나눠서 낸다.
4. text는 이 대화를 모르는 사람이 읽어도 이해되는 독립 문장(한두 문장, 길어도 세 문장)이다. 대명사 대신 고유명사를 쓰고 주어를 밝힌다. 사용자는 "사용자"라고 쓴다.
5. 숫자, 이름, ID, 경로, 날짜는 원문 그대로 옮긴다. 추측으로 바꾸거나 덧붙이지 않는다.
6. 남길 것이 없으면 {{"claims":[]}}를 낸다. 빈 결과는 정상이다. 억지로 주장을 만들지 않는다.

[추출하지 말 것]
- 인사, 잡담, 감정 표현, 맞장구, 대기 상태("알겠습니다", "대기 중")
- 대화 자체에 대한 서술("사용자와 대화를 나눴다", "질문을 했다", "답변했다")
- 파일·폴더·세션·도구 목록, 워크스페이스에 무엇이 있는지에 대한 나열
- 시세·지수·환율·뉴스 수치 그 자체. 사용자의 보유 자산이나 결정과 묶일 때만 날짜를 붙여 state로 낸다.
- 사용자가 동의하거나 확인하지 않은 어시스턴트의 추측·제안·분석·계획
- 어시스턴트가 질문에 답하며 설명한 일반 지식·상식·추천·사용법(누구나 찾을 수 있는 세상 지식, 프로그래밍 문법, 건강·생활 상식, 날씨 이야기). 사용자 자신, 사용자의 일·프로젝트·사람·약속, 사용자가 정하거나 알려 준 것만 남긴다. 그런 설명을 target agent의 교훈·절차로 바꿔 내지도 않는다
- 어시스턴트가 자기 능력·한계·작업 방식이나 이미 정해진 규칙을 되풀이한 말(사용자가 새로 정한 것이 아니면 내지 않는다)
- 비밀값(API 키, 토큰, 비밀번호, 인증 코드)과 [REDACTED:…] 표시가 들어간 내용
- 에이전트 자신의 기억·꿈·기억 정리 과정에 대한 주장("…를 기억하고 있다", "장기기억에 저장됨", "어젯밤 기억 정리에서…", yume, memory-context). 사용자가 "X 기억해"라고 하면 기억했다는 사실이 아니라 X 자체를 주장으로 낸다.
- 시스템 안내문, 도구 실행 결과, 오류 메시지 원문
- 이번 대화 안에서만 쓰이고 끝나는 일회성 요청

[kind: 아래 13개 중 하나]
{KIND_GUIDE}

[kind 고르는 법: 위에서부터 먼저 맞는 것]
- 포트, ID, 경로, 주소, URL, 계정 이름 같은 값을 알려 주는 주장 → reference (fact 아님)
- 단계를 차례로 나열한 순서(A → B → C, 번호 목록) → procedure. "앞으로 이 순서를 지켜"처럼 지키라는 말이 붙어도 rule이 아니다
- 단계 없는 원칙·금지·"항상 …해" → rule
- fact는 다른 kind가 하나도 맞지 않을 때만 쓴다

[level: 앞으로 얼마나 중요한가]
{LEVEL_GUIDE}

[필드]
- kind: 위 13개 중 하나
- target: user(사용자에 관한 것) | agent(에이전트 자신의 설정·작업 방식) | world(그 밖의 사실·프로젝트·세상)
- subject: 짧은 주제(명사구). 같은 대상에 대한 주장에는 같은 subject를 쓴다.
- text: 독립 문장
- event_time: 그 사실이 생기거나 말해진 시각. 근거 메시지 머리의 시각으로 YYYY-MM-DD 또는 YYYY-MM-DDTHH:MM
- valid_until: state와 schedule만 쓴다. 끝나는 날짜 YYYY-MM-DD. 모르거나 다른 kind면 null
- level: 위 기준의 등급 하나
- evidence: 근거 메시지 머리의 첫 토큰(대괄호와 시각을 뺀 U#…, A#…, L#…). [추출 대상] 블록의 메시지만 쓴다.
- explicit: 사용자가 기억하라고 직접 말했거나 앞으로 지킬 규칙을 직접 선언했으면 "true", 아니면 "false"
- steps: procedure일 때 절차의 단계 수, 아니면 null

[출력 전 확인]
- text와 subject에 "현재, 지금, 오늘, 요즘" 같은 상대 시간어가 남아 있으면 지우고 "YYYY-MM-DD 기준"(그 말을 한 메시지의 날짜)을 넣는다. 예: "지금은 X가 맡는다" → "YYYY-MM-DD 기준 X가 맡는다".
- kind가 fact인 주장을 다시 본다. 포트·ID·경로·주소처럼 찾아볼 값을 알려 주면 reference로, 단계 순서면 procedure로 바꾼다.

[출력] 설명 없이 JSON 객체 하나만 낸다.
{EXTRACT_SCHEMA}"""

EXTRACT_RETRY_USER = ('JSON만 다시: 앞 출력은 요구한 형식이 아니었다. 설명 없이 '
                      '{"claims":[…]} 형태의 JSON 객체 하나만 다시 내라. 각 주장은 kind, target, subject, '
                      'text, event_time, valid_until, level, evidence, explicit, steps 필드를 가진다. '
                      '남길 것이 없으면 {"claims":[]}를 낸다.')

JUDGE_SYSTEM = """\
너는 장기기억 저장소의 관계 판정기다. 새 주장 하나와 기존 기억 후보들을 비교해, 후보마다 관계를 하나씩 고른다. 설명 없이 JSON만 낸다.

[관계 type]
- duplicate: 같은 사실이다. 표현만 다르고 숫자, 날짜, 이름, 대상, 부정 여부가 모두 같다.
- state_change: 같은 대상의 같은 속성인데 값이 바뀌었다(담당자 교체, 보유 자산 변경, 일정 변경, 규칙 개정). 둘 중 하나만 지금 유효하다.
- different_aspects: 같은 대상에 대한 서로 다른 측면이다. 둘 다 사실이고 함께 유지할 수 있다.
- unrelated: 관련이 없다.
숫자나 날짜가 하나라도 다르면 duplicate가 아니다. 단, 상태(state) 주장끼리 "언제 기준"인지 밝힌 기준일만 다르고 나머지 값이 모두 같으면 같은 상태를 다시 확인한 것이므로 duplicate다. 주제만 비슷하고 속성이 다르면 state_change가 아니다.

[newer: 둘 중 어느 쪽이 더 나중 상태인가]
- new: 새 주장이 더 나중이다
- existing: 기존 후보가 더 나중이다
- same: 같은 시점이거나 판단할 수 없다
event_time(사실이 생긴 날짜)을 먼저 보고, 없으면 문장 내용으로 판단한다.

[출력] 입력의 모든 후보에 대해 정확히 한 번씩, 후보 id를 그대로 쓴다.
{"relations":[{"id":"<후보 id>","type":"duplicate|state_change|different_aspects|unrelated","newer":"new|existing|same"}]}"""

JUDGE_ENUM_SYSTEM = """\
두 기억의 관계를 하나만 고른다. 설명 없이 JSON만 낸다.
- type: duplicate(같은 사실, 숫자·날짜·이름이 모두 같음) | state_change(같은 대상의 같은 속성 값이 바뀜) | different_aspects(같은 대상의 다른 측면) | unrelated(관련 없음)
- newer: new(새 주장이 더 나중) | existing(기존 기억이 더 나중) | same(같거나 모름)
{"type":"duplicate|state_change|different_aspects|unrelated","newer":"new|existing|same"}"""

CONSOLIDATE_SYSTEM = """\
너는 같은 대상에 대한 두 기억을 하나로 합치는 편집기다. 두 기억의 사실을 하나도 잃지 않고 하나의 독립 문장(한두 문장, 길어도 세 문장)으로 합친다.
- 두 입력의 숫자, 날짜, 이름, ID, 경로, 따옴표 안 문자열, 영문 단어를 모두 그대로 남긴다.
- 새 사실을 지어내거나 추측을 덧붙이지 않는다.
- 상대 시간어(오늘, 현재, 지금 등)를 쓰지 않는다.
- "A / B"처럼 두 문장을 기호로 이어 붙이지 않는다.
- 사용자 메시지에 적힌 최대 길이를 넘지 않는다. 사실을 잃지 않고 그 길이 안에 합칠 수 없으면 text를 빈 문자열로 낸다.
[출력] 설명 없이 JSON만: {"text":"<합친 문장>"}"""

CORE_CLASSIFY_SYSTEM = f"""\
너는 사용자의 핵심 메모(USER.md·MEMORY.md 항목, 또는 사용자가 기억하라고 한 문장)를 분류한다. 항목의 글은 바꾸지 않고, 항목마다 kind와 짧은 subject만 정한다. 설명 없이 JSON만 낸다.

[kind: 아래 13개 중 하나]
{KIND_GUIDE}

[필드]
- i: 입력 항목의 i를 그대로
- kind: 위 13개 중 하나
- subject: 항목이 다루는 대상을 나타내는 짧은 명사구. 항목 앞에 **라벨:**이 있으면 그 라벨 글자를 쓴다.
- fragment: 항목이 값 없이 머리글(라벨)만 있으면 true, 아니면 false

[출력] 모든 항목에 대해 정확히 한 번씩:
{{"items":[{{"i":<입력의 i>,"kind":"<kind>","subject":"<짧은 주제>","fragment":<true 또는 false>}}]}}"""


def _sys(content: str) -> dict:
    return {"role": "system", "content": content}


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _date_str(v: Any) -> str | None:
    """Judge inputs: 'YYYY-MM-DD' or None. Accepts epoch seconds defensively."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        from .clock import kst_date
        return kst_date(float(v))
    return str(v)


def _judge_item(d: dict, *, with_id: bool) -> dict:
    out: dict[str, Any] = {}
    if with_id:
        out["id"] = str(d.get("id", ""))
    out["subject"] = d.get("subject") or ""
    out["kind"] = d.get("kind") or ""
    out["event_time"] = _date_str(d.get("event_time"))
    if with_id and d.get("status"):
        out["status"] = d.get("status")
    out["text"] = d.get("text") or ""
    return out


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)


# ── builders ─────────────────────────────────────────────────────────────────

def extract_messages(window_text: str) -> list[dict]:
    return [_sys(EXTRACT_SYSTEM), _user(window_text)]


def extract_retry_messages(window_text: str, bad_output: str) -> list[dict]:
    bad = (bad_output or "")[:2000] or "(빈 출력)"
    return extract_messages(window_text) + [{"role": "assistant", "content": bad},
                                            _user(EXTRACT_RETRY_USER)]


def judge_messages(new: dict, candidates: list[dict]) -> list[dict]:
    """new {"subject","text","kind","event_time"}; candidates [{"id","subject","text","kind",
    "event_time","status"}]. Expected output: {"relations":[{"id","type","newer"}]}."""
    body = {"new": _judge_item(new, with_id=False),
            "candidates": [_judge_item(c, with_id=True) for c in candidates]}
    return [_sys(JUDGE_SYSTEM), _user(_dumps(body))]


def judge_enum_messages(new: dict, candidate: dict) -> list[dict]:
    """Short retry: expects {"type": <relation>, "newer": <new|existing|same>}."""
    body = {"new": _judge_item(new, with_id=False), "existing": _judge_item(candidate, with_id=False)}
    return [_sys(JUDGE_ENUM_SYSTEM), _user(_dumps(body))]


def consolidate_messages(a: str, b: str, *, max_chars: int) -> list[dict]:
    """Expects {"text": "<≤max_chars>"} ("" when the facts cannot be kept)."""
    body = f"[최대 길이] {int(max_chars)}자\n\n[기억 A]\n{a}\n\n[기억 B]\n{b}"
    return [_sys(CONSOLIDATE_SYSTEM), _user(body)]


def core_classify_messages(entries: list[str]) -> list[dict]:
    """Expects {"items":[{"i":0,"kind":"<KINDS>","subject":"<짧은 주제>","fragment":false}]}."""
    body = {"items": [{"i": i, "text": t} for i, t in enumerate(entries)]}
    return [_sys(CORE_CLASSIFY_SYSTEM), _user(_dumps(body))]


__all__ = ["PROMPT_KINDS", "EXTRACT_SYSTEM", "EXTRACT_RETRY_USER", "JUDGE_SYSTEM",
           "JUDGE_ENUM_SYSTEM", "CONSOLIDATE_SYSTEM", "CORE_CLASSIFY_SYSTEM", "KIND_GUIDE",
           "LEVEL_GUIDE", "extract_messages", "extract_retry_messages", "judge_messages",
           "judge_enum_messages", "consolidate_messages", "core_classify_messages"]
