"""N4 deterministic gates: RawClaim → Claim | Rejection (PLAN-v2 §4.2 N4 + U2/U3; CONTRACTS §4.8).

Check order (first failure wins, reason ∈ types.GATE_REASONS):
kind_enum → length → evidence_outside → level1_not_explicit → relative_time → meta_pattern →
memory_meta → uuid → filenames → secret → threat → assistant_only.

U2: no claim is ever `candidate`. Assistant-only rule/profile/preference stay active and become tier
`decaying` in strength.compute_tier (source "dream", no user evidence); agent_log-only (md) protected
kinds become `slow`. Status here is only "active" or "expired".

U3 (DEVIATIONS E2E-7): a claim whose evidence is only assistant messages (no user, no agent_log) is
rejected (`assistant_only`) unless it is the agent's own work/learning — target `agent` with kind
procedure/lesson/reference/decision/project — or a U2 rule/profile/preference. Fact/opinion/event
and target `world` are always rejected: those are the assistant's general-knowledge explanations.
The check runs last so secret/threat rejections keep their own (redacted) Dream Log treatment.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from dataclasses import replace

from .clock import DAY, kst_date, parse_iso
from .types import (EXPIRING_KINDS, KINDS, TARGETS, TEXT_MAX_CHARS, TEXT_MIN_CHARS,
                    USER_EVIDENCE_REQUIRED_KINDS, Claim, RawClaim, Rejection, Window)

# §4.2 N4 list + other deictic Korean/English time words (DEVIATIONS F-34). "최근" is relative only
# when not followed by a duration number ("최근 7일치" is a rolling window, not a date).
RELATIVE_TIME_RE = re.compile(
    r"오늘|내일|어제|현재|지금|요즘|이번 ?주|다음 ?주|지난 ?주|저번 ?주|다음 ?달|이번 ?달|지난 ?달|저번 ?달|"
    r"올해|내년|작년|재작년|모레|글피|그저께|그제|엊그제|어젯밤|오늘밤|최근(?!\s*\d)|"
    r"(?<![가-힣])곧(?![가-힣])|"
    r"\b(?:today|tomorrow|yesterday|tonight|now|(?:this|next|last) (?:week|month|year))\b", re.I)
ABS_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{4}년\s*\d{1,2}월|\d{1,2}월\s*\d{1,2}일")
# "현재/지금 X" said at t means "t 기준 X" (DEVIATIONS E2E-2): anchored to the evidence date instead of
# rejected. Standalone word only (+은/는/도) — "지금까지", "현재가", "현재진행" stay untouched.
PRESENT_RE = re.compile(r"(?<![가-힣])(?:현재|지금)(?:은|는|도)?(?=[\s,])")
META_RES: list[re.Pattern] = [
    re.compile(r"워크스페이스에 .* (존재|있)"),
    re.compile(r"내용이 제공되지 않"),
    re.compile(r"생성할 수 없"),
    re.compile(r"대화(가|를) (나눴|했)"),
    re.compile(r"대기 중"),
    re.compile(r"평화롭게"),
]
UUID_RE = re.compile(r"\b[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}\b", re.I)
FILENAME_RE = re.compile(
    r"(?<![\w.])[\w.\-/~]*[\w\-]\.(?:md|py|json|ya?ml|sh|txt|csv|log|db|sqlite|js|ts|tsx|html|css|"
    r"toml|ini|cfg|conf|xlsx|xlsm|xls|pdf|docx|pptx|png|jpe?g|gif|zip|tar|gz|lance)\b", re.I)
EXPLICIT_RE = re.compile(r"앞으로|항상|절대|규칙|원칙|기억해|잊지 ?마|원장|단일 ?출처|기준은|always|never|remember", re.I)
MEMORY_META_RE = re.compile(                                  # U3: claims about the memory system itself
    r"장기\s*기억|기억\s*정리|꿈(속|에서|을\s*꾸)|기억(을|이|에)?\s*(저장|정리|삭제|보관|남겨)|"
    r"기억하고\s*있|기억해\s*(두었|뒀|둠)|어젯밤.{0,20}(정리|기억)|\byume\b|memory[-_ ]?context|"
    r"<memory|dream\s*log|드림\s*로그|회상(된|했|함)", re.I)
REDACTED_RE = re.compile(r"\[REDACTED:[a-z]+\]", re.I)
# U3 assistant-only policy (DEVIATIONS E2E-7); listed in types.GATE_REASONS.
ASSISTANT_ONLY_REASON = "assistant_only"
ASSISTANT_ONLY_REJECT_KINDS: frozenset[str] = frozenset({"fact", "opinion", "event"})
ASSISTANT_ONLY_AGENT_KINDS: frozenset[str] = frozenset({"procedure", "lesson", "reference", "decision",
                                                        "project"})
_HANGUL_RE = re.compile(r"[가-힣]")
SUBJECT_MAX_CHARS = 60


def anchor_present(text: str, ts: float | None) -> str:
    """'현재/지금 X' → '<YYYY-MM-DD of ts> 기준 X' when those are the only relative time words and the
    text has no absolute date (otherwise unchanged, so N4 still rejects e.g. '현재 계획은 다음 주')."""
    if not text or ts is None or ABS_DATE_RE.search(text) or not PRESENT_RE.search(text):
        return text
    if RELATIVE_TIME_RE.search(PRESENT_RE.sub(" ", text)):
        return text
    return PRESENT_RE.sub(f"{kst_date(float(ts))} 기준", text)


def _strip_present(subject: str) -> str:
    return re.sub(r"\s+", " ", PRESENT_RE.sub(" ", subject or "")).strip()


def _rej(raw: RawClaim, window: Window, reason: str, detail: str) -> Rejection:
    return Rejection(window_id=window.window_id, idx=raw.idx, reason=reason, detail=detail,
                     text=raw.text, kind=raw.kind)


def evidence_fields(refs: list[str], window: Window) -> dict:
    """Evidence bookkeeping for refs that exist in the window body (unknown refs ignored)."""
    index = window.evidence_index
    msgs = []
    seen_keys: set[str] = set()
    for r in refs:
        m = index.get(r)
        if m is None or m.key in seen_keys:
            continue
        seen_keys.add(m.key)
        msgs.append(m)
    roles: list[str] = []
    sessions: list[str] = []
    for m in msgs:
        if m.role not in roles:
            roles.append(m.role)
        sid = m.session_id or window.root
        if sid and sid not in sessions:
            sessions.append(sid)
    user_msgs = [m for m in msgs if m.role == "user"]
    user_sessions = {m.session_id or window.root for m in user_msgs}
    ts = [m.ts for m in msgs]
    return {
        "evidence_keys": [m.key for m in msgs],
        "evidence_roles": roles,
        "session_ids": sessions,
        "first_seen_at": min(ts) if ts else window.start_ts,
        "last_seen_at": max(ts) if ts else window.last_ts,
        "last_user_evidence_at": max(m.ts for m in user_msgs) if user_msgs else None,
        "user_evidence_count": len(user_msgs),
        "user_session_count": len(user_sessions),
        "user_session_ids": [s for s in sessions if s in user_sessions],
    }


def deadline(kind: str, event_time: float | None, valid_until: float | None, cfg: Any) -> float | None:
    """state → valid_until or event_time + state_default_ttl_days; schedule → valid_until or
    event_time (the due date; the grace is applied by strength). Other kinds carry no deadline."""
    if kind == "state":
        if valid_until is not None:
            return valid_until
        return None if event_time is None else event_time + float(cfg.state_default_ttl_days) * DAY
    if kind == "schedule":
        return valid_until if valid_until is not None else event_time
    return None


def _subject(raw: RawClaim) -> str:
    s = (raw.subject or "").strip()
    if not s:
        s = raw.text.strip().split("\n", 1)[0]
    s = re.sub(r"\s+", " ", s)
    return s[:SUBJECT_MAX_CHARS].strip()


def _lang(text: str) -> str:
    return "ko" if _HANGUL_RE.search(text or "") else "en"


def _target(raw: RawClaim) -> str:
    return raw.target if raw.target in TARGETS else "user"


def assistant_only_allowed(kind: str, target: str) -> bool:
    """U3 (E2E-7): may a claim backed only by assistant messages be stored (decaying, low importance)?
    Yes for the agent's own work/learning (target agent + procedure/lesson/reference/decision/project)
    and for U2 rule/profile/preference (target user/agent; tier decaying in strength.compute_tier).
    Never for fact/opinion/event or target world (general-knowledge explanations)."""
    if kind in ASSISTANT_ONLY_REJECT_KINDS or target == "world":
        return False
    if target == "agent" and kind in ASSISTANT_ONLY_AGENT_KINDS:
        return True
    return kind in USER_EVIDENCE_REQUIRED_KINDS


def _check(raw: RawClaim, window: Window, *, scanner: Any,
           explicit_user: bool, assistant_only: bool = False) -> tuple[str, str] | None:
    text = raw.text
    if raw.kind not in KINDS:
        return "kind_enum", raw.kind or "(없음)"
    n = len(text)
    if n < TEXT_MIN_CHARS or n > TEXT_MAX_CHARS:
        return "length", f"{n}자"
    index = window.evidence_index
    missing = [r for r in raw.evidence if r not in index]
    if not raw.evidence or missing:
        return "evidence_outside", ", ".join(missing[:5]) if missing else "근거 없음"
    if raw.level == 1 and not explicit_user:
        return "level1_not_explicit", "level 1"
    m = RELATIVE_TIME_RE.search(text)
    if m and not ABS_DATE_RE.search(text):
        return "relative_time", m.group(0)
    for pat in META_RES:
        m = pat.search(text)
        if m:
            return "meta_pattern", m.group(0)
    m = MEMORY_META_RE.search(text) or MEMORY_META_RE.search(raw.subject or "")
    if m:
        return "memory_meta", m.group(0)
    uuids = {u.lower().replace("-", "") for u in UUID_RE.findall(text)}
    if len(uuids) >= 2:
        return "uuid", f"UUID {len(uuids)}개"
    files = {f.lower() for f in FILENAME_RE.findall(text)}
    if len(files) >= 3:
        return "filenames", f"파일명 {len(files)}개"
    both = f"{text}\n{raw.subject or ''}"
    sec = scanner.secrets(both)
    if sec:
        return "secret", ",".join(sec)
    if REDACTED_RE.search(both):
        return "secret", "redacted_placeholder"
    thr = scanner.threats(both, "strict")
    if thr:
        return "threat", ",".join(thr[:5])
    if assistant_only and not assistant_only_allowed(raw.kind, _target(raw)):
        return ASSISTANT_ONLY_REASON, f"{raw.kind}/{_target(raw)}"
    return None


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。])\s+|\n+")
# words that point back at the conversation ("이건 꼭 기억해", "앞으로 그렇게 해") rather than state a fact
_DEICTIC = frozenset({"이건", "이거", "이것", "그건", "그거", "그것", "저거", "이걸", "그걸", "이렇게", "그렇게",
                      "저렇게", "위에", "방금", "this", "that", "it"})


def _pure_instruction(sentence: str) -> bool:
    """An explicit-word sentence with no fact of its own refers to the surrounding claims."""
    from .vecutil import fact_tokens
    if not EXPLICIT_RE.search(sentence):
        return False
    rest = {t for t in fact_tokens(sentence) if t not in _DEICTIC and not EXPLICIT_RE.search(t)}
    return not rest


def _claim_sentences(user_texts: list[str], claim_text: str, subject: str) -> list[str]:
    """User-evidence sentences that are about this claim: a sentence counts when it shares at
    least max(2, 30 %) of the claim's fact tokens, or when it is a bare instruction ("이건 꼭
    기억해") that points back at the claims around it (DEVIATIONS F-23)."""
    from .vecutil import fact_tokens
    ctoks = fact_tokens(f"{claim_text} {subject or ''}")
    need = min(max(2, int(0.3 * len(ctoks) + 0.999)), len(ctoks)) if ctoks else 1
    out = []
    for t in user_texts:
        for sent in _SENT_SPLIT_RE.split(t or ""):
            if not sent.strip():
                continue
            if (ctoks and len(fact_tokens(sent) & ctoks) >= need) or _pure_instruction(sent):
                out.append(sent)
    return out


def explicit_from_user(user_texts: list[str], claim_text: str, subject: str = "") -> bool:
    """§5.3 regex, applied only to the user sentences that state this claim — a "항상/절대" elsewhere
    in a long message must not make every claim of that message explicit."""
    return any(EXPLICIT_RE.search(s) for s in _claim_sentences(user_texts, claim_text, subject))


def gate_claim(raw: RawClaim, window: Window, *, scanner: Any, cfg: Any, now: float) -> Claim | Rejection:
    ev = evidence_fields(raw.evidence, window)
    has_user = ev["user_evidence_count"] > 0
    index = window.evidence_index
    user_texts = [index[r].text for r in raw.evidence if r in index and index[r].role == "user"]
    # D8: explicit only with user evidence (LLM flag OR user text regex on the claim's own sentences)
    explicit_user = has_user and (bool(raw.explicit)
                                  or explicit_from_user(user_texts, raw.text, raw.subject or ""))

    anchored = anchor_present(raw.text, ev["last_seen_at"])
    if anchored != raw.text:
        raw = replace(raw, text=anchored, subject=_strip_present(raw.subject or ""))

    # U3: evidence roles exactly {assistant} — no user message, no agent_log (E2E-7)
    assistant_only = set(ev["evidence_roles"]) == {"assistant"}
    bad = _check(raw, window, scanner=scanner, explicit_user=explicit_user,
                 assistant_only=assistant_only)
    if bad is not None:
        return _rej(raw, window, *bad)

    event_time = parse_iso(raw.event_time)
    if event_time is None:
        event_time = ev["last_seen_at"]
    valid_until = deadline(raw.kind, event_time, parse_iso(raw.valid_until, end_of_day=True), cfg)
    status = "active"
    if raw.kind == "state" and valid_until is not None and now > valid_until:
        status = "expired"
    elif raw.kind == "schedule" and valid_until is not None and \
            now > valid_until + float(cfg.schedule_grace_days) * DAY:
        status = "expired"

    text = unicodedata.normalize("NFC", raw.text.strip())
    return Claim(
        origin_key=f"{window.window_id}#{raw.idx}",
        source="dream" if window.source == "statedb" else "md",
        kind=raw.kind,
        target=_target(raw),
        subject=_subject(raw),
        text=text,
        level=int(raw.level),
        explicit=bool(raw.explicit),
        steps=raw.steps if raw.kind == "procedure" else None,
        event_time=event_time,
        valid_until=valid_until if raw.kind in EXPIRING_KINDS else None,
        status=status,
        window_id=window.window_id,
        evidence_refs=list(raw.evidence),
        evidence_keys=ev["evidence_keys"],
        evidence_roles=ev["evidence_roles"],
        session_ids=ev["session_ids"],
        first_seen_at=ev["first_seen_at"],
        last_seen_at=ev["last_seen_at"],
        last_user_evidence_at=ev["last_user_evidence_at"],
        user_evidence_count=ev["user_evidence_count"],
        user_session_count=ev["user_session_count"],
        user_session_ids=ev["user_session_ids"],
        explicit_user=explicit_user,
        lang=_lang(text),
    )


def gate_claims(raws: list[RawClaim], window: Window, *, scanner: Any, cfg: Any,
                now: float) -> tuple[list[Claim], list[Rejection]]:
    claims: list[Claim] = []
    rejects: list[Rejection] = []
    for raw in raws:
        out = gate_claim(raw, window, scanner=scanner, cfg=cfg, now=now)
        (rejects if isinstance(out, Rejection) else claims).append(out)
    return claims, rejects


__all__ = ["RELATIVE_TIME_RE", "ABS_DATE_RE", "PRESENT_RE", "anchor_present", "META_RES", "UUID_RE",
           "FILENAME_RE", "EXPLICIT_RE", "ASSISTANT_ONLY_REASON", "ASSISTANT_ONLY_REJECT_KINDS",
           "ASSISTANT_ONLY_AGENT_KINDS", "assistant_only_allowed",
           "MEMORY_META_RE", "evidence_fields", "deadline", "gate_claim", "gate_claims"]
