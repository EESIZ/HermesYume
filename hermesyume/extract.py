"""N3 extraction: one window → typed RawClaims (PLAN-v2 §4.2 N3; CONTRACTS §4.7).

- json_object, temperature cfg.llm_temperature (0), max_tokens cfg.extract_max_tokens (6000)
- output cut off at max_tokens (finish_reason "length") → one "extract_long" call with the same
  prompt and cfg.extract_long_max_tokens; still cut off → status "failed" (error "truncated")
- payload error (not a dict / no "claims" list / unparseable, incl. a top-level list) → exactly one
  "extract_retry" ("JSON만 다시"); still bad → status "failed" (the window keeps its watermark)
- per-claim coercion failures → Rejection(reason="schema"); the rest of the payload is kept
- LLMError → failed (llm.py already retried); LLMAuthError / BudgetExceeded propagate
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from . import prompts
from .llm import LLMAuthError, LLMError
from .types import BudgetExceeded, RawClaim, Rejection, Window

log = logging.getLogger("hermesyume.extract")

RAW_KEEP = 2000
_REF_RE = re.compile(r"[UAL]#[^\s,\[\]\"'`;]+")
_TRUE = frozenset({"true", "yes", "y", "1", "t", "참", "예"})
_FALSE = frozenset({"false", "no", "n", "0", "f", "거짓", "아니오", ""})
_NULLS = frozenset({"", "null", "none", "nil", "n/a", "na", "-"})


@dataclass
class ExtractResult:
    status: str                          # "ok" | "failed"
    claims: list[RawClaim] = field(default_factory=list)
    schema_rejections: list[Rejection] = field(default_factory=list)
    error: str | None = None
    llm_calls: int = 0
    raw: str = ""                        # last raw output (≤2000 chars), for the Dream Log on failure


class _Coerce(ValueError):
    """Per-claim coercion failure (→ Rejection reason 'schema')."""


# ── coercion helpers ─────────────────────────────────────────────────────────

def _opt_str(v: Any) -> str | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        v = str(v)
    if not isinstance(v, str):
        return None
    s = v.strip()
    return None if s.lower() in _NULLS else s


def _str(v: Any, name: str, *, required: bool = False) -> str:
    if v is None:
        if required:
            raise _Coerce(f"{name}_missing")
        return ""
    if isinstance(v, bool) or isinstance(v, (dict, list)):
        raise _Coerce(f"{name}_type")
    if isinstance(v, (int, float)):
        v = str(v)
    if not isinstance(v, str):
        raise _Coerce(f"{name}_type")
    s = v.strip()
    if required and not s:
        raise _Coerce(f"{name}_empty")
    return s


def _level(v: Any) -> int:
    if isinstance(v, bool) or v is None:
        raise _Coerce("level")
    if isinstance(v, (int, float)):
        f = float(v)
    elif isinstance(v, str):
        try:
            f = float(v.strip())
        except ValueError:
            raise _Coerce("level") from None
    else:
        raise _Coerce("level")
    if not math.isfinite(f) or f != int(f) or not 1 <= int(f) <= 5:
        raise _Coerce("level")
    return int(f)


def _bool(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
    raise _Coerce("explicit")


def _steps(v: Any) -> int | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v.strip()) if isinstance(v, str) else float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f < 1:
        return None
    return int(f)


def _one_ref(s: str) -> list[str]:
    s = s.strip()
    if not s:
        return []
    found = _REF_RE.findall(s)
    if found:
        return found
    return [s.strip("[]").split()[0]] if s.strip("[]").split() else []


def _evidence(v: Any) -> list[str]:
    if v is None:
        return []
    items = [v] if isinstance(v, (str, int, float)) and not isinstance(v, bool) else v
    if not isinstance(items, list):
        raise _Coerce("evidence_type")
    out: list[str] = []
    for it in items:
        if it is None or isinstance(it, (dict, list, bool)):
            continue
        for r in _one_ref(str(it)):
            if r not in out:
                out.append(r)
    return out


def _coerce_claim(idx: int, item: Any) -> RawClaim:
    if not isinstance(item, dict):
        raise _Coerce("claim_not_object")
    kind = _str(item.get("kind"), "kind", required=True).lower()
    target = (_opt_str(item.get("target")) or "").lower()
    text = _str(item.get("text"), "text", required=True)
    subject = _str(item.get("subject"), "subject")
    return RawClaim(idx=idx, kind=kind, target=target, subject=subject, text=text,
                    event_time=_opt_str(item.get("event_time")),
                    valid_until=_opt_str(item.get("valid_until")),
                    level=_level(item.get("level")), evidence=_evidence(item.get("evidence")),
                    explicit=_bool(item.get("explicit")), steps=_steps(item.get("steps")))


def validate_payload(data: Any, *, window_id: str | None = None
                     ) -> tuple[list[RawClaim], list[Rejection], str | None]:
    """Payload error → ([], [], "<error>"); per-claim coercion failure → Rejection(reason="schema").
    `idx` is the claim's position in the returned list."""
    if data is None:
        return [], [], "unparseable"
    if isinstance(data, list):
        return [], [], "payload_list"
    if not isinstance(data, dict):
        return [], [], f"payload_{type(data).__name__}"
    items = data.get("claims")
    if not isinstance(items, list):
        return [], [], "claims_missing" if items is None else "claims_not_list"
    claims: list[RawClaim] = []
    rejects: list[Rejection] = []
    for i, item in enumerate(items):
        try:
            claims.append(_coerce_claim(i, item))
        except _Coerce as e:
            text = item.get("text") if isinstance(item, dict) else item
            kind = item.get("kind") if isinstance(item, dict) else ""
            rejects.append(Rejection(window_id=window_id, idx=i, reason="schema", detail=str(e),
                                     text=str(text if text is not None else "")[:400],
                                     kind=str(kind or "")[:40]))
    return claims, rejects, None


# ── N3 ───────────────────────────────────────────────────────────────────────

def _call(llm: Any, kind: str, messages: list[dict], cfg: Any, max_tokens: int | None = None):
    return llm.chat_json(kind, messages, model=cfg.extract_model,
                         max_tokens=int(max_tokens or cfg.extract_max_tokens),
                         temperature=float(cfg.llm_temperature))


def _truncated(resp: Any) -> bool:
    return getattr(resp, "finish_reason", "") == "length"


def extract_window(window: Window, *, llm: Any, cfg: Any) -> ExtractResult:
    """N3 for one window. Raises LLMAuthError / BudgetExceeded; everything else → ExtractResult."""
    calls = 0
    try:
        calls += 1
        resp = _call(llm, "extract", prompts.extract_messages(window.text), cfg)
    except (LLMAuthError, BudgetExceeded):
        raise
    except LLMError as e:
        return ExtractResult("failed", error=f"llm_error: {e}", llm_calls=calls)
    if _truncated(resp):
        # A dense window produced more claims than the output budget: asking to "fix the JSON"
        # would be cut off again, so re-ask the same prompt once with a larger budget.
        log.info("extract output truncated for window %s; retrying with a larger budget",
                 window.window_id[:12])
        try:
            calls += 1
            resp = _call(llm, "extract_long", prompts.extract_messages(window.text), cfg,
                         max_tokens=int(cfg.extract_long_max_tokens))
        except (LLMAuthError, BudgetExceeded):
            raise
        except LLMError as e:
            return ExtractResult("failed", error=f"truncated; long retry llm_error: {e}", llm_calls=calls)
        if _truncated(resp):
            return ExtractResult("failed", error="truncated", llm_calls=calls,
                                 raw=(resp.text or "")[:RAW_KEEP])
    claims, rejects, err = validate_payload(resp.data, window_id=window.window_id)
    if err is None:
        return ExtractResult("ok", claims, rejects, llm_calls=calls)

    log.info("extract payload error (%s) for window %s; retrying once", err, window.window_id[:12])
    try:
        calls += 1
        resp2 = _call(llm, "extract_retry",
                      prompts.extract_retry_messages(window.text, resp.text or ""), cfg)
    except (LLMAuthError, BudgetExceeded):
        raise
    except LLMError as e:
        return ExtractResult("failed", error=f"schema: {err}; retry llm_error: {e}",
                             llm_calls=calls, raw=(resp.text or "")[:RAW_KEEP])
    claims, rejects, err2 = validate_payload(resp2.data, window_id=window.window_id)
    if err2 is None:
        return ExtractResult("ok", claims, rejects, llm_calls=calls)
    return ExtractResult("failed", error=f"schema: {err}; retry: {err2}", llm_calls=calls,
                         raw=(resp2.text or "")[:RAW_KEEP])


__all__ = ["ExtractResult", "validate_payload", "extract_window"]
