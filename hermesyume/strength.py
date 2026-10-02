"""Tier, strength and time transitions (PLAN-v2 §5.2–5.4, U2). Pure functions: no I/O, no clock,
no randomness — the same row and `now` always give the same answer (T1), so catch-up runs and
double runs never change the outcome. Strength is never stored-and-decremented; it is recomputed.

    t_ref = max(created_at, last_user_evidence_at, last_used_at)
    r     = recall_used_count + search_hit_count + 0.25·recall_injected_strong + max(0, usc − 1)
    hl    = HL_base[kind | tier] · (1 + 0.5·ln(1 + r))
    s     = importance · 2^(−(now − t_ref)/hl)
    pinned → max(importance, 0.9) (no decay)   durable → importance (no decay)   legacy → 0
"""

from __future__ import annotations

import math
from typing import Any

from .types import (DURABLE_SOURCES, EXPIRING_KINDS, KIND_HL_DAYS, LEGACY_KIND, PROTECTED_KINDS,
                    SLOW_HL_DAYS, USER_EVIDENCE_REQUIRED_KINDS, MemoryRow, StrengthResult,
                    Transition)

DAY = 86400.0
PINNED_FLOOR = 0.9
DECAYING_TIERS = frozenset({"slow", "decaying"})
_DEFAULT_HL = 60.0


def _cfg(cfg: Any, key: str, default: float) -> float:
    try:
        return float(cfg[key])
    except (KeyError, TypeError):
        return float(getattr(cfg, key, default))


# Evidence keys that mark agent_log text in a dream/md row without user evidence: md agent_log
# lines ("l:"), MEMORY.md legacy entries (M3, "x:memory_md:") and episodic core_add inbox items
# ("i:" — remember/core items carry it too, but those rows have user evidence / core sources).
AGENT_LOG_KEY_PREFIXES = ("l:", "x:memory_md:", "i:")


def has_agent_log_evidence(row: MemoryRow) -> bool:
    return any(str(k).startswith(AGENT_LOG_KEY_PREFIXES) for k in (row.source_message_ids or ()))


def compute_tier(row: MemoryRow) -> str:
    """§5.2 with U2 (D14), evaluated in order. "Assistant-only" = an extracted (dream/md) row
    with no user evidence and no agent_log evidence; agent_log evidence keeps protected kinds slow."""
    if row.pinned:
        return "pinned"
    if row.kind == LEGACY_KIND:
        return "legacy"
    protected = row.kind in PROTECTED_KINDS
    if protected and (row.source in DURABLE_SOURCES
                      or (row.user_evidence_count >= 1
                          and (row.explicit_user or row.user_session_count >= 2))):
        return "durable"
    if (row.kind in USER_EVIDENCE_REQUIRED_KINDS and row.user_evidence_count == 0
            and row.source in ("dream", "md") and not has_agent_log_evidence(row)):
        return "decaying"
    if protected:
        return "slow"
    if row.kind in EXPIRING_KINDS:
        return "expiring"
    return "decaying"


def t_ref(row: MemoryRow) -> float:
    return max(float(row.created_at or 0.0), float(row.last_user_evidence_at or 0.0),
               float(row.last_used_at or 0.0))


def reinforcement(row: MemoryRow) -> float:
    return (float(row.recall_used_count or 0) + float(row.search_hit_count or 0)
            + 0.25 * float(row.recall_injected_strong or 0)
            + max(0, int(row.user_session_count or 0) - 1))


def base_half_life_days(kind: str, tier: str) -> float | None:
    """HL before the spacing effect. None = no decay (pinned/durable) or legacy (strength 0)."""
    if tier in ("pinned", "durable", "legacy"):
        return None
    if tier == "slow":
        return SLOW_HL_DAYS
    return float(KIND_HL_DAYS.get(kind, _DEFAULT_HL))


def half_life_days(row: MemoryRow, tier: str) -> float | None:
    """Effective HL in days incl. the spacing effect (1 + 0.5·ln(1 + r)); None = no decay."""
    base = base_half_life_days(row.kind, tier)
    if base is None:
        return None
    return base * (1.0 + 0.5 * math.log1p(reinforcement(row)))


def strength(row: MemoryRow, now: float, *, tier: str | None = None) -> float:
    tier = tier or compute_tier(row)
    imp = float(row.importance or 0.0)
    if tier == "legacy":
        return 0.0
    if tier == "pinned":
        return max(imp, PINNED_FLOOR)
    if tier == "durable":
        return max(imp, 0.5 * imp)
    hl = half_life_days(row, tier)
    if hl is None or hl <= 0:
        return imp
    age_days = max(0.0, (float(now) - t_ref(row)) / DAY)
    return imp * math.pow(2.0, -age_days / hl)


def _deadline(row: MemoryRow, cfg: Any) -> float:
    if row.valid_until is not None:
        return float(row.valid_until)
    start = row.event_time if row.event_time is not None else row.created_at
    return float(start or 0.0) + _cfg(cfg, "state_default_ttl_days", 14) * DAY


def transition(row: MemoryRow, now: float, cfg: Any, *, tier: str | None = None,
               s: float | None = None) -> Transition | None:
    """§5.4 automatic transitions. Revival (dormant → active) is never automatic here."""
    tier = tier or compute_tier(row)
    if row.status == "active":
        if row.in_core and row.kind != LEGACY_KIND:
            # a core-file copy is in every session's system prompt: it is in use, so it neither
            # decays nor expires while there; demotion restarts its clock (DEVIATIONS F-15)
            return None
        if tier == "expiring":
            grace = _cfg(cfg, "schedule_grace_days", 2) if row.kind == "schedule" else 0.0
            if now > _deadline(row, cfg) + grace * DAY:
                return Transition(row.id, "active", "expired", "expired:valid_until")
            return None
        if tier in DECAYING_TIERS:
            s = strength(row, now, tier=tier) if s is None else s
            if (s < _cfg(cfg, "dormant_strength", 0.10)
                    and now - t_ref(row) >= _cfg(cfg, "dormant_min_days", 21) * DAY):
                return Transition(row.id, "active", "dormant", "dormant:strength")
        return None
    if row.status == "forgotten":
        if now - float(row.status_changed_at or 0.0) >= _cfg(cfg, "forget_purge_days", 30) * DAY:
            return Transition(row.id, "forgotten", "purge", "purge:forgotten")
        return None
    if row.status == "quarantined":
        return Transition(row.id, "quarantined", "purge", "purge:quarantined")
    return None


def evaluate(row: MemoryRow, now: float, cfg: Any) -> StrengthResult:
    tier = compute_tier(row)
    s = strength(row, now, tier=tier)
    return StrengthResult(tier=tier, strength=s, t_ref=t_ref(row), hl_days=half_life_days(row, tier),
                          transition=transition(row, now, cfg, tier=tier, s=s))
