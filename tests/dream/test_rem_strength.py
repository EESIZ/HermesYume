"""strength.py — T1 (pure, idempotent, §5.4 lifetime table ±1 day, 0.05 event → dormant),
T2 (durable/pinned never dormant, slow HL 180), U2 decaying protected kinds, transitions."""

import math

import pytest

from hermesyume import strength as S
from hermesyume.config import Config
from hermesyume.types import KIND_BASE, MemoryRow

DAY = 86400.0
T0 = 1_790_000_000.0
CFG = Config()


def mk(kind="fact", *, importance=None, source="dream", uec=1, usc=1, explicit=False, **kw):
    return MemoryRow(id=kw.pop("id", f"id-{kind}"), text="t", kind=kind,
                     importance=KIND_BASE.get(kind, 0.5) if importance is None else importance,
                     source=source, user_evidence_count=uec, user_session_count=usc,
                     explicit_user=explicit, status=kw.pop("status", "active"), created_at=T0,
                     status_changed_at=T0, **kw)


def dormant_day(row, limit=2000):
    """First whole day (from T0) at which R6 makes the row dormant, re-evaluating daily."""
    for d in range(limit + 1):
        tr = S.transition(row, T0 + d * DAY, CFG)
        if tr is not None and tr.to_status == "dormant":
            return d
    return None


@pytest.mark.parametrize("kind,importance,expected", [
    ("opinion", 0.35, 38), ("event", 0.40, 60), ("fact", 0.50, 139), ("project", 0.60, 155),
    ("decision", 0.65, 324), ("lesson", 0.65, 324),
])
def test_t1_lifetime_table(kind, importance, expected):
    r = mk(kind, importance=importance)
    assert S.compute_tier(r) == "decaying"
    d = dormant_day(r)
    assert d is not None and abs(d - expected) <= 1, (kind, d)


def test_t1_slow_reference_procedure_505():
    for kind in ("reference", "procedure"):
        r = mk(kind, importance=0.70, uec=0, usc=0, source="md")
        assert S.compute_tier(r) == "slow"
        assert abs(dormant_day(r) - 505) <= 1


def test_t1_project_used_at_50_days_lives_to_259():
    r = mk("project", importance=0.60, recall_used_count=1, last_used_at=T0 + 50 * DAY)
    assert abs(dormant_day(r) - 259) <= 1


def test_t1_pure_and_idempotent():
    r = mk("fact", importance=0.5)
    now = T0 + 77 * DAY
    a = S.evaluate(r, now, CFG)
    b = S.evaluate(r, now, CFG)
    assert a == b
    # 100 nightly evaluations vs one evaluation at +100d: same strength and same final status
    status_daily = "active"
    for d in range(1, 101):
        tr = S.transition(r, T0 + d * DAY, CFG)
        if tr is not None:
            status_daily = tr.to_status
            break
    once = S.transition(r, T0 + 100 * DAY, CFG)
    assert status_daily == "active" and once is None
    assert S.strength(r, T0 + 100 * DAY) == pytest.approx(0.5 * 2 ** (-100 / 60))
    # same at day 150 both ways (dormant), and evaluating twice changes nothing
    assert S.transition(r, T0 + 150 * DAY, CFG).to_status == "dormant"
    assert S.transition(r, T0 + 150 * DAY, CFG) == S.transition(r, T0 + 150 * DAY, CFG)


def test_t1_tiny_event_becomes_dormant():
    r = mk("event", importance=0.05)
    assert dormant_day(r) == 21           # s < 0.10 from the start; dormant after dormant_min_days


def test_t2_durable_and_pinned_never_dormant():
    durable = mk("rule", importance=0.97, explicit=True)
    assert S.compute_tier(durable) == "durable"
    pinned = mk("fact", importance=0.5, pinned=True)
    assert S.compute_tier(pinned) == "pinned"
    core = mk("profile", importance=0.85, source="core:user", uec=0, usc=0)
    assert S.compute_tier(core) == "durable"
    remembered = mk("reference", importance=0.8, source="tool:yume_remember", uec=0, usc=0)
    assert S.compute_tier(remembered) == "durable"
    two_sessions = mk("preference", usc=2)
    assert S.compute_tier(two_sessions) == "durable"
    for r in (durable, pinned, core, remembered, two_sessions):
        assert S.transition(r, T0 + 1000 * DAY, CFG) is None
        assert S.half_life_days(r, S.compute_tier(r)) is None
    assert S.strength(pinned, T0 + 1000 * DAY) == pytest.approx(0.9)
    assert S.strength(durable, T0 + 1000 * DAY) == pytest.approx(0.97)


def test_t2_slow_hl_180():
    r = mk("rule", importance=0.85, uec=0, usc=0, source="md",   # agent_log-only rule
           source_message_ids=["l:abcdef0123:30"])
    assert S.compute_tier(r) == "slow"
    # md assistant-only (no agent_log key) → decaying (U2)
    assert S.compute_tier(mk("rule", uec=0, usc=0, source="md", source_message_ids=["m:abcdef0123:31"])) == "decaying"
    assert S.half_life_days(r, "slow") == pytest.approx(180.0)
    d = dormant_day(r)
    assert abs(d - math.ceil(180 * math.log2(0.85 / 0.10))) <= 1


def test_u2_assistant_only_dream_rule_is_decaying_hl60():
    r = mk("rule", importance=0.75, uec=0, usc=0, source="dream")
    assert S.compute_tier(r) == "decaying"
    assert S.half_life_days(r, "decaying") == pytest.approx(60.0)
    assert abs(dormant_day(r) - math.ceil(60 * math.log2(0.75 / 0.10))) <= 1


def test_legacy_zero_and_no_auto_transition():
    r = mk("legacy", importance=0.2, status="dormant")
    assert S.compute_tier(r) == "legacy" and S.strength(r, T0) == 0.0
    assert S.transition(r, T0 + 9999 * DAY, CFG) is None


def test_injected_strong_and_sessions_extend_half_life():
    r = mk("fact", recall_injected_strong=4, usc=3)
    assert S.reinforcement(r) == pytest.approx(1 + 2)
    assert S.half_life_days(r, "decaying") == pytest.approx(60 * (1 + 0.5 * math.log(4)))
    # injection never moves t_ref
    r2 = mk("fact", last_recalled_at=T0 + 30 * DAY)
    assert S.t_ref(r2) == T0


def test_expiring_transitions():
    st = mk("state", importance=0.55, event_time=T0, valid_until=T0 + 14 * DAY)
    assert S.compute_tier(st) == "expiring"
    assert S.transition(st, T0 + 14 * DAY, CFG) is None
    assert S.transition(st, T0 + 14 * DAY + 1, CFG).to_status == "expired"
    sch = mk("schedule", event_time=T0, valid_until=T0 + 5 * DAY)
    assert S.transition(sch, T0 + 7 * DAY, CFG) is None               # 2-day grace
    assert S.transition(sch, T0 + 7 * DAY + 1, CFG).to_status == "expired"
    no_vu = mk("state", event_time=T0)
    assert S.transition(no_vu, T0 + 14 * DAY + 1, CFG).to_status == "expired"


def test_forgotten_and_quarantined_purge():
    f = mk("fact", status="forgotten")
    assert S.transition(f, T0 + 29 * DAY, CFG) is None
    assert S.transition(f, T0 + 30 * DAY, CFG).to_status == "purge"
    q = mk("fact", status="quarantined")
    assert S.transition(q, T0, CFG).to_status == "purge"
    for status in ("superseded", "expired", "dormant", "candidate"):
        assert S.transition(mk("fact", status=status), T0 + 5000 * DAY, CFG) is None
