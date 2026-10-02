"""Single time source. ``now()`` = --now override > $HERMESYUME_NOW > time.time().

All persisted times are epoch seconds (float, UTC) in Python; Lance stores timestamp(ms, UTC).
Human-facing times are KST. An override is resolved once and then frozen for the process so a
run is deterministic.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9), "KST")
DAY = 86400.0
HOUR = 3600.0
_WEEKDAY_KO = "월화수목금토일"

_override: float | None = None
_env_cache: tuple[str, float] | None = None

_REL_RE = re.compile(r"^([+-])\s*(\d+(?:\.\d+)?)\s*([smhdwy])$", re.I)
_UNIT = {"s": 1.0, "m": 60.0, "h": HOUR, "d": DAY, "w": 7 * DAY, "y": 365 * DAY}


def real_now() -> float:
    return time.time()


def parse_now_spec(spec: str | float | int, base: float | None = None) -> float:
    """``+Nd``/``-Nh``/``+5y`` (s,m,h,d,w,y; relative to `base` or real now), ISO date/datetime
    (naive → KST), or epoch seconds. Raises ValueError."""
    if isinstance(spec, (int, float)):
        return float(spec)
    s = str(spec).strip()
    if not s:
        raise ValueError("empty --now spec")
    m = _REL_RE.match(s)
    if m:
        sign = 1.0 if m.group(1) == "+" else -1.0
        return (real_now() if base is None else base) + sign * float(m.group(2)) * _UNIT[m.group(3).lower()]
    if re.fullmatch(r"\d{9,}(\.\d+)?", s):
        return float(s)
    ts = parse_iso(s)
    if ts is None:
        raise ValueError(f"bad --now spec: {spec!r}")
    return ts


def set_now(spec: str | float | int | None) -> None:
    """CLI --now. None clears the override."""
    global _override
    _override = None if spec is None else parse_now_spec(spec)


def override_active() -> bool:
    return _override is not None or bool(os.environ.get("HERMESYUME_NOW", "").strip())


def now() -> float:
    global _env_cache
    if _override is not None:
        return _override
    raw = os.environ.get("HERMESYUME_NOW", "").strip()
    if raw:
        if _env_cache is None or _env_cache[0] != raw:
            _env_cache = (raw, parse_now_spec(raw))
        return _env_cache[1]
    return real_now()


# ── conversions ──

def to_ms(ts: float | None) -> int | None:
    return None if ts is None else int(round(ts * 1000.0))


def from_ms(ms: int | None) -> float | None:
    return None if ms is None else ms / 1000.0


def to_datetime(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def from_datetime(dt: datetime | None) -> float | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def kst(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=KST)


def fmt_kst(ts: float, fmt: str = "%Y-%m-%d %H:%M") -> str:
    return kst(ts).strftime(fmt)


def kst_date(ts: float) -> str:
    return fmt_kst(ts, "%Y-%m-%d")


def kst_weekday_ko(ts: float) -> str:
    return _WEEKDAY_KO[kst(ts).weekday()]


def kst_date_ko(ts: float) -> str:
    """'2026-10-02(목)' — window header 정리 기준일 format."""
    return f"{kst_date(ts)}({kst_weekday_ko(ts)})"


def kst_stamp(ts: float) -> str:
    """'YYYY-MM-DD_HHMMSS' — dream-log file names (second resolution)."""
    return fmt_kst(ts, "%Y-%m-%d_%H%M%S")


def parse_iso(s: str | None, *, end_of_day: bool = False) -> float | None:
    """'YYYY-MM-DD', 'YYYY-MM-DDTHH:MM[:SS]', with optional offset/Z. Naive → KST.
    `end_of_day` maps a bare date to 23:59:59 KST (used for valid_until). None/garbage → None."""
    if not s:
        return None
    s = str(s).strip()
    if s.lower() in ("null", "none", ""):
        return None
    date_only = re.fullmatch(r"\d{4}-\d{2}-\d{2}", s) is not None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00").replace(" ", "T", 1))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    if date_only and end_of_day:
        dt = dt + timedelta(days=1) - timedelta(seconds=1)
    return dt.timestamp()


def kst_midnight(ts: float) -> float:
    d = kst(ts)
    return d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def days(n: float) -> float:
    return n * DAY
