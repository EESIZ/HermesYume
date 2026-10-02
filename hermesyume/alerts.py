"""Operator alerts (PLAN-v2 §10.6 + user principle U4; v1 alerts.py ADAPT).

- Only ``types.ALERT_CODES`` are alerts (system failures). Everything else is a Dream Log note
  (``ctx.note``): new pins, core_required removals, pin budget, guard holds, backlog, failure
  rates, row counts.
- Default output: one JSON line per alert in ``<data_dir>/alerts.log`` (0600). The agent never
  sees it and it never enters state.db.
- Telegram is opt-in (``alert_telegram``). The dream process or ``yume alert-flush`` sends the
  unsent lines as ONE caption-less ``.txt`` through Bot API ``sendDocument`` — never
  ``sendMessage``, no caption, no ``parse_mode``, neutral file name. The gateway/provider never
  calls Telegram.
- The bot token and chat id are read by name from ``$HERMES_HOME/.env``; neither they nor the
  request URL are ever logged (v1 M6).
"""

from __future__ import annotations

import fcntl
import html
import json
import logging
import math
import os
import re
import urllib.request
from typing import Any, Callable, Iterable

from . import clock, secrets_env
from .types import ALERT_CODES, Alert

log = logging.getLogger("hermesyume.alerts")

PREFIX = "[HermesYume 운영]"
CURSOR_FILE = "alerts.sent"           # <data_dir>/alerts.sent: byte offset of alerts.log already sent
TELEGRAM_API = "https://api.telegram.org"
DAY = 86400.0

ALERT_TITLES: dict[str, str] = {
    "run_failed": "야간 정리 실행 실패",
    "auth_401": "OpenAI 인증 실패(401/403)",
    "model_mismatch": "임베딩 모델·차원 또는 스키마 불일치",
    "scanner_unavailable": "위협·비밀값 검사기 로드 실패",
    "stalled": "처리 정체",
    "window_quarantined": "추출 창 격리",
    "secret_found": "비밀값 발견",
    "recall_embed_fail_rate": "회상 임베딩 실패율 높음",
    "serving_stale": "서빙 사본이 오래됨",
    "prefetch_p95": "회상 지연(p95) 높음",
}
assert set(ALERT_TITLES) == set(ALERT_CODES), "ALERT_TITLES must cover exactly types.ALERT_CODES"

LEVEL_KO = {"info": "정보", "warn": "경고", "error": "오류"}

# Telegram bot URLs carry the token in the path; strip them from anything we write.
_BOT_URL_RE = re.compile(r"(?i)(https?://)?api\.telegram\.org/(file/)?bot[^\s/]*")
_BOT_TOKEN_RE = re.compile(r"\bbot\d{5,}:[A-Za-z0-9_-]{10,}")


# ── scrubbing ────────────────────────────────────────────────────────────────

def scrub(text: Any) -> str:
    """Remove secrets (threat.redact_secrets) and Telegram bot URLs/tokens from free text."""
    s = "" if text is None else str(text)
    s = _BOT_URL_RE.sub("api.telegram.org/bot<redacted>", s)
    s = _BOT_TOKEN_RE.sub("bot<redacted>", s)
    try:
        from .threat import redact_secrets
        s = redact_secrets(s)[0]
    except Exception:  # noqa: BLE001 — scrubbing must never fail an alert
        pass
    return s


def _scrub_obj(obj: Any, depth: int = 0) -> Any:
    if depth > 6:
        return scrub(obj)
    if isinstance(obj, str):
        return scrub(obj)
    if isinstance(obj, dict):
        return {str(k): _scrub_obj(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_scrub_obj(v, depth + 1) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    return scrub(obj)


def _check_code(code: str) -> None:
    if code not in ALERT_CODES:
        raise ValueError(f"not an alert condition (U4): {code!r}; use ctx.note()")


def make_alert(code: str, message: str, *, level: str = "warn", run_id: str | None = None,
               ts: float | None = None, **details: Any) -> Alert:
    """Alert object for a code in ALERT_CODES (refuses others, U4)."""
    _check_code(code)
    return Alert(code=code, message=message, level=level, run_id=run_id,
                 ts=float(clock.now() if ts is None else ts), details=details)


# ── alerts.log ───────────────────────────────────────────────────────────────

def alert_record(alert: Alert) -> dict[str, Any]:
    ts = float(alert.ts or clock.now())
    return {"ts": ts, "kst": clock.fmt_kst(ts, "%Y-%m-%d %H:%M:%S"), "code": alert.code,
            "level": alert.level, "run_id": alert.run_id, "message": scrub(alert.message),
            "details": _scrub_obj(alert.details or {})}


def alert_from_record(d: dict[str, Any]) -> Alert | None:
    code = d.get("code")
    if code not in ALERT_CODES:
        return None
    try:
        ts = float(d.get("ts") or 0.0)
    except (TypeError, ValueError):
        ts = 0.0
    return Alert(code=code, message=str(d.get("message") or ""), level=str(d.get("level") or "warn"),
                 run_id=d.get("run_id"), ts=ts, details=d.get("details") or {})


class AlertSink:
    """Appends alerts to ``alerts.log`` (one JSON line each, 0600). ``dry_run`` → writes nothing."""

    def __init__(self, paths: Any, cfg: Any, *, dry_run: bool = False):
        self.paths = paths
        self.cfg = cfg
        self.dry_run = dry_run
        self.emitted: list[Alert] = []

    def emit(self, alert: Alert) -> None:
        _check_code(alert.code)
        self.emitted.append(alert)
        title = ALERT_TITLES.get(alert.code, alert.code)
        log.warning("운영 알림: %s (%s)", title, alert.code)
        if self.dry_run:
            return
        line = json.dumps(alert_record(alert), ensure_ascii=False, sort_keys=True, default=str) + "\n"
        self.paths.ensure_dir(self.paths.data_dir)
        fd = os.open(self.paths.alerts_log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)

    def emit_all(self, alerts: Iterable[Alert]) -> None:
        seen: set[int] = set()
        for a in alerts or []:
            if id(a) in seen:
                continue
            seen.add(id(a))
            self.emit(a)


def read_alerts(paths: Any, *, since_offset: int = 0) -> tuple[list[Alert], int]:
    """Parse complete lines of alerts.log from `since_offset`. Returns (alerts, end offset of the
    last complete line). Unparseable lines are skipped (and passed)."""
    p = paths.alerts_log
    if not p.exists():
        return [], 0
    with open(p, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        start = since_offset if 0 <= since_offset <= size else 0
        f.seek(start)
        data = f.read(size - start)
    cut = data.rfind(b"\n")
    if cut < 0:
        return [], start
    out: list[Alert] = []
    for raw in data[:cut + 1].splitlines():
        try:
            a = alert_from_record(json.loads(raw.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            continue
        if a is not None:
            out.append(a)
    return out, start + cut + 1


# ── Telegram .txt document (U4) ──────────────────────────────────────────────

def render_alert_text(alerts: list[Alert]) -> str:
    """Plain Korean body of the .txt: one block per alert (KST time, title, message). Free text is
    HTML-escaped and scrubbed (secrets, bot URLs)."""
    lines = [f"{PREFIX} 운영 알림 {len(alerts)}건", ""]
    for a in alerts:
        title = ALERT_TITLES.get(a.code, a.code)
        when = clock.fmt_kst(float(a.ts or 0.0), "%Y-%m-%d %H:%M KST")
        lvl = LEVEL_KO.get(a.level, a.level)
        lines.append(f"■ {when} · {html.escape(title, quote=False)} · {html.escape(lvl, quote=False)}")
        msg = html.escape(scrub(a.message), quote=False).strip()
        if msg:
            lines.append(msg)
        if a.run_id:
            lines.append(f"실행: {html.escape(scrub(a.run_id), quote=False)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def document_filename(now: float) -> str:
    """Neutral file name: 'yume-ops-YYYYMMDD-HHMM.txt' (KST)."""
    return f"yume-ops-{clock.fmt_kst(now, '%Y%m%d-%H%M')}.txt"


def build_send_document(text: str, *, chat_id: str, now: float
                        ) -> tuple[dict[str, str], tuple[str, str, bytes]]:
    """(form fields == {"chat_id"} ONLY, ("document", filename, utf-8 bytes)). No caption, no
    text, no parse_mode — a caption-less document yields no reply-quote injection (U4)."""
    return {"chat_id": str(chat_id)}, ("document", document_filename(now), text.encode("utf-8"))


def encode_multipart(fields: dict[str, str], file: tuple[str, str, bytes], *,
                     boundary: str | None = None) -> tuple[bytes, str]:
    """multipart/form-data body. Returns (body, content type)."""
    b = boundary or ("yume" + os.urandom(12).hex())
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n".encode("utf-8")
                     + str(v).encode("utf-8") + b"\r\n")
    name, filename, data = file
    parts.append(f"--{b}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                 f"filename=\"{filename}\"\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n"
                 .encode("utf-8") + data + b"\r\n")
    parts.append(f"--{b}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={b}"


def send_document(text: str, *, token: str, chat_id: str, now: float, timeout: float = 10,
                  api_base: str = TELEGRAM_API,
                  opener: Callable[..., Any] | None = None) -> bool:
    """POST sendDocument. On any failure log only the exception class (never URL/token/chat id)."""
    fields, file = build_send_document(text, chat_id=chat_id, now=now)
    body, ctype = encode_multipart(fields, file)
    url = f"{api_base.rstrip('/')}/bot{token}/sendDocument"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype}, method="POST")
    try:
        resp = (opener or urllib.request.urlopen)(req, timeout=timeout)
        try:
            raw = resp.read()
            status = getattr(resp, "status", None) or getattr(resp, "code", None) or 200
        finally:
            close = getattr(resp, "close", None)
            if close:
                close()
        try:
            ok = bool(json.loads(raw.decode("utf-8")).get("ok")) if raw else False
        except (ValueError, UnicodeDecodeError, AttributeError):
            ok = False
        if int(status) != 200 or not ok:
            log.warning("텔레그램 알림 전송 실패 (응답 status=%s)", int(status))
            return False
        return True
    except Exception as e:  # noqa: BLE001 — never let the URL (token) reach a log line
        log.warning("텔레그램 알림 전송 실패 (%s)", type(e).__name__)
        return False


def _read_cursor(fd: int) -> int:
    try:
        raw = os.pread(fd, 64, 0).decode("ascii", "ignore").strip()
        return int(raw) if raw else 0
    except (OSError, ValueError):
        return 0


def _write_cursor(fd: int, offset: int) -> None:
    os.ftruncate(fd, 0)
    os.pwrite(fd, f"{int(offset)}\n".encode("ascii"), 0)
    os.fsync(fd)


def flush_pending(paths: Any, cfg: Any, *, now: float, sender: Callable[..., bool] | None = None) -> int:
    """`yume alert-flush` and the end of every live dream: when `alert_telegram` is on and the bot
    token + chat id resolve, send the unsent alerts.log lines as ONE .txt and advance
    alerts.sent. Returns the number of alerts sent (0 when disabled/nothing/failed)."""
    if not bool(_cfg(cfg, "alert_telegram", False)):
        return 0
    if not paths.alerts_log.exists():
        return 0
    token = secrets_env.get_secret(secrets_env.TELEGRAM_BOT_TOKEN, paths)
    chat_id = secrets_env.alert_chat_id(paths)
    if not token or not chat_id:
        log.warning("alert_telegram이 켜져 있지만 봇 토큰 또는 알림 chat id가 없습니다 (토큰 %s)",
                    secrets_env.mask(token))
        return 0
    paths.ensure_dir(paths.data_dir)
    fd = os.open(paths.data_dir / CURSOR_FILE, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        offset = _read_cursor(fd)
        alerts, end = read_alerts(paths, since_offset=offset)
        if not alerts:
            if end != offset:
                _write_cursor(fd, end)      # only unparseable lines: skip them
            return 0
        text = render_alert_text(alerts)
        send = sender or send_document
        ok = send(text, token=token, chat_id=chat_id, now=now)
        if not ok:
            return 0
        _write_cursor(fd, end)
        return len(alerts)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# ── run-side evaluation (U4) ─────────────────────────────────────────────────

def _cfg(cfg: Any, key: str, default: Any) -> Any:
    try:
        return cfg[key]
    except (KeyError, TypeError):
        return getattr(cfg, key, default)


def _stats_dict(stats: Any) -> dict[str, Any]:
    if stats is None:
        return {}
    if isinstance(stats, dict):
        return stats
    to_dict = getattr(stats, "to_dict", None)
    return to_dict() if callable(to_dict) else dict(vars(stats))


ADMIN_RUN_SUFFIXES = ("-admin", "-debug", "-export", "-restore")


def is_stalled(stats: dict[str, Any]) -> bool:
    """A night with windows to process but none processed (ok/empty/quarantined all zero)."""
    pending = int(stats.get("windows_total") or 0)
    done = sum(int(stats.get(k) or 0) for k in ("windows_ok", "windows_empty", "windows_quarantined"))
    return pending > 0 and done == 0


def stall_streak(ctx: Any, *, limit: int) -> int:
    """Consecutive stalled nights ending with this run (this run's stats + earlier committed/held
    live runs from the ledger, newest first)."""
    if not is_stalled(_stats_dict(ctx.stats)):
        return 0
    streak = 1
    led = getattr(ctx, "ledger", None)
    if led is None or streak >= limit:
        return streak
    try:
        runs = led.runs(limit=max(10, limit * 4))
    except Exception:  # noqa: BLE001
        return streak
    for rec in runs:
        if rec.run_id == ctx.run_id or rec.mode != "live" or rec.status not in ("committed", "held"):
            continue
        if rec.run_id.endswith(ADMIN_RUN_SUFFIXES):     # `yume unpin/forget/debug/export` commits, not nights
            continue
        if not rec.stats_json:
            continue
        try:
            d = json.loads(rec.stats_json)
        except ValueError:
            continue
        if not is_stalled(d):
            break
        streak += 1
        if streak >= limit:
            break
    return streak


def evaluate_run_alerts(ctx: Any, plan: Any) -> list[Alert]:
    """U4 run-side checks. Only `stalled` is an alert (window_quarantined is raised by nrem);
    backlog, failure rates and row counts become Dream Log notes."""
    cfg, st = ctx.cfg, ctx.stats
    out: list[Alert] = []
    nights = max(1, int(_cfg(cfg, "stall_alert_nights", 2)))
    if not getattr(ctx, "dry_run", False) and getattr(ctx, "mode", "live") == "live":
        streak = stall_streak(ctx, limit=nights)
        if streak >= nights:
            out.append(make_alert(
                "stalled",
                f"처리할 창이 있는데 {streak}밤 연속으로 하나도 처리하지 못했습니다 "
                f"(이번 실행: 전체 {st.windows_total}, 실패 {st.windows_failed}, 미룸 {st.windows_deferred}). "
                "추출 LLM·예산·입력 상태를 확인하세요.",
                level="error", run_id=ctx.run_id, ts=ctx.now, nights=streak))

    per_night = max(1, int(_cfg(cfg, "max_windows_per_run", 60)))
    deferred = int(st.windows_deferred or 0)
    backlog_nights = math.ceil(deferred / per_night) if deferred else 0
    if backlog_nights > int(_cfg(cfg, "backlog_alert_nights", 3)):
        ctx.note(f"백로그: 미뤄진 창 {deferred}개, 하룻밤 {per_night}개 기준 약 {backlog_nights}밤 분량입니다.")

    rate = float(_cfg(cfg, "fail_rate_alert", 0.20))
    attempted = int(st.windows_ok or 0) + int(st.windows_failed or 0) + int(st.windows_quarantined or 0)
    failed = int(st.windows_failed or 0) + int(st.windows_quarantined or 0)
    if attempted and failed / attempted > rate:
        ctx.note(f"추출 실패율 {failed}/{attempted} ({failed / attempted:.0%})이 기준 {rate:.0%}를 넘었습니다.")
    jc, jf = int(st.judge_calls or 0), int(st.judge_failures or 0)
    if jc and jf / jc > rate:
        ctx.note(f"판정 실패율 {jf}/{jc} ({jf / jc:.0%})이 기준 {rate:.0%}를 넘었습니다.")

    store = getattr(ctx, "store", None)
    if store is not None:
        try:
            active = int(store.count("status = 'active'"))
            total = int(store.count())
        except Exception:  # noqa: BLE001 — informational only
            active = total = -1
        if active > int(_cfg(cfg, "active_rows_alert", 5000)):
            ctx.note(f"활성 행 {active}개가 {_cfg(cfg, 'active_rows_alert', 5000)}개를 넘었습니다 "
                     "(상시 회상 데몬 방식 검토, §8.1 B).")
        if total > int(_cfg(cfg, "total_rows_alert", 50000)):
            ctx.note(f"전체 행 {total}개가 {_cfg(cfg, 'total_rows_alert', 50000)}개를 넘었습니다.")
    return out


def health_summary(ctx: Any, *, now: float) -> dict[str, Any]:
    """Last-24h provider health aggregate (Dream Log + evaluate_health)."""
    live = getattr(ctx, "live", None)
    rows = []
    if live is not None:
        try:
            rows = live.health_since(float(now) - DAY)
        except Exception:  # noqa: BLE001
            rows = []
    agg = {"rows": len(rows), "prefetch": 0, "injected": 0, "empty": 0, "embed_fail": 0,
           "fts_fallback": 0, "timeout": 0, "p95_max": None, "last_error_class": None}
    for r in rows:
        agg["prefetch"] += int(r.prefetch_n or 0)
        agg["injected"] += int(r.injected_n or 0)
        agg["empty"] += int(r.empty_n or 0)
        agg["embed_fail"] += int(r.embed_fail_n or 0)
        agg["fts_fallback"] += int(r.fts_fallback_n or 0)
        agg["timeout"] += int(r.timeout_n or 0)
        if r.p95_ms is not None:
            agg["p95_max"] = max(int(r.p95_ms), agg["p95_max"] or 0)
        if r.last_error_class:
            agg["last_error_class"] = r.last_error_class
    agg["embed_fail_rate"] = (agg["embed_fail"] / agg["prefetch"]) if agg["prefetch"] else None
    return agg


def evaluate_health(ctx: Any, *, now: float) -> list[Alert]:
    """Recall health (last 24h of live.db health rows) and serving-copy age."""
    cfg = ctx.cfg
    out: list[Alert] = []
    h = health_summary(ctx, now=now)
    thr = float(_cfg(cfg, "health_embed_fail_alert", 0.20))
    rate = h["embed_fail_rate"]
    if rate is not None and rate >= thr:
        out.append(make_alert(
            "recall_embed_fail_rate",
            f"최근 24시간 회상 임베딩 실패 {h['embed_fail']}/{h['prefetch']} ({rate:.0%}). "
            f"키워드(FTS) 대체 {h['fts_fallback']}회, 시간 초과 {h['timeout']}회"
            + (f", 마지막 오류 {h['last_error_class']}" if h["last_error_class"] else "") + ".",
            run_id=ctx.run_id, ts=now, prefetch=h["prefetch"], embed_fail=h["embed_fail"]))
    p95_thr = int(_cfg(cfg, "prefetch_p95_alert_ms", 3000))
    if h["p95_max"] is not None and h["p95_max"] >= p95_thr:
        out.append(make_alert(
            "prefetch_p95",
            f"최근 24시간 회상 p95 지연 최대 {h['p95_max']}ms (기준 {p95_thr}ms).",
            run_id=ctx.run_id, ts=now, p95_ms=h["p95_max"]))
    p = ctx.paths.recall_sqlite
    try:
        if p.exists():
            age_h = (clock.real_now() - p.stat().st_mtime) / 3600.0   # file mtime is wall-clock
            stale = float(_cfg(cfg, "serving_stale_hours", 36))
            if age_h >= stale:
                out.append(make_alert(
                    "serving_stale",
                    f"서빙 사본(serving/recall.sqlite)이 {age_h:.0f}시간 동안 갱신되지 않았습니다 "
                    f"(기준 {stale:.0f}시간). 야간 정리가 돌고 있는지 확인하세요.",
                    run_id=ctx.run_id, ts=now, age_hours=round(age_h, 1)))
    except OSError:
        pass
    return out


def cli_flush(args: Any, paths: Any) -> int:
    """Ready-made handler for `yume alert-flush` (integrator may wire it into cli.HANDLERS)."""
    from .config import load_config
    cfg = load_config(paths)
    if not bool(_cfg(cfg, "alert_telegram", False)):
        print("alert_telegram이 꺼져 있습니다. alerts.log에만 기록합니다.")
        return 0
    n = flush_pending(paths, cfg, now=clock.now())
    print(f"전송한 운영 알림: {n}건")
    return 0


__all__ = ["ALERT_TITLES", "CURSOR_FILE", "AlertSink", "make_alert", "render_alert_text",
           "document_filename", "build_send_document", "send_document", "flush_pending",
           "evaluate_run_alerts", "evaluate_health", "health_summary", "read_alerts", "scrub",
           "cli_flush"]
