"""alerts.py (PLAN §10.6, U4): only ALERT_CODES, alerts.log JSON lines, caption-less .txt via
sendDocument only, HTML escape, never a token / bot URL / chat id in any log line."""

from __future__ import annotations

import json
import logging
import os
import re
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermesyume import alerts
from hermesyume.types import ALERT_CODES, Alert, RunRecord

TOKEN = "123456789:" + "A" * 35            # matches the telegram secret regex on purpose
CHAT_ID = "987654321"


def _env(fake_home, *, token=TOKEN, chat=CHAT_ID):
    lines = [f"OPENAI_API_KEY=sk-test-{'0' * 32}"]
    if token:
        lines.append(f"TELEGRAM_BOT_TOKEN={token}")
    if chat:
        lines.append(f"YUME_ALERT_CHAT_ID={chat}")
    (fake_home.root / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _alert(code="run_failed", msg="dream 실행 실패: X", ts=None, now=None):
    return Alert(code=code, message=msg, level="error", run_id="r1", ts=ts or now or 1.0)


def _no_secrets(text: str) -> None:
    assert TOKEN not in text
    assert TOKEN.split(":")[1] not in text
    assert CHAT_ID not in text
    assert "api.telegram.org/bot1" not in text


# ── codes ────────────────────────────────────────────────────────────────────

def test_titles_cover_exactly_alert_codes():
    assert set(alerts.ALERT_TITLES) == set(ALERT_CODES)


@pytest.mark.parametrize("code", ["new_pin", "core_required", "pin_budget", "candidate", "held",
                                  "backlog"])
def test_non_alert_conditions_are_refused(code, paths, cfg):
    with pytest.raises(ValueError):
        alerts.make_alert(code, "x")
    with pytest.raises(ValueError):
        alerts.AlertSink(paths, cfg).emit(Alert(code=code, message="x"))


# ── alerts.log ───────────────────────────────────────────────────────────────

def test_sink_writes_json_lines_0600_and_scrubs(paths, cfg, now):
    sink = alerts.AlertSink(paths, cfg)
    url = f"https://api.telegram.org/bot{TOKEN}/sendDocument"
    sink.emit(alerts.make_alert("auth_401", f"키 sk-{'x' * 30} 실패, {url}", ts=now, run_id="r1",
                                url=url))
    sink.emit_all([alerts.make_alert("stalled", "정체", ts=now)])
    raw = paths.alerts_log.read_text(encoding="utf-8")
    assert stat.S_IMODE(os.stat(paths.alerts_log).st_mode) == 0o600
    lines = [json.loads(x) for x in raw.splitlines()]
    assert [x["code"] for x in lines] == ["auth_401", "stalled"]
    assert set(lines[0]) == {"ts", "kst", "code", "level", "run_id", "message", "details"}
    assert "sk-xxxx" not in raw and TOKEN not in raw and "[REDACTED:openai]" in raw
    _no_secrets(raw)


def test_sink_dry_run_writes_nothing(paths, cfg):
    alerts.AlertSink(paths, cfg, dry_run=True).emit(alerts.make_alert("run_failed", "x"))
    assert not paths.alerts_log.exists()


# ── .txt body ────────────────────────────────────────────────────────────────

def test_render_alert_text_html_escapes_and_prefixes(now):
    text = alerts.render_alert_text([_alert(msg="<b>깨짐</b> & <memory-context>", now=now)])
    assert text.startswith(alerts.PREFIX)
    assert "&lt;b&gt;깨짐&lt;/b&gt; &amp; &lt;memory-context&gt;" in text
    assert "<b>" not in text and "<memory-context>" not in text
    assert alerts.ALERT_TITLES["run_failed"] in text


def test_render_alert_text_scrubs_token_and_url(now):
    text = alerts.render_alert_text([_alert(msg=f"https://api.telegram.org/bot{TOKEN}/getMe 실패", now=now)])
    _no_secrets(text)


def test_build_send_document_payload_has_no_caption_or_text(now):
    fields, file = alerts.build_send_document("본문", chat_id=CHAT_ID, now=now)
    assert fields == {"chat_id": CHAT_ID}
    assert set(fields) == {"chat_id"}
    for k in ("caption", "text", "parse_mode"):
        assert k not in fields
    name, filename, data = file
    assert name == "document"
    assert re.fullmatch(r"yume-ops-\d{8}-\d{4}\.txt", filename)
    assert data == "본문".encode("utf-8")
    body, ctype = alerts.encode_multipart(fields, file, boundary="BB")
    assert ctype == "multipart/form-data; boundary=BB"
    names = re.findall(rb'; name="([^"]+)"', body)
    assert names == [b"chat_id", b"document"]
    assert b'filename="' + filename.encode() + b'"' in body


def test_document_filename_is_neutral_kst(now):
    assert alerts.document_filename(now) == "yume-ops-20261002-0440.txt"


# ── sendDocument over HTTP ───────────────────────────────────────────────────

class _TG:
    def __init__(self, status=200, body=b'{"ok":true}'):
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                outer.requests.append((self.path, dict(self.headers), self.rfile.read(n)))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.t = threading.Thread(target=self.srv.serve_forever, daemon=True)

    @property
    def base(self):
        return f"http://127.0.0.1:{self.srv.server_address[1]}"

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *e):
        self.srv.shutdown()
        self.srv.server_close()


def test_send_document_uses_senddocument_only(now, caplog):
    caplog.set_level(logging.DEBUG)
    with _TG() as tg:
        ok = alerts.send_document("알림 본문", token=TOKEN, chat_id=CHAT_ID, now=now, api_base=tg.base)
    assert ok
    (path, headers, body), = tg.requests
    assert path == f"/bot{TOKEN}/sendDocument"
    assert "sendMessage" not in path
    assert headers["Content-Type"].startswith("multipart/form-data")
    assert re.findall(rb'; name="([^"]+)"', body) == [b"chat_id", b"document"]
    assert b"caption" not in body and b"parse_mode" not in body
    _no_secrets(caplog.text)


def test_send_document_failure_logs_class_only(now, caplog):
    caplog.set_level(logging.DEBUG)
    with _TG(status=500, body=b'{"ok":false,"description":"chat 987654321 not found"}') as tg:
        assert not alerts.send_document("x", token=TOKEN, chat_id=CHAT_ID, now=now, api_base=tg.base)
    assert not alerts.send_document("x", token=TOKEN, chat_id=CHAT_ID, now=now,
                                    api_base="http://127.0.0.1:9", timeout=0.5)
    assert "텔레그램 알림 전송 실패" in caplog.text
    _no_secrets(caplog.text)


# ── flush_pending ────────────────────────────────────────────────────────────

def test_flush_disabled_by_default_sends_nothing(paths, cfg, fake_home, now):
    _env(fake_home)
    alerts.AlertSink(paths, cfg).emit(alerts.make_alert("run_failed", "x", ts=now))
    calls = []
    assert cfg.alert_telegram is False
    assert alerts.flush_pending(paths, cfg, now=now, sender=lambda *a, **k: calls.append(1) or True) == 0
    assert calls == []
    assert not (paths.data_dir / alerts.CURSOR_FILE).exists()


def test_flush_sends_unsent_once_and_advances_cursor(paths, cfg, fake_home, now, caplog):
    caplog.set_level(logging.DEBUG)
    _env(fake_home)
    on = cfg.replace(alert_telegram=True)
    sink = alerts.AlertSink(paths, on)
    sink.emit(alerts.make_alert("run_failed", "첫 번째", ts=now))
    sink.emit(alerts.make_alert("secret_found", "두 번째", ts=now))
    sent = []

    def sender(text, *, token, chat_id, now):
        sent.append((text, token, chat_id))
        return True

    assert alerts.flush_pending(paths, on, now=now, sender=sender) == 2
    assert len(sent) == 1 and "첫 번째" in sent[0][0] and "두 번째" in sent[0][0]
    assert sent[0][1] == TOKEN and sent[0][2] == CHAT_ID
    cur = paths.data_dir / alerts.CURSOR_FILE
    assert stat.S_IMODE(os.stat(cur).st_mode) == 0o600
    assert int(cur.read_text()) == paths.alerts_log.stat().st_size
    assert alerts.flush_pending(paths, on, now=now, sender=sender) == 0     # nothing new
    sink.emit(alerts.make_alert("stalled", "세 번째", ts=now))
    assert alerts.flush_pending(paths, on, now=now, sender=sender) == 1
    assert "세 번째" in sent[-1][0] and "첫 번째" not in sent[-1][0]
    _no_secrets(caplog.text)


def test_flush_failure_keeps_cursor(paths, cfg, fake_home, now):
    _env(fake_home)
    on = cfg.replace(alert_telegram=True)
    alerts.AlertSink(paths, on).emit(alerts.make_alert("run_failed", "x", ts=now))
    assert alerts.flush_pending(paths, on, now=now, sender=lambda *a, **k: False) == 0
    got = []
    assert alerts.flush_pending(paths, on, now=now, sender=lambda t, **k: got.append(t) or True) == 1
    assert len(got) == 1


def test_flush_without_token_logs_no_values(paths, cfg, fake_home, now, caplog):
    caplog.set_level(logging.DEBUG)
    _env(fake_home, token=None)
    on = cfg.replace(alert_telegram=True)
    alerts.AlertSink(paths, on).emit(alerts.make_alert("run_failed", "x", ts=now))
    assert alerts.flush_pending(paths, on, now=now, sender=lambda *a, **k: True) == 0
    assert "<unset>" in caplog.text
    _no_secrets(caplog.text)


def test_flush_end_to_end_over_http(paths, cfg, fake_home, now, monkeypatch):
    _env(fake_home)
    on = cfg.replace(alert_telegram=True)
    alerts.AlertSink(paths, on).emit(alerts.make_alert("auth_401", "401", ts=now))
    with _TG() as tg:
        monkeypatch.setattr(alerts, "TELEGRAM_API", tg.base)
        real = alerts.send_document
        monkeypatch.setattr(alerts, "send_document",
                            lambda text, **kw: real(text, api_base=tg.base, **kw))
        assert alerts.flush_pending(paths, on, now=now) == 1
    (path, _h, body), = tg.requests
    assert path.endswith("/sendDocument")
    assert b'name="caption"' not in body and b'name="text"' not in body


# ── run-side evaluation ──────────────────────────────────────────────────────

def _stalled_stats():
    return {"windows_total": 4, "windows_ok": 0, "windows_empty": 0, "windows_quarantined": 0,
            "windows_failed": 4, "windows_deferred": 0}


def test_stalled_needs_two_consecutive_nights(ctx):
    ctx.stats.windows_total, ctx.stats.windows_failed = 3, 3
    assert alerts.evaluate_run_alerts(ctx, None) == []          # first stalled night
    ctx.ledger.insert_run(RunRecord(run_id="prev", started_at=ctx.now - 86400, mode="live",
                                    status="committed", stats_json=json.dumps(_stalled_stats())))
    out = alerts.evaluate_run_alerts(ctx, None)
    assert [a.code for a in out] == ["stalled"]
    assert out[0].details["nights"] == 2


def test_not_stalled_when_previous_night_processed(ctx):
    ctx.stats.windows_total, ctx.stats.windows_failed = 3, 3
    ok = dict(_stalled_stats(), windows_ok=2)
    ctx.ledger.insert_run(RunRecord(run_id="prev", started_at=ctx.now - 86400, mode="live",
                                    status="committed", stats_json=json.dumps(ok)))
    assert alerts.evaluate_run_alerts(ctx, None) == []


def test_other_conditions_are_notes_not_alerts(ctx):
    st = ctx.stats
    st.windows_deferred = 60 * 5
    st.windows_ok, st.windows_failed = 1, 4
    st.judge_calls, st.judge_failures = 10, 5
    out = alerts.evaluate_run_alerts(ctx, None)
    assert out == []
    notes = " ".join(ctx.report.notes)
    assert "백로그" in notes and "추출 실패율" in notes and "판정 실패율" in notes
    assert ctx.alerts == []


def _health(ctx, *, ts, prefetch, fail, p95):
    ctx.live.conn.execute(
        "INSERT INTO health(ts,pid,platform,prefetch_n,injected_n,empty_n,embed_fail_n,fts_fallback_n,"
        "timeout_n,p95_ms,last_error_class,snapshot_run) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (ts, 1, "telegram", prefetch, 0, 0, fail, fail, 0, p95, "URLError", "r0"))
    ctx.live.conn.commit()


def test_health_alerts(ctx):
    _health(ctx, ts=ctx.now - 3600, prefetch=10, fail=3, p95=3500)
    _health(ctx, ts=ctx.now - 3 * 86400, prefetch=100, fail=100, p95=9000)   # outside 24h
    codes = [a.code for a in alerts.evaluate_health(ctx, now=ctx.now)]
    assert codes == ["recall_embed_fail_rate", "prefetch_p95"]
    h = alerts.health_summary(ctx, now=ctx.now)
    assert h["prefetch"] == 10 and h["embed_fail"] == 3


def test_health_single_failed_prefetch_alerts(ctx):
    _health(ctx, ts=ctx.now - 60, prefetch=1, fail=1, p95=100)
    assert [a.code for a in alerts.evaluate_health(ctx, now=ctx.now)] == ["recall_embed_fail_rate"]


def test_serving_stale_uses_wall_clock_mtime(ctx):
    p = ctx.paths.recall_sqlite
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    assert alerts.evaluate_health(ctx, now=ctx.now) == []
    old = time.time() - 40 * 3600
    os.utime(p, (old, old))
    assert [a.code for a in alerts.evaluate_health(ctx, now=ctx.now)] == ["serving_stale"]


def test_health_quiet_when_ok(ctx):
    _health(ctx, ts=ctx.now - 60, prefetch=50, fail=1, p95=400)
    assert alerts.evaluate_health(ctx, now=ctx.now) == []
