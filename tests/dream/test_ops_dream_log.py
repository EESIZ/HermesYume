"""dream_log.py (PLAN §4.3 Dream Log, U2): sections in order, full texts, HTML escape, no secrets,
no forgotten text, no "확인 필요", file naming/permissions."""

from __future__ import annotations

import os
import re
import stat

from hermesyume import dream_log
from hermesyume.types import Alert, Plan, Rejection, SuppressRow, text_sha
from dataclasses import asdict

SECTIONS = ["요약", "입력", "추출된 주장", "게이트 거절", "새 기억", "강화", "대체", "통합", "연결",
            "만료 · 휴면 · 부활", "잊음 · 영구삭제", "회상 통계", "보류 연산", "새 pin", "핵심 파일 변화",
            "코사인 분포", "토큰 · 비용", "반복 줄 제거 상위", "참고", "알림"]
TG_TOKEN = "123456789:" + "B" * 35


def _populate(ctx):
    rep, st = ctx.report, ctx.stats
    st.created, st.reinforced, st.superseded = 2, 1, 1
    st.excluded = {"synthetic": 43, "source": 3}
    rep.inputs += [{"source": "statedb", "root": "s1", "title": "t", "messages": 5},
                   {"source": "md", "root": "/m.md", "title": "m", "messages": 2}]
    rep.claims.append({"origin_key": "w#0", "kind": "reference", "subject": "포트",
                       "text": "스테이징 서버 포트는 8081이다 <b>굵게</b> & 끝", "status": "active"})
    rep.claims.append({"origin_key": "w#1", "kind": "fact", "subject": "억제",
                       "text": "억제된 비밀 문장입니다 열다섯자 이상", "status": "active"})
    rep.rejections.append(asdict(Rejection("w", 2, "secret", "openai", "키는 sk-" + "z" * 30 + " 이다", "fact")))
    rep.rejections.append(asdict(Rejection("w", 3, "relative_time", "오늘", "오늘 회의가 있다 열다섯자", "event")))
    rep.created += [{"id": "id-keep", "kind": "rule", "tier": "durable", "text": "규칙 문장 전문", "importance": 0.97},
                    {"id": "id-forgot", "kind": "fact", "tier": "decaying", "text": "잊어야 할 비밀스러운 문장", "importance": 0.5}]
    rep.forgotten.append({"id": "id-forgot"})
    rep.reinforced.append({"id": "id-keep", "text": "규칙 문장 전문", "user": True})
    rep.superseded.append({"old_id": "o", "old_text": "당번은 7조", "new_id": "n", "new_text": "당번은 9조"})
    rep.consolidated.append({"id": "c", "a": "A문장", "b": "B문장", "result": "A와 B"})
    rep.related.append({"a_id": "a", "b_id": "b", "reason": "protected"})
    rep.dormant.append({"id": "d", "text": "휴면 문장", "strength": 0.05})
    rep.held.append({"seq": 3, "op": "supersede", "memory_id": "p", "reason": "x"})
    rep.new_pins.append({"id": "pin1", "text": "고정 문장"})
    rep.core_changes.append({"target": "user", "change": "remove", "text": "**가계부:** 원장"})
    rep.cos_by_relation = {"duplicate": [0.96, 0.97, 0.99], "unrelated": [0.5]}
    rep.strip_lines_top.append({"line": "🔁 [BACKUP BOT] 백업", "count": 9})
    rep.notes.append(f"메모 https://api.telegram.org/bot{TG_TOKEN}/x")
    ctx.alerts.append(Alert(code="stalled", message="<정체>", run_id=ctx.run_id, ts=ctx.now))


def test_sections_in_order_and_no_review_queue(ctx):
    _populate(ctx)
    out = dream_log.render(ctx, None, status="committed")
    heads = [m.group(1) for m in re.finditer(r"^## (.+?)(?: \([^)]*\))?$", out, re.M)]
    assert heads == SECTIONS
    assert "확인 필요" not in out
    assert "승인" not in out
    assert "커밋됨" in out and ctx.run_id in out


def test_full_texts_html_escaped(ctx):
    _populate(ctx)
    out = dream_log.render(ctx, None, status="committed")
    assert "스테이징 서버 포트는 8081이다 &lt;b&gt;굵게&lt;/b&gt; &amp; 끝" in out
    assert "<b>" not in out and "<정체>" not in out and "&lt;정체&gt;" in out
    assert "전: 당번은 7조 → 후: 당번은 9조" in out
    assert "A: A문장 + B: B문장 → 결과: A와 B" in out


def test_no_secrets_no_forgotten_text(ctx):
    _populate(ctx)
    plan = Plan(run_id=ctx.run_id, mode="live", created_at=ctx.now, now=ctx.now, lance_version_before=1,
                suppress=[SuppressRow(id="s", vector=None, text_sha=text_sha("억제된 비밀 문장입니다 열다섯자 이상"),
                                      kind="fact", created_at=ctx.now, reason="forget")])
    out = dream_log.render(ctx, plan, status="committed")
    assert "sk-zzzz" not in out and "[비밀값 포함" in out
    assert TG_TOKEN not in out and "BBBBBBBB" not in out
    assert "잊어야 할 비밀스러운 문장" not in out and "id-forgot" in out
    assert "억제된 비밀 문장" not in out
    assert "규칙 문장 전문" in out


def test_empty_report_and_failure(ctx):
    out = dream_log.render(ctx, None, status="failed", error="RuntimeError: boom")
    assert "## 오류" in out and "boom" in out
    assert out.count("- 없음") >= 10


def test_dry_marks_and_alerts_not_sent(ctx):
    ctx.dry_run, ctx.mode = True, "migrate"
    ctx.alerts.append(Alert(code="run_failed", message="x", ts=ctx.now))
    out = dream_log.render(ctx, None, status="dry")
    assert "마이그레이션" in out.splitlines()[0] and "리허설" in out.splitlines()[0]
    assert "보내지 않았습니다" in out


def test_write_names_permissions_collisions(paths, now):
    p1 = dream_log.write(paths, now, "a", dry=False)
    p2 = dream_log.write(paths, now, "b", dry=False)
    p3 = dream_log.write(paths, now, "c", dry=True)
    assert p1.name == "2026-10-02_044000.md"
    assert p2.name == "2026-10-02_044000_2.md"
    assert p3.name == "2026-10-02_044000_dry.md"
    assert p1.read_text(encoding="utf-8") == "a" and p2.read_text(encoding="utf-8") == "b"
    for p in (p1, p2, p3):
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(paths.dream_log_dir).st_mode) == 0o700


def test_quantile():
    assert dream_log.quantile([], 0.5) is None
    assert dream_log.quantile([1.0], 0.9) == 1.0
    assert dream_log.quantile([0.0, 1.0], 0.25) == 0.25
    assert abs(dream_log.quantile([0.1, 0.2, 0.3, 0.4, 0.5], 0.99) - 0.496) < 1e-9
