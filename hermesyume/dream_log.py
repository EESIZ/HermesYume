"""Dream Log (PLAN-v2 §4.3 "Dream Log", CONTRACTS §4.20): one Korean Markdown report per run.

File: ``dream-log/YYYY-MM-DD_HHMMSS[_dry].md`` (KST, second resolution; collision → ``_2``,
``_3`` …), 0600 in a 0700 directory. Full texts of what was created/changed — never the text of a
forgotten or purged row, never a secret (every text goes through ``threat.redact_secrets``; gate
rejections for secrets show no text at all). Free text is HTML-escaped so a Markdown viewer never
renders tags from conversations. There is no "확인 필요" section and nothing here asks the operator
to approve anything (U2).
"""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any, Iterable

from . import clock
from .types import KIND_LABEL_KO, Plan, text_sha

STATUS_KO = {
    "committed": "커밋됨",
    "held": "커밋됨 (일부 연산은 적용하지 않고 기록만 함)",
    "dry": "리허설 (dry-run, 아무것도 저장하지 않음)",
    "failed": "실패",
    "noop": "변경 없음",
    "planned": "계획됨",
}
MODE_KO = {"live": "야간 정리", "dry": "리허설", "migrate": "마이그레이션"}
TIER_KO = {"pinned": "고정", "durable": "보호", "slow": "느린 감쇠", "decaying": "감쇠",
           "expiring": "기한", "legacy": "레거시"}
REASON_KO = {
    "schema": "형식 오류", "kind_enum": "종류 열거형 밖", "length": "길이(15~400자) 밖",
    "evidence_outside": "근거가 창 밖", "level1_not_explicit": "중요도 1인데 명시 아님",
    "relative_time": "상대 시간어", "meta_pattern": "메타·목록 서술", "memory_meta": "기억 시스템 자체에 대한 서술",
    "uuid": "UUID 나열", "filenames": "파일명 나열", "secret": "비밀값", "threat": "주입·위협 패턴",
    "assistant_only": "어시스턴트 발화뿐(사용자 근거 없음)",
}
EXCLUDED_KO = {
    "source": "제외 원천(cron 등)", "owner": "소유자 아님", "chat_type": "그룹 대화",
    "synthetic": "합성 평가 세션", "deny_cwd": "차단 작업 폴더", "hidden_session": "숨김 세션",
    "tool_role": "도구 메시지", "hidden_message": "숨김 메시지", "compressed_summary": "압축 요약",
    "generation_copy": "압축 세대 복사본", "not_settled": "정착 전(다음 실행)", "empty": "빈 메시지",
    "inactive": "되감기로 지워진 행", "md_excluded": "md 제외 파일", "md_missing_root": "md 폴더 없음",
    "md_unreadable": "md 읽기 실패", "window_already_done": "이미 처리한 창",
}
# Lineage-level reasons are counted over the whole state.db every run (B1-S2), not per new message.
LINEAGE_LEVEL_EXCLUSIONS = frozenset({"source", "owner", "chat_type", "synthetic", "deny_cwd",
                                      "hidden_session"})
CHANGE_KO = {"add": "추가", "replace": "교체", "remove": "제거(강등)", "mirror_add": "추가(직접 편집 반영)",
             "mirror_remove": "제거(직접 편집 반영)"}
CORE_FILE = {"user": "USER.md", "memory": "MEMORY.md"}
HIDDEN = "[잊음 처리됨 — 원문 기록 안 함]"
REJECT_TEXT_MAX = 1000


def related_reason_ko(reason: Any) -> str:
    """upsert link reasons → Korean (unknown codes pass through)."""
    r = str(reason or "")
    if r == "protected":
        return "보호(pinned/durable) 기억이라 바꾸지 않고 따로 저장해 연결"
    if r == "different_aspects":
        return "다른 측면 — 연결"
    if r == "different_aspects:pinned":
        return "다른 측면 — 고정 기억이라 통합하지 않음"
    if r == "different_aspects:protected":
        return "다른 측면 — 보호 기억이라 통합하지 않음"
    if r.startswith("different_aspects:preservation"):
        missing = r.split(":", 2)[2] if r.count(":") >= 2 else ""
        return "다른 측면 — 통합 결과가 사실 보존 검사를 통과하지 못해 둘 다 유지" + (
            f" (빠진 토큰: {missing})" if missing else "")
    if r.startswith("different_aspects:"):
        return f"다른 측면 — 통합하지 않음 ({r.split(':', 1)[1]})"
    return r


# ── text helpers ─────────────────────────────────────────────────────────────

def _redact(text: str) -> str:
    """Secrets (threat patterns) and Telegram bot URLs/tokens removed."""
    try:
        from .alerts import scrub
        return scrub(text)
    except Exception:  # noqa: BLE001 — a log line must never fail the run
        return text


def t(text: Any, *, limit: int | None = None) -> str:
    """One-line, secret-redacted, HTML-escaped rendering of a free text."""
    s = " ".join(str(text or "").split())
    s = _redact(s)
    if limit is not None and len(s) > limit:
        s = s[:limit] + f"… (+{len(s) - limit}자)"
    if s.count("`") % 2:
        return html.escape(s, quote=False)
    # escape outside `code spans` only (entities are not decoded inside code spans)
    parts = s.split("`")
    return "`".join(p if i % 2 else html.escape(p, quote=False) for i, p in enumerate(parts))


def _code(s: Any) -> str:
    return "`" + str(s).replace("`", "'") + "`"


def _kind(k: Any) -> str:
    return KIND_LABEL_KO.get(str(k), str(k))


def _fmt_ts(ts: Any) -> str:
    try:
        return clock.fmt_kst(float(ts), "%Y-%m-%d %H:%M KST")
    except (TypeError, ValueError):
        return "-"


def _num(x: Any, nd: int = 3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def quantile(values: list[float], q: float) -> float | None:
    """Linear-interpolated quantile (numpy 'linear'); None for no data."""
    xs = sorted(float(v) for v in values if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * min(1.0, max(0.0, q))
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


class _Doc:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def h1(self, s: str) -> None:
        self.lines += [f"# {s}", ""]

    def h2(self, s: str) -> None:
        self.lines += [f"## {s}", ""]

    def h3(self, s: str) -> None:
        self.lines += [f"### {s}", ""]

    def items(self, xs: Iterable[str], *, empty: str = "없음") -> None:
        xs = list(xs)
        self.lines += [f"- {x}" for x in xs] if xs else [f"- {empty}"]
        self.lines.append("")

    def para(self, *xs: str) -> None:
        self.lines += list(xs) + [""]

    def text(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"


# ── render ───────────────────────────────────────────────────────────────────

def _hidden_ids(rep: Any) -> set[str]:
    ids = {str(x.get("id")) for x in (rep.forgotten or []) if isinstance(x, dict)}
    ids |= {str(x.get("id")) for x in (rep.purged or []) if isinstance(x, dict)}
    return ids


def _suppress_shas(ctx: Any, plan: Plan | None) -> set[str]:
    shas = {s.text_sha for s in (plan.suppress if plan is not None else []) if s.text_sha}
    store = getattr(ctx, "store", None)
    if store is not None:
        try:
            shas |= set(store.suppress_shas())
        except Exception:  # noqa: BLE001 — best effort
            pass
    return shas


def render(ctx: Any, plan: Plan | None, *, status: str, error: str | None = None) -> str:
    st, rep, cfg = ctx.stats, ctx.report, ctx.cfg
    hidden = _hidden_ids(rep)
    sup = _suppress_shas(ctx, plan)

    def txt(item: dict, key: str = "text", id_key: str = "id", *, limit: int | None = None) -> str:
        if str(item.get(id_key)) in hidden:
            return HIDDEN
        return t(item.get(key), limit=limit)

    mode = getattr(ctx, "mode", "live")
    dry = bool(getattr(ctx, "dry_run", False))
    d = _Doc()
    title = f"Dream Log — {clock.fmt_kst(ctx.now, '%Y-%m-%d %H:%M')} KST"
    if mode == "migrate":
        title += " · 마이그레이션"
    if dry:
        title += " · 리허설"
    d.h1(title)

    # 요약
    d.h2("요약")
    lv_b, lv_a = st.lance_version_before, st.lance_version_after
    if dry:
        lance = f"{lv_b if lv_b is not None else '-'} (리허설: 바뀌지 않음)"
    else:
        lance = f"{lv_b if lv_b is not None else '-'} → {lv_a if lv_a is not None else '-'}"
    when = clock.fmt_kst(ctx.now, "%Y-%m-%d %H:%M:%S KST")
    if clock.override_active():
        when += f" (시각 지정, 실제 {clock.fmt_kst(clock.real_now(), '%Y-%m-%d %H:%M KST')})"
    d.items([
        f"상태: {STATUS_KO.get(status, status)}",
        f"실행 id: {_code(ctx.run_id)}",
        f"모드: {MODE_KO.get(mode, mode)}" + (" · 리허설(dry-run)" if dry and mode != "dry" else ""),
        f"기준 시각: {when}",
        f"소요: {_num(st.duration_s, 1)}초",
        f"Lance 버전: {lance}",
        (f"새 기억 {st.created} · 강화 {st.reinforced} (기록만 {st.reinforce_noop}) · 대체 {st.superseded} · "
         f"통합 {st.consolidated} · 연결 {st.related_linked} · 만료 {st.expired} · 휴면 {st.dormant} · "
         f"부활 {st.revived} · 잊음 {st.forgotten} · 영구삭제 {st.purged}"),
    ])

    # 입력
    d.h2("입력")
    by_src: dict[str, list[int]] = {}
    for it in rep.inputs or []:
        s = str(it.get("source", "?"))
        acc = by_src.setdefault(s, [0, 0])
        acc[0] += 1
        acc[1] += int(it.get("messages") or 0)
    src_lines = [f"{_code(s)}: 단위 {n}개, 메시지 {m}개" for s, (n, m) in sorted(by_src.items())]
    d.items(src_lines + [
        f"state.db 세션 {st.sessions_seen}개 · 적격 메시지 {st.messages_in}개 · md 파일 {st.md_files}개 · "
        f"inbox 항목 {st.inbox_items}개",
        f"창: 전체 {st.windows_total} · 정상 {st.windows_ok} · 빈 창 {st.windows_empty} · 실패 {st.windows_failed} · "
        f"격리 {st.windows_quarantined} · 다음으로 미룸 {st.windows_deferred}",
    ])
    d.h3("제외 사유별 개수")
    d.items(f"{EXCLUDED_KO.get(k, k)} ({_code(k)}): {v}"
            + (" (state.db 전체 기준, 매일 같은 값)" if k in LINEAGE_LEVEL_EXCLUSIONS else "")
            for k, v in sorted((st.excluded or {}).items(), key=lambda kv: (-kv[1], kv[0])))

    # 추출된 주장
    claims = list(rep.claims or [])
    d.h2(f"추출된 주장 ({len(claims)})")
    lines = []
    for c in claims:
        status_ = str(c.get("status") or "")
        if status_ == "suppressed" or text_sha(str(c.get("text") or "")) in sup:
            lines.append(f"({_kind(c.get('kind'))}) {HIDDEN} — 억제 목록 적중")
            continue
        extra = f" · {status_}" if status_ and status_ != "active" else ""
        lines.append(f"({_kind(c.get('kind'))}{extra}) **{t(c.get('subject'))}** — {t(c.get('text'))}")
    d.items(lines)

    # 게이트 거절
    rejs = list(rep.rejections or [])
    d.h2(f"게이트 거절 ({len(rejs)})")
    lines = []
    for r in rejs:
        reason = str(r.get("reason") or "")
        body = "[비밀값 포함 — 원문 기록 안 함]" if reason == "secret" else t(r.get("text"), limit=REJECT_TEXT_MAX)
        detail = f" ({t(r.get('detail'), limit=200)})" if r.get("detail") else ""
        lines.append(f"{REASON_KO.get(reason, reason)}{detail}: {body}")
    d.items(lines)

    # 새 기억
    d.h2(f"새 기억 ({len(rep.created or [])})")
    d.items(f"({_kind(c.get('kind'))}·{TIER_KO.get(str(c.get('tier')), c.get('tier'))}) {txt(c)} "
            f"[중요도 {_num(c.get('importance'), 2)}] {_code(c.get('id'))}" for c in rep.created or [])

    # 강화
    d.h2(f"강화 ({len(rep.reinforced or [])})")
    d.items(f"{txt(r)} — {'사용자 근거로 강화' if r.get('user') else '기록만 (사용자 근거 아님)'} {_code(r.get('id'))}"
            for r in rep.reinforced or [])

    # 대체
    d.h2(f"대체 ({len(rep.superseded or [])})")
    d.items(f"전: {txt(s, 'old_text', 'old_id')} → 후: {txt(s, 'new_text', 'new_id')} "
            f"({_code(s.get('old_id'))} → {_code(s.get('new_id'))})" for s in rep.superseded or [])

    # 통합
    d.h2(f"통합 ({len(rep.consolidated or [])})")
    d.items(f"A: {txt(c, 'a')} + B: {txt(c, 'b')} → 결과: {txt(c, 'result')} {_code(c.get('id'))}"
            for c in rep.consolidated or [])

    # 연결
    d.h2(f"연결 ({len(rep.related or [])})")
    d.items(f"{_code(r.get('a_id'))} ↔ {_code(r.get('b_id'))} ({t(related_reason_ko(r.get('reason')))})"
            for r in rep.related or [])

    # 만료 / 휴면 / 부활
    d.h2("만료 · 휴면 · 부활")
    for key, label in (("expired", "만료"), ("dormant", "휴면"), ("revived", "부활")):
        xs = list(getattr(rep, key) or [])
        d.h3(f"{label} ({len(xs)})")
        d.items(f"{txt(x)} (강도 {_num(x.get('strength'), 3)}) {_code(x.get('id'))}" for x in xs)

    # 잊음 / 영구삭제 (id만)
    d.h2("잊음 · 영구삭제 (id만, 원문 기록 안 함)")
    d.items([
        "잊음: " + (", ".join(_code(x.get("id")) for x in rep.forgotten) if rep.forgotten else "없음"),
        "영구삭제: " + (", ".join(_code(x.get("id")) for x in rep.purged) if rep.purged else "없음"),
    ])

    # 회상 통계
    d.h2("회상 통계")
    from .alerts import health_summary
    h = health_summary(ctx, now=ctx.now)
    if h["prefetch"]:
        fail = (f"임베딩 실패 {h['embed_fail']}/{h['prefetch']} ({h['embed_fail_rate']:.1%}) · "
                f"키워드 대체 {h['fts_fallback']} · 시간 초과 {h['timeout']} · "
                f"p95 최대 {h['p95_max'] if h['p95_max'] is not None else '-'}ms")
    else:
        fail = "최근 24시간 health 기록 없음"
    d.items([
        f"주입 {st.recall_injected} · 사용 {st.recall_used} · 도구 검색 적중 {st.recall_tool_hit} · "
        f"shadow {st.recall_shadow} · 강화 제외(cron 등) {st.recall_ignored_platform}",
        f"회상 오류율(최근 24시간): {fail}",
    ])
    d.h3("상위 10")
    top = sorted(rep.recall_top or [], key=lambda x: (-(int(x.get("used") or 0)), -(int(x.get("injected") or 0))))[:10]
    d.items(f"{txt(x)} (주입 {x.get('injected', 0)}, 사용 {x.get('used', 0)}) {_code(x.get('id'))}" for x in top)
    d.h3("365일 동안 회상되지 않은 보호 기억")
    d.items(f"{txt(x)} ({x.get('days')}일) {_code(x.get('id'))}" for x in rep.durable_unrecalled or [])

    # 보류 연산
    d.h2(f"보류 연산 ({len(rep.held or [])})")
    if rep.held:
        d.para("고정·보호 기억을 사용자 근거 없이 바꾸려는 연산이라 적용하지 않고 기록만 했습니다.")
    d.items(f"seq {h_.get('seq')} · {_code(h_.get('op'))} · {_code(h_.get('memory_id'))} · {t(h_.get('reason'))}"
            for h_ in rep.held or [])

    # 새 pin
    d.h2(f"새 pin ({len(rep.new_pins or [])})")
    d.items(f"{txt(p)} {_code(p.get('id'))} — 해제는 `yume unpin {p.get('id')}`" for p in rep.new_pins or [])

    # 핵심 파일 변화
    d.h2(f"핵심 파일 변화 ({len(rep.core_changes or [])})")
    d.items(f"{CORE_FILE.get(str(c.get('target')), c.get('target'))} "
            f"{CHANGE_KO.get(str(c.get('change')), c.get('change'))}: {t(c.get('text'))}"
            for c in rep.core_changes or [])

    # 코사인 분포
    d.h2("코사인 분포 (판정별)")
    cos = rep.cos_by_relation or {}
    if cos:
        d.lines += ["| 판정 | n | 최소 | 25% | 50% | 75% | 최대 |", "|---|---|---|---|---|---|---|"]
        for rel in sorted(cos):
            vs = [float(v) for v in cos[rel] if v is not None]
            if not vs:
                continue
            q = [quantile(vs, p) for p in (0.25, 0.5, 0.75)]
            d.lines.append(f"| {rel} | {len(vs)} | {_num(min(vs))} | {_num(q[0])} | {_num(q[1])} | "
                           f"{_num(q[2])} | {_num(max(vs))} |")
        d.lines.append("")
    else:
        d.items([])

    # 토큰·비용
    d.h2("토큰 · 비용")
    d.items([
        f"LLM 호출 {st.llm_calls}회 (판정 {st.judge_calls}, 판정 실패 {st.judge_failures}, 재판정 대기 {st.judge_pending}) · "
        f"입력 토큰 {st.prompt_tokens} · 출력 토큰 {st.completion_tokens}",
        f"임베딩 입력 {st.embed_inputs}개 · 토큰 {st.embed_tokens}",
        f"추정 비용 ${_num(st.cost_usd, 4)} (단가: 입력 ${_cfgv(cfg, 'llm_price_in_per_mtok')}/M, "
        f"출력 ${_cfgv(cfg, 'llm_price_out_per_mtok')}/M, 임베딩 ${_cfgv(cfg, 'embed_price_per_mtok')}/M)",
    ])

    # 반복 줄 제거 상위
    d.h2("반복 줄 제거 상위")
    d.items(f"{_code(t(x.get('line'), limit=160))} × {x.get('count')}" for x in rep.strip_lines_top or [])

    # 참고
    d.h2("참고")
    d.items(t(n, limit=2000) for n in rep.notes or [])

    # 알림
    from .alerts import ALERT_TITLES
    alerts = list(getattr(ctx, "alerts", []) or [])
    d.h2(f"알림 ({len(alerts)})")
    if alerts and dry:
        d.para("리허설이라 alerts.log에 쓰거나 보내지 않았습니다.")
    d.items(f"[{ALERT_TITLES.get(a.code, a.code)}] {t(a.message, limit=1000)}" for a in alerts)

    if error or status == "failed":
        d.h2("오류")
        d.items([t(error or "알 수 없는 오류", limit=2000)])

    d.lines += ["---", f"*HermesYume {_version()} · {clock.fmt_kst(clock.real_now(), '%Y-%m-%d %H:%M:%S KST')} 작성*"]
    return d.text()


def _cfgv(cfg: Any, key: str) -> str:
    try:
        return str(cfg[key])
    except (KeyError, TypeError):
        return str(getattr(cfg, key, "-"))


def _version() -> str:
    try:
        from . import __version__
        return str(__version__)
    except Exception:  # noqa: BLE001
        return "?"


# ── write ────────────────────────────────────────────────────────────────────

def log_filename(now: float, *, dry: bool, n: int = 1) -> str:
    stem = clock.kst_stamp(now) + ("_dry" if dry else "")
    return f"{stem}.md" if n <= 1 else f"{stem}_{n}.md"


def write(paths: Any, now: float, text: str, *, dry: bool) -> Path:
    """Create dream-log/<stamp>[_dry][_n].md exclusively (0600, dir 0700) and return its path."""
    paths.ensure_dir(paths.data_dir)
    d = paths.ensure_dir(paths.dream_log_dir)
    data = text.encode("utf-8")
    n = 1
    while True:
        p = d / log_filename(now, dry=dry, n=n)
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            n += 1
            continue
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        return p


__all__ = ["render", "write", "log_filename", "quantile", "t"]
