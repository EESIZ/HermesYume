"""Extraction windows (PLAN-v2 §3.4; CONTRACTS §4.5).

A window is a run of whole exchanges (a user turn and its replies) whose rendered body is at most
``window_chars``. The exchange just before the window is prepended as read-only context. An
exchange that alone exceeds ``window_chars`` is split at message boundaries (DEVIATIONS: B1-W1),
so a window is only ever larger than the limit when a single message is.

Determinism matters for T5: windows are packed greedily from the first message, and the context
of a later window is the full prefix of the exchange that precedes it — exactly what the loader
returns as ``context_before`` when a later run resumes at that point. So the windows of a resumed
run are the deferred windows of the earlier run (same ids, same bodies).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .clock import fmt_kst, kst, kst_date, kst_date_ko
from .sanitize import redact, strip_injected_blocks
from .types import Message, Window, make_window_id, sha256_hex

if TYPE_CHECKING:  # pragma: no cover
    from .sources.markdown import MdSource

CONTEXT_HEADING = "[이전 맥락 · 추출 대상 아님]"
BODY_HEADING = "[추출 대상]"
_TITLE_MAX = 80


def _cfg(cfg: Any, key: str, default: Any) -> Any:
    try:
        return cfg[key]
    except (KeyError, TypeError):
        return getattr(cfg, key, default)


def split_exchanges(msgs: list[Message]) -> list[list[Message]]:
    """New exchange at each user message; agent_log messages are single-message exchanges;
    assistant messages before any user message form their own (leading) exchange."""
    out: list[list[Message]] = []
    cur: list[Message] = []
    for m in msgs:
        if m.role == "agent_log":
            if cur:
                out.append(cur)
                cur = []
            out.append([m])
        elif m.role == "user":
            if cur:
                out.append(cur)
            cur = [m]
        else:
            cur.append(m)
    if cur:
        out.append(cur)
    return out


def msg_head(m: Message) -> str:
    """"[U#1234 09-28 14:02]" for state.db; "[U#md:L12]" / "[L#inbox:7]" otherwise."""
    if m.source == "statedb" and m.ts:
        return f"[{m.ref} {fmt_kst(m.ts, '%m-%d %H:%M')}]"
    return f"[{m.ref}]"


def _line(m: Message) -> str:
    return f"{msg_head(m)} {m.text}"


def render_messages(msgs: list[Message]) -> str:
    return "\n".join(_line(m) for m in msgs)


def _rlen(lens: list[int]) -> int:
    return sum(lens) + max(0, len(lens) - 1)


def format_header(*, platform: str | None, title: str, start_ts: float, end_ts: float,
                  ref_ts: float, md_date: str | None = None) -> str:
    plat = platform or "unknown"
    t = " ".join((title or "").replace('"', "'").split())
    if len(t) > _TITLE_MAX:
        t = t[:_TITLE_MAX] + "…"
    if plat == "md" or md_date is not None:
        period = md_date or kst_date(start_ts)
    else:
        s, e = kst(start_ts), kst(end_ts)
        if s.date() == e.date():
            period = f"{s:%Y-%m-%d %H:%M}–{e:%H:%M} KST"
        else:
            period = f"{s:%Y-%m-%d %H:%M}–{e:%Y-%m-%d %H:%M} KST"
    return f'세션: {plat} "{t}" / 기간: {period} / 정리 기준일: {kst_date_ko(ref_ts)}'


def _defensive(msgs: list[Message]) -> list[Message]:
    """Callers pass sanitized messages; this only guarantees that no injected block or secret
    reaches the LLM if one did not (idempotent on sanitized text)."""
    out: list[Message] = []
    for m in msgs:
        text = m.text or ""
        clean = redact(strip_injected_blocks(text))[0]
        if clean != text:
            clean = clean.strip()
            if not clean:
                continue
            m = replace(m, text=clean)
        out.append(m)
    return out


def _render_context(ctx: list[Message], limit: int) -> str:
    text = render_messages(ctx)
    if limit > 0 and len(text) > limit:
        text = text[:limit].rstrip() + " …"
    return text


def build_windows(*, source: str, root: str, platform: str | None, title: str,
                  messages: list[Message], context_before: list[Message], cfg: Any, ref_ts: float,
                  end_offset: int | None = None, md: "MdSource | None" = None) -> list[Window]:
    """Pack `messages` (sanitized, ascending) into windows. `context_before` (sanitized) is the
    read-only context of the first window. md windows need `end_offset` (or `md`)."""
    window_chars = int(_cfg(cfg, "window_chars", 8000))
    ctx_chars = int(_cfg(cfg, "window_context_chars", 1500))
    if source == "md" and not platform:
        platform = "md"
    if md is not None and end_offset is None:
        end_offset = md.end_offset
    msgs = _defensive(list(messages))
    ctx0 = _defensive(list(context_before or []))
    if not msgs:
        return []

    # rendered line per message, computed once (render length = Σ lines + newlines)
    line_len = {id(m): len(_line(m)) for m in msgs}

    # units: (exchange index, messages); oversized exchanges split at message boundaries
    units: list[tuple[int, list[Message]]] = []
    for ei, ex in enumerate(split_exchanges(msgs)):
        lens = [line_len[id(m)] for m in ex]
        if _rlen(lens) <= window_chars:
            units.append((ei, ex))
            continue
        chunk: list[Message] = []
        size = 0
        for m, ln in zip(ex, lens):
            if chunk and size + 1 + ln > window_chars:
                units.append((ei, chunk))
                chunk, size = [], 0
            size = ln if not chunk else size + 1 + ln
            chunk.append(m)
        if chunk:
            units.append((ei, chunk))

    # greedy packing of whole units; the first unit of a window is always taken
    groups: list[list[int]] = []
    cur: list[int] = []
    size = 0
    for ui, (_, umsgs) in enumerate(units):
        usize = _rlen([line_len[id(m)] for m in umsgs])
        if cur and size + 1 + usize > window_chars:
            groups.append(cur)
            cur, size = [], 0
        size = usize if not cur else size + 1 + usize
        cur.append(ui)
    if cur:
        groups.append(cur)

    # messages starting with replies continue the exchange held in context_before
    lead_continues = bool(ctx0) and units[0][1][0].role == "assistant"
    units_of: dict[int, list[int]] = {}
    for ui, (ei, _) in enumerate(units):
        units_of.setdefault(ei, []).append(ui)

    def exchange_before(ei: int, upto: int) -> list[Message]:
        ms = [m for ui in units_of[ei] if ui < upto for m in units[ui][1]]
        return ctx0 + ms if ei == 0 and lead_continues else ms

    md_path = md.path if md is not None else (root if source == "md" else None)
    md_date = md.date if md is not None else None
    md_slug = md.slug if md is not None else None
    out: list[Window] = []
    for gi, group in enumerate(groups):
        body = [m for ui in group for m in units[ui][1]]
        u0 = group[0]
        ei0 = units[u0][0]
        if u0 == 0:
            ctx = ctx0
        elif units[u0 - 1][0] == ei0:      # continuation chunk: the exchange prefix so far
            ctx = exchange_before(ei0, u0)
        else:                              # the whole previous exchange
            ctx = exchange_before(units[u0 - 1][0], u0)
        body_text = render_messages(body)
        start_ts = min(m.ts for m in body)
        last_ts = max(m.ts for m in body)
        if source == "md":
            first_id = int(body[0].msg_id or 0)
            if gi + 1 < len(groups):
                last_id = int(units[groups[gi + 1][0]][1][0].msg_id or 0)
            elif end_offset is not None:
                last_id = int(end_offset)
            else:
                last_id = int(body[-1].msg_id or 0) + 1
            wid = make_window_id("md", root, first_id, last_id, content_sha=sha256_hex(body_text))
        else:
            ids = [int(m.msg_id or 0) for m in body]
            first_id, last_id = min(ids), max(ids)
            wid = make_window_id(source, root, first_id, last_id)
        header = format_header(platform=platform, title=title, start_ts=start_ts, end_ts=last_ts,
                               ref_ts=ref_ts, md_date=md_date)
        parts = [header, ""]
        if ctx:
            parts += [CONTEXT_HEADING, _render_context(ctx, ctx_chars), ""]
        parts += [BODY_HEADING, body_text]
        session_ids: list[str] = []
        for m in body:
            if m.session_id and m.session_id not in session_ids:
                session_ids.append(m.session_id)
        out.append(Window(window_id=wid, source=source, root=root, first_id=first_id,
                          last_id=last_id, start_ts=start_ts, last_ts=last_ts, platform=platform,
                          title=title or "", header=header, text="\n".join(parts), messages=body,
                          context=list(ctx), session_ids=session_ids, attempts=0,
                          md_path=md_path, md_date=md_date, md_slug=md_slug))
    return out
