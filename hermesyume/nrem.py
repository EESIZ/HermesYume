"""NREM: episodes → gated, normalized, embedded claims (PLAN-v2 §4.2 N0–N7; CONTRACTS §4.10).

N0 preflight   schema/meta/model guards, embedding + 1-token LLM ping (401 aborts), threat scanner
               (fail-closed), replay of `planned` runs, ledger backup, Lance version, live snapshot,
               run row (live/migrate only)
N1 collect     state.db lineages (read-only), md sources (offset ledger), episodic core_add inbox items
N2 windows     sanitize + window packing, per-root order, window cap, empty windows (no LLM)
N3–N5          extract → gates → normalize, failure accounting (3rd failure → quarantined + alert)
N6 embed       one batched call for every claim; EmbedError aborts the run (nothing is written)
N7             sort by event time, stats + Dream Log report, watermark / md offset deltas

Nothing here writes Lance, ledger tables (except the run row and backup in N0), live.db or Hermes
files; the deltas are committed by plan.commit_plan (R8). Cross-builder modules (sources.*,
sanitize, windows, plan) are resolved at call time through `_dep`.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from . import extract as _extract
from . import gates as _gates
from . import normalize as _normalize
from .clock import kst_date, parse_iso
from .embedder import embed_input
from .threat import ThreatScannerUnavailable, load_scanner
from .types import (WM_ADVANCING_STATUSES, BudgetExceeded, Claim, LiveSnapshot, MdFileState, Message,
                    Rejection, RunRecord, Window, WindowState, make_window_id, sha256_hex)

log = logging.getLogger("hermesyume.nrem")

_EPISODE_DATE_RE = re.compile(r"^Session: (\d{4}-\d{2}-\d{2})")
_ERR_KEEP = 500


def _dep(name: str) -> Any:
    """Other builders' modules, resolved at call time (tests inject stand-ins via sys.modules)."""
    return importlib.import_module(f"hermesyume.{name}")


def _accepts(fn: Any, kw: str) -> bool:
    try:
        return kw in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


@dataclass
class PreflightResult:
    snapshot: LiveSnapshot               # LiveSnapshot(0, 0) when live.db absent
    lance_version_before: int
    replayed_runs: list[str] = field(default_factory=list)
    ledger_backup: str | None = None


@dataclass
class NremResult:
    claims: list[Claim]                  # gated + normalized + embedded; sorted by (event_time or first_seen_at, origin_key)
    windows: list[Window]                # every window considered this run, in processing order
    window_states: dict[str, WindowState]   # window_id → state for windows processed THIS run (ledger delta)
    rejections: list[Rejection]
    wm_delta: dict[str, tuple[float, int]]  # only roots whose watermark advances
    session_roots: dict[str, str]
    md_states: list[MdFileState]         # only files whose state changes
    deferred_windows: int
    inbox_episodic_ids: list[int]        # pending episodic core_add items whose window reached ok/empty/quarantined
    sanitize_report: Any


# ── N0 ───────────────────────────────────────────────────────────────────────

def _writes_ledger(ctx: Any) -> bool:
    return not ctx.dry_run and ctx.mode in ("live", "migrate") and ctx.ledger is not None \
        and not getattr(ctx.ledger, "readonly", False)


def _safe_version(store: Any) -> int | None:
    try:
        return int(store.version())
    except Exception:  # noqa: BLE001 — a broken store is reported by check_schema right after
        return None


def preflight(ctx: Any) -> PreflightResult:
    """N0. The dream.lock is taken by the CLI. Auth errors (401/403) and guard mismatches raise;
    rem.run_dream turns them into alerts and a failed run row."""
    cfg = ctx.cfg
    if _writes_ledger(ctx):
        # insert first so that any N0 failure leaves a `failed` run row (CONTRACTS §3 lifecycle)
        from .clock import override_active
        ctx.ledger.insert_run(RunRecord(
            run_id=ctx.run_id, started_at=float(ctx.now), mode=ctx.mode,
            now_override=float(ctx.now) if override_active() else None, status="failed",
            lance_version_before=_safe_version(ctx.store),
            wm_before_json=ctx.ledger.wm_snapshot_json(), error="incomplete"))

    ctx.store.check_schema()
    ctx.ledger.check_meta(cfg.embed_model_id(), int(cfg.embed_dim))
    ctx.store.check_embed_model()
    ctx.embedder.ping()
    ctx.llm.ping(cfg.extract_model)

    if ctx.scanner is None:
        try:
            ctx.scanner = load_scanner(cfg.hermes_runtime_dir)
        except ThreatScannerUnavailable as e:
            ctx.alert("scanner_unavailable",
                      f"위협·비밀값 검사기를 불러오지 못해 실행을 멈췄습니다 ({type(e).__name__}).",
                      level="error")
            raise

    replayed: list[str] = []
    backup: str | None = None
    if _writes_ledger(ctx):
        replayed = list(_dep("plan").replay_planned(ctx) or [])
        backup = str(ctx.ledger.backup(ctx.paths.backups_dir, keep=int(cfg.ledger_backups_keep)))
        try:                                   # §10.4 weekly data-dir tar (serving copy excluded)
            from .backups import weekly_tar
            tar = weekly_tar(ctx.paths, cfg)
            if tar is not None:
                ctx.note(f"주간 데이터 백업: backups/{tar.name}")
        except Exception as e:  # noqa: BLE001 — a backup failure must not stop the night
            ctx.note(f"주간 데이터 백업 실패: {type(e).__name__}")

    version_before = int(ctx.store.version())
    snapshot = ctx.live.snapshot() if ctx.live is not None else LiveSnapshot(0, 0)
    if _writes_ledger(ctx):
        ctx.ledger.update_run(ctx.run_id, lance_version_before=version_before,
                              wm_before_json=ctx.ledger.wm_snapshot_json())
    ctx.stats.lance_version_before = version_before
    if replayed:
        ctx.note(f"이전 실행 재생: {', '.join(replayed)}")
    return PreflightResult(snapshot=snapshot, lance_version_before=version_before,
                           replayed_runs=replayed, ledger_backup=backup)


# ── helpers ──────────────────────────────────────────────────────────────────

def _content_chars(w: Window) -> int:
    return sum(len(m.text or "") for m in w.messages if m.role in ("user", "agent_log"))


def _processing_order(built: list[Window]) -> list[Window]:
    """Sort by (start_ts, root, first_id) while keeping each root's own window order."""
    running: dict[str, float] = {}
    keyed = []
    for seq, w in enumerate(built):
        t = max(running.get(w.root, float("-inf")), float(w.start_ts))
        running[w.root] = t
        keyed.append(((t, w.root, seq), w))
    keyed.sort(key=lambda kv: kv[0])
    return [w for _, w in keyed]


def _failed_attempts(ledger: Any) -> dict[tuple[str, str, int], int]:
    """Failed-attempt counts per (source, root, first_id). Stale failures of a span that was later
    committed under another window id are deleted by ledger.upsert_window (F-32)."""
    out: dict[tuple[str, str, int], int] = {}
    for ws in ledger.windows(status="failed"):
        k = (ws.source, ws.root_session_id, int(ws.first_id))
        out[k] = max(out.get(k, 0), int(ws.attempts or 0))
    return out


def _state(w: Window, status: str, *, run_id: str, attempts: int, error: str | None = None,
           n_claims: int = 0) -> WindowState:
    return WindowState(window_id=w.window_id, source=w.source, root_session_id=w.root,
                       first_id=int(w.first_id), last_id=int(w.last_id), last_ts=float(w.last_ts),
                       status=status, attempts=int(attempts),
                       last_error=(error or None) and error[:_ERR_KEEP], run_id=run_id,
                       n_claims=int(n_claims))


def _short(root: str) -> str:
    return root if len(root) <= 60 else "…" + root[-59:]


def _inbox_window(item: Any, *, ctx: Any, windows_mod: Any, sanitize_mod: Any,
                  repeat_lines: set[str], report: Any) -> Window:
    """Episodic core_add (e.g. 'Session: 2026-06-27 …') → one md-like window (CONTRACTS §4.10)."""
    raw = item.text or ""
    m = _EPISODE_DATE_RE.match(raw.lstrip())
    date = m.group(1) if m else kst_date(float(item.ts))
    base_ts = parse_iso(f"{date}T12:00") or float(item.ts)
    root = f"inbox:{item.id}"
    msg = Message(ref=f"L#inbox:{item.id}", key=f"i:{item.id}", role="agent_log", text=raw,
                  ts=base_ts, source="md", session_id="inbox", msg_id=0, line=None,
                  platform="inbox")
    msgs = sanitize_mod.sanitize_messages([msg], cfg=ctx.cfg, repeat_lines=repeat_lines,
                                          report=report)
    body = windows_mod.render_messages(msgs)
    title = f"core_add #{item.id}"
    header = windows_mod.format_header(platform="inbox", title=title, start_ts=base_ts,
                                       end_ts=base_ts, ref_ts=ctx.now, md_date=date)
    text = f"{header}\n\n{windows_mod.BODY_HEADING}\n{body}"
    nbytes = len(raw.encode("utf-8"))
    return Window(window_id=make_window_id("md", root, 0, nbytes, content_sha=sha256_hex(body)),
                  source="md", root=root, first_id=0, last_id=nbytes, start_ts=base_ts,
                  last_ts=base_ts, platform="inbox", title=title, header=header, text=text,
                  messages=list(msgs), context=[], session_ids=["inbox"], md_path=None,
                  md_date=date, md_slug=f"inbox-{item.id}")


def compute_watermarks(windows: list[Window], states: dict[str, WindowState]) -> dict[str, tuple[float, int]]:
    """Per state.db root: advance through the longest prefix of its windows (in order) whose state
    is ok/empty/quarantined; wm = (last_ts, last_id) of the last such window."""
    by_root: dict[str, list[Window]] = {}
    for w in windows:
        if w.source == "statedb":
            by_root.setdefault(w.root, []).append(w)
    out: dict[str, tuple[float, int]] = {}
    for root, ws in by_root.items():
        last: Window | None = None
        for w in ws:
            st = states.get(w.window_id)
            if st is None or st.status not in WM_ADVANCING_STATUSES:
                break
            last = w
        if last is not None:
            out[root] = (float(last.last_ts), int(last.last_id))
    return out


def _advancing_prefix(ws: list[Window], states: dict[str, WindowState]) -> tuple[Window | None, bool]:
    """(last advancing window of the prefix, whether every window advanced)."""
    last: Window | None = None
    for w in ws:
        st = states.get(w.window_id)
        if st is None or st.status not in WM_ADVANCING_STATUSES:
            return last, False
        last = w
    return last, True


# ── N1–N7 ────────────────────────────────────────────────────────────────────

def run_nrem(ctx: Any, pre: PreflightResult) -> NremResult:
    cfg, paths, stats, rep = ctx.cfg, ctx.paths, ctx.stats, ctx.report
    statedb = _dep("sources.statedb")
    markdown = _dep("sources.markdown")
    sanitize = _dep("sanitize")
    windows_mod = _dep("windows")
    settle = ctx.settle_minutes if ctx.settle_minutes is not None else int(cfg.settle_minutes)

    # ── N1 state.db ──
    lineages: list[Any] = []
    session_roots: dict[str, str] = {}
    recent: list[tuple[str, str]] = []
    session_end_ids = ctx.live.session_end_ids() if ctx.live is not None else set()
    if paths.state_db.exists():
        from .sqlite_util import open_for_read
        with open_for_read(paths.state_db, pure=bool(ctx.dry_run)) as conn:
            load = statedb.load_lineages(conn, ledger=ctx.ledger, cfg=cfg, now=ctx.now,
                                         settle_minutes=settle, session_end_ids=session_end_ids)
            recent = list(statedb.recent_texts(conn, cfg=cfg, now=ctx.now,
                                               days=int(cfg.repeat_line_days)))
        lineages = list(load.lineages)
        session_roots = dict(load.session_roots)
        stats.sessions_seen += int(load.sessions_seen)
        stats.messages_in += int(load.messages_in)
        for reason, n in (load.excluded or {}).items():
            stats.bump_reason("excluded", reason, int(n))

    # ── N1 md ──
    md_cfg = cfg if ctx.settle_minutes is None or not hasattr(cfg, "replace") \
        else cfg.replace(settle_minutes=int(ctx.settle_minutes))
    md_kw = {"now": ctx.now} if _accepts(markdown.scan_md_sources, "now") else {}
    md_sources, md_excluded = markdown.scan_md_sources(md_cfg, ctx.ledger, **md_kw)
    md_sources = list(md_sources)
    stats.md_files += len(md_sources)
    for reason, n in (md_excluded or {}).items():
        stats.bump_reason("excluded", reason, int(n))

    # ── N1 inbox: episodic core_add items (pending, id ≤ snapshot) ──
    episodic: list[Any] = []
    if ctx.live is not None and pre.snapshot.max_inbox_id > 0:
        from .paths import load_provider_module
        corefmt = load_provider_module("corefmt", paths)
        episodic = [it for it in ctx.live.inbox_range(0, pre.snapshot.max_inbox_id, ops=["core_add"])
                    if corefmt.is_episodic(it.text or "")]

    # ── N2 sanitize + windows ──
    texts = recent + [(m.key, m.text) for src in md_sources for m in src.messages]
    repeat_lines = sanitize.build_repeat_lines(texts, min_chars=int(cfg.repeat_line_min_chars),
                                               min_msgs=int(cfg.repeat_line_min_msgs))
    srep = sanitize.SanitizeReport()
    scratch = sanitize.SanitizeReport()        # context blocks: sanitized, not double-counted
    built: list[Window] = []
    for lin in lineages:
        msgs = sanitize.sanitize_messages(list(lin.messages), cfg=cfg, repeat_lines=repeat_lines,
                                          report=srep)
        ctx_msgs = sanitize.sanitize_messages(list(lin.context_before), cfg=cfg,
                                              repeat_lines=repeat_lines, report=scratch)
        built += windows_mod.build_windows(source="statedb", root=lin.root, platform=lin.platform,
                                           title=lin.title, messages=msgs, context_before=ctx_msgs,
                                           cfg=cfg, ref_ts=ctx.now)
        rep.inputs.append({"source": "statedb", "root": lin.root, "title": lin.title,
                           "messages": len(lin.messages)})
    for src in md_sources:
        msgs = sanitize.sanitize_messages(list(src.messages), cfg=cfg, repeat_lines=repeat_lines,
                                          report=srep)
        ctx_msgs = sanitize.sanitize_messages(list(getattr(src, "context_before", None) or []),
                                              cfg=cfg, repeat_lines=repeat_lines, report=scratch)
        built += windows_mod.build_windows(source="md", root=src.path, platform="md",
                                           title=src.slug, messages=msgs, context_before=ctx_msgs,
                                           cfg=cfg, ref_ts=ctx.now, end_offset=src.end_offset,
                                           md=src)
        rep.inputs.append({"source": "md", "root": src.path, "title": src.slug,
                           "messages": len(src.messages)})
    inbox_window_ids: dict[str, int] = {}
    for it in episodic:
        w = _inbox_window(it, ctx=ctx, windows_mod=windows_mod, sanitize_mod=sanitize,
                          repeat_lines=repeat_lines, report=srep)
        built.append(w)
        inbox_window_ids[w.window_id] = int(it.id)
        rep.inputs.append({"source": "inbox", "root": w.root, "title": w.title, "messages": 1})

    order = _processing_order(built)

    # ── N3–N5 ──
    failed_prev = _failed_attempts(ctx.ledger)
    states: dict[str, WindowState] = {}        # decided this run → ledger delta
    done_before: dict[str, WindowState] = {}   # already committed (same window id) → no LLM
    processed: list[tuple[Window, list[Claim], int, list[Rejection]]] = []   # (window, claims, n_extracted, rejections)
    deferred: list[Window] = []
    blocked: set[str] = set()
    stop = False
    llm_windows = 0
    max_windows = int(cfg.max_windows_per_run)
    max_attempts = int(cfg.window_max_attempts)

    for w in order:
        if stop or w.root in blocked:
            deferred.append(w)
            blocked.add(w.root)
            continue
        w.attempts = failed_prev.get((w.source, w.root, int(w.first_id)), 0)
        prev = ctx.ledger.get_window(w.window_id)
        if prev is not None and prev.status in WM_ADVANCING_STATUSES:
            done_before[w.window_id] = prev
            stats.bump_reason("excluded", "window_already_done")
            continue
        if _content_chars(w) < int(cfg.window_min_user_chars):
            states[w.window_id] = _state(w, "empty", run_id=ctx.run_id, attempts=w.attempts)
            continue
        if llm_windows >= max_windows or ctx.budget.expired():
            stop = True
            deferred.append(w)
            blocked.add(w.root)
            continue
        llm_windows += 1
        try:
            res = _extract.extract_window(w, llm=ctx.llm, cfg=cfg)
        except BudgetExceeded as e:
            ctx.note(f"예산 소진({e.what})으로 남은 창은 다음 실행으로 미룹니다.")
            stop = True
            deferred.append(w)
            blocked.add(w.root)
            continue
        if res.status == "ok":
            claims, rejs = _gates.gate_claims(res.claims, w, scanner=ctx.scanner, cfg=cfg, now=ctx.now)
            for c in claims:
                _normalize.normalize_claim(c, cfg=cfg, paths=paths)
            states[w.window_id] = _state(w, "ok", run_id=ctx.run_id, attempts=w.attempts,
                                         n_claims=len(claims))
            processed.append((w, claims, len(res.claims) + len(res.schema_rejections),
                              list(res.schema_rejections) + rejs))
            continue
        attempts = w.attempts + 1
        if res.raw:     # bad model output → Dream Log only (redacted, short)
            from .threat import redact_secrets
            snippet = " ".join(redact_secrets(res.raw)[0].split())[:300]
            ctx.note(f"추출 실패한 창의 모델 출력 앞부분 ({_short(w.root)}): {snippet}")
        if attempts >= max_attempts:
            states[w.window_id] = _state(w, "quarantined", run_id=ctx.run_id, attempts=attempts,
                                         error=res.error)
            ctx.alert("window_quarantined",
                      f"추출이 {attempts}회 실패한 창을 격리하고 넘어갑니다: {w.source} "
                      f"{_short(w.root)} ({w.first_id}–{w.last_id}). 원문은 그대로 남아 있습니다.",
                      window_id=w.window_id, source=w.source, root=w.root,
                      first_id=int(w.first_id), last_id=int(w.last_id),
                      error=(res.error or "")[:200])
        else:
            states[w.window_id] = _state(w, "failed", run_id=ctx.run_id, attempts=attempts,
                                         error=res.error)
            blocked.add(w.root)                 # the watermark cannot pass a failed window
            ctx.note(f"추출 실패({attempts}/{max_attempts}): {w.source} {_short(w.root)} "
                     f"({w.first_id}–{w.last_id}) — {(res.error or '')[:200]}")

    # ── N6 embed (one batch; respect the embed budget by deferring trailing windows) ──
    def _undo_last() -> None:
        w = processed.pop()[0]
        states.pop(w.window_id, None)
        deferred.append(w)
        blocked.add(w.root)

    b = ctx.budget
    while processed and b.embed_inputs + sum(len(p[1]) for p in processed) > b.max_embed_inputs:
        _undo_last()
    claims_all = [c for p in processed for c in p[1]]
    if claims_all:
        inputs = [embed_input(c.subject, c.text) for c in claims_all]
        try:
            vecs = ctx.embedder.embed(inputs)    # EmbedError / EmbedAuthError propagate (no writes)
        except BudgetExceeded as e:
            ctx.note(f"임베딩 예산 소진({e.what}): 이번 실행에서 추출한 창은 다음 실행에서 다시 처리합니다.")
            while processed:
                _undo_last()
            claims_all, vecs, inputs = [], [], []
        for c, t, v in zip(claims_all, inputs, vecs):
            c.embed_text = t
            c.vector = v
    rejections = [r for p in processed for r in p[3]]
    stats.claims_extracted += sum(p[2] for p in processed)

    # ── N7 order, watermarks, md offsets, stats, report ──
    claims_all.sort(key=lambda c: (c.event_time if c.event_time is not None else c.first_seen_at,
                                   c.origin_key))
    effective = {**done_before, **states}
    stat_windows = [w for w in order if w.source == "statedb"]
    wm = compute_watermarks(stat_windows, effective)
    for lin in lineages:
        ws = [w for w in stat_windows if w.root == lin.root]
        _, all_done = _advancing_prefix(ws, effective)
        if all_done and lin.messages:
            last_msg = lin.messages[-1]
            last = (float(last_msg.ts), int(last_msg.msg_id or 0))
            cur = wm.get(lin.root)
            if cur is None or last > cur:
                wm[lin.root] = last         # messages sanitized away after the last window
    wm_delta: dict[str, tuple[float, int]] = {}
    for root, (ts, mid) in wm.items():
        old = ctx.ledger.get_wm(root)
        if old is None or (ts, mid) > (float(old.last_ts), int(old.last_id)):
            wm_delta[root] = (ts, mid)

    md_states: list[MdFileState] = []
    for src in md_sources:
        ws = [w for w in order if w.source == "md" and w.root == src.path]
        last, all_done = _advancing_prefix(ws, effective)
        if all_done:
            pb = int(src.end_offset)
        elif last is not None:
            pb = int(last.last_id)
        else:
            continue
        if pb <= int(src.start_offset):
            continue
        status = "ok" if pb >= int(src.end_offset) else "partial"
        md_states.append(markdown.file_state_after(src, pb, ctx.run_id, status=status))

    inbox_ids = sorted(iid for wid, iid in inbox_window_ids.items()
                       if wid in effective and effective[wid].status in WM_ADVANCING_STATUSES)

    for st in states.values():
        stats.bump({"ok": "windows_ok", "empty": "windows_empty", "failed": "windows_failed",
                    "quarantined": "windows_quarantined"}[st.status])
    stats.windows_deferred += len(deferred)
    stats.windows_total += len(states) + len(deferred)
    stats.claims_rejected += len(rejections)
    for r in rejections:
        stats.bump_reason("rejected_by_reason", r.reason)
    for c in claims_all:
        rep.claims.append({"origin_key": c.origin_key, "kind": c.kind, "subject": c.subject,
                           "text": c.text, "status": c.status})
    rep.rejections += [asdict(r) for r in rejections]
    try:
        rep.strip_lines_top = list(srep.top_repeated(10))
    except Exception:  # noqa: BLE001 — report only
        pass
    if deferred:
        ctx.note(f"미뤄진 창 {len(deferred)}개(다음 실행에서 이어서 처리).")

    return NremResult(claims=claims_all, windows=order, window_states=states,
                      rejections=rejections, wm_delta=wm_delta, session_roots=session_roots,
                      md_states=md_states, deferred_windows=len(deferred),
                      inbox_episodic_ids=inbox_ids, sanitize_report=srep)


__all__ = ["PreflightResult", "NremResult", "preflight", "run_nrem", "compute_watermarks"]
