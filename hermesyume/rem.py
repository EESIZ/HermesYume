"""REM assembly (R0–R8) and the whole-run orchestration ``run_dream`` (PLAN-v2 §4.3, CONTRACTS §3,
§4.17).

    run_dream: nrem.preflight (N0) → nrem.run_nrem (N1–N7) → run_rem → post_commit
    run_rem:   R0 upsert claims → R1 inbox → R2 re-judge → R3 sweep → R4 recall fold →
               R5 core check (read-only) → R6 time transitions → secret re-scan → N8 doc refs →
               R7 guard → plan → R8 1–7 (dry-run: plan.json only)
    post_commit: docs/yume → serving export → alerts → Dream Log → alerts.log (+Telegram .txt)

MEMORY.md / USER.md are only ever read here (T14).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from . import core_check, docs_writer, plan as _plan, recall_fold, strength
from .clock import DAY, parse_iso, real_now
from .types import (FOLD_CURSOR_INBOX, FOLD_CURSOR_RECALL, KINDS, PIN_KINDS, WM_ADVANCING_STATUSES,
                    Claim, CommitResult, InboxItem, LedgerDelta, Plan, RunStats, SuppressRow,
                    Transition, UpsertOutcome, suppress_reason, text_sha)
from .upsert import Upserter, row_from_claim

log = logging.getLogger("hermesyume.rem")
REMEMBER_IMPORTANCE_FLOOR = 0.80
_INSERT_ACTIONS = frozenset({"inserted", "judge_pending", "state_change", "inserted_superseded", "related"})


@dataclass
class RemResult:
    plan: Plan
    commit: CommitResult | None
    outcomes: list[UpsertOutcome] = field(default_factory=list)
    status: str = "committed"          # committed | held | dry | noop


# ── R1 inbox ─────────────────────────────────────────────────────────────────

def _corefmt():
    from .paths import load_provider_module
    return load_provider_module("corefmt")


def _deadline(kind: str, ts: float, vu: float | None, cfg: Any) -> float | None:
    if kind == "state":
        return vu if vu is not None else ts + float(cfg.state_default_ttl_days) * DAY
    if kind == "schedule":
        return vu if vu is not None else ts
    return vu


def _is_expired(kind: str, vu: float | None, now: float, cfg: Any) -> bool:
    if vu is None or kind not in ("state", "schedule"):
        return False
    grace = float(cfg.schedule_grace_days) * DAY if kind == "schedule" else 0.0
    return now > vu + grace


def _unsafe(ctx: Any, text: str) -> bool:
    sc = ctx.scanner
    return bool(sc is not None and (sc.secrets(text) or sc.threats(text, "strict")))


def _remember_claim(ctx: Any, item: InboxItem, kind: str, subject: str) -> Claim:
    from . import normalize
    from .gates import anchor_present
    cfg, now = ctx.cfg, float(ctx.now)
    meta = item.meta or {}
    raw_vu = meta.get("valid_until")
    vu = parse_iso(str(raw_vu), end_of_day=True) if raw_vu else None
    ts = float(item.ts)
    vu = _deadline(kind, ts, vu, cfg)
    sid = item.session_id or "inbox"
    c = Claim(origin_key=f"inbox:{item.id}", source="tool:yume_remember", kind=kind, target="user",
              subject=subject, text=anchor_present((item.text or "").strip(), ts), level=3,
              explicit=True, event_time=ts,
              valid_until=vu, status="expired" if _is_expired(kind, vu, now, cfg) else "active",
              evidence_refs=[f"L#inbox:{item.id}"], evidence_keys=[f"i:{item.id}"],
              evidence_roles=["user"], session_ids=[sid], first_seen_at=ts, last_seen_at=ts,
              last_user_evidence_at=ts, user_evidence_count=1, user_session_count=1,
              explicit_user=True, pin=bool(item.pin) and kind in PIN_KINDS)
    normalize.normalize_claim(c, cfg=cfg, paths=ctx.paths)
    c.importance = round(max(float(c.importance or 0.0), REMEMBER_IMPORTANCE_FLOOR), 6)
    return c


def _claim_map(ctx: Any, items: list[InboxItem]) -> dict[int, Claim]:
    """item id → Claim for remember / non-episodic core_add / core_replace items, classified in one
    batched core_classify call and embedded in one batch."""
    cf = _corefmt()
    want: list[InboxItem] = []
    for it in items:
        text = (it.text or "").strip()
        if not text:
            continue
        if it.op == "remember":
            want.append(it)
        elif it.op in ("core_add", "core_replace") and it.target in ("memory", "user") \
                and not cf.is_episodic(text) and not core_check.is_fragment(text):
            want.append(it)
    if not want:
        return {}
    cls = core_check.classify_texts(ctx, [(it.text or "").strip() for it in want])
    out: dict[int, Claim] = {}
    for it, (kind, subj, _frag) in zip(want, cls):
        text = (it.text or "").strip()
        if it.op == "remember":
            k = it.kind if it.kind in KINDS else (kind or "fact")
            out[it.id] = _remember_claim(ctx, it, k, subj or text[:30])
        else:
            sha = cf.core_sha(text)
            out[it.id] = core_check.core_claim(ctx, target=it.target, text=text, kind=kind, subject=subj,
                                               origin_key=f"core:{it.target}:{sha}",
                                               evidence_key=f"i:{it.id}", ts=float(it.ts),
                                               session_id=it.session_id or "inbox")
    core_check.embed_claims(ctx, list(out.values()))
    return out


def inbox_claims(ctx: Any, items: list[InboxItem]) -> list[Claim]:
    return list(_claim_map(ctx, items).values())


def _pins_budget_ok(ctx: Any, ws: Any, text: str) -> bool:
    used = sum(len(r.text) for r in ws.rows.values()
               if r.pinned and r.status == "active" and not r.in_core)
    return used + len(text) <= int(ctx.cfg.pins_budget_chars)


def _forget_detail(reason: Any, text: str) -> str:
    """audit detail without the forgotten text (scrub a reason that quotes it)."""
    from .vecutil import fact_tokens
    r = str(reason or "").strip()[:80]
    if not r:
        return "reason="
    if fact_tokens(r) & fact_tokens(text):
        return "reason=[원문 일부 포함으로 생략]"
    return f"reason={r}"


def _new_pin(ctx: Any, ws: Any, mid: str) -> None:
    row = ws.get(mid)
    ctx.stats.pinned_new += 1
    ctx.report.new_pins.append({"id": mid, "text": row.text})
    ctx.note(f"새 pin: {row.text[:80]} (id {mid}). 해제는 `yume unpin {mid}`.")


def fold_inbox(ctx: Any, ws: Any, upserter: Upserter, items: list[InboxItem], *,
               episodic_done: set[int] | None = None) -> tuple[list[int], list[int]]:
    """R1 in inbox id order. Returns (consume_ids, skip_ids). Episodic core_add items are
    consumed only once their NREM window advanced (`episodic_done`); until then they stay pending."""
    cf = _corefmt()
    cfg, now, st, rep = ctx.cfg, float(ctx.now), ctx.stats, ctx.report
    items = sorted(items, key=lambda i: i.id)
    done_ep = set(episodic_done or ())
    consume: list[int] = []
    skip: list[int] = []
    todo = [it for it in items
            if not (it.op == "remember" and ws.by_origin_key(f"inbox:{it.id}") is not None)
            and not (it.op in ("remember", "core_add", "core_replace") and _unsafe(ctx, it.text or ""))]
    claims = _claim_map(ctx, todo)
    for it in items:
        text = (it.text or "").strip()
        if it.op == "remember":
            if ws.by_origin_key(f"inbox:{it.id}") is not None:
                consume.append(it.id)
                continue
            if not text or _unsafe(ctx, text):
                ctx.note(f"inbox {it.id}: 비어 있거나 비밀값/위협 패턴이 있어 저장하지 않았습니다.")
                skip.append(it.id)
                continue
            c = claims[it.id]
            if c.pin and not _pins_budget_ok(ctx, ws, c.text):
                c.pin = False
                ctx.note(f"pin 예산({cfg.pins_budget_chars}자) 초과로 고정하지 않았습니다 (inbox {it.id}). "
                         f"보호(durable)는 유지됩니다.")
            out = upserter.upsert(c)
            mid = out.memory_id
            if c.pin and mid:
                row = ws.get(mid)
                if mid in ws.new_ids and row is not None and row.pinned:
                    _new_pin(ctx, ws, mid)
                elif row is not None and not row.pinned and row.status == "active":
                    if ws.update(mid, {"pinned": True}, op="pin", reason="remember_pin",
                                 user_evidence=True) is not None:
                        _new_pin(ctx, ws, mid)
            consume.append(it.id)
        elif it.op == "forget":
            mid = it.memory_id or ""
            if mid.startswith("inbox:"):
                r = ws.by_origin_key(mid)
                mid = r.id if r is not None else ""
            row = ws.get(mid) if mid else None
            if row is None:
                skip.append(it.id)
                ctx.note(f"forget 대상 없음 (inbox {it.id})")
                continue
            meta = it.meta or {}
            if row.pinned and not bool(meta.get("confirm")):
                skip.append(it.id)
                ctx.note(f"pinned 기억은 confirm 없이 잊지 않습니다 (id {mid}).")
                continue
            o = ws.update(mid, {"status": "forgotten"}, op="forget", reason="forget", user_evidence=True)
            if o is not None and row.vector is not None:
                ws.add_suppress(SuppressRow(id=row.id, vector=row.vector.copy(), text_sha=text_sha(row.text),
                                            kind=row.kind, created_at=now,
                                            reason=suppress_reason(ctx.run_id, row.text)))
            if o is not None:
                ws.add_audit("forget", mid, _forget_detail(meta.get("reason"), row.text))
                st.forgotten += 1
                rep.forgotten.append({"id": mid})
            consume.append(it.id)
        elif it.op == "core_add":
            if cf.is_episodic(text):
                if it.id in done_ep:
                    consume.append(it.id)
                continue
            if not text or it.target not in ("memory", "user") or _unsafe(ctx, text):
                skip.append(it.id)
                continue
            if core_check.is_fragment(text):
                consume.append(it.id)
                continue
            c = claims[it.id]
            if core_check.find_core_row(ws, it.target, c.core_sha) is None:
                row = row_from_claim(c, ctx=ctx)
                ws.insert(row, reason="core_add", user_evidence=True)
                st.created += 1
                rep.created.append({"id": row.id, "kind": row.kind, "tier": row.tier, "text": row.text,
                                    "importance": row.importance})
                upserter.absorb_into_core_copy(row.id)
            st.core_changes += 1
            rep.core_changes.append({"target": it.target, "change": "add", "text": text})
            consume.append(it.id)
        elif it.op == "core_replace":
            if not text or it.target not in ("memory", "user") or _unsafe(ctx, text):
                skip.append(it.id)
                continue
            fresh_core_copy = None
            old_row = None
            if it.old_text:
                old_row = core_check.find_core_row(ws, it.target, cf.core_sha(it.old_text.strip()))
            new_row = None
            c = claims.get(it.id)
            if c is not None:
                new_row = core_check.find_core_row(ws, it.target, c.core_sha)
                if new_row is None:
                    new_row = row_from_claim(c, ctx=ctx)
                    ws.insert(new_row, reason="core_replace", user_evidence=True)
                    st.created += 1
                    rep.created.append({"id": new_row.id, "kind": new_row.kind, "tier": new_row.tier,
                                        "text": new_row.text, "importance": new_row.importance})
                    fresh_core_copy = new_row.id
            if old_row is not None and new_row is not None and old_row.id != new_row.id:
                if old_row.status in ("active", "dormant"):
                    o = ws.update(old_row.id, {"status": "superseded", "superseded_by": new_row.id,
                                               "valid_until": float(it.ts), "in_core": False},
                                  op="supersede", reason="core_replace", user_evidence=True)
                    if o is not None:
                        st.superseded += 1
                        rep.superseded.append({"old_id": old_row.id, "old_text": old_row.text,
                                               "new_id": new_row.id, "new_text": new_row.text})
                    cur_new = ws.get(new_row.id)
                    ws.update(new_row.id, {"supersedes": list(dict.fromkeys(cur_new.supersedes + [old_row.id]))},
                              op="text_update", reason="core_replace", user_evidence=True)
                if old_row.pinned:
                    ws.update(new_row.id, {"pinned": True}, op="pin", reason="core_replace", user_evidence=True)
                    ws.update(old_row.id, {"pinned": False}, op="unpin", reason="core_replace", user_evidence=True)
            if fresh_core_copy is not None:
                upserter.absorb_into_core_copy(fresh_core_copy)
            st.core_changes += 1
            rep.core_changes.append({"target": it.target, "change": "replace", "text": text})
            consume.append(it.id)
        elif it.op == "core_remove":
            row = core_check.find_core_row(ws, it.target or "", cf.core_sha(text)) if text else None
            if row is None and text:
                row = core_check.find_core_row(ws, "user", cf.core_sha(text)) or \
                    core_check.find_core_row(ws, "memory", cf.core_sha(text))
            if row is None:
                skip.append(it.id)
                continue
            if not row.in_core:         # already demoted (re-fold, or R5 mirrored it earlier): no change
                consume.append(it.id)
                continue
            core_check.demote_core_row(ctx, ws, row, at=float(it.ts), reason="core_remove")
            if row.core_required:
                fname = "USER.md" if (row.core_target or it.target) == "user" else "MEMORY.md"
                label = (cf.entry_label(row.text) or row.subject or row.text[:30]).strip("*: ").strip()
                state = "pinned로 남아 있음" if row.pinned else "활성 행으로 남아 있음"
                ctx.note(f"{fname}에서 '{label}'가 빠졌습니다. 장기기억에는 {state}. "
                         f"되돌리려면 `yume core-restore {row.id}`.")
            st.core_changes += 1
            rep.core_changes.append({"target": it.target or row.core_target or "", "change": "remove",
                                     "text": text})
            consume.append(it.id)
        elif it.op == "session_end":
            consume.append(it.id)
        else:
            skip.append(it.id)
    return consume, skip


# ── R6 + secret re-scan ──────────────────────────────────────────────────────

def time_transitions(ctx: Any, ws: Any) -> list[Transition]:
    """R6: expired / dormant / purge, computed by the pure strength functions and applied."""
    out: list[Transition] = []
    cfg, now, st, rep = ctx.cfg, float(ctx.now), ctx.stats, ctx.report
    for mid in sorted(list(ws.rows)):
        row = ws.rows.get(mid)
        if row is None:
            continue
        ev = strength.evaluate(row, now, cfg)
        tr = ev.transition
        if tr is None:
            continue
        if tr.to_status in ("expired", "dormant"):
            if ws.update(mid, {"status": tr.to_status}, op="status", reason=tr.reason) is None:
                continue
            st.bump(tr.to_status)
            getattr(rep, tr.to_status).append({"id": mid, "text": row.text,
                                               "strength": round(ev.strength, 4)})
        elif tr.to_status == "purge":
            reason = "quarantine" if row.status == "quarantined" else "forgotten+30d"
            ws.purge(mid, reason=reason)
            st.purged += 1
            rep.purged.append({"id": mid})
        out.append(tr)
    return out


def secret_rescan(ctx: Any, ws: Any) -> list[str]:
    """§10.2 nightly re-scan of stored rows: secret → quarantined → purged in the same run (a row
    inserted in this run is dropped before it is ever written)."""
    sc = ctx.scanner
    if sc is None:
        return []
    try:   # cheap necessary condition for threat.SECRET_PATTERNS (input builder)
        from .sanitize import may_contain_secret
    except ImportError:
        def may_contain_secret(_t: str) -> bool:
            return True
    hit: list[str] = []
    for mid in sorted(list(ws.rows)):
        row = ws.rows.get(mid)
        if row is None or row.status == "quarantined":
            continue
        text, subj = row.text or "", row.subject or ""
        if not (may_contain_secret(text) or may_contain_secret(subj)):
            continue
        types = sorted(set(sc.secrets(text)) | set(sc.secrets(subj)))
        if not types:
            continue
        hit.append(mid)
        if mid in ws.new_ids:
            ws.drop_new(mid)
            ctx.report.created = [c for c in ctx.report.created if c.get("id") != mid]
            ctx.stats.created = max(0, ctx.stats.created - 1)
        else:
            ws.update(mid, {"status": "quarantined"}, op="status", reason="quarantine:secret",
                      guard_exempt=True)
            ws.purge(mid, reason="quarantine:secret")
            ctx.stats.purged += 1
            ctx.report.purged.append({"id": mid})
        ctx.stats.quarantined += 1
        ctx.alert("secret_found",
                  f"저장 대상 기억에서 비밀값 패턴({', '.join(types)})을 발견해 격리하고 삭제했습니다. (id {mid})",
                  level="error", memory_id=mid, types=types)
    return hit


def _durable_unrecalled(ctx: Any, ws: Any) -> None:
    now = float(ctx.now)
    for row in sorted(ws.rows.values(), key=lambda r: r.id):
        if row.status != "active" or row.tier != "durable":
            continue
        last = row.last_recalled_at or row.last_used_at or row.created_at or now
        days = (now - float(last)) / DAY
        if days >= 365:
            ctx.report.durable_unrecalled.append({"id": row.id, "text": row.text, "days": int(days)})


# ── usage → stats ────────────────────────────────────────────────────────────

def _fill_usage(ctx: Any) -> None:
    st, cfg = ctx.stats, ctx.cfg
    lu = getattr(ctx.llm, "usage", None)
    eu = getattr(ctx.embedder, "usage", None)
    if lu is not None:
        st.llm_calls = int(getattr(lu, "content_calls", 0))
        st.prompt_tokens = int(getattr(lu, "prompt_tokens", 0))
        st.completion_tokens = int(getattr(lu, "completion_tokens", 0))
    if eu is not None:
        st.embed_inputs = int(getattr(eu, "content_inputs", 0))
        st.embed_tokens = int(getattr(eu, "tokens", 0))
    try:
        st.cost_usd = round(st.prompt_tokens / 1e6 * float(cfg.llm_price_in_per_mtok)
                            + st.completion_tokens / 1e6 * float(cfg.llm_price_out_per_mtok)
                            + st.embed_tokens / 1e6 * float(cfg.embed_price_per_mtok), 6)
    except (AttributeError, TypeError, ValueError):
        pass


# ── run_rem ──────────────────────────────────────────────────────────────────

def _episodic_done(ctx: Any, nres: Any, pending_ids: list[int]) -> set[int]:
    done: set[int] = set()
    states = getattr(nres, "window_states", {}) or {}
    for w in getattr(nres, "windows", []) or []:
        root = str(getattr(w, "root", ""))
        st = states.get(w.window_id)
        if root.startswith("inbox:") and st is not None and st.status in WM_ADVANCING_STATUSES:
            try:
                done.add(int(root.split(":", 1)[1]))
            except ValueError:
                pass
    if ctx.ledger is not None:
        for i in pending_ids:
            if i in done:
                continue
            ws_ = ctx.ledger.windows(root=f"inbox:{i}")
            if ws_ and all(w.status in WM_ADVANCING_STATUSES for w in ws_):
                done.add(i)
    return done


def run_rem(ctx: Any, nres: Any, pre: Any) -> RemResult:
    cfg, st, rep, now = ctx.cfg, ctx.stats, ctx.report, float(ctx.now)
    embed_model = getattr(ctx.embedder, "model_id", None) or cfg.embed_model_id()
    ws = _plan.WorkingSet(ctx.store.load_working_set(), run_id=ctx.run_id, now=now,
                          embed_model=embed_model)
    up = Upserter(ctx, ws)
    outcomes: list[UpsertOutcome] = []
    inserted: list[tuple[Claim, str]] = []

    # R0
    for c in getattr(nres, "claims", []) or []:
        out = up.upsert(c)
        outcomes.append(out)
        if out.action in _INSERT_ACTIONS and out.memory_id in ws.new_ids:
            inserted.append((c, out.memory_id))

    # R1
    cursors: dict[str, int] = {}
    snap = pre.snapshot
    live = ctx.live
    consume: list[int] = []
    skip: list[int] = []
    if live is not None:
        cur = ctx.ledger.get_cursor(FOLD_CURSOR_INBOX) if ctx.ledger is not None else 0
        if snap.max_inbox_id < cur:
            ctx.note("live.db inbox id가 원장 커서보다 작습니다(live.db 재생성?). 커서를 0부터 다시 셉니다.")
            cur = 0
        items = live.inbox_range(cur, snap.max_inbox_id)
        leftovers = live.inbox_range(0, cur) if cur > 0 else []   # pending ≤ cursor: crash leftovers / deferred episodes
        all_items = sorted({i.id: i for i in leftovers + items}.values(), key=lambda i: i.id)
        st.inbox_items = len(items)
        ep_ids = [i.id for i in all_items if i.op == "core_add"]
        consume, skip = fold_inbox(ctx, ws, up, all_items,
                                   episodic_done=_episodic_done(ctx, nres, ep_ids))
        led_cur = ctx.ledger.get_cursor(FOLD_CURSOR_INBOX) if ctx.ledger is not None else 0
        if snap.max_inbox_id != led_cur:
            cursors[FOLD_CURSOR_INBOX] = int(snap.max_inbox_id)

    # R2, R3
    outcomes += up.rejudge_pending(limit=int(cfg.rejudge_max))
    outcomes += up.sweep(days=float(cfg.sweep_days), min_cos=float(cfg.sweep_cos),
                         max_pairs=int(cfg.sweep_max_pairs))

    # R4
    if live is not None:
        rcur = ctx.ledger.get_cursor(FOLD_CURSOR_RECALL) if ctx.ledger is not None else 0
        base_cur = rcur
        if snap.max_recall_id < rcur:
            ctx.note("live.db recall_events id가 원장 커서보다 작습니다(live.db 재생성?). 커서를 0부터 다시 셉니다.")
            rcur = 0
        events = live.recall_range(rcur, snap.max_recall_id)
        prior = []
        if events and rcur > 0 and hasattr(live, "injected_before"):
            since = min(e.ts for e in events) - DAY        # same KST date as the oldest new event
            prior = live.injected_before(rcur, since)
        fold = recall_fold.fold_recall_events(ctx, ws, events, snapshot_max_id=snap.max_recall_id,
                                              prior_events=prior)
        if fold.new_cursor != base_cur:
            cursors[FOLD_CURSOR_RECALL] = int(fold.new_cursor)
        if fold.unknown_ids:
            ctx.note(f"회상 이벤트 중 대상 행을 찾지 못한 것 {fold.unknown_ids}건")

    # R5 (read-only on core files)
    from .sources.core_files import read_core
    ccr = core_check.check_core(ctx, ws, up, read_core(ctx.paths))

    # R6 + secret re-scan
    time_transitions(ctx, ws)
    secret_rescan(ctx, ws)
    from . import retention
    ctx.purged = retention.purged_info(ws)       # texts to scrub outside Lance after commit (F-5)

    # N8 doc refs (files are written after commit)
    windows = {w.window_id: w for w in getattr(nres, "windows", []) or []}
    docs = docs_writer.plan_docs(ctx, [(c, m) for c, m in inserted if m in ws.rows], windows)
    for d in docs:
        row = ws.get(d.memory_id) if d.memory_id else None
        ref = docs_writer.doc_ref(d.slug)
        if row is not None and ref not in row.refs:
            ws.update(row.id, {"refs": list(row.refs) + [ref]}, op="text_update", reason="doc_ref")

    _durable_unrecalled(ctx, ws)

    # R7
    guard = _plan.apply_guard(ws, cfg, mode=ctx.mode)
    st.held_ops = len(guard.held_seqs)
    for o in ws.ops:
        if o.held:
            rep.held.append({"seq": o.seq, "op": o.op, "memory_id": o.memory_id, "reason": o.reason})
    if guard.held:
        ctx.note(f"pinned/durable 행을 사용자 근거 없이 바꾸려는 연산 {len(guard.held_seqs)}건을 "
                 f"적용하지 않고 기록만 했습니다.")
    if guard.destructive_count > guard.threshold:
        ctx.note(f"이번 실행의 비활성 전이 {guard.destructive_count}건 (참고 기준 {guard.threshold}건). "
                 f"휴면 행은 검색으로 되살릴 수 있습니다.")

    delta = LedgerDelta(watermarks=dict(getattr(nres, "wm_delta", {}) or {}),
                        session_roots=dict(getattr(nres, "session_roots", {}) or {}),
                        windows=list((getattr(nres, "window_states", {}) or {}).values()),
                        md_files=list(getattr(nres, "md_states", []) or []),
                        cursors=cursors, core_seen=list(ccr.core_seen), audit=list(ws.audit))
    st.lance_version_before = pre.lance_version_before
    _fill_usage(ctx)
    p = _plan.build_plan(ctx, ws, lance_version_before=pre.lance_version_before, ledger_delta=delta,
                         inbox_consume_ids=consume, inbox_skip_ids=skip, docs=docs)
    if ctx.dry_run:
        p.status = "dry"
        _plan.write_plan(ctx.paths, p)
        return RemResult(plan=p, commit=None, outcomes=outcomes, status="dry")
    res = _plan.commit_plan(ctx, p)
    st.lance_version_after = res.lance_version_after
    status = "held" if guard.held else ("noop" if res.skipped else "committed")
    return RemResult(plan=p, commit=res, outcomes=outcomes, status=status)


# ── R8 8–10 ──────────────────────────────────────────────────────────────────

def _write_dream_log(ctx: Any, plan: Plan | None, *, status: str, error: str | None = None) -> None:
    from . import dream_log
    text = dream_log.render(ctx, plan, status=status, error=error)
    dream_log.write(ctx.paths, ctx.now, text, dry=bool(ctx.dry_run))


def _merge_alerts(ctx: Any, alerts: Any) -> None:
    for a in alerts or []:
        if not any(a is x for x in ctx.alerts):
            ctx.alerts.append(a)


def post_commit(ctx: Any, plan: Plan) -> None:
    """R8 8–10. Dry-run: Dream Log `_dry` only (alerts listed inside, never sent)."""
    if ctx.dry_run:
        _write_dream_log(ctx, plan, status="dry")
        return
    if plan is not None and plan.docs:
        for d, ok, err in docs_writer.write_docs(plan.docs, dry_run=False,
                                                 hermes_home=ctx.paths.hermes_home):
            if ok:
                ctx.stats.docs_written += 1
            else:
                ctx.note(f"절차 문서 쓰기 실패: {d.slug}.md ({err}). 참조는 export에서 빠집니다.")
    from . import alerts as _alerts, export, retention
    export.build_serving(ctx)
    if plan is not None:
        _plan.mark_inbox(ctx, plan)                 # R8 6, after the new serving copy is in place
        retention.after_commit(ctx, plan)
    _merge_alerts(ctx, _alerts.evaluate_run_alerts(ctx, plan))
    _merge_alerts(ctx, _alerts.evaluate_health(ctx, now=ctx.now))
    status = plan.status if plan is not None else "committed"
    _write_dream_log(ctx, plan, status=status)
    _alerts.AlertSink(ctx.paths, ctx.cfg).emit_all(ctx.alerts)
    _alerts.flush_pending(ctx.paths, ctx.cfg, now=ctx.now)


# ── run_dream ────────────────────────────────────────────────────────────────

def _safe_msg(e: BaseException) -> str:
    try:
        from .threat import redact_secrets
        return redact_secrets(str(e))[0][:300]
    except Exception:
        return type(e).__name__


def _failure(e: BaseException) -> tuple[str, str]:
    from .embedder import EmbedAuthError
    from .ledger import MetaMismatch
    from .llm import LLMAuthError
    from .store import SchemaMismatch
    from .threat import ThreatScannerUnavailable
    msg = _safe_msg(e)
    if isinstance(e, (LLMAuthError, EmbedAuthError)):
        return "auth_401", ("OpenAI 인증 실패(401/403)로 이번 실행을 쓰기 없이 중단했습니다. "
                            ".env의 OPENAI_API_KEY를 확인하세요.")
    if isinstance(e, (SchemaMismatch, MetaMismatch)):
        return "model_mismatch", f"임베딩 모델/차원 또는 Lance 스키마 불일치로 중단했습니다: {msg} (모델 변경은 yume reembed)"
    if isinstance(e, ThreatScannerUnavailable):
        return "scanner_unavailable", "위협 스캐너를 불러오지 못해 실행을 중단했습니다(fail-closed)."
    return "run_failed", f"dream 실행 실패: {type(e).__name__}: {msg}"


def _failure_outputs(ctx: Any, plan: Plan | None, status: str, error: str | None) -> None:
    try:
        _write_dream_log(ctx, plan, status=status, error=error)
    except Exception as e:
        log.error("dream log failed: %s", type(e).__name__)
    if ctx.dry_run:
        return
    try:
        from . import alerts as _alerts
        _alerts.AlertSink(ctx.paths, ctx.cfg).emit_all(ctx.alerts)
        _alerts.flush_pending(ctx.paths, ctx.cfg, now=ctx.now)
    except Exception as e:
        log.error("alert emit failed: %s", type(e).__name__)


def run_dream(ctx: Any) -> RunStats:
    """Whole run. Exceptions never escape: failures become an alert, a failed run row, a Dream
    Log and ``stats.status == "failed"`` (CLI exit 1); held → "held" (exit 3)."""
    t0 = time.monotonic()
    st = ctx.stats
    st.run_id, st.mode = ctx.run_id, ctx.mode
    plan: Plan | None = None
    status, error = "failed", None
    try:
        if ctx.scanner is None:
            from . import threat
            ctx.scanner = threat.load_scanner(ctx.cfg.hermes_runtime_dir)
        from . import nrem
        pre = nrem.preflight(ctx)
        st.lance_version_before = pre.lance_version_before
        nres = nrem.run_nrem(ctx, pre)
        rres = run_rem(ctx, nres, pre)
        plan = rres.plan
        status = "committed" if rres.status == "noop" else rres.status
    except Exception as e:
        code, message = _failure(e)
        error = f"{type(e).__name__}: {_safe_msg(e)}"
        log.error("dream failed: %s", error)
        if not any(a.code == code for a in ctx.alerts):     # a stage may already have raised it
            ctx.alert(code, message, level="error", error_class=type(e).__name__)
        status = "failed"
        if not ctx.dry_run and ctx.ledger is not None and not getattr(ctx.ledger, "readonly", False):
            try:
                cur = ctx.ledger.get_run(ctx.run_id)
                if cur is not None and cur.status == "planned":
                    # plan.json is durable and R8 started: keep 'planned' so N0 of the next run replays it
                    ctx.ledger.update_run(ctx.run_id, error=error)
                else:
                    ctx.ledger.update_run(ctx.run_id, status="failed", error=error, finished_at=real_now())
            except Exception as e2:
                log.error("could not mark run failed: %s", type(e2).__name__)
    _fill_usage(ctx)
    st.duration_s = round(time.monotonic() - t0, 3)
    if status != "failed":
        st.status = status
        try:
            post_commit(ctx, plan)
        except Exception as e:
            error = f"{type(e).__name__}: {_safe_msg(e)}"
            log.error("post_commit failed: %s", error)
            ctx.alert("run_failed", f"커밋 후 단계(export/Dream Log) 실패: {error}", level="error",
                      error_class=type(e).__name__)
            status = "failed"
            _failure_outputs(ctx, plan, status, error)
    else:
        st.status = status
        _failure_outputs(ctx, plan, status, error)
    st.status = status
    st.duration_s = round(time.monotonic() - t0, 3)
    try:
        if ctx.store is not None:
            st.lance_version_after = ctx.store.version()
    except Exception:
        pass
    return st
