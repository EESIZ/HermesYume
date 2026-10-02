"""R4: fold provider recall events into rows (PLAN-v2 §4.3 R4, §5.4).

- only platforms in ``reinforce_platforms`` count (cron etc. → ignored; a daily cron must not keep a
  memory alive forever)
- ``injected``: once per (memory, session, KST date); counts and ``last_recalled_at`` only — never
  touches the strength clock (t_ref); cos ≥ ``injected_strong_cos`` → ``recall_injected_strong``
- ``used``: ``recall_used_count``; ``tool_hit``: ``search_hit_count`` only (counted once in the
  strength reinforcement r, DEVIATIONS F-18); both set ``last_used_at`` (moves t_ref) and revive a
  dormant row
- the injected (memory, session, KST date) key also checks events folded by earlier runs
  (``prior_events``), so a date split by the 04:40 run is not counted twice (F-33)
- ``shadow``: counted, cos collected for calibration
Events with id > the N0 snapshot wait for the next run (``new_cursor = snapshot_max_id``).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .clock import kst_date
from .strength import strength as _strength
from .types import CANDIDATE_SEARCH_EXCLUDED, LEGACY_KIND, RecallEvent


@dataclass
class FoldResult:
    new_cursor: int
    injected: int = 0
    used: int = 0
    tool_hit: int = 0
    shadow: int = 0
    ignored_platform: int = 0
    unknown_ids: int = 0
    revived: list[str] = field(default_factory=list)
    shadow_cos: list[float] = field(default_factory=list)


def resolve_memory_id(ws: Any, memory_id: str) -> str | None:
    """Lance id, or "inbox:<n>" → the row created/reinforced from that inbox item."""
    if not memory_id:
        return None
    if memory_id.startswith("inbox:"):
        row = ws.by_origin_key(memory_id)
        return row.id if row is not None else None
    return memory_id if ws.get(memory_id) is not None else None


def fold_recall_events(ctx: Any, ws: Any, events: list[RecallEvent], *,
                       snapshot_max_id: int, prior_events: list[RecallEvent] = ()) -> FoldResult:
    cfg = ctx.cfg
    allowed = set(cfg.reinforce_platforms)
    strong = float(cfg.injected_strong_cos)
    res = FoldResult(new_cursor=int(snapshot_max_id))
    inj_seen: set[tuple[str, str, str]] = set()
    for ev in prior_events or ():           # already counted by an earlier run
        if ev.kind != "injected" or ev.platform not in allowed:
            continue
        mid = resolve_memory_id(ws, ev.memory_id)
        if mid:
            inj_seen.add((mid, ev.session_id or "", kst_date(ev.ts)))
    agg: dict[str, dict[str, Any]] = defaultdict(lambda: {"inj": 0, "strong": 0, "last_rec": None,
                                                          "used": 0, "hits": 0, "last_used": None,
                                                          "kinds": set()})
    for ev in sorted(events, key=lambda e: e.id):
        if ev.id > snapshot_max_id:
            continue
        if ev.platform not in allowed:
            res.ignored_platform += 1
            continue
        if ev.kind == "shadow":
            res.shadow += 1
            if ev.cos is not None:
                res.shadow_cos.append(float(ev.cos))
            continue
        mid = resolve_memory_id(ws, ev.memory_id)
        row = ws.get(mid) if mid else None
        if row is None or row.status in CANDIDATE_SEARCH_EXCLUDED:
            res.unknown_ids += 1
            continue
        a = agg[mid]
        if ev.kind == "injected":
            key = (mid, ev.session_id or "", kst_date(ev.ts))
            if key in inj_seen:
                continue
            inj_seen.add(key)
            res.injected += 1
            a["inj"] += 1
            if ev.cos is not None and float(ev.cos) >= strong:
                a["strong"] += 1
            a["last_rec"] = max(a["last_rec"] or 0.0, float(ev.ts))
            a["kinds"].add("injected")
        elif ev.kind in ("used", "tool_hit"):
            if ev.kind == "used":
                res.used += 1
                a["used"] += 1
            else:
                res.tool_hit += 1
                a["hits"] += 1
            a["last_used"] = max(a["last_used"] or 0.0, float(ev.ts))
            a["kinds"].add(ev.kind)
        else:
            res.unknown_ids += 1
    for mid in sorted(agg):
        a = agg[mid]
        row = ws.get(mid)
        ch: dict[str, Any] = {}
        if a["inj"]:
            ch["recall_injected_count"] = int(row.recall_injected_count or 0) + a["inj"]
            ch["last_recalled_at"] = max(float(row.last_recalled_at or 0.0), a["last_rec"])
            if a["strong"]:
                ch["recall_injected_strong"] = int(row.recall_injected_strong or 0) + a["strong"]
        used_kind = None
        if a["used"] or a["hits"]:
            if a["used"]:
                ch["recall_used_count"] = int(row.recall_used_count or 0) + a["used"]
            if a["hits"]:
                ch["search_hit_count"] = int(row.search_hit_count or 0) + a["hits"]
            ch["last_used_at"] = max(float(row.last_used_at or 0.0), a["last_used"])
            used_kind = "used" if a["used"] else "tool_hit"
        reason = "recall:" + "+".join(sorted(a["kinds"]))
        ws.update(mid, ch, op="reinforce", reason=reason, user_evidence=bool(a["used"] or a["hits"]),
                  detail={"injected": a["inj"], "used": a["used"], "tool_hit": a["hits"]})
        cur = ws.get(mid)
        if used_kind and cur is not None and cur.status == "dormant" and cur.kind != LEGACY_KIND:
            if ws.update(mid, {"status": "active"}, op="status", reason=f"revived:{used_kind}",
                         user_evidence=True) is not None:
                res.revived.append(mid)
                ctx.stats.revived += 1
                ctx.report.revived.append({"id": mid, "text": cur.text,
                                           "strength": round(_strength(cur, ctx.now), 4)})
    st = ctx.stats
    st.recall_injected += res.injected
    st.recall_used += res.used
    st.recall_tool_hit += res.tool_hit
    st.recall_shadow += res.shadow
    st.recall_ignored_platform += res.ignored_platform
    top = sorted(agg.items(), key=lambda kv: (-(kv[1]["inj"] + kv[1]["used"] + kv[1]["hits"]), kv[0]))[:10]
    for mid, a in top:
        row = ws.get(mid)
        if row is not None:
            ctx.report.recall_top.append({"id": mid, "text": row.text, "injected": a["inj"],
                                          "used": a["used"] + a["hits"]})
    return res
