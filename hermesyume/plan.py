"""Working set, op log, guard (R7), plan.json and the R8 commit/replay protocol (PLAN-v2 §4.3).

Every change of a run is an ``Op`` recorded on a ``WorkingSet``: ops apply to the working copy
immediately (R0: later decisions see earlier ones), and the final Lance rows are rebuilt by
replaying the non-held ops onto the committed base (``finalize``). A ``Plan`` is written
atomically to ``runs/<run_id>/plan.json`` *before* any Lance write; a crash anywhere in R8 2–7 is
repaired by ``replay_planned`` (every step is idempotent: merge_insert by key, absolute row
states, ledger UPSERTs).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import strength as _strength
from .types import (CANDIDATE_SEARCH_EXCLUDED, DESTRUCTIVE_TARGET_STATUSES, AuditRow,
                    CommitResult, CoreSeenRow, DocWrite, GuardResult, HistoryRow, LedgerDelta,
                    MdFileState, MemoryRow, Op, Plan, RunRecord, RunReport, RunStats, SuppressRow,
                    WindowState, make_history_id, vec_from_b64, vec_to_b64)

log = logging.getLogger("hermesyume.plan")

PLAN_VERSION = 1
PROTECTED_OPS = frozenset({"supersede", "consolidate", "text_update", "status", "unpin", "forget"})
# Changes to only these fields never count as changing a protected row's content (links/flags).
_BOOKKEEPING_FIELDS = frozenset({"related_ids", "judge_pending", "updated_at", "last_run_id",
                                 "version", "status_changed_at", "status_reason"})
_IMPLICIT_FIELDS = ("updated_at", "last_run_id")


def _vec_eq(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(np.array_equal(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))


def _jsonable(field: str, value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def is_destructive(before_status: str, after_status: str) -> bool:
    return before_status == "active" and after_status in DESTRUCTIVE_TARGET_STATUSES


class WorkingSet:
    """Committed rows (`base`, never mutated) + this run's working state (`rows`, copy-on-write)."""

    def __init__(self, base: dict[str, MemoryRow], *, run_id: str, now: float, embed_model: str):
        self.base = base
        self.rows: dict[str, MemoryRow] = dict(base)        # shares objects until first write
        self.run_id = run_id
        self.now = float(now)
        self.embed_model = embed_model
        self.new_ids: set[str] = set()
        self.ops: list[Op] = []
        self.new_suppress: list[SuppressRow] = []
        self.purge_ids: list[str] = []
        self.audit: list[AuditRow] = []                     # additive (forget/purge audit rows)
        self.guard: GuardResult | None = None
        self.vec_changed: set[str] = set()                  # base rows re-embedded this run
        self._copied: set[str] = set()
        self._seq = 0
        self._origin: dict[str, str] = {}
        self._subject: dict[str, set[str]] = defaultdict(set)
        for r in base.values():
            self._index(r)
        self._mat_ids: list[str] = []
        self._mat: np.ndarray | None = None
        self._mat_pos: dict[str, int] = {}
        self._mat_dirty = True

    # ── indexes ──
    def _index(self, row: MemoryRow) -> None:
        for k in row.origin_keys or ():
            self._origin[k] = row.id
        if row.subject_key:
            self._subject[row.subject_key].add(row.id)

    def _unindex(self, row: MemoryRow) -> None:
        for k in row.origin_keys or ():
            if self._origin.get(k) == row.id:
                del self._origin[k]
        if row.subject_key:
            self._subject.get(row.subject_key, set()).discard(row.id)

    def _writable(self, memory_id: str) -> MemoryRow:
        row = self.rows[memory_id]
        if memory_id not in self._copied and memory_id not in self.new_ids:
            row = row.copy()
            self.rows[memory_id] = row
            self._copied.add(memory_id)
        return row

    # ── lookups ──
    def get(self, memory_id: str) -> MemoryRow | None:
        return self.rows.get(memory_id)

    def by_origin_key(self, key: str) -> MemoryRow | None:
        mid = self._origin.get(key)
        return self.rows.get(mid) if mid else None

    def by_subject_key(self, key: str, *, exclude_statuses: Iterable[str] = CANDIDATE_SEARCH_EXCLUDED
                       ) -> list[MemoryRow]:
        if not key:
            return []
        ex = set(exclude_statuses)
        out = [self.rows[i] for i in self._subject.get(key, ()) if i in self.rows]
        return sorted((r for r in out if r.status not in ex), key=lambda r: (r.created_at, r.id))

    def _matrix(self) -> tuple[list[str], np.ndarray]:
        if self._mat_dirty or self._mat is None:
            ids = [i for i, r in self.rows.items() if r.vector is not None]
            self._mat_ids = ids
            self._mat_pos = {i: n for n, i in enumerate(ids)}
            self._mat = (np.stack([np.asarray(self.rows[i].vector, dtype=np.float32) for i in ids])
                         if ids else np.zeros((0, 1), dtype=np.float32))
            if len(ids):
                n = np.linalg.norm(self._mat, axis=1, keepdims=True)
                n[n == 0] = 1.0
                self._mat = self._mat / n
            self._mat_dirty = False
        return self._mat_ids, self._mat

    def vector_search(self, vec: Any, *, k: int, min_cos: float = -1.0, only_new: bool = False,
                      exclude_statuses: Iterable[str] = CANDIDATE_SEARCH_EXCLUDED,
                      exclude_ids: Iterable[str] = ()) -> list[tuple[MemoryRow, float]]:
        """Cosine over the working state. only_new: rows inserted or re-embedded in this run."""
        if vec is None or k <= 0:
            return []
        q = np.asarray(vec, dtype=np.float32).reshape(-1)
        qn = float(np.linalg.norm(q))
        if qn == 0:
            return []
        q = q / qn
        ex_status = set(exclude_statuses)
        ex_ids = set(exclude_ids)
        if only_new:
            ids = [i for i in sorted(self.new_ids | self.vec_changed)
                   if i in self.rows and self.rows[i].vector is not None]
            if not ids:
                return []
            mat = np.stack([np.asarray(self.rows[i].vector, dtype=np.float32) for i in ids])
            n = np.linalg.norm(mat, axis=1, keepdims=True)
            n[n == 0] = 1.0
            sims = (mat / n) @ q
        else:
            ids, mat = self._matrix()
            if not ids:
                return []
            sims = mat @ q
        order = np.argsort(-sims, kind="stable")
        out: list[tuple[MemoryRow, float]] = []
        for j in order:
            c = float(sims[j])
            if c < min_cos:
                break
            mid = ids[int(j)]
            row = self.rows.get(mid)
            if row is None or mid in ex_ids or row.status in ex_status:
                continue
            out.append((row, c))
            if len(out) >= k:
                break
        return out

    def active_count(self) -> int:
        return sum(1 for r in self.base.values() if r.status == "active")

    # ── mutations ──
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def insert(self, row: MemoryRow, *, reason: str = "", user_evidence: bool = False,
               detail: dict | None = None) -> Op:
        if row.id in self.rows:
            raise ValueError(f"insert: id exists {row.id}")
        row.tier = _strength.compute_tier(row)
        row.last_run_id = self.run_id
        if not row.embed_model:
            row.embed_model = self.embed_model
        self.rows[row.id] = row
        self.new_ids.add(row.id)
        self._index(row)
        self._mat_dirty = True
        op = Op(seq=self._next_seq(), op="insert", memory_id=row.id,
                changes=row.snapshot(include_vector=True), before={}, reason=reason,
                user_evidence=user_evidence, detail=dict(detail or {}))
        self.ops.append(op)
        return op

    def update(self, memory_id: str, changes: dict[str, Any], *, op: str, reason: str = "",
               user_evidence: bool = False, guard_exempt: bool = False,
               detail: dict | None = None) -> Op | None:
        """Apply `changes` (field → new value; "vector" may be an array) now. Returns the Op, or
        None when nothing actually changes (or the row is gone)."""
        if memory_id not in self.rows:
            return None
        cur = self.rows[memory_id]
        diff: dict[str, Any] = {}
        for k, v in changes.items():
            if k in ("id", "version"):
                continue
            if k == "vector":
                if not _vec_eq(cur.vector, v):
                    diff[k] = None if v is None else np.asarray(v, dtype=np.float32).copy()
                continue
            if not hasattr(cur, k):
                raise KeyError(f"unknown memory field {k!r}")
            if getattr(cur, k) != v:
                diff[k] = list(v) if isinstance(v, (list, tuple)) else v
        if not diff:
            return None
        row = self._writable(memory_id)
        before_status = row.status
        before: dict[str, Any] = {}
        recorded: dict[str, Any] = {}
        if "status" in diff:
            diff.setdefault("status_changed_at", self.now)
            if "status_reason" not in changes:
                diff["status_reason"] = reason
        old_subject_key, old_origins = row.subject_key, list(row.origin_keys)
        for k, v in diff.items():
            old = getattr(row, k)
            if k == "vector":
                before["vector_b64"] = vec_to_b64(old)
                recorded["vector_b64"] = vec_to_b64(v)
                self.vec_changed.add(memory_id)
                self._mat_dirty = True
            else:
                before[k] = _jsonable(k, old)
                recorded[k] = _jsonable(k, v)
            setattr(row, k, v)
        new_tier = _strength.compute_tier(row)
        if new_tier != row.tier:
            before["tier"], recorded["tier"] = row.tier, new_tier
            row.tier = new_tier
        for k, v in (("updated_at", self.now), ("last_run_id", self.run_id)):
            if getattr(row, k) != v:
                before[k], recorded[k] = getattr(row, k), v
                setattr(row, k, v)
        row.version = int(row.version or 0) + 1
        if row.subject_key != old_subject_key or row.origin_keys != old_origins:
            tmp = MemoryRow(id=row.id, text="", subject_key=old_subject_key, origin_keys=old_origins)
            self._unindex(tmp)
            self._index(row)
        base_row = self.base.get(memory_id)
        new_row = memory_id in self.new_ids
        destructive = (not new_row and not guard_exempt
                       and is_destructive(before_status, row.status))
        content = set(recorded) - _BOOKKEEPING_FIELDS - {"tier"}
        protected = bool(base_row is not None and not new_row
                         and (base_row.pinned or base_row.tier == "durable")
                         and not user_evidence and not guard_exempt
                         and op in PROTECTED_OPS and content)
        o = Op(seq=self._next_seq(), op=op, memory_id=memory_id, changes=recorded, before=before,
               reason=reason, destructive=destructive, protected_change=protected,
               user_evidence=user_evidence, detail=dict(detail or {}))
        self.ops.append(o)
        return o

    def add_suppress(self, row: SuppressRow) -> None:
        if any(s.id == row.id for s in self.new_suppress):
            return
        self.new_suppress.append(row)

    def add_audit(self, op: str, memory_id: str | None, detail: str) -> None:
        self.audit.append(AuditRow(ts=self.now, run_id=self.run_id, op=op, memory_id=memory_id,
                                   detail=detail))

    def purge(self, memory_id: str, *, reason: str) -> None:
        """Hard delete at commit (forgotten+30d, quarantined). Audit only — no history row."""
        row = self.rows.pop(memory_id, None)
        if row is None:
            return
        self._unindex(row)
        self._mat_dirty = True
        self.new_ids.discard(memory_id)
        if memory_id in self.base and memory_id not in self.purge_ids:
            self.purge_ids.append(memory_id)
        self.add_audit("purge", memory_id, reason)

    def drop_new(self, memory_id: str) -> None:
        """Forget a row inserted in this run completely (e.g. secret found): no op is committed."""
        if memory_id not in self.new_ids:
            raise ValueError("drop_new: not a new row")
        row = self.rows.pop(memory_id)
        self._unindex(row)
        self._mat_dirty = True
        self.new_ids.discard(memory_id)
        self.ops = [o for o in self.ops if o.memory_id != memory_id]

    def touched_ids(self) -> list[str]:
        seen: list[str] = []
        for o in self.ops:
            if o.memory_id in self.rows and o.memory_id not in seen:
                seen.append(o.memory_id)
        return seen


# ── guard (R7, U2) ───────────────────────────────────────────────────────────

def apply_guard(ws: WorkingSet, cfg: Any, *, mode: str) -> GuardResult:
    """Holds ONLY ops changing a pinned/durable row without user evidence (U2). Mass
    dormant/expired is counted for the Dream Log but never held; inserts/reinforcements never."""
    held: list[int] = []
    for o in ws.ops:
        o.held = bool(o.protected_change)
        if o.held:
            held.append(o.seq)
    destructive = sum(1 for o in ws.ops if o.destructive and not o.held)
    if mode == "migrate":
        destructive = sum(1 for o in ws.ops if o.destructive and not o.held
                          and o.memory_id not in ws.new_ids)
    threshold = int(cfg.mass_change_threshold(ws.active_count()))
    reasons: list[str] = []
    if held:
        reasons.append(f"pinned/durable 행을 사용자 근거 없이 바꾸는 연산 {len(held)}건 보류")
    if destructive > threshold:
        reasons.append(f"비활성 전이 {destructive}건 (기준 {threshold}건 초과, 보류하지 않음)")
    ws.guard = GuardResult(held=bool(held), reasons=reasons, destructive_count=destructive,
                           threshold=threshold, held_seqs=held)
    return ws.guard


# ── finalize ─────────────────────────────────────────────────────────────────

def _apply_changes(row: MemoryRow, changes: dict[str, Any]) -> None:
    for k, v in changes.items():
        if k == "vector_b64":
            row.vector = vec_from_b64(v)
        elif k in ("id", "vector"):
            continue
        else:
            setattr(row, k, list(v) if isinstance(v, list) else v)


def _history_dict(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != "vector_b64"}


def _row_differs(a: MemoryRow, b: MemoryRow) -> bool:
    return (a.snapshot() != b.snapshot()) or not _vec_eq(a.vector, b.vector)


def finalize(ws: WorkingSet, *, held_seqs: set[int] | Iterable[int]) -> tuple[list[MemoryRow], list[HistoryRow]]:
    """Replay non-held ops onto copies of base → final rows (only those differing from base)
    + one HistoryRow per non-held op."""
    from .store import history_json
    held = set(held_seqs)
    final: dict[str, MemoryRow] = {}
    history: list[HistoryRow] = []
    order: list[str] = []
    for o in sorted(ws.ops, key=lambda x: x.seq):
        if o.seq in held:
            continue
        if o.op == "insert":
            row = MemoryRow.from_snapshot(o.changes)
            final[o.memory_id] = row
            after = _history_dict(row.snapshot())
            before: dict[str, Any] = {}
        else:
            row = final.get(o.memory_id)
            if row is None:
                base = ws.base.get(o.memory_id)
                if base is None:
                    continue
                row = base.copy()
                final[o.memory_id] = row
            _apply_changes(row, o.changes)
            row.version = int(row.version or 0) + 1
            after = _history_dict(o.changes)
            before = _history_dict(o.before)
        if o.memory_id not in order:
            order.append(o.memory_id)
        history.append(HistoryRow(history_id=make_history_id(ws.run_id, o.memory_id, o.op, o.seq),
                                  memory_id=o.memory_id, run_id=ws.run_id, op=o.op,
                                  before_json=history_json(before), after_json=history_json(after),
                                  at=ws.now))
    purged = set(ws.purge_ids)
    rows: list[MemoryRow] = []
    for mid in order:
        if mid in purged or mid not in final:
            continue
        r = final[mid]
        base = ws.base.get(mid)
        if base is None or _row_differs(r, base):
            rows.append(r)
    return rows, history


def build_plan(ctx: Any, ws: WorkingSet, *, lance_version_before: int | None,
               ledger_delta: LedgerDelta, inbox_consume_ids: list[int], inbox_skip_ids: list[int],
               docs: list[DocWrite]) -> Plan:
    guard = ws.guard or apply_guard(ws, ctx.cfg, mode=ctx.mode)
    rows, history = finalize(ws, held_seqs=set(guard.held_seqs))
    from . import clock
    return Plan(run_id=ctx.run_id, mode=ctx.mode, created_at=clock.real_now(), now=ctx.now,
                lance_version_before=lance_version_before, upserts=rows, history=history,
                suppress=list(ws.new_suppress), purge_ids=list(ws.purge_ids), ops=list(ws.ops),
                guard=guard, ledger_delta=ledger_delta,
                inbox_consume_ids=sorted(set(int(i) for i in inbox_consume_ids)),
                inbox_skip_ids=sorted(set(int(i) for i in inbox_skip_ids)), docs=list(docs),
                stats=ctx.stats, report=ctx.report,
                status="dry" if ctx.dry_run else "planned")


# ── plan.json ────────────────────────────────────────────────────────────────

def _op_to_json(o: Op) -> dict:
    return {"seq": o.seq, "op": o.op, "memory_id": o.memory_id, "changes": o.changes,
            "before": o.before, "reason": o.reason, "destructive": o.destructive,
            "protected_change": o.protected_change, "user_evidence": o.user_evidence,
            "held": o.held, "detail": o.detail}


def _suppress_to_json(s: SuppressRow) -> dict:
    return {"id": s.id, "vector_b64": vec_to_b64(s.vector), "text_sha": s.text_sha, "kind": s.kind,
            "created_at": s.created_at, "reason": s.reason}


def _delta_to_json(d: LedgerDelta) -> dict:
    return {"watermarks": {k: [float(v[0]), int(v[1])] for k, v in d.watermarks.items()},
            "session_roots": dict(d.session_roots),
            "windows": [asdict(w) for w in d.windows],
            "md_files": [asdict(m) for m in d.md_files],
            "cursors": {k: int(v) for k, v in d.cursors.items()},
            "core_seen": [asdict(c) for c in d.core_seen],
            "audit": [asdict(a) for a in d.audit]}


def _delta_from_json(d: dict) -> LedgerDelta:
    return LedgerDelta(
        watermarks={k: (float(v[0]), int(v[1])) for k, v in (d.get("watermarks") or {}).items()},
        session_roots=dict(d.get("session_roots") or {}),
        windows=[WindowState(**w) for w in d.get("windows") or []],
        md_files=[MdFileState(**m) for m in d.get("md_files") or []],
        cursors={k: int(v) for k, v in (d.get("cursors") or {}).items()},
        core_seen=[CoreSeenRow(**c) for c in d.get("core_seen") or []],
        audit=[AuditRow(**a) for a in d.get("audit") or []])


def _dc_from(cls: type, d: dict | None) -> Any:
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in (d or {}).items() if k in names})


def _redacted(obj: Any) -> Any:
    """Every string of the report with secret spans masked (a rejected claim or a core_remove
    text can carry one; plan.json must not keep it, §10.2)."""
    from .threat import redact_secrets
    try:
        from .sanitize import may_contain_secret
    except ImportError:          # pragma: no cover
        def may_contain_secret(_t: str) -> bool:
            return True
    if isinstance(obj, str):
        return redact_secrets(obj)[0] if may_contain_secret(obj) else obj
    if isinstance(obj, list):
        return [_redacted(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _redacted(v) for k, v in obj.items()}
    return obj


def plan_to_json(plan: Plan) -> dict:
    return {"plan_version": PLAN_VERSION, "run_id": plan.run_id, "mode": plan.mode,
            "status": plan.status, "created_at": plan.created_at, "now": plan.now,
            "lance_version_before": plan.lance_version_before,
            "upserts": [r.snapshot(include_vector=True) for r in plan.upserts],
            "history": [asdict(h) for h in plan.history],
            "suppress": [_suppress_to_json(s) for s in plan.suppress],
            "purge_ids": list(plan.purge_ids),
            "ops": [_op_to_json(o) for o in plan.ops],
            "guard": asdict(plan.guard),
            "ledger_delta": _delta_to_json(plan.ledger_delta),
            "inbox_consume_ids": list(plan.inbox_consume_ids),
            "inbox_skip_ids": list(plan.inbox_skip_ids),
            "docs": [asdict(d) for d in plan.docs],
            "stats": plan.stats.to_dict(), "report": _redacted(plan.report.to_dict())}


def plan_from_json(d: dict) -> Plan:
    if int(d.get("plan_version", PLAN_VERSION)) != PLAN_VERSION:
        raise ValueError(f"unsupported plan_version {d.get('plan_version')}")
    return Plan(
        run_id=d["run_id"], mode=d.get("mode", "live"), created_at=float(d.get("created_at") or 0.0),
        now=float(d.get("now") or 0.0), lance_version_before=d.get("lance_version_before"),
        upserts=[MemoryRow.from_snapshot(r) for r in d.get("upserts") or []],
        history=[HistoryRow(**h) for h in d.get("history") or []],
        suppress=[SuppressRow(id=s["id"], vector=vec_from_b64(s.get("vector_b64")),
                              text_sha=s["text_sha"], kind=s.get("kind", ""),
                              created_at=float(s.get("created_at") or 0.0), reason=s.get("reason", ""))
                  for s in d.get("suppress") or []],
        purge_ids=list(d.get("purge_ids") or []),
        ops=[_dc_from(Op, o) for o in d.get("ops") or []],
        guard=_dc_from(GuardResult, d.get("guard")),
        ledger_delta=_delta_from_json(d.get("ledger_delta") or {}),
        inbox_consume_ids=[int(i) for i in d.get("inbox_consume_ids") or []],
        inbox_skip_ids=[int(i) for i in d.get("inbox_skip_ids") or []],
        docs=[_dc_from(DocWrite, x) for x in d.get("docs") or []],
        stats=_dc_from(RunStats, d.get("stats")), report=_dc_from(RunReport, d.get("report")),
        status=d.get("status", "planned"))


def _fsync_dir(d: Path) -> None:
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_plan(paths: Any, plan: Plan) -> Path:
    """runs/<id>/plan.json: tmp + fsync + os.replace, 0600 (dirs 0700). Also used in dry-run."""
    paths.ensure_dir(paths.data_dir)
    paths.ensure_dir(paths.runs_dir)
    d = paths.ensure_dir(paths.run_dir(plan.run_id))
    dst = paths.plan_json(plan.run_id)
    data = json.dumps(plan_to_json(plan), ensure_ascii=False, sort_keys=True, default=str)
    fd, tmp = tempfile.mkstemp(prefix=".plan_", suffix=".tmp", dir=str(d))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    _fsync_dir(d)
    return dst


def read_plan(path: str | os.PathLike) -> Plan:
    with open(path, "r", encoding="utf-8") as f:
        return plan_from_json(json.load(f))


# ── R8 commit / replay ───────────────────────────────────────────────────────

def _quarantine_purge(plan: Plan) -> bool:
    return any(a.op == "purge" and (a.detail or "").startswith("quarantine")
               for a in plan.ledger_delta.audit)


def _stats_json(plan: Plan) -> str:
    return json.dumps(plan.stats.to_dict(), ensure_ascii=False, sort_keys=True, default=str)


def _apply_steps(ctx: Any, plan: Plan) -> None:
    """R8 2–5 and 7 (all idempotent). Step 6 (inbox consumed) runs in rem.post_commit after the
    serving copy is replaced (mark_inbox), so a same-day forget/remember stays visible to the
    provider through its pending list until the new serving copy carries it (DEVIATIONS F-25)."""
    store, ledger = ctx.store, ctx.ledger
    store.commit_history(plan.history)                       # 2
    store.commit_suppress(plan.suppress)                     # 3
    store.commit_memories(plan.upserts)                      # 4 (one merge_insert)
    store.purge(plan.purge_ids)
    store.purge_history(plan.purge_ids)                      # purged text must not survive in history
    ledger.apply_delta(plan.run_id, plan.ledger_delta,       # 5 (one transaction)
                       lance_version_after=store.version(), stats_json=_stats_json(plan),
                       status="held" if plan.guard.held else "committed")
    if _quarantine_purge(plan):                              # 7
        from .store import T_HISTORY, T_MEMORIES, T_SUPPRESS
        _prepurge_snapshot(ctx, plan)
        # only the tables holding text; suppress (vectors + sha) keeps its normal time travel
        store.optimize(cleanup_older_than_days=0, delete_unverified=True,
                       tables=(T_MEMORIES, T_HISTORY))
        store.optimize(cleanup_older_than_days=float(ctx.cfg.lance_cleanup_days), tables=(T_SUPPRESS,))
    else:
        store.optimize(cleanup_older_than_days=float(ctx.cfg.lance_cleanup_days))


def _prepurge_snapshot(ctx: Any, plan: Plan) -> None:
    """§5.4: a fresh backup before cleanup_older_than=0 wipes the time travel (F-2)."""
    from . import backups
    from .types import AuditRow
    try:
        dest = backups.snapshot_lancedb(ctx.paths, plan.run_id,
                                        keep=int(getattr(ctx.cfg, "prepurge_backups_keep", 2)))
        detail = f"lancedb-prepurge:{dest.name}"
    except Exception as e:  # noqa: BLE001 — the secret purge itself must still happen
        detail = f"lancedb-prepurge:failed:{type(e).__name__}"
        ctx.note(f"격리 삭제 전 Lance 백업을 만들지 못했습니다 ({type(e).__name__}). 삭제는 그대로 진행합니다.")
    try:
        if ctx.ledger is not None and not getattr(ctx.ledger, "readonly", False):
            ctx.ledger.add_audit(AuditRow(ts=float(ctx.now), run_id=plan.run_id, op="backup",
                                          memory_id=None, detail=detail))
    except Exception:  # noqa: BLE001
        pass


def mark_inbox(ctx: Any, plan: Plan) -> None:
    """R8 6, after the serving copy is in place. Failure leaves the items pending; the next run's
    R1 folds them again (every inbox op is idempotent)."""
    if ctx.live is None or plan is None or getattr(ctx, "dry_run", False):
        return
    try:
        if plan.inbox_consume_ids:
            ctx.live.mark_consumed(plan.inbox_consume_ids, plan.run_id)
        if plan.inbox_skip_ids:
            ctx.live.mark_consumed(plan.inbox_skip_ids, plan.run_id, status="skipped")
    except Exception as e:  # noqa: BLE001
        ctx.note(f"live.db inbox consumed 표시 실패({type(e).__name__}) — 다음 실행에서 다시 반영합니다.")


def commit_plan(ctx: Any, plan: Plan) -> CommitResult:
    """R8 1–7 for live/migrate runs. A noop plan writes plan.json, skips 2–7 and ends committed
    with the unchanged Lance version (G1b/T13)."""
    if ctx.dry_run:
        raise RuntimeError("commit_plan in dry-run")
    ledger, store = ctx.ledger, ctx.store
    ledger.insert_run(RunRecord(run_id=plan.run_id, started_at=ctx.now, mode=plan.mode,
                                status="planned", lance_version_before=plan.lance_version_before))
    plan.status = "planned"
    write_plan(ctx.paths, plan)                              # 1
    ledger.update_run(plan.run_id, status="planned", error=None)   # clears N0's "incomplete"
    skipped = plan.is_noop()
    if skipped:
        from . import clock
        ledger.update_run(plan.run_id, status="held" if plan.guard.held else "committed",
                          stats_json=_stats_json(plan), finished_at=clock.real_now())
    else:
        _apply_steps(ctx, plan)
    after = store.version()
    ledger.update_run(plan.run_id, lance_version_after=after)
    plan.status = "held" if plan.guard.held else "committed"
    return CommitResult(run_id=plan.run_id, versions_after=store.versions(),
                        lance_version_after=after, skipped=skipped)


def replay_planned(ctx: Any) -> list[str]:
    """N0: finish runs left in status 'planned' (crash during R8): steps 2–7, then post_commit."""
    if ctx.dry_run:
        return []
    from . import rem  # plan ↔ rem cycle
    done: list[str] = []
    for rec in ctx.ledger.planned_runs():
        p = ctx.paths.plan_json(rec.run_id)
        if not p.exists():
            ctx.ledger.update_run(rec.run_id, status="failed", error="plan.json missing")
            ctx.note(f"재생 불가: 실행 {rec.run_id}의 plan.json이 없습니다.")
            continue
        plan = read_plan(p)
        if not plan.is_noop():
            _apply_steps(ctx, plan)
        else:
            ctx.ledger.update_run(plan.run_id, status="held" if plan.guard.held else "committed")
        ctx.ledger.update_run(plan.run_id, lance_version_after=ctx.store.version())
        plan.status = "held" if plan.guard.held else "committed"
        sub = dataclasses.replace(ctx, run_id=plan.run_id, stats=plan.stats, report=plan.report,
                                  alerts=[])
        try:
            rem.post_commit(sub, plan)
        except Exception as e:  # the commit itself is complete; surface on the current run
            ctx.note(f"재생한 실행 {plan.run_id}의 커밋 후 단계 실패: {type(e).__name__}")
        ctx.alerts.extend(sub.alerts)
        done.append(plan.run_id)
    return done


def approve_held(ctx: Any, run_id: str) -> CommitResult:
    """Operator escape hatch (U2: nothing prompts for it). Applies a held run's held ops to the
    current rows when their current values still equal `op.before`; commits as this run."""
    rec = ctx.ledger.get_run(run_id)
    if rec is None or rec.status != "held":
        raise ValueError(f"보류된 실행이 아닙니다: {run_id}")
    held_plan = read_plan(ctx.paths.plan_json(run_id))
    ws = WorkingSet(ctx.store.load_working_set(), run_id=ctx.run_id, now=ctx.now,
                    embed_model=getattr(ctx.embedder, "model_id", None) or ctx.cfg.embed_model_id())
    applied = 0
    for o in held_plan.ops:
        if not o.held:
            continue
        row = ws.get(o.memory_id)
        if row is None:
            ctx.note(f"approve:{run_id} seq {o.seq} 대상 행 없음(stale)")
            continue
        cur = row.snapshot(include_vector=True)
        stale = [k for k, v in o.before.items()
                 if k not in ("updated_at", "last_run_id", "status_changed_at", "status_reason",
                              "tier") and cur.get(k) != v]
        if stale:
            ctx.note(f"approve:{run_id} seq {o.seq} 현재 값이 달라 건너뜀(stale: {', '.join(stale)})")
            continue
        changes: dict[str, Any] = {}
        for k, v in o.changes.items():
            if k == "vector_b64":
                changes["vector"] = vec_from_b64(v)
            elif k not in ("updated_at", "last_run_id", "tier"):
                changes[k] = v
        if ws.update(o.memory_id, changes, op=o.op, reason=f"approve:{run_id}", user_evidence=True,
                     guard_exempt=True, detail={"approved_seq": o.seq}) is not None:
            applied += 1
    ctx.note(f"approve:{run_id} 보류 연산 {applied}건 적용")
    apply_guard(ws, ctx.cfg, mode="live")
    plan = build_plan(ctx, ws, lance_version_before=ctx.store.version(),
                      ledger_delta=LedgerDelta(audit=list(ws.audit)), inbox_consume_ids=[],
                      inbox_skip_ids=[], docs=[])
    res = commit_plan(ctx, plan)
    ctx.ledger.update_run(run_id, status="committed")
    return res
