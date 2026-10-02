"""R0 upsert, R2 re-judge, R3 sweep (PLAN-v2 §4.3, U2).

    claim → idempotency(origin key) → suppress → candidates(Lance top-k + this run's new rows +
    same subject_key) → auto-dup(cos ≥ 0.95 & numbers/negation equal) | judge(enum JSON) →
    duplicate(live row) > state_change > different_aspects > unknown > unrelated
    (a duplicate of a superseded/expired row is history only when the claim is not newer than the
    state's current holder — otherwise it is a state change, DEVIATIONS F-13)

Every decision is an op on the WorkingSet, so the next claim sees the result of the previous one
(fixes the original 8-B stale-snapshot overwrite). A judge failure is never treated as
"unrelated": the row is inserted with judge_pending and re-judged next night (R2).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from . import vecutil
from .embedder import embed_input
from .llm import LLMAuthError, LLMError
from .strength import compute_tier
from .store import DEFAULT_CANDIDATE_WHERE
from .types import (CANDIDATE_SEARCH_EXCLUDED, CORE_SOURCES, DURABLE_SOURCES, EXPIRING_KINDS, KIND_BASE,
                    LEGACY_KIND, LIST_CAPS, NEWER_VALUES, PIN_KINDS, RELATIONS, STATUSES, TEXT_MIN_CHARS,
                    BudgetExceeded, Candidate, Claim, Judgement, MemoryRow, UpsertOutcome, append_capped,
                    new_memory_id, suppress_guard, suppress_reason_guard, text_sha)

log = logging.getLogger("hermesyume.upsert")

DAY = 86400.0
PRIORITY = ("duplicate", "state_change", "different_aspects", "unknown", "unrelated")
LIVE_TARGET_STATUSES = frozenset({"active", "dormant"})
CANDIDATE_WHERE = f"{DEFAULT_CANDIDATE_WHERE} AND kind != '{LEGACY_KIND}'"
ONLY_ACTIVE_EXCLUDED = tuple(s for s in STATUSES if s != "active")


def _round_imp(x: float) -> float:
    return round(float(x), 6)


def _kst_date(ts: float | None) -> str | None:
    if ts is None:
        return None
    from .clock import kst_date
    return kst_date(ts)


def _et(obj: Any, default: float) -> float:
    for f in ("event_time", "first_seen_at", "created_at"):
        v = getattr(obj, f, None)
        if v:
            return float(v)
    return float(default)


def row_from_claim(claim: Claim, *, ctx: Any, status: str | None = None) -> MemoryRow:
    """CONTRACTS §4.13.1 field mapping."""
    now = float(ctx.now)
    imp = claim.importance if claim.importance and claim.importance > 0 else KIND_BASE.get(claim.kind, 0.5)
    row = MemoryRow(
        id=new_memory_id(), text=claim.text, subject=claim.subject, subject_key=claim.subject_key,
        vector=None if claim.vector is None else claim.vector.copy(),
        embed_model=getattr(ctx.embedder, "model_id", "") or ctx.cfg.embed_model_id(),
        kind=claim.kind, target=claim.target, importance=_round_imp(imp), level=int(claim.level),
        pinned=bool(claim.pin), core_required=bool(claim.core_required),
        core_target=claim.core_target, core_sha=claim.core_sha, in_core=bool(claim.in_core),
        status=status or claim.status or "active", status_reason="new", status_changed_at=now,
        created_at=now, updated_at=now, event_time=claim.event_time, valid_from=claim.event_time,
        valid_until=claim.valid_until, first_seen_at=claim.first_seen_at or now,
        last_seen_at=claim.last_seen_at or claim.first_seen_at or now,
        last_user_evidence_at=claim.last_user_evidence_at,
        evidence_count=max(1, len(claim.evidence_keys)),
        user_evidence_count=int(claim.user_evidence_count), user_session_count=int(claim.user_session_count),
        explicit_user=bool(claim.explicit_user), source=claim.source,
        source_session_ids=append_capped([], _row_sessions(claim), LIST_CAPS["source_session_ids"]),
        source_message_ids=append_capped([], claim.evidence_keys, LIST_CAPS["source_message_ids"]),
        origin_keys=append_capped([], [claim.origin_key] if claim.origin_key else [], LIST_CAPS["origin_keys"]),
        refs=list(dict.fromkeys(claim.refs)), version=1, lang=claim.lang or "ko",
        scope=str(ctx.cfg.scope), last_run_id=ctx.run_id, schema_version=2)
    row.tier = compute_tier(row)
    return row


def _user_sessions(claim: Claim) -> list[str]:
    """Sessions that carried USER evidence for this claim (§5.2: protection counts user sessions
    only). Gates fill ``user_session_ids``; remember/core claims are user evidence throughout."""
    if claim.user_session_ids:
        return list(dict.fromkeys(claim.user_session_ids))
    roles = set(claim.evidence_roles or ())
    if claim.has_user_evidence and (claim.source in DURABLE_SOURCES or roles <= {"user", "core"}):
        return list(dict.fromkeys(claim.session_ids))
    return []


def _row_sessions(claim: Claim) -> list[str]:
    """A row's source_session_ids: the user-evidence sessions once it has user evidence (so a later
    user message from an assistant-only session still counts as a new user session), otherwise the
    evidence sessions for provenance (DEVIATIONS F-29)."""
    if claim.has_user_evidence:
        return _user_sessions(claim) or list(claim.session_ids)
    return list(claim.session_ids)


def _importance(**kw: Any) -> float:
    from .normalize import compute_importance
    return float(compute_importance(**kw))


def reinforce_changes(row: MemoryRow, claim: Claim, *, now: float) -> tuple[dict, bool]:
    """Duplicate merge (§4.3 table). Returns (changes, user_evidence). Counters move only for
    evidence messages not already recorded (distinct-message counting, §2.2); strength-relevant
    fields move only with user evidence."""
    keys = list(claim.evidence_keys or [])
    known = set(row.source_message_ids or [])
    new_keys = [k for k in keys if k not in known]
    fresh = bool(new_keys) or not keys
    user_ev = bool(claim.has_user_evidence) and fresh
    ch: dict[str, Any] = {}
    if fresh:
        ch["evidence_count"] = int(row.evidence_count or 0) + (len(new_keys) if keys else 1)
    if claim.origin_key:
        ch["origin_keys"] = append_capped(row.origin_keys, [claim.origin_key], LIST_CAPS["origin_keys"])
    row_usc = int(row.user_session_count or 0)
    user_sessions = _user_sessions(claim) if user_ev else []
    if user_ev:
        # first user evidence: the list becomes the user-evidence sessions (provenance → protection)
        base = list(row.source_session_ids) if row_usc > 0 else []
        ch["source_session_ids"] = append_capped(base, user_sessions or claim.session_ids,
                                                 LIST_CAPS["source_session_ids"])
    elif claim.session_ids and row_usc == 0:
        ch["source_session_ids"] = append_capped(row.source_session_ids, claim.session_ids,
                                                 LIST_CAPS["source_session_ids"])
    if keys:
        ch["source_message_ids"] = append_capped(row.source_message_ids, keys,
                                                 LIST_CAPS["source_message_ids"])
    if claim.last_seen_at:
        ch["last_seen_at"] = max(float(row.last_seen_at or 0.0), float(claim.last_seen_at))
    if claim.first_seen_at and (not row.first_seen_at or claim.first_seen_at < row.first_seen_at):
        ch["first_seen_at"] = float(claim.first_seen_at)
    if claim.refs:
        ch["refs"] = list(dict.fromkeys(list(row.refs) + list(claim.refs)))
    if user_ev:
        known = set(row.source_session_ids) if row_usc > 0 else set()
        new_sessions = [s for s in user_sessions if s not in known]
        inc = min(len(new_sessions), max(int(claim.user_session_count), 1))
        usc = row_usc + (max(inc, 1) if row_usc == 0 else inc)
        ch["user_evidence_count"] = int(row.user_evidence_count or 0) + max(int(claim.user_evidence_count), 1)
        ch["user_session_count"] = usc
        lue = claim.last_user_evidence_at or claim.last_seen_at or now
        ch["last_user_evidence_at"] = max(float(row.last_user_evidence_at or 0.0), float(lue))
        explicit = bool(row.explicit_user or claim.explicit_user)
        ch["explicit_user"] = explicit
        level = max(int(row.level or 3), int(claim.level or 3))
        ch["level"] = level
        imp = _importance(kind=row.kind, level=level, explicit_user=explicit, user_session_count=usc,
                          assistant_only=False, source=row.source)
        ch["importance"] = _round_imp(max(float(row.importance or 0.0), float(claim.importance or 0.0), imp))
        if (row.kind == "state" and claim.valid_until is not None
                and (row.valid_until is None or claim.valid_until > row.valid_until)):
            ch["valid_until"] = float(claim.valid_until)
    return ch, user_ev


def claim_from_row(row: MemoryRow) -> Claim:
    """Pseudo-claim for R2/R3 (a stored row acting as the "new" side)."""
    return Claim(origin_key="", source=row.source, kind=row.kind, target=row.target,
                 subject=row.subject, text=row.text, level=int(row.level or 3),
                 explicit=bool(row.explicit_user), event_time=row.event_time,
                 valid_until=row.valid_until, status=row.status,
                 evidence_keys=list(row.source_message_ids),
                 evidence_roles=["user"] if row.user_evidence_count > 0 else ["assistant"],
                 session_ids=list(row.source_session_ids),
                 user_session_ids=list(row.source_session_ids) if row.user_session_count > 0 else [],
                 first_seen_at=row.first_seen_at,
                 last_seen_at=row.last_seen_at, last_user_evidence_at=row.last_user_evidence_at,
                 user_evidence_count=int(row.user_evidence_count),
                 user_session_count=int(row.user_session_count),
                 explicit_user=bool(row.explicit_user), subject_key=row.subject_key,
                 refs=list(row.refs), importance=float(row.importance or 0.0), vector=row.vector,
                 pin=bool(row.pinned), lang=row.lang)


class Upserter:
    def __init__(self, ctx: Any, ws: Any):
        self.ctx = ctx
        self.ws = ws
        self.cfg = ctx.cfg
        self.judge_calls = 0
        self._suppress_shas: set[str] | None = None
        self.judged_pairs: set[frozenset] = set()
        self.judged_ids: set[str] = set()

    # ── helpers ──
    @property
    def _stats(self):
        return self.ctx.stats

    @property
    def _report(self):
        return self.ctx.report

    def _vector(self, claim: Claim) -> Any:
        if claim.vector is None:
            text = embed_input(claim.subject, claim.text)
            claim.embed_text = text
            claim.vector = self.ctx.embedder.embed([text])[0]
        return claim.vector

    def _cos_record(self, relation: str, c: float | None) -> None:
        if c is None:
            return
        self._report.cos_by_relation.setdefault(relation, []).append(round(float(c), 4))

    def _suppressed(self, claim: Claim, vec: Any) -> str | None:
        """R0-2: same text_sha → blocked. A cosine hit (≥ suppress_cos) blocks only when the
        numbers/dates and negation also match the forgotten text (guard hash in the suppress
        reason; rows written before the guard existed block on cosine alone)."""
        sha = text_sha(claim.text)
        if self._suppress_shas is None:
            self._suppress_shas = set(self.ctx.store.suppress_shas()) if self.ctx.store is not None else set()
        for s in self.ws.new_suppress:
            if s.text_sha == sha:
                return s.id
        if sha in self._suppress_shas:
            return "text_sha"
        th = float(self.cfg.suppress_cos)
        guard = suppress_guard(claim.text)

        def blocks(reason: str | None) -> bool:
            g = suppress_reason_guard(reason)
            return g is None or g == guard

        if self.ctx.store is not None:
            for hit, c in self.ctx.store.search_suppress(vec, k=3):
                if c >= th and blocks(hit.reason):
                    return hit.id
        for s in self.ws.new_suppress:
            if s.vector is not None and vecutil.cos(vec, s.vector) >= th and blocks(s.reason):
                return s.id
        return None

    def _subject_dict(self, c: Claim | MemoryRow) -> dict:
        return {"subject": c.subject, "text": c.text, "kind": c.kind,
                "event_time": _kst_date(getattr(c, "event_time", None))}

    # ── public API ──
    def candidates(self, claim: Claim, *, exclude_ids: Iterable[str] = (),
                   full_working_set: bool = False) -> list[Candidate]:
        vec = self._vector(claim)
        ex = set(exclude_ids)
        found: dict[str, Candidate] = {}

        def add(row: MemoryRow, c: float, via: str) -> None:
            # legacy rows are raw archives ("dormant 고정", §5.1): never a merge/supersede target
            if row.id in ex or row.status in CANDIDATE_SEARCH_EXCLUDED or row.kind == LEGACY_KIND:
                return
            prev = found.get(row.id)
            if prev is None or c > prev.cos:
                found[row.id] = Candidate(row=row, cos=float(c), via=via)

        k = int(self.cfg.candidate_k)
        if full_working_set:
            for row, c in self.ws.vector_search(vec, k=k, exclude_ids=ex):
                add(row, c, "working_set")
        elif self.ctx.store is not None:
            for hit, c in self.ctx.store.search(vec, k=k, where=CANDIDATE_WHERE):
                cur = self.ws.get(hit.id)
                if cur is None:
                    continue
                add(cur, vecutil.cos(vec, cur.vector) if cur.vector is not None else c, "lance")
        for row, c in self.ws.vector_search(vec, k=k, min_cos=float(self.cfg.candidate_cos),
                                            only_new=True, exclude_ids=ex):
            add(row, c, "run_new")
        for row in self.ws.by_subject_key(claim.subject_key):
            if row.vector is not None:
                add(row, vecutil.cos(vec, row.vector), "subject_key")
        return sorted(found.values(), key=lambda x: (-x.cos, x.row.id))

    def _to_judge(self, claim: Claim, cands: list[Candidate]) -> list[Candidate]:
        th = float(self.cfg.candidate_cos)
        sk = claim.subject_key
        sel = [c for c in cands if c.cos >= th or (sk and c.row.subject_key == sk)]
        return sel[: int(self.cfg.candidate_k)]

    def _parse_relations(self, data: Any, sid_map: dict[str, Candidate]) -> dict[str, tuple[str, str]] | None:
        if not isinstance(data, dict) or not isinstance(data.get("relations"), list):
            return None
        rels = data["relations"]
        full = {c.row.id: sid for sid, c in sid_map.items()}
        out: dict[str, tuple[str, str]] = {}
        for item in rels:
            if not isinstance(item, dict):
                continue
            rid = str(item.get("id", "")).strip()
            sid = rid if rid in sid_map else full.get(rid)
            typ = str(item.get("type", "")).strip().lower()
            if sid is None or typ not in RELATIONS:
                continue
            newer = str(item.get("newer", "same")).strip().lower()
            out[sid] = (typ, newer if newer in NEWER_VALUES else "same")
        if rels and not out:
            return None
        return out

    def _call(self, kind: str, messages: list[dict], max_tokens: int) -> Any:
        resp = self.ctx.llm.chat_json(kind, messages, model=self.cfg.judge_model, max_tokens=max_tokens,
                                      temperature=float(self.cfg.llm_temperature))
        return resp.data

    def judge(self, subject: dict, cands: list[Candidate]) -> list[Judgement]:
        """One judge call for ≤ candidate_k candidates. Failure → one judge_enum call for the top
        candidate; still bad (or over budget) → unknown. Never 'unrelated' by default."""
        unknown = [Judgement(c.row.id, "unknown", "same", c.cos) for c in cands]
        if not cands:
            return []
        from . import prompts
        maxj = int(self.cfg.max_judge_calls)
        if self.judge_calls >= maxj:
            return self._record(unknown)
        sid_map = {f"c{i + 1}": c for i, c in enumerate(cands)}
        cdicts = [{"id": sid, "subject": c.row.subject, "text": c.row.text, "kind": c.row.kind,
                   "event_time": _kst_date(c.row.event_time), "status": c.row.status}
                  for sid, c in sid_map.items()]
        parsed: dict[str, tuple[str, str]] | None = None
        try:
            self.judge_calls += 1
            self._stats.judge_calls += 1
            data = self._call("judge", prompts.judge_messages(subject, cdicts), int(self.cfg.judge_max_tokens))
            parsed = self._parse_relations(data, sid_map)
        except LLMAuthError:
            raise
        except BudgetExceeded:
            return self._record(unknown)
        except LLMError as e:
            log.warning("judge failed: %s", type(e).__name__)
            self._stats.judge_failures += 1
            return self._record(unknown)
        if parsed is None:
            self._stats.judge_failures += 1
            if self.judge_calls >= maxj:
                return self._record(unknown)
            top_sid, top = next(iter(sid_map.items()))
            try:
                self.judge_calls += 1
                self._stats.judge_calls += 1
                data = self._call("judge_enum", prompts.judge_enum_messages(subject, cdicts[0]),
                                  int(self.cfg.judge_max_tokens))
            except LLMAuthError:
                raise
            except (BudgetExceeded, LLMError):
                return self._record(unknown)
            typ = str((data or {}).get("type", "")).strip().lower() if isinstance(data, dict) else ""
            newer = str((data or {}).get("newer", "same")).strip().lower() if isinstance(data, dict) else "same"
            if typ not in RELATIONS:
                return self._record(unknown)
            parsed = {top_sid: (typ, newer if newer in NEWER_VALUES else "same")}
        out = []
        for sid, c in sid_map.items():
            typ, newer = parsed.get(sid, ("unknown", "same"))
            out.append(Judgement(c.row.id, typ, newer, c.cos))
        return self._record(out)

    def _record(self, js: list[Judgement]) -> list[Judgement]:
        for j in js:
            self._cos_record(j.relation, j.cos)
        return js

    def _pick(self, judgements: list[Judgement]) -> str:
        rels = {j.relation for j in judgements}
        for r in PRIORITY:
            if r in rels:
                return r
        return "unrelated"

    # ── R0 ──
    def upsert(self, claim: Claim) -> UpsertOutcome:
        ws, st = self.ws, self._stats
        hit = ws.by_origin_key(claim.origin_key) if claim.origin_key else None
        if hit is not None:
            st.idempotent_skips += 1
            return UpsertOutcome("skip_idempotent", hit.id)
        vec = self._vector(claim)
        sup = self._suppressed(claim, vec)
        if sup is not None:
            st.suppressed_hits += 1
            self._report.notes.append(f"억제 목록 적중으로 버림 (suppress id {sup})")
            for item in self._report.claims:     # lets the Dream Log hide the forgotten text
                if claim.origin_key and item.get("origin_key") == claim.origin_key:
                    item["status"] = "suppressed"
            return UpsertOutcome("suppressed", None)
        cands = self.candidates(claim)
        if cands:
            top = cands[0]
            if (top.cos >= float(self.cfg.auto_dup_cos)
                    and vecutil.auto_dup_eligible(claim.text, top.row.text, claim.kind, top.row.kind,
                                                  new_ref_date=_kst_date(claim.event_time),
                                                  old_ref_date=_kst_date(top.row.event_time))):
                js = self._record([Judgement(top.row.id, "duplicate", "same", top.cos)])
                return self._resolve_claim(claim, js, auto=True)
        to_judge = self._to_judge(claim, cands)
        if not to_judge:
            return self._insert(claim, action="inserted", reason="no_candidate")
        js = self.judge(self._subject_dict(claim), to_judge)
        return self._resolve_claim(claim, js)

    def _insert(self, claim: Claim, *, action: str, reason: str, status: str | None = None,
                superseded_by: str | None = None, supersedes: Iterable[str] = (),
                related: Iterable[str] = (), judge_pending: bool = False,
                judgements: list[Judgement] | None = None, valid_until: float | None = None) -> UpsertOutcome:
        row = row_from_claim(claim, ctx=self.ctx, status=status)
        row.supersedes = list(dict.fromkeys(supersedes))
        row.related_ids = list(dict.fromkeys(related))
        row.superseded_by = superseded_by
        row.judge_pending = bool(judge_pending)
        if valid_until is not None:
            row.valid_until = valid_until
        if status and status != "active":
            row.status_reason = reason
        op = self.ws.insert(row, reason=reason, user_evidence=claim.has_user_evidence,
                            detail={"origin_key": claim.origin_key,
                                    "relations": [[j.candidate_id, j.relation] for j in judgements or []]})
        self._stats.created += 1
        if judge_pending:
            self._stats.judge_pending += 1
        self._report.created.append({"id": row.id, "kind": row.kind, "tier": row.tier,
                                     "text": row.text, "importance": row.importance})
        for j in judgements or []:
            self.judged_pairs.add(frozenset((row.id, j.candidate_id)))
        self.judged_ids.add(row.id)
        return UpsertOutcome(action, row.id, list(judgements or []), [op.seq])

    def _live(self, memory_id: str) -> bool:
        r = self.ws.get(memory_id)
        return r is not None and r.status in LIVE_TARGET_STATUSES

    def _chain_end(self, row: MemoryRow) -> MemoryRow:
        """Follow superseded_by to the last row of the chain (the current holder of the state when
        it is active/dormant; an expired/superseded end means nothing holds it now)."""
        seen: set[str] = set()
        cur = row
        while cur.status not in LIVE_TARGET_STATUSES and cur.superseded_by and cur.id not in seen:
            seen.add(cur.id)
            nxt = self.ws.get(cur.superseded_by)
            if nxt is None:
                break
            cur = nxt
        return cur

    @staticmethod
    def _grounded(row: MemoryRow) -> bool:
        """A row backed by the user (user evidence, core/remember source, pin or core copy)."""
        return bool((row.user_evidence_count or 0) > 0 or row.source in DURABLE_SOURCES
                    or row.pinned or row.in_core)

    def _echo_skip(self, claim: Claim, js: list[Judgement]) -> UpsertOutcome | None:
        """U3 echo guard (DEVIATIONS E2E-5): a claim with no user evidence that the judge relates
        to a user-grounded row (same / changed / other aspect) is the assistant (or its journal)
        restating recalled memory. It is recorded only — no new row, no supersede, no merge — so a
        recall → answer → re-extraction cycle cannot add paraphrase rows, retire a user-stated row
        or resurrect an expired one. Unrelated/unknown claims are inserted as before (decaying)."""
        if claim.has_user_evidence:
            return None
        rel = [j for j in js if j.relation in ("duplicate", "state_change", "different_aspects")]
        for j in sorted(rel, key=lambda j: -(j.cos or 0.0)):
            row = self.ws.get(j.candidate_id)
            if row is None or not self._grounded(row):
                continue
            self._stats.reinforce_noop += 1
            self._report.reinforced.append({"id": row.id, "text": row.text, "user": False})
            for x in js:
                self.judged_pairs.add(frozenset((row.id, x.candidate_id)))
            return UpsertOutcome("reinforce_noop", row.id, js, [])
        return None

    def _resolve_claim(self, claim: Claim, js: list[Judgement], *, auto: bool = False) -> UpsertOutcome:
        by_rel = lambda r: sorted((j for j in js if j.relation == r), key=lambda j: -(j.cos or 0.0))  # noqa: E731
        live_dups = [j for j in by_rel("duplicate") if self._live(j.candidate_id)]
        if live_dups:
            out = self._duplicate(claim, live_dups[0], js, auto=auto)
            self._explicit_supersede_protected(claim, live_dups[0].candidate_id, js, out)
            return out
        echo = self._echo_skip(claim, js)
        if echo is not None:
            return echo
        dead_dups = by_rel("duplicate")
        if dead_dups:
            # only superseded/expired rows match: never let them swallow a newer claim (A→B→A)
            out = self._duplicate_of_inactive(claim, dead_dups[0], js, auto=auto)
            if out is not None:
                return out
            js = [j for j in js if j.relation != "duplicate"]
        rel = self._pick(js)
        if rel == "state_change":
            return self._state_change(claim, by_rel("state_change"), js)
        if rel == "different_aspects":
            return self._different_aspects(claim, by_rel("different_aspects"), js)
        if rel == "unknown":
            return self._insert(claim, action="judge_pending", reason="judge_unknown", judge_pending=True,
                                judgements=js)
        return self._insert(claim, action="inserted", reason="unrelated", judgements=js)

    def _revive_if_dormant(self, memory_id: str, ops: list[int], claim: Claim | None = None) -> bool:
        """dormant → active on fresh user evidence (§5.4). An expired state whose deadline the new
        user evidence pushed into the future is revived too (it would not re-expire in R6)."""
        row = self.ws.get(memory_id)
        if row is None:
            return False
        if row.status == "expired":
            if claim is None or claim.status != "active" or row.kind not in EXPIRING_KINDS:
                return False
            probe = row.copy()
            probe.status = "active"
            from .strength import transition
            if transition(probe, self.ctx.now, self.cfg) is not None:
                return False
        elif row.status != "dormant" or row.kind == LEGACY_KIND:
            return False
        o = self.ws.update(memory_id, {"status": "active"}, op="status", reason="revived:evidence",
                           user_evidence=True)
        if o is not None:
            ops.append(o.seq)
            self._stats.revived += 1
            from .strength import strength as _s
            self._report.revived.append({"id": memory_id, "text": row.text,
                                         "strength": round(_s(row, self.ctx.now), 4)})
            return True
        return False

    def _duplicate(self, claim: Claim, j: Judgement, js: list[Judgement], *, auto: bool = False) -> UpsertOutcome:
        target = self.ws.get(j.candidate_id)
        changes, user_ev = reinforce_changes(target, claim, now=self.ctx.now)
        ops: list[int] = []
        o = self.ws.update(target.id, changes, op="reinforce", reason="duplicate", user_evidence=user_ev,
                           detail={"relation": "duplicate", "cos": j.cos, "auto": auto,
                                   "origin_key": claim.origin_key})
        if o is not None:
            ops.append(o.seq)
        if user_ev:
            self._stats.reinforced += 1
            self._revive_if_dormant(target.id, ops, claim)
        else:
            self._stats.reinforce_noop += 1
        self._report.reinforced.append({"id": target.id, "text": target.text, "user": user_ev})
        for x in js:
            self.judged_pairs.add(frozenset((target.id, x.candidate_id)))
        return UpsertOutcome("duplicate" if user_ev else "reinforce_noop", target.id, js, ops)

    def _duplicate_of_inactive(self, claim: Claim, j: Judgement, js: list[Judgement], *,
                               auto: bool) -> UpsertOutcome | None:
        """Duplicate of a superseded/expired row (DEVIATIONS F-13). A claim not newer than the
        current holder of that state is history (backlog) → absorbed as before. A newer claim is
        the state coming back: superseded → state_change against the live successor; expired →
        absorbed only when that revives it (user evidence pushing valid_until past now). Returns
        None to fall through to the remaining judgements / an insert."""
        target = self.ws.get(j.candidate_id)
        if target is None:
            return None
        now = self.ctx.now
        if target.status == "superseded":
            end = self._chain_end(target)
            if not self._claim_newer(claim, end, "same", now):
                return self._duplicate(claim, j, js, auto=auto)       # backlog: history
            if end.status not in LIVE_TARGET_STATUSES:
                return None                                            # nothing holds it now: new row
            sc = Judgement(end.id, "state_change", "new", j.cos)
            rest = [x for x in js if x.candidate_id != end.id]
            self._record([sc])
            return self._state_change(claim, [sc], rest + [sc])
        if target.status == "expired":
            if not self._claim_newer(claim, target, "same", now):
                return self._duplicate(claim, j, js, auto=auto)
            changes, user_ev = reinforce_changes(target, claim, now=now)
            probe = target.copy()
            for k, v in changes.items():
                setattr(probe, k, v)
            probe.status = "active"
            from .strength import transition
            if user_ev and claim.status == "active" and target.kind in EXPIRING_KINDS \
                    and transition(probe, now, self.cfg) is None:
                return self._duplicate(claim, j, js, auto=auto)     # revives (R-5)
            return None
        return self._duplicate(claim, j, js, auto=auto)

    def _explicit_supersede_protected(self, claim: Claim, holder_id: str, js: list[Judgement],
                                      out: UpsertOutcome) -> None:
        """U2 divergence repair (DEVIATIONS F-14): an explicit user restatement that duplicates
        one row while the judge called another live pinned/durable row a state_change it is newer
        than — that protected row was only *linked* earlier for lack of explicit evidence. Now it
        is superseded by the row holding the restated value (pin follows, F-17)."""
        if not (claim.has_user_evidence and claim.explicit_user):
            return
        holder = self.ws.get(holder_id)
        if holder is None or holder.status not in LIVE_TARGET_STATUSES:
            return
        now = self.ctx.now
        for j in js:
            if j.relation != "state_change" or j.candidate_id == holder_id:
                continue
            old = self.ws.get(j.candidate_id)
            if old is None or old.status not in LIVE_TARGET_STATUSES or not self._protected(old):
                continue
            if not self._claim_newer(claim, old, j.newer, now):
                continue
            o = self.ws.update(old.id, {"status": "superseded", "superseded_by": holder.id,
                                        "valid_until": _et(claim, now)},
                               op="supersede", reason="state_change:explicit", user_evidence=True,
                               detail={"by": holder.id})
            if o is None:
                continue
            out.ops.append(o.seq)
            self._stats.superseded += 1
            self._report.superseded.append({"old_id": old.id, "old_text": old.text,
                                            "new_id": holder.id, "new_text": holder.text})
            cur = self.ws.get(holder.id)
            o2 = self.ws.update(holder.id, {"supersedes": list(dict.fromkeys(list(cur.supersedes) + [old.id]))},
                                op="text_update", reason="state_change:explicit", user_evidence=True)
            if o2 is not None:
                out.ops.append(o2.seq)
            self._move_pin(old, holder.id, out.ops)

    def _pins_budget_ok(self, text: str, *, freed_id: str | None = None) -> bool:
        used = sum(len(r.text) for r in self.ws.rows.values()
                   if r.pinned and r.status == "active" and not r.in_core and r.id != freed_id)
        return used + len(text) <= int(self.cfg.pins_budget_chars)

    def _move_pin(self, old: MemoryRow, new_id: str, ops: list[int]) -> None:
        """A pinned row replaced with explicit user evidence hands its pin to the replacement
        (as core_replace already does), within the pin budget (DEVIATIONS F-17)."""
        new = self.ws.get(new_id)
        if not old.pinned or new is None or new.pinned or new.kind not in PIN_KINDS:
            return
        if not self._pins_budget_ok(new.text, freed_id=old.id):
            self.ctx.note(f"pin 예산({self.cfg.pins_budget_chars}자) 초과로 대체된 pin을 새 행에 옮기지 "
                          f"않았습니다 (id {new_id}).")
            return
        o = self.ws.update(new_id, {"pinned": True}, op="pin", reason="supersede_pin", user_evidence=True,
                           detail={"from": old.id})
        if o is not None:
            ops.append(o.seq)
        o2 = self.ws.update(old.id, {"pinned": False}, op="unpin", reason="supersede_pin",
                            user_evidence=True, detail={"to": new_id})
        if o2 is not None:
            ops.append(o2.seq)

    def _protected(self, row: MemoryRow) -> bool:
        return bool(row.pinned or row.tier == "durable")

    @staticmethod
    def _claim_newer(claim: Claim, row: MemoryRow, newer: str, now: float) -> bool:
        a, b = _et(claim, now), _et(row, now)
        if a != b:
            return a > b
        return newer != "existing"

    def _state_change(self, claim: Claim, scs: list[Judgement], js: list[Judgement]) -> UpsertOutcome:
        ws, now = self.ws, self.ctx.now
        allow = bool(claim.has_user_evidence and claim.explicit_user)
        supersede, link, newer_rows = [], [], []
        for j in scs:
            row = ws.get(j.candidate_id)
            if row is None:
                continue
            if self._claim_newer(claim, row, j.newer, now):
                if row.status in LIVE_TARGET_STATUSES:
                    if self._protected(row) and not allow:
                        link.append(row)          # U2: never supersede, keep both, link
                    else:
                        supersede.append(row)
            else:
                newer_rows.append(row)
        status, sup_by, vu = None, None, None
        if newer_rows:
            status = "superseded"
            sup_by = max(newer_rows, key=lambda r: (_et(r, now), r.id)).id
            first_after = min(_et(r, now) for r in newer_rows)
            vu = first_after if claim.valid_until is None or claim.valid_until > first_after else None
        action = ("state_change" if supersede else "inserted_superseded" if newer_rows
                  else "related" if link else "inserted")
        out = self._insert(claim, action=action, reason="state_change", status=status,
                           superseded_by=sup_by, supersedes=[r.id for r in supersede],
                           related=[r.id for r in link], judgements=js, valid_until=vu)
        new_id = out.memory_id
        new_et = _et(claim, now)
        for r in supersede:
            o = ws.update(r.id, {"status": "superseded", "superseded_by": new_id, "valid_until": new_et},
                          op="supersede", reason="state_change", user_evidence=claim.has_user_evidence,
                          detail={"by": new_id})
            if o is not None:
                out.ops.append(o.seq)
                self._stats.superseded += 1
                self._report.superseded.append({"old_id": r.id, "old_text": r.text,
                                                "new_id": new_id, "new_text": claim.text})
                if r.pinned and allow:
                    self._move_pin(r, new_id, out.ops)
        for r in link:
            self._link(r.id, new_id, "protected", out.ops)
        if status == "superseded":
            self._report.superseded.append({"old_id": new_id, "old_text": claim.text,
                                            "new_id": sup_by, "new_text": ws.get(sup_by).text})
        return out

    def _link(self, a_id: str, b_id: str, reason: str, ops: list[int]) -> None:
        """Two-way related_ids link (b already carries a when inserted with related=[a])."""
        for x, y in ((a_id, b_id), (b_id, a_id)):
            row = self.ws.get(x)
            if row is None or y in row.related_ids:
                continue
            o = self.ws.update(x, {"related_ids": list(row.related_ids) + [y]}, op="text_update",
                               reason=f"related:{reason}")
            if o is not None:
                ops.append(o.seq)
        self._stats.related_linked += 1
        self._report.related.append({"a_id": a_id, "b_id": b_id, "reason": reason})

    def _consolidate(self, a: str, b: str) -> tuple[str | None, str]:
        """LLM merge (≤ consolidate_max) + fact preservation + scanner. (text|None, why)."""
        from . import prompts
        maxc = int(self.cfg.consolidate_max)
        try:
            resp = self.ctx.llm.chat_json("consolidate", prompts.consolidate_messages(a, b, max_chars=maxc),
                                          model=self.cfg.judge_model,
                                          max_tokens=int(self.cfg.consolidate_max_tokens),
                                          temperature=float(self.cfg.llm_temperature))
        except LLMAuthError:
            raise
        except (BudgetExceeded, LLMError) as e:
            return None, f"llm:{type(e).__name__}"
        data = resp.data
        text = data.get("text") if isinstance(data, dict) else None
        if not isinstance(text, str):
            return None, "parse"
        text = text.strip()
        if not (TEXT_MIN_CHARS <= len(text) <= maxc):
            return None, "length"
        ok, missing = vecutil.preservation_check(a, b, text)
        if not ok:
            return None, "preservation:" + ",".join(missing[:8])
        sc = self.ctx.scanner
        if sc is not None and not sc.is_clean(text):
            return None, "threat"
        return text, "ok"

    def _can_consolidate(self, target: MemoryRow, claim: Claim) -> bool:
        if target.status not in LIVE_TARGET_STATUSES or target.pinned:
            return False
        if target.tier == "durable" and not claim.has_user_evidence:
            return False   # would be a guarded change (U2) → keep both instead of losing the claim
        if target.status == "dormant" and not claim.has_user_evidence:
            return False   # merging would hide a new fact inside a row auto-recall skips (F-20)
        return True

    def _merge_into(self, target: MemoryRow, claim: Claim, merged: str, ops: list[int], *,
                    reason: str) -> bool:
        old_text = target.text
        changes, user_ev = reinforce_changes(target, claim, now=self.ctx.now)
        changes["text"] = merged
        changes["vector"] = self.ctx.embedder.embed([embed_input(target.subject, merged)])[0]
        changes["importance"] = _round_imp(max(float(target.importance or 0.0), float(claim.importance or 0.0),
                                               float(changes.get("importance", 0.0))))
        o = self.ws.update(target.id, changes, op="consolidate", reason=reason,
                           user_evidence=bool(claim.has_user_evidence),
                           detail={"origin_key": claim.origin_key})
        if o is None:
            return False
        ops.append(o.seq)
        self._stats.consolidated += 1
        self._report.consolidated.append({"id": target.id, "a": old_text, "b": claim.text,
                                          "result": merged})
        if user_ev:
            self._revive_if_dormant(target.id, ops, claim)
        return True

    def _different_aspects(self, claim: Claim, das: list[Judgement], js: list[Judgement]) -> UpsertOutcome:
        ws = self.ws
        top = ws.get(das[0].candidate_id)
        why = "pinned" if top is not None and top.pinned else "protected"
        if top is not None and self._can_consolidate(top, claim):
            merged, why = self._consolidate(top.text, claim.text)
            if merged is not None:
                ops: list[int] = []
                if self._merge_into(top, claim, merged, ops, reason="different_aspects"):
                    for j in das[1:]:
                        self._link(top.id, j.candidate_id, "different_aspects", ops)
                    for x in js:
                        self.judged_pairs.add(frozenset((top.id, x.candidate_id)))
                    return UpsertOutcome("consolidated", top.id, js, ops)
        ids = [j.candidate_id for j in das if ws.get(j.candidate_id) is not None]
        out = self._insert(claim, action="related", reason="different_aspects", related=ids, judgements=js)
        for mid in ids:
            self._link(mid, out.memory_id, f"different_aspects:{why}", out.ops)
        return out

    # ── R2 / R3: a stored row as the "new" side ──
    def _retire(self, loser: MemoryRow, into: str, reason: str, ops: list[int]) -> None:
        o = self.ws.update(loser.id, {"status": "superseded", "superseded_by": into, "judge_pending": False},
                           op="supersede", reason=reason,
                           user_evidence=claim_from_row(loser).has_user_evidence, detail={"into": into})
        if o is not None:
            ops.append(o.seq)
            self._stats.superseded += 1
            self._report.superseded.append({"old_id": loser.id, "old_text": loser.text,
                                            "new_id": into, "new_text": self.ws.get(into).text})

    def _resolve_existing(self, row: MemoryRow, js: list[Judgement]) -> UpsertOutcome:
        """R2/R3 resolution with a stored row as the "new" side. A pinned/protected row is never
        retired in favour of an unprotected one without explicit user evidence (U2)."""
        ws, now = self.ws, self.ctx.now
        self.judged_ids.add(row.id)
        for j in js:
            self.judged_pairs.add(frozenset((row.id, j.candidate_id)))
        rel = self._pick(js)
        ops: list[int] = []
        if rel == "unknown":
            return UpsertOutcome("judge_pending", row.id, js, ops)
        best = sorted((j for j in js if j.relation == rel), key=lambda j: -(j.cos or 0.0))

        def clear_pending() -> None:
            cur = ws.get(row.id)
            if cur is not None and cur.judge_pending:
                o = ws.update(row.id, {"judge_pending": False}, op="status", reason=f"judged:{rel}")
                if o is not None:
                    ops.append(o.seq)

        def rank(r: MemoryRow) -> tuple:
            # a live row always survives a superseded/expired one (F-13); then pin, protection
            return (r.status == "active", r.status in LIVE_TARGET_STATUSES, bool(r.pinned),
                    self._protected(r))

        if rel == "duplicate":
            target = ws.get(best[0].candidate_id)
            if target is None:
                clear_pending()
                return UpsertOutcome("inserted", row.id, js, ops)
            survivor, loser = (row, target) if rank(row) > rank(target) else (target, row)
            changes, user_ev = reinforce_changes(survivor, claim_from_row(loser), now=now)
            o = ws.update(survivor.id, changes, op="reinforce", reason="duplicate_of",
                          user_evidence=user_ev, detail={"from": loser.id})
            if o is not None:
                ops.append(o.seq)
            if user_ev:
                self._stats.reinforced += 1
                self._revive_if_dormant(survivor.id, ops)
            else:
                self._stats.reinforce_noop += 1
            self._report.reinforced.append({"id": survivor.id, "text": survivor.text, "user": user_ev})
            self._retire(ws.get(loser.id), survivor.id, "duplicate_of", ops)
            if survivor.id == row.id:
                clear_pending()
            return UpsertOutcome("duplicate", survivor.id, js, ops)

        if rel == "state_change":
            acted = False
            for j in best:
                other = ws.get(j.candidate_id)
                cur = ws.get(row.id)
                if other is None or cur is None:
                    continue
                if self._claim_newer(claim_from_row(cur), other, j.newer, now):
                    newer, older = cur, other
                else:
                    newer, older = other, cur
                if older.status not in LIVE_TARGET_STATUSES:
                    continue
                nc = claim_from_row(newer)
                if self._protected(older) and not (nc.has_user_evidence and nc.explicit_user):
                    self._link(older.id, newer.id, "protected", ops)
                    acted = True
                    continue
                o = ws.update(older.id, {"status": "superseded", "superseded_by": newer.id,
                                         "valid_until": _et(newer, now), "judge_pending": False},
                              op="supersede", reason="state_change",
                              user_evidence=nc.has_user_evidence, detail={"by": newer.id})
                if o is not None:
                    ops.append(o.seq)
                    acted = True
                    self._stats.superseded += 1
                    self._report.superseded.append({"old_id": older.id, "old_text": older.text,
                                                    "new_id": newer.id, "new_text": newer.text})
                    o2 = ws.update(newer.id, {"supersedes": list(dict.fromkeys(list(newer.supersedes) + [older.id]))},
                                   op="text_update", reason="state_change")
                    if o2 is not None:
                        ops.append(o2.seq)
                if ws.get(row.id).status not in LIVE_TARGET_STATUSES:
                    break
            clear_pending()
            return UpsertOutcome("state_change" if acted else "related", row.id, js, ops)

        if rel == "different_aspects":
            target = ws.get(best[0].candidate_id)
            if target is not None and not row.pinned and self._can_consolidate(target, claim_from_row(row)):
                merged, _why = self._consolidate(target.text, row.text)
                if merged is not None and self._merge_into(target, claim_from_row(row), merged, ops,
                                                           reason="different_aspects"):
                    self._retire(ws.get(row.id), target.id, "consolidated_into", ops)
                    return UpsertOutcome("consolidated", target.id, js, ops)
            for j in best:
                if ws.get(j.candidate_id) is not None:
                    self._link(j.candidate_id, row.id, "different_aspects", ops)
            clear_pending()
            return UpsertOutcome("related", row.id, js, ops)

        clear_pending()
        return UpsertOutcome("inserted", row.id, js, ops)

    def absorb_into_core_copy(self, core_id: str) -> list[UpsertOutcome]:
        """A new verbatim core copy (core_add / core_replace / R5 mirror) is inserted without R0, so
        a conversation row stating the same fact would be recalled again through <memory-context>
        while the entry already sits in the system prompt (§0.3 double injection). Live,
        unpinned, non-core rows judged duplicate of the copy (auto-dup rule first, judge
        otherwise) are folded into it and retired (``duplicate_of_core``), as R2 does for
        duplicates (DEVIATIONS F-21)."""
        ws = self.ws
        core = ws.get(core_id)
        if core is None or core.vector is None or core.status != "active":
            return []
        pseudo = claim_from_row(core)
        cands = [c for c in self.candidates(pseudo, exclude_ids={core.id}, full_working_set=True)
                 if c.row.status in LIVE_TARGET_STATUSES and not c.row.pinned and not c.row.in_core
                 and c.row.source not in CORE_SOURCES and c.row.kind != LEGACY_KIND]
        if not cands:
            return []
        top = cands[0]
        if top.cos >= float(self.cfg.auto_dup_cos) and vecutil.auto_dup_eligible(
                core.text, top.row.text, core.kind, top.row.kind,
                new_ref_date=_kst_date(core.event_time), old_ref_date=_kst_date(top.row.event_time)):
            js = self._record([Judgement(top.row.id, "duplicate", "same", top.cos)])
        else:
            to_judge = self._to_judge(pseudo, cands)
            if not to_judge:
                return []
            js = self.judge(self._subject_dict(core), to_judge)
        self.judged_ids.add(core.id)
        outs: list[UpsertOutcome] = []
        for j in js:
            self.judged_pairs.add(frozenset((core.id, j.candidate_id)))
            if j.relation != "duplicate":
                continue
            x = ws.get(j.candidate_id)
            cur = ws.get(core.id)
            if x is None or cur is None or x.status not in LIVE_TARGET_STATUSES or x.pinned:
                continue
            ops: list[int] = []
            changes, user_ev = reinforce_changes(cur, claim_from_row(x), now=self.ctx.now)
            o = ws.update(cur.id, changes, op="reinforce", reason="duplicate_of_core", user_evidence=True,
                          detail={"from": x.id})
            if o is not None:
                ops.append(o.seq)
            self._report.reinforced.append({"id": cur.id, "text": cur.text, "user": user_ev})
            self._retire(x, cur.id, "duplicate_of_core", ops)
            outs.append(UpsertOutcome("duplicate", cur.id, js, ops))
        return outs

    def rejudge_pending(self, *, limit: int) -> list[UpsertOutcome]:
        """R2: rows with judge_pending from earlier runs (oldest first)."""
        ws = self.ws
        pending = sorted((r for r in ws.rows.values()
                          if r.judge_pending and r.status not in CANDIDATE_SEARCH_EXCLUDED
                          and r.id not in ws.new_ids and r.id not in self.judged_ids),
                         key=lambda r: (r.created_at, r.id))[: max(0, int(limit))]
        outs: list[UpsertOutcome] = []
        for r0 in pending:
            row = ws.get(r0.id)
            if row is None or not row.judge_pending or row.status in CANDIDATE_SEARCH_EXCLUDED:
                continue
            pseudo = claim_from_row(row)
            cands = self._to_judge(pseudo, self.candidates(pseudo, exclude_ids={row.id},
                                                           full_working_set=True))
            if not cands:
                outs.append(self._resolve_existing(row, []))
                continue
            js = self.judge(self._subject_dict(row), cands)
            outs.append(self._resolve_existing(row, js))
        return outs

    def sweep(self, *, days: float, min_cos: float, max_pairs: int) -> list[UpsertOutcome]:
        """R3: pairs (cos ≥ min_cos) between rows whose text/vector was created or changed in this
        run and other active rows created/updated within `days`. Pairs already judged this run or
        already linked are skipped, so an unchanged re-run makes no LLM call (DEVIATIONS)."""
        ws, now = self.ws, float(self.ctx.now)
        cutoff = now - float(days) * DAY
        anchors = [ws.rows[i] for i in sorted(ws.new_ids | ws.vec_changed)
                   if i in ws.rows and ws.rows[i].status == "active"]
        pairs: dict[frozenset, tuple[float, str, str]] = {}
        k = max(int(self.cfg.candidate_k) * 2, 10)
        for a in anchors:
            if a.vector is None:
                continue
            for b, c in ws.vector_search(a.vector, k=k, min_cos=float(min_cos), exclude_ids={a.id},
                                         exclude_statuses=ONLY_ACTIVE_EXCLUDED):
                if max(float(b.created_at or 0), float(b.updated_at or 0)) < cutoff:
                    continue
                key = frozenset((a.id, b.id))
                if key in self.judged_pairs or key in pairs:
                    continue
                if (b.id in a.related_ids or a.id in b.related_ids or b.id in a.supersedes
                        or a.id in b.supersedes):
                    continue
                pairs[key] = (c, a.id, b.id)
        outs: list[UpsertOutcome] = []
        for c, a_id, b_id in sorted(pairs.values(), key=lambda t: (-t[0], t[1], t[2]))[: max(0, int(max_pairs))]:
            a, b = ws.get(a_id), ws.get(b_id)
            if a is None or b is None or a.status != "active" or b.status != "active":
                continue
            new, old = (a, b) if (_et(a, now), a.created_at, a.id) >= (_et(b, now), b.created_at, b.id) else (b, a)
            js = self.judge(self._subject_dict(new), [Candidate(row=old, cos=c, via="sweep")])
            self.judged_pairs.add(frozenset((a_id, b_id)))
            outs.append(self._resolve_existing(new, js))
        return outs
