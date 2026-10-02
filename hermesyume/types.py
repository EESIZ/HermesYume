"""Shared enums, constants and in-memory record types passed between dream modules.

Rules:
- All times are epoch seconds (float, UTC). Lance timestamp(ms, UTC) conversion happens only in store.py.
- Vectors are ``numpy.ndarray`` float32, L2-normalized, shape (dim,). numpy is imported lazily so
  this module stays importable without numpy.
- Evidence keys: ``s:<state.db message id>`` | ``m:<sha1(abs md path)[:10]>:<line>`` |
  ``i:<inbox id>`` | ``c:<core target>:<entry sha>`` | ``x:<migration source>:<n>``.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Iterable

# ── kinds (§5.1) ─────────────────────────────────────────────────────────────

KINDS: tuple[str, ...] = ("rule", "profile", "preference", "reference", "procedure", "decision",
                          "lesson", "project", "fact", "state", "schedule", "event", "opinion")
LEGACY_KIND = "legacy"
ALL_KINDS: tuple[str, ...] = KINDS + (LEGACY_KIND,)
PROTECTED_KINDS: frozenset[str] = frozenset({"rule", "profile", "preference", "reference", "procedure"})
EXPIRING_KINDS: frozenset[str] = frozenset({"state", "schedule"})
# rule/profile/preference without user evidence: active + tier decaying (U2; `candidate` is never produced)
USER_EVIDENCE_REQUIRED_KINDS: frozenset[str] = frozenset({"rule", "profile", "preference"})
# kinds a pin may sit on (§4.3 R1 remember pin; a pin follows an explicit supersede of such a row)
PIN_KINDS: frozenset[str] = frozenset({"rule", "profile", "preference", "reference"})

KIND_BASE: dict[str, float] = {
    "rule": 0.85, "profile": 0.85, "preference": 0.75, "reference": 0.70, "procedure": 0.70,
    "decision": 0.65, "lesson": 0.65, "project": 0.60, "fact": 0.50, "state": 0.55,
    "schedule": 0.55, "event": 0.40, "opinion": 0.35, "legacy": 0.20,
}
# Half-life (days) used when a row's tier is `decaying` / `expiring`. Protected kinds are normally
# durable (no decay) or slow (SLOW_HL_DAYS); they are `decaying` only under U2 (assistant-only
# conversation evidence for rule/profile/preference) and then use DECAYING_PROTECTED_HL_DAYS.
DECAYING_PROTECTED_HL_DAYS = 60.0
KIND_HL_DAYS: dict[str, float] = {
    "decision": 120, "lesson": 120, "project": 60, "fact": 60, "state": 14, "schedule": 14,
    "event": 30, "opinion": 21,
    "rule": DECAYING_PROTECTED_HL_DAYS, "profile": DECAYING_PROTECTED_HL_DAYS,
    "preference": DECAYING_PROTECTED_HL_DAYS, "reference": DECAYING_PROTECTED_HL_DAYS,
    "procedure": DECAYING_PROTECTED_HL_DAYS,
}
SLOW_HL_DAYS = 180.0

# "같은 kind 계열" for auto-duplicate (R0-4). Plan leaves families undefined; defined here.
KIND_FAMILY: dict[str, str] = {
    "rule": "norm", "preference": "norm", "procedure": "norm",
    "profile": "identity", "reference": "identity",
    "fact": "knowledge", "project": "knowledge", "decision": "knowledge", "lesson": "knowledge",
    "state": "temporal", "schedule": "temporal", "event": "temporal",
    "opinion": "opinion", "legacy": "legacy",
}

# Korean labels used in Dream Log and the provider recall block.
KIND_LABEL_KO: dict[str, str] = {
    "rule": "규칙", "profile": "프로필", "preference": "선호", "reference": "참조",
    "procedure": "절차", "decision": "결정", "lesson": "교훈", "project": "프로젝트",
    "fact": "사실", "state": "상태", "schedule": "일정", "event": "사건", "opinion": "의견",
    "legacy": "레거시",
}

# ── tiers / targets / statuses (§2.2, §5.2) ──────────────────────────────────

TIERS: tuple[str, ...] = ("pinned", "durable", "slow", "decaying", "expiring", "legacy")
TARGETS: tuple[str, ...] = ("user", "agent", "world")
# `candidate` stays in the enum (schema §2.2) but is never produced (U2).
STATUSES: tuple[str, ...] = ("active", "candidate", "superseded", "expired", "dormant",
                             "forgotten", "quarantined")
SERVING_STATUSES: tuple[str, ...] = ("active", "superseded", "expired", "dormant")
CANDIDATE_SEARCH_EXCLUDED: tuple[str, ...] = ("forgotten", "quarantined")
# R7: an active row moving to one of these is "destructive" (counted for the Dream Log; U2: never held)
DESTRUCTIVE_TARGET_STATUSES: frozenset[str] = frozenset({"superseded", "expired", "dormant", "forgotten"})
CORE_TARGETS: tuple[str, ...] = ("memory", "user")   # Hermes memory-tool targets → MEMORY.md / USER.md

SOURCES: tuple[str, ...] = ("dream", "md", "tool:yume_remember", "core:user", "core:memory",
                            "legacy:memory_md", "legacy:dreamer")
DEBUG_SOURCE_PREFIX = "debug:"      # `yume debug plant` rows: source = "debug:<tag>" (sandbox only)
CORE_SOURCES: frozenset[str] = frozenset({"core:user", "core:memory"})
DURABLE_SOURCES: frozenset[str] = frozenset({"core:user", "core:memory", "tool:yume_remember"})

HISTORY_OPS: tuple[str, ...] = ("insert", "reinforce", "supersede", "consolidate", "text_update",
                                "status", "pin", "unpin", "restore", "core_mirror", "forget")
RELATIONS: tuple[str, ...] = ("duplicate", "state_change", "different_aspects", "unrelated")
RELATION_UNKNOWN = "unknown"
NEWER_VALUES: tuple[str, ...] = ("new", "existing", "same")
EVIDENCE_ROLES: tuple[str, ...] = ("user", "assistant", "agent_log", "core")

RUN_MODES: tuple[str, ...] = ("live", "dry", "migrate")
RUN_STATUSES: tuple[str, ...] = ("planned", "committed", "failed", "dry", "held")
WINDOW_STATUSES: tuple[str, ...] = ("ok", "empty", "failed", "quarantined")
WM_ADVANCING_STATUSES: frozenset[str] = frozenset({"ok", "empty", "quarantined"})

RECALL_EVENT_KINDS: tuple[str, ...] = ("injected", "used", "tool_hit", "shadow")
RECALL_MODES: tuple[str, ...] = ("vector", "keyword", "inbox")
INBOX_OPS: tuple[str, ...] = ("remember", "forget", "core_add", "core_replace", "core_remove",
                              "session_end")
FOLD_CURSOR_RECALL = "recall_events"
FOLD_CURSOR_INBOX = "inbox"

GATE_REASONS: tuple[str, ...] = ("schema", "kind_enum", "length", "evidence_outside",
                                 "level1_not_explicit", "relative_time", "meta_pattern",
                                 "memory_meta", "uuid", "filenames", "secret", "threat",
                                 "assistant_only")
# U4: the only conditions that become operator alerts (everything else → Dream Log notes).
ALERT_CODES: tuple[str, ...] = ("run_failed", "auth_401", "model_mismatch", "scanner_unavailable",
                                "stalled", "window_quarantined", "secret_found",
                                "recall_embed_fail_rate", "serving_stale", "prefetch_p95")

LIST_CAPS: dict[str, int] = {"source_session_ids": 32, "source_message_ids": 64, "origin_keys": 64}
TEXT_MIN_CHARS, TEXT_MAX_CHARS = 15, 400


# ── small pure helpers ───────────────────────────────────────────────────────

def sha256_hex(s: str | bytes) -> str:
    return hashlib.sha256(s.encode("utf-8") if isinstance(s, str) else s).hexdigest()


def norm_text(text: str) -> str:
    """NFKC, casefold, whitespace collapsed, stripped. Used for text_sha and containment checks."""
    t = unicodedata.normalize("NFKC", text or "").casefold()
    return re.sub(r"\s+", " ", t).strip()


def text_sha(text: str) -> str:
    """suppress.text_sha and dedupe keys: sha256(norm_text(text))."""
    return sha256_hex(norm_text(text))


def make_history_id(run_id: str, memory_id: str, op: str, seq: int) -> str:
    """§2.2: sha256(run_id + memory_id + op + seq) — plain concatenation."""
    return sha256_hex(f"{run_id}{memory_id}{op}{seq}")


def make_window_id(source: str, root: str, first_id: int, last_id: int,
                   content_sha: str | None = None) -> str:
    """§3.4: sha256(source + root + first_id + last_id). md windows append the window content
    sha256 (DEVIATIONS.md D2) so a changed prefix re-extracts instead of matching an old id."""
    base = f"{source}{root}{first_id}{last_id}"
    if content_sha:
        base += content_sha
    return sha256_hex(base)


def suppress_guard(text: str) -> str:
    """Numbers/dates + negation flag of a forgotten text, hashed (no text kept). A cosine hit on a
    suppress row blocks a new claim only when this guard also matches, so a forgotten "사물함 12" does
    not swallow a later "사물함 13" (DEVIATIONS F-16)."""
    from .vecutil import has_negation, numeric_tokens
    payload = "|".join(sorted(numeric_tokens(text))) + ("|neg" if has_negation(text) else "|pos")
    return hashlib.sha256(("yume-suppress-guard:" + payload).encode("utf-8")).hexdigest()[:16]


def suppress_reason(run_id: str, text: str) -> str:
    """SuppressRow.reason for a forget: ``forget|run:<run_id>|g:<guard>``."""
    return f"forget|run:{run_id}|g:{suppress_guard(text)}"


def suppress_reason_guard(reason: str | None) -> str | None:
    """The ``g:`` guard of a suppress reason (None for rows written before the guard existed)."""
    for part in str(reason or "").split("|"):
        if part.startswith("g:") and len(part) > 2:
            return part[2:]
    return None


def suppress_reason_run(reason: str | None) -> str | None:
    for part in str(reason or "").split("|"):
        if part.startswith("run:"):
            return part[4:]
    return None


def new_memory_id() -> str:
    import uuid
    return uuid.uuid4().hex


def make_run_id(now: float | None = None) -> str:
    """'20261002-044000-1a2b' (KST stamp + 4 hex). Sortable; unique per second+random."""
    from .clock import fmt_kst
    ts = time.time() if now is None else now
    return f"{fmt_kst(ts, '%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"


def append_capped(existing: Iterable[str], new: Iterable[str], cap: int) -> list[str]:
    """Order-preserving union keeping the most recent `cap` items (new items go last)."""
    out = [x for x in existing]
    for x in new:
        if x in out:
            out.remove(x)
        out.append(x)
    return out[-cap:] if cap and len(out) > cap else out


def vec_to_b64(vec: Any) -> str | None:
    if vec is None:
        return None
    import numpy as np
    return base64.b64encode(np.asarray(vec, dtype="<f4").tobytes()).decode("ascii")


def vec_from_b64(s: str | None) -> Any:
    if not s:
        return None
    import numpy as np
    return np.frombuffer(base64.b64decode(s), dtype="<f4").copy()


# ── input records ────────────────────────────────────────────────────────────

@dataclass
class Message:
    """One sanitized input message (state.db row or md line block)."""
    ref: str                       # "U#1234" | "A#1235" | "U#md:L12" | "A#md:L13" | "L#md:L30"
    key: str                       # canonical evidence key ("s:1234", "m:<sha10>:12")
    role: str                      # "user" | "assistant" | "agent_log"
    text: str
    ts: float
    source: str                    # "statedb" | "md"
    session_id: str | None = None
    msg_id: int | None = None      # state.db messages.id
    line: int | None = None        # md 1-based line number
    platform: str | None = None    # sessions.source


@dataclass
class Window:
    """Extraction unit (§3.4). `text` is exactly what extract sends as the user prompt body."""
    window_id: str
    source: str                    # "statedb" | "md"
    root: str                      # root_session_id | absolute md path
    first_id: int                  # statedb: first extractable msg id | md: start byte offset
    last_id: int                   # statedb: last extractable msg id | md: end byte offset (exclusive)
    start_ts: float
    last_ts: float
    platform: str | None
    title: str
    header: str
    text: str
    messages: list[Message] = field(default_factory=list)   # extractable (evidence allowed)
    context: list[Message] = field(default_factory=list)    # previous exchange, NOT extractable
    session_ids: list[str] = field(default_factory=list)
    attempts: int = 0              # failed attempts so far (from ledger)
    md_path: str | None = None
    md_date: str | None = None
    md_slug: str | None = None

    @property
    def evidence_index(self) -> dict[str, Message]:
        return {m.ref: m for m in self.messages}

    def user_chars(self) -> int:
        return sum(len(m.text) for m in self.messages if m.role == "user")


@dataclass
class RawClaim:
    """Type-checked EXTRACT output (extract.py), before gates."""
    idx: int
    kind: str
    target: str
    subject: str
    text: str
    event_time: str | None
    valid_until: str | None
    level: int
    evidence: list[str]
    explicit: bool
    steps: int | None = None


@dataclass
class Claim:
    """A gated, normalized, (later) embedded candidate memory. Produced by gates/normalize,
    by inbox `remember`/`core_*` folding (R1) and by migrate; consumed by upsert (R0)."""
    origin_key: str                # "<window_id>#<idx>" | "inbox:<id>" | "core:<target>:<sha>" | "x:<src>:<n>"
    source: str                    # SOURCES
    kind: str
    target: str
    subject: str
    text: str
    level: int = 3
    explicit: bool = False         # LLM 'explicit' flag
    steps: int | None = None
    event_time: float | None = None
    valid_until: float | None = None
    status: str = "active"         # active | candidate | expired (set by gates / R1)
    window_id: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    evidence_keys: list[str] = field(default_factory=list)
    evidence_roles: list[str] = field(default_factory=list)   # distinct roles, EVIDENCE_ROLES
    session_ids: list[str] = field(default_factory=list)
    first_seen_at: float = 0.0
    last_seen_at: float = 0.0
    last_user_evidence_at: float | None = None
    user_evidence_count: int = 0   # distinct user messages
    user_session_count: int = 0    # distinct sessions with user evidence
    user_session_ids: list[str] = field(default_factory=list)  # those sessions (⊆ session_ids)
    explicit_user: bool = False    # §5.3 (LLM explicit OR user text regex)
    subject_key: str = ""          # N5
    refs: list[str] = field(default_factory=list)              # N5 verified paths
    importance: float = 0.0        # N5 §5.3
    embed_text: str = ""           # N6 "subject: text" with tails stripped
    vector: Any = None             # N6 np.ndarray float32 (dim,)
    pin: bool = False              # remember(pin=1) request / migration pin
    core_target: str | None = None
    core_sha: str | None = None
    core_required: bool = False
    in_core: bool = False
    lang: str = "ko"
    notes: list[str] = field(default_factory=list)

    @property
    def has_user_evidence(self) -> bool:
        return "user" in self.evidence_roles or self.source in DURABLE_SOURCES

    @property
    def assistant_only(self) -> bool:
        return bool(self.evidence_roles) and set(self.evidence_roles) == {"assistant"}

    @property
    def agent_log_only(self) -> bool:
        return bool(self.evidence_roles) and set(self.evidence_roles) == {"agent_log"}


@dataclass
class Rejection:
    """Gate rejection (N4), listed verbatim in the Dream Log."""
    window_id: str | None
    idx: int
    reason: str                    # GATE_REASONS
    detail: str
    text: str
    kind: str = ""


# ── canonical store rows (§2.2) ──────────────────────────────────────────────

# Field order == Lance schema order (store.py builds the Arrow schema from this list).
MEMORY_FIELDS: tuple[str, ...] = (
    "id", "text", "subject", "subject_key", "vector", "embed_model", "kind", "tier", "target",
    "importance", "level", "pinned", "core_required", "core_target", "core_sha", "in_core",
    "status", "status_reason", "status_changed_at", "created_at", "updated_at",
    "event_time", "valid_from", "valid_until", "first_seen_at", "last_seen_at",
    "last_user_evidence_at", "evidence_count", "user_evidence_count", "user_session_count",
    "explicit_user", "source", "source_session_ids", "source_message_ids", "origin_keys", "refs",
    "recall_injected_count", "recall_injected_strong", "recall_used_count", "search_hit_count",
    "last_recalled_at", "last_used_at", "supersedes", "superseded_by", "related_ids",
    "judge_pending", "needs_review", "version", "lang", "scope", "last_run_id", "schema_version",
)
TS_FIELDS: frozenset[str] = frozenset({
    "status_changed_at", "created_at", "updated_at", "event_time", "valid_from", "valid_until",
    "first_seen_at", "last_seen_at", "last_user_evidence_at", "last_recalled_at", "last_used_at",
})
LIST_FIELDS: frozenset[str] = frozenset({
    "source_session_ids", "source_message_ids", "origin_keys", "refs", "supersedes", "related_ids",
})


@dataclass
class MemoryRow:
    id: str
    text: str
    subject: str = ""
    subject_key: str = ""
    vector: Any = None                     # np.ndarray float32 (dim,), required on commit
    embed_model: str = ""
    kind: str = "fact"
    tier: str = "decaying"
    target: str = "user"
    importance: float = 0.5
    level: int = 3
    pinned: bool = False
    core_required: bool = False
    core_target: str | None = None
    core_sha: str | None = None
    in_core: bool = False
    status: str = "active"
    status_reason: str = ""
    status_changed_at: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0
    event_time: float | None = None
    valid_from: float | None = None
    valid_until: float | None = None
    first_seen_at: float = 0.0
    last_seen_at: float = 0.0
    last_user_evidence_at: float | None = None
    evidence_count: int = 0
    user_evidence_count: int = 0
    user_session_count: int = 0
    explicit_user: bool = False
    source: str = "dream"
    source_session_ids: list[str] = field(default_factory=list)
    source_message_ids: list[str] = field(default_factory=list)   # evidence keys
    origin_keys: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    recall_injected_count: int = 0
    recall_injected_strong: int = 0
    recall_used_count: int = 0
    search_hit_count: int = 0
    last_recalled_at: float | None = None
    last_used_at: float | None = None
    supersedes: list[str] = field(default_factory=list)
    superseded_by: str | None = None
    related_ids: list[str] = field(default_factory=list)
    judge_pending: bool = False
    needs_review: bool = False
    version: int = 1
    lang: str = "ko"
    scope: str = "default"
    last_run_id: str = ""
    schema_version: int = 2

    def copy(self) -> "MemoryRow":
        c = copy.copy(self)
        for f in LIST_FIELDS:
            setattr(c, f, list(getattr(self, f)))
        if self.vector is not None:
            c.vector = self.vector.copy()
        return c

    def snapshot(self, *, include_vector: bool = False) -> dict[str, Any]:
        """JSON-able dict (history before/after_json, plan.json, inspect --json)."""
        d: dict[str, Any] = {}
        for f in MEMORY_FIELDS:
            if f == "vector":
                if include_vector:
                    d["vector_b64"] = vec_to_b64(self.vector)
                continue
            v = getattr(self, f)
            d[f] = list(v) if f in LIST_FIELDS else v
        return d

    @classmethod
    def from_snapshot(cls, d: dict[str, Any]) -> "MemoryRow":
        kw = {k: v for k, v in d.items() if k in MEMORY_FIELDS and k != "vector"}
        row = cls(**kw)
        if d.get("vector_b64"):
            row.vector = vec_from_b64(d["vector_b64"])
        return row


@dataclass
class HistoryRow:
    history_id: str
    memory_id: str
    run_id: str
    op: str                         # HISTORY_OPS
    before_json: str                # json.dumps(snapshot or {}), ensure_ascii=False
    after_json: str
    at: float


@dataclass
class SuppressRow:
    id: str                         # = forgotten memory id (or uuid hex for core/manual)
    vector: Any                     # np.ndarray float32 (dim,)
    text_sha: str                   # types.text_sha(text); original text is NEVER stored
    kind: str
    created_at: float
    reason: str


# ── plan / ops (R0–R8) ───────────────────────────────────────────────────────

@dataclass
class Op:
    """One change to the working set. Final rows = base rows + changes of non-held ops, in seq order."""
    seq: int
    op: str                          # HISTORY_OPS
    memory_id: str
    changes: dict[str, Any]          # field → new value (insert: full snapshot incl. "vector_b64")
    before: dict[str, Any] = field(default_factory=dict)   # field → old value of changed fields
    reason: str = ""
    destructive: bool = False        # active → DESTRUCTIVE_TARGET_STATUSES (counts for mass_change)
    protected_change: bool = False   # changes a pinned/durable row without user evidence
    user_evidence: bool = False
    held: bool = False
    detail: dict[str, Any] = field(default_factory=dict)   # judge relation, cos, other ids …


@dataclass
class Judgement:
    candidate_id: str
    relation: str                    # RELATIONS + "unknown"
    newer: str = "same"              # NEWER_VALUES
    cos: float | None = None


@dataclass
class Candidate:
    row: "MemoryRow"
    cos: float
    via: str                         # "lance" | "run_new" | "subject_key"


@dataclass
class UpsertOutcome:
    action: str   # skip_idempotent|suppressed|duplicate|reinforce_noop|inserted|state_change|
                  # inserted_superseded|consolidated|related|candidate|judge_pending|revived
    memory_id: str | None
    judgements: list[Judgement] = field(default_factory=list)
    ops: list[int] = field(default_factory=list)            # Op.seq values produced


@dataclass
class StrengthResult:
    tier: str
    strength: float
    t_ref: float
    hl_days: float | None            # None = no decay (durable/pinned) or legacy
    transition: "Transition | None" = None


@dataclass
class Transition:
    memory_id: str
    from_status: str
    to_status: str                   # "expired" | "dormant" | "active" | "purge"
    reason: str


@dataclass
class GuardResult:
    held: bool = False
    reasons: list[str] = field(default_factory=list)
    destructive_count: int = 0
    threshold: int = 0
    held_seqs: list[int] = field(default_factory=list)


@dataclass
class DocWrite:
    """N8 procedure doc, written after commit to {workspace}/docs/yume/<slug>.md."""
    slug: str
    path: str                        # absolute target path
    title: str
    body: str                        # masked + threat-checked
    memory_id: str | None = None
    origin_key: str = ""


# ── ledger records (§2.3) ────────────────────────────────────────────────────

@dataclass
class Watermark:
    root_session_id: str
    last_ts: float
    last_id: int
    updated_run: str | None = None


@dataclass
class MdFileState:
    path: str
    sha256: str
    processed_bytes: int
    prefix_sha256: str
    status: str                      # "ok" | "partial" | "failed"
    run_id: str | None = None


@dataclass
class WindowState:
    window_id: str
    source: str
    root_session_id: str
    first_id: int
    last_id: int
    last_ts: float
    status: str                      # WINDOW_STATUSES
    attempts: int = 0
    last_error: str | None = None
    run_id: str | None = None
    n_claims: int = 0


@dataclass
class RunRecord:
    run_id: str
    started_at: float
    finished_at: float | None = None
    mode: str = "live"
    now_override: float | None = None
    status: str = "planned"
    lance_version_before: int | None = None
    lance_version_after: int | None = None
    wm_before_json: str | None = None
    stats_json: str | None = None
    error: str | None = None


@dataclass
class CoreSeenRow:
    target: str
    entry_sha: str
    text: str
    memory_id: str | None
    first_seen_run: str | None
    last_seen_run: str | None
    present: bool


@dataclass
class AuditRow:
    ts: float
    run_id: str
    op: str
    memory_id: str | None
    detail: str                      # never the forgotten text


@dataclass
class LedgerDelta:
    """Everything R8-5 writes in ONE ledger transaction."""
    watermarks: dict[str, tuple[float, int]] = field(default_factory=dict)  # root → (last_ts, last_id)
    session_roots: dict[str, str] = field(default_factory=dict)
    windows: list[WindowState] = field(default_factory=list)
    md_files: list[MdFileState] = field(default_factory=list)
    cursors: dict[str, int] = field(default_factory=dict)                   # fold_cursor
    core_seen: list[CoreSeenRow] = field(default_factory=list)
    audit: list[AuditRow] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.watermarks or self.session_roots or self.windows or self.md_files
                    or self.cursors or self.core_seen or self.audit)


# ── live.db records (§2.4) ───────────────────────────────────────────────────

@dataclass
class LiveSnapshot:
    max_inbox_id: int
    max_recall_id: int


@dataclass
class RecallEvent:
    id: int
    ts: float
    session_id: str | None
    platform: str | None
    turn_no: int | None
    memory_id: str                   # Lance id, or "inbox:<inbox id>" for mode='inbox'
    kind: str                        # RECALL_EVENT_KINDS
    cos: float | None
    mode: str | None                 # RECALL_MODES
    snapshot_run: str | None


@dataclass
class InboxItem:
    id: int
    ts: float
    session_id: str | None
    platform: str | None
    op: str                          # INBOX_OPS
    text: str | None
    old_text: str | None
    kind: str | None
    pin: bool
    memory_id: str | None
    target: str | None               # core ops: "memory" | "user"
    vec: Any                         # np.ndarray float32 | None (decoded BLOB)
    embed_model: str | None
    meta: dict[str, Any]
    status: str
    consumed_run: str | None


@dataclass
class HealthRow:
    id: int
    ts: float
    pid: int | None
    platform: str | None
    prefetch_n: int
    injected_n: int
    empty_n: int
    embed_fail_n: int
    fts_fallback_n: int
    timeout_n: int
    p95_ms: int | None
    last_error_class: str | None
    snapshot_run: str | None


# ── core files ───────────────────────────────────────────────────────────────

@dataclass
class CoreEntry:
    target: str                      # "memory" | "user"
    index: int
    text: str                        # exact entry text (stripped)
    sha: str                         # corefmt.core_sha(text)
    label: str | None                # leading "**...:**" label, if any


# ── operator alerts / run accounting ─────────────────────────────────────────

@dataclass
class Alert:
    code: str                        # alerts.ALERT_CODES
    message: str                     # Korean, operator-facing
    level: str = "warn"              # "info" | "warn" | "error"
    run_id: str | None = None
    ts: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


class BudgetExceeded(RuntimeError):
    def __init__(self, what: str):
        super().__init__(f"budget exceeded: {what}")
        self.what = what


@dataclass
class RunBudget:
    """Per-run caps (§4.1). Deadline uses time.monotonic (independent of --now)."""
    max_llm_calls: int = 400
    max_embed_inputs: int = 3000
    max_runtime_s: float = 40 * 60
    llm_calls: int = 0
    embed_inputs: int = 0
    started_mono: float = field(default_factory=time.monotonic)

    def time_left(self) -> float:
        return self.max_runtime_s - (time.monotonic() - self.started_mono)

    def expired(self) -> bool:
        return self.time_left() <= 0

    def can_llm(self, n: int = 1) -> bool:
        return self.llm_calls + n <= self.max_llm_calls and not self.expired()

    def take_llm(self, n: int = 1) -> None:
        if self.expired():
            raise BudgetExceeded("runtime")
        if self.llm_calls + n > self.max_llm_calls:
            raise BudgetExceeded("llm_calls")
        self.llm_calls += n

    def can_embed(self, n: int) -> bool:
        return self.embed_inputs + n <= self.max_embed_inputs and not self.expired()

    def take_embed(self, n: int) -> None:
        if self.expired():
            raise BudgetExceeded("runtime")
        if self.embed_inputs + n > self.max_embed_inputs:
            raise BudgetExceeded("embed_inputs")
        self.embed_inputs += n


@dataclass
class RunStats:
    """Counters. `yume dream --json` prints {"run_id","status","stats": RunStats.to_dict()}."""
    run_id: str = ""
    mode: str = "live"
    status: str = ""
    # inputs (N1)
    sessions_seen: int = 0
    messages_in: int = 0
    md_files: int = 0
    inbox_items: int = 0
    excluded: dict[str, int] = field(default_factory=dict)        # reason → count
    # windows (N2/N3)
    windows_total: int = 0
    windows_ok: int = 0
    windows_empty: int = 0
    windows_failed: int = 0
    windows_quarantined: int = 0
    windows_deferred: int = 0       # over max_windows_per_run / budget
    # claims (N3–N6)
    claims_extracted: int = 0
    claims_rejected: int = 0
    rejected_by_reason: dict[str, int] = field(default_factory=dict)
    # REM (R0–R6)
    created: int = 0
    reinforced: int = 0
    reinforce_noop: int = 0         # duplicate with non-user evidence (recorded, no strength gain)
    superseded: int = 0
    consolidated: int = 0
    related_linked: int = 0
    judge_calls: int = 0
    judge_pending: int = 0
    judge_failures: int = 0
    suppressed_hits: int = 0
    idempotent_skips: int = 0
    expired: int = 0
    dormant: int = 0
    revived: int = 0
    forgotten: int = 0
    purged: int = 0
    quarantined: int = 0
    pinned_new: int = 0
    core_changes: int = 0
    # recall fold (R4)
    recall_injected: int = 0
    recall_used: int = 0
    recall_tool_hit: int = 0
    recall_shadow: int = 0
    recall_ignored_platform: int = 0
    # guard / commit
    held_ops: int = 0
    docs_written: int = 0
    # cost
    llm_calls: int = 0
    embed_inputs: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embed_tokens: int = 0
    cost_usd: float = 0.0
    lance_version_before: int | None = None
    lance_version_after: int | None = None
    duration_s: float = 0.0

    def bump(self, name: str, n: int = 1) -> None:
        setattr(self, name, getattr(self, name) + n)

    def bump_reason(self, table: str, reason: str, n: int = 1) -> None:
        d = getattr(self, table)
        d[reason] = d.get(reason, 0) + n

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunReport:
    """Verbatim lists for the Dream Log (Korean rendering in dream_log.py). Items are plain dicts
    so they serialize into plan.json; each producer documents its dict keys in CONTRACTS.md."""
    inputs: list[dict] = field(default_factory=list)          # {"source","root","title","messages"}
    claims: list[dict] = field(default_factory=list)          # {"origin_key","kind","subject","text","status"}
    rejections: list[dict] = field(default_factory=list)      # asdict(Rejection)
    created: list[dict] = field(default_factory=list)         # {"id","kind","tier","text","importance"}
    reinforced: list[dict] = field(default_factory=list)      # {"id","text","user": bool}
    superseded: list[dict] = field(default_factory=list)      # {"old_id","old_text","new_id","new_text"}
    consolidated: list[dict] = field(default_factory=list)    # {"id","a","b","result"}
    related: list[dict] = field(default_factory=list)         # {"a_id","b_id","reason"}
    expired: list[dict] = field(default_factory=list)         # {"id","text","strength"}
    dormant: list[dict] = field(default_factory=list)
    revived: list[dict] = field(default_factory=list)
    forgotten: list[dict] = field(default_factory=list)       # {"id"} ONLY (no text)
    purged: list[dict] = field(default_factory=list)          # {"id"}
    held: list[dict] = field(default_factory=list)            # {"seq","op","memory_id","reason"}
    new_pins: list[dict] = field(default_factory=list)        # {"id","text"} (Dream Log only, U4)
    core_changes: list[dict] = field(default_factory=list)    # {"target","change","text"}
    recall_top: list[dict] = field(default_factory=list)      # {"id","text","injected","used"}
    durable_unrecalled: list[dict] = field(default_factory=list)
    cos_by_relation: dict[str, list[float]] = field(default_factory=dict)
    strip_lines_top: list[dict] = field(default_factory=list) # {"line","count"}
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunContext:
    """Everything a dream stage needs. Typed as Any where the type lives in a heavier module
    (llm.LLM, embedder.Embedder, threat.ThreatScanner, store.Store, ledger.Ledger, livedb.LiveDB)."""
    paths: Any                       # paths.Paths
    cfg: Any                         # config.Config
    run_id: str
    now: float
    mode: str = "live"               # RUN_MODES
    dry_run: bool = False
    offline: bool = False
    llm: Any = None
    embedder: Any = None
    scanner: Any = None
    store: Any = None
    ledger: Any = None
    live: Any = None
    budget: RunBudget = field(default_factory=RunBudget)
    stats: RunStats = field(default_factory=RunStats)
    report: RunReport = field(default_factory=RunReport)
    alerts: list[Alert] = field(default_factory=list)
    settle_minutes: int | None = None     # --settle-minutes override
    approve_migration: bool = False
    log: Any = None                  # logging.Logger
    purged: dict[str, dict] = field(default_factory=dict)   # rows hard-deleted this run (retention scrub; never serialized)

    def alert(self, code: str, message: str, level: str = "warn", **details: Any) -> Alert:
        """Operator alert — only for ALERT_CODES (U4). Anything else → note()."""
        if code not in ALERT_CODES:
            raise ValueError(f"not an alert condition (U4): {code!r}; use ctx.note()")
        a = Alert(code=code, message=message, level=level, run_id=self.run_id, ts=self.now,
                  details=details)
        self.alerts.append(a)
        return a

    def note(self, message: str) -> None:
        """Dream Log-only remark (new pin, pin budget, guard hold, backlog, fail rates …)."""
        self.report.notes.append(message)


@dataclass
class Plan:
    """One run's complete, replayable change set (serialized to runs/<run_id>/plan.json by plan.py).
    `upserts` are FINAL row states (base + non-held ops); held ops stay only in `ops` (held=True)."""
    run_id: str
    mode: str
    created_at: float
    now: float
    lance_version_before: int | None
    upserts: list[MemoryRow] = field(default_factory=list)
    history: list[HistoryRow] = field(default_factory=list)
    suppress: list[SuppressRow] = field(default_factory=list)
    purge_ids: list[str] = field(default_factory=list)
    ops: list[Op] = field(default_factory=list)
    guard: GuardResult = field(default_factory=GuardResult)
    ledger_delta: LedgerDelta = field(default_factory=LedgerDelta)
    inbox_consume_ids: list[int] = field(default_factory=list)
    inbox_skip_ids: list[int] = field(default_factory=list)      # marked status='skipped'
    docs: list[DocWrite] = field(default_factory=list)
    stats: RunStats = field(default_factory=RunStats)
    report: RunReport = field(default_factory=RunReport)
    status: str = "planned"          # planned | dry | committed | held

    def store_noop(self) -> bool:
        return not (self.upserts or self.history or self.suppress or self.purge_ids)

    def is_noop(self) -> bool:
        """No Lance change, no ledger change, nothing to consume → R8 steps 2–7 are skipped."""
        return (self.store_noop() and self.ledger_delta.is_empty() and not self.inbox_consume_ids
                and not self.inbox_skip_ids)


@dataclass
class CommitResult:
    run_id: str
    versions_after: dict[str, int] = field(default_factory=dict)   # Store.versions()
    lance_version_after: int | None = None                         # memories table
    skipped: bool = False                                          # plan.is_noop()
    replayed: bool = False


def dataclass_fields(cls: type) -> list[str]:
    return [f.name for f in fields(cls)]
