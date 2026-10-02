"""Episodes from Hermes' state.db, read-only (PLAN-v2 §3.1–3.2; CONTRACTS §4.1; v1 sessions.py ADAPT).

The caller passes a connection from ``sqlite_util.open_for_read(paths.state_db, pure=dry_run)``
(``mode=ro`` + ``query_only``, or a private snapshot copy for dry-runs). Only ``content`` is read —
never ``api_content``, so recalled memory that Hermes injected into the API copy of a turn cannot
be learned back.

Progress is tracked per compression lineage, not per session:
- root: follow ``parent_session_id`` upward while the parent ended with ``end_reason='compression'``
  (cached in ledger ``session_root`` so a pruned parent never splits a lineage);
- dedupe: generation copies share (role, content, timestamp) — keep the lowest id, across all
  sessions of the lineage (compression children copy the parent's tail with identical timestamps);
- watermark ``(last_ts, last_id)``: ``ts > last_ts``, or ``ts == last_ts and id > last_id`` and the
  (role, sha256(content), ts) key was not already seen at or before the watermark.
So neither a generation copy nor a compression child's tail is ever processed twice, whatever run
it shows up in.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..clock import DAY
from ..types import Message, Watermark

log = logging.getLogger("hermesyume.sources.statedb")

CONTENT_JSON_PREFIX = "\x00json:"
_TEXT_PART_TYPES = ("text", "input_text", "output_text")
ROLES = ("user", "assistant")
_IN_CHUNK = 400   # max bound parameters per IN (...) list


@dataclass
class SessionInfo:
    id: str
    source: str
    title: str | None
    user_id: str | None
    chat_type: str | None
    cwd: str | None
    parent_session_id: str | None
    end_reason: str | None
    started_at: float
    ended_at: float | None
    hidden: bool


@dataclass
class Lineage:
    root: str                       # root_session_id
    platform: str                   # root session source
    title: str                      # root title, else first non-empty title in lineage, else ""
    chat_type: str | None
    sessions: list[SessionInfo]     # root first, then descendants by started_at
    messages: list[Message]         # eligible: after watermark, deduped, settled, ascending (ts, id); RAW text
    context_before: list[Message]   # last already-processed exchange before messages[0]; [] if none
    wm: Watermark | None
    fully_settled: bool             # False if a trailing exchange was deferred


@dataclass
class StateDBLoad:
    lineages: list[Lineage] = field(default_factory=list)
    excluded: dict[str, int] = field(default_factory=dict)
    session_roots: dict[str, str] = field(default_factory=dict)
    sessions_seen: int = 0
    messages_in: int = 0


# ── content ──────────────────────────────────────────────────────────────────

def decode_content(content: Any) -> str:
    """Flatten Hermes message content to plain text (``\\x00json:`` multimodal parts → text parts)."""
    if content is None:
        return ""
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    if not isinstance(content, str):
        return str(content)
    if not content.startswith(CONTENT_JSON_PREFIX):
        return content
    try:
        parts = json.loads(content[len(CONTENT_JSON_PREFIX):])
    except (json.JSONDecodeError, ValueError):
        return ""
    if isinstance(parts, dict):
        parts = [parts]
    if isinstance(parts, str):
        return parts
    texts: list[str] = []
    for p in parts if isinstance(parts, list) else []:
        if isinstance(p, dict) and p.get("type") in _TEXT_PART_TYPES:
            t = p.get("text")
            if isinstance(t, str):
                texts.append(t)
        elif isinstance(p, str):
            texts.append(p)
    return "\n".join(t for t in texts if t)


def _content_sha(content: Any) -> str:
    if content is None:
        raw = b""
    elif isinstance(content, bytes):
        raw = content
    else:
        raw = str(content).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(raw).hexdigest()


# ── schema probing ───────────────────────────────────────────────────────────

def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _col(cols: set[str], name: str, default: str) -> str:
    return name if name in cols else f"{default} AS {name}"


_SESSION_COLS = (("title", "NULL"), ("user_id", "NULL"), ("chat_type", "NULL"), ("cwd", "NULL"),
                 ("parent_session_id", "NULL"), ("end_reason", "NULL"), ("ended_at", "NULL"),
                 ("hidden", "0"))
_MSG_OPT_COLS = (("active", "1"), ("compacted", "0"), ("_compressed_summary", "0"),
                 ("display_kind", "NULL"))


def _load_sessions(conn: sqlite3.Connection) -> dict[str, SessionInfo]:
    scols = _columns(conn, "sessions")
    if not scols:
        return {}
    sel = ", ".join(["id", "source", "started_at"] + [_col(scols, c, d) for c, d in _SESSION_COLS])
    out: dict[str, SessionInfo] = {}
    for r in conn.execute(f"SELECT {sel} FROM sessions"):
        out[str(r[0])] = SessionInfo(
            id=str(r[0]), source=r[1] or "", started_at=float(r[2] or 0.0), title=r[3],
            user_id=None if r[4] is None else str(r[4]), chat_type=r[5], cwd=r[6],
            parent_session_id=r[7], end_reason=r[8],
            ended_at=None if r[9] is None else float(r[9]), hidden=bool(r[10]))
    return out


def _msg_select(conn: sqlite3.Connection) -> str:
    mcols = _columns(conn, "messages")
    opt = ", ".join(_col(mcols, c, d) for c, d in _MSG_OPT_COLS)
    # NOTE: api_content is deliberately never selected (§3.1).
    return f"SELECT id, session_id, role, content, timestamp, {opt} FROM messages"


# ── lineage roots ────────────────────────────────────────────────────────────

def _root_of(sid: str, sessions: dict[str, SessionInfo], cache: dict[str, str]) -> str:
    """Climb compression edges. A cached session keeps its cached root (a parent pruned from
    state.db must not split its lineage); a missing parent that the cache knows is joined."""
    chain: list[str] = []
    cur = sid
    while True:
        if cur in cache:
            root = cache[cur]
            break
        chain.append(cur)
        s = sessions.get(cur)
        parent = s.parent_session_id if s is not None else None
        if not parent or parent in chain:
            root = cur
            break
        p = sessions.get(parent)
        if p is None:
            root = cache.get(parent, cur)
            break
        if p.end_reason != "compression":
            root = cur
            break
        cur = parent
    for c in chain:
        cache[c] = root
    return root


def resolve_root(conn: sqlite3.Connection, session_id: str, cache: dict[str, str]) -> str:
    """Follow parent_session_id while the parent's end_reason is 'compression'. `cache` maps
    session → root (seed it with ledger.all_roots()); new entries are added to it."""
    if session_id in cache:
        return cache[session_id]
    return _root_of(session_id, _load_sessions(conn), cache)


# ── config helpers ───────────────────────────────────────────────────────────

def _cfg(cfg: Any, key: str, default: Any) -> Any:
    try:
        return cfg[key]
    except (KeyError, TypeError):
        return getattr(cfg, key, default)


@dataclass
class _Filters:
    include_sources: set[str]
    owners: set[str]
    chat_types: list[Any]
    synthetic_re: re.Pattern | None
    deny_globs: list[str]

    @classmethod
    def from_cfg(cls, cfg: Any) -> "_Filters":
        rx = _cfg(cfg, "exclude_first_message_regex", "") or ""
        try:
            synth = re.compile(rx) if rx else None
        except re.error as e:
            raise ValueError(f"exclude_first_message_regex invalid: {e}") from e
        return cls(include_sources=set(_cfg(cfg, "include_sources", ["telegram", "cli", "tui"]) or []),
                   owners={str(x) for x in (_cfg(cfg, "owner_user_ids", []) or [])},
                   chat_types=list(_cfg(cfg, "allowed_chat_types", [None, "dm", "private"])),
                   synthetic_re=synth,
                   deny_globs=list(_cfg(cfg, "deny_cwd_globs", []) or []))


def _bump(d: dict[str, int], key: str, n: int = 1) -> None:
    if n:
        d[key] = d.get(key, 0) + n


def _chunks(seq: list[str], n: int = _IN_CHUNK) -> Iterable[list[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ── row classification ───────────────────────────────────────────────────────

def _exclusion(r: sqlite3.Row | tuple) -> str | None:
    """None if the row passes §3.1 message filters, else the counter key."""
    role = r[2]
    if role not in ROLES:
        return "tool_role"
    if not (int(r[5] or 0) == 1 or int(r[6] or 0) == 1):
        return "inactive"            # rewound / soft-deleted (active=0, compacted=0)
    if int(r[7] or 0):
        return "compressed_summary"
    if (r[8] or "") == "hidden":
        return "hidden_message"
    return None


def _key(r) -> tuple[str, str, float]:
    return (r[2], _content_sha(r[3]), float(r[4]))


def _fetch_rows(conn: sqlite3.Connection, select: str, sids: list[str], *,
                ts_min: float | None = None, ts_max: float | None = None,
                role: str | None = None) -> list[tuple]:
    rows: list[tuple] = []
    for chunk in _chunks(sids):
        q = f"{select} WHERE session_id IN ({','.join('?' * len(chunk))})"
        args: list[Any] = list(chunk)
        if ts_min is not None:
            q += " AND timestamp >= ?"
            args.append(ts_min)
        if ts_max is not None:
            q += " AND timestamp <= ?"
            args.append(ts_max)
        if role is not None:
            q += " AND role = ?"
            args.append(role)
        rows.extend(tuple(r) for r in conn.execute(q, args))
    rows.sort(key=lambda r: (float(r[4]), int(r[0])))
    return rows


def _to_message(r: tuple, text: str, platform: str) -> Message:
    mid = int(r[0])
    return Message(ref=f"{'U' if r[2] == 'user' else 'A'}#{mid}", key=f"s:{mid}", role=r[2],
                   text=text, ts=float(r[4]), source="statedb", session_id=str(r[1]),
                   msg_id=mid, platform=platform)


def _after(ts: float, mid: int, wm: Watermark | None) -> bool:
    return wm is None or ts > wm.last_ts or (ts == wm.last_ts and mid > wm.last_id)


# ── lineage assembly ─────────────────────────────────────────────────────────

class _Index:
    """Sessions grouped into lineages (shared by load_lineages and recent_texts)."""

    def __init__(self, conn: sqlite3.Connection, cache: dict[str, str]):
        self.conn = conn
        self.sessions = _load_sessions(conn)
        self.cache = cache
        self.groups: dict[str, list[SessionInfo]] = {}
        for sid in self.sessions:
            root = _root_of(sid, self.sessions, cache)
            self.groups.setdefault(root, []).append(self.sessions[sid])
        for root, ss in self.groups.items():
            ss.sort(key=lambda s: (0 if s.id == root else 1, s.started_at, s.id))
        self.select = _msg_select(conn)

    def lineage_reason(self, root: str, f: _Filters) -> str | None:
        ss = self.groups[root]
        head = ss[0]
        if head.source not in f.include_sources:
            return "source"
        if f.owners and head.user_id is not None and head.user_id not in f.owners:
            return "owner"
        if head.chat_type not in f.chat_types:
            return "chat_type"
        # the regex is the primary filter (§0.2); the narrow cwd deny list comes after it
        if f.synthetic_re is not None:
            first = self.first_user_text(root)
            if first and (f.synthetic_re.search(first) or f.synthetic_re.search(first.lstrip())):
                return "synthetic"
        if f.deny_globs and any(s.cwd and any(fnmatch.fnmatch(s.cwd, g) for g in f.deny_globs)
                                for s in ss):
            return "deny_cwd"
        return None

    def visible_sids(self, root: str) -> list[str]:
        return [s.id for s in self.groups[root] if not s.hidden]

    def first_user_text(self, root: str) -> str | None:
        sids = self.visible_sids(root)
        best: tuple | None = None
        for chunk in _chunks(sids):
            for r in self.conn.execute(
                    f"{self.select} WHERE session_id IN ({','.join('?' * len(chunk))}) AND role='user' "
                    "ORDER BY timestamp, id", chunk):
                r = tuple(r)
                if _exclusion(r) is not None:
                    continue
                if not decode_content(r[3]).strip():
                    continue
                if best is None or (float(r[4]), int(r[0])) < (float(best[4]), int(best[0])):
                    best = r
                break
        return decode_content(best[3]) if best is not None else None


def _dedupe(rows: list[tuple]) -> tuple[list[tuple], list[tuple]]:
    """(kept rows, dropped copies) — rows already sorted by (ts, id); keep the lowest id."""
    seen: set[tuple] = set()
    kept, dropped = [], []
    for r in rows:
        k = _key(r)
        if k in seen:
            dropped.append(r)
        else:
            seen.add(k)
            kept.append(r)
    return kept, dropped


def _context_before(idx: _Index, sids: list[str], wm: Watermark, platform: str) -> list[Message]:
    """Last user message with (ts, id) <= watermark plus its following replies up to the watermark."""
    cand = None
    for chunk in _chunks(sids):
        for r in idx.conn.execute(
                f"{idx.select} WHERE session_id IN ({','.join('?' * len(chunk))}) AND role='user' "
                "AND (timestamp < ? OR (timestamp = ? AND id <= ?)) ORDER BY timestamp DESC, id DESC",
                (*chunk, wm.last_ts, wm.last_ts, wm.last_id)):
            r = tuple(r)
            if _exclusion(r) is None and decode_content(r[3]).strip():
                if cand is None or (float(r[4]), int(r[0])) > (float(cand[4]), int(cand[0])):
                    cand = r
                break
    if cand is None:
        return []
    rows = [r for r in _fetch_rows(idx.conn, idx.select, sids, ts_min=float(cand[4]),
                                   ts_max=wm.last_ts)
            if (float(r[4]), int(r[0])) <= (wm.last_ts, wm.last_id) and _exclusion(r) is None]
    rows, _ = _dedupe(rows)
    msgs: list[Message] = []
    started = False
    for r in rows:
        text = decode_content(r[3]).strip()
        if not text:
            continue
        if r[2] == "user":
            msgs = []          # keep only the LAST user message and what follows it
            started = True
        if started:
            msgs.append(_to_message(r, text, platform))
    return msgs


def _marker_ts(session_end_ids: Any, sid: str | None) -> float | None:
    """session_end marker time (live.db) for a session; a plain set (legacy callers) = no limit."""
    if sid is None or sid not in session_end_ids:
        return None
    if isinstance(session_end_ids, dict):
        v = session_end_ids.get(sid)
        return float("inf") if v is None else float(v)
    return float("inf")


def _settle(msgs: list[Message], *, cutoff: float,
            session_end_ids: Any) -> tuple[list[Message], int, bool]:
    """(eligible, deferred_count, fully_settled).

    A session_end marker is a flush point, not proof the session is over (Hermes also calls
    on_session_end at context compaction / cache eviction while the session id lives on). It
    settles only that session's messages up to the marker time, and it waives the "last exchange
    may still be growing" cut only when no message of the session is newer than the marker
    (DEVIATIONS F-8)."""
    last_ts: dict[str | None, float] = {}
    for m in msgs:
        last_ts[m.session_id] = max(last_ts.get(m.session_id, float("-inf")), float(m.ts))

    def by_marker(m: Message) -> bool:
        t = _marker_ts(session_end_ids, m.session_id)
        return t is not None and float(m.ts) <= t

    def ended(sid: str | None) -> bool:
        t = _marker_ts(session_end_ids, sid)
        return t is not None and last_ts.get(sid, float("-inf")) <= t

    n = 0
    for m in msgs:
        if m.ts <= cutoff or by_marker(m):
            n += 1
        else:
            break
    if n == len(msgs):
        return msgs, 0, True
    eligible = msgs[:n]
    rest0 = msgs[n]
    if eligible and not ended(eligible[-1].session_id) and rest0.role != "user":
        # the last exchange may still be growing: defer it from its user message on
        cut = 0
        for i in range(len(eligible) - 1, -1, -1):
            if eligible[i].role == "user":
                cut = i
                break
        eligible = eligible[:cut]
    return eligible, len(msgs) - len(eligible), False


def load_lineages(conn: sqlite3.Connection, *, ledger: Any, cfg: Any, now: float,
                  settle_minutes: int, session_end_ids: Any) -> StateDBLoad:
    """Eligible messages per lineage (RAW text; the caller sanitizes). See module docstring."""
    out = StateDBLoad()
    if not _columns(conn, "sessions") or not _columns(conn, "messages"):
        return out
    filters = _Filters.from_cfg(cfg)
    known_roots: dict[str, str] = dict(ledger.all_roots()) if ledger is not None else {}
    cache = dict(known_roots)
    idx = _Index(conn, cache)
    out.sessions_seen = len(idx.sessions)
    cutoff = now - float(settle_minutes) * 60.0
    ends = dict(session_end_ids) if isinstance(session_end_ids, dict) else set(session_end_ids or ())
    exc = out.excluded

    for root in sorted(idx.groups):
        ss = idx.groups[root]
        reason = idx.lineage_reason(root, filters)
        if reason is not None:
            _bump(exc, reason)
            continue
        hidden = [s for s in ss if s.hidden]
        _bump(exc, "hidden_session", len(hidden))
        sids = idx.visible_sids(root)
        if not sids:
            continue
        head = ss[0]
        platform = head.source
        wm = ledger.get_wm(root) if ledger is not None else None
        rows = _fetch_rows(conn, idx.select, sids, ts_min=wm.last_ts if wm else None)

        included: list[tuple] = []
        for r in rows:
            why = _exclusion(r)
            if why is None:
                included.append(r)
            elif _after(float(r[4]), int(r[0]), wm):
                _bump(exc, why)
        kept, dropped = _dedupe(included)
        _bump(exc, "generation_copy",
              sum(1 for r in dropped if _after(float(r[4]), int(r[0]), wm)))
        wm_keys: set[tuple] = set()
        if wm is not None:
            wm_keys = {_key(r) for r in rows
                       if float(r[4]) == wm.last_ts and int(r[0]) <= wm.last_id}
        msgs: list[Message] = []
        for r in kept:
            ts, mid = float(r[4]), int(r[0])
            if not _after(ts, mid, wm):
                continue
            if wm is not None and ts == wm.last_ts and _key(r) in wm_keys:
                _bump(exc, "generation_copy")
                continue
            text = decode_content(r[3]).strip()
            if not text:
                _bump(exc, "empty")
                continue
            msgs.append(_to_message(r, text, platform))
        if not msgs:
            continue
        eligible, deferred, fully = _settle(msgs, cutoff=cutoff, session_end_ids=ends)
        _bump(exc, "not_settled", deferred)
        if not eligible:
            continue
        title = head.title or next((s.title for s in ss if s.title), None) or ""
        lin = Lineage(root=root, platform=platform, title=title, chat_type=head.chat_type,
                      sessions=list(ss), messages=eligible,
                      context_before=_context_before(idx, sids, wm, platform) if wm else [],
                      wm=wm, fully_settled=fully)
        out.lineages.append(lin)
        out.messages_in += len(eligible)
        for s in ss:
            if s.id not in known_roots:
                out.session_roots[s.id] = root

    out.lineages.sort(key=lambda lin: (lin.messages[0].ts, lin.messages[0].msg_id or 0, lin.root))
    return out


def recent_texts(conn: sqlite3.Connection, *, cfg: Any, now: float,
                 days: int) -> list[tuple[str, str]]:
    """(evidence key "s:<id>", decoded content) of included user/assistant messages with
    ts >= now - days·DAY (repeat-line statistics, §3.3-3). Generation copies counted once."""
    if not _columns(conn, "sessions") or not _columns(conn, "messages"):
        return []
    filters = _Filters.from_cfg(cfg)
    idx = _Index(conn, {})
    since = now - float(days) * DAY
    out: list[tuple[str, str]] = []
    for root in sorted(idx.groups):
        if idx.lineage_reason(root, filters) is not None:
            continue
        sids = idx.visible_sids(root)
        if not sids:
            continue
        rows = [r for r in _fetch_rows(conn, idx.select, sids, ts_min=since)
                if _exclusion(r) is None]
        kept, _ = _dedupe(rows)
        for r in kept:
            text = decode_content(r[3])
            if text.strip():
                out.append((f"s:{int(r[0])}", text))
    return out
