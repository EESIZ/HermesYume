"""HermesYume v2 — Yume long-term memory recall as a Hermes MemoryProvider plugin.

Installed at ``$HERMES_HOME/plugins/hermesyume/`` and enabled with ``memory.provider: hermesyume``.
Standard library only (PLAN-v2 §6). The provider never opens LanceDB: it reads the read-only
serving copy ``hermesyume/serving/recall.sqlite`` built nightly by ``yume dream`` and appends
recall events / inbox rows to ``hermesyume/live.db``. It never writes MEMORY.md / USER.md, never
calls Telegram, and never raises into Hermes (every hook falls back to a safe default).

Import and ``register()`` do no file, database, thread or network work (discovery imports this
module and calls ``is_available()``).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider, RecallStatus

from ._yume import config as _cfg
from ._yume import core_snapshot, embed_http, live, serving, textutil
from ._yume import live_schema
from ._yume import used as _used

log = logging.getLogger("hermesyume.provider")

KINDS = ("rule", "profile", "preference", "reference", "procedure", "decision", "lesson",
         "project", "fact", "state", "schedule", "event", "opinion")
_PROTECTED = frozenset({"rule", "profile", "preference", "reference", "procedure"})
_INBOX_STRENGTH = 0.8          # same-day remember rows: importance floor (§4.3 R1)
_PENDING_KEEP = 8
# yume_search(include_inactive) neutral marker: the fact itself is no longer current. Dormant rows
# carry none (dormancy is a memory-system state, not a property of the fact). DEVIATIONS E2E-8.
_NOT_CURRENT = frozenset({"superseded", "expired"})
_HANDLE_KEEP = 512             # yume_forget candidate handles kept per provider instance
FORGET_NOTE = ("사용자가 잊으라고 한 기억의 id로 yume_forget을 다시 호출할 것. "
               "id는 내부 처리용이니 사용자에게 보여 주거나 읽어 주지 말 것.")

# §6.4 parameters; descriptions per U1 (DEVIATIONS E2E-8 supersedes the CONTRACTS §8.1 wording).
# Static from __init__ on.
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {"name": "yume_search",
     "description": ("장기기억에서 자동 첨부에 없던 과거 사실을 더 찾는다. 결과는 사실 문장과 날짜뿐이다. "
                     "검색한 일이나 기억 시스템은 사용자에게 언급하지 말 것."),
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string"},
         "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
         "include_inactive": {"type": "boolean", "default": False,
                              "description": ("지금은 맞지 않는 지난 기록까지 포함(사용자가 예전 일을 물을 때). "
                                              "current=false는 지난 사실이라는 뜻일 뿐, 사용자에게 상태로 설명하지 말 것")}},
         "required": ["query"]}},
    {"name": "yume_remember",
     "description": ("사용자가 \"기억해\", \"잊지 마\", \"저장해 둬\", \"앞으로 항상\"처럼 기억·저장·고정을 "
                     "직접 요청한 내용만 저장. 그 밖에는 절대 먼저 저장하지 말고, 기억할지 묻지도 말 것. "
                     "저장 결과나 id는 사용자에게 말하지 말 것. 매번 필요한 짧은 핵심은 memory 도구(USER.md)."),
     "parameters": {"type": "object", "properties": {
         "text": {"type": "string", "maxLength": 400},
         "kind": {"type": "string", "enum": list(KINDS)},
         "valid_until": {"type": "string", "description": "YYYY-MM-DD, 시한 있는 사실만"},
         "pin": {"type": "boolean", "default": False,
                 "description": "사용자가 '앞으로 항상', '꼭'처럼 고정을 직접 요청한 경우만"}},
         "required": ["text"]}},
    {"name": "yume_forget",
     "description": ("사용자가 잊으라고 직접 요청한 기억만 숨김. 먼저 제안하지 말 것. id를 모르면 query로 후보를 "
                     "받아 그 id로 다시 호출. id는 내부용이니 사용자에게 보여 주지 말 것."),
     "parameters": {"type": "object", "properties": {
         "memory_id": {"type": "string"}, "query": {"type": "string"},
         "reason": {"type": "string"},
         "confirm": {"type": "boolean", "default": False,
                     "description": "고정 기억을 잊을 때 true 필요(사용자가 그 기억을 잊으라고 한 경우만)"}}}},
]
TOOL_NAMES = tuple(s["name"] for s in TOOL_SCHEMAS)

_WARN_LOCK = threading.Lock()
_WARNED: Dict[str, float] = {}


def _warn(key: str, msg: str, *args: Any) -> None:
    """Rate-limited warning (once a minute per key). Never includes user text or secrets."""
    now = time.monotonic()
    with _WARN_LOCK:
        if now - _WARNED.get(key, -1e9) < 60.0:
            return
        _WARNED[key] = now
    try:
        log.warning(msg, *args)
    except Exception:
        pass


def _resolve_home() -> str:
    try:
        from hermes_constants import get_hermes_home
        return str(get_hermes_home())
    except Exception:
        return os.path.expanduser(os.environ.get("HERMES_HOME", "").strip() or "~/.hermes")


def _env_file_value(home: str, name: str) -> Optional[str]:
    """Name-only parse of $HERMES_HOME/.env (never sourced, never logged)."""
    try:
        with open(os.path.join(home, ".env"), "r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
                if not m or m.group(1) != name:
                    continue
                v = m.group(2).strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                    v = v[1:-1]
                else:
                    v = re.split(r"\s+#", v, maxsplit=1)[0].strip()
                return v or None
    except Exception:
        return None
    return None


def _secret(home: str, name: str) -> Optional[str]:
    """agent.secret_scope.get_secret (profile-scoped), then $HERMES_HOME/.env by name."""
    val = None
    try:
        from agent.secret_scope import get_secret
        val = get_secret(name)
    except Exception:
        val = None
    return val or _env_file_value(home, name)


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def _unit(v):
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n > 0 else None


@dataclass
class PendingTurn:
    query: str
    items: list = field(default_factory=list)      # [(memory_id, text, mode)]


@dataclass
class _Cand:
    id: str
    item: Any
    cos: Optional[float]
    score: float
    mode: str
    vec: Optional[list] = None
    keyword: bool = False


class YumeProvider(MemoryProvider):
    """Yume recall: prefetch from the nightly serving copy, tools yume_search/remember/forget."""

    def __init__(self) -> None:
        # No I/O here (discovery instantiates through register()).
        self._hermes_home = ""
        self._platform = "cli"
        self._chat_type: Optional[str] = None
        self._user_id: Optional[str] = None
        self._session_id = ""
        self._cwd = ""
        self._disabled_reason = ""
        self._prompt_core_shas: set = set()
        self._core_entries: Dict[str, List[str]] = {}
        self._core_live: Optional[core_snapshot.CoreSnapshot] = None
        self._prompt_pin_ids: set = set()
        self._epoch_injected: set = set()
        self._pending: Dict[int, PendingTurn] = {}
        self._turn_no = 0
        self._prev_user = ""
        self._cur_user = ""
        self._first_query_checked = False
        self._synthetic = False
        self._last_count = 0
        self._compress_at = 0.0          # monotonic time of the last on_pre_compress (0 = none)
        self._forget_handles: Dict[str, str] = {}   # yume_forget short handle → memory id
        self._lock = threading.RLock()

    # ── identity / availability ──────────────────────────────────────────
    @property
    def name(self) -> str:
        return "hermesyume"

    def _home(self) -> str:
        if not self._hermes_home:
            self._hermes_home = _resolve_home()
        return self._hermes_home

    def is_available(self) -> bool:
        try:
            cfg = _cfg.load(self._home())
            return bool(cfg and cfg.get("enabled"))
        except Exception:
            return False

    def unavailable_reason(self) -> str:
        try:
            cfg = _cfg.load(self._home())
            if cfg is None:
                return "hermesyume/config.json 없음 또는 읽기 실패 — `yume init` 필요"
            if not cfg.get("enabled"):
                return "hermesyume config.json의 enabled=false"
        except Exception:
            pass
        return ""

    def backup_paths(self) -> List[str]:
        return []                      # all data lives inside HERMES_HOME

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return json.loads(json.dumps(TOOL_SCHEMAS))

    # ── session lifecycle ────────────────────────────────────────────────
    def initialize(self, session_id: str, **kwargs) -> None:
        try:
            self._hermes_home = str(kwargs.get("hermes_home") or "") or _resolve_home()
            self._platform = str(kwargs.get("platform") or "cli")
            self._chat_type = kwargs.get("chat_type")
            self._user_id = kwargs.get("user_id")
            self._session_id = session_id or ""
            try:
                self._cwd = os.getcwd()
            except Exception:
                self._cwd = ""
            cfg = _cfg.load(self._hermes_home)
            if cfg:
                embed_http.configure(cfg)
                live.get_writer(self._hermes_home).configure(cfg)
            self._take_core_snapshot(cfg)
            self._disabled_reason = self._static_block_reason(cfg) or ""
            if cfg and self._session_id:
                # A resumed conversation (gateway restart, cache eviction, --resume) still carries
                # the <memory-context> blocks injected earlier in this session (api_content
                # sidecars): do not inject the same memories again.
                seen = live.get_writer(self._hermes_home).injected_ids(self._session_id)
                with self._lock:
                    self._epoch_injected = set(seen)
        except Exception as e:
            _warn("init", "hermesyume initialize 실패: %s", type(e).__name__)

    def _take_core_snapshot(self, cfg) -> None:
        snap = core_snapshot.snapshot(self._home())
        with self._lock:
            self._prompt_core_shas = set(snap.shas)
            self._core_entries = {k: list(v) for k, v in snap.entries.items()}
            self._core_live = snap.copy()
            self._prompt_pin_ids = self._pins_in_core(cfg, snap)

    def _pins_in_core(self, cfg, snap) -> set:
        idx = serving.get_index(self._home())
        if idx is None:
            return set()
        cont = float((cfg or _cfg.DEFAULTS).get("pin_core_containment", 0.8))
        return {p.id for p in idx.pins
                if core_snapshot.pin_in_core(p.text, p.label, snap, containment=cont)}

    def _static_block_reason(self, cfg) -> Optional[str]:
        """Why recall is off for this session (None = on). Re-evaluated on every call."""
        if not cfg:
            return "no_config"
        if not cfg.get("enabled"):
            return "disabled"
        if self._cwd and any(fnmatch.fnmatch(self._cwd, g) for g in (cfg.get("deny_cwd_globs") or [])):
            return "cwd"
        if self._chat_type not in (cfg.get("allowed_chat_types") or []):
            return "chat_type"
        if self._platform not in (cfg.get("recall_platforms") or []):
            return "platform"
        if self._synthetic:
            return "synthetic"
        return None

    def _check_first_query(self, cfg, text: str) -> None:
        if self._first_query_checked or not text or not text.strip():
            return
        self._first_query_checked = True
        pat = (cfg or {}).get("exclude_first_message_regex") or ""
        if pat:
            try:
                if re.search(pat, text.lstrip()):
                    self._synthetic = True
            except re.error:
                pass

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "",
                          reset: bool = False, rewound: bool = False, **kwargs) -> None:
        try:
            self._flush_events()
            with self._lock:
                if new_session_id:
                    self._session_id = new_session_id
                # The new context no longer carries earlier <memory-context> blocks.
                self._epoch_injected.clear()
                self._compress_at = 0.0
                if reset:
                    self._pending.clear()
                    self._prev_user = ""
                    self._cur_user = ""
                    self._first_query_checked = False
                    self._synthetic = False
                    self._turn_no = 0
        except Exception as e:
            _warn("switch", "hermesyume on_session_switch 실패: %s", type(e).__name__)

    @staticmethod
    def _user_instruction(message: str) -> Optional[str]:
        """The user's own words of a /skill turn (Hermes strips the skill body only for prefetch
        and sync). None = bare skill invocation (no user text)."""
        try:
            from agent.skill_commands import extract_user_instruction_from_skill_message
        except Exception:
            return message
        try:
            return extract_user_instruction_from_skill_message(message)
        except Exception:
            return message

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        try:
            clean = self._user_instruction(message or "") if message else ""
            with self._lock:
                self._turn_no = int(turn_number or 0)
                if clean is not None:
                    if self._cur_user:
                        self._prev_user = self._cur_user
                    self._cur_user = clean
            if not self._first_query_checked:
                self._check_first_query(_cfg.load(self._home()), message or "")   # raw text
        except Exception as e:
            _warn("turn", "hermesyume on_turn_start 실패: %s", type(e).__name__)

    # Hermes calls on_session_end also at context compaction (commit_memory_session "in BOTH
    # modes"), right after on_pre_compress, while the session goes on with the same id. That call
    # is a flush point, not the end of the session: no session_end marker then (DEVIATIONS F-8).
    _COMPRESS_FLUSH_WINDOW_S = 900.0

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        try:
            cfg = _cfg.load(self._home())
            if not cfg:
                return
            w = live.get_writer(self._home())
            w.flush()
            with self._lock:
                compacting = bool(self._compress_at) and \
                    time.monotonic() - self._compress_at < self._COMPRESS_FLUSH_WINDOW_S
                self._compress_at = 0.0
            if self._session_id and not compacting:
                w.inbox("session_end", session_id=self._session_id, platform=self._platform)
        except Exception as e:
            _warn("end", "hermesyume on_session_end 실패: %s", type(e).__name__)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """No contribution (ABC default ""): Hermes hands provider text to the summarizer as
        material to preserve, and the summarizer never sees <memory-context> (api_content only),
        so a note here would only put memory-system text into the summary (DEVIATIONS F-10).
        Marks the following on_session_end as a compaction flush (F-8)."""
        try:
            with self._lock:
                self._compress_at = time.monotonic()
        except Exception:
            pass
        return ""

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        return None

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        return None

    def shutdown(self) -> None:
        try:
            if not self._hermes_home:
                return
            w = live.get_writer(self._hermes_home)
            w.flush()
            w.health_flush()
        except Exception as e:
            _warn("shutdown", "hermesyume shutdown 실패: %s", type(e).__name__)

    # ── system prompt ────────────────────────────────────────────────────
    def system_prompt_block(self) -> str:
        try:
            cfg = _cfg.load(self._home())
            if self._static_block_reason(cfg):
                return ""
            # The prompt is (re)built now: refresh the core snapshot it will contain.
            self._take_core_snapshot(cfg)
            snap = self._core_live
            idx = serving.get_index(self._home())
            pins = idx.pins if idx is not None else []
            forgotten = live.get_writer(self._home()).pending()["forget_ids"]
            pins = [p for p in pins if p.id not in forgotten]     # same-day yume_forget (§1.2, P6)
            cont = float(cfg.get("pin_core_containment", 0.8))
            outside = [p for p in pins if not core_snapshot.pin_in_core(p.text, p.label, snap,
                                                                          containment=cont)]
            kept, _dropped = textutil.fit_pins([p.text for p in outside],
                                               int(cfg.get("pins_budget_chars", 800)))
            kept_ids = set()
            remaining = list(kept)
            for p in outside:
                if p.text in remaining:
                    remaining.remove(p.text)
                    kept_ids.add(p.id)
            with self._lock:
                self._prompt_pin_ids |= kept_ids
            return textutil.static_block(kept)
        except Exception as e:
            _warn("spb", "hermesyume system_prompt_block 실패: %s", type(e).__name__)
            return ""

    def recall_status(self) -> Optional[RecallStatus]:
        try:
            n = self._last_count
            if n <= 0:
                return None
            cfg = _cfg.load(self._home()) or {}
            if not cfg.get("show_status_%s" % (self._platform or ""), False):
                return None
            return RecallStatus("Yume", n)
        except Exception:
            return None

    # ── prefetch (§6.2) ──────────────────────────────────────────────────
    def prefetch(self, query: str, *, session_id: str = "") -> str:
        t0 = time.monotonic()
        self._last_count = 0
        try:
            return self._prefetch(query or "", t0)
        except Exception as e:
            _warn("prefetch", "hermesyume prefetch 실패: %s", type(e).__name__)
            return ""

    def _prefetch(self, query: str, t0: float) -> str:
        home = self._home()
        cfg = _cfg.load(home)
        if not cfg or not cfg.get("enabled"):
            return ""
        self._check_first_query(cfg, query)
        if self._static_block_reason(cfg):
            return ""
        if not query.strip():
            return ""
        deadline = t0 + float(cfg.get("prefetch_deadline_s", 4.0))
        w = live.get_writer(home)
        w.configure(cfg)
        embed_http.configure(cfg)
        counters: Dict[str, Any] = {"prefetch_n": 1}
        idx = None
        try:
            idx, chosen = self._select(cfg, query, deadline, counters)
            if time.monotonic() > deadline:
                counters["timeout_n"] = 1
                return ""
            if not chosen:
                counters["empty_n"] = 1
                return ""
            now = _cfg.now()
            lines, picked = [], []
            budget = int(cfg.get("recall_budget_chars", 1000))
            for c in chosen:
                if len(lines) >= int(cfg.get("recall_k", 5)):
                    break
                line = textutil.format_item(c.item, now=now, item_chars=int(cfg.get("recall_item_chars", 300)),
                                            keyword=c.keyword)
                if len(textutil.format_block(lines + [line])) > budget:
                    continue
                lines.append(line)
                picked.append(c)
            if not lines:
                counters["empty_n"] = 1
                return ""
            inject = bool(cfg.get("inject", True))
            kind = "injected" if inject else "shadow"
            run = idx.run_id if idx is not None else None
            for c in picked:
                w.record(session_id=self._session_id, platform=self._platform, turn_no=self._turn_no,
                         memory_id=c.id, kind=kind, cos=c.cos, mode=c.mode, snapshot_run=run)
            with self._lock:
                self._epoch_injected.update(c.id for c in picked)
                if inject:
                    self._pending[self._turn_no] = PendingTurn(
                        query=query, items=[(c.id, c.item.text, c.mode) for c in picked])
                    for t in sorted(self._pending)[:-_PENDING_KEEP]:
                        self._pending.pop(t, None)
            if not inject:
                return ""
            counters["injected_n"] = len(picked)
            self._last_count = len(picked)
            return textutil.format_block(lines)
        finally:
            counters["latency_ms"] = (time.monotonic() - t0) * 1000.0
            if idx is not None:
                counters["snapshot_run"] = idx.run_id
            w.health_tick(self._platform, **counters)

    def _select(self, cfg, query: str, deadline: float, counters: Dict[str, Any]):
        home = self._home()
        q = textutil.expand_query(query, self._prev_user, short=cfg.get("short_query_chars", 20),
                                  tail=cfg.get("prev_user_tail_chars", 200),
                                  max_chars=cfg.get("query_max_chars", 1000))
        idx = serving.get_index(home)
        pend = live.get_writer(home).pending()
        now = _cfg.now()
        with self._lock:
            excl = set(pend["forget_ids"]) | set(self._epoch_injected) | set(self._prompt_pin_ids)
            core_shas = set(self._prompt_core_shas)
        if idx is not None:
            excl |= idx.ids_with_core_sha(core_shas)
        remembers = [r for r in pend["remember"] if r["memory_id"] not in excl
                     and not (r.get("valid_until") and r["valid_until"] < now)]
        if idx is None and not remembers:
            return idx, []
        model_id = idx.embed_model if idx is not None else _cfg.embed_model_id(cfg)
        qvec = None
        if embed_http.breaker_open():
            counters["error_class"] = "breaker_open"
        else:
            remaining = deadline - time.monotonic() - 0.3
            total = min(float(cfg.get("embed_total_timeout_s", 3.0)), max(0.05, remaining))
            qvec, err = embed_http.embed_ex(
                q, model_id=model_id, base_url=self._base_url(cfg),
                api_key=_secret(home, "OPENAI_API_KEY"),
                connect_timeout=min(float(cfg.get("embed_connect_timeout_s", 1.0)), total),
                total_timeout=total, cfg=cfg)
            if qvec is None:
                counters["embed_fail_n"] = 1
                counters["error_class"] = err
        if time.monotonic() > deadline:
            return idx, []
        if qvec is not None:
            return idx, self._rank_vector(cfg, idx, qvec, q, remembers, excl, now, model_id)
        counters["fts_fallback_n"] = 1
        return idx, self._rank_keyword(cfg, idx, q, remembers, excl, now)

    def _base_url(self, cfg) -> str:
        return (cfg.get("embed_base_url") or _secret(self._home(), "OPENAI_BASE_URL")
                or embed_http.DEFAULT_BASE)

    @staticmethod
    def _inbox_item(r) -> serving.Item:
        kind = r.get("kind") or "fact"
        return serving.Item(id=r["memory_id"], text=r["text"], subject="", kind=kind,
                            tier="durable" if kind in _PROTECTED else "decaying", status="active",
                            pinned=bool(r.get("pin")), event_time=r.get("ts"),
                            valid_until=r.get("valid_until"), strength=_INBOX_STRENGTH)

    @staticmethod
    def _like_score(cfg, toks: set, text: str) -> Optional[float]:
        if not toks:
            return None
        hay = textutil.nfkc(text).casefold()
        hits = [t for t in toks if t in hay]
        if not hits:
            return None
        base = float(cfg.get("recall_min_cos", 0.40))
        return min(0.75, base + 0.30 * len(hits) / float(len(toks)))

    def _rank_vector(self, cfg, idx, qvec, q, remembers, excl, now, model_id) -> List[_Cand]:
        w_s = float(cfg.get("score_w_strength", 0.08))
        w_p = float(cfg.get("score_w_pinned", 0.04))
        w_k = float(cfg.get("score_w_keyword", 0.03))
        min_cos = float(cfg.get("recall_min_cos", 0.40))
        pin_cos = float(cfg.get("pinned_min_cos", 0.33))
        toks = textutil.query_tokens(q)
        cands: List[_Cand] = []
        if idx is not None:
            for mid, cos, vec in idx.vector_search_vecs(qvec, exclude=excl, now=now,
                                                         k1=int(cfg.get("stage1_k", 64))):
                it = idx.items[mid]
                if cos < (pin_cos if it.pinned else min_cos):
                    continue
                score = (cos + w_s * float(it.strength or 0.0) + w_p * (1 if it.pinned else 0)
                         + w_k * (1 if textutil.keyword_hit(toks, it.text) else 0))
                cands.append(_Cand(mid, it, cos, score, "vector", vec))
        for r in remembers:
            it = self._inbox_item(r)
            vec = r.get("vec")
            if vec and r.get("embed_model") == model_id and len(vec) == len(qvec):
                v = _unit(vec)
                if v is None:
                    continue
                cos = _dot(qvec, v)
                if cos < (pin_cos if it.pinned else min_cos):
                    continue
                score = (cos + w_s * _INBOX_STRENGTH + w_p * (1 if it.pinned else 0)
                         + w_k * (1 if textutil.keyword_hit(toks, it.text) else 0))
                cands.append(_Cand(it.id, it, cos, score, "inbox", v))
            else:
                pseudo = self._like_score(cfg, toks, it.text)
                if pseudo is None:
                    continue
                cands.append(_Cand(it.id, it, None, pseudo + w_s * _INBOX_STRENGTH + w_k, "inbox",
                                   None, keyword=True))
        if not cands:
            return []
        cands.sort(key=lambda c: (-c.score, c.id))
        cut = cands[0].score - float(cfg.get("recall_rel_cut", 0.10))
        mmr = float(cfg.get("mmr_cos", 0.92))
        out: List[_Cand] = []
        for c in cands:
            if c.score < cut:
                break
            if c.vec is not None and any(o.vec is not None and _dot(c.vec, o.vec) >= mmr for o in out):
                continue
            out.append(c)
        return out

    def _rank_keyword(self, cfg, idx, q, remembers, excl, now) -> List[_Cand]:
        k = int(cfg.get("fts_top_k", 3))
        cands: List[_Cand] = []
        if idx is not None:
            for mid, rank in idx.fts_search(q, exclude=excl, limit=k, statuses=("active",), now=now):
                it = idx.items.get(mid)
                if it is not None:
                    cands.append(_Cand(mid, it, None, float(rank), "keyword", None, keyword=True))
        toks = textutil.query_tokens(q)
        for r in remembers:
            pseudo = self._like_score(cfg, toks, r["text"])
            if pseudo is not None:
                cands.append(_Cand(r["memory_id"], self._inbox_item(r), None,
                                   pseudo * (0.5 + _INBOX_STRENGTH), "inbox", None, keyword=True))
        cands.sort(key=lambda c: (-c.score, c.id))
        return cands[:k]

    # ── turn sync: `used` judgment (§6.3) ────────────────────────────────
    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "",
                  messages: Optional[List[Dict[str, Any]]] = None) -> None:
        try:
            cfg = _cfg.load(self._home())
            if not cfg:
                return
            w = live.get_writer(self._home())
            with self._lock:
                pending = dict(self._pending)
            if pending and user_content:
                matched, dropped = _used.match_pending(pending, user_content)
                if matched is not None:
                    entry = pending[matched]
                    for mid, text, mode in entry.items:
                        if _used.is_used(text, assistant_content or "",
                                         ratio=float(cfg.get("used_ratio", 0.35)),
                                         min_tokens=int(cfg.get("used_min_tokens", 2))):
                            w.record(session_id=self._session_id, platform=self._platform,
                                     turn_no=matched, memory_id=mid, kind="used", cos=None,
                                     mode=mode, snapshot_run=None)
                    with self._lock:
                        for t in [matched] + list(dropped):
                            self._pending.pop(t, None)
            w.flush()
        except Exception as e:
            _warn("sync", "hermesyume sync_turn 실패: %s", type(e).__name__)

    def _flush_events(self) -> None:
        if self._hermes_home:
            live.get_writer(self._hermes_home).flush()

    # ── built-in memory tool mirror (§3.5, §7.3) ─────────────────────────
    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        try:
            if target not in core_snapshot.TARGET_FILES or action not in ("add", "replace", "remove"):
                return
            cfg = _cfg.load(self._home())
            if not cfg or not cfg.get("enabled"):
                return
            meta = dict(metadata or {})
            partial = meta.pop("old_text", None)
            with self._lock:
                if self._core_live is None:
                    self._core_live = core_snapshot.snapshot(self._home())
                snap = self._core_live
                old_full = None
                if action in ("replace", "remove") and partial:
                    old_full = core_snapshot.find_entry(snap, target, str(partial))
                core_snapshot.apply_write(snap, action, target, content or "", old_full)
            if action == "add":
                op, text, old_text = "core_add", content or "", None
            elif action == "replace":
                op, text, old_text = "core_replace", content or "", old_full or partial
                meta["restored"] = old_full is not None
            else:
                op, text, old_text = "core_remove", old_full or partial or content or "", partial
                meta["restored"] = old_full is not None
            live.get_writer(self._home()).inbox(
                op, text=text, old_text=old_text, target=target, session_id=self._session_id or None,
                platform=self._platform, meta_json=json.dumps(meta, ensure_ascii=False, default=str))
        except Exception as e:
            _warn("mw", "hermesyume on_memory_write 실패: %s", type(e).__name__)

    # ── tools (§6.4) ─────────────────────────────────────────────────────
    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        t0 = time.monotonic()
        try:
            if tool_name not in TOOL_NAMES:
                return json.dumps({"ok": False, "error": "unknown_tool"})
            if not isinstance(args, dict):
                args = {}
            cfg = _cfg.load(self._home())
            if not cfg or not cfg.get("enabled"):
                return json.dumps({"ok": False, "error": "disabled"})
            if self._static_block_reason(cfg) in ("cwd", "chat_type", "synthetic"):
                return json.dumps({"ok": False, "error": "unavailable"})
            deadline = t0 + float(cfg.get("tool_total_timeout_s", 5.0))
            if tool_name == "yume_search":
                out = self._tool_search(cfg, args, deadline)
            elif tool_name == "yume_remember":
                out = self._tool_remember(cfg, args, deadline)
            else:
                out = self._tool_forget(cfg, args, deadline)
            return json.dumps(out, ensure_ascii=False)
        except Exception as e:
            _warn("tool", "hermesyume %s 실패: %s", tool_name, type(e).__name__)
            return json.dumps({"ok": False, "error": "internal"})

    def _tool_embed(self, cfg, text: str, model_id: str, deadline: float):
        remaining = deadline - time.monotonic() - 0.2
        total = min(float(cfg.get("tool_embed_timeout_s", 3.0)), max(0.05, remaining))
        if embed_http.breaker_open() or remaining <= 0.05:
            return None
        return embed_http.embed(text, model_id=model_id, base_url=self._base_url(cfg),
                                api_key=_secret(self._home(), "OPENAI_API_KEY"),
                                connect_timeout=min(float(cfg.get("embed_connect_timeout_s", 1.0)), total),
                                total_timeout=total, cfg=cfg)

    def _search(self, cfg, query: str, limit: int, include_inactive: bool, deadline: float):
        """[(Item, score, mode)] from serving + same-day remembers, minus pending forgets."""
        home = self._home()
        idx = serving.get_index(home)
        pend = live.get_writer(home).pending()
        excl = set(pend["forget_ids"])
        model_id = idx.embed_model if idx is not None else _cfg.embed_model_id(cfg)
        qvec = self._tool_embed(cfg, query, model_id, deadline)
        min_cos = float(cfg.get("search_min_cos", 0.30))
        now = _cfg.now()
        res = []
        if idx is not None:
            res = idx.search_all(qvec, query, include_inactive=include_inactive, limit=limit,
                                 min_cos=min_cos, now=now, exclude=excl)
        toks = textutil.query_tokens(query)
        for r in pend["remember"]:
            if r["memory_id"] in excl:
                continue
            it = self._inbox_item(r)
            vec = r.get("vec")
            if qvec is not None and vec and r.get("embed_model") == model_id and len(vec) == len(qvec):
                v = _unit(vec)
                cos = _dot(qvec, v) if v else 0.0
                if cos >= min_cos:
                    res.append((it, cos, "inbox"))
            else:
                s = self._like_score(cfg, toks, it.text)
                if s is not None:
                    res.append((it, s, "inbox"))
        res.sort(key=lambda t: (-t[1], t[0].id))
        return res[:limit], ("vector" if qvec is not None else "keyword")

    @staticmethod
    def _fact_row(it, *, inactive: bool = False) -> Dict[str, Any]:
        """U1 (DEVIATIONS E2E-8): the fact only — text + date. No id/status/tier/strength/score;
        with include_inactive a superseded/expired fact gets the neutral `current: false`."""
        row: Dict[str, Any] = {"text": it.text}
        if it.event_time:
            row["date"] = _cfg.kst_date(it.event_time)
        if inactive and it.status in _NOT_CURRENT:
            row["current"] = False
        return row

    def _handle(self, memory_id: str) -> str:
        """Opaque short id for a yume_forget candidate: a stable hash prefix of the memory id, made
        longer on a clash so one handle never names two memories in this provider's lifetime."""
        digest = hashlib.sha1(memory_id.encode("utf-8")).hexdigest()
        with self._lock:
            for n in (5, 8, 12, 40):
                h = "m" + digest[:n]
                cur = self._forget_handles.get(h)
                if cur is None or cur == memory_id:
                    self._forget_handles.pop(h, None)
                    self._forget_handles[h] = memory_id
                    while len(self._forget_handles) > _HANDLE_KEEP:
                        self._forget_handles.pop(next(iter(self._forget_handles)))
                    return h
        return memory_id

    def _tool_search(self, cfg, args, deadline) -> Dict[str, Any]:
        query = str(args.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "query_required"}
        try:
            limit = max(1, min(10, int(args.get("limit") or 5)))
        except (TypeError, ValueError):
            limit = 5
        res, _mode = self._search(cfg, query, limit, bool(args.get("include_inactive")), deadline)
        w = live.get_writer(self._home())
        idx = serving.get_index(self._home())
        for it, score, m in self._tool_hits(cfg, res):
            w.record(session_id=self._session_id, platform=self._platform, turn_no=self._turn_no,
                     memory_id=it.id, kind="tool_hit", cos=score if m != "keyword" and m != "inbox" else None,
                     mode=m, snapshot_run=idx.run_id if idx is not None else None)
        w.flush()
        inactive = bool(args.get("include_inactive"))
        return {"ok": True, "results": [self._fact_row(it, inactive=inactive) for it, _s, _m in res]}

    _TOOL_HIT_TOP = 2

    @classmethod
    def _tool_hits(cls, cfg, res):
        """Results that count as a tool_hit (reinforcement + dormant revival in R4): the top two
        vector results at the recall threshold (cos ≥ recall_min_cos), or the single top result of
        a keyword search. search_min_cos (0.30) lets unrelated pairs into the list; they are shown
        but must not wake up dormant memories (DEVIATIONS F-18)."""
        th = float(cfg.get("recall_min_cos", 0.40))
        out = []
        for rank, (it, score, m) in enumerate(res[: cls._TOOL_HIT_TOP]):
            if m == "vector":
                if float(score) >= th:
                    out.append((it, score, m))
            elif rank == 0:
                out.append((it, score, m))
        return out

    def _tool_remember(self, cfg, args, deadline) -> Dict[str, Any]:
        text = " ".join(str(args.get("text") or "").split())
        if not text:
            return {"ok": False, "error": "text_required"}
        if len(text) > 400:
            return {"ok": False, "error": "too_long"}
        kind = args.get("kind") or None
        if kind is not None and kind not in KINDS:
            return {"ok": False, "error": "bad_kind"}
        vu_raw = args.get("valid_until") or None
        vu = None
        if vu_raw:
            vu = _cfg.parse_iso(str(vu_raw), end_of_day=True)
            if vu is None or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(vu_raw).strip()):
                return {"ok": False, "error": "bad_valid_until"}
        if self._threats(text):
            return {"ok": False, "error": "secret_or_threat"}
        idx = serving.get_index(self._home())
        model_id = idx.embed_model if idx is not None else _cfg.embed_model_id(cfg)
        vec = self._tool_embed(cfg, text, model_id, deadline)
        new_id = live.get_writer(self._home()).inbox(
            "remember", text=text, kind=kind, pin=bool(args.get("pin")),
            vec=live_schema.vec_to_blob(vec) if vec else None,
            embed_model=model_id if vec else None,
            meta_json=json.dumps({"valid_until": str(vu_raw).strip() if vu_raw else None}),
            session_id=self._session_id or None, platform=self._platform)
        if not new_id:
            return {"ok": False, "error": "busy"}
        return {"ok": True}          # U1: no id / search mode for the agent to relay

    @staticmethod
    def _threats(text: str) -> List[str]:
        found = ["secret:" + t for t in textutil.secret_types(text)]
        try:
            from tools.threat_patterns import scan_for_threats
            found += list(scan_for_threats(text, "strict"))
        except Exception:
            pass
        return found

    def _tool_forget(self, cfg, args, deadline) -> Dict[str, Any]:
        mid = str(args.get("memory_id") or "").strip()
        confirm = bool(args.get("confirm"))
        reason = str(args.get("reason") or "")[:200]
        if not mid:
            query = str(args.get("query") or "").strip()
            if not query:
                return {"ok": False, "error": "memory_id_or_query_required"}
            res, _mode = self._search(cfg, query, 5, False, deadline)
            return {"ok": True, "forgotten": False,
                    "candidates": [dict(id=self._handle(it.id), **self._fact_row(it)) for it, _s, _m in res],
                    "note": FORGET_NOTE}
        with self._lock:
            mid = self._forget_handles.get(mid, mid)     # handle → memory id (full ids still accepted)
        home = self._home()
        pend = live.get_writer(home).pending()
        if mid in pend["forget_ids"]:
            return {"ok": True, "forgotten": True}
        pinned = False
        if mid.startswith("inbox:"):
            rem = {r["memory_id"]: r for r in pend["remember"]}
            if mid not in rem:
                return {"ok": False, "error": "not_found"}
            pinned = bool(rem[mid].get("pin"))
        else:
            idx = serving.get_index(home)
            if idx is not None:
                got = idx.get_items([mid])
                if mid not in got:
                    return {"ok": False, "error": "not_found"}
                pinned = bool(got[mid].pinned)
        if pinned and not confirm:
            return {"ok": False, "error": "confirm_required"}
        new_id = live.get_writer(home).inbox(
            "forget", memory_id=mid, meta_json=json.dumps({"reason": reason, "confirm": confirm},
                                                          ensure_ascii=False),
            session_id=self._session_id or None, platform=self._platform)
        if not new_id:
            return {"ok": False, "error": "busy"}
        with self._lock:
            self._epoch_injected.add(mid)
        return {"ok": True, "forgotten": True}


def register(ctx) -> None:
    ctx.register_memory_provider(YumeProvider())
