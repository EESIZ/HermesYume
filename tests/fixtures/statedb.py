"""Fake Hermes state.db with the REAL live column set (stdlib only; usable from the Hermes venv).

DDL below mirrors ``sessions``/``messages`` of a Hermes 0.21.0 state.db (schema_version 30). FTS
tables/triggers are omitted (not read by us). ``live_columns()`` re-reads a real schema (the home in
$HERMESYUME_TEST_LIVE_HOME) without creating any file next to it (immutable=1), for the parity test.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

# A real HERMES_HOME to compare the column set with (opt-in, read-only; the parity test skips when
# $HERMESYUME_TEST_LIVE_HOME is unset).
LIVE_STATE_DB = (os.path.join(os.environ["HERMESYUME_TEST_LIVE_HOME"], "state.db")
                 if os.environ.get("HERMESYUME_TEST_LIVE_HOME", "").strip() else "")
CONTENT_JSON_PREFIX = "\x00json:"

SESSIONS_DDL = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    session_key TEXT,
    chat_id TEXT,
    chat_type TEXT,
    thread_id TEXT,
    display_name TEXT,
    origin_json TEXT,
    expiry_finalized INTEGER DEFAULT 0,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    system_prompt_hash TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    cwd TEXT,
    git_branch TEXT,
    git_repo_root TEXT,
    git_metadata_generation INTEGER NOT NULL DEFAULT 0,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    title_source TEXT,
    last_activity_at REAL,
    last_activity_description TEXT,
    last_activity_provenance TEXT,
    api_call_count INTEGER DEFAULT 0,
    handoff_state TEXT,
    handoff_platform TEXT,
    handoff_error TEXT,
    compression_failure_cooldown_until REAL,
    compression_failure_error TEXT,
    compression_fallback_streak INTEGER NOT NULL DEFAULT 0,
    compression_ineffective_count INTEGER NOT NULL DEFAULT 0,
    compression_recovery_deadline REAL,
    profile_name TEXT,
    rewind_count INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    hidden INTEGER NOT NULL DEFAULT 0,
    last_read_at REAL,
    tool_names TEXT,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id),
    FOREIGN KEY (system_prompt_hash) REFERENCES system_prompts(hash)
);
"""

MESSAGES_DDL = """
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    effect_disposition TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT,
    platform_message_id TEXT,
    observed INTEGER DEFAULT 0,
    _compressed_summary INTEGER NOT NULL DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1,
    compacted INTEGER NOT NULL DEFAULT 0,
    api_content TEXT,
    display_kind TEXT,
    display_metadata TEXT
);
"""

INDEX_DDL = """
CREATE INDEX idx_messages_session ON messages(session_id, timestamp);
CREATE INDEX idx_messages_session_active ON messages(session_id, active, timestamp);
CREATE INDEX idx_messages_session_id ON messages(session_id, id);
CREATE INDEX idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX idx_sessions_source ON sessions(source);
CREATE INDEX idx_sessions_started ON sessions(started_at DESC);
CREATE UNIQUE INDEX idx_sessions_title_unique ON sessions(title) WHERE title IS NOT NULL;
CREATE TABLE schema_version (version INTEGER NOT NULL);
INSERT INTO schema_version(version) VALUES (30);
"""

SYNTHETIC_FIRST_MESSAGE = ("[synthetic-eval] You are a worker in a scripted test run. "
                           "Use only files inside the current workspace.")


def live_columns(path: str = LIVE_STATE_DB) -> dict[str, list[str]] | None:
    """Column names of live sessions/messages, read with immutable=1 (creates no -shm/-wal).
    None if no live db is configured or it is absent."""
    if not path or not os.path.exists(path):
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        return {t: [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
                for t in ("sessions", "messages")}
    finally:
        conn.close()


def encode_json_content(parts: list | dict) -> str:
    """Hermes multimodal content encoding (\\x00json: prefix)."""
    return CONTENT_JSON_PREFIX + json.dumps(parts, ensure_ascii=False)


@dataclass
class StateDB:
    """Builder over a fake state.db. All timestamps are epoch seconds."""
    path: Path
    conn: sqlite3.Connection = field(init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        self.conn = sqlite3.connect(str(self.path))
        if new:
            self.conn.executescript(SESSIONS_DDL + MESSAGES_DDL + INDEX_DDL)
            self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> "StateDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def session(self, sid: str, source: str = "telegram", *, started_at: float,
                title: str | None = None, user_id: str | None = None,
                chat_type: str | None = None, cwd: str | None = None,
                parent_session_id: str | None = None, end_reason: str | None = None,
                ended_at: float | None = None, hidden: int = 0,
                last_activity_at: float | None = None, system_prompt: str | None = None) -> str:
        self.conn.execute(
            "INSERT INTO sessions(id, source, user_id, chat_type, cwd, parent_session_id, started_at, "
            "ended_at, end_reason, title, hidden, last_activity_at, system_prompt) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, source, user_id, chat_type, cwd, parent_session_id, started_at, ended_at,
             end_reason, title, hidden, last_activity_at if last_activity_at is not None else started_at,
             system_prompt))
        self.conn.commit()
        return sid

    def end_session(self, sid: str, *, ended_at: float, end_reason: str = "agent_close") -> None:
        self.conn.execute("UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
                          (ended_at, end_reason, sid))
        self.conn.commit()

    def message(self, sid: str, role: str, content: str | None, ts: float, *, active: int = 1,
                compacted: int = 0, compressed_summary: int = 0, display_kind: str | None = None,
                api_content: str | None = None, tool_name: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO messages(session_id, role, content, timestamp, active, compacted, "
            "_compressed_summary, display_kind, api_content, tool_name) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (sid, role, content, ts, active, compacted, compressed_summary, display_kind,
             api_content, tool_name))
        self.conn.execute("UPDATE sessions SET message_count=message_count+1, last_activity_at=? "
                          "WHERE id=?", (ts, sid))
        self.conn.commit()
        return int(cur.lastrowid)

    def exchange(self, sid: str, user: str, assistant: str | None, ts: float, *,
                 reply_after: float = 30.0, user_api_content: str | None = None) -> tuple[int, int | None]:
        u = self.message(sid, "user", user, ts, api_content=user_api_content)
        a = self.message(sid, "assistant", assistant, ts + reply_after) if assistant is not None else None
        return u, a

    def compress_generation(self, sid: str, *, keep_tail: int = 2, summary_ts: float | None = None,
                            summary: str = "[요약] 이전 대화 요약") -> list[int]:
        """Same-session generation copy (hermes_state compression): old rows → active=0,
        compacted=1; a summary row (_compressed_summary=1); the last `keep_tail` rows copied as NEW
        ids with identical role/content/timestamp. Returns the new tail ids."""
        rows = self.conn.execute(
            "SELECT id, role, content, timestamp FROM messages WHERE session_id=? AND active=1 "
            "ORDER BY timestamp, id", (sid,)).fetchall()
        self.conn.execute("UPDATE messages SET active=0, compacted=1 WHERE session_id=? AND active=1", (sid,))
        ts0 = summary_ts if summary_ts is not None else (rows[-1][3] if rows else time.time())
        self.message(sid, "user", summary, ts0, compressed_summary=1)
        new_ids = [self.message(sid, r[1], r[2], r[3]) for r in rows[-keep_tail:]] if keep_tail else []
        return new_ids

    def compression_child(self, parent: str, child: str, *, started_at: float, keep_tail: int = 2,
                          source: str | None = None) -> list[int]:
        """Compression into a child session: parent end_reason='compression'; child has
        parent_session_id=parent; the parent's last `keep_tail` rows are copied with identical
        timestamps. Returns the child's copied ids."""
        src = source or self.conn.execute("SELECT source FROM sessions WHERE id=?", (parent,)).fetchone()[0]
        self.end_session(parent, ended_at=started_at, end_reason="compression")
        self.session(child, src, started_at=started_at, parent_session_id=parent)
        rows = self.conn.execute(
            "SELECT role, content, timestamp FROM messages WHERE session_id=? AND active=1 "
            "AND _compressed_summary=0 ORDER BY timestamp, id", (parent,)).fetchall()
        self.message(child, "user", "[요약] 압축 요약", started_at, compressed_summary=1)
        return [self.message(child, r[0], r[1], r[2]) for r in rows[-keep_tail:]] if keep_tail else []


def build_basic(path: str | os.PathLike, now: float) -> dict:
    """Canonical scenario covering §3.1 filters. Returns ids/handles for assertions.

    - tg1: telegram dm, settled (2h ago), 2 exchanges with durable facts (included)
    - tg_recent: telegram dm, 5 min ago (NOT settled unless session_end)
    - cli_synth: cli, first user message matches the synthetic-eval regex in TEST_FILTERS (excluded)
    - cli_probe: cli, cwd /tmp/agent-probe-7 (excluded by deny_cwd_globs in TEST_FILTERS)
    - cli_tmp: cli, cwd /tmp (included — the cwd deny list is narrow)
    - cron1: cron (excluded by include_sources)
    - hidden1: telegram, sessions.hidden=1 (excluded)
    - tg1 also has: a tool message, a display_kind='hidden' assistant message, a user message
      whose api_content carries <memory-context> (content clean), and one whose content
      itself contains a <memory-context> block (must be stripped by sanitize)
    """
    t = now - 2 * 3600
    db = StateDB(Path(path))
    ids: dict = {"now": now}
    db.session("tg1", "telegram", started_at=t, title="Orion 회의", user_id="u1", chat_type="dm")
    ids["tg1_u1"], ids["tg1_a1"] = db.exchange(
        "tg1", "Orion 결제 스테이징 서버 포트는 8081이야. 앞으로 이걸로 접속해.", "알겠습니다. 8081로 기억할게요.",
        t + 60, user_api_content="Orion … <memory-context>[System note: …]\n- old\n</memory-context>")
    ids["tg1_tool"] = db.message("tg1", "tool", '{"ok": true}', t + 100, tool_name="terminal")
    ids["tg1_hidden"] = db.message("tg1", "assistant", "(hidden status line)", t + 110, display_kind="hidden")
    ids["tg1_u2"], ids["tg1_a2"] = db.exchange(
        "tg1", "Orion 데모 마감은 2026-10-10이야.\n<memory-context>\n[Yume 장기기억] 주입본\n</memory-context>",
        "마감 2026-10-10 확인했습니다.", t + 300)
    db.end_session("tg1", ended_at=t + 400, end_reason="session_reset")

    db.session("tg_recent", "telegram", started_at=now - 300, user_id="u1", chat_type="dm")
    ids["tg_recent_u"], ids["tg_recent_a"] = db.exchange("tg_recent", "방금 한 말: 보라색 고래 7341", "네.", now - 290)

    db.session("cli_synth", "cli", started_at=t, cwd="/srv/agent")
    db.exchange("cli_synth", SYNTHETIC_FIRST_MESSAGE + " Orion 포트는 9999다.", "ok", t + 10)

    db.session("cli_probe", "cli", started_at=t, cwd="/tmp/agent-probe-7")
    db.exchange("cli_probe", "probe 세션 내용입니다", "ok", t + 10)

    db.session("cli_tmp", "cli", started_at=t, cwd="/tmp")
    ids["cli_tmp_u"], ids["cli_tmp_a"] = db.exchange("cli_tmp", "Orion 요금 확인은 매뉴얼을 먼저 본다. 항상 요금표부터.", "네, 요금표부터 확인하겠습니다.", t + 20)

    db.session("cron1", "cron", started_at=t)
    db.exchange("cron1", "nightly report Orion", "done", t + 5)

    db.session("hidden1", "telegram", started_at=t, hidden=1, user_id="u1", chat_type="dm")
    db.exchange("hidden1", "숨김 세션 내용", "ok", t + 5)
    db.close()
    ids["path"] = str(path)
    return ids
