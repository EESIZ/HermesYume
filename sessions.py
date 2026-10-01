"""Episodic intake: read finished conversations from Hermes' state.db.

Hermes persists every CLI / messaging session in SQLite ($HERMES_HOME/state.db):
  sessions(id, source, title, started_at, ended_at, last_activity_at, hidden, ...)
  messages(id, session_id, role, content, timestamp, active, compacted,
           _compressed_summary, ...)

This is the "hippocampus": raw episodes. The database is opened READ-ONLY;
Hermesume never writes to it. A cursor (last processed activity timestamp)
lives in $HERMESUME_HOME/state.json.

Optional: markdown episode files (YYYY-MM-DD*.md) in $HERMESUME_HOME/episodes/
are read as well, for notes produced outside Hermes.
"""

import glob
import json
import logging
import os
import re
import shutil
import sqlite3
import time

from config import (
    EPISODE_ARCHIVE_DIR,
    EPISODE_DIR,
    EXCLUDE_SOURCES,
    HERMES_STATE_DB,
    INCLUDE_TOOL_MESSAGES,
    MAX_EPISODES_PER_RUN,
    MAX_MESSAGE_CHARS,
    MAX_SESSIONS_PER_RUN,
    SESSION_SETTLE_SECONDS,
    STATE_PATH,
)
from hermes_memory import read_json, write_json

log = logging.getLogger("hermesume.sessions")

# Hermes stores list/dict (multimodal) content as this prefix + JSON.
_CONTENT_JSON_PREFIX = "\x00json:"


def load_cursor() -> float:
    return float(read_json(STATE_PATH, {}).get("session_cursor", 0.0))


def save_cursor(value: float):
    state = read_json(STATE_PATH, {})
    state["session_cursor"] = value
    state["updated_at"] = time.time()
    write_json(STATE_PATH, state)


def _columns(conn, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _decode_content(content) -> str:
    """Flatten Hermes message content to plain text."""
    if content is None:
        return ""
    if not isinstance(content, str):
        return str(content)
    if not content.startswith(_CONTENT_JSON_PREFIX):
        return content
    try:
        parts = json.loads(content[len(_CONTENT_JSON_PREFIX):])
    except json.JSONDecodeError:
        return ""
    if isinstance(parts, dict):
        parts = [parts]
    texts = []
    for p in parts if isinstance(parts, list) else []:
        if isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text"):
            texts.append(p.get("text", ""))
        elif isinstance(p, str):
            texts.append(p)
    return "\n".join(t for t in texts if t)


def connect_readonly(db_path: str = HERMES_STATE_DB):
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def load_sessions(db_path: str = HERMES_STATE_DB, cursor: float | None = None,
                  now: float | None = None) -> list[dict]:
    """Settled sessions with activity after the cursor, oldest first.

    Returns [{"id", "source", "title", "activity", "messages": [(role, text)]}].
    """
    if not os.path.exists(db_path):
        log.warning("Hermes state.db not found: %s", db_path)
        return []
    cursor = load_cursor() if cursor is None else cursor
    now = time.time() if now is None else now
    settled_before = now - SESSION_SETTLE_SECONDS

    conn = connect_readonly(db_path)
    try:
        scols = _columns(conn, "sessions")
        mcols = _columns(conn, "messages")

        activity_parts = [c for c in ("last_activity_at", "ended_at") if c in scols]
        activity = f"COALESCE({', '.join(activity_parts + ['started_at'])})"
        title = "title" if "title" in scols else "NULL"
        where = [f"{activity} > ?", f"{activity} <= ?"]
        params: list = [cursor, settled_before]
        if "hidden" in scols:
            where.append("COALESCE(hidden, 0) = 0")
        if EXCLUDE_SOURCES:
            where.append(f"source NOT IN ({','.join('?' * len(EXCLUDE_SOURCES))})")
            params += EXCLUDE_SOURCES

        rows = conn.execute(
            f"SELECT id, source, {title} AS title, {activity} AS activity "
            f"FROM sessions WHERE {' AND '.join(where)} "
            f"ORDER BY activity ASC LIMIT ?",
            params + [MAX_SESSIONS_PER_RUN],
        ).fetchall()

        roles = ["user", "assistant"] + (["tool"] if INCLUDE_TOOL_MESSAGES else [])
        mwhere = ["session_id = ?", f"role IN ({','.join('?' * len(roles))})"]
        # Compression archives originals as active=0, compacted=1 and inserts a
        # summary row; dream over the originals, not the summary.
        if "active" in mcols:
            mwhere.append("(active = 1" + (" OR compacted = 1)" if "compacted" in mcols else ")"))
        if "_compressed_summary" in mcols:
            mwhere.append("COALESCE(_compressed_summary, 0) = 0")

        sessions = []
        for row in rows:
            msgs = conn.execute(
                f"SELECT role, content FROM messages WHERE {' AND '.join(mwhere)} "
                f"ORDER BY timestamp ASC, id ASC",
                [row["id"], *roles],
            ).fetchall()
            messages = []
            for m in msgs:
                text = _decode_content(m["content"]).strip()
                if text:
                    messages.append((m["role"], text[:MAX_MESSAGE_CHARS]))
            sessions.append({
                "id": row["id"],
                "source": row["source"],
                "title": row["title"] or "",
                "activity": float(row["activity"]),
                "messages": messages,
            })
    finally:
        conn.close()

    log.info("Loaded %d settled sessions from state.db", len(sessions))
    return sessions


def chunk_session(session: dict) -> list[str]:
    """One chunk per exchange: a user turn plus the assistant replies to it."""
    chunks, current = [], []
    for role, text in session["messages"]:
        if role == "user" and current:
            chunks.append("\n".join(current))
            current = []
        label = {"user": "User", "assistant": "Assistant"}.get(role, role.title())
        current.append(f"{label}: {text}")
    if current:
        chunks.append("\n".join(current))
    return chunks


# ── optional markdown episodes ──

def load_episode_files(episode_dir: str = EPISODE_DIR) -> list[dict]:
    date_re = re.compile(r"\d{4}-\d{2}-\d{2}(-.+)?\.md$")
    files = sorted(f for f in glob.glob(os.path.join(episode_dir, "*.md"))
                   if date_re.search(f))
    episodes = []
    for path in files[:MAX_EPISODES_PER_RUN]:
        with open(path, "r", encoding="utf-8") as f:
            episodes.append({"path": path, "content": f.read()})
    return episodes


def chunk_episode(content: str) -> list[str]:
    chunks = []
    for section in re.split(r"\n(?=## )", content):
        for para in re.split(r"\n\n+", section.strip()):
            if para.strip():
                chunks.append(para.strip())
    return chunks


def archive_episode_files(paths: list[str]) -> int:
    os.makedirs(EPISODE_ARCHIVE_DIR, exist_ok=True)
    for p in paths:
        shutil.move(p, os.path.join(EPISODE_ARCHIVE_DIR, os.path.basename(p)))
    return len(paths)
