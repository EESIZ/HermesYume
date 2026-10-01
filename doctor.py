#!/usr/bin/env python3
"""HermesYume doctor: check that Hermes is where we think it is.

Creates $HERMESYUME_HOME and verifies, read-only:
  - $HERMES_HOME/state.db exists and has the sessions/messages tables
  - MEMORY.md / USER.md parse and round-trip in Hermes' §-delimited format
  - char limits / enabled flags from $HERMES_HOME/config.yaml
  - an API key is present for the configured providers

Usage:
    python doctor.py
"""

import os
import sqlite3
import sys

from config import (
    DREAM_LOG_DIR,
    EMBEDDING_PROVIDER,
    EPISODE_ARCHIVE_DIR,
    HERMES_CONFIG_PATH,
    HERMES_HOME,
    HERMES_STATE_DB,
    HERMESYUME_HOME,
    LLM_PROVIDER,
    MEMORY_ARCHIVE_DIR,
    OPENAI_API_KEY,
)
from hermes_memory import TARGETS, char_count, load_limits, parse_entries, path_for, serialize


def main() -> int:
    problems = 0
    print(f"HERMES_HOME:    {HERMES_HOME}")
    print(f"HERMESYUME_HOME: {HERMESYUME_HOME}")

    for d in (DREAM_LOG_DIR, MEMORY_ARCHIVE_DIR, EPISODE_ARCHIVE_DIR):
        os.makedirs(d, exist_ok=True)

    # state.db
    if not os.path.exists(HERMES_STATE_DB):
        print(f"\n[!] state.db not found: {HERMES_STATE_DB}")
        problems += 1
    else:
        conn = sqlite3.connect(f"file:{HERMES_STATE_DB}?mode=ro", uri=True)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        missing = {"sessions", "messages"} - tables
        if missing:
            print(f"\n[!] state.db is missing tables: {', '.join(sorted(missing))}")
            problems += 1
        else:
            n_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            n_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            print(f"\n[ok] state.db: {n_sessions} sessions, {n_messages} messages")
        conn.close()

    # memory files
    print(f"\nconfig.yaml: {HERMES_CONFIG_PATH}"
          f"{'' if os.path.exists(HERMES_CONFIG_PATH) else ' (not found, using defaults)'}")
    limits = load_limits()
    for target in TARGETS:
        path = path_for(target)
        lim = limits[target]
        state = "enabled" if lim["enabled"] else "DISABLED"
        if not os.path.exists(path):
            print(f"[ok] {os.path.basename(path)}: not created yet ({state}, limit {lim['limit']})")
            continue
        with open(path, "r", encoding="utf-8-sig") as f:
            raw = f.read()
        entries = parse_entries(raw)
        used = char_count(entries)
        print(f"[ok] {os.path.basename(path)}: {len(entries)} entries, "
              f"{used}/{lim['limit']} chars ({state})")
        if raw.strip() != serialize(entries):
            print(f"[!] {os.path.basename(path)} does not round-trip through the "
                  "§-delimited format; Hermes will refuse to write it (drift guard). "
                  "Fix it before running HermesYume.")
            problems += 1

    # providers
    print(f"\nEmbedding provider: {EMBEDDING_PROVIDER} / LLM provider: {LLM_PROVIDER}")
    if "openai" in (EMBEDDING_PROVIDER, LLM_PROVIDER) and not OPENAI_API_KEY:
        print("[!] OPENAI_API_KEY is not set")
        problems += 1

    print("\nAll good. Try: python hermesyume.py --dry-run" if not problems
          else f"\n{problems} problem(s) found.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
