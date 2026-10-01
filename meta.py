"""Sidecar metadata for Hermes memory entries.

MEMORY.md / USER.md are plain text -- no importance, no timestamps. Hermesume
keeps that bookkeeping next to them, keyed by a hash of the entry text:

  {"<target>:<sha1>": {"importance": 0.7, "first_seen": ts,
                       "last_reinforced": ts, "vector": [...] | null}}

Entries the agent wrote itself (unknown hash) are adopted on first sight.
"""

import hashlib
import time

from config import (
    AGENT_ENTRY_IMPORTANCE,
    IMPORTANCE_DECAY_RATE,
    META_PATH,
    REINFORCE_BOOST,
)
from hermes_memory import read_json, write_json

DAY = 86400.0


def key(target: str, text: str) -> str:
    return f"{target}:{hashlib.sha1(text.encode('utf-8')).hexdigest()}"


class Meta:
    def __init__(self, path: str = META_PATH):
        self.path = path
        self.data: dict = read_json(path, {})

    def get(self, target: str, text: str, now: float | None = None) -> dict:
        """Metadata for an entry, adopting it if never seen before."""
        now = time.time() if now is None else now
        k = key(target, text)
        if k not in self.data:
            self.data[k] = {"importance": AGENT_ENTRY_IMPORTANCE,
                            "first_seen": now, "last_reinforced": now,
                            "vector": None}
        return self.data[k]

    def set(self, target: str, text: str, importance: float,
            now: float | None = None, vector=None, first_seen=None):
        now = time.time() if now is None else now
        self.data[key(target, text)] = {
            "importance": max(0.0, min(1.0, float(importance))),
            "first_seen": first_seen or now,
            "last_reinforced": now,
            "vector": vector,
        }

    def reinforce(self, target: str, text: str, now: float | None = None):
        now = time.time() if now is None else now
        m = self.get(target, text, now)
        m["importance"] = min(1.0, m["importance"] + REINFORCE_BOOST)
        m["last_reinforced"] = now

    def score(self, target: str, text: str, now: float | None = None) -> float:
        """Decayed importance: fades linearly with days since last reinforced."""
        now = time.time() if now is None else now
        m = self.get(target, text, now)
        days = max(0.0, (now - m["last_reinforced"]) / DAY)
        return max(0.0, m["importance"] - IMPORTANCE_DECAY_RATE * days)

    def drop(self, target: str, text: str):
        self.data.pop(key(target, text), None)

    def prune(self, live: dict[str, list[str]]):
        """Forget metadata for entries no longer present in any file."""
        keep = {key(t, e) for t, entries in live.items() for e in entries}
        self.data = {k: v for k, v in self.data.items() if k in keep}

    def save(self):
        write_json(self.path, self.data)
