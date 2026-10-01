"""Read/write Hermes Agent's curated memory files (MEMORY.md / USER.md).

Mirrors the on-disk contract of Hermes' tools/memory_tool_store.py:
  - entries are stripped strings joined by "\\n§\\n"
  - writers take an exclusive flock on "<file>.lock" and replace the file
    atomically (temp file + rename), so readers never see a torn file
  - Hermes refuses to touch a file that doesn't round-trip through that format
    ("drift guard"), so everything written here must round-trip exactly
  - each target has a char budget (memory_char_limit / user_char_limit)

Hermes injects these files into the system prompt as a frozen snapshot at
session start, so changes made here show up from the agent's next session.
"""

import json
import logging
import os
import re
import shutil
import tempfile
import time
from contextlib import contextmanager

from config import (
    DEFAULT_MEMORY_CHAR_LIMIT,
    DEFAULT_USER_CHAR_LIMIT,
    ENTRY_DELIMITER,
    HERMES_CONFIG_PATH,
    HERMES_MEMORY_DIR,
    MEMORY_ARCHIVE_DIR,
)

try:
    import fcntl
except ImportError:  # Windows: Hermes uses msvcrt there; we degrade to no lock
    fcntl = None

log = logging.getLogger("hermesume.memory")

TARGETS = ("memory", "user")
FILENAMES = {"memory": "MEMORY.md", "user": "USER.md"}


# ── config.yaml ──

def _read_memory_section(path: str) -> dict:
    """Return the `memory:` section of Hermes' config.yaml ({} if unavailable)."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        return {}
    try:
        import yaml  # optional dependency
        data = yaml.safe_load(raw) or {}
        section = data.get("memory") if isinstance(data, dict) else None
        return section if isinstance(section, dict) else {}
    except ImportError:
        pass
    except Exception as e:
        log.warning("Could not parse %s: %s", path, e)
        return {}
    # Minimal fallback: flat `key: value` lines indented under `memory:`.
    section, inside = {}, False
    for line in raw.splitlines():
        if re.match(r"^memory:\s*(#.*)?$", line):
            inside = True
            continue
        if inside:
            if line and not line[0].isspace():
                break
            m = re.match(r"^\s+(\w+):\s*([^#]*)", line)
            if m:
                section[m.group(1)] = m.group(2).strip().strip("'\"")
    return section


def _truthy(value, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def load_limits(config_path: str = HERMES_CONFIG_PATH) -> dict:
    """{"memory": {"enabled": bool, "limit": int}, "user": {...}} from Hermes config."""
    section = _read_memory_section(config_path)
    return {
        "memory": {
            "enabled": _truthy(section.get("memory_enabled"), True),
            "limit": int(section.get("memory_char_limit") or DEFAULT_MEMORY_CHAR_LIMIT),
        },
        "user": {
            "enabled": _truthy(section.get("user_profile_enabled"), True),
            "limit": int(section.get("user_char_limit") or DEFAULT_USER_CHAR_LIMIT),
        },
    }


# ── entry format ──

def path_for(target: str, memory_dir: str = HERMES_MEMORY_DIR) -> str:
    return os.path.join(memory_dir, FILENAMES[target])


def parse_entries(raw: str) -> list[str]:
    """Split on the FULL delimiter (a bare "§" inside an entry survives)."""
    return [e for e in (x.strip() for x in raw.split(ENTRY_DELIMITER)) if e]


def serialize(entries: list[str]) -> str:
    return ENTRY_DELIMITER.join(entries)


def char_count(entries: list[str]) -> int:
    return len(serialize(entries))


def read_entries(target: str, memory_dir: str = HERMES_MEMORY_DIR) -> list[str]:
    """Entries of a memory file. Raises if the file exists but can't be read,
    so we never mistake an unreadable file for an empty one and wipe it."""
    path = path_for(target, memory_dir)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig") as f:
        return parse_entries(f.read())


def clean_entry(text: str) -> str:
    """Normalize LLM output into a single valid entry."""
    text = (text or "").replace("\r\n", "\n").strip()
    # The delimiter must never appear inside an entry.
    text = re.sub(r"\n\s*§\s*\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── threat scan ──
# Memory is injected into every future system prompt, so a poisoned entry is
# persistent. Prefer Hermes' own scanner when hermes-agent is importable;
# otherwise fall back to a small subset of its "all"-scope patterns.

_FILLER = r"(?:\w+\s+){0,8}"
_SECRET_VAR = r"\$\{?\w*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)S?\b"
_FALLBACK_PATTERNS = [
    (rf"ignore\s+{_FILLER}(previous|all|above|prior)\s+{_FILLER}instructions", "prompt_injection"),
    (r"system\s+prompt\s+override", "sys_prompt_override"),
    (rf"disregard\s+{_FILLER}(your|all|any)\s+{_FILLER}(instructions|rules|guidelines)", "disregard_rules"),
    (rf"do\s+not\s+{_FILLER}tell\s+{_FILLER}the\s+user", "deception_hide"),
    (rf"curl\s+[^\n]{{0,2048}}{_SECRET_VAR}", "exfil_curl"),
    (rf"wget\s+[^\n]{{0,2048}}{_SECRET_VAR}", "exfil_wget"),
    (r"cat\s+[^\n]{0,2048}(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)", "read_secrets"),
    (r"(send|post|upload|transmit)\s+[^\n]{0,2048}\s+(to|at)\s+https?://", "send_to_url"),
]
_INVISIBLE = {"​", "‌", "‍", "⁠", "﻿", "‪",
              "‫", "‬", "‭", "‮"}
# Obvious secrets that should never be written into a prompt.
_SECRET_LITERALS = re.compile(
    r"(sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{30,}|xox[abp]-[A-Za-z0-9-]{10,}|"
    r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")


def threat_findings(text: str) -> list[str]:
    try:
        from tools.threat_patterns import scan_for_threats  # hermes-agent
        findings = list(scan_for_threats(text, scope="strict"))
    except Exception:
        findings = [pid for pat, pid in _FALLBACK_PATTERNS
                    if re.search(pat, text, re.IGNORECASE)]
        findings += [f"invisible_unicode_U+{ord(c):04X}" for c in set(text) & _INVISIBLE]
    if _SECRET_LITERALS.search(text):
        findings.append("secret_literal")
    return findings


# ── locked, atomic writes ──

@contextmanager
def file_lock(path: str):
    """Same lock Hermes uses: exclusive flock on '<path>.lock'."""
    lock_path = path + ".lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    if fcntl is None:
        yield
        return
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _atomic_write(path: str, content: str):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    mode = (os.stat(path).st_mode & 0o777) if os.path.exists(path) else 0o600
    fd, tmp = tempfile.mkstemp(prefix=".mem_", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def backup(target: str, memory_dir: str = HERMES_MEMORY_DIR,
           archive_dir: str = MEMORY_ARCHIVE_DIR, stamp: str | None = None) -> str | None:
    """Copy the current file into memory-archive/<stamp>/ before modifying it."""
    src = path_for(target, memory_dir)
    if not os.path.exists(src):
        return None
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    dst_dir = os.path.join(archive_dir, stamp)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, FILENAMES[target])
    shutil.copy2(src, dst)
    return dst


def apply_ops(target: str, ops: list[dict], limit: int,
              memory_dir: str = HERMES_MEMORY_DIR, dry_run: bool = False,
              stamp: str | None = None) -> dict:
    """Apply planned operations against the CURRENT file, under Hermes' lock.

    The plan was computed from a snapshot; the agent may have written to the
    file since. Each op therefore targets an exact entry string and is skipped
    if that entry is gone. Ops:
      {"op": "add", "text": str}
      {"op": "replace", "old": str, "new": [str, ...]}
      {"op": "remove", "old": str}
    """
    path = path_for(target, memory_dir)
    result = {"applied": [], "skipped": [], "before_chars": 0, "after_chars": 0,
              "backup": None}

    with file_lock(path):
        entries = read_entries(target, memory_dir)
        result["before_chars"] = char_count(entries)
        working = list(entries)

        for op in ops:
            kind = op["op"]
            if kind == "add":
                if op["text"] in working:
                    result["skipped"].append({**op, "reason": "already present"})
                    continue
                working.append(op["text"])
            elif kind in ("replace", "remove"):
                if op["old"] not in working:
                    result["skipped"].append({**op, "reason": "entry changed since planning"})
                    continue
                idx = working.index(op["old"])
                others = working[:idx] + working[idx + 1:]
                new = ([t for t in op.get("new", []) if t not in others]
                       if kind == "replace" else [])
                working[idx:idx + 1] = new
            else:
                raise ValueError(f"unknown op: {kind}")
            result["applied"].append(op)

        # The agent may have added entries since planning: never exceed the hard
        # limit -- roll back our own adds (newest first) until it fits.
        while char_count(working) > limit:
            adds = [o for o in result["applied"] if o["op"] == "add"]
            if not adds:
                break
            last = adds[-1]
            working.remove(last["text"])
            result["applied"].remove(last)
            result["skipped"].append({**last, "reason": "would exceed char limit"})

        # Round-trip guarantee (Hermes drift guard).
        assert parse_entries(serialize(working)) == working

        result["after_chars"] = char_count(working)
        if working == entries:
            return result
        if not dry_run:
            result["backup"] = backup(target, memory_dir, stamp=stamp)
            _atomic_write(path, serialize(working))
    return result


def write_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
