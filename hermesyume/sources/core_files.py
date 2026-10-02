"""MEMORY.md / USER.md, read-only (PLAN-v2 §7, §4.3 R5; v1 hermes_memory parse helpers KEEP).

Parsing and hashing live in ``provider/_yume/corefmt.py`` (single source, loaded by path) so the
dream and the provider can never disagree on ``core_sha``. This module only reads: plain
``open(…, "r", encoding="utf-8-sig")``, no ``<file>.lock``, no write path (T14).
"""

from __future__ import annotations

import logging
import re
from types import ModuleType
from typing import Any

from ..paths import Paths, load_provider_module
from ..types import CORE_TARGETS, CoreEntry

log = logging.getLogger("hermesyume.sources.core_files")

# Hermes defaults (tools/memory_tool.py) when config.yaml says nothing.
DEFAULT_LIMITS = {"memory": 2200, "user": 1375}
_ENABLED_KEYS = {"memory": "memory_enabled", "user": "user_profile_enabled"}
_LIMIT_KEYS = {"memory": "memory_char_limit", "user": "user_char_limit"}


def corefmt(paths: Paths | None = None) -> ModuleType:
    return load_provider_module("corefmt", paths)


def entry_sha(text: str) -> str:
    return corefmt().core_sha(text)


def read_core(paths: Paths) -> dict[str, list[CoreEntry]]:
    """{"memory": [...], "user": [...]}. Missing file → []. An undecodable file raises (never
    mistaken for an empty one)."""
    fmt = corefmt(paths)
    out: dict[str, list[CoreEntry]] = {}
    for target in CORE_TARGETS:
        entries = fmt.read_entries(paths.core_file(target))
        out[target] = [CoreEntry(target=target, index=i, text=e, sha=fmt.core_sha(e),
                                 label=fmt.entry_label(e)) for i, e in enumerate(entries)]
    return out


def _read_memory_section(path) -> dict[str, Any]:
    """`memory:` section of Hermes config.yaml ({} if unavailable). pyyaml if present, else a
    flat `  key: value` fallback (v1 behaviour)."""
    try:
        with open(str(path), "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        return {}
    try:
        import yaml  # optional
    except ImportError:
        yaml = None
    if yaml is not None:
        try:
            data = yaml.safe_load(raw) or {}
        except Exception as e:  # malformed yaml → defaults
            log.warning("config.yaml 파싱 실패: %s", type(e).__name__)
            return {}
        section = data.get("memory") if isinstance(data, dict) else None
        return section if isinstance(section, dict) else {}
    section: dict[str, Any] = {}
    inside = False
    for line in raw.splitlines():
        if re.match(r"^memory:\s*(#.*)?$", line):
            inside = True
            continue
        if inside:
            if line and not line[0].isspace() and not line.lstrip().startswith("#"):
                break
            m = re.match(r"^\s+(\w+):\s*([^#]*)", line)
            if m:
                section[m.group(1)] = m.group(2).strip().strip("'\"")
    return section


def _truthy(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _int(value: Any, default: int) -> int:
    try:
        n = int(str(value).strip())
        return n if n > 0 else default
    except (TypeError, ValueError):
        return default


def load_limits(paths: Paths) -> dict[str, dict]:
    """{"memory": {"enabled": bool, "limit": 2200}, "user": {"enabled": bool, "limit": 1375}}."""
    section = _read_memory_section(paths.hermes_config_yaml)
    return {t: {"enabled": _truthy(section.get(_ENABLED_KEYS[t]), True),
                "limit": _int(section.get(_LIMIT_KEYS[t]), DEFAULT_LIMITS[t])}
            for t in CORE_TARGETS}
