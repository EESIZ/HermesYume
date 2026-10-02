"""Read named secrets from ``$HERMES_HOME/.env`` by *name only* (PLAN-v2 §10.3).

Never ``source``s the file, never exports anything, never logs values. Resolution for a name:
``$HERMES_HOME/.env`` value → ``os.environ`` → None. (.env wins so a sandbox HERMES_HOME is
self-contained even when the caller's shell exports real keys.)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Mapping

from .paths import Paths

OPENAI_API_KEY = "OPENAI_API_KEY"
OPENAI_BASE_URL = "OPENAI_BASE_URL"
DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL = "DEEPSEEK_BASE_URL"
TELEGRAM_BOT_TOKEN = "TELEGRAM_BOT_TOKEN"
YUME_ALERT_CHAT_ID = "YUME_ALERT_CHAT_ID"          # optional explicit alert chat id
TELEGRAM_ALLOWED_USERS = "TELEGRAM_ALLOWED_USERS"  # fallback: first id (a private chat id equals the user id)
KNOWN_NAMES = (OPENAI_API_KEY, OPENAI_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
               TELEGRAM_BOT_TOKEN, YUME_ALERT_CHAT_ID, TELEGRAM_ALLOWED_USERS)
DEFAULT_OPENAI_BASE = "https://api.openai.com/v1"
DEFAULT_DEEPSEEK_BASE = "https://api.deepseek.com"
SOURCE_ENV_FILE = "$HERMES_HOME/.env"
SOURCE_PROCESS = "environment"

_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def _unquote(raw: str) -> str:
    v = raw.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        inner = v[1:-1]
        if v[0] == '"':
            inner = (inner.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\"))
        return inner
    # unquoted: strip inline comment (" #")
    m = re.search(r"\s+#", v)
    if m:
        v = v[: m.start()]
    return v.strip()


def parse_env_text(text: str, names: Iterable[str] | None = None) -> dict[str, str]:
    """Parse KEY=VALUE lines; returns only `names` (all keys if None). Later lines win."""
    wanted = None if names is None else set(names)
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        key = m.group(1)
        if wanted is not None and key not in wanted:
            continue
        out[key] = _unquote(m.group(2))
    return out


def parse_env_file(path: str | os.PathLike, names: Iterable[str] | None = None) -> dict[str, str]:
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        text = p.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return {}
    return parse_env_text(text, names)


def usable(value: str | None) -> bool:
    """False for empty values and example placeholders such as ``your_openai_api_key``."""
    v = (value or "").strip()
    return bool(v) and not v.lower().startswith(("your_", "your-", "<"))


def secret_with_source(name: str, paths: Paths | None = None, *,
                       env: Mapping[str, str] | None = None) -> tuple[str | None, str | None]:
    """(value, where) — where is ``SOURCE_ENV_FILE`` or ``SOURCE_PROCESS`` (a *name* of the place,
    never the value), (None, None) when unset or a placeholder."""
    env = os.environ if env is None else env
    if paths is not None:
        val = parse_env_file(paths.hermes_env, [name]).get(name)
        if usable(val):
            return val, SOURCE_ENV_FILE
    val = env.get(name)
    if usable(val):
        return val, SOURCE_PROCESS
    return None, None


def get_secret(name: str, paths: Paths | None = None, *,
               env: Mapping[str, str] | None = None) -> str | None:
    return secret_with_source(name, paths, env=env)[0]


def secret_source(name: str, paths: Paths | None = None, *,
                  env: Mapping[str, str] | None = None) -> str | None:
    """Where `name` would be read from (for doctor), or None. Never the value."""
    return secret_with_source(name, paths, env=env)[1]


def has_secret(name: str, paths: Paths | None = None, *,
               env: Mapping[str, str] | None = None) -> bool:
    return secret_with_source(name, paths, env=env)[0] is not None


def openai_base_url(cfg_value: str | None, paths: Paths | None = None, *,
                    env: Mapping[str, str] | None = None) -> str:
    """config value (embed_base_url / llm_base_url) → OPENAI_BASE_URL (.env, env) → default."""
    if cfg_value:
        return cfg_value.rstrip("/")
    return (get_secret(OPENAI_BASE_URL, paths, env=env) or DEFAULT_OPENAI_BASE).rstrip("/")


def deepseek_base_url(paths: Paths | None = None, *, env: Mapping[str, str] | None = None) -> str:
    """DEEPSEEK_BASE_URL (.env, env) → https://api.deepseek.com. llm_base_url never applies here,
    so a DeepSeek key is only ever sent to a DeepSeek endpoint."""
    return (get_secret(DEEPSEEK_BASE_URL, paths, env=env) or DEFAULT_DEEPSEEK_BASE).rstrip("/")


def alert_chat_id(paths: Paths | None = None, *, env: Mapping[str, str] | None = None) -> str | None:
    """U4 alert destination: YUME_ALERT_CHAT_ID, else the first TELEGRAM_ALLOWED_USERS entry."""
    explicit = get_secret(YUME_ALERT_CHAT_ID, paths, env=env)
    if explicit:
        return explicit.strip()
    allowed = get_secret(TELEGRAM_ALLOWED_USERS, paths, env=env) or ""
    for part in re.split(r"[,\s]+", allowed):
        if part.strip():
            return part.strip()
    return None


def mask(value: str | None) -> str:
    """Safe description of a secret for logs: never any characters of the value."""
    if not value:
        return "<unset>"
    return f"<set len={len(value)}>"
