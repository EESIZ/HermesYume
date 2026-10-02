"""Provider-side config: ``$HERMES_HOME/hermesyume/config.json`` (stdlib only, PLAN-v2 §4.1/§6).

- ``DEFAULTS`` equals ``{k: hermesyume.config.DEFAULTS[k] for k in PROVIDER_KEYS}`` (parity test).
- ``load()`` is mtime-cached and thread-safe; a missing file → None, invalid JSON → None (+ one
  warning). Values with the wrong type fall back to the default for that key.
- ``now()`` honors ``HERMESYUME_NOW`` with the same grammar as ``hermesyume.clock.parse_now_spec``.
- ``embed_provider`` is resolved exactly like the dream side (``hash_embed.apply_model``, one
  source): "auto" → openai when OPENAI_API_KEY is available in the process environment or
  ``$HERMES_HOME/.env`` (by name — the same places the dream looks, so both sides agree), else
  hash; for hash the model's own threshold defaults apply to every key the file does not set.
  ``yume init`` writes a concrete provider, so "auto" is rare.

The returned dict is shared (cached): callers must treat it as read-only.
"""

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone

try:
    from . import hash_embed
except ImportError:      # loaded by path without its package (provider/cli.py fallback)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "_hermesyume_hash_embed", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hash_embed.py"))
    hash_embed = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(hash_embed)

log = logging.getLogger("hermesyume.provider.config")

DATA_DIRNAME = "hermesyume"
CONFIG_NAME = "config.json"

DEFAULTS = {
    "schema_version": 2,
    "enabled": True,
    "inject": True,
    "embed_provider": "auto",
    "embed_model": "text-embedding-3-small",
    "embed_dim": 1536,
    "embed_base_url": "",
    "yume_bin": "~/.local/share/hermesyume/venv/bin/yume",
    "scope": "default",
    "exclude_first_message_regex": "",
    "deny_cwd_globs": [],
    "recall_min_cos": 0.40,
    "pinned_min_cos": 0.33,
    "recall_rel_cut": 0.10,
    "recall_k": 5,
    "recall_budget_chars": 1000,
    "recall_item_chars": 300,
    "injected_strong_cos": 0.50,
    "pins_budget_chars": 800,
    "recall_platforms": ["telegram", "cli", "tui", "cron"],
    "allowed_chat_types": [None, "dm", "private"],
    "show_status_cli": True,
    "show_status_tui": False,
    "show_status_telegram": False,
    "prefetch_deadline_s": 4.0,
    "embed_connect_timeout_s": 1.0,
    "embed_total_timeout_s": 3.0,
    "embed_lru_size": 256,
    "breaker_failures": 3,
    "breaker_cooldown_s": 60,
    "tool_embed_timeout_s": 3.0,
    "tool_total_timeout_s": 5.0,
    "search_min_cos": 0.30,
    "short_query_chars": 20,
    "query_max_chars": 1000,
    "prev_user_tail_chars": 200,
    "stage1_k": 64,
    "mmr_cos": 0.92,
    "fts_top_k": 3,
    "score_w_strength": 0.08,
    "score_w_pinned": 0.04,
    "score_w_keyword": 0.03,
    "used_ratio": 0.35,
    "used_min_tokens": 2,
    "live_busy_timeout_ms": 300,
    "event_flush_every": 10,
    "health_flush_s": 300,
    "static_block_chars": 400,
    "pin_core_containment": 0.8,
}

_LOCK = threading.Lock()
_CACHE = {}          # path -> (stat_key, dict | None); stat_key includes .env when "auto" needs it
_WARNED = set()      # (path, stat_key) already warned about


def data_dir(hermes_home):
    return os.path.join(str(hermes_home), DATA_DIRNAME)


def config_path(hermes_home):
    return os.path.join(data_dir(hermes_home), CONFIG_NAME)


def _type_ok(default, value):
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(default, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(default, str):
        return isinstance(value, str)
    if isinstance(default, list):
        return isinstance(value, list)
    return True


def _usable(value):
    v = (value or "").strip()
    return bool(v) and not v.lower().startswith(("your_", "your-", "<"))


def _env_has(hermes_home, name):
    """True when `name` has a usable value in ``$HERMES_HOME/.env`` (last assignment wins) or the
    process environment — the same rule as the dream's ``secrets_env``. The value itself is never
    kept or returned."""
    file_val = None
    try:
        with open(os.path.join(str(hermes_home), ".env"), "r", encoding="utf-8-sig") as f:
            for line in f:
                m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
                if not m or m.group(1) != name or line.lstrip().startswith("#"):
                    continue
                v = m.group(2).strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                    v = v[1:-1]
                else:
                    v = re.split(r"\s+#", v, maxsplit=1)[0].strip()
                file_val = v
    except OSError:
        pass
    return _usable(file_val) or _usable(os.environ.get(name))


def _wants_key(raw):
    return str(raw.get("embed_provider", DEFAULTS["embed_provider"]) or "").strip().lower() == "auto"


def _merge(raw, path, has_openai_key=None):
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()}
    bad = []
    explicit = set()
    for k, v in raw.items():
        if k in DEFAULTS:
            if not _type_ok(DEFAULTS[k], v):
                bad.append(k)
                continue
            if isinstance(DEFAULTS[k], float) and isinstance(v, int):
                v = float(v)
        out[k] = v
        explicit.add(k)
    if bad:
        log.warning("hermesyume config.json: 잘못된 타입이라 기본값 사용: %s", ", ".join(sorted(bad)))
    hash_embed.apply_model(out, explicit=explicit, has_openai_key=has_openai_key)
    return out


def load(hermes_home):
    """Merged config dict, or None when config.json is missing/unreadable/invalid.
    Never raises."""
    try:
        path = config_path(hermes_home)
        try:
            st = os.stat(path)
        except OSError:
            with _LOCK:
                _CACHE.pop(path, None)
            return None
        key = (st.st_ino, st.st_mtime_ns, st.st_size)
        with _LOCK:
            hit = _CACHE.get(path)
        if hit is not None and hit[0][:3] == key:
            # an "auto" entry is also keyed on $HERMES_HOME/.env (the key decides the provider)
            if len(hit[0]) == 3 or hit[0][3] == _env_stat(hermes_home):
                return hit[1]
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raise ValueError("config.json is not a JSON object")
            has_key = None
            if _wants_key(raw):
                key = key + (_env_stat(hermes_home),)
                has_key = _env_has(hermes_home, "OPENAI_API_KEY")
            val = _merge(raw, path, has_key)
        except Exception as e:  # invalid JSON etc.
            val = None
            with _LOCK:
                warn = (path, key) not in _WARNED
                _WARNED.add((path, key))
            if warn:
                log.warning("hermesyume config.json 읽기 실패(%s) — provider 비활성", type(e).__name__)
        with _LOCK:
            _CACHE[path] = (key, val)
        return val
    except Exception:
        return None


def _env_stat(hermes_home):
    try:
        st = os.stat(os.path.join(str(hermes_home), ".env"))
        return (st.st_ino, st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def clear_cache():
    with _LOCK:
        _CACHE.clear()
        _WARNED.clear()


def embed_model_id(cfg):
    """Model id of a loaded (resolved) config dict; a raw "auto" counts as openai."""
    prov = hash_embed.resolve_provider(cfg.get("embed_provider") or "auto", None)
    if prov == hash_embed.PROVIDER:
        return hash_embed.MODEL_ID
    return "%s/%s@%d" % (prov, cfg.get("embed_model") or "", int(cfg.get("embed_dim") or 0))


# ── time (stdlib copy of hermesyume.clock: same grammar, parity-tested) ──

KST = timezone(timedelta(hours=9), "KST")
DAY = 86400.0
HOUR = 3600.0
_REL_RE = re.compile(r"^([+-])\s*(\d+(?:\.\d+)?)\s*([smhdwy])$", re.I)
_UNIT = {"s": 1.0, "m": 60.0, "h": HOUR, "d": DAY, "w": 7 * DAY, "y": 365 * DAY}
_env_cache = None   # (raw, value)


def parse_iso(s, end_of_day=False):
    """'YYYY-MM-DD' / 'YYYY-MM-DDTHH:MM[:SS]' (+offset/Z). Naive → KST. Garbage → None."""
    if not s:
        return None
    s = str(s).strip()
    if s.lower() in ("null", "none", ""):
        return None
    date_only = re.fullmatch(r"\d{4}-\d{2}-\d{2}", s) is not None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00").replace(" ", "T", 1))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    if date_only and end_of_day:
        dt = dt + timedelta(days=1) - timedelta(seconds=1)
    return dt.timestamp()


def parse_now_spec(spec, base=None):
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return float(spec)
    s = str(spec).strip()
    if not s:
        raise ValueError("empty --now spec")
    m = _REL_RE.match(s)
    if m:
        sign = 1.0 if m.group(1) == "+" else -1.0
        return (time.time() if base is None else base) + sign * float(m.group(2)) * _UNIT[m.group(3).lower()]
    if re.fullmatch(r"\d{9,}(\.\d+)?", s):
        return float(s)
    ts = parse_iso(s)
    if ts is None:
        raise ValueError("bad --now spec: %r" % (spec,))
    return ts


def now():
    """HERMESYUME_NOW (resolved once per distinct value, then frozen) else time.time()."""
    global _env_cache
    raw = os.environ.get("HERMESYUME_NOW", "").strip()
    if raw:
        try:
            if _env_cache is None or _env_cache[0] != raw:
                _env_cache = (raw, parse_now_spec(raw))
            return _env_cache[1]
        except ValueError:
            return time.time()
    return time.time()


def kst_date(ts):
    return datetime.fromtimestamp(ts, tz=KST).strftime("%Y-%m-%d")


def kst_fmt(ts, fmt):
    return datetime.fromtimestamp(ts, tz=KST).strftime(fmt)
