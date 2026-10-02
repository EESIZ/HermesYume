"""``$HERMES_HOME/hermesyume/config.json`` — flat keys shared by dream and provider (PLAN-v2 §4.1).

No secrets here. Unknown keys are preserved (and reported by ``validate``). The provider reads the
same file with its own stdlib loader; every key in ``PROVIDER_KEYS`` must have the same default in
``provider/_yume/config.py`` (parity test in tests/dream).

Providers (DEVIATIONS PR-1, PR-3):
- ``embed_provider`` "auto" | "openai" | "hash". "auto" = openai when OPENAI_API_KEY is available,
  else the local ``hash/ngram-v1@1024`` embedder. A ``Config`` always holds the *resolved* provider;
  for hash, embed_model/embed_dim are fixed and the model's threshold defaults
  (``provider/_yume/hash_embed.MODEL_DEFAULTS``, the single source) replace the neural ones for
  every key the file does not set. ``yume init`` writes the resolved provider, so adding or
  removing a key later never flips the stored model (the ledger guard would refuse anyway).
- ``llm_provider`` "auto" | "openai" | "deepseek", resolved per run by ``llm.resolve_llm``
  ("auto" = deepseek when DEEPSEEK_API_KEY is available, else openai).
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping

from .paths import Paths


class ConfigError(ValueError):
    pass


DEFAULTS: dict[str, Any] = {
    # ── shared / identity ──
    "schema_version": 2,
    "enabled": True,                 # provider master switch (mtime-reloaded)
    "inject": True,                  # False = shadow mode (compute, log 'shadow', inject nothing)
    "embed_provider": "auto",        # auto | openai | hash (auto: openai if OPENAI_API_KEY, else hash)
    "embed_model": "text-embedding-3-small",
    "embed_dim": 1536,
    "embed_base_url": "",            # "" → $OPENAI_BASE_URL → https://api.openai.com/v1
    "llm_base_url": "",              # same fallback chain as embed_base_url
    "yume_bin": "~/.local/share/hermesyume/venv/bin/yume",
    "workspace_dir": "",             # the agent's workspace (docs/yume); "" = unset, opt in via config.json
    "hermes_runtime_dir": "",        # "" = discover (threat.default_runtime_dir), else the vendored copy
    "protect_homes": [],             # extra live homes guarded like ~/.hermes (paths.live_homes)
    "protect_workspaces": [],        # extra workspaces a sandbox home must never write into
    "scope": "default",

    # ── input (§3.1, §3.3, §3.4) ──
    "include_sources": ["telegram", "cli", "tui"],
    "owner_user_ids": [],            # [] = no owner filter
    "exclude_first_message_regex": "",   # "" = no filter; e.g. the first line of synthetic eval runs
    "deny_cwd_globs": [],            # sessions started in these cwds are skipped (fnmatch)
    "md_sources": [],                # markdown note folders; [] = none (opt in via config.json)
    "md_exclude_globs": [],
    "strip_line_regex": [],          # bot/status lines to drop before extraction (regex, per line)
    "repeat_line_min_chars": 20,
    "repeat_line_min_msgs": 5,
    "repeat_line_days": 30,
    "msg_max_chars": 3000,
    "msg_head_chars": 2000,
    "msg_tail_chars": 500,
    "code_max_lines": 15,
    "json_max_chars": 500,
    "blob_min_chars": 64,
    "window_min_user_chars": 20,
    "window_context_chars": 1500,

    # ── §4.1 thresholds and caps ──
    "settle_minutes": 30,
    "window_chars": 8000,
    "max_windows_per_run": 60,
    "max_llm_calls": 400,
    "max_embed_inputs": 3000,
    "max_runtime_min": 40,
    "llm_provider": "auto",          # auto | openai | deepseek (auto: deepseek if DEEPSEEK_API_KEY, else openai)
    "extract_model": "gpt-4.1-mini",  # OpenAI models; with deepseek only a deepseek-* name here is used
    "judge_model": "gpt-4.1-mini",
    "deepseek_model": "deepseek-v4-flash",
    "auto_dup_cos": 0.95,
    "candidate_cos": 0.72,
    "candidate_k": 5,
    "sweep_cos": 0.82,
    "sweep_max_pairs": 100,
    "max_judge_calls": 150,
    "consolidate_max": 400,
    "max_docs_per_run": 3,
    "dormant_strength": 0.10,
    "dormant_min_days": 21,
    "state_default_ttl_days": 14,
    "schedule_grace_days": 2,
    "forget_purge_days": 30,
    "lance_cleanup_days": 14,
    "mass_change_min": 10,           # mass_change = max(mass_change_min, ceil(ratio · active)); U2: informational only (never holds)
    "mass_change_ratio": 0.10,
    "recall_min_cos": 0.40,          # never below 0.40 (calibrate floor)
    "pinned_min_cos": 0.33,
    "recall_rel_cut": 0.10,
    "recall_k": 5,
    "recall_budget_chars": 1000,
    "recall_item_chars": 300,
    "injected_strong_cos": 0.50,
    "pins_budget_chars": 800,

    # ── dream internals ──
    "extract_max_tokens": 2000,
    "judge_max_tokens": 400,
    "consolidate_max_tokens": 600,
    "core_classify_max_tokens": 1500,
    "llm_temperature": 0.0,
    "llm_timeout_s": 60,
    "llm_tokens_param": "max_completion_tokens",   # or "max_tokens" for older compatible servers
    "embed_batch": 64,
    "embed_timeout_s": 30,
    "window_max_attempts": 3,        # 3rd failure → quarantined + alert
    "rejudge_max": 50,               # R2
    "sweep_days": 7,                 # R3
    "suppress_cos": 0.90,            # R0-2
    "core_match_cos": 0.90,          # R5
    "reinforce_platforms": ["telegram", "cli", "tui"],
    "backlog_alert_nights": 3,
    "stall_alert_nights": 2,
    "fail_rate_alert": 0.20,
    "active_rows_alert": 5000,
    "total_rows_alert": 50000,
    "health_embed_fail_alert": 0.20,
    "serving_stale_hours": 36,
    "prefetch_p95_alert_ms": 3000,
    "ledger_backups_keep": 7,
    "weekly_tar_keep": 4,
    "prepurge_backups_keep": 2,      # Lance tar taken before a quarantine purge's cleanup(0) (§5.4)
    "runs_keep_days": 14,            # committed/failed/dry runs/<id>/ (plan.json holds row text) pruned after
    "inbox_keep_days": 30,           # consumed/skipped live.db inbox rows (their text) deleted after
    "alert_telegram": False,         # U4: also send alerts as a caption-less .txt via sendDocument (token/chat id from .env by name)
    "llm_price_in_per_mtok": 0.40,   # USD, for Dream Log cost lines only
    "llm_price_out_per_mtok": 1.60,
    "embed_price_per_mtok": 0.02,

    # ── provider runtime (§6) ──
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

# Keys the provider (stdlib side) reads. Its own DEFAULTS must equal these values.
PROVIDER_KEYS: frozenset[str] = frozenset({
    "schema_version", "enabled", "inject", "embed_provider", "embed_model", "embed_dim",
    "embed_base_url", "yume_bin", "scope", "exclude_first_message_regex", "deny_cwd_globs",
    "recall_min_cos", "pinned_min_cos", "recall_rel_cut", "recall_k", "recall_budget_chars",
    "recall_item_chars", "injected_strong_cos", "pins_budget_chars",
    "recall_platforms", "allowed_chat_types", "show_status_cli", "show_status_tui",
    "show_status_telegram", "prefetch_deadline_s", "embed_connect_timeout_s",
    "embed_total_timeout_s", "embed_lru_size", "breaker_failures", "breaker_cooldown_s",
    "tool_embed_timeout_s", "tool_total_timeout_s", "search_min_cos", "short_query_chars",
    "query_max_chars", "prev_user_tail_chars", "stage1_k", "mmr_cos", "fts_top_k",
    "score_w_strength", "score_w_pinned", "score_w_keyword", "used_ratio", "used_min_tokens",
    "live_busy_timeout_ms", "event_flush_every", "health_flush_s", "static_block_chars",
    "pin_core_containment",
})

# Keys whose value may legitimately be null.
NULLABLE_KEYS: frozenset[str] = frozenset()
RECALL_MIN_COS_FLOOR = 0.40          # neural embeddings; hash has its own (hash_embed.RECALL_MIN_COS_FLOOR)
EMBED_PROVIDERS = ("auto", "openai", "hash")
LLM_PROVIDERS = ("auto", "openai", "deepseek")


def hash_embed():
    """``provider/_yume/hash_embed.py`` loaded by path (stdlib; the provider imports the same file)."""
    from .paths import load_provider_module
    return load_provider_module("hash_embed")


def recall_min_cos_floor(embed_provider: str | None) -> float:
    """0.40 for neural embeddings; the hash model's own measured floor for "hash"."""
    return float(hash_embed().recall_min_cos_floor(embed_provider))


_COS_KEYS = ("auto_dup_cos", "candidate_cos", "sweep_cos", "recall_min_cos", "pinned_min_cos",
             "injected_strong_cos", "search_min_cos", "mmr_cos", "suppress_cos", "core_match_cos",
             "pin_core_containment", "used_ratio", "mass_change_ratio", "dormant_strength")


class Config(Mapping[str, Any]):
    """Immutable view: ``cfg.settle_minutes`` or ``cfg["settle_minutes"]``."""

    __slots__ = ("_v", "source_path", "file_exists", "embed_provider_setting")

    def __init__(self, values: Mapping[str, Any] | None = None, *,
                 source_path: Path | None = None, file_exists: bool = False,
                 has_openai_key: bool | None = None):
        """``has_openai_key`` resolves embed_provider "auto" (None = unknown → openai, the classic
        default; ``load_config`` passes the real answer)."""
        merged = copy.deepcopy(DEFAULTS)
        if values:
            merged.update(copy.deepcopy(dict(values)))
        setting = hash_embed().apply_model(merged, explicit=set(values or ()),
                                           has_openai_key=has_openai_key)
        object.__setattr__(self, "_v", merged)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "file_exists", file_exists)
        object.__setattr__(self, "embed_provider_setting", setting)

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self._v[key]
        except KeyError:
            raise AttributeError(f"unknown config key {key!r}") from None

    def __setattr__(self, key, value):
        raise AttributeError("Config is immutable; use replace()")

    def __getitem__(self, key: str) -> Any:
        return self._v[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._v)

    def __len__(self) -> int:
        return len(self._v)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._v)

    def replace(self, **changes: Any) -> "Config":
        v = self.as_dict()
        v.update(changes)
        c = Config(v, source_path=self.source_path, file_exists=self.file_exists)
        if "embed_provider" not in changes:
            object.__setattr__(c, "embed_provider_setting", self.embed_provider_setting)
        return c

    def unknown_keys(self) -> list[str]:
        return sorted(k for k in self._v if k not in DEFAULTS)

    # ── derived values ──
    def mass_change_threshold(self, active_count: int) -> int:
        return max(int(self._v["mass_change_min"]),
                   math.ceil(float(self._v["mass_change_ratio"]) * max(0, active_count)))

    def embed_model_id(self) -> str:
        return f"{self._v['embed_provider']}/{self._v['embed_model']}@{int(self._v['embed_dim'])}"

    def show_status(self, platform: str | None) -> bool:
        return bool(self._v.get(f"show_status_{platform or ''}", False))


def _type_ok(default: Any, value: Any) -> bool:
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
    if isinstance(default, dict):
        return isinstance(value, dict)
    return True


def validate(values: Mapping[str, Any], *, force: bool = False,
             has_openai_key: bool | None = None) -> list[str]:
    """Return error strings (empty = valid). Unknown keys are warnings, not errors.
    The recall_min_cos floor follows the (resolved) embed provider: 0.40 for neural embeddings,
    the hash model's own floor for hash."""
    errs: list[str] = []
    for k, v in values.items():
        if k not in DEFAULTS:
            continue
        if v is None and k in NULLABLE_KEYS:
            continue
        if not _type_ok(DEFAULTS[k], v):
            errs.append(f"{k}: expected {type(DEFAULTS[k]).__name__}, got {type(v).__name__}")
    for k in _COS_KEYS:
        v = values.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and not (0.0 <= v <= 1.0):
            errs.append(f"{k}: must be within [0, 1]")
    ep = values.get("embed_provider", DEFAULTS["embed_provider"])
    if isinstance(ep, str) and ep.strip().lower() not in EMBED_PROVIDERS:
        errs.append(f"embed_provider: one of {', '.join(EMBED_PROVIDERS)}")
    lp = values.get("llm_provider", DEFAULTS["llm_provider"])
    if isinstance(lp, str) and lp.strip().lower() not in LLM_PROVIDERS:
        errs.append(f"llm_provider: one of {', '.join(LLM_PROVIDERS)}")
    rmc = values.get("recall_min_cos")
    if not force and isinstance(rmc, (int, float)) and isinstance(ep, str):
        floor = recall_min_cos_floor(hash_embed().resolve_provider(ep, has_openai_key))
        if rmc < floor:
            errs.append(f"recall_min_cos: must be >= {floor} (use force to override)")
    dim = values.get("embed_dim")
    if isinstance(dim, int) and dim < 256:
        errs.append("embed_dim: must be >= 256 (serving vec256 prefix)")
    return errs


def openai_key_available(paths: Paths | None) -> bool:
    from .secrets_env import OPENAI_API_KEY, has_secret
    return has_secret(OPENAI_API_KEY, paths)


def _key_for(values: Mapping[str, Any], paths: Paths | None) -> bool | None:
    """Look the key up only when "auto" needs it (no .env read otherwise)."""
    ep = str(values.get("embed_provider", DEFAULTS["embed_provider"]) or "").strip().lower()
    return openai_key_available(paths) if ep == "auto" else None


def load_config(paths: Paths, *, strict: bool = True) -> Config:
    """Defaults ← config.json, providers resolved (see module docstring). Missing file → defaults
    (file_exists=False). Invalid JSON or type errors raise ConfigError when strict."""
    p = paths.config_json
    if not p.exists():
        return Config(source_path=p, file_exists=False, has_openai_key=_key_for({}, paths))
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigError(f"config.json 읽기 실패: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError("config.json must be a JSON object")
    errs = validate(raw, force=True)
    if errs and strict:
        raise ConfigError("; ".join(errs))
    return Config(raw, source_path=p, file_exists=True, has_openai_key=_key_for(raw, paths))


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".cfg_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def save_config(paths: Paths, values: Mapping[str, Any], *, force: bool = False) -> Config:
    key = _key_for(values, paths)
    errs = validate(values, force=force, has_openai_key=key)
    if errs:
        raise ConfigError("; ".join(errs))
    _atomic_write_json(paths.config_json, dict(values))
    return Config(values, source_path=paths.config_json, file_exists=True, has_openai_key=key)


def initial_values(paths: Paths) -> dict[str, Any]:
    """Every key, as `yume init` writes it. The defaults carry no machine-specific paths:
    workspace_dir/md_sources start empty (the operator sets them with `yume config set`) and
    hermes_runtime_dir "" means "discover". embed_provider "auto" is resolved now and written as
    "openai" or "hash" (with the hash model's thresholds), so the stored model does not change when
    a key is added or removed later. llm_provider stays "auto" (nothing stored depends on it)."""
    values = copy.deepcopy(DEFAULTS)
    hash_embed().apply_model(values, explicit=(), has_openai_key=_key_for(values, paths))
    return values


def write_default_config(paths: Paths, *, overwrite: bool = False) -> bool:
    """`yume init`: write every default key explicitly (dashboard flat_json friendly). Returns
    False if the file existed and overwrite is False."""
    if paths.config_json.exists() and not overwrite:
        return False
    paths.ensure_dir(paths.data_dir)
    _atomic_write_json(paths.config_json, initial_values(paths))
    return True


def parse_cli_value(raw: str) -> Any:
    """`yume config set k v`: JSON if it parses (numbers, true/false, null, lists, quoted strings),
    otherwise the raw string."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def set_key(paths: Paths, key: str, raw_value: str, *, force: bool = False) -> Config:
    """Read-modify-write one key. Unknown keys are refused unless force."""
    if key not in DEFAULTS and not force:
        raise ConfigError(f"unknown key {key!r}")
    value = parse_cli_value(raw_value)
    if key in DEFAULTS and isinstance(DEFAULTS[key], float) and isinstance(value, int) \
            and not isinstance(value, bool):
        value = float(value)
    current: dict[str, Any] = {}
    if paths.config_json.exists():
        current = json.loads(paths.config_json.read_text(encoding="utf-8"))
    current[key] = value
    return save_config(paths, current, force=force)
