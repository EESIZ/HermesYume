"""config.py: §4.1 defaults, provider keys, load/save/set, recall_min_cos floor."""

import json
import os
import stat

import pytest

from hermesyume.config import (DEFAULTS, PROVIDER_KEYS, Config, ConfigError, load_config,
                               parse_cli_value, save_config, set_key, write_default_config)

PLAN_4_1 = {
    "settle_minutes": 30, "window_chars": 8000, "max_windows_per_run": 60, "max_llm_calls": 400,
    "max_embed_inputs": 3000, "max_runtime_min": 40, "extract_model": "gpt-4.1-mini",
    "judge_model": "gpt-4.1-mini", "auto_dup_cos": 0.95, "candidate_cos": 0.72, "candidate_k": 5,
    "sweep_cos": 0.82, "sweep_max_pairs": 100, "max_judge_calls": 150, "consolidate_max": 400,
    "max_docs_per_run": 3, "dormant_strength": 0.10, "dormant_min_days": 21,
    "state_default_ttl_days": 14, "schedule_grace_days": 2, "forget_purge_days": 30,
    "lance_cleanup_days": 14, "recall_min_cos": 0.40, "pinned_min_cos": 0.33, "recall_rel_cut": 0.10,
    "recall_k": 5, "recall_budget_chars": 1000, "recall_item_chars": 300, "injected_strong_cos": 0.50,
    "pins_budget_chars": 800,
}
PLAN_OTHER = {
    "include_sources": ["telegram", "cli", "tui"],
    # no deployment-specific paths or filters ship as defaults (opt in via config.json)
    "workspace_dir": "", "md_sources": [], "hermes_runtime_dir": "",
    "exclude_first_message_regex": "", "deny_cwd_globs": [], "md_exclude_globs": [],
    "strip_line_regex": [], "protect_homes": [], "protect_workspaces": [],
    "reinforce_platforms": ["telegram", "cli", "tui"],
    "recall_platforms": ["telegram", "cli", "tui", "cron"],
    "allowed_chat_types": [None, "dm", "private"],
    "enabled": True, "embed_model": "text-embedding-3-small", "embed_dim": 1536,
    "show_status_cli": True, "show_status_telegram": False,
    "score_w_strength": 0.08, "score_w_pinned": 0.04, "score_w_keyword": 0.03,
    "stage1_k": 64, "mmr_cos": 0.92, "search_min_cos": 0.30, "prefetch_deadline_s": 4.0,
    "embed_connect_timeout_s": 1.0, "embed_total_timeout_s": 3.0, "embed_lru_size": 256,
    "breaker_failures": 3, "breaker_cooldown_s": 60, "tool_embed_timeout_s": 3.0,
    "tool_total_timeout_s": 5.0, "used_ratio": 0.35, "live_busy_timeout_ms": 300,
    "alert_telegram": False,                       # U4: off by default
}


def test_plan_defaults():
    for k, v in {**PLAN_4_1, **PLAN_OTHER}.items():
        assert DEFAULTS[k] == v, k
    assert PROVIDER_KEYS <= set(DEFAULTS)
    c = Config()
    assert c.mass_change_threshold(0) == 10 and c.mass_change_threshold(250) == 25
    assert c.embed_model_id() == "openai/text-embedding-3-small@1536"
    assert c.show_status("cli") is True and c.show_status("telegram") is False
    assert c.show_status("cron") is False


def test_removed_v1_keys_absent():
    for k in ("FILL_RATIO", "PREV_STATE_MAX_CHARS", "ENTRY_MAX_CHARS", "MAX_NEW_FACTS",
              "fill_ratio", "prev_state_max_chars", "entry_max_chars", "max_new_facts"):
        assert k not in DEFAULTS


def test_config_immutable_and_attr():
    c = Config({"settle_minutes": 5})
    assert c.settle_minutes == 5 and c["window_chars"] == 8000
    with pytest.raises(AttributeError):
        c.settle_minutes = 3
    with pytest.raises(AttributeError):
        c.no_such_key
    assert c.replace(settle_minutes=0).settle_minutes == 0 and c.settle_minutes == 5


def test_load_missing_is_defaults(paths):
    c = load_config(paths)
    assert c.file_exists is False and c.settle_minutes == 30


def test_write_default_and_load(paths):
    assert write_default_config(paths) is True
    assert write_default_config(paths) is False
    mode = stat.S_IMODE(os.stat(paths.config_json).st_mode)
    assert mode == 0o600
    data = json.loads(paths.config_json.read_text(encoding="utf-8"))
    assert set(data) == set(DEFAULTS)
    c = load_config(paths)
    # F-1: no home inherits a workspace / md path (defaults carry none)
    assert c.file_exists and c.workspace_dir == "" and c.md_sources == [] and c.hermes_runtime_dir == ""
    # PR-1: "auto" is resolved at init (the fake home's .env has OPENAI_API_KEY) and written concretely
    assert data["embed_provider"] == "openai" and data["llm_provider"] == "auto"
    assert {k: v for k, v in c.as_dict().items() if k != "embed_provider"} == \
        {k: v for k, v in DEFAULTS.items() if k != "embed_provider"}


def test_initial_values_same_for_protected_home(paths, monkeypatch):
    from hermesyume.config import initial_values
    plain = initial_values(paths)
    monkeypatch.setenv("HERMESYUME_PROTECT_HOMES", str(paths.hermes_home))  # pretend: no write happens
    assert initial_values(paths) == plain == {**DEFAULTS, "embed_provider": "openai"}
    assert not any("/home/" in json.dumps(v) for v in plain.values())


def test_set_key_parsing(paths):
    write_default_config(paths)
    c = set_key(paths, "md_sources", '["/sb/md"]')
    assert c.md_sources == ["/sb/md"]
    c = set_key(paths, "embed_base_url", "http://10.255.255.1:9")
    assert c.embed_base_url == "http://10.255.255.1:9"
    c = set_key(paths, "recall_min_cos", "0.45")
    assert c.recall_min_cos == 0.45
    c = set_key(paths, "pinned_min_cos", "1")
    assert isinstance(c.pinned_min_cos, float)
    c = set_key(paths, "inject", "false")
    assert c.inject is False
    assert load_config(paths).inject is False


def test_validation(paths):
    write_default_config(paths)
    with pytest.raises(ConfigError, match="0.4"):
        set_key(paths, "recall_min_cos", "0.39")
    set_key(paths, "recall_min_cos", "0.39", force=True)
    with pytest.raises(ConfigError):
        set_key(paths, "settle_minutes", '"thirty"')
    with pytest.raises(ConfigError):
        set_key(paths, "nope", "1")
    with pytest.raises(ConfigError):
        save_config(paths, {"candidate_cos": 1.5})


def test_bad_json_raises(paths):
    paths.ensure_dir(paths.data_dir)
    paths.config_json.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(paths)


def test_parse_cli_value():
    assert parse_cli_value("12") == 12
    assert parse_cli_value("null") is None
    assert parse_cli_value("abc") == "abc"
    assert parse_cli_value('"x"') == "x"
