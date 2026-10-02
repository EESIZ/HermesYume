"""LLM / embedding providers (DEVIATIONS PR-1..PR-3): DeepSeek and the local hash embedder.

- keys and base URLs by *name* from $HERMES_HOME/.env; doctor shows provider + key source names only
- llm_provider auto = deepseek when DEEPSEEK_API_KEY, else openai; thinking disabled, max_tokens,
  OpenAI model names mapped to deepseek_model, JSON-mode fallback, key never sent elsewhere
- embed_provider auto = openai when OPENAI_API_KEY, else hash/ngram-v1@1024 (resolved and written
  at init), the hash model's own thresholds and recall floor, Lance/serving dims follow the model,
  and the ledger guard refuses mixing models
- the dream's hash vectors equal the provider's bit for bit (one source file)
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from hermesyume import cli
from hermesyume.config import (DEFAULTS, Config, ConfigError, initial_values, load_config, set_key,
                               validate, write_default_config)
from hermesyume.embedder import EmbedError, HashNgramEmbedder, OpenAIEmbedder, make_embedder
from hermesyume.ledger import Ledger, MetaMismatch
from hermesyume.llm import DeepSeekLLM, LLMAuthError, OpenAILLM, make_llm, resolve_llm
from hermesyume.paths import Paths, find_provider_file
from hermesyume.secrets_env import secret_source
from hermesyume.store import SchemaMismatch, Store
from tests.fakes import FakeOpenAIServer
from tests.fixtures.hermes_home import FAKE_OPENAI_KEY

REPO = Path(__file__).resolve().parents[2]
DS_KEY = "sk-ds-" + "1" * 32           # fake


def _env(paths: Paths, text: str) -> None:
    paths.hermes_env.write_text(text, encoding="utf-8")


def _provider_hash_module():
    spec = importlib.util.spec_from_file_location("_t_prov_hash", REPO / "provider" / "_yume" / "hash_embed.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── hash embedder: one source, identical vectors ──────────────────────────────

def test_hash_embedder_is_the_provider_file_bit_for_bit(paths):
    assert find_provider_file("hash_embed") == REPO / "provider" / "_yume" / "hash_embed.py"
    emb = HashNgramEmbedder(paths)
    assert emb.model_id == "hash/ngram-v1@1024" and emb.dim == 1024
    texts = ["사용자는 Neovim으로 코딩한다.", "Project database is Postgres 17 on db.internal.", "!!!", ""]
    got = emb.embed(texts)
    want = np.asarray([_provider_hash_module().embed(t) for t in texts], dtype=np.float32)
    assert got.dtype == np.float32 and got.shape == (4, 1024)
    assert got.tobytes() == want.tobytes()
    assert np.allclose(np.linalg.norm(got[:3], axis=1), 1.0, atol=1e-6) and not got[3].any()


def test_hash_embedder_free_and_offline(paths, cfg):
    from hermesyume.types import RunBudget
    budget = RunBudget(max_llm_calls=0, max_embed_inputs=1, max_runtime_s=60)
    hcfg = cfg.replace(embed_provider="hash")
    for offline in (False, True):
        emb = make_embedder(hcfg, paths, budget=budget, offline=offline)
        assert isinstance(emb, HashNgramEmbedder)
        emb.embed(["a", "b", "c"])                      # never charged: no BudgetExceeded
        emb.ping()
        assert emb.usage.content_inputs == 3
    bad = Config({"embed_provider": "openai", "embed_model": "ngram-v1", "embed_dim": 1024})
    assert isinstance(make_embedder(bad, paths), OpenAIEmbedder)    # a model name alone never picks hash
    from types import SimpleNamespace
    odd = SimpleNamespace(embed_provider="hash", embed_model_id=lambda: "hash/ngram-v2@512")
    with pytest.raises(EmbedError, match="ngram-v1@1024"):
        make_embedder(odd, paths)


# ── config resolution ────────────────────────────────────────────────────────

def test_config_auto_resolution():
    assert Config().embed_model_id() == "openai/text-embedding-3-small@1536"        # unknown key → openai
    c = Config(has_openai_key=False)
    assert c.embed_model_id() == "hash/ngram-v1@1024" and c.embed_provider_setting == "auto"
    assert (c.recall_min_cos, c.candidate_cos, c.auto_dup_cos, c.sweep_cos, c.injected_strong_cos) == \
        (0.30, 0.40, 0.90, 0.55, 0.40)
    assert Config(has_openai_key=True).embed_model_id() == "openai/text-embedding-3-small@1536"
    h = Config({"embed_provider": "hash", "embed_dim": 1536, "recall_min_cos": 0.27})
    assert h.embed_model_id() == "hash/ngram-v1@1024" and h.recall_min_cos == 0.27   # explicit wins
    assert h.pinned_min_cos == 0.25
    r = h.replace(embed_model="text-embedding-3-small", embed_dim=1536)             # reembed-style replace
    assert r.embed_model_id() == "hash/ngram-v1@1024" and r.embed_provider_setting == "hash"
    o = Config({"embed_provider": "openai"})
    assert o.recall_min_cos == 0.40 and o.embed_dim == 1536


def test_load_config_resolves_auto_from_env_names(paths):
    paths.ensure_dir(paths.data_dir)
    paths.config_json.write_text(json.dumps({"embed_provider": "auto"}), encoding="utf-8")
    assert load_config(paths).embed_model_id() == "openai/text-embedding-3-small@1536"   # fake key in .env
    _env(paths, "OPENAI_API_KEY=your_openai_api_key\n")                               # placeholder
    c = load_config(paths)
    assert c.embed_model_id() == "hash/ngram-v1@1024" and c.embed_provider_setting == "auto"


def test_recall_floor_is_for_neural_embeddings_only(paths):
    assert validate({"embed_provider": "hash", "recall_min_cos": 0.30}) == []
    assert validate({"embed_provider": "hash", "recall_min_cos": 0.20})          # below the hash floor
    assert validate({"embed_provider": "openai", "recall_min_cos": 0.30})
    assert validate({"recall_min_cos": 0.30})                                      # auto, key unknown
    assert validate({"recall_min_cos": 0.30}, has_openai_key=False) == []
    assert validate({"embed_provider": "ollama"}) and validate({"llm_provider": "minimax"})
    _env(paths, "OTHER=1\n")
    write_default_config(paths)
    assert set_key(paths, "recall_min_cos", "0.28").recall_min_cos == 0.28
    with pytest.raises(ConfigError, match="0.25"):
        set_key(paths, "recall_min_cos", "0.2")


def test_init_writes_resolved_provider(paths):
    _env(paths, "OTHER=1\n")                           # no OpenAI key → hash
    assert write_default_config(paths)
    data = json.loads(paths.config_json.read_text(encoding="utf-8"))
    assert (data["embed_provider"], data["embed_model"], data["embed_dim"]) == ("hash", "ngram-v1", 1024)
    assert data["recall_min_cos"] == 0.30 and data["llm_provider"] == "auto"
    _env(paths, f"OPENAI_API_KEY={FAKE_OPENAI_KEY}\n")   # adding a key later changes nothing stored
    assert load_config(paths).embed_model_id() == "hash/ngram-v1@1024"


def test_initial_values_never_leak_deployment_paths(tmp_path):
    home = tmp_path / "u" / ".hermes"
    home.mkdir(parents=True)
    (home / "gateway.pid").write_text("1")              # a live home gets the same generic defaults
    v = initial_values(Paths(home))
    assert v["workspace_dir"] == "" and v["md_sources"] == [] and v["hermes_runtime_dir"] == ""


# ── Lance / serving dimensions follow the model; the guard refuses mixing ─────

def test_hash_store_export_and_guard(paths, tmp_path):
    from types import SimpleNamespace
    from hermesyume import export
    from tests.fakes import make_row
    _env(paths, "OTHER=1\n")
    write_default_config(paths)
    cfg = load_config(paths)
    assert int(cfg.embed_dim) == 1024
    paths.ensure_data_dirs()
    with Ledger.from_paths(paths) as led:
        led.init_meta(cfg.embed_model_id(), int(cfg.embed_dim))
        with pytest.raises(MetaMismatch):
            led.check_meta("openai/text-embedding-3-small@1536", 1536)
    st = Store.from_config(paths, cfg, create=True)
    st.check_schema()
    emb = make_embedder(cfg, paths)
    text = "데모 서버 포트는 8123이다."
    row = make_row(text, embedder=emb, subject="데모 포트", vector=emb.embed([f"데모 포트: {text}"])[0])
    st.commit(upserts=[row])
    st.check_embed_model()
    hit = st.search(emb.embed(["데모 서버 포트"])[0], k=1)
    assert hit and hit[0][0].id == row.id and hit[0][0].vector.shape == (1024,)
    with pytest.raises(SchemaMismatch):
        Store.open(paths.lancedb_dir, dim=1536, embed_model="openai/text-embedding-3-small@1536").check_schema()
    ctx = SimpleNamespace(cfg=cfg, paths=paths, store=st, run_id="t-hash", now=1_790_000_000.0, dry_run=False,
                          note=lambda *_: None)
    out = export.build_serving(ctx)
    con = sqlite3.connect(f"file:{out}?mode=ro", uri=True)
    meta = dict(con.execute("SELECT key, value FROM meta"))
    blob = con.execute("SELECT vec FROM items").fetchone()[0]
    con.close()
    assert meta["embed_model"] == "hash/ngram-v1@1024" and meta["dim"] == "1024"
    assert len(blob) == 1024 * 4


# ── LLM providers ────────────────────────────────────────────────────────────

def _cfg_for(cfg, server, **kw):
    return cfg.replace(llm_base_url=server.base_url, **kw)


def test_resolve_llm_auto_and_sources(paths, cfg):
    st = resolve_llm(cfg, paths)
    assert (st.provider, st.model, st.key_name, st.key_source) == \
        ("openai", "gpt-4.1-mini", "OPENAI_API_KEY", "$HERMES_HOME/.env")
    _env(paths, f"OPENAI_API_KEY={FAKE_OPENAI_KEY}\nDEEPSEEK_API_KEY={DS_KEY}\n")
    st = resolve_llm(cfg, paths)
    assert (st.setting, st.provider, st.model, st.base_url) == \
        ("auto", "deepseek", "deepseek-v4-flash", "https://api.deepseek.com")
    assert st.key_source == "$HERMES_HOME/.env" and DS_KEY not in repr(st)
    assert resolve_llm(cfg.replace(llm_provider="openai"), paths).provider == "openai"
    _env(paths, "DEEPSEEK_API_KEY=\n")
    assert resolve_llm(cfg, paths, env={"DEEPSEEK_API_KEY": DS_KEY}).key_source == "environment"
    assert secret_source("DEEPSEEK_API_KEY", paths, env={}) is None


def test_deepseek_wire_format_and_key_isolation(paths, cfg):
    with FakeOpenAIServer() as ds, FakeOpenAIServer() as oa:
        _env(paths, f"OPENAI_API_KEY={FAKE_OPENAI_KEY}\nDEEPSEEK_API_KEY={DS_KEY}\n"
                    f"DEEPSEEK_BASE_URL={ds.base_url}\n")
        c = cfg.replace(llm_base_url=oa.base_url)        # llm_base_url is for openai only
        llm = make_llm(c, paths)
        assert isinstance(llm, DeepSeekLLM) and DS_KEY not in repr(llm)
        ds.chat_responses.append({"claims": [{"text": "x"}]})
        r = llm.chat_json("extract", [{"role": "user", "content": "json please"}], model=c.extract_model,
                          max_tokens=123)
        assert r.data == {"claims": [{"text": "x"}]}
        path, body, had_auth = ds.requests[-1]
        assert path.endswith("/chat/completions") and had_auth
        assert body["model"] == "deepseek-v4-flash"                       # gpt-4.1-mini mapped
        assert body["thinking"] == {"type": "disabled"}
        assert body["max_tokens"] == 123 and "max_completion_tokens" not in body
        assert body["response_format"] == {"type": "json_object"}
        llm.chat_json("judge", [{"role": "user", "content": "json"}], model="deepseek-v4-pro")
        assert ds.requests[-1][1]["model"] == "deepseek-v4-pro"           # explicit deepseek name kept
        llm.ping(c.extract_model)
        assert ds.requests[-1][1]["max_tokens"] == 1 and "response_format" not in ds.requests[-1][1]
        assert oa.requests == []                                         # the DeepSeek key went nowhere else


def test_deepseek_json_mode_fallback_and_auth(paths, cfg):
    with FakeOpenAIServer() as ds:
        _env(paths, f"DEEPSEEK_API_KEY={DS_KEY}\nDEEPSEEK_BASE_URL={ds.base_url}\n")
        llm = make_llm(cfg, paths)
        ds.push_status("chat", 400)
        ds.chat_responses.append('Sure! {"claims": []} done')
        r = llm.chat_json("extract", [{"role": "user", "content": "x"}])
        assert r.data == {"claims": []} and llm.json_mode_ok is False
        assert "response_format" in ds.requests[0][1] and "response_format" not in ds.requests[1][1]
        ds.push_status("chat", 401)
        with pytest.raises(LLMAuthError, match="DEEPSEEK_API_KEY"):
            llm.chat_json("judge", [{"role": "user", "content": "x"}])
    _env(paths, "OTHER=1\n")
    nokey = make_llm(cfg.replace(llm_provider="deepseek"), paths)
    with pytest.raises(LLMAuthError, match="DEEPSEEK_API_KEY 없음"):
        nokey.ping()


def test_openai_unchanged(paths, cfg):
    with FakeOpenAIServer() as oa:
        llm = make_llm(cfg.replace(llm_base_url=oa.base_url), paths)
        assert type(llm) is OpenAILLM
        llm.chat_json("extract", [{"role": "user", "content": "json"}], model="gpt-4.1-mini", max_tokens=9)
        body = oa.requests[-1][1]
        assert body["model"] == "gpt-4.1-mini" and body["max_completion_tokens"] == 9
        assert "thinking" not in body


# ── doctor: provider names and key *sources*, never values ───────────────────

def test_doctor_shows_resolved_providers_without_values(paths, capsys):
    _env(paths, f"DEEPSEEK_API_KEY={DS_KEY}\n")
    assert cli.main(["--hermes-home", str(paths.hermes_home), "init"]) == 0
    capsys.readouterr()
    cli.main(["--hermes-home", str(paths.hermes_home), "doctor", "--offline", "--json"])
    out = capsys.readouterr().out
    assert DS_KEY not in out
    checks = {c["name"]: c for c in json.loads(out)["checks"]}
    assert checks["임베딩"]["level"] == "OK" and "hash/ngram-v1@1024" in checks["임베딩"]["detail"]
    llm = checks["LLM"]
    assert llm["level"] == "OK"
    assert "deepseek deepseek-v4-flash (llm_provider=auto)" in llm["detail"]
    assert "DEEPSEEK_API_KEY ($HERMES_HOME/.env)" in llm["detail"]
    _env(paths, "OTHER=1\n")
    cli.main(["--hermes-home", str(paths.hermes_home), "doctor", "--offline", "--json"])
    checks = {c["name"]: c for c in json.loads(capsys.readouterr().out)["checks"]}
    assert checks["LLM"]["level"] == "FAIL" and "OPENAI_API_KEY 없음" in checks["LLM"]["detail"]


def test_example_and_defaults_list_the_new_keys():
    ex = json.loads((REPO / "config.json.example").read_text(encoding="utf-8"))
    for k in ("llm_provider", "deepseek_model", "embed_provider"):
        assert ex[k] == DEFAULTS[k]
    assert DEFAULTS["embed_provider"] == "auto" and DEFAULTS["llm_provider"] == "auto"
    assert DEFAULTS["deepseek_model"] == "deepseek-v4-flash"
    from hermesyume.config import PROVIDER_KEYS, _COS_KEYS, hash_embed
    he = hash_embed()
    assert set(he.MODEL_DEFAULTS) <= set(_COS_KEYS) <= set(DEFAULTS)
    assert validate(Config({"embed_provider": "hash"}).as_dict()) == []
    for k in set(he.MODEL_DEFAULTS) & PROVIDER_KEYS:                # provider resolves the same values
        assert Config({"embed_provider": "hash"})[k] == he.MODEL_DEFAULTS[k]
