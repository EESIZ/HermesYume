"""Shared pytest fixtures for dream-side tests (PLAN §11.2: `pytest -q tests/dream`).

Isolation (autouse): HERMES_HOME → tmp, no HERMESYUME_NOW / OPENAI_* / protect-list / runtime-dir
variables from the caller's shell, clock override cleared after each test. Nothing here reads or
writes a real Hermes home. A test that needs a real home's path reads $HERMESYUME_TEST_LIVE_HOME
(captured below, before isolation) and skips when it is unset.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from hermesyume import clock, threat
from hermesyume.config import Config, save_config
from hermesyume.ledger import Ledger
from hermesyume.livedb import LiveDB
from hermesyume.paths import Paths
from hermesyume.store import Store
from hermesyume.types import RunBudget, RunContext, RunReport, RunStats
from tests.fakes import FakeEmbedder, ScriptedLLM
from tests.fixtures.hermes_home import TEST_FILTERS, make_hermes_home
from tests.fixtures.statedb import build_basic

# Fixed reference time: 2026-10-02 04:40 KST (the nightly slot).
NOW = clock.parse_iso("2026-10-02T04:40")
TEST_LIVE_HOME = os.environ.get("HERMESYUME_TEST_LIVE_HOME", "").strip()
LIVE_HOMES = tuple(p for p in (os.path.expanduser("~/.hermes"), TEST_LIVE_HOME,
                               *os.environ.get("HERMESYUME_PROTECT_HOMES", "").split(os.pathsep),
                               *os.environ.get("HERMESYUME_LIVE_HOMES", "").split(os.pathsep)) if p.strip())
ISOLATED_ENV = ("HERMESYUME_NOW", "OPENAI_API_KEY", "OPENAI_BASE_URL", "HERMESYUME_PROVIDER_DIR",
                "HERMESYUME_LIVE_HOMES", "HERMESYUME_PROTECT_HOMES", "HERMESYUME_LIVE_WORKSPACES",
                "HERMESYUME_PROTECT_WORKSPACES", "HERMES_RUNTIME_DIR")


@pytest.fixture(autouse=True)
def _isolation(monkeypatch, tmp_path):
    for k in ISOLATED_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    clock.set_now(None)
    yield
    clock.set_now(None)


@pytest.fixture
def now() -> float:
    return NOW


@pytest.fixture
def fake_home(tmp_path):
    home = make_hermes_home(tmp_path)
    assert str(home.root) not in LIVE_HOMES
    return home


@pytest.fixture
def paths(fake_home) -> Paths:
    return Paths.from_env(fake_home.root)


@pytest.fixture
def cfg(fake_home, tmp_path) -> Config:
    """Defaults with workspace/md paths inside tmp, no runtime dir (threat → vendored copy) and the
    fixture input filters (TEST_FILTERS)."""
    return Config({
        "workspace_dir": str(fake_home.workspace),
        "md_sources": [str(fake_home.md_dir)],
        "hermes_runtime_dir": str(tmp_path / "no-runtime"),
        **TEST_FILTERS,
    })


@pytest.fixture
def statedb(fake_home, now):
    """Basic §3.1 scenario in <HERMES_HOME>/state.db; returns the ids dict of build_basic()."""
    return build_basic(fake_home.state_db, now)


@pytest.fixture
def fake_embedder(cfg) -> FakeEmbedder:
    return FakeEmbedder(dim=int(cfg.embed_dim), model_id=cfg.embed_model_id())


@pytest.fixture
def scripted_llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def scanner():
    return threat.load_scanner(None)   # vendored, sha-pinned copy


@pytest.fixture
def initialized(paths, cfg):
    """`yume init` equivalent built from foundation pieces."""
    paths.ensure_data_dirs()
    save_config(paths, cfg.as_dict())
    led = Ledger.from_paths(paths)
    led.init_meta(cfg.embed_model_id(), int(cfg.embed_dim))
    led.close()
    Store.from_config(paths, cfg, create=True)
    LiveDB.open(paths, mode="rw").close()
    return SimpleNamespace(paths=paths, cfg=cfg)


@pytest.fixture
def ctx(initialized, paths, cfg, now, fake_embedder, scripted_llm, scanner):
    """A live-mode RunContext with every handle open (closed on teardown)."""
    clock.set_now(now)
    budget = RunBudget(max_llm_calls=int(cfg.max_llm_calls), max_embed_inputs=int(cfg.max_embed_inputs),
                       max_runtime_s=float(cfg.max_runtime_min) * 60)
    fake_embedder.budget = budget
    scripted_llm.budget = budget
    c = RunContext(paths=paths, cfg=cfg, run_id="20261002-044000-test", now=now, mode="live",
                   llm=scripted_llm, embedder=fake_embedder, scanner=scanner,
                   store=Store.from_config(paths, cfg), ledger=Ledger.from_paths(paths),
                   live=LiveDB.open(paths, mode="rw"), budget=budget, stats=RunStats(run_id="20261002-044000-test"),
                   report=RunReport())
    yield c
    c.ledger.close()
    c.live.close()
