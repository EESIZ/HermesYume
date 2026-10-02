"""clock.py (--now / HERMESYUME_NOW / KST helpers) and paths.py (layout, live guard, lock, loader)."""

import os

import pytest

from hermesyume import clock
from hermesyume.paths import (AlreadyRunning, LiveHomeRefused, Paths, dream_lock,
                              find_provider_file, is_live_home, load_provider_module,
                              refuse_if_live)


def test_parse_specs():
    base = 1_000_000.0
    assert clock.parse_now_spec("+70d", base) == base + 70 * 86400
    assert clock.parse_now_spec("-2h", base) == base - 7200
    assert clock.parse_now_spec("+5y", base) == base + 5 * 365 * 86400
    ts = clock.parse_now_spec("2026-10-02T04:40")
    assert clock.fmt_kst(ts, "%Y-%m-%d %H:%M") == "2026-10-02 04:40"     # naive → KST
    assert clock.parse_now_spec("2026-10-01T19:40:00+00:00") == ts
    assert clock.parse_now_spec("1790000000") == 1790000000.0
    with pytest.raises(ValueError):
        clock.parse_now_spec("tomorrow")


def test_set_now_freezes_and_env(monkeypatch):
    clock.set_now("2026-10-02")
    a = clock.now()
    assert a == clock.now() and clock.override_active()
    clock.set_now(None)
    monkeypatch.setenv("HERMESYUME_NOW", "+10d")
    x = clock.now()
    assert x == clock.now()                                   # frozen once resolved
    assert abs(x - (clock.real_now() + 10 * 86400)) < 5
    monkeypatch.delenv("HERMESYUME_NOW")
    assert abs(clock.now() - clock.real_now()) < 1


def test_kst_helpers():
    ts = clock.parse_iso("2026-10-02T04:40")
    assert clock.kst_date(ts) == "2026-10-02"
    assert clock.kst_date_ko(ts) == "2026-10-02(금)"
    assert clock.kst_stamp(ts) == "2026-10-02_044000"
    assert clock.parse_iso("2026-10-10", end_of_day=True) == clock.parse_iso("2026-10-10T23:59:59")
    assert clock.parse_iso("null") is None and clock.parse_iso("garbage") is None
    assert clock.from_ms(clock.to_ms(ts)) == ts
    assert clock.kst_midnight(ts) == clock.parse_iso("2026-10-02")


def test_paths_layout(tmp_path):
    p = Paths.from_env(tmp_path / "h")
    assert p.data_dir == tmp_path / "h" / "hermesyume"
    assert p.recall_sqlite == p.data_dir / "serving" / "recall.sqlite"
    assert p.serving_tmp("r1").name == "recall.r1.sqlite"
    assert p.plan_json("r1") == p.data_dir / "runs" / "r1" / "plan.json"
    assert p.core_file("user").name == "USER.md" and p.core_file("memory").name == "MEMORY.md"
    assert p.provider_dir == tmp_path / "h" / "plugins" / "hermesyume"
    p.ensure_data_dirs()
    assert oct(os.stat(p.data_dir).st_mode & 0o777) == "0o700"


def test_env_resolution(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "x"))
    assert Paths.from_env().hermes_home == tmp_path / "x"
    monkeypatch.delenv("HERMES_HOME")
    assert Paths.from_env().hermes_home == Paths.from_env("~/.hermes").hermes_home


def test_live_guard(tmp_path, monkeypatch):
    assert is_live_home(os.path.expanduser("~/.hermes"))          # Hermes' default home, built in
    assert not is_live_home(tmp_path)
    (tmp_path / "gateway_state.json").write_text("{}")
    assert is_live_home(tmp_path)
    with pytest.raises(LiveHomeRefused):
        refuse_if_live(tmp_path, "debug plant")
    other = tmp_path / "other"
    other.mkdir()
    assert not is_live_home(other)
    monkeypatch.setenv("HERMESYUME_PROTECT_HOMES", str(other))
    assert is_live_home(other)
    monkeypatch.delenv("HERMESYUME_PROTECT_HOMES")
    monkeypatch.setenv("HERMESYUME_LIVE_HOMES", str(other))        # older name, still read
    assert is_live_home(other)


def test_protect_lists_from_config(tmp_path, monkeypatch):
    """`protect_homes` / `protect_workspaces` in the active config.json, plus the workspace_dir a
    protected home's own config.json names (read-only)."""
    import json

    from hermesyume.paths import in_live_workspace, live_homes, workspace_write_refusal
    live, ws, extra_ws = tmp_path / "live", tmp_path / "live-ws", tmp_path / "extra-ws"
    for d in (live / "hermesyume", ws, extra_ws):
        d.mkdir(parents=True)
    (live / "hermesyume" / "config.json").write_text(json.dumps({"workspace_dir": str(ws)}))
    sandbox = tmp_path / "sandbox"
    (sandbox / "hermesyume").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(sandbox))
    assert not is_live_home(live) and not in_live_workspace(ws / "docs")
    (sandbox / "hermesyume" / "config.json").write_text(
        json.dumps({"protect_homes": [str(live)], "protect_workspaces": [str(extra_ws)]}))
    assert live.resolve() in live_homes() and is_live_home(live)
    assert in_live_workspace(ws / "docs" / "yume") and in_live_workspace(extra_ws)
    assert workspace_write_refusal(sandbox, ws) and workspace_write_refusal(live, ws) is None


def test_real_live_home_opt_in():
    """Only with $HERMESYUME_TEST_LIVE_HOME (a real agent home on this machine, never written)."""
    from tests.dream.conftest import TEST_LIVE_HOME
    if not TEST_LIVE_HOME:
        pytest.skip("HERMESYUME_TEST_LIVE_HOME not set")
    assert is_live_home(TEST_LIVE_HOME)          # a gateway has run there (LIVE_MARKERS)


def test_dream_lock_non_blocking(tmp_path):
    p = Paths.from_env(tmp_path)
    with dream_lock(p):
        with pytest.raises(AlreadyRunning):
            with dream_lock(p):
                pass
    with dream_lock(p):       # released
        pass


def test_provider_module_loader():
    f = find_provider_file("live_schema")
    assert f.name == "live_schema.py" and f.parent.name == "_yume"
    m = load_provider_module("live_schema")
    assert m is load_provider_module("live_schema")       # cached
    assert m.USER_VERSION == 1
    assert load_provider_module("corefmt").ENTRY_DELIMITER == "\n§\n"
    assert load_provider_module("serving_schema").VEC256_DIM == 256
