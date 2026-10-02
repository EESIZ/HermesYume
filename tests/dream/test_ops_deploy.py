"""deploy/* and config.json.example (PLAN §8.2), plus the ready-made ops CLI handlers."""

from __future__ import annotations

import argparse
import configparser
import functools
import json
import os
import subprocess
from pathlib import Path

import pytest

from hermesyume.config import DEFAULTS, validate

REPO = Path(__file__).resolve().parents[2]
PATH_KEYS = {"workspace_dir", "md_sources", "hermes_runtime_dir"}


def test_config_example_mirrors_defaults():
    ex = json.loads((REPO / "config.json.example").read_text(encoding="utf-8"))
    assert set(ex) == set(DEFAULTS)
    for k, v in DEFAULTS.items():
        if k not in PATH_KEYS:
            assert ex[k] == v, k
    assert validate(ex) == []
    assert ex["alert_telegram"] is False and ex["inject"] is True and ex["recall_min_cos"] >= 0.40
    raw = (REPO / "config.json.example").read_text(encoding="utf-8")
    from hermesyume.threat import find_secrets
    assert "/home/" not in raw and not find_secrets(raw)


def test_public_repo_carries_no_machine_specific_values():
    """Public repository: defaults and shipped files name no user's home, host or private network."""
    import re
    flat = json.dumps(DEFAULTS, ensure_ascii=False)
    assert '"/' not in flat                                   # only ~-relative or empty paths
    for k in ("workspace_dir", "hermes_runtime_dir", "exclude_first_message_regex"):
        assert DEFAULTS[k] == "", k
    for k in ("md_sources", "md_exclude_globs", "strip_line_regex", "deny_cwd_globs", "protect_homes",
              "protect_workspaces"):
        assert DEFAULTS[k] == [], k
    home_re = re.compile(r"/home/(?!YOUR_USER\b|user\b)[a-z_][a-z0-9_-]*")
    cgnat_re = re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b")   # Tailscale range
    shipped = [REPO / "install.sh", REPO / "config.json.example", REPO / "README.md", REPO / "README.ko.md",
               REPO / "DESIGN.md", REPO / "pyproject.toml", *(REPO / "deploy").iterdir(),
               *(REPO / "hermesyume").rglob("*.py"), *(REPO / "provider").rglob("*.py"),
               *(REPO / "provider").rglob("*.yaml"), *(REPO / "tests").rglob("*.py")]
    for f in shipped:
        if not f.is_file():
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        assert not home_re.search(text), f
        assert not cgnat_re.search(text), f
        assert "docs-" + "local/" not in text, f            # private planning notes stay private


def _denylist_patterns(path: Path) -> list:
    import re
    pats = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            pats.append(None)                                 # keep line numbers for the report
        elif t.isdigit():
            pats.append(re.compile(rf"(?<!\d){re.escape(t)}(?!\d)"))
        elif t.isascii():
            pats.append(re.compile(rf"(?<![A-Za-z0-9_]){re.escape(t)}(?![A-Za-z0-9_])", re.I))
        else:
            pats.append(re.compile(re.escape(t)))
    return pats


def test_public_repo_carries_no_private_terms():
    """Opt-in on the owner's machine: no term of a private, untracked deny-list (one per line, '#'
    comments; privacy-denylist.txt in the gitignored private notes folder, or the file named by
    $HERMESYUME_PRIVACY_DENYLIST) appears in a shipped file. The list never ships, and a failure
    names only the file and the list line."""
    src = os.environ.get("HERMESYUME_PRIVACY_DENYLIST") or str(REPO / ("docs-" + "local") / "privacy-denylist.txt")
    if not Path(src).is_file():
        pytest.skip("no local privacy deny-list")
    pats = _denylist_patterns(Path(src))
    files = [REPO / n for n in ("install.sh", "config.json.example", "README.md", "README.ko.md", "DESIGN.md",
                                "pyproject.toml", ".gitignore", "LICENSE", "requirements-dream.in")]
    files += [*(REPO / "deploy").iterdir(), *(REPO / "provider").rglob("*.yaml")]
    for top in ("hermesyume", "provider", "tests"):
        files += [p for p in (REPO / top).rglob("*") if p.suffix in (".py", ".md", ".json", ".sh", ".txt")
                  and "__pycache__" not in p.parts]
    hits = []
    for f in files:
        if not f.is_file():
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        hits += [(str(f.relative_to(REPO)), n) for n, p in enumerate(pats, start=1) if p and p.search(text)]
    assert hits == [], f"private deny-list hits (file, list line): {hits}"


def _unit(name):
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.optionxform = str
    cp.read(REPO / "deploy" / name, encoding="utf-8")
    return cp


def test_systemd_units():
    svc = _unit("hermesyume-dream.service")["Service"]
    assert svc["Type"] == "oneshot"
    assert svc["ExecStart"].endswith("/venv/bin/yume dream")
    assert svc["Environment"].startswith("HERMES_HOME=")
    assert svc["UMask"] == "0077" and svc["Nice"] == "10" and svc["TimeoutStartSec"] == "1h"
    assert "EnvironmentFile" not in svc
    tm = _unit("hermesyume-dream.timer")
    assert tm["Timer"]["OnCalendar"] == "*-*-* 04:40:00 Asia/Seoul"
    assert tm["Timer"]["Persistent"] == "true"
    assert tm["Install"]["WantedBy"] == "timers.target"


@pytest.mark.parametrize("script", ["setup_venv.sh", "install_provider.sh"])
def test_scripts_parse_and_are_executable(script):
    p = REPO / "deploy" / script
    assert os.access(p, os.X_OK)
    subprocess.run(["bash", "-n", str(p)], check=True)


def _install(home, *extra):
    env = {k: v for k, v in os.environ.items() if k != "HERMES_HOME"}
    return subprocess.run(["bash", str(REPO / "deploy" / "install_provider.sh"), "--hermes-home", str(home), *extra],
                          capture_output=True, text=True, env=env)


def test_install_provider_atomic_swap_and_prev(tmp_path):
    home = tmp_path / "h"
    home.mkdir()
    (home / "config.yaml").write_text("memory:\n  provider: builtin\n", encoding="utf-8")
    cfg_before = (home / "config.yaml").read_bytes()
    r1 = _install(home)
    assert r1.returncode == 0, r1.stderr
    dest = home / "plugins" / "hermesyume"
    assert (dest / "__init__.py").exists() and (dest / "_yume" / "live_schema.py").exists()
    assert "hermesyume-provider" in (dest / "VERSION").read_text()
    assert not list(dest.rglob("__pycache__"))
    assert oct((home / "hermesyume" / "staging").stat().st_mode & 0o777) == "0o700"
    (dest / "MARK").write_text("old")
    r2 = _install(home)
    assert r2.returncode == 0, r2.stderr
    assert not (dest / "MARK").exists()
    assert (home / "hermesyume" / "staging" / "provider.prev" / "MARK").exists()
    assert (home / "config.yaml").read_bytes() == cfg_before      # never touched


def test_install_provider_refuses_live_home_without_yes(tmp_path):
    home = tmp_path / "h"
    home.mkdir()
    (home / "gateway.pid").write_text("1")
    r = _install(home)
    assert r.returncode != 0 and "--yes" in r.stderr
    assert not (home / "plugins").exists()
    r = _install(home, "--dry-run", "--yes")
    assert r.returncode == 0 and "[dry-run]" in r.stdout
    assert not (home / "plugins" / "hermesyume").exists()


def test_setup_venv_refuses_hermes_venv(tmp_path):
    fake = tmp_path / "venv"
    (fake / "bin").mkdir(parents=True)
    (fake / "bin" / "hermes").write_text("#!/bin/sh\n")
    r = subprocess.run(["bash", str(REPO / "deploy" / "setup_venv.sh"), "--venv", str(fake), "--dry-run",
                        "--from-worktree"], capture_output=True, text=True)
    assert r.returncode != 0 and "Hermes" in r.stderr


# ── ready-made CLI handlers (integrator wires them into cli.HANDLERS) ───────────

def _args(**kw):
    base = dict(dry_run=False, approve_migration=False, estimate=False, only=None, max_llm_calls=None,
                statedb_start=None, json=True)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def offline_migrate(monkeypatch):
    from hermesyume import migrate
    monkeypatch.setattr(migrate, "build_context", functools.partial(migrate.build_context, offline=True))
    return migrate


def test_migrate_cli_modes(initialized, paths, fake_home, offline_migrate, capsys):
    from tests.fixtures.hermes_home import tree_hash
    m = offline_migrate
    assert m.cli_handler(_args(), paths) == 1                                    # a mode is required
    assert m.cli_handler(_args(dry_run=True, approve_migration=True), paths) == 1
    capsys.readouterr()
    assert m.cli_handler(_args(estimate=True), paths) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "estimate" and out["estimate"]["by_step"]["core"]["entries"] == 21
    before = tree_hash(fake_home.root)
    assert m.cli_handler(_args(dry_run=True, only="core,memory_md"), paths) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "dry" and out["core_map"]["accounted"] == 35
    assert tree_hash(fake_home.root) == before
    assert m.cli_handler(_args(approve_migration=True, only="core,memory_md"), paths) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "committed" and Path(out["core_map_path"]).exists()


def test_calibrate_and_alert_flush_handlers(initialized, paths, capsys):
    from hermesyume import alerts, calibrate
    assert calibrate.cli_handler(argparse.Namespace(json=True), paths) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["recall_min_cos"] >= 0.40
    assert alerts.cli_flush(argparse.Namespace(), paths) == 0
    assert "alert_telegram" in capsys.readouterr().out
