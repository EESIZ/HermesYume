"""sources/core_files.py — read-only MEMORY.md / USER.md (CONTRACTS §4.3)."""

from __future__ import annotations

import builtins
import os

import pytest

from hermesyume.paths import load_provider_module
from hermesyume.sources import core_files
from hermesyume.sources.core_files import entry_sha, load_limits, read_core
from tests.fixtures.hermes_home import MEMORY_ENTRIES, USER_ENTRIES, tree_hash


def test_read_core_fixture(paths, fake_home):
    before = tree_hash(fake_home.memories)
    core = read_core(paths)
    assert [e.text for e in core["user"]] == USER_ENTRIES
    assert [e.text for e in core["memory"]] == MEMORY_ENTRIES
    assert len(core["user"]) + len(core["memory"]) == 35
    fmt = load_provider_module("corefmt")
    inv = core["user"][4]
    assert inv.target == "user" and inv.index == 4
    assert inv.label == "**가계부 관리:**" and inv.sha == fmt.core_sha(inv.text)
    assert entry_sha(inv.text) == inv.sha and len(inv.sha) == 40
    assert core["memory"][0].label is None
    assert tree_hash(fake_home.memories) == before                 # nothing written
    assert not any(n.endswith(".lock") for n in os.listdir(fake_home.memories))


def test_read_core_opens_read_only(paths, monkeypatch):
    real_open = builtins.open
    modes = []

    def spy(file, mode="r", *a, **kw):
        if str(file).endswith((".md",)):
            modes.append(mode)
        return real_open(file, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", spy)
    read_core(paths)
    assert modes and all(m == "r" for m in modes)


def test_missing_and_bom(paths, fake_home):
    fake_home.memory_md.unlink()
    fake_home.user_md.write_bytes("﻿**이름:** 테스트\n§\n둘째 항목".encode("utf-8"))
    core = read_core(paths)
    assert core["memory"] == []
    assert [e.text for e in core["user"]] == ["**이름:** 테스트", "둘째 항목"]


def test_undecodable_core_file_raises(paths, fake_home):
    fake_home.user_md.write_bytes(b"\xff\xfe\xfa not utf8")
    with pytest.raises(UnicodeDecodeError):
        read_core(paths)


def test_load_limits_from_config_yaml(paths, fake_home):
    assert load_limits(paths) == {"memory": {"enabled": True, "limit": 2200},
                                  "user": {"enabled": True, "limit": 1375}}
    (fake_home.root / "config.yaml").write_text(
        "memory:\n  memory_enabled: false\n  user_char_limit: 999\n", encoding="utf-8")
    assert load_limits(paths) == {"memory": {"enabled": False, "limit": 2200},
                                  "user": {"enabled": True, "limit": 999}}


def test_load_limits_defaults_and_fallback_parser(paths, fake_home, monkeypatch):
    (fake_home.root / "config.yaml").unlink()
    assert load_limits(paths)["user"]["limit"] == 1375
    (fake_home.root / "config.yaml").write_text(
        "model:\n  default: x\nmemory:\n  memory_char_limit: 1800  # comment\n"
        "  user_profile_enabled: 'no'\nother: 1\n", encoding="utf-8")
    real_import = builtins.__import__

    def no_yaml(name, *a, **kw):
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_yaml)
    assert load_limits(paths) == {"memory": {"enabled": True, "limit": 1800},
                                  "user": {"enabled": False, "limit": 1375}}


def test_load_limits_malformed_yaml(paths, fake_home):
    (fake_home.root / "config.yaml").write_text("memory: [unclosed\n", encoding="utf-8")
    assert load_limits(paths)["memory"] == {"enabled": True, "limit": 2200}


def test_corefmt_loader():
    assert core_files.corefmt().ENTRY_DELIMITER == "\n§\n"
