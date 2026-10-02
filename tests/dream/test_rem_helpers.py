"""Shared helpers for the REM builder's tests (no tests here).

Cross-builder modules (prompts, normalize, sources.core_files) are used when importable; when a
module is missing a minimal contract-shaped stub is installed for the test only.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import sys
import types
import unicodedata
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import hermesyume
from hermesyume.embedder import embed_input
from hermesyume.types import (KIND_BASE, Claim, LiveSnapshot, MemoryRow, RunBudget, RunReport,
                              RunStats, new_memory_id)
from tests.fakes import ScriptedLLM, make_row, vector_with_cos


# ── stubs for not-yet-present cross-builder modules ──────────────────────────

def _stub_prompts() -> types.ModuleType:
    m = types.ModuleType("hermesyume.prompts")
    m.PROMPT_KINDS = ("extract", "extract_retry", "judge", "judge_enum", "consolidate", "core_classify", "ping")
    m.judge_messages = lambda new, cands: [{"role": "user", "content": json.dumps({"new": new, "candidates": cands}, ensure_ascii=False)}]
    m.judge_enum_messages = lambda new, cand: [{"role": "user", "content": json.dumps({"new": new, "existing": cand}, ensure_ascii=False)}]
    m.consolidate_messages = lambda a, b, *, max_chars: [{"role": "user", "content": f"{max_chars}\n{a}\n{b}"}]
    m.core_classify_messages = lambda entries: [{"role": "user", "content": json.dumps(entries, ensure_ascii=False)}]
    return m


def _subject_key(s: str) -> str:
    t = unicodedata.normalize("NFKC", s or "").casefold()
    return "".join(ch for ch in t if not ch.isspace() and unicodedata.category(ch)[0] not in "PS")


def _stub_normalize() -> types.ModuleType:
    m = types.ModuleType("hermesyume.normalize")

    def compute_importance(*, kind, level, explicit_user, user_session_count, assistant_only, source):
        v = (KIND_BASE.get(kind, 0.5) + 0.08 * (int(level) - 3) + 0.12 * bool(explicit_user)
             + 0.05 * max(0, min(int(user_session_count) - 1, 3)) - 0.10 * bool(assistant_only))
        v = min(1.0, max(0.05, v))
        if (source or "").startswith("core:"):
            v = max(v, 0.85)
        elif source == "tool:yume_remember":
            v = max(v, 0.80)
        return round(v, 4)

    def normalize_claim(claim, *, cfg, paths):
        claim.subject_key = _subject_key(claim.subject) or _subject_key(claim.text[:30])
        claim.importance = max(float(claim.importance or 0.0), compute_importance(
            kind=claim.kind, level=claim.level, explicit_user=claim.explicit_user,
            user_session_count=claim.user_session_count, assistant_only=claim.assistant_only,
            source=claim.source))
        return claim

    m.subject_key = _subject_key
    m.compute_importance = compute_importance
    m.normalize_claim = normalize_claim
    m.verify_refs = lambda text, *, paths, workspace_dir: []
    return m


def _stub_core_files() -> types.ModuleType:
    from hermesyume.paths import load_provider_module
    from hermesyume.types import CoreEntry
    m = types.ModuleType("hermesyume.sources.core_files")
    cf = load_provider_module("corefmt")

    def read_core(paths):
        out = {}
        for t in ("memory", "user"):
            out[t] = [CoreEntry(t, i, e, cf.core_sha(e), cf.entry_label(e))
                      for i, e in enumerate(cf.read_entries(paths.core_file(t)))]
        return out

    m.read_core = read_core
    m.load_limits = lambda paths: {"memory": {"enabled": True, "limit": 2200},
                                   "user": {"enabled": True, "limit": 1375}}
    return m


_STUBS = {"hermesyume.prompts": _stub_prompts, "hermesyume.normalize": _stub_normalize,
          "hermesyume.sources.core_files": _stub_core_files}


def install_missing(monkeypatch, *names: str) -> None:
    for name in names or tuple(_STUBS):
        try:
            found = importlib.util.find_spec(name) is not None
        except ModuleNotFoundError:
            found = False
        if found:
            continue
        mod = _STUBS[name]()
        monkeypatch.setitem(sys.modules, name, mod)
        parent_name, _, attr = name.rpartition(".")
        monkeypatch.setattr(sys.modules[parent_name], attr, mod, raising=False)


def install_module(monkeypatch, name: str, mod: types.ModuleType) -> None:
    """Force a (stub) module for one test, e.g. hermesyume.export / dream_log / alerts / nrem."""
    monkeypatch.setitem(sys.modules, name, mod)
    parent_name, _, attr = name.rpartition(".")
    monkeypatch.setattr(sys.modules[parent_name], attr, mod, raising=False)


@pytest.fixture
def deps(monkeypatch):
    install_missing(monkeypatch)
    return True


def skey(s: str) -> str:
    try:
        from hermesyume.normalize import subject_key
        return subject_key(s)
    except ImportError:
        return _subject_key(s)


# ── builders ─────────────────────────────────────────────────────────────────

def unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    return a / float(np.linalg.norm(a))


def vec_like(anchor, cos: float, seed: str) -> np.ndarray:
    return unit(vector_with_cos([float(x) for x in unit(anchor)], cos, seed))


def row(ctx, text: str, *, kind: str = "fact", subject: str | None = None, now: float | None = None,
        **kw) -> MemoryRow:
    subject = subject or text[:12]
    t = ctx.now if now is None else now
    r = make_row(text, embedder=ctx.embedder, subject=subject, now=t, kind=kind,
                 subject_key=kw.pop("subject_key", skey(subject)), event_time=kw.pop("event_time", t),
                 **kw)
    from hermesyume.strength import compute_tier
    r.tier = compute_tier(r)
    return r


def commit(ctx, *rows: MemoryRow) -> None:
    ctx.store.commit(upserts=list(rows))


_N = [0]


def claim(ctx, text: str, *, kind: str = "fact", subject: str | None = None, origin: str | None = None,
          user: bool = True, explicit: bool = False, et: float | None = None, vector=None,
          keys: list[str] | None = None, sessions: list[str] | None = None, source: str = "dream",
          roles: list[str] | None = None, importance: float | None = None, steps: int | None = None,
          window_id: str | None = None, status: str = "active", valid_until: float | None = None,
          pin: bool = False) -> Claim:
    _N[0] += 1
    n = _N[0]
    subject = subject or text[:12]
    et = ctx.now - 3600 if et is None else et
    roles = roles or (["user"] if user else ["assistant"])
    has_user = "user" in roles
    c = Claim(origin_key=origin or f"w{n}#0", source=source, kind=kind, target="user", subject=subject,
              text=text, level=3, explicit=explicit, steps=steps, event_time=et, valid_until=valid_until,
              status=status, window_id=window_id, evidence_refs=[f"U#{n}"],
              evidence_keys=keys if keys is not None else [f"s:{1000 + n}"],
              evidence_roles=roles, session_ids=sessions or [f"sess{n}"], first_seen_at=et,
              last_seen_at=et, last_user_evidence_at=et if has_user else None,
              user_evidence_count=1 if has_user else 0, user_session_count=1 if has_user else 0,
              explicit_user=bool(explicit and has_user), subject_key=skey(subject), pin=pin)
    imp = KIND_BASE.get(kind, 0.5) + (0.12 if c.explicit_user else 0.0) - (0.10 if c.assistant_only else 0.0)
    c.importance = round(importance if importance is not None else imp, 4)
    c.vector = unit(vector) if vector is not None else ctx.embedder.vector(embed_input(subject, text))
    return c


def next_ctx(ctx, *, days: float = 0.0, run_id: str | None = None, llm=None, now: float | None = None):
    """A fresh RunContext for "the next night" sharing the open handles."""
    t = (ctx.now + days * 86400.0) if now is None else now
    rid = run_id or f"run-{int(t)}-{new_memory_id()[:4]}"
    from hermesyume import clock
    clock.set_now(t)
    budget = RunBudget(max_llm_calls=int(ctx.cfg.max_llm_calls), max_embed_inputs=int(ctx.cfg.max_embed_inputs),
                       max_runtime_s=float(ctx.cfg.max_runtime_min) * 60)
    llm = llm or ScriptedLLM(budget=budget)
    llm.budget = budget
    ctx.embedder.budget = budget
    return dataclasses.replace(ctx, run_id=rid, now=t, llm=llm, budget=budget,
                               stats=RunStats(run_id=rid), report=RunReport(), alerts=[])


def nres(claims=(), *, windows=(), window_states=None, **kw) -> SimpleNamespace:
    return SimpleNamespace(claims=list(claims), windows=list(windows), window_states=dict(window_states or {}),
                           rejections=[], wm_delta=dict(kw.get("wm_delta", {})),
                           session_roots=dict(kw.get("session_roots", {})), md_states=list(kw.get("md_states", [])),
                           deferred_windows=0, inbox_episodic_ids=[], sanitize_report=None)


def pre(ctx) -> SimpleNamespace:
    snap = ctx.live.snapshot() if ctx.live is not None else LiveSnapshot(0, 0)
    return SimpleNamespace(snapshot=snap, lance_version_before=ctx.store.version(), replayed_runs=[],
                           ledger_backup=None)


def rows_by_text(ctx) -> dict[str, MemoryRow]:
    return {r.text: r for r in ctx.store.load_working_set().values()}


def live_insert(ctx, table: str, **fields) -> int:
    cols = ", ".join(fields)
    qs = ", ".join("?" * len(fields))
    cur = ctx.live.conn.execute(f"INSERT INTO {table}({cols}) VALUES({qs})", tuple(fields.values()))
    ctx.live.conn.commit()
    return int(cur.lastrowid)


def no_core_writes(monkeypatch, paths) -> list[str]:
    """T14 harness: any write/replace/rename/flock touching MEMORY.md/USER.md (or their .lock)
    raises. Returns the list of violations (also raised)."""
    import builtins
    import fcntl
    import io
    import os
    from pathlib import Path

    core = set()
    for t in ("memory", "user"):
        p = paths.core_file(t)
        for q in (p, Path(str(p) + ".lock")):
            core.add(os.path.abspath(str(q)))
            core.add(os.path.realpath(str(q)))
    hits: list[str] = []

    def is_core(p) -> bool:
        try:
            s = os.fspath(p)
        except TypeError:
            return False
        if isinstance(s, bytes):
            s = s.decode()
        return os.path.abspath(s) in core or os.path.realpath(s) in core

    real_open, real_os_open = builtins.open, os.open
    real_replace, real_rename = os.replace, os.rename

    def guarded_open(file, mode="r", *a, **k):
        if is_core(file) and any(c in mode for c in "wax+"):
            hits.append(f"open({file}, {mode})")
            raise AssertionError(f"core file write: {file}")
        return real_open(file, mode, *a, **k)

    def guarded_os_open(path, flags, *a, **k):
        if is_core(path) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC):
            hits.append(f"os.open({path})")
            raise AssertionError(f"core file os.open: {path}")
        return real_os_open(path, flags, *a, **k)

    def guarded_replace(src, dst, *a, **k):
        if is_core(dst) or is_core(src):
            hits.append(f"replace({src}, {dst})")
            raise AssertionError("core file replace")
        return real_replace(src, dst, *a, **k)

    def guarded_rename(src, dst, *a, **k):
        if is_core(dst) or is_core(src):
            hits.append(f"rename({src}, {dst})")
            raise AssertionError("core file rename")
        return real_rename(src, dst, *a, **k)

    real_flock, real_lockf = fcntl.flock, fcntl.lockf

    def fd_path(fd) -> str:
        n = fd if isinstance(fd, int) else fd.fileno()
        try:
            return os.readlink(f"/proc/self/fd/{n}")
        except OSError:
            return ""

    def guarded_flock(fd, op, *a, **k):
        if is_core(fd_path(fd)):
            hits.append(f"flock({fd_path(fd)})")
            raise AssertionError("flock on a core file during dream")
        return real_flock(fd, op, *a, **k)

    def guarded_lockf(fd, op, *a, **k):
        if is_core(fd_path(fd)):
            hits.append(f"lockf({fd_path(fd)})")
            raise AssertionError("lockf on a core file during dream")
        return real_lockf(fd, op, *a, **k)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    monkeypatch.setattr(os, "replace", guarded_replace)
    monkeypatch.setattr(os, "rename", guarded_rename)
    monkeypatch.setattr(fcntl, "flock", guarded_flock)
    monkeypatch.setattr(fcntl, "lockf", guarded_lockf)
    return hits


def classify_all(kind_for=lambda text: "fact"):
    """core_classify handler: every entry gets kind_for(text) and a short subject."""
    def fn(messages):
        content = messages[-1]["content"]
        try:
            data = json.loads(content)
            entries = data["items"] if isinstance(data, dict) else data
            texts = [e["text"] if isinstance(e, dict) else e for e in entries]
        except (json.JSONDecodeError, KeyError, TypeError):
            texts = []
        items = []
        for i, t in enumerate(texts):
            m = re.match(r"\*\*([^*]+)\*\*", t)
            items.append({"i": i, "kind": kind_for(t), "subject": (m.group(1).strip(": ") if m else t[:20]),
                          "fragment": False})
        return {"items": items}
    return fn
