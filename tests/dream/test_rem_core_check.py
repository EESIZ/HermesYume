"""core_check.py — R5 mirror (hook-less adds/removes, core_seen delta), core_required note (U4: Dream
Log only), restore()/apply_proposal() (lock, limit, symlink, backup), and T14: the dream never
writes, replaces, renames or flocks MEMORY.md/USER.md (30-night simulation, v1 C1 regression)."""

import hashlib
import os
import stat
from pathlib import Path

import pytest

from hermesyume import core_check, plan as P, rem
from hermesyume.types import CoreEntry, CoreSeenRow
from tests.dream.test_rem_helpers import (classify_all, claim, commit, deps, live_insert, next_ctx,  # noqa: F401
                                          no_core_writes, nres, pre, row)
from tests.fixtures.hermes_home import USER_ENTRIES, read_core, write_core

DAY = 86400.0


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def kind_for(text: str) -> str:
    if text.startswith(("**이름", "**호칭", "**시간대", "**직업")):
        return "profile"
    if "원장" in text or "원칙" in text or "금지" in text:
        return "rule"
    return "preference"


def night(ctx, *, claims=(), days=1.0, llm_setup=None):
    c = next_ctx(ctx, days=days)
    c.llm.on("core_classify", classify_all(kind_for))
    if llm_setup:
        llm_setup(c.llm)
    res = rem.run_rem(c, nres(claims), pre(c))
    return c, res


def test_diff_core():
    e1, e2 = CoreEntry("user", 0, "a", "s1", None), CoreEntry("user", 1, "b", "s2", None)
    seen = {("user", "s1"): CoreSeenRow("user", "s1", "a", "m1", "r", "r", True),
            ("user", "s3"): CoreSeenRow("user", "s3", "c", "m3", "r", "r", True),
            ("user", "s4"): CoreSeenRow("user", "s4", "d", "m4", "r", "r", False)}
    adds, removes = core_check.diff_core({"user": [e1, e2], "memory": []}, seen)
    assert [e.sha for e in adds] == ["s2"] and [s.entry_sha for s in removes] == ["s3"]


def test_r5_mirror_first_run_then_noop(ctx, deps):
    c1, r1 = night(ctx, days=0)
    rows_ = ctx.store.load_working_set()
    mirrored = [r for r in rows_.values() if r.source == "core:user"]
    assert len(mirrored) == 21                                   # 22 minus the header-only fragment
    assert all(r.in_core and r.core_target == "user" and r.core_sha for r in mirrored)
    assert all(r.tier in ("durable", "pinned") for r in mirrored)
    assert not any(r.source == "core:memory" for r in rows_.values())   # 13 episodic entries not facts
    seen = ctx.ledger.core_seen()
    assert len(seen) == 35 and all(s.present for s in seen.values())
    assert len(c1.llm.calls_of("core_classify")) == 1
    # unchanged core files: no LLM, no Lance write, empty ledger delta
    v = ctx.store.versions()
    c2, r2 = night(ctx, days=1)
    assert c2.llm.calls_of("core_classify") == [] and r2.status == "noop"
    assert ctx.store.versions() == v


def test_r5_hookless_remove_and_required_note(ctx, fake_home, deps):
    night(ctx, days=0)
    rows_ = ctx.store.load_working_set()
    inv = next(r for r in rows_.values() if "가계부" in r.text)
    inv.core_required = True
    inv.pinned = True
    commit(ctx, inv)
    entries = read_core(fake_home.user_md)
    write_core(fake_home.user_md, [e for e in entries if "가계부" not in e])   # a human edits USER.md
    c2, _ = night(ctx, days=1)
    r = ctx.store.load_working_set()[inv.id]
    assert r.in_core is False and r.status == "active" and r.pinned
    assert any(ch["change"] == "mirror_remove" for ch in c2.report.core_changes)
    assert any("USER.md에서 '가계부 관리'가 빠졌습니다" in n and inv.id in n for n in c2.report.notes)
    assert c2.alerts == []                                       # U4: Dream Log only
    seen = ctx.ledger.core_seen()
    assert seen[("user", inv.core_sha)].present is False
    # entry comes back (e.g. core-restore): row back in core, no duplicate row
    write_core(fake_home.user_md, entries)
    night(ctx, days=2)
    after = ctx.store.load_working_set()
    assert after[inv.id].in_core is True
    assert sum(1 for x in after.values() if x.core_sha == inv.core_sha) == 1


def test_t14_dream_never_writes_core_files(ctx, fake_home, paths, monkeypatch, deps):
    """30 nights: R1 core ops, claims touching USER entries, recall, decay. Core files untouched,
    no lock file created, every pinned row active and recallable (strength ≥ 0.9)."""
    before = {t: sha(paths.core_file(t)) for t in ("memory", "user")}
    with monkeypatch.context() as m:
        hits = no_core_writes(m, paths)
        night(ctx, days=0)
        assert hits == []
    # migration-style auto-pin of profile/rule rows (U2) — done by the test, not by the dream
    rows_ = ctx.store.load_working_set()
    pins = [r for r in rows_.values() if r.kind in ("profile", "rule") and r.source == "core:user"]
    for r in pins:
        r.pinned = True
        r.tier = "pinned"
    commit(ctx, *pins)
    assert pins
    with monkeypatch.context() as m:
        hits = no_core_writes(m, paths)
        for n in range(1, 31):
            claims = []
            if n % 5 == 0:
                p = pins[n % len(pins)]
                claims.append(claim(ctx, f"{p.text} 라고 다시 말했다 {n}일째.", kind=p.kind,
                                    subject=p.subject, user=False, et=ctx.now + n * DAY - 3600))
            if n == 3:
                live_insert(ctx, "inbox", ts=ctx.now + 2 * DAY, session_id="s", platform="cli",
                            op="core_remove", text=USER_ENTRIES[7], target="user", status="pending")
            if n == 4:
                live_insert(ctx, "recall_events", ts=ctx.now + 3 * DAY, session_id="s", platform="telegram",
                            turn_no=1, memory_id=pins[0].id, kind="used", mode="vector")
            night(ctx, claims=claims, days=n)
        assert hits == []
    assert {t: sha(paths.core_file(t)) for t in ("memory", "user")} == before
    for t in ("memory", "user"):
        assert not Path(str(paths.core_file(t)) + ".lock").exists()
    final = ctx.store.load_working_set()
    from hermesyume import strength
    for p in pins:
        r = final[p.id]
        assert r.status == "active" and r.pinned
        assert strength.strength(r, ctx.now + 30 * DAY) >= 0.9


def _user_row(ctx, text):
    from hermesyume.paths import load_provider_module
    cf = load_provider_module("corefmt")
    return row(ctx, text, kind="rule", core_target="user", core_sha=cf.core_sha(text), source="core:user",
               in_core=False)


def test_restore_appends_with_lock_backup_and_mode(ctx, paths, fake_home):
    entries = read_core(fake_home.user_md)
    text = "**복원 테스트:** 짧은 규칙 하나."
    r = _user_row(ctx, text)
    commit(ctx, r)
    os.chmod(fake_home.user_md, 0o640)
    res = core_check.restore(paths, r.id, target=None, store=ctx.store, ledger=ctx.ledger, now=ctx.now)
    assert res.ok and res.reason is None and res.target == "user"
    assert read_core(fake_home.user_md) == entries + [text]
    assert Path(str(fake_home.user_md) + ".lock").exists()          # Hermes lock convention
    assert res.backup and read_core(res.backup) == entries
    assert stat.S_IMODE(os.stat(fake_home.user_md).st_mode) == 0o640
    again = core_check.restore(paths, r.id, target=None, store=ctx.store, ledger=ctx.ledger, now=ctx.now)
    assert again.ok and again.reason == "already_present"
    assert any(a.op == "core_restore" for a in ctx.ledger.audits(r.id))


def test_restore_refuses_over_limit(ctx, paths, fake_home):
    before = sha(fake_home.user_md)
    used = len("\n§\n".join(read_core(fake_home.user_md)))
    r = _user_row(ctx, "**아주 긴 규칙:** " + "가" * max(10, 1375 - used))
    commit(ctx, r)
    res = core_check.restore(paths, r.id, target="user", store=ctx.store, ledger=ctx.ledger, now=ctx.now)
    assert not res.ok and res.reason == "limit"
    assert sha(fake_home.user_md) == before
    missing = core_check.restore(paths, "0" * 32, target="user", store=ctx.store, ledger=None, now=ctx.now)
    assert not missing.ok and missing.reason == "not_found"


def test_restore_preserves_symlink(ctx, paths, fake_home, tmp_path):
    real = tmp_path / "dotfiles" / "USER.md"
    real.parent.mkdir()
    real.write_bytes(fake_home.user_md.read_bytes())
    fake_home.user_md.unlink()
    fake_home.user_md.symlink_to(real)
    r = _user_row(ctx, "**심링크 테스트:** 링크는 유지된다.")
    commit(ctx, r)
    res = core_check.restore(paths, r.id, target="user", store=ctx.store, ledger=None, now=ctx.now)
    assert res.ok and fake_home.user_md.is_symlink()
    assert res.path == str(real)
    assert read_core(real)[-1] == "**심링크 테스트:** 링크는 유지된다."


def test_apply_proposal_verifies_rows_exist(ctx, paths, fake_home):
    from tests.fixtures.hermes_home import MEMORY_ENTRIES
    paths.ensure_data_dirs()
    prop = paths.proposals_dir / "MEMORY.md.proposed"
    write_core(prop, MEMORY_ENTRIES[:2])
    before = sha(fake_home.memory_md)
    res = core_check.apply_proposal(paths, None, store=ctx.store, now=ctx.now)
    assert not res.ok and res.reason.startswith("not_in_store") and sha(fake_home.memory_md) == before
    legacy = [row(ctx, e, kind="legacy", status="dormant", source="legacy:memory_md") for e in MEMORY_ENTRIES[2:]]
    commit(ctx, *legacy)
    res = core_check.apply_proposal(paths, str(prop), store=ctx.store, now=ctx.now)
    assert res.ok and read_core(fake_home.memory_md) == MEMORY_ENTRIES[:2] and res.backup


def test_t14_harness_catches_the_only_writer(ctx, paths, monkeypatch):
    """Sanity: the T14 guard would catch restore() (the one legitimate writer, manual only)."""
    r = _user_row(ctx, "**하네스 확인:** 쓰기 감지.")
    commit(ctx, r)
    with monkeypatch.context() as m:
        hits = no_core_writes(m, paths)
        with pytest.raises(AssertionError):
            core_check.restore(paths, r.id, target="user", store=ctx.store, ledger=None, now=ctx.now)
        assert hits
