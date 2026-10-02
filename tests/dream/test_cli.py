"""cli.py (integrator) — `yume` end to end on a fake HERMES_HOME, offline (no network):

- `yume dream --offline` twice on build_basic + the md episode: the 2nd run is a no-op (G1b/T13:
  LLM 0, created/reinforced 0, Lance version unchanged); the Orion fact is one active row and
  reaches the serving copy.
- T19: `dream --dry-run` leaves the whole home (incl. lock files) byte-identical except
  dream-log/ and runs/.
- T14: dream + migrate + read-only admin commands never write MEMORY.md/USER.md; only
  `core-restore` does, under dream.lock.
- admin commands: debug plant/event, inspect, pin list, forget (confirm for pins), unpin,
  restore (incl. suppress rows), export, status, search, config, core-check, reembed,
  dream --approve-run, refusals (not initialized, live home, lock held).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

from hermesyume import cli
from hermesyume.paths import dream_lock
from hermesyume.store import Store
from tests.dream.test_rem_helpers import no_core_writes
from tests.fixtures.hermes_home import read_core, tree_hash, write_core

NOWS = "2026-10-02T04:40"          # = conftest NOW (build_basic timestamps are relative to it)


def run(capsys, *argv: str) -> tuple[int, str, str]:
    rc = cli.main(list(argv))
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def J(capsys, home, *argv: str, rc: int = 0):
    code, out, err = run(capsys, "--hermes-home", str(home), *argv, "--json")
    assert code == rc, (code, out, err)
    return json.loads(out)


def core_shas(paths) -> dict:
    return {t: hashlib.sha256(paths.core_file(t).read_bytes()).hexdigest() for t in ("memory", "user")}


def dream(capsys, home, *extra: str, rc: int = 0, now: str = NOWS):
    return J(capsys, home, "dream", "--offline", "--settle-minutes", "0", "--now", now, *extra, rc=rc)


# ── dream ────────────────────────────────────────────────────────────────────

def test_offline_dream_twice_second_is_noop(initialized, statedb, paths, capsys, monkeypatch):
    home = paths.hermes_home
    before = core_shas(paths)
    with monkeypatch.context() as m:
        hits = no_core_writes(m, paths)
        r1 = dream(capsys, home)
        r2 = dream(capsys, home)
    assert hits == []
    assert core_shas(paths) == before
    s1, s2 = r1["stats"], r2["stats"]
    assert r1["status"] == "committed" and s1["created"] >= 5 and s1["llm_calls"] >= 1
    assert r2["status"] == "committed"
    assert (s2["llm_calls"], s2["created"], s2["reinforced"]) == (0, 0, 0)
    assert s2["lance_version_after"] == s1["lance_version_after"] == s2["lance_version_before"]
    rows = J(capsys, home, "inspect", "--query", "8081")
    active = [r for r in rows if r["status"] == "active"]
    assert len(active) == 1 and "8081" in active[0]["text"] and "strength" in active[0]
    assert active[0]["source_session_ids"] == ["tg1"]
    assert not J(capsys, home, "inspect", "--query", "9999")          # synthetic-eval session excluded
    con = sqlite3.connect(f"file:{paths.recall_sqlite}?mode=ro", uri=True)
    ids = {r[0] for r in con.execute("SELECT id FROM items")}
    con.close()
    assert active[0]["id"] in ids
    logs = sorted(p.name for p in paths.dream_log_dir.glob("*.md"))
    assert len(logs) == 2 and not any(n.endswith("_dry.md") for n in logs)
    run_rec = J(capsys, home, "status")["runs"][0]
    assert run_rec["status"] == "committed" and run_rec["error"] is None


def test_dry_run_is_pure_before_and_after_a_live_run(initialized, statedb, paths, fake_home, capsys):
    home = paths.hermes_home

    def snap():
        return tree_hash(fake_home.root), tree_hash(fake_home.workspace)

    a = snap()
    r = dream(capsys, home, "--dry-run")
    assert r["status"] == "dry" and r["stats"]["created"] >= 5
    assert snap() == a                                   # lock files included, no dream.lock created
    assert not paths.dream_lock.exists()
    assert list(paths.dream_log_dir.glob("*_dry.md")) and list(paths.runs_dir.glob("*/plan.json"))
    dream(capsys, home)
    b = snap()
    r = dream(capsys, home, "--dry-run", now="2026-10-03T04:40")
    assert r["status"] == "dry"
    assert snap() == b


def test_dream_refusals_and_lock(paths, fake_home, initialized, statedb, capsys):
    home = paths.hermes_home
    with dream_lock(paths):
        rc, out, _ = run(capsys, "--hermes-home", str(home), "dream", "--offline", "--now", NOWS)
    assert rc == 0 and "already running" in out
    (home / "gateway.pid").write_text("1\n")             # looks like a live home now
    rc, _, err = run(capsys, "--hermes-home", str(home), "dream", "--offline")
    assert rc == 1 and "--dry-run" in err
    rc, _, err = run(capsys, "--hermes-home", str(home), "debug", "plant", "--offline")
    assert rc == 1 and "라이브" in err
    (home / "gateway.pid").unlink()


def test_not_initialized(paths, fake_home, capsys):
    rc, _, err = run(capsys, "--hermes-home", str(paths.hermes_home), "dream", "--offline")
    assert rc == 1 and "yume init" in err
    assert not paths.data_dir.exists() or not paths.ledger_db.exists()


def test_init_is_idempotent(paths, fake_home, capsys):
    home = paths.hermes_home
    out = J(capsys, home, "init")
    assert out["config_created"] is True and paths.ledger_db.exists() and paths.lancedb_dir.exists()
    out = J(capsys, home, "init")
    assert out["config_created"] is False
    assert (paths.config_json.stat().st_mode & 0o777) == 0o600


# ── T14 across dream + migrate + read-only admin ─────────────────────────────

def test_t14_cli_dream_migrate_never_write_core(initialized, statedb, paths, capsys, monkeypatch):
    home = paths.hermes_home
    before = core_shas(paths)
    with monkeypatch.context() as m:
        hits = no_core_writes(m, paths)
        mig = J(capsys, home, "migrate", "--approve-migration", "--only", "core,memory_md", "--offline")
        assert mig["status"] == "committed" and mig["core_map"]["accounted"] == 35
        dream(capsys, home)
        dream(capsys, home, now="2026-11-20T04:40")
        J(capsys, home, "core-check")
        J(capsys, home, "export")
        J(capsys, home, "status", "--recall")
        J(capsys, home, "pin", "list")
    assert hits == []
    assert core_shas(paths) == before


# ── admin commands ───────────────────────────────────────────────────────────

def test_admin_commands(initialized, statedb, paths, capsys):
    home = paths.hermes_home
    dream(capsys, home)
    ids = J(capsys, home, "debug", "plant", "--kind", "event", "--importance", "0.4", "--text",
            "E2E event 디버그 기억입니다", "--offline", "--now", NOWS)["ids"]
    assert len(ids) == 1
    many = J(capsys, home, "debug", "plant", "--kind", "fact", "--count", "3", "--tag", "g9", "--offline")
    assert len(many["ids"]) == 3
    tagged = J(capsys, home, "inspect", "--tag", "g9")
    assert len(tagged) == 3 and all(r["source"] == "debug:g9" for r in tagged)
    ev = J(capsys, home, "debug", "event", "--target", ids[0], "--kind", "used", "--at", "+50d")
    assert ev["kind"] == "used"

    # make one row pinned (no CLI pin command under U2: emulate a migration auto-pin)
    store = Store.from_config(paths, initialized.cfg)
    tgt = J(capsys, home, "inspect", "--query", "8081")[0]["id"]
    row = store.get([tgt])[tgt]
    row.pinned = True
    store.commit(upserts=[row])
    pins = J(capsys, home, "pin", "list")
    assert [p["id"] for p in pins] == [tgt]

    out = J(capsys, home, "forget", tgt, rc=1)
    assert out["result"] == "confirm_required"
    out = J(capsys, home, "unpin", tgt)
    assert out["result"] == "ok" and not J(capsys, home, "pin", "list")
    out = J(capsys, home, "forget", tgt, "--reason", "테스트")
    assert out["result"] == "ok"
    forget_run = out["run_id"]
    assert J(capsys, home, "inspect", "--id", tgt)[0]["status"] == "forgotten"
    store = Store.from_config(paths, initialized.cfg)
    assert any(s.id == tgt for s in store.load_suppress())
    con = sqlite3.connect(f"file:{paths.recall_sqlite}?mode=ro", uri=True)
    assert tgt not in {r[0] for r in con.execute("SELECT id FROM items")}
    con.close()

    res = J(capsys, home, "restore", "--run", forget_run)
    assert res["lance_version_restored"] is not None
    # F-3: a restore never brings back what the user asked to forget — the forget is re-applied
    assert res["reforgotten"] == [tgt]
    assert J(capsys, home, "inspect", "--id", tgt)[0]["status"] == "forgotten"
    store = Store.from_config(paths, initialized.cfg)
    assert any(s.id == tgt for s in store.load_suppress())
    con = sqlite3.connect(f"file:{paths.recall_sqlite}?mode=ro", uri=True)
    assert tgt not in {r[0] for r in con.execute("SELECT id FROM items")}
    con.close()
    # … unless the operator explicitly undoes the forget too
    res = J(capsys, home, "restore", "--run", forget_run, "--unforget")
    assert res["unforget"] and res["reforgotten"] == []
    assert J(capsys, home, "inspect", "--id", tgt)[0]["status"] == "active"
    store = Store.from_config(paths, initialized.cfg)
    assert not any(s.id == tgt for s in store.load_suppress())       # the undone forget's suppress row

    exp = J(capsys, home, "export")
    assert exp["items"] >= 5
    st = J(capsys, home, "status", "--recall")
    assert st["rows"]["active"] >= 5 and "health_24h" in st
    hits = J(capsys, home, "search", "Orion 스테이징 서버 포트", "--offline", "--limit", "3")
    assert hits and "8081" in hits[0]["text"]

    assert J(capsys, home, "config", "set", "recall_k", "4") == {"key": "recall_k", "value": 4}
    rc, out, _ = run(capsys, "--hermes-home", str(home), "config", "get", "recall_k")
    assert rc == 0 and json.loads(out) == 4
    rc, _, err = run(capsys, "--hermes-home", str(home), "config", "set", "no_such_key", "1")
    assert rc == 1
    rc, _, err = run(capsys, "--hermes-home", str(home), "config", "set", "recall_min_cos", "0.3")
    assert rc == 1                                          # floor 0.40 needs --force

    cc = J(capsys, home, "core-check")
    assert cc["files"]["user"]["entries"] == 22 and cc["new_entries"] == []


def test_core_restore_is_the_only_writer_and_takes_the_lock(initialized, statedb, paths, fake_home, capsys):
    home = paths.hermes_home
    dream(capsys, home)
    row = J(capsys, home, "inspect", "--label", "가계부")[0]
    entries = [e for e in read_core(paths.core_file("user")) if "가계부" not in e]
    write_core(paths.core_file("user"), entries)
    before = paths.core_file("user").read_bytes()
    with dream_lock(paths):
        rc, out, _ = run(capsys, "--hermes-home", str(home), "core-restore", row["id"])
    assert rc == 0 and "already running" in out and paths.core_file("user").read_bytes() == before
    out = J(capsys, home, "core-restore", row["id"])
    assert out["ok"] is True and out["backup"]
    assert "가계부 관리" in paths.core_file("user").read_text(encoding="utf-8")
    out = J(capsys, home, "core-restore", row["id"])
    assert out["ok"] is True and out["reason"] == "already_present"
    from hermesyume.ledger import Ledger
    with Ledger.from_paths(paths, readonly=True) as led:
        assert any(a.op == "core_restore" and a.memory_id == row["id"] for a in led.audits())


def test_reembed_offline_new_dim_then_dream(initialized, statedb, paths, capsys):
    home = paths.hermes_home
    dream(capsys, home)
    n_before = len(J(capsys, home, "inspect", "--status", "active,dormant,forgotten,superseded,expired"))
    out = J(capsys, home, "reembed", "--dim", "256", "--offline")
    assert out["model"].endswith("@256") and out["rows"] == n_before
    rc, o, _ = run(capsys, "--hermes-home", str(home), "config", "get", "embed_dim")
    assert json.loads(o) == 256
    r = dream(capsys, home, now="2026-10-03T04:40")
    assert r["status"] == "committed"
    con = sqlite3.connect(f"file:{paths.recall_sqlite}?mode=ro", uri=True)
    meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
    con.close()
    assert meta["embed_model"].endswith("@256") and meta["dim"] == "256"
    assert J(capsys, home, "reembed", "--dim", "256", "--offline")["changed"] is False


def test_approve_run(ctx, capsys):
    from hermesyume import plan as P
    from hermesyume.types import LedgerDelta
    from tests.dream.test_rem_helpers import commit, row
    pin = row(ctx, "**시간대:** Asia/Seoul 기준이다.", kind="profile", pinned=True, source="core:user")
    commit(ctx, pin)
    ws = P.WorkingSet(ctx.store.load_working_set(), run_id=ctx.run_id, now=ctx.now,
                      embed_model=ctx.embedder.model_id)
    ws.update(pin.id, {"pinned": False}, op="unpin", reason="bug")       # no user evidence → held
    P.apply_guard(ws, ctx.cfg, mode="live")
    pl = P.build_plan(ctx, ws, lance_version_before=ctx.store.version(), ledger_delta=LedgerDelta(),
                      inbox_consume_ids=[], inbox_skip_ids=[], docs=[])
    P.commit_plan(ctx, pl)
    assert ctx.ledger.get_run(ctx.run_id).status == "held"
    home = ctx.paths.hermes_home
    code, _, err = run(capsys, "--hermes-home", str(home), "dream", "--approve-run", "nope")
    assert code == 1
    out = J(capsys, home, "dream", "--approve-run", ctx.run_id)
    assert out["approved"] == ctx.run_id
    fresh = Store.from_config(ctx.paths, ctx.cfg)          # another process committed: reopen
    assert fresh.get([pin.id])[pin.id].pinned is False
    assert ctx.ledger.get_run(ctx.run_id).status == "committed"
    assert ctx.paths.recall_sqlite.exists()


def test_global_flags_before_or_after_subcommand(initialized, paths, capsys):
    home = str(paths.hermes_home)
    rc, out, _ = run(capsys, "--json", "--hermes-home", home, "status")
    assert rc == 0 and json.loads(out)["hermes_home"] == home
    rc, out, _ = run(capsys, "status", "--hermes-home", home, "--json")
    assert rc == 0 and json.loads(out)["hermes_home"] == home


def test_heuristic_offline_llm_shapes():
    from hermesyume.offline import HeuristicLLM
    from hermesyume.prompts import core_classify_messages, extract_messages, judge_messages
    llm = HeuristicLLM()
    body = ('세션: telegram "t" / 기간: 2026-09-28 14:02–15:10 KST / 정리 기준일: 2026-10-02(금)\n\n'
            '[추출 대상]\n[U#1 09-28 14:02] Orion 결제 스테이징 서버 포트는 8081이야. 포트 몇 번?\n'
            '[A#2 09-28 14:03] 알겠습니다 그렇게 기억하겠습니다.')
    data = llm.chat_json("extract", extract_messages(body)).data
    assert [c["text"] for c in data["claims"]] == ["Orion 결제 스테이징 서버 포트는 8081이야."]
    assert data["claims"][0]["evidence"] == ["U#1"] and data["claims"][0]["event_time"] == "2026-09-28"
    j = llm.chat_json("judge", judge_messages({"text": "가나다라마바사아자차카타파하"},
                                              [{"id": "c1", "text": "가나다라마바사아자차카타파하"},
                                               {"id": "c2", "text": "전혀 다른 문장입니다 열다섯자"}])).data
    assert [r["type"] for r in j["relations"]] == ["duplicate", "unrelated"]
    cc = llm.chat_json("core_classify", core_classify_messages(["**호칭:** 사장님", "**Reading List:**",
                                                                 "**원칙:** 항상 표로"])).data
    assert [(i["kind"], i["fragment"]) for i in cc["items"]] == [("profile", False), ("preference", True),
                                                                  ("rule", False)]
    assert llm.usage.content_calls == 3
