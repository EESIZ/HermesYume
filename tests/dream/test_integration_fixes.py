"""Integrator fixes (DEVIATIONS I-8…I-13): purge also deletes history, legacy rows are never
candidates nor revived, suppressed claims are marked in the report, the telegram token regex
matches inside a bot URL, and loading the runtime threat patterns writes no bytecode."""

from __future__ import annotations

import shutil

from hermesyume import plan as P, recall_fold, threat
from hermesyume.types import LedgerDelta, RecallEvent, SuppressRow, text_sha
from hermesyume.upsert import Upserter
from tests.dream.test_rem_helpers import claim, commit, deps, next_ctx, row  # noqa: F401

DAY = 86400.0


def _commit_ws(ctx, ws):
    P.apply_guard(ws, ctx.cfg, mode="live")
    pl = P.build_plan(ctx, ws, lance_version_before=ctx.store.version(), ledger_delta=LedgerDelta(audit=list(ws.audit)),
                      inbox_consume_ids=[], inbox_skip_ids=[], docs=[])
    return P.commit_plan(ctx, pl)


def _ws(ctx):
    return P.WorkingSet(ctx.store.load_working_set(), run_id=ctx.run_id, now=ctx.now,
                        embed_model=ctx.embedder.model_id)


def test_purge_deletes_history_rows(ctx, deps):
    r = row(ctx, "Orion 테스트 비밀 문장은 지워져야 한다.", kind="fact")
    ws = _ws(ctx)
    ws.insert(r, reason="t", user_evidence=True)
    _commit_ws(ctx, ws)
    assert ctx.store.history(memory_id=r.id)
    c2 = next_ctx(ctx, days=1, run_id="purge-run")
    ws2 = _ws(c2)
    ws2.update(r.id, {"status": "quarantined"}, op="status", reason="quarantine:secret", guard_exempt=True)
    ws2.purge(r.id, reason="quarantine:secret")
    _commit_ws(c2, ws2)
    assert r.id not in ctx.store.load_working_set()
    assert ctx.store.history(memory_id=r.id) == []
    assert ctx.store.purge_history(["nope"]) is None and ctx.store.purge_history([]) is None


def test_legacy_rows_are_not_candidates(ctx, deps):
    text = "Orion 결제 스테이징 서버 포트는 8081이다."
    leg = row(ctx, text, kind="legacy", status="dormant", source="legacy:memory_md")
    commit(ctx, leg)
    ws = _ws(ctx)
    up = Upserter(ctx, ws)
    c = claim(ctx, text, kind="reference")
    assert all(cd.row.id != leg.id for cd in up.candidates(c))
    out = up.upsert(c)
    assert out.action == "inserted" and out.memory_id != leg.id
    assert ws.get(leg.id).status == "dormant"


def test_used_event_does_not_revive_legacy(ctx, deps):
    leg = row(ctx, "옛 Dreamer 덤프 원문 한 줄이 여기에 있다.", kind="legacy", status="dormant",
              source="legacy:dreamer")
    commit(ctx, leg)
    ws = _ws(ctx)
    ev = RecallEvent(id=1, ts=ctx.now, session_id="s", platform="cli", turn_no=1, memory_id=leg.id,
                     kind="tool_hit", cos=0.5, mode="vector", snapshot_run=None)
    res = recall_fold.fold_recall_events(ctx, ws, [ev], snapshot_max_id=1)
    assert res.revived == [] and ws.get(leg.id).status == "dormant"
    assert ws.get(leg.id).search_hit_count == 1


def test_suppressed_claim_is_marked_in_report(ctx, deps):
    text = "보라색 고래 다음 숫자는 7341이다 테스트."
    c = claim(ctx, text)
    ws = _ws(ctx)
    ws.add_suppress(SuppressRow(id="sup1", vector=c.vector.copy(), text_sha=text_sha(text), kind="fact",
                                created_at=ctx.now, reason="forget|run:x"))
    ctx.report.claims.append({"origin_key": c.origin_key, "kind": "fact", "subject": "s", "text": text,
                              "status": "active"})
    out = Upserter(ctx, ws).upsert(c)
    assert out.action == "suppressed"
    assert ctx.report.claims[-1]["status"] == "suppressed"


def test_telegram_token_inside_bot_url_is_a_secret():
    tok = "123456789:" + "A" * 35
    assert threat.secret_types(f"https://api.telegram.org/bot{tok}/sendDocument") == ["telegram"]
    assert threat.secret_types(f"토큰 {tok} 끝") == ["telegram"]
    assert threat.secret_types("12345678901:" + "A" * 35) == []        # 11 digits: not a bot id


def test_runtime_threat_patterns_load_writes_no_bytecode(tmp_path):
    rt = tmp_path / "runtime"
    (rt / "tools").mkdir(parents=True)
    shutil.copy(threat.VENDOR_PATH, rt / "tools" / "threat_patterns.py")
    sc = threat.load_scanner(rt)
    assert sc.source == "runtime"
    assert not (rt / "tools" / "__pycache__").exists()
