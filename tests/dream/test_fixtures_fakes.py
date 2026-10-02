"""Test infrastructure itself: fake embedder (Korean), scripted LLM, state.db fixture parity with
the live schema, fake HERMES_HOME, corefmt, shared types helpers."""

import json
import sqlite3

import numpy as np
import pytest

from hermesyume.offline import HashEmbedder, NullLLM
from hermesyume.paths import load_provider_module
from hermesyume.types import (Claim, RunBudget, append_capped, make_window_id, text_sha)
from tests.fakes import (FakeEmbedder, ScriptedLLM, claim, cosine, extract_json, judge_json,
                         vector_with_cos)
from tests.fixtures.hermes_home import (MEMORY_ENTRIES, USER_ENTRIES, make_hermes_home, read_core,
                                        tree_hash)
from tests.fixtures.statedb import (LIVE_STATE_DB, MESSAGES_DDL, SESSIONS_DDL, StateDB,
                                    build_basic, encode_json_content, live_columns)


def test_fake_embedder_korean_similarity():
    e = FakeEmbedder()
    a = e.vector("Orion 결제 스테이징 서버 포트는 8081이다.")
    b = e.vector("Orion 스테이징 서버 포트는 8081번이다.")
    c = e.vector("오늘 저녁은 김치찌개를 먹었다.")
    assert float(a @ b) > 0.6
    assert abs(float(a @ c)) < 0.2
    assert np.allclose(a, e.vector("Orion 결제 스테이징 서버 포트는 8081이다."))   # deterministic
    assert np.isclose(np.linalg.norm(a), 1.0)
    e.pin_cos("새 문장", "기준 문장", 0.73)
    assert float(e.vector("새 문장") @ e.vector("기준 문장")) == pytest.approx(0.73, abs=1e-5)
    e.alias("별칭", "기준 문장")
    assert float(e.vector("별칭") @ e.vector("기준 문장")) == pytest.approx(1.0, abs=1e-6)
    v = vector_with_cos(list(e.vector("x")), 0.95, "s")
    assert cosine(v, e.vector("x")) == pytest.approx(0.95, abs=1e-6)


def test_fake_embedder_failure_and_budget():
    from hermesyume.embedder import EmbedAuthError
    from hermesyume.types import BudgetExceeded
    e = FakeEmbedder(budget=RunBudget(max_embed_inputs=2))
    e.fail_with, e.fail_once = EmbedAuthError("401", status=401), True
    with pytest.raises(EmbedAuthError):
        e.embed(["a"])
    assert e.embed(["a", "b"]).shape == (2, 1536)
    with pytest.raises(BudgetExceeded):
        e.embed(["c"])


def test_scripted_llm_routing():
    llm = ScriptedLLM()
    llm.queue("extract", extract_json(claim("rule", "앞으로 Orion 요금 질문은 요금표부터 확인한다.",
                                            evidence=["U#12"], explicit=True)), "not json")
    llm.on("judge", lambda msgs: judge_json(("m1", "duplicate", "same")))
    r1 = llm.chat_json("extract", [{"role": "user", "content": "w"}])
    assert r1.data["claims"][0]["kind"] == "rule" and r1.data["claims"][0]["explicit"] == "true"
    assert r1.data["claims"][0]["level"] == "3"
    assert llm.chat_json("extract", []).data is None          # invalid JSON on purpose
    assert llm.chat_json("extract", []).data == {"claims": []}   # default
    assert llm.chat_json("judge", []).data["relations"][0]["type"] == "duplicate"
    llm.queue("consolidate", RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        llm.chat_json("consolidate", [])
    assert len(llm.calls_of("extract")) == 3 and llm.usage.failures == 1


def test_offline_null_llm_and_hash_embedder():
    n = NullLLM()
    assert n.chat_json("judge", []).data == {"relations": []}
    h = HashEmbedder(dim=256)
    assert h.embed(["가나다"]).shape == (1, 256)


@pytest.mark.skipif(live_columns() is None, reason="set HERMESYUME_TEST_LIVE_HOME to compare with a real state.db")
def test_statedb_fixture_matches_live_columns(tmp_path):
    live = live_columns(LIVE_STATE_DB)
    db = StateDB(tmp_path / "state.db")
    fake = {t: [r[1] for r in db.conn.execute(f"PRAGMA table_info({t})")] for t in ("sessions", "messages")}
    db.close()
    assert fake == live


def test_statedb_builder_scenarios(tmp_path, now):
    ids = build_basic(tmp_path / "state.db", now)
    c = sqlite3.connect(tmp_path / "state.db")
    src = dict(c.execute("SELECT id, source FROM sessions"))
    assert src["cron1"] == "cron" and src["tg1"] == "telegram"
    assert c.execute("SELECT cwd FROM sessions WHERE id='cli_probe'").fetchone()[0].startswith("/tmp/agent-probe")
    row = c.execute("SELECT content, api_content FROM messages WHERE id=?", (ids["tg1_u1"],)).fetchone()
    assert "<memory-context>" not in row[0] and "<memory-context>" in row[1]
    assert c.execute("SELECT display_kind FROM messages WHERE id=?", (ids["tg1_hidden"],)).fetchone()[0] == "hidden"
    c.close()
    # compression shapes
    db = StateDB(tmp_path / "c.db")
    db.session("p", "telegram", started_at=now - 1000)
    for i in range(3):
        db.exchange("p", f"질문 {i}", f"답 {i}", now - 900 + i * 60)
    tail = db.compress_generation("p", keep_tail=2)
    rows = db.conn.execute("SELECT id, role, content, timestamp, active, compacted, _compressed_summary "
                           "FROM messages WHERE session_id='p' ORDER BY id").fetchall()
    olds = [r for r in rows if r[5] == 1]
    copies = [r for r in rows if r[0] in tail]
    assert len(olds) == 6 and all(r[4] == 0 for r in olds)
    assert [(r[1], r[2], r[3]) for r in copies] == [(r[1], r[2], r[3]) for r in olds[-2:]]
    child_tail = db.compression_child("p", "c1", started_at=now - 100, keep_tail=1)
    assert db.conn.execute("SELECT end_reason FROM sessions WHERE id='p'").fetchone()[0] == "compression"
    assert db.conn.execute("SELECT parent_session_id FROM sessions WHERE id='c1'").fetchone()[0] == "p"
    ct = db.conn.execute("SELECT timestamp FROM messages WHERE id=?", (child_tail[0],)).fetchone()[0]
    assert ct == copies[-1][3]
    db.close()
    assert encode_json_content([{"type": "text", "text": "a"}]).startswith("\x00json:")
    assert "CREATE TABLE sessions" in SESSIONS_DDL and "api_content" in MESSAGES_DDL


def test_fake_home_core_format(tmp_path):
    home = make_hermes_home(tmp_path)
    cf = load_provider_module("corefmt")
    user = cf.read_entries(home.user_md)
    mem = cf.read_entries(home.memory_md)
    assert user == USER_ENTRIES and len(user) == 22 and len(mem) == 13 and len(user) + len(mem) == 35
    assert len("\n§\n".join(USER_ENTRIES)) <= 1375 and len("\n§\n".join(MEMORY_ENTRIES)) <= 2200
    assert read_core(home.user_md) == USER_ENTRIES
    assert all(cf.is_episodic(e) for e in mem)
    assert "**Reading List:**" in user
    assert (home.root / ".env").read_text().startswith("OPENAI_API_KEY=")
    assert (home.md_dir / "scratch-notes.md").exists()
    t1 = tree_hash(home.root)
    assert "memories/USER.md" in t1 and tree_hash(home.root) == t1


def test_corefmt():
    cf = load_provider_module("corefmt")
    assert cf.parse_entries("a\n§\nb § c\n§\n\n") == ["a", "b § c"]
    assert cf.core_sha("  **라벨:**  값\t\n") == cf.core_sha("**라벨:** 값")
    assert cf.core_sha("Abc") != cf.core_sha("abc")
    assert len(cf.core_sha("x")) == 40
    assert cf.entry_label("**가계부 관리:** 공용 가계부 DB") == "**가계부 관리:**"
    assert cf.entry_label("라벨 없음") is None
    assert cf.is_episodic("Session: 2026-06-27 요약") and not cf.is_episodic("Sessions are fun")
    assert cf.containment("공용 가계부 DB", "지출은 공용 가계부 db가 원장") == 1.0
    assert cf.containment("완전히 다른 내용", "공용 가계부 DB") < 0.2
    assert cf.containment("", "x") == 0.0


def test_types_helpers():
    assert append_capped(["a", "b"], ["b", "c"], 2) == ["b", "c"]
    assert append_capped([], ["x"], 0) == ["x"]
    w1 = make_window_id("statedb", "root", 1, 5)
    assert w1 == __import__("hashlib").sha256(b"statedbroot15").hexdigest()
    assert make_window_id("md", "/p", 0, 10, "abc") != make_window_id("md", "/p", 0, 10, "abd")
    assert text_sha("A  b") == text_sha("a b")
    c = Claim(origin_key="w#0", source="dream", kind="rule", target="user", subject="s", text="t",
              evidence_roles=["assistant"])
    assert c.assistant_only and not c.has_user_evidence
    c2 = Claim(origin_key="i:1", source="tool:yume_remember", kind="rule", target="user", subject="s", text="t")
    assert c2.has_user_evidence
    b = RunBudget(max_llm_calls=1, max_runtime_s=1000)
    assert b.can_llm() and not b.expired()
    assert json.dumps(b.__dict__)


def test_run_context_alert_is_u4_only(ctx):
    from hermesyume.types import ALERT_CODES
    a = ctx.alert("window_quarantined", "창 격리")
    assert a.code in ALERT_CODES and ctx.alerts == [a]
    for banned in ("new_pin", "core_required_removed", "pin_budget_exceeded", "guard_held", "candidate"):
        with pytest.raises(ValueError):
            ctx.alert(banned, "x")
    ctx.note("새 pin: …")
    assert ctx.report.notes == ["새 pin: …"]


def test_u2_constants():
    from hermesyume.types import GATE_REASONS, KIND_HL_DAYS, PROTECTED_KINDS, RunReport, RunStats
    assert "memory_meta" in GATE_REASONS
    assert all(KIND_HL_DAYS[k] == 60 for k in PROTECTED_KINDS)
    assert "review" not in RunReport.__dataclass_fields__
    assert "candidates" not in RunStats.__dataclass_fields__
