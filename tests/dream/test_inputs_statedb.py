"""T3 — state.db loader (PLAN-v2 §3.1–3.2, §11.2 T3; CONTRACTS §4.1)."""

from __future__ import annotations

import hashlib
import sqlite3

import pytest

from hermesyume.ledger import Ledger
from hermesyume.sanitize import SanitizeReport, sanitize_messages
from hermesyume.sources import statedb
from hermesyume.sources.statedb import decode_content, load_lineages, recent_texts, resolve_root
from hermesyume.sqlite_util import connect_ro, open_for_read
from hermesyume.types import Watermark
from tests.fixtures.statedb import SYNTHETIC_FIRST_MESSAGE, StateDB, encode_json_content

MIN = 60.0


@pytest.fixture
def ledger(tmp_path):
    led = Ledger.open(tmp_path / "ledger.db")
    yield led
    led.close()


def _load(db_path, ledger, cfg, now, *, settle=30, ends=()):
    with open_for_read(db_path, pure=False) as conn:
        return load_lineages(conn, ledger=ledger, cfg=cfg, now=now, settle_minutes=settle,
                             session_end_ids=set(ends))


def _by_root(load):
    return {lin.root: lin for lin in load.lineages}


def _commit_all(ledger, load, run_id="r1"):
    """Advance every lineage watermark to its last eligible message (all windows ok)."""
    for lin in load.lineages:
        last = lin.messages[-1]
        ledger.set_wm(lin.root, last.ts, last.msg_id, run_id)
    for sid, root in load.session_roots.items():
        ledger.set_root(sid, root)


# ── §3.1 filters ─────────────────────────────────────────────────────────────

def test_basic_filters_and_counters(statedb, fake_home, cfg, now, ledger):
    load = _load(fake_home.state_db, ledger, cfg, now)
    roots = _by_root(load)
    assert set(roots) == {"tg1", "cli_tmp"}          # /tmp cwd stays included (narrow deny list)
    ex = load.excluded
    assert ex["source"] == 1                          # cron1
    assert ex["synthetic"] == 1                       # cli_synth
    assert ex["deny_cwd"] == 1                        # cli_probe
    assert ex["hidden_session"] == 1                  # hidden1
    assert ex["tool_role"] == 1 and ex["hidden_message"] == 1
    assert ex["not_settled"] == 2                     # tg_recent (5 min old)
    tg = roots["tg1"]
    assert [m.msg_id for m in tg.messages] == [statedb["tg1_u1"], statedb["tg1_a1"],
                                               statedb["tg1_u2"], statedb["tg1_a2"]]
    m0 = tg.messages[0]
    assert (m0.ref, m0.key, m0.role, m0.source, m0.session_id, m0.platform) == \
        (f"U#{statedb['tg1_u1']}", f"s:{statedb['tg1_u1']}", "user", "statedb", "tg1", "telegram")
    assert tg.messages[1].ref.startswith("A#")
    assert tg.platform == "telegram" and tg.title == "Orion 회의" and tg.chat_type == "dm"
    assert tg.fully_settled and tg.wm is None and tg.context_before == []
    assert load.messages_in == 6 and load.sessions_seen == 7
    assert load.session_roots == {"tg1": "tg1", "cli_tmp": "cli_tmp"}
    # oldest lineage first
    assert [lin.root for lin in load.lineages] == ["cli_tmp", "tg1"]


def test_api_content_is_never_read(statedb, fake_home, cfg, now, ledger):
    conn = connect_ro(fake_home.state_db)

    def deny_api_content(action, arg1, arg2, dbname, source):
        if action == sqlite3.SQLITE_READ and arg2 == "api_content":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    conn.set_authorizer(deny_api_content)
    try:
        load = load_lineages(conn, ledger=ledger, cfg=cfg, now=now, settle_minutes=30,
                             session_end_ids=set())
        texts = recent_texts(conn, cfg=cfg, now=now, days=30)
    finally:
        conn.close()
    u1 = _by_root(load)["tg1"].messages[0]
    assert "memory-context" not in u1.text and "System note" not in u1.text   # api_content had it
    assert u1.text.startswith("Orion 결제")
    assert all("api_content" not in t for _, t in texts)


def test_memory_context_in_content_removed_by_sanitize(statedb, fake_home, cfg, now, ledger):
    tg = _by_root(_load(fake_home.state_db, ledger, cfg, now))["tg1"]
    raw = next(m for m in tg.messages if m.msg_id == statedb["tg1_u2"])
    assert "<memory-context>" in raw.text                     # loader returns RAW text
    rep = SanitizeReport()
    clean = sanitize_messages(tg.messages, cfg=cfg, repeat_lines=set(), report=rep)
    u2 = next(m for m in clean if m.msg_id == statedb["tg1_u2"])
    assert u2.text == "Orion 데모 마감은 2026-10-10이야."
    assert rep.blocks_removed["memory_context"] == 1
    assert (u2.ref, u2.key, u2.ts) == (raw.ref, raw.key, raw.ts)


def test_session_end_marker_settles_recent_session(statedb, fake_home, cfg, now, ledger):
    load = _load(fake_home.state_db, ledger, cfg, now, ends={"tg_recent"})
    tg = _by_root(load)["tg_recent"]
    assert [m.text for m in tg.messages] == ["방금 한 말: 보라색 고래 7341", "네."]
    assert "not_settled" not in load.excluded
    # --settle-minutes 0 also settles it
    load0 = _load(fake_home.state_db, ledger, cfg, now, settle=0)
    assert "tg_recent" in _by_root(load0)


def test_owner_and_chat_type_filters(statedb, fake_home, cfg, now, ledger):
    load = _load(fake_home.state_db, ledger, cfg.replace(owner_user_ids=["someone-else"]), now)
    assert set(_by_root(load)) == {"cli_tmp"}          # user_id NULL sessions are not owner-filtered
    assert load.excluded["owner"] == 3                 # tg1, tg_recent, hidden1 (user u1)
    db = StateDB(fake_home.state_db)
    db.session("grp", "telegram", started_at=now - 3 * 3600, user_id="u1", chat_type="group")
    db.exchange("grp", "그룹 대화의 사적인 내용입니다.", "네.", now - 3 * 3600 + 10)
    db.close()
    load = _load(fake_home.state_db, ledger, cfg, now)
    assert "grp" not in _by_root(load) and load.excluded["chat_type"] == 1


def test_owner_filter_counts(statedb, fake_home, cfg, now, ledger):
    load = _load(fake_home.state_db, ledger, cfg.replace(owner_user_ids=["u1"]), now)
    assert set(_by_root(load)) == {"tg1", "cli_tmp"}
    assert "owner" not in load.excluded


def test_synthetic_regex_checks_first_user_message_of_whole_lineage(fake_home, cfg, now, ledger):
    db = StateDB(fake_home.state_db)
    t = now - 5 * 3600
    db.session("s1", "cli", started_at=t)
    db.exchange("s1", SYNTHETIC_FIRST_MESSAGE, "ok", t + 1)
    db.exchange("s1", "정상처럼 보이는 두 번째 질문입니다", "ok", t + 100)
    db.compression_child("s1", "s1c", started_at=t + 200)
    db.exchange("s1c", "압축 뒤 새 질문 (첫 메시지가 아님)", "ok", t + 300)
    db.close()
    # pretend the first part was already processed: synthetic must still apply to the lineage
    ledger.set_wm("s1", t + 130, 4, "r0")
    load = _load(fake_home.state_db, ledger, cfg, now)
    assert "s1" not in _by_root(load) and load.excluded["synthetic"] == 1


def test_synthetic_counted_before_deny_cwd(fake_home, cfg, now, ledger):
    db = StateDB(fake_home.state_db)
    t = now - 5 * 3600
    db.session("probe42", "cli", started_at=t, cwd="/tmp/agent-probe-42")
    db.exchange("probe42", SYNTHETIC_FIRST_MESSAGE, "ok", t + 1)
    db.close()
    load = _load(fake_home.state_db, ledger, cfg, now)
    assert load.excluded == {"synthetic": 1}


def test_decode_content_json_prefix():
    enc = encode_json_content([{"type": "text", "text": "첫 줄"}, {"type": "image_url", "image_url": "x"},
                               {"type": "input_text", "text": "둘째"}, "셋째"])
    assert decode_content(enc) == "첫 줄\n둘째\n셋째"
    assert decode_content(encode_json_content({"type": "output_text", "text": "단일"})) == "단일"
    assert decode_content("\x00json:{broken") == ""
    assert decode_content(None) == "" and decode_content("plain") == "plain"


def test_json_content_rows_are_decoded(fake_home, cfg, now, ledger):
    db = StateDB(fake_home.state_db)
    t = now - 3 * 3600
    db.session("mm", "telegram", started_at=t, chat_type="dm")
    db.message("mm", "user", encode_json_content([{"type": "text", "text": "사진과 함께: 포트 8081"},
                                                  {"type": "image_url", "image_url": {"url": "x"}}]), t)
    db.message("mm", "assistant", encode_json_content([{"type": "image_url", "image_url": "x"}]), t + 5)
    db.close()
    load = _load(fake_home.state_db, ledger, cfg, now)
    mm = _by_root(load)["mm"]
    assert [m.text for m in mm.messages] == ["사진과 함께: 포트 8081"]
    assert load.excluded["empty"] == 1


# ── settle (§3.2) ────────────────────────────────────────────────────────────

def test_settle_defers_possibly_incomplete_exchange(fake_home, cfg, now, ledger):
    db = StateDB(fake_home.state_db)
    db.session("a", "telegram", started_at=now - 3600, chat_type="dm")
    u1 = db.message("a", "user", "첫 질문입니다 충분히 길게", now - 50 * MIN)
    a1 = db.message("a", "assistant", "첫 답", now - 49 * MIN)
    u2 = db.message("a", "user", "두 번째 질문", now - 40 * MIN)
    a2 = db.message("a", "assistant", "두 번째 답 (아직 정착 전)", now - 10 * MIN)
    db.close()
    load = _load(fake_home.state_db, ledger, cfg, now)
    lin = _by_root(load)["a"]
    # u2 is old enough, but its reply is not: the u2 exchange is deferred from u2 on
    assert [m.msg_id for m in lin.messages] == [u1, a1]
    assert not lin.fully_settled and load.excluded["not_settled"] == 2
    # later everything settles
    load = _load(fake_home.state_db, ledger, cfg, now + 30 * MIN)
    assert [m.msg_id for m in _by_root(load)["a"].messages] == [u1, a1, u2, a2]


def test_settle_keeps_complete_exchange_before_new_user_turn(fake_home, cfg, now, ledger):
    db = StateDB(fake_home.state_db)
    db.session("b", "telegram", started_at=now - 3600, chat_type="dm")
    u1 = db.message("b", "user", "질문", now - 50 * MIN)
    a1 = db.message("b", "assistant", "답", now - 45 * MIN)
    db.message("b", "user", "방금 보낸 질문", now - 2 * MIN)
    db.close()
    lin = _by_root(_load(fake_home.state_db, ledger, cfg, now))["b"]
    assert [m.msg_id for m in lin.messages] == [u1, a1] and not lin.fully_settled


def test_settle_nothing_when_only_partial_exchange(fake_home, cfg, now, ledger):
    db = StateDB(fake_home.state_db)
    db.session("c", "telegram", started_at=now - 3600, chat_type="dm")
    db.message("c", "user", "질문", now - 50 * MIN)
    db.message("c", "assistant", "답 1", now - 45 * MIN)
    db.message("c", "assistant", "답 2 (진행 중)", now - 1 * MIN)
    db.close()
    load = _load(fake_home.state_db, ledger, cfg, now)
    assert "c" not in _by_root(load) and load.excluded["not_settled"] == 3


# ── lineage watermark: generation copies and compression children (§3.2) ────

def _session_with_exchanges(db, sid, t0, n, *, source="telegram", prefix="교환"):
    db.session(sid, source, started_at=t0, chat_type="dm" if source == "telegram" else None,
               title=f"{sid} 제목")
    out = []
    for i in range(n):
        out.append(db.exchange(sid, f"{prefix} {i} 사용자 발화입니다", f"{prefix} {i} 답변", t0 + i * 100))
    return out


def test_generation_copies_never_reprocessed_across_runs(fake_home, cfg, now, ledger):
    t0 = now - 10 * 3600
    db = StateDB(fake_home.state_db)
    _session_with_exchanges(db, "g", t0, 4)
    db.close()
    run1 = _load(fake_home.state_db, ledger, cfg, now)
    g1 = _by_root(run1)["g"]
    assert len(g1.messages) == 8
    _commit_all(ledger, run1)

    db = StateDB(fake_home.state_db)
    copies = db.compress_generation("g", keep_tail=2)          # tail copied with new ids
    new_u, new_a = db.exchange("g", "압축 뒤 새 질문", "압축 뒤 새 답", t0 + 1000)
    db.close()
    assert len(copies) == 2
    run2 = _load(fake_home.state_db, ledger, cfg, now)
    g2 = _by_root(run2)["g"]
    assert [m.msg_id for m in g2.messages] == [new_u, new_a]
    # the copy of the last processed message shares its timestamp: fetched, then dropped
    assert run2.excluded["generation_copy"] == 1
    assert run2.excluded["compressed_summary"] == 1
    # context is the last processed exchange (original ids, not copies)
    assert [m.text for m in g2.context_before] == ["교환 3 사용자 발화입니다", "교환 3 답변"]
    assert {m.msg_id for m in g2.context_before}.isdisjoint(copies)
    assert "[요약]" not in " ".join(m.text for m in g2.messages)    # _compressed_summary rows skipped


def test_generation_copies_within_one_run_are_deduped(fake_home, cfg, now, ledger):
    t0 = now - 10 * 3600
    db = StateDB(fake_home.state_db)
    ids = _session_with_exchanges(db, "g", t0, 3)
    copies = db.compress_generation("g", keep_tail=2)
    db.close()
    load = _load(fake_home.state_db, ledger, cfg, now)
    g = _by_root(load)["g"]
    assert [m.msg_id for m in g.messages] == [i for pair in ids for i in pair]   # lowest ids kept
    assert load.excluded["generation_copy"] == len(copies)
    assert load.excluded["compressed_summary"] == 1


def test_compression_child_tail_never_reprocessed(fake_home, cfg, now, ledger):
    t0 = now - 10 * 3600
    db = StateDB(fake_home.state_db)
    _session_with_exchanges(db, "p", t0, 3)
    db.close()
    run1 = _load(fake_home.state_db, ledger, cfg, now)
    _commit_all(ledger, run1)

    db = StateDB(fake_home.state_db)
    tail = db.compression_child("p", "p_child", started_at=t0 + 500, keep_tail=2)
    nu, na = db.exchange("p_child", "자식 세션의 새 질문", "자식 세션의 새 답", t0 + 600)
    db.close()
    run2 = _load(fake_home.state_db, ledger, cfg, now)
    roots = _by_root(run2)
    assert set(roots) == {"p"}                                  # child joins the parent lineage
    p = roots["p"]
    assert [m.msg_id for m in p.messages] == [nu, na]
    assert set(tail).isdisjoint(m.msg_id for m in p.messages)
    assert [s.id for s in p.sessions] == ["p", "p_child"]
    assert run2.session_roots == {"p_child": "p"}
    _commit_all(ledger, run2, "r2")

    # Hermes prunes the parent session: the cached root keeps the lineage (and its watermark)
    db = StateDB(fake_home.state_db)
    db.conn.execute("DELETE FROM messages WHERE session_id='p'")
    db.conn.execute("DELETE FROM sessions WHERE id='p'")          # child keeps a dangling parent id
    db.conn.commit()
    nu3, na3 = db.exchange("p_child", "부모가 지워진 뒤의 질문", "답", t0 + 700)
    db.close()
    run3 = _load(fake_home.state_db, ledger, cfg, now)
    roots3 = _by_root(run3)
    assert set(roots3) == {"p"}
    assert [m.msg_id for m in roots3["p"].messages] == [nu3, na3]


def test_compression_child_in_first_run_dedupes_against_parent(fake_home, cfg, now, ledger):
    t0 = now - 10 * 3600
    db = StateDB(fake_home.state_db)
    ids = _session_with_exchanges(db, "p", t0, 2)
    tail = db.compression_child("p", "pc", started_at=t0 + 500, keep_tail=2)
    nu, na = db.exchange("pc", "새 질문", "새 답", t0 + 600)
    db.close()
    load = _load(fake_home.state_db, ledger, cfg, now)
    p = _by_root(load)["p"]
    assert [m.msg_id for m in p.messages] == [i for pair in ids for i in pair] + [nu, na]
    assert load.excluded["generation_copy"] == len(tail)


def test_non_compression_parent_starts_new_lineage(fake_home, cfg, now, ledger):
    t0 = now - 10 * 3600
    db = StateDB(fake_home.state_db)
    _session_with_exchanges(db, "old", t0, 1)
    db.end_session("old", ended_at=t0 + 50, end_reason="session_reset")
    db.session("new", "telegram", started_at=t0 + 100, parent_session_id="old", chat_type="dm")
    db.exchange("new", "리셋 뒤 대화입니다", "네", t0 + 120)
    db.close()
    load = _load(fake_home.state_db, ledger, cfg, now)
    assert set(_by_root(load)) == {"old", "new"}
    with open_for_read(fake_home.state_db, pure=False) as conn:
        cache: dict[str, str] = {}
        assert resolve_root(conn, "new", cache) == "new"
        assert cache["new"] == "new"


def test_resolve_root_climbs_compression_chain(fake_home, now):
    t0 = now - 10 * 3600
    db = StateDB(fake_home.state_db)
    _session_with_exchanges(db, "a", t0, 1)
    db.compression_child("a", "b", started_at=t0 + 200)
    db.compression_child("b", "c", started_at=t0 + 300)
    db.close()
    with open_for_read(fake_home.state_db, pure=False) as conn:
        cache: dict[str, str] = {}
        assert resolve_root(conn, "c", cache) == "a"
        assert cache == {"c": "a", "b": "a", "a": "a"}
        assert resolve_root(conn, "zzz-missing", {}) == "zzz-missing"
        assert resolve_root(conn, "c", {"c": "cached"}) == "cached"


def test_watermark_tie_on_timestamp_uses_key(fake_home, cfg, now, ledger):
    t0 = now - 10 * 3600
    db = StateDB(fake_home.state_db)
    db.session("w", "telegram", started_at=t0, chat_type="dm")
    u = db.message("w", "user", "같은 시각 원본", t0)
    a = db.message("w", "assistant", "답", t0 + 10)
    db.close()
    ledger.set_wm("w", t0 + 10, a, "r1")
    db = StateDB(fake_home.state_db)
    # an identical (role, content, ts) row with a higher id whose original no longer passes the
    # filters (rewound): dedupe cannot see it, the watermark key check must
    db.conn.execute("UPDATE messages SET active=0, compacted=0 WHERE id=?", (a,))
    db.conn.commit()
    copy_a = db.message("w", "assistant", "답", t0 + 10)
    other = db.message("w", "assistant", "같은 시각의 다른 내용", t0 + 10)
    later = db.message("w", "user", "나중 질문", t0 + 20)
    db.close()
    lin = _by_root(_load(fake_home.state_db, ledger, cfg, now))["w"]
    ids = [m.msg_id for m in lin.messages]
    assert copy_a not in ids and other in ids and later in ids and u not in ids


def test_rerun_without_new_messages_is_empty(statedb, fake_home, cfg, now, ledger):
    run1 = _load(fake_home.state_db, ledger, cfg, now)
    _commit_all(ledger, run1)
    run2 = _load(fake_home.state_db, ledger, cfg, now)
    assert run2.lineages == [] and run2.session_roots == {} and run2.messages_in == 0
    # row-level counters only count rows after the watermark; lineage-level ones stay
    for k in ("tool_role", "hidden_message", "generation_copy", "empty"):
        assert k not in run2.excluded
    assert run2.excluded["synthetic"] == 1 and run2.excluded["source"] == 1


def test_context_before_after_partial_processing(fake_home, cfg, now, ledger):
    t0 = now - 10 * 3600
    db = StateDB(fake_home.state_db)
    ids = _session_with_exchanges(db, "k", t0, 3)
    db.close()
    ledger.set_wm("k", t0 + 100 + 30, ids[1][1], "r1")      # exchanges 0 and 1 processed
    lin = _by_root(_load(fake_home.state_db, ledger, cfg, now))["k"]
    assert [m.msg_id for m in lin.messages] == list(ids[2])
    assert [m.msg_id for m in lin.context_before] == list(ids[1])
    assert lin.wm == Watermark("k", t0 + 130, ids[1][1], "r1")


def test_hidden_messages_and_tool_rows_excluded(fake_home, cfg, now, ledger):
    t0 = now - 10 * 3600
    db = StateDB(fake_home.state_db)
    db.session("h", "telegram", started_at=t0, chat_type="dm")
    u = db.message("h", "user", "질문입니다", t0)
    db.message("h", "tool", '{"result": 1}', t0 + 1, tool_name="x")
    db.message("h", "assistant", "숨김 상태줄", t0 + 2, display_kind="hidden")
    db.message("h", "assistant", "되감기로 지워진 답", t0 + 3, active=0, compacted=0)
    a = db.message("h", "assistant", "최종 답", t0 + 4)
    db.message("h", "system", "system row", t0 + 5)
    db.close()
    load = _load(fake_home.state_db, ledger, cfg, now)
    assert [m.msg_id for m in _by_root(load)["h"].messages] == [u, a]
    assert load.excluded["tool_role"] == 2 and load.excluded["hidden_message"] == 1
    assert load.excluded["inactive"] == 1


def test_recent_texts_for_repeat_line_stats(statedb, fake_home, cfg, now):
    db = StateDB(fake_home.state_db)
    db.session("old", "telegram", started_at=now - 40 * 86400, chat_type="dm")
    db.exchange("old", "40일 전 메시지", "답", now - 40 * 86400)
    db.close()
    with open_for_read(fake_home.state_db, pure=False) as conn:
        texts = dict(recent_texts(conn, cfg=cfg, now=now, days=30))
    assert f"s:{statedb['tg1_u1']}" in texts and f"s:{statedb['cli_tmp_u']}" in texts
    assert f"s:{statedb['tg_recent_u']}" in texts                 # no settle for statistics
    joined = "\n".join(texts.values())
    assert "40일 전" not in joined and "9999" not in joined and "nightly report" not in joined
    assert "숨김 세션" not in joined and "probe 세션" not in joined
    assert all(k.startswith("s:") for k in texts)


def test_schema_drift_tolerated(tmp_path, cfg, now, ledger):
    p = tmp_path / "old_state.db"
    conn = sqlite3.connect(p)
    conn.executescript("""
        CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT NOT NULL, started_at REAL NOT NULL);
        CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT,
                               content TEXT, timestamp REAL);
    """)
    conn.execute("INSERT INTO sessions VALUES('o','cli',?)", (now - 7200,))
    conn.execute("INSERT INTO messages(session_id,role,content,timestamp) VALUES('o','user','옛 스키마 질문',?)",
                 (now - 7000,))
    conn.execute("INSERT INTO messages(session_id,role,content,timestamp) VALUES('o','assistant','답',?)",
                 (now - 6990,))
    conn.commit()
    conn.close()
    load = _load(p, ledger, cfg, now)
    assert [m.text for m in _by_root(load)["o"].messages] == ["옛 스키마 질문", "답"]


def test_loader_is_read_only(statedb, fake_home, cfg, now, ledger):
    before = hashlib.sha256(fake_home.state_db.read_bytes()).hexdigest()
    files_before = sorted(p.name for p in fake_home.state_db.parent.iterdir())
    with open_for_read(fake_home.state_db, pure=False) as conn:
        load_lineages(conn, ledger=ledger, cfg=cfg, now=now, settle_minutes=30, session_end_ids=set())
        recent_texts(conn, cfg=cfg, now=now, days=30)
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM messages")             # mode=ro + query_only
    with open_for_read(fake_home.state_db, pure=True) as conn:  # dry-run snapshot path
        load_lineages(conn, ledger=ledger, cfg=cfg, now=now, settle_minutes=30, session_end_ids=set())
    assert hashlib.sha256(fake_home.state_db.read_bytes()).hexdigest() == before
    assert sorted(p.name for p in fake_home.state_db.parent.iterdir()) == files_before


def test_missing_tables_yield_empty(tmp_path, cfg, now, ledger):
    p = tmp_path / "empty.db"
    sqlite3.connect(p).close()
    load = _load(p, ledger, cfg, now)
    assert load.lineages == [] and load.sessions_seen == 0
    with open_for_read(p, pure=False) as conn:
        assert recent_texts(conn, cfg=cfg, now=now, days=30) == []


def test_ledger_none_is_accepted(statedb, fake_home, cfg, now):
    with open_for_read(fake_home.state_db, pure=False) as conn:
        load = load_lineages(conn, ledger=None, cfg=cfg, now=now, settle_minutes=30, session_end_ids=set())
    assert {lin.root for lin in load.lineages} == {"tg1", "cli_tmp"}


def test_module_exports_contract_names():
    assert statedb.CONTENT_JSON_PREFIX == "\x00json:"
    for name in ("SessionInfo", "Lineage", "StateDBLoad", "decode_content", "resolve_root",
                 "load_lineages", "recent_texts"):
        assert hasattr(statedb, name)
