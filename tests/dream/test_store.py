"""store.py: exact §2.2 schema, round-trip, merge_insert idempotency, search, restore, guards."""

import json

import lancedb
import numpy as np
import pyarrow as pa
import pytest

from hermesyume.store import (DEFAULT_CANDIDATE_WHERE, SchemaMismatch, Store, StoreMissing,
                              expected_schemas, history_json, memories_schema)
from hermesyume.types import (MEMORY_FIELDS, HistoryRow, MemoryRow, SuppressRow,
                              make_history_id, text_sha)
from tests.fakes import FakeEmbedder, make_row
from tests.fixtures.hermes_home import tree_hash

MODEL = "openai/text-embedding-3-small@1536"


@pytest.fixture
def emb():
    return FakeEmbedder()


@pytest.fixture
def store(tmp_path):
    return Store.open(tmp_path / "lancedb", dim=1536, embed_model=MODEL, create=True)


def full_row(emb, now):
    """Every field set to a non-default value (incl. None-able timestamps)."""
    return make_row("Orion 결제 스테이징 서버 포트는 8081이다.", embedder=emb, now=now,
                    id="a" * 32, subject="Orion 포트", subject_key="orion포트", kind="reference",
                    tier="durable", target="world", importance=0.85, level=4, pinned=True,
                    core_required=True, core_target="user", core_sha="c" * 40, in_core=True,
                    status="active", status_reason="test", status_changed_at=now - 5,
                    created_at=now - 100, updated_at=now - 50, event_time=now - 1000,
                    valid_from=now - 1000, valid_until=None, first_seen_at=now - 1000,
                    last_seen_at=now - 900, last_user_evidence_at=now - 900, evidence_count=3,
                    user_evidence_count=2, user_session_count=2, explicit_user=True, source="dream",
                    source_session_ids=["s1", "s2"], source_message_ids=["s:1", "s:2"],
                    origin_keys=["w1#0", "w2#1"], refs=["docs/yume/x.md"], recall_injected_count=4,
                    recall_injected_strong=2, recall_used_count=1, search_hit_count=1,
                    last_recalled_at=now - 10, last_used_at=None, supersedes=["b" * 32],
                    superseded_by=None, related_ids=["c" * 32], judge_pending=True,
                    needs_review=True, version=3, lang="ko", scope="default", last_run_id="r1",
                    schema_version=2)


def test_schema_is_exact_plan_2_2(store):
    store.check_schema()
    sch = store.table("memories").schema
    assert [f.name for f in sch] == list(MEMORY_FIELDS)
    assert sch.field("vector").type == pa.list_(pa.float32(), 1536)
    assert sch.field("importance").type == pa.float32()
    assert sch.field("level").type == pa.int8()
    assert sch.field("evidence_count").type == pa.int32()
    assert sch.field("created_at").type == pa.timestamp("ms", tz="UTC")
    assert sch.field("origin_keys").type == pa.list_(pa.string())
    assert sch.field("pinned").type == pa.bool_()
    h = store.table("memory_history").schema
    assert [f.name for f in h] == ["history_id", "memory_id", "run_id", "op", "before_json", "after_json", "at"]
    assert "vector" not in h.names
    s = store.table("suppress").schema
    assert [f.name for f in s] == ["id", "vector", "text_sha", "kind", "created_at", "reason"]
    assert "text" not in s.names


def test_roundtrip_all_fields(store, emb, now):
    r = full_row(emb, now)
    store.commit(upserts=[r])
    got = store.get([r.id])[r.id]
    for f in MEMORY_FIELDS:
        a, b = getattr(r, f), getattr(got, f)
        if f == "vector":
            assert np.allclose(a, b)
        elif isinstance(a, float):
            assert b == pytest.approx(a, abs=1e-3), f
        else:
            assert a == b, f
    assert got.valid_until is None and got.last_used_at is None and got.superseded_by is None


def test_merge_insert_idempotent_and_noop_keeps_version(store, emb, now):
    rows = [make_row(f"사실 {i}번: 값은 {i * 7}", embedder=emb, now=now) for i in range(12)]
    v0 = store.version()
    store.commit(upserts=rows)
    v1 = store.version()
    assert v1 > v0 and store.count() == 12
    store.commit(upserts=rows)                     # replay of the same step
    assert store.count() == 12
    before = store.versions()
    assert store.commit() == before                # empty commit: no version bump anywhere
    # update path: same id, changed fields
    rows[0].text = "사실 0번: 값은 999"
    rows[0].version = 2
    store.commit(upserts=[rows[0]])
    assert store.count() == 12
    assert store.get([rows[0].id])[rows[0].id].text == "사실 0번: 값은 999"


def test_history_merge_insert_no_duplicates(store, now):
    hist = [HistoryRow(make_history_id("r1", "m1", "insert", i), "m1", "r1", "insert",
                       history_json({}), history_json({"text": "x"}), now) for i in range(3)]
    store.commit_history(hist)
    store.commit_history(hist)                     # crash-replay
    assert store.count(name="memory_history") == 3
    assert [h.op for h in store.history(memory_id="m1")] == ["insert"] * 3
    assert make_history_id("r1", "m1", "insert", 0) == __import__("hashlib").sha256(b"r1m1insert0").hexdigest()


def test_suppress_has_no_text_and_search(store, emb, now):
    v = emb.vector("보라색 고래 7341")
    store.commit_suppress([SuppressRow("m1", v, text_sha("보라색 고래 7341"), "fact", now, "forget|run:r1")])
    store.commit_suppress([SuppressRow("m1", v, text_sha("보라색 고래 7341"), "fact", now, "forget|run:r1")])
    assert store.count(name="suppress") == 1
    hits = store.search_suppress(v)
    assert hits and hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert text_sha("보라색  고래 7341") in store.suppress_shas()   # whitespace-normalized


def test_search_cosine_prefilter_excludes_forgotten(store, emb, now):
    a = make_row("Orion 스테이징 서버 포트는 8081", embedder=emb, now=now)
    b = make_row("Orion 스테이징 서버 포트는 8081", embedder=emb, now=now, status="forgotten")
    c = make_row("오늘 저녁 메뉴는 김치찌개", embedder=emb, now=now, status="quarantined")
    d = make_row("전혀 무관한 문장 abc", embedder=emb, now=now)
    store.commit(upserts=[a, b, c, d])
    hits = store.search(a.vector, k=5)
    ids = [r.id for r, _ in hits]
    assert ids[0] == a.id and b.id not in ids and c.id not in ids
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert all(-1.0 <= cos <= 1.0001 for _, cos in hits)
    assert DEFAULT_CANDIDATE_WHERE == "status NOT IN ('forgotten', 'quarantined')"
    # explicit filter
    hits2 = store.search(a.vector, k=5, where="status = 'forgotten'")
    assert [r.id for r, _ in hits2] == [b.id]


def test_restore_and_purge(store, emb, now):
    r = make_row("복원 테스트 행", embedder=emb, now=now)
    store.commit(upserts=[r])
    v_before = store.version()
    r2 = r.copy()
    r2.status = "dormant"
    store.commit(upserts=[r2])
    assert store.get([r.id])[r.id].status == "dormant"
    store.restore(v_before)
    assert store.get([r.id])[r.id].status == "active"
    store.purge([r.id])
    assert store.count() == 0
    assert store.purge([]) is None


def test_working_set_without_vectors(store, emb, now):
    store.commit(upserts=[make_row(f"행 {i}", embedder=emb, now=now) for i in range(15)])
    ws = store.load_working_set(with_vectors=False)
    assert len(ws) == 15 and all(r.vector is None for r in ws.values())
    ws2 = store.load_working_set()
    assert all(r.vector is not None and r.vector.shape == (1536,) for r in ws2.values())


def test_commit_requires_vectors(store, now):
    with pytest.raises(SchemaMismatch):
        store.commit_memories([MemoryRow(id="x", text="벡터 없는 행", created_at=now)])


def test_schema_guard_list_float_vector(tmp_path):
    """The openclaw-era failure: vector stored as list<float> (not fixed_size_list)."""
    d = tmp_path / "bad"
    db = lancedb.connect(str(d))
    bad = memories_schema(1536)
    bad = bad.set(bad.get_field_index("vector"), pa.field("vector", pa.list_(pa.float32())))
    db.create_table("memories", schema=bad)
    for name, sch in expected_schemas(1536).items():
        if name != "memories":
            db.create_table(name, schema=sch)
    s = Store.open(d, dim=1536, embed_model=MODEL)
    with pytest.raises(SchemaMismatch, match="vector"):
        s.check_schema()


def test_schema_guard_wrong_dim(tmp_path):
    Store.open(tmp_path / "l", dim=1536, embed_model=MODEL, create=True)
    s = Store.open(tmp_path / "l", dim=3072, embed_model=MODEL)
    with pytest.raises(SchemaMismatch):
        s.check_schema()


def test_embed_model_guard(store, emb, now):
    store.commit(upserts=[make_row("모델 확인", embedder=emb, now=now, embed_model="openai/other@1536")])
    with pytest.raises(SchemaMismatch, match="reembed"):
        store.check_embed_model()


def test_missing_store_is_not_created(tmp_path):
    with pytest.raises(StoreMissing):
        Store.open(tmp_path / "nope", dim=1536, embed_model=MODEL)
    assert not (tmp_path / "nope").exists()


def test_reads_create_no_files(store, emb, now, tmp_path):
    store.commit(upserts=[make_row("읽기 순수성", embedder=emb, now=now)])
    before = tree_hash(tmp_path / "lancedb")
    s = Store.open(tmp_path / "lancedb", dim=1536, embed_model=MODEL)
    s.check_schema()
    s.load_working_set()
    s.search(emb.vector("읽기"))
    s.search_suppress(emb.vector("읽기"))
    s.history()
    s.versions()
    assert tree_hash(tmp_path / "lancedb") == before


def test_snapshot_roundtrip_through_json(emb, now):
    r = full_row(emb, now)
    snap = json.loads(json.dumps(r.snapshot(include_vector=True)))
    back = MemoryRow.from_snapshot(snap)
    assert back.text == r.text and back.origin_keys == r.origin_keys
    assert np.allclose(back.vector, r.vector)
    assert "vector_b64" not in r.snapshot()
