"""calibrate.py (§4.1 보정): recall_min_cos = max(0.40, p99(unrelated) + 0.05), never below the
floor, labeled pairs, read-only."""

from __future__ import annotations

import json

from hermesyume import calibrate
from tests.fixtures.hermes_home import tree_hash

DAY = 86400.0


def _ev(ctx, *, ts, kind, cos, sid="s1", mid="m1"):
    ctx.live.conn.execute(
        "INSERT INTO recall_events(ts,session_id,platform,turn_no,memory_id,kind,cos,mode,snapshot_run) "
        "VALUES(?,?,?,?,?,?,?,?,?)", (ts, sid, "telegram", 1, mid, kind, cos, "vector", "r0"))
    ctx.live.conn.commit()


def test_recommendation_from_unused_events(ctx):
    for i, c in enumerate([0.41, 0.45, 0.52, 0.58]):
        _ev(ctx, ts=ctx.now - 3600, kind="shadow", cos=c, mid=f"u{i}")
    _ev(ctx, ts=ctx.now - 3600, kind="injected", cos=0.80, mid="hit")
    _ev(ctx, ts=ctx.now - 3000, kind="used", cos=None, mid="hit")
    _ev(ctx, ts=ctx.now - 30 * DAY, kind="shadow", cos=0.95, mid="old")       # outside 7 days
    res = calibrate.calibrate(ctx)
    assert res.n_shadow == 4 and res.n_injected == 1
    assert res.unrelated_source == "events" and res.n_unrelated == 4 and res.n_related == 1
    p99 = res.quantiles["unrelated"]["p99"]
    assert abs(p99 - 0.5782) < 1e-3
    assert res.recall_min_cos == 0.63                                          # ceil2(p99 + 0.05)
    assert res.candidate_cos_floor is None
    assert "recall_min_cos" in calibrate.format_result(res)


def test_never_below_floor(ctx):
    for i in range(5):
        _ev(ctx, ts=ctx.now - 60, kind="shadow", cos=0.10 + i * 0.01, mid=f"x{i}")
    assert calibrate.calibrate(ctx).recall_min_cos == 0.40


def test_no_samples_keeps_current(ctx):
    c = ctx.cfg.replace(recall_min_cos=0.45)
    ctx2 = type(ctx)(**{**ctx.__dict__, "cfg": c})
    res = calibrate.calibrate(ctx2)
    assert res.recall_min_cos == 0.45 and res.unrelated_source == "none"


def test_labeled_pairs_drive_both_thresholds(ctx, fake_embedder):
    fake_embedder.pin_cos("관련 B", "관련 A", 0.80)
    fake_embedder.pin_cos("관련 D", "관련 C", 0.70)
    fake_embedder.pin_cos("무관 B", "무관 A", 0.30)
    pairs = [{"a": "관련 A", "b": "관련 B", "label": "related"},
             {"a": "관련 C", "b": "관련 D", "label": "related"},
             {"a": "무관 A", "b": "무관 B", "label": "unrelated"},
             {"a": "x", "b": "y", "label": "maybe"}]
    p = ctx.paths.data_dir / calibrate.LABELS_RELPATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in pairs) + "\n# comment\n", encoding="utf-8")
    _ev(ctx, ts=ctx.now - 60, kind="shadow", cos=0.9, mid="z")                 # labels win over events
    res = calibrate.calibrate(ctx)
    assert res.n_labeled == 3 and res.unrelated_source == "labels"
    assert res.recall_min_cos == 0.40                                          # 0.30 + 0.05 < floor
    assert abs(res.candidate_cos_floor - 0.71) < 1e-3                          # p10 of [0.70, 0.80]


def test_calibrate_writes_nothing(ctx, fake_home):
    _ev(ctx, ts=ctx.now - 60, kind="shadow", cos=0.5)
    def files():   # sqlite's own -shm/-wal bookkeeping of the open rw handle is not a write by us
        return {k: v for k, v in tree_hash(fake_home.root, exclude_dirs=()).items()
                if not k.endswith(("-shm", "-wal"))}

    before = files()
    calibrate.calibrate(ctx)
    assert files() == before
