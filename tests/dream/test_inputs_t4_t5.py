"""T4 / T5 — loader side (PLAN-v2 §3.2, §11.2; CONTRACTS §4.1/§4.2/§4.5/§4.10).

nrem (B2) owns extraction, alerts and the run loop. These tests drive the input modules through a
minimal stand-in of that loop, written strictly from the CONTRACTS rules (watermark advances only
through the leading run of ok/empty/quarantined windows; md processed_bytes likewise; attempts
from ledger windows; window cap defers the rest), and check what the inputs must guarantee:
  T4  a failed window keeps the watermark, comes back with the SAME window id next run, and after
      the 3rd failure it is quarantined and the watermark moves past it;
  T5  windows over the per-run cap continue exactly in the next run — no loss, no duplicate —
      including a cap that falls inside an oversized exchange, appends between runs, and a
      compression (generation copy) between runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from hermesyume import clock
from hermesyume.config import Config
from hermesyume.ledger import Ledger
from hermesyume.sanitize import SanitizeReport, sanitize_messages
from hermesyume.sources.markdown import file_state_after, scan_md_sources
from hermesyume.sources.statedb import load_lineages
from hermesyume.sqlite_util import open_for_read
from hermesyume.types import WM_ADVANCING_STATUSES, LedgerDelta, Window, WindowState
from hermesyume.windows import build_windows
from tests.fixtures.statedb import StateDB

NOW = clock.parse_iso("2026-10-02T04:40")
MAX_ATTEMPTS = 3


# ── contract stand-ins for nrem.compute_watermarks / md progress ────────────

def compute_watermarks(windows: list[Window], states: dict[str, WindowState]) -> dict[str, tuple[float, int]]:
    out: dict[str, tuple[float, int]] = {}
    roots: dict[str, list[Window]] = {}
    for w in windows:
        if w.source == "statedb":
            roots.setdefault(w.root, []).append(w)
    for root, ws in roots.items():
        last = None
        for w in ws:
            st = states.get(w.window_id)
            if st is None or st.status not in WM_ADVANCING_STATUSES:
                break
            last = w
        if last is not None:
            out[root] = (last.last_ts, last.last_id)
    return out


def md_progress(windows: list[Window], states: dict[str, WindowState]) -> dict[str, int]:
    out: dict[str, int] = {}
    roots: dict[str, list[Window]] = {}
    for w in windows:
        if w.source == "md":
            roots.setdefault(w.root, []).append(w)
    for root, ws in roots.items():
        last = None
        for w in ws:
            st = states.get(w.window_id)
            if st is None or st.status not in WM_ADVANCING_STATUSES:
                break
            last = w
        if last is not None:
            out[root] = last.last_id
    return out


@dataclass
class RunResult:
    windows: list[Window] = field(default_factory=list)        # considered, processing order
    processed: list[Window] = field(default_factory=list)      # extracted this run
    states: dict[str, WindowState] = field(default_factory=dict)
    deferred: int = 0
    alerts: list[str] = field(default_factory=list)


class Driver:
    """Minimal nrem stand-in (inputs → windows → fake extraction → ledger delta)."""

    def __init__(self, db_path, ledger: Ledger, cfg: Config):
        self.db_path, self.ledger, self.cfg = db_path, ledger, cfg
        self.extracted: dict[str, int] = {}       # window_id → successful extractions
        self.fail: set[str] = set()               # window ids that fail extraction
        self.run_no = 0

    def build(self, now: float) -> tuple[list[Window], dict]:
        rep = SanitizeReport()
        wins: list[Window] = []
        mds = {}
        if self.db_path is not None:
            with open_for_read(self.db_path, pure=False) as conn:
                load = load_lineages(conn, ledger=self.ledger, cfg=self.cfg, now=now,
                                     settle_minutes=30, session_end_ids=set())
            self.session_roots = load.session_roots
            for lin in load.lineages:
                msgs = sanitize_messages(lin.messages, cfg=self.cfg, repeat_lines=set(), report=rep)
                ctx = sanitize_messages(lin.context_before, cfg=self.cfg, repeat_lines=set(), report=rep)
                wins += build_windows(source="statedb", root=lin.root, platform=lin.platform,
                                      title=lin.title, messages=msgs, context_before=ctx,
                                      cfg=self.cfg, ref_ts=NOW)
        else:
            self.session_roots = {}
        srcs, _ = scan_md_sources(self.cfg, self.ledger, now=now)
        for src in srcs:
            mds[src.path] = src
            msgs = sanitize_messages(src.messages, cfg=self.cfg, repeat_lines=set(), report=rep)
            ctx = sanitize_messages(src.context_before, cfg=self.cfg, repeat_lines=set(), report=rep)
            wins += build_windows(source="md", root=src.path, platform="md", title=src.slug,
                                  messages=msgs, context_before=ctx, cfg=self.cfg, ref_ts=NOW, md=src)
        # contract order: (start_ts, root, first_id), per-root order preserved by construction
        wins.sort(key=lambda w: (w.start_ts, w.root, w.first_id))
        return wins, mds

    def run(self, now: float = NOW, *, cap: int | None = None) -> RunResult:
        self.run_no += 1
        run_id = f"r{self.run_no}"
        wins, mds = self.build(now)
        res = RunResult(windows=wins)
        cap = int(self.cfg.max_windows_per_run) if cap is None else cap
        stopped_roots: set[str] = set()
        for w in wins:
            if w.root in stopped_roots or len(res.processed) >= cap:
                stopped_roots.add(w.root)
                res.deferred += 1
                continue
            prev = self.ledger.get_window(w.window_id)
            attempts = prev.attempts if prev else 0
            if prev is not None and prev.status in WM_ADVANCING_STATUSES:
                status = prev.status                         # already extracted: no second call
            elif w.window_id in self.fail:
                attempts += 1
                status = "quarantined" if attempts >= MAX_ATTEMPTS else "failed"
                if status == "quarantined":
                    res.alerts.append(f"window_quarantined:{w.window_id}")
            else:
                status = "ok"
                self.extracted[w.window_id] = self.extracted.get(w.window_id, 0) + 1
            res.processed.append(w)
            res.states[w.window_id] = WindowState(w.window_id, w.source, w.root, w.first_id,
                                                  w.last_id, w.last_ts, status, attempts,
                                                  None, run_id, 0)
        delta = LedgerDelta(watermarks=compute_watermarks(wins, res.states),
                            session_roots=dict(self.session_roots),
                            windows=list(res.states.values()))
        for path, pb in md_progress(wins, res.states).items():
            delta.md_files.append(file_state_after(mds[path], pb, run_id))
        self.ledger.apply_delta(run_id, delta, lance_version_after=None)
        return res


@pytest.fixture
def ledger(tmp_path):
    led = Ledger.open(tmp_path / "ledger.db")
    yield led
    led.close()


def _cfg(tmp_path, **kw) -> Config:
    base = {"md_sources": [str(tmp_path / "md")], "window_chars": 400, "window_context_chars": 1500,
            "max_windows_per_run": 60}
    base.update(kw)
    (tmp_path / "md").mkdir(exist_ok=True)
    return Config(base)


def _lineage(db: StateDB, sid: str, t0: float, n: int, *, size: int = 120, big_at: int | None = None):
    db.session(sid, "telegram", started_at=t0, chat_type="dm", title=sid)
    ids = []
    for i in range(n):
        if big_at is not None and i == big_at:
            u = db.message(sid, "user", f"{sid} 큰 질문 {i}", t0 + i * 100)
            ids.append(u)
            for j in range(6):
                ids.append(db.message(sid, "assistant", f"{sid} 긴 답 {i}.{j} " + "나" * 150,
                                      t0 + i * 100 + j + 1))
            continue
        u, a = db.exchange(sid, f"{sid} 질문 {i} " + "가" * size, f"{sid} 답 {i}", t0 + i * 100)
        ids += [u, a]
    return ids


def _sig(w: Window):
    return (w.window_id, [m.msg_id for m in w.messages], [m.msg_id for m in w.context])


# ── T5 ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cap", [1, 2, 3])
def test_t5_cap_continues_exactly_statedb(tmp_path, ledger, cap):
    cfg = _cfg(tmp_path)
    db = StateDB(tmp_path / "state.db")
    ids = _lineage(db, "L", NOW - 20 * 3600, 12, big_at=5)
    db.close()
    empty = Ledger.open(tmp_path / "absent-ledger.db", readonly=True)
    full, _ = Driver(tmp_path / "state.db", empty, cfg).build(NOW)
    assert len(full) >= 6
    assert any(w.messages[0].role == "assistant" for w in full)     # a split exchange exists

    drv = Driver(tmp_path / "state.db", ledger, cfg)
    seen: list[int] = []
    runs_windows: list[Window] = []
    for _ in range(30):
        res = drv.run(cap=cap)
        if not res.windows:
            break
        assert len(res.processed) <= cap
        runs_windows += res.processed
        seen += [m.msg_id for w in res.processed for m in w.messages]
    assert seen == ids                                  # no loss, no duplicate, in order
    assert [_sig(w) for w in runs_windows] == [_sig(w) for w in full]   # same windows as one pass
    assert all(v == 1 for v in drv.extracted.values()) and len(drv.extracted) == len(full)
    # body text identical too (only the 정리 기준일 header could differ between runs)
    by_id = {w.window_id: w for w in full}
    for w in runs_windows:
        assert w.text == by_id[w.window_id].text


def test_t5_cap_with_appends_and_compression_between_runs(tmp_path, ledger):
    cfg = _cfg(tmp_path)
    db = StateDB(tmp_path / "state.db")
    t0 = NOW - 20 * 3600
    ids = _lineage(db, "L", t0, 8)
    db.close()
    drv = Driver(tmp_path / "state.db", ledger, cfg)
    seen = [m.msg_id for w in drv.run(cap=2).processed for m in w.messages]

    db = StateDB(tmp_path / "state.db")
    copies = db.compress_generation("L", keep_tail=4)               # generation copies mid-backlog
    more = []
    for i in range(3):
        more += list(db.exchange("L", f"압축 뒤 질문 {i} " + "다" * 100, f"압축 뒤 답 {i}", t0 + 5000 + i * 100))
    db.close()
    for _ in range(20):
        res = drv.run(cap=2)
        if not res.windows:
            break
        seen += [m.msg_id for w in res.processed for m in w.messages]
    assert seen == ids + more
    assert set(copies).isdisjoint(seen)


def test_t5_multi_source_cap_defers_rest_of_root(tmp_path, ledger):
    cfg = _cfg(tmp_path)
    db = StateDB(tmp_path / "state.db")
    a_ids = _lineage(db, "A", NOW - 30 * 3600, 6)
    b_ids = _lineage(db, "B", NOW - 29 * 3600 + 50, 6)
    db.close()
    md_lines = [f"user: md 질문 {i} " + "라" * 120 + f"\nassistant: md 답 {i}\n" for i in range(6)]
    p = tmp_path / "md" / "2026-09-01-log.md"
    p.write_text("".join(md_lines), encoding="utf-8")
    drv = Driver(tmp_path / "state.db", ledger, cfg)
    got: dict[str, list] = {"A": [], "B": [], "md": []}
    for _ in range(30):
        res = drv.run(cap=3)
        if not res.windows:
            break
        # within a run each root is processed as a prefix of its windows
        for root in {w.root for w in res.windows}:
            flags = [w in res.processed for w in res.windows if w.root == root]
            assert flags == sorted(flags, reverse=True)
        for w in res.processed:
            key = "md" if w.source == "md" else w.root
            got[key] += [m.key for m in w.messages]
    assert got["A"] == [f"s:{i}" for i in a_ids] and got["B"] == [f"s:{i}" for i in b_ids]
    assert len(got["md"]) == 12 and len(set(got["md"])) == 12
    st = ledger.get_md(str(p))
    assert st.processed_bytes == p.stat().st_size and st.status == "ok"


def test_t5_md_cap_continues_exactly(tmp_path, ledger):
    cfg = _cfg(tmp_path)
    p = tmp_path / "md" / "2026-09-02-big.md"
    body = "# 일지\n" + "".join(f"user: 질문 {i} " + "마" * 120 + f"\nassistant: 답 {i}\n\n"
                                for i in range(8)) + "마지막 일지 문단\n"
    p.write_text(body, encoding="utf-8")
    full, _ = Driver(None, Ledger.open(tmp_path / "absent-ledger.db", readonly=True),
                     cfg).build(NOW)
    drv = Driver(None, ledger, cfg)
    windows = []
    for _ in range(20):
        res = drv.run(cap=2)
        if not res.windows:
            break
        windows += res.processed
    assert [_sig(w) for w in windows] == [_sig(w) for w in full]
    assert [w.text for w in windows] == [w.text for w in full]
    assert ledger.get_md(str(p)).processed_bytes == len(body.encode())
    # appending later continues after the processed bytes only
    with open(p, "a", encoding="utf-8") as f:
        f.write("user: 추가된 질문입니다 충분히 길게\n")
    res = drv.run()
    assert [m.text for w in res.processed for m in w.messages] == ["추가된 질문입니다 충분히 길게"]


# ── T4 ───────────────────────────────────────────────────────────────────────

def test_t4_failure_keeps_watermark_then_quarantine(tmp_path, ledger):
    cfg = _cfg(tmp_path)
    db = StateDB(tmp_path / "state.db")
    ids = _lineage(db, "L", NOW - 20 * 3600, 6)
    db.close()
    drv = Driver(tmp_path / "state.db", ledger, cfg)
    full, _ = drv.build(NOW)
    assert len(full) >= 3
    w1, w2 = full[0], full[1]
    drv.fail.add(w2.window_id)

    r1 = drv.run()
    assert r1.states[w1.window_id].status == "ok"
    assert r1.states[w2.window_id].status == "failed" and r1.states[w2.window_id].attempts == 1
    wm = ledger.get_wm("L")
    assert (wm.last_ts, wm.last_id) == (w1.last_ts, w1.last_id)      # held at the failed window

    r2 = drv.run()
    assert r2.windows[0].window_id == w2.window_id                     # same window comes back
    assert [_sig(w) for w in r2.windows] == [_sig(w) for w in full[1:]]
    assert r2.states[w2.window_id].attempts == 2
    assert ledger.get_wm("L").last_id == w1.last_id

    r3 = drv.run()
    assert r3.states[w2.window_id].status == "quarantined" and r3.states[w2.window_id].attempts == 3
    assert r3.alerts == [f"window_quarantined:{w2.window_id}"]
    wm = ledger.get_wm("L")
    assert (wm.last_ts, wm.last_id) == (full[-1].last_ts, full[-1].last_id)   # moved past it

    r4 = drv.run()
    assert r4.windows == []
    # windows after the failed one were re-offered but never extracted twice (stable ids)
    assert all(v == 1 for v in drv.extracted.values())
    assert w2.window_id not in drv.extracted
    assert ledger.get_window(w2.window_id).status == "quarantined"
    assert sorted(m.msg_id for w in full for m in w.messages) == sorted(ids)


def test_t4_md_failure_keeps_processed_bytes(tmp_path, ledger):
    cfg = _cfg(tmp_path)
    p = tmp_path / "md" / "2026-08-23-x.md"
    p.write_text("".join(f"user: 질문 {i} " + "바" * 150 + f"\nassistant: 답 {i}\n" for i in range(5)),
                 encoding="utf-8")
    drv = Driver(None, ledger, cfg)
    full, _ = drv.build(NOW)
    assert len(full) >= 3
    drv.fail.add(full[1].window_id)
    drv.run()
    st = ledger.get_md(str(p))
    assert st.processed_bytes == full[0].last_id and st.status == "partial"
    r2 = drv.run()
    assert r2.windows[0].window_id == full[1].window_id
    drv.run()
    st = ledger.get_md(str(p))
    assert st.processed_bytes == p.stat().st_size and st.status == "ok"


def test_watermark_helper_matches_nrem_if_present(tmp_path, ledger):
    """Cross-check the stand-in against the real nrem.compute_watermarks once B2 lands."""
    nrem = pytest.importorskip("hermesyume.nrem")
    fn = getattr(nrem, "compute_watermarks", None)
    if fn is None:
        pytest.skip("nrem.compute_watermarks not available yet")
    cfg = _cfg(tmp_path)
    db = StateDB(tmp_path / "state.db")
    _lineage(db, "L", NOW - 20 * 3600, 6)
    db.close()
    full, _ = Driver(tmp_path / "state.db", ledger, cfg).build(NOW)
    states = {w.window_id: WindowState(w.window_id, w.source, w.root, w.first_id, w.last_id,
                                       w.last_ts, s, 1, None, "r", 0)
              for w, s in zip(full, ["ok", "empty", "failed", "ok"])}
    assert fn(full, states) == compute_watermarks(full, states)
