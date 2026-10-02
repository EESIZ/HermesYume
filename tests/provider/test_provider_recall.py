"""P2 (failure tolerance), P3 (ranking / gating), P4 (scoring speed) for prefetch (PLAN-v2 §6.2)."""

import atexit
import os
import random
import sys
import time
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from tests.provider import yume_support as S  # noqa: E402

atexit.register(S.cleanup)

Q = "Orion 스테이징 서버 포트 몇 번이었지?"          # ≥ 20 chars → no query expansion
A = S.anchor(Q)
NOW_EV = 1790000000.0


def item(mid, text, cos, **kw):
    d = {"id": mid, "text": text, "vec": S.with_cos(A, cos, mid), "event_time": NOW_EV,
         "strength": 0.5, "kind": "fact"}
    d.update(kw)
    return d


class _Base(unittest.TestCase):
    def setUp(self):
        S.reset_singletons()
        self.srv = S.fakes.FakeOpenAIServer().start()
        self.sb = S.Sandbox(self.id().rsplit(".", 1)[-1], config={"embed_base_url": self.srv.base_url})

    def tearDown(self):
        self.srv.stop()
        S.reset_singletons()

    def ids_in(self, block, sb=None):
        return [e[0] for e in (sb or self.sb).events() if e[1] in ("injected", "shadow")]

    def fresh(self, **kw):
        p = S.new_provider(self.sb, **kw)
        p.on_turn_start(1, Q)
        return p


class Ranking(_Base):
    """P3"""

    def test_threshold_and_relative_cut(self):
        self.sb.build_serving([
            item("a", "알파 설명 문장입니다 하나", 0.70),
            item("b", "베타 설명 문장입니다 둘", 0.65),
            item("c", "감마 설명 문장입니다 셋", 0.55),     # below top − 0.10
            item("d", "델타 설명 문장입니다 넷", 0.35)])    # below 0.40
        p = self.fresh()
        block = p.prefetch(Q)
        self.assertIn("알파", block)
        self.assertIn("베타", block)
        self.assertNotIn("감마", block)
        self.assertNotIn("델타", block)
        self.assertTrue(block.startswith("[Yume 장기기억 · 관련 2건]"))
        p.shutdown()
        inj = self.sb.events("injected")
        self.assertEqual(sorted(e[0] for e in inj), ["a", "b"])
        self.assertAlmostEqual(dict((e[0], e[2]) for e in inj)["a"], 0.70, places=4)
        self.assertTrue(all(e[3] == "vector" and e[4] == "sess-1" and e[5] == 1 for e in inj))

    def test_below_threshold_injects_nothing(self):
        self.sb.build_serving([item("x", "무관한 기억 하나", 0.39), item("y", "무관한 기억 둘", 0.20)])
        p = self.fresh()
        self.assertEqual(p.prefetch(Q), "")
        self.assertIsNone(p.recall_status())
        p.shutdown()
        self.assertEqual(self.sb.events("injected"), [])
        h = self.sb.live_rows("SELECT prefetch_n, injected_n, empty_n FROM health")
        self.assertEqual(h, [(1, 0, 1)])

    def test_no_reinjection_same_session_until_reset_or_compression(self):
        self.sb.build_serving([item("a", "알파 설명 문장입니다", 0.70), item("c", "감마 설명 문장입니다", 0.45)])
        p = self.fresh()
        self.assertIn("알파", p.prefetch(Q))
        p.on_turn_start(2, Q)
        second = p.prefetch(Q)
        self.assertNotIn("알파", second)          # already in this context
        self.assertIn("감마", second)             # next best becomes eligible
        p.on_turn_start(3, Q)
        self.assertEqual(p.prefetch(Q), "")
        p.on_session_switch("sess-compressed", parent_session_id="sess-1", reset=False)
        p.on_turn_start(4, Q)
        self.assertIn("알파", p.prefetch(Q))      # compression: new context → allowed again
        p.on_session_switch("sess-new", reset=True)
        p.on_turn_start(1, Q)
        self.assertIn("알파", p.prefetch(Q))

    def test_core_sha_excluded(self):
        corefmt = S.yume_mod("corefmt")
        self.sb.build_serving([
            item("core", S.USER_TEXT_LEDGER, 0.80, core_sha=corefmt.core_sha(S.USER_TEXT_LEDGER),
                 kind="rule", tier="pinned", pinned=True),
            item("a", "알파 설명 문장입니다", 0.72)])
        p = self.fresh()
        block = p.prefetch(Q)
        self.assertIn("알파", block)
        self.assertNotIn("가계부", block)

    def test_budget_1000_chars_and_k(self):
        items = [item("L%d" % i, ("긴 기억 %d " % i) + "가나다라마바사아자차" * 30, 0.70 - i * 0.001)
                 for i in range(6)]
        self.sb.build_serving(items)
        p = self.fresh()
        block = p.prefetch(Q)
        self.assertLessEqual(len(block), 1000)
        lines = block.split("\n")[1:]
        self.assertGreaterEqual(len(lines), 2)
        self.assertTrue(all(len(l) <= 300 for l in lines))
        self.assertTrue(lines[0].endswith("…(yume_search로 전문)"))
        short = [item("s%d" % i, "짧은 기억 %d번" % i, 0.70 - i * 0.002) for i in range(8)]
        sb2 = S.Sandbox("k5", config={"embed_base_url": self.srv.base_url})
        sb2.build_serving(short)
        p2 = S.new_provider(sb2)
        self.assertTrue(p2.prefetch(Q).startswith("[Yume 장기기억 · 관련 5건]"))

    def test_mmr_skips_near_duplicates(self):
        import math
        a = S.with_cos(A, 0.70, "a")
        u = S.unit([x - 0.70 * y for x, y in zip(a, A)])            # a = 0.70·A + s·u
        w = S.anchor("직교 성분")
        for b in (A, u):                                              # w ⟂ A, u
            d = sum(x * y for x, y in zip(w, b))
            w = [x - d * y for x, y in zip(w, b)]
        w = S.unit(w)
        s2, th = math.sqrt(1 - 0.69 ** 2), 0.95
        a2 = [0.69 * x + s2 * (th * y + math.sqrt(1 - th * th) * z) for x, y, z in zip(A, u, w)]
        self.assertAlmostEqual(S.fakes.cosine(a2, A), 0.69, places=4)
        self.assertGreater(S.fakes.cosine(a, a2), 0.92)
        self.sb.build_serving([
            {"id": "a", "text": "알파 설명 문장입니다", "vec": a, "event_time": NOW_EV, "strength": 0.5},
            {"id": "a2", "text": "알파 설명 문장 다른 표현", "vec": a2, "event_time": NOW_EV, "strength": 0.5},
            item("b", "베타 설명 문장입니다", 0.66)])
        block = self.fresh().prefetch(Q)
        self.assertEqual(block.count("알파"), 1)
        self.assertIn("베타", block)

    def test_strength_pinned_keyword_scoring(self):
        self.sb.build_serving([
            item("weak", "알파 설명 문장입니다", 0.60, strength=0.0),
            item("strong", "베타 설명 문장입니다", 0.58, strength=1.0)])
        block = self.fresh().prefetch(Q)
        self.assertLess(block.index("베타"), block.index("알파"))   # 0.58+0.08 > 0.60
        sb2 = S.Sandbox("kw", config={"embed_base_url": self.srv.base_url})
        sb2.build_serving([item("plain", "알파 설명 문장입니다", 0.60),
                           item("kw", "Orion 관련 설명입니다", 0.58)])
        b2 = S.new_provider(sb2).prefetch(Q)
        self.assertLess(b2.index("Orion"), b2.index("알파"))  # keyword_hit +0.03

    def test_pinned_floor_and_prompt_pins(self):
        self.sb.build_serving([item("pin", "Orion 요금 질문은 항상 요금표부터 확인한다.", 0.36,
                                    pinned=True, tier="pinned", kind="rule")],
                              pins=[{"id": "pin", "text": "Orion 요금 질문은 항상 요금표부터 확인한다.",
                                     "label": "Orion 요금표"}])
        p = self.fresh()
        self.assertIn("요금표부터", p.prefetch(Q))            # not in prompt → floor 0.33
        p2 = S.new_provider(self.sb, session_id="s2")
        self.assertIn("요금표부터", p2.system_prompt_block())   # now in [고정 기억]
        p2.on_turn_start(1, Q)
        self.assertEqual(p2.prefetch(Q), "")                 # no duplicate of the prompt pin

    def test_expired_valid_until_filtered(self):
        self.sb.build_serving([item("old", "지난 일정 설명", 0.70, kind="schedule", valid_until=time.time() - 60),
                               item("ok", "현재 일정 설명", 0.69, kind="schedule",
                                    valid_until=time.time() + 86400)])
        block = self.fresh().prefetch(Q)
        self.assertNotIn("지난", block)
        self.assertIn("현재", block)

    def test_gating_chat_type_platform_cwd_synthetic_disabled(self):
        self.sb.build_serving([item("a", "알파 설명 문장입니다", 0.70)])
        self.assertEqual(S.new_provider(self.sb, chat_type="group").prefetch(Q), "")
        self.assertEqual(S.new_provider(self.sb, chat_type="group").system_prompt_block(), "")
        self.assertIn("알파", S.new_provider(self.sb, chat_type="dm", session_id="sess-dm").prefetch(Q))
        self.assertEqual(S.new_provider(self.sb, platform="discord", session_id="sess-dc").prefetch(Q), "")
        self.assertIn("알파", S.new_provider(self.sb, platform="cron", session_id="sess-cron").prefetch(Q))
        syn = S.new_provider(self.sb)
        syn.on_turn_start(1, "[synthetic-eval] You are a worker in a scripted test run. " + Q)
        self.assertEqual(syn.prefetch(Q), "")
        self.sb.write_config(deny_cwd_globs=[os.getcwd()])
        self.assertEqual(S.new_provider(self.sb).prefetch(Q), "")
        self.sb.write_config(deny_cwd_globs=[], enabled=False)
        self.assertEqual(S.new_provider(self.sb).prefetch(Q), "")

    def test_shadow_mode(self):
        self.sb.build_serving([item("a", "알파 설명 문장입니다", 0.70)])
        self.sb.write_config(inject=False)
        p = self.fresh()
        self.assertEqual(p.prefetch(Q), "")
        self.assertIsNone(p.recall_status())
        p.shutdown()
        self.assertEqual([e[:2] for e in self.sb.events()], [("a", "shadow")])

    def test_recall_status_per_platform(self):
        self.sb.build_serving([item("a", "알파 설명 문장입니다", 0.70)])
        p = self.fresh()
        p.prefetch(Q)
        st = p.recall_status()
        self.assertEqual((st.provider_label, st.count), ("Yume", 1))
        t = S.new_provider(self.sb, platform="telegram", chat_type="dm", session_id="sess-tg")
        t.on_turn_start(1, Q)
        self.assertIn("알파", t.prefetch(Q))
        self.assertIsNone(t.recall_status())

    def test_snapshot_reload_on_replace(self):
        self.sb.build_serving([item("a", "알파 설명 문장입니다", 0.70)], run_id="r1")
        p = self.fresh()
        self.assertIn("알파", p.prefetch(Q))
        self.sb.build_serving([item("b", "베타 설명 문장입니다", 0.70)], run_id="r2")
        p.on_session_switch("s2", reset=True)
        p.on_turn_start(1, Q)
        self.assertIn("베타", p.prefetch(Q))
        p.shutdown()
        self.assertEqual(self.sb.live_rows("SELECT snapshot_run FROM recall_events ORDER BY id"),
                         [("r1",), ("r2",)])


class Failures(_Base):
    """P2: every failure → "" or keyword results, no exception, < 4 s."""

    def timed(self, p, q=Q):
        t0 = time.monotonic()
        out = p.prefetch(q)
        dt = time.monotonic() - t0
        self.assertLess(dt, 4.0)
        return out, dt

    def test_no_serving_copy(self):
        out, _ = self.timed(self.fresh())
        self.assertEqual(out, "")
        self.assertEqual(self.srv.requests, [])      # nothing to recall → no embedding call

    def test_corrupt_serving_copy(self):
        self.sb.recall_sqlite.write_bytes(b"this is not a sqlite database" * 100)
        out, _ = self.timed(self.fresh())
        self.assertEqual(out, "")

    def test_truncated_serving_copy(self):
        self.sb.build_serving([item("a", "알파 설명 문장입니다", 0.70)])
        data = self.sb.recall_sqlite.read_bytes()
        self.sb.recall_sqlite.write_bytes(data[: len(data) // 3])
        out, _ = self.timed(self.fresh())
        self.assertIsInstance(out, str)

    def test_embedding_timeout_falls_back_to_fts(self):
        self.sb.build_serving([item("a", "Orion 결제 스테이징 서버 포트는 8081이다.", 0.70)])
        self.srv.delay_s = 6.0
        out, dt = self.timed(self.fresh())
        self.assertIn("8081", out)
        self.assertIn("(키워드)", out)
        self.assertGreater(dt, 2.5)

    def test_401_falls_back_to_fts_and_breaker(self):
        self.srv.require_key = "the-right-key"
        self.sb.build_serving([item("a", "Orion 결제 스테이징 서버 포트는 8081이다.", 0.70)])
        for i in range(4):
            p = S.new_provider(self.sb, session_id="s%d" % i)
            out, _ = self.timed(p)
            self.assertIn("(키워드)", out)
        self.assertEqual(len(self.srv.requests), 3)      # breaker open after 3 failures
        p.shutdown()
        h = self.sb.live_rows("SELECT SUM(prefetch_n), SUM(embed_fail_n), SUM(fts_fallback_n) FROM health")
        self.assertEqual(h, [(4, 3, 4)])
        errs = [r[0] for r in self.sb.live_rows("SELECT last_error_class FROM health")]
        self.assertTrue(any(e and ("http_401" in e or "breaker" in e) for e in errs), errs)

    def test_no_api_key(self):
        (self.sb.home / ".env").write_text("OTHER=1\n", encoding="utf-8")
        self.sb.build_serving([item("a", "Orion 결제 스테이징 서버 포트는 8081이다.", 0.70)])
        out, _ = self.timed(self.fresh())
        self.assertIn("(키워드)", out)
        self.assertEqual(self.srv.requests, [])

    def test_live_db_locked(self):
        self.sb.build_serving([item("a", "Orion 결제 스테이징 서버 포트는 8081이다.", 0.70)])
        ls = S.yume_mod("live_schema")
        holder = ls.connect(str(self.sb.live_db))
        holder.execute("BEGIN EXCLUSIVE")
        try:
            p = self.fresh()
            out, _ = self.timed(p)
            self.assertIn("8081", out)
            t0 = time.monotonic()
            p.sync_turn(Q, "8081번입니다")
            p.on_memory_write("add", "user", "**새 항목:** 테스트")
            p.on_session_end([])
            self.assertLess(time.monotonic() - t0, 3.0)
            res = p.handle_tool_call("yume_remember", {"text": "잠금 중 기억 요청 테스트 문장"})
            self.assertIn('"ok": false', res)
        finally:
            holder.rollback()
            holder.close()
        p.on_session_switch("s2", reset=True)
        p.on_turn_start(1, Q)
        self.assertIn("8081", p.prefetch(Q))
        p.shutdown()
        self.assertTrue(self.sb.events("injected"))

    def test_exceptions_never_escape(self):
        p = self.fresh()
        mod = S.provider_module()
        orig = mod.serving.get_index
        mod.serving.get_index = lambda home: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            self.assertEqual(p.prefetch(Q), "")
            self.assertEqual(p.system_prompt_block(), "")
            self.assertIn('"ok": false', p.handle_tool_call("yume_search", {"query": "x"}))
        finally:
            mod.serving.get_index = orig
        self.assertIn('"ok": false', p.handle_tool_call("yume_search", None))
        self.assertIn('"ok": false', p.handle_tool_call("nope", {}))
        p.sync_turn(None, None)
        p.on_turn_start(None, None)
        p.on_memory_write(None, None, None)
        p.on_session_switch(None)
        p.on_session_end(None)


class Performance(unittest.TestCase):
    """P4: stage 1 (256-d) + stage 2 rerank: 2,000 rows < 150 ms, 5,000 rows < 300 ms."""

    def build(self, n):
        rng = random.Random(n)
        sb = S.Sandbox("perf%d" % n)
        items = [{"id": "r%05d" % i, "text": "기억 %d" % i,
                  "vec": [rng.random() - 0.5 for _ in range(S.DIM)], "strength": 0.5}
                 for i in range(n)]
        sb.build_serving(items)
        serving = S.yume_mod("serving")
        idx = serving.ServingIndex.load(str(sb.recall_sqlite))
        self.assertEqual(len(idx.active_ids), n)
        return idx

    def best_of(self, idx, q, runs=3):
        best = 1e9
        for _ in range(runs):
            t0 = time.perf_counter()
            res = idx.vector_search(q, exclude=set(), now=time.time(), k1=64)
            best = min(best, time.perf_counter() - t0)
        self.assertEqual(len(res), 64)
        return best

    def test_2000_and_5000(self):
        q = S.unit([random.Random(7).random() - 0.5 for _ in range(S.DIM)])
        t2 = self.best_of(self.build(2000), q)
        t5 = self.best_of(self.build(5000), q)
        sys.stderr.write("\n[P4] 2000 rows %.1f ms, 5000 rows %.1f ms\n" % (t2 * 1000, t5 * 1000))
        self.assertLess(t2, 0.150)
        self.assertLess(t5, 0.300)


if __name__ == "__main__":
    unittest.main()
