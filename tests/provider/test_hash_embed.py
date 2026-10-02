"""Local hash embeddings on the provider side (stdlib, Hermes venv): hash/ngram-v1@1024 needs no key
and no HTTP, its defaults/thresholds resolve like the dream side, and the serving index searches
non-text-embedding-3 models single-stage (exact cosine over the full vector)."""

import atexit
import importlib.util
import math
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from tests.provider import yume_support as S  # noqa: E402

atexit.register(S.cleanup)

HE_PATH = S.PROVIDER_SRC / "_yume" / "hash_embed.py"


def he():
    return S.yume_mod("hash_embed")


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class HashEmbedModule(unittest.TestCase):
    def test_stdlib_and_loadable_by_path(self):
        src = HE_PATH.read_text(encoding="utf-8")
        import re
        self.assertIsNone(re.search(r"^\s*from\s+\.", src, re.M))          # loadable by path
        self.assertIsNone(re.search(r"^\s*(import|from)\s+(numpy|requests)", src, re.M))
        spec = importlib.util.spec_from_file_location("_t_hash_embed", HE_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.embed("같은 문장"), he().embed("같은 문장"))

    def test_shape_norm_determinism(self):
        m = he()
        self.assertEqual(m.MODEL_ID, "hash/ngram-v1@1024")
        v = m.embed("사용자는 Neovim으로 코딩한다.")
        self.assertEqual(len(v), 1024)
        self.assertAlmostEqual(math.sqrt(dot(v, v)), 1.0, places=9)
        self.assertEqual(v, m.embed("사용자는 Neovim으로 코딩한다."))
        self.assertEqual(m.embed(""), [0.0] * 1024)
        self.assertTrue(any(m.embed("!!! ???")))           # symbols only → still a vector

    def test_frozen_algorithm_fingerprint(self):
        """ngram-v1 vectors are stored; any change to the algorithm needs a new model name."""
        v = he().embed("Project database is Postgres 17 on db.internal. 사용자는 간결한 답변을 선호한다.")
        nz = [(i, round(x, 6)) for i, x in enumerate(v) if x][:6]
        self.assertEqual(sum(1 for x in v if x), FINGERPRINT_NNZ)
        self.assertEqual(nz, FINGERPRINT_HEAD)

    def test_lexical_separation(self):
        m = he()
        th = m.MODEL_DEFAULTS
        sim = lambda a, b: dot(m.embed(a), m.embed(b))       # noqa: E731
        related = [("Project database is Postgres 17 on db.internal.",
                    "Project database is Postgres 16 on db.internal."),
                   ("사용자의 딸은 초등학교 2학년이다.", "사용자의 딸은 초등학교 3학년이다.")]
        unrelated = [("User's timezone is Asia/Seoul.", "The API rate limit is 100 requests per minute."),
                     ("사용자는 간결한 한국어 답변을 선호한다.", "배포는 docker-compose와 nginx로 한다.")]
        for a, b in related:
            self.assertGreater(sim(a, b), th["candidate_cos"], (a, b))
        for a, b in unrelated:
            self.assertLess(sim(a, b), th["recall_min_cos"], (a, b))
        self.assertGreater(sim("데모 서버 포트 몇 번이었지?", "데모 서버 포트는 8123이다."),
                           th["recall_min_cos"])

    def test_threshold_order(self):
        m = he()
        t = m.MODEL_DEFAULTS
        self.assertLess(t["search_min_cos"], t["pinned_min_cos"])
        self.assertLessEqual(m.RECALL_MIN_COS_FLOOR, t["pinned_min_cos"])
        self.assertLess(t["pinned_min_cos"], t["recall_min_cos"])
        self.assertLess(t["recall_min_cos"], t["injected_strong_cos"])
        self.assertLessEqual(t["candidate_cos"], t["sweep_cos"])
        self.assertLess(t["sweep_cos"], t["auto_dup_cos"])
        self.assertEqual(m.recall_min_cos_floor("openai"), 0.40)
        self.assertTrue(m.two_stage("openai/text-embedding-3-small@1536"))
        self.assertFalse(m.two_stage("hash/ngram-v1@1024"))
        self.assertFalse(m.two_stage("openai/nomic-embed-text@768"))


class ProviderConfig(unittest.TestCase):
    def setUp(self):
        S.reset_singletons()

    def test_auto_resolution_and_hash_defaults(self):
        cfgm = S.yume_mod("config")
        sb = S.Sandbox("cfg-auto", config={"recall_k": 4})      # .env has OPENAI_API_KEY
        cfg = cfgm.load(str(sb.home))
        self.assertEqual(cfgm.embed_model_id(cfg), "openai/text-embedding-3-small@1536")
        self.assertEqual(cfg["recall_min_cos"], 0.40)
        (sb.home / ".env").write_text("OPENAI_API_KEY=your_openai_api_key\n", encoding="utf-8")   # placeholder
        cfgm.clear_cache()
        cfg = cfgm.load(str(sb.home))
        self.assertEqual(cfgm.embed_model_id(cfg), "hash/ngram-v1@1024")
        self.assertEqual((cfg["embed_dim"], cfg["recall_min_cos"], cfg["mmr_cos"]), (1024, 0.30, 0.85))
        self.assertEqual(cfg["recall_k"], 4)
        sb.write_config(embed_provider="hash", recall_min_cos=0.27)       # explicit value wins
        cfg = cfgm.load(str(sb.home))
        self.assertEqual((cfg["recall_min_cos"], cfg["pinned_min_cos"]), (0.27, 0.25))
        sb.write_config(embed_provider="openai")
        cfg = cfgm.load(str(sb.home))
        self.assertEqual(cfgm.embed_model_id(cfg), "openai/text-embedding-3-small@1536")

    def test_env_change_invalidates_auto_cache(self):
        cfgm = S.yume_mod("config")
        sb = S.Sandbox("cfg-envcache")
        self.assertEqual(cfgm.load(str(sb.home))["embed_provider"], "openai")
        (sb.home / ".env").write_text("OTHER=1\n", encoding="utf-8")
        self.assertEqual(cfgm.load(str(sb.home))["embed_provider"], "hash")

    def test_cli_fallback_loads_config_by_path(self):
        spec = importlib.util.spec_from_file_location("_t_standalone_cfg", S.PROVIDER_SRC / "_yume" / "config.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.embed_model_id({"embed_provider": "hash"}), "hash/ngram-v1@1024")


class LocalEmbedding(unittest.TestCase):
    def setUp(self):
        S.reset_singletons()
        self.srv = S.fakes.FakeOpenAIServer().start()

    def tearDown(self):
        self.srv.stop()
        S.reset_singletons()

    def test_embed_http_hash_needs_no_key_and_no_request(self):
        eh = S.yume_mod("embed_http")
        q = "데모 서버 포트 몇 번이었지?"
        vec, err = eh.embed_ex(q, model_id="hash/ngram-v1@1024", base_url=self.srv.base_url, api_key=None,
                               connect_timeout=0.5, total_timeout=1.0)
        self.assertIsNone(err)
        self.assertEqual(vec, he().embed(q))                  # bit for bit, no renormalization
        self.assertEqual(self.srv.requests, [])
        self.assertEqual(eh.embed_ex("", model_id="hash/ngram-v1@1024", base_url="", api_key=None,
                                     connect_timeout=0.5, total_timeout=1.0), (None, "zero_vector"))
        self.assertEqual(eh.embed_ex("x", model_id="hash/ngram-v2@512", base_url="", api_key="k",
                                     connect_timeout=0.5, total_timeout=1.0)[1], "unknown_model")
        self.assertEqual(self.srv.requests, [])

    def test_prefetch_with_hash_model_end_to_end(self):
        m = he()
        mems = {"a": "데모 서버 포트는 8123이다.",
                "b": "사용자는 간결한 한국어 답변을 선호한다.",
                "c": "API 호출 한도는 분당 100회다."}
        sb = S.Sandbox("hash-e2e", config={"embed_provider": "hash", "embed_base_url": self.srv.base_url})
        (sb.home / ".env").write_text("OTHER=1\n", encoding="utf-8")       # no key at all
        sb.build_serving([{"id": k, "text": t, "vec": m.embed(t), "strength": 0.5} for k, t in mems.items()],
                         embed_model=m.MODEL_ID, dim=m.DIM)
        p = S.new_provider(sb)
        q = "데모 서버 포트 몇 번이었지?"
        p.on_turn_start(1, q)
        block = p.prefetch(q)
        p.shutdown()
        self.assertIn("8123", block)
        self.assertNotIn("간결한", block)
        self.assertEqual(self.srv.requests, [])
        inj = sb.events("injected")
        self.assertEqual([e[0] for e in inj], ["a"])
        self.assertAlmostEqual(inj[0][2], dot(m.embed(q), m.embed(mems["a"])), places=5)


class SingleStageServing(unittest.TestCase):
    def setUp(self):
        S.reset_singletons()

    def test_hash_index_is_single_stage_and_exact(self):
        m = he()
        texts = ["%s 항목 %d 설명 문장" % (w, i) for i, w in enumerate(
            ["서버", "포트", "고양이", "커피", "배포", "회의록", "블로그", "알레르기", "달리기", "커밋"] * 3)]
        sb = S.Sandbox("single-stage")
        items = [{"id": "m%02d" % i, "text": t, "vec": m.embed(t)} for i, t in enumerate(texts)]
        # a row whose first 256 dims are all zero would vanish from a 256-d prefilter
        zero_head = [0.0] * 256 + [1.0] + [0.0] * (m.DIM - 257)
        items.append({"id": "zz", "text": "머리 차원이 0인 행", "vec": zero_head})
        sb.build_serving(items, embed_model=m.MODEL_ID, dim=m.DIM)
        idx = S.yume_mod("serving").get_index(str(sb.home))
        self.assertFalse(idx.two_stage)
        self.assertTrue(idx.sparse)
        self.assertEqual(len(idx.active_ids), len(items))
        q = m.embed("고양이 커피 설명")
        got = idx.vector_search(q, exclude=set(), now=None, k1=5)
        exact = {it["id"]: dot(q, S.unit(it["vec"])) for it in items}
        brute = sorted(((c, i) for i, c in exact.items()), reverse=True)[:5]
        for (i, c), (bc, _bi) in zip(got, brute):        # same scores in the same order (ties may swap ids)
            self.assertAlmostEqual(c, bc, places=5)
            self.assertAlmostEqual(c, exact[i], places=5)
        self.assertEqual(len(got), 5)
        zq = [0.0] * 256 + [1.0] + [0.0] * (m.DIM - 257)
        self.assertEqual(idx.vector_search(zq, exclude=set(), now=None, k1=1)[0][0], "zz")
        res = idx.search_all(q, "", include_inactive=True, limit=3, min_cos=0.0)
        self.assertEqual([round(r[1], 5) for r in res], [round(c, 5) for c, _ in brute[:3]])

    def test_openai_index_stays_two_stage(self):
        sb = S.Sandbox("two-stage")
        a = S.anchor("질의")
        sb.build_serving([{"id": "x", "text": "기억", "vec": S.with_cos(a, 0.7, "x")}])
        idx = S.yume_mod("serving").get_index(str(sb.home))
        self.assertTrue(idx.two_stage)
        self.assertFalse(idx.sparse)
        self.assertEqual(len(idx.vec256[0]), 256)
        self.assertEqual(idx.vector_search(a, exclude=set(), now=None, k1=8)[0][0], "x")


FINGERPRINT_NNZ = 112
FINGERPRINT_HEAD = [(50, -0.077152), (65, -0.077152), (72, 0.154303), (98, 0.077152), (102, -0.154303),
                    (112, 0.154303)]

if __name__ == "__main__":
    unittest.main()
