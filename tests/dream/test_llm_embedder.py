"""llm.py + embedder.py against a local fake OpenAI server: JSON mode, retry/backoff, 401 abort,
budget, batching, dimension guard, helpers."""

import numpy as np
import pytest

from hermesyume.embedder import (EmbedAuthError, EmbedError, OpenAIEmbedder, embed_input,
                                 l2_normalize, make_embedder, parse_model_id, prefix_renorm,
                                 strip_tails)
from hermesyume.llm import LLMAuthError, LLMError, OpenAILLM, make_llm, parse_llm_json
from hermesyume.types import BudgetExceeded, RunBudget
from tests.fakes import FakeOpenAIServer, cosine, hash_embed

KEY = "sk-test-" + "1" * 24


@pytest.fixture
def server():
    with FakeOpenAIServer(require_key=KEY) as s:
        yield s


def test_parse_llm_json():
    assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_llm_json('설명 {"claims": []} 끝') == {"claims": []}
    assert parse_llm_json('[{"x":1}]') == [{"x": 1}]
    assert parse_llm_json("not json") is None
    assert parse_llm_json("") is None
    assert parse_llm_json('{"a": ') is None


def test_chat_json_mode_and_usage(server):
    server.chat_responses.append({"claims": [{"kind": "fact"}]})
    sleeps = []
    llm = OpenAILLM(KEY, server.base_url, "gpt-4.1-mini", sleep=sleeps.append)
    r = llm.chat_json("extract", [{"role": "user", "content": "x"}], max_tokens=2000)
    assert r.data == {"claims": [{"kind": "fact"}]} and r.model == "gpt-4.1-mini"
    path, body, had_auth = server.requests[-1]
    assert path.endswith("/chat/completions") and had_auth
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0.0 and body["max_completion_tokens"] == 2000
    assert llm.usage.calls == 1 and llm.usage.by_kind == {"extract": 1} and sleeps == []


def test_retry_backoff_then_success(server):
    server.push_status("chat", 429)
    server.push_status("chat", 503)
    sleeps = []
    llm = OpenAILLM(KEY, server.base_url, "m", sleep=sleeps.append)
    r = llm.chat_json("judge", [{"role": "user", "content": "x"}])
    assert r.data == {"claims": []} and sleeps == [1.0, 4.0]


def test_retry_exhausted(server):
    server.push_status("chat", 500, n=4)
    sleeps = []
    llm = OpenAILLM(KEY, server.base_url, "m", sleep=sleeps.append)
    with pytest.raises(LLMError):
        llm.chat_json("judge", [])
    assert sleeps == [1.0, 4.0, 16.0] and llm.usage.failures == 1


def test_401_aborts_immediately(server):
    sleeps = []
    llm = OpenAILLM("sk-wrong-" + "2" * 24, server.base_url, "m", sleep=sleeps.append)
    with pytest.raises(LLMAuthError):
        llm.chat_json("extract", [])
    assert sleeps == [] and len(server.requests) == 1
    with pytest.raises(LLMAuthError):
        OpenAILLM(None, server.base_url, "m").chat_json("extract", [])


def test_non_retryable_400(server):
    server.push_status("chat", 400)
    with pytest.raises(LLMError) as ei:
        OpenAILLM(KEY, server.base_url, "m", sleep=lambda s: None).chat_json("extract", [])
    assert ei.value.status == 400 and not isinstance(ei.value, LLMAuthError)


def test_budget_and_ping_is_free(server):
    b = RunBudget(max_llm_calls=1)
    llm = OpenAILLM(KEY, server.base_url, "m", budget=b)
    llm.ping()
    assert b.llm_calls == 0 and llm.usage.content_calls == 0
    llm.chat_json("extract", [])
    with pytest.raises(BudgetExceeded):
        llm.chat_json("extract", [])
    assert len([r for r in server.requests if r[0].endswith("/chat/completions")]) == 2


def test_embedder_batches_dims_and_normalizes(server):
    b = RunBudget(max_embed_inputs=1000)
    e = OpenAIEmbedder(KEY, server.base_url, "text-embedding-3-small", 1536, batch=64, budget=b)
    texts = [f"문장 {i}" for i in range(130)]
    m = e.embed(texts)
    assert m.shape == (130, 1536) and m.dtype == np.float32
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0, atol=1e-5)
    reqs = [r for r in server.requests if r[0].endswith("/embeddings")]
    assert [len(r[1]["input"]) for r in reqs] == [64, 64, 2]
    assert reqs[0][1]["dimensions"] == 1536
    assert e.usage.content_inputs == 130 and b.embed_inputs == 130
    assert cosine(m[5], hash_embed("문장 5")) == pytest.approx(1.0, abs=1e-5)
    assert e.model_id == "openai/text-embedding-3-small@1536"
    assert e.embed([]).shape == (0, 1536)


def test_embedder_401_retry_dim(server):
    with pytest.raises(EmbedAuthError):
        OpenAIEmbedder("sk-bad-" + "3" * 24, server.base_url).embed(["x"])
    server.push_status("embeddings", 429)
    sleeps = []
    e = OpenAIEmbedder(KEY, server.base_url, sleep=sleeps.append)
    e.embed(["x"])
    assert sleeps == [1.0]
    wrong = OpenAIEmbedder(KEY, server.base_url, "custom-model", 1536)   # no dimensions → server 1536
    wrong.dim = 768
    with pytest.raises(EmbedError, match="차원"):
        wrong.embed(["x"])


def test_embedder_budget(server):
    b = RunBudget(max_embed_inputs=3)
    e = OpenAIEmbedder(KEY, server.base_url, budget=b)
    e.ping()
    e.embed(["a", "b", "c"])
    with pytest.raises(BudgetExceeded):
        e.embed(["d"])


def test_helpers():
    assert strip_tails("포트는 8081이다. 상세: docs/yume/a.md") == "포트는 8081이다."
    assert strip_tails("새 값 (prev: 옛 값)") == "새 값"
    assert strip_tails("(이전: 전부 꼬리)") == "(이전: 전부 꼬리)"     # never empties
    assert embed_input("Orion 포트", "8081이다. (ref: x)") == "Orion 포트: 8081이다."
    assert embed_input("", "본문") == "본문"
    v = l2_normalize(np.arange(1, 1537, dtype=np.float32))
    p = prefix_renorm(v)
    assert p.shape == (256,) and np.isclose(np.linalg.norm(p), 1.0)
    assert parse_model_id("openai/text-embedding-3-small@1536") == ("openai", "text-embedding-3-small", 1536)


def test_factories_offline(cfg, paths):
    llm = make_llm(cfg, paths, offline=True)
    assert llm.chat_json("extract", []).data == {"claims": []}
    emb = make_embedder(cfg, paths, offline=True)
    assert emb.model_id == cfg.embed_model_id() and emb.embed(["가"]).shape == (1, 1536)


def test_factory_reads_key_from_dotenv(cfg, paths, server):
    (paths.hermes_home / ".env").write_text(f"OPENAI_API_KEY={KEY}\n", encoding="utf-8")
    c = cfg.replace(llm_base_url=server.base_url, embed_base_url=server.base_url)
    assert make_llm(c, paths).chat_json("extract", []).data == {"claims": []}
    assert make_embedder(c, paths).embed(["x"]).shape == (1, 1536)
    assert KEY not in repr(make_llm(c, paths)) and KEY not in repr(make_embedder(c, paths))
