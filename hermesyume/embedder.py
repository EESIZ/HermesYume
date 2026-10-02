"""Embeddings for the dream side (PLAN-v2 §4.2 N6).

Two implementations of the ``Embedder`` protocol, picked by the resolved ``cfg.embed_provider``:

- ``OpenAIEmbedder`` ("openai"): batches of `embed_batch` (64), base URL cfg.embed_base_url →
  $OPENAI_BASE_URL → api.openai.com/v1; retry 1s/4s/16s on 429/5xx/timeouts; 401/403 →
  ``EmbedAuthError`` (abort run, alert); wrong dimension → ``EmbedError``
- ``HashNgramEmbedder`` ("hash"): the local ``hash/ngram-v1@1024`` embedder, no key, no network.
  It runs ``provider/_yume/hash_embed.py`` loaded by path — the very file the provider embeds
  queries with — so stored and query vectors are identical bit for bit (as float32).

Both return float32 ``np.ndarray`` (n, dim) with L2-normalized rows. ``model_id`` (e.g.
"openai/text-embedding-3-small@1536", "hash/ngram-v1@1024") is what Lance rows, ledger meta, serving
meta and live.db inbox store; the ledger guard refuses a run whose configured model differs.
``prefix_renorm`` gives the serving ``vec256`` (first 256 dims re-normalized; only text-embedding-3
vectors are searched through it).
"""

from __future__ import annotations

import json
import logging
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from .types import RunBudget

log = logging.getLogger("hermesyume.embedder")

BACKOFF_S = (1.0, 4.0, 16.0)
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
PING_KIND = "ping"
PREFIX_DIM = 256
_TAIL_RE = re.compile(r"\s*(?:상세:|\(ref:|\(prev:|\(이전:)[\s\S]*$")


class EmbedError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class EmbedAuthError(EmbedError):
    pass


@dataclass
class EmbedUsage:
    calls: int = 0
    inputs: int = 0
    tokens: int = 0
    failures: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)

    @property
    def content_inputs(self) -> int:
        """Inputs excluding preflight pings (what RunStats.embed_inputs reports)."""
        return self.inputs - self.by_kind.get(PING_KIND, 0)


@runtime_checkable
class Embedder(Protocol):
    model_id: str
    dim: int
    usage: EmbedUsage

    def embed(self, texts: Sequence[str], *, kind: str = "content") -> np.ndarray:
        """(len(texts), dim) float32, L2-normalized rows. Raises EmbedAuthError/EmbedError/BudgetExceeded."""
        ...

    def ping(self) -> None:
        ...


def make_model_id(model: str, dim: int, provider: str = "openai") -> str:
    return f"{provider}/{model}@{int(dim)}"


def parse_model_id(model_id: str) -> tuple[str, str, int]:
    """"openai/text-embedding-3-small@1536" → ("openai", "text-embedding-3-small", 1536)."""
    prov, rest = model_id.split("/", 1)
    model, dim = rest.rsplit("@", 1)
    return prov, model, int(dim)


def l2_normalize(m: np.ndarray) -> np.ndarray:
    a = np.asarray(m, dtype=np.float32)
    if a.ndim == 1:
        n = float(np.linalg.norm(a))
        return a / n if n > 0 else a
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (a / n).astype(np.float32)


def prefix_renorm(vec: Any, k: int = PREFIX_DIM) -> np.ndarray:
    """First k dims, re-normalized (text-embedding-3 vectors are prefix-truncatable)."""
    return l2_normalize(np.asarray(vec, dtype=np.float32)[..., :k])


def strip_tails(text: str) -> str:
    """Drop pointer/history tails (`상세:`, `(ref:`, `(prev:`, `(이전:`) before embedding."""
    out = _TAIL_RE.sub("", text or "").strip()
    return out or (text or "").strip()


def embed_input(subject: str, text: str) -> str:
    """Canonical N6 embedding input: "subject: text" (tails stripped)."""
    body = strip_tails(text)
    subject = (subject or "").strip()
    return f"{subject}: {body}" if subject else body


class OpenAIEmbedder:
    def __init__(self, api_key: str | None, base_url: str, model: str = "text-embedding-3-small",
                 dim: int = 1536, *, provider: str = "openai", batch: int = 64,
                 timeout: float = 30.0, budget: RunBudget | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 backoff: tuple[float, ...] = BACKOFF_S):
        self._key = api_key or ""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = int(dim)
        self.model_id = make_model_id(model, dim, provider)
        self.batch = max(1, int(batch))
        self.timeout = timeout
        self.budget = budget
        self._sleep = sleep
        self._backoff = backoff
        self.usage = EmbedUsage()

    def __repr__(self) -> str:
        return f"OpenAIEmbedder(base_url={self.base_url!r}, model_id={self.model_id!r})"

    def _post(self, inputs: list[str]) -> dict:
        if not self._key:
            raise EmbedAuthError("OPENAI_API_KEY 없음", status=401)
        body: dict[str, Any] = {"model": self.model, "input": inputs, "encoding_format": "float"}
        if self.model.startswith("text-embedding-3"):
            body["dimensions"] = self.dim
        req = urllib.request.Request(
            f"{self.base_url}/embeddings", data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            method="POST")
        attempts = len(self._backoff) + 1
        last: Exception | None = None
        for i in range(attempts):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise EmbedAuthError(f"임베딩 인증 실패 HTTP {e.code}", status=e.code) from None
                if e.code not in RETRY_STATUS:
                    raise EmbedError(f"임베딩 HTTP {e.code}", status=e.code) from None
                last = EmbedError(f"임베딩 HTTP {e.code}", status=e.code)
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError,
                    json.JSONDecodeError) as e:
                last = EmbedError(f"임베딩 연결 오류: {type(e).__name__}")
            if i < attempts - 1:
                self._sleep(self._backoff[i])
        assert last is not None
        raise last

    def embed(self, texts: Sequence[str], *, kind: str = "content") -> np.ndarray:
        texts = [t if (t and t.strip()) else " " for t in texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(texts), self.batch):
            chunk = texts[start:start + self.batch]
            if self.budget is not None and kind != PING_KIND:
                self.budget.take_embed(len(chunk))
            self.usage.calls += 1
            self.usage.inputs += len(chunk)
            self.usage.by_kind[kind] = self.usage.by_kind.get(kind, 0) + len(chunk)
            try:
                data = self._post(chunk)
            except EmbedError:
                self.usage.failures += 1
                raise
            rows = sorted(data.get("data") or [], key=lambda r: r.get("index", 0))
            if len(rows) != len(chunk):
                raise EmbedError(f"임베딩 응답 개수 불일치 {len(rows)}/{len(chunk)}")
            for j, r in enumerate(rows):
                v = np.asarray(r.get("embedding") or [], dtype=np.float32)
                if v.shape != (self.dim,):
                    raise EmbedError(f"임베딩 차원 불일치 {v.shape} != ({self.dim},)")
                out[start + j] = v
            self.usage.tokens += int((data.get("usage") or {}).get("total_tokens") or 0)
        return l2_normalize(out) if len(texts) else out

    def ping(self) -> None:
        self.embed(["ping"], kind=PING_KIND)


class HashNgramEmbedder:
    """Local ``hash/ngram-v1@1024`` (see ``provider/_yume/hash_embed.py``). Free and offline, so
    nothing is charged to the run's embedding budget; usage is still counted for the stats."""

    def __init__(self, paths: Any = None, *, budget: RunBudget | None = None):
        from .paths import load_provider_module
        self._he = load_provider_module("hash_embed", paths)
        self.dim = int(self._he.DIM)
        self.model = self._he.MODEL
        self.model_id = self._he.MODEL_ID
        self.budget = budget            # kept for interface parity; never charged
        self.usage = EmbedUsage()

    def __repr__(self) -> str:
        return f"HashNgramEmbedder(model_id={self.model_id!r})"

    def embed(self, texts: Sequence[str], *, kind: str = "content") -> np.ndarray:
        texts = list(texts)
        self.usage.calls += 1
        self.usage.inputs += len(texts)
        self.usage.by_kind[kind] = self.usage.by_kind.get(kind, 0) + len(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(self._he.embed_many(texts), dtype=np.float32).reshape(len(texts), self.dim)

    def ping(self) -> None:
        self.embed(["ping"], kind=PING_KIND)


def make_embedder(cfg: Any, paths: Any, *, budget: RunBudget | None = None,
                  offline: bool = False) -> Embedder:
    """cfg.embed_provider (resolved by config): "hash" → HashNgramEmbedder (also with --offline: it
    is already network-free and deterministic); offline → offline.HashEmbedder posing as the
    configured model id; else OpenAIEmbedder."""
    if str(cfg.embed_provider) == "hash":
        emb = HashNgramEmbedder(paths, budget=budget)
        if cfg.embed_model_id() != emb.model_id:
            raise EmbedError(f"hash 임베더는 {emb.model_id}만 지원합니다 (설정: {cfg.embed_model_id()})")
        return emb
    if offline:
        from .offline import HashEmbedder
        return HashEmbedder(dim=int(cfg.embed_dim), model_id=cfg.embed_model_id(), budget=budget)
    from .secrets_env import OPENAI_API_KEY, get_secret, openai_base_url
    return OpenAIEmbedder(get_secret(OPENAI_API_KEY, paths), openai_base_url(cfg.embed_base_url, paths),
                          cfg.embed_model, int(cfg.embed_dim), provider=cfg.embed_provider,
                          batch=int(cfg.embed_batch), timeout=float(cfg.embed_timeout_s),
                          budget=budget)
