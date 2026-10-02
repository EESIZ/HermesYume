"""Deterministic fakes for tests. Module top is stdlib-only so provider tests (Hermes venv, no
numpy) can import ``hash_embed``, ``FakeOpenAIServer`` and the response builders.

- ``hash_embed(text, dim)``      stdlib unit vector (= hermesyume.offline.hash_embed_list)
- ``FakeEmbedder``               dream ``Embedder`` (numpy) with exact-cosine overrides and failure injection
- ``ScriptedLLM``                dream ``LLM`` routed by prompt kind ("extract", "judge", …)
- ``FakeOpenAIServer``           local HTTP server for /v1/embeddings and /v1/chat/completions
- ``claim()``, ``extract_json()``, ``judge_json()``  response builders matching PLAN §4.2/§4.3 shapes
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hermesyume.offline import cosine_list, hash_embed_list  # noqa: E402  (stdlib-only module top)

DEFAULT_DIM = 1536
DEFAULT_MODEL_ID = "openai/text-embedding-3-small@1536"


def hash_embed(text: str, dim: int = DEFAULT_DIM) -> list[float]:
    return hash_embed_list(text, dim)


def cosine(a, b) -> float:
    return cosine_list(list(a), list(b))


def _orthonormal_to(u: list[float], seed: str) -> list[float]:
    """Deterministic unit vector orthogonal to unit vector u."""
    w = hash_embed_list("__ortho__" + seed, len(u))
    d = sum(x * y for x, y in zip(w, u))
    w = [x - d * y for x, y in zip(w, u)]
    n = math.sqrt(sum(x * x for x in w)) or 1.0
    return [x / n for x in w]


def vector_with_cos(anchor: list[float], cos: float, seed: str) -> list[float]:
    """Unit vector v with cos(v, anchor) == cos exactly (anchor must be unit)."""
    w = _orthonormal_to(anchor, seed)
    s = math.sqrt(max(0.0, 1.0 - cos * cos))
    return [cos * a + s * b for a, b in zip(anchor, w)]


# ── embedder ─────────────────────────────────────────────────────────────────

class FakeEmbedder:
    """``hermesyume.embedder.Embedder`` protocol. Overrides take precedence over hashing:
    ``set_vector(text, vec)``, ``alias(text, other)``, ``pin_cos(text, anchor, cos)``.
    Failure injection: ``fail_with = EmbedAuthError(...)`` (raised on next call; sticky unless
    ``fail_once``)."""

    def __init__(self, dim: int = DEFAULT_DIM, model_id: str | None = None, budget: Any = None):
        from hermesyume.embedder import EmbedUsage
        self.dim = dim
        self.model_id = model_id or f"openai/text-embedding-3-small@{dim}"
        self.budget = budget
        self.usage = EmbedUsage()
        self.calls: list[list[str]] = []
        self._over: dict[str, list[float]] = {}
        self.fail_with: Exception | None = None
        self.fail_once = False

    def base(self, text: str) -> list[float]:
        return self._over.get(text) or hash_embed_list(text, self.dim)

    def set_vector(self, text: str, vec) -> None:
        v = [float(x) for x in vec]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        self._over[text] = [x / n for x in v]

    def alias(self, text: str, other: str) -> None:
        self._over[text] = list(self.base(other))

    def pin_cos(self, text: str, anchor_text: str, cos: float) -> None:
        self._over[text] = vector_with_cos(self.base(anchor_text), cos, seed=text)

    def vector(self, text: str):
        import numpy as np
        return np.asarray(self.base(text), dtype=np.float32)

    def embed(self, texts, *, kind: str = "content"):
        import numpy as np
        texts = list(texts)
        if self.fail_with is not None:
            exc = self.fail_with
            if self.fail_once:
                self.fail_with = None
            self.usage.failures += 1
            raise exc
        if self.budget is not None and kind != "ping" and texts:
            self.budget.take_embed(len(texts))
        self.calls.append(texts)
        self.usage.calls += 1
        self.usage.inputs += len(texts)
        self.usage.by_kind[kind] = self.usage.by_kind.get(kind, 0) + len(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.vector(t) for t in texts]).astype(np.float32)

    def ping(self) -> None:
        self.embed(["ping"], kind="ping")


# ── LLM ──────────────────────────────────────────────────────────────────────

Responder = Callable[[list[dict]], Any]   # messages -> dict | list | str | Exception


class Truncated:
    """Queue item for ScriptedLLM: an answer cut off at max_tokens (finish_reason "length")."""

    def __init__(self, text: str):
        self.text = text


class ScriptedLLM:
    """``hermesyume.llm.LLM`` protocol routed by `kind`.

    Resolution per call: queued responses for the kind (FIFO) → handler set via ``on(kind, fn)`` →
    ``DEFAULTS[kind]``. A response may be a dict/list (serialized), a raw str (parsed leniently,
    may be invalid JSON on purpose), or an Exception instance (raised).
    """

    DEFAULTS: dict[str, Any] = {
        "extract": {"claims": []},
        "extract_retry": {"claims": []},
        "extract_long": {"claims": []},
        "judge": {"relations": []},
        "judge_enum": {"type": "unknown", "newer": "same"},
        "consolidate": {"text": ""},
        "core_classify": {"items": []},
        "ping": "",
    }

    def __init__(self, budget: Any = None):
        from hermesyume.llm import LLMUsage
        self.usage = LLMUsage()
        self.budget = budget
        self.calls: list[dict] = []
        self._queues: dict[str, deque] = defaultdict(deque)
        self._handlers: dict[str, Responder] = {}

    def queue(self, kind: str, *responses: Any) -> "ScriptedLLM":
        self._queues[kind].extend(responses)
        return self

    def on(self, kind: str, fn: Responder) -> "ScriptedLLM":
        self._handlers[kind] = fn
        return self

    def calls_of(self, kind: str) -> list[dict]:
        return [c for c in self.calls if c["kind"] == kind]

    def chat_json(self, kind: str, messages: list[dict], *, model: str | None = None,
                  max_tokens: int = 400, temperature: float = 0.0, json_mode: bool = True):
        from hermesyume.llm import LLMResponse, parse_llm_json
        if self.budget is not None and kind != "ping":
            self.budget.take_llm(1)
        self.calls.append({"kind": kind, "messages": messages, "model": model,
                           "max_tokens": max_tokens, "temperature": temperature})
        if self._queues[kind]:
            r = self._queues[kind].popleft()
        elif kind in self._handlers:
            r = self._handlers[kind](messages)
        else:
            r = self.DEFAULTS.get(kind, {})
        if isinstance(r, BaseException):
            self.usage.add(kind, None, failed=True)
            raise r
        finish = "length" if isinstance(r, Truncated) else "stop"
        if isinstance(r, Truncated):
            r = r.text
        text = r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)
        resp = LLMResponse(text=text, data=parse_llm_json(text) if json_mode else None,
                           model=model or "scripted", prompt_tokens=len(json.dumps(messages, ensure_ascii=False)) // 4,
                           completion_tokens=len(text) // 4, finish_reason=finish)
        self.usage.add(kind, resp)
        return resp

    def ping(self, model: str | None = None) -> None:
        self.chat_json("ping", [], model=model, max_tokens=1, json_mode=False)


def claim(kind: str, text: str, *, subject: str = "", target: str = "user",
          event_time: str = "2026-09-28", valid_until: str | None = None, level: int | str = 3,
          evidence: list[str] | None = None, explicit: bool | str = False,
          steps: int | str | None = None) -> dict:
    """One EXTRACT claim in the exact §4.2 N3 output shape (strings where the prompt uses them)."""
    return {"kind": kind, "target": target, "subject": subject or text[:12], "text": text,
            "event_time": event_time, "valid_until": valid_until, "level": str(level),
            "evidence": evidence or [], "explicit": str(explicit).lower() if isinstance(explicit, bool) else explicit,
            "steps": None if steps is None else str(steps)}


def extract_json(*claims: dict) -> dict:
    return {"claims": list(claims)}


def judge_json(*relations: tuple[str, str, str]) -> dict:
    """judge_json(("<cand id>", "duplicate", "same"), …) → §4.3 R0-5 shape."""
    return {"relations": [{"id": i, "type": t, "newer": n} for i, t, n in relations]}


# ── local OpenAI-compatible HTTP server ──────────────────────────────────────

class FakeOpenAIServer:
    """Threaded local server. ``base_url`` = http://127.0.0.1:<port>/v1.

    - POST /v1/embeddings: hash embeddings (``dimensions`` honored, default 1536)
    - POST /v1/chat/completions: pops ``chat_responses`` (dict/str) else '{"claims": []}'
    - ``push_status(path_suffix, code, n=1)``: next n requests to that endpoint fail with code
    - ``delay_s``: sleep before answering (timeouts); ``require_key``: bearer must equal it, else 401
    - ``requests``: list of (path, json body, had_auth: bool) — never stores the key
    """

    def __init__(self, *, require_key: str | None = None):
        self.require_key = require_key
        self.chat_responses: deque = deque()
        self.status_queue: dict[str, deque] = defaultdict(deque)
        self.delay_s = 0.0
        self.requests: list[tuple[str, dict, bool]] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b"{}"
                try:
                    body = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    body = {}
                auth = self.headers.get("Authorization") or ""
                outer.requests.append((self.path, body, bool(auth)))
                if outer.delay_s:
                    time.sleep(outer.delay_s)
                key = "embeddings" if self.path.endswith("/embeddings") else "chat"
                q = outer.status_queue[key]
                if q:
                    code = q.popleft()
                    return self._send(code, {"error": {"message": f"scripted {code}"}})
                if outer.require_key is not None and auth != f"Bearer {outer.require_key}":
                    return self._send(401, {"error": {"message": "bad key"}})
                if key == "embeddings":
                    inputs = body.get("input") or []
                    if isinstance(inputs, str):
                        inputs = [inputs]
                    dim = int(body.get("dimensions") or DEFAULT_DIM)
                    data = [{"object": "embedding", "index": i, "embedding": hash_embed_list(t, dim)}
                            for i, t in enumerate(inputs)]
                    return self._send(200, {"object": "list", "data": data, "model": body.get("model"),
                                            "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)}})
                r = outer.chat_responses.popleft() if outer.chat_responses else {"claims": []}
                text = r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)
                return self._send(200, {"id": "x", "object": "chat.completion", "model": body.get("model"),
                                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                                     "finish_reason": "stop"}],
                                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

            def _send(self, code: int, obj: dict):
                data = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    def push_status(self, endpoint: str, code: int, n: int = 1) -> None:
        """endpoint: "embeddings" | "chat"."""
        self.status_queue[endpoint].extend([code] * n)

    def start(self) -> "FakeOpenAIServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def __enter__(self) -> "FakeOpenAIServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def sha_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── row helpers (dream side; numpy imported lazily) ─────────────────────────

def make_row(text: str, *, embedder: FakeEmbedder | None = None, subject: str = "",
             now: float | None = None, **kw: Any):
    """A committed-looking MemoryRow with a vector of embed_input(subject, text)."""
    from hermesyume.embedder import embed_input
    from hermesyume.types import KIND_BASE, MemoryRow, new_memory_id
    emb = embedder or FakeEmbedder()
    ts = time.time() if now is None else now
    kind = kw.pop("kind", "fact")
    defaults = dict(id=new_memory_id(), text=text, subject=subject or text[:12],
                    subject_key=(subject or text[:12]).replace(" ", "").lower(), kind=kind,
                    importance=KIND_BASE.get(kind, 0.5), embed_model=emb.model_id,
                    status="active", status_changed_at=ts, created_at=ts, updated_at=ts,
                    first_seen_at=ts, last_seen_at=ts, evidence_count=1)
    defaults.update(kw)
    row = MemoryRow(**defaults)
    if row.vector is None:
        row.vector = emb.vector(embed_input(row.subject, row.text))
    return row
