"""Query embedding over HTTP (stdlib only, PLAN-v2 §6.2.4).

- POST ``{base_url}/embeddings`` with the same body shape the dream embedder sends (model,
  input, encoding_format, ``dimensions`` for text-embedding-3*), so query and stored vectors match.
- Connect timeout and total timeout are separate; the total is a hard cap (worker thread + join).
- No retries. Module singletons (thread-safe): LRU cache and a process-wide circuit breaker
  (N consecutive failures → open for cooldown seconds).
- Never raises; never logs the key or the URL.
- A ``hash/...`` model id (``hash_embed``, the local n-gram embedder) is computed in-process: no
  key, no HTTP, no breaker. It is the same file the dream embeds with, so vectors match exactly.
"""

import http.client
import json
import logging
import math
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict

from . import hash_embed

log = logging.getLogger("hermesyume.provider.embed")

DEFAULT_BASE = "https://api.openai.com/v1"

_LOCK = threading.Lock()
_LRU = OrderedDict()
_lru_size = 256
_fail_count = 0
_open_until = 0.0
_breaker_failures = 3
_breaker_cooldown = 60.0
_last_warn = {}


def _warn(key, msg, *args):
    now = time.monotonic()
    with _LOCK:
        last = _last_warn.get(key, -1e9)
        if now - last < 60.0:
            return
        _last_warn[key] = now
    log.warning(msg, *args)


def configure(cfg):
    """Pick breaker/LRU parameters from the provider config dict."""
    global _lru_size, _breaker_failures, _breaker_cooldown
    try:
        with _LOCK:
            _lru_size = max(0, int(cfg.get("embed_lru_size", 256)))
            _breaker_failures = max(1, int(cfg.get("breaker_failures", 3)))
            _breaker_cooldown = float(cfg.get("breaker_cooldown_s", 60))
    except Exception:
        pass


def breaker_open():
    with _LOCK:
        return time.monotonic() < _open_until


def record_failure():
    global _fail_count, _open_until
    with _LOCK:
        _fail_count += 1
        if _fail_count >= _breaker_failures:
            _open_until = time.monotonic() + _breaker_cooldown
            _fail_count = 0


def record_success():
    global _fail_count, _open_until
    with _LOCK:
        _fail_count = 0
        _open_until = 0.0


def reset():
    """Tests: clear breaker + LRU."""
    global _fail_count, _open_until
    with _LOCK:
        _fail_count = 0
        _open_until = 0.0
        _LRU.clear()
        _last_warn.clear()


def parse_model_id(model_id):
    """"openai/text-embedding-3-small@1536" → ("openai", "text-embedding-3-small", 1536)."""
    prov, rest = str(model_id).split("/", 1)
    model, dim = rest.rsplit("@", 1)
    return prov, model, int(dim)


def _normalize(v):
    n = math.sqrt(sum(x * x for x in v))
    if n <= 0:
        return None
    return [x / n for x in v]


def _uses_proxy(url):
    try:
        parts = urllib.parse.urlsplit(url)
        proxies = urllib.request.getproxies()
        if parts.scheme not in proxies:
            return False
        return not urllib.request.proxy_bypass(parts.hostname or "")
    except Exception:
        return False


def _post_direct(url, body, headers, connect_timeout, deadline):
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname
    port = parts.port
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    if parts.scheme == "https":
        conn = http.client.HTTPSConnection(host, port, timeout=connect_timeout,
                                           context=ssl.create_default_context())
    else:
        conn = http.client.HTTPConnection(host, port, timeout=connect_timeout)
    try:
        conn.connect()
        conn.sock.settimeout(max(0.05, deadline - time.monotonic()))
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, data
    finally:
        conn.close()


def _post_urllib(url, body, headers, deadline):
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=max(0.05, deadline - time.monotonic())) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def _request(url, body, headers, connect_timeout, total_timeout):
    """(status, bytes) or raises. Hard total cap via a daemon worker thread."""
    deadline = time.monotonic() + total_timeout
    box = {}

    def run():
        try:
            if _uses_proxy(url):
                box["v"] = _post_urllib(url, body, headers, deadline)
            else:
                box["v"] = _post_direct(url, body, headers, connect_timeout, deadline)
        except BaseException as e:  # noqa: BLE001 - reported to caller
            box["e"] = e

    t = threading.Thread(target=run, daemon=True, name="yume-embed")
    t.start()
    t.join(total_timeout)
    if t.is_alive():
        raise TimeoutError("embedding total timeout")
    if "e" in box:
        raise box["e"]
    return box["v"]


def _embed_local(text, model_id):
    """hash/ngram-v1@1024 in-process (LRU-cached like the HTTP path)."""
    _prov, model, dim = parse_model_id(model_id)
    if model != hash_embed.MODEL or dim != hash_embed.DIM:
        _warn("hashmodel", "hermesyume 알 수 없는 로컬 임베딩 모델")
        return None, "unknown_model"
    key = (model_id, "", text)
    with _LOCK:
        hit = _LRU.get(key)
        if hit is not None:
            _LRU.move_to_end(key)
            return list(hit), None
    vec = hash_embed.embed(text or "")          # already unit length; kept bit-for-bit
    if not any(vec):
        return None, "zero_vector"
    with _LOCK:
        if _lru_size > 0:
            _LRU[key] = tuple(vec)
            while len(_LRU) > _lru_size:
                _LRU.popitem(last=False)
    return vec, None


def embed_ex(text, *, model_id, base_url, api_key, connect_timeout, total_timeout, cfg=None):
    """(vector | None, error_class | None). Never raises."""
    try:
        if cfg is not None:
            configure(cfg)
        if hash_embed.is_model_id(model_id):
            return _embed_local(text, model_id)
        if not api_key:
            return None, "no_key"
        _prov, model, dim = parse_model_id(model_id)
        base = (base_url or DEFAULT_BASE).rstrip("/")
        key = (model_id, base, text)
        with _LOCK:
            hit = _LRU.get(key)
            if hit is not None:
                _LRU.move_to_end(key)
                return list(hit), None
        if breaker_open():
            return None, "breaker_open"
        body = {"model": model, "input": [text or " "], "encoding_format": "float"}
        if model.startswith("text-embedding-3"):
            body["dimensions"] = dim
        headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
        try:
            status, raw = _request(base + "/embeddings",
                                   json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                   headers, float(connect_timeout), float(total_timeout))
        except (TimeoutError, socket.timeout):
            record_failure()
            _warn("timeout", "hermesyume 임베딩 시간 초과")
            return None, "timeout"
        except Exception as e:
            record_failure()
            _warn("conn", "hermesyume 임베딩 연결 오류: %s", type(e).__name__)
            return None, type(e).__name__
        if status != 200:
            record_failure()
            _warn("http%s" % status, "hermesyume 임베딩 HTTP %s", status)
            return None, "http_%s" % status
        try:
            data = json.loads(raw.decode("utf-8"))
            vec = [float(x) for x in data["data"][0]["embedding"]]
        except Exception as e:
            record_failure()
            _warn("parse", "hermesyume 임베딩 응답 해석 실패: %s", type(e).__name__)
            return None, "bad_response"
        if len(vec) != dim:
            record_failure()
            _warn("dim", "hermesyume 임베딩 차원 불일치 %d != %d", len(vec), dim)
            return None, "dim_mismatch"
        vec = _normalize(vec)
        if vec is None:
            record_failure()
            return None, "zero_vector"
        record_success()
        with _LOCK:
            if _lru_size > 0:
                _LRU[key] = tuple(vec)
                while len(_LRU) > _lru_size:
                    _LRU.popitem(last=False)
        return vec, None
    except Exception as e:  # pragma: no cover - defensive
        return None, type(e).__name__


def embed(text, *, model_id, base_url, api_key, connect_timeout, total_timeout, cfg=None):
    """list[float] (L2-normalized) or None on any failure."""
    return embed_ex(text, model_id=model_id, base_url=base_url, api_key=api_key,
                    connect_timeout=connect_timeout, total_timeout=total_timeout, cfg=cfg)[0]
