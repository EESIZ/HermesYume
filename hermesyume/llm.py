"""LLM client for the dream side: OpenAI-compatible chat completions in JSON mode.

- `response_format={"type":"json_object"}`, temperature 0 by default
- retry with backoff 1s/4s/16s on 429/5xx/timeouts/connection errors
- 401/403 → ``LLMAuthError`` immediately (the run aborts and alerts; PLAN §4.2 N0)
- every call is charged to the RunBudget *before* it is sent
- ``parse_llm_json`` (v1 ``_parse_llm_json``, extended to arrays) never raises

Implementations: ``OpenAILLM`` (urllib, stdlib), ``DeepSeekLLM`` (same wire format, DeepSeek
endpoint), ``offline.NullLLM`` and ``tests.fakes.ScriptedLLM``. All share the ``LLM`` protocol;
`kind` names the prompt type (``prompts.PROMPT_KINDS``) so fakes can route by it and usage can be
accounted per kind.

Provider choice (``resolve_llm``, DEVIATIONS PR-1): ``cfg.llm_provider`` "auto" = deepseek when
DEEPSEEK_API_KEY is available, else openai. Keys and base URLs are read by *name* from
``$HERMES_HOME/.env`` (then the process environment); a DeepSeek key is only ever sent to
DEEPSEEK_BASE_URL or https://api.deepseek.com (``llm_base_url`` applies to openai only).
"""

from __future__ import annotations

import json
import logging
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from .types import BudgetExceeded, RunBudget

log = logging.getLogger("hermesyume.llm")

BACKOFF_S = (1.0, 4.0, 16.0)
PING_KIND = "ping"   # preflight; excluded from budget and from RunStats.llm_calls
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """Call failed after retries (or non-retryable 4xx other than auth)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class LLMAuthError(LLMError):
    """401/403: abort the run immediately, write nothing, alert."""


@dataclass
class LLMResponse:
    text: str
    data: Any                        # parse_llm_json(text): dict | list | None
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""          # "length" = cut off at max_tokens


@dataclass
class LLMUsage:
    calls: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)

    @property
    def content_calls(self) -> int:
        """Calls excluding preflight pings (what RunStats.llm_calls reports)."""
        return self.calls - self.by_kind.get(PING_KIND, 0)

    def add(self, kind: str, resp: LLMResponse | None, failed: bool = False) -> None:
        self.calls += 1
        self.by_kind[kind] = self.by_kind.get(kind, 0) + 1
        if failed:
            self.failures += 1
        if resp is not None:
            self.prompt_tokens += resp.prompt_tokens
            self.completion_tokens += resp.completion_tokens


@runtime_checkable
class LLM(Protocol):
    usage: LLMUsage

    def chat_json(self, kind: str, messages: list[dict], *, model: str | None = None,
                  max_tokens: int = 400, temperature: float = 0.0,
                  json_mode: bool = True) -> LLMResponse:
        """Raises LLMAuthError, LLMError, BudgetExceeded. `data` may be None (unparseable)."""
        ...

    def ping(self, model: str | None = None) -> None:
        """1-token call (N0 / doctor). Raises LLMAuthError / LLMError."""
        ...


def parse_llm_json(raw: str | None) -> Any:
    """Parse JSON from LLM output (code fences, surrounding prose). dict | list | None."""
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if "```" in cleaned:
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(open_c), cleaned.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                continue
    log.warning("LLM JSON parse failed (len=%d)", len(raw))
    return None


_parse_llm_json = parse_llm_json  # v1 name kept


class OpenAILLM:
    """OpenAI(-compatible) /chat/completions via urllib."""

    provider = "openai"
    key_name = "OPENAI_API_KEY"

    def __init__(self, api_key: str | None, base_url: str, default_model: str, *,
                 budget: RunBudget | None = None, timeout: float = 60.0,
                 tokens_param: str = "max_completion_tokens",
                 sleep: Callable[[float], None] = time.sleep,
                 backoff: tuple[float, ...] = BACKOFF_S):
        self._key = api_key or ""
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.budget = budget
        self.timeout = timeout
        self.tokens_param = tokens_param
        self._sleep = sleep
        self._backoff = backoff
        self.usage = LLMUsage()

    def __repr__(self) -> str:  # never show the key
        return f"{type(self).__name__}(base_url={self.base_url!r}, model={self.default_model!r})"

    # hooks for compatible servers (DeepSeekLLM)
    def model_for(self, model: str | None) -> str:
        return model or self.default_model

    def _body(self, model: str, messages: list[dict], max_tokens: int, temperature: float,
              json_mode: bool) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                                self.tokens_param: max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def _post(self, body: dict) -> dict:
        if not self._key:
            raise LLMAuthError(f"{self.key_name} 없음", status=401)
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            method="POST",
        )
        attempts = len(self._backoff) + 1
        last: Exception | None = None
        for i in range(attempts):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                status = e.code
                if status in (401, 403):
                    raise LLMAuthError(f"LLM({self.provider}) 인증 실패 HTTP {status} — {self.key_name} 확인",
                                       status=status) from None
                if status not in RETRY_STATUS:
                    raise LLMError(f"LLM HTTP {status}", status=status) from None
                last = LLMError(f"LLM HTTP {status}", status=status)
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError,
                    json.JSONDecodeError) as e:
                last = LLMError(f"LLM 연결 오류: {type(e).__name__}")
            if i < attempts - 1:
                self._sleep(self._backoff[i])
        assert last is not None
        raise last

    def chat_json(self, kind: str, messages: list[dict], *, model: str | None = None,
                  max_tokens: int = 400, temperature: float = 0.0,
                  json_mode: bool = True) -> LLMResponse:
        if self.budget is not None and kind != PING_KIND:
            self.budget.take_llm(1)        # raises BudgetExceeded; preflight pings are free
        m = self.model_for(model)
        try:
            data = self._send(m, messages, max_tokens, temperature, json_mode)
        except LLMError:
            self.usage.add(kind, None, failed=True)
            raise
        try:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
            finish = str(choice.get("finish_reason") or "")
        except (KeyError, IndexError, TypeError, AttributeError):
            text, finish = "", ""
        u = data.get("usage") or {}
        resp = LLMResponse(text=text, data=parse_llm_json(text) if json_mode else None, model=m,
                           prompt_tokens=int(u.get("prompt_tokens") or 0),
                           completion_tokens=int(u.get("completion_tokens") or 0),
                           finish_reason=finish)
        self.usage.add(kind, resp)
        return resp

    def _send(self, model: str, messages: list[dict], max_tokens: int, temperature: float,
              json_mode: bool) -> dict:
        return self._post(self._body(model, messages, max_tokens, temperature, json_mode))

    def ping(self, model: str | None = None) -> None:
        self.chat_json(PING_KIND, [{"role": "user", "content": "ping"}], model=model,
                       max_tokens=1, json_mode=False)


class DeepSeekLLM(OpenAILLM):
    """DeepSeek's OpenAI-compatible /chat/completions.

    - thinking is switched off explicitly (DeepSeek V4 turns it on when the toggle is omitted;
      these are short extraction/judgement calls that want the plain JSON answer)
    - ``max_tokens`` (not max_completion_tokens)
    - the config's model names (extract_model / judge_model) are OpenAI names unless they start
      with "deepseek"; anything else is replaced by ``default_model`` (cfg.deepseek_model)
    - JSON mode (``response_format`` json_object) is used; if the server rejects it with HTTP 400 the
      call is retried once without it and every later call relies on ``parse_llm_json`` alone
    """

    provider = "deepseek"
    key_name = "DEEPSEEK_API_KEY"
    THINKING_OFF = {"type": "disabled"}

    def __init__(self, api_key: str | None, base_url: str, default_model: str, **kw: Any):
        kw.setdefault("tokens_param", "max_tokens")
        super().__init__(api_key, base_url, default_model, **kw)
        self.json_mode_ok = True

    def model_for(self, model: str | None) -> str:
        return model if model and model.lower().startswith("deepseek") else self.default_model

    def _body(self, model: str, messages: list[dict], max_tokens: int, temperature: float,
              json_mode: bool) -> dict[str, Any]:
        body = super()._body(model, messages, max_tokens, temperature,
                             json_mode and self.json_mode_ok)
        body["thinking"] = dict(self.THINKING_OFF)
        return body

    def _send(self, model: str, messages: list[dict], max_tokens: int, temperature: float,
              json_mode: bool) -> dict:
        try:
            return super()._send(model, messages, max_tokens, temperature, json_mode)
        except LLMError as e:
            if not (json_mode and self.json_mode_ok and e.status == 400):
                raise
            log.warning("DeepSeek가 JSON 모드를 거부함 — JSON 모드 없이 다시 시도")
            self.json_mode_ok = False
            return super()._send(model, messages, max_tokens, temperature, json_mode)


@dataclass(frozen=True)
class LLMSettings:
    """Resolved LLM provider. Holds key *names* and where they come from, never values."""
    setting: str                     # cfg.llm_provider as written ("auto", "openai", "deepseek")
    provider: str                    # "openai" | "deepseek"
    model: str                       # default model for this provider
    base_url: str
    key_name: str
    key_source: str | None           # secrets_env.SOURCE_* or None (missing)

    @property
    def key_present(self) -> bool:
        return self.key_source is not None


def resolve_llm(cfg: Any, paths: Any, *, env: Mapping[str, str] | None = None) -> LLMSettings:
    from .config import LLM_PROVIDERS
    from .secrets_env import (DEEPSEEK_API_KEY, OPENAI_API_KEY, deepseek_base_url, openai_base_url,
                              secret_source)
    setting = str(getattr(cfg, "llm_provider", "auto") or "auto").strip().lower()
    ds_src = secret_source(DEEPSEEK_API_KEY, paths, env=env)
    prov = ("deepseek" if ds_src else "openai") if setting == "auto" else setting
    if prov == "deepseek":
        return LLMSettings(setting, "deepseek", str(cfg.deepseek_model), deepseek_base_url(paths, env=env),
                           DEEPSEEK_API_KEY, ds_src)
    if prov != "openai":
        raise LLMError(f"알 수 없는 llm_provider: {setting!r} ({', '.join(LLM_PROVIDERS)})")
    return LLMSettings(setting, "openai", str(cfg.extract_model),
                       openai_base_url(cfg.llm_base_url, paths, env=env), OPENAI_API_KEY,
                       secret_source(OPENAI_API_KEY, paths, env=env))


def make_llm(cfg: Any, paths: Any, *, budget: RunBudget | None = None,
             offline: bool = False) -> "LLM":
    """Factory used by the CLI: offline → offline.NullLLM; else the provider ``resolve_llm`` picks,
    with its key read by name from $HERMES_HOME/.env (then the environment)."""
    if offline:
        from .offline import NullLLM
        return NullLLM()
    from .secrets_env import get_secret
    st = resolve_llm(cfg, paths)
    key = get_secret(st.key_name, paths)
    if st.provider == "deepseek":
        return DeepSeekLLM(key, st.base_url, st.model, budget=budget, timeout=float(cfg.llm_timeout_s))
    return OpenAILLM(key, st.base_url, st.model, budget=budget, timeout=float(cfg.llm_timeout_s),
                     tokens_param=cfg.llm_tokens_param)


__all__ = ["LLM", "LLMAuthError", "LLMError", "LLMResponse", "LLMUsage", "OpenAILLM", "DeepSeekLLM",
           "LLMSettings", "resolve_llm", "parse_llm_json", "BudgetExceeded", "make_llm", "PING_KIND"]
