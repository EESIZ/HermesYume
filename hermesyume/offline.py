"""Deterministic network-free stand-ins (`yume dream --offline`, and the base of tests/fakes.py).

Module top is stdlib-only (provider tests in the Hermes venv import ``hash_embed_list``);
numpy is imported lazily inside ``HashEmbedder``.

Hash embedding: NFKC+casefold text → word tokens (weight 1.0) + character 2/3-grams over the
whitespace-collapsed string (weight 0.5) → signed feature hashing (blake2b) into `dim` buckets →
L2 normalize. Works for Korean (syllable n-grams); paraphrases sharing many n-grams score high,
unrelated texts ≈ 0.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from typing import Any, Sequence

_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣ㄱ-ㆎ]+")


def _features(text: str) -> list[tuple[str, float]]:
    t = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "").casefold()).strip()
    feats: list[tuple[str, float]] = [("w:" + w, 1.0) for w in _TOKEN_RE.findall(t)]
    compact = t
    for n in (2, 3):
        for i in range(max(0, len(compact) - n + 1)):
            g = compact[i:i + n]
            if g.strip():
                feats.append((f"g{n}:" + g, 0.5))
    return feats


def hash_embed_list(text: str, dim: int = 1536, *, salt: str = "yume") -> list[float]:
    """Deterministic unit vector (python list). Empty text → e0."""
    v = [0.0] * dim
    for feat, w in _features(text):
        h = hashlib.blake2b(f"{salt}|{feat}".encode("utf-8"), digest_size=8).digest()
        x = int.from_bytes(h, "little")
        v[x % dim] += w if (x >> 63) & 1 else -w
    n = math.sqrt(sum(a * a for a in v))
    if n == 0:
        v[0] = 1.0
        return v
    return [a / n for a in v]


def cosine_list(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


class HashEmbedder:
    """`Embedder` protocol implementation without network."""

    def __init__(self, dim: int = 1536, model_id: str | None = None, budget: Any = None):
        from .embedder import EmbedUsage
        self.dim = int(dim)
        self.model_id = model_id or f"openai/text-embedding-3-small@{self.dim}"
        self.budget = budget
        self.usage = EmbedUsage()

    def vector(self, text: str) -> Any:
        import numpy as np
        return np.asarray(hash_embed_list(text, self.dim), dtype=np.float32)

    def embed(self, texts: Sequence[str], *, kind: str = "content") -> Any:
        import numpy as np
        texts = list(texts)
        if self.budget is not None and kind != "ping" and texts:
            self.budget.take_embed(len(texts))
        self.usage.calls += 1
        self.usage.inputs += len(texts)
        self.usage.by_kind[kind] = self.usage.by_kind.get(kind, 0) + len(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.vector(t) for t in texts]).astype(np.float32)

    def ping(self) -> None:
        self.embed(["ping"], kind="ping")


class NullLLM:
    """`LLM` protocol implementation that stores nothing: extract → no claims, judge → no
    relations (upsert treats as unknown → judge_pending), consolidate → empty (keep both)."""

    CANNED: dict[str, Any] = {
        "extract": {"claims": []},
        "extract_retry": {"claims": []},
        "extract_long": {"claims": []},
        "judge": {"relations": []},
        "judge_enum": {"type": "unknown", "newer": "same"},
        "consolidate": {"text": ""},
        "core_classify": {"items": []},
    }

    def __init__(self) -> None:
        from .llm import LLMUsage
        self.usage = LLMUsage()

    def chat_json(self, kind: str, messages: list[dict], *, model: str | None = None,
                  max_tokens: int = 400, temperature: float = 0.0, json_mode: bool = True):
        import json
        from .llm import LLMResponse
        data = self.CANNED.get(kind, {}) if json_mode else None
        resp = LLMResponse(text=json.dumps(data, ensure_ascii=False) if json_mode else "",
                           data=data, model=model or "offline")
        self.usage.add(kind, resp)
        return resp

    def ping(self, model: str | None = None) -> None:
        self.chat_json("ping", [], model=model, max_tokens=1, json_mode=False)


# ── `yume dream --offline`: deterministic rule-based stand-in (no network) ─────────────────

_HEAD_RE = re.compile(r"^\[([ULA])#([^\s\]]+)(?: [^\]]*)?\] ?(.*)$")
_RULE_RE = re.compile(r"앞으로|항상|절대|규칙|원칙|원장|기준은|always|never", re.IGNORECASE)
_PROFILE_RE = re.compile(r"이름|호칭|시간대|직업|생일|\bname\b|timezone", re.IGNORECASE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!。])\s+|\n+")
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_LABEL_ONLY_RE = re.compile(r"^\*\*[^*]+\*\*\s*$")
_LABEL_RE = re.compile(r"^\*\*([^*]+?)\*\*")


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", t or "").casefold()).strip()


class HeuristicLLM:
    """`LLM` protocol stand-in for ``yume dream --offline`` (sandbox smoke runs). Deterministic:

    - extract: every user sentence (15–400 chars, not a question) in the ``[추출 대상]`` part
      becomes one claim (kind rule when it carries a rule word, else fact; evidence = its ref;
      event date = the window header date). Gates still apply.
    - judge / judge_enum: identical normalized text → duplicate, else unrelated (never unknown,
      so a re-run makes no LLM calls).
    - consolidate → "" (keep both). core_classify → profile / rule / preference by keywords;
      header-only entries are fragments.
    Calls are charged to the run budget like the real client (pings are free)."""

    MAX_CLAIMS = 12

    def __init__(self, budget: Any = None) -> None:
        from .llm import LLMUsage
        self.usage = LLMUsage()
        self.budget = budget

    # handlers return the JSON object the real model would
    def _extract(self, content: str) -> dict:
        marker = "[추출 대상]"
        body = content.split(marker, 1)[1] if marker in content else content
        dm = _DATE_RE.search(content.split(marker, 1)[0] if marker in content else "")
        date = dm.group(1) if dm else None
        msgs: list[list[str]] = []        # [role, ref, text]
        for line in body.splitlines():
            m = _HEAD_RE.match(line)
            if m:
                msgs.append([m.group(1), f"{m.group(1)}#{m.group(2)}", m.group(3)])
            elif msgs:
                msgs[-1][2] += "\n" + line
        claims: list[dict] = []
        for role, ref, text in msgs:
            if role != "U":
                continue
            for s in _SENT_SPLIT_RE.split(text or ""):
                s = s.strip()
                if not 15 <= len(s) <= 400 or "?" in s:
                    continue
                rule = bool(_RULE_RE.search(s))
                subject = " ".join(s.split()[:2])[:30]
                claims.append({"kind": "rule" if rule else "fact", "target": "user", "subject": subject,
                               "text": s, "event_time": date, "valid_until": None, "level": "3",
                               "evidence": [ref], "explicit": "true" if rule else "false",
                               "steps": None})
                if len(claims) >= self.MAX_CLAIMS:
                    return {"claims": claims}
        return {"claims": claims}

    @staticmethod
    def _judge(content: str) -> dict:
        try:
            body = json.loads(content)
        except ValueError:
            return {"relations": []}
        new = _norm((body.get("new") or {}).get("text", ""))
        rels = []
        for c in body.get("candidates") or []:
            same = new and _norm(c.get("text", "")) == new
            rels.append({"id": c.get("id"), "type": "duplicate" if same else "unrelated", "newer": "same"})
        return {"relations": rels}

    @staticmethod
    def _judge_enum(content: str) -> dict:
        try:
            body = json.loads(content)
        except ValueError:
            return {"type": "unrelated", "newer": "same"}
        same = _norm((body.get("new") or {}).get("text", "")) == _norm((body.get("existing") or {}).get("text", ""))
        return {"type": "duplicate" if same else "unrelated", "newer": "same"}

    @staticmethod
    def _core_classify(content: str) -> dict:
        try:
            body = json.loads(content)
        except ValueError:
            return {"items": []}
        items = []
        for it in body.get("items") or []:
            text = str(it.get("text", "")).strip()
            frag = bool(_LABEL_ONLY_RE.match(text))
            kind = "profile" if _PROFILE_RE.search(text) else ("rule" if _RULE_RE.search(text) else "preference")
            lm = _LABEL_RE.match(text)
            subject = (lm.group(1).strip(" :") if lm else text[:30]).strip() or text[:30]
            items.append({"i": it.get("i"), "kind": kind, "subject": subject, "fragment": frag})
        return {"items": items}

    def chat_json(self, kind: str, messages: list[dict], *, model: str | None = None,
                  max_tokens: int = 400, temperature: float = 0.0, json_mode: bool = True):
        from .llm import LLMResponse
        if kind != "ping" and self.budget is not None:
            self.budget.take_llm(1)
        content = ""
        for m in messages or []:
            if m.get("role") == "user":
                content = str(m.get("content") or "")
                if kind in ("extract", "extract_retry", "extract_long"):
                    break          # the window text is the first user message
        if not json_mode or kind == "ping":
            data = None
        elif kind in ("extract", "extract_retry", "extract_long"):
            data = self._extract(content)
        elif kind == "judge":
            data = self._judge(content)
        elif kind == "judge_enum":
            data = self._judge_enum(content)
        elif kind == "core_classify":
            data = self._core_classify(content)
        elif kind == "consolidate":
            data = {"text": ""}
        else:
            data = {}
        resp = LLMResponse(text=json.dumps(data, ensure_ascii=False) if data is not None else "",
                           data=data, model=model or "offline-heuristic")
        self.usage.add(kind, resp)
        return resp

    def ping(self, model: str | None = None) -> None:
        self.chat_json("ping", [], model=model, max_tokens=1, json_mode=False)
