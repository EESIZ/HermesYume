"""Local embeddings without an embeddings API: ``hash/ngram-v1@1024`` (stdlib only).

The ONLY copy. The provider imports it (``from . import hash_embed``); the dream loads this same
file by path (``hermesyume.paths.load_provider_module("hash_embed")``), so a stored vector and a
query vector always come from the same code. No relative imports, no third-party modules.

Algorithm (frozen — any change needs a new model name such as ``ngram-v2``, because vectors that
are already stored would silently stop matching):

1. ``unicodedata.normalize("NFKC", text).casefold()``
2. word tokens ``\\w+`` (Unicode: Hangul, Latin, digits, ``_``); each word → feature ``w:<word>``
   (weight 1.0) plus the character 2-grams and 3-grams of `` <word> `` (padded with one space on
   each side) → ``c:<gram>`` (weight 0.5 each)
3. text without any word token (only symbols) → one ``s:<char>`` feature per non-space character
4. signed feature hashing: ``x = int.from_bytes(blake2b(feature, digest_size=8), "little")``,
   bucket ``x % 1024``, sign ``+`` when bit 63 of x is set, else ``-``; then L2 normalize
   (empty text → the zero vector)

It is lexical, not semantic: "Postgres 16" vs "Postgres 17" or the same Korean phrase with
different endings score high; a paraphrase that shares no words scores ≈ 0. Thresholds are
therefore separate from the neural ones (``MODEL_DEFAULTS``, measured; see DEVIATIONS PR-2).
"""

import hashlib
import math
import re
import unicodedata

PROVIDER = "hash"
MODEL = "ngram-v1"
DIM = 1024
MODEL_ID = "%s/%s@%d" % (PROVIDER, MODEL, DIM)

WORD_WEIGHT = 1.0
GRAM_WEIGHT = 0.5
_WORD_RE = re.compile(r"\w+")

# Config defaults that differ for this model (cosine geometry is lexical, so related pairs score
# lower than with a neural model). Applied for every key the config file does not set itself.
# Numbers come from the synthetic Korean/English pair measurement recorded in DEVIATIONS PR-2.
MODEL_DEFAULTS = {
    "recall_min_cos": 0.30,       # unrelated query→memory: 0.19 % of pairs ≥ 0.30; related: 43 %
    "pinned_min_cos": 0.25,       # unrelated 0.47 % (pins are few)
    "search_min_cos": 0.20,       # yume_search list floor (tool_hit still needs recall_min_cos)
    "injected_strong_cos": 0.40,  # unrelated 0.03 %; related p75 0.36, p90 0.49
    "candidate_cos": 0.40,        # state changes 93 %, near-duplicates 97 %; unrelated memory pairs 0.13 %
    "sweep_cos": 0.55,            # state changes 77 %, near-duplicates 93 %; unrelated 0 % (max 0.456)
    "auto_dup_cos": 0.90,         # a state change with unchanged numbers reached 0.783 → keep the judge
    "suppress_cos": 0.80,         # above that 0.783 too (text_sha still blocks identical text)
    "core_match_cos": 0.75,       # unrelated max 0.456; a reworded entry is 0.56–0.92
    "mmr_cos": 0.85,              # near-duplicate p75 0.80
}
RECALL_MIN_COS_FLOOR = 0.25       # calibrate rule p99(unrelated query→memory)+0.05 = 0.202+0.05 (neural: 0.40)

PROVIDERS = ("auto", "openai", "hash")


def features(text):
    """[(feature, weight)] for one text (see module docstring)."""
    t = unicodedata.normalize("NFKC", text or "").casefold()
    out = []
    for word in _WORD_RE.findall(t):
        out.append(("w:" + word, WORD_WEIGHT))
        padded = " " + word + " "
        for n in (2, 3):
            for i in range(len(padded) - n + 1):
                out.append(("c:" + padded[i:i + n], GRAM_WEIGHT))
    if not out:
        out = [("s:" + ch, WORD_WEIGHT) for ch in t if not ch.isspace()]
    return out


def _bucket(feature):
    x = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "little")
    return x % DIM, (1.0 if (x >> 63) & 1 else -1.0)


def embed(text):
    """L2-normalized ``DIM``-float list (python floats). Deterministic, never raises for str input."""
    v = [0.0] * DIM
    for feat, w in features(text):
        b, sign = _bucket(feat)
        v[b] += sign * w
    n = math.sqrt(sum(x * x for x in v))
    if n <= 0:
        return v
    return [x / n for x in v]


def embed_many(texts):
    return [embed(t) for t in texts]


def is_model_id(model_id):
    return str(model_id or "").split("/", 1)[0] == PROVIDER


def two_stage(model_id):
    """True when the model's vectors are prefix-truncatable (OpenAI text-embedding-3*), so the
    256-dim prefix prefilter + full rerank is valid. Every other model is searched single-stage."""
    try:
        prov, rest = str(model_id).split("/", 1)
    except ValueError:
        return False
    return prov == "openai" and rest.startswith("text-embedding-3")


def resolve_provider(setting, has_openai_key):
    """``embed_provider`` setting → concrete provider. "auto" = openai when an OPENAI_API_KEY is
    available, else hash. ``has_openai_key=None`` (unknown) keeps the classic default, openai."""
    s = str(setting or "auto").strip().lower()
    if s == "auto":
        return "hash" if has_openai_key is False else "openai"
    return s


def apply_model(values, explicit=(), has_openai_key=None):
    """Resolve ``embed_provider`` in a merged config dict in place. For hash: embed_model/embed_dim
    are fixed to this model and ``MODEL_DEFAULTS`` replace every default the caller did not set
    explicitly (``explicit`` = keys present in the config file). Returns the setting as written
    ("auto", "openai", "hash")."""
    setting = str(values.get("embed_provider") or "auto").strip().lower()
    prov = resolve_provider(setting, has_openai_key)
    values["embed_provider"] = prov
    if prov == PROVIDER:
        values["embed_model"] = MODEL
        values["embed_dim"] = DIM
        for k, v in MODEL_DEFAULTS.items():
            if k in values and k not in explicit:
                values[k] = v
    return setting


def recall_min_cos_floor(provider):
    """0.40 for neural embeddings (PLAN §4.1); this model's own floor for hash."""
    return RECALL_MIN_COS_FLOOR if provider == PROVIDER else 0.40
