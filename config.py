"""HermesYume configuration.

HermesYume = Hermes + Yume (夢, "dream").
It consolidates a Hermes Agent's memory while the agent "sleeps".

Inputs  (episodic):  $HERMES_HOME/state.db       (SQLite: sessions + messages)
Outputs (semantic):  $HERMES_HOME/memories/MEMORY.md, USER.md  (§-delimited)
Own state:           $HERMESYUME_HOME/                            (cursor, metadata, logs)
"""

import os


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ── Hermes paths ──
# HERMES_HOME follows Hermes' own resolution (env var -> ~/.hermes).
# Point it at a profile directory to dream for a specific Hermes profile.
HERMES_HOME = os.path.expanduser(os.path.expandvars(
    os.environ.get("HERMES_HOME", "").strip() or "~/.hermes"))
HERMES_STATE_DB = os.environ.get(
    "HERMESYUME_STATE_DB", os.path.join(HERMES_HOME, "state.db"))
HERMES_MEMORY_DIR = os.path.join(HERMES_HOME, "memories")
HERMES_CONFIG_PATH = os.path.join(HERMES_HOME, "config.yaml")

# ── HermesYume paths ──
HERMESYUME_HOME = os.path.expanduser(
    os.environ.get("HERMESYUME_HOME", "~/.hermesyume"))
STATE_PATH = os.path.join(HERMESYUME_HOME, "state.json")        # session cursor
META_PATH = os.path.join(HERMESYUME_HOME, "meta.json")          # per-entry importance
DREAM_LOG_DIR = os.path.join(HERMESYUME_HOME, "dream-log")
MEMORY_ARCHIVE_DIR = os.path.join(HERMESYUME_HOME, "memory-archive")
# Optional: extra markdown episodes (YYYY-MM-DD*.md) dropped here are dreamed too.
EPISODE_DIR = os.path.join(HERMESYUME_HOME, "episodes")
EPISODE_ARCHIVE_DIR = os.path.join(EPISODE_DIR, "archive")

# ── Hermes memory format (must match tools/memory_tool_store.py) ──
ENTRY_DELIMITER = "\n§\n"
# Defaults used when config.yaml is missing / has no memory section.
DEFAULT_MEMORY_CHAR_LIMIT = 2200
DEFAULT_USER_CHAR_LIMIT = 1375

# ── API Keys ──
# Keys come from the environment, falling back to Hermes' own $HERMES_HOME/.env,
# so a Hermes install that already has e.g. DEEPSEEK_API_KEY needs no extra setup.
# Only these names are read from that file; nothing else in it is touched.
_HERMES_ENV_KEYS = ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "MINIMAX_API_KEY")


def _read_hermes_env(path: str) -> dict:
    found = {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[7:].lstrip()
                key, sep, value = line.partition("=")
                key = key.strip()
                if sep and key in _HERMES_ENV_KEYS:
                    value = value.strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                        value = value[1:-1]
                    else:
                        value = value.split(" #", 1)[0].strip()
                    found[key] = value
    except OSError:
        pass
    return found


def _usable(key: str) -> bool:
    """False for empty values and .env.example placeholders."""
    return bool(key) and not key.startswith("your_")


_hermes_env = _read_hermes_env(os.path.join(HERMES_HOME, ".env"))
KEY_SOURCE = {}
for _k in _HERMES_ENV_KEYS:
    if _usable(os.environ.get(_k, "")):
        KEY_SOURCE[_k] = "environment"
    elif _usable(_hermes_env.get(_k, "")):
        os.environ[_k] = _hermes_env[_k]
        KEY_SOURCE[_k] = "$HERMES_HOME/.env"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "") if _usable(os.environ.get("OPENAI_API_KEY", "")) else ""
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "") if _usable(os.environ.get("DEEPSEEK_API_KEY", "")) else ""
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")


def _st_installed() -> bool:
    import importlib.util
    return importlib.util.find_spec("sentence_transformers") is not None


# LLM (extraction / classification / merging)
#   auto (default): deepseek if DEEPSEEK_API_KEY is available, else openai
LLM_PROVIDER = os.environ.get("HERMESYUME_LLM_PROVIDER", "auto")
if LLM_PROVIDER == "auto":
    LLM_PROVIDER = "deepseek" if DEEPSEEK_API_KEY else "openai"
OPENAI_LLM_MODEL = os.environ.get("HERMESYUME_OPENAI_LLM_MODEL", "gpt-4.1-nano")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "") or "https://api.deepseek.com"
# deepseek-chat / deepseek-reasoner were retired on 2026-07-24.
DEEPSEEK_MODEL = os.environ.get("HERMESYUME_DEEPSEEK_MODEL", "deepseek-v4-flash")
OLLAMA_LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:3b")
MINIMAX_BASE_URL = "https://api.minimax.io/anthropic"

# Embeddings: "openai", "ollama", "sentence-transformers", or "hash"
#   auto (default): openai if OPENAI_API_KEY, else sentence-transformers if
#   installed, else "hash" (stdlib char n-gram hashing -- no API, no download;
#   cruder, so REM leans more on the LLM classifier). DeepSeek has no
#   embedding API.
EMBEDDING_PROVIDER = os.environ.get("HERMESYUME_EMBEDDING_PROVIDER", "auto")
if EMBEDDING_PROVIDER == "auto":
    EMBEDDING_PROVIDER = ("openai" if OPENAI_API_KEY
                          else "sentence-transformers" if _st_installed() else "hash")
EMBEDDING_MODEL = os.environ.get("HERMESYUME_EMBEDDING_MODEL", "text-embedding-3-small")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
ST_MODEL_NAME = os.environ.get("ST_MODEL_NAME", "all-MiniLM-L6-v2")

# ── Session intake (hippocampus replay) ──
# Sessions idle for less than this are considered "still awake" and skipped.
SESSION_SETTLE_SECONDS = _env_int("HERMESYUME_SESSION_SETTLE_SECONDS", 1800)
MAX_SESSIONS_PER_RUN = _env_int("HERMESYUME_MAX_SESSIONS_PER_RUN", 50)
# Comma-separated session sources to ignore (Hermes: cli, telegram, discord, cron, ...).
EXCLUDE_SOURCES = [s.strip() for s in os.environ.get(
    "HERMESYUME_EXCLUDE_SOURCES", "cron").split(",") if s.strip()]
# Tool output is untrusted (web pages, files) -- by default only user/assistant
# turns are dreamed, so a prompt injection in a fetched page can't become a
# permanent memory.
INCLUDE_TOOL_MESSAGES = _env_bool("HERMESYUME_INCLUDE_TOOL_MESSAGES", False)
MAX_MESSAGE_CHARS = _env_int("HERMESYUME_MAX_MESSAGE_CHARS", 1500)
MAX_EPISODES_PER_RUN = 7

# ── NREM parameters ──
CHUNK_MIN_LENGTH = 20        # minimum chars per chunk
# Hash embeddings score related pairs much lower than neural ones (measured:
# related 0.28-0.94, unrelated <= 0.18 vs. ~0.7+ for neural), so they get
# their own thresholds.
_HASH = EMBEDDING_PROVIDER == "hash"
CLUSTER_SIMILARITY = 0.40 if _HASH else 0.75  # cosine threshold for clustering
DEDUP_SIMILARITY = 0.88      # fact counts as "already known" above this
MAX_CLUSTERS_PER_RUN = _env_int("HERMESYUME_MAX_CLUSTERS_PER_RUN", 40)
MAX_NEW_FACTS = _env_int("HERMESYUME_MAX_NEW_FACTS", 12)
ENTRY_MAX_CHARS = _env_int("HERMESYUME_ENTRY_MAX_CHARS", 220)

# ── REM parameters ──
CONTRADICTION_SIMILARITY = 0.25 if _HASH else 0.70  # same-topic -> ask classifier
KEEP_PREV_STATE = _env_bool("HERMESYUME_KEEP_PREV_STATE", True)
PREV_STATE_MAX_CHARS = 40        # "(prev: ...)" suffix length -- budget is tight
# Importance a never-seen entry gets (the agent chose to save it itself).
AGENT_ENTRY_IMPORTANCE = 0.7
REINFORCE_BOOST = 0.1            # importance gain when a fact is re-observed
IMPORTANCE_DECAY_RATE = _env_float("HERMESYUME_DECAY_RATE", 0.01)  # per day unreinforced
# Fill MEMORY.md / USER.md only up to this fraction of Hermes' char limit, so
# the agent still has room for its own `memory add` calls during the day.
FILL_RATIO = _env_float("HERMESYUME_FILL_RATIO", 0.85)
# Optional absolute forgetting: evict entries whose decayed score falls below
# this even when there is budget left. 0 = forget only under budget pressure.
FORGET_THRESHOLD = _env_float("HERMESYUME_FORGET_THRESHOLD", 0.0)

# ── Error Alerts ──
# HermesYume runs as a background process invisible to the agent.
# When something breaks, alert the *operator* -- never the agent.
#   telegram  - HERMESYUME_ALERT_TELEGRAM_BOT_TOKEN + HERMESYUME_ALERT_TELEGRAM_CHAT_ID
#   slack     - HERMESYUME_ALERT_SLACK_WEBHOOK_URL
#   webhook   - HERMESYUME_ALERT_WEBHOOK_URL (generic POST)
#   (empty)   - alerts disabled
ALERT_PROVIDER = os.environ.get("HERMESYUME_ALERT_PROVIDER", "")
ALERT_TELEGRAM_BOT_TOKEN = os.environ.get("HERMESYUME_ALERT_TELEGRAM_BOT_TOKEN", "")
ALERT_TELEGRAM_CHAT_ID = os.environ.get("HERMESYUME_ALERT_TELEGRAM_CHAT_ID", "")
ALERT_SLACK_WEBHOOK_URL = os.environ.get("HERMESYUME_ALERT_SLACK_WEBHOOK_URL", "")
ALERT_WEBHOOK_URL = os.environ.get("HERMESYUME_ALERT_WEBHOOK_URL", "")
