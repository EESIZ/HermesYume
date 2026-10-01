"""Hermesume configuration.

Hermesume = Hermes + Yume (夢, "dream").
It consolidates a Hermes Agent's memory while the agent "sleeps".

Inputs  (episodic):  $HERMES_HOME/state.db       (SQLite: sessions + messages)
Outputs (semantic):  $HERMES_HOME/memories/MEMORY.md, USER.md  (§-delimited)
Own state:           $HERMESUME_HOME/                            (cursor, metadata, logs)
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
    "HERMESUME_STATE_DB", os.path.join(HERMES_HOME, "state.db"))
HERMES_MEMORY_DIR = os.path.join(HERMES_HOME, "memories")
HERMES_CONFIG_PATH = os.path.join(HERMES_HOME, "config.yaml")

# ── Hermesume paths ──
HERMESUME_HOME = os.path.expanduser(
    os.environ.get("HERMESUME_HOME", "~/.hermesume"))
STATE_PATH = os.path.join(HERMESUME_HOME, "state.json")        # session cursor
META_PATH = os.path.join(HERMESUME_HOME, "meta.json")          # per-entry importance
DREAM_LOG_DIR = os.path.join(HERMESUME_HOME, "dream-log")
MEMORY_ARCHIVE_DIR = os.path.join(HERMESUME_HOME, "memory-archive")
# Optional: extra markdown episodes (YYYY-MM-DD*.md) dropped here are dreamed too.
EPISODE_DIR = os.path.join(HERMESUME_HOME, "episodes")
EPISODE_ARCHIVE_DIR = os.path.join(EPISODE_DIR, "archive")

# ── Hermes memory format (must match tools/memory_tool_store.py) ──
ENTRY_DELIMITER = "\n§\n"
# Defaults used when config.yaml is missing / has no memory section.
DEFAULT_MEMORY_CHAR_LIMIT = 2200
DEFAULT_USER_CHAR_LIMIT = 1375

# ── API Keys ──
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")

# Embedding provider: "openai", "ollama", or "sentence-transformers"
EMBEDDING_PROVIDER = os.environ.get("HERMESUME_EMBEDDING_PROVIDER", "openai")
EMBEDDING_MODEL = os.environ.get("HERMESUME_EMBEDDING_MODEL", "text-embedding-3-small")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
ST_MODEL_NAME = os.environ.get("ST_MODEL_NAME", "all-MiniLM-L6-v2")

# LLM (extraction / classification / merging)
LLM_PROVIDER = os.environ.get("HERMESUME_LLM_PROVIDER", "openai")  # openai, ollama, minimax
OPENAI_LLM_MODEL = os.environ.get("HERMESUME_OPENAI_LLM_MODEL", "gpt-4.1-nano")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OLLAMA_LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:3b")
MINIMAX_BASE_URL = "https://api.minimax.io/anthropic"

# ── Session intake (hippocampus replay) ──
# Sessions idle for less than this are considered "still awake" and skipped.
SESSION_SETTLE_SECONDS = _env_int("HERMESUME_SESSION_SETTLE_SECONDS", 1800)
MAX_SESSIONS_PER_RUN = _env_int("HERMESUME_MAX_SESSIONS_PER_RUN", 50)
# Comma-separated session sources to ignore (Hermes: cli, telegram, discord, cron, ...).
EXCLUDE_SOURCES = [s.strip() for s in os.environ.get(
    "HERMESUME_EXCLUDE_SOURCES", "cron").split(",") if s.strip()]
# Tool output is untrusted (web pages, files) -- by default only user/assistant
# turns are dreamed, so a prompt injection in a fetched page can't become a
# permanent memory.
INCLUDE_TOOL_MESSAGES = _env_bool("HERMESUME_INCLUDE_TOOL_MESSAGES", False)
MAX_MESSAGE_CHARS = _env_int("HERMESUME_MAX_MESSAGE_CHARS", 1500)
MAX_EPISODES_PER_RUN = 7

# ── NREM parameters ──
CHUNK_MIN_LENGTH = 20        # minimum chars per chunk
CLUSTER_SIMILARITY = 0.75    # cosine similarity threshold for clustering
DEDUP_SIMILARITY = 0.88      # fact counts as "already known" above this
MAX_CLUSTERS_PER_RUN = _env_int("HERMESUME_MAX_CLUSTERS_PER_RUN", 40)
MAX_NEW_FACTS = _env_int("HERMESUME_MAX_NEW_FACTS", 12)
ENTRY_MAX_CHARS = _env_int("HERMESUME_ENTRY_MAX_CHARS", 220)

# ── REM parameters ──
CONTRADICTION_SIMILARITY = 0.70  # same-topic detection threshold
KEEP_PREV_STATE = _env_bool("HERMESUME_KEEP_PREV_STATE", True)
PREV_STATE_MAX_CHARS = 40        # "(prev: ...)" suffix length -- budget is tight
# Importance a never-seen entry gets (the agent chose to save it itself).
AGENT_ENTRY_IMPORTANCE = 0.7
REINFORCE_BOOST = 0.1            # importance gain when a fact is re-observed
IMPORTANCE_DECAY_RATE = _env_float("HERMESUME_DECAY_RATE", 0.01)  # per day unreinforced
# Fill MEMORY.md / USER.md only up to this fraction of Hermes' char limit, so
# the agent still has room for its own `memory add` calls during the day.
FILL_RATIO = _env_float("HERMESUME_FILL_RATIO", 0.85)
# Optional absolute forgetting: evict entries whose decayed score falls below
# this even when there is budget left. 0 = forget only under budget pressure.
FORGET_THRESHOLD = _env_float("HERMESUME_FORGET_THRESHOLD", 0.0)

# ── Error Alerts ──
# Hermesume runs as a background process invisible to the agent.
# When something breaks, alert the *operator* -- never the agent.
#   telegram  - HERMESUME_ALERT_TELEGRAM_BOT_TOKEN + HERMESUME_ALERT_TELEGRAM_CHAT_ID
#   slack     - HERMESUME_ALERT_SLACK_WEBHOOK_URL
#   webhook   - HERMESUME_ALERT_WEBHOOK_URL (generic POST)
#   (empty)   - alerts disabled
ALERT_PROVIDER = os.environ.get("HERMESUME_ALERT_PROVIDER", "")
ALERT_TELEGRAM_BOT_TOKEN = os.environ.get("HERMESUME_ALERT_TELEGRAM_BOT_TOKEN", "")
ALERT_TELEGRAM_CHAT_ID = os.environ.get("HERMESUME_ALERT_TELEGRAM_CHAT_ID", "")
ALERT_SLACK_WEBHOOK_URL = os.environ.get("HERMESUME_ALERT_SLACK_WEBHOOK_URL", "")
ALERT_WEBHOOK_URL = os.environ.get("HERMESUME_ALERT_WEBHOOK_URL", "")
