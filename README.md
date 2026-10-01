# Hermesume

**Hermes + Yume (夢, "dream")** -- sleep-time memory consolidation for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

> AI agents never sleep. They never dream. → That's actually their biggest problem.

Hermesume is the Hermes port of [Dreamer (clawdreamer)](https://github.com/EESIZ/clawdreamer), which did the same for OpenClaw.
Same idea: dreams are (hypothetically) the brain compressing and reorganizing the day's memories. Let the agent do the same every night.

[한국어](README.ko.md)

## Why Hermes needs this

Hermes' built-in memory is deliberately tiny and curated:

| File | What | Default limit |
|------|------|---------------|
| `~/.hermes/memories/MEMORY.md` | agent's own notes (environment, conventions, lessons) | 2,200 chars |
| `~/.hermes/memories/USER.md` | user profile (preferences, style) | 1,375 chars |

Both are injected into the system prompt at the start of every session. All conversations are also kept in `~/.hermes/state.db` (SQLite + FTS5) and are searchable on demand via `session_search`.

So Hermes already has a hippocampus (`state.db`, raw episodes) and a small neocortex (`MEMORY.md`/`USER.md`, always-on knowledge). What's missing is **sleep**: nothing systematically moves the important parts of the day from one into the other, merges outdated facts, or decides what to forget once the files are full. Writing them is left to the agent mid-conversation, when it is busy doing something else.

Hermesume is that offline process.

## How it works

```
~/.hermes/state.db  (sessions + messages, read-only)
        │
        ▼
   ┌─────────┐
   │  NREM   │  settled sessions → exchanges → embed → cluster → LLM extracts durable facts
   └────┬────┘  (routed to "memory" or "user")
        ▼
   ┌─────────┐
   │   REM   │  each fact vs. its closest existing entry:
   └────┬────┘    duplicate → reinforce · state_change → merge (newer wins, "(prev: …)")
        │         different_aspects → consolidate · unrelated → add
        │       homeostasis: over budget → shorten long entries → forget weakest
        ▼
~/.hermes/memories/MEMORY.md, USER.md  (written under Hermes' own lock, atomically)
        │
        ▼
   Dream Log  (~/.hermesume/dream-log/YYYY-MM-DD_HHMM.md)
```

### Phase 1: NREM -- "What happened today?"

- Reads sessions from `state.db` whose last activity is after the previous run and at least 30 min ago (so live conversations aren't dreamed mid-way). Opened **read-only**.
- Only `user` / `assistant` turns are used by default. Tool output (web pages, files) is untrusted, and a prompt injection in a fetched page must not become a permanent memory. `cron` sessions are skipped by default.
- Uses the original messages of compressed sessions, not the compression summary.
- Splits into exchanges (a user turn + the replies), embeds, clusters similar exchanges, and asks the LLM for *durable* facts only. Returning nothing is a valid answer: the budget is a few thousand characters.

### Phase 2: REM -- "Does this fit with what I already know?"

- Each new fact is compared only to existing entries (O(N·M)), and an LLM classifies the closest one as `duplicate` / `state_change` / `different_aspects` / `unrelated`.
  - Near-duplicates are **not** dropped by embedding similarity alone. "Postgres 16" vs "Postgres 17" embed almost identically, so only the classifier can tell a duplicate from a state change.
- **Synaptic homeostasis**: Hermes memory is bounded, so forgetting is necessary. If a file exceeds `FILL_RATIO` (default 85%) of its limit, Hermesume first asks the LLM to tighten the longest entries, then evicts the entries with the lowest *decayed importance*. Importance rises when a fact comes up again and decays linearly with days since. Entries the agent wrote itself are adopted with importance 0.7. Evicted entries go to `memory-archive/forgotten.jsonl`. The 15% headroom leaves room for the agent's own `memory add` calls during the day.
- Every candidate entry passes a prompt-injection / exfiltration / secret scan before it is written. Hermes' own `tools.threat_patterns` is used if importable; otherwise a vendored subset.

### Phase 3: Dream Log -- "What did I dream about?"

A markdown report per run: facts extracted, entries added / merged / consolidated / shortened / forgotten / blocked, and size before → after.

## Safety with a running Hermes

- `state.db` is opened with `mode=ro`; Hermesume never writes to it.
- Memory files are written exactly like Hermes does: exclusive `flock` on `MEMORY.md.lock`, temp file + `os.replace`. The output round-trips through the `§` format, so Hermes' drift guard accepts it. This was verified by loading the output with Hermes' own `MemoryStore`.
- The plan is computed from a snapshot but applied against the *current* file. If the agent changed an entry in the meantime, that operation is skipped, and our additions are rolled back rather than exceeding the hard limit.
- A copy of each file is saved to `~/.hermesume/memory-archive/<timestamp>/` before every write.
- Hermes injects memory as a frozen snapshot at session start, so changes appear from the **next** session.

## Quick start

```bash
pip install -r requirements.txt      # only pyyaml (optional); core is stdlib
cp .env.example .env                 # set OPENAI_API_KEY or use ollama
python doctor.py                     # checks state.db, memory files, limits, keys
python hermesume.py --dry-run -v     # plan only, writes just a dream log
python hermesume.py -v               # real run
```

Run it nightly with cron or the systemd units in `examples/`:

```bash
0 3 * * * /path/to/hermesume/examples/run-hermesume.sh
```

Unlike the OpenClaw version, no `session-flush` step is needed: Hermes persists every message to `state.db` as it happens.

### Hermes profiles

Point `HERMES_HOME` at the profile directory and give each profile its own `HERMESUME_HOME`:

```bash
HERMES_HOME=~/.hermes/profiles/work HERMESUME_HOME=~/.hermesume-work python hermesume.py
```

## Configuration

Limits and enabled flags are read from `$HERMES_HOME/config.yaml` (`memory.memory_char_limit`, `memory.user_char_limit`, `memory.memory_enabled`, `memory.user_profile_enabled`).

| Variable | Default | Description |
|----------|---------|-------------|
| `HERMES_HOME` | `~/.hermes` | Hermes home (or profile dir) |
| `HERMESUME_HOME` | `~/.hermesume` | cursor, metadata, dream logs, archives |
| `HERMESUME_EMBEDDING_PROVIDER` | `openai` | `openai`, `ollama`, `sentence-transformers` |
| `HERMESUME_LLM_PROVIDER` | `openai` | `openai` (any OpenAI-compatible URL), `ollama`, `minimax` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | e.g. OpenRouter |
| `HERMESUME_OPENAI_LLM_MODEL` | `gpt-4.1-nano` | |
| `HERMESUME_FILL_RATIO` | `0.85` | fill memory up to this fraction of Hermes' limit |
| `HERMESUME_DECAY_RATE` | `0.01` | importance lost per day without reinforcement |
| `HERMESUME_FORGET_THRESHOLD` | `0` | >0: also forget faded entries when under budget |
| `HERMESUME_KEEP_PREV_STATE` | `true` | keep a short "(prev: …)" on merged state changes |
| `HERMESUME_EXCLUDE_SOURCES` | `cron` | comma-separated session sources to skip |
| `HERMESUME_INCLUDE_TOOL_MESSAGES` | `false` | dream over tool output too (riskier) |
| `HERMESUME_SESSION_SETTLE_SECONDS` | `1800` | skip sessions active more recently than this |
| `HERMESUME_MAX_NEW_FACTS` | `12` | cap per run |
| `HERMESUME_ENTRY_MAX_CHARS` | `220` | max length of one entry |
| `HERMESUME_ALERT_PROVIDER` | (off) | `telegram`, `slack`, `webhook` -- errors go to the operator, never the agent |

Optional: markdown notes named `YYYY-MM-DD*.md` in `$HERMESUME_HOME/episodes/` are dreamed too, then archived.

### Directory structure

```
$HERMESUME_HOME/
  state.json            # session cursor
  meta.json             # per-entry importance / reinforcement / cached embeddings
  dream-log/            # nightly reports
  memory-archive/
    <timestamp>/        # MEMORY.md / USER.md before each write
    forgotten.jsonl     # everything that was forgotten, with its score
  episodes/             # optional extra markdown episodes
```

## Limitations

- The threat scan is pattern-based. A malicious instruction paraphrased into harmless-looking wording can get past it; skipping tool output is the main defense.
- Quality depends on the LLM. Small local models may extract trivia or misclassify relationships. Read the dream logs for the first few nights, ideally with `--dry-run`.
- Importance is a heuristic (re-mentions + time). Hermesume cannot see when the agent *used* an entry, because Hermes doesn't record that.
- External Hermes memory providers (Honcho, Mem0, …) are not touched; only the built-in files are.

## Tests

```bash
python -m unittest discover tests
```

Network-free end-to-end run against a synthetic Hermes home: state-change merge, user/agent routing, injection blocking, budget eviction, round-trip format, and concurrent-edit handling.

## Donations
If you find it useful...
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/V7V21XAPRC)

## License

MIT
