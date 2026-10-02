# HermesYume v2: Design

Nightly memory consolidation for Hermes Agent. The neuroscience framing is unchanged from Dreamer
and v1; the storage model is not.

## Background: why v2 is not "v1, but better"

| | Dreamer (OpenClaw) | HermesYume v1 | HermesYume v2 |
|---|---|---|---|
| Episodic input | daily markdown files | `state.db` | `state.db` (read-only) + markdown notes, both tracked in place |
| Semantic store | LanceDB, recalled by vector search | `MEMORY.md` / `USER.md` (bounded, always in the prompt) | **LanceDB** (canonical) + read-only SQLite serving copy |
| Recall | top 3, no real threshold | the whole file, every session | relevant facts only (cos ≥ 0.40, relative cut, ≤ 5 / 1,000 chars) |
| Forgetting | cumulative nightly decrement, hard delete | eviction under budget pressure | computed strength, `dormant` state (searchable, revivable) |
| Core files | — | rewritten every night | **never written** by the nightly job |

v1's budget homeostasis turned out to be the wrong analogy for a store that the agent and the user
also edit: the files were already full, so every automatic write evicted something, often a
standing rule. v2 keeps `MEMORY.md` / `USER.md` as the small always-on core and moves long-term
memory to a store with per-turn recall.

## Scientific basis

- **Complementary Learning Systems** (McClelland, 1995): hippocampus = `state.db` (fast, episodic);
  neocortex = LanceDB (slow, schematic); the core files are the always-active part of the cortex.
- **NREM**: replay of the day's episodes; extraction of what generalizes.
- **REM**: integration with existing knowledge, conflict resolution (state changes as chains).
- **Forgetting = reduced accessibility, not deletion.** Dormant traces stay retrievable by search
  and come back with new evidence or use.
- **Spacing effect**: real use lengthens the half-life (`1 + 0.5·ln(1 + r)`); mere exposure
  (injection without use) does not reset the clock.

## Components

| | Where | Runtime |
|---|---|---|
| Provider | `$HERMES_HOME/plugins/hermesyume/` (`__init__.py`, `plugin.yaml`, `cli.py`, `_yume/`) | inside Hermes, **standard library only** |
| Nightly job `yume dream` | separate venv outside `$HERMES_HOME` | lancedb, pyarrow, numpy, openai, pyyaml |
| Admin CLI `yume` | same venv, same lock | |
| Data | `$HERMES_HOME/hermesyume/` (0700 / 0600) | |

No daemon. The gateway never opens LanceDB: the nightly job exports a read-only
`serving/recall.sqlite` (256-dim and 1536-dim vectors, FTS5 trigram index, pins) and replaces it
atomically; the provider reopens it when the inode changes.

### Write ownership

| Store | Writer | Readers |
|---|---|---|
| `state.db` | Hermes | dream (`mode=ro` + `query_only`) |
| `MEMORY.md` / `USER.md` | Hermes' memory tool, humans | dream, provider (read only) |
| `lancedb/`, `ledger.db` | dream / `yume` (one non-blocking `flock`) | dream, `yume` |
| `live.db` | provider processes (WAL, append), dream marks inbox rows consumed | dream, provider |
| `serving/recall.sqlite` | dream (build aside, `os.replace`) | provider |

## Storage

**LanceDB `memories`** (one row per fact): verbatim `text` (1–3 standalone sentences, absolute
dates), `subject`/`subject_key`, `vector fixed_size_list<float32>[1536]` of `"subject: text"`,
`kind` (13 kinds + `legacy`), `tier`, `importance` (only ever raised), status
(`active / superseded / expired / dormant / forgotten / quarantined`), event/validity times,
evidence counters split into *all* vs *user* evidence, provenance lists (sessions, messages,
idempotency keys), recall counters, supersede chain, `related_ids`, core-file mirror fields
(`core_target`, `core_sha`, `in_core`, `core_required`), `pinned`.

**`memory_history`**: append-only, keyed by `sha256(run + id + op + seq)` (no truncated ids).
**`suppress`**: vectors and text hashes of forgotten facts (never the text), so they cannot be
re-learned.

**`ledger.db`**: per-lineage watermarks (`last_ts`, `last_id`), session → lineage root cache,
markdown offsets with prefix hashes, window states (attempts, quarantine), runs (status, Lance
version before/after, watermark snapshot, stats), core-file observations, fold cursors, audit
(never forgotten text).

**`live.db`** (DDL single source `provider/_yume/live_schema.py`): `recall_events`
(injected / used / tool_hit / shadow), `inbox` (remember / forget / core_add / core_replace /
core_remove / session_end), `health`.

## NREM: episodes → claims

1. **Inputs.** `state.db` lineages (follow `parent_session_id` only through compression parents),
   watermark by `(timestamp, id)` so compression generation copies and child-session tails are not
   re-read; source allow-list, optional synthetic-eval regex and cwd deny-list (empty by default),
   chat-type allow-list, hidden sessions/messages and tool rows excluded; settle 30 min or
   `session_end`. Markdown notes by byte offset + prefix hash (append-only fast path; a changed
   prefix re-reads the file and upsert absorbs duplicates). Episodic `core_add` items from the inbox.
2. **Sanitize.** Strip injected `<memory-context>` blocks and metadata envelopes, elide long code /
   JSON / blobs, drop lines repeated across many messages, cap long messages with an explicit
   marker, redact secrets.
3. **Windows.** Whole exchanges up to 8,000 chars, one previous exchange as read-only context,
   header with platform, title, period and the reference date. `window_id = sha256(source + root +
   first_id + last_id)` (markdown adds the content hash).
4. **Extract** (JSON mode, temperature 0): 0..N claims `{kind, target, subject, text, event_time,
   valid_until, level, evidence, explicit, steps}`; one retry on malformed output, then the window
   is `failed` (3rd failure → `quarantined` + alert). The watermark only advances over a contiguous
   prefix of committed windows.
5. **Gates** (deterministic): kind enum, 15–400 chars, evidence inside the window, relative time
   without an absolute date, meta/listing patterns, claims about the memory system itself, UUID /
   filename dumps, secrets, injection patterns.
6. **Normalize**: subject key, refs only to paths that exist, importance
   `clamp(base[kind] + 0.08·(level−3) + 0.12·explicit_user + 0.05·min(sessions−1, 3) − 0.10·assistant_only)`.
7. **Embed** in batches; any failure aborts the run before anything is written.

## REM: claims → store

- **Upsert** per claim, applied immediately to a working copy: idempotency key → suppress list →
  candidates (Lance top-k, this run's new rows, same subject key) → auto-duplicate only at
  cos ≥ 0.95 with identical numbers/dates and negation → otherwise an enum judgement
  (`duplicate | state_change | different_aspects | unrelated`, `newer`), unparseable → `unknown` →
  inserted with `judge_pending` and re-judged next night.
- **Duplicate** reinforces; only user evidence raises counters and protection.
- **State change** supersedes by event time; a backlog claim older than the current row is inserted
  already superseded. Pinned/durable rows are never superseded without explicit user evidence: the
  new claim becomes a separate row linked by `related_ids`.
- **Different aspects** consolidate only when a fact-preservation check passes (every number, date,
  Latin token, quoted string and proper-noun candidate of both inputs appears in the result).
- **Inbox**: `remember` (user evidence, importance ≥ 0.80, optional pin), `forget` (status +
  suppress row + audit without text), core-file add / replace / remove (a removal demotes, it does
  not delete).
- **Recall fold**: `injected` counts once per (memory, session, day) and never touches the decay
  clock; `used` / `tool_hit` reset it and revive dormant rows. Cron events are ignored.
- **Core check** (read-only): mirror edits that bypassed the hooks; note when a `core_required` row
  left the core files.
- **Transitions** (pure functions of the row and `now`): tier
  `pinned > legacy > durable > decaying(assistant-only rule/profile/preference) > slow > expiring >
  decaying`; strength `importance · 2^(−Δt/hl)` with `hl = base · (1 + 0.5·ln(1 + r))`; active →
  expired after `valid_until` (+ grace for schedules); active → dormant below 0.10 after ≥ 21 days;
  forgotten → purged after 30 days; a secret found later → quarantined → purged immediately.
- **Guard**: ops that would change a pinned/durable row without user evidence are held (recorded,
  never auto-applied). Mass dormancy is never held — dormant rows are searchable, so it is reversible.
- **Commit** (each step idempotent, replayed after a crash): `plan.json` (atomic) → history →
  suppress → memories (`merge_insert` once) → one ledger transaction → mark inbox consumed →
  optimize → export → procedure docs → Dream Log and alerts. An unchanged re-run makes zero LLM
  calls and keeps the Lance version.

## Recall (provider)

`prefetch(query)`: skip when disabled, denied cwd, group chat, platform not allowed, synthetic
session or open circuit breaker → expand very short queries with the previous user turn → embed
(urllib, 1 s connect / 3 s total, LRU, breaker after 3 failures) → stage 1: 256-dim dot products
over active rows → top 64 → stage 2: 1536-dim rerank → drop pending forgets, expired validity,
entries already in this session's core snapshot (`core_sha`), ids already injected this epoch →
eligibility cos ≥ 0.40 (0.33 for pins not in the core) → `score = cos + 0.08·strength +
0.04·pinned + 0.03·keyword_hit` → relative cut (top − 0.10) → MMR (0.92) → ≤ 5 items / 1,000 chars,
each ≤ 300 chars. The block says the facts are reference material, not instructions, with dates.
Embedding failure → FTS5 trigram / LIKE fallback. Everything is wrapped so a broken copy, a locked
`live.db` or a dead API yields `""` within the deadline.

`sync_turn` decides `used` by distinctive-token overlap between the injected memory and the reply.
`system_prompt_block` carries a static note (the agent must not bring up its own memory
housekeeping, and uses the tools only when the user asks first) plus pins that are not in the core
files. `on_memory_write(remove)` restores the full entry from the session snapshot so a removal is
recorded as a demotion.

## Operations

- **Dream Log** per run (Korean, full texts, ids only for forgotten rows, no secrets, no review
  queue).
- **Alerts** only for system failures: run failure, 401, model/dimension mismatch, scanner
  unavailable, two stalled nights, window quarantine, secret found, recall health (embedding failure
  rate, p95, stale serving copy). Output is `alerts.log`; optionally a caption-less `.txt` via the
  Telegram Bot API `sendDocument` from the nightly process — never a chat message, never from the
  gateway.
- **Rollback**: `yume restore --run <id>` (Lance time travel, 14 days) undoes that run and every
  later one; forgets made by those runs are re-applied unless `--unforget`; `--reprocess` rewinds
  watermarks and md offsets and puts their inbox items back to pending. Refused across a
  `yume reembed`. Ledger backups (7), weekly data archives (4), and a Lance archive before every
  secret purge.
- **Calibration**: shadow mode logs would-be injections; `yume calibrate` recommends
  `recall_min_cos = max(0.40, p99(unrelated) + 0.05)`.

## Migration

M0 inventory → M1 core entries seeded verbatim → M2 user profile/rule entries pinned → M3 old
episodic `MEMORY.md` entries kept as dormant legacy rows and extracted → M4 markdown backlog →
M5 old Dreamer dump as dormant legacy rows → M6 `state.db` backfill → M7 full REM, export,
calibration hint, `core_map` (every core entry accounted for) and a `MEMORY.md` clean-up proposal
that only a human applies. Runs as two committed plans (direct inserts, then the regular NREM/REM
pipeline); `--dry-run` applies the first plan to a private temporary copy so the preview is exact.

## Files

```
hermesyume/            nightly job + admin CLI (yume)
  cli.py               commands
  paths.py clock.py config.py secrets_env.py sqlite_util.py
  threat.py vendor/    threat scanner (runtime by path → pinned vendored copy → fail closed)
  llm.py embedder.py offline.py
  store.py ledger.py livedb.py
  sources/             statedb.py markdown.py core_files.py
  sanitize.py windows.py prompts.py extract.py gates.py normalize.py nrem.py
  vecutil.py upsert.py strength.py recall_fold.py core_check.py plan.py rem.py docs_writer.py
  export.py dream_log.py alerts.py migrate.py calibrate.py
provider/              Hermes memory provider (stdlib only)
  __init__.py plugin.yaml cli.py
  _yume/               config embed_http serving live core_snapshot textutil used
                       + single-source schemas: live_schema serving_schema corefmt
deploy/                setup_venv.sh install_provider.sh hermesyume-dream.{service,timer}
tests/dream/           pytest (offline: fake LLM, fake embeddings, fake Hermes home)
tests/provider/        unittest, runs with Hermes' Python
```
