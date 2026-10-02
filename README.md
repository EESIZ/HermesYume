# HermesYume

**Hermes + Yume (夢, "dream")** -- sleep-time memory consolidation for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

> AI agents never sleep. They never dream. → That's actually their biggest problem.

HermesYume is the Hermes port of [Dreamer (clawdreamer)](https://github.com/EESIZ/clawdreamer), which did the same for OpenClaw.
Same idea: dreams are (hypothetically) the brain compressing and reorganizing the day's memories. Let the agent do the same every night.

[한국어](README.ko.md) · [Design notes](DESIGN.md)

## What changed in v2

v1 wrote its dreams back into Hermes' tiny `MEMORY.md` / `USER.md` (2,200 / 1,375 chars) and kept them under budget by evicting the weakest entries. In practice those files are already full, so every new fact pushed something out -- and what got pushed out was often a standing rule.

v2 flips it around:

- **`MEMORY.md` / `USER.md` stay the small, always-on core.** The nightly job never writes them. Not one byte.
- **Long-term memory lives in LanceDB** (`$HERMES_HOME/hermesyume/lancedb/`), one row per fact, with provenance, history and a computed strength.
- **A Hermes memory provider recalls it per turn** -- only facts that are actually relevant to the current message (cosine ≥ 0.40 with OpenAI embeddings, at most 5 items / 1,000 chars), wrapped in Hermes' `<memory-context>` block.
- **Forgetting is a state, not a delete.** Faded facts become `dormant`: no longer auto-recalled, still searchable, revived when they come up again.

## How it works

```
[Telegram / CLI / TUI]
        │
Hermes (its own venv, unchanged)
  ├─ MEMORY.md / USER.md  ← always in the system prompt, as before
  └─ hermesyume provider  ($HERMES_HOME/plugins/hermesyume, stdlib only)
        prefetch → serving/recall.sqlite (read-only) → <memory-context> (≤5 facts)
        hooks    → live.db (recall events, remember/forget inbox)
                                     │
state.db (read-only) ────────────────┤
workspace/memory/*.md (read-only) ───┤
                                     ▼
   yume dream  (systemd user timer, nightly, separate venv, one lock)
     NREM: settled conversations → windows → LLM extracts 0..N typed claims → gates
     REM:  upsert into LanceDB (duplicate / state change / different aspects / unrelated),
           recall feedback, strength + transitions, guard, one plan → one commit
     → export serving/recall.sqlite (atomic replace) → Dream Log
```

### Phase 1: NREM -- "What happened today?"

- Reads `state.db` **read-only**, `content` only (never the API copy that already contains injected memories), user/assistant turns only, settled sessions only (30 min idle or ended). Cron and synthetic-eval sessions are skipped.
- Optional markdown notes (`md_sources`) are read in place with a byte-offset ledger. Files are never moved.
- Conversations are cut into windows of whole exchanges (≤ 8,000 chars) with an absolute date header, then the LLM returns **zero or more** typed claims (13 kinds: rule, profile, preference, reference, procedure, decision, lesson, project, fact, state, schedule, event, opinion). "Nothing worth keeping" is a normal answer.
- Deterministic gates reject relative dates without an absolute one, file/session listings, chit-chat about the conversation itself, claims about the memory system itself, secrets and injection patterns.

### Phase 2: REM -- "Does this fit with what I already know?"

- Each claim is compared to its nearest existing memories (vector top-k + same subject). An LLM labels the relation with an enum: `duplicate` / `state_change` / `different_aspects` / `unrelated`. A parse failure becomes `unknown` and is re-judged next night -- it is **never** silently treated as unrelated.
- **Duplicate** → no new row; the existing one gets the evidence. Only things the *user* said strengthen a memory; the agent repeating itself does not.
- **State change** → the old row becomes `superseded` (kept, linked), ordered by when the event happened, not by when it was processed.
- **Different aspects** → merged into one text only if every number, date and name of both inputs survives; otherwise both are kept and linked.

### Forgetting: strength is computed, never decremented

```
t_ref = max(created, last user evidence, last real use)
hl    = half-life[kind] · (1 + 0.5·ln(1 + times actually used))
s     = importance · 2^(−(now − t_ref)/hl)
```

Same inputs, same answer -- a missed night or a double run changes nothing. Below 0.10 (and ≥ 21 days old) a fact goes `dormant`. Protection is earned: rules and profile facts the user stated explicitly (or in two separate sessions), entries of the core files and things the user asked to remember never decay. Being *injected* does not reset the clock; being *used* in a reply does.

### Recall (the provider)

- Pure standard library inside the Hermes process. No packages are installed into Hermes' venv and there is no daemon.
- Search runs over a read-only SQLite copy the nightly job builds: two-stage for OpenAI embeddings (256-dim prefilter → 1536-dim rerank), exact single-stage for the local `hash` embeddings; then relative cut, MMR, already-in-core and already-injected items skipped. If the embedding API is down, it falls back to FTS keyword recall.
- Tools: `yume_search`, `yume_remember`, `yume_forget` -- used only when the user asks first. The agent is told not to bring up its own memory housekeeping.
- `inject: false` runs everything in shadow mode (computed and logged, nothing injected), which is how you calibrate before turning it on.

### Dream Log

Every run writes `dream-log/YYYY-MM-DD_HHMMSS.md`: inputs and exclusion counts, every extracted claim, every gate rejection with its reason, new / reinforced / superseded / merged rows (full text), what expired, went dormant or came back, recall statistics, cosine distribution per relation, tokens and cost. Forgotten items appear by id only; secrets never appear.

## Is it safe to run next to a live Hermes?

- `state.db` is opened `mode=ro` + `query_only`. `MEMORY.md` / `USER.md` are only read (no lock files either). The only code path that can write them is the manual `yume core-restore` / `yume core-proposal apply`.
- `yume dream --dry-run` makes the real LLM calls but writes only its plan and a `_dry` Dream Log -- the Hermes home is byte-for-byte unchanged afterwards (tested).
- Every run is a plan written to disk first and committed once; a crash is replayed. `yume restore --run <id>` rolls LanceDB back to before a run.
- Secrets are masked before anything goes to the LLM, rejected before storage, re-scanned nightly.
- Operator alerts (run failure, 401, model mismatch, stalls, quarantined windows, secrets found, recall health) go to `alerts.log` only. Optionally the nightly job sends them to you as a caption-less `.txt` document through the Telegram Bot API -- never a chat message the agent could see or quote.

## Quick start

### One-line install (on the machine running Hermes)

```bash
curl -fsSL https://raw.githubusercontent.com/EESIZ/HermesYume/main/install.sh | bash -s -- --timer
```

What it does, as the user that runs Hermes (re-running just updates):

- checks `$HERMES_HOME/state.db` (`HERMES_HOME` defaults to `~/.hermes`; use `--hermes-home DIR` for a profile)
- clones / fast-forwards the code into `~/HermesYume`
- creates the nightly venv `~/.local/share/hermesyume/venv` (with [uv](https://docs.astral.sh/uv/) if you have it, otherwise `python3 -m venv` + pip; needs Python 3.11 or 3.12) and installs the locked dependencies
- copies the provider into `$HERMES_HOME/plugins/hermesyume` -- **installed, not activated**
- `yume init` (an existing `config.json` is kept as is) and `yume doctor`
- `--timer`: a systemd user timer (04:40 Asia/Seoul); without a user systemd it adds one crontab line instead, after saving your old crontab to `$HERMES_HOME/hermesyume/crontab.bak`

What it never does: run `hermes config set memory.provider`, touch `config.yaml` or `$HERMES_HOME/.env`, restart Hermes. At the end it prints the exact commands for the next steps ([Turning it on](#turning-it-on)).

Usually there is no key to enter: HermesYume reads `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` by name from the `.env` Hermes already has (see [Providers](#providers)). If neither is there, add one to that file; `yume doctor` tells you which one it found.

### Manual

```bash
# 1. separate venv for the nightly job (never Hermes' own venv); installs a tagged commit
deploy/setup_venv.sh --ref v2.0.0
Y=~/.local/share/hermesyume/venv/bin/yume

# 2. data dir, config, empty stores, checks (1-token API call)
export HERMES_HOME=~/.hermes          # or your profile directory
$Y init && $Y doctor

# 3. migrate what you already have: review first, then apply
$Y migrate --estimate
$Y migrate --dry-run                  # read dream-log/*_dry.md
$Y migrate --approve-migration

# 4. install the provider, start in shadow mode
deploy/install_provider.sh --hermes-home "$HERMES_HOME"
$Y config set inject false
hermes config set memory.provider hermesyume

# 5. nightly timer (systemd user unit)
cp deploy/hermesyume-dream.{service,timer} ~/.config/systemd/user/
systemctl --user edit hermesyume-dream.service   # set HERMES_HOME if not ~/.hermes
systemctl --user daemon-reload && systemctl --user enable --now hermesyume-dream.timer

# 6. after a few days of shadow mode
$Y calibrate                          # recommended recall_min_cos (never below 0.40)
$Y config set inject true
```

### Turning it on

Installing changes nothing in Hermes. Turning it on is three steps, and the first one is shadow mode:

1. `$Y config set inject false` -- the provider computes recall for every message and logs it to `live.db`, but injects nothing.
2. `hermes config set memory.provider hermesyume` -- activates the provider from the next message on.
3. After a few nights, look at `$Y status --recall` and the Dream Logs, run `$Y calibrate`, then `$Y config set inject true`.

`calibrate` never recommends less than 0.40, which is right for OpenAI embeddings; with `hash` embeddings keep the measured default (0.30) unless the shadow numbers say otherwise.

Stop instantly: `$Y config set enabled false` (the provider re-reads the file; no restart). Remove completely: delete the `memory.provider` line; the data stays. After a code update, restart the Hermes gateway yourself (Python caches the provider modules).

## Providers

Keys are read **by name** from `$HERMES_HOME/.env` (the file Hermes already uses), then from the environment. The file is never sourced and values are never logged; `yume doctor` shows which provider was picked and where each key came from, never the key.

**LLM** (extraction and relation judging, nightly only) -- `llm_provider`:

| | when | model | notes |
|---|---|---|---|
| `deepseek` | `auto` picks it when `DEEPSEEK_API_KEY` is set | `deepseek-v4-flash` (`deepseek_model`) | cheapest; called with thinking disabled and JSON mode (falls back to plain parsing if the server refuses JSON mode). `DEEPSEEK_BASE_URL` overrides `https://api.deepseek.com`; the DeepSeek key is never sent anywhere else |
| `openai` | `auto` otherwise | `gpt-4.1-mini` (`extract_model` / `judge_model`) | any OpenAI-compatible endpoint via `llm_base_url` or `OPENAI_BASE_URL` |

`extract_model` / `judge_model` are OpenAI model names; with DeepSeek they only apply if they name a `deepseek-*` model. For accurate cost lines in the Dream Log, set `llm_price_in_per_mtok` / `llm_price_out_per_mtok` to your provider's prices.

**Embeddings** (stored with every memory, and one per message for recall) -- `embed_provider`:

| | when | model | trade-off |
|---|---|---|---|
| `openai` | `auto` picks it when `OPENAI_API_KEY` is set | `text-embedding-3-small`, 1536 dims | semantic: finds a memory phrased in completely different words. One small, cheap HTTP call per message |
| `hash` | `auto` otherwise (DeepSeek has no embeddings API) | `hash/ngram-v1`, 1024 dims, standard library | free, local, no key, no network, same code in the nightly job and the provider. **Lexical**: word and character 2/3-gram overlap, so "Postgres 16" vs "17" or the same Korean phrase with different endings match, but a question that shares no words with the memory does not. Its own thresholds (below) |

The choice is made once, at `yume init`, and written to `config.json` (`"embed_provider": "openai"` or `"hash"`), so adding or removing a key later never changes the stored vectors. Every vector carries its model id (`openai/text-embedding-3-small@1536`, `hash/ngram-v1@1024`) and the nightly job refuses to run when the configured model differs from the stored one -- models are never mixed. `yume reembed` re-embeds for a new model within the same provider; moving between `hash` and an API is not wired up yet (start from a fresh data directory).

`hash` thresholds, measured on synthetic Korean and English pairs (related vs. unrelated cosine distributions; any key you set in `config.json` wins):

| key | OpenAI | hash | why (hash) |
|---|---|---|---|
| `recall_min_cos` | 0.40 | 0.30 | 0.19 % of unrelated question→memory pairs reach it; 43 % of related ones do (lexical recall is the price of no API) |
| `pinned_min_cos` / `search_min_cos` | 0.33 / 0.30 | 0.25 / 0.20 | |
| `injected_strong_cos` | 0.50 | 0.40 | |
| `candidate_cos` | 0.72 | 0.40 | catches 93 % of state changes for the judge |
| `sweep_cos` | 0.82 | 0.55 | |
| `auto_dup_cos` | 0.95 | 0.90 | a state change with unchanged numbers scored up to 0.78, so near-duplicates still go to the judge |
| `suppress_cos` / `core_match_cos` / `mmr_cos` | 0.90 / 0.90 / 0.92 | 0.80 / 0.75 / 0.85 | |

The `recall_min_cos` floor is 0.40 for neural embeddings and 0.25 for `hash`.

## Migration

`yume migrate` loses nothing:

| Step | What |
|---|---|
| M0 | inventory of every source with sha256 (`migration/inventory.json`) |
| M1 | every core-file entry becomes one row, text verbatim; header-only fragments are listed |
| M2 | `USER.md` entries classified as profile or rule are pinned automatically (`yume unpin` to undo) |
| M3 | old episodic `MEMORY.md` entries ("Session: …") are kept verbatim as dormant legacy rows and also extracted |
| M4 | markdown notes backlog (`md_sources`) |
| M5 | an old Dreamer dump (`migration/dreamer_memories.json`) → dormant legacy rows |
| M6 | `state.db` backfill (or `--statedb-start now` to start fresh) |
| M7 | full REM, export, calibration hint, migration Dream Log, `proposals/MEMORY.md.proposed` |

`--only core,dump,memory_md,md,statedb` picks steps; `--estimate` prints windows, calls and cost; a run whose estimate exceeds `--max-llm-calls` is refused. The `core_map` must account for every core-file entry. The clean-up proposal for `MEMORY.md` is only a file; apply it yourself with `yume core-proposal apply` once you have checked it.

## Commands

| Command | |
|---|---|
| `yume dream [--dry-run] [--offline] [--now +70d] [--settle-minutes N]` | nightly run (`--json` prints run stats) |
| `yume status [--recall]` | last runs, backlog, recall health |
| `yume search <q> [--include-inactive]` / `yume inspect --query/--id/--label` | look at memories |
| `yume forget <id>` / `yume unpin <id>` / `yume pin list` | admin edits |
| `yume restore --run <id> [--reprocess] [--unforget]` | roll back a run and every later one (forgets are re-applied unless `--unforget`) |
| `yume core-check` / `yume core-restore <id>` / `yume core-proposal apply` | core files (the last two are the only writers) |
| `yume migrate`, `yume calibrate`, `yume export`, `yume reembed` | see above |
| `yume alert-flush` | send pending operator alerts (when `alert_telegram` is on) |
| `yume doctor [--offline]`, `yume init`, `yume config set/get/show` | setup |
| `yume debug plant/event` | sandbox-only test helpers (refused on a live home) |

`--offline` swaps in a rule-based fake LLM and hash embeddings (no network) for sandbox smoke runs;
on a live home it is only allowed together with `--dry-run`.

A *live home* is `~/.hermes`, any home a gateway has run in (`gateway_state.json`, `gateway.pid` or
`gateway.lock`), and any home listed in `$HERMESYUME_PROTECT_HOMES` (`:`-separated) or in the
`protect_homes` key. A sandbox home never writes procedure docs into a protected home's
`workspace_dir`, nor into `$HERMESYUME_PROTECT_WORKSPACES` / `protect_workspaces`.

## Configuration

`$HERMES_HOME/hermesyume/config.json` -- flat keys, no secrets, shared by the nightly job and the provider. `config.json.example` lists every key with its default. The ones you are most likely to touch:

| Key | Default | |
|---|---|---|
| `enabled` / `inject` | `true` / `true` | master switch / shadow mode |
| `llm_provider` / `embed_provider` | `auto` / `auto` | see [Providers](#providers); `init` writes the resolved `embed_provider` |
| `extract_model` / `judge_model` | `gpt-4.1-mini` | OpenAI |
| `deepseek_model` | `deepseek-v4-flash` | DeepSeek |
| `include_sources` | `["telegram","cli","tui"]` | session sources to learn from |
| `workspace_dir` / `md_sources` | | the agent's workspace (procedure docs go to `docs/yume/`) / markdown note folders; empty after a fresh install -- set them with `yume config set` |
| `hermes_runtime_dir` | | Hermes checkout whose `tools/threat_patterns.py` is used; empty = `$HERMES_RUNTIME_DIR`, then the importable `hermes_cli`, else the vendored copy |
| `exclude_first_message_regex` / `deny_cwd_globs` / `strip_line_regex` | empty | skip sessions whose first message matches (e.g. synthetic evals) / sessions started in matching cwds / drop bot status lines before extraction |
| `protect_homes` / `protect_workspaces` | `[]` | extra live homes and workspaces (see [Commands](#commands)) |
| `recall_min_cos` / `recall_k` / `recall_budget_chars` | `0.40` (hash `0.30`) / `5` / `1000` | |
| `max_windows_per_run` / `max_llm_calls` | `60` / `400` | per-run caps (backlog continues next night) |
| `alert_telegram` | `false` | send alerts as a `.txt` document |

Keys are read from `$HERMES_HOME/.env` by name only (`DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`; with alerts on, `TELEGRAM_BOT_TOKEN` and `YUME_ALERT_CHAT_ID`). The file is never sourced and values are never logged.

## Data layout

```
$HERMES_HOME/hermesyume/            (0700; files 0600)
  config.json
  lancedb/                          memories, memory_history, suppress  (canonical)
  ledger.db                         watermarks, windows, runs, core_seen, audit
  live.db                           provider → dream: recall events, inbox, health
  serving/recall.sqlite             read-only copy for the provider (atomic replace)
  runs/<run_id>/plan.json           replayable commit plans
  dream-log/                        nightly reports
  alerts.log                        operator alerts
  migration/  proposals/  backups/
```

## Limits

- Injection checks are pattern based. Tool output is never used as input, which is the real line of defence.
- Quality follows the LLM. A small model will extract junk or mislabel relations; the gates and the decay keep junk from piling up, but they do not make it smart.
- "Used" is a token-overlap heuristic on the reply. A false positive only extends a memory's life a little.
- Semantic recall needs an embeddings API (OpenAI-compatible). Without one, `hash` embeddings recall by shared words only; if the API fails at runtime the provider falls back to keyword search.

## Tests

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements-dream.txt pytest -e .
.venv/bin/python -m pytest -q tests/dream            # nightly side, offline, fake LLM/embeddings
cd <hermes-runtime> && venv/bin/python -B -m unittest discover -s <repo>/tests/provider -t <repo>   # provider side, Hermes' Python
```

Three tests compare against a real installation and skip unless asked: `HERMESYUME_TEST_LIVE_HOME=<home>`
(state.db column parity and the live-home guard, read-only) and `HERMES_RUNTIME_DIR=<checkout>`
(threat-pattern parity).

## Requirements

- Python 3.11 or 3.12 for the nightly venv (LanceDB, pyarrow, numpy, openai, pyyaml); [uv](https://docs.astral.sh/uv/) is used when present (and required by `deploy/setup_venv.sh`)
- Hermes Agent with memory-provider plugins (the provider itself is standard library only)
- One LLM key: `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` (any OpenAI-compatible endpoint). Embeddings: `OPENAI_API_KEY`, or nothing (`hash`)

## Donations
If you find it useful...
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/V7V21XAPRC)

## License

MIT
