# HermesYume: Neuroscience-Inspired Memory Consolidation for Hermes Agent

## Background

Port of Dreamer (OpenClaw + LanceDB) to Hermes Agent. The neuroscience model is
unchanged; the storage model is different, and that changes the design.

| | Dreamer (OpenClaw) | HermesYume (Hermes) |
|---|---|---|
| Episodic store | daily markdown files | `state.db` sessions/messages (SQLite) |
| Semantic store | LanceDB, unbounded, recalled by vector search | `MEMORY.md` + `USER.md`, bounded (2,200 / 1,375 chars), always in the prompt |
| Forgetting | importance below a threshold | **budget pressure** (homeostasis) |
| Episode capture | needs a 2 AM `/new` session-flush | none: Hermes persists every message |

## Scientific Basis

### Complementary Learning Systems (McClelland, 1995)
- Hippocampus (fast, episodic) = `state.db`
- Neocortex (slow, schematic) = `MEMORY.md` / `USER.md`
- Transfer between the two during sleep = HermesYume

### Sleep Stage Roles
- **NREM**: hippocampal replay; extraction of what generalizes.
- **REM**: integration with existing knowledge; resolving conflicts.
- **Synaptic Homeostasis Hypothesis (SHY)**: global downscaling with selective
  preservation. Hermes' fixed char budget makes this literal: total memory is
  constant, so strengthening one trace means weakening others.

### Engram Lifecycle
- Encoding -> Consolidation -> Retrieval -> Forgetting
- Forgetting = reduced accessibility, not deletion. Evicted entries stay in
  `memory-archive/forgotten.jsonl`, and the raw episodes stay in `state.db`,
  where Hermes' `session_search` can still reach them.

## Dream Process

### Phase 1: NREM

```
1. Select settled sessions: activity > cursor AND idle >= 30 min,
   not hidden, source not excluded (cron)
2. Messages: user/assistant only; originals of compressed sessions
   (active=1 OR compacted=1, not _compressed_summary)
3. Chunk into exchanges; embed; greedy cosine clustering (>= 0.75)
4. LLM extracts durable facts per cluster -> {target: memory|user, text, importance}
5. Verbatim-known facts reinforce the existing entry; near-duplicates go to REM
```

### Phase 2: REM

```
6.  For each fact (highest importance first): closest entry by cosine
7.  sim >= 0.70 -> LLM: duplicate / state_change / different_aspects / unrelated
8.  duplicate -> reinforce; state_change -> "<new> (prev: <old, 40 chars>)";
    different_aspects -> consolidate only if it saves space; else add
9.  Security scan on every candidate entry (injection / exfil / secrets)
10. Homeostasis: while chars > limit * FILL_RATIO:
      a. LLM-shorten the longest entries (max 5)
      b. evict min(score) among entries not touched tonight
    score = importance - DECAY_RATE * days_since_reinforced
11. Apply ops to the CURRENT file under flock(<file>.lock), atomic replace,
    skip ops whose target entry changed, never exceed the hard limit
```

### Phase 3: Dream Log

```
12. dream-log/YYYY-MM-DD_HHMM.md: facts, per-file ops, size before/after
```

## State

- `state.json`: session cursor (max activity timestamp processed)
- `meta.json`: `"<target>:<sha1(entry)>" -> importance, first_seen,
  last_reinforced, cached embedding`. Pruned to live entries after each run.

## File Structure

```
hermesyume/
├── hermesyume.py     # entry point (NREM -> REM -> Dream Log)
├── config.py         # configuration
├── sessions.py       # state.db reader (read-only) + cursor
├── hermes_memory.py  # MEMORY.md/USER.md format, lock, atomic write, threat scan
├── meta.py           # sidecar importance / reinforcement / decay
├── nrem.py           # Phase 1
├── rem.py            # Phase 2
├── llm.py            # extraction / classification / merging prompts
├── embedder.py       # embeddings (openai / ollama / sentence-transformers)
├── dream_log.py      # Phase 3
├── alerts.py         # operator alerts
├── doctor.py         # environment check
└── tests/
```
