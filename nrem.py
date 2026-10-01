"""NREM Phase: episodes -> candidate semantic facts.

Like NREM sleep (hippocampal replay -> neocortical transfer):
1. Load settled Hermes sessions from state.db (+ optional markdown episodes)
2. Chunk into exchanges (user turn + assistant replies)
3. Embed and cluster similar chunks (sharp-wave ripple replay)
4. LLM distills each cluster into durable facts, routed to MEMORY.md or USER.md
5. Facts already stored verbatim REINFORCE the existing entry instead

Nothing is written to Hermes here; REM decides how facts are integrated.
"""

import logging

from config import (
    CHUNK_MIN_LENGTH,
    CLUSTER_SIMILARITY,
    DEDUP_SIMILARITY,
    MAX_CLUSTERS_PER_RUN,
    MAX_NEW_FACTS,
)
from embedder import cosine_similarity, embed_texts
from llm import extract_facts
from sessions import chunk_episode, chunk_session, load_episode_files, load_sessions

log = logging.getLogger("hermesyume.nrem")


def entry_vectors(target: str, entries: list[str], meta) -> list[list[float]]:
    """Embeddings for existing memory entries, cached in the sidecar metadata."""
    missing = [e for e in entries if not meta.get(target, e).get("vector")]
    if missing:
        for text, vec in zip(missing, embed_texts(missing)):
            meta.get(target, text)["vector"] = vec
    vectors = [meta.get(target, e)["vector"] for e in entries]
    # Embedding provider changed since the cache was written -> re-embed all.
    dims = {len(v) for v in vectors}
    if len(dims) > 1:
        for text, vec in zip(entries, embed_texts(entries)):
            meta.get(target, text)["vector"] = vec
        vectors = [meta.get(target, e)["vector"] for e in entries]
    return vectors


def cluster_chunks(vectors: list[list[float]]) -> list[list[int]]:
    """Greedy single-pass clustering on cosine similarity."""
    n = len(vectors)
    assigned = [False] * n
    clusters = []
    for i in range(n):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if not assigned[j] and cosine_similarity(vectors[i], vectors[j]) >= CLUSTER_SIMILARITY:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)
    multi = sum(1 for c in clusters if len(c) > 1)
    log.info("Formed %d clusters (%d multi-chunk, %d singletons)",
             len(clusters), multi, len(clusters) - multi)
    return clusters


def best_match(vector, vectors: list[list[float]]) -> tuple[int, float]:
    best_i, best_sim = -1, 0.0
    for i, v in enumerate(vectors):
        if v and len(v) == len(vector):
            sim = cosine_similarity(vector, v)
            if sim > best_sim:
                best_i, best_sim = i, sim
    return best_i, best_sim


def run_nrem(snapshot: dict[str, list[str]], meta, now: float) -> dict:
    """Execute the NREM phase.

    Args:
        snapshot: {"memory": [entries], "user": [entries]} for enabled targets
        meta:     Meta sidecar (reinforcement is recorded on it)
    """
    log.info("=== NREM Phase: replaying the day ===")
    empty = {"sessions": 0, "episode_files": [], "chunks": 0, "clusters": 0,
             "facts": [], "reinforced": [], "skipped_dup": 0, "cursor": None}

    sessions = load_sessions(now=now)
    episode_files = load_episode_files()
    chunks = []
    for s in sessions:
        chunks += chunk_session(s)
    for ep in episode_files:
        chunks += chunk_episode(ep["content"])
    chunks = [c for c in chunks if len(c) >= CHUNK_MIN_LENGTH]

    cursor = max((s["activity"] for s in sessions), default=None)
    result = {**empty, "sessions": len(sessions),
              "episode_files": [e["path"] for e in episode_files],
              "chunks": len(chunks), "cursor": cursor}
    if not chunks:
        log.info("Nothing to dream about")
        return result

    vectors = embed_texts(chunks)
    clusters = cluster_chunks(vectors)
    # Bigger clusters = topics that came up repeatedly; replay those first.
    clusters.sort(key=len, reverse=True)
    result["clusters"] = len(clusters)

    facts, fact_vectors = [], []
    reinforced = set()

    for idx in clusters[:MAX_CLUSTERS_PER_RUN]:
        if len(facts) >= MAX_NEW_FACTS:
            log.info("Reached max new facts (%d)", MAX_NEW_FACTS)
            break
        texts = [chunks[i] for i in idx]
        if len(idx) == 1 and len(texts[0]) < 50:
            continue
        extracted = extract_facts(texts)
        if not extracted:
            continue
        ex_vectors = embed_texts([f["text"] for f in extracted])
        for fact, vec in zip(extracted, ex_vectors):
            target = fact["target"]
            if target not in snapshot:  # target disabled in Hermes config
                target = "memory" if "memory" in snapshot else None
                if target is None:
                    continue
                fact["target"] = target
            # Verbatim already known -> reinforce. Near-duplicates go on to REM:
            # "Postgres 16" vs "Postgres 17" embed almost identically, and only
            # the classifier can tell a duplicate from a state change.
            norm = " ".join(fact["text"].lower().split())
            same = [e for e in snapshot[target] if " ".join(e.lower().split()) == norm]
            if same:
                if (target, same[0]) not in reinforced:
                    meta.reinforce(target, same[0], now)
                    reinforced.add((target, same[0]))
                result["skipped_dup"] += 1
                continue
            # Same fact extracted twice tonight.
            same_tonight = [v for f, v in zip(facts, fact_vectors) if f["target"] == target]
            if best_match(vec, same_tonight)[1] >= DEDUP_SIMILARITY:
                result["skipped_dup"] += 1
                continue
            facts.append({**fact, "vector": vec})
            fact_vectors.append(vec)
            if len(facts) >= MAX_NEW_FACTS:
                break

    result["facts"] = facts
    result["reinforced"] = sorted(reinforced)
    log.info("NREM complete: %d sessions, %d chunks, %d clusters, %d new facts, "
             "%d reinforced", len(sessions), len(chunks), len(clusters),
             len(facts), len(reinforced))
    return result
