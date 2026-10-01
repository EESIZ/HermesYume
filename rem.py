"""REM Phase: integrate new facts into MEMORY.md / USER.md, then downscale.

Like REM sleep:
1. Compare each new fact only against existing entries (O(N*M))
2. Classify: duplicate / state_change / different_aspects / unrelated
3. Resolve: reinforce, merge (newer state wins), consolidate, or add
4. Synaptic homeostasis: Hermes memory is bounded (2,200 / 1,375 chars by
   default). If a file is over its budget, compress long entries first, then
   forget the entries with the lowest decayed importance (archived, not lost)
5. Write the result back under Hermes' own file lock, atomically

Hermes reads these files once per session (frozen snapshot), so the agent
wakes up with the consolidated memory at its next session.
"""

import json
import logging
import os

from config import (
    CONTRADICTION_SIMILARITY,
    ENTRY_MAX_CHARS,
    FILL_RATIO,
    FORGET_THRESHOLD,
    KEEP_PREV_STATE,
    MEMORY_ARCHIVE_DIR,
    PREV_STATE_MAX_CHARS,
)
from embedder import embed_texts
from hermes_memory import apply_ops, char_count, clean_entry, threat_findings
from llm import classify_relationship, consolidate_aspects, merge_state_change, shorten_entry
from nrem import best_match, entry_vectors

log = logging.getLogger("hermesyume.rem")

MAX_SHORTEN_PER_RUN = 5
# A merged state change may exceed ENTRY_MAX_CHARS by its "(prev: ...)" trace.
MERGED_MAX_CHARS = ENTRY_MAX_CHARS + PREV_STATE_MAX_CHARS + 12


class Plan:
    """Planned edits for one target, simulated on a working copy."""

    def __init__(self, target: str, entries: list[str], vectors: list, meta, now: float):
        self.target = target
        self.entries = list(entries)
        self.vectors = list(vectors)
        self.meta = meta
        self.now = now
        self.ops: list[dict] = []
        self.touched: set[str] = set()   # entries created/changed tonight
        self.details = {"added": [], "merged": [], "consolidated": [],
                        "shortened": [], "forgotten": [], "blocked": [],
                        "reinforced": 0}

    def add(self, text: str, vector, importance: float):
        self.ops.append({"op": "add", "text": text})
        self.entries.append(text)
        self.vectors.append(vector)
        self.touched.add(text)
        self.meta.set(self.target, text, importance, self.now, vector)

    def replace(self, old: str, new: list[str], new_vectors: list, importance: float):
        i = self.entries.index(old)
        old_meta = self.meta.get(self.target, old, self.now)
        others = self.entries[:i] + self.entries[i + 1:]
        kept = [(t, v) for t, v in zip(new, new_vectors) if t not in others]
        new, new_vectors = [t for t, _ in kept], [v for _, v in kept]
        self.ops.append({"op": "replace", "old": old, "new": new})
        self.entries[i:i + 1] = new
        self.vectors[i:i + 1] = new_vectors
        for text, vec in zip(new, new_vectors):
            self.touched.add(text)
            self.meta.set(self.target, text, max(importance, old_meta["importance"]),
                          self.now, vec, first_seen=old_meta["first_seen"])

    def remove(self, old: str):
        i = self.entries.index(old)
        self.ops.append({"op": "remove", "old": old})
        del self.entries[i]
        del self.vectors[i]


def _safe(text: str, plan: Plan, max_chars: int) -> str | None:
    """Clean and vet a candidate entry; None if it must not be written."""
    text = clean_entry(text)
    if len(text) > max_chars:
        shorter = shorten_entry(text, ENTRY_MAX_CHARS)
        if not shorter or len(shorter) > max_chars:
            plan.details["blocked"].append({"text": text, "reason": "too long"})
            return None
        text = clean_entry(shorter)
    findings = threat_findings(text)
    if findings:
        log.warning("Blocked entry (%s): %s", ", ".join(findings), text[:80])
        plan.details["blocked"].append({"text": text, "reason": ", ".join(findings)})
        return None
    return text


def integrate(plan: Plan, fact: dict):
    text = _safe(fact["text"], plan, ENTRY_MAX_CHARS)
    if text is None or text in plan.entries:
        return
    vector = fact["vector"]
    importance = fact.get("importance", 0.5)

    i, sim = best_match(vector, plan.vectors)
    if i < 0 or sim < CONTRADICTION_SIMILARITY:
        plan.add(text, vector, importance)
        plan.details["added"].append(text)
        return

    existing = plan.entries[i]
    kind = classify_relationship(text, existing)["type"]
    log.info("Relation (sim=%.3f, %s): '%s' vs '%s'", sim, kind, text[:50], existing[:50])

    if kind == "duplicate":
        plan.meta.reinforce(plan.target, existing, plan.now)
        plan.details["reinforced"] += 1
    elif kind == "state_change":
        merged = _safe(merge_state_change(text, existing, KEEP_PREV_STATE), plan,
                       MERGED_MAX_CHARS)
        if merged is None:
            return
        plan.replace(existing, [merged], [embed_texts([merged])[0]], importance)
        plan.details["merged"].append({"before": [existing, text], "after": merged})
    elif kind == "different_aspects":
        texts = consolidate_aspects(text, existing)
        texts = [t for t in (_safe(t, plan, ENTRY_MAX_CHARS) for t in texts or []) if t]
        # Consolidation must actually save space, otherwise just add the fact.
        if not texts or char_count(texts) >= len(text) + len(existing) + 3:
            plan.add(text, vector, importance)
            plan.details["added"].append(text)
            return
        plan.replace(existing, texts, embed_texts(texts), importance)
        plan.details["consolidated"].append({"before": [existing, text], "after": texts})
    else:
        plan.add(text, vector, importance)
        plan.details["added"].append(text)


def downscale(plan: Plan, budget: int):
    """Synaptic homeostasis: bring the file back under its budget."""
    # 0. Optional absolute forgetting of faded entries.
    if FORGET_THRESHOLD > 0:
        for e in list(plan.entries):
            score = plan.meta.score(plan.target, e, plan.now)
            if e not in plan.touched and score < FORGET_THRESHOLD:
                plan.remove(e)
                plan.details["forgotten"].append({"text": e, "score": round(score, 3),
                                                  "reason": "faded"})
    if char_count(plan.entries) <= budget:
        return

    # 1. Compress: tighten the longest entries first.
    for e in sorted(plan.entries, key=len, reverse=True)[:MAX_SHORTEN_PER_RUN]:
        if char_count(plan.entries) <= budget or len(e) <= ENTRY_MAX_CHARS // 2:
            break
        shorter = shorten_entry(e)
        if not shorter:
            continue
        shorter = clean_entry(shorter)
        if threat_findings(shorter) or shorter in plan.entries:
            continue
        importance = plan.meta.get(plan.target, e, plan.now)["importance"]
        plan.replace(e, [shorter], [embed_texts([shorter])[0]], importance)
        plan.details["shortened"].append({"before": e, "after": shorter})

    # 2. Forget: evict the weakest entries, sparing tonight's work if possible.
    while char_count(plan.entries) > budget and plan.entries:
        candidates = [e for e in plan.entries if e not in plan.touched] or plan.entries
        weakest = min(candidates, key=lambda e: plan.meta.score(plan.target, e, plan.now))
        score = plan.meta.score(plan.target, weakest, plan.now)
        plan.remove(weakest)
        plan.details["forgotten"].append({"text": weakest, "score": round(score, 3),
                                          "reason": "over budget"})


def _archive_forgotten(target: str, forgotten: list[dict], now: float):
    if not forgotten:
        return
    os.makedirs(MEMORY_ARCHIVE_DIR, exist_ok=True)
    with open(os.path.join(MEMORY_ARCHIVE_DIR, "forgotten.jsonl"), "a", encoding="utf-8") as f:
        for item in forgotten:
            f.write(json.dumps({"target": target, "at": now, **item},
                               ensure_ascii=False) + "\n")


def run_rem(nrem_result: dict, snapshot: dict[str, list[str]], limits: dict,
            meta, now: float, dry_run: bool = False, stamp: str | None = None) -> dict:
    log.info("=== REM Phase: integration + homeostasis ===")
    summary = {"targets": {}}

    for target, entries in snapshot.items():
        limit = limits[target]["limit"]
        budget = int(limit * FILL_RATIO)
        plan = Plan(target, entries, entry_vectors(target, entries, meta), meta, now)

        facts = sorted((f for f in nrem_result["facts"] if f["target"] == target),
                       key=lambda f: f.get("importance", 0.5), reverse=True)
        for fact in facts:
            integrate(plan, fact)
        downscale(plan, budget)

        applied = apply_ops(target, plan.ops, limit, dry_run=dry_run, stamp=stamp)
        if not dry_run:
            _archive_forgotten(target, plan.details["forgotten"], now)

        summary["targets"][target] = {
            "limit": limit,
            "budget": budget,
            "before_chars": applied["before_chars"],
            "after_chars": applied["after_chars"],
            "entries_before": len(entries),
            "entries_after": len(plan.entries),
            "ops_applied": len(applied["applied"]),
            "ops_skipped": applied["skipped"],
            "backup": applied["backup"],
            **plan.details,
        }
        log.info("%s: %d -> %d chars (budget %d / limit %d), %d ops applied, %d skipped",
                 target, applied["before_chars"], applied["after_chars"], budget, limit,
                 len(applied["applied"]), len(applied["skipped"]))
    return summary
