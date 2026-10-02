"""Read-only snapshot of the core files (MEMORY.md / USER.md) for one session (PLAN-v2 §6.1, §7.2).

- ``shas``: corefmt.core_sha of every entry → prefetch excludes serving rows already in the prompt.
- ``pin_in_core``: a pin counts as present when an entry carries the same leading label, or the
  pin's normalized 3-grams are ≥ `containment` covered by the core text.
- ``find_entry``: restore a full entry from a partial ``old_text`` (memory remove/replace hooks).
Never writes and never creates lock files.
"""

from dataclasses import dataclass, field
import os

from . import corefmt

TARGET_FILES = {"memory": "MEMORY.md", "user": "USER.md"}


@dataclass
class CoreSnapshot:
    entries: dict = field(default_factory=dict)    # target -> [entry text]
    shas: set = field(default_factory=set)
    labels: set = field(default_factory=set)

    def copy(self):
        return CoreSnapshot({k: list(v) for k, v in self.entries.items()}, set(self.shas),
                            set(self.labels))

    def all_text(self):
        return "\n".join(e for t in sorted(self.entries) for e in self.entries[t])


def core_path(hermes_home, target):
    return os.path.join(str(hermes_home), "memories", TARGET_FILES[target])


def from_entries(entries):
    snap = CoreSnapshot({t: list(v) for t, v in entries.items()})
    for es in snap.entries.values():
        for e in es:
            snap.shas.add(corefmt.core_sha(e))
            lab = corefmt.entry_label(e)
            if lab:
                snap.labels.add(corefmt.core_norm(lab))
    return snap


def snapshot(hermes_home):
    """Entries of both core files (missing/undecodable file → no entries for it)."""
    entries = {}
    for target in TARGET_FILES:
        try:
            entries[target] = corefmt.read_entries(core_path(hermes_home, target))
        except Exception:
            entries[target] = []
    return from_entries(entries)


def pin_in_core(pin_text, pin_label, snap, *, containment=0.8):
    if snap is None:
        return False
    if pin_label:
        if corefmt.core_norm(pin_label) in snap.labels:
            return True
    lab = corefmt.entry_label(pin_text or "")
    if lab and corefmt.core_norm(lab) in snap.labels:
        return True
    if corefmt.core_sha(pin_text or "") in snap.shas:
        return True
    hay = snap.all_text()
    if not hay:
        return False
    return corefmt.containment(pin_text or "", hay) >= float(containment)


def find_entry(snap, target, partial):
    """Full entry of `target` containing `partial` (exact, then whitespace/NFKC-normalized)."""
    if snap is None or not partial:
        return None
    entries = snap.entries.get(target) or []
    for e in entries:
        if partial in e:
            return e
    p = corefmt.core_norm(partial)
    if not p:
        return None
    for e in entries:
        if p in corefmt.core_norm(e):
            return e
    return None


def apply_write(snap, action, target, content, old_full):
    """Keep a live view in step with memory-tool writes (the prompt snapshot stays frozen)."""
    if snap is None or target not in TARGET_FILES:
        return
    es = snap.entries.setdefault(target, [])
    if action == "add" and content:
        es.append(content)
    elif action == "replace" and old_full is not None:
        for i, e in enumerate(es):
            if e == old_full:
                es[i] = content or e
                break
    elif action == "remove" and old_full is not None:
        for i, e in enumerate(es):
            if e == old_full:
                del es[i]
                break
