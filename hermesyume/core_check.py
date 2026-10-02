"""R5 core-file check (read-only) and the package's ONLY MEMORY.md/USER.md writer (PLAN-v2 §4.3
R5, §7.1, §7.3, T14).

Dream never writes the core files: ``check_core`` only reads them (``core_files.read_core``, plain
read, no lock file), mirrors hook-less additions/removals into Lance rows and records
``core_seen``. ``restore`` / ``apply_proposal`` are called only by the human-run
``yume core-restore`` / ``yume core-proposal apply``; they use Hermes' own convention
(``<file>.lock`` flock, mkstemp + fsync + ``os.replace`` onto the symlink-resolved path).
"""

from __future__ import annotations

import fcntl
import logging
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .embedder import embed_input
from .llm import LLMAuthError, LLMError
from .types import (CANDIDATE_SEARCH_EXCLUDED, KINDS, BudgetExceeded, Claim, CoreEntry,
                    CoreSeenRow, MemoryRow, text_sha)

log = logging.getLogger("hermesyume.core_check")

CLASSIFY_BATCH = 40
DEFAULT_LIMITS = {"memory": 2200, "user": 1375}


def _corefmt():
    from .paths import load_provider_module
    return load_provider_module("corefmt")


# ── classification helper (R1 remember/core_add, R5 mirror) ─────────────────

def classify_texts(ctx: Any, texts: list[str]) -> list[tuple[str | None, str | None, bool]]:
    """One batched ``core_classify`` call per ≤40 texts → [(kind|None, subject|None, fragment)].
    Failures (bad JSON, LLM error, budget) → (None, None, False) — callers fall back."""
    out: list[tuple[str | None, str | None, bool]] = [(None, None, False)] * len(texts)
    if not texts:
        return out
    from . import prompts
    cfg = ctx.cfg
    for start in range(0, len(texts), CLASSIFY_BATCH):
        chunk = texts[start:start + CLASSIFY_BATCH]
        try:
            resp = ctx.llm.chat_json("core_classify", prompts.core_classify_messages(chunk),
                                     model=cfg.extract_model,
                                     max_tokens=int(cfg.core_classify_max_tokens),
                                     temperature=float(cfg.llm_temperature))
        except LLMAuthError:
            raise
        except (LLMError, BudgetExceeded) as e:
            log.warning("core_classify failed: %s", type(e).__name__)
            continue
        data = resp.data
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            try:
                i = int(it.get("i"))
            except (TypeError, ValueError):
                continue
            if not 0 <= i < len(chunk):
                continue
            kind = str(it.get("kind") or "").strip().lower()
            subj = it.get("subject")
            subj = subj.strip() if isinstance(subj, str) and subj.strip() else None
            frag = it.get("fragment")
            frag = frag is True or (isinstance(frag, str) and frag.strip().lower() == "true")
            out[start + i] = (kind if kind in KINDS else None, subj, bool(frag))
    return out


def is_fragment(text: str) -> bool:
    """Header-only entries like '**Reading List:**' carry no fact."""
    cf = _corefmt()
    lab = cf.entry_label(text)
    return bool(lab) and cf.core_norm(text) == cf.core_norm(lab)


def core_claim(ctx: Any, *, target: str, text: str, kind: str | None, subject: str | None,
               origin_key: str, evidence_key: str, ts: float, session_id: str = "core") -> Claim:
    """Verbatim core-copy claim (source core:<target>, in_core, core_sha, user evidence)."""
    from . import normalize
    cf = _corefmt()
    subj = subject or (cf.entry_label(text) or "").strip("*: ").strip() or text[:30]
    c = Claim(origin_key=origin_key, source=f"core:{target}", kind=kind or "fact",
              target="user" if target == "user" else "agent",
              subject=subj, text=text, level=3, explicit=True, event_time=ts,
              evidence_refs=[], evidence_keys=[evidence_key], evidence_roles=["core"],
              session_ids=[session_id], first_seen_at=ts, last_seen_at=ts, last_user_evidence_at=ts,
              user_evidence_count=1, user_session_count=1, explicit_user=True,
              core_target=target, core_sha=cf.core_sha(text), in_core=True)
    normalize.normalize_claim(c, cfg=ctx.cfg, paths=ctx.paths)
    return c


def embed_claims(ctx: Any, claims: list[Claim]) -> None:
    todo = [c for c in claims if c.vector is None]
    if not todo:
        return
    texts = [embed_input(c.subject, c.text) for c in todo]
    vecs = ctx.embedder.embed(texts)
    for c, t, v in zip(todo, texts, vecs):
        c.embed_text = t
        c.vector = v


def demote_core_row(ctx: Any, ws: Any, row: MemoryRow, *, at: float, reason: str) -> None:
    """Entry left MEMORY.md/USER.md: in_core=False, the row stays (demotion, §7.3). Until now the
    entry was in every system prompt, i.e. in use: last_used_at = removal time restarts its strength
    clock, and a copy that went dormant meanwhile is active again (DEVIATIONS F-15)."""
    cur = ws.get(row.id) or row
    changes: dict[str, Any] = {"in_core": False}
    if cur.kind != "legacy":
        changes["last_used_at"] = max(float(cur.last_used_at or 0.0), float(at))
        if cur.status == "dormant":
            changes["status"] = "active"
    ws.update(row.id, changes, op="core_mirror", reason=reason, user_evidence=True)


def find_core_row(ws: Any, target: str, sha: str) -> MemoryRow | None:
    """Row mirroring core entry (target, sha): by origin key first, then by core_sha."""
    r = ws.by_origin_key(f"core:{target}:{sha}")
    if r is not None and r.status not in CANDIDATE_SEARCH_EXCLUDED:
        return r
    best = None
    for row in ws.rows.values():
        if row.core_sha == sha and (row.core_target in (None, target)) \
                and row.status not in CANDIDATE_SEARCH_EXCLUDED:
            if best is None or (row.status == "active", row.created_at) > (best.status == "active", best.created_at):
                best = row
    return best


# ── R5 ───────────────────────────────────────────────────────────────────────

@dataclass
class CoreCheckResult:
    mirrored_adds: list[CoreEntry] = field(default_factory=list)
    mirrored_removes: list[CoreSeenRow] = field(default_factory=list)
    core_seen: list[CoreSeenRow] = field(default_factory=list)     # → LedgerDelta.core_seen (changed only)
    required_missing: list[str] = field(default_factory=list)


def diff_core(core: dict[str, list[CoreEntry]], seen: dict[tuple[str, str], CoreSeenRow]
              ) -> tuple[list[CoreEntry], list[CoreSeenRow]]:
    """(entries new or previously absent, seen rows present before but missing now)."""
    current = {(e.target, e.sha) for es in core.values() for e in es}
    adds = [e for es in core.values() for e in es
            if (e.target, e.sha) not in seen or not seen[(e.target, e.sha)].present]
    removes = [s for k, s in sorted(seen.items()) if s.present and k not in current]
    return adds, removes


def _entry_vectors(core: dict[str, list[CoreEntry]], embedder: Any, cache: dict | None) -> list[tuple[CoreEntry, Any]]:
    entries = [e for es in core.values() for e in es]
    if cache is not None and "vecs" in cache:
        return cache["vecs"]
    vecs = embedder.embed([e.text for e in entries]) if entries else []
    pairs = list(zip(entries, vecs))
    if cache is not None:
        cache["vecs"] = pairs
    return pairs


def core_required_present(row: MemoryRow, core: dict[str, list[CoreEntry]], *, embedder: Any,
                          min_cos: float, _cache: dict | None = None) -> bool:
    """Present iff some entry's core_norm contains the row's core_norm text, or cos(entry
    embedding, row vector) ≥ min_cos (embedding only when the substring test fails)."""
    from .vecutil import cos
    cf = _corefmt()
    needle = cf.core_norm(row.text)
    entries = [e for es in core.values() for e in es]
    if needle and any(needle in cf.core_norm(e.text) for e in entries):
        return True
    if row.vector is None or embedder is None or not entries:
        return False
    return any(cos(v, row.vector) >= min_cos for _e, v in _entry_vectors(core, embedder, _cache))


def _seen_text(text: str) -> str:
    """core_seen.text / report text with secret spans masked (ledger + its backups and plan.json
    must never hold a secret a core entry happened to contain)."""
    try:
        from .threat import redact_secrets
        return redact_secrets(text or "")[0]
    except Exception:  # noqa: BLE001
        return text or ""


def check_core(ctx: Any, ws: Any, upserter: Any, core: dict[str, list[CoreEntry]]) -> CoreCheckResult:
    from .upsert import row_from_claim
    cf = _corefmt()
    run_id, now = ctx.run_id, float(ctx.now)
    seen = ctx.ledger.core_seen() if ctx.ledger is not None else {}
    res = CoreCheckResult()
    adds, removes = diff_core(core, seen)

    def seen_row(target: str, sha: str, text: str, mid: str | None, present: bool) -> None:
        text = _seen_text(text)
        old = seen.get((target, sha))
        if old is not None and old.present == present and old.memory_id == mid and old.text == text:
            return
        res.core_seen.append(CoreSeenRow(target=target, entry_sha=sha, text=text, memory_id=mid,
                                         first_seen_run=old.first_seen_run if old else run_id,
                                         last_seen_run=run_id, present=present))

    # unchanged present entries whose memory_id was never resolved get linked when a row exists
    for es in core.values():
        for e in es:
            old = seen.get((e.target, e.sha))
            if old is not None and old.present and old.memory_id is None:
                r = find_core_row(ws, e.target, e.sha)
                if r is not None:
                    seen_row(e.target, e.sha, e.text, r.id, True)

    to_mirror: list[CoreEntry] = []
    for e in adds:
        if cf.is_episodic(e.text) or is_fragment(e.text):
            seen_row(e.target, e.sha, e.text, None, True)
            continue
        if ctx.scanner is not None and ctx.scanner.secrets(e.text):
            ctx.note(f"핵심 파일 항목에 비밀값 패턴이 있어 장기기억으로 옮기지 않았습니다 ({e.target} #{e.index}).")
            seen_row(e.target, e.sha, _seen_text(e.text), None, True)     # never the raw secret (F-5)
            continue
        row = find_core_row(ws, e.target, e.sha)
        if row is not None:
            if not row.in_core:
                ws.update(row.id, {"in_core": True, "core_target": e.target}, op="core_mirror",
                          reason="core_mirror_add", user_evidence=True)
            seen_row(e.target, e.sha, e.text, row.id, True)
            continue
        to_mirror.append(e)

    if to_mirror:
        cls = classify_texts(ctx, [e.text for e in to_mirror])
        claims: list[tuple[CoreEntry, Claim]] = []
        for e, (kind, subj, _frag) in zip(to_mirror, cls):
            c = core_claim(ctx, target=e.target, text=e.text, kind=kind, subject=subj,
                           origin_key=f"core:{e.target}:{e.sha}", evidence_key=f"c:{e.target}:{e.sha}",
                           ts=now)
            claims.append((e, c))
        embed_claims(ctx, [c for _e, c in claims])
        for e, c in claims:
            existing = ws.by_origin_key(c.origin_key)
            if existing is None:
                row = row_from_claim(c, ctx=ctx)
                ws.insert(row, reason="core_mirror_add", user_evidence=True)
                ctx.stats.created += 1
                ctx.report.created.append({"id": row.id, "kind": row.kind, "tier": row.tier,
                                           "text": row.text, "importance": row.importance})
                mid = row.id
                if upserter is not None and hasattr(upserter, "absorb_into_core_copy"):
                    upserter.absorb_into_core_copy(row.id)
            else:
                mid = existing.id
            seen_row(e.target, e.sha, e.text, mid, True)
            res.mirrored_adds.append(e)
            ctx.stats.core_changes += 1
            ctx.report.core_changes.append({"target": e.target, "change": "mirror_add", "text": e.text})

    for s in removes:
        row = ws.get(s.memory_id) if s.memory_id else None
        if row is None:
            row = find_core_row(ws, s.target, s.entry_sha)
        seen_row(s.target, s.entry_sha, s.text, row.id if row else s.memory_id, False)
        if row is not None and not row.in_core:
            continue        # already demoted (R1 core_remove this run): one change, reported once (E2E-4)
        if row is not None:
            demote_core_row(ctx, ws, row, at=now, reason="core_mirror_remove")
        res.mirrored_removes.append(s)
        ctx.stats.core_changes += 1
        ctx.report.core_changes.append({"target": s.target, "change": "mirror_remove",
                                        "text": _seen_text(s.text)})

    cache: dict = {}
    for row in sorted(ws.rows.values(), key=lambda r: r.id):
        if not row.core_required or row.status in CANDIDATE_SEARCH_EXCLUDED:
            continue
        if core_required_present(row, core, embedder=ctx.embedder,
                                 min_cos=float(ctx.cfg.core_match_cos), _cache=cache):
            continue
        res.required_missing.append(row.id)
        target = row.core_target or "user"
        fname = "USER.md" if target == "user" else "MEMORY.md"
        label = (cf.entry_label(row.text) or row.subject or row.text[:30]).strip("*: ").strip()
        pin_note = "pinned로 남아 있음" if row.pinned else "활성 행으로 남아 있음"
        msg = (f"{fname}에서 '{label}'가 빠졌습니다. 장기기억에는 {pin_note}. "
               f"되돌리려면 `yume core-restore {row.id}`.")
        if msg not in ctx.report.notes:       # R1 may have noted it this run; a standing state, not a change
            ctx.note(msg)
    return res


# ── manual writers (yume core-restore / core-proposal apply) ────────────────

@dataclass
class RestoreResult:
    ok: bool
    target: str
    path: str
    backup: str | None = None
    reason: str | None = None


def _limits(paths: Any) -> dict[str, int]:
    try:
        from .sources.core_files import load_limits
        lim = load_limits(paths)
        return {t: int((lim.get(t) or {}).get("limit", DEFAULT_LIMITS[t])) for t in DEFAULT_LIMITS}
    except ImportError:
        return dict(DEFAULT_LIMITS)


def _backup(paths: Any, real: Path, now: float) -> str | None:
    if not real.exists():
        return None
    from .clock import kst_stamp
    d = paths.backups_dir / "core" / kst_stamp(now)
    d.mkdir(parents=True, exist_ok=True)
    for p in (paths.backups_dir, paths.backups_dir / "core", d):
        os.chmod(p, 0o700)
    dst = d / real.name
    n = 2
    while dst.exists():
        dst = d / f"{real.name}.{n}"
        n += 1
    shutil.copy2(real, dst)
    os.chmod(dst, 0o600)
    return str(dst)


class _CoreLock:
    """Hermes MemoryStore._file_lock convention: exclusive flock on "<path>.lock" (unresolved)."""

    def __init__(self, path: Path):
        self.lock_path = Path(str(path) + ".lock")
        self.fd = None

    def __enter__(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = open(self.lock_path, "a+", encoding="utf-8")
        fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        try:
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
        finally:
            self.fd.close()


def _write_entries(path: Path, entries: list[str], limit: int) -> tuple[bool, str | None]:
    cf = _corefmt()
    content = cf.ENTRY_DELIMITER.join(entries)
    if len(content) > limit:
        return False, "limit"
    if cf.parse_entries(content) != [e.strip() for e in entries if e.strip()]:
        return False, "roundtrip"
    real = Path(os.path.realpath(str(path)))
    mode = stat.S_IMODE(real.stat().st_mode) if real.exists() else 0o600
    fd, tmp = tempfile.mkstemp(prefix=".mem_", suffix=".tmp", dir=str(real.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, str(real))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return True, None


def restore(paths: Any, memory_id: str, *, target: str | None, store: Any, ledger: Any,
            now: float) -> RestoreResult:
    """`yume core-restore <id>`: append the row's text back as a core entry (never automatic)."""
    from .types import AuditRow
    cf = _corefmt()
    rows = store.get([memory_id]) if store is not None else {}
    row = rows.get(memory_id)
    tgt = target or (row.core_target if row is not None else None) or "user"
    if tgt not in DEFAULT_LIMITS:
        return RestoreResult(False, tgt, "", reason="bad_target")
    path = paths.core_file(tgt)
    if row is None:
        return RestoreResult(False, tgt, str(path), reason="not_found")
    if row.status in CANDIDATE_SEARCH_EXCLUDED:
        return RestoreResult(False, tgt, str(path), reason=f"status:{row.status}")
    limit = _limits(paths)[tgt]
    real = Path(os.path.realpath(str(path)))
    with _CoreLock(path):
        entries = cf.read_entries(real)
        if any(cf.core_norm(e) == cf.core_norm(row.text) for e in entries):
            return RestoreResult(True, tgt, str(real), reason="already_present")
        new_entries = entries + [row.text.strip()]
        if len(cf.ENTRY_DELIMITER.join(new_entries)) > limit:
            return RestoreResult(False, tgt, str(real), reason="limit")
        backup = _backup(paths, real, now)
        ok, why = _write_entries(path, new_entries, limit)
    if ok and ledger is not None and not getattr(ledger, "readonly", False):
        ledger.add_audit(AuditRow(ts=now, run_id="manual", op="core_restore", memory_id=memory_id,
                                  detail=f"target={tgt}"))
    return RestoreResult(ok, tgt, str(real), backup=backup, reason=why)


def _proposal_target(p: Path) -> str | None:
    name = p.name.upper()
    if name.startswith("MEMORY.MD"):
        return "memory"
    if name.startswith("USER.MD"):
        return "user"
    return None


def apply_proposal(paths: Any, proposal_path: str | None, *, store: Any, now: float) -> RestoreResult:
    """`yume core-proposal apply`: replace a core file with a reviewed proposal. Every entry the
    proposal drops must already exist in Lance (core_sha or identical text), else refused."""
    cf = _corefmt()
    p = Path(proposal_path) if proposal_path else paths.proposals_dir / "MEMORY.md.proposed"
    tgt = _proposal_target(p)
    if tgt is None:
        return RestoreResult(False, "", str(p), reason="bad_proposal_name")
    path = paths.core_file(tgt)
    if not p.exists():
        return RestoreResult(False, tgt, str(path), reason="proposal_missing")
    proposed = cf.read_entries(p)
    limit = _limits(paths)[tgt]
    if len(cf.ENTRY_DELIMITER.join(proposed)) > limit:
        return RestoreResult(False, tgt, str(path), reason="limit")
    rows = store.load_working_set(with_vectors=False) if store is not None else {}
    shas = {r.core_sha for r in rows.values() if r.core_sha and r.status not in ("forgotten", "quarantined")}
    norms = {cf.core_norm(r.text) for r in rows.values() if r.status not in ("forgotten", "quarantined")}
    txt_shas = {text_sha(r.text) for r in rows.values() if r.status not in ("forgotten", "quarantined")}
    real = Path(os.path.realpath(str(path)))
    with _CoreLock(path):
        current = cf.read_entries(real)
        keep = {cf.core_norm(e) for e in proposed}
        dropped = [e for e in current if cf.core_norm(e) not in keep]
        missing = [e for e in dropped if cf.core_sha(e) not in shas and cf.core_norm(e) not in norms
                   and text_sha(e) not in txt_shas]
        if missing:
            return RestoreResult(False, tgt, str(real), reason=f"not_in_store:{len(missing)}")
        backup = _backup(paths, real, now)
        ok, why = _write_entries(path, proposed, limit)
    return RestoreResult(ok, tgt, str(real), backup=backup, reason=why)
