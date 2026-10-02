"""`yume migrate` — M0–M7 (PLAN-v2 §9, CONTRACTS §4.22, user principle U2). Data loss 0.

    M0  inventory: every source enumerated read-only with sha256 (migration/inventory.json)
    M1  core seed: each USER.md / MEMORY.md (non-episodic) entry → one row, text verbatim
        (source core:*, in_core, core_sha); kind/subject via CORE_CLASSIFY; header-only fragments
        are listed in core_map (no row — see DEVIATIONS "migrate")
    M2  auto-pin (U2): seeded USER.md rows of kind profile/rule → pinned (+ core_required).
        No proposal file, no approval; operators only `yume unpin`
    M3  MEMORY.md episodic "Session: …" entries → raw text kept as legacy dormant rows
        (source legacy:memory_md) + md-like windows for extraction
    M4  workspace md backlog (NREM md source)          M6  state.db backfill (NREM state.db)
    M5  old Dreamer dump → legacy dormant rows (source legacy:dreamer, createdAt kept)
    M7  full REM (sweep, guard, R8), export, calibrate note, migration Dream Log, MEMORY.md
        clean-up proposal file (proposals/MEMORY.md.proposed — applied only by a human)

Execution is two committed plans in one invocation: phase A (`<run_id>-m`: direct inserts
M1/M2/M3/M5 + `--statedb-start` watermarks) and phase B (`<run_id>`: nrem.run_nrem + M3 windows
→ rem.run_rem → rem.post_commit). `--dry-run` writes only runs/<id>/plan.json and the `_dry`
Dream Log; phase A is applied to a private temp copy of Lance + ledger so phase B sees it.
MEMORY.md / USER.md are never written (T14).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import clock
from .embedder import embed_input
from .types import (CANDIDATE_SEARCH_EXCLUDED, KIND_BASE, KINDS, WM_ADVANCING_STATUSES,
                    BudgetExceeded, Claim, LedgerDelta, MemoryRow, Message, Plan, RunReport,
                    RunStats, Window, WindowState, make_window_id, new_memory_id, sha256_hex)

log = logging.getLogger("hermesyume.migrate")

STEPS = ("core", "dump", "memory_md", "md", "statedb")
AUTO_PIN_KINDS = frozenset({"profile", "rule"})
SEED_SUFFIX = "-m"
DUMP_FILENAME = "dreamer_memories.json"
PROPOSAL_FILENAME = "MEMORY.md.proposed"
CLASSIFY_BATCH = 40
FALLBACK_KIND = {"user": "preference", "memory": "reference"}   # protected kinds → durable (core:*)
EPISODE_DATE_RE = re.compile(r"^\s*Session:\s*(\d{4}-\d{2}-\d{2})")
_HANGUL_RE = re.compile(r"[가-힣]")

# --estimate assumptions (rough, deliberately pessimistic)
CHARS_PER_TOKEN = 1.5
EST_CLAIMS_PER_WINDOW = 1.5
EST_JUDGE_PER_CLAIM = 0.6
EXTRACT_COMPLETION_TOKENS = 300
JUDGE_PROMPT_TOKENS = 900
JUDGE_COMPLETION_TOKENS = 60
CLASSIFY_COMPLETION_PER_ITEM = 30
EMBED_TOKENS_PER_CLAIM = 80


@dataclass
class MigrateResult:
    plan: Plan | None
    core_map_path: str | None
    pins_proposed_path: str | None          # always None (U2); kept for shape stability
    estimate: dict | None
    ok: bool
    problems: list[str]
    status: str = ""                        # committed | held | dry | failed | refused | estimate
    seed_plan: Plan | None = None
    core_map: dict | None = None
    inventory: dict | None = None
    pinned_ids: list[str] = field(default_factory=list)
    proposal_path: str | None = None


# ── small helpers ────────────────────────────────────────────────────────────

def _corefmt(paths: Any = None):
    from .paths import load_provider_module
    return load_provider_module("corefmt", paths)


def _cfg(cfg: Any, key: str, default: Any) -> Any:
    try:
        return cfg[key]
    except (KeyError, TypeError):
        return getattr(cfg, key, default)


def _lang(text: str) -> str:
    return "ko" if _HANGUL_RE.search(text or "") else "en"


def _sha256_file(p: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd, tmp = tempfile.mkstemp(prefix=".mig_", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def parse_only(spec: str | Iterable[str] | None) -> set[str]:
    """`--only core,dump` → {"core","dump"}; None/"" → every step. Unknown names raise ValueError."""
    if spec is None or spec == "":
        return set(STEPS)
    items = [s.strip() for s in (spec.split(",") if isinstance(spec, str) else spec) if s and s.strip()]
    bad = [s for s in items if s not in STEPS]
    if bad:
        raise ValueError(f"알 수 없는 단계: {', '.join(bad)} (가능: {', '.join(STEPS)})")
    return set(items) or set(STEPS)


def parse_statedb_start(spec: str | None, now: float) -> float | None:
    """`--statedb-start now | ISO | -Nd` → epoch seconds (None when not given)."""
    if spec is None or str(spec).strip() == "":
        return None
    s = str(spec).strip()
    if s.lower() == "now":
        return float(now)
    return float(clock.parse_now_spec(s, base=now))


def _is_header_only(text: str, cf: Any) -> bool:
    """'**Reading List:**' — a label with no value (deterministic fragment rule)."""
    lab = cf.entry_label(text)
    if not lab:
        return False
    rest = text.strip()[len(lab):] if text.strip().startswith(lab) else text.replace(lab, "", 1)
    return not re.sub(r"[\s*:：\-–—·•|]+", "", rest)


def _label_subject(text: str, cf: Any) -> str | None:
    lab = cf.entry_label(text)
    if not lab:
        return None
    s = lab.strip().strip("*").strip().rstrip(":：").strip().strip("*").strip()
    return s or None


def _secret_types(ctx: Any, text: str) -> list[str]:
    sc = getattr(ctx, "scanner", None)
    if sc is not None:
        try:
            return sorted(set(sc.secrets(text or "")))
        except Exception:  # noqa: BLE001
            pass
    from .threat import secret_types
    return sorted(set(secret_types(text or "")))


def _secret_free(text: str) -> bool:
    """Deterministic secret check (same regexes as the scanner's secrets())."""
    from .threat import secret_types
    return not secret_types(text or "")


def _model_id(ctx: Any) -> str:
    return getattr(ctx.embedder, "model_id", None) or ctx.cfg.embed_model_id()


# ── M0 inventory ─────────────────────────────────────────────────────────────

def _md_files(cfg: Any) -> tuple[list[Path], list[str]]:
    import fnmatch
    globs = list(_cfg(cfg, "md_exclude_globs", []) or [])
    files: list[Path] = []
    excluded: list[str] = []
    for root in _cfg(cfg, "md_sources", []) or []:
        r = Path(os.path.expanduser(str(root)))
        if not r.is_dir():
            continue
        for p in sorted(r.rglob("*.md")):
            if not p.is_file():
                continue
            if any(fnmatch.fnmatch(p.name, g) for g in globs):
                excluded.append(str(p))
            else:
                files.append(p)
    return files, excluded


def default_dump_path(paths: Any) -> Path:
    return Path(paths.migration_dir) / DUMP_FILENAME


def load_dump(path: str | os.PathLike) -> list[dict]:
    """Old Dreamer dump: a JSON list of {id, text, importance, category, createdAt(ms)} (or an
    object wrapping such a list under memories/rows/items/data)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k in ("memories", "rows", "items", "data"):
            if isinstance(data.get(k), list):
                data = data[k]
                break
    if not isinstance(data, list):
        raise ValueError("Dreamer 덤프 형식이 아닙니다 (JSON 배열 필요)")
    return [r for r in data if isinstance(r, dict) and isinstance(r.get("text"), str) and r["text"].strip()]


def inventory(ctx: Any, *, dump_path: str | None = None) -> dict:
    """M0: read-only enumeration of every migration source with sha256 (state.db: size and row
    counts — a live WAL database has no stable file hash)."""
    paths, cfg = ctx.paths, ctx.cfg
    inv: dict[str, Any] = {"generated_at": float(ctx.now), "kst": clock.fmt_kst(ctx.now),
                           "run_id": ctx.run_id}
    sdb = Path(paths.state_db)
    info: dict[str, Any] = {"path": str(sdb), "exists": sdb.exists()}
    if sdb.exists():
        info["size"] = sdb.stat().st_size
        try:
            from .sqlite_util import open_for_read
            with open_for_read(sdb, pure=bool(ctx.dry_run)) as conn:
                for t in ("sessions", "messages"):
                    try:
                        info[t] = int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
                    except sqlite3.Error:
                        info[t] = None
        except Exception as e:  # noqa: BLE001 — inventory is informational
            info["error"] = type(e).__name__
    inv["state_db"] = info

    from .sources.core_files import read_core
    core = read_core(paths)
    inv["core"] = {}
    for target in ("user", "memory"):
        p = Path(paths.core_file(target))
        ents = core.get(target, [])
        inv["core"][target] = {"path": str(p), "exists": p.exists(), "sha256": _sha256_file(p) if p.exists() else None,
                               "entries": len(ents), "chars": sum(len(e.text) for e in ents)}

    files, excluded = _md_files(cfg)
    inv["md"] = [{"path": str(p), "size": p.stat().st_size, "sha256": _sha256_file(p)} for p in files]
    inv["md_excluded"] = excluded
    dp = Path(dump_path) if dump_path else default_dump_path(paths)
    dinfo: dict[str, Any] = {"path": str(dp), "exists": dp.is_file()}
    if dp.is_file():
        dinfo["sha256"] = _sha256_file(dp)
        try:
            dinfo["rows"] = len(load_dump(dp))
        except (OSError, ValueError) as e:
            dinfo["error"] = type(e).__name__
    inv["dreamer_dump"] = dinfo
    cy = Path(paths.hermes_config_yaml)
    inv["config_yaml"] = {"path": str(cy), "sha256": _sha256_file(cy) if cy.exists() else None}
    ws_dir = str(_cfg(cfg, "workspace_dir", "") or "").strip()
    v1 = Path(ws_dir) / "hermesyume-state" if ws_dir else None
    inv["v1_state"] = {"path": str(v1) if v1 else None, "exists": bool(v1 and v1.is_dir()),
                       "files": sorted(str(p.relative_to(v1)) for p in v1.rglob("*") if p.is_file())
                       if v1 is not None and v1.is_dir() else []}
    return inv


# ── M1 core seed ─────────────────────────────────────────────────────────────

def _classify(ctx: Any, texts: list[str]) -> list[tuple[str | None, str | None]]:
    """Batched CORE_CLASSIFY → [(kind|None, subject|None)]. LLM/budget failures → (None, None);
    401 propagates (the run aborts)."""
    from . import prompts
    from .llm import LLMAuthError, LLMError
    out: list[tuple[str | None, str | None]] = [(None, None)] * len(texts)
    cfg = ctx.cfg
    for start in range(0, len(texts), CLASSIFY_BATCH):
        chunk = texts[start:start + CLASSIFY_BATCH]
        try:
            resp = ctx.llm.chat_json("core_classify", prompts.core_classify_messages(chunk),
                                     model=cfg.extract_model,
                                     max_tokens=int(_cfg(cfg, "core_classify_max_tokens", 1500)),
                                     temperature=float(_cfg(cfg, "llm_temperature", 0.0)))
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
            out[start + i] = (kind if kind in KINDS else None, subj)
    return out


def _find_core_row(ws: Any, target: str, sha: str) -> MemoryRow | None:
    r = ws.by_origin_key(f"core:{target}:{sha}")
    if r is not None and r.status not in CANDIDATE_SEARCH_EXCLUDED:
        return r
    best = None
    for row in ws.rows.values():
        if row.core_sha == sha and row.core_target in (None, target) and row.kind != "legacy" \
                and row.status not in CANDIDATE_SEARCH_EXCLUDED:
            if best is None or (row.status == "active", row.created_at) > (best.status == "active", best.created_at):
                best = row
    return best


def _legacy_key(sha: str) -> str:
    return f"x:memory_md:{sha}"


def build_core_map(ws: Any, core: dict[str, list[Any]], *, steps: Iterable[str] = STEPS,
                   cf: Any = None, skipped: dict[tuple[str, str], str] | None = None) -> dict:
    """core_map: every USER.md/MEMORY.md entry → its row (core copy or legacy), a fragment, or a
    reason. `ok` = every entry in scope of `steps` is accounted; `complete` = all are."""
    cf = cf or _corefmt()
    steps = set(steps)
    skipped = skipped or {}
    entries = []
    for target in ("user", "memory"):
        for e in core.get(target, []):
            episodic = bool(cf.is_episodic(e.text))
            rec: dict[str, Any] = {"target": target, "index": e.index, "sha": e.sha, "memory_id": None,
                                   "fragment": False, "kind": None, "status": "pending",
                                   "label": e.label}
            if episodic:
                row = ws.by_origin_key(_legacy_key(e.sha))
                if row is not None and row.status not in CANDIDATE_SEARCH_EXCLUDED:
                    rec.update(memory_id=row.id, kind=row.kind, status="legacy")
                elif (target, e.sha) in skipped:
                    rec["status"] = skipped[(target, e.sha)]
                rec["in_scope"] = "memory_md" in steps
            elif _is_header_only(e.text, cf):
                rec.update(fragment=True, status="fragment", text=e.text)
                rec["in_scope"] = "core" in steps
            else:
                row = _find_core_row(ws, target, e.sha)
                if row is not None:
                    rec.update(memory_id=row.id, kind=row.kind, status="row",
                               pinned=bool(row.pinned))
                elif (target, e.sha) in skipped or _secret_free(e.text) is False:
                    rec["status"] = skipped.get((target, e.sha), "secret")
                rec["in_scope"] = "core" in steps
            rec["accounted"] = rec["status"] in ("row", "legacy", "fragment", "secret")
            entries.append(rec)
    total = len(entries)
    accounted = sum(1 for r in entries if r["accounted"])
    return {"entries": entries, "total": total, "accounted": accounted,
            "rows": sum(1 for r in entries if r["status"] in ("row", "legacy")),
            "fragments": sum(1 for r in entries if r["fragment"]),
            "secret_skipped": sum(1 for r in entries if r["status"] == "secret"),
            "pending": sum(1 for r in entries if not r["accounted"]),
            "ok": all(r["accounted"] for r in entries if r["in_scope"]),
            "complete": accounted == total}


def seed_core(ctx: Any, ws: Any, upserter: Any, core: dict[str, list[Any]], *,
              steps: Iterable[str] = STEPS) -> dict:
    """M1: one row per non-episodic core entry, text verbatim (source core:<target>, in_core,
    core_sha), kind/subject via CORE_CLASSIFY. Idempotent: entries that already have a row are
    reused. Returns the core_map. `upserter` is unused (seeds are inserted directly, no judge)."""
    from . import normalize
    from .upsert import row_from_claim
    cf = _corefmt(ctx.paths)
    now = float(ctx.now)
    todo: list[tuple[str, Any]] = []
    skipped: dict[tuple[str, str], str] = {}
    secret_hits: list[str] = []
    seen: set[tuple[str, str]] = set()
    for target in ("user", "memory"):
        for e in core.get(target, []):
            if cf.is_episodic(e.text) or _is_header_only(e.text, cf) or (target, e.sha) in seen:
                continue
            seen.add((target, e.sha))
            types = _secret_types(ctx, e.text)
            if types:
                skipped[(target, e.sha)] = "secret"
                secret_hits.append(f"{target} #{e.index} ({', '.join(types)})")
                continue
            if _find_core_row(ws, target, e.sha) is not None:
                continue
            todo.append((target, e))
    if secret_hits:
        ctx.alert("secret_found",
                  f"핵심 파일 항목에서 비밀값 패턴을 발견해 장기기억으로 옮기지 않았습니다: {', '.join(secret_hits)}. "
                  "핵심 파일에서 직접 지우고 키를 재발급하세요.", level="error")
    if todo:
        cls = _classify(ctx, [e.text for _t, e in todo])
        claims: list[Claim] = []
        fallback = 0
        for (target, e), (kind, subj) in zip(todo, cls):
            if kind is None:
                kind = FALLBACK_KIND[target]
                fallback += 1
            subject = subj or _label_subject(e.text, cf) or " ".join(e.text.split())[:30]
            c = Claim(origin_key=f"core:{target}:{e.sha}", source=f"core:{target}", kind=kind,
                      target="user" if target == "user" else "agent", subject=subject, text=e.text,
                      level=3, explicit=True, event_time=now, evidence_refs=[],
                      evidence_keys=[f"c:{target}:{e.sha}"], evidence_roles=["core"],
                      session_ids=[f"core:{target}"], first_seen_at=now, last_seen_at=now,
                      last_user_evidence_at=now, user_evidence_count=1, user_session_count=1,
                      explicit_user=True, core_target=target, core_sha=e.sha, in_core=True,
                      lang=_lang(e.text))
            normalize.normalize_claim(c, cfg=ctx.cfg, paths=ctx.paths)
            claims.append(c)
        if fallback:
            ctx.note(f"핵심 항목 {fallback}개는 분류에 실패해 보호 종류(USER.md→선호, MEMORY.md→참조)로 넣었습니다.")
        inputs = [embed_input(c.subject, c.text) for c in claims]
        vecs = ctx.embedder.embed(inputs)
        for c, t, v in zip(claims, inputs, vecs):
            c.embed_text, c.vector = t, v
            row = row_from_claim(c, ctx=ctx)
            ws.insert(row, reason="migration:core_seed", user_evidence=True)
            ctx.stats.created += 1
            ctx.report.created.append({"id": row.id, "kind": row.kind, "tier": row.tier,
                                       "text": row.text, "importance": row.importance})
    return build_core_map(ws, core, steps=steps, cf=cf, skipped=skipped)


def seeded_ids(core_map: dict) -> dict[str, str]:
    """{"<target>:<sha>": memory_id} for core-copy rows (not legacy, not fragments)."""
    return {f"{r['target']}:{r['sha']}": r["memory_id"] for r in core_map.get("entries", [])
            if r.get("status") == "row" and r.get("memory_id")}


# ── M2 auto-pin (U2) ─────────────────────────────────────────────────────────

def auto_pin(ctx: Any, ws: Any, seeded: dict[str, str]) -> list[str]:
    """Pin every seeded USER.md row of kind profile/rule (also `core_required`, so a later removal
    from USER.md is noted in the Dream Log). No proposal file, no approval."""
    pinned: list[str] = []
    for key, mid in seeded.items():                  # core-file order
        target = key.split(":", 1)[0]
        row = ws.get(mid)
        if target != "user" or row is None or row.status != "active" or row.kind not in AUTO_PIN_KINDS:
            continue
        changes: dict[str, Any] = {}
        if not row.pinned:
            changes["pinned"] = True
        if not row.core_required:
            changes["core_required"] = True
        if not changes:
            continue
        if ws.update(mid, changes, op="pin", reason="migration:auto_pin", user_evidence=True) is None:
            continue
        if "pinned" in changes:
            pinned.append(mid)
            ctx.stats.pinned_new += 1
            ctx.report.new_pins.append({"id": mid, "text": row.text})
    if pinned:
        total = sum(len(ws.get(m).text) for m in pinned)
        budget = int(_cfg(ctx.cfg, "pins_budget_chars", 800))
        ctx.note(f"USER.md의 프로필·규칙 항목 {len(pinned)}개를 자동으로 pin했습니다 (해제는 `yume unpin <id>`).")
        if total > budget:
            ctx.note(f"자동 pin 텍스트 합계 {total}자가 pin 예산 {budget}자를 넘습니다. 핵심 파일에 있는 동안은 "
                     "영향이 없고, 빠지면 [고정 기억]에 다 들어가지 못한 pin은 회상으로 받칩니다.")
    return pinned


# ── M3 MEMORY.md legacy, M5 Dreamer dump ─────────────────────────────────────

def _legacy_row(ctx: Any, *, text: str, subject: str, source: str, origin_key: str, vector: Any,
                created_at: float, event_time: float | None, session_id: str,
                core_target: str | None = None, core_sha: str | None = None,
                in_core: bool = False) -> MemoryRow:
    from .normalize import subject_key
    now = float(ctx.now)
    return MemoryRow(
        id=new_memory_id(), text=text, subject=subject, subject_key=subject_key(subject) if subject else "",
        vector=vector, embed_model=_model_id(ctx), kind="legacy", tier="legacy", target="user",
        importance=float(KIND_BASE["legacy"]), level=3, core_target=core_target, core_sha=core_sha,
        in_core=in_core, status="dormant", status_reason=source, status_changed_at=now,
        created_at=float(created_at), updated_at=now, event_time=event_time, valid_from=event_time,
        first_seen_at=float(created_at), last_seen_at=float(created_at), evidence_count=1,
        source=source, source_session_ids=[session_id], source_message_ids=[origin_key],
        origin_keys=[origin_key], lang=_lang(text), scope=str(_cfg(ctx.cfg, "scope", "default")),
        last_run_id=ctx.run_id, schema_version=2)


def _report_created(ctx: Any, row: MemoryRow) -> None:
    ctx.stats.created += 1
    ctx.report.created.append({"id": row.id, "kind": row.kind, "tier": row.tier, "text": row.text,
                               "importance": row.importance})


def _episode_date(text: str, now: float) -> tuple[str, float]:
    m = EPISODE_DATE_RE.match(text or "")
    date = m.group(1) if m else clock.kst_date(now)
    return date, clock.parse_iso(f"{date}T12:00") or float(now)


def _redacted(ctx: Any, text: str, where: str, hits: list[str]) -> str:
    types = _secret_types(ctx, text)
    if not types:
        return text
    from .threat import redact_secrets
    hits.append(f"{where} ({', '.join(types)})")
    return redact_secrets(text)[0]


def legacy_memory_md(ctx: Any, ws: Any, entries: list[Any]) -> int:
    """M3 raw part: each episodic MEMORY.md entry → one legacy dormant row (source
    legacy:memory_md, core_sha so the clean-up proposal can verify it). Extraction of the same
    entries happens in phase B (`extract_memory_md`). Returns rows inserted."""
    cf = _corefmt(ctx.paths)
    todo = []
    hits: list[str] = []
    for e in entries:
        if not cf.is_episodic(e.text) or ws.by_origin_key(_legacy_key(e.sha)) is not None:
            continue
        if any(e.sha == x.sha for x in todo):
            continue
        todo.append(e)
    if not todo:
        return 0
    texts = [_redacted(ctx, e.text, f"MEMORY.md #{e.index}", hits) for e in todo]
    subjects = [f"레거시 세션 요약 {_episode_date(e.text, ctx.now)[0]}" for e in todo]
    vecs = ctx.embedder.embed([embed_input(s, t) for s, t in zip(subjects, texts)])
    for e, text, subj, v in zip(todo, texts, subjects, vecs):
        _date, ets = _episode_date(e.text, ctx.now)
        row = _legacy_row(ctx, text=text, subject=subj, source="legacy:memory_md",
                          origin_key=_legacy_key(e.sha), vector=v, created_at=float(ctx.now),
                          event_time=ets, session_id="core:memory", core_target="memory",
                          core_sha=e.sha, in_core=True)
        ws.insert(row, reason="migration:legacy_memory_md")
        _report_created(ctx, row)
    if hits:
        ctx.alert("secret_found", f"MEMORY.md 레거시 항목에서 비밀값 패턴을 가려서 보존했습니다: {', '.join(hits)}",
                  level="error")
    ctx.note(f"MEMORY.md 레거시 세션 조각 {len(todo)}개를 원문 그대로 휴면(레거시) 행으로 보존했습니다.")
    return len(todo)


def import_dreamer_dump(ctx: Any, ws: Any, path: str | os.PathLike | None) -> int:
    """M5: old Dreamer dump rows → kind legacy, status dormant, source legacy:dreamer,
    created_at = createdAt (ms). Searchable only with include_inactive. Returns rows inserted."""
    p = Path(path) if path else default_dump_path(ctx.paths)
    if not p.is_file():
        ctx.note(f"Dreamer 덤프 파일이 없어 M5를 건너뜁니다 ({p.name}).")
        return 0
    rows = load_dump(p)
    hits: list[str] = []
    todo = []
    for n, r in enumerate(rows):
        key = f"x:dreamer:{r.get('id') or n}"
        if ws.by_origin_key(key) is not None or any(k == key for k, _r in todo):
            continue
        todo.append((key, r))
    if not todo:
        return 0
    texts = [_redacted(ctx, str(r["text"]).strip(), f"덤프 {key}", hits) for key, r in todo]
    vecs = ctx.embedder.embed([embed_input("", t) for t in texts])
    for (key, r), text, v in zip(todo, texts, vecs):
        try:
            created = float(r.get("createdAt")) / 1000.0
        except (TypeError, ValueError):
            created = float(ctx.now)
        row = _legacy_row(ctx, text=text, subject="", source="legacy:dreamer", origin_key=key,
                          vector=v, created_at=created, event_time=None, session_id="legacy:dreamer")
        ws.insert(row, reason="migration:dreamer_dump")
        _report_created(ctx, row)
    if hits:
        ctx.alert("secret_found", f"Dreamer 덤프에서 비밀값 패턴을 가려서 보존했습니다: {len(hits)}건", level="error")
    ctx.note(f"옛 Dreamer 덤프 {len(todo)}행을 휴면(레거시) 행으로 보존했습니다 (include_inactive 검색으로만 보임).")
    return len(todo)


# ── M3 extraction windows (phase B) ──────────────────────────────────────────

def memory_md_windows(ctx: Any, entries: list[Any]) -> list[Window]:
    """Each episodic MEMORY.md entry → one md-like window (single agent_log message)."""
    from . import sanitize, windows
    cf = _corefmt(ctx.paths)
    out: list[Window] = []
    seen: set[str] = set()
    rep = sanitize.SanitizeReport()
    for e in entries:
        if not cf.is_episodic(e.text):
            continue
        date, base_ts = _episode_date(e.text, ctx.now)
        root = f"core:memory:{e.sha}"
        msg = Message(ref=f"L#mem:{e.index}", key=_legacy_key(e.sha), role="agent_log", text=e.text,
                      ts=base_ts, source="md", session_id="core:memory", msg_id=0, line=None,
                      platform="md")
        msgs = sanitize.sanitize_messages([msg], cfg=ctx.cfg, repeat_lines=set(), report=rep)
        if not msgs:
            continue
        body = windows.render_messages(msgs)
        title = f"MEMORY.md 레거시 #{e.index}"
        header = windows.format_header(platform="md", title=title, start_ts=base_ts, end_ts=base_ts,
                                       ref_ts=ctx.now, md_date=date)
        nbytes = len(e.text.encode("utf-8"))
        wid = make_window_id("md", root, 0, nbytes, content_sha=sha256_hex(body))
        if wid in seen:
            continue
        seen.add(wid)
        out.append(Window(window_id=wid, source="md", root=root, first_id=0, last_id=nbytes,
                          start_ts=base_ts, last_ts=base_ts, platform="md", title=title, header=header,
                          text=f"{header}\n\n{windows.BODY_HEADING}\n{body}", messages=list(msgs),
                          context=[], session_ids=["core:memory"], md_path=None, md_date=date,
                          md_slug=f"memory-md-{e.index}"))
    return out


def _wstate(w: Window, status: str, *, run_id: str, attempts: int, error: str | None = None,
            n_claims: int = 0) -> WindowState:
    return WindowState(window_id=w.window_id, source=w.source, root_session_id=w.root,
                       first_id=int(w.first_id), last_id=int(w.last_id), last_ts=float(w.last_ts),
                       status=status, attempts=int(attempts), last_error=(error or None) and error[:500],
                       run_id=run_id, n_claims=int(n_claims))


def extract_memory_md(ctx: Any, nres: Any, entries: list[Any]) -> int:
    """N3–N6 for the M3 windows; results are merged into `nres` (claims, windows, window_states,
    rejections) before rem.run_rem. Returns windows extracted this run."""
    from dataclasses import asdict
    from . import extract, gates, normalize
    cfg, st, rep, led = ctx.cfg, ctx.stats, ctx.report, ctx.ledger
    wins = memory_md_windows(ctx, entries)
    if not wins:
        return 0
    failed_prev: dict[str, int] = {}
    if led is not None:
        for w in led.windows(status="failed"):
            failed_prev[w.root_session_id] = max(failed_prev.get(w.root_session_id, 0), int(w.attempts or 0))
    cap = max(1, int(_cfg(cfg, "max_windows_per_run", 60)))
    max_attempts = int(_cfg(cfg, "window_max_attempts", 3))
    states: dict[str, WindowState] = {}
    processed: list[tuple[Window, list[Claim]]] = []
    rejections: list[Any] = []
    considered: list[Window] = []
    deferred = 0
    n_llm = 0
    stop = False
    for w in wins:
        prev = led.get_window(w.window_id) if led is not None else None
        if prev is not None and prev.status in WM_ADVANCING_STATUSES:
            st.bump_reason("excluded", "window_already_done")
            continue
        considered.append(w)
        rep.inputs.append({"source": "memory_md", "root": w.root, "title": w.title, "messages": 1})
        attempts = failed_prev.get(w.root, 0)
        if sum(len(m.text or "") for m in w.messages) < int(_cfg(cfg, "window_min_user_chars", 20)):
            states[w.window_id] = _wstate(w, "empty", run_id=ctx.run_id, attempts=attempts)
            continue
        if stop or n_llm >= cap or ctx.budget.expired():
            deferred += 1
            continue
        n_llm += 1
        try:
            res = extract.extract_window(w, llm=ctx.llm, cfg=cfg)
        except BudgetExceeded as e:
            ctx.note(f"예산 소진({e.what})으로 MEMORY.md 레거시 창 일부를 다음 실행으로 미룹니다.")
            stop = True
            deferred += 1
            continue
        if res.status == "ok":
            cs, rj = gates.gate_claims(res.claims, w, scanner=ctx.scanner, cfg=cfg, now=ctx.now)
            for c in cs:
                normalize.normalize_claim(c, cfg=cfg, paths=ctx.paths)
            states[w.window_id] = _wstate(w, "ok", run_id=ctx.run_id, attempts=attempts, n_claims=len(cs))
            processed.append((w, cs))
            rejections += list(res.schema_rejections) + list(rj)
            st.claims_extracted += len(res.claims) + len(res.schema_rejections)
            continue
        attempts += 1
        if attempts >= max_attempts:
            states[w.window_id] = _wstate(w, "quarantined", run_id=ctx.run_id, attempts=attempts, error=res.error)
            ctx.alert("window_quarantined",
                      f"추출이 {attempts}회 실패한 MEMORY.md 레거시 창을 격리하고 넘어갑니다 ({w.title}). "
                      "원문은 휴면 행과 MEMORY.md에 그대로 남아 있습니다.",
                      window_id=w.window_id, source=w.source, root=w.root)
        else:
            states[w.window_id] = _wstate(w, "failed", run_id=ctx.run_id, attempts=attempts, error=res.error)
            ctx.note(f"MEMORY.md 레거시 창 추출 실패({attempts}/{max_attempts}): {w.title}")

    claims = [c for _w, cs in processed for c in cs]
    if claims:
        inputs = [embed_input(c.subject, c.text) for c in claims]
        try:
            vecs = ctx.embedder.embed(inputs)
        except BudgetExceeded as e:
            ctx.note(f"임베딩 예산 소진({e.what}): MEMORY.md 레거시 창은 다음 실행에서 다시 처리합니다.")
            for w, _cs in processed:
                states.pop(w.window_id, None)
                deferred += 1
            claims, vecs, inputs = [], [], []
        for c, t, v in zip(claims, inputs, vecs):
            c.embed_text, c.vector = t, v

    for s in states.values():
        st.bump({"ok": "windows_ok", "empty": "windows_empty", "failed": "windows_failed",
                 "quarantined": "windows_quarantined"}[s.status])
    st.windows_deferred += deferred
    st.windows_total += len(states) + deferred
    st.claims_rejected += len(rejections)
    for r in rejections:
        st.bump_reason("rejected_by_reason", r.reason)
    for c in claims:
        rep.claims.append({"origin_key": c.origin_key, "kind": c.kind, "subject": c.subject,
                           "text": c.text, "status": c.status})
    rep.rejections += [asdict(r) for r in rejections]
    if deferred:
        ctx.note(f"MEMORY.md 레거시 창 {deferred}개를 다음 실행으로 미뤘습니다.")

    merged = list(getattr(nres, "claims", []) or []) + claims
    merged.sort(key=lambda c: (c.event_time if c.event_time is not None else c.first_seen_at, c.origin_key))
    nres.claims = merged
    nres.windows = list(getattr(nres, "windows", []) or []) + considered
    ws_states = dict(getattr(nres, "window_states", {}) or {})
    ws_states.update(states)
    nres.window_states = ws_states
    nres.rejections = list(getattr(nres, "rejections", []) or []) + rejections
    return n_llm


# ── --statedb-start ──────────────────────────────────────────────────────────

def statedb_start_watermarks(ctx: Any, start: float) -> tuple[dict[str, tuple[float, int]], dict[str, str]]:
    """Every lineage watermark → its latest (ts, id) at or before `start`, without extraction.
    Only advances (never rewinds). Returns (watermarks, new session→root pairs)."""
    paths = ctx.paths
    if not Path(paths.state_db).exists():
        return {}, {}
    from .sources import statedb
    from .sqlite_util import open_for_read
    known = dict(ctx.ledger.all_roots()) if ctx.ledger is not None else {}
    cache = dict(known)
    best: dict[str, tuple[float, int]] = {}
    with open_for_read(paths.state_db, pure=bool(ctx.dry_run)) as conn:
        rows = conn.execute("SELECT session_id, id, timestamp FROM messages WHERE timestamp <= ?",
                            (float(start),)).fetchall()
        for sid, mid, ts in rows:
            if sid is None or ts is None:
                continue
            root = statedb.resolve_root(conn, str(sid), cache)
            cur = best.get(root)
            key = (float(ts), int(mid))
            if cur is None or key > cur:
                best[root] = key
    wm: dict[str, tuple[float, int]] = {}
    for root, key in best.items():
        old = ctx.ledger.get_wm(root) if ctx.ledger is not None else None
        if old is None or key > (float(old.last_ts), int(old.last_id)):
            wm[root] = key
    roots = {s: r for s, r in cache.items() if s not in known}
    return wm, roots


# ── estimate ─────────────────────────────────────────────────────────────────

def _window_chars(w: Window) -> int:
    return sum(len(m.text or "") for m in w.messages if m.role in ("user", "agent_log"))


def _llm_windows_md(ctx: Any) -> list[Window]:
    from . import sanitize, windows
    from .sources import markdown
    cfg = ctx.cfg
    try:
        srcs, _ex = markdown.scan_md_sources(cfg, ctx.ledger, now=ctx.now)
    except TypeError:
        srcs, _ex = markdown.scan_md_sources(cfg, ctx.ledger)
    out: list[Window] = []
    rep = sanitize.SanitizeReport()
    for src in srcs:
        msgs = sanitize.sanitize_messages(list(src.messages), cfg=cfg, repeat_lines=set(), report=rep)
        out += windows.build_windows(source="md", root=src.path, platform="md", title=src.slug,
                                     messages=msgs, context_before=[], cfg=cfg, ref_ts=ctx.now,
                                     end_offset=src.end_offset, md=src)
    return out


def _llm_windows_statedb(ctx: Any, start: float | None) -> list[Window]:
    from . import sanitize, windows
    from .sources import statedb
    from .sqlite_util import open_for_read
    cfg, paths = ctx.cfg, ctx.paths
    if not Path(paths.state_db).exists():
        return []
    ends = ctx.live.session_end_ids() if getattr(ctx, "live", None) is not None else set()
    settle = ctx.settle_minutes if getattr(ctx, "settle_minutes", None) is not None else int(cfg.settle_minutes)
    with open_for_read(paths.state_db, pure=bool(ctx.dry_run)) as conn:
        load = statedb.load_lineages(conn, ledger=ctx.ledger, cfg=cfg, now=ctx.now,
                                     settle_minutes=settle, session_end_ids=ends)
    out: list[Window] = []
    rep = sanitize.SanitizeReport()
    for lin in load.lineages:
        msgs = [m for m in lin.messages if start is None or float(m.ts) > start]
        msgs = sanitize.sanitize_messages(msgs, cfg=cfg, repeat_lines=set(), report=rep)
        out += windows.build_windows(source="statedb", root=lin.root, platform=lin.platform,
                                     title=lin.title, messages=msgs, context_before=[], cfg=cfg,
                                     ref_ts=ctx.now)
    return out


def estimate(ctx: Any, only: set[str], *, statedb_start: float | None = None,
             dump_path: str | None = None) -> dict:
    """Rough cost of the selected steps: {"windows","llm_calls","embed_inputs","usd"} + details.
    Reads only (no LLM, no embedding)."""
    from . import prompts
    from .sources.core_files import read_core
    cfg = ctx.cfg
    only = set(only)
    cf = _corefmt(ctx.paths)
    core = read_core(ctx.paths)
    min_chars = int(_cfg(cfg, "window_min_user_chars", 20))
    led = ctx.ledger

    def todo_windows(ws_: list[Window]) -> list[Window]:
        out = []
        for w in ws_:
            prev = led.get_window(w.window_id) if led is not None else None
            if prev is not None and prev.status in WM_ADVANCING_STATUSES:
                continue
            if _window_chars(w) >= min_chars:
                out.append(w)
        return out

    by_step: dict[str, dict[str, Any]] = {}
    sys_tokens = len(prompts.EXTRACT_SYSTEM) / CHARS_PER_TOKEN
    cls_tokens = len(prompts.CORE_CLASSIFY_SYSTEM) / CHARS_PER_TOKEN

    def windows_cost(ws_: list[Window]) -> dict[str, Any]:
        n = len(ws_)
        claims = n * EST_CLAIMS_PER_WINDOW
        judges = claims * EST_JUDGE_PER_CLAIM
        return {"windows": n, "llm_calls": math.ceil(n + judges),
                "prompt_tokens": int(sum(sys_tokens + len(w.text) / CHARS_PER_TOKEN for w in ws_)
                                     + judges * JUDGE_PROMPT_TOKENS),
                "completion_tokens": int(n * EXTRACT_COMPLETION_TOKENS + judges * JUDGE_COMPLETION_TOKENS),
                "embed_inputs": math.ceil(claims), "embed_tokens": int(claims * EMBED_TOKENS_PER_CLAIM)}

    if "core" in only:
        ents = [e for t in ("user", "memory") for e in core.get(t, [])
                if not cf.is_episodic(e.text) and not _is_header_only(e.text, cf)]
        batches = math.ceil(len(ents) / CLASSIFY_BATCH) if ents else 0
        chars = sum(len(e.text) for e in ents)
        by_step["core"] = {"entries": len(ents), "windows": 0, "llm_calls": batches,
                           "prompt_tokens": int(batches * cls_tokens + chars / CHARS_PER_TOKEN),
                           "completion_tokens": int(len(ents) * CLASSIFY_COMPLETION_PER_ITEM),
                           "embed_inputs": len(ents), "embed_tokens": int(chars / CHARS_PER_TOKEN)}
    if "memory_md" in only:
        eps = [e for e in core.get("memory", []) if cf.is_episodic(e.text)]
        wc = windows_cost(todo_windows(memory_md_windows(ctx, eps)))
        wc["entries"] = len(eps)
        wc["embed_inputs"] += len(eps)
        wc["embed_tokens"] += int(sum(len(e.text) for e in eps) / CHARS_PER_TOKEN)
        by_step["memory_md"] = wc
    if "dump" in only:
        p = Path(dump_path) if dump_path else default_dump_path(ctx.paths)
        rows = []
        if p.is_file():
            try:
                rows = load_dump(p)
            except (OSError, ValueError):
                rows = []
        by_step["dump"] = {"rows": len(rows), "windows": 0, "llm_calls": 0, "prompt_tokens": 0,
                           "completion_tokens": 0, "embed_inputs": len(rows),
                           "embed_tokens": int(sum(len(r["text"]) for r in rows) / CHARS_PER_TOKEN),
                           "exists": p.is_file()}
    if "md" in only:
        by_step["md"] = windows_cost(todo_windows(_llm_windows_md(ctx)))
    if "statedb" in only:
        by_step["statedb"] = windows_cost(todo_windows(_llm_windows_statedb(ctx, statedb_start)))

    keys = ("windows", "llm_calls", "prompt_tokens", "completion_tokens", "embed_inputs", "embed_tokens")
    tot = {k: sum(int(s.get(k, 0)) for s in by_step.values()) for k in keys}
    usd = (tot["prompt_tokens"] / 1e6 * float(_cfg(cfg, "llm_price_in_per_mtok", 0.40))
           + tot["completion_tokens"] / 1e6 * float(_cfg(cfg, "llm_price_out_per_mtok", 1.60))
           + tot["embed_tokens"] / 1e6 * float(_cfg(cfg, "embed_price_per_mtok", 0.02)))
    cap = max(1, int(_cfg(cfg, "max_windows_per_run", 60)))
    nrem_windows = sum(by_step.get(s, {}).get("windows", 0) for s in ("md", "statedb"))
    runs = max(math.ceil(nrem_windows / cap), math.ceil(by_step.get("memory_md", {}).get("windows", 0) / cap), 1)
    return {**tot, "usd": round(usd, 4), "runs_needed": runs, "by_step": by_step,
            "steps": sorted(only),
            "assumptions": (f"토큰≈글자/{CHARS_PER_TOKEN}, 창당 주장 {EST_CLAIMS_PER_WINDOW}개, "
                            f"주장당 판정 {EST_JUDGE_PER_CLAIM}회, 실행당 창 상한 {cap}개")}


# ── dry-run sandbox (phase A visible to phase B without touching the data dir) ──

class _DrySandbox:
    """Private temp copy of Lance + ledger outside HERMES_HOME (0700, removed on close)."""

    def __init__(self, ctx: Any):
        from .ledger import Ledger
        from .store import Store
        self.dir = Path(tempfile.mkdtemp(prefix="yume-migrate-dry-"))
        try:
            src = Path(ctx.paths.lancedb_dir)
            if src.exists():
                shutil.copytree(src, self.dir / "lancedb")
            dst = sqlite3.connect(str(self.dir / "ledger.db"))
            try:
                ctx.ledger.conn.backup(dst)
            finally:
                dst.close()
            self.store = Store.open(self.dir / "lancedb", dim=int(ctx.cfg.embed_dim),
                                    embed_model=ctx.cfg.embed_model_id(), create=not src.exists())
            self.ledger = Ledger.open(self.dir / "ledger.db")
        except BaseException:
            shutil.rmtree(self.dir, ignore_errors=True)
            raise

    def apply(self, plan: Plan) -> None:
        self.store.commit(upserts=plan.upserts, history=plan.history, suppress=plan.suppress,
                          purge_ids=plan.purge_ids)
        self.ledger.apply_delta(plan.run_id, plan.ledger_delta, lance_version_after=self.store.version(),
                                status="dry")

    def close(self) -> None:
        try:
            self.ledger.close()
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(self.dir, ignore_errors=True)


# ── report merge, artifacts ──────────────────────────────────────────────────

_STAT_SUMS = ("created", "pinned_new", "quarantined", "core_changes", "purged")


def _merge_into(dst_rep: RunReport, dst_st: RunStats, src_rep: RunReport, src_st: RunStats) -> None:
    for f in dataclasses.fields(RunReport):
        a, b = getattr(dst_rep, f.name), getattr(src_rep, f.name)
        if isinstance(a, list):
            setattr(dst_rep, f.name, list(b) + list(a))
        elif isinstance(a, dict):
            merged = {k: list(v) for k, v in b.items()}
            for k, v in a.items():
                merged.setdefault(k, []).extend(v)
            setattr(dst_rep, f.name, merged)
    for name in _STAT_SUMS:
        setattr(dst_st, name, int(getattr(dst_st, name) or 0) + int(getattr(src_st, name) or 0))


def memory_md_proposal(ctx: Any, core: dict[str, list[Any]], rows: dict[str, MemoryRow]) -> tuple[list[str], int, int]:
    """(entries to keep, dropped count, chars freed): episodic MEMORY.md entries whose text is
    preserved in Lance (core_sha) are dropped from the proposal."""
    cf = _corefmt(ctx.paths)
    kept_shas = {r.core_sha for r in rows.values() if r.core_sha and r.status not in CANDIDATE_SEARCH_EXCLUDED}
    keep, dropped, freed = [], 0, 0
    for e in core.get("memory", []):
        if cf.is_episodic(e.text) and e.sha in kept_shas:
            dropped += 1
            freed += len(e.text) + len(cf.ENTRY_DELIMITER)
        else:
            keep.append(e.text)
    return keep, dropped, freed


def copy_v1_logs(ctx: Any) -> int:
    """v1 dry-run logs in <workspace>/hermesyume-state → dream-log/legacy-v1/ (never overwrite)."""
    ws_dir = str(_cfg(ctx.cfg, "workspace_dir", "") or "").strip()
    if not ws_dir:
        return 0
    src = Path(ws_dir) / "hermesyume-state"
    if not src.is_dir():
        return 0
    dst = ctx.paths.ensure_dir(Path(ctx.paths.dream_log_dir) / "legacy-v1")
    n = 0
    for f in sorted(src.rglob("*.md")):
        target = dst / f.name
        if target.exists() or not f.is_file():
            continue
        shutil.copyfile(f, target)
        os.chmod(target, 0o600)
        n += 1
    return n


def _core_file_shas(paths: Any) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for t in ("user", "memory"):
        p = Path(paths.core_file(t))
        out[t] = _sha256_file(p) if p.exists() else None
    return out


class _RowsView:
    """Read-only stand-in for a WorkingSet (rows + by_origin_key) over committed rows."""

    def __init__(self, rows: dict[str, MemoryRow]):
        self.rows = rows
        self._by_key: dict[str, MemoryRow] = {}
        for r in rows.values():
            for k in r.origin_keys or []:
                self._by_key.setdefault(k, r)

    def by_origin_key(self, key: str) -> MemoryRow | None:
        return self._by_key.get(key)

    def get(self, memory_id: str) -> MemoryRow | None:
        return self.rows.get(memory_id)


# ── run ──────────────────────────────────────────────────────────────────────

def _safe_msg(e: BaseException) -> str:
    try:
        from .threat import redact_secrets
        return redact_secrets(str(e))[0][:300]
    except Exception:  # noqa: BLE001
        return type(e).__name__


def _failure(e: BaseException) -> tuple[str, str]:
    from .embedder import EmbedAuthError
    from .ledger import MetaMismatch
    from .llm import LLMAuthError
    from .store import SchemaMismatch
    from .threat import ThreatScannerUnavailable
    msg = _safe_msg(e)
    if isinstance(e, (LLMAuthError, EmbedAuthError)):
        return "auth_401", "OpenAI 인증 실패(401/403)로 마이그레이션을 중단했습니다. .env의 OPENAI_API_KEY를 확인하세요."
    if isinstance(e, (SchemaMismatch, MetaMismatch)):
        return "model_mismatch", f"임베딩 모델/차원 또는 Lance 스키마 불일치로 중단했습니다: {msg}"
    if isinstance(e, ThreatScannerUnavailable):
        return "scanner_unavailable", "위협 스캐너를 불러오지 못해 마이그레이션을 중단했습니다(fail-closed)."
    return "run_failed", f"마이그레이션 실패: {type(e).__name__}: {msg}"


def _refuse(problems: list[str], est: dict | None = None, inv: dict | None = None) -> MigrateResult:
    return MigrateResult(plan=None, core_map_path=None, pins_proposed_path=None, estimate=est,
                         ok=False, problems=problems, status="refused", inventory=inv)


def run_migrate(ctx: Any, *, only: set[str] | Iterable[str] | str | None, estimate_only: bool = False,
                statedb_start: float | None = None, dump_path: str | None = None) -> MigrateResult:
    """M0–M7. `ctx.mode` should be "migrate"; `ctx.dry_run` → plan.json + `_dry` Dream Log only;
    otherwise `ctx.approve_migration` must be set (the CLI's --approve-migration)."""
    from . import nrem, plan as plan_mod, rem
    only = parse_only(only) if not isinstance(only, set) else (only or set(STEPS))
    bad = sorted(set(only) - set(STEPS))
    if bad:
        return _refuse([f"알 수 없는 단계: {', '.join(bad)}"])
    dry = bool(ctx.dry_run)
    if not estimate_only and not dry and not getattr(ctx, "approve_migration", False):
        return _refuse(["실행하려면 --approve-migration이 필요합니다 (먼저 --dry-run으로 검토)."])

    est = estimate(ctx, only, statedb_start=statedb_start, dump_path=dump_path)
    if estimate_only:
        return MigrateResult(plan=None, core_map_path=None, pins_proposed_path=None, estimate=est,
                             ok=True, problems=[], status="estimate")
    budget = getattr(ctx, "budget", None)
    if budget is not None and est["llm_calls"] > int(budget.max_llm_calls):
        return _refuse([f"예상 LLM 호출 {est['llm_calls']}회(약 ${est['usd']})가 상한 {budget.max_llm_calls}회를 넘습니다. "
                        "--estimate로 확인한 뒤 --max-llm-calls를 올려 다시 실행하세요."], est)
    if dump_path and "dump" in only and not Path(dump_path).is_file():
        return _refuse([f"덤프 파일이 없습니다: {dump_path}"], est)

    t0 = time.monotonic()
    st, rep = ctx.stats, ctx.report
    st.run_id, st.mode = ctx.run_id, ctx.mode
    sandbox: _DrySandbox | None = None
    main_plan: Plan | None = None
    seed_plan: Plan | None = None
    core_map: dict | None = None
    pinned: list[str] = []
    problems: list[str] = []
    inv: dict | None = None
    core_map_path = proposal_path = None
    status = "failed"
    error: str | None = None
    try:
        inv = inventory(ctx, dump_path=dump_path)
        pre = nrem.preflight(ctx)                       # N0 (main run row, checks, pings, replay, backup)
        from .sources.core_files import read_core
        core = read_core(ctx.paths)

        # ── phase A: direct inserts (M3 raw, M1, M2, M5) + --statedb-start watermarks ──
        sctx = dataclasses.replace(ctx, run_id=ctx.run_id + SEED_SUFFIX,
                                   stats=RunStats(run_id=ctx.run_id + SEED_SUFFIX, mode=ctx.mode),
                                   report=RunReport())
        ws = plan_mod.WorkingSet(ctx.store.load_working_set(), run_id=sctx.run_id, now=float(ctx.now),
                                 embed_model=_model_id(ctx))
        episodic = [e for e in core.get("memory", []) if _corefmt(ctx.paths).is_episodic(e.text)]
        if "memory_md" in only:
            legacy_memory_md(sctx, ws, episodic)
        if "core" in only:
            core_map = seed_core(sctx, ws, None, core, steps=only)
            pinned = auto_pin(sctx, ws, seeded_ids(core_map))
        if "dump" in only:
            import_dreamer_dump(sctx, ws, dump_path)
        wm, roots = ({}, {})
        if statedb_start is not None:
            wm, roots = statedb_start_watermarks(sctx, statedb_start)
            sctx.note(f"--statedb-start {clock.fmt_kst(statedb_start)}: state.db 계보 {len(wm)}개의 "
                      "워터마크를 그 시각까지 올렸습니다 (그 이전 대화는 추출하지 않음).")
        core_map = build_core_map(ws, core, steps=only)
        plan_mod.apply_guard(ws, ctx.cfg, mode=ctx.mode)
        seed_plan = plan_mod.build_plan(
            sctx, ws, lance_version_before=int(ctx.store.version()),
            ledger_delta=LedgerDelta(watermarks=wm, session_roots=roots, audit=list(getattr(ws, "audit", []) or [])),
            inbox_consume_ids=[], inbox_skip_ids=[], docs=[])
        if not seed_plan.is_noop():
            if dry:
                plan_mod.write_plan(ctx.paths, seed_plan)
                sandbox = _DrySandbox(ctx)
                sandbox.apply(seed_plan)
            else:
                plan_mod.commit_plan(sctx, seed_plan)
            sctx.note(f"1단계(핵심 seed·레거시·덤프) 실행 id `{sctx.run_id}`: 행 {len(seed_plan.upserts)}개 "
                      + ("(리허설: 임시 사본에만 적용)" if dry else "커밋"))
        _merge_into(rep, st, sctx.report, sctx.stats)

        # ── phase B: NREM (md / state.db / M3 windows) → REM → commit ──
        over: dict[str, Any] = {}
        if "md" not in only:
            over["md_sources"] = []
        if "statedb" not in only:
            over["include_sources"] = []
        cfg_b = ctx.cfg.replace(**over) if over and hasattr(ctx.cfg, "replace") else ctx.cfg
        ctx_b = dataclasses.replace(ctx, cfg=cfg_b,
                                    store=sandbox.store if sandbox else ctx.store,
                                    ledger=sandbox.ledger if sandbox else ctx.ledger)
        skipped_steps = [s for s in ("md", "statedb") if s not in only]
        if skipped_steps:
            ctx.note(f"이번 마이그레이션에서 제외한 단계: {', '.join(skipped_steps)} (--only)")
        nres = nrem.run_nrem(ctx_b, pre)
        if "memory_md" in only:
            extract_memory_md(ctx_b, nres, episodic)
        rres = rem.run_rem(ctx_b, nres, pre)
        main_plan = rres.plan
        status = "committed" if rres.status == "noop" else rres.status

        # ── M7 wrap-up ──
        try:
            from .calibrate import calibrate
            cal = calibrate(ctx_b)
            ctx.note(f"calibrate: 표본 shadow {cal.n_shadow}·주입 {cal.n_injected}, "
                     f"추천 recall_min_cos {cal.recall_min_cos:.2f}")
        except Exception as e:  # noqa: BLE001 — informational
            ctx.note(f"calibrate 생략 ({type(e).__name__})")
        # committed state (dry-run: the temp copy holding phase A, or the untouched store)
        final_rows = ctx_b.store.load_working_set(with_vectors=False)
        core_map = build_core_map(_RowsView(final_rows), core, steps=only)
        cm_note = (f"core_map {core_map['accounted']}/{core_map['total']} "
                   f"(행 {core_map['rows']}, 머리글 조각 {core_map['fragments']}, 비밀값 제외 {core_map['secret_skipped']}, "
                   f"대기 {core_map['pending']}) — {'통과' if core_map['ok'] else '불일치'}")
        ctx.note(cm_note)
        if not core_map["ok"]:
            problems.append(cm_note)
        keep, dropped, freed = memory_md_proposal(ctx, core, final_rows)
        if dropped:
            if dry:
                ctx.note(f"MEMORY.md 정리 제안(실행 시 생성): 레거시 {dropped}개 제거, 약 {freed}자 확보. "
                         "적용은 사람이 `yume core-proposal apply`로만.")
            else:
                cf = _corefmt(ctx.paths)
                pdir = ctx.paths.ensure_dir(Path(ctx.paths.proposals_dir))
                pp = pdir / PROPOSAL_FILENAME
                fd, tmp = tempfile.mkstemp(prefix=".prop_", dir=str(pdir))
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(cf.ENTRY_DELIMITER.join(keep))
                    f.flush()
                    os.fsync(f.fileno())
                os.chmod(tmp, 0o600)
                os.replace(tmp, pp)
                proposal_path = str(pp)
                ctx.note(f"MEMORY.md 정리 제안: proposals/{PROPOSAL_FILENAME} (레거시 {dropped}개 제거, 약 {freed}자 확보). "
                         "적용은 사람이 `yume core-proposal apply`로만.")
        if not dry:
            ctx.paths.ensure_dir(Path(ctx.paths.migration_dir))
            core_map_path = str(_write_json(Path(ctx.paths.migration_dir) / "core_map.json",
                                            {**core_map, "run_id": ctx.run_id}))
            _write_json(Path(ctx.paths.migration_dir) / "inventory.json", inv)
            n_v1 = copy_v1_logs(ctx)
            if n_v1:
                ctx.note(f"v1 로그 {n_v1}개를 dream-log/legacy-v1/로 복사했습니다.")
        before = {t: (inv.get("core", {}).get(t) or {}).get("sha256") for t in ("user", "memory")}
        if _core_file_shas(ctx.paths) != before:
            ctx.note("마이그레이션 중에 MEMORY.md/USER.md가 바뀌었습니다(에이전트 또는 사람). 다음 밤 R5가 반영합니다.")
        st.duration_s = round(time.monotonic() - t0, 3)
        rem.post_commit(ctx_b, main_plan)
    except Exception as e:  # noqa: BLE001 — becomes an alert + failed run row + Dream Log
        code, message = _failure(e)
        error = f"{type(e).__name__}: {_safe_msg(e)}"
        log.error("migrate failed: %s", type(e).__name__)
        if not any(a.code == code for a in ctx.alerts):
            ctx.alert(code, message, level="error", error_class=type(e).__name__)
        status = "failed"
        problems.append(message)
        st.status = "failed"
        st.duration_s = round(time.monotonic() - t0, 3)
        if not dry and ctx.ledger is not None and not getattr(ctx.ledger, "readonly", False):
            try:
                rec = ctx.ledger.get_run(ctx.run_id)
                if rec is not None and rec.status in ("planned", "committed", "held"):
                    # planned: R8 crashed mid-way → replayed by the next N0; committed/held: only a
                    # post-commit step failed. Keep the status, record the error (REM R-9).
                    ctx.ledger.update_run(ctx.run_id, error=error[:500])
                else:
                    ctx.ledger.update_run(ctx.run_id, status="failed", error=error[:500],
                                          finished_at=clock.real_now())
            except Exception:  # noqa: BLE001
                pass
        try:
            from . import dream_log
            dream_log.write(ctx.paths, ctx.now, dream_log.render(ctx, main_plan, status="failed", error=error),
                            dry=dry)
        except Exception as e2:  # noqa: BLE001
            log.error("dream log failed: %s", type(e2).__name__)
        if not dry:
            try:
                from . import alerts as _alerts
                _alerts.AlertSink(ctx.paths, ctx.cfg).emit_all(ctx.alerts)
                _alerts.flush_pending(ctx.paths, ctx.cfg, now=ctx.now)
            except Exception as e3:  # noqa: BLE001
                log.error("alert emit failed: %s", type(e3).__name__)
    finally:
        if sandbox is not None:
            sandbox.close()
    st.status = status
    ok = status in ("committed", "held", "dry") and not problems
    return MigrateResult(plan=main_plan, core_map_path=core_map_path, pins_proposed_path=None,
                         estimate=est, ok=ok, problems=problems, status=status, seed_plan=seed_plan,
                         core_map=core_map, inventory=inv, pinned_ids=pinned, proposal_path=proposal_path)


# ── CLI (ready-made handler; the integrator may wire it into cli.HANDLERS) ──

def build_context(paths: Any, cfg: Any, *, run_id: str, now: float, dry_run: bool, approve: bool,
                  max_llm_calls: int | None = None, estimate_only: bool = False,
                  offline: bool = False) -> Any:
    """RunContext for migrate (CONTRACTS §3 pipeline; mode "migrate")."""
    from .embedder import make_embedder
    from .ledger import Ledger
    from .livedb import LiveDB
    from .llm import make_llm
    from .store import Store, StoreMissing
    from .types import RunBudget, RunContext
    budget = RunBudget(max_llm_calls=int(max_llm_calls or cfg.max_llm_calls),
                       max_embed_inputs=int(cfg.max_embed_inputs),
                       max_runtime_s=float(cfg.max_runtime_min) * 60)
    ledger = Ledger.from_paths(paths, readonly=dry_run or estimate_only)
    live = LiveDB.open(paths, mode="pure" if (dry_run or estimate_only) else "rw")
    store = None
    try:
        store = Store.from_config(paths, cfg)
    except StoreMissing:
        if not estimate_only:
            ledger.close()
            if live is not None:
                live.close()
            raise
    ctx = RunContext(paths=paths, cfg=cfg, run_id=run_id, now=now, mode="migrate", dry_run=dry_run,
                     offline=offline, ledger=ledger, live=live, store=store, budget=budget,
                     stats=RunStats(run_id=run_id, mode="migrate"), approve_migration=approve,
                     log=log)
    if not estimate_only:
        from .threat import load_scanner
        ctx.llm = make_llm(cfg, paths, budget=budget, offline=offline)
        ctx.embedder = make_embedder(cfg, paths, budget=budget, offline=offline)
        ctx.scanner = load_scanner(cfg.hermes_runtime_dir)
    return ctx


def _close(ctx: Any) -> None:
    for h in (getattr(ctx, "ledger", None), getattr(ctx, "live", None)):
        try:
            if h is not None:
                h.close()
        except Exception:  # noqa: BLE001
            pass


def format_result(res: MigrateResult) -> str:
    lines = [f"마이그레이션: {res.status} ({'정상' if res.ok else '문제 있음'})"]
    if res.estimate:
        e = res.estimate
        lines.append(f"예상: 창 {e['windows']}개 · LLM {e['llm_calls']}회 · 임베딩 {e['embed_inputs']}개 · "
                     f"약 ${e['usd']} · 실행 {e['runs_needed']}회")
        for step, s in sorted(e.get("by_step", {}).items()):
            lines.append(f"  {step}: " + ", ".join(f"{k}={v}" for k, v in s.items()))
    if res.core_map:
        cm = res.core_map
        lines.append(f"core_map {cm['accounted']}/{cm['total']} (머리글 조각 {cm['fragments']}) "
                     f"{'통과' if cm['ok'] else '불일치'}")
    if res.pinned_ids:
        lines.append(f"자동 pin {len(res.pinned_ids)}개")
    if res.seed_plan is not None and not res.seed_plan.is_noop():
        lines.append(f"1단계: {len(res.seed_plan.upserts)}행 (run {res.seed_plan.run_id})")
    if res.plan is not None:
        lines.append(f"2단계: {len(res.plan.upserts)}행 변경 (run {res.plan.run_id})")
    if res.proposal_path:
        lines.append(f"MEMORY.md 정리 제안: {res.proposal_path}")
    for p in res.problems:
        lines.append(f"문제: {p}")
    return "\n".join(lines)


def cli_handler(args: Any, paths: Any) -> int:
    """`yume migrate [--dry-run|--approve-migration|--estimate] [--only …] [--max-llm-calls N]
    [--statedb-start now|ISO]`. Exit 0 ok, 1 failure/refusal, 3 held."""
    from .config import load_config
    from .paths import AlreadyRunning, dream_lock
    from .store import StoreMissing
    from .types import make_run_id
    dry, approve, est_only = bool(args.dry_run), bool(args.approve_migration), bool(args.estimate)
    if dry and approve:
        print("--dry-run과 --approve-migration은 함께 쓸 수 없습니다.", file=sys.stderr)
        return 1
    if not (dry or approve or est_only):
        print("--dry-run(검토), --estimate(비용) 또는 --approve-migration(실행) 중 하나가 필요합니다.",
              file=sys.stderr)
        return 1
    try:
        only = parse_only(getattr(args, "only", None))
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    cfg = load_config(paths)
    now = clock.now()
    start = parse_statedb_start(getattr(args, "statedb_start", None), now)
    run_id = make_run_id(now)

    def go() -> MigrateResult:
        extra = {"offline": True} if getattr(args, "offline", False) else {}   # `yume migrate --offline`
        ctx = build_context(paths, cfg, run_id=run_id, now=now, dry_run=dry or est_only,
                            approve=approve, max_llm_calls=getattr(args, "max_llm_calls", None),
                            estimate_only=est_only, **extra)
        try:
            return run_migrate(ctx, only=only, estimate_only=est_only, statedb_start=start)
        finally:
            _close(ctx)

    try:
        if dry or est_only:
            res = go()
        else:
            with dream_lock(paths):
                res = go()
    except AlreadyRunning:
        print("already running")
        return 0
    except StoreMissing:
        print("Lance 저장소가 없습니다. 먼저 `yume init`을 실행하세요.", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        out = {"run_id": run_id, "status": res.status, "ok": res.ok, "problems": res.problems,
               "estimate": res.estimate, "core_map_path": res.core_map_path,
               "proposal_path": res.proposal_path, "pinned_ids": res.pinned_ids,
               "core_map": {k: v for k, v in (res.core_map or {}).items() if k != "entries"} or None}
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_result(res))
    if res.status == "held":
        return 3
    return 0 if res.ok else 1


__all__ = ["STEPS", "MigrateResult", "inventory", "seed_core", "auto_pin", "legacy_memory_md",
           "import_dreamer_dump", "estimate", "run_migrate", "build_core_map", "parse_only",
           "parse_statedb_start", "memory_md_windows", "extract_memory_md", "statedb_start_watermarks",
           "cli_handler", "build_context", "format_result"]
