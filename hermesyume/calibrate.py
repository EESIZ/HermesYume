"""`yume calibrate` (PLAN-v2 §4.1 보정, CONTRACTS §4.23): recommend recall thresholds from the
shadow/injected cosine distribution. Read-only — the CLI prints, `yume config set` applies.

    recall_min_cos      = max(0.40, p99(unrelated) + 0.05)        never below 0.40
    candidate_cos_floor = p10(related labeled pairs)              None without labeled pairs

"unrelated": labeled pairs (`<data_dir>/calibration/pairs.jsonl`, {"a","b","label"}) when any
are labeled unrelated; otherwise shadow/injected events never followed by `used` for the same
memory in the same session. "related": labeled related pairs, else events followed by `used`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import RECALL_MIN_COS_FLOOR
from .dream_log import quantile

DAY = 86400.0
MARGIN = 0.05
QUANTILES = (("p10", 0.10), ("p50", 0.50), ("p90", 0.90), ("p99", 0.99))
LABELS_RELPATH = "calibration/pairs.jsonl"


@dataclass
class CalibrationResult:
    n_shadow: int
    n_labeled: int
    quantiles: dict[str, dict[str, float]]   # {"shadow"|"injected"|"unrelated"|"related": {"p10","p50","p90","p99"}}
    recall_min_cos: float                    # max(0.40, p99(unrelated) + 0.05)
    candidate_cos_floor: float | None
    n_injected: int = 0
    n_unrelated: int = 0
    n_related: int = 0
    unrelated_source: str = "none"           # "labels" | "events" | "none"
    current_recall_min_cos: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def _qdict(values: list[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, q in QUANTILES:
        v = quantile(values, q)
        if v is not None:
            out[name] = round(v, 4)
    return out


def _ceil2(x: float) -> float:
    return math.ceil(round(x * 100, 6)) / 100.0


def _cfg(cfg: Any, key: str, default: Any) -> Any:
    try:
        return cfg[key]
    except (KeyError, TypeError):
        return getattr(cfg, key, default)


def labels_path_for(paths: Any) -> Path:
    return Path(paths.data_dir) / LABELS_RELPATH


def load_labeled_pairs(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.is_file():
        return []
    out: list[dict[str, str]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        a, b, lab = d.get("a"), d.get("b"), str(d.get("label") or "").strip().lower()
        if isinstance(a, str) and isinstance(b, str) and a.strip() and b.strip() \
                and lab in ("related", "unrelated"):
            out.append({"a": a, "b": b, "label": lab})
    return out


def _pair_cosines(embedder: Any, pairs: list[dict[str, str]]) -> list[tuple[str, float]]:
    texts: list[str] = []
    for p in pairs:
        texts += [p["a"], p["b"]]
    vecs = embedder.embed(texts, kind="calibrate")
    out = []
    for i, p in enumerate(pairs):
        va, vb = vecs[2 * i], vecs[2 * i + 1]
        dot = sum(float(x) * float(y) for x, y in zip(va, vb))
        na = math.sqrt(sum(float(x) * float(x) for x in va)) or 1.0
        nb = math.sqrt(sum(float(x) * float(x) for x in vb)) or 1.0
        out.append((p["label"], dot / (na * nb)))
    return out


def _events(ctx: Any, days: int) -> list[Any]:
    live = getattr(ctx, "live", None)
    if live is None:
        return []
    snap = live.snapshot()
    since = float(ctx.now) - float(days) * DAY
    return [e for e in live.recall_range(0, snap.max_recall_id) if float(e.ts) >= since]


def calibrate(ctx: Any, *, days: int = 7, labels_path: str | None = None) -> CalibrationResult:
    """Shadow/injected cosine distribution (+ optional labeled pairs) → recommended thresholds.
    Writes nothing."""
    cfg = ctx.cfg
    current = float(_cfg(cfg, "recall_min_cos", RECALL_MIN_COS_FLOOR))
    events = _events(ctx, days)
    shadow = [float(e.cos) for e in events if e.kind == "shadow" and e.cos is not None]
    injected = [float(e.cos) for e in events if e.kind == "injected" and e.cos is not None]

    # used follow-ups per (session, memory)
    used_at: dict[tuple[str | None, str], list[float]] = {}
    for e in events:
        if e.kind == "used":
            used_at.setdefault((e.session_id, e.memory_id), []).append(float(e.ts))
    ev_unrel: list[float] = []
    ev_rel: list[float] = []
    for e in events:
        if e.kind not in ("shadow", "injected") or e.cos is None:
            continue
        later = [ts for ts in used_at.get((e.session_id, e.memory_id), []) if ts >= float(e.ts)]
        (ev_rel if later else ev_unrel).append(float(e.cos))

    notes: list[str] = []
    lp = Path(labels_path) if labels_path else labels_path_for(ctx.paths)
    pairs = load_labeled_pairs(lp)
    lab_rel: list[float] = []
    lab_unrel: list[float] = []
    if pairs:
        if getattr(ctx, "embedder", None) is None:
            notes.append(f"라벨 쌍 {len(pairs)}개가 있지만 임베더가 없어 건너뜀")
        else:
            for label, c in _pair_cosines(ctx.embedder, pairs):
                (lab_rel if label == "related" else lab_unrel).append(c)

    if lab_unrel:
        unrelated, source = lab_unrel, "labels"
    elif ev_unrel:
        unrelated, source = ev_unrel, "events"
    else:
        unrelated, source = [], "none"
    related = lab_rel or ev_rel

    if unrelated:
        p99 = quantile(unrelated, 0.99) or 0.0
        rec = max(RECALL_MIN_COS_FLOOR, _ceil2(p99 + MARGIN))
    else:
        rec = max(RECALL_MIN_COS_FLOOR, current)
        notes.append("무관 표본이 없어 현재 값을 유지합니다")
    rec = round(min(1.0, rec), 2)
    floor = round(quantile(lab_rel, 0.10), 4) if lab_rel else None
    if source == "events" and not ev_rel:
        notes.append("'used'로 이어진 회상이 없어 무관 표본이 과대평가될 수 있습니다(shadow 모드에서는 정상)")

    q: dict[str, dict[str, float]] = {}
    for name, vals in (("shadow", shadow), ("injected", injected), ("unrelated", unrelated),
                       ("related", related)):
        if vals:
            q[name] = _qdict(vals)
    return CalibrationResult(n_shadow=len(shadow), n_labeled=len(lab_rel) + len(lab_unrel), quantiles=q,
                             recall_min_cos=rec, candidate_cos_floor=floor, n_injected=len(injected),
                             n_unrelated=len(unrelated), n_related=len(related),
                             unrelated_source=source, current_recall_min_cos=current, notes=notes)


def format_result(res: CalibrationResult) -> str:
    lines = [f"보정 표본: shadow {res.n_shadow} · 주입 {res.n_injected} · 라벨 쌍 {res.n_labeled}",
             f"무관 표본 {res.n_unrelated}개 (출처: {res.unrelated_source}) · 관련 표본 {res.n_related}개"]
    for name, qd in res.quantiles.items():
        lines.append(f"  {name}: " + ", ".join(f"{k}={v:.3f}" for k, v in qd.items()))
    lines.append(f"추천 recall_min_cos = {res.recall_min_cos:.2f} (현재 {res.current_recall_min_cos})")
    if res.candidate_cos_floor is not None:
        lines.append(f"candidate_cos 하한 참고값 = {res.candidate_cos_floor:.3f} (라벨 관련 쌍 p10)")
    for n in res.notes:
        lines.append(f"참고: {n}")
    lines.append(f"적용: yume config set recall_min_cos {res.recall_min_cos:.2f}")
    return "\n".join(lines)


def cli_handler(args: Any, paths: Any) -> int:
    """Ready-made handler for `yume calibrate` (read-only; integrator may wire it)."""
    from . import clock
    from .config import load_config
    from .livedb import LiveDB
    from .types import RunContext
    cfg = load_config(paths)
    live = LiveDB.open(paths, mode="pure")
    embedder = None
    if labels_path_for(paths).is_file():
        from .embedder import make_embedder
        embedder = make_embedder(cfg, paths)
    ctx = RunContext(paths=paths, cfg=cfg, run_id="calibrate", now=clock.now(), mode="dry",
                     dry_run=True, live=live, embedder=embedder)
    try:
        res = calibrate(ctx, days=int(getattr(args, "days", 7) or 7))
    finally:
        if live is not None:
            live.close()
    if getattr(args, "json", False):
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_result(res))
    return 0


__all__ = ["CalibrationResult", "calibrate", "format_result", "load_labeled_pairs", "cli_handler"]
