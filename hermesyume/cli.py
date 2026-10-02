"""`yume` CLI (PLAN-v2 §11.1 #38; CONTRACTS §3, §4.24). Every handler has the signature
``handler(args: argparse.Namespace, paths: Paths) -> int`` and returns the process exit code.

Write rules: Lance/ledger writers (dream, unpin, forget, restore, export, reembed, debug plant,
core-restore, core-proposal apply, init) hold ``dream.lock``; ``dream --dry-run`` never takes it
and writes only ``dream-log/*_dry.md`` + ``runs/<id>/plan.json`` (§10.5). MEMORY.md/USER.md are
written only by ``core-restore`` / ``core-proposal apply`` (core_check, T14).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import unicodedata
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from . import __version__, clock
from .paths import AlreadyRunning, LiveHomeRefused, Paths, dream_lock, is_live_home, refuse_if_live

Handler = Callable[[argparse.Namespace, Paths], int]

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_NOT_IMPLEMENTED = 2
EXIT_HELD = 3          # dream finished but destructive ops were held (R7)

log = logging.getLogger("hermesyume.cli")


# ── parser ───────────────────────────────────────────────────────────────────

def _common(s: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Global flags also accepted after the sub-command (`yume dream … --json`). SUPPRESS keeps
    a value given before the sub-command."""
    s.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="JSON 출력")
    s.add_argument("--hermes-home", default=argparse.SUPPRESS, help="HERMES_HOME")
    s.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)
    return s


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="yume", description="HermesYume v2 관리 CLI")
    p.add_argument("--version", action="version", version=f"hermesyume {__version__}")
    p.add_argument("--hermes-home", help="HERMES_HOME (기본: $HERMES_HOME → ~/.hermes)")
    p.add_argument("--json", action="store_true", help="JSON 출력")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, **kw: Any) -> argparse.ArgumentParser:
        return _common(sub.add_parser(name, **kw))

    s = add("doctor", help="환경·키·스키마·스캐너 점검 (1토큰 호출 포함)")
    s.add_argument("--offline", action="store_true", help="네트워크 호출(임베딩·LLM 1토큰) 생략")
    s = add("init", help="데이터 디렉터리·config.json·ledger·Lance·live.db 생성")
    s.add_argument("--force", action="store_true", help="ledger meta(모델·차원)를 설정값으로 다시 기록")

    s = add("dream", help="야간 정리 실행")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--offline", action="store_true", help="가짜 LLM·임베더 (네트워크 없음)")
    s.add_argument("--now", help="ISO | +Nd | +Nh | +Ny")
    s.add_argument("--settle-minutes", type=int)
    s.add_argument("--approve-run", metavar="RUN_ID")
    s.add_argument("--max-llm-calls", type=int)

    s = add("status", help="최근 실행·백로그·회상 health")
    s.add_argument("--recall", action="store_true")

    s = add("search", help="장기기억 검색")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=5)
    s.add_argument("--include-inactive", action="store_true")
    s.add_argument("--offline", action="store_true", help="가짜 임베더 (네트워크 없음)")

    s = add("inspect", help="행 상세 (G1 검사용)")
    s.add_argument("--id")
    s.add_argument("--query", help="text/subject 부분 일치")
    s.add_argument("--label", help="핵심 파일 라벨(**…**) 또는 subject 부분 일치")
    s.add_argument("--tag", help="debug plant 태그 (source=debug:<tag>)")
    s.add_argument("--status", help="상태 필터 (쉼표)")
    s.add_argument("--now")

    s = add("pin", help="pin 목록 (승인 절차 없음, U2)")
    s.add_argument("action", choices=["list"])

    s = add("unpin")
    s.add_argument("memory_id")

    s = add("forget")
    s.add_argument("memory_id")
    s.add_argument("--confirm", action="store_true")
    s.add_argument("--reason", default="")

    s = add("restore", help="실행 단위 되돌리기 (table.restore)")
    s.add_argument("--run", required=True)
    s.add_argument("--reprocess", action="store_true",
                   help="되돌린 실행들의 입력(창·md·inbox)을 다음 dream에서 다시 처리")
    s.add_argument("--unforget", action="store_true",
                   help="되돌린 실행들의 forget도 취소 (기본: forget은 다시 적용)")

    add("core-check", help="R5 핵심 파일 점검 (읽기 전용)")
    s = add("core-restore", help="핵심 파일 항목 복원 (사람만 실행)")
    s.add_argument("memory_id")
    s.add_argument("--target", choices=["user", "memory"])
    s = add("core-proposal", help="MEMORY.md 정리 제안 적용")
    s.add_argument("action", choices=["apply"])
    s.add_argument("path", nargs="?")

    add("alert-flush", help="쌓인 운영 알림을 .txt 문서로 텔레그램 전송 (U4)")
    s = add("calibrate", help="shadow 분포로 임계값 추천")
    s.add_argument("--days", type=int, default=7)
    add("export", help="서빙 사본 재생성")
    s = add("reembed", help="임베딩 모델 변경")
    s.add_argument("--model")
    s.add_argument("--dim", type=int)
    s.add_argument("--offline", action="store_true", help="가짜 임베더 (샌드박스 시험용)")

    s = add("migrate", help="M0~M7 마이그레이션")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--approve-migration", action="store_true")
    s.add_argument("--only", help="core,dump,memory_md,md,statedb (쉼표)")
    s.add_argument("--estimate", action="store_true")
    s.add_argument("--max-llm-calls", type=int)
    s.add_argument("--statedb-start", help="now | ISO")
    s.add_argument("--offline", action="store_true", help="가짜 LLM·임베더 (라이브 홈에서는 --dry-run만)")

    s = add("config", help="config.json")
    s.add_argument("action", choices=["set", "get", "show"])
    s.add_argument("key", nargs="?")
    s.add_argument("value", nargs="?")
    s.add_argument("--force", action="store_true")

    s = add("debug", help="샌드박스 전용 (라이브 HERMES_HOME 거부)")
    s.add_argument("action", choices=["plant", "event"])
    s.add_argument("--kind")
    s.add_argument("--importance", type=float)
    s.add_argument("--text")
    s.add_argument("--count", type=int, default=1)
    s.add_argument("--tag", default="e2e")
    s.add_argument("--target", help="event: memory id")
    s.add_argument("--at", help="event: ISO | +Nd")
    s.add_argument("--offline", action="store_true", help="plant: 가짜 임베더")
    s.add_argument("--now", help="plant: 생성 시각 (ISO | +Nd)")
    for name in ("status", "pin", "search", "export", "core-check"):
        sub.choices[name].add_argument("--now", help="기준 시각 (ISO | +Nd)")
    return p


# ── shared helpers ───────────────────────────────────────────────────────────

def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _json_out(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "json", False))


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def _cfg(paths: Paths):
    from .config import load_config
    return load_config(paths)


def _not_initialized(paths: Paths) -> str | None:
    missing = [n for n, p in (("config.json", paths.config_json), ("ledger.db", paths.ledger_db),
                              ("lancedb", paths.lancedb_dir)) if not p.exists()]
    if missing:
        return f"초기화되지 않았습니다({', '.join(missing)} 없음). 먼저 `yume init`을 실행하세요."
    return None


def _open_ctx(paths: Paths, cfg: Any, *, mode: str = "live", dry_run: bool = False,
              offline: bool = False, need_llm: bool = False, need_embedder: bool = True,
              max_llm_calls: int | None = None, run_id: str | None = None,
              live_mode: str | None = None, ledger_readonly: bool | None = None) -> Any:
    """RunContext with store/ledger/live opened (CONTRACTS §3). Raises StoreMissing."""
    from .embedder import make_embedder
    from .ledger import Ledger
    from .livedb import LiveDB
    from .store import Store
    from .types import RunBudget, RunContext, RunReport, RunStats, make_run_id
    now = clock.now()
    run_id = run_id or make_run_id(now)
    budget = RunBudget(max_llm_calls=int(max_llm_calls or cfg.max_llm_calls),
                       max_embed_inputs=int(cfg.max_embed_inputs),
                       max_runtime_s=float(cfg.max_runtime_min) * 60)
    store = Store.from_config(paths, cfg)
    ledger = Ledger.from_paths(paths, readonly=dry_run if ledger_readonly is None else ledger_readonly)
    live = None
    try:
        live = LiveDB.open(paths, mode=live_mode or ("pure" if dry_run else "rw"))
        llm = None
        if need_llm:
            if offline:
                from .offline import HeuristicLLM
                llm = HeuristicLLM(budget=budget)
            else:
                from .llm import make_llm
                llm = make_llm(cfg, paths, budget=budget)
        emb = make_embedder(cfg, paths, budget=budget, offline=offline) if need_embedder else None
    except BaseException:
        ledger.close()
        if live is not None:
            live.close()
        raise
    return RunContext(paths=paths, cfg=cfg, run_id=run_id, now=now, mode=mode, dry_run=dry_run,
                      offline=offline, llm=llm, embedder=emb, scanner=None, store=store,
                      ledger=ledger, live=live, budget=budget,
                      stats=RunStats(run_id=run_id, mode=mode), report=RunReport(), log=log)


def _close_ctx(ctx: Any) -> None:
    for h in (getattr(ctx, "ledger", None), getattr(ctx, "live", None)):
        try:
            if h is not None:
                h.close()
        except Exception:  # noqa: BLE001
            pass


@contextmanager
def _locked(paths: Paths) -> Iterator[None]:
    with dream_lock(paths):
        yield


def _guards(ctx: Any) -> None:
    """Schema/model guards before any admin write (§2.1)."""
    ctx.store.check_schema()
    ctx.ledger.check_meta(ctx.cfg.embed_model_id(), int(ctx.cfg.embed_dim))
    ctx.store.check_embed_model()


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").casefold()


def _row_view(row: Any, now: float, cfg: Any) -> dict:
    from . import strength
    ev = strength.evaluate(row, now, cfg)
    d = row.snapshot()
    d["strength"] = round(float(ev.strength), 6)
    d["tier"] = ev.tier
    return d


def _short(text: str, n: int = 120) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _fmt_row(row: Any, now: float, cfg: Any) -> str:
    from . import strength
    from .types import KIND_LABEL_KO
    ev = strength.evaluate(row, now, cfg)
    flags = []
    if row.pinned:
        flags.append("pin")
    if row.in_core:
        flags.append("core")
    if row.judge_pending:
        flags.append("판정대기")
    f = f" [{','.join(flags)}]" if flags else ""
    when = clock.kst_date(row.event_time) if row.event_time else "-"
    return (f"{row.id}  {row.status}/{ev.tier}  ({KIND_LABEL_KO.get(row.kind, row.kind)}) "
            f"강도 {ev.strength:.3f}  {when}{f}\n    {_short(row.text, 200)}")


def _latest_dream_log(paths: Paths) -> str | None:
    d = paths.dream_log_dir
    if not d.is_dir():
        return None
    files = sorted(d.glob("*.md"), key=lambda p: p.stat().st_mtime)
    return str(files[-1]) if files else None


class PlannedRunPending(RuntimeError):
    """A crashed run is still `planned`: replaying it later would overwrite an admin change."""


def _refuse_if_planned(ctx: Any) -> None:
    """Admin writes and restore refuse while a run waits for replay (its plan.json holds absolute
    row states that the next dream's N0 would write back over this change, F-30)."""
    pend = [r.run_id for r in ctx.ledger.planned_runs()]
    if pend:
        raise PlannedRunPending(
            f"재생을 기다리는 실행이 있습니다({', '.join(pend)}). 먼저 `yume dream`을 실행해 마무리하세요.")


def _commit_ws(ctx: Any, ws: Any) -> Any:
    from . import plan as _plan
    from .types import LedgerDelta
    _plan.apply_guard(ws, ctx.cfg, mode="live")
    p = _plan.build_plan(ctx, ws, lance_version_before=ctx.store.version(),
                         ledger_delta=LedgerDelta(audit=list(ws.audit)), inbox_consume_ids=[],
                         inbox_skip_ids=[], docs=[])
    return _plan.commit_plan(ctx, p)


def _admin_commit(paths: Paths, cfg: Any, tag: str, mutate: Callable[[Any, Any], Any]) -> tuple[Any, Any]:
    """Run ``mutate(ctx, ws)`` on the working set and commit it as a small plan (R8 1–7) under
    dream.lock, then re-export the serving copy. Returns (mutate result, CommitResult|None)."""
    from . import export, plan as _plan
    from .types import make_run_id
    with _locked(paths):
        now = clock.now()
        ctx = _open_ctx(paths, cfg, mode="live", offline=True, run_id=f"{make_run_id(now)}-{tag}")
        try:
            _guards(ctx)
            _refuse_if_planned(ctx)
            ws = _plan.WorkingSet(ctx.store.load_working_set(), run_id=ctx.run_id, now=ctx.now,
                                  embed_model=cfg.embed_model_id())
            out = mutate(ctx, ws)
            if not ws.ops:
                return out, None
            res = _commit_ws(ctx, ws)
            export.build_serving(ctx)
            return out, res
        finally:
            _close_ctx(ctx)


# ── doctor / init ────────────────────────────────────────────────────────────

def cmd_doctor(args: argparse.Namespace, paths: Paths) -> int:
    from . import threat
    from .config import validate
    from .secrets_env import (OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, alert_chat_id, get_secret, mask,
                              secret_source)
    checks: list[tuple[str, str, str]] = []      # (level OK|WARN|FAIL, name, detail)

    def add(level: str, name: str, detail: str = "") -> None:
        checks.append((level, name, detail))

    home = paths.hermes_home
    add("OK", "HERMES_HOME", f"{home}{' (라이브 홈)' if is_live_home(home) else ''}")
    cfg = _cfg(paths)
    if not cfg.file_exists:
        add("WARN", "config.json", "없음 — `yume init`")
    else:
        problems = validate(cfg.as_dict())
        add("FAIL" if problems else "OK", "config.json", "; ".join(problems) or str(paths.config_json))
        unk = cfg.unknown_keys()
        if unk:
            add("WARN", "config 알 수 없는 키", ", ".join(sorted(unk)))
    add("OK" if paths.state_db.exists() else "WARN", "state.db",
        "있음 (읽기 전용으로만 연다)" if paths.state_db.exists() else "없음")
    from .paths import workspace_write_refusal
    why = workspace_write_refusal(home, cfg.workspace_dir)
    add("OK" if why is None else "WARN", "workspace_dir",
        str(cfg.workspace_dir) if why is None else f"{why} — 절차 문서(docs/yume)를 쓰지 않음")
    if not cfg.md_sources:
        add("WARN", "md_sources", "비어 있음 — md 입력 없음 (`yume config set md_sources`)")
    for t in ("user", "memory"):
        p = paths.core_file(t)
        if p.exists():
            try:
                from .sources.core_files import load_limits, read_core
                n = len(read_core(paths).get(t, []))
                lim = load_limits(paths).get(t, {}).get("limit")
                size = len(p.read_text(encoding="utf-8-sig"))
                add("OK", p.name, f"{n}항목 {size}/{lim}자")
            except Exception as e:  # noqa: BLE001
                add("WARN", p.name, f"읽기 실패: {type(e).__name__}")
        else:
            add("WARN", p.name, "없음")
    # providers: resolved choice + key *names* and where they come from (never values)
    from .llm import LLMError, resolve_llm
    setting = getattr(cfg, "embed_provider_setting", cfg.embed_provider)
    via = f" (embed_provider={setting})" if setting != cfg.embed_provider else ""
    if cfg.embed_provider == "hash":
        add("OK", "임베딩", f"{cfg.embed_model_id()}{via} — 로컬 해싱, 키 불필요 (어휘 일치만 봄)")
    else:
        src = secret_source(OPENAI_API_KEY, paths)
        add("OK" if src else "FAIL", "임베딩",
            f"{cfg.embed_model_id()}{via} — {OPENAI_API_KEY} " + (f"({src})" if src else "없음"))
    try:
        ls = resolve_llm(cfg, paths)
        lvia = f" (llm_provider={ls.setting})" if ls.setting != ls.provider else ""
        add("OK" if ls.key_present else "FAIL", "LLM",
            f"{ls.provider} {ls.model}{lvia} — {ls.key_name} "
            + (f"({ls.key_source})" if ls.key_present else "없음"))
    except LLMError as e:
        add("FAIL", "LLM", str(e))
    if bool(cfg.alert_telegram):
        tok = get_secret(TELEGRAM_BOT_TOKEN, paths)
        add("OK" if tok and alert_chat_id(paths) else "WARN", "텔레그램 알림",
            f"token {mask(tok)}, chat_id {'있음' if alert_chat_id(paths) else '없음'}")
    try:
        cmp_ = threat.compare_runtime_vendor(cfg.hermes_runtime_dir)
        if not cmp_.get("vendor_ok"):
            add("FAIL", "threat 동봉 사본", "sha256 불일치")
        elif cmp_.get("runtime_sha") and not cmp_.get("same"):
            add("WARN", "threat_patterns", "런타임 파일과 동봉 사본이 다름 (런타임 우선 사용)")
        else:
            add("OK", "threat_patterns", "런타임과 동일" if cmp_.get("same") else "동봉 사본 사용")
        sc = threat.load_scanner(cfg.hermes_runtime_dir)
        add("OK", "위협 스캐너", f"{sc.source}")
    except Exception as e:  # noqa: BLE001
        add("FAIL", "위협 스캐너", f"{type(e).__name__} (dream은 fail-closed로 멈춤)")
    init_msg = _not_initialized(paths)
    if init_msg:
        add("WARN", "데이터", init_msg)
    else:
        try:
            from .ledger import Ledger
            from .store import Store
            st = Store.from_config(paths, cfg)
            st.check_schema()
            st.check_embed_model()
            with Ledger.from_paths(paths, readonly=True) as led:
                led.check_meta(cfg.embed_model_id(), int(cfg.embed_dim))
                runs = led.runs(limit=1)
            add("OK", "Lance·ledger", f"행 {st.count()}개, 모델 {cfg.embed_model_id()}, "
                f"최근 실행 {runs[0].run_id + ' ' + runs[0].status if runs else '없음'}")
        except Exception as e:  # noqa: BLE001
            add("FAIL", "Lance·ledger", f"{type(e).__name__}: {e}")
    if paths.recall_sqlite.exists():
        try:
            import sqlite3
            con = sqlite3.connect(f"file:{paths.recall_sqlite}?mode=ro", uri=True)
            meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
            con.close()
            age_h = (clock.real_now() - paths.recall_sqlite.stat().st_mtime) / 3600
            add("OK" if age_h < float(cfg.serving_stale_hours) else "WARN", "서빙 사본",
                f"{meta.get('count')}행, run {meta.get('run_id')}, {age_h:.1f}시간 전")
        except Exception as e:  # noqa: BLE001
            add("FAIL", "서빙 사본", f"읽기 실패: {type(e).__name__}")
    else:
        add("WARN", "서빙 사본", "없음 (첫 `yume dream` 후 생김)")
    prov = paths.provider_dir
    if (prov / "__init__.py").exists():
        ver = (prov / "VERSION").read_text(encoding="utf-8").splitlines()[0] if (prov / "VERSION").exists() else "?"
        add("OK", "provider", f"{prov} ({ver})")
    else:
        add("WARN", "provider", "설치 안 됨 — deploy/install_provider.sh")
    import os
    ybin = os.path.expanduser(str(cfg.yume_bin))
    add("OK" if os.access(ybin, os.X_OK) else "WARN", "yume_bin", ybin)
    if not getattr(args, "offline", False):
        try:
            from .embedder import make_embedder
            from .llm import make_llm
            make_embedder(cfg, paths).ping()
            add("OK", "임베딩 호출", cfg.embed_model_id())
            llm = make_llm(cfg, paths)
            llm.ping(cfg.extract_model)
            add("OK", "LLM 1토큰 호출", f"{getattr(llm, 'provider', '?')} {llm.model_for(cfg.extract_model)}")
        except Exception as e:  # noqa: BLE001
            from .threat import redact_secrets
            add("FAIL", "네트워크 점검", f"{type(e).__name__}: {redact_secrets(str(e))[0][:160]}")
    fails = [c for c in checks if c[0] == "FAIL"]
    if _json_out(args):
        _print_json({"ok": not fails, "checks": [{"level": a, "name": b, "detail": c} for a, b, c in checks]})
    else:
        for a, b, c in checks:
            print(f"[{a:4}] {b}: {c}")
        print("정상" if not fails else f"문제 {len(fails)}건")
    return EXIT_FAIL if fails else EXIT_OK


def cmd_init(args: argparse.Namespace, paths: Paths) -> int:
    from .config import write_default_config
    from .ledger import Ledger
    from .livedb import LiveDB
    from .store import Store
    paths.ensure_data_dirs()
    try:
        with _locked(paths):
            created = write_default_config(paths, overwrite=False)
            cfg = _cfg(paths)
            with Ledger.from_paths(paths) as led:
                led.init_meta(cfg.embed_model_id(), int(cfg.embed_dim), force=bool(args.force))
            Store.from_config(paths, cfg, create=True)
            lv = LiveDB.open(paths, mode="rw")
            if lv is not None:
                lv.close()
    except AlreadyRunning:
        print("already running")
        return EXIT_OK
    out = {"hermes_home": str(paths.hermes_home), "data_dir": str(paths.data_dir),
           "config_created": bool(created), "embed_model": cfg.embed_model_id()}
    if _json_out(args):
        _print_json(out)
    else:
        print(f"초기화 완료: {paths.data_dir}")
        print(f"config.json {'생성' if created else '기존 유지'} · 임베딩 {cfg.embed_model_id()}")
    return EXIT_OK


# ── dream ────────────────────────────────────────────────────────────────────

def _dream_summary(stats: Any, paths: Paths) -> str:
    s = stats
    lines = [f"실행 {s.run_id}: {s.status} ({s.mode})",
             f"입력: 메시지 {s.messages_in} · md {s.md_files} · inbox {s.inbox_items} · "
             f"창 {s.windows_total} (ok {s.windows_ok}, 빈 창 {s.windows_empty}, 실패 {s.windows_failed}, "
             f"격리 {s.windows_quarantined}, 미룸 {s.windows_deferred})",
             f"주장 {s.claims_extracted} (거절 {s.claims_rejected}) → 새 기억 {s.created} · 강화 {s.reinforced} · "
             f"대체 {s.superseded} · 통합 {s.consolidated} · 만료 {s.expired} · 휴면 {s.dormant} · "
             f"부활 {s.revived} · 잊음 {s.forgotten}",
             f"LLM {s.llm_calls}회 · 임베딩 {s.embed_inputs}건 · 약 ${s.cost_usd} · "
             f"Lance v{s.lance_version_before}→v{s.lance_version_after}"]
    if s.held_ops:
        lines.append(f"보류 연산 {s.held_ops}건 (pinned/durable 보호)")
    dl = _latest_dream_log(paths)
    if dl:
        lines.append(f"Dream Log: {dl}")
    return "\n".join(lines)


def _approve_run(args: argparse.Namespace, paths: Paths, cfg: Any) -> int:
    from . import plan as _plan, rem
    try:
        with _locked(paths):
            ctx = _open_ctx(paths, cfg, mode="live", need_llm=False, offline=True)
            try:
                _guards(ctx)
                try:
                    res = _plan.approve_held(ctx, args.approve_run)
                except (ValueError, FileNotFoundError) as e:
                    _err(str(e))
                    return EXIT_FAIL
                p = _plan.read_plan(paths.plan_json(ctx.run_id))
                ctx.stats.run_id, ctx.stats.status = ctx.run_id, p.status
                rem.post_commit(ctx, p)
            finally:
                _close_ctx(ctx)
    except AlreadyRunning:
        print("already running")
        return EXIT_OK
    out = {"run_id": res.run_id, "approved": args.approve_run, "status": "committed",
           "lance_version_after": res.lance_version_after}
    if _json_out(args):
        _print_json(out)
    else:
        print(f"보류 실행 {args.approve_run} 승인 → 실행 {res.run_id} 커밋 (Lance v{res.lance_version_after})")
    return EXIT_OK


def cmd_dream(args: argparse.Namespace, paths: Paths) -> int:
    from . import rem
    from .store import StoreMissing
    dry, offline = bool(args.dry_run), bool(args.offline)
    if offline and not dry and is_live_home(paths.hermes_home):
        _err("라이브 HERMES_HOME에서는 --offline을 --dry-run과 함께만 쓸 수 있습니다.")
        return EXIT_FAIL
    msg = _not_initialized(paths)
    if msg:
        _err(msg)
        return EXIT_FAIL
    cfg = _cfg(paths)
    if args.approve_run:
        if dry:
            _err("--approve-run은 --dry-run과 함께 쓸 수 없습니다.")
            return EXIT_FAIL
        return _approve_run(args, paths, cfg)

    def go() -> Any:
        ctx = _open_ctx(paths, cfg, mode="dry" if dry else "live", dry_run=dry, offline=offline,
                        need_llm=True, max_llm_calls=args.max_llm_calls)
        ctx.settle_minutes = args.settle_minutes
        try:
            return rem.run_dream(ctx)
        finally:
            _close_ctx(ctx)

    try:
        if dry:
            stats = go()
        else:
            with _locked(paths):
                stats = go()
    except AlreadyRunning:
        print("already running")
        return EXIT_OK
    except StoreMissing:
        _err("Lance 저장소가 없습니다. 먼저 `yume init`을 실행하세요.")
        return EXIT_FAIL
    if _json_out(args):
        _print_json({"run_id": stats.run_id, "status": stats.status, "stats": stats.to_dict()})
    else:
        print(_dream_summary(stats, paths))
    if stats.status == "failed":
        return EXIT_FAIL
    if stats.status == "held":
        return EXIT_HELD
    return EXIT_OK


# ── read-only views ──────────────────────────────────────────────────────────

def _open_store_ro(paths: Paths, cfg: Any):
    from .store import Store
    return Store.from_config(paths, cfg)


def cmd_status(args: argparse.Namespace, paths: Paths) -> int:
    from collections import Counter
    from .ledger import Ledger
    msg = _not_initialized(paths)
    if msg:
        _err(msg)
        return EXIT_FAIL
    cfg = _cfg(paths)
    now = clock.now()
    out: dict[str, Any] = {"hermes_home": str(paths.hermes_home), "now": clock.fmt_kst(now)}
    with Ledger.from_paths(paths, readonly=True) as led:
        out["runs"] = [{"run_id": r.run_id, "mode": r.mode, "status": r.status,
                        "started": clock.fmt_kst(r.started_at) if r.started_at else None,
                        "lance": [r.lance_version_before, r.lance_version_after],
                        "error": r.error} for r in led.runs(limit=5)]
        out["windows"] = {st: len(led.windows(status=st)) for st in ("failed", "quarantined")}
        out["watermarks"] = len(led.all_wms())
        out["md_files"] = len(led.all_md())
    store = _open_store_ro(paths, cfg)
    rows = store.load_working_set(with_vectors=False)
    out["rows"] = dict(Counter(r.status for r in rows.values()))
    out["tiers_active"] = dict(Counter(r.tier for r in rows.values() if r.status == "active"))
    out["pinned"] = sum(1 for r in rows.values() if r.pinned and r.status == "active")
    out["lance_version"] = store.version()
    if paths.recall_sqlite.exists():
        import sqlite3
        try:
            con = sqlite3.connect(f"file:{paths.recall_sqlite}?mode=ro", uri=True)
            meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
            con.close()
            out["serving"] = {"run_id": meta.get("run_id"), "count": meta.get("count"),
                              "built_at": clock.fmt_kst(float(meta.get("built_at") or 0))}
        except Exception as e:  # noqa: BLE001
            out["serving"] = {"error": type(e).__name__}
    from .livedb import LiveDB
    live = LiveDB.open(paths, mode="ro")
    try:
        if live is not None:
            out["inbox_pending"] = len(live.pending_inbox())
            if args.recall:
                from types import SimpleNamespace
                from . import alerts
                out["health_24h"] = alerts.health_summary(SimpleNamespace(live=live, cfg=cfg, paths=paths),
                                                          now=now)
                snap = live.snapshot()
                ev = live.recall_range(0, snap.max_recall_id)
                recent = [e for e in ev if e.ts >= now - 86400]
                out["recall_24h"] = dict(Counter(e.kind for e in recent))
    finally:
        if live is not None:
            live.close()
    if args.recall:
        top = sorted((r for r in rows.values() if r.status == "active"),
                     key=lambda r: (-(r.recall_used_count or 0), -(r.recall_injected_count or 0), r.id))[:10]
        out["recall_top"] = [{"id": r.id, "injected": r.recall_injected_count, "used": r.recall_used_count,
                              "text": _short(r.text, 80)} for r in top if r.recall_injected_count or r.recall_used_count]
    if paths.alerts_log.exists():
        lines = paths.alerts_log.read_text(encoding="utf-8").splitlines()[-5:]
        out["alerts_tail"] = [json.loads(x).get("message", "") for x in lines if x.strip().startswith("{")]
    if _json_out(args):
        _print_json(out)
        return EXIT_OK
    print(f"HERMES_HOME {out['hermes_home']} · 기준 {out['now']} · Lance v{out['lance_version']}")
    print("행: " + ", ".join(f"{k} {v}" for k, v in sorted(out["rows"].items())) + f" · pin {out['pinned']}")
    print(f"워터마크 {out['watermarks']}개 · md {out['md_files']}개 · 실패 창 {out['windows']['failed']} · "
          f"격리 창 {out['windows']['quarantined']} · inbox 대기 {out.get('inbox_pending', 0)}")
    if "serving" in out:
        print(f"서빙 사본: {out['serving']}")
    for r in out["runs"]:
        print(f"  {r['run_id']} {r['mode']} {r['status']} {r['started']} Lance {r['lance'][0]}→{r['lance'][1]}"
              + (f" ({r['error']})" if r.get("error") else ""))
    if args.recall:
        print(f"회상 24시간: {out.get('recall_24h')} · health {out.get('health_24h')}")
        for t in out.get("recall_top", []):
            print(f"  {t['id']} 주입 {t['injected']} 사용 {t['used']}  {t['text']}")
    for a in out.get("alerts_tail", []):
        print(f"알림: {a}")
    return EXIT_OK


def cmd_search(args: argparse.Namespace, paths: Paths) -> int:
    from .embedder import embed_input, make_embedder
    msg = _not_initialized(paths)
    if msg:
        _err(msg)
        return EXIT_FAIL
    cfg = _cfg(paths)
    store = _open_store_ro(paths, cfg)
    emb = make_embedder(cfg, paths, offline=bool(args.offline))
    q = embed_input("", args.query)
    vec = emb.embed([q], kind="search")[0]
    where = ("status NOT IN ('forgotten', 'quarantined')" if args.include_inactive
             else "status = 'active'")
    hits = store.search(vec, k=max(1, min(int(args.limit), 50)), where=where)
    now = clock.now()
    if _json_out(args):
        _print_json([dict(_row_view(r, now, cfg), cos=round(float(c), 4)) for r, c in hits])
        return EXIT_OK
    if not hits:
        print("결과 없음")
    for r, c in hits:
        print(f"cos {c:.3f}  " + _fmt_row(r, now, cfg))
    return EXIT_OK


def _match_rows(rows: dict, args: argparse.Namespace) -> list:
    from .paths import load_provider_module
    cf = load_provider_module("corefmt")
    out = list(rows.values())
    if args.id:
        out = [r for r in out if r.id == args.id or r.id.startswith(args.id)]
    if args.query:
        q = _norm(args.query)
        out = [r for r in out if q in _norm(r.text) or q in _norm(r.subject)]
    if args.label:
        q = _norm(args.label)
        out = [r for r in out if q in _norm(cf.entry_label(r.text) or "") or q in _norm(r.subject)]
    if args.tag:
        out = [r for r in out if r.source == f"debug:{args.tag}"]
    if getattr(args, "status", None):
        want = {s.strip() for s in args.status.split(",") if s.strip()}
        out = [r for r in out if r.status in want]
    return sorted(out, key=lambda r: (r.created_at or 0.0, r.id))


def cmd_inspect(args: argparse.Namespace, paths: Paths) -> int:
    msg = _not_initialized(paths)
    if msg:
        _err(msg)
        return EXIT_FAIL
    if not (args.id or args.query or args.label or args.tag or args.status):
        _err("--id, --query, --label, --tag, --status 중 하나가 필요합니다.")
        return EXIT_FAIL
    cfg = _cfg(paths)
    rows = _open_store_ro(paths, cfg).load_working_set(with_vectors=False)
    found = _match_rows(rows, args)
    now = clock.now()
    if _json_out(args):
        _print_json([_row_view(r, now, cfg) for r in found])
        return EXIT_OK
    if not found:
        print("해당 행 없음")
    for r in found:
        print(_fmt_row(r, now, cfg))
        print(f"    source {r.source} · 근거 {r.evidence_count} (사용자 {r.user_evidence_count}) · "
              f"세션 {len(r.source_session_ids)} · 처음 {clock.fmt_kst(r.first_seen_at) if r.first_seen_at else '-'}"
              + (f" · 대체됨→{r.superseded_by}" if r.superseded_by else "")
              + (f" · 기한 {clock.kst_date(r.valid_until)}" if r.valid_until else ""))
    return EXIT_OK


def cmd_pin(args: argparse.Namespace, paths: Paths) -> int:
    msg = _not_initialized(paths)
    if msg:
        _err(msg)
        return EXIT_FAIL
    cfg = _cfg(paths)
    rows = _open_store_ro(paths, cfg).load_working_set(with_vectors=False)
    pins = sorted((r for r in rows.values() if r.pinned), key=lambda r: (r.created_at or 0.0, r.id))
    now = clock.now()
    if _json_out(args):
        _print_json([_row_view(r, now, cfg) for r in pins])
        return EXIT_OK
    total = sum(len(r.text) for r in pins if r.status == "active" and not r.in_core)
    print(f"pin {len(pins)}개 (핵심 파일 밖 {total}/{cfg.pins_budget_chars}자)")
    for r in pins:
        print(_fmt_row(r, now, cfg))
    return EXIT_OK


# ── admin writes ─────────────────────────────────────────────────────────────

def _find_row(ws: Any, memory_id: str) -> Any:
    row = ws.get(memory_id)
    if row is None and memory_id:
        cands = [r for r in ws.rows.values() if r.id.startswith(memory_id)]
        row = cands[0] if len(cands) == 1 else None
    return row


def cmd_unpin(args: argparse.Namespace, paths: Paths) -> int:
    msg = _not_initialized(paths)
    if msg:
        _err(msg)
        return EXIT_FAIL
    cfg = _cfg(paths)

    def mutate(ctx: Any, ws: Any) -> str:
        row = _find_row(ws, args.memory_id)
        if row is None:
            return "not_found"
        if not row.pinned:
            return "not_pinned"
        ws.update(row.id, {"pinned": False}, op="unpin", reason="cli:unpin", user_evidence=True,
                  guard_exempt=True)
        ws.add_audit("unpin", row.id, "cli")
        return "ok"

    try:
        out, res = _admin_commit(paths, cfg, "admin", mutate)
    except AlreadyRunning:
        print("already running")
        return EXIT_OK
    if _json_out(args):
        _print_json({"result": out, "run_id": res.run_id if res else None})
    else:
        print({"ok": "pin을 해제했습니다 (행은 그대로, 보호 등급은 근거에 따라 다시 계산).",
               "not_found": "해당 id의 행이 없습니다.", "not_pinned": "pin이 아닙니다."}[out])
    return EXIT_FAIL if out == "not_found" else EXIT_OK


def cmd_forget(args: argparse.Namespace, paths: Paths) -> int:
    from .rem import _forget_detail
    from .types import SuppressRow, suppress_reason, text_sha
    msg = _not_initialized(paths)
    if msg:
        _err(msg)
        return EXIT_FAIL
    cfg = _cfg(paths)

    def mutate(ctx: Any, ws: Any) -> str:
        row = _find_row(ws, args.memory_id)
        if row is None:
            return "not_found"
        if row.status == "forgotten":
            return "already"
        if row.pinned and not args.confirm:
            return "confirm_required"
        if ws.update(row.id, {"status": "forgotten"}, op="forget", reason="cli:forget",
                     user_evidence=True, guard_exempt=True) is None:
            return "already"
        if row.vector is not None:
            ws.add_suppress(SuppressRow(id=row.id, vector=row.vector.copy(), text_sha=text_sha(row.text),
                                        kind=row.kind, created_at=ctx.now,
                                        reason=suppress_reason(ctx.run_id, row.text)))
        ws.add_audit("forget", row.id, _forget_detail(args.reason, row.text))
        return "ok"

    try:
        out, res = _admin_commit(paths, cfg, "admin", mutate)
    except AlreadyRunning:
        print("already running")
        return EXIT_OK
    if _json_out(args):
        _print_json({"result": out, "run_id": res.run_id if res else None})
    else:
        print({"ok": f"잊음으로 표시했습니다. {cfg.forget_purge_days}일 뒤 영구 삭제됩니다.",
               "already": "이미 잊은 기억입니다.", "not_found": "해당 id의 행이 없습니다.",
               "confirm_required": "pinned 기억은 --confirm이 필요합니다."}[out])
    return EXIT_OK if out in ("ok", "already") else EXIT_FAIL


def cmd_restore(args: argparse.Namespace, paths: Paths) -> int:
    """`yume restore --run X [--reprocess] [--unforget]` (§8.3, DEVIATIONS F-3/F-4/F-30).

    memories ← the version before X (X and every later run are undone together). Forgets made by
    the undone runs are applied again unless --unforget (content the user asked to forget never
    comes back by accident). --reprocess also rewinds watermarks and md offsets, drops the undone
    runs' window rows and puts their inbox items back to pending, so the next dream re-reads
    everything those runs consumed. Refused for runs before a `yume reembed` (another Lance
    lineage), for versions Lance no longer has, and while a run waits for replay."""
    from . import export, plan as _plan
    from .store import sql_quote
    from .types import AuditRow, suppress_reason_run
    msg = _not_initialized(paths)
    if msg:
        _err(msg)
        return EXIT_FAIL
    cfg = _cfg(paths)
    out: dict[str, Any] = {}
    from .types import make_run_id
    try:
        with _locked(paths):
            ctx = _open_ctx(paths, cfg, offline=True, run_id=f"{make_run_id(clock.now())}-restore")
            try:
                rec = ctx.ledger.get_run(args.run)
                if rec is None or rec.lance_version_before is None:
                    _err(f"실행 {args.run}을(를) 찾을 수 없거나 되돌릴 버전이 없습니다.")
                    return EXIT_FAIL
                try:
                    _refuse_if_planned(ctx)
                except PlannedRunPending as e:
                    _err(str(e))
                    return EXIT_FAIL
                rowid = ctx.ledger.run_rowid(args.run) or 0
                if rowid < ctx.ledger.lance_epoch_min_rowid():
                    prev = sorted(p.name for p in paths.data_dir.glob("lancedb.prev-*"))
                    _err(f"실행 {args.run}은(는) `yume reembed` 이전 저장소의 실행입니다. 현재 Lance의 버전 "
                         f"번호와 맞지 않아 되돌릴 수 없습니다. 이전 저장소: {', '.join(prev) or '없음'}")
                    return EXIT_FAIL
                target = int(rec.lance_version_before)
                if target not in set(ctx.store.list_versions()):
                    pre = sorted(p.name for p in paths.backups_dir.glob("lancedb-prepurge-*.tar.gz"))
                    _err(f"Lance에 버전 {target}이(가) 더 이상 없습니다(정리 기간 {cfg.lance_cleanup_days}일 경과 "
                         f"또는 비밀값 격리 삭제). 격리 전 백업: {', '.join(pre) or '없음'}")
                    return EXIT_FAIL
                later = [r.run_id for r in ctx.ledger.runs(limit=1000)
                         if (r.started_at or 0) > (rec.started_at or 0) and r.status in ("committed", "held")
                         and r.run_id != args.run]
                undone = [args.run] + later
                forgot = sorted({a.memory_id for a in ctx.ledger.audits()
                                 if a.op == "forget" and a.run_id in set(undone) and a.memory_id})
                before = ctx.store.version()
                after = ctx.store.restore(target)
                reforgot: list[str] = []
                if args.unforget:
                    for s in ctx.store.load_suppress():
                        if suppress_reason_run(s.reason) in set(undone):
                            ctx.store.delete_suppress(f"id = {sql_quote(s.id)}")
                else:
                    ws = _plan.WorkingSet(ctx.store.load_working_set(), run_id=ctx.run_id, now=ctx.now,
                                          embed_model=cfg.embed_model_id())
                    for mid in forgot:
                        row = ws.get(mid)
                        if row is None or row.status in ("forgotten", "quarantined"):
                            continue
                        if ws.update(mid, {"status": "forgotten"}, op="forget", reason="restore:reforget",
                                     user_evidence=True, guard_exempt=True) is not None:
                            ws.add_audit("forget", mid, "reason=restore:reforget")
                            reforgot.append(mid)
                    if ws.ops:
                        _commit_ws(ctx, ws)
                        after = ctx.store.version()
                requeued = 0
                if args.reprocess:
                    if rec.wm_before_json:
                        ctx.ledger.restore_wms(rec.wm_before_json, rolled_back_run=args.run,
                                               undone_runs=later)
                    if ctx.live is not None:
                        ops = None if args.unforget else ["remember", "core_add", "core_replace",
                                                          "core_remove"]
                        requeued = ctx.live.requeue(undone, ops=ops)
                seen_dropped = ctx.ledger.forget_core_seen_from(undone)
                ctx.ledger.add_audit(AuditRow(ts=ctx.now, run_id=args.run, op="restore", memory_id=None,
                                              detail=f"lance {before}->{target} (v{after})"
                                                     f"{' reprocess' if args.reprocess else ''}"
                                                     f"{' unforget' if args.unforget else ''}"))
                for rid in undone:
                    ctx.ledger.update_run(rid, error=f"restored to v{target} (restore {args.run})")
                export.build_serving(ctx)
                out = {"run_id": args.run, "lance_version_restored": target, "new_version": after,
                       "reprocess": bool(args.reprocess), "later_runs_undone": later,
                       "reforgotten": reforgot, "unforget": bool(args.unforget),
                       "inbox_requeued": requeued, "core_seen_reset": seen_dropped}
            finally:
                _close_ctx(ctx)
    except AlreadyRunning:
        print("already running")
        return EXIT_OK
    if _json_out(args):
        _print_json(out)
    else:
        print(f"Lance를 실행 {args.run} 이전(v{out['lance_version_restored']})으로 되돌렸습니다 → "
              f"v{out['new_version']}. 서빙 사본 재생성 완료.")
        if later:
            print(f"주의: 이후 실행 {len(later)}개도 함께 되돌려졌습니다: {', '.join(later)}")
        if out["reforgotten"]:
            print(f"되돌린 실행에서 잊은 기억 {len(out['reforgotten'])}개는 다시 잊음으로 표시했습니다 "
                  f"(취소하려면 --unforget).")
        if args.reprocess:
            print(f"워터마크·md 오프셋을 되돌리고 inbox 항목 {out['inbox_requeued']}개를 다시 대기로 돌렸습니다 "
                  f"(다음 dream에서 다시 처리).")
        else:
            print("입력(대화 창·md·inbox)은 다시 처리하지 않습니다. 다시 처리하려면 --reprocess.")
    return EXIT_OK


def cmd_core_check(args: argparse.Namespace, paths: Paths) -> int:
    from . import core_check
    from .ledger import Ledger
    from .sources.core_files import load_limits, read_core
    msg = _not_initialized(paths)
    if msg:
        _err(msg)
        return EXIT_FAIL
    cfg = _cfg(paths)
    core = read_core(paths)
    with Ledger.from_paths(paths, readonly=True) as led:
        seen = led.core_seen()
    adds, removes = core_check.diff_core(core, seen)
    rows = _open_store_ro(paths, cfg).load_working_set(with_vectors=False)
    req_missing = [r for r in rows.values()
                   if r.core_required and r.status == "active"
                   and not core_check.core_required_present(r, core, embedder=None,
                                                            min_cos=float(cfg.core_match_cos))]
    limits = load_limits(paths)
    from .paths import load_provider_module
    cf = load_provider_module("corefmt")
    files = {t: {"entries": len(es), "chars": len(cf.ENTRY_DELIMITER.join(e.text for e in es)),
                 "limit": (limits.get(t) or {}).get("limit")} for t, es in core.items()}
    out = {"files": files,
           "new_entries": [{"target": e.target, "text": e.text} for e in adds],
           "removed_entries": [{"target": s.target, "text": s.text, "memory_id": s.memory_id} for s in removes],
           "core_required_missing": [{"id": r.id, "text": r.text, "pinned": r.pinned} for r in req_missing]}
    if _json_out(args):
        _print_json(out)
        return EXIT_OK
    for t, f in files.items():
        print(f"{'USER.md' if t == 'user' else 'MEMORY.md'}: {f['entries']}항목 {f['chars']}/{f['limit']}자")
    print(f"아직 반영 안 된 항목 {len(adds)}개 · 빠진 항목 {len(removes)}개 (다음 dream에서 반영)")
    for s in removes:
        print(f"  빠짐: {_short(s.text, 100)}")
    for r in req_missing:
        print(f"  필수 항목 이탈: {_short(r.text, 100)} → `yume core-restore {r.id}`")
    return EXIT_OK


def cmd_core_restore(args: argparse.Namespace, paths: Paths) -> int:
    from . import core_check
    msg = _not_initialized(paths)
    if msg:
        _err(msg)
        return EXIT_FAIL
    cfg = _cfg(paths)
    try:
        with _locked(paths):           # single ledger writer (audit row)
            ctx = _open_ctx(paths, cfg, offline=True, need_embedder=False, run_id="core-restore")
            try:
                res = core_check.restore(paths, args.memory_id, target=args.target, store=ctx.store,
                                         ledger=ctx.ledger, now=ctx.now)
            finally:
                _close_ctx(ctx)
    except AlreadyRunning:
        print("already running")
        return EXIT_OK
    out = {"ok": res.ok, "target": res.target, "path": res.path, "backup": res.backup, "reason": res.reason}
    if _json_out(args):
        _print_json(out)
    elif res.ok:
        print(f"{res.path}에 복원했습니다{' (이미 있음)' if res.reason == 'already_present' else ''}. "
              f"백업: {res.backup or '-'}")
    else:
        print(f"복원하지 않았습니다: {res.reason}")
    return EXIT_OK if res.ok else EXIT_FAIL


def cmd_core_proposal(args: argparse.Namespace, paths: Paths) -> int:
    from . import core_check
    msg = _not_initialized(paths)
    if msg:
        _err(msg)
        return EXIT_FAIL
    cfg = _cfg(paths)
    try:
        with _locked(paths):
            store = _open_store_ro(paths, cfg)
            res = core_check.apply_proposal(paths, args.path, store=store, now=clock.now())
    except AlreadyRunning:
        print("already running")
        return EXIT_OK
    out = {"ok": res.ok, "target": res.target, "path": res.path, "backup": res.backup, "reason": res.reason}
    if _json_out(args):
        _print_json(out)
    elif res.ok:
        print(f"제안을 적용했습니다: {res.path} (백업 {res.backup or '-'})")
    else:
        print(f"적용하지 않았습니다: {res.reason}")
    return EXIT_OK if res.ok else EXIT_FAIL


def cmd_export(args: argparse.Namespace, paths: Paths) -> int:
    from . import export
    from .types import make_run_id
    msg = _not_initialized(paths)
    if msg:
        _err(msg)
        return EXIT_FAIL
    cfg = _cfg(paths)
    try:
        with _locked(paths):
            ctx = _open_ctx(paths, cfg, offline=True, need_embedder=False,
                            run_id=f"{make_run_id(clock.now())}-export")
            try:
                _guards(ctx)
                path = export.build_serving(ctx)
            finally:
                _close_ctx(ctx)
    except AlreadyRunning:
        print("already running")
        return EXIT_OK
    import sqlite3
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    n = con.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    con.close()
    if _json_out(args):
        _print_json({"path": str(path), "items": n, "notes": ctx.report.notes})
    else:
        print(f"서빙 사본 재생성: {path} ({n}행)")
        for note in ctx.report.notes:
            print(f"  참고: {note}")
    return EXIT_OK


def cmd_reembed(args: argparse.Namespace, paths: Paths) -> int:
    """Re-embed every row (and suppress rows whose text still exists) with a new model into a new
    Lance directory, then swap it in (old directory kept as lancedb.prev-<stamp>)."""
    import hashlib
    import os
    import shutil
    import numpy as np
    from . import export
    from .config import save_config
    from .embedder import embed_input, make_embedder
    from .ledger import Ledger
    from .store import Store
    from .types import RunBudget, SuppressRow
    if args.offline and is_live_home(paths.hermes_home):
        _err("라이브 HERMES_HOME에서는 --offline(가짜 임베더)로 다시 임베딩할 수 없습니다.")
        return EXIT_FAIL
    msg = _not_initialized(paths)
    if msg:
        _err(msg)
        return EXIT_FAIL
    cfg = _cfg(paths)
    new_cfg = cfg.replace(embed_model=args.model or cfg.embed_model, embed_dim=int(args.dim or cfg.embed_dim))
    try:
        with _locked(paths):
            old = Store.from_config(paths, cfg)
            old.check_schema()
            rows = old.load_working_set(with_vectors=True)
            if new_cfg.embed_model_id() == cfg.embed_model_id() and all(
                    r.embed_model == cfg.embed_model_id() for r in rows.values()):
                if _json_out(args):
                    _print_json({"model": cfg.embed_model_id(), "changed": False})
                else:
                    print(f"이미 {cfg.embed_model_id()}입니다. 바꿀 것이 없습니다.")
                return EXIT_OK
            budget = RunBudget(max_llm_calls=0, max_embed_inputs=10 ** 7, max_runtime_s=6 * 3600)
            emb = make_embedder(new_cfg, paths, budget=budget, offline=bool(args.offline))
            emb.ping()
            ids = sorted(rows)
            texts = [embed_input(rows[i].subject, rows[i].text) for i in ids]
            vecs = emb.embed(texts, kind="reembed") if texts else np.zeros((0, int(new_cfg.embed_dim)), np.float32)
            new_id = new_cfg.embed_model_id()
            for i, v in zip(ids, vecs):
                rows[i].vector = np.asarray(v, dtype=np.float32)
                rows[i].embed_model = new_id
            sup_old = old.load_suppress()
            sup_new: list[SuppressRow] = []
            lost = 0
            for s in sup_old:
                src = rows.get(s.id)
                if src is not None:
                    v = src.vector
                else:     # original text is gone (by design): keep text_sha, use a neutral unit vector
                    lost += 1
                    seed = int.from_bytes(hashlib.sha256(s.id.encode()).digest()[:8], "little")
                    v = np.random.default_rng(seed).standard_normal(int(new_cfg.embed_dim)).astype(np.float32)
                    v /= np.linalg.norm(v) or 1.0
                sup_new.append(SuppressRow(id=s.id, vector=v, text_sha=s.text_sha, kind=s.kind,
                                           created_at=s.created_at, reason=s.reason))
            hist = old.history()
            stamp = clock.kst_stamp(clock.real_now())
            new_dir = paths.data_dir / f"lancedb.reembed-{stamp}"
            if new_dir.exists():
                shutil.rmtree(new_dir)
            ns = Store.open(new_dir, dim=int(new_cfg.embed_dim), embed_model=new_id, create=True)
            ns.commit(upserts=[rows[i] for i in ids], history=hist, suppress=sup_new)
            ns.check_schema()
            prev = paths.data_dir / f"lancedb.prev-{stamp}"
            os.rename(paths.lancedb_dir, prev)
            try:
                os.rename(new_dir, paths.lancedb_dir)
            except OSError:
                os.rename(prev, paths.lancedb_dir)      # put the old store back
                raise
            with Ledger.from_paths(paths) as led:
                led.init_meta(new_id, int(new_cfg.embed_dim), force=True)
                led.start_lance_epoch()       # older runs' version numbers belong to lancedb.prev-*
            cur = json.loads(paths.config_json.read_text(encoding="utf-8"))
            cur.update(embed_model=new_cfg.embed_model, embed_dim=int(new_cfg.embed_dim))
            save_config(paths, cur)
            ctx = _open_ctx(paths, _cfg(paths), offline=True, need_embedder=False, run_id=f"reembed-{stamp}")
            try:
                export.build_serving(ctx)
            finally:
                _close_ctx(ctx)
    except AlreadyRunning:
        print("already running")
        return EXIT_OK
    out = {"model": new_id, "changed": True, "rows": len(ids), "suppress": len(sup_new),
           "suppress_text_lost": lost, "previous_dir": str(prev)}
    if _json_out(args):
        _print_json(out)
    else:
        print(f"{len(ids)}행을 {new_id}로 다시 임베딩했습니다. 이전 저장소: {prev}")
        if lost:
            print(f"원문이 없는 억제 항목 {lost}개는 text_sha로만 막습니다.")
    return EXIT_OK


# ── config / debug / wired handlers ──────────────────────────────────────────

def cmd_config(args: argparse.Namespace, paths: Paths) -> int:
    from .config import ConfigError, set_key
    if args.action == "show":
        cfg = _cfg(paths)
        _print_json(cfg.as_dict())
        return EXIT_OK
    if not args.key:
        _err("key가 필요합니다.")
        return EXIT_FAIL
    if args.action == "get":
        cfg = _cfg(paths)
        if args.key not in cfg:
            _err(f"알 수 없는 키: {args.key}")
            return EXIT_FAIL
        _print_json(cfg[args.key])
        return EXIT_OK
    if args.value is None:
        _err("value가 필요합니다.")
        return EXIT_FAIL
    try:
        cfg = set_key(paths, args.key, args.value, force=bool(args.force))
    except (ConfigError, ValueError) as e:
        _err(f"설정하지 않았습니다: {e}")
        return EXIT_FAIL
    if _json_out(args):
        _print_json({"key": args.key, "value": cfg[args.key]})
    else:
        print(f"{args.key} = {json.dumps(cfg[args.key], ensure_ascii=False)}")
    return EXIT_OK


def _plant(args: argparse.Namespace, paths: Paths) -> int:
    from .embedder import embed_input
    from .clock import DAY
    from .types import ALL_KINDS, KIND_BASE, Claim
    from .upsert import row_from_claim
    from .normalize import subject_key
    kind = args.kind or "fact"
    if kind not in ALL_KINDS:
        _err(f"알 수 없는 kind: {kind}")
        return EXIT_FAIL
    n = max(1, int(args.count))
    cfg = _cfg(paths)
    tag = args.tag or "e2e"

    def mutate(ctx: Any, ws: Any) -> list[str]:
        emb = ctx.embedder if args.offline else None
        if emb is None:
            from .embedder import make_embedder
            emb = make_embedder(cfg, paths, budget=ctx.budget)
        now = float(ctx.now)
        claims = []
        for i in range(n):
            base = args.text or f"E2E 디버그 {kind} 기억 ({tag})"
            text = base if n == 1 else f"{base} #{i + 1:03d}"
            subj = text[:30]
            vu = now + float(cfg.state_default_ttl_days) * DAY if kind in ("state", "schedule") else None
            c = Claim(origin_key=f"x:debug:{tag}:{ctx.run_id}:{i}", source=f"debug:{tag}", kind=kind,
                      target="user", subject=subj, text=text, event_time=now, valid_until=vu,
                      evidence_keys=[f"x:debug:{ctx.run_id}:{i}"], evidence_roles=["user"],
                      session_ids=[f"debug:{tag}"], first_seen_at=now, last_seen_at=now,
                      last_user_evidence_at=now, user_evidence_count=1, user_session_count=1,
                      importance=float(args.importance if args.importance is not None else KIND_BASE.get(kind, 0.5)),
                      subject_key=subject_key(subj))
            claims.append(c)
        vecs = emb.embed([embed_input(c.subject, c.text) for c in claims], kind="debug")
        ids = []
        for c, v in zip(claims, vecs):
            c.vector = v
            row = row_from_claim(c, ctx=ctx)
            ws.insert(row, reason="debug_plant", user_evidence=True)
            ids.append(row.id)
        return ids

    out, res = _admin_commit(paths, cfg, "debug", mutate)
    if _json_out(args):
        _print_json({"ids": out, "run_id": res.run_id if res else None})
    else:
        for i in out:
            print(i)
    return EXIT_OK


def _event(args: argparse.Namespace, paths: Paths) -> int:
    from .paths import load_provider_module
    from .types import RECALL_EVENT_KINDS
    kind = args.kind or "used"
    if kind not in RECALL_EVENT_KINDS:
        _err(f"알 수 없는 이벤트 kind: {kind} ({', '.join(RECALL_EVENT_KINDS)})")
        return EXIT_FAIL
    if not args.target:
        _err("--target <memory id>가 필요합니다.")
        return EXIT_FAIL
    ts = clock.parse_now_spec(args.at, base=clock.now()) if args.at else clock.now()
    ls = load_provider_module("live_schema")
    paths.ensure_data_dirs()
    conn = ls.connect(str(paths.live_db))
    try:
        cur = conn.execute(
            "INSERT INTO recall_events(ts, session_id, platform, turn_no, memory_id, kind, cos, mode, snapshot_run)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (float(ts), f"debug:{args.tag}", "cli", 0, args.target, kind, None,
             "vector" if kind in ("injected", "tool_hit", "shadow") else None, None))
        conn.commit()
        eid = cur.lastrowid
    finally:
        conn.close()
    if _json_out(args):
        _print_json({"id": eid, "ts": ts, "kind": kind, "memory_id": args.target})
    else:
        print(f"recall_events {eid}: {kind} {args.target} @ {clock.fmt_kst(ts)}")
    return EXIT_OK


def cmd_debug(args: argparse.Namespace, paths: Paths) -> int:
    try:
        refuse_if_live(paths.hermes_home, f"yume debug {args.action}")
    except LiveHomeRefused as e:
        _err(str(e))
        return EXIT_FAIL
    msg = _not_initialized(paths)
    if msg:
        _err(msg)
        return EXIT_FAIL
    try:
        return _plant(args, paths) if args.action == "plant" else _event(args, paths)
    except AlreadyRunning:
        print("already running")
        return EXIT_OK


def cmd_migrate(args: argparse.Namespace, paths: Paths) -> int:
    from . import migrate
    if getattr(args, "offline", False) and not args.dry_run and is_live_home(paths.hermes_home):
        _err("라이브 HERMES_HOME에서는 --offline을 --dry-run과 함께만 쓸 수 있습니다.")
        return EXIT_FAIL
    return migrate.cli_handler(args, paths)


def cmd_calibrate(args: argparse.Namespace, paths: Paths) -> int:
    from . import calibrate
    return calibrate.cli_handler(args, paths)


def cmd_alert_flush(args: argparse.Namespace, paths: Paths) -> int:
    from . import alerts
    return alerts.cli_flush(args, paths)


COMMANDS = ("doctor", "init", "dream", "status", "search", "inspect", "pin", "unpin", "forget",
            "restore", "core-check", "core-restore", "core-proposal", "alert-flush", "calibrate",
            "export", "reembed", "migrate", "config", "debug")

HANDLERS: dict[str, Handler] = {
    "doctor": cmd_doctor, "init": cmd_init, "dream": cmd_dream, "status": cmd_status,
    "search": cmd_search, "inspect": cmd_inspect, "pin": cmd_pin, "unpin": cmd_unpin,
    "forget": cmd_forget, "restore": cmd_restore, "core-check": cmd_core_check,
    "core-restore": cmd_core_restore, "core-proposal": cmd_core_proposal,
    "alert-flush": cmd_alert_flush, "calibrate": cmd_calibrate, "export": cmd_export,
    "reembed": cmd_reembed, "migrate": cmd_migrate, "config": cmd_config, "debug": cmd_debug,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for k, v in (("json", False), ("hermes_home", None), ("verbose", False)):
        if not hasattr(args, k):
            setattr(args, k, v)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)
    if getattr(args, "now", None):
        try:
            clock.set_now(args.now)
        except ValueError as e:
            _err(f"--now 해석 실패: {e}")
            return EXIT_FAIL
    paths = Paths.from_env(args.hermes_home)
    try:
        return int(HANDLERS[args.command](args, paths))
    except (LiveHomeRefused, PlannedRunPending) as e:
        _err(str(e))
        return EXIT_FAIL
    except Exception as e:  # noqa: BLE001 — last resort: class + redacted message, no traceback with secrets
        from .threat import redact_secrets
        log.debug("unhandled", exc_info=True)
        _err(f"yume {args.command} 실패: {type(e).__name__}: {redact_secrets(str(e))[0][:300]}")
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
