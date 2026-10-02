"""HERMES_HOME resolution, data-directory layout (PLAN-v2 §1.1), live-home guard, dream.lock,
and the by-path loader for stdlib-only provider modules (``provider/_yume/*.py``)."""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator, Mapping

DATA_DIRNAME = "hermesyume"
PROVIDER_NAME = "hermesyume"

# Agent homes that `debug` and other destructive helpers refuse to touch (PLAN §11.1 #3): Hermes'
# default home, every home listed in $HERMESYUME_PROTECT_HOMES (os.pathsep-separated;
# $HERMESYUME_LIVE_HOMES is the older name and is still read) or in the `protect_homes` key of the
# active config.json. No machine-specific paths are built in; a home a gateway has run in is always
# live anyway (LIVE_MARKERS).
DEFAULT_LIVE_HOMES = ("~/.hermes",)
PROTECT_HOMES_ENV = ("HERMESYUME_PROTECT_HOMES", "HERMESYUME_LIVE_HOMES")
# Files only a home that a gateway has actually run in contains.
LIVE_MARKERS = ("gateway_state.json", "gateway.pid", "gateway.lock")
# Workspaces a sandbox HERMES_HOME must never write into (N8 docs/yume): $HERMESYUME_PROTECT_WORKSPACES
# ($HERMESYUME_LIVE_WORKSPACES is the older name), the `protect_workspaces` key of the active
# config.json, and the `workspace_dir` that each protected home's own config.json names.
DEFAULT_LIVE_WORKSPACES: tuple[str, ...] = ()
PROTECT_WORKSPACES_ENV = ("HERMESYUME_PROTECT_WORKSPACES", "HERMESYUME_LIVE_WORKSPACES")


class LiveHomeRefused(RuntimeError):
    """A debug/destructive helper was pointed at a live HERMES_HOME."""


class AlreadyRunning(RuntimeError):
    """dream.lock is held by another process."""


def resolve_hermes_home(explicit: str | os.PathLike | None = None,
                        env: Mapping[str, str] | None = None) -> Path:
    """explicit > $HERMES_HOME > ~/.hermes (same fallback as Hermes). Not resolved through symlinks."""
    env = os.environ if env is None else env
    raw = str(explicit) if explicit else (env.get("HERMES_HOME", "").strip() or "~/.hermes")
    return Path(os.path.expanduser(os.path.expandvars(raw))).absolute()


def _env_paths(env: Mapping[str, str], names: tuple[str, ...]) -> list[str]:
    return [p for n in names for p in env.get(n, "").split(os.pathsep) if p.strip()]


def _config_paths(home: str | os.PathLike, key: str) -> list[str]:
    """Path list (or single path) under `key` in <home>/hermesyume/config.json — read-only and
    best effort: a missing or unreadable file means nothing."""
    try:
        data = json.loads((Path(home) / DATA_DIRNAME / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    v = data.get(key) if isinstance(data, dict) else None
    if isinstance(v, str):
        v = [v]
    return [p for p in v if isinstance(p, str) and p.strip()] if isinstance(v, list) else []


def _norm(p: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(p)))).resolve()


def _unique(paths: list[Path]) -> list[Path]:
    return list(dict.fromkeys(paths))


def live_homes(env: Mapping[str, str] | None = None) -> list[Path]:
    """~/.hermes + $HERMESYUME_PROTECT_HOMES + `protect_homes` of the active config.json."""
    env = os.environ if env is None else env
    active = resolve_hermes_home(env=env)
    raw = (*DEFAULT_LIVE_HOMES, *_env_paths(env, PROTECT_HOMES_ENV), *_config_paths(active, "protect_homes"))
    return _unique([_norm(p) for p in raw])


def is_live_home(home: str | os.PathLike, env: Mapping[str, str] | None = None) -> bool:
    """True if `home` is a protected path (see live_homes; a home may also list itself in its own
    `protect_homes`) or looks like one a gateway ran in."""
    p = Path(home).resolve()
    if p in live_homes(env) or any((p / m).exists() for m in LIVE_MARKERS):
        return True
    return any(p == _norm(x) for x in _config_paths(p, "protect_homes"))


def live_workspaces(env: Mapping[str, str] | None = None) -> list[Path]:
    env = os.environ if env is None else env
    active = resolve_hermes_home(env=env)
    raw = [*DEFAULT_LIVE_WORKSPACES, *_env_paths(env, PROTECT_WORKSPACES_ENV),
           *_config_paths(active, "protect_workspaces")]
    for h in live_homes(env):
        raw += _config_paths(h, "workspace_dir")
    return _unique([_norm(p) for p in raw])


def in_live_workspace(path: str | os.PathLike, env: Mapping[str, str] | None = None) -> bool:
    """True if `path` is a live workspace or lies inside one (symlinks resolved)."""
    p = Path(os.path.expanduser(str(path))).resolve()
    return any(p == w or w in p.parents for w in live_workspaces(env))


def workspace_write_refusal(hermes_home: str | os.PathLike, workspace_dir: str | os.PathLike | None,
                            env: Mapping[str, str] | None = None) -> str | None:
    """Why dream must not write under `workspace_dir` (None = allowed): unset, or a sandbox
    HERMES_HOME pointing at a protected (live) agent's workspace."""
    if not str(workspace_dir or "").strip():
        return "workspace_dir 미설정"
    if not is_live_home(hermes_home, env) and in_live_workspace(workspace_dir, env):
        return "샌드박스 HERMES_HOME이 라이브 워크스페이스를 가리킴"
    return None


def refuse_if_live(home: str | os.PathLike, what: str) -> None:
    if is_live_home(home):
        raise LiveHomeRefused(f"{what}: 라이브 HERMES_HOME({home})에서는 실행할 수 없습니다.")


@dataclass(frozen=True)
class Paths:
    """Every on-disk location HermesYume touches, derived from HERMES_HOME only."""

    hermes_home: Path

    @classmethod
    def from_env(cls, hermes_home: str | os.PathLike | None = None,
                 env: Mapping[str, str] | None = None) -> "Paths":
        return cls(resolve_hermes_home(hermes_home, env))

    # ── Hermes-owned (read-only for us, except core_check.restore on core files) ──
    @property
    def state_db(self) -> Path: return self.hermes_home / "state.db"
    @property
    def memories_dir(self) -> Path: return self.hermes_home / "memories"
    @property
    def memory_md(self) -> Path: return self.memories_dir / "MEMORY.md"
    @property
    def user_md(self) -> Path: return self.memories_dir / "USER.md"
    @property
    def hermes_config_yaml(self) -> Path: return self.hermes_home / "config.yaml"
    @property
    def hermes_env(self) -> Path: return self.hermes_home / ".env"
    @property
    def skills_dir(self) -> Path: return self.hermes_home / "skills"
    @property
    def plugins_dir(self) -> Path: return self.hermes_home / "plugins"
    @property
    def provider_dir(self) -> Path: return self.plugins_dir / PROVIDER_NAME

    def core_file(self, target: str) -> Path:
        """target: "memory" → MEMORY.md, "user" → USER.md (Hermes memory tool targets)."""
        return {"memory": self.memory_md, "user": self.user_md}[target]

    # ── HermesYume data dir C4 (0700 dirs, 0600 files) ──
    @property
    def data_dir(self) -> Path: return self.hermes_home / DATA_DIRNAME
    @property
    def config_json(self) -> Path: return self.data_dir / "config.json"
    @property
    def lancedb_dir(self) -> Path: return self.data_dir / "lancedb"
    @property
    def ledger_db(self) -> Path: return self.data_dir / "ledger.db"
    @property
    def live_db(self) -> Path: return self.data_dir / "live.db"
    @property
    def serving_dir(self) -> Path: return self.data_dir / "serving"
    @property
    def recall_sqlite(self) -> Path: return self.serving_dir / "recall.sqlite"
    @property
    def runs_dir(self) -> Path: return self.data_dir / "runs"
    @property
    def dream_log_dir(self) -> Path: return self.data_dir / "dream-log"
    @property
    def backups_dir(self) -> Path: return self.data_dir / "backups"
    @property
    def alerts_log(self) -> Path: return self.data_dir / "alerts.log"
    @property
    def dream_lock(self) -> Path: return self.data_dir / "dream.lock"
    @property
    def proposals_dir(self) -> Path: return self.data_dir / "proposals"
    @property
    def migration_dir(self) -> Path: return self.data_dir / "migration"
    @property
    def staging_dir(self) -> Path: return self.data_dir / "staging"

    def serving_tmp(self, run_id: str) -> Path:
        return self.serving_dir / f"recall.{run_id}.sqlite"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def plan_json(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "plan.json"

    def ensure_data_dirs(self) -> None:
        """Create the data dir tree with 0700. Never called in --dry-run except for runs/ and dream-log/."""
        for d in (self.data_dir, self.lancedb_dir, self.serving_dir, self.runs_dir,
                  self.dream_log_dir, self.backups_dir, self.proposals_dir, self.migration_dir):
            d.mkdir(parents=True, exist_ok=True)
            os.chmod(d, 0o700)

    def ensure_dir(self, d: Path) -> Path:
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
        return d


# ── dream.lock (non-blocking flock; live/migrate runs only, never in --dry-run) ──

@contextmanager
def dream_lock(paths: Paths) -> Iterator[Path]:
    paths.ensure_dir(paths.data_dir)
    fd = os.open(paths.dream_lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise AlreadyRunning("already running") from e
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
            yield paths.dream_lock
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


# ── stdlib-only provider modules loaded by file path (single source of truth, no copies) ──

_MOD_LOCK = threading.Lock()
_MOD_CACHE: dict[str, ModuleType] = {}


def provider_source_dirs(paths: Paths | None = None,
                         env: Mapping[str, str] | None = None) -> list[Path]:
    """Candidate provider roots, in priority order."""
    env = os.environ if env is None else env
    here = Path(__file__).resolve().parent
    cands: list[Path] = []
    if env.get("HERMESYUME_PROVIDER_DIR"):
        cands.append(Path(env["HERMESYUME_PROVIDER_DIR"]).expanduser())
    cands.append(here.parent / "provider")          # repo checkout / editable install
    cands.append(here / "_provider")                # wheel install (package-dir mapping)
    if paths is not None:
        cands.append(paths.provider_dir)            # installed provider in HERMES_HOME
    return cands


def find_provider_file(name: str, paths: Paths | None = None) -> Path:
    for root in provider_source_dirs(paths):
        f = root / "_yume" / f"{name}.py"
        if f.is_file():
            return f
    raise FileNotFoundError(f"provider/_yume/{name}.py not found in {provider_source_dirs(paths)}")


def load_provider_module(name: str, paths: Paths | None = None) -> ModuleType:
    """Exec ``provider/_yume/<name>.py`` standalone (no package context, no sys.path change).

    Such modules must be stdlib-only and must not use relative imports.
    """
    f = find_provider_file(name, paths)
    key = str(f)
    with _MOD_LOCK:
        mod = _MOD_CACHE.get(key)
        if mod is not None:
            return mod
        mod_name = f"hermesyume._bypath_{name}"
        spec = importlib.util.spec_from_file_location(mod_name, f)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {f}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod  # dataclasses etc. need the module registered
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(mod_name, None)
            raise
        _MOD_CACHE[key] = mod
        return mod
