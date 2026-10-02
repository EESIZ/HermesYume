"""Data-directory backups (PLAN-v2 §10.4, §5.4).

- ``weekly_tar``: one ``backups/data-<stamp>.tar.gz`` per 7 days (real time), newest
  ``weekly_tar_keep`` kept. The serving copy, backups/, staging/ and old Lance directories
  (``lancedb.prev-*`` / ``lancedb.reembed-*``) are left out; ledger.db and live.db go in through the
  sqlite backup API (consistent copies while providers keep appending to live.db).
- ``snapshot_lancedb``: ``backups/lancedb-prepurge-<run>.tar.gz`` taken right before a quarantine
  purge runs ``cleanup_older_than=0`` (§5.4 "백업을 새로 만든 뒤"), newest
  ``prepurge_backups_keep`` kept. A wrong secret match can be undone from it by hand.

Archives are 0600 in a 0700 directory. Nothing here runs in --dry-run.
"""

from __future__ import annotations

import os
import sqlite3
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

WEEK = 7 * 86400.0
_SKIP_TOP = {"serving", "backups", "staging", "dream.lock"}
_SQLITE = ("ledger.db", "live.db")


def _cfg(cfg: Any, key: str, default: Any) -> Any:
    try:
        return cfg[key]
    except (KeyError, TypeError):
        return getattr(cfg, key, default)


def _prune(d: Path, pattern: str, keep: int) -> None:
    files = sorted(d.glob(pattern), key=lambda p: (p.stat().st_mtime, p.name))
    for old in files[:-keep] if keep > 0 else files:
        old.unlink(missing_ok=True)


def _sqlite_copy(src: Path, tmpdir: Path) -> Path | None:
    if not src.exists():
        return None
    dst = tmpdir / src.name
    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=30)
    try:
        out = sqlite3.connect(str(dst))
        try:
            con.backup(out)
        finally:
            out.close()
    finally:
        con.close()
    return dst


def _write_tar(dest: Path, add) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(dest.parent, 0o700)
    fd, tmp = tempfile.mkstemp(prefix=".tar_", suffix=".tmp", dir=str(dest.parent))
    os.close(fd)
    try:
        os.chmod(tmp, 0o600)
        with tarfile.open(tmp, "w:gz") as tf:
            add(tf)
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return dest


def weekly_tar(paths: Any, cfg: Any, *, now: float | None = None, force: bool = False) -> Path | None:
    """§10.4 weekly tar of the data directory. Returns the new archive, or None when the newest
    one is younger than 7 days."""
    keep = int(_cfg(cfg, "weekly_tar_keep", 4))
    if keep <= 0:
        return None
    now = time.time() if now is None else float(now)
    bdir = paths.backups_dir
    existing = sorted(bdir.glob("data-*.tar.gz"), key=lambda p: p.stat().st_mtime) if bdir.is_dir() else []
    if existing and not force and now - existing[-1].stat().st_mtime < WEEK:
        return None
    data = paths.data_dir
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    dest = bdir / f"data-{stamp}.tar.gz"

    def add(tf: tarfile.TarFile) -> None:
        with tempfile.TemporaryDirectory(prefix="yume-tar-") as td:
            for name in _SQLITE:
                cp = _sqlite_copy(data / name, Path(td))
                if cp is not None:
                    tf.add(str(cp), arcname=f"hermesyume/{name}")
            for child in sorted(data.iterdir()):
                n = child.name
                if (n in _SKIP_TOP or n.startswith(("ledger.db", "live.db", "lancedb.prev-", "lancedb.reembed-"))
                        or n.startswith(".")):
                    continue
                tf.add(str(child), arcname=f"hermesyume/{n}")

    _write_tar(dest, add)
    _prune(bdir, "data-*.tar.gz", keep)
    return dest


def snapshot_lancedb(paths: Any, run_id: str, *, keep: int) -> Path:
    """Tar of the whole Lance directory (all versions) before a cleanup_older_than=0."""
    dest = paths.backups_dir / f"lancedb-prepurge-{run_id}.tar.gz"
    src = paths.lancedb_dir

    def add(tf: tarfile.TarFile) -> None:
        tf.add(str(src), arcname="lancedb")

    _write_tar(dest, add)
    _prune(paths.backups_dir, "lancedb-prepurge-*.tar.gz", max(1, int(keep)))
    return dest
