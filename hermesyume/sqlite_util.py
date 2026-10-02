"""Read-only SQLite access that never creates files in someone else's directory.

Verified on sqlite 3.53.1: opening a WAL database with ``mode=ro`` when its ``-shm``/``-wal`` do
not exist CREATES them next to the database and leaves them behind. That breaks dry-run purity
(PLAN §10.5, G10). So:

- ``connect_ro(path)``: live runs — ``file:…?mode=ro`` + ``PRAGMA query_only=ON`` (PLAN §1.2).
- ``open_for_read(path, pure=True)``: dry-run — copy the db (+ ``-wal`` if present) into a private
  temp dir outside HERMES_HOME and open the copy; the temp dir is removed on close.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def connect_ro(path: str | os.PathLike, *, timeout: float = 30.0) -> sqlite3.Connection:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=timeout)
    conn.execute("PRAGMA query_only=ON")
    conn.row_factory = sqlite3.Row
    return conn


class SnapshotConnection:
    """A connection to a private copy; ``close()`` also deletes the copy."""

    def __init__(self, src: Path):
        self.src = src
        self.tmpdir = tempfile.mkdtemp(prefix="yume-ro-")
        os.chmod(self.tmpdir, 0o700)
        dst = Path(self.tmpdir) / src.name
        shutil.copyfile(src, dst)
        wal = src.with_name(src.name + "-wal")
        if wal.exists():
            try:
                shutil.copyfile(wal, dst.with_name(dst.name + "-wal"))
            except FileNotFoundError:
                pass  # checkpointed between exists() and copy
        self.conn = sqlite3.connect(str(dst), timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA query_only=ON")

    def close(self) -> None:
        try:
            self.conn.close()
        finally:
            shutil.rmtree(self.tmpdir, ignore_errors=True)


@contextmanager
def open_for_read(path: str | os.PathLike, *, pure: bool) -> Iterator[sqlite3.Connection]:
    """pure=True (dry-run) → snapshot copy; pure=False → connect_ro. Missing file → FileNotFoundError."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if pure:
        snap = SnapshotConnection(p)
        try:
            yield snap.conn
        finally:
            snap.close()
    else:
        conn = connect_ro(p)
        try:
            yield conn
        finally:
            conn.close()
