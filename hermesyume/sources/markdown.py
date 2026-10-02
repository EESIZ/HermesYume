"""Workspace markdown episodes (PLAN-v2 §3.1(b); CONTRACTS §4.2).

Files are never moved or written. Progress per file = ``processed_bytes`` + ``prefix_sha256``
(sha256 of the processed prefix) in ledger ``md_files``: an append continues where the last run
stopped; a changed prefix reprocesses the whole file (upsert absorbs duplicates).

Message parsing (DEVIATIONS B1-M1 adapts the CONTRACTS decision to the real openclaw files):
- ``#``…``######`` header lines are not messages; the latest header is remembered.
- ``Session Key/ID/Source`` lines (plain or ``- **Session Key**:`` / ``- **Source**:``) are dropped.
- ``user:`` / ``assistant:`` (also ``**User:**``, ``- user:``) starts a user/assistant message that
  continues — across blank lines — until the next label or header. In session-memory files the
  real user text follows "(untrusted metadata)" JSON blocks after a blank line.
- any other blank-line-separated paragraph outside a labeled message is one ``agent_log``
  message, prefixed ``"[<latest header>] "`` when a header exists.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..clock import kst_date, now as clock_now, parse_iso
from ..types import Message, MdFileState

log = logging.getLogger("hermesyume.sources.markdown")

_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")
_HEADER_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
_LABEL_RE = re.compile(
    r"^[ \t]*(?:[-*][ \t]+)?(?:\*\*)?(user|assistant)(?:[ \t]*:[ \t]*\*\*|\*\*[ \t]*:|[ \t]*:)[ \t]?(.*)$",
    re.I)
_DROP_RE = re.compile(
    r"^[ \t]*(?:[-*][ \t]+)?(?:(?:\*\*)?Session[ \t]+(?:Key|ID|Source)(?:\*\*)?[ \t]*:"
    r"|\*\*Source(?:\*\*[ \t]*:|[ \t]*:\*\*))", re.I)
_REF_PREFIX = {"user": "U", "assistant": "A", "agent_log": "L"}


@dataclass
class MdSource:
    path: str               # absolute
    rel: str                # relative to its md_sources root
    date: str | None        # basename[:10] if valid 'YYYY-MM-DD'
    slug: str               # basename without date prefix/'-'/'.md' (stem if that leaves nothing)
    sha256: str             # whole file
    size: int
    start_offset: int       # bytes already processed (0 if new or prefix changed)
    end_offset: int         # end of last complete line; new processed_bytes candidate
    prefix_changed: bool
    base_ts: float          # date 12:00 KST, or file mtime if no date
    messages: list[Message]  # parsed from bytes [start_offset, end_offset) — RAW text
    context_before: list[Message] = field(default_factory=list)   # last exchange before start_offset (RAW)
    header: str | None = None       # latest header before start_offset
    mtime: float = 0.0
    _data: bytes = field(default=b"", repr=False, compare=False)


def md_key(path: str, line: int, role: str | None = None) -> str:
    """Evidence key of an md line: "m:<sha10>:<line>" for user/assistant lines, "l:<sha10>:<line>"
    for agent_log text (lets strength.compute_tier tell agent_log from assistant-only evidence)."""
    prefix = "l" if role == "agent_log" else "m"
    return f"{prefix}:{hashlib.sha1(str(path).encode('utf-8')).hexdigest()[:10]}:{int(line)}"


def _cfg(cfg: Any, key: str, default: Any) -> Any:
    try:
        return cfg[key]
    except (KeyError, TypeError):
        return getattr(cfg, key, default)


def _valid_date(s: str) -> bool:
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _date_slug(basename: str) -> tuple[str | None, str]:
    stem = basename[:-3] if basename.lower().endswith(".md") else basename
    m = _DATE_RE.match(stem)
    if m and _valid_date(m.group(1)):
        slug = stem[10:].lstrip("-_ ").strip()
        return m.group(1), slug or stem
    return None, stem


def _parse(data: bytes, *, path: str, start_offset: int, first_line: int, base_ts: float,
           header: str | None) -> tuple[list[Message], str | None]:
    out: list[Message] = []
    cur: dict | None = None
    latest = header

    def flush() -> None:
        nonlocal cur
        if cur is None:
            return
        text = "\n".join(cur["lines"]).strip()
        if text:
            role = cur["role"]
            if role == "agent_log" and cur["header"]:
                text = f"[{cur['header']}] {text}"
            line = cur["line"]
            out.append(Message(ref=f"{_REF_PREFIX[role]}#md:L{line}", key=md_key(path, line, role),
                               role=role, text=text, ts=base_ts, source="md",
                               session_id=f"md:{path}", msg_id=start_offset + cur["off"],
                               line=line, platform="md"))
        cur = None

    off = 0
    parts = data.split(b"\n")
    for i, chunk in enumerate(parts):
        line_no = first_line + i
        line_off = off
        off += len(chunk) + 1
        if i == len(parts) - 1 and not chunk:
            break                                  # nothing after the final newline
        line = chunk.decode("utf-8", errors="replace").rstrip("\r")
        hm = _HEADER_RE.match(line)
        if hm:
            flush()
            latest = hm.group(2).strip() or latest
            continue
        if _DROP_RE.match(line):
            continue
        lm = _LABEL_RE.match(line)
        if lm:
            flush()
            cur = {"role": lm.group(1).lower(), "lines": [lm.group(2)], "off": line_off,
                   "line": line_no, "header": latest}
            continue
        if not line.strip():
            if cur is not None and cur["role"] != "agent_log":
                cur["lines"].append("")
            else:
                flush()
            continue
        if cur is None:
            cur = {"role": "agent_log", "lines": [line], "off": line_off, "line": line_no,
                   "header": latest}
        else:
            cur["lines"].append(line)
    flush()
    return out, latest


def parse_md_messages(data: bytes, *, path: str, start_offset: int, first_line: int,
                      base_ts: float, header: str | None = None) -> list[Message]:
    """Messages of `data` = file bytes [start_offset, end). `first_line` = 1-based line number of
    data's first line; `header` = latest header before start_offset (agent_log prefix)."""
    return _parse(data, path=path, start_offset=start_offset, first_line=first_line,
                  base_ts=base_ts, header=header)[0]


def _iter_md_files(root: str) -> list[str]:
    if os.path.isfile(root):
        return [root] if root.lower().endswith(".md") else []
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))   # .git, .obsidian …
        for fn in sorted(filenames):
            if fn.lower().endswith(".md"):
                p = os.path.join(dirpath, fn)
                if os.path.isfile(p):
                    found.append(p)
    return found


def _last_exchange(msgs: list[Message]) -> list[Message]:
    from ..windows import split_exchanges
    ex = split_exchanges(msgs)
    return list(ex[-1]) if ex else []


def scan_md_sources(cfg: Any, ledger: Any, *, now: float | None = None
                    ) -> tuple[list[MdSource], dict[str, int]]:
    """Files with unprocessed complete lines, sorted by (date or mtime date, path), and
    exclusion counts. A missing trailing newline counts as a line end once the file has been
    untouched for `settle_minutes` (relative to `now`, default clock.now())."""
    excluded: dict[str, int] = {}
    exclude_globs = list(_cfg(cfg, "md_exclude_globs", []) or [])
    settle_s = float(_cfg(cfg, "settle_minutes", 30)) * 60.0
    ref_now = clock_now() if now is None else float(now)
    sources: list[MdSource] = []
    seen: set[str] = set()
    for root in _cfg(cfg, "md_sources", []) or []:
        root = os.path.abspath(os.path.expanduser(str(root)))
        if not os.path.exists(root):
            excluded["md_missing_root"] = excluded.get("md_missing_root", 0) + 1
            continue
        base_dir = root if os.path.isdir(root) else os.path.dirname(root)
        for p in _iter_md_files(root):
            path = os.path.abspath(p)
            if path in seen:
                continue
            seen.add(path)
            name = os.path.basename(path)
            if any(fnmatch.fnmatch(name, g) for g in exclude_globs):
                excluded["md_excluded"] = excluded.get("md_excluded", 0) + 1
                continue
            try:
                st = os.stat(path)
                with open(path, "rb") as f:
                    data = f.read()
            except OSError as e:
                log.warning("md 읽기 실패: %s (%s)", path, type(e).__name__)
                excluded["md_unreadable"] = excluded.get("md_unreadable", 0) + 1
                continue
            src = _build_source(path, os.path.relpath(path, base_dir), data, st.st_mtime,
                                ledger, now=ref_now, settle_s=settle_s)
            if src is not None:
                sources.append(src)
    sources.sort(key=lambda s: (s.date or kst_date(s.mtime), s.path))
    return sources, excluded


def _build_source(path: str, rel: str, data: bytes, mtime: float, ledger: Any, *, now: float,
                  settle_s: float) -> MdSource | None:
    size = len(data)
    if size == 0 or data.endswith(b"\n"):
        end = size
    else:
        end = data.rfind(b"\n") + 1
        if mtime <= now - settle_s:
            end = size
    state = ledger.get_md(path) if ledger is not None else None
    start, prefix_changed = 0, False
    if state is not None:
        pb = int(state.processed_bytes or 0)
        if 0 <= pb <= size and state.prefix_sha256 == hashlib.sha256(data[:pb]).hexdigest():
            start = pb
        else:
            prefix_changed = True
    if start >= end:
        return None
    date, slug = _date_slug(os.path.basename(path))
    base_ts = parse_iso(f"{date}T12:00") if date else float(mtime)
    prefix = data[:start]
    first_line = prefix.count(b"\n") + 1
    header: str | None = None
    context: list[Message] = []
    if start > 0:
        pre_msgs, header = _parse(prefix, path=path, start_offset=0, first_line=1,
                                  base_ts=base_ts, header=None)
        context = _last_exchange(pre_msgs)
    msgs, _ = _parse(data[start:end], path=path, start_offset=start, first_line=first_line,
                     base_ts=base_ts, header=header)
    if not msgs:
        return None
    return MdSource(path=path, rel=rel, date=date, slug=slug,
                    sha256=hashlib.sha256(data).hexdigest(), size=size, start_offset=start,
                    end_offset=end, prefix_changed=prefix_changed, base_ts=float(base_ts),
                    messages=msgs, context_before=context, header=header, mtime=float(mtime),
                    _data=data)


def file_state_after(src: MdSource, processed_bytes: int, run_id: str, *,
                     status: str | None = None) -> MdFileState:
    """Ledger state once windows up to `processed_bytes` are committed. status defaults to
    "ok" when everything up to end_offset is processed, else "partial"."""
    data = src._data
    if not data and processed_bytes:
        with open(src.path, "rb") as f:
            data = f.read()
        if hashlib.sha256(data).hexdigest() != src.sha256:
            raise ValueError(f"md file changed since scan: {src.path}")
    pb = max(0, min(int(processed_bytes), len(data) if data else 0))
    st = status or ("ok" if pb >= src.end_offset else "partial")
    return MdFileState(path=src.path, sha256=src.sha256, processed_bytes=pb,
                       prefix_sha256=hashlib.sha256(data[:pb]).hexdigest(), status=st,
                       run_id=run_id)
