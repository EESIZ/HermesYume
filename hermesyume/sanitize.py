"""Input cleaning before extraction (PLAN-v2 §3.3; CONTRACTS §4.4).

Order inside ``sanitize_text``:
  1. injected blocks: <memory-context>…</memory-context>, <relevant-memories>…</relevant-memories>,
     [System note: …], "<label> (untrusted metadata):" + its JSON block, Session Key/ID/Source lines
  2. secrets → [REDACTED:<type>] (threat.redact_secrets)
  3. fenced code > code_max_lines → [코드 N줄 생략]; JSON literal > json_max_chars → [JSON 생략];
     base64/hex run > blob_min_chars → [blob]
  4. drop lines that repeat across ≥ N messages (stats over recent input) or match strip_line_regex
  5. > msg_max_chars → head + [중간 N자 생략] + tail (never cut silently)
The window-level "user chars < 20 → empty" check is done by nrem.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from .threat import redact_secrets
from .types import Message

log = logging.getLogger("hermesyume.sanitize")

BLOCK_KEYS = ("memory_context", "relevant_memories", "system_note", "untrusted_metadata",
              "session_lines")

_MEMCTX_RE = re.compile(r"<\s*memory-context\s*>[\s\S]*?(?:<\s*/\s*memory-context\s*>|\Z)", re.I)
_RELMEM_RE = re.compile(r"<\s*relevant-memories\s*>[\s\S]*?(?:<\s*/\s*relevant-memories\s*>|\Z)", re.I)
_STRAY_TAG_RE = re.compile(r"<\s*/?\s*(?:memory-context|relevant-memories)\s*>", re.I)
# one level of nested [...] allowed inside the note
_SYSNOTE_RE = re.compile(r"\[System note:[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\][ \t]*\n?", re.I)
# "Conversation info (untrusted metadata):", "Sender (untrusted metadata):", … at line start
_UNTRUSTED_LABEL_RE = re.compile(
    r"(?m)^[ \t]*[^\n(]{0,60}\(untrusted[^)\n]{0,40}\)[ \t]*:?[ \t]*", re.I)
# "Session Key: …" and the openclaw markdown forms "- **Session Key**: …" / "- **Source**: …"
# (a bare "Source: …" line is ordinary prose and is kept)
_SESSION_LINE_RE = re.compile(
    r"(?mi)^[ \t]*(?:[-*][ \t]+)?(?:"
    r"(?:\*\*)?Session[ \t]+(?:Key|ID|Source)(?:\*\*)?[ \t]*:(?:\*\*)?"
    r"|\*\*Source(?:\*\*[ \t]*:|[ \t]*:\*\*)"
    r").*(?:\n|\Z)")
_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")
_BLOB_CAND_RE = re.compile(r"[A-Za-z0-9+/=_-]+")
_HEX_RE = re.compile(r"[0-9a-fA-F]+")
_JSONISH_RE = re.compile(r"[\"']\s*:")
# md agent_log messages carry a "[<latest header>] " prefix (sources.markdown)
_MD_HEADER_PREFIX_RE = re.compile(r"^\[[^\]\n]{1,160}\] ")
_BLANKS_RE = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
_JSON_START_RE = re.compile(r"(?:^|[:=(])[ \t]*[\[{]", re.M)
_STRUCT_RE = re.compile(r'[\\"{}\[\]\n]')
_MAX_JSON_SCANS = 400
_REPORT_LINE_MAX = 200
_BRACKET_SCAN_LIMIT = 100_000
# Texts above this are pre-cut to head/tail halves before steps 1–4 (they end up truncated at
# step 5 anyway); the cut is included in the "[중간 N자 생략]" count.
_PRE_CAP = 60_000


@dataclass
class SanitizeReport:
    blocks_removed: dict[str, int] = field(default_factory=lambda: {k: 0 for k in BLOCK_KEYS})
    code_elided: int = 0
    json_elided: int = 0
    blobs: int = 0
    truncated: int = 0
    repeated_removed: dict[str, int] = field(default_factory=dict)   # line → removals
    strip_regex_removed: int = 0
    secrets: dict[str, int] = field(default_factory=dict)            # type → redactions
    dropped_tool: int = 0
    dropped_empty: int = 0
    regex_removed_lines: dict[str, int] = field(default_factory=dict)   # strip_line_regex hits by line

    def _block(self, key: str, n: int = 1) -> None:
        if n:
            self.blocks_removed[key] = self.blocks_removed.get(key, 0) + n

    def top_repeated(self, n: int = 10) -> list[dict]:
        """[{"line","count"}] for RunReport.strip_lines_top (repeat-line + strip-regex removals)."""
        merged: dict[str, int] = dict(self.repeated_removed)
        for line, c in self.regex_removed_lines.items():
            merged[line] = merged.get(line, 0) + c
        items = sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
        return [{"line": line, "count": c} for line, c in items]

    def merge(self, other: "SanitizeReport") -> None:
        for k, v in other.blocks_removed.items():
            self._block(k, v)
        self.code_elided += other.code_elided
        self.json_elided += other.json_elided
        self.blobs += other.blobs
        self.truncated += other.truncated
        self.strip_regex_removed += other.strip_regex_removed
        self.dropped_tool += other.dropped_tool
        self.dropped_empty += other.dropped_empty
        for src, dst in ((other.repeated_removed, self.repeated_removed),
                         (other.secrets, self.secrets),
                         (other.regex_removed_lines, self.regex_removed_lines)):
            for k, v in src.items():
                dst[k] = dst.get(k, 0) + v


def _cfg(cfg: Any, key: str, default: Any) -> Any:
    try:
        return cfg[key]
    except (KeyError, TypeError):
        return getattr(cfg, key, default)


def compile_strip_regex(cfg: Any) -> list[re.Pattern]:
    out: list[re.Pattern] = []
    for pat in _cfg(cfg, "strip_line_regex", []) or []:
        try:
            out.append(re.compile(pat))
        except re.error as e:
            log.warning("strip_line_regex 무시(잘못된 정규식): %r (%s)", pat, e)
    return out


def _report_key(line: str) -> str:
    return line if len(line) <= _REPORT_LINE_MAX else line[:_REPORT_LINE_MAX] + "…"


_GENERIC_WORDS = ("api", "token", "password", "secret", "비밀번호")


def may_contain_secret(text: str) -> bool:
    """Cheap necessary condition for threat.SECRET_PATTERNS (substring tests in C). False means
    redact_secrets would change nothing; True means run it. Python's re scans non-ASCII text
    slowly, and most messages contain no secret."""
    if not text:
        return False
    if ("sk-" in text or "ntn_" in text or "secret_" in text or "AKIA" in text
            or "eyJ" in text):
        return True
    if "gh" in text and ("ghp_" in text or "gho_" in text or "ghu_" in text or "ghs_" in text
                         or "ghr_" in text):
        return True
    colon = ":" in text
    if colon:                                          # telegram: 8–10 digits then ':'
        i = text.find(":")
        while i != -1:
            if i >= 8 and text[i - 8:i].isdigit():
                return True
            i = text.find(":", i + 1)
    if colon or "=" in text:                           # generic: label then ':' or '='
        folded = text.casefold().replace("ı", "i")      # re IGNORECASE also maps ı/İ/ſ/K
        if any(w in folded for w in _GENERIC_WORDS):
            return True
    return False


def redact(text: str) -> tuple[str, dict[str, int]]:
    """threat.redact_secrets, skipped when no secret pattern can match."""
    if not may_contain_secret(text):
        return text, {}
    return redact_secrets(text)


def build_repeat_lines(texts: Iterable[tuple[str, str]], *, min_chars: int = 20,
                       min_msgs: int = 5) -> set[str]:
    """Stripped lines of ≥ min_chars appearing in ≥ min_msgs distinct messages (by key).
    Lines are compared after secret redaction (the same form sanitize_text sees at step 4)."""
    seen_keys: set[str] = set()
    counts: dict[str, int] = {}
    for key, text in texts:
        if not text or key in seen_keys:
            continue
        seen_keys.add(key)
        lines = set()
        for raw in text.split("\n"):
            s = raw.strip()
            if len(s) >= min_chars:
                lines.add(redact(s)[0])
        for s in lines:
            counts[s] = counts.get(s, 0) + 1
    return {s for s, c in counts.items() if c >= min_msgs}


# ── step 1 ──────────────────────────────────────────────────────────────────

def _match_bracket(text: str, start: int, work: list[int] | None = None) -> int | None:
    """Index just past the bracket matching text[start] ('{' or '['); string-aware (a raw newline
    inside a string means "not JSON"). None if unbalanced within the scan limit. Jumps between
    structural characters with a regex; `work` (a one-item counter) accumulates the steps."""
    stack = ["}" if text[start] == "{" else "]"]
    end_lim = min(len(text), start + _BRACKET_SCAN_LIMIT)
    pos = start + 1
    in_str = False
    steps = 0
    try:
        while True:
            m = _STRUCT_RE.search(text, pos, end_lim)
            if m is None:
                return None
            steps += 1
            i = m.start()
            c = text[i]
            pos = i + 1
            if in_str:
                if c == "\\":
                    pos = i + 2
                elif c == '"':
                    in_str = False
                elif c == "\n":
                    return None
            elif c == '"':
                in_str = True
            elif c == "{" or c == "[":
                stack.append("}" if c == "{" else "]")
            elif c == "}" or c == "]":
                if c != stack[-1]:
                    return None
                stack.pop()
                if not stack:
                    return i + 1
    finally:
        if work is not None:
            work[0] += steps


def _remove_untrusted(text: str) -> tuple[str, int]:
    n = 0
    pos = 0
    out: list[str] = []
    while True:
        m = _UNTRUSTED_LABEL_RE.search(text, pos)
        if not m:
            out.append(text[pos:])
            break
        out.append(text[pos:m.start()])
        end = m.end()
        j = end
        while j < len(text) and text[j] in " \t\r\n":
            j += 1
        bracket_end = _match_bracket(text, j) if j < len(text) and text[j] in "{[" else None
        if text.startswith("```", j) or text.startswith("~~~", j):
            fence = text[j:j + 3]
            close = text.find("\n" + fence, j + 3)
            if close == -1:
                end = len(text)
            else:
                k = text.find("\n", close + 4)
                end = len(text) if k == -1 else k + 1
        elif bracket_end is not None:
            end = bracket_end
            if end < len(text) and text[end] == "\n":
                end += 1
        elif end < len(text) and text[end] == "\n":
            end += 1                       # bare label line
        n += 1
        pos = end
    return "".join(out), n


def strip_injected_blocks(text: str, report: SanitizeReport | None = None) -> str:
    """Step 1 only (also used defensively by windows.build_windows)."""
    if not text:
        return text or ""
    rep = report if report is not None else SanitizeReport()
    text, n = _MEMCTX_RE.subn("", text)
    rep._block("memory_context", n)
    text, n = _RELMEM_RE.subn("", text)
    rep._block("relevant_memories", n)
    text = _STRAY_TAG_RE.sub("", text)
    text, n = _SYSNOTE_RE.subn("", text)
    rep._block("system_note", n)
    if "ntrusted" in text or "NTRUSTED" in text:
        text, n = _remove_untrusted(text)
        rep._block("untrusted_metadata", n)
    if "ession" in text or "ource" in text:
        text, n = _SESSION_LINE_RE.subn("", text)
        rep._block("session_lines", n)
    return text


# ── step 3 ──────────────────────────────────────────────────────────────────

def _elide_code(text: str, max_lines: int, rep: SanitizeReport) -> str:
    if "```" not in text and "~~~" not in text:
        return text
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = _FENCE_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        fence = m.group(1)
        j = i + 1
        while j < len(lines) and not lines[j].strip().startswith(fence):
            j += 1
        body_n = j - i - 1
        closed = j < len(lines)
        if body_n > max_lines:
            out.append(f"[코드 {body_n}줄 생략]")
            rep.code_elided += 1
        else:
            out.extend(lines[i:j + 1] if closed else lines[i:j])
        i = j + 1 if closed else j
    return "\n".join(out)


def _looks_json(span: str) -> bool:
    try:
        json.loads(span)
        return True
    except (ValueError, TypeError):
        return len(_JSONISH_RE.findall(span)) >= 3


def _elide_json(text: str, max_chars: int, rep: SanitizeReport) -> str:
    """Replace JSON object/array literals longer than max_chars. Candidates are brackets that
    open a line or follow ':', '=' or '(' (where pasted JSON / tool output starts); the number of
    long scans per text is capped so prose with stray brackets stays linear."""
    if "{" not in text and "[" not in text:
        return text
    out: list[str] = []
    last = 0
    scans = 0
    work = [0]
    budget = 2 * len(text) + 10_000          # structural-char steps over all scans
    for m in _JSON_START_RE.finditer(text):
        i = m.end() - 1
        if i < last:
            continue
        if scans >= _MAX_JSON_SCANS or work[0] > budget:
            break
        scans += 1
        end = _match_bracket(text, i, work)
        if end is None or end - i <= max_chars:
            continue
        if _looks_json(text[i:end]):
            out.append(text[last:i])
            out.append("[JSON 생략]")
            rep.json_elided += 1
            last = end
    out.append(text[last:])
    return "".join(out)


def _is_blob(tok: str) -> bool:
    if _HEX_RE.fullmatch(tok):
        return True
    has_digit = any(ch.isdigit() for ch in tok)
    has_upper = any(ch.isupper() for ch in tok)
    has_lower = any(ch.islower() for ch in tok)
    return has_digit and has_upper and has_lower and tok.count("/") * 20 < len(tok)


def _elide_blobs(text: str, min_chars: int, rep: SanitizeReport) -> str:
    def _sub(m: re.Match) -> str:
        tok = m.group(0)
        if len(tok) > min_chars and _is_blob(tok.strip("=")):
            rep.blobs += 1
            return "[blob]"
        return tok
    return _BLOB_CAND_RE.sub(_sub, text)


# ── step 4 ──────────────────────────────────────────────────────────────────

def _drop_lines(text: str, repeat_lines: set[str], strip_res: list[re.Pattern],
                rep: SanitizeReport) -> str:
    if not repeat_lines and not strip_res:
        return text
    kept: list[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if s and s in repeat_lines:
            k = _report_key(s)
            rep.repeated_removed[k] = rep.repeated_removed.get(k, 0) + 1
            continue
        if s and any(p.search(s) for p in strip_res):
            rep.strip_regex_removed += 1
            k = _report_key(s)
            rep.regex_removed_lines[k] = rep.regex_removed_lines.get(k, 0) + 1
            continue
        kept.append(line)
    return "\n".join(kept)


# ── public ──────────────────────────────────────────────────────────────────

def sanitize_text(text: str, *, cfg: Any, repeat_lines: set[str], strip_res: list[re.Pattern],
                  report: SanitizeReport) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n")
    pre_cut, sentinel = 0, ""
    if len(text) > _PRE_CAP:
        half = _PRE_CAP // 2
        pre_cut = len(text) - 2 * half
        sentinel = f"[중간 {pre_cut}자 생략]"
        text = f"{text[:half]}\n{sentinel}\n{text[len(text) - half:]}"
    text = strip_injected_blocks(text, report)
    text, counts = redact(text)
    for k, v in counts.items():
        report.secrets[k] = report.secrets.get(k, 0) + v
    text = _elide_code(text, int(_cfg(cfg, "code_max_lines", 15)), report)
    text = _elide_json(text, int(_cfg(cfg, "json_max_chars", 500)), report)
    text = _elide_blobs(text, int(_cfg(cfg, "blob_min_chars", 64)), report)
    text = _drop_lines(text, repeat_lines, strip_res, report)
    text = _BLANKS_RE.sub("\n\n", text).strip()
    max_chars = int(_cfg(cfg, "msg_max_chars", 3000))
    head_n = int(_cfg(cfg, "msg_head_chars", 2000))
    tail_n = int(_cfg(cfg, "msg_tail_chars", 500))
    if len(text) > max_chars and len(text) > head_n + tail_n:
        middle = text[head_n:len(text) - tail_n]
        removed = len(middle)
        if sentinel and f"\n{sentinel}\n" in middle:
            removed += pre_cut - len(sentinel) - 2
        elif sentinel and sentinel in middle:
            removed += pre_cut - len(sentinel)
        text = f"{text[:head_n]}\n[중간 {removed}자 생략]\n{text[len(text) - tail_n:]}"
        report.truncated += 1
    elif pre_cut:
        report.truncated += 1
    return text


def sanitize_messages(msgs: list[Message], *, cfg: Any, repeat_lines: set[str],
                      report: SanitizeReport) -> list[Message]:
    """New Message objects (same ref/key/ts). Drops role 'tool' and messages empty after
    cleaning. md agent_log messages keep their "[<header>] " prefix, which is excluded from
    line matching and dropped together with an emptied body."""
    strip_res = compile_strip_regex(cfg)
    out: list[Message] = []
    for m in msgs:
        if m.role == "tool":
            report.dropped_tool += 1
            continue
        text = m.text or ""
        prefix = ""
        if m.role == "agent_log":
            pm = _MD_HEADER_PREFIX_RE.match(text)
            if pm:
                prefix, text = pm.group(0), text[pm.end():]
        clean = sanitize_text(text, cfg=cfg, repeat_lines=repeat_lines, strip_res=strip_res,
                              report=report)
        if not clean.strip():
            report.dropped_empty += 1
            continue
        out.append(replace(m, text=prefix + clean if prefix else clean))
    return out
