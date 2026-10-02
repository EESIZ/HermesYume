"""Read-only access to ``serving/recall.sqlite`` (PLAN-v2 §2.5, §6.2; stdlib only).

One ``ServingIndex`` per snapshot file, shared by every provider instance in the process (module
singleton). A new file is detected by ``st_ino`` (plus mtime/size) and reloaded; a missing or
corrupt file yields ``None`` (callers fall back to inbox-only / no recall).

Two-stage search only for prefix-truncatable models (OpenAI text-embedding-3*, ``hash_embed.two_stage``):
active rows' ``vec256`` are held as Python lists; stage 1 ranks them by 256-d dot product, stage 2
re-ranks the top ``k1`` with the full vectors read by id. Every other model (e.g. the local
``hash/ngram-v1@1024``) is searched single-stage: the full ``vec`` of each active row is held in
memory (sparse index/value arrays for hash vectors, ~140 non-zeros of 1024; dense float32 arrays
otherwise) and stage 1 is already the exact cosine; the same stage 2 then reads the winners' full
vectors. Inactive rows are read only when ``search_all(include_inactive=True)`` asks for them.
"""

import heapq
import json
import logging
import math
import os
import threading
import time
from array import array
from dataclasses import dataclass, field, replace
from itertools import repeat
from operator import mul

from . import config as _cfg
from . import hash_embed
from . import serving_schema
from . import textutil

log = logging.getLogger("hermesyume.provider.serving")

_ITEM_META_COLS = ("id", "text", "subject", "kind", "tier", "status", "pinned", "core_sha",
                   "event_time", "valid_until", "strength", "refs")
_LIKE_WEIGHT = 0.5


@dataclass
class Item:
    id: str
    text: str
    subject: str = ""
    kind: str = "fact"
    tier: str = "decaying"
    status: str = "active"
    pinned: bool = False
    core_sha: str = None
    event_time: float = None
    valid_until: float = None
    strength: float = 0.0
    refs: list = field(default_factory=list)


@dataclass
class Pin:
    id: str
    text: str
    label: str = None
    core_target: str = None


def _parse_refs(raw):
    if not raw:
        return []
    try:
        v = json.loads(raw)
        return [str(x) for x in v] if isinstance(v, list) else []
    except Exception:
        return []


def _item_from_row(r):
    return Item(id=r[0], text=r[1] or "", subject=r[2] or "", kind=r[3] or "fact",
                tier=r[4] or "decaying", status=r[5] or "active", pinned=bool(r[6]),
                core_sha=r[7], event_time=r[8], valid_until=r[9],
                strength=float(r[10] or 0.0), refs=_parse_refs(r[11]))


def _normalize(v):
    n = math.sqrt(sum(map(mul, v, v)))
    if n <= 0 or abs(n - 1.0) < 1e-5:
        return v
    return [x / n for x in v]


def _escape_like(t):
    return t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class ServingIndex:
    def __init__(self, path, stat_key, conn, meta):
        self.path = str(path)
        self.stat_key = stat_key
        self.ino = stat_key[0]
        self.meta = meta
        self.embed_model = meta.get("embed_model") or ""
        self.dim = int(meta.get("dim") or 0)
        self.run_id = meta.get("run_id") or ""
        self.two_stage = hash_embed.two_stage(self.embed_model)
        self.sparse = (not self.two_stage) and hash_embed.is_model_id(self.embed_model)
        self.items = {}          # active rows only
        self.active_ids = []
        self.vec256 = []         # stage-1 vectors: 256-d prefix (two-stage) or the full vector
        self.pins = []
        self._pos = {}           # id -> index in active_ids
        self._vu = []            # (index, valid_until) for active rows that expire
        self._conn = conn
        self._lock = threading.Lock()

    # ── load ──
    @classmethod
    def load(cls, path):
        st = os.stat(path)
        key = (st.st_ino, st.st_mtime_ns, st.st_size)
        conn = serving_schema.open_readonly(path)
        try:
            meta = serving_schema.read_meta(conn)
            idx = cls(path, key, conn, meta)
            if not idx.embed_model or idx.dim <= 0:
                raise ValueError("serving meta incomplete")
            want = idx._stage1_dim()
            cols = ", ".join(_ITEM_META_COLS)
            col = "vec256" if idx.two_stage else "vec"
            for r in conn.execute("SELECT %s, %s FROM items WHERE status='active' ORDER BY id" % (cols, col)):
                v = serving_schema.blob_to_vec(r[12])
                if not v or len(v) != want:
                    continue
                it = _item_from_row(r)
                idx.items[it.id] = it
                idx._pos[it.id] = len(idx.active_ids)
                if it.valid_until is not None:
                    idx._vu.append((len(idx.active_ids), float(it.valid_until)))
                idx.active_ids.append(it.id)
                idx.vec256.append(idx._pack(v))
            for r in conn.execute("SELECT id, text, label, core_target FROM pins ORDER BY rowid"):
                idx.pins.append(Pin(id=r[0], text=r[1] or "", label=r[2], core_target=r[3]))
            return idx
        except Exception:
            conn.close()
            raise

    # ── stage-1 representation ──
    def _stage1_dim(self):
        return min(serving_schema.VEC256_DIM, self.dim) if self.two_stage else self.dim

    def _pack(self, v):
        if self.two_stage:
            return v
        if self.sparse:
            ix = array("H", [i for i, x in enumerate(v) if x])
            return (ix, array("f", [v[i] for i in ix]))
        return array("f", v)

    def _stage1_query(self, qvec):
        sub = self._stage1_dim()
        return _normalize(list(qvec[:sub]))

    def _scores(self, q, vecs):
        """Dot products of q with packed stage-1 vectors (C-level map/sum, no per-row frames)."""
        if self.sparse:
            qg = q.__getitem__
            return [sum(map(mul, map(qg, ix), w)) for ix, w in vecs]
        return list(map(sum, map(map, repeat(mul), repeat(q), vecs)))

    # ── reads ──
    def _query(self, sql, params=()):
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def full_vectors(self, ids):
        out = {}
        ids = list(ids)
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            q = "SELECT id, vec FROM items WHERE id IN (%s)" % ",".join("?" * len(chunk))
            for rid, blob in self._query(q, chunk):
                v = serving_schema.blob_to_vec(blob)
                if v and len(v) == self.dim:
                    out[rid] = _normalize(v)
        return out

    def get_items(self, ids):
        """Item metadata for any status (inactive rows are read on demand)."""
        out = {}
        missing = []
        for i in ids:
            if i in self.items:
                out[i] = self.items[i]
            else:
                missing.append(i)
        for i in range(0, len(missing), 500):
            chunk = missing[i:i + 500]
            q = "SELECT %s FROM items WHERE id IN (%s)" % (", ".join(_ITEM_META_COLS),
                                                          ",".join("?" * len(chunk)))
            for r in self._query(q, chunk):
                out[r[0]] = _item_from_row(r)
        return out

    def ids_with_core_sha(self, shas):
        if not shas:
            return set()
        return {i for i, it in self.items.items() if it.core_sha and it.core_sha in shas}

    @staticmethod
    def _stage1(q256, pairs, k1):
        """pairs: list of (id, stage-1 vector as a list) → [(dot, id)] top k1."""
        scored = [(sum(map(mul, q256, v)), i) for i, v in pairs]
        return heapq.nlargest(int(k1), scored)

    def vector_search_vecs(self, qvec, *, exclude=frozenset(), now=None, k1=64):
        """[(id, cos, vec1536)] best first over active rows (exclude ids, valid_until < now)."""
        if not qvec or not self.active_ids or len(qvec) != self.dim:
            return []
        q1 = self._stage1_query(qvec)
        skip = {self._pos[i] for i in exclude if i in self._pos}
        if now is not None:
            skip.update(j for j, vu in self._vu if vu < now)
        # stage 1: 256-d prefix dot (two-stage) or exact full dot (single-stage) over active rows
        scores = self._scores(q1, self.vec256)
        want = int(k1)
        order = heapq.nlargest(want + len(skip), range(len(scores)), key=scores.__getitem__)
        top = [self.active_ids[j] for j in order if j not in skip][:want]
        # stage 2: exact cosine with the full vectors
        vecs = self.full_vectors(top)
        out = [(i, sum(map(mul, qvec, vecs[i])), vecs[i]) for i in top if i in vecs]
        out.sort(key=lambda t: -t[1])
        return out

    def vector_search(self, qvec, *, exclude=frozenset(), now=None, k1=64):
        return [(i, c) for i, c, _ in self.vector_search_vecs(qvec, exclude=exclude, now=now, k1=k1)]

    def fts_search(self, query, *, exclude=frozenset(), limit=3, statuses=("active",), now=None):
        """[(id, rank)] by bm25 relevance × (0.5 + strength). MATCH for ≥3-char tokens, LIKE for
        2-char tokens."""
        match, like = textutil.search_tokens(query)
        rel = {}
        if match:
            expr = " OR ".join('"%s"' % t.replace('"', '""') for t in match)
            try:
                rows = self._query("SELECT id, bm25(items_fts) FROM items_fts WHERE items_fts MATCH ? "
                                   "ORDER BY 2 LIMIT 200", (expr,))
            except Exception:
                rows = []
            best = max([-r for _, r in rows] + [0.0])
            for rid, r in rows:
                v = (-r / best) if best > 0 else 1.0
                rel[rid] = rel.get(rid, 0.0) + max(v, 0.1)
        st = list(statuses)
        for t in like:
            pat = "%" + _escape_like(t) + "%"
            q = ("SELECT id FROM items WHERE status IN (%s) AND (text LIKE ? ESCAPE '\\' "
                 "OR subject LIKE ? ESCAPE '\\') LIMIT 200" % ",".join("?" * len(st)))
            for (rid,) in self._query(q, st + [pat, pat]):
                rel[rid] = rel.get(rid, 0.0) + _LIKE_WEIGHT
        if not rel:
            return []
        metas = self.get_items([i for i in rel if i not in exclude])
        out = []
        for rid, it in metas.items():
            if it.status not in statuses:
                continue
            if now is not None and it.status == "active" and it.valid_until is not None \
                    and it.valid_until < now:
                continue
            out.append((rid, rel[rid] * (0.5 + float(it.strength or 0.0))))
        out.sort(key=lambda t: (-t[1], t[0]))
        return out[: int(limit)]

    def search_all(self, qvec, query, *, include_inactive, limit, min_cos, now=None,
                   exclude=frozenset(), k1=64):
        """yume_search: [(Item, score, mode)]. Vector (cos ≥ min_cos) when qvec, else keyword.
        Active rows past valid_until are reported as status 'expired' (only with include_inactive)."""
        now = _cfg.now() if now is None else now
        results = []
        if qvec and len(qvec) != self.dim:
            qvec = None
        if qvec:
            k = max(int(k1), int(limit))
            act = self.vector_search_vecs(qvec, exclude=exclude, now=None, k1=k)
            for i, c, _ in act:
                results.append((self.items[i], c))
            if include_inactive:
                col = "vec256" if self.two_stage else "vec"
                rows = self._query("SELECT id, %s FROM items WHERE status != 'active'" % col)
                sub = self._stage1_dim()
                q256 = self._stage1_query(qvec)
                pairs = []
                for rid, blob in rows:
                    if rid in exclude:
                        continue
                    v = serving_schema.blob_to_vec(blob)
                    if v and len(v) == sub:
                        pairs.append((rid, v))
                top = self._stage1(q256, pairs, k)
                vecs = self.full_vectors([i for _, i in top])
                metas = self.get_items(list(vecs))
                for i in vecs:
                    if i in metas:
                        results.append((metas[i], sum(map(mul, qvec, vecs[i]))))
            out = []
            for it, c in results:
                if c < float(min_cos):
                    continue
                if it.status == "active" and it.valid_until is not None and it.valid_until < now:
                    if not include_inactive:
                        continue
                    it = replace(it, status="expired")
                out.append((it, c, "vector"))
            out.sort(key=lambda t: (-t[1], t[0].id))
            return out[: int(limit)]
        statuses = serving_schema.SERVING_STATUSES if include_inactive else ("active",)
        hits = self.fts_search(query, exclude=exclude, limit=limit, statuses=statuses,
                               now=None if include_inactive else now)
        metas = self.get_items([i for i, _ in hits])
        out = []
        for i, s in hits:
            it = metas.get(i)
            if it is None:
                continue
            if it.status == "active" and it.valid_until is not None and it.valid_until < now:
                it = replace(it, status="expired")
            out.append((it, s, "keyword"))
        return out


# ── module singleton ──

_LOCK = threading.Lock()
_INDEX = {}          # path -> (stat_key, ServingIndex | None)
_last_warn = {}


def serving_path(hermes_home):
    return os.path.join(_cfg.data_dir(hermes_home), "serving", "recall.sqlite")


def get_index(hermes_home):
    """Current ServingIndex for this home, reloading when the file changed. None when the file is
    missing or unreadable (never raises)."""
    try:
        path = serving_path(hermes_home)
        try:
            st = os.stat(path)
        except OSError:
            with _LOCK:
                _INDEX.pop(path, None)
            return None
        key = (st.st_ino, st.st_mtime_ns, st.st_size)
        with _LOCK:
            hit = _INDEX.get(path)
            if hit is not None and hit[0] == key:
                return hit[1]
            try:
                idx = ServingIndex.load(path)
            except Exception as e:
                idx = None
                now = time.monotonic()
                if now - _last_warn.get(path, -1e9) > 60:
                    _last_warn[path] = now
                    log.warning("hermesyume 서빙 사본을 열 수 없음(%s) — 회상 생략", type(e).__name__)
            _INDEX[path] = (key, idx)
            return idx
    except Exception:
        return None


def clear():
    with _LOCK:
        _INDEX.clear()
