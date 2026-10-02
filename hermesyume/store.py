"""LanceDB canonical store (PLAN-v2 §2.1, §2.2).

Tables: ``memories`` (vector fixed_size_list<float32>[dim]), ``memory_history`` (no vector),
``suppress`` (vector + text_sha, never the text). Single writer under dream.lock; the gateway never
opens Lance.

Commit protocol (R8 steps 2–4): history.merge_insert("history_id") → suppress.merge_insert("id")
→ memories.merge_insert("id") ONCE → delete purged ids. Each step is idempotent; empty steps are
skipped so an unchanged re-run leaves every table version unchanged.
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import lancedb
import numpy as np
import pyarrow as pa

from .types import (CANDIDATE_SEARCH_EXCLUDED, LIST_FIELDS, MEMORY_FIELDS, TS_FIELDS,
                    HistoryRow, MemoryRow, SuppressRow)

T_MEMORIES = "memories"
T_HISTORY = "memory_history"
T_SUPPRESS = "suppress"
TABLES = (T_MEMORIES, T_HISTORY, T_SUPPRESS)
TS_TYPE = pa.timestamp("ms", tz="UTC")
DEFAULT_CANDIDATE_WHERE = "status NOT IN ({})".format(
    ", ".join(f"'{s}'" for s in CANDIDATE_SEARCH_EXCLUDED))

_INT8 = frozenset({"level"})
_INT32 = frozenset({"evidence_count", "user_evidence_count", "user_session_count",
                    "recall_injected_count", "recall_injected_strong", "recall_used_count",
                    "search_hit_count", "version", "schema_version"})
_FLOAT32 = frozenset({"importance"})
_BOOL = frozenset({"pinned", "core_required", "in_core", "explicit_user", "judge_pending",
                   "needs_review"})


class SchemaMismatch(RuntimeError):
    """Lance schema differs from the expected one (e.g. list<float> vector, wrong dim). Write nothing."""


class StoreMissing(RuntimeError):
    """lancedb dir or a table does not exist (run `yume init`)."""


def sql_quote(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def sql_in(column: str, values: Iterable[str]) -> str:
    vals = list(values)
    if not vals:
        return "FALSE"
    return f"{column} IN ({', '.join(sql_quote(v) for v in vals)})"


def vector_type(dim: int) -> pa.DataType:
    return pa.list_(pa.float32(), int(dim))


def _field_type(name: str, dim: int) -> pa.DataType:
    if name == "vector":
        return vector_type(dim)
    if name in TS_FIELDS:
        return TS_TYPE
    if name in LIST_FIELDS:
        return pa.list_(pa.string())
    if name in _INT8:
        return pa.int8()
    if name in _INT32:
        return pa.int32()
    if name in _FLOAT32:
        return pa.float32()
    if name in _BOOL:
        return pa.bool_()
    return pa.string()


def memories_schema(dim: int) -> pa.Schema:
    return pa.schema([pa.field(n, _field_type(n, dim), nullable=(n != "id")) for n in MEMORY_FIELDS])


def history_schema() -> pa.Schema:
    return pa.schema([
        pa.field("history_id", pa.string(), nullable=False),
        pa.field("memory_id", pa.string()),
        pa.field("run_id", pa.string()),
        pa.field("op", pa.string()),
        pa.field("before_json", pa.string()),
        pa.field("after_json", pa.string()),
        pa.field("at", TS_TYPE),
    ])


def suppress_schema(dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string(), nullable=False),
        pa.field("vector", vector_type(dim)),
        pa.field("text_sha", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("created_at", TS_TYPE),
        pa.field("reason", pa.string()),
    ])


def expected_schemas(dim: int) -> dict[str, pa.Schema]:
    return {T_MEMORIES: memories_schema(dim), T_HISTORY: history_schema(),
            T_SUPPRESS: suppress_schema(dim)}


# ── Arrow ⇄ Python conversion ────────────────────────────────────────────────

def _ts_array(values: Sequence[float | None]) -> pa.Array:
    return pa.array([None if v is None else int(round(float(v) * 1000.0)) for v in values],
                    type=pa.int64()).cast(TS_TYPE)


def _vec_array(values: Sequence[Any], dim: int) -> pa.Array:
    mask = [v is None for v in values]
    flat = np.zeros((len(values), dim), dtype=np.float32)
    for i, v in enumerate(values):
        if v is not None:
            a = np.asarray(v, dtype=np.float32)
            if a.shape != (dim,):
                raise SchemaMismatch(f"vector shape {a.shape} != ({dim},)")
            flat[i] = a
    arr = pa.FixedSizeListArray.from_arrays(pa.array(flat.reshape(-1), type=pa.float32()), dim)
    if any(mask):
        # rebuild with validity bitmap
        arr = pa.FixedSizeListArray.from_arrays(pa.array(flat.reshape(-1), type=pa.float32()), dim,
                                                mask=pa.array(mask, type=pa.bool_()))
    return arr


def _column(name: str, values: list[Any], typ: pa.DataType, dim: int) -> pa.Array:
    if pa.types.is_fixed_size_list(typ):
        return _vec_array(values, dim)
    if pa.types.is_timestamp(typ):
        return _ts_array(values)
    if pa.types.is_list(typ):
        return pa.array([list(v) if v is not None else [] for v in values], type=typ)
    return pa.array(values, type=typ)


def build_table(schema: pa.Schema, records: list[dict[str, Any]], dim: int) -> pa.Table:
    cols = [_column(f.name, [r.get(f.name) for r in records], f.type, dim) for f in schema]
    return pa.Table.from_arrays(cols, schema=schema)


def _col_values(tbl: pa.Table, name: str, dim: int) -> list[Any]:
    col = tbl.column(name)
    typ = col.type
    if pa.types.is_fixed_size_list(typ):
        chunks = col.combine_chunks()
        valid = chunks.is_valid().to_pylist()
        flat = chunks.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
        mat = flat.reshape(-1, typ.list_size) if len(flat) else np.zeros((0, typ.list_size), np.float32)
        out: list[Any] = []
        j = 0
        for ok in valid:
            if ok:
                out.append(mat[j].copy())
                j += 1
            else:
                out.append(None)
        return out
    if pa.types.is_timestamp(typ):
        return [None if v is None else v / 1000.0 for v in col.cast(pa.int64()).to_pylist()]
    return col.to_pylist()


def table_to_dicts(tbl: pa.Table, dim: int) -> list[dict[str, Any]]:
    names = tbl.column_names
    cols = {n: _col_values(tbl, n, dim) for n in names}
    return [{n: cols[n][i] for n in names} for i in range(tbl.num_rows)]


def row_to_record(row: MemoryRow) -> dict[str, Any]:
    return {f: getattr(row, f) for f in MEMORY_FIELDS}


def record_to_row(d: dict[str, Any]) -> MemoryRow:
    kw: dict[str, Any] = {}
    for f in MEMORY_FIELDS:
        if f not in d:
            continue
        v = d[f]
        if f in LIST_FIELDS:
            v = list(v or [])
        elif f == "importance" and v is not None:
            v = round(float(v), 6)
        elif f in _BOOL:
            v = bool(v) if v is not None else False
        kw[f] = v
    return MemoryRow(**kw)


# ── Store ────────────────────────────────────────────────────────────────────

class Store:
    def __init__(self, lancedb_dir: Path, db: Any, dim: int, embed_model: str):
        self.dir = lancedb_dir
        self.db = db
        self.dim = int(dim)
        self.embed_model = embed_model
        self._tables: dict[str, Any] = {}

    @classmethod
    def open(cls, lancedb_dir: str | os.PathLike, *, dim: int, embed_model: str,
             create: bool = False) -> "Store":
        """create=False: missing dir/table → StoreMissing (dry-run never creates anything).
        create=True (`yume init`): dir 0700 + any missing table with the exact schema."""
        d = Path(lancedb_dir)
        if not d.exists():
            if not create:
                raise StoreMissing(f"{d} 없음 (yume init 필요)")
            d.mkdir(parents=True, exist_ok=True)
            os.chmod(d, 0o700)
        s = cls(d, lancedb.connect(str(d)), dim, embed_model)
        schemas = expected_schemas(dim)
        for name in TABLES:
            if s._exists(name):
                s._tables[name] = s.db.open_table(name)
            elif create:
                s._tables[name] = s.db.create_table(name, schema=schemas[name])
            else:
                raise StoreMissing(f"Lance 테이블 {name} 없음 (yume init 필요)")
        return s

    @classmethod
    def from_config(cls, paths: Any, cfg: Any, *, create: bool = False) -> "Store":
        return cls.open(paths.lancedb_dir, dim=int(cfg.embed_dim), embed_model=cfg.embed_model_id(),
                        create=create)

    def _exists(self, name: str) -> bool:
        names: list[str] = []
        token = None
        while True:  # list_tables is paginated
            resp = self.db.list_tables(page_token=token) if token else self.db.list_tables()
            names += list(resp.tables)
            token = getattr(resp, "page_token", None)
            if not token:
                break
        return name in names

    def table(self, name: str = T_MEMORIES) -> Any:
        return self._tables[name]

    # ── guards ──
    def check_schema(self) -> None:
        """Every table must match expected_schemas(dim) exactly (names, order-insensitive types).
        Raises SchemaMismatch listing all differences."""
        problems: list[str] = []
        for name, want in expected_schemas(self.dim).items():
            got = self._tables[name].schema
            got_f = {f.name: f.type for f in got}
            want_f = {f.name: f.type for f in want}
            for n, t in want_f.items():
                if n not in got_f:
                    problems.append(f"{name}.{n} missing")
                elif got_f[n] != t:
                    problems.append(f"{name}.{n}: {got_f[n]} != {t}")
            for n in got_f:
                if n not in want_f:
                    problems.append(f"{name}.{n} unexpected")
        if problems:
            raise SchemaMismatch("; ".join(problems))

    def check_embed_model(self) -> None:
        """Every row's embed_model must equal the configured model id (sampled via filter)."""
        bad = self.count(f"embed_model != {sql_quote(self.embed_model)}")
        if bad:
            raise SchemaMismatch(f"{bad}개 행의 embed_model이 {self.embed_model}와 다름 (yume reembed)")

    # ── versions / counts ──
    def version(self, name: str = T_MEMORIES) -> int:
        return int(self._tables[name].version)

    def versions(self) -> dict[str, int]:
        return {n: self.version(n) for n in TABLES}

    def count(self, where: str | None = None, name: str = T_MEMORIES) -> int:
        t = self._tables[name]
        return int(t.count_rows(where) if where else t.count_rows())

    # ── reads ──
    def _scan(self, name: str, where: str | None = None, columns: list[str] | None = None) -> pa.Table:
        t = self._tables[name]
        if t.count_rows() == 0:
            return expected_schemas(self.dim)[name].empty_table()
        q = t.search()
        if where:
            q = q.where(where)
        if columns:
            q = q.select(columns)
        return q.limit(None).to_arrow() if hasattr(q, "limit") else q.to_arrow()

    def load_working_set(self, *, where: str | None = None,
                         with_vectors: bool = True) -> dict[str, MemoryRow]:
        """All memories (any status unless `where`) keyed by id. R0 working copy base."""
        cols = None if with_vectors else [f for f in MEMORY_FIELDS if f != "vector"]
        tbl = self._scan(T_MEMORIES, where, cols)
        return {d["id"]: record_to_row(d) for d in table_to_dicts(tbl, self.dim)}

    def get(self, ids: Iterable[str]) -> dict[str, MemoryRow]:
        ids = list(ids)
        if not ids:
            return {}
        return self.load_working_set(where=sql_in("id", ids))

    def search(self, vector: Any, *, k: int = 5, where: str | None = DEFAULT_CANDIDATE_WHERE,
               prefilter: bool = True) -> list[tuple[MemoryRow, float]]:
        """Cosine top-k over committed rows: [(row, cos)] best first. cos = 1 - _distance."""
        t = self._tables[T_MEMORIES]
        if t.count_rows() == 0:
            return []
        q = t.search(np.asarray(vector, dtype=np.float32), vector_column_name="vector") \
             .distance_type("cosine")
        if where:
            q = q.where(where, prefilter=prefilter)
        tbl = q.limit(int(k)).to_arrow()
        dists = tbl.column("_distance").to_pylist()
        rows = table_to_dicts(tbl.drop_columns(["_distance"]), self.dim)
        return [(record_to_row(d), 1.0 - float(dist)) for d, dist in zip(rows, dists)]

    def search_suppress(self, vector: Any, *, k: int = 1) -> list[tuple[SuppressRow, float]]:
        t = self._tables[T_SUPPRESS]
        if t.count_rows() == 0:
            return []
        tbl = (t.search(np.asarray(vector, dtype=np.float32), vector_column_name="vector")
               .distance_type("cosine").limit(int(k)).to_arrow())
        dists = tbl.column("_distance").to_pylist()
        out = []
        for d, dist in zip(table_to_dicts(tbl.drop_columns(["_distance"]), self.dim), dists):
            out.append((SuppressRow(**{k2: d[k2] for k2 in ("id", "vector", "text_sha", "kind",
                                                           "created_at", "reason")}), 1.0 - float(dist)))
        return out

    def load_suppress(self) -> list[SuppressRow]:
        tbl = self._scan(T_SUPPRESS)
        return [SuppressRow(**d) for d in table_to_dicts(tbl, self.dim)]

    def suppress_shas(self) -> set[str]:
        tbl = self._scan(T_SUPPRESS, columns=["text_sha"])
        return {v for v in tbl.column("text_sha").to_pylist() if v} if tbl.num_rows else set()

    def history(self, *, memory_id: str | None = None, run_id: str | None = None) -> list[HistoryRow]:
        conds = []
        if memory_id:
            conds.append(f"memory_id = {sql_quote(memory_id)}")
        if run_id:
            conds.append(f"run_id = {sql_quote(run_id)}")
        tbl = self._scan(T_HISTORY, " AND ".join(conds) or None)
        rows = [HistoryRow(**d) for d in table_to_dicts(tbl, self.dim)]
        return sorted(rows, key=lambda h: (h.at, h.history_id))

    # ── writes (R8 steps 2–4) ──
    def _merge(self, name: str, key: str, records: list[dict[str, Any]]) -> int | None:
        if not records:
            return None
        data = build_table(expected_schemas(self.dim)[name], records, self.dim)
        (self._tables[name].merge_insert(key).when_matched_update_all()
         .when_not_matched_insert_all().execute(data))
        return self.version(name)

    def commit_history(self, rows: Sequence[HistoryRow]) -> int | None:
        return self._merge(T_HISTORY, "history_id", [vars(r).copy() for r in rows])

    def commit_suppress(self, rows: Sequence[SuppressRow]) -> int | None:
        return self._merge(T_SUPPRESS, "id", [vars(r).copy() for r in rows])

    def commit_memories(self, rows: Sequence[MemoryRow]) -> int | None:
        for r in rows:
            if r.vector is None:
                raise SchemaMismatch(f"memory {r.id} has no vector")
        return self._merge(T_MEMORIES, "id", [row_to_record(r) for r in rows])

    def purge(self, ids: Sequence[str]) -> int | None:
        """Hard delete (forgotten+30d, quarantined). Only R6/R8 and `yume` admin call this."""
        ids = list(ids)
        if not ids:
            return None
        self._tables[T_MEMORIES].delete(sql_in("id", ids))
        return self.version(T_MEMORIES)

    def purge_history(self, memory_ids: Sequence[str]) -> int | None:
        """Delete every history row of purged memories (their before/after JSON holds the text).
        Called with R8 purge (forgotten+30d, quarantine). No-op when nothing matches."""
        ids = list(memory_ids)
        if not ids:
            return None
        where = sql_in("memory_id", ids)
        if not self._tables[T_HISTORY].count_rows(where):
            return None
        self._tables[T_HISTORY].delete(where)
        return self.version(T_HISTORY)

    def delete_suppress(self, where: str) -> None:
        self._tables[T_SUPPRESS].delete(where)

    def commit(self, *, upserts: Sequence[MemoryRow] = (), history: Sequence[HistoryRow] = (),
               suppress: Sequence[SuppressRow] = (), purge_ids: Sequence[str] = ()) -> dict[str, int]:
        """R8 steps 2–4 in order. Returns versions after."""
        self.commit_history(history)
        self.commit_suppress(suppress)
        self.commit_memories(upserts)
        self.purge(purge_ids)
        return self.versions()

    # ── maintenance ──
    def restore(self, version: int, name: str = T_MEMORIES) -> int:
        """`yume restore --run`: table.restore(version) creates a new latest version equal to it."""
        self._tables[name].restore(int(version))
        return self.version(name)

    def optimize(self, *, cleanup_older_than_days: float = 14, delete_unverified: bool = False,
                 tables: Iterable[str] = TABLES) -> None:
        """R8-7 (14 days) / quarantine purge (0 days + delete_unverified, text tables only)."""
        for name in tables:
            self._tables[name].optimize(cleanup_older_than=timedelta(days=cleanup_older_than_days),
                                        delete_unverified=delete_unverified)

    def list_versions(self, name: str = T_MEMORIES) -> list[int]:
        out = []
        for v in self._tables[name].list_versions():
            out.append(int(v["version"] if isinstance(v, dict) else v.version))
        return out


def history_json(snapshot: dict[str, Any] | None) -> str:
    return json.dumps(snapshot or {}, ensure_ascii=False, sort_keys=True, default=str)
