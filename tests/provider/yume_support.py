"""Shared helpers for provider tests (Hermes venv, stdlib only).

Safety: importing this module (1) disables bytecode writing, (2) points HERMES_HOME at a private
sandbox before any Hermes runtime module is imported, (3) removes OPENAI_* variables so nothing can
reach a real endpoint with a real key, (4) puts the repo before the runtime on sys.path (both have a
``tests`` package). Runtime modules are imported read-only from the Hermes runtime checkout:
$HERMESYUME_TEST_RUNTIME, else $HERMES_RUNTIME_DIR, else where ``hermes_cli`` is importable from
(the Hermes venv / the runtime directory the tests are started in).

Run: cd <hermes-runtime> && PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m unittest \
     discover -s <repo>/tests/provider -t <repo> -v
"""

import importlib.util
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
PROVIDER_SRC = REPO / "provider"


def _runtime_dir() -> Path:
    for k in ("HERMESYUME_TEST_RUNTIME", "HERMES_RUNTIME_DIR"):
        if os.environ.get(k, "").strip():
            return Path(os.environ[k]).expanduser()
    spec = importlib.util.find_spec("hermes_cli")       # no import: just where the package lives
    if spec is not None and spec.origin:
        return Path(spec.origin).resolve().parent.parent
    raise RuntimeError("Hermes runtime not found: run from the runtime checkout with its venv, "
                       "or set HERMESYUME_TEST_RUNTIME")


RUNTIME = _runtime_dir()
# Real agent homes the sandbox must never be: Hermes' default plus any the caller protects.
LIVE_HOMES = {Path(p).expanduser().resolve()
              for p in ("~/.hermes", os.environ.get("HERMESYUME_TEST_LIVE_HOME", ""),
                        *os.environ.get("HERMESYUME_PROTECT_HOMES", "").split(os.pathsep))
              if p.strip()}

for _k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "HERMESYUME_NOW"):
    os.environ.pop(_k, None)

_BASE_TMP = Path(tempfile.mkdtemp(prefix="yume-provider-tests-"))
os.environ["HERMES_HOME"] = str(_BASE_TMP / "base-home")
(_BASE_TMP / "base-home").mkdir(parents=True, exist_ok=True)

for _p in (str(RUNTIME), str(REPO)):        # REPO ends up first
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from tests import fakes  # noqa: E402  (stdlib-only module top)
from tests.fixtures.hermes_home import TEST_FILTERS, make_hermes_home  # noqa: E402

DIM = 1536
MODEL_ID = "openai/text-embedding-3-small@1536"
USER_TEXT_LEDGER = "**가계부 관리:** 지출 기록은 공용 가계부 DB를 단일 원장으로 사용한다."


def assert_sandbox(home):
    p = Path(home).resolve()
    assert p not in LIVE_HOMES and str(p).startswith(str(_BASE_TMP.resolve())), p


def load_yume_pkg(name="_yume_test_pkg"):
    """provider/_yume as a standalone package (no provider/__init__ → no agent import)."""
    import importlib.util
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, str(PROVIDER_SRC / "_yume" / "__init__.py"),
        submodule_search_locations=[str(PROVIDER_SRC / "_yume")])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def yume_mod(sub):
    import importlib
    pkg = load_yume_pkg()
    return importlib.import_module(pkg.__name__ + "." + sub)


def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def vec_blob(v):
    from array import array
    a = array("f", [float(x) for x in v])
    if sys.byteorder != "little":
        a.byteswap()
    return a.tobytes()


class Sandbox:
    """Fake HERMES_HOME with the provider installed under plugins/hermesyume and a config.json."""

    def __init__(self, name, *, config=None, with_config=True, user_entries=None):
        self.base = _BASE_TMP / ("%s-%d" % (name, time.monotonic_ns()))
        fh = make_hermes_home(self.base, user_entries=user_entries)
        self.home = fh.root
        self.workspace = fh.workspace
        assert_sandbox(self.home)
        self.data = self.home / "hermesyume"
        self.data.mkdir(mode=0o700, parents=True, exist_ok=True)
        (self.data / "serving").mkdir(mode=0o700, exist_ok=True)
        plug = self.home / "plugins" / "hermesyume"
        shutil.copytree(PROVIDER_SRC, plug, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        self.config = {"embed_base_url": "http://127.0.0.1:9/v1",
                       # the public defaults filter nothing; the fixtures use these filters
                       "exclude_first_message_regex": TEST_FILTERS["exclude_first_message_regex"],
                       "deny_cwd_globs": list(TEST_FILTERS["deny_cwd_globs"])}
        if config:
            self.config.update(config)
        if with_config:
            self.write_config()

    def write_config(self, **changes):
        self.config.update(changes)
        p = self.data / "config.json"
        p.write_text(json.dumps(self.config, ensure_ascii=False), encoding="utf-8")
        st = p.stat()   # make sure an mtime-cached reader sees the change
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    @property
    def recall_sqlite(self):
        return self.data / "serving" / "recall.sqlite"

    @property
    def live_db(self):
        return self.data / "live.db"

    def build_serving(self, items, *, pins=(), run_id="run-1", embed_model=MODEL_ID, dim=DIM):
        """items: dicts with id, text, vec (dim floats) and optional item columns."""
        ss = yume_mod("serving_schema")
        tmp = self.data / "serving" / ("recall.%s.sqlite" % run_id)
        if tmp.exists():
            tmp.unlink()
        conn = sqlite3.connect(str(tmp))
        conn.execute("PRAGMA journal_mode=DELETE")
        ss.create_schema(conn)
        for it in items:
            v = unit(it["vec"])
            row = {"id": it["id"], "text": it["text"], "subject": it.get("subject", ""),
                   "kind": it.get("kind", "fact"), "tier": it.get("tier", "decaying"),
                   "status": it.get("status", "active"), "pinned": 1 if it.get("pinned") else 0,
                   "core_sha": it.get("core_sha"), "event_time": it.get("event_time"),
                   "valid_until": it.get("valid_until"), "strength": it.get("strength", 0.5),
                   "refs": json.dumps(it.get("refs", []), ensure_ascii=False),
                   "vec256": vec_blob(unit(v[:256])), "vec": vec_blob(v)}
            cols = ss.ITEM_COLUMNS
            conn.execute("INSERT INTO items(%s) VALUES(%s)" % (",".join(cols), ",".join("?" * len(cols))),
                         [row[c] for c in cols])
            conn.execute("INSERT INTO items_fts(id, text, subject) VALUES(?,?,?)",
                         (row["id"], row["text"], row["subject"]))
        for p in pins:
            conn.execute("INSERT INTO pins(id, text, label, core_target) VALUES(?,?,?,?)",
                         (p["id"], p["text"], p.get("label"), p.get("core_target")))
        meta = {"embed_model": embed_model, "dim": str(dim), "run_id": run_id, "lance_version": "1",
                "built_at": str(time.time()), "count": str(len(items))}
        conn.executemany("INSERT INTO meta(key, value) VALUES(?,?)", sorted(meta.items()))
        conn.commit()
        conn.close()
        os.replace(tmp, self.recall_sqlite)

    def live_rows(self, sql, params=()):
        if not self.live_db.exists():
            return []
        conn = sqlite3.connect("file:%s?mode=ro" % self.live_db, uri=True)
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def events(self, kind=None):
        q = "SELECT memory_id, kind, cos, mode, session_id, turn_no FROM recall_events"
        if kind:
            return self.live_rows(q + " WHERE kind=? ORDER BY id", (kind,))
        return self.live_rows(q + " ORDER BY id")

    def inbox(self, op=None):
        q = "SELECT id, op, text, old_text, target, memory_id, vec, embed_model, meta_json, pin, kind FROM inbox"
        if op:
            return self.live_rows(q + " WHERE op=? ORDER BY id", (op,))
        return self.live_rows(q + " ORDER BY id")


def reset_singletons():
    """Forget module singletons of the provider copy loaded by Hermes (and of the test package)."""
    for name, mod in list(sys.modules.items()):
        if not name.endswith(("._yume.live", "._yume.serving", "._yume.embed_http", "._yume.config")):
            continue
        for fn in ("reset_all", "clear", "reset", "clear_cache"):
            f = getattr(mod, fn, None)
            if callable(f):
                try:
                    f()
                except Exception:
                    pass


def load_provider(sb, register_skills=False):
    """The real Hermes loader against the sandbox HERMES_HOME."""
    os.environ["HERMES_HOME"] = str(sb.home)
    from plugins.memory import load_memory_provider
    return load_memory_provider("hermesyume", register_skills=register_skills)


def new_provider(sb, *, session_id="sess-1", platform="cli", chat_type=None, init=True, **kw):
    p = load_provider(sb)
    assert p is not None, "provider failed to load"
    if init:
        kwargs = {"platform": platform, "hermes_home": str(sb.home)}
        if chat_type is not None:
            kwargs["chat_type"] = chat_type
        kwargs.update(kw)
        p.initialize(session_id, **kwargs)
    return p


def provider_module():
    return sys.modules.get("_hermes_user_memory.hermesyume")


def anchor(text):
    return fakes.hash_embed(text, DIM)


def with_cos(anchor_vec, cos, seed):
    return fakes.vector_with_cos(anchor_vec, cos, seed)


class FakeServer(fakes.FakeOpenAIServer):
    pass


def wait_until(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def cleanup():
    shutil.rmtree(_BASE_TMP, ignore_errors=True)
