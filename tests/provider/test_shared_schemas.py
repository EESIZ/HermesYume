"""Provider-side (Hermes venv, stdlib only): the shared single-source modules written by the
foundation load without numpy and agree with the dream side.

Run (from the Hermes runtime checkout): cd <hermes-runtime> && PYTHONDONTWRITEBYTECODE=1 \
     venv/bin/python -m unittest discover -s <repo>/tests/provider -t <repo> -v
"""

import importlib.util
import re
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
YUME = REPO / "provider" / "_yume"


def load(name):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", YUME / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SharedSchemas(unittest.TestCase):
    def test_stdlib_only_imports(self):
        for name in ("live_schema", "serving_schema", "corefmt"):
            src = (YUME / f"{name}.py").read_text(encoding="utf-8")
            self.assertIsNone(re.search(r"^\s*(import|from)\s+numpy", src, re.M))
            self.assertIsNone(re.search(r"^\s*from\s+\.", src, re.M))   # loadable by path
            load(name)

    def test_live_schema_creates_db(self):
        ls = load("live_schema")
        with tempfile.TemporaryDirectory() as d:
            conn = ls.connect(Path(d) / "live.db")
            conn.execute("INSERT INTO inbox(ts, op, text, vec) VALUES(1, 'remember', 'x', ?)",
                         (ls.vec_to_blob([0.5, -0.25]),))
            conn.commit()
            blob = conn.execute("SELECT vec FROM inbox").fetchone()[0]
            self.assertEqual(ls.blob_to_vec(blob), [0.5, -0.25])
            conn.close()
            ro = ls.connect(Path(d) / "live.db", readonly=True)
            with self.assertRaises(sqlite3.OperationalError):
                ro.execute("INSERT INTO inbox(ts, op) VALUES(1, 'forget')")
            ro.close()

    def test_serving_schema_fts_trigram(self):
        ss = load("serving_schema")
        conn = sqlite3.connect(":memory:")
        ss.create_schema(conn)
        conn.execute("INSERT INTO items_fts(id, text, subject) VALUES('a', '가계부 DB가 단일 원장', '가계')")
        self.assertEqual(conn.execute("SELECT id FROM items_fts WHERE items_fts MATCH '가계부'").fetchone()[0], "a")

    def test_corefmt_sha_matches_known_value(self):
        cf = load("corefmt")
        import hashlib
        self.assertEqual(cf.core_sha(" a  b "), hashlib.sha1(b"a b").hexdigest())

    def test_fakes_importable_without_numpy(self):
        from tests import fakes
        v = fakes.hash_embed("한국어 문장", 64)
        self.assertEqual(len(v), 64)
        self.assertAlmostEqual(sum(x * x for x in v), 1.0, places=6)
        with fakes.FakeOpenAIServer() as s:
            self.assertTrue(s.base_url.startswith("http://127.0.0.1:"))


if __name__ == "__main__":
    unittest.main()
