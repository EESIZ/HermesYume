"""P1 contract: the real Hermes loader/manager accept the provider (Hermes venv, stdlib only).

Run (from the Hermes runtime checkout): cd <hermes-runtime> && PYTHONDONTWRITEBYTECODE=1 \
     venv/bin/python -m unittest discover -s <repo>/tests/provider -t <repo> -v
"""

import ast
import atexit
import builtins
import inspect
import json
import socket
import sqlite3
import sys
import threading
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from tests.provider import yume_support as S  # noqa: E402

atexit.register(S.cleanup)

STDLIB_OK = {"__future__", "argparse", "array", "collections", "dataclasses", "datetime", "fnmatch",
             "hashlib", "heapq", "http", "importlib", "itertools", "json", "logging", "math", "operator", "os",
             "re", "socket", "sqlite3", "ssl", "struct", "subprocess", "sys", "threading", "time",
             "typing", "unicodedata", "urllib"}
HERMES_OK = {"agent", "hermes_constants", "tools"}   # runtime modules the provider may touch


def _imports(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name, 0) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            out.append((node.module or "", node.level))
    return out


class ProviderContract(unittest.TestCase):
    def setUp(self):
        S.reset_singletons()
        self.sb = S.Sandbox("p1")

    def test_loader_finds_and_instantiates(self):
        for name in [n for n in sys.modules if n.startswith("_hermes_user_memory.hermesyume")]:
            sys.modules.pop(name, None)
        p = S.load_provider(self.sb)
        self.assertIsNotNone(p)
        self.assertEqual(p.name, "hermesyume")
        self.assertTrue(p.is_available())
        mod = S.provider_module()
        self.assertTrue(mod.__file__.startswith(str(self.sb.home)))

    def test_memory_provider_in_first_8192_chars(self):
        src = (S.PROVIDER_SRC / "__init__.py").read_text(encoding="utf-8")[:8192]
        self.assertIn("MemoryProvider", src)

    def test_import_and_register_without_network_or_io(self):
        for name in [n for n in sys.modules if n.startswith("_hermes_user_memory.hermesyume")]:
            sys.modules.pop(name, None)

        def boom(*a, **k):
            raise AssertionError("network access during import")

        orig_connect, orig_cc = socket.socket.connect, socket.create_connection
        socket.socket.connect = boom
        socket.create_connection = boom
        try:
            p = S.load_provider(self.sb)
            self.assertIsNotNone(p)
            self.assertTrue(p.is_available())
            mod = S.provider_module()
        finally:
            socket.socket.connect, socket.create_connection = orig_connect, orig_cc

        class Ctx:
            provider = None

            def register_memory_provider(self, prov):
                self.provider = prov

        def no_io(*a, **k):
            raise AssertionError("I/O during register()")

        saved = (builtins.open, sqlite3.connect, threading.Thread.start, socket.socket.connect)
        builtins.open, sqlite3.connect, threading.Thread.start, socket.socket.connect = no_io, no_io, no_io, no_io
        try:
            ctx = Ctx()
            mod.register(ctx)
            schemas = ctx.provider.get_tool_schemas()
            ctx.provider.backup_paths()
        finally:
            builtins.open, sqlite3.connect, threading.Thread.start, socket.socket.connect = saved
        self.assertIsNotNone(ctx.provider)
        self.assertEqual(len(schemas), 3)

    def test_unavailable_without_config_or_disabled(self):
        sb = S.Sandbox("p1-nocfg", with_config=False)
        p = S.load_provider(sb)
        self.assertFalse(p.is_available())
        self.assertIn("yume init", p.unavailable_reason())
        self.sb.write_config(enabled=False)
        p2 = S.load_provider(self.sb)
        self.assertFalse(p2.is_available())

    def test_tool_schemas_static_and_not_core(self):
        p = S.load_provider(self.sb)
        before = p.get_tool_schemas()
        p.initialize("s1", platform="cli", hermes_home=str(self.sb.home))
        after = p.get_tool_schemas()
        self.assertEqual(before, after)
        names = [s["name"] for s in before]
        self.assertEqual(names, ["yume_search", "yume_remember", "yume_forget"])
        from toolsets import _HERMES_CORE_TOOLS
        self.assertFalse(set(names) & set(_HERMES_CORE_TOOLS))
        from agent.memory_manager import normalize_tool_schema
        for s in before:
            self.assertIsNotNone(normalize_tool_schema(s))
        by = {s["name"]: s for s in before}
        self.assertEqual(by["yume_search"]["parameters"]["required"], ["query"])
        self.assertEqual(by["yume_remember"]["parameters"]["properties"]["text"]["maxLength"], 400)
        self.assertEqual(len(by["yume_remember"]["parameters"]["properties"]["kind"]["enum"]), 13)
        self.assertNotIn("required", by["yume_forget"]["parameters"])

    def test_u1_tool_descriptions_exact(self):
        """DEVIATIONS E2E-8 (U1): remember/forget only on an explicit user request, never
        proactively or as a question; no memory-system talk; ids are internal."""
        schemas = {s["name"]: s for s in S.load_provider(self.sb).get_tool_schemas()}
        by = {n: s["description"] for n, s in schemas.items()}
        self.assertEqual(by["yume_search"],
                         "장기기억에서 자동 첨부에 없던 과거 사실을 더 찾는다. 결과는 사실 문장과 날짜뿐이다. "
                         "검색한 일이나 기억 시스템은 사용자에게 언급하지 말 것.")
        self.assertEqual(by["yume_remember"],
                         '사용자가 "기억해", "잊지 마", "저장해 둬", "앞으로 항상"처럼 기억·저장·고정을 직접 요청한 '
                         "내용만 저장. 그 밖에는 절대 먼저 저장하지 말고, 기억할지 묻지도 말 것. "
                         "저장 결과나 id는 사용자에게 말하지 말 것. 매번 필요한 짧은 핵심은 memory 도구(USER.md).")
        self.assertEqual(by["yume_forget"],
                         "사용자가 잊으라고 직접 요청한 기억만 숨김. 먼저 제안하지 말 것. id를 모르면 query로 후보를 "
                         "받아 그 id로 다시 호출. id는 내부용이니 사용자에게 보여 주지 말 것.")
        for word in ("기억해", "잊지 마", "저장해 둬", "앞으로 항상"):
            self.assertIn(word, by["yume_remember"])
        # no memory-status vocabulary the agent could relay
        for name, d in by.items():
            for w in ("휴면", "만료", "대체", "밤에", "tier", "score"):
                self.assertNotIn(w, d, (name, w))
        inc = schemas["yume_search"]["parameters"]["properties"]["include_inactive"]["description"]
        self.assertIn("current=false", inc)
        self.assertIn("상태로 설명하지 말 것", inc)

    def test_backup_paths_empty(self):
        p = S.load_provider(self.sb)
        self.assertEqual(p.backup_paths(), [])
        p.initialize("s1", platform="cli", hermes_home=str(self.sb.home))
        self.assertEqual(p.backup_paths(), [])

    def test_hook_signatures_match_abc(self):
        from agent.memory_provider import MemoryProvider
        cls = type(S.load_provider(self.sb))
        for name in ("is_available", "initialize", "system_prompt_block", "prefetch", "queue_prefetch",
                     "recall_status", "sync_turn", "get_tool_schemas", "handle_tool_call", "shutdown",
                     "on_turn_start", "on_session_end", "on_session_switch", "on_pre_compress",
                     "on_delegation", "on_memory_write", "backup_paths", "unavailable_reason"):
            base = inspect.signature(getattr(MemoryProvider, name))
            ours = inspect.signature(getattr(cls, name))
            self.assertEqual([(q.name, q.kind, q.default) for q in base.parameters.values()],
                             [(q.name, q.kind, q.default) for q in ours.parameters.values()], name)
        from agent.memory_manager import MemoryManager, _accepts_require_checkpoint
        p = S.load_provider(self.sb)
        self.assertTrue(MemoryManager._provider_sync_accepts_messages(p))
        self.assertEqual(MemoryManager._provider_memory_write_metadata_mode(p), "keyword")
        self.assertFalse(_accepts_require_checkpoint(p.on_pre_compress))

    def test_stdlib_only_and_no_telegram(self):
        files = [S.PROVIDER_SRC / "__init__.py", S.PROVIDER_SRC / "cli.py"] + \
            sorted((S.PROVIDER_SRC / "_yume").glob("*.py"))
        for f in files:
            for mod, level in _imports(f):
                if level:
                    continue
                top = mod.split(".")[0]
                if f.parent.name == "_yume":
                    self.assertIn(top, STDLIB_OK, "%s imports %s" % (f.name, mod))
                else:
                    self.assertIn(top, STDLIB_OK | HERMES_OK, "%s imports %s" % (f.name, mod))
            src = f.read_text(encoding="utf-8")
            for bad in ("numpy", "lancedb", "pyarrow", "api.telegram.org", "sendMessage",
                        "sendDocument", "import telegram", "from telegram"):
                self.assertFalse(bad in src, "%s mentions %s" % (f.name, bad))
            self.assertFalse(any(m.split(".")[0] == "hermesyume" for m, lv in _imports(f) if not lv),
                             "%s imports the dream package" % f.name)

    def test_plugin_yaml(self):
        y = (S.PROVIDER_SRC / "plugin.yaml").read_text(encoding="utf-8")
        self.assertIn("name: hermesyume", y)
        self.assertIn("version: 2.0.0", y)
        self.assertIn("kind: exclusive", y)
        self.assertIn("description:", y)
        self.assertNotIn("pip_dependencies", y)
        import yaml   # available in the Hermes venv (the loader reads plugin.yaml with it)
        meta = yaml.safe_load(y)
        self.assertEqual((meta["name"], str(meta["version"]), meta["kind"]), ("hermesyume", "2.0.0", "exclusive"))

    def test_memory_manager_end_to_end(self):
        """Real MemoryManager: init → system prompt → prefetch_all → sync_all(used) → tool → write hook."""
        from agent.memory_manager import MemoryManager, build_memory_context_block
        q = "Orion 스테이징 서버 포트 몇 번이었지?"
        a = S.anchor(q)
        self.sb.build_serving([
            {"id": "m8081", "text": "Orion 결제 스테이징 서버 포트는 8081이다.", "kind": "reference",
             "tier": "durable", "vec": S.with_cos(a, 0.72, "a"), "event_time": 1790000000.0,
             "strength": 0.7}])
        with S.fakes.FakeOpenAIServer() as srv:
            self.sb.write_config(embed_base_url=srv.base_url)
            p = S.load_provider(self.sb)
            mm = MemoryManager()
            mm.add_provider(p)
            self.assertTrue(mm.has_tool("yume_search"))
            mm.initialize_all(session_id="sess-mm", platform="cli", hermes_home=str(self.sb.home))
            sp = mm.build_system_prompt()
            self.assertIn("[장기기억]", sp)
            mm.on_turn_start(1, q)
            ctx = mm.prefetch_all(q)
            self.assertIn("8081", ctx)
            block = build_memory_context_block(ctx)
            self.assertTrue(block.startswith("<memory-context>"))
            self.assertIn("8081", block)
            self.assertIn("Yume — recalled 1 memory", mm.describe_recall())
            mm.sync_all(q, "스테이징 서버 포트는 8081번입니다.", session_id="sess-mm")
            self.assertTrue(mm.flush_pending(5))
            res = json.loads(mm.handle_tool_call("yume_search", {"query": "Orion 스테이징 서버 포트"}))
            self.assertTrue(res["ok"])
            self.assertEqual(res["results"][0], {"text": "Orion 결제 스테이징 서버 포트는 8081이다.",
                                                 "date": "2026-09-21"})
            mm.notify_memory_tool_write({"success": True}, {"action": "remove", "target": "user",
                                                            "old_text": "공용 가계부"})
            self.assertEqual(mm.on_pre_compress([]), "")          # F-10: no summarizer material
            mm.shutdown_all()
        kinds = [e[1] for e in self.sb.events()]
        self.assertIn("injected", kinds)
        self.assertIn("used", kinds)
        self.assertIn("tool_hit", kinds)
        rm = self.sb.inbox("core_remove")
        self.assertEqual(len(rm), 1)
        self.assertEqual(rm[0][2], S.USER_TEXT_LEDGER)


if __name__ == "__main__":
    unittest.main()
