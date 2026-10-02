"""`hermes hermesyume …` → runs the dream-side `yume` CLI (config `yume_bin`) as a subprocess with
the same HERMES_HOME (PLAN-v2 §1.1 C3, §6.1). No side effects at import (the Hermes loader
pre-executes this file)."""

import argparse
import os


def register_cli(subparser):
    subparser.add_argument("yume_args", nargs=argparse.REMAINDER,
                           help="yume 하위 명령과 인자 (예: status, search \"질의\", dream --dry-run)")


def _hermes_home():
    try:
        from hermes_constants import get_hermes_home
        return str(get_hermes_home())
    except Exception:
        return os.path.expanduser(os.environ.get("HERMES_HOME", "").strip() or "~/.hermes")


def _load_config(home):
    try:
        from ._yume import config as _cfg
    except Exception:
        import importlib.util
        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location("_hermesyume_cli_config",
                                                      os.path.join(here, "_yume", "config.py"))
        _cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_cfg)
    return _cfg.load(home) or dict(_cfg.DEFAULTS)


def yume_command_argv(args, home=None):
    """argv for the subprocess (exposed for tests)."""
    home = home or _hermes_home()
    cfg = _load_config(home)
    rest = list(getattr(args, "yume_args", None) or [])
    if rest and rest[0] == "--":
        rest = rest[1:]
    return [os.path.expanduser(str(cfg.get("yume_bin") or "yume")), *rest], home


def hermesyume_command(args):
    import subprocess
    import sys
    argv, home = yume_command_argv(args)
    if not (os.path.isfile(argv[0]) and os.access(argv[0], os.X_OK)):
        print("hermesyume: yume 실행 파일을 찾을 수 없습니다: %s (config.json의 yume_bin 확인, "
              "deploy/setup_venv.sh로 설치)" % argv[0], file=sys.stderr)
        return 1
    env = dict(os.environ)
    env["HERMES_HOME"] = home
    try:
        return subprocess.run(argv, env=env).returncode
    except KeyboardInterrupt:
        return 130
    except OSError as e:
        print("hermesyume: yume 실행 실패: %s" % type(e).__name__, file=sys.stderr)
        return 1
