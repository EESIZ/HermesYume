"""install.sh (one-line installer) in a fully fake HOME / HERMES_HOME sandbox.

Network-free: HERMESYUME_SRC points the installer at this checkout (no git), and stub ``uv`` /
``python3*`` / ``systemctl`` / ``crontab`` / ``loginctl`` / ``git`` / ``hermes`` / ``curl`` in a
sandbox bin dir log their arguments and fake the venv (its ``yume`` runs this checkout's CLI with the
test interpreter). When bubblewrap is usable the installer runs with the whole filesystem read-only
except the sandbox and with no network, so any write outside the sandbox fails the run; the stub
logs and before/after snapshots are checked either way.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fixtures.hermes_home import make_hermes_home
from tests.fixtures.statedb import build_basic

REPO = Path(__file__).resolve().parents[2]
INSTALL = REPO / "install.sh"

STUB_UV = r"""#!/bin/sh
echo "uv $*" >> "$STUB_LOG"
if [ "$1" = venv ]; then
  for a in "$@"; do d="$a"; done
  mkdir -p "$d/bin" && : > "$d/pyvenv.cfg" && ln -sf "$STUB_REAL_PY" "$d/bin/python"
  exit 0
fi
if [ "$1" = pip ]; then
  py=""; prev=""
  for a in "$@"; do [ "$prev" = "--python" ] && py="$a"; prev="$a"; done
  case "$*" in *--no-deps*)
    v="$(dirname "$(dirname "$py")")"
    printf '#!/bin/sh\nexec "%s" -m hermesyume.cli "$@"\n' "$STUB_REAL_PY" > "$v/bin/yume"
    chmod +x "$v/bin/yume" ;;
  esac
  exit 0
fi
exit 1
"""

STUB_PY = r"""#!/bin/sh
echo "python $*" >> "$STUB_LOG"
case "$1" in
  -c) exec "$STUB_REAL_PY" "$@" ;;
  -m)
    case "$2" in
      venv) d="$3"; mkdir -p "$d/bin"; : > "$d/pyvenv.cfg"; cp "$0" "$d/bin/python"; chmod +x "$d/bin/python"; exit 0 ;;
      pip)
        case "$*" in *--no-deps*)
          v="$(cd "$(dirname "$0")/.." && pwd)"
          printf '#!/bin/sh\nexec "%s" -m hermesyume.cli "$@"\n' "$STUB_REAL_PY" > "$v/bin/yume"
          chmod +x "$v/bin/yume" ;;
        esac
        exit 0 ;;
    esac ;;
esac
exit 1
"""

STUB_SYSTEMCTL = r"""#!/bin/sh
echo "systemctl $*" >> "$STUB_LOG"
case "$*" in *show-environment*) exit "${STUB_SYSTEMD_OK:-1}" ;; esac
exit 0
"""

STUB_CRONTAB = r"""#!/bin/sh
echo "crontab $*" >> "$STUB_LOG"
if [ "$1" = "-l" ]; then
  if [ -f "$STUB_CRONTAB_FILE" ]; then cat "$STUB_CRONTAB_FILE"; exit 0; fi
  echo "no crontab for sandbox" >&2; exit 1
fi
cp "$1" "$STUB_CRONTAB_FILE"
"""

STUB_LOGGER = r"""#!/bin/sh
echo "%s $*" >> "$STUB_LOG"
exit %d
"""


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _bwrap_ok() -> bool:
    if not shutil.which("bwrap"):
        return False
    r = subprocess.run(["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
                        "--unshare-net", "true"], capture_output=True)
    return r.returncode == 0


BWRAP = _bwrap_ok()


class Sandbox:
    def __init__(self, tmp: Path, *, uv: bool, systemd: bool, state_db: bool = True,
                 crontab: str | None = None):
        self.root = tmp / "sb"
        self.home = self.root / "home"
        self.bin = self.root / "bin"
        self.tmp = self.root / "tmp"
        for d in (self.home, self.bin, self.tmp):
            d.mkdir(parents=True)
        fh = make_hermes_home(self.root / "seed", env_text="SOME_OTHER_SETTING=1\n")
        self.hh = self.home / ".hermes"
        shutil.move(str(fh.root), str(self.hh))
        if state_db:
            build_basic(self.hh / "state.db", 1_790_000_000.0)
        self.log = self.root / "stub.log"
        self.log.write_text("")
        self.cron = self.root / "crontab.txt"
        if crontab is not None:
            self.cron.write_text(crontab)
        stubs = {"systemctl": STUB_SYSTEMCTL, "crontab": STUB_CRONTAB,
                 "git": STUB_LOGGER % ("git", 1), "hermes": STUB_LOGGER % ("hermes", 0),
                 "curl": STUB_LOGGER % ("curl", 7), "wget": STUB_LOGGER % ("wget", 4),
                 "loginctl": "#!/bin/sh\necho Linger=no\n",
                 "pip": STUB_LOGGER % ("pip", 1), "pip3": STUB_LOGGER % ("pip3", 1)}
        if uv:
            stubs["uv"] = STUB_UV
        for name in ("python3", "python3.11", "python3.12", "python"):
            stubs[name] = STUB_PY
        for name, body in stubs.items():
            p = self.bin / name
            p.write_text(body)
            p.chmod(0o755)
        self.systemd = systemd
        self.protected = {p: _sha(p) for p in (self.hh / ".env", self.hh / "config.yaml",
                                              self.hh / "memories" / "USER.md",
                                              self.hh / "memories" / "MEMORY.md")}
        if state_db:
            self.protected[self.hh / "state.db"] = _sha(self.hh / "state.db")

    @property
    def venv(self) -> Path:
        return self.home / ".local" / "share" / "hermesyume" / "venv"

    def env(self) -> dict[str, str]:
        return {
            "HOME": str(self.home), "PATH": f"{self.bin}:/usr/bin:/bin", "TMPDIR": str(self.tmp),
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "USER": "sandbox", "LOGNAME": "sandbox",
            "HERMESYUME_SRC": str(REPO), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
            "STUB_LOG": str(self.log), "STUB_REAL_PY": sys.executable,
            "STUB_CRONTAB_FILE": str(self.cron), "STUB_SYSTEMD_OK": "0" if self.systemd else "1",
            "HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9",
        }

    def run(self, *args: str) -> subprocess.CompletedProcess:
        cmd = ["bash", str(INSTALL), *args]
        if BWRAP:
            cmd = ["bwrap", "--ro-bind", "/", "/", "--bind", str(self.root), str(self.root),
                   "--dev", "/dev", "--proc", "/proc", "--unshare-net", "--die-with-parent",
                   "--chdir", str(self.root), *cmd]
        return subprocess.run(cmd, env=self.env(), cwd=str(self.root), capture_output=True,
                              text=True, timeout=600)

    def log_lines(self) -> list[str]:
        return self.log.read_text().splitlines()

    def assert_contained(self) -> None:
        """Hermes-owned files unchanged; every absolute path a stub saw lies in the sandbox."""
        for p, h in self.protected.items():
            assert _sha(p) == h, f"{p} was modified"
        for line in self.log_lines():
            for tok in line.split():
                tok = tok.strip("'\"")
                if not tok.startswith("/") or tok == sys.executable:
                    continue
                if line.startswith("git ") and tok == str(REPO):   # install_provider.sh: read-only rev-parse
                    continue
                assert tok.startswith(str(self.root)), f"path outside the sandbox: {line}"
        assert not [ln for ln in self.log_lines() if ln.startswith("hermes ")], "hermes CLI was called"
        assert not [ln for ln in self.log_lines()
                    if ln.startswith("git ") and any(w in ln.split() for w in ("clone", "fetch", "pull", "merge"))]
        assert not [ln for ln in self.log_lines() if ln.startswith(("curl ", "wget ", "pip "))]


def _outside_snapshot() -> dict[str, tuple]:
    real_home = Path(os.path.expanduser("~"))
    out = {}
    for p in (real_home / ".local" / "share" / "hermesyume", real_home / "HermesYume",
              real_home / ".config" / "systemd" / "user" / "hermesyume-dream.service",
              real_home / ".config" / "systemd" / "user" / "hermesyume-dream.timer",
              REPO / "build", REPO / "dist"):
        out[str(p)] = (p.exists(), p.stat().st_mtime_ns if p.exists() else None)
    out["repo-top"] = tuple(sorted(os.listdir(REPO)))
    return out


def test_install_sh_is_valid_bash():
    assert os.access(INSTALL, os.X_OK)
    subprocess.run(["bash", "-n", str(INSTALL)], check=True)
    text = INSTALL.read_text(encoding="utf-8")
    # the one-liner in the header and README must match the published path
    assert "raw.githubusercontent.com/EESIZ/HermesYume/main/install.sh | bash -s -- --timer" in text
    assert str(Path.home()) not in text and str(REPO.parent) not in text     # no machine paths


def test_install_uv_systemd_idempotent(tmp_path):
    before = _outside_snapshot()
    sb = Sandbox(tmp_path, uv=True, systemd=True)
    r = sb.run("--timer")
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    # never activated: the exact command is printed, shadow mode recommended first
    assert "hermes config set memory.provider hermesyume" in out
    assert "config set inject false" in out
    data = sb.hh / "hermesyume"
    cfg = json.loads((data / "config.json").read_text(encoding="utf-8"))
    assert cfg["embed_provider"] == "hash" and cfg["embed_dim"] == 1024      # no OpenAI key → hash
    assert cfg["llm_provider"] == "auto" and cfg["recall_min_cos"] == 0.30
    assert cfg["workspace_dir"] == "" and cfg["md_sources"] == []            # never this deployment's paths
    assert (data / "ledger.db").exists() and (data / "lancedb").is_dir()
    plug = sb.hh / "plugins" / "hermesyume"
    assert (plug / "__init__.py").exists() and (plug / "_yume" / "hash_embed.py").exists()
    assert (sb.venv / "bin" / "yume").exists()
    unit_dir = sb.home / ".config" / "systemd" / "user"
    svc = (unit_dir / "hermesyume-dream.service").read_text(encoding="utf-8")
    assert f'Environment="HERMES_HOME={sb.hh}"' in svc
    assert f'ExecStart="{sb.venv}/bin/yume" dream' in svc
    assert (unit_dir / "hermesyume-dream.timer").read_text() == \
        (REPO / "deploy" / "hermesyume-dream.timer").read_text()
    log = sb.log_lines()
    assert "systemctl --user daemon-reload" in log
    assert "systemctl --user enable --now hermesyume-dream.timer" in log
    assert not [ln for ln in log if ln.startswith("crontab ")]
    sb.assert_contained()

    # re-run: the operator's config is kept, nothing is duplicated
    cfg["inject"] = False
    (data / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    cfg_sha = _sha(data / "config.json")
    r2 = sb.run("--timer")
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert _sha(data / "config.json") == cfg_sha
    assert "기존 유지" in r2.stdout
    assert (sb.hh / "hermesyume" / "staging" / "provider.prev").is_dir()
    assert not (unit_dir / "hermesyume-dream.service.bak").exists()     # identical units: no backup
    sb.assert_contained()
    assert _outside_snapshot() == before


def test_install_pip_fallback_and_cron(tmp_path):
    before = _outside_snapshot()
    existing = "# my jobs\n15 * * * * /usr/bin/true\n"
    sb = Sandbox(tmp_path, uv=False, systemd=False, crontab=existing)
    r = sb.run("--timer")
    assert r.returncode == 0, r.stdout + r.stderr
    log = sb.log_lines()
    assert any(ln.startswith("python") and "-m venv" in ln for ln in log)
    assert any("-m pip install -q --no-deps --force-reinstall" in ln for ln in log)
    tab = sb.cron.read_text()
    assert tab.startswith(existing)
    ours = [ln for ln in tab.splitlines() if ln.endswith("# hermesyume-dream")]
    assert len(ours) == 1 and f"HERMES_HOME='{sb.hh}'" in ours[0] and "/bin/yume' dream" in ours[0]
    assert (sb.hh / "hermesyume" / "crontab.bak").read_text() == existing
    assert not (sb.home / ".config" / "systemd").exists()
    r2 = sb.run("--timer")
    assert r2.returncode == 0, r2.stdout + r2.stderr
    tab2 = sb.cron.read_text()
    assert tab2.startswith(existing) and len([ln for ln in tab2.splitlines()
                                              if ln.endswith("# hermesyume-dream")]) == 1
    sb.assert_contained()
    assert _outside_snapshot() == before


def test_install_refuses_without_state_db(tmp_path):
    sb = Sandbox(tmp_path, uv=True, systemd=True, state_db=False)
    r = sb.run()
    assert r.returncode != 0 and "state.db not found" in r.stderr
    assert not (sb.home / ".local").exists() and not (sb.hh / "plugins").exists()
    assert not (sb.hh / "hermesyume").exists()
    assert sb.log_lines() == []
    sb.assert_contained()


@pytest.mark.skipif(not BWRAP, reason="bubblewrap not usable here")
def test_sandbox_really_blocks_outside_writes(tmp_path):
    """Guard for the guard: the bwrap wrapper used above does reject a write outside the sandbox."""
    sb = Sandbox(tmp_path, uv=True, systemd=True)
    outside = REPO / ".install-sh-probe"
    r = subprocess.run(["bwrap", "--ro-bind", "/", "/", "--bind", str(sb.root), str(sb.root),
                        "--dev", "/dev", "--proc", "/proc", "--unshare-net",
                        "bash", "-c", f"touch {outside}"], capture_output=True, text=True)
    assert r.returncode != 0 and not outside.exists()
