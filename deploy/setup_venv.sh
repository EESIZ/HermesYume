#!/usr/bin/env bash
# HermesYume v2 — separate venv for `yume dream` and the `yume` admin CLI (PLAN-v2 §8.2).
#
#   deploy/setup_venv.sh [--venv DIR] [--ref TAG|COMMIT] [--python 3.11] [--force] [--dry-run]
#   deploy/setup_venv.sh --from-worktree --venv DIR        # development only
#
# - The venv lives OUTSIDE $HERMES_HOME (default ~/.local/share/hermesyume/venv), so Hermes backups
#   do not drag it along, and the Hermes venv is never touched (refused if it looks like one).
# - Locked dependencies come from requirements-dream.txt (uv pip compile).
# - The package itself is installed NON-editable from a tagged commit (git archive), so code you are
#   still editing in the work tree never reaches the nightly run by accident.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${HERMESYUME_VENV:-$HOME/.local/share/hermesyume/venv}"
REF=""
PYVER="3.11"
FROM_WORKTREE=0
FORCE=0
DRY=0

die() { echo "setup_venv: $*" >&2; exit 1; }
run() { if [[ $DRY == 1 ]]; then printf '[dry-run]'; printf ' %q' "$@"; echo; else "$@"; fi; }
usage() { sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --python) PYVER="$2"; shift 2 ;;
    --from-worktree) FROM_WORKTREE=1; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "알 수 없는 옵션: $1 (--help)" ;;
  esac
done

command -v uv >/dev/null 2>&1 || die "uv가 필요합니다 (https://docs.astral.sh/uv/)"
VENV="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$VENV")"

# ── never the Hermes venv ──
if [[ -e "$VENV/bin/hermes" ]] || compgen -G "$VENV/lib/python*/site-packages/hermes_cli" >/dev/null; then
  die "$VENV 에 Hermes가 설치돼 있습니다. Hermes venv에는 설치하지 않습니다 (§8.1 A 기각)."
fi
if [[ -n "${HERMES_HOME:-}" ]]; then
  case "$VENV/" in "$(cd "$HERMES_HOME" 2>/dev/null && pwd)/"*) die "venv를 HERMES_HOME 안에 두지 않습니다: $VENV" ;; esac
fi

# ── source: tagged commit (default) or the work tree (dev) ──
TMP="$(mktemp -d "${TMPDIR:-/tmp}/hermesyume-src.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
SRC="$TMP/src"
mkdir -p "$SRC"
if [[ $FROM_WORKTREE == 1 ]]; then
  echo "주의: 작업 트리에서 설치합니다 (개발용). 라이브에는 태그를 쓰세요." >&2
  (cd "$REPO" && tar --exclude=.git --exclude=.venv --exclude='__pycache__' --exclude='*.egg-info' \
       --exclude=docs-local --exclude=.pytest_cache --exclude=build --exclude=dist -cf - .) | tar -xf - -C "$SRC"
  DESC="worktree $(git -C "$REPO" rev-parse --short=12 HEAD 2>/dev/null || echo nogit)"
else
  if [[ -z "$REF" ]]; then
    REF="$(git -C "$REPO" describe --tags --exact-match HEAD 2>/dev/null)" \
      || die "HEAD에 태그가 없습니다. --ref <태그>로 설치할 커밋을 지정하세요."
  fi
  SHA="$(git -C "$REPO" rev-parse --verify "${REF}^{commit}")" || die "알 수 없는 ref: $REF"
  git -C "$REPO" archive --format=tar "$SHA" | tar -xf - -C "$SRC"
  DESC="$REF ($SHA)"
fi
[[ -f "$SRC/pyproject.toml" && -f "$SRC/requirements-dream.txt" && -d "$SRC/hermesyume" && -d "$SRC/provider" ]] \
  || die "$DESC 에 v2 패키지가 없습니다 (pyproject.toml / requirements-dream.txt / hermesyume / provider)"

# ── venv ──
if [[ -d "$VENV" && $FORCE == 1 ]]; then
  [[ -f "$VENV/pyvenv.cfg" ]] || die "$VENV 는 venv가 아닙니다 (pyvenv.cfg 없음) — 지우지 않습니다"
  run rm -rf "$VENV"
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  run mkdir -p "$(dirname "$VENV")"
  run uv venv --python "$PYVER" "$VENV"
fi
run uv pip install --python "$VENV/bin/python" -r "$SRC/requirements-dream.txt"
run uv pip install --python "$VENV/bin/python" --no-deps --reinstall "$SRC"
if [[ $DRY == 0 ]]; then
  printf 'source %s\ninstalled %s\n' "$DESC" "$(date -Iseconds)" > "$VENV/hermesyume-install.txt"
  "$VENV/bin/yume" --version
fi

cat <<MSG
완료: $VENV ($DESC)
다음 (승인 후, 직접):
  HERMES_HOME=<hermes home> $VENV/bin/yume doctor
  HERMES_HOME=<hermes home> $VENV/bin/yume init
  HERMES_HOME=<hermes home> $VENV/bin/yume migrate --dry-run
  deploy/install_provider.sh --hermes-home <hermes home>
MSG
