#!/usr/bin/env bash
# HermesYume v2 — install the stdlib-only memory provider into $HERMES_HOME/plugins/hermesyume
# (PLAN-v2 §8.2).
#
#   deploy/install_provider.sh --hermes-home H [--yes] [--dry-run]
#
# - Copies provider/ to H/hermesyume/staging/provider.<sha> with a VERSION file, then swaps it in
#   with a rename on the same filesystem (a half-copied folder is never importable).
# - The previous plugin is kept as H/hermesyume/staging/provider.prev (rollback: swap it back).
# - Never touches config.yaml, never restarts anything. Activation and restarts are the
#   operator's call (§8.3): `hermes config set memory.provider hermesyume`; after a code update
#   restart the gateway/dashboard yourself (Python caches imported modules).
# - A home that a gateway has run in (gateway.pid / gateway_state.json / gateway.lock) needs --yes.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/provider"
HH="${HERMES_HOME:-}"
YES=0
DRY=0

die() { echo "install_provider: $*" >&2; exit 1; }
run() { if [[ $DRY == 1 ]]; then printf '[dry-run]'; printf ' %q' "$@"; echo; else "$@"; fi; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hermes-home) HH="$2"; shift 2 ;;
    --yes) YES=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "알 수 없는 옵션: $1 (--help)" ;;
  esac
done

[[ -n "$HH" ]] || die "--hermes-home이 필요합니다"
[[ -d "$HH" ]] || die "HERMES_HOME이 없습니다: $HH"
HH="$(cd "$HH" && pwd)"

# ── provider sanity (§6.1, §8.2) ──
for f in __init__.py plugin.yaml cli.py _yume/__init__.py; do
  [[ -f "$SRC/$f" ]] || die "provider/$f 없음"
done
# (process substitution, not a pipe: with pipefail, head dying of SIGPIPE after grep -q matched
#  would fail the check)
grep -q "MemoryProvider" < <(head -c 8192 "$SRC/__init__.py") || die "__init__.py 앞 8,192자에 MemoryProvider가 없습니다"
grep -Eq '^name:[[:space:]]*hermesyume[[:space:]]*$' "$SRC/plugin.yaml" || die "plugin.yaml name이 hermesyume가 아닙니다"
if grep -Eq '^[[:space:]]*pip_dependencies' "$SRC/plugin.yaml"; then
  die "plugin.yaml에 pip_dependencies가 있습니다 (Hermes venv에 설치하지 않음, §8.1)"
fi

# ── live-home guard ──
for m in gateway.pid gateway_state.json gateway.lock; do
  if [[ -e "$HH/$m" && $YES != 1 ]]; then
    die "$HH 는 게이트웨이가 쓰는 홈으로 보입니다 ($m). 승인을 받았다면 --yes를 붙이세요."
  fi
done

VERSION="$(sed -n 's/^version:[[:space:]]*//p' "$SRC/plugin.yaml" | head -1)"
SHA="$(git -C "$REPO" rev-parse --short=12 HEAD 2>/dev/null || echo nogit)"
if [[ "$SHA" != nogit && -n "$(git -C "$REPO" status --porcelain -- provider 2>/dev/null)" ]]; then
  SHA="$SHA-dirty"
fi
DATA="$HH/hermesyume"
STAGING="$DATA/staging"
PLUGINS="$HH/plugins"
DEST="$PLUGINS/hermesyume"
NEW="$STAGING/provider.$SHA"
PREV="$STAGING/provider.prev"

run mkdir -p "$STAGING" "$PLUGINS"
run chmod 700 "$DATA" "$STAGING"
if [[ $DRY == 0 ]]; then
  rm -rf "$NEW"
  mkdir -p "$NEW"
  (cd "$SRC" && tar --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pyo' -cf - .) | tar -xf - -C "$NEW"
  printf 'hermesyume-provider %s\ngit %s\ninstalled %s\n' "$VERSION" "$SHA" "$(date -Iseconds)" > "$NEW/VERSION"
  chmod -R go-w "$NEW"
  [[ "$(stat -c %d "$STAGING")" == "$(stat -c %d "$PLUGINS")" ]] \
    || die "staging과 plugins가 다른 파일시스템입니다 (원자적 교체 불가)"
else
  echo "[dry-run] copy $SRC -> $NEW (+VERSION $VERSION/$SHA)"
fi

if [[ -e "$DEST" || -L "$DEST" ]]; then
  run rm -rf "$PREV"
  if grep -q -- '--exchange' < <(mv --help 2>/dev/null); then
    run mv --exchange -T "$NEW" "$DEST"      # atomic swap (renameat2 RENAME_EXCHANGE)
    run mv -T "$NEW" "$PREV"
  else
    run mv -T "$DEST" "$PREV"
    run mv -T "$NEW" "$DEST"
  fi
else
  run mv -T "$NEW" "$DEST"
fi

cat <<MSG
설치: $DEST (hermesyume-provider $VERSION, git $SHA)
이전 버전: $( [[ -e "$PREV" ]] && echo "$PREV" || echo 없음 )
config.yaml은 바꾸지 않았습니다.
- 처음 켤 때(승인 후): hermes config set memory.provider hermesyume  (다음 메시지부터 적용)
- 코드 갱신 후: 게이트웨이·대시보드 재시작은 직접 (모듈 캐시)
- 즉시 정지: $DATA/config.json 의 "enabled": false
MSG
