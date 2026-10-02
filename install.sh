#!/usr/bin/env bash
# HermesYume v2 installer — run it as the user that runs Hermes, on the machine where Hermes lives.
#
#   curl -fsSL https://raw.githubusercontent.com/EESIZ/HermesYume/main/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/EESIZ/HermesYume/main/install.sh | bash -s -- --timer
#
# Idempotent: re-running updates the code, the venv and the provider files and changes nothing else.
#   1. checks $HERMES_HOME/state.db                      (HERMES_HOME defaults to ~/.hermes)
#   2. clones or fast-forwards the code into ~/HermesYume ($HERMESYUME_DIR, branch $HERMESYUME_REF)
#   3. nightly venv ~/.local/share/hermesyume/venv ($HERMESYUME_VENV): uv if present, else
#      python3 -m venv + pip; locked deps (requirements-dream.txt) + the package
#   4. deploy/install_provider.sh → $HERMES_HOME/plugins/hermesyume (stdlib only; NOT activated)
#   5. yume init (an existing config.json is kept as is) and yume doctor
#   6. --timer: systemd user timer (04:40 Asia/Seoul); without a user systemd, a crontab line
#      (the old crontab is backed up first and the new one is built in a temp file)
# It never runs `hermes config set memory.provider`, never edits config.yaml or $HERMES_HOME/.env,
# never restarts Hermes. Activation is yours: shadow mode first (printed at the end).
#
# Options: --timer  --hermes-home DIR  -h|--help
# Env:     HERMES_HOME, HERMESYUME_DIR, HERMESYUME_REF (default main), HERMESYUME_VENV,
#          HERMESYUME_SRC=<checkout>  use that local v2 checkout as is (no git, no network for code)
set -euo pipefail

main() {
  local REPO_URL="https://github.com/EESIZ/HermesYume.git"
  local TIMER=0
  local HH="${HERMES_HOME:-$HOME/.hermes}"
  local DIR="${HERMESYUME_DIR:-$HOME/HermesYume}"
  local REF="${HERMESYUME_REF:-main}"
  local VENV="${HERMESYUME_VENV:-$HOME/.local/share/hermesyume/venv}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --timer|--cron) TIMER=1; shift ;;
      --hermes-home) HH="${2:?--hermes-home needs a directory}"; shift 2 ;;
      -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' \
                   "${BASH_SOURCE[0]:-/dev/null}" 2>/dev/null | grep . \
                   || echo "usage: install.sh [--timer] [--hermes-home DIR]"; return 0 ;;
      *) die "unknown option: $1 (try --help)" ;;
    esac
  done

  # ── 1. Hermes home ──
  [[ -f "$HH/state.db" ]] || die "$HH/state.db not found — is Hermes installed for this user?
   Set HERMES_HOME (or --hermes-home) to your Hermes home or profile directory."
  HH="$(cd "$HH" && pwd)"
  export HERMES_HOME="$HH"
  say "HERMES_HOME  $HH"

  # ── 2. code ──
  local SRC
  if [[ -n "${HERMESYUME_SRC:-}" ]]; then
    SRC="$(cd "$HERMESYUME_SRC" && pwd)"
    say "source       $SRC (HERMESYUME_SRC: used as is, no git)"
  else
    command -v git >/dev/null 2>&1 || die "git is required"
    if [[ -d "$DIR/.git" ]]; then
      git -C "$DIR" fetch -q origin "$REF" </dev/null
      git -C "$DIR" checkout -q "$REF" </dev/null 2>/dev/null \
        || git -C "$DIR" checkout -q -b "$REF" "origin/$REF" </dev/null
      git -C "$DIR" merge -q --ff-only "origin/$REF" </dev/null \
        || die "$DIR has local changes or diverged from origin/$REF — update it yourself, then re-run"
    elif [[ -e "$DIR" ]]; then
      die "$DIR exists and is not a git checkout (set HERMESYUME_DIR to another place)"
    else
      git clone -q --branch "$REF" "$REPO_URL" "$DIR" </dev/null
    fi
    SRC="$(cd "$DIR" && pwd)"
    say "source       $SRC ($(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo '?'))"
  fi
  for f in pyproject.toml requirements-dream.txt hermesyume/cli.py provider/__init__.py \
           deploy/install_provider.sh deploy/hermesyume-dream.service deploy/hermesyume-dream.timer; do
    [[ -f "$SRC/$f" ]] || die "$SRC is not a HermesYume v2 checkout ($f missing)"
  done

  # ── 3. nightly venv (never Hermes' own venv, never inside HERMES_HOME) ──
  VENV="$(abspath "$VENV")"
  case "$VENV/" in "$HH/"*) die "the venv must not live inside HERMES_HOME: $VENV" ;; esac
  if [[ -e "$VENV/bin/hermes" ]] || compgen -G "$VENV/lib/python*/site-packages/hermes_cli" >/dev/null; then
    die "$VENV contains Hermes itself — HermesYume never installs into Hermes' venv"
  fi
  if [[ -e "$VENV" && ! -f "$VENV/pyvenv.cfg" ]]; then
    die "$VENV exists but is not a venv (no pyvenv.cfg) — not touching it"
  fi
  local TMP
  TMP="$(mktemp -d "${TMPDIR:-/tmp}/hermesyume-install.XXXXXX")"
  # shellcheck disable=SC2064
  trap "rm -rf '$TMP'" EXIT
  # build from a clean copy so pip/uv never write build/ or *.egg-info into the checkout
  mkdir -p "$TMP/src"
  (cd "$SRC" && tar --exclude=.git --exclude=.venv --exclude='__pycache__' --exclude='*.pyc' \
      --exclude='*.egg-info' --exclude=docs-local --exclude=.pytest_cache --exclude=.ruff_cache \
      --exclude=build --exclude=dist -cf - .) | tar -xf - -C "$TMP/src"
  if command -v uv >/dev/null 2>&1; then
    if [[ ! -x "$VENV/bin/python" ]]; then
      mkdir -p "$(dirname "$VENV")"
      uv venv -q --python '>=3.11,<3.13' "$VENV" </dev/null
    fi
    py_ok "$VENV/bin/python" || die "$VENV uses an unsupported Python (need 3.11 or 3.12) — remove it and re-run"
    uv pip install -q --python "$VENV/bin/python" -r "$TMP/src/requirements-dream.txt" </dev/null
    uv pip install -q --python "$VENV/bin/python" --no-deps --reinstall "$TMP/src" </dev/null
  else
    if [[ ! -x "$VENV/bin/python" ]]; then
      local PY="" c
      for c in python3.12 python3.11 python3; do
        if command -v "$c" >/dev/null 2>&1 && py_ok "$c"; then PY="$c"; break; fi
      done
      [[ -n "$PY" ]] || die "Python 3.11 or 3.12 not found (install one, or install uv: https://docs.astral.sh/uv/)"
      mkdir -p "$(dirname "$VENV")"
      "$PY" -m venv "$VENV" </dev/null
    fi
    py_ok "$VENV/bin/python" || die "$VENV uses an unsupported Python (need 3.11 or 3.12) — remove it and re-run"
    "$VENV/bin/python" -m pip install -q -r "$TMP/src/requirements-dream.txt" </dev/null
    "$VENV/bin/python" -m pip install -q --no-deps --force-reinstall "$TMP/src" </dev/null
  fi
  local Y="$VENV/bin/yume"
  [[ -x "$Y" ]] || die "$Y was not installed"
  say "venv         $VENV"

  # ── 4. provider files (copied, not activated) ──
  bash "$SRC/deploy/install_provider.sh" --hermes-home "$HH" --yes </dev/null >/dev/null
  say "provider     $HH/plugins/hermesyume (installed, not activated)"

  # ── 5. data dir + config (kept if it exists) + checks ──
  local had_cfg=0
  [[ -f "$HH/hermesyume/config.json" ]] && had_cfg=1
  "$Y" init </dev/null
  if [[ $had_cfg == 0 && "$VENV" != "$HOME/.local/share/hermesyume/venv" ]]; then
    "$Y" config set yume_bin "$VENV/bin/yume" </dev/null >/dev/null   # only in the config just created
  fi
  echo
  "$Y" doctor </dev/null || true
  echo

  # ── 6. nightly schedule ──
  if [[ $TIMER == 1 ]]; then
    install_schedule "$SRC" "$HH" "$VENV" "$TMP"
  else
    say "no schedule installed (re-run with --timer for the nightly 04:40 Asia/Seoul run)"
  fi

  cat <<MSG

Done. The provider is installed but not active — Hermes behaves exactly as before.

Next steps (yours to run):
  Y=$Y
  export HERMES_HOME=$HH
  \$Y migrate --estimate && \$Y migrate --dry-run     # optional: bring in what Hermes already knows
  \$Y migrate --approve-migration                     #   (read dream-log/*_dry.md first)
  \$Y config set inject false                         # shadow mode first: recall is computed and logged, nothing injected
  hermes config set memory.provider hermesyume       # activate the provider (takes effect on the next message)
  # a few nights later:
  \$Y calibrate && \$Y config set inject true

Stop any time: \$Y config set enabled false   ·   remove: delete the memory.provider line (data stays)
After a code update, restart the Hermes gateway yourself (Python caches the provider modules).
MSG
}

say() { printf '%s\n' "$*"; }
die() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }

abspath() {
  local p="${1/#\~/$HOME}"
  case "$p" in /*) ;; *) p="$PWD/$p" ;; esac
  printf '%s' "$p"
}

py_ok() {   # $1 = python executable: 3.11 <= version < 3.13
  "$1" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)' </dev/null 2>/dev/null
}

shq() {     # single-quote for /bin/sh (cron)
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

install_schedule() {
  local SRC="$1" HH="$2" VENV="$3" TMP="$4"
  if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1 </dev/null; then
    local UD="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
    mkdir -p "$UD"
    # the repo units, with this HERMES_HOME and venv filled in (% escaped for systemd)
    local env_line="Environment=\"HERMES_HOME=${HH//%/%%}\""
    local exec_line="ExecStart=\"${VENV//%/%%}/bin/yume\" dream"
    awk -v env="$env_line" -v exe="$exec_line" '
      /^Environment=HERMES_HOME=/ { print env; next }
      /^ExecStart=/ { print exe; next }
      { print }' "$SRC/deploy/hermesyume-dream.service" > "$TMP/hermesyume-dream.service"
    cp "$SRC/deploy/hermesyume-dream.timer" "$TMP/hermesyume-dream.timer"
    local u
    for u in hermesyume-dream.service hermesyume-dream.timer; do
      if [[ -f "$UD/$u" ]] && ! cmp -s "$TMP/$u" "$UD/$u"; then
        cp -p "$UD/$u" "$UD/$u.bak"
        say "kept your previous $u as $UD/$u.bak"
      fi
      cp "$TMP/$u" "$UD/$u"
    done
    systemctl --user daemon-reload </dev/null
    systemctl --user enable --now hermesyume-dream.timer </dev/null
    say "timer        systemd user timer hermesyume-dream.timer (04:40 Asia/Seoul; change: systemctl --user edit hermesyume-dream.timer)"
    if command -v loginctl >/dev/null 2>&1 \
        && ! loginctl show-user "$(id -un)" --property=Linger 2>/dev/null </dev/null | grep -q 'Linger=yes'; then
      say "             hint: user timers stop when you log out unless lingering is on: loginctl enable-linger $(id -un)"
    fi
  elif command -v crontab >/dev/null 2>&1; then
    # same moment as the systemd timer (04:40 Asia/Seoul) in this machine's local time
    local when
    when="$(date -d 'TZ="Asia/Seoul" 04:40' '+%M %H' 2>/dev/null | sed 's/^0\([0-9]\)/\1/; s/ 0\([0-9]\)$/ \1/')" || when=""
    [[ -n "$when" ]] || when="40 4"
    local line
    line="$when * * * HERMES_HOME=$(shq "$HH") $(shq "$VENV/bin/yume") dream >/dev/null 2>&1 # hermesyume-dream"
    line="${line//%/\\%}"
    local data="$HH/hermesyume" old new err
    old="$data/crontab.bak"
    new="$TMP/crontab.new"
    err="$TMP/crontab.err"
    # back up the current crontab first; build the new one in a file (a failure halfway through a
    # pipe into `crontab -` would otherwise replace the user's crontab with nothing)
    if crontab -l > "$TMP/crontab.cur" 2> "$err" </dev/null; then
      cp "$TMP/crontab.cur" "$old"
    elif grep -qi 'no crontab' "$err"; then
      : > "$old"
    else
      die "crontab -l failed ($(head -c 200 "$err")) — not touching your crontab"
    fi
    chmod 600 "$old"
    { grep -v '# hermesyume-dream$' "$old" || true; printf '%s\n' "$line"; } > "$new"
    crontab "$new" </dev/null
    say "timer        crontab: $line"
    say "             (previous crontab saved to $old; cron uses this machine's local time)"
  else
    say "WARNING: no user systemd and no crontab — schedule \`$VENV/bin/yume dream\` (HERMES_HOME=$HH) yourself"
  fi
}

main "$@"
