#!/usr/bin/env bash
# HermesYume installer -- run on the machine where Hermes Agent lives.
#
#   bash install.sh            # install + doctor
#   bash install.sh --cron     # ... and register a nightly 03:00 cron job
#
# Idempotent: re-running updates the code and never overwrites your .env.

set -euo pipefail

REPO_URL="https://github.com/EESIZ/HermesYume.git"
INSTALL_DIR="${HERMESYUME_DIR:-$HOME/HermesYume}"
export HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
export HERMESYUME_HOME="${HERMESYUME_HOME:-$HOME/.hermesyume}"
CRON=0
[ "${1:-}" = "--cron" ] && CRON=1

echo "HERMES_HOME:    $HERMES_HOME"
echo "HERMESYUME_HOME: $HERMESYUME_HOME"
echo "install dir:    $INSTALL_DIR"

if [ ! -f "$HERMES_HOME/state.db" ]; then
    echo "!! $HERMES_HOME/state.db not found -- is Hermes installed for this user?"
    echo "   (set HERMES_HOME to your Hermes home or profile directory)"
    exit 1
fi

# 1. code
if [ -d "$INSTALL_DIR/.git" ]; then
    git -C "$INSTALL_DIR" pull --ff-only -q
elif [ -f "$(dirname "$0")/hermesyume.py" ] && [ "$(cd "$(dirname "$0")" && pwd)" = "$INSTALL_DIR" ]; then
    :  # running from the install dir itself
else
    git clone -q "$REPO_URL" "$INSTALL_DIR"
fi

# 2. venv
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
    || { echo "!! Python 3.10+ required"; exit 1; }
[ -x "$INSTALL_DIR/.venv/bin/python" ] || python3 -m venv "$INSTALL_DIR/.venv"
"$INSTALL_DIR/.venv/bin/pip" install -q -r "$INSTALL_DIR/requirements.txt"

# 3. env file (never overwritten)
mkdir -p "$HERMESYUME_HOME/dream-log"
ENV_FILE="$HERMESYUME_HOME/.env"
if [ ! -f "$ENV_FILE" ]; then
    cp "$INSTALL_DIR/.env.example" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo
    echo ">> Created $ENV_FILE"
    if grep -qE '^(export )?(DEEPSEEK|OPENAI)_API_KEY=.+' "$HERMES_HOME/.env" 2>/dev/null; then
        echo "   API key will be taken from $HERMES_HOME/.env -- nothing to fill in."
    else
        echo "   Put an API key in it (DEEPSEEK_API_KEY or OPENAI_API_KEY)."
    fi
fi

# 4. doctor
echo
set -a; . "$ENV_FILE"; set +a
"$INSTALL_DIR/.venv/bin/python" "$INSTALL_DIR/doctor.py" || true

# 5. cron
if [ "$CRON" = 1 ]; then
    LINE="0 3 * * * HERMES_HOME=$HERMES_HOME HERMESYUME_HOME=$HERMESYUME_HOME $INSTALL_DIR/examples/run-hermesyume.sh"
    # Build the new crontab in a file first: a failure halfway through a pipe
    # into `crontab -` would otherwise replace the user's crontab with nothing.
    OLD_CRON="$HERMESYUME_HOME/crontab.bak"
    NEW_CRON="$(mktemp)"
    crontab -l > "$OLD_CRON" 2>/dev/null || : > "$OLD_CRON"
    { grep -v 'run-hermesyume.sh' "$OLD_CRON" || true; echo "$LINE"; } > "$NEW_CRON"
    crontab "$NEW_CRON"
    rm -f "$NEW_CRON"
    echo
    echo ">> cron registered: $LINE"
    echo "   (previous crontab saved to $OLD_CRON)"
fi

echo
echo "Next: $INSTALL_DIR/.venv/bin/python $INSTALL_DIR/hermesyume.py --dry-run -v"
