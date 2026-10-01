#!/bin/bash
# HermesYume daily run wrapper
# Cron example: 0 3 * * * /path/to/HermesYume/examples/run-hermesyume.sh

set -euo pipefail

export HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
export HERMESYUME_HOME="${HERMESYUME_HOME:-$HOME/.hermesyume}"

# Load API keys: $HERMESYUME_HOME/.env (install.sh), falling back to ~/.env
for f in "$HERMESYUME_HOME/.env" "$HOME/.env"; do
    if [ -f "$f" ]; then
        set -a; source "$f"; set +a
        break
    fi
done

cd "$(dirname "$0")/.."
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python

mkdir -p "$HERMESYUME_HOME/dream-log"
"$PY" hermesyume.py --verbose >> "$HERMESYUME_HOME/dream-log/cron.log" 2>&1
