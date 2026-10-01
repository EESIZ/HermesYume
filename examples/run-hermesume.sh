#!/bin/bash
# Hermesume daily run wrapper
# Cron example: 0 3 * * * /path/to/hermesume/examples/run-hermesume.sh

set -euo pipefail

export HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
export HERMESUME_HOME="${HERMESUME_HOME:-$HOME/.hermesume}"

# Load API keys from .env
if [ -f "$HOME/.env" ]; then
    set -a
    source "$HOME/.env"
    set +a
fi

mkdir -p "$HERMESUME_HOME/dream-log"
cd "$(dirname "$0")/.."
python3 hermesume.py --verbose >> "$HERMESUME_HOME/dream-log/cron.log" 2>&1
