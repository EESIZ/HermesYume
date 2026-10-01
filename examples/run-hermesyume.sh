#!/bin/bash
# HermesYume daily run wrapper
# Cron example: 0 3 * * * /path/to/hermesyume/examples/run-hermesyume.sh

set -euo pipefail

export HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
export HERMESYUME_HOME="${HERMESYUME_HOME:-$HOME/.hermesyume}"

# Load API keys from .env
if [ -f "$HOME/.env" ]; then
    set -a
    source "$HOME/.env"
    set +a
fi

mkdir -p "$HERMESYUME_HOME/dream-log"
cd "$(dirname "$0")/.."
python3 hermesyume.py --verbose >> "$HERMESYUME_HOME/dream-log/cron.log" 2>&1
