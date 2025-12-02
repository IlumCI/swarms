#!/usr/bin/env bash
set -euo pipefail
# Simple local runner for research use
# Usage: ./run_local.sh

# load env if present
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

PYTHON=${PYTHON:-python3}
PORT=${PORT:-8765}
DB_PATH=${DB_PATH:-time_engine.db}

echo "Starting Time Engine on port ${PORT} with DB ${DB_PATH}"
${PYTHON} main.py --host 0.0.0.0 --port ${PORT} --db ${DB_PATH}


