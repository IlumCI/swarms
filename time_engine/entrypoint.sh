#!/usr/bin/env bash
set -euo pipefail
# Optional: source .env if present
ENV_FILE="$(dirname "$0")/env_example.txt"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
DB_PATH="${DB_PATH:-time_engine.db}"

echo "Starting time_engine server on ${HOST}:${PORT} (DB=${DB_PATH})"
exec python3 "$(dirname "$0")/main.py" --host "$HOST" --port "$PORT" --db "$DB_PATH"


