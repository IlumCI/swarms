#!/usr/bin/env bash
set -euo pipefail
IN=${1:-}
if [ -z "$IN" ]; then
  echo "Usage: $0 backup-file.db.gz"
  exit 1
fi
OUT=${DB_PATH:-time_engine.db}
echo "Restoring ${IN} -> ${OUT}"
gunzip -c "$IN" > "$OUT"
echo "Restore complete."


