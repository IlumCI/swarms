#!/usr/bin/env bash
set -euo pipefail
DB=${DB_PATH:-time_engine.db}
OUT=${1:-backup-$(date +%Y%m%dT%H%M%SZ).db.gz}
echo "Backing up ${DB} -> ${OUT}"
gzip -c "${DB}" > "${OUT}"
echo "Backup complete: ${OUT}"


