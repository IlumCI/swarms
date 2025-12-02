#!/usr/bin/env bash
set -euo pipefail
echo "Running tests..."
PYTHON=${PYTHON:-python3}
${PYTHON} -m pytest -q tests || { echo "Tests failed"; exit 1; }
echo "Tests passed."


