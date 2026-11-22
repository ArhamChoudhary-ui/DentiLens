#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${EGOSIGHT_PYTHON:-}" ]]; then
  PYTHON="$EGOSIGHT_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON=python3.11
fi
exec "$PYTHON" "$ROOT/code/main.py" "$@"
