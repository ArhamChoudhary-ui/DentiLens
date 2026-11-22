#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${EGOSIGHT_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  printf '%s\n' 'Create the project Python environment using the README setup commands.' >&2
  exit 1
fi
FRONTEND_ENTRY=frontend/dist/index.html
if [[ ! -f "$FRONTEND_ENTRY" ]] || [[ -n "$(find frontend/src frontend/public frontend/index.html frontend/package.json frontend/package-lock.json -newer "$FRONTEND_ENTRY" -print -quit)" ]]; then
  npm --prefix frontend run build
fi
exec "$PYTHON" -m uvicorn backend.main:app --host 127.0.0.1 --port 8502
