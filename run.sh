#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${DENTILENS_PYTHON:-}" ]]; then
    PYTHON="$DENTILENS_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
else
    PYTHON=python3.11
fi
exec "$PYTHON" "$ROOT/code/dental_video.py" "$@"
