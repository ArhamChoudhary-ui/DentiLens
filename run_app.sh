#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${DENTILENS_PYTHON:-$ROOT/.venv/bin/python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    printf '%s\n' 'DentiLens could not find Python. Create the .venv from the README setup instructions.' >&2
    exit 1
fi
export YOLO_CONFIG_DIR="$ROOT/runtime/yolo-config"
export MPLCONFIGDIR="$ROOT/runtime/matplotlib"
exec "$PYTHON" -m streamlit run "$ROOT/app.py" --server.address 127.0.0.1 --server.port 8501 --server.headless true --server.fileWatcherType poll --browser.gatherUsageStats false
