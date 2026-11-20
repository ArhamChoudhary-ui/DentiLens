#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${DENTILENS_PYTHON:-$ROOT/.venv/bin/python}"
export YOLO_CONFIG_DIR="$ROOT/runtime/yolo-config"
export MPLCONFIGDIR="$ROOT/runtime/matplotlib"
exec "$PYTHON" -m streamlit run "$ROOT/app.py" --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
