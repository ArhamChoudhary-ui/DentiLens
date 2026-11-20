#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
URL="http://127.0.0.1:8502"
if curl -fsS --max-time 1 "$URL/api/health" >/dev/null 2>&1; then
  open "$URL/"
  exit 0
fi
(
  for attempt in {1..60}; do
    if curl -fsS --max-time 1 "$URL/api/health" >/dev/null 2>&1; then
      open "$URL/"
      exit 0
    fi
    sleep 0.25
  done
) &
exec bash "$ROOT/run_full.sh"
