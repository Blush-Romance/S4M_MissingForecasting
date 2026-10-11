#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/runtime/stage22_20_ard_diagnostic/state.json"
LOG="$ROOT/logs/stage22_20_ard_diagnostic/scheduler.log"
if [[ -f "$STATE" ]]; then
  /mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python -m json.tool "$STATE"
else
  echo "state not found: $STATE"
fi
if [[ -f "$LOG" ]]; then tail -n 40 "$LOG"; fi

