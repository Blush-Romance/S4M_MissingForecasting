#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="$ROOT/runtime/stage22_26_traffic_v2/state.json"
if [[ ! -f "$STATE" ]]; then echo "stage22.26-v2 state not created"; exit 0; fi
jq '{queue_state,jobs: (.jobs | with_entries(.value |= {state,attempt_count,oom_retry_count,run_dir,failure_summary,scientific_dependency}))}' "$STATE"
