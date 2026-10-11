#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/zza/code/S4M_MissingForecasting"
STOP_FILE="$ROOT/runtime/stage5_baseline_queue/stop.requested"
mkdir -p "$(dirname "$STOP_FILE")"
touch "$STOP_FILE"
echo "stop requested; running experiments will finish and will not be killed"
