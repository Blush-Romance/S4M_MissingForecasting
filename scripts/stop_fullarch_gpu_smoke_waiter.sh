#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/zza/code/S4M_MissingForecasting"
STOP_FILE="$ROOT/runtime/fullarch_gpu_smoke_waiter/stop.requested"
mkdir -p "$(dirname "$STOP_FILE")"
touch "$STOP_FILE"
echo "stop requested; a running experiment will not be killed"

