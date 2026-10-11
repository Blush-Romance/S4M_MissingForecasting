#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage22_25_traffic_pipeline"
touch "$ROOT/runtime/stage22_25_traffic_pipeline/stop.requested"
echo "stage22.25 scheduler stop requested; running experiment is not killed"
