#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="$ROOT/runtime/stage22_24_traffic_pipeline"
mkdir -p "$RUNTIME"
touch "$RUNTIME/stop.requested"
echo "stage22.24 scheduler stop requested; running experiment is not killed"
