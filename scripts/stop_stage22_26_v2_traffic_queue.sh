#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage22_26_traffic_v2"
touch "$ROOT/runtime/stage22_26_traffic_v2/stop.requested"
echo "stage22.26-v2 scheduler stop requested; running experiment is not killed"
