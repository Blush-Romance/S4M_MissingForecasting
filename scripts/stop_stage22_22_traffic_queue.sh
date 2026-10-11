#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage22_22_traffic_pipeline"
touch "$ROOT/runtime/stage22_22_traffic_pipeline/stop.requested"
echo "stage22.22 Traffic queue stop requested; running jobs will finish safely"
