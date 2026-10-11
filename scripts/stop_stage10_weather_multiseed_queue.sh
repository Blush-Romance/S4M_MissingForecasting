#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage10_weather_multiseed_queue"
touch "$ROOT/runtime/stage10_weather_multiseed_queue/stop.requested"
echo "stage10 Weather multiseed queue stop requested; running experiments will finish safely"
