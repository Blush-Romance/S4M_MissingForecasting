#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage21_electricity_pipeline"
touch "$ROOT/runtime/stage21_electricity_pipeline/stop.requested"
echo "stage21 queue stop requested; running jobs will finish safely"
