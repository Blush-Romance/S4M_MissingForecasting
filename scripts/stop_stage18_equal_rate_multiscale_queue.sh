#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage18_equal_rate_multiscale_queue"
touch "$ROOT/runtime/stage18_equal_rate_multiscale_queue/stop.requested"
echo "stage18 queue stop requested; running validation will finish safely"
