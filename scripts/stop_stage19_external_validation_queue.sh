#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage19_external_validation_queue"
touch "$ROOT/runtime/stage19_external_validation_queue/stop.requested"
echo "stage19 queue stop requested; running evaluation will finish safely"
