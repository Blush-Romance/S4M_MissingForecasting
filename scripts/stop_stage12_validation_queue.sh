#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage12_validation_queue"
touch "$ROOT/runtime/stage12_validation_queue/stop.requested"
echo "stage12 validation queue stop requested; running diagnostics will finish safely"
