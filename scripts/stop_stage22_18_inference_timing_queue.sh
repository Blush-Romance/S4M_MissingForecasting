#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="$ROOT/runtime/stage22_18_inference_timing"
mkdir -p "$RUNTIME"
touch "$RUNTIME/stop.requested"
echo "stop requested; a running timing job will finish normally"
