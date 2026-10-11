#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="$ROOT/runtime/stage22_mrra_v3_diagnostic"
mkdir -p "$RUNTIME"
touch "$RUNTIME/stop.requested"
echo "stage22.14 MRRA-v3 diagnostic queue will stop after active jobs finish"
