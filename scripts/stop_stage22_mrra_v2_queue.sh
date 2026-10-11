#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="$ROOT/runtime/stage22_mrra_v2"
mkdir -p "$RUNTIME"
touch "$RUNTIME/stop.requested"
echo "stage22 MRRA-v2 queue will stop after its active jobs finish"
