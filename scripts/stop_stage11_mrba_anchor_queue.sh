#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage11_mrba_anchor_queue"
touch "$ROOT/runtime/stage11_mrba_anchor_queue/stop.requested"
echo "stage11 MRBA anchor queue stop requested; running experiments will finish safely"
