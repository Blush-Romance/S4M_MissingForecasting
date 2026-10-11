#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage14_uncertainty_queue"
touch "$ROOT/runtime/stage14_uncertainty_queue/stop.requested"
echo "stage14 uncertainty queue stop requested; running jobs will finish safely"
