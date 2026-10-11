#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/runtime/stage20_etth1_pg95_pipeline"
touch "$ROOT/runtime/stage20_etth1_pg95_pipeline/stop.requested"
echo "stage20 queue stop requested; running job will finish safely"
