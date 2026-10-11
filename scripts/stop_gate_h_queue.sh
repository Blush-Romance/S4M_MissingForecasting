#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/zza/code/S4M_MissingForecasting"
mkdir -p "$ROOT/runtime/gate_h_queue"
touch "$ROOT/runtime/gate_h_queue/stop.requested"
echo "Gate H queue stop requested; running diagnostics will finish before the scheduler exits"
