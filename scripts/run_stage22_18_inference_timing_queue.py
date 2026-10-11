#!/usr/bin/env python3
"""Single-GPU serial smoke-to-formal queue for Step-22.18 latency closure."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime" / "stage22_18_inference_timing"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs" / "stage22_18_inference_timing"
scheduler.RUNNER = ROOT / "scripts" / "run_stage22_18_inference_timing.py"
scheduler.AGGREGATOR = ROOT / "scripts" / "aggregate_stage22_18_ttrr.py"
scheduler.QUEUE_NAME = "stage22_18_same_gpu_serial_inference_timing"
scheduler.JOB_DEFINITIONS = [
    {
        "id": "pg95_timing_smoke",
        "config": "configs/stage22_ttrr/timing_pg95_electricity_seed2024_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "mcpr_mse_timing_smoke",
        "config": "configs/stage22_ttrr/timing_mcpr_mse_electricity_seed2024_smoke.yaml",
        "depends_on": ["pg95_timing_smoke"],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "ttrr_mcpr_timing_smoke",
        "config": "configs/stage22_ttrr/timing_ttrr_mcpr_electricity_seed2024_smoke.yaml",
        "depends_on": ["mcpr_mse_timing_smoke"],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "pg95_timing_formal",
        "config": "configs/stage22_ttrr/timing_pg95_electricity_seed2024.yaml",
        "depends_on": ["ttrr_mcpr_timing_smoke"],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "mcpr_mse_timing_formal",
        "config": "configs/stage22_ttrr/timing_mcpr_mse_electricity_seed2024.yaml",
        "depends_on": ["pg95_timing_formal"],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "ttrr_mcpr_timing_formal",
        "config": "configs/stage22_ttrr/timing_ttrr_mcpr_electricity_seed2024.yaml",
        "depends_on": ["mcpr_mse_timing_formal"],
        "required_memory_estimate_mib": 6144,
    },
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
