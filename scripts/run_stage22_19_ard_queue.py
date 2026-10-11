#!/usr/bin/env python3
"""Smoke-gated dynamic GPU queue for Step 22.19 ARD confirmation."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime" / "stage22_19_ard"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs" / "stage22_19_ard"
scheduler.RUNNER = ROOT / "scripts" / "run_s4m_experiment.py"
scheduler.AGGREGATOR = ROOT / "scripts" / "aggregate_stage22_19_ard.py"
scheduler.QUEUE_NAME = "stage22_19_ard_preregistered_crossmask_confirmation"
CACHE_RUNNER = ROOT / "scripts" / "run_stage12_validation_cache.py"
scheduler.JOB_DEFINITIONS = [
    {
        "id": "pg95_smoke",
        "config": "configs/stage22_ard/pg95_electricity_seed2025_smoke.yaml",
        "runner": str(CACHE_RUNNER),
        "depends_on": [],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "mcpr_mse_smoke",
        "config": "configs/stage22_ard/mcpr_mse_electricity_seed2025_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "ard_smoke",
        "config": "configs/stage22_ard/ard_mcpr_electricity_seed2025_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "pg95_formal",
        "config": "configs/stage22_ard/pg95_electricity_seed2025.yaml",
        "runner": str(CACHE_RUNNER),
        "depends_on": ["pg95_smoke"],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "mcpr_mse_formal",
        "config": "configs/stage22_ard/mcpr_mse_electricity_seed2025.yaml",
        "depends_on": ["mcpr_mse_smoke"],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "ard_formal",
        "config": "configs/stage22_ard/ard_mcpr_electricity_seed2025.yaml",
        "depends_on": ["ard_smoke"],
        "required_memory_estimate_mib": 6144,
    },
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
