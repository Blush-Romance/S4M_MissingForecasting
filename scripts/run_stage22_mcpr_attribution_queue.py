#!/usr/bin/env python3
"""Durable smoke-to-formal queue for frozen MCPR failure attribution."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime" / "stage22_mcpr_attribution"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs" / "stage22_mcpr_attribution"
scheduler.RUNNER = ROOT / "scripts" / "run_stage12_validation_cache.py"
scheduler.AGGREGATOR = ROOT / "scripts" / "aggregate_stage22_mcpr_attribution.py"
scheduler.QUEUE_NAME = "stage22_17_mcpr_frozen_failure_attribution"
scheduler.JOB_DEFINITIONS = [
    {
        "id": "mcpr_mse_attribution_smoke",
        "config": "configs/stage22_mcpr_attribution/mcpr_mse_electricity_seed2023_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "mcpr_pcd_attribution_smoke",
        "config": "configs/stage22_mcpr_attribution/mcpr_pcd_electricity_seed2023_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "mcpr_mse_attribution_formal",
        "config": "configs/stage22_mcpr_attribution/mcpr_mse_electricity_seed2023.yaml",
        "depends_on": ["mcpr_mse_attribution_smoke"],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "mcpr_pcd_attribution_formal",
        "config": "configs/stage22_mcpr_attribution/mcpr_pcd_electricity_seed2023.yaml",
        "depends_on": ["mcpr_pcd_attribution_smoke"],
        "required_memory_estimate_mib": 6144,
    },
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
