#!/usr/bin/env python3
"""Smoke-to-formal queue for Stage22.16 MCPR variants."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime" / "stage22_mcpr"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs" / "stage22_mcpr"
scheduler.RUNNER = ROOT / "scripts" / "run_s4m_experiment.py"
scheduler.AGGREGATOR = ROOT / "scripts" / "aggregate_stage22_mcpr_results.py"
scheduler.QUEUE_NAME = "stage22_16_mcpr_validation_screen"
scheduler.JOB_DEFINITIONS = [
    {"id": "mcpr_mse_smoke", "config": "configs/stage22_mcpr/mcpr_mse_electricity_seed2023_smoke.yaml", "depends_on": [], "required_memory_estimate_mib": 6144},
    {"id": "mcpr_pcd_smoke", "config": "configs/stage22_mcpr/mcpr_pcd_electricity_seed2023_smoke.yaml", "depends_on": [], "required_memory_estimate_mib": 8192},
    {"id": "mcpr_mse_formal", "config": "configs/stage22_mcpr/mcpr_mse_electricity_seed2023.yaml", "depends_on": ["mcpr_mse_smoke"], "required_memory_estimate_mib": 6144},
    {"id": "mcpr_pcd_formal", "config": "configs/stage22_mcpr/mcpr_pcd_electricity_seed2023.yaml", "depends_on": ["mcpr_pcd_smoke"], "required_memory_estimate_mib": 8192},
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
