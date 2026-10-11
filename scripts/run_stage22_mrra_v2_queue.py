#!/usr/bin/env python3
"""Dependency queue for the MRRA-v2 stability smoke and formal validation."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime" / "stage22_mrra_v2"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs" / "stage22_mrra_v2"
scheduler.RUNNER = ROOT / "scripts" / "run_s4m_experiment.py"
scheduler.AGGREGATOR = ROOT / "scripts" / "aggregate_stage22_mrra_v2_results.py"
scheduler.QUEUE_NAME = "stage22_mrra_v2_stability_then_validation"
scheduler.JOB_DEFINITIONS = [
    {
        "id": "pg95_default_off_post_v2_regression_smoke",
        "config": (
            "configs/stage21_electricity/"
            "pg95_electricity_seed2023_smoke.yaml"
        ),
        "depends_on": [],
        "required_memory_estimate_mib": 13312,
    },
    {
        "id": "mrra_v2_pg95_electricity_seed2023_stability_smoke",
        "config": (
            "configs/stage22_innovation/"
            "mrra_v2_pg95_electricity_seed2023_stability_smoke.yaml"
        ),
        "depends_on": ["pg95_default_off_post_v2_regression_smoke"],
        "required_memory_estimate_mib": 13312,
    },
    {
        "id": "mrra_v2_pg95_electricity_seed2023_formal",
        "config": (
            "configs/stage22_innovation/"
            "mrra_v2_pg95_electricity_seed2023.yaml"
        ),
        "depends_on": ["mrra_v2_pg95_electricity_seed2023_stability_smoke"],
        "required_memory_estimate_mib": 13312,
    },
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
