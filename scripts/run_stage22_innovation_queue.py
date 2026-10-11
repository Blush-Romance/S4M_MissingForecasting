#!/usr/bin/env python3
"""Dependency-aware queue for Step-22 MRRA/RADR validation experiments."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime" / "stage22_innovation"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs" / "stage22_innovation"
scheduler.RUNNER = ROOT / "scripts" / "run_s4m_experiment.py"
scheduler.AGGREGATOR = ROOT / "scripts" / "aggregate_stage22_innovation_results.py"
scheduler.QUEUE_NAME = "stage22_pg95_baseline_regression_then_mrra_radr_validation"
scheduler.JOB_DEFINITIONS = [
    {
        "id": "pg95_default_off_regression_smoke",
        "config": (
            "configs/stage21_electricity/"
            "pg95_electricity_seed2023_smoke.yaml"
        ),
        "depends_on": [],
        "required_memory_estimate_mib": 13312,
    },
    {
        "id": "mrra_pg95_electricity_seed2023_smoke",
        "config": (
            "configs/stage22_innovation/"
            "mrra_pg95_electricity_seed2023_smoke.yaml"
        ),
        "depends_on": ["pg95_default_off_regression_smoke"],
        "required_memory_estimate_mib": 13312,
    },
    {
        "id": "radr_pg95_electricity_seed2023_smoke",
        "config": (
            "configs/stage22_innovation/"
            "radr_pg95_electricity_seed2023_smoke.yaml"
        ),
        "depends_on": ["pg95_default_off_regression_smoke"],
        "required_memory_estimate_mib": 13312,
    },
    {
        "id": "mrra_pg95_electricity_seed2023_formal",
        "config": (
            "configs/stage22_innovation/"
            "mrra_pg95_electricity_seed2023.yaml"
        ),
        "depends_on": ["mrra_pg95_electricity_seed2023_smoke"],
        "required_memory_estimate_mib": 13312,
    },
    {
        "id": "radr_pg95_electricity_seed2023_formal",
        "config": (
            "configs/stage22_innovation/"
            "radr_pg95_electricity_seed2023.yaml"
        ),
        "depends_on": ["radr_pg95_electricity_seed2023_smoke"],
        "required_memory_estimate_mib": 13312,
    },
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
