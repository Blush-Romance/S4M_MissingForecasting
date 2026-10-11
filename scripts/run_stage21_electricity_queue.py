#!/usr/bin/env python3
"""Dependency-aware Stage-21 Electricity validation and gated-test queue."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime" / "stage21_electricity_pipeline"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs" / "stage21_electricity_pipeline"
scheduler.RUNNER = ROOT / "scripts" / "run_stage21_electricity_pipeline.py"
scheduler.AGGREGATOR = ROOT / "scripts" / "aggregate_stage21_electricity.py"
scheduler.QUEUE_NAME = "stage21_electricity_validation_then_conditional_single_test"
scheduler.JOB_DEFINITIONS = [
    {
        "id": "electricity_s4m_seed2023_smoke",
        "config": (
            "configs/stage21_electricity/"
            "s4m_electricity_seed2023_smoke.yaml"
        ),
        "depends_on": [],
        "required_memory_estimate_mib": 9216,
    },
    {
        "id": "electricity_pg95_seed2023_smoke",
        "config": (
            "configs/stage21_electricity/"
            "pg95_electricity_seed2023_smoke.yaml"
        ),
        "depends_on": [],
        "required_memory_estimate_mib": 13312,
    },
    {
        "id": "electricity_s4m_seed2023_formal",
        "config": (
            "configs/stage21_electricity/s4m_electricity_seed2023.yaml"
        ),
        "depends_on": ["electricity_s4m_seed2023_smoke"],
        "required_memory_estimate_mib": 9216,
    },
    {
        "id": "electricity_pg95_seed2023_formal",
        "config": (
            "configs/stage21_electricity/pg95_electricity_seed2023.yaml"
        ),
        "depends_on": ["electricity_pg95_seed2023_smoke"],
        "required_memory_estimate_mib": 13312,
    },
    {
        "id": "electricity_pg95_multiscale_validation_seed2023_smoke",
        "config": (
            "configs/stage21_electricity/"
            "pg95_electricity_multiscale_validation_seed2023_smoke.yaml"
        ),
        "depends_on": ["electricity_pg95_seed2023_formal"],
        "required_memory_estimate_mib": 8192,
    },
    {
        "id": "electricity_pg95_multiscale_validation_seed2023_formal",
        "config": (
            "configs/stage21_electricity/"
            "pg95_electricity_multiscale_validation_seed2023.yaml"
        ),
        "depends_on": [
            "electricity_pg95_multiscale_validation_seed2023_smoke"
        ],
        "required_memory_estimate_mib": 8192,
    },
    {
        "id": "electricity_conditional_single_test_seed2023_formal",
        "config": (
            "configs/stage21_electricity/"
            "pg95_electricity_multiscale_test_seed2023.yaml"
        ),
        "depends_on": [
            "electricity_s4m_seed2023_formal",
            "electricity_pg95_multiscale_validation_seed2023_formal",
        ],
        "required_memory_estimate_mib": 8192,
    },
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
