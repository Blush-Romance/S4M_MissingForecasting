#!/usr/bin/env python3
"""Dependency-aware Step-22.21 Traffic validation and gated-test queue."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime/stage22_21_traffic_pipeline"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs/stage22_21_traffic_pipeline"
scheduler.RUNNER = ROOT / "scripts/run_stage22_21_traffic_pipeline.py"
scheduler.AGGREGATOR = ROOT / "scripts/aggregate_stage22_21_traffic.py"
scheduler.QUEUE_NAME = "stage22_21_traffic_validation_then_conditional_single_test"
scheduler.JOB_DEFINITIONS = [
    {
        "id": "traffic_s4m_seed2026_smoke",
        "config": "configs/stage22_21_traffic/s4m_traffic_seed2026_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 12288,
    },
    {
        "id": "traffic_pg95_seed2026_smoke",
        "config": "configs/stage22_21_traffic/pg95_traffic_seed2026_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 20480,
    },
    {
        "id": "traffic_s4m_seed2026_formal",
        "config": "configs/stage22_21_traffic/s4m_traffic_seed2026.yaml",
        "depends_on": ["traffic_s4m_seed2026_smoke"],
        "required_memory_estimate_mib": 12288,
    },
    {
        "id": "traffic_pg95_seed2026_formal",
        "config": "configs/stage22_21_traffic/pg95_traffic_seed2026.yaml",
        "depends_on": ["traffic_pg95_seed2026_smoke"],
        "required_memory_estimate_mib": 20480,
    },
    {
        "id": "traffic_pg95_multiscale_validation_seed2026_smoke",
        "config": (
            "configs/stage22_21_traffic/"
            "pg95_traffic_multiscale_validation_seed2026_smoke.yaml"
        ),
        "depends_on": ["traffic_pg95_seed2026_formal"],
        "required_memory_estimate_mib": 16384,
    },
    {
        "id": "traffic_pg95_multiscale_validation_seed2026_formal",
        "config": (
            "configs/stage22_21_traffic/"
            "pg95_traffic_multiscale_validation_seed2026.yaml"
        ),
        "depends_on": ["traffic_pg95_multiscale_validation_seed2026_smoke"],
        "required_memory_estimate_mib": 16384,
    },
    {
        "id": "traffic_conditional_single_test_seed2026_formal",
        "config": (
            "configs/stage22_21_traffic/"
            "pg95_traffic_multiscale_test_seed2026.yaml"
        ),
        "depends_on": [
            "traffic_pg95_multiscale_validation_seed2026_formal",
            "traffic_s4m_seed2026_formal",
        ],
        "required_memory_estimate_mib": 16384,
    },
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
