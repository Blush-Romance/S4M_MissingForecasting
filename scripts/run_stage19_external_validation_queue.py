#!/usr/bin/env python3
"""Dynamic smoke-to-formal queue for Stage-19 ETTh1 external validation."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime" / "stage19_external_validation_queue"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs" / "stage19_external_validation_queue"
scheduler.RUNNER = ROOT / "scripts" / "run_stage14_uncertainty_cache.py"
scheduler.AGGREGATOR = ROOT / "scripts" / "aggregate_stage19_external_validation.py"
scheduler.QUEUE_NAME = "stage19_etth1_frozen_multiscale_external_validation"
scheduler.JOB_DEFINITIONS = [
    {
        "id": "etth1_equal_rate_multiscale_seed2023_smoke",
        "config": (
            "configs/stage19_external_validation/"
            "s4m_etth1_seed2023_smoke.yaml"
        ),
        "depends_on": [],
        "required_memory_estimate_mib": 4096,
    },
    {
        "id": "etth1_equal_rate_multiscale_seed2023_formal",
        "config": (
            "configs/stage19_external_validation/"
            "s4m_etth1_seed2023.yaml"
        ),
        "depends_on": ["etth1_equal_rate_multiscale_seed2023_smoke"],
        "required_memory_estimate_mib": 4096,
    },
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
