#!/usr/bin/env python3
"""Dynamic four-stage queue for ETTh1 PG95 plus frozen multiscale CNM-SE."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime" / "stage20_etth1_pg95_pipeline"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs" / "stage20_etth1_pg95_pipeline"
scheduler.RUNNER = ROOT / "scripts" / "run_stage20_etth1_pg95_pipeline.py"
scheduler.AGGREGATOR = ROOT / "scripts" / "aggregate_stage20_etth1_pg95.py"
scheduler.QUEUE_NAME = "stage20_etth1_pg95_then_frozen_multiscale_cnm_se"
scheduler.JOB_DEFINITIONS = [
    {
        "id": "etth1_pg95_seed2023_smoke",
        "config": (
            "configs/stage20_etth1_pg95/"
            "s4m_etth1_pg95_seed2023_smoke.yaml"
        ),
        "depends_on": [],
        "required_memory_estimate_mib": 7168,
    },
    {
        "id": "etth1_pg95_seed2023_formal",
        "config": (
            "configs/stage20_etth1_pg95/"
            "s4m_etth1_pg95_seed2023.yaml"
        ),
        "depends_on": ["etth1_pg95_seed2023_smoke"],
        "required_memory_estimate_mib": 7168,
    },
    {
        "id": "etth1_pg95_multiscale_seed2023_smoke",
        "config": (
            "configs/stage20_etth1_pg95/"
            "s4m_etth1_pg95_multiscale_seed2023_smoke.yaml"
        ),
        "depends_on": ["etth1_pg95_seed2023_formal"],
        "required_memory_estimate_mib": 4096,
    },
    {
        "id": "etth1_pg95_multiscale_seed2023_formal",
        "config": (
            "configs/stage20_etth1_pg95/"
            "s4m_etth1_pg95_multiscale_seed2023.yaml"
        ),
        "depends_on": ["etth1_pg95_multiscale_seed2023_smoke"],
        "required_memory_estimate_mib": 4096,
    },
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
