#!/usr/bin/env python3
"""Durable smoke-to-formal queue for Stage-16 CNM-SE structure ablations."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime" / "stage16_cnm_se_ablation_queue"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs" / "stage16_cnm_se_ablation_queue"
scheduler.RUNNER = ROOT / "scripts" / "run_stage14_uncertainty_cache.py"
scheduler.AGGREGATOR = ROOT / "scripts" / "aggregate_stage16_cnm_se_ablation.py"
scheduler.QUEUE_NAME = "stage16_weather_pg95_cnm_se_structure_ablation"
scheduler.JOB_DEFINITIONS = [
    {
        "id": f"weather_cnm_se_l{block}_seed2024_smoke",
        "config": (
            "configs/stage16_cnm_se_ablation/"
            f"s4m_weather_cnm_se_l{block}_seed2024_smoke.yaml"
        ),
        "depends_on": [],
        "required_memory_estimate_mib": 4096,
    }
    for block in (1, 20)
] + [
    {
        "id": f"weather_cnm_se_l{block}_seed{seed}_formal",
        "config": (
            "configs/stage16_cnm_se_ablation/"
            f"s4m_weather_cnm_se_l{block}_seed{seed}.yaml"
        ),
        "depends_on": [f"weather_cnm_se_l{block}_seed2024_smoke"],
        "required_memory_estimate_mib": 4096,
    }
    for block in (1, 20)
    for seed in (2023, 2024, 2025)
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
