#!/usr/bin/env python3
"""Durable smoke-to-formal queue for Stage-15 CNM-SE."""

from __future__ import annotations

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / "runtime" / "stage15_cnm_se_queue"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs" / "stage15_cnm_se_queue"
scheduler.RUNNER = ROOT / "scripts" / "run_stage14_uncertainty_cache.py"
scheduler.AGGREGATOR = ROOT / "scripts" / "aggregate_stage15_cnm_se_results.py"
scheduler.QUEUE_NAME = "stage15_weather_pg95_causal_nested_mask_self_ensemble"
scheduler.JOB_DEFINITIONS = [
    {
        "id": "weather_cnm_se_seed2024_smoke",
        "config": "configs/stage15_cnm_se/s4m_weather_cnm_se_seed2024_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 4096,
    },
] + [
    {
        "id": f"weather_cnm_se_seed{seed}_formal",
        "config": f"configs/stage15_cnm_se/s4m_weather_cnm_se_seed{seed}.yaml",
        "depends_on": ["weather_cnm_se_seed2024_smoke"],
        "required_memory_estimate_mib": 4096,
    }
    for seed in (2023, 2024, 2025)
]


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
