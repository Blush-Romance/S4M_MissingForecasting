#!/usr/bin/env python3
"""Step-8 Weather FUSR V2 frozen-router smoke-to-formal queue."""

from __future__ import annotations

from pathlib import Path

import run_stage5_baseline_queue as engine


ROOT = Path(__file__).resolve().parents[1]
engine.RUNTIME_ROOT = ROOT / "runtime" / "stage8_fusr_v2_queue"
engine.STATE_PATH = engine.RUNTIME_ROOT / "state.json"
engine.EVENTS_PATH = engine.RUNTIME_ROOT / "events.jsonl"
engine.STOP_PATH = engine.RUNTIME_ROOT / "stop.requested"
engine.SCHEDULER_LOCK_PATH = engine.RUNTIME_ROOT / "scheduler.lock"
engine.LOG_ROOT = ROOT / "logs" / "stage8_fusr_v2_queue"
engine.QUEUE_NAME = "stage8_weather_fusr_v2_frozen_router_independent_screen"
engine.JOB_DEFINITIONS = [
    {
        "id": "weather_fusr_v2_smoke",
        "config": "configs/fusr/s4m_weather_fusr_v2_frozen_router_seed2023_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": "weather_fusr_v2_formal",
        "config": "configs/fusr/s4m_weather_fusr_v2_frozen_router_seed2023.yaml",
        "depends_on": ["weather_fusr_v2_smoke"],
        "required_memory_estimate_mib": 6144,
    },
]


if __name__ == "__main__":
    raise SystemExit(engine.main())
