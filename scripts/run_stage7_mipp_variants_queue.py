#!/usr/bin/env python3
"""Step-7 queue for two additional Weather MIPP mechanism variants."""

from __future__ import annotations

from pathlib import Path

import run_stage5_baseline_queue as engine


ROOT = Path(__file__).resolve().parents[1]
engine.RUNTIME_ROOT = ROOT / "runtime" / "stage7_mipp_variants_queue"
engine.STATE_PATH = engine.RUNTIME_ROOT / "state.json"
engine.EVENTS_PATH = engine.RUNTIME_ROOT / "events.jsonl"
engine.STOP_PATH = engine.RUNTIME_ROOT / "stop.requested"
engine.SCHEDULER_LOCK_PATH = engine.RUNTIME_ROOT / "scheduler.lock"
engine.LOG_ROOT = ROOT / "logs" / "stage7_mipp_variants_queue"
engine.QUEUE_NAME = "stage7_weather_mipp_two_additional_mechanism_variants"
engine.JOB_DEFINITIONS = [
    {
        "id": "strict_gate_t095_smoke",
        "config": "configs/mipp/s4m_weather_mipp_strict_gate_t095_seed2023_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6656,
    },
    {
        "id": "pair_gate_only_t095_smoke",
        "config": "configs/mipp/s4m_weather_mipp_pair_gate_only_t095_seed2023_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 6656,
    },
    {
        "id": "strict_gate_t095_formal",
        "config": "configs/mipp/s4m_weather_mipp_strict_gate_t095_seed2023.yaml",
        "depends_on": ["strict_gate_t095_smoke"],
        "required_memory_estimate_mib": 6656,
    },
    {
        "id": "pair_gate_only_t095_formal",
        "config": "configs/mipp/s4m_weather_mipp_pair_gate_only_t095_seed2023.yaml",
        "depends_on": ["pair_gate_only_t095_smoke"],
        "required_memory_estimate_mib": 6656,
    },
]


if __name__ == "__main__":
    raise SystemExit(engine.main())
