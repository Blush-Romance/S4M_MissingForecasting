#!/usr/bin/env python3
"""Step-22.25 queue with a hard smoke-equivalence scientific dependency."""

from __future__ import annotations

import json
from pathlib import Path

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
SMOKE_JOB = "traffic_cuda_stream_validation_seed2027_smoke"
FORMAL_JOB = "traffic_cuda_stream_validation_seed2027_formal"
scheduler.RUNTIME_ROOT = ROOT / "runtime/stage22_25_traffic_pipeline"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs/stage22_25_traffic_pipeline"
scheduler.RUNNER = ROOT / "scripts/run_stage22_25_traffic_pipeline.py"
scheduler.AGGREGATOR = ROOT / "scripts/aggregate_stage22_25_cuda_stream.py"
scheduler.QUEUE_NAME = "stage22_25_traffic_cuda_stream_seed2027"
scheduler.JOB_DEFINITIONS = [
    {
        "id": SMOKE_JOB,
        "config": (
            "configs/stage22_25_traffic/"
            "cuda_stream_traffic_validation_seed2027_smoke.yaml"
        ),
        "depends_on": [],
        "required_memory_estimate_mib": 12288,
    },
    {
        "id": FORMAL_JOB,
        "config": (
            "configs/stage22_25_traffic/"
            "cuda_stream_traffic_validation_seed2027.yaml"
        ),
        "depends_on": [SMOKE_JOB],
        "required_memory_estimate_mib": 12288,
    },
]

_base_update_dependency_states = scheduler.update_dependency_states


def enforce_scientific_gate(state: dict) -> None:
    """Block formal before launch when smoke is not numerically equivalent."""
    _base_update_dependency_states(state)
    smoke = state["jobs"][SMOKE_JOB]
    formal = state["jobs"][FORMAL_JOB]
    if smoke.get("state") != "SUCCESS" or formal.get("state") not in scheduler.WAITING_STATES:
        return
    metrics_path = (
        Path(smoke["run_dir"]) / "metrics.json" if smoke.get("run_dir") else None
    )
    passed = False
    reason = "smoke metrics missing"
    if metrics_path and metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        passed = bool(
            metrics.get("artifact_status") == "PASS"
            and metrics.get("prediction_equivalence_pass") is True
        )
        reason = (
            "smoke equivalence PASS"
            if passed else "smoke prediction_equivalence_pass is not true"
        )
    formal["scientific_dependency"] = {
        "source_job": SMOKE_JOB,
        "passed": passed,
        "reason": reason,
        "metrics_path": str(metrics_path) if metrics_path else None,
    }
    if not passed:
        formal["state"] = "BLOCKED_DEPENDENCY"
        formal["failure_summary"] = f"scientific gate blocked formal: {reason}"
        formal["finished_at"] = scheduler.iso_now()
        scheduler.append_event(
            "blocked_scientific_dependency",
            job_id=FORMAL_JOB,
            source_job=SMOKE_JOB,
            reason=reason,
        )


scheduler.update_dependency_states = enforce_scientific_gate


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
