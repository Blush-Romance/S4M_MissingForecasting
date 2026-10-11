#!/usr/bin/env python3
"""Smoke-gated UUID queue for Step 22.27 BF-CMVD failure attribution."""

from __future__ import annotations

import json
from pathlib import Path

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
SMOKE_JOB = "bf_cmvd_v2_failure_attribution_seed2027_smoke"
FORMAL_JOB = "bf_cmvd_v2_failure_attribution_seed2027_formal"
scheduler.RUNTIME_ROOT = ROOT / "runtime/stage22_27_bf_cmvd_diagnostic"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs/stage22_27_bf_cmvd_diagnostic"
scheduler.RUNNER = ROOT / "scripts/run_stage22_27_bf_cmvd_diagnostic.py"
scheduler.AGGREGATOR = ROOT / "scripts/analyze_stage22_27_bf_cmvd_failure.py"
scheduler.QUEUE_NAME = "stage22_27_bf_cmvd_failure_attribution_validation_only"
scheduler.JOB_DEFINITIONS = [
    {
        "id": SMOKE_JOB,
        "config": (
            "configs/stage22_27_bf_cmvd_diagnostic/"
            "bf_cmvd_v2_seed2027_smoke.yaml"
        ),
        "depends_on": [],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": FORMAL_JOB,
        "config": (
            "configs/stage22_27_bf_cmvd_diagnostic/"
            "bf_cmvd_v2_seed2027.yaml"
        ),
        "depends_on": [SMOKE_JOB],
        "required_memory_estimate_mib": 6144,
    },
]


_base_update_dependency_states = scheduler.update_dependency_states


def enforce_artifact_gate(state: dict) -> None:
    _base_update_dependency_states(state)
    smoke = state["jobs"][SMOKE_JOB]
    formal = state["jobs"][FORMAL_JOB]
    if smoke.get("state") != "SUCCESS" or formal.get("state") not in scheduler.WAITING_STATES:
        return
    metrics_path = Path(smoke["run_dir"]) / "metrics.json" if smoke.get("run_dir") else None
    passed = False
    reason = "smoke metrics missing"
    if metrics_path and metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        passed = bool(
            metrics.get("artifact_status") == "PASS"
            and all(metrics.get("artifact_checks", {}).values())
            and metrics.get("test_split_accessed") is False
            and metrics.get("updates_model_or_memory") is False
            and int(metrics.get("validation_batches_processed", 0)) == 3
            and int(metrics.get("validation_samples_processed", 0)) == 48
        )
        reason = "Step22.27 smoke artifact gate PASS" if passed else "Step22.27 smoke artifact gate failed"
    formal["scientific_dependency"] = {
        "source_job": SMOKE_JOB,
        "passed": passed,
        "reason": reason,
        "metrics_path": str(metrics_path) if metrics_path else None,
    }
    if not passed:
        formal["state"] = "BLOCKED_DEPENDENCY"
        formal["failure_summary"] = f"artifact gate blocked formal: {reason}"
        formal["finished_at"] = scheduler.iso_now()
        scheduler.append_event(
            "blocked_scientific_dependency",
            job_id=FORMAL_JOB,
            source_job=SMOKE_JOB,
            reason=reason,
        )


scheduler.update_dependency_states = enforce_artifact_gate


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
