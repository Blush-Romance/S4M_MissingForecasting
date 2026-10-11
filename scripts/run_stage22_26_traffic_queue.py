#!/usr/bin/env python3
"""Step-22.26 BF-CMVD queue with a mechanism-level smoke dependency."""

from __future__ import annotations

import json
from pathlib import Path

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
SMOKE_JOB = "bf_cmvd_traffic_validation_seed2027_smoke"
FORMAL_JOB = "bf_cmvd_traffic_validation_seed2027_formal"
scheduler.RUNTIME_ROOT = ROOT / "runtime/stage22_26_traffic"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs/stage22_26_traffic"
scheduler.RUNNER = ROOT / "scripts/run_s4m_experiment.py"
scheduler.AGGREGATOR = ROOT / "scripts/aggregate_stage22_26_bf_cmvd.py"
scheduler.QUEUE_NAME = "stage22_26_bf_cmvd_traffic_seed2027"
scheduler.JOB_DEFINITIONS = [
    {
        "id": SMOKE_JOB,
        "config": "configs/stage22_26_traffic/bf_cmvd_traffic_seed2027_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 14336,
    },
    {
        "id": FORMAL_JOB,
        "config": "configs/stage22_26_traffic/bf_cmvd_traffic_seed2027.yaml",
        "depends_on": [SMOKE_JOB],
        "required_memory_estimate_mib": 14336,
    },
]


_base_update_dependency_states = scheduler.update_dependency_states


def enforce_scientific_gate(state: dict) -> None:
    """Block formal before GPU launch unless the BF-CMVD mechanism ran."""
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
        beneficial = int(metrics.get("bf_cmvd_beneficial_samples", 0))
        total = int(metrics.get("bf_cmvd_total_samples", 0))
        passed = bool(
            metrics.get("bf_cmvd_enabled") is True
            and int(metrics.get("bf_cmvd_diagnostic_steps", 0)) > 0
            and 0 < beneficial < total
            and float(metrics.get("bf_cmvd_distill_l1_mean", 0.0)) > 0.0
            and float(metrics.get("bf_cmvd_teacher_delta_energy_mean", 0.0)) > 0.0
            and float(metrics.get("mcpr_gradient_norm_max", 0.0)) > 0.0
            and float(metrics.get("mcpr_residual_energy_mean", 0.0)) > 0.0
            and metrics.get("checkpoint_load_verified") is True
        )
        reason = "BF-CMVD smoke mechanism PASS" if passed else "BF-CMVD smoke mechanism evidence failed"
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
