#!/usr/bin/env python3
"""Smoke-gated UUID queue for Step 22.29 DMRB failure attribution."""

from __future__ import annotations

import json
from pathlib import Path

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
SMOKE_JOB = "dmrb_failure_attribution_seed2027_smoke"
FORMAL_JOB = "dmrb_failure_attribution_seed2027_formal"
scheduler.RUNTIME_ROOT = ROOT / "runtime/stage22_29_dmrb_diagnostic"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs/stage22_29_dmrb_diagnostic"
scheduler.RUNNER = ROOT / "scripts/run_stage22_29_dmrb_diagnostic.py"
scheduler.AGGREGATOR = ROOT / "scripts/aggregate_stage22_29_dmrb.py"
scheduler.QUEUE_NAME = "stage22_29_dmrb_failure_attribution_validation_only"
scheduler.JOB_DEFINITIONS = [
    {
        "id": SMOKE_JOB,
        "config": (
            "configs/stage22_29_dmrb_diagnostic/dmrb_seed2027_smoke.yaml"
        ),
        "depends_on": [],
        "required_memory_estimate_mib": 6144,
    },
    {
        "id": FORMAL_JOB,
        "config": "configs/stage22_29_dmrb_diagnostic/dmrb_seed2027.yaml",
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
    checks = {}
    if metrics_path and metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        checks = {
            "artifact_status": metrics.get("artifact_status") == "PASS",
            "all_artifact_checks": bool(
                metrics.get("artifact_checks")
                and all(metrics.get("artifact_checks", {}).values())
            ),
            "validation_only": metrics.get("test_split_accessed") is False,
            "frozen": metrics.get("updates_model_or_memory") is False,
            "three_batches": int(
                metrics.get("validation_batches_processed", 0)
            ) == 3,
            "forty_eight_samples": int(
                metrics.get("validation_samples_processed", 0)
            ) == 48,
            "both_dmrb_branches": (
                metrics.get("dmrb_l5_enabled") is True
                and metrics.get("dmrb_l20_enabled") is True
                and metrics.get("checkpoint_dmrb_branches_verified") is True
            ),
        }
        passed = all(checks.values())
        reason = (
            "Step22.29 smoke artifact gate PASS"
            if passed else "Step22.29 smoke artifact gate failed"
        )
    formal["scientific_dependency"] = {
        "source_job": SMOKE_JOB,
        "passed": passed,
        "reason": reason,
        "checks": checks,
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
            checks=checks,
        )


scheduler.update_dependency_states = enforce_artifact_gate


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
