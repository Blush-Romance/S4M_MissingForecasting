#!/usr/bin/env python3
"""Step-22.28 DMRB smoke-to-formal validation queue."""

from __future__ import annotations

import json
from pathlib import Path

import run_stage5_baseline_queue as scheduler


ROOT = scheduler.PROJECT_ROOT
SMOKE_JOB = "dmrb_traffic_validation_seed2027_smoke"
FORMAL_JOB = "dmrb_traffic_validation_seed2027_formal"
DMRB_TEACHER_RECONSTRUCTION_ATOL = 32 * 1.1920928955078125e-7
scheduler.RUNTIME_ROOT = ROOT / "runtime/stage22_28_dmrb"
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / "state.json"
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / "events.jsonl"
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / "stop.requested"
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / "scheduler.lock"
scheduler.LOG_ROOT = ROOT / "logs/stage22_28_dmrb"
scheduler.RUNNER = ROOT / "scripts/run_s4m_experiment.py"
scheduler.AGGREGATOR = ROOT / "scripts/aggregate_stage22_28_dmrb.py"
scheduler.QUEUE_NAME = "stage22_28_dmrb_traffic_seed2027"
scheduler.JOB_DEFINITIONS = [
    {
        "id": SMOKE_JOB,
        "config": "configs/stage22_28_dmrb/dmrb_traffic_seed2027_smoke.yaml",
        "depends_on": [],
        "required_memory_estimate_mib": 14336,
    },
    {
        "id": FORMAL_JOB,
        "config": "configs/stage22_28_dmrb/dmrb_traffic_seed2027.yaml",
        "depends_on": [SMOKE_JOB],
        "required_memory_estimate_mib": 14336,
    },
]


_base_update_dependency_states = scheduler.update_dependency_states


def enforce_scientific_gate(state: dict) -> None:
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
        beneficial = int(metrics.get("bf_cmvd_beneficial_units", 0))
        total = int(metrics.get("bf_cmvd_total_units", 0))
        checks = {
            "dmrb_enabled": metrics.get("dmrb_enabled") is True,
            "both_branches_enabled": (
                metrics.get("dmrb_l5_enabled") is True
                and metrics.get("dmrb_l20_enabled") is True
            ),
            "token_filter_active": (
                metrics.get("bf_cmvd_filter_granularity") == "token"
                and 0 < beneficial < total
            ),
            "both_distill_losses_positive": all(
                float(metrics.get(key, 0.0)) > 0.0
                for key in (
                    "dmrb_l5_distill_l1_mean", "dmrb_l20_distill_l1_mean"
                )
            ),
            "both_residual_energies_positive": all(
                float(metrics.get(key, 0.0)) > 0.0
                for key in (
                    "dmrb_l5_residual_energy_mean",
                    "dmrb_l20_residual_energy_mean",
                )
            ),
            "both_gradient_norms_positive": all(
                float(metrics.get(key, 0.0)) > 0.0
                for key in (
                    "dmrb_l5_gradient_norm_max", "dmrb_l20_gradient_norm_max"
                )
            ),
            "teacher_factorization_exact": float(
                metrics.get("dmrb_teacher_reconstruction_max_abs", 1.0)
            ) <= DMRB_TEACHER_RECONSTRUCTION_ATOL,
            "exact_trainable_parameter_count": int(
                metrics.get("mcpr_trainable_parameter_count", -1)
            ) == 90364,
            "frozen_backbone_and_memory": (
                metrics.get("mcpr_backbone_hash_before")
                == metrics.get("mcpr_backbone_hash_after")
                and metrics.get("mcpr_memory_hash_before")
                == metrics.get("mcpr_memory_hash_after")
            ),
            "single_backbone_inference_path": int(
                metrics.get("dmrb_inference_backbone_forward_count", -1)
            ) == 1,
            "delta_checkpoint_verified": (
                metrics.get("mcpr_checkpoint_format") == "mcpr_delta_v1"
                and metrics.get("checkpoint_format") == "mcpr_delta_v1"
                and metrics.get("checkpoint_dmrb_branches_verified") is True
                and int(metrics.get("checkpoint_size_bytes", 10**9)) < 8 * 1024 * 1024
            ),
            "validation_only": (
                metrics.get("evaluation_split") == "validation"
                and metrics.get("test_split_accessed") is False
            ),
        }
        passed = all(checks.values())
        reason = "DMRB smoke PASS" if passed else "DMRB mechanism/artifact evidence failed"
    formal["scientific_dependency"] = {
        "source_job": SMOKE_JOB,
        "passed": passed,
        "reason": reason,
        "checks": checks,
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
            checks=checks,
        )


scheduler.update_dependency_states = enforce_scientific_gate


if __name__ == "__main__":
    raise SystemExit(scheduler.main())
