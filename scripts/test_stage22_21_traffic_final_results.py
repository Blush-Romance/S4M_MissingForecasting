#!/usr/bin/env python3
"""Audit the one authorized Step-22.21 Traffic test and final comparison."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/stage22_21_traffic_pipeline/state.json"
GATE = ROOT / "reports/stage22_21_traffic_validation_gate.json"
RAW_RESULTS = ROOT / "reports/stage22_21_traffic_raw_results.json"
REPORT = ROOT / "reports/stage22_21_traffic_final_audit.json"
TEST_JOB = "traffic_conditional_single_test_seed2026_formal"
PG95_JOB = "traffic_pg95_seed2026_formal"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    state = load(STATE)
    gate = load(GATE)
    raw = load(RAW_RESULTS)
    test_job = state["jobs"][TEST_JOB]
    pg95_job = state["jobs"][PG95_JOB]
    test_run = Path(test_job["run_dir"]).resolve()
    pg95_run = Path(pg95_job["run_dir"]).resolve()
    test_metrics = load(test_run / "metrics.json")
    test_status = load(test_run / "status.json")
    pg95_metrics = load(pg95_run / "metrics.json")
    original = {
        metric: float(test_metrics[f"complete_target_{metric}_on"])
        for metric in ("mse", "mae")
    }
    ensemble_l5 = {
        metric: float(test_metrics[f"complete_target_{metric}_ensemble_l5"])
        for metric in ("mse", "mae")
    }
    multiscale = {
        metric: float(test_metrics[f"complete_target_{metric}_multiscale"])
        for metric in ("mse", "mae")
    }
    real_test_runs = list(
        (ROOT / "runs").glob(
            "cnm_se_equal_rate_multiscale_pg95__traffic__test__seed2026__*"
        )
    )
    skipped_gate_runs = list(
        (ROOT / "runs").glob(
            "stage22_21_scientific_gate__traffic__test_not_authorized__seed2026__*"
        )
    )
    attempt_logs = list(
        (ROOT / "logs/stage22_21_traffic_pipeline").glob(
            "traffic_conditional_single_test_seed2026_formal__attempt02__*.log"
        )
    )
    authorization_logged = any(
        "TRAFFIC_VALIDATION_PASS_ONE_TEST_AUTHORIZED"
        in path.read_text(encoding="utf-8", errors="replace")
        for path in attempt_logs
    )
    checks = {
        "queue_and_required_jobs_success": (
            state.get("queue_state") == "SUCCESS"
            and test_job.get("state") == "SUCCESS"
            and pg95_job.get("state") == "SUCCESS"
        ),
        "validation_gate_passed_before_test": (
            gate.get("status") == "PASS"
            and gate.get("protocol_pass") is True
            and gate.get("test_authorized") is True
            and gate.get("decision")
            == "TRAFFIC_VALIDATION_PASS_ONE_TEST_AUTHORIZED"
            and all(gate.get("protocol_checks", {}).values())
        ),
        "authorization_is_persisted_in_attempt_log": authorization_logged,
        "exactly_one_real_test_run": len(real_test_runs) == 1,
        "previous_safe_skip_is_preserved": len(skipped_gate_runs) >= 1,
        "test_status_and_checkpoint_verified": (
            test_status.get("state") == "SUCCESS"
            and test_status.get("return_code") == 0
            and test_status.get("verification_error") is None
            and test_status.get("checkpoint_load_verified") is True
        ),
        "exact_pg95_checkpoint_handoff": (
            Path(test_metrics["source_checkpoint"]).resolve()
            == Path(pg95_metrics["checkpoint"]).resolve()
        ),
        "complete_test_budget": (
            test_metrics.get("source_split") == "test"
            and test_metrics.get("uses_test_split") is True
            and int(test_metrics["test_batches_processed"]) == 213
            and int(test_metrics["sample_count"]) == 3408
        ),
        "test_artifacts_all_pass": (
            test_metrics.get("artifact_status") == "PASS"
            and all(test_metrics.get("artifact_checks", {}).values())
        ),
        "frozen_test_protocol": (
            test_metrics.get("updates_model_or_memory") is False
            and test_metrics.get("block_lengths") == [1, 5, 20]
            and test_metrics.get("multiscale_weights")
            == {"original": 0.5, "l5": 0.25, "l20": 0.25}
            and float(test_metrics["target_additional_missing_rate"]) == 0.15
        ),
        "multiscale_strictly_improves_original_on_test": all(
            multiscale[metric] < original[metric] for metric in ("mse", "mae")
        ),
        "multiscale_strictly_improves_original_plus_l5_on_test": all(
            multiscale[metric] < ensemble_l5[metric]
            for metric in ("mse", "mae")
        ),
        "aggregated_final_decision_matches": (
            raw.get("test_summary", {}).get("decision")
            == "TRAFFIC_TEST_MULTISCALE_CONFIRMED"
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "test_run": str(test_run),
        "test_batches": int(test_metrics["test_batches_processed"]),
        "test_samples": int(test_metrics["sample_count"]),
        "original": original,
        "original_plus_l5": ensemble_l5,
        "multiscale": multiscale,
        "gains_percent": {
            "multiscale_vs_original": {
                metric: (original[metric] - multiscale[metric])
                / original[metric] * 100.0
                for metric in ("mse", "mae")
            },
            "multiscale_vs_original_plus_l5": {
                metric: (ensemble_l5[metric] - multiscale[metric])
                / ensemble_l5[metric] * 100.0
                for metric in ("mse", "mae")
            },
        },
        "decision": (
            "TRAFFIC_TEST_MULTISCALE_CONFIRMED" if not failures
            else "TRAFFIC_FINAL_AUDIT_FAILED"
        ),
    }
    REPORT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
