#!/usr/bin/env python3
"""Positive and negative tests for the frozen Stage-15 decision."""

from __future__ import annotations

import json
import os
from pathlib import Path

from analyze_stage15_cnm_se import decide, evaluate_seed


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "stage15_cnm_se_analysis_test.json"


def metrics(mse: float, mae: float) -> dict:
    return {
        "complete_target_mse_on": 1.0,
        "complete_target_mae_on": 1.0,
        "complete_target_mse_nested": 0.99,
        "complete_target_mae_nested": 0.995,
        "complete_target_mse_ensemble": mse,
        "complete_target_mae_ensemble": mae,
        "artifact_status": "PASS",
        "uses_test_split": True,
        "source_split": "test",
        "updates_model_or_memory": False,
        "ensemble_weight_on": 0.5,
        "ensemble_weight_nested": 0.5,
        "evaluation_elapsed_seconds": 1.0,
        "inference_forward_multiplier": 2.0,
        "peak_gpu_memory_mb": 1.0,
    }


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    positive_results = {
        "2024": evaluate_seed(metrics(0.98, 0.99)),
        "2025": evaluate_seed(metrics(0.98, 0.99)),
    }
    one_seed_regresses = {
        "2024": evaluate_seed(metrics(0.98, 0.99)),
        "2025": evaluate_seed(metrics(1.01, 0.99)),
    }
    weak_mean = {
        "2024": evaluate_seed(metrics(0.995, 0.996)),
        "2025": evaluate_seed(metrics(0.995, 0.996)),
    }
    checks = {
        "positive_passes": decide(positive_results)["gate_pass"] is True,
        "one_seed_regression_fails": decide(one_seed_regresses)["gate_pass"] is False,
        "weak_mean_fails": decide(weak_mean)["gate_pass"] is False,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    atomic_json(REPORT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
