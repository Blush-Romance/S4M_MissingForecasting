#!/usr/bin/env python3
"""Positive and negative tests for the Stage-14 equal-ensemble gate."""

from __future__ import annotations

import json
import os
from pathlib import Path

from analyze_stage14_equal_ensemble import evaluate_metrics


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "stage14_equal_ensemble_analysis_test.json"


def metrics(ensemble_mse: float, ensemble_mae: float) -> dict:
    return {
        "complete_target_mse_on": 1.0,
        "complete_target_mae_on": 1.0,
        "complete_target_mse_nested": 0.99,
        "complete_target_mae_nested": 0.995,
        "complete_target_mse_ensemble": ensemble_mse,
        "complete_target_mae_ensemble": ensemble_mae,
        "artifact_status": "PASS",
        "uses_test_split": False,
        "source_split": "val",
        "updates_model_or_memory": False,
        "ensemble_weight_on": 0.5,
        "ensemble_weight_nested": 0.5,
    }


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    positive = evaluate_metrics(metrics(0.98, 0.99), 0.98)
    weak_mae = evaluate_metrics(metrics(0.98, 0.997), 0.98)
    wrong_identity = evaluate_metrics(metrics(0.98, 0.99), 0.97)
    checks = {
        "positive_passes": positive["gate_pass"] is True,
        "weak_mae_fails": weak_mae["gate_pass"] is False,
        "identity_mismatch_fails": wrong_identity["gate_pass"] is False,
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
