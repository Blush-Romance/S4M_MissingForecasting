#!/usr/bin/env python3
"""Zero-GPU helper checks for the Step-22 scientific gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_stage22_innovation import (  # noqa: E402
    candidate_gate,
    normalized_training_args,
)


def main() -> int:
    anchor = {"mse": 1.0, "mae": 0.5}
    base_args = {
        "batch_size": 16, "train_epochs": 15, "loss": "MSE",
        "model_id": "base", "des": "BASE",
    }
    candidate_args = {
        "batch_size": 16, "train_epochs": 15,
        "model_id": "candidate", "des": "CAND", "use_mrra": True,
        "mrra_rank": 32, "mrra_kernel_size": 5,
        "mrra_reconstruction_weight": 0.1,
    }
    checks = {
        "dual_improvement_passes": candidate_gate(
            anchor, {"mse": 0.99, "mae": 0.49}
        )["pass"],
        "mse_only_mae_degrade_fails": not candidate_gate(
            anchor, {"mse": 0.90, "mae": 0.51}
        )["pass"],
        "mae_only_mse_degrade_fails": not candidate_gate(
            anchor, {"mse": 1.01, "mae": 0.45}
        )["pass"],
        "exact_tie_fails_strict_clause": not candidate_gate(
            anchor, {"mse": 1.0, "mae": 0.5}
        )["pass"],
        "innovation_identity_is_ignored": (
            normalized_training_args(base_args)
            == normalized_training_args(candidate_args)
        ),
        "real_budget_drift_is_detected": (
            normalized_training_args(base_args)
            != normalized_training_args({**candidate_args, "batch_size": 8})
        ),
    }
    payload = {
        "step": 22,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    report = ROOT / "reports" / "stage22_innovation_analysis_test.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(report)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
