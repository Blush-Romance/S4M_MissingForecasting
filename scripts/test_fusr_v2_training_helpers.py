#!/usr/bin/env python3
"""CPU tests for the frozen-router temporal split, loss, and calibration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
S4M_ROOT = ROOT / "baselines" / "S4M" / "s4m"
MODEL_ROOT = S4M_ROOT / "model"
S4_ROOT = MODEL_ROOT / "S4" / "s4"
KERNEL_ROOT = S4_ROOT / "extensions" / "kernels"
for path in (S4M_ROOT, MODEL_ROOT, S4_ROOT, KERNEL_ROOT):
    sys.path.insert(0, str(path))

from experiments.exp_pretrain1 import Exp_Long_Term_Forecast
from model.FUSR import ForecastUtilityRouter, RetrievalInterventionFeaturizer


def main() -> int:
    torch.manual_seed(2023)
    checks = {}

    split = Exp_Long_Term_Forecast._fusr_v2_split_indices(
        sample_count=1000,
        fit_fraction=0.55,
        dev_fraction=0.20,
        requested_embargo=80,
    )
    fit = split["fit"]
    dev = split["dev"]
    calibration = split["calibration"]
    checks["chronological_blocks_are_ordered"] = bool(
        fit[-1] < dev[0] < dev[-1] < calibration[0]
    )
    checks["requested_embargo_is_preserved"] = bool(
        split["effective_embargo_windows"] == 80
        and int(dev[0] - fit[-1] - 1) == 80
        and int(calibration[0] - dev[-1] - 1) == 80
    )
    all_indices = torch.cat((fit, dev, calibration))
    checks["split_has_no_overlap"] = bool(torch.unique(all_indices).numel() == all_indices.numel())

    feature_dim = RetrievalInterventionFeaturizer.FEATURE_DIM
    features = torch.randn(64, feature_dim)
    utility = torch.linspace(-0.8, 0.8, 64)
    labels = (utility > 0.0).float()
    router = ForecastUtilityRouter(
        objective_version="v2_frozen_balanced_rank",
        feature_set="intervention14",
        gate_score="sign_logit",
        gate_prior_mode="none",
    )
    router.fit_feature_normalizer(features)
    total, decision, ranking, pairs, outputs = (
        Exp_Long_Term_Forecast._fusr_v2_router_loss(
            router,
            features,
            utility,
            labels,
            rank_weight=0.25,
            class_weights=(1.0, 1.0),
        )
    )
    total.backward()
    gradient_norm = sum(
        float(parameter.grad.abs().sum().item())
        for parameter in router.parameters()
        if parameter.grad is not None
    )
    checks["balanced_rank_loss_is_finite"] = bool(
        torch.isfinite(total) and torch.isfinite(decision) and torch.isfinite(ranking)
    )
    checks["balanced_rank_loss_generates_pairs"] = pairs == 32
    checks["balanced_rank_loss_backpropagates"] = gradient_norm > 0.0
    checks["router_score_is_target_free_tensor"] = outputs["gate_score"].shape == (64,)

    scores = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
    mse_on = np.array([0.1, 0.2, 0.3, 0.9, 1.0, 1.1])
    mse_off = np.array([0.8, 0.7, 0.6, 0.2, 0.2, 0.2])
    calibrated = Exp_Long_Term_Forecast._fusr_v2_calibrate_threshold(
        scores, mse_on, mse_off, "prefer_on"
    )
    checks["calibration_selects_expected_gate"] = calibrated["gate"].tolist() == [True, True, True, False, False, False]
    checks["calibration_improves_over_always_on"] = calibrated["risk"] < calibrated["always_on_risk"]
    checks["calibration_threshold_is_finite"] = bool(np.isfinite(calibrated["threshold"]))

    passed = sum(bool(value) for value in checks.values())
    payload = {
        "status": "PASS" if passed == len(checks) else "FAIL",
        "passed": passed,
        "total": len(checks),
        "checks": checks,
        "evidence": {
            "effective_embargo_windows": split["effective_embargo_windows"],
            "gradient_norm": gradient_norm,
            "rank_pairs": pairs,
            "calibrated_threshold": calibrated["threshold"],
            "calibration_gate_rate": calibrated["gate_rate"],
            "calibration_risk": calibrated["risk"],
            "always_on_risk": calibrated["always_on_risk"],
        },
    }
    report = ROOT / "reports" / "stage8_fusr_v2_training_helper_test.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"report={report}")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
