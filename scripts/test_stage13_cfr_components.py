#!/usr/bin/env python3
"""CPU regression tests for Step-13 counterfactual regret regularization."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
S4M = ROOT / "baselines" / "S4M" / "s4m"
S4_MODEL = S4M / "model"
S4_SOURCE = S4_MODEL / "S4" / "s4"
STRUCTURED_KERNELS = S4_SOURCE / "extensions" / "kernels"
for source_root in (S4M, S4_MODEL, S4_SOURCE, STRUCTURED_KERNELS):
    sys.path.insert(0, str(source_root))

from experiments.exp_pretrain1 import Exp_Long_Term_Forecast  # noqa: E402


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    prediction_on = torch.tensor(
        [[[2.0], [0.0]], [[0.5], [9.0]]], requires_grad=True
    )
    prediction_off = torch.tensor(
        [[[1.0], [0.0]], [[1.0], [7.0]]], requires_grad=True
    )
    target = torch.zeros_like(prediction_on)
    target_mask = torch.tensor(
        [[[1.0], [1.0]], [[1.0], [0.0]]]
    )
    result = Exp_Long_Term_Forecast._counterfactual_regret_objective(
        prediction_on, prediction_off, target, target_mask
    )
    result["total_loss"].backward()

    beneficial_on = torch.tensor([[[0.0], [0.0]]], requires_grad=True)
    harmful_off = torch.tensor([[[1.0], [1.0]]], requires_grad=True)
    zero_result = Exp_Long_Term_Forecast._counterfactual_regret_objective(
        beneficial_on,
        harmful_off,
        torch.zeros_like(beneficial_on),
        torch.ones_like(beneficial_on),
    )

    mismatched_shape_rejected = False
    try:
        Exp_Long_Term_Forecast._counterfactual_regret_objective(
            prediction_on,
            prediction_off[:, :1],
            target,
            target_mask,
        )
    except ValueError:
        mismatched_shape_rejected = True

    empty_target_rejected = False
    try:
        Exp_Long_Term_Forecast._counterfactual_regret_objective(
            prediction_on,
            prediction_off,
            target,
            torch.zeros_like(target_mask),
        )
    except ValueError:
        empty_target_rejected = True

    scalar = {
        key: (float(value.detach().item()) if torch.is_tensor(value) else int(value))
        for key, value in result.items()
    }
    checks = {
        "sample_count_respects_per_sample_mask": scalar["sample_count"] == 2,
        "prototype_on_mse_exact": math.isclose(
            scalar["prototype_on_mse"], 1.125, rel_tol=0.0, abs_tol=1e-7
        ),
        "prototype_off_mse_exact": math.isclose(
            scalar["prototype_off_mse"], 0.75, rel_tol=0.0, abs_tol=1e-7
        ),
        "signed_regret_exact": math.isclose(
            scalar["signed_regret"], 0.375, rel_tol=0.0, abs_tol=1e-7
        ),
        "positive_regret_exact": math.isclose(
            scalar["positive_regret"], 0.75, rel_tol=0.0, abs_tol=1e-7
        ),
        "harmful_rate_exact": math.isclose(
            scalar["harmful_retrieval_rate"], 0.5, rel_tol=0.0, abs_tol=1e-7
        ),
        "loss_equals_unweighted_positive_regret": math.isclose(
            scalar["total_loss"], scalar["positive_regret"], rel_tol=0.0, abs_tol=1e-8
        ),
        "prototype_on_receives_gradient": (
            prediction_on.grad is not None
            and float(prediction_on.grad.abs().sum().item()) > 0.0
        ),
        "prototype_off_reference_is_detached": prediction_off.grad is None,
        "beneficial_retrieval_has_zero_penalty": math.isclose(
            float(zero_result["total_loss"].item()), 0.0, rel_tol=0.0, abs_tol=1e-8
        ),
        "mismatched_shape_rejected": mismatched_shape_rejected,
        "empty_target_rejected": empty_target_rejected,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": scalar,
        "frozen_objective": (
            "forecast_mse + mean(relu(sample_mse_on - "
            "stopgrad(sample_mse_off)))"
        ),
    }
    atomic_json(ROOT / "reports" / "stage13_cfr_component_test.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
