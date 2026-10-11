#!/usr/bin/env python3
"""CPU component checks for MIPP before any formal GPU experiment."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
S4M = ROOT / "baselines" / "S4M" / "s4m"
S4_MODEL = S4M / "model"
S4_SOURCE = S4_MODEL / "S4" / "s4"
STRUCTURED_KERNELS = S4_SOURCE / "extensions" / "kernels"
for source_root in (S4M, S4_MODEL, S4_SOURCE, STRUCTURED_KERNELS):
    sys.path.insert(0, str(source_root))

from model.MIPP import MIPPPurifier  # noqa: E402
from model.Memory import FeatureMemory  # noqa: E402
from experiments.exp_pretrain1 import Exp_Long_Term_Forecast  # noqa: E402


def main() -> int:
    torch.manual_seed(2023)
    purifier = MIPPPurifier(hidden_size=16, num_variables=5)
    features_a = torch.randn(4, 12, 16, requires_grad=True)
    features_b = (features_a.detach() + 0.05 * torch.randn(4, 12, 16)).requires_grad_()
    mask_a = (torch.rand(4, 12, 5) > 0.2).float()
    mask_b = (torch.rand(4, 12, 5) > 0.4).float()

    raw_a, content_a = purifier.encode_content(features_a)
    raw_b, content_b = purifier.encode_content(features_b)
    identity_error = float((raw_a.detach() - features_a.detach()).abs().max().item())
    outputs = purifier.paired_objective(
        raw_a, content_a, mask_a, raw_b, content_b, mask_b, use_factorization=True
    )
    stability = torch.tensor([[0.10, 0.90, 0.30, 0.80]])
    absolute_mask, absolute_diagnostics = purifier.select_write_candidates(
        stability,
        mode="absolute",
        threshold=0.85,
        budget=2,
    )
    budget_mask, budget_diagnostics = purifier.select_write_candidates(
        stability,
        mode="budget_topk",
        threshold=0.85,
        budget=2,
    )
    empty_absolute_mask, empty_absolute_diagnostics = purifier.select_write_candidates(
        torch.tensor([[0.10, 0.20]]),
        mode="absolute",
        threshold=0.85,
        budget=2,
    )
    total = sum(
        outputs[key]
        for key in (
            "invariance_loss",
            "variance_loss",
            "covariance_loss",
            "decorrelation_loss",
            "mask_reconstruction_loss",
        )
    )
    total.backward()

    memory_config = type(
        "MemoryConfig",
        (),
        {
            "memory_size": 10,
            "K": 1,
            "M": 10,
            "per_mem_size": 3,
            "thres1": 0.9,
            "thres2": 0.1,
        },
    )()
    memory = FeatureMemory(memory_config)
    memory.memory = [torch.tensor([[1.0, 0.0]])]
    memory.repre = torch.tensor([[1.0, 0.0]])
    memory.num_dic[0] = 1
    write = memory.add_features_from_sample_learned(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]),
        eligible_mask=torch.tensor([[True, False, False]]),
    )
    empty_write = memory.add_features_from_sample_learned(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        eligible_mask=torch.tensor([[False, False]]),
    )
    experiment = object.__new__(Exp_Long_Term_Forecast)
    experiment.args = SimpleNamespace(
        use_mipp=True,
        use_mask_intervention_pair=True,
        mipp_pair_block_lengths=[20],
        mipp_pair_rate_min=0.35,
        mipp_pair_rate_max=0.35,
    )
    paired_view = experiment._build_mipp_pair(torch.randn(4, 192, 5))

    metrics = {key: float(value.detach().item()) for key, value in outputs.items()}
    checks = {
        "zero_initialized_content_identity": identity_error == 0.0,
        "content_shape_preserved": tuple(content_a.shape) == (4, 12, 16),
        "mask_statistics_shape": tuple(purifier.mask_statistics(mask_a).shape) == (4, 10),
        "all_losses_finite": all(math.isfinite(value) for value in metrics.values()),
        "content_adapter_receives_gradient": (
            purifier.content_adapter.weight.grad is not None
            and bool(torch.isfinite(purifier.content_adapter.weight.grad).all())
            and float(purifier.content_adapter.weight.grad.abs().sum().item()) > 0.0
        ),
        "mask_encoder_receives_gradient": (
            purifier.mask_encoder[0].weight.grad is not None
            and float(purifier.mask_encoder[0].weight.grad.abs().sum().item()) > 0.0
        ),
        "write_gate_filters_candidates": write == {
            "eligible_count": 1,
            "sampled_count": 1,
            "updated_count": 1,
        },
        "absolute_gate_preserves_threshold_behavior": absolute_mask.tolist()
        == [[False, True, False, False]],
        "absolute_gate_allows_empty_admission": empty_absolute_mask.tolist()
        == [[False, False]],
        "empty_absolute_gate_reports_threshold_cutoff": math.isclose(
            float(empty_absolute_diagnostics["write_effective_cutoff"].item()),
            0.85,
            rel_tol=0.0,
            abs_tol=1e-7,
        ),
        "empty_absolute_gate_reports_zero_acceptance": math.isclose(
            float(
                empty_absolute_diagnostics[
                    "write_effective_acceptance_rate"
                ].item()
            ),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-8,
        ),
        "memory_accepts_empty_eligible_set": empty_write == {
            "eligible_count": 0,
            "sampled_count": 0,
            "updated_count": 0,
        },
        "budget_topk_selects_exact_budget": budget_mask.tolist()
        == [[False, True, False, True]],
        "budget_topk_reports_half_acceptance": math.isclose(
            float(budget_diagnostics["write_effective_acceptance_rate"].item()),
            0.5,
            rel_tol=0.0,
            abs_tol=1e-8,
        ),
        "budget_topk_cutoff_is_selected_minimum": math.isclose(
            float(budget_diagnostics["write_effective_cutoff"].item()),
            0.8,
            rel_tol=0.0,
            abs_tol=1e-7,
        ),
        "length20_block_pair_shape": tuple(paired_view["seq_x_mask"].shape) == (4, 192, 5),
        "length20_block_pair_rate_is_finite": math.isfinite(
            paired_view["realized_missing_rate"]
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "identity_error": identity_error,
        "metrics": metrics,
        "write_gate_diagnostics": write,
        "absolute_selection": {
            "mask": absolute_mask.tolist(),
            "acceptance_rate": float(
                absolute_diagnostics["write_effective_acceptance_rate"].item()
            ),
        },
        "empty_absolute_selection": {
            "mask": empty_absolute_mask.tolist(),
            "acceptance_rate": float(
                empty_absolute_diagnostics[
                    "write_effective_acceptance_rate"
                ].item()
            ),
            "effective_cutoff": float(
                empty_absolute_diagnostics["write_effective_cutoff"].item()
            ),
        },
        "budget_topk_selection": {
            "mask": budget_mask.tolist(),
            "acceptance_rate": float(
                budget_diagnostics["write_effective_acceptance_rate"].item()
            ),
            "effective_cutoff": float(
                budget_diagnostics["write_effective_cutoff"].item()
            ),
        },
        "length20_block_realized_missing_rate": paired_view["realized_missing_rate"],
    }
    output = ROOT / "reports" / "stage9_bmsa_component_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
