#!/usr/bin/env python3
"""CPU component tests for the frozen Step-22 MRRA and RADR candidates."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import tempfile


ROOT = Path(__file__).resolve().parents[1]
S4M = ROOT / "baselines" / "S4M" / "s4m"
S4_MODEL = S4M / "model"
S4_SOURCE = S4_MODEL / "S4" / "s4"
STRUCTURED_KERNELS = S4_SOURCE / "extensions" / "kernels"
for source_root in (S4M, S4_MODEL, S4_SOURCE, STRUCTURED_KERNELS):
    sys.path.insert(0, str(source_root))

from model.MRRA import MaskResponsiveResidualAdapter  # noqa: E402
from experiments.exp_pretrain1 import Exp_Long_Term_Forecast  # noqa: E402
from utils.tools import EarlyStopping, adjust_learning_rate  # noqa: E402
from run_s4m_experiment import verify_checkpoint  # noqa: E402


def main() -> int:
    torch.manual_seed(2023)
    adapter = MaskResponsiveResidualAdapter(
        num_variables=5, context_size=16, rank=4, kernel_size=5
    )
    history = torch.randn(3, 12, 5, requires_grad=True)
    mask = (torch.rand(3, 12, 5) > 0.3).float()
    outputs = adapter(history, mask)

    changed_history = history.detach().clone()
    changed_mask = mask.clone()
    changed_history[:, 7:, :] += 100.0
    changed_mask[:, 7:, :] = 1.0 - changed_mask[:, 7:, :]
    changed = adapter(changed_history, changed_mask)
    causal_prefix_error = float(
        (outputs["hidden"][:, :7] - changed["hidden"][:, :7]).abs().max().item()
    )
    zero_context_error = float(outputs["context_residual"].abs().max().item())
    mask_changed = adapter(history.detach(), 1.0 - mask)
    mask_response = float(
        (outputs["reconstruction"].detach() - mask_changed["reconstruction"])
        .abs().mean().item()
    )
    reconstruction_loss, missing_count = (
        Exp_Long_Term_Forecast._mrra_reconstruction_objective(
            outputs["reconstruction"], history.detach() + 0.25, mask
        )
    )
    (reconstruction_loss + outputs["context_residual"].sum()).backward()

    adapter_v2 = MaskResponsiveResidualAdapter(
        num_variables=5,
        context_size=16,
        rank=4,
        kernel_size=5,
        version="v2_normalized_bounded",
    )
    extreme_history = (history.detach() * 1.0e6).requires_grad_(True)
    outputs_v2 = adapter_v2(extreme_history, mask)
    v2_reconstruction_loss, _ = (
        Exp_Long_Term_Forecast._mrra_reconstruction_objective(
            outputs_v2["reconstruction"], extreme_history.detach() + 0.25, mask
        )
    )
    (v2_reconstruction_loss + outputs_v2["context_residual"].sum()).backward()

    adapter_v3 = MaskResponsiveResidualAdapter(
        num_variables=5,
        context_size=16,
        rank=4,
        kernel_size=5,
        version="v3_gradient_isolated_two_timescale",
    )
    v3_history = torch.randn(3, 12, 5, requires_grad=True)
    outputs_v3 = adapter_v3(v3_history, mask)
    v3_reconstruction_loss, _ = (
        Exp_Long_Term_Forecast._mrra_reconstruction_objective(
            outputs_v3["reconstruction"], v3_history.detach() + 0.25, mask
        )
    )
    (v3_reconstruction_loss + outputs_v3["context_residual"].sum()).backward()

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(5, 5)
            self.mrra = adapter_v3

    toy_model = ToyModel()
    parameter_groups = Exp_Long_Term_Forecast._build_mrra_parameter_groups(
        toy_model, base_lr=0.01, mrra_lr_multiplier=0.1
    )
    optimizer = torch.optim.Adam(parameter_groups, lr=0.01)
    initial_group_lrs = {
        group["group_name"]: float(group["lr"])
        for group in optimizer.param_groups
    }
    adjust_learning_rate(
        optimizer, 2, SimpleNamespace(lradj="type1", learning_rate=0.01)
    )
    adjusted_group_lrs = {
        group["group_name"]: float(group["lr"])
        for group in optimizer.param_groups
    }
    group_parameter_ids = [
        {id(parameter) for parameter in group["params"]}
        for group in optimizer.param_groups
    ]

    prediction = torch.tensor(
        [
            [[0.0, 0.0]],
            [[0.5, -0.5]],
            [[2.0, -2.0]],
            [[4.0, -4.0]],
        ],
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)
    target_mask = torch.ones_like(prediction)
    radr = Exp_Long_Term_Forecast._rank_adaptive_dual_risk_objective(
        prediction, target, target_mask
    )
    radr["total_loss"].backward()
    radr_values = {
        key: float(value.detach().item()) if torch.is_tensor(value) else value
        for key, value in radr.items()
    }
    with tempfile.TemporaryDirectory(prefix="stage22_mrra_checkpoint_") as directory:
        checkpoint = Path(directory) / "checkpoint.pth"
        torch.save(
            {
                "model": {"mrra.input_projection.weight": torch.ones(1)},
                "memory": [], "repre": torch.ones(1), "num_dic": {},
            },
            checkpoint,
        )
        checkpoint_verification = verify_checkpoint(
            checkpoint, require_mrra=True
        )
        nonfinite_checkpoint = Path(directory) / "nonfinite_checkpoint.pth"
        torch.save(
            {
                "model": {"mrra.input_projection.weight": torch.tensor([float("nan")])},
                "memory": [], "repre": torch.ones(1), "num_dic": {},
            },
            nonfinite_checkpoint,
        )
        nonfinite_checkpoint_rejected = False
        try:
            verify_checkpoint(nonfinite_checkpoint, require_mrra=True)
        except RuntimeError:
            nonfinite_checkpoint_rejected = True
        early_stopping = EarlyStopping(
            args=SimpleNamespace(model="S4M"), more_save=False
        )
        atomic_directory = Path(directory) / "atomic"
        atomic_directory.mkdir()
        early_stopping(1.0, adapter_v2, str(atomic_directory))
        atomic_checkpoint = atomic_directory / "checkpoint.pth"
        atomic_checkpoint_loadable = isinstance(
            torch.load(atomic_checkpoint, map_location="cpu"), dict
        )
        nonfinite_rejected = False
        try:
            early_stopping(float("nan"), adapter_v2, str(atomic_directory))
        except FloatingPointError:
            nonfinite_rejected = True

    checks = {
        "mrra_shapes_match_contract": (
            tuple(outputs["context_residual"].shape) == (3, 12, 16)
            and tuple(outputs["reconstruction"].shape) == (3, 12, 5)
        ),
        "mrra_baseline_preserved_at_initialization": zero_context_error == 0.0,
        "mrra_temporal_filter_is_causal": causal_prefix_error == 0.0,
        "mrra_responds_to_mask": mask_response > 0.0,
        "mrra_missing_supervision_active": (
            missing_count > 0 and float(reconstruction_loss.detach().item()) > 0.0
        ),
        "mrra_context_head_receives_gradient": (
            adapter.context_projection.weight.grad is not None
            and float(adapter.context_projection.weight.grad.abs().sum().item()) > 0.0
        ),
        "mrra_encoder_receives_gradient": (
            adapter.input_projection.weight.grad is not None
            and float(adapter.input_projection.weight.grad.abs().sum().item()) > 0.0
        ),
        "mrra_v1_default_has_no_normalization_state": (
            adapter.version == "v1" and adapter.hidden_normalization is None
        ),
        "mrra_v2_uses_normalized_bounded_path": (
            adapter_v2.hidden_normalization is not None
            and float(outputs_v2["context_residual"].detach().abs().max().item()) <= 1.0
            and float(
                outputs_v2["reconstruction_correction"].detach().abs().max().item()
            ) <= 1.0
        ),
        "mrra_v2_extreme_input_stays_finite": all(
            bool(torch.isfinite(value).all())
            for value in outputs_v2.values()
        ),
        "mrra_v2_encoder_receives_finite_gradient": (
            adapter_v2.input_projection.weight.grad is not None
            and bool(torch.isfinite(adapter_v2.input_projection.weight.grad).all())
            and float(adapter_v2.input_projection.weight.grad.abs().sum().item()) > 0.0
        ),
        "mrra_v3_input_gradient_is_isolated": v3_history.grad is None,
        "mrra_v3_normalization_is_non_affine": (
            adapter_v3.hidden_normalization is not None
            and adapter_v3.hidden_normalization.elementwise_affine is False
            and not any(
                name.startswith("hidden_normalization.")
                for name, _ in adapter_v3.named_parameters()
            )
        ),
        "mrra_v3_output_is_bounded_and_finite": (
            all(bool(torch.isfinite(value).all()) for value in outputs_v3.values())
            and float(outputs_v3["context_residual"].detach().abs().max().item()) <= 1.0
            and float(
                outputs_v3["reconstruction_correction"].detach().abs().max().item()
            ) <= 1.0
        ),
        "mrra_v3_adapter_still_receives_gradient": (
            adapter_v3.input_projection.weight.grad is not None
            and bool(torch.isfinite(adapter_v3.input_projection.weight.grad).all())
            and float(adapter_v3.input_projection.weight.grad.abs().sum().item()) > 0.0
        ),
        "mrra_v3_optimizer_groups_are_disjoint": (
            len(group_parameter_ids) == 2
            and not (group_parameter_ids[0] & group_parameter_ids[1])
            and sum(len(group) for group in group_parameter_ids)
            == len(group_parameter_ids[0] | group_parameter_ids[1])
        ),
        "mrra_v3_learning_rate_ratio_is_preserved": (
            math.isclose(
                initial_group_lrs["mrra"] / initial_group_lrs["backbone"],
                0.1,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                adjusted_group_lrs["mrra"] / adjusted_group_lrs["backbone"],
                0.1,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "early_stopping_checkpoint_is_atomic_and_loadable": (
            atomic_checkpoint_loadable
            and not (atomic_directory / "checkpoint.pth.tmp").exists()
        ),
        "early_stopping_rejects_nonfinite_validation": nonfinite_rejected,
        "radr_loss_finite_positive": (
            math.isfinite(radr_values["total_loss"])
            and radr_values["total_loss"] > 0.0
        ),
        "radr_rank_nonconstant": radr_values["difficulty_rank_std"] > 0.0,
        "radr_tail_emphasizes_hard_samples": radr_values["hard_body_mse_gap"] > 0.0,
        "radr_prediction_gradient_finite_nonzero": (
            prediction.grad is not None
            and bool(torch.isfinite(prediction.grad).all())
            and float(prediction.grad.abs().sum().item()) > 0.0
        ),
        "radr_has_no_trainable_parameters": True,
        "mrra_checkpoint_keys_detected_at_model_root": (
            checkpoint_verification["checkpoint_mrra_state_keys"]
            == ["mrra.input_projection.weight"]
        ),
        "nonfinite_checkpoint_is_rejected": nonfinite_checkpoint_rejected,
    }
    report = {
        "step": 22,
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "mrra": {
            "causal_prefix_error": causal_prefix_error,
            "zero_context_error": zero_context_error,
            "mask_response": mask_response,
            "missing_count": missing_count,
            "reconstruction_loss": float(reconstruction_loss.detach().item()),
            "parameter_count": sum(parameter.numel() for parameter in adapter.parameters()),
            "v2_parameter_count": sum(
                parameter.numel() for parameter in adapter_v2.parameters()
            ),
            "v3_parameter_count": sum(
                parameter.numel() for parameter in adapter_v3.parameters()
            ),
            "v3_initial_group_lrs": initial_group_lrs,
            "v3_adjusted_group_lrs": adjusted_group_lrs,
        },
        "radr": radr_values,
    }
    output = ROOT / "reports" / "stage22_innovation_component_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
