#!/usr/bin/env python3
"""CPU contracts for the Step-22.28 dual-scale residual basis."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "baselines/S4M/s4m"),
    str(ROOT / "baselines/S4M/s4m/model"),
    str(ROOT / "baselines/S4M/s4m/model/S4/s4"),
    str(ROOT / "baselines/S4M/s4m/model/S4/s4/extensions/kernels"),
]

from experiments import exp_pretrain1 as exp_module  # noqa: E402
from experiments.exp_pretrain1 import Exp_Long_Term_Forecast as Exp  # noqa: E402
from model.MCPR import (  # noqa: E402
    DualScaleMaskConditionedResidualBasis,
    MaskConditionedParetoResidual,
)
from utils.tools import EarlyStopping  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from run_s4m_experiment import verify_checkpoint  # noqa: E402


class MemoryBox(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.memory = torch.tensor([1.0])
        self.repre = torch.tensor([2.0])
        self.num_dic = torch.tensor([3.0])


class TinyDeltaModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(3, 3)
        self.register_buffer("frozen_blob", torch.zeros(4096))
        self.mcpr = DualScaleMaskConditionedResidualBasis(
            hidden_size=3, num_variables=2, rank=2, kernel_size=2
        )
        self.mem_net = torch.nn.Module()
        self.mem_net.mipp = torch.nn.Linear(1, 1, bias=False)
        self.mem_net.memory = MemoryBox()


def gradient_norm(parameters) -> float:
    return float(sum(
        parameter.grad.detach().double().square().sum().item()
        for parameter in parameters if parameter.grad is not None
    ) ** 0.5)


def main() -> int:
    torch.manual_seed(2027)
    hidden = torch.randn(2, 7, 8)
    mask = torch.ones(2, 7, 5)
    mask[:, 2:4] = 0.0
    backbone = torch.randn(2, 7, 5)
    dmrb = DualScaleMaskConditionedResidualBasis(
        hidden_size=8, num_variables=5, rank=4, kernel_size=3
    )
    initial = dmrb(hidden, mask, backbone)
    l5_pointer = dmrb.l5_basis.hidden_projection.weight.data_ptr()
    l20_pointer = dmrb.l20_basis.hidden_projection.weight.data_ptr()

    with torch.no_grad():
        dmrb.l5_basis.output_projection.weight.fill_(0.02)
        dmrb.l20_basis.output_projection.weight.fill_(-0.01)
    active = dmrb(hidden, mask, backbone)
    dmrb.zero_grad(set_to_none=True)
    active["residual_l5"].square().mean().backward()
    l5_only_gradient = gradient_norm(dmrb.l5_basis.parameters())
    l20_from_l5_gradient = gradient_norm(dmrb.l20_basis.parameters())

    dmrb.zero_grad(set_to_none=True)
    active = dmrb(hidden, mask, backbone)
    (active["residual_l5"].square().mean()
     + active["residual_l20"].square().mean()).backward()
    both_l5_gradient = gradient_norm(dmrb.l5_basis.parameters())
    both_l20_gradient = gradient_norm(dmrb.l20_basis.parameters())

    l5_only = DualScaleMaskConditionedResidualBasis(
        hidden_size=8, num_variables=5, rank=4, kernel_size=3,
        l5_enabled=True, l20_enabled=False,
    )
    with torch.no_grad():
        l5_only.l5_basis.output_projection.bias.fill_(0.1)
    l5_only_outputs = l5_only(hidden, mask, backbone)

    checkpoint_equal = False
    with tempfile.TemporaryDirectory(prefix="dmrb_checkpoint_") as temporary:
        path = Path(temporary) / "dmrb.pt"
        torch.save(dmrb.state_dict(), path)
        restored = DualScaleMaskConditionedResidualBasis(
            hidden_size=8, num_variables=5, rank=4, kernel_size=3
        )
        restored.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
        restored_outputs = restored(hidden, mask, backbone)
        checkpoint_equal = torch.equal(
            active["residual"].detach(), restored_outputs["residual"].detach()
        )

    delta_checkpoint_verified = False
    delta_checkpoint_reconstructs = False
    delta_checkpoint_is_small = False
    with tempfile.TemporaryDirectory(prefix="dmrb_delta_checkpoint_") as temporary:
        temporary_root = Path(temporary)
        source_path = temporary_root / "source.pth"
        trained = TinyDeltaModel()
        source_model_state = {
            key: value.detach().clone()
            for key, value in trained.state_dict().items()
            if not (key.startswith("mcpr.") or ".mcpr." in key)
        }
        torch.save({
            "model": source_model_state,
            "memory": torch.tensor([11.0]),
            "repre": torch.tensor([12.0]),
            "num_dic": torch.tensor([13.0]),
        }, source_path)
        with torch.no_grad():
            trained.backbone.weight.add_(5.0)
            trained.mcpr.l5_basis.output_projection.bias.fill_(0.2)
            trained.mcpr.l20_basis.output_projection.bias.fill_(-0.1)
        arguments = SimpleNamespace(
            model="S4M",
            mcpr_delta_checkpoint=True,
            mcpr_backbone_checkpoint=str(source_path),
        )
        saver = EarlyStopping(arguments, patience=1, more_save=True)
        saver.save_checkpoint(1.0, trained, str(temporary_root))
        delta_path = temporary_root / "checkpoint.pth"
        verification = verify_checkpoint(
            delta_path, require_mipp=True, require_mcpr=True
        )
        delta_checkpoint_verified = (
            verification["checkpoint_format"] == "mcpr_delta_v1"
            and verification["checkpoint_load_verified"] is True
            and verification["checkpoint_adapter_model_key_count"] > 0
        )
        reconstructed = TinyDeltaModel()
        fake_experiment = SimpleNamespace(
            model=reconstructed,
            device=torch.device("cpu"),
            _file_sha256=Exp._file_sha256,
        )
        Exp.load_model(fake_experiment, str(delta_path))
        expected_adapter = {
            key: value for key, value in trained.state_dict().items()
            if key.startswith("mcpr.") or ".mcpr." in key
        }
        reconstructed_state = reconstructed.state_dict()
        delta_checkpoint_reconstructs = (
            all(torch.equal(reconstructed_state[key], value)
                for key, value in source_model_state.items())
            and all(torch.equal(reconstructed_state[key], value)
                    for key, value in expected_adapter.items())
            and torch.equal(reconstructed.mem_net.memory.memory, torch.tensor([11.0]))
            and torch.equal(reconstructed.mem_net.memory.repre, torch.tensor([12.0]))
            and torch.equal(reconstructed.mem_net.memory.num_dic, torch.tensor([13.0]))
        )
        delta_checkpoint_is_small = delta_path.stat().st_size < source_path.stat().st_size

    target = torch.zeros(2, 2, 3)
    objective_backbone = torch.tensor([
        [[2.0, 2.0, 2.0], [1.0, 1.0, 1.0]],
        [[3.0, 3.0, 3.0], [2.0, 2.0, 2.0]],
    ], requires_grad=True)
    teacher_values = torch.tensor([
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    ])
    correction_total = teacher_values - objective_backbone.detach()
    correction_l5 = (0.4 * correction_total).requires_grad_()
    correction_l20 = (0.6 * correction_total).requires_grad_()
    objective_teacher = teacher_values.clone().requires_grad_()
    residual_l5 = torch.zeros_like(target, requires_grad=True)
    residual_l20 = torch.zeros_like(target, requires_grad=True)
    student = objective_backbone.detach() + residual_l5 + residual_l20
    observed = torch.ones_like(target, dtype=torch.bool)
    observed[1, 1] = False
    diagnostics = Exp._dmrb_objective(
        student_prediction=student,
        backbone_prediction=objective_backbone,
        teacher_prediction=objective_teacher,
        residual_l5=residual_l5,
        residual_l20=residual_l20,
        target_correction_l5=correction_l5,
        target_correction_l20=correction_l20,
        target=target,
        target_mask=observed,
    )
    diagnostics["total_loss"].backward()

    bad_reconstruction_rejected = False
    try:
        Exp._dmrb_objective(
            student_prediction=student.detach(),
            backbone_prediction=objective_backbone.detach(),
            teacher_prediction=objective_teacher.detach() + 0.1,
            residual_l5=residual_l5.detach(),
            residual_l20=residual_l20.detach(),
            target_correction_l5=correction_l5.detach(),
            target_correction_l20=correction_l20.detach(),
            target=target,
            target_mask=observed,
        )
    except RuntimeError:
        bad_reconstruction_rejected = True

    empty_switch_rejected = False
    try:
        DualScaleMaskConditionedResidualBasis(
            hidden_size=8, num_variables=5, rank=4, kernel_size=3,
            l5_enabled=False, l20_enabled=False,
        )
    except ValueError:
        empty_switch_rejected = True

    formal_dmrb = DualScaleMaskConditionedResidualBasis(
        hidden_size=512, num_variables=862, rank=32, kernel_size=5
    )
    formal_parameter_count = sum(p.numel() for p in formal_dmrb.parameters())
    single_mcpr = MaskConditionedParetoResidual(
        hidden_size=512, num_variables=862, rank=32, kernel_size=5
    )
    single_parameter_count = sum(p.numel() for p in single_mcpr.parameters())

    checks = {
        "zero_initialization_recovers_backbone": (
            torch.count_nonzero(initial["residual"]).item() == 0
            and torch.count_nonzero(initial["residual_l5"]).item() == 0
            and torch.count_nonzero(initial["residual_l20"]).item() == 0
        ),
        "basis_parameters_are_not_shared": l5_pointer != l20_pointer,
        "fixed_sum_identity": torch.equal(
            active["residual"],
            active["residual_l5"] + active["residual_l20"],
        ),
        "l5_loss_does_not_reach_l20_basis": (
            l5_only_gradient > 0.0 and l20_from_l5_gradient == 0.0
        ),
        "both_basis_gradients_are_nonzero": (
            both_l5_gradient > 0.0 and both_l20_gradient > 0.0
        ),
        "independent_l20_switch_returns_exact_zero": (
            torch.count_nonzero(l5_only_outputs["residual_l20"]).item() == 0
            and torch.equal(
                l5_only_outputs["residual"], l5_only_outputs["residual_l5"]
            )
        ),
        "checkpoint_round_trip_is_exact": checkpoint_equal,
        "delta_checkpoint_is_verified": delta_checkpoint_verified,
        "delta_checkpoint_reconstructs_source_plus_adapter": (
            delta_checkpoint_reconstructs
        ),
        "delta_checkpoint_does_not_duplicate_full_source": delta_checkpoint_is_small,
        "strict_bimetric_token_filter_is_reused": torch.equal(
            diagnostics["beneficial_mask"],
            torch.tensor([[True, False], [True, False]]),
        ),
        "scale_losses_are_positive": (
            float(diagnostics["dmrb_l5_distill_l1"].item()) > 0.0
            and float(diagnostics["dmrb_l20_distill_l1"].item()) > 0.0
        ),
        "both_scale_targets_backpropagate": (
            residual_l5.grad is not None
            and residual_l20.grad is not None
            and float(residual_l5.grad.abs().sum().item()) > 0.0
            and float(residual_l20.grad.abs().sum().item()) > 0.0
        ),
        "teacher_and_backbone_are_detached": (
            objective_teacher.grad is None
            and objective_backbone.grad is None
            and correction_l5.grad is None
            and correction_l20.grad is None
        ),
        "teacher_factorization_is_exact": float(
            diagnostics["dmrb_teacher_reconstruction_max_abs"].item()
        ) == 0.0,
        "fp32_reconstruction_tolerance_is_frozen": (
            exp_module.DMRB_TEACHER_RECONSTRUCTION_ATOL
            == float(32.0 * torch.finfo(torch.float32).eps)
            and 1.9073486328125e-6
            < exp_module.DMRB_TEACHER_RECONSTRUCTION_ATOL
            < 4e-6
        ),
        "bad_teacher_factorization_is_rejected": bad_reconstruction_rejected,
        "empty_scale_selection_is_rejected": empty_switch_rejected,
        "formal_single_head_parameter_count_is_frozen": (
            single_parameter_count == 45182
        ),
        "formal_dual_head_parameter_count_is_frozen": (
            formal_parameter_count == 90364
            and formal_parameter_count == 2 * single_parameter_count
        ),
    }
    report = {
        "step": "22.28",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "single_mcpr_parameter_count": single_parameter_count,
        "dmrb_parameter_count": formal_parameter_count,
    }
    output = ROOT / "reports/stage22_28_dmrb_component_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
