#!/usr/bin/env python3
"""CPU contracts for Step-22.32 HFD-PBA."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "baselines/S4M/s4m"),
    str(ROOT / "baselines/S4M/s4m/model"),
    str(ROOT / "baselines/S4M/s4m/model/S4/s4"),
    str(ROOT / "baselines/S4M/s4m/model/S4/s4/extensions/kernels"),
]
from experiments.exp_pretrain1 import Exp_Long_Term_Forecast as Exp  # noqa: E402
from model.S4M import Model as S4MModel  # noqa: E402


class FirstLayer(torch.nn.Module):
    def forward(self, value, _mask):
        return value + 1.0


class TailLayer(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))

    def forward(self, value):
        return value * self.scale


class TinyCapture(torch.nn.Module):
    set_hfd_feature_capture = S4MModel.set_hfd_feature_capture
    get_last_hfd_features = S4MModel.get_last_hfd_features
    _forecast_from_context = S4MModel._forecast_from_context

    def __init__(self):
        super().__init__()
        self.configs = SimpleNamespace(e_layers=8)
        self.s4_layers = torch.nn.ModuleList(
            [FirstLayer()] + [TailLayer(1.0 + index * 0.01) for index in range(1, 8)]
        )
        self.linear1 = torch.nn.Identity()
        self.norm = torch.nn.Identity()
        self.out_proj = torch.nn.Identity()
        self.classification = False
        self.use_mcpr = False
        self.last_mcpr_outputs = None
        self.hfd_feature_capture_layers = ()
        self.last_hfd_features = None


def rejected(callable_):
    try:
        callable_()
    except (ValueError, FloatingPointError):
        return True
    return False


def main() -> int:
    torch.manual_seed(2032)
    batch, pred_len, full_len, hidden = 2, 3, 5, 4
    source = {layer: torch.randn(batch, full_len, hidden, requires_grad=True) for layer in (6, 7)}
    l5 = {
        layer: torch.nn.Parameter(source[layer].detach().clone() + 0.20)
        for layer in (6, 7)
    }
    l20 = {
        layer: torch.nn.Parameter(source[layer].detach().clone() - 0.10)
        for layer in (6, 7)
    }
    student = {
        layer: torch.nn.Parameter(source[layer].detach().clone() + 0.03)
        for layer in (6, 7)
    }
    beneficial = torch.tensor([[True, False, True], [False, True, False]])
    output_scale = torch.nn.Parameter(torch.tensor(0.2))
    diagnostics = Exp._hfd_pba_objective(
        student, source, l5, l20, beneficial, output_scale, pred_len,
    )
    diagnostics["hfd_loss"].backward()

    zero_student = {
        layer: torch.nn.Parameter(source[layer].detach().clone())
        for layer in (6, 7)
    }
    zero_diagnostics = Exp._hfd_pba_objective(
        zero_student, source, l5, l20,
        torch.zeros_like(beneficial), output_scale, pred_len,
    )
    zero_diagnostics["hfd_loss"].backward()

    capture = TinyCapture()
    value = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5, 1)
    state_keys_before = tuple(capture.state_dict())
    output_off = capture._forecast_from_context(value, mask)
    capture.set_hfd_feature_capture(True, [6, 7])
    output_on = capture._forecast_from_context(value, mask)
    captured = capture.get_last_hfd_features()
    capture.set_hfd_feature_capture(False)
    state_keys_after = tuple(capture.state_dict())

    bad_layers = {6: student[6]}
    wrong_shape = {layer: tensor.clone() for layer, tensor in student.items()}
    wrong_shape[7] = wrong_shape[7][:1]
    nonfinite = {layer: tensor.detach().clone() for layer, tensor in student.items()}
    nonfinite[6][0, 0, 0] = float("nan")

    checks = {
        "positive_finite_hidden_loss": (
            torch.isfinite(diagnostics["hfd_loss"])
            and float(diagnostics["hfd_loss"]) > 0.0
        ),
        "both_layers_have_positive_raw_loss": all(
            float(diagnostics["hfd_layer_raw"][layer]) > 0.0 for layer in (6, 7)
        ),
        "both_student_features_receive_gradient": all(
            student[layer].grad is not None
            and torch.isfinite(student[layer].grad).all()
            and float(student[layer].grad.abs().sum()) > 0.0
            for layer in (6, 7)
        ),
        "all_teacher_features_are_detached": (
            all(source[layer].grad is None for layer in (6, 7))
            and all(l5[layer].grad is None for layer in (6, 7))
            and all(l20[layer].grad is None for layer in (6, 7))
        ),
        "scale_is_detached": output_scale.grad is None,
        "no_benefit_returns_graph_zero": (
            float(zero_diagnostics["hfd_loss"]) == 0.0
            and all(
                zero_student[layer].grad is not None
                and float(zero_student[layer].grad.abs().sum()) == 0.0
                for layer in (6, 7)
            )
        ),
        "capture_default_off_preserves_output": torch.equal(output_off, output_on),
        "capture_contains_only_frozen_layers": set(captured) == {6, 7},
        "captured_student_features_require_grad": all(
            captured[layer].requires_grad for layer in (6, 7)
        ),
        "capture_adds_no_state_keys": state_keys_before == state_keys_after,
        "capture_disable_clears_ephemeral_state": (
            capture.hfd_feature_capture_layers == ()
            and capture.last_hfd_features is None
        ),
        "wrong_layer_set_rejected": rejected(lambda: Exp._hfd_pba_objective(
            bad_layers, source, l5, l20, beneficial, output_scale, pred_len,
        )),
        "wrong_shape_rejected": rejected(lambda: Exp._hfd_pba_objective(
            wrong_shape, source, l5, l20, beneficial, output_scale, pred_len,
        )),
        "nonfinite_feature_rejected": rejected(lambda: Exp._hfd_pba_objective(
            nonfinite, source, l5, l20, beneficial, output_scale, pred_len,
        )),
    }
    report = {
        "step": "22.32",
        "status": "PASS" if all(bool(value) for value in checks.values()) else "FAIL",
        "checks": {key: bool(value) for key, value in checks.items()},
    }
    output = ROOT / "reports/stage22_32_hfd_component_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
