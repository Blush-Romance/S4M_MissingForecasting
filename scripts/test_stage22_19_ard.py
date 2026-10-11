#!/usr/bin/env python3
"""CPU contracts for Step 22.19 ARD-MCPR."""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path[:0] = [
    str(ROOT / "baselines/S4M/s4m"),
    str(ROOT / "baselines/S4M/s4m/model"),
    str(ROOT / "baselines/S4M/s4m/model/S4/s4"),
    str(ROOT / "baselines/S4M/s4m/model/S4/s4/extensions/kernels"),
]

from run_s4m_experiment import validate_experiment_config  # noqa: E402
from run_stage12_validation_cache import validate_config as validate_cache_config  # noqa: E402
from model.ARD import AnalyticResidualDecisionGate  # noqa: E402
from model.MCPR import MaskConditionedParetoResidual  # noqa: E402
from experiments.exp_pretrain1 import Exp_Long_Term_Forecast as Exp  # noqa: E402
import run_stage22_19_ard_queue as queue  # noqa: E402


def expect_rejected(config: dict) -> bool:
    try:
        validate_experiment_config(config)
    except (FileNotFoundError, TypeError, ValueError):
        return True
    return False


def ard_config_fixture() -> dict:
    source = (
        ROOT
        / "configs/stage22_ttrr/mcpr_mse_electricity_seed2024_smoke.yaml"
    )
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["experiment"]["model"] = "ard_mcpr_component_fixture"
    config["official_args"].update({
        "use_ard": True,
        "use_ttrr": False,
        "mcpr_objective": "analytic_decision_distill",
        "ard_hidden_dim": 16,
        "ard_distill_warmup_epochs": 1,
        "ard_energy_epsilon": 1e-8,
        "ard_action_epsilon": 1e-6,
        "ard_action_weight": 1.0,
        "ard_scale_weight": 1.0,
    })
    return config


def synthetic_objective(distill_enabled: bool) -> tuple[dict, dict]:
    # target=0. For raw residual r=1, alpha*=-e/r. These samples yield
    # OFF=0, FULL=1, PARTIAL=0.25; the final zero-energy sample is excluded.
    backbone = torch.tensor([1.0, -2.0, -0.25, 0.5]).view(4, 1, 1)
    raw = torch.tensor([1.0, 1.0, 1.0, 0.0]).view(4, 1, 1)
    raw.requires_grad_()
    student_gate = torch.full((4, 1, 1), 0.5, requires_grad=True)
    applied = student_gate * raw
    deployed = backbone + applied
    logits = torch.zeros(4, 3, requires_grad=True)
    target = torch.zeros_like(backbone)
    observed = torch.ones_like(backbone, dtype=torch.bool)
    diagnostics = Exp._ard_analytic_distillation_objective(
        raw_residual=raw,
        applied_residual=applied,
        deployed_prediction=deployed,
        action_logits=logits,
        student_gate=student_gate,
        target=target,
        target_mask=observed,
        distill_enabled=distill_enabled,
        energy_epsilon=1e-8,
        action_epsilon=1e-6,
        action_weight=1.0,
        scale_weight=1.0,
    )
    diagnostics["total_loss"].backward()
    tensors = {
        "raw": raw,
        "student_gate": student_gate,
        "logits": logits,
    }
    return diagnostics, tensors


def main() -> int:
    torch.manual_seed(2219)
    disabled = MaskConditionedParetoResidual(
        hidden_size=8,
        num_variables=5,
        rank=4,
        kernel_size=3,
        use_ard=False,
    )
    torch.manual_seed(2219)
    historical = MaskConditionedParetoResidual(
        hidden_size=8,
        num_variables=5,
        rank=4,
        kernel_size=3,
    )
    disabled_exact = (
        set(disabled.state_dict()) == set(historical.state_dict())
        and all(
            torch.equal(disabled.state_dict()[key], historical.state_dict()[key])
            for key in disabled.state_dict()
        )
    )

    enabled = MaskConditionedParetoResidual(
        hidden_size=8,
        num_variables=5,
        rank=4,
        kernel_size=3,
        use_ard=True,
        ard_hidden_dim=16,
    )
    hidden = torch.randn(6, 12, 8, requires_grad=True)
    mask = torch.ones(6, 12, 5)
    mask[1:, 3:8, :] = 0.0
    backbone = torch.randn(6, 12, 5, requires_grad=True)
    initial = enabled(hidden, mask, backbone)
    initial_probabilities = torch.full_like(
        initial["ard_action_probabilities"], 1.0 / 3.0
    )
    initial_partial = torch.full_like(initial["ard_partial_alpha"], 0.5)
    initial_gate = torch.full_like(initial["ard_gate"], 0.5)

    with torch.no_grad():
        enabled.output_projection.weight.fill_(0.05)
        enabled.output_projection.bias.fill_(0.01)
    activated = enabled(hidden, mask, backbone)
    actual_target = torch.randn_like(activated["residual"])
    actual_observed = torch.ones_like(actual_target, dtype=torch.bool)
    actual_diagnostics = Exp._ard_analytic_distillation_objective(
        raw_residual=activated["raw_residual"],
        applied_residual=activated["residual"],
        deployed_prediction=backbone + activated["residual"],
        action_logits=activated["ard_action_logits"],
        student_gate=activated["ard_gate"],
        target=actual_target,
        target_mask=actual_observed,
        distill_enabled=True,
        energy_epsilon=1e-8,
        action_epsilon=1e-6,
        action_weight=1.0,
        scale_weight=1.0,
    )
    actual_diagnostics["total_loss"].backward()

    distilled, distilled_tensors = synthetic_objective(True)
    warmup, warmup_tensors = synthetic_objective(False)
    expected_teacher = torch.tensor([0.0, 1.0, 0.25])
    expected_actions = torch.tensor([0, 2, 1])

    buffer = io.BytesIO()
    torch.save(enabled.state_dict(), buffer)
    buffer.seek(0)
    reloaded = MaskConditionedParetoResidual(
        hidden_size=8,
        num_variables=5,
        rank=4,
        kernel_size=3,
        use_ard=True,
        ard_hidden_dim=16,
    )
    reloaded.load_state_dict(torch.load(buffer, map_location="cpu"), strict=True)
    checkpoint_exact = all(
        torch.equal(enabled.state_dict()[key], reloaded.state_dict()[key])
        for key in enabled.state_dict()
    )

    simultaneous_rejected = False
    try:
        MaskConditionedParetoResidual(
            hidden_size=8,
            num_variables=5,
            rank=4,
            kernel_size=3,
            use_ttrr=True,
            use_ard=True,
        )
    except ValueError:
        simultaneous_rejected = True

    valid_config = ard_config_fixture()
    validate_experiment_config(valid_config)

    training_paths = sorted(
        path for path in (ROOT / "configs/stage22_ard").glob("*.yaml")
        if path.name.startswith(("mcpr_mse_", "ard_mcpr_"))
    )
    training_configs = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in training_paths
    ]
    for config in training_configs:
        validate_experiment_config(config)
    cache_paths = sorted((ROOT / "configs/stage22_ard").glob("pg95_*.yaml"))
    cache_configs = []
    for path in cache_paths:
        cache_config = yaml.safe_load(path.read_text(encoding="utf-8"))
        source_path = ROOT / cache_config["source"]["resolved_config"]
        source_config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        validate_cache_config(cache_config, source_config)
        cache_configs.append(cache_config)
    controls = [
        config for config in training_configs
        if not config["official_args"]["use_ard"]
    ]
    candidates = [
        config for config in training_configs
        if config["official_args"]["use_ard"]
    ]
    formal_control = next(
        config for config in controls if not config["experiment"]["smoke_test"]
    )
    formal_candidate = next(
        config for config in candidates if not config["experiment"]["smoke_test"]
    )
    allowed_official_differences = {
        "model_id", "des", "use_ard", "mcpr_objective"
    }
    formal_unexpected_differences = sorted(
        key
        for key in set(formal_control["official_args"]) | set(formal_candidate["official_args"])
        if formal_control["official_args"].get(key)
        != formal_candidate["official_args"].get(key)
        and key not in allowed_official_differences
    )
    negative_without_mcpr = copy.deepcopy(valid_config)
    negative_without_mcpr["official_args"]["use_mcpr"] = False
    negative_wrong_objective = copy.deepcopy(valid_config)
    negative_wrong_objective["official_args"]["mcpr_objective"] = "mse_only"
    negative_without_switch = copy.deepcopy(valid_config)
    negative_without_switch["official_args"]["use_ard"] = False
    negative_with_ttrr = copy.deepcopy(valid_config)
    negative_with_ttrr["official_args"]["use_ttrr"] = True
    negative_hidden = copy.deepcopy(valid_config)
    negative_hidden["official_args"]["ard_hidden_dim"] = 32
    negative_missing_field = copy.deepcopy(valid_config)
    del negative_missing_field["official_args"]["ard_scale_weight"]
    jobs = {job["id"]: job for job in queue.scheduler.JOB_DEFINITIONS}

    action_gradient = float(
        enabled.ard_gate.action_head.weight.grad.detach().abs().sum().item()
    )
    partial_gradient = float(
        enabled.ard_gate.partial_head.weight.grad.detach().abs().sum().item()
    )
    residual_gradient = float(
        enabled.output_projection.weight.grad.detach().abs().sum().item()
    )
    checks = {
        "disabled_state_dict_matches_historical_default": disabled_exact,
        "disabled_has_no_ard_parameters": not any(
            "ard_gate" in key for key in disabled.state_dict()
        ),
        "feature_schema_is_eight_dimensional": (
            AnalyticResidualDecisionGate().feature_dim == 8
            and activated["ard_features"].shape == (6, 8)
        ),
        "initial_action_probabilities_are_uniform": torch.equal(
            initial["ard_action_probabilities"], initial_probabilities
        ),
        "initial_partial_alpha_is_half": torch.equal(
            initial["ard_partial_alpha"], initial_partial
        ),
        "initial_gate_is_exactly_half": torch.equal(
            initial["ard_gate"], initial_gate
        ),
        "zero_residual_preserves_backbone": torch.equal(
            initial["residual"], torch.zeros_like(initial["residual"])
        ),
        "features_are_gradient_detached": not activated["ard_features"].requires_grad,
        "gate_is_closed_interval_bounded": bool(
            ((activated["ard_gate"] >= 0.0) & (activated["ard_gate"] <= 1.0))
            .all().item()
        ),
        "analytic_teacher_is_exact": torch.allclose(
            distilled["teacher_alpha"], expected_teacher, atol=0.0, rtol=0.0
        ),
        "analytic_actions_are_exact": torch.equal(
            distilled["teacher_actions"], expected_actions
        ),
        "zero_energy_sample_is_excluded": distilled["valid_sample_count"] == 3,
        "distillation_losses_are_finite_positive": all(
            bool(torch.isfinite(distilled[key]).item())
            and float(distilled[key].item()) > 0.0
            for key in ("action_loss", "scale_loss", "total_loss")
        ),
        "synthetic_raw_mse_backpropagates": (
            distilled_tensors["raw"].grad is not None
            and bool(torch.isfinite(distilled_tensors["raw"].grad).all().item())
        ),
        "synthetic_action_head_backpropagates": (
            distilled_tensors["logits"].grad is not None
            and float(distilled_tensors["logits"].grad.abs().sum().item()) > 0.0
        ),
        "synthetic_scale_head_backpropagates": (
            distilled_tensors["student_gate"].grad is not None
            and float(distilled_tensors["student_gate"].grad.abs().sum().item()) > 0.0
        ),
        "warmup_uses_raw_mse_only": (
            warmup["valid_sample_count"] == 0
            and torch.equal(warmup["total_loss"], warmup["raw_mse"])
            and float(warmup["action_loss"].item()) == 0.0
            and float(warmup["scale_loss"].item()) == 0.0
        ),
        "warmup_does_not_update_gate": (
            warmup_tensors["logits"].grad is None
            and (
                warmup_tensors["student_gate"].grad is None
                or torch.equal(
                    warmup_tensors["student_gate"].grad,
                    torch.zeros_like(warmup_tensors["student_gate"]),
                )
            )
        ),
        "warmup_still_updates_raw_residual": (
            warmup_tensors["raw"].grad is not None
            and float(warmup_tensors["raw"].grad.abs().sum().item()) > 0.0
        ),
        "actual_action_head_receives_gradient": action_gradient > 0.0,
        "actual_partial_head_receives_gradient": partial_gradient > 0.0,
        "actual_raw_residual_receives_gradient": residual_gradient > 0.0,
        "checkpoint_strict_roundtrip_is_exact": checkpoint_exact,
        "module_rejects_ttrr_ard_combination": simultaneous_rejected,
        "valid_frozen_config_is_accepted": True,
        "four_training_configs_validate": len(training_configs) == 4,
        "two_pg95_cache_configs_validate": len(cache_configs) == 2,
        "two_controls_and_two_candidates": (
            len(controls) == 2 and len(candidates) == 2
        ),
        "all_configs_use_seed2025_asset": (
            len({json.dumps(config["data"], sort_keys=True) for config in training_configs})
            == 1
            and all(config["experiment"]["seed"] == 2025 for config in training_configs)
            and all(config["experiment"]["seed"] == 2025 for config in cache_configs)
        ),
        "formal_control_candidate_budget_is_identical": not formal_unexpected_differences,
        "formal_runtime_caps_are_disabled": all(
            config["runtime"]["max_train_batches"] == -1
            and config["runtime"]["max_eval_batches"] == -1
            and config["official_args"]["train_epochs"] == 15
            for config in (formal_control, formal_candidate)
        ),
        "smoke_covers_warmup_and_distillation": all(
            config["official_args"]["train_epochs"] == 2
            and config["official_args"]["max_train_batches"] == 3
            and config["official_args"]["max_eval_batches"] == 3
            for config in training_configs
            if config["experiment"]["smoke_test"]
        ),
        "pg95_cache_uses_fixed_seed2025_probe": all(
            config["diagnostic"]["nested_seed"] == 2025
            and config["diagnostic"]["stage_identity"] == "stage22_crossmask"
            for config in cache_configs
        ),
        "queue_has_six_jobs": len(jobs) == 6,
        "all_formals_depend_on_matching_smoke": (
            jobs["pg95_formal"]["depends_on"] == ["pg95_smoke"]
            and jobs["mcpr_mse_formal"]["depends_on"] == ["mcpr_mse_smoke"]
            and jobs["ard_formal"]["depends_on"] == ["ard_smoke"]
        ),
        "pg95_jobs_use_read_only_cache_runner": all(
            Path(jobs[name]["runner"]).name == "run_stage12_validation_cache.py"
            for name in ("pg95_smoke", "pg95_formal")
        ),
        "training_jobs_use_default_runner": all(
            "runner" not in jobs[name]
            for name in (
                "mcpr_mse_smoke", "mcpr_mse_formal", "ard_smoke", "ard_formal"
            )
        ),
        "queue_runtime_is_isolated": (
            queue.scheduler.RUNTIME_ROOT == ROOT / "runtime/stage22_19_ard"
        ),
        "rejects_ard_without_mcpr": expect_rejected(negative_without_mcpr),
        "rejects_wrong_objective": expect_rejected(negative_wrong_objective),
        "rejects_objective_without_ard_switch": expect_rejected(
            negative_without_switch
        ),
        "rejects_ttrr_ard_combination": expect_rejected(negative_with_ttrr),
        "rejects_unfrozen_hidden_width": expect_rejected(negative_hidden),
        "rejects_missing_ard_field": expect_rejected(negative_missing_field),
    }
    report = {
        "step": "22.19.3",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "ard_parameter_count": sum(p.numel() for p in enabled.ard_gate.parameters()),
        "gradients_l1": {
            "action_head": action_gradient,
            "partial_head": partial_gradient,
            "raw_residual_output": residual_gradient,
        },
        "synthetic_teacher_alpha": distilled["teacher_alpha"].detach().tolist(),
        "synthetic_teacher_actions": distilled["teacher_actions"].detach().tolist(),
        "formal_unexpected_official_arg_differences": formal_unexpected_differences,
    }
    path = ROOT / "reports/stage22_19_ard_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
