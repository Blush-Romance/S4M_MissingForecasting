#!/usr/bin/env python3
"""CPU/static contracts for Step 22.26 BF-CMVD."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
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

from experiments.exp_pretrain1 import Exp_Long_Term_Forecast as Exp  # noqa: E402
from run_s4m_experiment import validate_experiment_config  # noqa: E402
import run_stage22_26_traffic_queue as queue  # noqa: E402


def rejected(config: dict) -> bool:
    try:
        validate_experiment_config(config)
    except (FileNotFoundError, TypeError, ValueError):
        return True
    return False


def main() -> int:
    config_paths = sorted((ROOT / "configs/stage22_26_traffic").glob("*.yaml"))
    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in config_paths]
    for config in configs:
        validate_experiment_config(config)

    formal = next(config for config in configs if not config["experiment"]["smoke_test"])
    smoke = next(config for config in configs if config["experiment"]["smoke_test"])
    negative_no_switch = copy.deepcopy(formal)
    negative_no_switch["official_args"]["use_bf_cmvd"] = False
    negative_wrong_objective = copy.deepcopy(formal)
    negative_wrong_objective["official_args"]["mcpr_objective"] = "mse_only"
    negative_wrong_rate = copy.deepcopy(formal)
    negative_wrong_rate["official_args"]["bf_cmvd_target_rate"] = 0.2
    negative_wrong_blocks = copy.deepcopy(formal)
    negative_wrong_blocks["official_args"]["bf_cmvd_block_lengths"] = [1, 20]
    negative_wrong_seed = copy.deepcopy(formal)
    negative_wrong_seed["official_args"]["bf_cmvd_nested_seed"] = 2028
    negative_wrong_weight = copy.deepcopy(formal)
    negative_wrong_weight["official_args"]["bf_cmvd_l20_weight"] = 0.30

    target = torch.zeros(4, 1, 2)
    student = torch.zeros_like(target, requires_grad=True)
    backbone = torch.tensor([
        [[2.0, 2.0]],
        [[1.0, 1.0]],
        [[0.0, 2.0]],
        [[2.0, 0.0]],
    ], requires_grad=True)
    teacher = torch.tensor([
        [[1.0, 1.0]],
        [[2.0, 2.0]],
        [[1.3, 1.3]],
        [[0.5, 0.5]],
    ], requires_grad=True)
    observed = torch.ones_like(target, dtype=torch.bool)
    diagnostics = Exp._bf_cmvd_objective(
        student, backbone, teacher, target, observed
    )
    diagnostics["total_loss"].backward()

    no_benefit_student = torch.ones(2, 1, 2, requires_grad=True)
    no_benefit = Exp._bf_cmvd_objective(
        no_benefit_student,
        torch.zeros(2, 1, 2),
        torch.ones(2, 1, 2),
        torch.zeros(2, 1, 2),
        torch.ones(2, 1, 2, dtype=torch.bool),
    )
    no_benefit["total_loss"].backward()

    token_student = torch.zeros(1, 3, 2, requires_grad=True)
    token_backbone = torch.tensor([[[2.0, 2.0], [0.0, 0.0], [4.0, 4.0]]])
    token_teacher = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]]])
    token_observed = torch.tensor([[[True, True], [True, True], [False, False]]])
    token_objective = Exp._bf_cmvd_objective(
        token_student,
        token_backbone,
        token_teacher,
        torch.zeros_like(token_student),
        token_observed,
        filter_granularity="token",
    )
    token_objective["total_loss"].backward()

    mask = torch.ones(3, 192, 7)
    mask[:, 20:30, :] = 0.0
    sample_indices = np.array([0, 17, 1001], dtype=np.int64)
    nested_l5_a = Exp._stage18_equal_count_observation_mask(
        mask, sample_indices, 0.15, 5, 2027
    )
    nested_l5_b = Exp._stage18_equal_count_observation_mask(
        mask, sample_indices, 0.15, 5, 2027
    )
    nested_l20 = Exp._stage18_equal_count_observation_mask(
        mask, sample_indices, 0.15, 20, 2027
    )
    original_observed_rows = mask[:, :, 0].sum(dim=1)
    expected_removed = torch.from_numpy(
        np.rint(0.15 * original_observed_rows.numpy()).astype(np.int64)
    )
    removed_l5 = (mask[:, :, 0] - nested_l5_a[:, :, 0]).sum(dim=1).to(torch.int64)
    removed_l20 = (mask[:, :, 0] - nested_l20[:, :, 0]).sum(dim=1).to(torch.int64)

    fair_keys = (
        "seed", "seq_len", "pred_len", "batch_size", "learning_rate",
        "optimizer", "patience", "lradj", "mcpr_rank", "mcpr_kernel_size",
        "mcpr_learning_rate", "mcpr_backbone_checkpoint",
        "post_train_evaluation",
    )
    shape_rejected = False
    try:
        Exp._bf_cmvd_objective(
            torch.zeros(1, 1, 1), torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 2), torch.zeros(1, 1, 1),
            torch.ones(1, 1, 1, dtype=torch.bool),
        )
    except ValueError:
        shape_rejected = True

    with tempfile.TemporaryDirectory(prefix="stage22_26_gate_") as temporary:
        original_append_event = queue.scheduler.append_event
        queue.scheduler.append_event = lambda *args, **kwargs: None
        temporary_root = Path(temporary)
        failed_run = temporary_root / "failed"
        failed_run.mkdir()
        (failed_run / "metrics.json").write_text(json.dumps({
            "bf_cmvd_enabled": True,
            "bf_cmvd_diagnostic_steps": 2,
            "bf_cmvd_beneficial_samples": 0,
            "bf_cmvd_total_samples": 32,
            "bf_cmvd_distill_l1_mean": 0.0,
            "bf_cmvd_teacher_delta_energy_mean": 0.01,
            "mcpr_gradient_norm_max": 0.2,
            "mcpr_residual_energy_mean": 0.01,
            "checkpoint_load_verified": True,
        }), encoding="utf-8")
        failed_state = {"jobs": {
            queue.SMOKE_JOB: {"state": "SUCCESS", "run_dir": str(failed_run)},
            queue.FORMAL_JOB: {
                "state": "PENDING", "depends_on": [queue.SMOKE_JOB],
            },
        }}
        queue.enforce_scientific_gate(failed_state)

        passed_run = temporary_root / "passed"
        passed_run.mkdir()
        (passed_run / "metrics.json").write_text(json.dumps({
            "bf_cmvd_enabled": True,
            "bf_cmvd_diagnostic_steps": 2,
            "bf_cmvd_beneficial_samples": 9,
            "bf_cmvd_total_samples": 32,
            "bf_cmvd_distill_l1_mean": 0.02,
            "bf_cmvd_teacher_delta_energy_mean": 0.01,
            "mcpr_gradient_norm_max": 0.2,
            "mcpr_residual_energy_mean": 0.01,
            "checkpoint_load_verified": True,
        }), encoding="utf-8")
        passed_state = {"jobs": {
            queue.SMOKE_JOB: {"state": "SUCCESS", "run_dir": str(passed_run)},
            queue.FORMAL_JOB: {
                "state": "PENDING", "depends_on": [queue.SMOKE_JOB],
            },
        }}
        queue.enforce_scientific_gate(passed_state)
        queue.scheduler.append_event = original_append_event

    checks = {
        "two_configs_validate": len(configs) == 2,
        "smoke_limits_are_isolated": (
            smoke["official_args"]["train_epochs"] == 2
            and smoke["official_args"]["max_train_batches"] == 2
            and smoke["official_args"]["max_eval_batches"] == 3
            and formal["official_args"]["train_epochs"] == 15
            and formal["official_args"]["max_train_batches"] == -1
            and formal["official_args"]["max_eval_batches"] == -1
        ),
        "formal_uses_validation_only": (
            formal["official_args"]["post_train_evaluation"] == "validation"
        ),
        "smoke_and_formal_core_protocol_equal": all(
            smoke["official_args"][key] == formal["official_args"][key]
            for key in fair_keys
        ),
        "benefit_filter_is_strict_bi_metric": torch.equal(
            diagnostics["beneficial_mask"], torch.tensor([True, False, False, True])
        ),
        "beneficial_fraction_exact": torch.isclose(
            diagnostics["beneficial_fraction"], torch.tensor(0.5)
        ).item(),
        "distillation_is_positive": float(diagnostics["distill_l1"].item()) > 0.0,
        "student_receives_finite_nonzero_gradient": (
            student.grad is not None
            and torch.isfinite(student.grad).all().item()
            and float(student.grad.abs().sum().item()) > 0.0
        ),
        "teacher_is_detached": teacher.grad is None,
        "backbone_is_detached": backbone.grad is None,
        "empty_benefit_set_has_zero_distillation": (
            float(no_benefit["distill_l1"].item()) == 0.0
        ),
        "empty_benefit_set_keeps_direct_gradient": (
            no_benefit_student.grad is not None
            and float(no_benefit_student.grad.abs().sum().item()) > 0.0
        ),
        "token_filter_recovers_local_benefit_hidden_by_sample_average": (
            torch.equal(
                token_objective["beneficial_mask"],
                torch.tensor([[True, False, False]]),
            )
            and torch.equal(
                token_objective["eligible_mask"],
                torch.tensor([[True, True, False]]),
            )
            and torch.isclose(
                token_objective["beneficial_fraction"], torch.tensor(0.5)
            ).item()
        ),
        "token_distillation_backpropagates": (
            token_student.grad is not None
            and torch.isfinite(token_student.grad).all().item()
            and float(token_student.grad.abs().sum().item()) > 0.0
        ),
        "equal_count_mask_is_deterministic": torch.equal(nested_l5_a, nested_l5_b),
        "l5_removes_exact_count": torch.equal(removed_l5, expected_removed),
        "l20_removes_exact_count": torch.equal(removed_l20, expected_removed),
        "different_scales_are_not_identical": not torch.equal(nested_l5_a, nested_l20),
        "shape_mismatch_is_rejected": shape_rejected,
        "missing_switch_is_rejected": rejected(negative_no_switch),
        "wrong_objective_is_rejected": rejected(negative_wrong_objective),
        "wrong_rate_is_rejected": rejected(negative_wrong_rate),
        "wrong_blocks_are_rejected": rejected(negative_wrong_blocks),
        "wrong_nested_seed_is_rejected": rejected(negative_wrong_seed),
        "wrong_weight_is_rejected": rejected(negative_wrong_weight),
        "failed_smoke_blocks_formal_before_gpu": (
            failed_state["jobs"][queue.FORMAL_JOB]["state"] == "BLOCKED_DEPENDENCY"
            and failed_state["jobs"][queue.FORMAL_JOB]
            ["scientific_dependency"]["passed"] is False
        ),
        "passed_smoke_leaves_formal_pending": (
            passed_state["jobs"][queue.FORMAL_JOB]["state"] == "PENDING"
            and passed_state["jobs"][queue.FORMAL_JOB]
            ["scientific_dependency"]["passed"] is True
        ),
        "queue_is_smoke_then_formal": (
            len(queue.scheduler.JOB_DEFINITIONS) == 2
            and queue.scheduler.JOB_DEFINITIONS[1]["depends_on"] == [queue.SMOKE_JOB]
        ),
    }
    report = {
        "step": "22.26",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    output = ROOT / "reports/stage22_26_bf_cmvd_component_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
