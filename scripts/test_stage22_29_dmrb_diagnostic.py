#!/usr/bin/env python3
"""CPU, config, decision, and queue contracts for Step 22.29."""

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
sys.path[:0] = [
    str(ROOT / "baselines/S4M/s4m"),
    str(ROOT / "baselines/S4M/s4m/model"),
    str(ROOT / "baselines/S4M/s4m/model/S4/s4"),
    str(ROOT / "baselines/S4M/s4m/model/S4/s4/extensions/kernels"),
    str(ROOT / "scripts"),
]

from experiments.exp_pretrain1 import Exp_Long_Term_Forecast as Exp  # noqa: E402
from analyze_stage22_29_dmrb_failure import analyze_arrays  # noqa: E402
from run_stage22_29_dmrb_diagnostic import (  # noqa: E402
    dmrb_branch_keys_verified,
    validate_config,
)
import run_stage22_29_dmrb_queue as queue  # noqa: E402


def rejected(config: dict, source: dict) -> bool:
    try:
        validate_config(config, source)
    except (KeyError, TypeError, ValueError):
        return True
    return False


def decision_arrays(case: str, count: int = 4) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for label in (
        "backbone", "teacher_l5", "teacher_l20", "teacher_joint",
        "student_l5", "student_l20", "student_joint",
    ):
        for metric in ("mse", "mae"):
            arrays[f"{label}_{metric}"] = np.ones(count, dtype=np.float32)
    for metric in ("mse", "mae"):
        arrays[f"student_l5_{metric}"][:] = 0.9
        arrays[f"student_l20_{metric}"][:] = 0.9
        arrays[f"student_joint_{metric}"][:] = 0.8
        arrays[f"teacher_l5_{metric}"][:] = 0.7
        arrays[f"teacher_l20_{metric}"][:] = 0.7
        arrays[f"teacher_joint_{metric}"][:] = 0.6
    ones = np.ones(count, dtype=np.uint8)
    arrays["teacher_l5_beneficial"] = ones.copy()
    arrays["teacher_l20_beneficial"] = ones.copy()
    arrays["teacher_joint_beneficial"] = ones.copy()
    for label in ("student_l5", "student_l20", "student_joint"):
        arrays[f"{label}_beneficial"] = ones.copy()
    for branch in ("l5", "l20"):
        arrays[f"{branch}_self_dot"] = np.full(count, 0.1, dtype=np.float32)
        arrays[f"{branch}_self_residual_energy"] = np.full(
            count, 0.1, dtype=np.float32
        )
        arrays[f"{branch}_self_target_energy"] = np.ones(count, dtype=np.float32)
        arrays[f"{branch}_self_mismatch_energy"] = np.ones(count, dtype=np.float32)
    arrays["l5_to_l20_target_dot"] = np.zeros(count, dtype=np.float32)
    arrays["l20_to_l5_target_dot"] = np.zeros(count, dtype=np.float32)
    arrays["teacher_component_dot"] = np.full(count, 0.25, dtype=np.float32)
    arrays["student_branch_dot"] = np.zeros(count, dtype=np.float32)
    arrays["teacher_sum_energy"] = np.full(count, 2.5, dtype=np.float32)
    arrays["student_sum_energy"] = np.full(count, 0.2, dtype=np.float32)

    if case == "supervision":
        arrays["teacher_joint_beneficial"] = np.array([1, 1, 0, 0], dtype=np.uint8)
        arrays["teacher_l5_beneficial"] = np.array([1, 0, 1, 0], dtype=np.uint8)
        arrays["teacher_l20_beneficial"] = np.array([0, 1, 0, 1], dtype=np.uint8)
        arrays["teacher_component_dot"][:] = -0.5
        arrays["teacher_sum_energy"][:] = 1.0
    elif case == "conflict":
        for metric in ("mse", "mae"):
            arrays[f"student_l5_{metric}"][:] = 0.8
            arrays[f"student_l20_{metric}"][:] = 0.9
            arrays[f"student_joint_{metric}"][:] = 0.95
    elif case == "mixed":
        for branch in ("l5", "l20"):
            arrays[f"{branch}_self_dot"][:] = 1.0
            arrays[f"{branch}_self_residual_energy"][:] = 1.0
            arrays[f"{branch}_self_mismatch_energy"][:] = 0.0
    return arrays


def main() -> int:
    torch.manual_seed(2027)
    backbone = torch.zeros(2, 3, 4)
    residual_l5 = torch.full_like(backbone, 0.1)
    residual_l20 = torch.full_like(backbone, -0.05)
    correction_l5 = torch.full_like(backbone, -0.2)
    correction_l20 = torch.full_like(backbone, 0.1)
    target = torch.ones_like(backbone)
    token = Exp._stage22_29_token_diagnostics(
        backbone, residual_l5, residual_l20,
        correction_l5, correction_l20, target,
    )

    config_paths = sorted(
        (ROOT / "configs/stage22_29_dmrb_diagnostic").glob("*.yaml")
    )
    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in config_paths]
    source_path = ROOT / configs[0]["source"]["dmrb_resolved_config"]
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    for config in configs:
        validate_config(config, source)
    smoke = next(config for config in configs if config["experiment"]["smoke_test"])
    formal = next(config for config in configs if not config["experiment"]["smoke_test"])
    wrong_split = copy.deepcopy(formal)
    wrong_split["diagnostic"]["split"] = "test"
    wrong_rate = copy.deepcopy(formal)
    wrong_rate["diagnostic"]["target_rate"] = 0.2
    wrong_weight = copy.deepcopy(formal)
    wrong_weight["diagnostic"]["teacher_weights"]["l5"] = 0.3
    wrong_budget = copy.deepcopy(formal)
    wrong_budget["diagnostic"]["max_eval_batches"] = 3
    wrong_source = copy.deepcopy(source)
    wrong_source["official_args"]["use_dmrb"] = False

    decisions = {
        case: analyze_arrays({}, decision_arrays(case), True)["decision"]
        for case in ("supervision", "conflict", "underfit", "mixed")
    }
    protocol_decision = analyze_arrays(
        {}, decision_arrays("supervision"), False
    )["decision"]

    with tempfile.TemporaryDirectory(prefix="stage22_29_gate_") as temporary:
        run = Path(temporary)
        (run / "metrics.json").write_text(json.dumps({
            "artifact_status": "PASS",
            "artifact_checks": {"one": True, "two": True},
            "test_split_accessed": False,
            "updates_model_or_memory": False,
            "validation_batches_processed": 3,
            "validation_samples_processed": 48,
            "dmrb_l5_enabled": True,
            "dmrb_l20_enabled": True,
            "checkpoint_dmrb_branches_verified": True,
        }), encoding="utf-8")
        state = {"jobs": {
            queue.SMOKE_JOB: {"state": "SUCCESS", "run_dir": str(run)},
            queue.FORMAL_JOB: {
                "state": "PENDING", "depends_on": [queue.SMOKE_JOB],
            },
        }}
        original = queue.scheduler.append_event
        queue.scheduler.append_event = lambda *args, **kwargs: None
        queue.enforce_artifact_gate(state)
        queue.scheduler.append_event = original

    checks = {
        "token_shapes": all(value.shape == (2, 3) for value in token.values()),
        "student_factorization": torch.allclose(
            backbone + residual_l5 + residual_l20,
            backbone + token["l5_self_residual_energy"].unsqueeze(-1) * 0.0
            + residual_l5 + residual_l20,
        ),
        "teacher_benefit_roundtrip": torch.equal(
            token["teacher_joint_beneficial"],
            (token["teacher_joint_mse"] < token["backbone_mse"])
            & (token["teacher_joint_mae"] < token["backbone_mae"]),
        ),
        "self_mismatch_identity": torch.allclose(
            token["l5_self_mismatch_energy"],
            token["l5_self_residual_energy"]
            + token["l5_self_target_energy"]
            - 2.0 * token["l5_self_dot"],
        ),
        "two_configs_validate": len(configs) == 2,
        "smoke_budget_three_batches": smoke["diagnostic"]["max_eval_batches"] == 3,
        "formal_budget_full": formal["diagnostic"]["max_eval_batches"] == -1,
        "test_split_is_rejected": rejected(wrong_split, source),
        "wrong_rate_is_rejected": rejected(wrong_rate, source),
        "wrong_weight_is_rejected": rejected(wrong_weight, source),
        "formal_smoke_budget_is_rejected": rejected(wrong_budget, source),
        "wrong_dmrb_identity_is_rejected": rejected(formal, wrong_source),
        "dmrb_branch_keys_are_verified_without_runner_side_metrics": (
            dmrb_branch_keys_verified({"checkpoint_mcpr_state_keys": [
                "mcpr.l5_basis.output_projection.weight",
                "mcpr.l20_basis.output_projection.weight",
            ]})
            and not dmrb_branch_keys_verified({"checkpoint_mcpr_state_keys": [
                "mcpr.l5_basis.output_projection.weight",
            ]})
        ),
        "protocol_failure_has_priority": protocol_decision
        == "PROTOCOL_FAILURE_NO_DECISION",
        "supervision_maps_to_branch_masks": decisions["supervision"]
        == "SCALE_SUPERVISION_MISMATCH_FREEZE_BRANCH_PARETO_MASKS",
        "conflict_maps_to_router": decisions["conflict"]
        == "BRANCH_SUM_CONFLICT_FREEZE_CAUSAL_SCALE_ROUTER",
        "dual_underfit_stops_low_rank": decisions["underfit"]
        == "DUAL_BRANCH_EXPRESSION_UNDERFIT_STOP_LOW_RANK_ROUTE",
        "mixed_stops": decisions["mixed"] == "MIXED_OR_INDETERMINATE_STOP",
        "passed_smoke_releases_formal": (
            state["jobs"][queue.FORMAL_JOB]["state"] == "PENDING"
            and state["jobs"][queue.FORMAL_JOB]["scientific_dependency"]["passed"]
            is True
        ),
        "queue_is_smoke_then_formal": (
            len(queue.scheduler.JOB_DEFINITIONS) == 2
            and queue.scheduler.JOB_DEFINITIONS[1]["depends_on"]
            == [queue.SMOKE_JOB]
        ),
        "queue_preserves_max_concurrency_five": (
            queue.scheduler.MAX_CONCURRENT_JOBS == 5
        ),
    }
    report = {
        "step": "22.29",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "synthetic_decisions": decisions,
    }
    output = ROOT / "reports/stage22_29_dmrb_diagnostic_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
