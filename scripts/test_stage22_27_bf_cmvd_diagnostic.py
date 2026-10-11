#!/usr/bin/env python3
"""CPU/static contracts for Step 22.27 BF-CMVD failure attribution."""

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
from analyze_stage22_27_bf_cmvd_failure import decide_diagnosis  # noqa: E402
import run_stage22_27_bf_cmvd_diagnostic as runner  # noqa: E402
import run_stage22_27_bf_cmvd_queue as queue  # noqa: E402


def rejected(config: dict, source: dict) -> bool:
    try:
        runner.validate_config(config, source)
    except (KeyError, TypeError, ValueError):
        return True
    return False


def decision_inputs() -> tuple[dict, dict, dict, dict, dict]:
    coverage = {
        "complete_benefit_observed_support_recall": 0.8,
        "benefit_rate_observed_unobserved_abs_gap": 0.01,
    }
    region = {
        "student_minus_backbone_mean": -0.01,
        "token_count": 10,
    }
    gap_mse = {
        "beneficial_positive_gap_share": 0.7,
        "nonbeneficial_positive_gap_share": 0.3,
        "nonbeneficial_region": copy.deepcopy(region),
    }
    gap_mae = copy.deepcopy(gap_mse)
    beneficial = {
        "cosine": 0.4,
        "projection_coefficient": 0.4,
        "normalized_mismatch": 0.9,
    }
    nonbeneficial = {"student_energy": 1.0}
    return coverage, gap_mse, gap_mae, beneficial, nonbeneficial


def main() -> int:
    config_paths = sorted(
        (ROOT / "configs/stage22_27_bf_cmvd_diagnostic").glob("*.yaml")
    )
    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in config_paths]
    source_path = runner.project_path(
        configs[0]["source"]["student_resolved_config"]
    )
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    for config in configs:
        runner.validate_config(config, source)
    formal = next(config for config in configs if not config["experiment"]["smoke_test"])
    smoke = next(config for config in configs if config["experiment"]["smoke_test"])

    negative_split = copy.deepcopy(formal)
    negative_split["diagnostic"]["split"] = "test"
    negative_rate = copy.deepcopy(formal)
    negative_rate["diagnostic"]["target_rate"] = 0.2
    negative_blocks = copy.deepcopy(formal)
    negative_blocks["diagnostic"]["block_lengths"] = [1, 20]
    negative_seed = copy.deepcopy(formal)
    negative_seed["diagnostic"]["nested_seed"] = 2028
    negative_weight = copy.deepcopy(formal)
    negative_weight["diagnostic"]["teacher_weights"]["l20"] = 0.3
    negative_formal_budget = copy.deepcopy(formal)
    negative_formal_budget["diagnostic"]["max_eval_batches"] = 3
    negative_identity = copy.deepcopy(source)
    negative_identity["official_args"]["mcpr_objective"] = "mse_only"

    target = torch.zeros(1, 4, 2)
    backbone = torch.tensor([[
        [2.0, 2.0], [0.0, 2.0], [1.0, 1.0], [2.0, 2.0],
    ]])
    teacher = torch.tensor([[
        [1.0, 1.0], [1.3, 1.3], [2.0, 2.0], [0.5, 0.5],
    ]])
    student = torch.tensor([[
        [1.5, 1.5], [0.2, 1.8], [1.2, 1.2], [1.0, 1.0],
    ]])
    observed_mask = torch.tensor([[
        [True, True], [True, True], [True, True], [False, False],
    ]])
    diagnostics = Exp._stage22_27_token_diagnostics(
        backbone, teacher, student, target, target, observed_mask
    )
    shape_rejected = False
    try:
        Exp._stage22_27_token_diagnostics(
            torch.zeros(1, 1, 1), torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1), torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 2), torch.ones(1, 1, 1),
        )
    except ValueError:
        shape_rejected = True

    coverage, gap_mse, gap_mae, beneficial, nonbeneficial = decision_inputs()
    expression_decision = decide_diagnosis(
        coverage, gap_mse, gap_mae, beneficial, nonbeneficial, 0.2, True
    )
    coverage_case = copy.deepcopy(coverage)
    coverage_case.update({
        "complete_benefit_observed_support_recall": 0.6,
        "benefit_rate_observed_unobserved_abs_gap": 0.06,
    })
    coverage_decision = decide_diagnosis(
        coverage_case, gap_mse, gap_mae, beneficial, nonbeneficial, 0.2, True
    )
    spill_gap_mse = copy.deepcopy(gap_mse)
    spill_gap_mae = copy.deepcopy(gap_mae)
    for gap in (spill_gap_mse, spill_gap_mae):
        gap["beneficial_positive_gap_share"] = 0.25
        gap["nonbeneficial_positive_gap_share"] = 0.75
        gap["nonbeneficial_region"]["student_minus_backbone_mean"] = 0.01
    spill_decision = decide_diagnosis(
        coverage, spill_gap_mse, spill_gap_mae, beneficial, nonbeneficial, 0.4, True
    )
    mixed_gap_mse = copy.deepcopy(gap_mse)
    mixed_gap_mae = copy.deepcopy(gap_mae)
    for gap in (mixed_gap_mse, mixed_gap_mae):
        gap["beneficial_positive_gap_share"] = 0.55
        gap["nonbeneficial_positive_gap_share"] = 0.45
    mixed_decision = decide_diagnosis(
        coverage, mixed_gap_mse, mixed_gap_mae, beneficial, nonbeneficial, 0.2, True
    )

    with tempfile.TemporaryDirectory(prefix="stage22_27_gate_") as temporary:
        original_append_event = queue.scheduler.append_event
        queue.scheduler.append_event = lambda *args, **kwargs: None
        temporary_root = Path(temporary)
        failed_run = temporary_root / "failed"
        failed_run.mkdir()
        (failed_run / "metrics.json").write_text(json.dumps({
            "artifact_status": "FAIL",
            "artifact_checks": {"validation_only": False},
            "test_split_accessed": False,
            "updates_model_or_memory": False,
            "validation_batches_processed": 3,
            "validation_samples_processed": 48,
        }), encoding="utf-8")
        failed_state = {"jobs": {
            queue.SMOKE_JOB: {"state": "SUCCESS", "run_dir": str(failed_run)},
            queue.FORMAL_JOB: {"state": "PENDING", "depends_on": [queue.SMOKE_JOB]},
        }}
        queue.enforce_artifact_gate(failed_state)

        passed_run = temporary_root / "passed"
        passed_run.mkdir()
        (passed_run / "metrics.json").write_text(json.dumps({
            "artifact_status": "PASS",
            "artifact_checks": {"validation_only": True, "fixed_masks": True},
            "test_split_accessed": False,
            "updates_model_or_memory": False,
            "validation_batches_processed": 3,
            "validation_samples_processed": 48,
        }), encoding="utf-8")
        passed_state = {"jobs": {
            queue.SMOKE_JOB: {"state": "SUCCESS", "run_dir": str(passed_run)},
            queue.FORMAL_JOB: {"state": "PENDING", "depends_on": [queue.SMOKE_JOB]},
        }}
        queue.enforce_artifact_gate(passed_state)
        queue.scheduler.append_event = original_append_event

    run_source = (ROOT / "baselines/S4M/s4m/run.py").read_text(encoding="utf-8")
    exp_source = (
        ROOT / "baselines/S4M/s4m/experiments/exp_pretrain1.py"
    ).read_text(encoding="utf-8")
    checks = {
        "two_configs_validate": len(configs) == 2,
        "smoke_and_formal_budgets_are_isolated": (
            smoke["diagnostic"]["max_eval_batches"] == 3
            and formal["diagnostic"]["max_eval_batches"] == -1
        ),
        "validation_only_is_frozen": formal["diagnostic"]["split"] == "val",
        "teacher_protocol_equal": all(
            smoke["diagnostic"][key] == formal["diagnostic"][key]
            for key in ("target_rate", "block_lengths", "nested_seed", "teacher_weights")
        ),
        "complete_benefit_is_strict_bimetric": torch.equal(
            diagnostics["complete_teacher_beneficial"],
            torch.tensor([[True, False, False, True]]),
        ),
        "observed_benefit_is_supported_subset": torch.equal(
            diagnostics["observed_teacher_beneficial"],
            torch.tensor([[True, False, False, False]]),
        ),
        "student_benefit_is_computed": diagnostics[
            "complete_student_beneficial"
        ].shape == (1, 4),
        "quadratic_mismatch_identity": torch.allclose(
            diagnostics["correction_mismatch_energy"],
            diagnostics["student_correction_energy"]
            + diagnostics["teacher_correction_energy"]
            - 2.0 * diagnostics["correction_dot"],
            atol=1e-6,
        ),
        "float32_residual_identity_tolerance_covers_two_ulp": bool(
            isinstance(runner.FLOAT32_RESIDUAL_IDENTITY_ATOL, float)
            and
            runner.FLOAT32_RESIDUAL_IDENTITY_ATOL
            >= 2.0 * np.finfo(np.float32).eps
            and runner.FLOAT32_RESIDUAL_IDENTITY_ATOL <= 1e-6
        ),
        "shape_mismatch_is_rejected": shape_rejected,
        "test_split_is_rejected": rejected(negative_split, source),
        "wrong_rate_is_rejected": rejected(negative_rate, source),
        "wrong_blocks_are_rejected": rejected(negative_blocks, source),
        "wrong_seed_is_rejected": rejected(negative_seed, source),
        "wrong_weight_is_rejected": rejected(negative_weight, source),
        "formal_smoke_budget_is_rejected": rejected(negative_formal_budget, source),
        "wrong_source_identity_is_rejected": rejected(formal, negative_identity),
        "expression_gate_maps_only_to_dmrb": (
            expression_decision["decision"]
            == "STUDENT_EXPRESSION_UNDERFIT_FREEZE_DMRB"
        ),
        "coverage_gate_has_priority": (
            coverage_decision["decision"]
            == "SUPERVISION_COVERAGE_MISMATCH_FREEZE_OBSERVABILITY_BALANCED_BF_CMVD"
        ),
        "spillover_gate_maps_only_to_cbr": (
            spill_decision["decision"]
            == "NONBENEFICIAL_SPILLOVER_FREEZE_CBR"
        ),
        "mixed_evidence_stops": (
            mixed_decision["decision"] == "MIXED_OR_INDETERMINATE_STOP"
        ),
        "failed_smoke_blocks_formal": (
            failed_state["jobs"][queue.FORMAL_JOB]["state"]
            == "BLOCKED_DEPENDENCY"
        ),
        "passed_smoke_leaves_formal_pending": (
            passed_state["jobs"][queue.FORMAL_JOB]["state"] == "PENDING"
            and passed_state["jobs"][queue.FORMAL_JOB][
                "scientific_dependency"
            ]["passed"] is True
        ),
        "queue_is_smoke_then_formal": (
            len(queue.scheduler.JOB_DEFINITIONS) == 2
            and queue.scheduler.JOB_DEFINITIONS[1]["depends_on"]
            == [queue.SMOKE_JOB]
        ),
        "run_parser_and_dispatch_are_wired": (
            "--stage22_27_bf_cmvd_diagnostic" in run_source
            and "exp.stage22_27_bf_cmvd_diagnostic(setting)" in run_source
        ),
        "diagnostic_never_constructs_test_loader": (
            "def stage22_27_bf_cmvd_diagnostic" in exp_source
            and "self._get_data(flag='val')" in exp_source.split(
                "def stage22_27_bf_cmvd_diagnostic", 1
            )[1].split("def stage22_inference_timing_validation", 1)[0]
            and "flag='test'" not in exp_source.split(
                "def stage22_27_bf_cmvd_diagnostic", 1
            )[1].split("def stage22_inference_timing_validation", 1)[0]
        ),
    }
    report = {
        "step": "22.27",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    output = ROOT / "reports/stage22_27_bf_cmvd_component_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
