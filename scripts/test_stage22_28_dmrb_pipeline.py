#!/usr/bin/env python3
"""Config, queue, checkpoint, and scientific-gate contracts for Step 22.28."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_s4m_experiment import validate_experiment_config  # noqa: E402
import run_stage22_28_dmrb_queue as queue  # noqa: E402


def rejected(config: dict) -> bool:
    try:
        validate_experiment_config(config)
    except (FileNotFoundError, TypeError, ValueError):
        return True
    return False


def gate_metrics(**overrides) -> dict:
    metrics = {
        "dmrb_enabled": True,
        "dmrb_l5_enabled": True,
        "dmrb_l20_enabled": True,
        "bf_cmvd_filter_granularity": "token",
        "bf_cmvd_beneficial_units": 100,
        "bf_cmvd_total_units": 500,
        "dmrb_l5_distill_l1_mean": 0.01,
        "dmrb_l20_distill_l1_mean": 0.02,
        "dmrb_l5_residual_energy_mean": 0.001,
        "dmrb_l20_residual_energy_mean": 0.002,
        "dmrb_l5_gradient_norm_max": 0.1,
        "dmrb_l20_gradient_norm_max": 0.2,
        "dmrb_teacher_reconstruction_max_abs": 1e-7,
        "mcpr_trainable_parameter_count": 90364,
        "mcpr_backbone_hash_before": "backbone",
        "mcpr_backbone_hash_after": "backbone",
        "mcpr_memory_hash_before": "memory",
        "mcpr_memory_hash_after": "memory",
        "dmrb_inference_backbone_forward_count": 1,
        "mcpr_checkpoint_format": "mcpr_delta_v1",
        "checkpoint_format": "mcpr_delta_v1",
        "checkpoint_dmrb_branches_verified": True,
        "checkpoint_size_bytes": 500000,
        "evaluation_split": "validation",
        "test_split_accessed": False,
    }
    metrics.update(overrides)
    return metrics


def queue_state(run_dir: Path) -> dict:
    return {"jobs": {
        queue.SMOKE_JOB: {"state": "SUCCESS", "run_dir": str(run_dir)},
        queue.FORMAL_JOB: {
            "state": "PENDING", "depends_on": [queue.SMOKE_JOB],
        },
    }}


def main() -> int:
    paths = sorted((ROOT / "configs/stage22_28_dmrb").glob("*.yaml"))
    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    for config in configs:
        validate_experiment_config(config)
    smoke = next(config for config in configs if config["experiment"]["smoke_test"])
    formal = next(config for config in configs if not config["experiment"]["smoke_test"])

    no_dmrb = copy.deepcopy(formal)
    no_dmrb["official_args"]["use_dmrb"] = False
    wrong_objective = copy.deepcopy(formal)
    wrong_objective["official_args"]["mcpr_objective"] = "token_benefit_filtered_multiview_distill"
    no_branches = copy.deepcopy(formal)
    no_branches["official_args"]["dmrb_l5_enabled"] = False
    no_branches["official_args"]["dmrb_l20_enabled"] = False
    wrong_rank = copy.deepcopy(formal)
    wrong_rank["official_args"]["mcpr_rank"] = 16
    no_delta_checkpoint = copy.deepcopy(formal)
    no_delta_checkpoint["official_args"]["mcpr_delta_checkpoint"] = False
    test_access = copy.deepcopy(formal)
    test_access["official_args"]["post_train_evaluation"] = "test"
    wrong_teacher_weight = copy.deepcopy(formal)
    wrong_teacher_weight["official_args"]["bf_cmvd_l5_weight"] = 0.30

    fair_keys = (
        "seed", "seq_len", "pred_len", "batch_size", "learning_rate",
        "optimizer", "patience", "lradj", "mcpr_rank", "mcpr_kernel_size",
        "mcpr_learning_rate", "mcpr_backbone_checkpoint", "mcpr_objective",
        "mcpr_delta_checkpoint", "bf_cmvd_target_rate",
        "bf_cmvd_block_lengths", "bf_cmvd_nested_seed",
        "bf_cmvd_original_weight", "bf_cmvd_l5_weight",
        "bf_cmvd_l20_weight", "use_dmrb", "dmrb_l5_enabled",
        "dmrb_l20_enabled", "post_train_evaluation",
    )

    with tempfile.TemporaryDirectory(prefix="stage22_28_gate_") as temporary:
        root = Path(temporary)
        original_append_event = queue.scheduler.append_event
        queue.scheduler.append_event = lambda *args, **kwargs: None
        passed_run = root / "passed"
        passed_run.mkdir()
        (passed_run / "metrics.json").write_text(
            json.dumps(gate_metrics()), encoding="utf-8"
        )
        passed_state = queue_state(passed_run)
        queue.enforce_scientific_gate(passed_state)

        failed_run = root / "failed"
        failed_run.mkdir()
        (failed_run / "metrics.json").write_text(
            json.dumps(gate_metrics(dmrb_l20_gradient_norm_max=0.0)),
            encoding="utf-8",
        )
        failed_state = queue_state(failed_run)
        queue.enforce_scientific_gate(failed_state)

        full_checkpoint_run = root / "full_checkpoint"
        full_checkpoint_run.mkdir()
        (full_checkpoint_run / "metrics.json").write_text(
            json.dumps(gate_metrics(
                checkpoint_format="full_model_v1",
                checkpoint_size_bytes=120 * 1024 * 1024,
            )),
            encoding="utf-8",
        )
        full_checkpoint_state = queue_state(full_checkpoint_run)
        queue.enforce_scientific_gate(full_checkpoint_state)

        bad_factorization_run = root / "bad_factorization"
        bad_factorization_run.mkdir()
        (bad_factorization_run / "metrics.json").write_text(
            json.dumps(gate_metrics(dmrb_teacher_reconstruction_max_abs=4e-6)),
            encoding="utf-8",
        )
        bad_factorization_state = queue_state(bad_factorization_run)
        queue.enforce_scientific_gate(bad_factorization_state)
        queue.scheduler.append_event = original_append_event

    checks = {
        "two_configs_validate": len(configs) == 2,
        "smoke_budget_is_isolated": (
            smoke["official_args"]["train_epochs"] == 2
            and smoke["official_args"]["max_train_batches"] == 2
            and smoke["official_args"]["max_eval_batches"] == 3
        ),
        "formal_budget_is_full": (
            formal["official_args"]["train_epochs"] == 15
            and formal["official_args"]["max_train_batches"] == -1
            and formal["official_args"]["max_eval_batches"] == -1
        ),
        "same_frozen_protocol": all(
            smoke["official_args"][key] == formal["official_args"][key]
            for key in fair_keys
        ),
        "validation_only_is_frozen": (
            formal["official_args"]["post_train_evaluation"] == "validation"
        ),
        "both_branches_are_explicit": (
            formal["official_args"]["use_dmrb"] is True
            and formal["official_args"]["dmrb_l5_enabled"] is True
            and formal["official_args"]["dmrb_l20_enabled"] is True
        ),
        "delta_checkpoint_is_explicit": (
            formal["official_args"]["mcpr_delta_checkpoint"] is True
        ),
        "missing_dmrb_switch_is_rejected": rejected(no_dmrb),
        "wrong_objective_is_rejected": rejected(wrong_objective),
        "empty_branch_set_is_rejected": rejected(no_branches),
        "wrong_rank_is_rejected": rejected(wrong_rank),
        "missing_delta_checkpoint_is_rejected": rejected(no_delta_checkpoint),
        "test_evaluation_is_rejected_by_protocol_test": rejected(test_access),
        "wrong_teacher_weight_is_rejected": rejected(wrong_teacher_weight),
        "passing_smoke_releases_formal": (
            passed_state["jobs"][queue.FORMAL_JOB]["state"] == "PENDING"
            and passed_state["jobs"][queue.FORMAL_JOB]
            ["scientific_dependency"]["passed"] is True
        ),
        "dead_branch_blocks_formal": (
            failed_state["jobs"][queue.FORMAL_JOB]["state"]
            == "BLOCKED_DEPENDENCY"
        ),
        "full_checkpoint_blocks_formal": (
            full_checkpoint_state["jobs"][queue.FORMAL_JOB]["state"]
            == "BLOCKED_DEPENDENCY"
        ),
        "fp32_tolerance_is_frozen_and_strict": (
            queue.DMRB_TEACHER_RECONSTRUCTION_ATOL
            == 32 * 1.1920928955078125e-7
            and 1.9073486328125e-6
            < queue.DMRB_TEACHER_RECONSTRUCTION_ATOL
            < 4e-6
            and bad_factorization_state["jobs"][queue.FORMAL_JOB]["state"]
            == "BLOCKED_DEPENDENCY"
        ),
        "queue_is_smoke_then_formal": (
            len(queue.scheduler.JOB_DEFINITIONS) == 2
            and queue.scheduler.JOB_DEFINITIONS[1]["depends_on"]
            == [queue.SMOKE_JOB]
        ),
        "queue_preserves_user_max_concurrency": (
            queue.scheduler.MAX_CONCURRENT_JOBS == 5
        ),
    }
    report = {
        "step": "22.28",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    output = ROOT / "reports/stage22_28_dmrb_pipeline_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
