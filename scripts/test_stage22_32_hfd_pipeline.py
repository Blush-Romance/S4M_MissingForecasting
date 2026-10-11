#!/usr/bin/env python3
"""Configuration, metric and queue contracts for Step 22.32."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_s4m_experiment import (  # noqa: E402
    validate_experiment_config,
    verify_hfd_pba_metrics,
)
import run_stage22_32_hfd_pba_queue as queue  # noqa: E402


def rejected(config: dict) -> bool:
    try:
        validate_experiment_config(config)
    except (FileNotFoundError, TypeError, ValueError):
        return True
    return False


def gate_metrics(**overrides) -> dict:
    metrics = {
        "optimizer_steps": 4,
        "pba_cmvd_enabled": True,
        "pba_tail_layers": 2,
        "pba_tail_start_index": 6,
        "pba_trainable_parameter_count": 4127582,
        "pba_trainable_group_gradient_norm_max": {
            "s4_layers_6": 1.0, "s4_layers_7": 1.0,
            "norm": 1.0, "out_proj": 1.0,
        },
        "pba_filter_granularity": "token",
        "pba_beneficial_units": 100,
        "pba_eligible_units": 500,
        "pba_distill_l1_mean": 0.1,
        "pba_frozen_hash_before": "frozen",
        "pba_frozen_hash_after": "frozen",
        "pba_memory_hash_before": "memory",
        "pba_memory_hash_after": "memory",
        "pba_trainable_hash_before": "before",
        "pba_trainable_hash_after": "after",
        "pba_teacher_hash_before": "teacher",
        "pba_teacher_hash_after": "teacher",
        "pba_teacher_memory_hash_before": "tmemory",
        "pba_teacher_memory_hash_after": "tmemory",
        "pba_teacher_shared_parameter_storage": False,
        "pba_teacher_output_requires_grad": False,
        "pba_teacher_gradient_found": False,
        "pba_inference_student_forward_count": 1,
        "pba_validation_teacher_forward_count": 0,
        "checkpoint_format": "pba_delta_v1",
        "checkpoint_pba_tail_layers": 2,
        "checkpoint_pba_state_keys": ["s4_layers.6.weight"],
        "checkpoint_size_bytes": 16 * 1024 * 1024,
        "evaluation_split": "validation",
        "test_split_accessed": False,
        "validation_batches_processed": 3,
        "hfd_pba_enabled": True,
        "hfd_layers": [6, 7],
        "hfd_loss_mode": "correction_energy_normalized_mse",
        "hfd_scale_mode": "output_distill_detached",
        "hfd_feature_capture_default_off": True,
        "hfd_validation_feature_capture_enabled": False,
        "hfd_student_feature_requires_grad": True,
        "hfd_teacher_feature_requires_grad": False,
        "hfd_active_steps": 4,
        "hfd_loss_mean": 0.05,
        "hfd_raw_mean": 0.5,
        "hfd_layer_raw_mean": {"6": 0.4, "7": 0.6},
        "hfd_layer_teacher_energy_mean": {"6": 2.0, "7": 3.0},
        "hfd_layer_feature_gradient_norm_max": {"6": 1.0, "7": 1.0},
        "hfd_training_student_feature_forward_count": 4,
        "hfd_training_teacher_feature_forward_count": 12,
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


def metric_rejected(metrics: dict, official: dict) -> bool:
    try:
        verify_hfd_pba_metrics(metrics, official)
    except RuntimeError:
        return True
    return False


def main() -> int:
    paths = sorted((ROOT / "configs/stage22_32_hfd_pba").glob("*.yaml"))
    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    for config in configs:
        validate_experiment_config(config)
    smoke = next(config for config in configs if config["experiment"]["smoke_test"])
    formal = next(config for config in configs if not config["experiment"]["smoke_test"])

    no_hfd = copy.deepcopy(formal)
    no_hfd["official_args"]["use_hfd_pba"] = False
    wrong_layers = copy.deepcopy(formal)
    wrong_layers["official_args"]["hfd_layers"] = [7]
    wrong_loss = copy.deepcopy(formal)
    wrong_loss["official_args"]["hfd_loss_mode"] = "cosine"
    wrong_scale = copy.deepcopy(formal)
    wrong_scale["official_args"]["hfd_scale_mode"] = "fixed_weight"
    no_pba = copy.deepcopy(formal)
    no_pba["official_args"]["use_pba_cmvd"] = False
    test_access = copy.deepcopy(formal)
    test_access["official_args"]["post_train_evaluation"] = "test"
    reduced_formal = copy.deepcopy(formal)
    reduced_formal["official_args"]["train_epochs"] = 2

    official = formal["official_args"]
    valid_metrics = gate_metrics()
    verify_hfd_pba_metrics(valid_metrics, official)
    zero_gradient_metrics = gate_metrics(
        hfd_layer_feature_gradient_norm_max={"6": 1.0, "7": 0.0}
    )
    teacher_grad_metrics = gate_metrics(hfd_teacher_feature_requires_grad=True)

    with tempfile.TemporaryDirectory(prefix="stage22_32_gate_") as temporary:
        root = Path(temporary)
        original_append_event = queue.scheduler.append_event
        queue.scheduler.append_event = lambda *args, **kwargs: None
        passed_run = root / "passed"
        passed_run.mkdir()
        (passed_run / "metrics.json").write_text(
            json.dumps(valid_metrics), encoding="utf-8"
        )
        passed_state = queue_state(passed_run)
        queue.enforce_scientific_gate(passed_state)

        failed_run = root / "failed"
        failed_run.mkdir()
        (failed_run / "metrics.json").write_text(
            json.dumps(zero_gradient_metrics), encoding="utf-8"
        )
        failed_state = queue_state(failed_run)
        queue.enforce_scientific_gate(failed_state)
        queue.scheduler.append_event = original_append_event

    fair_keys = (
        "seed", "seq_len", "pred_len", "batch_size", "learning_rate",
        "optimizer", "patience", "lradj", "pba_backbone_checkpoint",
        "pba_tail_layers", "pba_delta_checkpoint", "pba_target_rate",
        "pba_block_lengths", "pba_nested_seed", "pba_original_weight",
        "pba_l5_weight", "pba_l20_weight", "use_hfd_pba", "hfd_layers",
        "hfd_loss_mode", "hfd_scale_mode", "post_train_evaluation",
    )
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
        "missing_hfd_switch_rejected": rejected(no_hfd),
        "wrong_layers_rejected": rejected(wrong_layers),
        "wrong_loss_rejected": rejected(wrong_loss),
        "wrong_scale_rejected": rejected(wrong_scale),
        "missing_pba_rejected": rejected(no_pba),
        "test_access_rejected": rejected(test_access),
        "reduced_formal_rejected": rejected(reduced_formal),
        "positive_metric_contract_passes": not metric_rejected(valid_metrics, official),
        "zero_layer_gradient_rejected": metric_rejected(zero_gradient_metrics, official),
        "teacher_feature_gradient_rejected": metric_rejected(teacher_grad_metrics, official),
        "passing_smoke_releases_formal": (
            passed_state["jobs"][queue.FORMAL_JOB]["state"] == "PENDING"
            and passed_state["jobs"][queue.FORMAL_JOB]["scientific_dependency"]["passed"] is True
        ),
        "failed_hfd_level_blocks_formal": (
            failed_state["jobs"][queue.FORMAL_JOB]["state"] == "BLOCKED_DEPENDENCY"
        ),
        "queue_is_smoke_then_formal": (
            len(queue.scheduler.JOB_DEFINITIONS) == 2
            and queue.scheduler.JOB_DEFINITIONS[1]["depends_on"] == [queue.SMOKE_JOB]
        ),
        "queue_preserves_user_max_concurrency": queue.scheduler.MAX_CONCURRENT_JOBS == 5,
    }
    report = {
        "step": "22.32",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    output = ROOT / "reports/stage22_32_hfd_pipeline_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
