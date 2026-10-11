#!/usr/bin/env python3
"""Config and scientific-gate contracts for Step 22.30."""

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
import run_stage22_30_pba_cmvd_queue as queue  # noqa: E402


def rejected(config: dict) -> bool:
    try:
        validate_experiment_config(config)
    except (FileNotFoundError, TypeError, ValueError):
        return True
    return False


def gate_metrics(**overrides) -> dict:
    metrics = {
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
        "pba_checkpoint_format": "pba_delta_v1",
        "checkpoint_format": "pba_delta_v1",
        "checkpoint_pba_tail_layers": 2,
        "checkpoint_pba_state_keys": ["s4_layers.6.weight"],
        "checkpoint_size_bytes": 16 * 1024 * 1024,
        "evaluation_split": "validation",
        "test_split_accessed": False,
        "validation_batches_processed": 3,
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
    paths = sorted((ROOT / "configs/stage22_30_pba_cmvd").glob("*.yaml"))
    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    for config in configs:
        validate_experiment_config(config)
    smoke = next(config for config in configs if config["experiment"]["smoke_test"])
    formal = next(config for config in configs if not config["experiment"]["smoke_test"])

    no_switch = copy.deepcopy(formal)
    no_switch["official_args"]["use_pba_cmvd"] = False
    wrong_tail = copy.deepcopy(formal)
    wrong_tail["official_args"]["pba_tail_layers"] = 1
    no_delta = copy.deepcopy(formal)
    no_delta["official_args"]["pba_delta_checkpoint"] = False
    test_access = copy.deepcopy(formal)
    test_access["official_args"]["post_train_evaluation"] = "test"
    wrong_weight = copy.deepcopy(formal)
    wrong_weight["official_args"]["pba_l20_weight"] = 0.3
    mixed_mcpr = copy.deepcopy(formal)
    mixed_mcpr["official_args"]["use_mcpr"] = True
    wrong_lr = copy.deepcopy(formal)
    wrong_lr["official_args"]["learning_rate"] = 0.0001
    reduced_formal = copy.deepcopy(formal)
    reduced_formal["official_args"]["train_epochs"] = 2

    fair_keys = (
        "seed", "seq_len", "pred_len", "batch_size", "learning_rate",
        "optimizer", "patience", "lradj", "pba_backbone_checkpoint",
        "pba_tail_layers", "pba_delta_checkpoint", "pba_target_rate",
        "pba_block_lengths", "pba_nested_seed", "pba_original_weight",
        "pba_l5_weight", "pba_l20_weight", "post_train_evaluation",
    )

    with tempfile.TemporaryDirectory(prefix="stage22_30_gate_") as temporary:
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
            json.dumps(gate_metrics(pba_distill_l1_mean=0.0)),
            encoding="utf-8",
        )
        failed_state = queue_state(failed_run)
        queue.enforce_scientific_gate(failed_state)

        shared_teacher_run = root / "shared_teacher"
        shared_teacher_run.mkdir()
        (shared_teacher_run / "metrics.json").write_text(
            json.dumps(gate_metrics(pba_teacher_shared_parameter_storage=True)),
            encoding="utf-8",
        )
        shared_teacher_state = queue_state(shared_teacher_run)
        queue.enforce_scientific_gate(shared_teacher_state)
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
        "validation_only": formal["official_args"]["post_train_evaluation"] == "validation",
        "pba_is_single_active_innovation": (
            formal["official_args"]["use_pba_cmvd"] is True
            and not any(formal["official_args"][key] for key in (
                "use_mcpr", "use_dmrb", "use_mrra", "use_radr",
                "use_fusr", "use_cfr", "use_bf_cmvd",
            ))
        ),
        "missing_switch_rejected": rejected(no_switch),
        "wrong_tail_rejected": rejected(wrong_tail),
        "missing_delta_rejected": rejected(no_delta),
        "test_access_rejected": rejected(test_access),
        "wrong_weight_rejected": rejected(wrong_weight),
        "mixed_mcpr_rejected": rejected(mixed_mcpr),
        "wrong_lr_rejected": rejected(wrong_lr),
        "reduced_formal_rejected": rejected(reduced_formal),
        "passing_smoke_releases_formal": (
            passed_state["jobs"][queue.FORMAL_JOB]["state"] == "PENDING"
            and passed_state["jobs"][queue.FORMAL_JOB]
            ["scientific_dependency"]["passed"] is True
        ),
        "zero_distillation_blocks_formal": (
            failed_state["jobs"][queue.FORMAL_JOB]["state"]
            == "BLOCKED_DEPENDENCY"
        ),
        "shared_teacher_blocks_formal": (
            shared_teacher_state["jobs"][queue.FORMAL_JOB]["state"]
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
        "step": "22.30",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    output = ROOT / "reports/stage22_30_pba_pipeline_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
