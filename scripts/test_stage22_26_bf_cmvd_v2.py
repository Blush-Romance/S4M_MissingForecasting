#!/usr/bin/env python3
"""Static/config/scientific-gate contracts for BF-CMVD-v2."""

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
import run_stage22_26_v2_traffic_queue as queue  # noqa: E402


def main() -> int:
    paths = sorted((ROOT / "configs/stage22_26_traffic_v2").glob("*.yaml"))
    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    for config in configs:
        validate_experiment_config(config)
    smoke = next(c for c in configs if c["experiment"]["smoke_test"])
    formal = next(c for c in configs if not c["experiment"]["smoke_test"])

    wrong_objective = copy.deepcopy(formal)
    wrong_objective["official_args"]["mcpr_objective"] = "mse_only"
    rejected = False
    try:
        validate_experiment_config(wrong_objective)
    except ValueError:
        rejected = True

    with tempfile.TemporaryDirectory(prefix="stage22_26_v2_gate_") as temporary:
        original_append_event = queue.scheduler.append_event
        queue.scheduler.append_event = lambda *args, **kwargs: None
        root = Path(temporary)

        def state_with(metrics: dict) -> dict:
            run = root / str(len(list(root.iterdir())))
            run.mkdir()
            (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            return {"jobs": {
                queue.SMOKE_JOB: {"state": "SUCCESS", "run_dir": str(run)},
                queue.FORMAL_JOB: {
                    "state": "PENDING", "depends_on": [queue.SMOKE_JOB],
                },
            }}

        common = {
            "bf_cmvd_enabled": True,
            "bf_cmvd_filter_granularity": "token",
            "bf_cmvd_diagnostic_steps": 4,
            "bf_cmvd_beneficial_units": 17,
            "bf_cmvd_total_units": 100,
            "bf_cmvd_distill_l1_mean": 0.01,
            "bf_cmvd_teacher_delta_energy_mean": 0.02,
            "mcpr_gradient_norm_max": 0.1,
            "mcpr_residual_energy_mean": 0.01,
            "checkpoint_load_verified": True,
        }
        failed_state = state_with({**common, "bf_cmvd_beneficial_units": 0})
        queue.enforce_scientific_gate(failed_state)
        passed_state = state_with(common)
        queue.enforce_scientific_gate(passed_state)
        queue.scheduler.append_event = original_append_event

    frozen_keys = (
        "seed", "seq_len", "pred_len", "batch_size", "learning_rate",
        "optimizer", "patience", "lradj", "mcpr_rank", "mcpr_kernel_size",
        "mcpr_learning_rate", "mcpr_backbone_checkpoint", "mcpr_objective",
        "bf_cmvd_target_rate", "bf_cmvd_block_lengths", "bf_cmvd_nested_seed",
        "bf_cmvd_original_weight", "bf_cmvd_l5_weight", "bf_cmvd_l20_weight",
        "post_train_evaluation",
    )
    checks = {
        "two_v2_configs_validate": len(configs) == 2,
        "token_objective_explicit": all(
            c["official_args"]["mcpr_objective"]
            == "token_benefit_filtered_multiview_distill"
            for c in configs
        ),
        "same_frozen_protocol": all(
            smoke["official_args"][key] == formal["official_args"][key]
            for key in frozen_keys
        ),
        "formal_budget_full": (
            formal["official_args"]["train_epochs"] == 15
            and formal["official_args"]["max_train_batches"] == -1
            and formal["official_args"]["max_eval_batches"] == -1
        ),
        "wrong_objective_rejected": rejected,
        "failed_token_smoke_blocks_formal": (
            failed_state["jobs"][queue.FORMAL_JOB]["state"] == "BLOCKED_DEPENDENCY"
        ),
        "passed_token_smoke_leaves_formal_pending": (
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
        "step": "22.26-v2",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
