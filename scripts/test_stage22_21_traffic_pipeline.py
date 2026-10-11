#!/usr/bin/env python3
"""Static, negative, handoff, and gate tests for Step-22.21 Traffic."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

import yaml

from analyze_stage22_21_traffic_validation import (
    audit_method_differences,
    matches_frozen_training_budget,
    scientific_decision,
)
from run_s4m_experiment import validate_experiment_config
from run_stage14_uncertainty_cache import validate_config as validate_eval_config
from run_stage22_21_traffic_pipeline import materialize_evaluation_config
import run_stage22_21_traffic_queue as queue


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/stage22_21_traffic"
REPORT = ROOT / "reports/stage22_21_traffic_pipeline_test.json"


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def expect_rejected(callback, label: str) -> None:
    try:
        callback()
    except (KeyError, RuntimeError, TypeError, ValueError):
        return
    raise AssertionError(f"negative case was accepted: {label}")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def setting_name(args: dict, data_path: str) -> str:
    return (
        f"{args['model']}_custom4_ft{args['seq_len']}_ll{args['pred_len']}"
        f"_pl{args['d_model']}_dm{args['e_layers']}_eltimeF"
        f"_eb{args['des']}_dtprojection_0_data_{data_path[:-4]}tag1_fill_"
        f"{args['fillna']}_drop_False_dG_{args['num_dG']}"
        f"_pattn_{args['num_pattn']}_n_10_slen_{args['short_len']}"
        f"_W_{args['W']}_ts1_{args['thres1']}_ts2_{args['thres2']}"
        f"_K_{args['K']}_n_10_M_{args['M']}_pre_False"
        f"_en_{args['en_conv_hidden_size']}_s_25_top{args['topK']}"
        f"_P_{args['per_mem_size']}_m_{args['memnet']}_t_test"
    )


def main() -> int:
    paths = {
        "s4m_smoke": CONFIG_ROOT / "s4m_traffic_seed2026_smoke.yaml",
        "s4m_formal": CONFIG_ROOT / "s4m_traffic_seed2026.yaml",
        "pg95_smoke": CONFIG_ROOT / "pg95_traffic_seed2026_smoke.yaml",
        "pg95_formal": CONFIG_ROOT / "pg95_traffic_seed2026.yaml",
        "validation_smoke": (
            CONFIG_ROOT / "pg95_traffic_multiscale_validation_seed2026_smoke.yaml"
        ),
        "validation_formal": (
            CONFIG_ROOT / "pg95_traffic_multiscale_validation_seed2026.yaml"
        ),
        "gated_test": CONFIG_ROOT / "pg95_traffic_multiscale_test_seed2026.yaml",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    configs = {name: load(path) for name, path in paths.items()}
    for name in ("s4m_smoke", "s4m_formal", "pg95_smoke", "pg95_formal"):
        validate_experiment_config(configs[name])
    for name in ("validation_smoke", "validation_formal", "gated_test"):
        validate_eval_config(configs[name], configs["pg95_formal"])

    checks = {
        "training_dataset_seed_target_width": all(
            config["experiment"]["dataset"] == "traffic"
            and int(config["experiment"]["seed"]) == 2026
            and str(config["official_args"]["target"]) == "OT"
            and config["official_args"]["enc_in"]
            == config["official_args"]["dec_in"]
            == config["official_args"]["c_out"] == 862
            for config in (
                configs["s4m_smoke"], configs["s4m_formal"],
                configs["pg95_smoke"], configs["pg95_formal"],
            )
        ),
        "validation_only_training": all(
            configs[name]["official_args"]["post_train_evaluation"] == "validation"
            for name in ("s4m_smoke", "s4m_formal", "pg95_smoke", "pg95_formal")
        ),
        "extended_stability_smokes_and_complete_formals": all(
            configs[f"{family}_smoke"]["official_args"]["max_train_batches"] == 64
            and configs[f"{family}_smoke"]["official_args"]["max_eval_batches"] == 3
            and configs[f"{family}_smoke"]["official_args"]["train_epochs"] == 1
            and configs[f"{family}_formal"]["official_args"]["max_train_batches"] == -1
            and configs[f"{family}_formal"]["official_args"]["max_eval_batches"] == -1
            and configs[f"{family}_formal"]["official_args"]["train_epochs"] == 15
            for family in ("s4m", "pg95")
        ),
        "shared_authorized_lr_amendment": all(
            configs[name]["official_args"]["optimizer"] == "adam"
            and configs[name]["official_args"]["learning_rate"] == 0.001
            and configs[name]["experiment"]["training_stability_amendment"]
            == "adam_lr_0.01_to_0.001_user_authorized_20260816"
            for name in ("s4m_smoke", "s4m_formal", "pg95_smoke", "pg95_formal")
        ),
        "analyzer_accepts_authorized_formal_budget": all(
            matches_frozen_training_budget(configs[name])
            for name in ("s4m_formal", "pg95_formal")
        ),
        "validation_variants_use_val": all(
            configs[name]["diagnostic"]["variant"]
            == "stage22_21_traffic_validation"
            and configs[name]["diagnostic"]["split"] == "val"
            for name in ("validation_smoke", "validation_formal")
        ),
        "test_variant_is_unique_and_gated": (
            configs["gated_test"]["diagnostic"]["variant"]
            == "stage22_21_traffic_test"
            and configs["gated_test"]["diagnostic"]["split"] == "test"
            and "only_after" in configs["gated_test"]["experiment"]["purpose"]
        ),
        "all_eval_sources_are_pg95_formal": all(
            configs[name]["source"] == {"job_id": "traffic_pg95_seed2026_formal"}
            for name in ("validation_smoke", "validation_formal", "gated_test")
        ),
        "frozen_multiscale_protocol": all(
            configs[name]["diagnostic"]["target_rate"] == 0.15
            and configs[name]["diagnostic"]["block_lengths"] == [1, 5, 20]
            and configs[name]["diagnostic"]["multiscale_weights"]
            == {"original": 0.5, "l5": 0.25, "l20": 0.25}
            for name in ("validation_smoke", "validation_formal", "gated_test")
        ),
        "formal_setting_names_below_filesystem_limit": all(
            len(setting_name(
                configs[name]["official_args"],
                Path(configs[name]["data"]["missing_csv"]).name,
            ).encode("utf-8")) <= 255
            for name in ("s4m_formal", "pg95_formal")
        ),
    }

    baseline_args = copy.deepcopy(configs["s4m_formal"]["official_args"])
    pg95_args = copy.deepcopy(configs["pg95_formal"]["official_args"])
    actual, unexpected, required = audit_method_differences(
        baseline_args, pg95_args
    )
    checks["formal_difference_is_only_pg95"] = not unexpected and required
    checks["formal_assets_identical"] = (
        configs["s4m_formal"]["data"] == configs["pg95_formal"]["data"]
    )
    drift = copy.deepcopy(pg95_args)
    drift["batch_size"] = 8
    _, drift_unexpected, _ = audit_method_differences(baseline_args, drift)
    checks["budget_drift_is_detected"] = drift_unexpected == {"batch_size"}

    stale_lr_config = copy.deepcopy(configs["pg95_formal"])
    stale_lr_config["official_args"]["learning_rate"] = 0.01
    missing_amendment_config = copy.deepcopy(configs["pg95_formal"])
    missing_amendment_config["experiment"].pop("training_stability_amendment")
    changed_epoch_config = copy.deepcopy(configs["pg95_formal"])
    changed_epoch_config["official_args"]["train_epochs"] = 14
    checks["analyzer_rejects_stale_lr_and_budget_drift"] = all(
        not matches_frozen_training_budget(config)
        for config in (
            stale_lr_config, missing_amendment_config, changed_epoch_config
        )
    )

    bad = copy.deepcopy(configs["validation_formal"])
    bad["diagnostic"]["split"] = "test"
    expect_rejected(
        lambda: validate_eval_config(bad, configs["pg95_formal"]),
        "validation_with_test_split",
    )
    bad = copy.deepcopy(configs["pg95_formal"])
    bad["official_args"]["target"] = "861"
    expect_rejected(
        lambda: validate_eval_config(configs["validation_formal"], bad),
        "traffic_nonexistent_numeric_target",
    )
    bad = copy.deepcopy(configs["pg95_formal"])
    bad["official_args"]["enc_in"] = 321
    expect_rejected(
        lambda: validate_eval_config(configs["validation_formal"], bad),
        "traffic_wrong_width",
    )
    bad = copy.deepcopy(configs["gated_test"])
    bad["diagnostic"]["multiscale_weights"]["l20"] = 0.30
    expect_rejected(
        lambda: validate_eval_config(bad, configs["pg95_formal"]),
        "changed_test_weight",
    )
    checks["negative_protocol_cases_rejected"] = True

    decisions = {
        "protocol": scientific_decision(
            protocol_pass=False, pg95_gate=True,
            view_response_gate=True, multiscale_gate=True,
        ),
        "pg95": scientific_decision(
            protocol_pass=True, pg95_gate=False,
            view_response_gate=True, multiscale_gate=True,
        ),
        "view": scientific_decision(
            protocol_pass=True, pg95_gate=True,
            view_response_gate=False, multiscale_gate=True,
        ),
        "multiscale": scientific_decision(
            protocol_pass=True, pg95_gate=True,
            view_response_gate=True, multiscale_gate=False,
        ),
        "pass": scientific_decision(
            protocol_pass=True, pg95_gate=True,
            view_response_gate=True, multiscale_gate=True,
        ),
    }
    checks["decision_precedence_and_authorization"] = (
        decisions["protocol"] == ("TRAFFIC_PROTOCOL_FAIL_TEST_FORBIDDEN", False)
        and decisions["pg95"] == ("TRAFFIC_PG95_FAIL_TEST_FORBIDDEN", False)
        and decisions["view"] == ("TRAFFIC_VIEW_COLLAPSE_TEST_FORBIDDEN", False)
        and decisions["multiscale"]
        == ("TRAFFIC_MULTISCALE_FAIL_TEST_FORBIDDEN", False)
        and decisions["pass"]
        == ("TRAFFIC_VALIDATION_PASS_ONE_TEST_AUTHORIZED", True)
    )

    definitions = {job["id"]: job for job in queue.scheduler.JOB_DEFINITIONS}
    checks["queue_dependencies_freeze_test_order"] = (
        definitions["traffic_s4m_seed2026_formal"]["depends_on"]
        == ["traffic_s4m_seed2026_smoke"]
        and definitions["traffic_pg95_seed2026_formal"]["depends_on"]
        == ["traffic_pg95_seed2026_smoke"]
        and definitions["traffic_pg95_multiscale_validation_seed2026_formal"]
        ["depends_on"]
        == ["traffic_pg95_multiscale_validation_seed2026_smoke"]
        and set(definitions["traffic_conditional_single_test_seed2026_formal"]
                ["depends_on"])
        == {
            "traffic_s4m_seed2026_formal",
            "traffic_pg95_multiscale_validation_seed2026_formal",
        }
    )

    with tempfile.TemporaryDirectory(prefix="stage22_21_handoff_") as temporary:
        temp_root = Path(temporary)
        source_run = temp_root / "formal_pg95"
        checkpoint = source_run / "checkpoints/setting/checkpoint.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        (source_run / "resolved_config.yaml").write_text(
            yaml.safe_dump(configs["pg95_formal"]), encoding="utf-8"
        )
        source_metrics = {
            "checkpoint": str(checkpoint),
            "evaluation_split": "validation",
            "post_train_evaluation": "validation",
            "test_split_accessed": False,
            "chronological_loader": True,
        }
        write_json(source_run / "metrics.json", source_metrics)
        write_json(source_run / "status.json", {"state": "SUCCESS"})
        state_path = temp_root / "state.json"
        write_json(state_path, {"jobs": {
            "traffic_pg95_seed2026_formal": {
                "state": "SUCCESS", "run_dir": str(source_run),
            }
        }})
        destination = temp_root / "materialized.yaml"
        materialize_evaluation_config(
            paths["validation_smoke"],
            state_path=state_path,
            destination=destination,
        )
        materialized = load(destination)
        checks["exact_checkpoint_handoff"] = (
            Path(materialized["source"]["checkpoint"]).resolve()
            == checkpoint.resolve()
            and materialized["provenance"]["source_is_validation_only"] is True
        )
        source_metrics["test_split_accessed"] = True
        write_json(source_run / "metrics.json", source_metrics)
        expect_rejected(
            lambda: materialize_evaluation_config(
                paths["validation_smoke"],
                state_path=state_path,
                destination=temp_root / "must_not_exist.yaml",
            ),
            "test_accessed_source",
        )
        checks["test_accessed_source_rejected"] = True

    failures = [name for name, passed in checks.items() if not passed]
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "negative_cases": 8,
        "formal_method_differences": sorted(actual),
        "setting_name_bytes": {
            name: len(setting_name(
                configs[name]["official_args"],
                Path(configs[name]["data"]["missing_csv"]).name,
            ).encode("utf-8"))
            for name in ("s4m_formal", "pg95_formal")
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
