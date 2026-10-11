#!/usr/bin/env python3
"""Validate Step-21 Electricity training configs before GPU smoke tests."""

from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from run_s4m_experiment import validate_experiment_config  # noqa: E402


CONFIG_ROOT = PROJECT_ROOT / "configs/stage21_electricity"
CONFIGS = {
    "s4m_smoke": CONFIG_ROOT / "s4m_electricity_seed2023_smoke.yaml",
    "s4m_formal": CONFIG_ROOT / "s4m_electricity_seed2023.yaml",
    "pg95_smoke": CONFIG_ROOT / "pg95_electricity_seed2023_smoke.yaml",
    "pg95_formal": CONFIG_ROOT / "pg95_electricity_seed2023.yaml",
}
RUN_SOURCE = PROJECT_ROOT / "baselines/S4M/s4m/run.py"
EXPERIMENT_SOURCE = (
    PROJECT_ROOT / "baselines/S4M/s4m/experiments/exp_pretrain1.py"
)
REPORT = PROJECT_ROOT / "reports/stage21_electricity_training_config_test.json"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_experiment_config(config)
    return config


def function_source(path: Path, function_name: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                return ast.get_source_segment(text, node) or ""
    raise KeyError(f"function not found: {function_name}")


def setting_name(args: dict) -> str:
    return (
        f"{args['model']}_custom4_ft{args['seq_len']}_ll{args['pred_len']}"
        f"_pl{args['d_model']}_dm{args['e_layers']}_eltimeF"
        f"_eb{args['des']}_dtprojection_0_data_"
        f"electricity_point_paper_r0.06_l5_seed2023tag1_fill_"
        f"{args['fillna']}_drop_False_dG_{args['num_dG']}"
        f"_pattn_{args['num_pattn']}_n_10_slen_{args['short_len']}"
        f"_W_{args['W']}_ts1_{args['thres1']}_ts2_{args['thres2']}"
        f"_K_{args['K']}_n_10_M_{args['M']}_pre_False"
        f"_en_{args['en_conv_hidden_size']}_s_25_top{args['topK']}"
        f"_P_{args['per_mem_size']}_m_{args['memnet']}_t_test"
    )


def main() -> None:
    configs = {name: read_config(path) for name, path in CONFIGS.items()}
    checks: dict[str, bool] = {}
    for name, config in configs.items():
        experiment = config["experiment"]
        data = config["data"]
        args = config["official_args"]
        checks[f"{name}_dataset"] = experiment["dataset"] == "electricity"
        checks[f"{name}_seed"] = int(experiment["seed"]) == 2023
        checks[f"{name}_validation_only"] = (
            args["post_train_evaluation"] == "validation"
        )
        checks[f"{name}_full_width"] = (
            args["enc_in"] == args["dec_in"] == args["c_out"] == 321
        )
        checks[f"{name}_paper_hidden"] = (
            args["d_model"] == args["d_ff"] == args["en_conv_hidden_size"] == 512
        )
        checks[f"{name}_frozen_architecture"] = (
            args["e_layers"] == 8 and args["n_heads"] == 8
        )
        checks[f"{name}_paper_optimizer"] = (
            args["optimizer"] == "adam"
            and float(args["learning_rate"]) == 0.01
            and int(args["batch_size"]) == 16
        )
        checks[f"{name}_fixed_assets"] = all(
            (PROJECT_ROOT / data[key]).is_file()
            for key in (
                "missing_csv", "complete_csv", "mask_asset", "mask_metadata",
            )
        )
        checks[f"{name}_protocol_disclosure"] = (
            experiment["protocol_match"]
            == "approximate_missing_mask_and_architecture_seed_unavailable"
        )
        checks[f"{name}_setting_below_name_max"] = (
            len(setting_name(args).encode("utf-8")) <= 255
        )

    for family in ("s4m", "pg95"):
        smoke = configs[f"{family}_smoke"]
        formal = configs[f"{family}_formal"]
        checks[f"{family}_smoke_is_limited"] = (
            smoke["experiment"]["smoke_test"] is True
            and smoke["official_args"]["train_epochs"] == 2
            and smoke["official_args"]["max_train_batches"] == 2
            and smoke["official_args"]["max_eval_batches"] == 3
        )
        checks[f"{family}_formal_is_full"] = (
            formal["experiment"]["smoke_test"] is False
            and formal["official_args"]["train_epochs"] == 15
            and formal["official_args"]["max_train_batches"] == -1
            and formal["official_args"]["max_eval_batches"] == -1
        )

    allowed_method_differences = {
        "model_id", "des", "use_mipp", "use_content_mask_factorization",
        "use_mask_intervention_pair", "use_stability_write_gate",
        "mipp_write_gate_mode", "mipp_inv_weight", "mipp_var_weight",
        "mipp_cov_weight", "mipp_dec_weight", "mipp_mask_weight",
        "mipp_variance_target", "mipp_stability_threshold",
        "mipp_pair_rate_min", "mipp_pair_rate_max",
        "mipp_pair_block_lengths",
    }
    left = configs["s4m_formal"]["official_args"]
    right = configs["pg95_formal"]["official_args"]
    actual_differences = {
        key
        for key in set(left) | set(right)
        if left.get(key) != right.get(key)
    }
    checks["formal_difference_is_only_pg95"] = (
        actual_differences <= allowed_method_differences
        and actual_differences
        >= {"model_id", "des", "use_mipp"}
    )
    checks["formal_data_assets_identical"] = (
        configs["s4m_formal"]["data"] == configs["pg95_formal"]["data"]
    )
    checks["baseline_modules_disabled"] = all(
        configs["s4m_formal"]["official_args"].get(key, False) is False
        for key in ("use_mipp", "use_fusr", "use_cfr")
    )
    checks["pg95_identity"] = (
        configs["pg95_formal"]["official_args"]["use_mipp"] is True
        and configs["pg95_formal"]["official_args"]["use_fusr"] is False
        and configs["pg95_formal"]["official_args"]["use_cfr"] is False
        and configs["pg95_formal"]["official_args"]["mipp_write_gate_mode"]
        == "absolute"
        and float(
            configs["pg95_formal"]["official_args"][
                "mipp_stability_threshold"
            ]
        )
        == 0.95
    )

    invalid = copy.deepcopy(configs["s4m_smoke"])
    invalid["official_args"]["post_train_evaluation"] = "test_after_validation"
    invalid_rejected = False
    try:
        validate_experiment_config(invalid)
    except ValueError:
        invalid_rejected = True
    checks["invalid_post_train_split_rejected"] = invalid_rejected

    train_source = function_source(EXPERIMENT_SOURCE, "train")
    validation_source = function_source(
        EXPERIMENT_SOURCE, "post_train_validation"
    )
    run_source = RUN_SOURCE.read_text(encoding="utf-8")
    checks["train_does_not_construct_test_loader"] = (
        "_get_data(flag='test')" not in train_source
    )
    checks["post_train_validation_uses_val"] = (
        "_get_data(flag='val')" in validation_source
        and "shuffle=False" in validation_source
        and "'test_split_accessed': False" in validation_source
    )
    checks["run_dispatch_has_validation_branch"] = (
        "args.post_train_evaluation == 'validation'" in run_source
        and "exp.post_train_validation(setting)" in run_source
    )

    failures = [name for name, passed in checks.items() if not passed]
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "formal_method_differences": sorted(actual_differences),
        "setting_name_bytes": {
            name: len(setting_name(config["official_args"]).encode("utf-8"))
            for name, config in configs.items()
        },
    }
    atomic_json(REPORT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
