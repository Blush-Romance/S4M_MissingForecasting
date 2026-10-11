#!/usr/bin/env python3
"""Static fairness checks for the Step-10 Weather paired-seed study."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_s4m_experiment import validate_experiment_config  # noqa: E402


SEEDS = (2024, 2025)
METHODS = ("pg95", "bmsa50")
REPORT = ROOT / "reports" / "stage10_multiseed_config_test.json"
FROZEN = {
    "pg95": ROOT / "configs" / "mipp" / "s4m_weather_mipp_pair_gate_only_t095_seed2023.yaml",
    "bmsa50": ROOT / "configs" / "mipp" / "s4m_weather_mipp_bmsa50_seed2023.yaml",
}
ALLOWED_PAIR_ARG_DIFFERENCES = {
    "model_id",
    "des",
    "mipp_write_gate_mode",
    "mipp_write_budget",
}
ALLOWED_FROZEN_EXPERIMENT_DIFFERENCES = {
    "seed",
    "realized_missing_rate",
    "protocol_scope",
    "purpose",
}
ALLOWED_FROZEN_DATA_DIFFERENCES = {
    "missing_csv",
    "mask_asset",
    "mask_metadata",
    "generation_seed",
}
ALLOWED_FROZEN_ARG_DIFFERENCES = {"seed", "model_id", "des"}


def path(method: str, seed: int, smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    return ROOT / "configs" / "mipp" / f"s4m_weather_mipp_{method}_seed{seed}{suffix}.yaml"


def load(path_: Path) -> dict:
    value = yaml.safe_load(path_.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path_)
    return value


def diff_keys(left: dict, right: dict) -> set[str]:
    keys = set(left) | set(right)
    return {key for key in keys if left.get(key, "<missing>") != right.get(key, "<missing>")}


def atomic_json(path_: Path, payload: dict) -> None:
    path_.parent.mkdir(parents=True, exist_ok=True)
    temporary = path_.with_name(f".{path_.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path_)


def main() -> int:
    frozen = {method: load(path_) for method, path_ in FROZEN.items()}
    configs = {}
    checks = {}
    for seed in SEEDS:
        for method in METHODS:
            for smoke in (True, False):
                config_path = path(method, seed, smoke)
                config = load(config_path)
                validate_experiment_config(config)
                configs[(method, seed, smoke)] = config
                tag = f"{method}_seed{seed}_{'smoke' if smoke else 'formal'}"
                checks[f"{tag}_valid"] = True
                checks[f"{tag}_seed_consistent"] = (
                    int(config["experiment"]["seed"]) == seed
                    and int(config["data"]["generation_seed"]) == seed
                    and int(config["official_args"]["seed"]) == seed
                )
                checks[f"{tag}_assets_exist"] = all(
                    (ROOT / config["data"][key]).is_file()
                    for key in (
                        "missing_csv",
                        "complete_csv",
                        "mask_asset",
                        "mask_metadata",
                        "source_manifest",
                    )
                )

        pg95 = configs[("pg95", seed, False)]
        bmsa = configs[("bmsa50", seed, False)]
        checks[f"seed{seed}_paired_data_identical"] = pg95["data"] == bmsa["data"]
        checks[f"seed{seed}_only_bmsa_fields_differ"] = (
            diff_keys(pg95["official_args"], bmsa["official_args"])
            == ALLOWED_PAIR_ARG_DIFFERENCES
        )
        checks[f"seed{seed}_bmsa_budget_matches_M"] = (
            bmsa["official_args"]["mipp_write_gate_mode"] == "budget_topk"
            and int(bmsa["official_args"]["mipp_write_budget"])
            == int(bmsa["official_args"]["M"])
        )
        for method in METHODS:
            formal = configs[(method, seed, False)]
            reference = frozen[method]
            checks[f"{method}_seed{seed}_frozen_experiment"] = (
                diff_keys(reference["experiment"], formal["experiment"])
                <= ALLOWED_FROZEN_EXPERIMENT_DIFFERENCES
            )
            checks[f"{method}_seed{seed}_frozen_data_protocol"] = (
                diff_keys(reference["data"], formal["data"])
                == ALLOWED_FROZEN_DATA_DIFFERENCES
            )
            checks[f"{method}_seed{seed}_frozen_training_args"] = (
                diff_keys(reference["official_args"], formal["official_args"])
                == ALLOWED_FROZEN_ARG_DIFFERENCES
            )
            args = formal["official_args"]
            checks[f"{method}_seed{seed}_formal_not_reduced"] = (
                int(args["batch_size"]) == 16
                and int(args["train_epochs"]) == 15
                and int(args["max_train_batches"]) == -1
                and int(args["max_eval_batches"]) == -1
                and args["optimizer"] == "adam"
                and int(args["patience"]) == 3
            )

    asset_report = json.loads(
        (ROOT / "reports" / "stage10_multiseed_asset_test.json").read_text(encoding="utf-8")
    )
    checks["asset_gate_passed"] = asset_report.get("status") == "PASS"
    passed = all(checks.values())
    report = {
        "status": "PASS" if passed else "FAIL",
        "pre_registered_seeds": [2023, *SEEDS],
        "checks": checks,
        "configs": [
            str(path(method, seed, smoke))
            for seed in SEEDS
            for method in METHODS
            for smoke in (True, False)
        ],
    }
    atomic_json(REPORT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
