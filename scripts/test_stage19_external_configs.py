#!/usr/bin/env python3
"""Static protocol, isolation, and asset gates for Stage-19."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import yaml

from run_stage14_uncertainty_cache import resolve_path, validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "stage19_external_validation"
CONFIGS = [
    CONFIG_ROOT / "s4m_etth1_seed2023_smoke.yaml",
    CONFIG_ROOT / "s4m_etth1_seed2023.yaml",
]
STAGE18_CONFIGS = sorted(
    (ROOT / "configs" / "stage18_equal_rate_multiscale").glob("*.yaml")
)
REPORT = ROOT / "reports" / "stage19_external_config_test.json"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def source(config: dict) -> dict:
    return load(resolve_path(config["source"]["resolved_config"]))


def rejected(config: dict, source_override: dict | None = None) -> bool:
    try:
        validate_config(config, source_override or source(config))
    except (KeyError, TypeError, ValueError):
        return True
    return False


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    configs = [load(path) for path in CONFIGS]
    for config in configs:
        validate_config(config, source(config))
    for path in STAGE18_CONFIGS:
        config = load(path)
        validate_config(config, source(config))

    smoke, formal = configs
    wrong_dataset = copy.deepcopy(formal)
    wrong_dataset["experiment"]["dataset"] = "weather"
    wrong_method = copy.deepcopy(formal)
    wrong_method["experiment"]["method"] = "pg95"
    wrong_split = copy.deepcopy(formal)
    wrong_split["diagnostic"]["split"] = "val"
    wrong_seed = copy.deepcopy(formal)
    wrong_seed["experiment"]["seed"] = 2024
    wrong_weight = copy.deepcopy(formal)
    wrong_weight["diagnostic"]["multiscale_weights"]["l20"] = 0.30
    mipp_source = copy.deepcopy(source(formal))
    mipp_source["official_args"]["use_mipp"] = True

    checks = {
        "all_stage19_configs_valid": True,
        "stage18_configs_still_valid": len(STAGE18_CONFIGS) == 3,
        "all_assets_exist": all(
            resolve_path(config[section][key]).is_file()
            for config in configs
            for section, key in (
                ("source", "resolved_config"),
                ("source", "metrics"),
                ("source", "checkpoint"),
                ("data", "missing_csv"),
                ("data", "complete_csv"),
                ("data", "mask_asset"),
            )
        ),
        "frozen_etth1_external_identity": all(
            config["experiment"]["dataset"] == "etth1"
            and config["experiment"]["method"] == "s4m_baseline"
            and int(config["experiment"]["seed"]) == 2023
            and config["diagnostic"]["variant"] == "stage19_external_test"
            and config["diagnostic"]["split"] == "test"
            and float(config["diagnostic"]["target_rate"]) == 0.15
            and config["diagnostic"]["block_lengths"] == [1, 5, 20]
            and config["diagnostic"]["multiscale_weights"]
            == {"original": 0.5, "l5": 0.25, "l20": 0.25}
            for config in configs
        ),
        "smoke_is_three_batches": (
            smoke["experiment"]["smoke_test"] is True
            and int(smoke["diagnostic"]["max_eval_batches"]) == 3
            and int(smoke["diagnostic"]["timing_warmup_batches"]) == 1
        ),
        "formal_is_complete": (
            formal["experiment"]["smoke_test"] is False
            and int(formal["diagnostic"]["max_eval_batches"]) == -1
            and int(formal["diagnostic"]["timing_warmup_batches"]) == 10
        ),
        "source_is_original_s4m": not bool(
            source(formal)["official_args"].get("use_mipp", False)
        ),
        "wrong_dataset_rejected": rejected(wrong_dataset),
        "wrong_method_rejected": rejected(wrong_method),
        "validation_split_rejected": rejected(wrong_split),
        "wrong_seed_rejected": rejected(wrong_seed),
        "wrong_weight_rejected": rejected(wrong_weight),
        "mipp_source_rejected": rejected(formal, mipp_source),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "configs": [str(path) for path in CONFIGS],
        "stage18_regression_configs": [
            str(path) for path in STAGE18_CONFIGS
        ],
    }
    atomic_json(REPORT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
