#!/usr/bin/env python3
"""Static protocol and asset gates for Stage-18 validation experiments."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import yaml

from run_stage14_uncertainty_cache import resolve_path, validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "stage18_equal_rate_multiscale"
CONFIGS = [
    CONFIG_ROOT / "s4m_weather_pg95_seed2024_smoke.yaml",
    CONFIG_ROOT / "s4m_weather_pg95_seed2024.yaml",
    CONFIG_ROOT / "s4m_weather_pg95_seed2025.yaml",
]
REPORT = ROOT / "reports" / "stage18_equal_rate_config_test.json"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def source(config: dict) -> dict:
    return load(resolve_path(config["source"]["resolved_config"]))


def rejected(config: dict) -> bool:
    try:
        validate_config(config, source(config))
    except (KeyError, TypeError, ValueError):
        return True
    return False


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    configs = [load(path) for path in CONFIGS]
    for config in configs:
        validate_config(config, source(config))
    smoke = configs[0]
    formals = configs[1:]
    wrong_split = copy.deepcopy(formals[0])
    wrong_split["diagnostic"]["split"] = "test"
    wrong_rate = copy.deepcopy(formals[0])
    wrong_rate["diagnostic"]["target_rate"] = 0.14
    wrong_lengths = copy.deepcopy(formals[0])
    wrong_lengths["diagnostic"]["block_lengths"] = [1, 5, 10]
    wrong_weight = copy.deepcopy(formals[0])
    wrong_weight["diagnostic"]["multiscale_weights"]["l20"] = 0.3
    wrong_seed = copy.deepcopy(formals[0])
    wrong_seed["diagnostic"]["nested_seed"] = 2025
    no_timed_batch = copy.deepcopy(smoke)
    no_timed_batch["diagnostic"]["timing_warmup_batches"] = 3
    checks = {
        "all_configs_valid": True,
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
        "validation_only_identity_frozen": all(
            config["diagnostic"]["variant"] == "stage18_validation"
            and config["diagnostic"]["split"] == "val"
            and config["diagnostic"]["block_lengths"] == [1, 5, 20]
            and float(config["diagnostic"]["target_rate"]) == 0.15
            and config["diagnostic"]["multiscale_weights"]
            == {"original": 0.5, "l5": 0.25, "l20": 0.25}
            for config in configs
        ),
        "one_smoke_three_batches": (
            smoke["experiment"]["smoke_test"] is True
            and int(smoke["diagnostic"]["max_eval_batches"]) == 3
            and int(smoke["diagnostic"]["timing_warmup_batches"]) == 1
        ),
        "two_complete_formals": (
            {int(config["experiment"]["seed"]) for config in formals}
            == {2024, 2025}
            and all(
                config["experiment"]["smoke_test"] is False
                and int(config["diagnostic"]["max_eval_batches"]) == -1
                and int(config["diagnostic"]["timing_warmup_batches"]) == 10
                for config in formals
            )
        ),
        "test_split_rejected": rejected(wrong_split),
        "wrong_rate_rejected": rejected(wrong_rate),
        "wrong_lengths_rejected": rejected(wrong_lengths),
        "wrong_weight_rejected": rejected(wrong_weight),
        "wrong_seed_rejected": rejected(wrong_seed),
        "warmup_consuming_all_batches_rejected": rejected(no_timed_batch),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "configs": [str(path) for path in CONFIGS],
    }
    atomic_json(REPORT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
