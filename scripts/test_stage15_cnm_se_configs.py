#!/usr/bin/env python3
"""Static assets and protocol gates for Stage-15 CNM-SE test runs."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import yaml

from run_stage14_uncertainty_cache import resolve_path, validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "stage15_cnm_se"
CONFIGS = [
    CONFIG_ROOT / "s4m_weather_cnm_se_seed2024_smoke.yaml",
    CONFIG_ROOT / "s4m_weather_cnm_se_seed2023.yaml",
    CONFIG_ROOT / "s4m_weather_cnm_se_seed2024.yaml",
    CONFIG_ROOT / "s4m_weather_cnm_se_seed2025.yaml",
]
REPORT = ROOT / "reports" / "stage15_cnm_se_config_test.json"


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
    except (TypeError, ValueError):
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
    smoke, seed2023, seed2024, seed2025 = configs
    validation_split = copy.deepcopy(seed2024)
    validation_split["diagnostic"]["split"] = "val"
    missing_variant = copy.deepcopy(seed2024)
    del missing_variant["diagnostic"]["variant"]
    wrong_weight = copy.deepcopy(seed2024)
    wrong_weight["diagnostic"]["ensemble_weight"] = 0.4
    disabled_ensemble = copy.deepcopy(seed2024)
    disabled_ensemble["diagnostic"]["save_equal_ensemble"] = False
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
        "test_variant_frozen": all(
            config["diagnostic"]["variant"] == "stage15_test"
            and config["diagnostic"]["split"] == "test"
            and config["diagnostic"]["save_equal_ensemble"] is True
            and float(config["diagnostic"]["ensemble_weight"]) == 0.5
            for config in configs
        ),
        "formal_budgets_complete": all(
            int(config["diagnostic"]["max_eval_batches"]) == -1
            for config in (seed2023, seed2024, seed2025)
        ),
        "smoke_only_reduces_eval_batches": (
            {
                key: value
                for key, value in smoke["diagnostic"].items()
                if key != "max_eval_batches"
            }
            == {
                key: value
                for key, value in seed2024["diagnostic"].items()
                if key != "max_eval_batches"
            }
            and int(smoke["diagnostic"]["max_eval_batches"]) == 2
        ),
        "validation_split_rejected": rejected(validation_split),
        "missing_variant_rejected": rejected(missing_variant),
        "wrong_weight_rejected": rejected(wrong_weight),
        "disabled_ensemble_rejected": rejected(disabled_ensemble),
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
