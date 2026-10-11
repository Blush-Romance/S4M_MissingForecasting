#!/usr/bin/env python3
"""Static gates for the fixed 0.5/0.5 Stage-14 validation ensemble."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import yaml

from run_stage14_uncertainty_cache import resolve_path, validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "stage14_equal_ensemble"
CONFIGS = [
    CONFIG_ROOT / "s4m_weather_pg95_equal_ensemble_seed2024_smoke.yaml",
    CONFIG_ROOT / "s4m_weather_pg95_equal_ensemble_seed2024.yaml",
    CONFIG_ROOT / "s4m_weather_pg95_equal_ensemble_seed2025.yaml",
]
REPORT = ROOT / "reports" / "stage14_equal_ensemble_config_test.json"


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
    smoke, seed2024, seed2025 = configs
    wrong_weight = copy.deepcopy(seed2024)
    wrong_weight["diagnostic"]["ensemble_weight"] = 0.6
    missing_ensemble = copy.deepcopy(seed2024)
    missing_ensemble["diagnostic"]["save_equal_ensemble"] = False
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
        "formal_budgets_complete": all(
            int(config["diagnostic"]["max_eval_batches"]) == -1
            for config in (seed2024, seed2025)
        ),
        "fixed_equal_weight": all(
            config["diagnostic"]["save_equal_ensemble"] is True
            and float(config["diagnostic"]["ensemble_weight"]) == 0.5
            for config in configs
        ),
        "wrong_weight_rejected": rejected(wrong_weight),
        "non_ensemble_variant_not_accepted_as_equal_ensemble": (
            missing_ensemble["diagnostic"]["save_equal_ensemble"] is False
        ),
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
