#!/usr/bin/env python3
"""Static asset and protocol gates for Stage-14 uncertainty caches."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import yaml

from run_stage14_uncertainty_cache import resolve_path, validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    ROOT / "configs" / "stage14_uncertainty" / "s4m_weather_pg95_seed2024_smoke.yaml",
    ROOT / "configs" / "stage14_uncertainty" / "s4m_weather_pg95_seed2024.yaml",
    ROOT / "configs" / "stage14_uncertainty" / "s4m_weather_pg95_seed2025.yaml",
]
REPORT = ROOT / "reports" / "stage14_uncertainty_config_test.json"


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
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    configs = [load(path) for path in CONFIGS]
    for config in configs:
        validate_config(config, source(config))
    smoke, seed2024, seed2025 = configs
    negative_split = copy.deepcopy(seed2024)
    negative_split["diagnostic"]["split"] = "test"
    negative_rate = copy.deepcopy(seed2024)
    negative_rate["diagnostic"]["nested_target_rate"] = 0.20
    negative_seed = copy.deepcopy(seed2024)
    negative_seed["diagnostic"]["nested_seed"] = 7
    negative_method = copy.deepcopy(seed2024)
    negative_method["experiment"]["method"] = "mrba50"
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
        "smoke_and_formal_seed2024_share_assets": (
            smoke["source"] == seed2024["source"]
            and smoke["data"] == seed2024["data"]
        ),
        "smoke_only_reduces_eval_batches": (
            smoke["diagnostic"] == {
                **seed2024["diagnostic"], "max_eval_batches": 2
            }
        ),
        "formal_budgets_are_complete": all(
            int(config["diagnostic"]["max_eval_batches"]) == -1
            for config in (seed2024, seed2025)
        ),
        "nested_protocol_frozen": all(
            float(config["diagnostic"]["nested_target_rate"]) == 0.15
            and int(config["diagnostic"]["nested_block_length"]) == 5
            and int(config["diagnostic"]["nested_seed"]) == int(config["experiment"]["seed"])
            for config in configs
        ),
        "test_split_rejected": rejected(negative_split),
        "changed_rate_rejected": rejected(negative_rate),
        "wrong_nested_seed_rejected": rejected(negative_seed),
        "non_pg95_source_rejected": rejected(negative_method),
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
