#!/usr/bin/env python3
"""Static assets and protocol gates for Stage-16 CNM-SE structure ablations."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import yaml

from run_stage14_uncertainty_cache import resolve_path, validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "stage16_cnm_se_ablation"
CONFIGS = [
    CONFIG_ROOT / f"s4m_weather_cnm_se_l{block}_seed2024_smoke.yaml"
    for block in (1, 20)
] + [
    CONFIG_ROOT / f"s4m_weather_cnm_se_l{block}_seed{seed}.yaml"
    for block in (1, 20)
    for seed in (2023, 2024, 2025)
]
REPORT = ROOT / "reports" / "stage16_cnm_se_ablation_config_test.json"


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
    smoke_configs = [config for config in configs if config["experiment"]["smoke_test"]]
    formal_configs = [
        config for config in configs if not config["experiment"]["smoke_test"]
    ]
    wrong_main_length = copy.deepcopy(formal_configs[0])
    wrong_main_length["diagnostic"]["nested_block_length"] = 5
    illegal_length = copy.deepcopy(formal_configs[0])
    illegal_length["diagnostic"]["nested_block_length"] = 7
    wrong_rate = copy.deepcopy(formal_configs[0])
    wrong_rate["diagnostic"]["nested_target_rate"] = 0.2
    wrong_weight = copy.deepcopy(formal_configs[0])
    wrong_weight["diagnostic"]["ensemble_weight"] = 0.4
    validation_split = copy.deepcopy(formal_configs[0])
    validation_split["diagnostic"]["split"] = "val"
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
        "ablation_identity_frozen": all(
            config["diagnostic"]["variant"] == "stage16_ablation"
            and config["diagnostic"]["split"] == "test"
            and int(config["diagnostic"]["nested_block_length"]) in (1, 20)
            and float(config["diagnostic"]["nested_target_rate"]) == 0.15
            and float(config["diagnostic"]["ensemble_weight"]) == 0.5
            for config in configs
        ),
        "two_independent_smokes": (
            len(smoke_configs) == 2
            and {int(c["diagnostic"]["nested_block_length"]) for c in smoke_configs}
            == {1, 20}
            and all(int(c["diagnostic"]["max_eval_batches"]) == 2 for c in smoke_configs)
        ),
        "six_complete_formals": (
            len(formal_configs) == 6
            and all(int(c["diagnostic"]["max_eval_batches"]) == -1 for c in formal_configs)
            and {
                (
                    int(c["diagnostic"]["nested_block_length"]),
                    int(c["experiment"]["seed"]),
                )
                for c in formal_configs
            }
            == {
                (block, seed)
                for block in (1, 20)
                for seed in (2023, 2024, 2025)
            }
        ),
        "block_length_5_rejected_for_ablation": rejected(wrong_main_length),
        "unregistered_block_length_rejected": rejected(illegal_length),
        "wrong_rate_rejected": rejected(wrong_rate),
        "wrong_weight_rejected": rejected(wrong_weight),
        "validation_split_rejected": rejected(validation_split),
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
