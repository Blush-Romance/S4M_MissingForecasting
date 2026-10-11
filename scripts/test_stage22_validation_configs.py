#!/usr/bin/env python3
"""Static and negative tests for the Stage-22 Electricity validation caches."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_stage12_validation_cache import resolve_project_path, validate_config


CONFIG_DIR = PROJECT_ROOT / "configs" / "stage22_validation"
CONFIGS = {
    "s4m_smoke": CONFIG_DIR / "s4m_electricity_seed2023_smoke.yaml",
    "pg95_smoke": CONFIG_DIR / "pg95_electricity_seed2023_smoke.yaml",
    "s4m_formal": CONFIG_DIR / "s4m_electricity_seed2023.yaml",
    "pg95_formal": CONFIG_DIR / "pg95_electricity_seed2023.yaml",
    "mrra_v3_smoke": CONFIG_DIR / "mrra_v3_electricity_seed2023_smoke.yaml",
    "mrra_v3_formal": CONFIG_DIR / "mrra_v3_electricity_seed2023.yaml",
}
REPORT = PROJECT_ROOT / "reports" / "stage22_validation_config_test.json"


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def rejected(config: dict, source: dict) -> bool:
    try:
        validate_config(config, source)
    except ValueError:
        return True
    return False


def main() -> int:
    loaded = {name: load(path) for name, path in CONFIGS.items()}
    sources = {
        name: load(resolve_project_path(config["source"]["resolved_config"]))
        for name, config in loaded.items()
    }
    checks: dict[str, bool] = {}
    for name, config in loaded.items():
        validate_config(config, sources[name])
        required = [
            CONFIGS[name],
            resolve_project_path(config["source"]["resolved_config"]),
            resolve_project_path(config["source"]["metrics"]),
            resolve_project_path(config["source"]["checkpoint"]),
            resolve_project_path(config["data"]["missing_csv"]),
            resolve_project_path(config["data"]["complete_csv"]),
            resolve_project_path(config["data"]["mask_asset"]),
        ]
        checks[f"{name}_all_assets_exist"] = all(path.is_file() for path in required)
        checks[f"{name}_validation_only"] = (
            config["diagnostic"]["stage_identity"] == "stage22"
            and config["diagnostic"]["split"] == "val"
        )
        checks[f"{name}_frozen_probe"] = (
            float(config["diagnostic"]["nested_target_rate"]) == 0.15
            and int(config["diagnostic"]["nested_block_length"]) == 5
            and int(config["diagnostic"]["nested_seed"]) == 2023
        )

    for method in ("s4m", "pg95", "mrra_v3"):
        smoke = loaded[f"{method}_smoke"]
        formal = loaded[f"{method}_formal"]
        checks[f"{method}_smoke_two_batches"] = int(
            smoke["diagnostic"]["max_eval_batches"]
        ) == 2
        checks[f"{method}_formal_full_validation"] = int(
            formal["diagnostic"]["max_eval_batches"]
        ) == -1
        smoke_copy = copy.deepcopy(smoke)
        formal_copy = copy.deepcopy(formal)
        for item in (smoke_copy, formal_copy):
            item["experiment"].pop("smoke_test")
            item["experiment"].pop("purpose")
            item["diagnostic"].pop("max_eval_batches")
        checks[f"{method}_smoke_formal_identity"] = smoke_copy == formal_copy

    s4m = loaded["s4m_formal"]
    pg95 = loaded["pg95_formal"]
    mrra_v3 = loaded["mrra_v3_formal"]
    checks["paired_data_identical"] = (
        s4m["data"] == pg95["data"] == mrra_v3["data"]
    )
    checks["paired_seed_identical"] = (
        int(s4m["experiment"]["seed"]) == int(pg95["experiment"]["seed"]) == 2023
    )
    checks["paired_probe_identical"] = s4m["diagnostic"] == pg95["diagnostic"]
    checks["mrra_v3_probe_identical"] = (
        mrra_v3["diagnostic"] == pg95["diagnostic"]
    )

    bad = copy.deepcopy(pg95)
    bad["diagnostic"]["split"] = "test"
    checks["test_split_rejected"] = rejected(bad, sources["pg95_formal"])
    bad = copy.deepcopy(pg95)
    bad["experiment"]["dataset"] = "weather"
    checks["wrong_dataset_rejected"] = rejected(bad, sources["pg95_formal"])
    bad = copy.deepcopy(pg95)
    bad["experiment"]["seed"] = 2024
    checks["wrong_seed_rejected"] = rejected(bad, sources["pg95_formal"])
    bad = copy.deepcopy(pg95)
    bad["diagnostic"]["nested_block_length"] = 20
    checks["unfrozen_block_length_rejected"] = rejected(
        bad, sources["pg95_formal"]
    )
    bad_source = copy.deepcopy(sources["pg95_formal"])
    bad_source["official_args"]["use_mipp"] = False
    checks["pg95_without_mipp_rejected"] = rejected(pg95, bad_source)
    bad_source = copy.deepcopy(sources["s4m_formal"])
    bad_source["official_args"]["use_mipp"] = True
    checks["s4m_with_mipp_rejected"] = rejected(s4m, bad_source)
    bad_source = copy.deepcopy(sources["mrra_v3_formal"])
    bad_source["official_args"]["use_mrra"] = False
    checks["mrra_v3_without_module_rejected"] = rejected(mrra_v3, bad_source)
    bad_source = copy.deepcopy(sources["mrra_v3_formal"])
    bad_source["official_args"]["mrra_version"] = "v2_normalized_bounded"
    checks["mrra_v3_wrong_version_rejected"] = rejected(mrra_v3, bad_source)
    bad_source = copy.deepcopy(sources["mrra_v3_formal"])
    bad_source["official_args"]["mrra_lr_multiplier"] = 1.0
    checks["mrra_v3_wrong_lr_multiplier_rejected"] = rejected(mrra_v3, bad_source)
    bad_source = copy.deepcopy(sources["mrra_v3_formal"])
    bad_source["official_args"]["use_radr"] = True
    checks["mrra_v3_radr_mixture_rejected"] = rejected(mrra_v3, bad_source)

    payload = {
        "step": 22,
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "config_count": len(loaded),
        "checks": checks,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temp = REPORT.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(REPORT)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"REPORT={REPORT}")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
