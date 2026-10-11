#!/usr/bin/env python3
"""Static fairness and failure tests for the frozen stage12 cache configs."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

from run_stage12_validation_cache import (
    EXPECTED_METHOD_MODES,
    PROJECT_ROOT,
    resolve_project_path,
    validate_config,
)


CONFIG_ROOT = PROJECT_ROOT / "configs" / "stage12_validation"
REPORT = PROJECT_ROOT / "reports" / "stage12_validation_config_test.json"
SEEDS = (2024, 2025)
METHODS = ("pg95", "mrba50")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    configs: dict[tuple[str, int, str], dict[str, Any]] = {}
    checks: dict[str, bool] = {}
    for method in METHODS:
        for seed in SEEDS:
            variants = ("formal", "smoke") if seed == 2024 else ("formal",)
            for variant in variants:
                suffix = "_smoke" if variant == "smoke" else ""
                path = CONFIG_ROOT / f"s4m_weather_{method}_seed{seed}{suffix}.yaml"
                config = load(path)
                source = load(resolve_project_path(config["source"]["resolved_config"]))
                validate_config(config, source)
                configs[(method, seed, variant)] = config
                checks[f"{method}_seed{seed}_{variant}_all_assets_exist"] = all(
                    resolve_project_path(value).is_file()
                    for section, key in (
                        ("source", "resolved_config"),
                        ("source", "metrics"),
                        ("source", "checkpoint"),
                        ("data", "missing_csv"),
                        ("data", "complete_csv"),
                        ("data", "mask_asset"),
                    )
                    for value in (config[section][key],)
                )
                effective_mode = source["official_args"].get(
                    "mipp_write_gate_mode", "absolute"
                )
                checks[f"{method}_seed{seed}_{variant}_mode"] = (
                    effective_mode == EXPECTED_METHOD_MODES[method]
                )
                checks[f"{method}_seed{seed}_{variant}_validation_only"] = (
                    config["diagnostic"]["split"] == "val"
                    and "test" not in config["experiment"]["domain"]
                )

    for seed in SEEDS:
        pg = configs[("pg95", seed, "formal")]
        mr = configs[("mrba50", seed, "formal")]
        checks[f"seed{seed}_paired_data_identical"] = pg["data"] == mr["data"]
        checks[f"seed{seed}_paired_budget_full_validation"] = (
            int(pg["diagnostic"]["max_eval_batches"]) == -1
            and int(mr["diagnostic"]["max_eval_batches"]) == -1
        )
        checks[f"seed{seed}_paired_seed"] = (
            int(pg["experiment"]["seed"]) == seed
            and int(mr["experiment"]["seed"]) == seed
        )

    for method in METHODS:
        smoke = configs[(method, 2024, "smoke")]
        formal = configs[(method, 2024, "formal")]
        smoke_copy = copy.deepcopy(smoke)
        formal_copy = copy.deepcopy(formal)
        smoke_copy["experiment"].pop("smoke_test")
        formal_copy["experiment"].pop("smoke_test")
        smoke_copy["experiment"].pop("purpose")
        formal_copy["experiment"].pop("purpose")
        smoke_copy["diagnostic"].pop("max_eval_batches")
        formal_copy["diagnostic"].pop("max_eval_batches")
        checks[f"{method}_smoke_formal_identity"] = smoke_copy == formal_copy
        checks[f"{method}_smoke_is_two_batches"] = (
            int(smoke["diagnostic"]["max_eval_batches"]) == 2
        )

    invalid_split = copy.deepcopy(configs[("pg95", 2024, "formal")])
    invalid_split["diagnostic"]["split"] = "test"
    source = load(resolve_project_path(invalid_split["source"]["resolved_config"]))
    try:
        validate_config(invalid_split, source)
        checks["invalid_test_split_rejected"] = False
    except ValueError:
        checks["invalid_test_split_rejected"] = True

    invalid_mode = copy.deepcopy(
        load(resolve_project_path(configs[("mrba50", 2024, "formal")]["source"]["resolved_config"]))
    )
    invalid_mode["official_args"]["mipp_write_gate_mode"] = "budget_topk"
    try:
        validate_config(configs[("mrba50", 2024, "formal")], invalid_mode)
        checks["invalid_mrba_mode_rejected"] = False
    except ValueError:
        checks["invalid_mrba_mode_rejected"] = True

    report = {
        "schema_version": 1,
        "step": 12,
        "artifact_status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "config_count": len(configs),
    }
    atomic_json(REPORT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"REPORT={REPORT}")
    return 0 if report["artifact_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
