#!/usr/bin/env python3
"""Static configuration and fairness gates for the Step-13 CFR screen."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_s4m_experiment import validate_experiment_config  # noqa: E402


PG95 = ROOT / "configs" / "mipp" / "s4m_weather_mipp_pair_gate_only_t095_seed2023.yaml"
PG95_REGRESSION_SMOKE = ROOT / "configs" / "stage13_cfr" / "s4m_weather_pg95_seed2023_regression_smoke.yaml"
SMOKE = ROOT / "configs" / "stage13_cfr" / "s4m_weather_pg95_cfr_seed2023_smoke.yaml"
FORMAL = ROOT / "configs" / "stage13_cfr" / "s4m_weather_pg95_cfr_seed2023.yaml"
REPORT = ROOT / "reports" / "stage13_cfr_config_test.json"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def rejected(value: dict) -> bool:
    try:
        validate_experiment_config(value)
    except (TypeError, ValueError):
        return True
    return False


def main() -> int:
    pg95 = load(PG95)
    pg95_regression_smoke = load(PG95_REGRESSION_SMOKE)
    smoke = load(SMOKE)
    formal = load(FORMAL)
    for config in (pg95, pg95_regression_smoke, smoke, formal):
        validate_experiment_config(config)

    pg_args = pg95["official_args"]
    cfr_args = formal["official_args"]
    all_keys = sorted(set(pg_args) | set(cfr_args))
    differences = {
        key: {"pg95": pg_args.get(key, "<missing>"), "cfr": cfr_args.get(key, "<missing>")}
        for key in all_keys
        if pg_args.get(key, "<missing>") != cfr_args.get(key, "<missing>")
    }
    expected_differences = {"model_id", "des", "use_cfr"}

    false_identity = copy.deepcopy(formal)
    false_identity["official_args"]["use_cfr"] = False
    no_mipp = copy.deepcopy(formal)
    no_mipp["official_args"]["use_mipp"] = False
    cfr_with_fusr = copy.deepcopy(formal)
    cfr_with_fusr["official_args"]["use_fusr"] = True

    smoke_args = smoke["official_args"]
    formal_identity_exclusions = {
        "model_id", "des", "train_epochs", "num_workers",
        "max_train_batches", "max_eval_batches",
    }
    smoke_formal_differences = {
        key
        for key in set(smoke_args) | set(cfr_args)
        if smoke_args.get(key, "<missing>") != cfr_args.get(key, "<missing>")
    }
    checks = {
        "historical_pg95_config_still_valid": True,
        "pg95_default_off_regression_smoke_valid": True,
        "smoke_config_valid": True,
        "formal_config_valid": True,
        "only_identity_and_cfr_switch_differ_from_pg95": (
            set(differences) == expected_differences
        ),
        "formal_data_assets_exactly_match_pg95": formal["data"] == pg95["data"],
        "smoke_data_assets_exactly_match_formal": smoke["data"] == formal["data"],
        "pg95_regression_smoke_data_assets_match_formal": (
            pg95_regression_smoke["data"] == formal["data"]
        ),
        "pg95_regression_smoke_keeps_cfr_disabled": (
            pg95_regression_smoke["official_args"].get("use_cfr", False) is False
            and "cfr" not in pg95_regression_smoke["experiment"]["model"].lower()
        ),
        "formal_budget_not_reduced": (
            int(cfr_args["batch_size"]) == 16
            and int(cfr_args["train_epochs"]) == 15
            and int(cfr_args["max_train_batches"]) == -1
            and int(cfr_args["max_eval_batches"]) == -1
        ),
        "smoke_changes_only_engineering_budget_and_identity": (
            smoke_formal_differences == formal_identity_exclusions
        ),
        "cfr_is_single_new_scientific_switch": (
            cfr_args["use_cfr"] is True
            and cfr_args.get("use_fusr", False) is False
            and cfr_args["use_mipp"] is True
        ),
        "false_cfr_identity_rejected": rejected(false_identity),
        "cfr_without_mipp_rejected": rejected(no_mipp),
        "cfr_with_fusr_rejected": rejected(cfr_with_fusr),
        "all_data_assets_exist": all(
            (ROOT / formal["data"][key]).is_file()
            for key in (
                "missing_csv", "complete_csv", "mask_asset",
                "mask_metadata", "source_manifest",
            )
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "pg95_to_cfr_official_arg_differences": differences,
        "smoke_to_formal_official_arg_differences": sorted(smoke_formal_differences),
        "configs": [str(PG95), str(PG95_REGRESSION_SMOKE), str(SMOKE), str(FORMAL)],
    }
    atomic_json(REPORT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
