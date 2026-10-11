#!/usr/bin/env python3
"""Static configuration and fairness checks for the Step-9 BMSA screen."""

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
SMOKE = ROOT / "configs" / "mipp" / "s4m_weather_mipp_bmsa50_seed2023_smoke.yaml"
FORMAL = ROOT / "configs" / "mipp" / "s4m_weather_mipp_bmsa50_seed2023.yaml"
REPORT = ROOT / "reports" / "stage9_bmsa_config_test.json"


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


def main() -> int:
    pg95 = load(PG95)
    smoke = load(SMOKE)
    formal = load(FORMAL)
    for config in (pg95, smoke, formal):
        validate_experiment_config(config)

    invalid_budget = copy.deepcopy(formal)
    invalid_budget["official_args"]["mipp_write_budget"] = 49
    invalid_budget_rejected = False
    try:
        validate_experiment_config(invalid_budget)
    except ValueError:
        invalid_budget_rejected = True

    pg_args = pg95["official_args"]
    bmsa_args = formal["official_args"]
    all_keys = sorted(set(pg_args) | set(bmsa_args))
    differences = {
        key: {"pg95": pg_args.get(key, "<missing>"), "bmsa50": bmsa_args.get(key, "<missing>")}
        for key in all_keys
        if pg_args.get(key, "<missing>") != bmsa_args.get(key, "<missing>")
    }
    expected_differences = {
        "model_id",
        "des",
        "mipp_write_gate_mode",
        "mipp_write_budget",
    }
    data_assets_exist = all(
        (ROOT / formal["data"][key]).is_file()
        for key in ("missing_csv", "complete_csv", "mask_asset", "mask_metadata", "source_manifest")
    )
    checks = {
        "pg95_config_still_valid": True,
        "smoke_config_valid": True,
        "formal_config_valid": True,
        "invalid_budget_rejected": invalid_budget_rejected,
        "only_predeclared_pg95_to_bmsa_args_differ": set(differences) == expected_differences,
        "formal_data_assets_match_pg95": formal["data"] == pg95["data"],
        "smoke_and_formal_use_same_data_assets": smoke["data"] == formal["data"],
        "budget_matches_existing_M": int(bmsa_args["mipp_write_budget"]) == int(bmsa_args["M"]),
        "fusr_not_enabled": not bool(bmsa_args.get("use_fusr", False)),
        "data_assets_exist": data_assets_exist,
        "formal_budget_not_reduced": (
            int(bmsa_args["batch_size"]) == 16
            and int(bmsa_args["train_epochs"]) == 15
            and int(bmsa_args["max_train_batches"]) == -1
            and int(bmsa_args["max_eval_batches"]) == -1
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "pg95_to_bmsa_official_arg_differences": differences,
        "configs": [str(PG95), str(SMOKE), str(FORMAL)],
    }
    atomic_json(REPORT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
