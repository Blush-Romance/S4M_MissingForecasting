#!/usr/bin/env python3
"""Static fairness and validation gates for Step-11 MRBA anchor experiments."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from run_s4m_experiment import validate_experiment_config


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2024, 2025)


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def differing_keys(left: dict, right: dict) -> set[str]:
    return {
        key
        for key in set(left).union(right)
        if left.get(key) != right.get(key)
    }


def rejects(config: dict) -> bool:
    try:
        validate_experiment_config(config)
    except (TypeError, ValueError):
        return True
    return False


def main() -> int:
    checks: dict[str, bool] = {}
    records = {}
    for seed in SEEDS:
        for smoke in (True, False):
            suffix = "_smoke" if smoke else ""
            bmsa_path = ROOT / "configs" / "mipp" / f"s4m_weather_mipp_bmsa50_seed{seed}{suffix}.yaml"
            mrba_path = ROOT / "configs" / "mipp" / f"s4m_weather_mipp_mrba50_seed{seed}{suffix}.yaml"
            bmsa = load(bmsa_path)
            mrba = load(mrba_path)
            validate_experiment_config(mrba)
            label = f"seed{seed}_{'smoke' if smoke else 'formal'}"
            experiment_differences = differing_keys(bmsa["experiment"], mrba["experiment"])
            official_differences = differing_keys(bmsa["official_args"], mrba["official_args"])
            checks[f"{label}_same_data"] = bmsa["data"] == mrba["data"]
            checks[f"{label}_same_runtime"] = bmsa["runtime"] == mrba["runtime"]
            checks[f"{label}_only_expected_experiment_identity_changes"] = (
                experiment_differences <= {"model", "protocol_scope", "purpose"}
            )
            checks[f"{label}_only_expected_official_changes"] = official_differences == {
                "model_id",
                "des",
                "mipp_write_gate_mode",
            }
            checks[f"{label}_new_mode_and_budget"] = (
                mrba["official_args"]["mipp_write_gate_mode"]
                == "mask_regime_balanced_rank"
                and int(mrba["official_args"]["mipp_write_budget"])
                == int(mrba["official_args"]["M"])
                == 50
            )
            checks[f"{label}_module_prerequisites"] = all(
                bool(mrba["official_args"][key])
                for key in (
                    "use_mipp",
                    "use_mask_intervention_pair",
                    "use_stability_write_gate",
                )
            )
            for asset in ("missing_csv", "complete_csv", "mask_asset", "mask_metadata"):
                checks[f"{label}_{asset}_exists"] = (ROOT / mrba["data"][asset]).is_file()
            if smoke:
                checks[f"{label}_smoke_is_lifecycle_only"] = (
                    mrba["experiment"]["smoke_test"] is True
                    and int(mrba["official_args"]["train_epochs"]) == 2
                    and int(mrba["official_args"]["max_train_batches"]) == 2
                    and int(mrba["official_args"]["max_eval_batches"]) == 1
                    and int(mrba["experiment"]["required_memory_bank_warmups"]) == 2
                )
            else:
                args = mrba["official_args"]
                checks[f"{label}_formal_budget_not_reduced"] = (
                    mrba["experiment"]["smoke_test"] is False
                    and int(args["batch_size"]) == 16
                    and int(args["train_epochs"]) == 15
                    and int(args["patience"]) == 3
                    and args["optimizer"] == "adam"
                    and int(args["max_train_batches"]) == -1
                    and int(args["max_eval_batches"]) == -1
                )
            records[label] = {
                "bmsa_path": str(bmsa_path),
                "mrba_path": str(mrba_path),
                "experiment_differences": sorted(experiment_differences),
                "official_arg_differences": sorted(official_differences),
            }

    invalid_budget = load(
        ROOT / "configs" / "mipp" / "s4m_weather_mipp_mrba50_seed2024.yaml"
    )
    invalid_budget = copy.deepcopy(invalid_budget)
    invalid_budget["official_args"]["mipp_write_budget"] = 49
    checks["invalid_budget_is_rejected"] = rejects(invalid_budget)

    missing_gate = load(
        ROOT / "configs" / "mipp" / "s4m_weather_mipp_mrba50_seed2024.yaml"
    )
    missing_gate = copy.deepcopy(missing_gate)
    missing_gate["official_args"]["use_stability_write_gate"] = False
    checks["missing_stability_gate_is_rejected"] = rejects(missing_gate)

    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "records": records,
    }
    output = ROOT / "reports" / "stage11_mrba_config_test.json"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
