#!/usr/bin/env python3
"""Static, fairness, and negative tests for Step-22 innovation configs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_s4m_experiment import (  # noqa: E402
    classify_run_failure,
    validate_experiment_config,
)


CONFIG_DIR = ROOT / "configs" / "stage22_innovation"
CONFIGS = {
    "mrra_smoke": CONFIG_DIR / "mrra_pg95_electricity_seed2023_smoke.yaml",
    "mrra_formal": CONFIG_DIR / "mrra_pg95_electricity_seed2023.yaml",
    "radr_smoke": CONFIG_DIR / "radr_pg95_electricity_seed2023_smoke.yaml",
    "radr_formal": CONFIG_DIR / "radr_pg95_electricity_seed2023.yaml",
    "mrra_v2_stability_smoke": CONFIG_DIR / "mrra_v2_pg95_electricity_seed2023_stability_smoke.yaml",
    "mrra_v2_formal": CONFIG_DIR / "mrra_v2_pg95_electricity_seed2023.yaml",
    "mrra_v3_stability_smoke": CONFIG_DIR / "mrra_v3_pg95_electricity_seed2023_stability_smoke.yaml",
    "mrra_v3_formal": CONFIG_DIR / "mrra_v3_pg95_electricity_seed2023.yaml",
}
BASE_CONFIGS = {
    "smoke": ROOT / "configs" / "stage21_electricity" / "pg95_electricity_seed2023_smoke.yaml",
    "formal": ROOT / "configs" / "stage21_electricity" / "pg95_electricity_seed2023.yaml",
}
REPORT = ROOT / "reports" / "stage22_innovation_config_test.json"


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def rejected(config: dict) -> bool:
    try:
        validate_experiment_config(config)
    except (TypeError, ValueError):
        return True
    return False


def normalized_args(config: dict) -> dict:
    result = copy.deepcopy(config["official_args"])
    defaults = {
        "loss": "MSE",
        "use_mrra": False,
        "mrra_version": "v1",
        "mrra_rank": 32,
        "mrra_kernel_size": 5,
        "mrra_reconstruction_weight": 0.1,
        "mrra_lr_multiplier": 1.0,
        "use_radr": False,
    }
    for key, value in defaults.items():
        result.setdefault(key, value)
    for key in (
        "model_id", "des", "use_mrra", "use_radr", "mrra_version", "mrra_rank",
        "mrra_kernel_size", "mrra_reconstruction_weight",
        "mrra_lr_multiplier",
    ):
        result.pop(key, None)
    return result


def main() -> int:
    configs = {name: load(path) for name, path in CONFIGS.items()}
    bases = {name: load(path) for name, path in BASE_CONFIGS.items()}
    checks: dict[str, bool] = {}
    for name, config in configs.items():
        validate_experiment_config(config)
        args = config["official_args"]
        checks[f"{name}_assets_exist"] = all(
            (ROOT / config["data"][key]).is_file()
            for key in ("missing_csv", "complete_csv", "mask_asset", "mask_metadata")
        )
        checks[f"{name}_validation_only"] = (
            args["post_train_evaluation"] == "validation"
            and config["experiment"]["dataset"] == "electricity"
            and int(config["experiment"]["seed"]) == 2023
        )
        checks[f"{name}_single_candidate"] = (
            int(bool(args["use_mrra"])) + int(bool(args["use_radr"])) == 1
        )
        if name not in {"mrra_v2_stability_smoke", "mrra_v3_stability_smoke"}:
            budget = "smoke" if config["experiment"]["smoke_test"] else "formal"
            checks[f"{name}_same_base_budget"] = (
                normalized_args(config) == normalized_args(bases[budget])
            )

    for candidate in ("mrra", "radr"):
        smoke = configs[f"{candidate}_smoke"]
        formal = configs[f"{candidate}_formal"]
        checks[f"{candidate}_smoke_budget"] = (
            smoke["official_args"]["train_epochs"] == 2
            and smoke["official_args"]["max_train_batches"] == 2
            and smoke["official_args"]["max_eval_batches"] == 3
        )
        checks[f"{candidate}_formal_full_budget"] = (
            formal["official_args"]["train_epochs"] == 15
            and formal["official_args"]["max_train_batches"] == -1
            and formal["official_args"]["max_eval_batches"] == -1
        )

    v2_smoke = configs["mrra_v2_stability_smoke"]
    v2_formal = configs["mrra_v2_formal"]
    checks["mrra_v2_stability_smoke_covers_v1_failure_window"] = (
        v2_smoke["official_args"]["train_epochs"] == 3
        and v2_smoke["official_args"]["max_train_batches"] == 800
        and v2_smoke["official_args"]["max_eval_batches"] == 3
        and v2_smoke["experiment"]["required_memory_bank_warmups"] == 3
    )
    checks["mrra_v2_stability_smoke_is_not_formal"] = (
        v2_smoke["experiment"]["smoke_test"] is True
    )
    v2_smoke_core = normalized_args(v2_smoke)
    v2_formal_core = normalized_args(v2_formal)
    for key in ("train_epochs", "max_train_batches", "max_eval_batches"):
        v2_smoke_core.pop(key, None)
        v2_formal_core.pop(key, None)
    checks["mrra_v2_smoke_only_changes_smoke_budget"] = (
        v2_smoke_core == v2_formal_core
    )
    checks["mrra_v2_formal_full_budget"] = (
        v2_formal["official_args"]["train_epochs"] == 15
        and v2_formal["official_args"]["max_train_batches"] == -1
        and v2_formal["official_args"]["max_eval_batches"] == -1
    )
    checks["mrra_v1_behavior_remains_default"] = all(
        config["official_args"].get("mrra_version", "v1") == "v1"
        for name, config in configs.items()
        if name.startswith("mrra_")
        and not name.startswith(("mrra_v2", "mrra_v3"))
    )
    checks["mrra_v2_version_is_explicit"] = all(
        config["official_args"].get("mrra_version") == "v2_normalized_bounded"
        for name, config in configs.items()
        if name.startswith("mrra_v2")
    )

    v3_smoke = configs["mrra_v3_stability_smoke"]
    v3_formal = configs["mrra_v3_formal"]
    checks["mrra_v3_stability_smoke_covers_v1_failure_window"] = (
        v3_smoke["official_args"]["train_epochs"] == 3
        and v3_smoke["official_args"]["max_train_batches"] == 800
        and v3_smoke["official_args"]["max_eval_batches"] == 3
        and v3_smoke["experiment"]["required_memory_bank_warmups"] == 3
    )
    checks["mrra_v3_stability_smoke_is_not_formal"] = (
        v3_smoke["experiment"]["smoke_test"] is True
    )
    v3_smoke_core = normalized_args(v3_smoke)
    v3_formal_core = normalized_args(v3_formal)
    for key in ("train_epochs", "max_train_batches", "max_eval_batches"):
        v3_smoke_core.pop(key, None)
        v3_formal_core.pop(key, None)
    checks["mrra_v3_smoke_only_changes_smoke_budget"] = (
        v3_smoke_core == v3_formal_core
    )
    checks["mrra_v3_formal_full_budget"] = (
        v3_formal["official_args"]["train_epochs"] == 15
        and v3_formal["official_args"]["max_train_batches"] == -1
        and v3_formal["official_args"]["max_eval_batches"] == -1
    )
    checks["mrra_v3_version_is_explicit"] = all(
        config["official_args"].get("mrra_version")
        == "v3_gradient_isolated_two_timescale"
        for name, config in configs.items()
        if name.startswith("mrra_v3")
    )
    checks["mrra_v3_lr_multiplier_is_explicit"] = all(
        float(config["official_args"].get("mrra_lr_multiplier", -1.0)) == 0.1
        for name, config in configs.items()
        if name.startswith("mrra_v3")
    )

    bad = copy.deepcopy(configs["mrra_formal"])
    bad["official_args"]["use_mrra"] = False
    checks["mrra_identity_without_switch_rejected"] = rejected(bad)
    bad = copy.deepcopy(configs["radr_formal"])
    bad["official_args"]["use_radr"] = False
    checks["radr_identity_without_switch_rejected"] = rejected(bad)
    bad = copy.deepcopy(configs["mrra_formal"])
    bad["official_args"]["mrra_rank"] = 0
    checks["invalid_mrra_rank_rejected"] = rejected(bad)
    bad = copy.deepcopy(configs["mrra_formal"])
    bad["official_args"]["mrra_reconstruction_weight"] = -0.1
    checks["negative_reconstruction_weight_rejected"] = rejected(bad)
    bad = copy.deepcopy(configs["mrra_v2_formal"])
    bad["official_args"]["mrra_version"] = "v1"
    checks["mrra_v2_identity_with_v1_rejected"] = rejected(bad)
    bad = copy.deepcopy(configs["mrra_v2_formal"])
    bad["official_args"]["mrra_version"] = "unregistered"
    checks["invalid_mrra_version_rejected"] = rejected(bad)
    bad = copy.deepcopy(configs["mrra_v3_formal"])
    bad["official_args"]["mrra_version"] = "v2_normalized_bounded"
    checks["mrra_v3_identity_with_v2_rejected"] = rejected(bad)
    bad = copy.deepcopy(configs["mrra_v3_formal"])
    bad["official_args"]["mrra_lr_multiplier"] = 1.0
    checks["mrra_v3_wrong_lr_multiplier_rejected"] = rejected(bad)
    bad = copy.deepcopy(configs["mrra_v3_formal"])
    bad["official_args"].pop("mrra_lr_multiplier")
    checks["mrra_v3_missing_lr_multiplier_rejected"] = rejected(bad)
    bad = copy.deepcopy(configs["mrra_v3_formal"])
    bad["official_args"]["mrra_lr_multiplier"] = 0.0
    checks["nonpositive_mrra_lr_multiplier_rejected"] = rejected(bad)
    bad = copy.deepcopy(configs["radr_formal"])
    bad["official_args"]["use_fusr"] = True
    checks["radr_fusr_mixture_rejected"] = rejected(bad)
    checks["nonfinite_failure_is_classified_scientifically"] = (
        classify_run_failure(
            "FloatingPointError: NAN_OR_DIVERGENCE: non-finite training objective"
        ) == "NAN_OR_DIVERGENCE"
    )
    checks["oom_classification_keeps_priority"] = (
        classify_run_failure("CUDA out of memory; non-finite loss") == "OOM"
    )

    payload = {
        "step": 22,
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "config_count": len(configs),
        "checks": checks,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(REPORT)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
