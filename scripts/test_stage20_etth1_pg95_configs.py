#!/usr/bin/env python3
"""Static and negative checks for the frozen Stage-20 configs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from run_s4m_experiment import validate_experiment_config
from run_stage14_uncertainty_cache import validate_config as validate_eval_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "stage20_etth1_pg95"
BASELINE = (
    ROOT / "configs" / "baseline"
    / "s4m_etth1_point_r006_paper_adam_seed2023.yaml"
)
REPORT = ROOT / "reports" / "stage20_etth1_pg95_config_test.json"


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def expect_rejected(callback, label: str) -> None:
    try:
        callback()
    except (TypeError, ValueError):
        return
    raise AssertionError(f"negative case was accepted: {label}")


def main() -> int:
    paths = {
        "train_smoke": CONFIG_ROOT / "s4m_etth1_pg95_seed2023_smoke.yaml",
        "train_formal": CONFIG_ROOT / "s4m_etth1_pg95_seed2023.yaml",
        "eval_smoke": (
            CONFIG_ROOT
            / "s4m_etth1_pg95_multiscale_seed2023_smoke.yaml"
        ),
        "eval_formal": (
            CONFIG_ROOT / "s4m_etth1_pg95_multiscale_seed2023.yaml"
        ),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline = load(BASELINE)
    train_smoke = load(paths["train_smoke"])
    train_formal = load(paths["train_formal"])
    eval_smoke = load(paths["eval_smoke"])
    eval_formal = load(paths["eval_formal"])
    validate_experiment_config(train_smoke)
    validate_experiment_config(train_formal)
    validate_eval_config(eval_smoke, train_formal)
    validate_eval_config(eval_formal, train_formal)

    for key in ("missing_csv", "complete_csv", "mask_asset", "mask_metadata"):
        if train_formal["data"][key] != baseline["data"][key]:
            raise AssertionError(f"training data drift: {key}")
        if not (ROOT / train_formal["data"][key]).is_file():
            raise FileNotFoundError(ROOT / train_formal["data"][key])
    if train_formal["data"]["split"] != baseline["data"]["split"]:
        raise AssertionError("split drift")

    allowed_official_changes = {
        "model_id", "des",
        "use_mipp", "use_content_mask_factorization",
        "use_mask_intervention_pair", "use_stability_write_gate",
        "mipp_write_gate_mode", "mipp_inv_weight", "mipp_var_weight",
        "mipp_cov_weight", "mipp_dec_weight", "mipp_mask_weight",
        "mipp_variance_target", "mipp_stability_threshold",
        "mipp_pair_rate_min", "mipp_pair_rate_max",
        "mipp_pair_block_lengths", "use_fusr", "use_cfr",
    }
    baseline_official = baseline["official_args"]
    formal_official = train_formal["official_args"]
    shared_keys = set(baseline_official).intersection(formal_official)
    unexpected_changes = {
        key: {
            "baseline": baseline_official[key],
            "pg95": formal_official[key],
        }
        for key in shared_keys - allowed_official_changes
        if baseline_official[key] != formal_official[key]
    }
    if unexpected_changes:
        raise AssertionError(
            f"unexpected baseline-to-PG95 changes: {unexpected_changes}"
        )
    frozen_pg95 = {
        "use_mipp": True,
        "use_content_mask_factorization": False,
        "use_mask_intervention_pair": True,
        "use_stability_write_gate": True,
        "mipp_write_gate_mode": "absolute",
        "mipp_inv_weight": 0.05,
        "mipp_var_weight": 0.0,
        "mipp_cov_weight": 0.0,
        "mipp_dec_weight": 0.0,
        "mipp_mask_weight": 0.0,
        "mipp_stability_threshold": 0.95,
        "mipp_pair_rate_min": 0.15,
        "mipp_pair_rate_max": 0.50,
        "mipp_pair_block_lengths": [1, 5, 20],
        "use_fusr": False,
        "use_cfr": False,
    }
    actual_pg95 = {
        key: formal_official.get(key) for key in frozen_pg95
    }
    if actual_pg95 != frozen_pg95:
        raise AssertionError(
            {"expected_pg95": frozen_pg95, "actual_pg95": actual_pg95}
        )
    if not (
        formal_official["batch_size"] == 16
        and formal_official["learning_rate"] == 0.005
        and formal_official["optimizer"] == "adam"
        and formal_official["train_epochs"] == 15
        and formal_official["patience"] == 3
        and formal_official["max_train_batches"] == -1
        and formal_official["max_eval_batches"] == -1
        and train_formal["experiment"]["smoke_test"] is False
    ):
        raise AssertionError("formal training budget changed")
    if not (
        train_smoke["experiment"]["smoke_test"] is True
        and train_smoke["official_args"]["train_epochs"] == 2
        and train_smoke["official_args"]["max_train_batches"] == 2
        and train_smoke["official_args"]["max_eval_batches"] == 1
        and train_smoke["experiment"]["required_memory_bank_warmups"] == 2
    ):
        raise AssertionError("training smoke lifecycle is incomplete")
    for config in (eval_smoke, eval_formal):
        if config["source"] != {"job_id": "etth1_pg95_seed2023_formal"}:
            raise AssertionError("dynamic source job drift")
        diagnostic = config["diagnostic"]
        if not (
            diagnostic["variant"] == "stage20_pg95_external_test"
            and diagnostic["split"] == "test"
            and diagnostic["target_rate"] == 0.15
            and diagnostic["block_lengths"] == [1, 5, 20]
            and diagnostic["multiscale_weights"]
            == {"original": 0.5, "l5": 0.25, "l20": 0.25}
        ):
            raise AssertionError("frozen evaluation protocol changed")
    if not (
        eval_smoke["diagnostic"]["max_eval_batches"] == 3
        and eval_formal["diagnostic"]["max_eval_batches"] == -1
    ):
        raise AssertionError("smoke/formal evaluation budgets changed")

    bad_eval = copy.deepcopy(eval_formal)
    bad_eval["diagnostic"]["multiscale_weights"]["l20"] = 0.30
    expect_rejected(
        lambda: validate_eval_config(bad_eval, train_formal),
        "changed_multiscale_weight",
    )
    bad_eval = copy.deepcopy(eval_formal)
    bad_eval["experiment"]["method"] = "s4m_baseline"
    expect_rejected(
        lambda: validate_eval_config(bad_eval, train_formal),
        "wrong_stage20_method",
    )
    bad_source = copy.deepcopy(train_formal)
    bad_source["official_args"]["use_mipp"] = False
    expect_rejected(
        lambda: validate_eval_config(eval_formal, bad_source),
        "non_mipp_source",
    )
    bad_source = copy.deepcopy(train_formal)
    bad_source["official_args"]["mipp_inv_weight"] = 0.06
    expect_rejected(
        lambda: validate_eval_config(eval_formal, bad_source),
        "changed_pg95_pair_weight",
    )

    payload = {
        "status": "PASS",
        "configs": {key: str(path) for key, path in paths.items()},
        "unexpected_baseline_changes": unexpected_changes,
        "frozen_pg95": frozen_pg95,
        "negative_cases": 4,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
