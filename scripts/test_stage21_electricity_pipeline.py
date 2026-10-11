#!/usr/bin/env python3
"""Static, negative, and handoff checks for the Stage-21 pipeline."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

import yaml

from run_s4m_experiment import validate_experiment_config
from run_stage14_uncertainty_cache import validate_config as validate_eval_config
from analyze_stage21_electricity_validation import audit_method_differences
from run_stage21_electricity_pipeline import materialize_evaluation_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "stage21_electricity"
REPORT = ROOT / "reports" / "stage21_electricity_pipeline_test.json"


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def expect_rejected(callback, label: str) -> None:
    try:
        callback()
    except (KeyError, RuntimeError, TypeError, ValueError):
        return
    raise AssertionError(f"negative case was accepted: {label}")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> int:
    paths = {
        "s4m_smoke": CONFIG_ROOT / "s4m_electricity_seed2023_smoke.yaml",
        "s4m_formal": CONFIG_ROOT / "s4m_electricity_seed2023.yaml",
        "pg95_smoke": CONFIG_ROOT / "pg95_electricity_seed2023_smoke.yaml",
        "pg95_formal": CONFIG_ROOT / "pg95_electricity_seed2023.yaml",
        "validation_smoke": (
            CONFIG_ROOT
            / "pg95_electricity_multiscale_validation_seed2023_smoke.yaml"
        ),
        "validation_formal": (
            CONFIG_ROOT
            / "pg95_electricity_multiscale_validation_seed2023.yaml"
        ),
        "gated_test": (
            CONFIG_ROOT / "pg95_electricity_multiscale_test_seed2023.yaml"
        ),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    configs = {name: load(path) for name, path in paths.items()}
    for name in ("s4m_smoke", "s4m_formal", "pg95_smoke", "pg95_formal"):
        validate_experiment_config(configs[name])
    for name in ("validation_smoke", "validation_formal", "gated_test"):
        validate_eval_config(configs[name], configs["pg95_formal"])

    checks = {
        "validation_variants_use_val": all(
            configs[name]["diagnostic"]["variant"]
            == "stage21_electricity_validation"
            and configs[name]["diagnostic"]["split"] == "val"
            for name in ("validation_smoke", "validation_formal")
        ),
        "test_variant_is_unique_and_gated": (
            configs["gated_test"]["diagnostic"]["variant"]
            == "stage21_electricity_test"
            and configs["gated_test"]["diagnostic"]["split"] == "test"
            and "only_after" in configs["gated_test"]["experiment"]["purpose"]
        ),
        "all_eval_sources_are_pg95_formal": all(
            configs[name]["source"]
            == {"job_id": "electricity_pg95_seed2023_formal"}
            for name in ("validation_smoke", "validation_formal", "gated_test")
        ),
        "frozen_multiscale_protocol": all(
            configs[name]["diagnostic"]["target_rate"] == 0.15
            and configs[name]["diagnostic"]["block_lengths"] == [1, 5, 20]
            and configs[name]["diagnostic"]["multiscale_weights"]
            == {"original": 0.5, "l5": 0.25, "l20": 0.25}
            for name in ("validation_smoke", "validation_formal", "gated_test")
        ),
        "formal_validation_is_complete": (
            configs["validation_formal"]["diagnostic"]["max_eval_batches"] == -1
            and configs["validation_formal"]["diagnostic"][
                "timing_warmup_batches"
            ]
            == 10
        ),
        "formal_test_is_complete_if_authorized": (
            configs["gated_test"]["diagnostic"]["max_eval_batches"] == -1
        ),
    }

    baseline_args = copy.deepcopy(configs["s4m_formal"]["official_args"])
    pg95_args = copy.deepcopy(configs["pg95_formal"]["official_args"])
    for key in ("checkpoints", "metrics_path", "results_root", "test_results_root"):
        baseline_args[key] = f"/runs/s4m/{key}"
        pg95_args[key] = f"/runs/pg95/{key}"
    actual, unexpected, required_present = audit_method_differences(
        baseline_args, pg95_args
    )
    checks["run_identity_paths_are_not_budget_drift"] = (
        not unexpected
        and required_present
        and {
            "checkpoints", "metrics_path", "results_root", "test_results_root"
        }
        <= actual
    )
    drifted_pg95_args = copy.deepcopy(pg95_args)
    drifted_pg95_args["batch_size"] = 8
    _, drifted_unexpected, _ = audit_method_differences(
        baseline_args, drifted_pg95_args
    )
    checks["real_training_budget_drift_is_rejected"] = (
        drifted_unexpected == {"batch_size"}
    )

    bad = copy.deepcopy(configs["validation_formal"])
    bad["diagnostic"]["split"] = "test"
    expect_rejected(
        lambda: validate_eval_config(bad, configs["pg95_formal"]),
        "validation_variant_with_test_split",
    )
    bad = copy.deepcopy(configs["gated_test"])
    bad["diagnostic"]["multiscale_weights"]["l20"] = 0.30
    expect_rejected(
        lambda: validate_eval_config(bad, configs["pg95_formal"]),
        "changed_test_weight",
    )
    bad_source = copy.deepcopy(configs["pg95_formal"])
    bad_source["official_args"]["post_train_evaluation"] = "test"
    expect_rejected(
        lambda: validate_eval_config(
            configs["validation_formal"], bad_source
        ),
        "source_not_validation_only",
    )

    with tempfile.TemporaryDirectory(prefix="stage21_handoff_") as temporary:
        temp_root = Path(temporary)
        source_run = temp_root / "formal_pg95"
        checkpoint = (
            source_run / "checkpoints" / "setting" / "checkpoint.pth"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        (source_run / "resolved_config.yaml").write_text(
            yaml.safe_dump(configs["pg95_formal"]), encoding="utf-8"
        )
        validation_metrics = {
            "checkpoint": str(checkpoint),
            "evaluation_split": "validation",
            "post_train_evaluation": "validation",
            "test_split_accessed": False,
            "chronological_loader": True,
        }
        write_json(source_run / "metrics.json", validation_metrics)
        write_json(source_run / "status.json", {"state": "SUCCESS"})
        state = {
            "jobs": {
                "electricity_pg95_seed2023_formal": {
                    "state": "SUCCESS",
                    "run_dir": str(source_run),
                }
            }
        }
        state_path = temp_root / "state.json"
        write_json(state_path, state)
        destination = temp_root / "materialized.yaml"
        materialize_evaluation_config(
            paths["validation_smoke"],
            state_path=state_path,
            destination=destination,
        )
        materialized = load(destination)
        checks["exact_checkpoint_handoff"] = (
            Path(materialized["source"]["checkpoint"]).resolve()
            == checkpoint.resolve()
        )
        checks["source_identity_recorded"] = (
            materialized["provenance"]["source_is_validation_only"] is True
        )

        bad_metrics = dict(validation_metrics)
        bad_metrics["test_split_accessed"] = True
        write_json(source_run / "metrics.json", bad_metrics)
        expect_rejected(
            lambda: materialize_evaluation_config(
                paths["validation_smoke"],
                state_path=state_path,
                destination=temp_root / "must_not_exist.yaml",
            ),
            "test_accessed_source",
        )
        checks["test_accessed_source_rejected"] = True

    failures = [name for name, passed in checks.items() if not passed]
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "negative_cases": 5,
        "gate": {
            "pg95": "both non-worse and at least one strict improvement",
            "multiscale": "both strict vs original and original+L5",
            "test_runs": "only when all protocol and scientific gates pass",
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
