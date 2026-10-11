#!/usr/bin/env python3
"""Static, negative, handoff, queue, and decision tests for Step-22.22."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile

import yaml

from aggregate_stage22_22_traffic import scientific_decision
from analyze_stage22_21_traffic_validation import audit_method_differences
from run_s4m_experiment import validate_experiment_config
from run_stage14_uncertainty_cache import validate_config as validate_eval_config
from run_stage22_22_traffic_pipeline import materialize_evaluation_config
import run_stage22_22_traffic_queue as queue


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/stage22_22_traffic"
OLD_CONFIG_ROOT = ROOT / "configs/stage22_21_traffic"
REPORT = ROOT / "reports/stage22_22_traffic_pipeline_test.json"


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def expect_rejected(callback, label: str) -> None:
    try:
        callback()
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError):
        return
    raise AssertionError(f"negative case was accepted: {label}")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def source_config(config: dict) -> dict:
    seed = int(config["experiment"]["seed"])
    family = (
        "s4m" if config["experiment"]["method"] == "s4m_baseline" else "pg95"
    )
    root = OLD_CONFIG_ROOT if seed == 2026 else CONFIG_ROOT
    return load(root / f"{family}_traffic_seed{seed}.yaml")


def main() -> int:
    paths = sorted(CONFIG_ROOT.glob("*.yaml"))
    if len(paths) != 22:
        raise AssertionError(f"expected 22 configs, found {len(paths)}")
    configs = {path.stem: load(path) for path in paths}
    training = {name: config for name, config in configs.items() if "official_args" in config}
    evaluations = {name: config for name, config in configs.items() if "source" in config}
    for config in training.values():
        validate_experiment_config(config)
    for config in evaluations.values():
        validate_eval_config(config, source_config(config))

    checks: dict[str, bool] = {
        "exact_config_counts": len(training) == 8 and len(evaluations) == 14,
        "all_training_protocol_frozen": all(
            config["experiment"]["dataset"] == "traffic"
            and config["experiment"]["seed"] in (2027, 2028)
            and config["official_args"]["seed"] == config["experiment"]["seed"]
            and config["official_args"]["target"] == "OT"
            and config["official_args"]["enc_in"]
            == config["official_args"]["dec_in"]
            == config["official_args"]["c_out"] == 862
            and config["official_args"]["batch_size"] == 16
            and config["official_args"]["learning_rate"] == 0.001
            and config["official_args"]["optimizer"] == "adam"
            and config["official_args"]["patience"] == 3
            and config["official_args"]["post_train_evaluation"] == "validation"
            and config["experiment"]["training_stability_amendment"]
            == "adam_lr_0.01_to_0.001_user_authorized_20260816"
            for config in training.values()
        ),
        "smoke_and_formal_budgets_separated": all(
            (
                config["official_args"]["train_epochs"] == 1
                and config["official_args"]["max_train_batches"] == 2
                and config["official_args"]["max_eval_batches"] == 3
            ) if config["experiment"]["smoke_test"] else (
                config["official_args"]["train_epochs"] == 15
                and config["official_args"]["max_train_batches"] == -1
                and config["official_args"]["max_eval_batches"] == -1
            )
            for config in training.values()
        ),
        "evaluation_protocol_frozen": all(
            config["experiment"]["seed"] in (2026, 2027, 2028)
            and config["diagnostic"]["nested_seed"]
            == config["experiment"]["seed"]
            and config["diagnostic"]["target_rate"] == 0.15
            and config["diagnostic"]["block_lengths"] == [1, 5, 20]
            and config["diagnostic"]["multiscale_weights"]
            == {"original": 0.5, "l5": 0.25, "l20": 0.25}
            and (
                config["diagnostic"]["max_eval_batches"] == 3
                and config["diagnostic"]["timing_warmup_batches"] == 1
                if config["experiment"]["smoke_test"]
                else config["diagnostic"]["max_eval_batches"] == -1
                and config["diagnostic"]["timing_warmup_batches"] == 10
            )
            for config in evaluations.values()
        ),
        "s4m_has_explicit_readonly_test_identity": all(
            config["experiment"]["method"] == "s4m_baseline"
            and config["diagnostic"]["variant"] == "stage22_22_traffic_s4m_test"
            and config["diagnostic"]["split"] == "test"
            and config["source"]["job_id"]
            == f"traffic_s4m_seed{config['experiment']['seed']}_formal"
            for name, config in evaluations.items() if name.startswith("s4m_")
        ),
        "pg95_validation_and_test_are_precommitted": all(
            config["experiment"]["method"] == "pg95"
            and config["diagnostic"]["variant"] in {
                "stage22_21_traffic_validation", "stage22_21_traffic_test"
            }
            and config["source"]["job_id"]
            == f"traffic_pg95_seed{config['experiment']['seed']}_formal"
            for name, config in evaluations.items() if name.startswith("pg95_")
        ),
    }

    for seed in (2027, 2028):
        s4m = training[f"s4m_traffic_seed{seed}"]["official_args"]
        pg95 = training[f"pg95_traffic_seed{seed}"]["official_args"]
        _, unexpected, required = audit_method_differences(s4m, pg95)
        checks[f"seed{seed}_only_pg95_method_difference"] = not unexpected and required
        checks[f"seed{seed}_assets_equal_between_methods"] = (
            training[f"s4m_traffic_seed{seed}"]["data"]
            == training[f"pg95_traffic_seed{seed}"]["data"]
        )

    reference = copy.deepcopy(training["s4m_traffic_seed2027"]["official_args"])
    candidate = copy.deepcopy(training["s4m_traffic_seed2028"]["official_args"])
    for payload in (reference, candidate):
        for key in ("seed", "model_id", "des"):
            payload.pop(key)
    checks["cross_seed_s4m_formal_budget_identical"] = reference == candidate
    reference = copy.deepcopy(training["pg95_traffic_seed2027"]["official_args"])
    candidate = copy.deepcopy(training["pg95_traffic_seed2028"]["official_args"])
    for payload in (reference, candidate):
        for key in ("seed", "model_id", "des"):
            payload.pop(key)
    checks["cross_seed_pg95_formal_budget_identical"] = reference == candidate

    bad = copy.deepcopy(evaluations["s4m_traffic_multiscale_test_seed2027"])
    bad["experiment"]["method"] = "pg95"
    expect_rejected(lambda: validate_eval_config(bad, source_config(
        evaluations["s4m_traffic_multiscale_test_seed2027"]
    )), "s4m_identity_masquerades_as_pg95")
    bad = copy.deepcopy(evaluations["pg95_traffic_multiscale_test_seed2027"])
    bad["diagnostic"]["multiscale_weights"]["l20"] = 0.30
    expect_rejected(lambda: validate_eval_config(bad, source_config(bad)),
                    "changed_multiscale_weight")
    bad = copy.deepcopy(evaluations["pg95_traffic_multiscale_test_seed2027"])
    bad["experiment"]["seed"] = 2029
    bad["diagnostic"]["nested_seed"] = 2029
    bad_source = source_config(evaluations["pg95_traffic_multiscale_test_seed2027"])
    bad_source["official_args"]["seed"] = 2029
    expect_rejected(lambda: validate_eval_config(bad, bad_source),
                    "non_preregistered_seed")
    bad_source = copy.deepcopy(source_config(
        evaluations["s4m_traffic_multiscale_test_seed2027"]
    ))
    bad_source["official_args"]["use_mipp"] = True
    expect_rejected(lambda: validate_eval_config(
        evaluations["s4m_traffic_multiscale_test_seed2027"], bad_source
    ), "s4m_source_with_mipp")
    checks["negative_eval_cases_rejected"] = True

    definitions = {job["id"]: job for job in queue.scheduler.JOB_DEFINITIONS}
    checks["queue_has_exactly_22_unique_jobs"] = len(definitions) == 22
    checks["seed2026_reuses_source_but_smoke_precedes_formal"] = (
        definitions["traffic_s4m_multiscale_test_seed2026_smoke"]["depends_on"] == []
        and definitions["traffic_s4m_multiscale_test_seed2026_formal"]["depends_on"]
        == ["traffic_s4m_multiscale_test_seed2026_smoke"]
    )
    checks["replication_dependencies_complete"] = all(
        definitions[f"traffic_{family}_seed{seed}_formal"]["depends_on"]
        == [f"traffic_{family}_seed{seed}_smoke"]
        and definitions[f"traffic_{family}_multiscale_test_seed{seed}_smoke"]
        ["depends_on"] == [f"traffic_{family}_seed{seed}_formal"]
        and definitions[f"traffic_{family}_multiscale_test_seed{seed}_formal"]
        ["depends_on"] == [f"traffic_{family}_multiscale_test_seed{seed}_smoke"]
        for seed in (2027, 2028) for family in ("s4m", "pg95")
    ) and all(
        definitions[f"traffic_pg95_multiscale_validation_seed{seed}_smoke"]
        ["depends_on"] == [f"traffic_pg95_seed{seed}_formal"]
        and definitions[f"traffic_pg95_multiscale_validation_seed{seed}_formal"]
        ["depends_on"]
        == [f"traffic_pg95_multiscale_validation_seed{seed}_smoke"]
        for seed in (2027, 2028)
    )

    decisions = {
        "incomplete": scientific_decision(
            complete=False, protocol_pass=False,
            cnm_all_test_wins=False, pg95_all_test_wins=False,
        ),
        "protocol": scientific_decision(
            complete=True, protocol_pass=False,
            cnm_all_test_wins=True, pg95_all_test_wins=True,
        ),
        "unstable": scientific_decision(
            complete=True, protocol_pass=True,
            cnm_all_test_wins=False, pg95_all_test_wins=True,
        ),
        "confirmed": scientific_decision(
            complete=True, protocol_pass=True,
            cnm_all_test_wins=True, pg95_all_test_wins=False,
        ),
        "strong": scientific_decision(
            complete=True, protocol_pass=True,
            cnm_all_test_wins=True, pg95_all_test_wins=True,
        ),
    }
    checks["decision_precedence_frozen"] = decisions == {
        "incomplete": "TRAFFIC_MULTISEED_INCOMPLETE",
        "protocol": "TRAFFIC_MULTISEED_PROTOCOL_FAIL",
        "unstable": "TRAFFIC_MULTISEED_UNSTABLE",
        "confirmed": "TRAFFIC_MULTISEED_CONFIRMED",
        "strong": "TRAFFIC_MULTISEED_STRONG_CHAIN_CONFIRMED",
    }

    with tempfile.TemporaryDirectory(prefix="stage22_22_handoff_") as temporary:
        temp_root = Path(temporary)
        source_run = temp_root / "formal_s4m"
        checkpoint = source_run / "checkpoints/setting/checkpoint.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        formal_source = training["s4m_traffic_seed2027"]
        (source_run / "resolved_config.yaml").write_text(
            yaml.safe_dump(formal_source), encoding="utf-8"
        )
        metrics = {
            "checkpoint": str(checkpoint),
            "evaluation_split": "validation",
            "post_train_evaluation": "validation",
            "test_split_accessed": False,
            "chronological_loader": True,
        }
        write_json(source_run / "metrics.json", metrics)
        write_json(source_run / "status.json", {"state": "SUCCESS"})
        state_path = temp_root / "state.json"
        write_json(state_path, {"jobs": {"traffic_s4m_seed2027_formal": {
            "state": "SUCCESS", "run_dir": str(source_run),
        }}})
        destination = temp_root / "materialized.yaml"
        materialize_evaluation_config(
            CONFIG_ROOT / "s4m_traffic_multiscale_test_seed2027.yaml",
            default_state_path=state_path,
            destination=destination,
        )
        materialized = load(destination)
        checks["exact_validation_only_checkpoint_handoff"] = (
            Path(materialized["source"]["checkpoint"]).resolve()
            == checkpoint.resolve()
            and materialized["provenance"]
            ["test_precommitted_without_validation_selection"] is True
        )
        metrics["test_split_accessed"] = True
        write_json(source_run / "metrics.json", metrics)
        expect_rejected(lambda: materialize_evaluation_config(
            CONFIG_ROOT / "s4m_traffic_multiscale_test_seed2027.yaml",
            default_state_path=state_path,
            destination=temp_root / "reject_test_source.yaml",
        ), "source_that_already_accessed_test")
        checks["test_accessed_source_rejected"] = True

    failures = [name for name, passed in checks.items() if not bool(passed)]
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "checks": {name: bool(value) for name, value in checks.items()},
        "failures": failures,
        "config_count": len(configs),
        "training_config_count": len(training),
        "evaluation_config_count": len(evaluations),
        "queue_job_count": len(definitions),
        "negative_cases": 4,
        "decision_cases": decisions,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
