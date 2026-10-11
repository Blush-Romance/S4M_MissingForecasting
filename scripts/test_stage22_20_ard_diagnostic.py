#!/usr/bin/env python3
"""CPU contracts for the Step 22.20 frozen ARD information diagnostic."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "baselines/S4M/s4m"),
    str(ROOT / "baselines/S4M/s4m/model"),
    str(ROOT / "baselines/S4M/s4m/model/S4/s4"),
    str(ROOT / "baselines/S4M/s4m/model/S4/s4/extensions/kernels"),
]

from experiments.exp_pretrain1 import Exp_Long_Term_Forecast as Exp  # noqa: E402
from run_s4m_experiment import verify_checkpoint  # noqa: E402
from run_stage22_20_ard_diagnostic import (  # noqa: E402
    project_path,
    validate_config,
)
from analyze_stage22_20_ard_information import analyse  # noqa: E402
import run_stage22_20_ard_diagnostic_queue as queue  # noqa: E402


def main() -> int:
    backbone = torch.tensor([1.0, -2.0, -0.25, 0.5]).view(4, 1, 1)
    raw = torch.tensor([1.0, 1.0, 1.0, 0.0]).view(4, 1, 1)
    target = torch.zeros_like(backbone)
    teacher = Exp._stage22_ard_analytic_targets(
        backbone,
        raw,
        target,
        torch.ones_like(target, dtype=torch.bool),
    )
    teacher_exact = (
        torch.equal(teacher["valid"], torch.tensor([True, True, True, False]))
        and torch.equal(teacher["actions"], torch.tensor([0, 2, 1, -1]))
        and torch.allclose(
            teacher["alpha"], torch.tensor([0.0, 1.0, 0.25, 0.0])
        )
        and torch.allclose(
            teacher["alignment_cross"], torch.tensor([2.0, -4.0, -0.5, 0.0])
        )
        and torch.allclose(
            teacher["residual_energy"], torch.tensor([1.0, 1.0, 1.0, 0.0])
        )
    )

    observed_mask = torch.tensor([True, False, True, True]).view(4, 1, 1)
    observed_teacher = Exp._stage22_ard_analytic_targets(
        backbone, raw, target, observed_mask
    )
    observed_mask_respected = (
        torch.equal(
            observed_teacher["valid"], torch.tensor([True, False, True, False])
        )
        and torch.equal(
            observed_teacher["actions"], torch.tensor([0, -1, 1, -1])
        )
    )

    torch.manual_seed(2220)
    batch_size, variables, hidden = 3, 7, 32
    current = torch.randn(batch_size, 8, requires_grad=True)
    raw_forecast = torch.randn(batch_size, 96, variables, requires_grad=True)
    backbone_forecast = torch.randn(
        batch_size, 96, variables, requires_grad=True
    )
    mask_features = torch.rand(batch_size, 192, 4, requires_grad=True)
    latent = torch.randn(batch_size, 192, hidden, requires_grad=True)
    structured = Exp._stage22_ard_structured_features(
        current,
        raw_forecast,
        backbone_forecast,
        mask_features,
        latent,
    )
    structure_contract = (
        structured.shape == (batch_size, 108)
        and torch.equal(structured[:, :8], current.detach())
        and bool(torch.isfinite(structured).all())
        and not structured.requires_grad
    )

    # The frozen 8-D descriptor is unchanged, but reversing temporal order must
    # change at least one of the new time-structured fields.
    reversed_structured = Exp._stage22_ard_structured_features(
        current,
        raw_forecast.flip(1),
        backbone_forecast.flip(1),
        mask_features.flip(1),
        latent.flip(1),
    )
    temporal_information_present = (
        torch.equal(reversed_structured[:, :8], structured[:, :8])
        and not torch.equal(reversed_structured[:, 8:], structured[:, 8:])
    )

    invalid_shape_rejected = False
    try:
        Exp._stage22_ard_structured_features(
            current,
            raw_forecast,
            backbone_forecast,
            mask_features[:, :-1],
            latent,
        )
    except ValueError:
        invalid_shape_rejected = True

    mismatched_teacher_shape_rejected = False
    try:
        Exp._stage22_ard_analytic_targets(
            backbone, raw, target[:3], torch.ones_like(target, dtype=torch.bool)
        )
    except ValueError:
        mismatched_teacher_shape_rejected = True

    config_paths = sorted(
        (ROOT / "configs/stage22_ard_diagnostic").glob("*.yaml")
    )
    configs_valid = len(config_paths) == 2
    loaded_configs = []
    for config_path in config_paths:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        source_path = project_path(config["source"]["resolved_config"])
        source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        validate_config(config, source)
        loaded_configs.append((config, source))

    invalid_split_rejected = False
    if loaded_configs:
        invalid = copy.deepcopy(loaded_configs[0][0])
        invalid["diagnostic"]["split"] = "test"
        try:
            validate_config(invalid, loaded_configs[0][1])
        except ValueError:
            invalid_split_rejected = True

    checkpoint_identity = False
    if loaded_configs:
        checkpoint = project_path(loaded_configs[0][0]["source"]["checkpoint"])
        evidence = verify_checkpoint(
            checkpoint,
            require_mipp=True,
            require_mcpr=True,
            require_ard=True,
        )
        checkpoint_identity = (
            evidence["checkpoint_all_model_tensors_finite"] is True
            and bool(evidence["checkpoint_mipp_state_keys"])
            and bool(evidence["checkpoint_mcpr_state_keys"])
            and bool(evidence["checkpoint_ard_state_keys"])
            and not bool(evidence["checkpoint_ttrr_state_keys"])
        )

    # End-to-end CPU fixture for the preregistered split/probe/decision path.
    rng = np.random.default_rng(2220)
    sample_count = 2528
    actions = np.arange(sample_count, dtype=np.int64) % 3
    alpha = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)[actions]
    global8 = rng.normal(size=(sample_count, 8)).astype(np.float32)
    structured108 = np.zeros((sample_count, 108), dtype=np.float32)
    structured108[:, :8] = global8
    structured108[np.arange(sample_count), 8 + actions] = 1.0
    structured108[:, 11] = alpha
    full_contribution = np.linspace(0.15, 0.35, sample_count)
    partial_contribution = 0.5 - full_contribution
    partial_probability = np.full(sample_count, 0.6)
    probabilities = np.stack((
        1.0 - full_contribution - partial_probability,
        partial_probability,
        full_contribution,
    ), axis=1)
    partial_alpha = partial_contribution / partial_probability
    valid = np.ones(sample_count, dtype=bool)
    synthetic_arrays = {
        "sample_index": np.arange(sample_count, dtype=np.int64),
        "global8": global8,
        "structured108": structured108,
        "action_logits": np.log(np.maximum(probabilities, 1e-8)),
        "action_probabilities": probabilities,
        "partial_alpha": partial_alpha,
        "student_gate": np.full(sample_count, 0.5),
        "complete_teacher_alpha": alpha,
        "complete_teacher_actions": actions,
        "complete_teacher_valid": valid,
        "observed_teacher_alpha": alpha.copy(),
        "observed_teacher_actions": actions.copy(),
        "observed_teacher_valid": valid.copy(),
        "complete_count": np.ones(sample_count),
        "observed_count": np.ones(sample_count),
        "backbone_sse": np.full(sample_count, 2.0),
        "backbone_sae": np.ones(sample_count),
        "raw_sse": np.ones(sample_count),
        "raw_sae": np.ones(sample_count),
        "ard_sse": np.ones(sample_count),
        "ard_sae": np.ones(sample_count),
        "alignment_cross": -2.0 * alpha,
        "residual_energy": np.ones(sample_count),
    }
    with tempfile.TemporaryDirectory(prefix="stage22_20_analysis_") as temp_dir:
        fixture_dir = Path(temp_dir)
        (fixture_dir / "status.json").write_text(
            json.dumps({"state": "SUCCESS"}), encoding="utf-8"
        )
        (fixture_dir / "metrics.json").write_text(json.dumps({
            "artifact_status": "PASS",
            "test_split_accessed": False,
            "uses_test_split": False,
            "source_split": "val",
            "updates_model_or_memory": False,
            "chronological_loader": True,
            "validation_batches_processed": 158,
            "mse": 1.0,
            "mae": 1.0,
        }), encoding="utf-8")
        np.savez_compressed(
            fixture_dir / "ard_information_samples.npz", **synthetic_arrays
        )
        synthetic_report = analyse(fixture_dir)
    analysis_path_exact = (
        synthetic_report["status"] == "PASS"
        and synthetic_report["feature_information_bottleneck"]
        == "FEATURE_INFORMATION_BOTTLENECK_CONFIRMED"
        and synthetic_report["mixture_cancellation"]
        == "MIXTURE_CANCELLATION_CONFIRMED"
        and synthetic_report["decision"]
        == "STRUCTURED_TOKEN_ENCODER_PLUS_DISCRETE_ACTION_EXECUTION"
        and synthetic_report["test_accessed"] is False
    )
    queue_contract = (
        len(queue.scheduler.JOB_DEFINITIONS) == 2
        and queue.scheduler.JOB_DEFINITIONS[0]["id"] == "ard_information_smoke"
        and queue.scheduler.JOB_DEFINITIONS[1]["depends_on"]
        == ["ard_information_smoke"]
        and queue.scheduler.RUNTIME_ROOT.name == "stage22_20_ard_diagnostic"
        and queue.scheduler.RUNNER.name == "run_stage22_20_ard_diagnostic.py"
        and queue.scheduler.AGGREGATOR.name
        == "analyze_stage22_20_ard_information.py"
    )

    checks = {
        "analytic_teacher_exact": teacher_exact,
        "observed_mask_respected": observed_mask_respected,
        "structured_108_contract": structure_contract,
        "structured_contains_temporal_information": temporal_information_present,
        "invalid_structured_shape_rejected": invalid_shape_rejected,
        "mismatched_teacher_shape_rejected": mismatched_teacher_shape_rejected,
        "two_frozen_configs_valid": configs_valid,
        "test_split_config_rejected": invalid_split_rejected,
        "source_checkpoint_identity": checkpoint_identity,
        "frozen_analysis_decision_path": analysis_path_exact,
        "queue_smoke_gates_formal": queue_contract,
    }
    for name, passed in checks.items():
        print(f"{name}={passed}")
    if not all(checks.values()):
        raise SystemExit(f"Step 22.20 CPU contract failed: {checks}")
    print("stage22_20_ard_diagnostic_cpu_contracts=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
