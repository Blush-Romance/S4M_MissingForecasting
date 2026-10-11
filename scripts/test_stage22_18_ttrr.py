#!/usr/bin/env python3
"""CPU/static contracts for Step 22.18 TTRR-MCPR."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path[:0] = [
    str(ROOT / "baselines/S4M/s4m"),
    str(ROOT / "baselines/S4M/s4m/model"),
    str(ROOT / "baselines/S4M/s4m/model/S4/s4"),
    str(ROOT / "baselines/S4M/s4m/model/S4/s4/extensions/kernels"),
]

from run_s4m_experiment import validate_experiment_config  # noqa: E402
from run_stage12_validation_cache import validate_config as validate_cache_config  # noqa: E402
from model.MCPR import MaskConditionedParetoResidual  # noqa: E402
from model.TTRR import TailAwareTrustRegionGate  # noqa: E402
from experiments.exp_pretrain1 import Exp_Long_Term_Forecast as Exp  # noqa: E402
import run_stage22_18_ttrr_queue as queue  # noqa: E402


def expect_rejected(config: dict) -> bool:
    try:
        validate_experiment_config(config)
    except (TypeError, ValueError):
        return True
    return False


def main() -> int:
    config_paths = sorted(
        path for path in (ROOT / "configs/stage22_ttrr").glob("*.yaml")
        if path.name.startswith(("mcpr_mse_", "ttrr_mcpr_"))
    )
    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in config_paths]
    for config in configs:
        validate_experiment_config(config)
    cache_paths = sorted(
        path for path in (ROOT / "configs/stage22_ttrr").glob("pg95_*.yaml")
        if not path.name.startswith("timing_")
    )
    for path in cache_paths:
        cache_config = yaml.safe_load(path.read_text(encoding="utf-8"))
        source_path = ROOT / cache_config["source"]["resolved_config"]
        source_config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        validate_cache_config(cache_config, source_config)

    controls = [c for c in configs if not c["official_args"]["use_ttrr"]]
    candidates = [c for c in configs if c["official_args"]["use_ttrr"]]
    formal = [c for c in configs if not c["experiment"]["smoke_test"]]

    torch.manual_seed(2218)
    disabled = MaskConditionedParetoResidual(
        hidden_size=8, num_variables=5, rank=4, kernel_size=3, use_ttrr=False
    )
    torch.manual_seed(2218)
    historical_default = MaskConditionedParetoResidual(
        hidden_size=8, num_variables=5, rank=4, kernel_size=3
    )
    disabled_state_exact = all(
        torch.equal(disabled.state_dict()[key], historical_default.state_dict()[key])
        for key in disabled.state_dict()
    ) and set(disabled.state_dict()) == set(historical_default.state_dict())

    enabled = MaskConditionedParetoResidual(
        hidden_size=8, num_variables=5, rank=4, kernel_size=3, use_ttrr=True
    )
    hidden = torch.randn(4, 12, 8, requires_grad=True)
    mask = torch.ones(4, 12, 5)
    mask[1:, 3:8, :] = 0.0
    backbone = torch.randn(4, 12, 5, requires_grad=True)
    initial = enabled(hidden, mask, backbone)
    initial_gate_half = torch.equal(
        initial["trust_gate"], torch.full_like(initial["trust_gate"], 0.5)
    )
    initial_baseline_equivalence = bool(
        torch.equal(initial["residual"], torch.zeros_like(initial["residual"]))
    )
    trust_features_detached = not initial["trust_features"].requires_grad

    with torch.no_grad():
        enabled.output_projection.weight.fill_(0.05)
        enabled.output_projection.bias.fill_(0.01)
    after = enabled(hidden, mask, backbone)
    exact_half_scaling = bool(torch.equal(
        after["residual"], 0.5 * after["raw_residual"]
    ))
    gate_loss = after["residual"].square().mean()
    gate_loss.backward()
    gate_gradient = float(enabled.trust_gate.projection.weight.grad.abs().sum().item())

    target = torch.zeros(4, 1, 1)
    backbone_prediction = torch.zeros_like(target)
    candidate = torch.tensor([0.0, 1.0, 2.0, 3.0]).view(4, 1, 1).requires_grad_()
    observed = torch.ones_like(target, dtype=torch.bool)
    objective = Exp._ttrr_tail_trust_objective(
        prediction=candidate,
        backbone_prediction=backbone_prediction,
        target=target,
        target_mask=observed,
        tail_fraction=0.25,
        tail_weight=1.0,
    )
    objective["total_loss"].backward()
    better_candidate = torch.zeros(4, 1, 1)
    worse_backbone = torch.ones_like(better_candidate)
    non_harm = Exp._ttrr_tail_trust_objective(
        prediction=better_candidate,
        backbone_prediction=worse_backbone,
        target=torch.zeros_like(better_candidate),
        target_mask=torch.ones_like(better_candidate, dtype=torch.bool),
        tail_fraction=0.10,
        tail_weight=1.0,
    )

    negative_without_mcpr = copy.deepcopy(candidates[0])
    negative_without_mcpr["official_args"]["use_mcpr"] = False
    negative_wrong_objective = copy.deepcopy(candidates[0])
    negative_wrong_objective["official_args"]["mcpr_objective"] = "mse_only"
    negative_missing_gate = copy.deepcopy(candidates[0])
    negative_missing_gate["official_args"]["use_ttrr"] = False
    negative_fraction = copy.deepcopy(candidates[0])
    negative_fraction["official_args"]["ttrr_tail_fraction"] = 0.0
    negative_weight = copy.deepcopy(candidates[0])
    negative_weight["official_args"]["ttrr_tail_weight"] = -1.0

    fair_keys = (
        "seed", "seq_len", "pred_len", "batch_size", "learning_rate",
        "optimizer", "train_epochs", "patience", "lradj", "mcpr_rank",
        "mcpr_kernel_size", "mcpr_learning_rate", "mcpr_backbone_checkpoint",
        "post_train_evaluation",
    )
    formal_control = next(c for c in formal if not c["official_args"]["use_ttrr"])
    formal_candidate = next(c for c in formal if c["official_args"]["use_ttrr"])
    jobs = {job["id"]: job for job in queue.scheduler.JOB_DEFINITIONS}
    checks = {
        "four_configs": len(configs) == 4,
        "two_pg95_cache_configs_validate": len(cache_paths) == 2,
        "two_controls_two_candidates": len(controls) == 2 and len(candidates) == 2,
        "configs_validate": True,
        "same_seed2024_fixed_assets": (
            len({json.dumps(c["data"], sort_keys=True) for c in configs}) == 1
            and all(c["experiment"]["seed"] == 2024 for c in configs)
        ),
        "formal_budget_equal": all(
            formal_control["official_args"][key]
            == formal_candidate["official_args"][key]
            for key in fair_keys
        ),
        "formal_runtime_caps_disabled": all(
            c["runtime"]["max_train_batches"] == -1
            and c["runtime"]["max_eval_batches"] == -1
            for c in formal
        ),
        "disabled_state_dict_matches_historical_default": disabled_state_exact,
        "disabled_has_no_trust_parameters": not any(
            "trust_gate" in key for key in disabled.state_dict()
        ),
        "feature_schema_is_eight_dimensional": (
            TailAwareTrustRegionGate().feature_dim == 8
            and after["trust_features"].shape == (4, 8)
        ),
        "zero_initialized_gate_is_exactly_half": initial_gate_half,
        "zero_initialized_residual_reproduces_backbone": initial_baseline_equivalence,
        "trust_features_are_gradient_detached": trust_features_detached,
        "half_gate_scales_raw_residual_exactly": exact_half_scaling,
        "trust_gate_receives_gradient_after_residual_activates": gate_gradient > 0.0,
        "tail_count_is_exact": objective["tail_count"] == 1,
        "candidate_mse_is_exact": torch.isclose(
            objective["candidate_mse"].detach(), torch.tensor(3.5)
        ).item(),
        "positive_tail_cvar_is_exact": torch.isclose(
            objective["tail_cvar"].detach(), torch.tensor(9.0)
        ).item(),
        "total_objective_is_exact": torch.isclose(
            objective["total_loss"].detach(), torch.tensor(12.5)
        ).item(),
        "tail_objective_backpropagates": bool(torch.isfinite(candidate.grad).all()),
        "non_harmful_candidate_has_zero_tail_penalty": (
            float(non_harm["tail_cvar"].item()) == 0.0
        ),
        "rejects_ttrr_without_mcpr": expect_rejected(negative_without_mcpr),
        "rejects_ttrr_with_wrong_objective": expect_rejected(negative_wrong_objective),
        "rejects_tail_objective_without_gate": expect_rejected(negative_missing_gate),
        "rejects_zero_tail_fraction": expect_rejected(negative_fraction),
        "rejects_negative_tail_weight": expect_rejected(negative_weight),
        "queue_has_six_jobs": len(jobs) == 6,
        "all_formals_depend_on_matching_smoke": (
            jobs["pg95_formal"]["depends_on"] == ["pg95_smoke"]
            and jobs["mcpr_mse_formal"]["depends_on"] == ["mcpr_mse_smoke"]
            and jobs["ttrr_formal"]["depends_on"] == ["ttrr_smoke"]
        ),
        "pg95_jobs_use_read_only_cache_runner": all(
            Path(jobs[name]["runner"]).name == "run_stage12_validation_cache.py"
            for name in ("pg95_smoke", "pg95_formal")
        ),
        "train_jobs_use_default_training_runner": all(
            "runner" not in jobs[name]
            for name in (
                "mcpr_mse_smoke", "mcpr_mse_formal", "ttrr_smoke", "ttrr_formal"
            )
        ),
        "queue_runtime_isolated": (
            queue.scheduler.RUNTIME_ROOT == ROOT / "runtime/stage22_18_ttrr"
        ),
    }
    report = {
        "step": "22.18",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "gate_parameter_count": sum(
            p.numel() for p in enabled.trust_gate.parameters()
        ),
        "gate_gradient_l1": gate_gradient,
        "synthetic_tail_objective": {
            key: (float(value.detach().item()) if torch.is_tensor(value) else int(value))
            for key, value in objective.items()
            if key not in {"candidate_sample_mse_mean", "backbone_sample_mse_mean"}
        },
    }
    path = ROOT / "reports/stage22_18_ttrr_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
