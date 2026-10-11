#!/usr/bin/env python3
"""CPU/static contracts for Stage22.16 MCPR."""

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
from model.MCPR import MaskConditionedParetoResidual  # noqa: E402
from experiments.exp_pretrain1 import Exp_Long_Term_Forecast as Exp  # noqa: E402
import run_stage22_mcpr_queue as queue  # noqa: E402


def main() -> int:
    config_paths = sorted((ROOT / "configs/stage22_mcpr").glob("*.yaml"))
    configs = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in config_paths]
    for config in configs:
        validate_experiment_config(config)

    torch.manual_seed(22)
    module = MaskConditionedParetoResidual(8, 5, rank=4, kernel_size=3)
    hidden = torch.randn(3, 12, 8)
    observed = torch.ones(3, 12, 5)
    missing = observed.clone()
    missing[:, 3:8] = 0.0
    initial_observed = module(hidden, observed)["residual"]
    initial_missing = module(hidden, missing)["residual"]
    exact_zero_initialization = bool(
        torch.equal(initial_observed, torch.zeros_like(initial_observed))
        and torch.equal(initial_missing, torch.zeros_like(initial_missing))
    )
    target = torch.randn_like(initial_observed)
    loss = (initial_observed - target).square().mean()
    loss.backward()
    output_gradient = float(module.output_projection.weight.grad.abs().sum())
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    optimizer.step()
    mask_response = float(
        (module(hidden, observed)["residual"] - module(hidden, missing)["residual"])
        .abs().mean().item()
    )
    prediction = module(hidden, observed)["residual"]
    mse = (prediction - target).square().mean()
    mae = (prediction - target).abs().mean()
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    gradients, diagnostics = Exp._mcpr_common_descent_gradients(
        mse, mae, parameters
    )

    negative = copy.deepcopy(configs[0])
    negative["official_args"]["use_mcpr"] = False
    rejected_identity = False
    try:
        validate_experiment_config(negative)
    except ValueError:
        rejected_identity = True

    jobs = {job["id"]: job for job in queue.scheduler.JOB_DEFINITIONS}
    checks = {
        "four_configs": len(config_paths) == 4,
        "configs_validate": True,
        "same_fixed_assets": len({json.dumps(config["data"], sort_keys=True) for config in configs}) == 1,
        "objectives_exact": {config["official_args"]["mcpr_objective"] for config in configs} == {"mse_only", "pareto_common_descent"},
        "zero_initialized_baseline_equivalence": exact_zero_initialization,
        "output_projection_receives_gradient": output_gradient > 0.0,
        "mask_response_after_update": mask_response > 0.0,
        "common_mse_descent": diagnostics["common_mse_dot"] >= -1e-10,
        "common_mae_descent": diagnostics["common_mae_dot"] >= -1e-10,
        "combined_gradients_finite": all(bool(torch.isfinite(value).all()) for value in gradients),
        "disabled_identity_rejected": rejected_identity,
        "queue_has_four_jobs": len(jobs) == 4,
        "formal_depends_on_matching_smoke": (
            jobs["mcpr_mse_formal"]["depends_on"] == ["mcpr_mse_smoke"]
            and jobs["mcpr_pcd_formal"]["depends_on"] == ["mcpr_pcd_smoke"]
        ),
        "queue_runtime_isolated": queue.scheduler.RUNTIME_ROOT == ROOT / "runtime/stage22_mcpr",
    }
    report = {
        "step": "22.16",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "pareto_diagnostics": diagnostics,
    }
    path = ROOT / "reports/stage22_mcpr_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
