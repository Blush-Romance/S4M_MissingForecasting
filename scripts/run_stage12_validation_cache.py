#!/usr/bin/env python3
"""Build one auditable chronological validation cache from a frozen checkpoint."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

# The orchestration parent must not reserve CUDA before the scheduler selects
# a stable GPU UUID for the child process.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import yaml

from run_s4m_experiment import (
    atomic_json,
    atomic_text,
    cli_from_mapping,
    git_evidence,
    resolve_gpu_metadata,
    run_capture,
    safe_environment_text,
    verify_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "baselines" / "S4M"
S4M_ROOT = BASELINE_ROOT / "s4m"
STRUCTURED_KERNEL_ROOT = S4M_ROOT / "model" / "S4" / "s4" / "extensions" / "kernels"
EXPECTED_METHOD_MODES = {
    "s4m": None,
    "pg95": "absolute",
    "mrra_v3": "absolute",
    "mcpr_mse": "absolute",
    "mcpr_pcd": "absolute",
    "mrba50": "mask_regime_balanced_rank",
}
MCPR_REDUCTION_RTOL = 2e-5
# A sample contains 96*321 float32 forecast elements.  Subtracting two
# separately reduced SSE values is less stable than reducing their elementwise
# difference; 8e-3 safely covers the measured <=6.35e-3 reduction-order gap
# while remaining below 2.6e-7 per target element.
MCPR_REDUCTION_ATOL = 8e-3


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_id(config: dict[str, Any], timestamp: str) -> str:
    experiment = config["experiment"]
    smoke = "__smoke" if experiment.get("smoke_test") else ""
    stage_identity = str(config["diagnostic"].get("stage_identity", "stage12"))
    prefix = (
        "stage22_internal_valcache"
        if stage_identity == "stage22_internal"
        else "stage22_mcpr_valcache" if stage_identity == "stage22_mcpr"
        else "stage22_crossmask_valcache" if stage_identity == "stage22_crossmask"
        else "stage22_valcache" if stage_identity == "stage22"
        else "stage12_valcache"
    )
    return (
        f"{prefix}_{experiment['method']}__{experiment['dataset']}"
        f"__validation__seed{experiment['seed']}{smoke}__{timestamp}"
    )


def validate_config(config: dict[str, Any], source_config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    diagnostic = config["diagnostic"]
    method = str(experiment["method"])
    stage_identity = str(diagnostic.get("stage_identity", "stage12"))
    if stage_identity not in (
        "stage12", "stage22", "stage22_internal", "stage22_mcpr",
        "stage22_crossmask",
    ):
        raise ValueError(f"unsupported validation-cache stage identity: {stage_identity}")
    if stage_identity == "stage22_mcpr":
        allowed_methods = {"mcpr_mse", "mcpr_pcd"}
    elif stage_identity == "stage22_crossmask":
        allowed_methods = {"pg95"}
    elif stage_identity in {"stage22", "stage22_internal"}:
        allowed_methods = {"s4m", "pg95", "mrra_v3"}
    else:
        allowed_methods = {"pg95", "mrba50"}
    if method not in allowed_methods:
        raise ValueError(f"unsupported {stage_identity} method: {method}")
    if stage_identity in {
        "stage22", "stage22_internal", "stage22_mcpr", "stage22_crossmask"
    }:
        if experiment["dataset"] != "electricity":
            raise ValueError("stage22 frozen screen is registered only for Electricity")
        expected_evaluation_seed = (
            int(experiment["seed"])
            if stage_identity == "stage22_crossmask"
            else 2023
        )
        if stage_identity == "stage22_crossmask" and expected_evaluation_seed not in {
            2024, 2025,
        }:
            raise ValueError(
                "stage22 cross-mask screen is registered only for seeds 2024/2025"
            )
        if int(experiment["seed"]) != expected_evaluation_seed:
            raise ValueError(
                f"{stage_identity} is registered only for seed{expected_evaluation_seed}"
            )
        if float(diagnostic.get("nested_target_rate", -1.0)) != 0.15:
            raise ValueError("stage22 nested target rate is frozen at 0.15")
        if int(diagnostic.get("nested_block_length", -1)) != 5:
            raise ValueError("stage22 nested block length is frozen at L=5")
        if int(diagnostic.get("nested_seed", -1)) != expected_evaluation_seed:
            raise ValueError(
                f"{stage_identity} nested seed is frozen at {expected_evaluation_seed}"
            )
    else:
        if experiment["dataset"] != "weather":
            raise ValueError("stage12 frozen screen is registered only for Weather")
        if int(experiment["seed"]) not in (2024, 2025):
            raise ValueError("stage12 frozen screen is registered only for seeds 2024/2025")
    if diagnostic.get("split") != "val":
        raise ValueError(f"{stage_identity} cache must use the validation split")
    max_batches = int(diagnostic["max_eval_batches"])
    if max_batches == 0 or max_batches < -1:
        raise ValueError("max_eval_batches must be -1 or a positive integer")
    official = source_config["official_args"]
    if stage_identity == "stage22_crossmask":
        if int(official["seed"]) != 2023:
            raise ValueError("stage22 cross-mask source checkpoint must be seed2023")
    elif int(official["seed"]) != int(experiment["seed"]):
        raise ValueError(f"source checkpoint seed does not match {stage_identity} config")
    # PG95 predates the explicit mode field; its original parser default was
    # `absolute`.  Newer modes must always be explicit in their resolved config.
    effective_mode = official.get("mipp_write_gate_mode", "absolute")
    expected_mode = EXPECTED_METHOD_MODES[method]
    if method == "s4m":
        if bool(official.get("use_mipp", False)):
            raise ValueError("stage22 S4M source must keep MIPP disabled")
    else:
        if effective_mode != expected_mode:
            raise ValueError("source checkpoint write mode does not match method identity")
        if not bool(official.get("use_mipp")):
            raise ValueError(f"{stage_identity} PG95/MRBA sources must use MIPP")
    if method == "mrra_v3":
        if not bool(official.get("use_mrra", False)):
            raise ValueError("stage22 MRRA-v3 source must enable MRRA")
        if str(official.get("mrra_version", "")) != "v3_gradient_isolated_two_timescale":
            raise ValueError("stage22 MRRA-v3 source has the wrong version")
        if float(official.get("mrra_lr_multiplier", -1.0)) != 0.1:
            raise ValueError("stage22 MRRA-v3 source must retain lr multiplier 0.1")
        if bool(official.get("use_radr", False)):
            raise ValueError("stage22 MRRA-v3 source must keep RADR disabled")
        if stage_identity == "stage22_internal" and not bool(
            diagnostic.get("internal_features", False)
        ):
            raise ValueError("stage22 internal cache must enable internal_features")
    elif bool(official.get("use_mrra", False)):
        raise ValueError(f"{stage_identity} {method} source must keep MRRA disabled")
    if method in {"mcpr_mse", "mcpr_pcd"}:
        expected_objective = {
            "mcpr_mse": "mse_only",
            "mcpr_pcd": "pareto_common_descent",
        }[method]
        if not bool(official.get("use_mcpr", False)):
            raise ValueError(f"{method} source must enable MCPR")
        if str(official.get("mcpr_objective", "")) != expected_objective:
            raise ValueError(f"{method} source has the wrong MCPR objective")
        if any(bool(official.get(key, False)) for key in (
            "use_mrra", "use_radr", "use_fusr", "use_cfr"
        )):
            raise ValueError(f"{method} source combines an excluded innovation module")
    elif bool(official.get("use_mcpr", False)):
        raise ValueError(f"{stage_identity} {method} source must keep MCPR disabled")
    if method == "mrba50" and int(official.get("mipp_write_budget", -1)) != 50:
        raise ValueError("stage12 MRBA source must retain the frozen M=50 budget")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-uuid", default=None)
    parser.add_argument("--scheduler-attempt", type=int, default=0)
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage_identity = str(config["diagnostic"].get("stage_identity", "stage12"))
    source_config_path = resolve_project_path(config["source"]["resolved_config"])
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    checkpoint = resolve_project_path(config["source"]["checkpoint"])
    source_metrics_path = resolve_project_path(config["source"]["metrics"])
    backbone_metrics_path = (
        resolve_project_path(config["source"]["backbone_metrics"])
        if stage_identity == "stage22_mcpr"
        else None
    )
    missing_csv = resolve_project_path(config["data"]["missing_csv"])
    complete_csv = resolve_project_path(config["data"]["complete_csv"])
    mask_asset = resolve_project_path(config["data"]["mask_asset"])
    required_files = (
        config_path,
        source_config_path,
        checkpoint,
        source_metrics_path,
        missing_csv,
        complete_csv,
        mask_asset,
    ) + ((backbone_metrics_path,) if backbone_metrics_path is not None else ())
    for required in required_files:
        if not required.is_file():
            raise FileNotFoundError(f"required stage12 asset is missing: {required}")
    validate_config(config, source_config)

    if args.gpu_uuid:
        config["runtime"]["gpu_uuid"] = args.gpu_uuid
    if config["runtime"].get("device") == "cuda":
        config["runtime"].update(resolve_gpu_metadata(str(config["runtime"]["gpu_uuid"])))
    config["runtime"]["scheduler_attempt"] = int(args.scheduler_attempt)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = experiment_id(config, timestamp)
    run_dir = PROJECT_ROOT / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"experiment ID already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    metrics_path = run_dir / "metrics.json"
    samples_path = run_dir / (
        "stage22_validation_samples.npz"
        if stage_identity in {
            "stage22", "stage22_internal", "stage22_mcpr", "stage22_crossmask"
        }
        else "validation_samples.npz"
    )
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    backbone_metrics = (
        json.loads(backbone_metrics_path.read_text(encoding="utf-8"))
        if backbone_metrics_path is not None
        else None
    )
    source_checkpoint_sha256 = file_sha256(checkpoint)
    mask_asset_sha256 = file_sha256(mask_asset)
    checkpoint_verification = verify_checkpoint(
        checkpoint,
        require_mrra=str(config["experiment"]["method"]) == "mrra_v3",
        require_mcpr=str(config["experiment"]["method"]) in {"mcpr_mse", "mcpr_pcd"},
    )

    official_args = copy.deepcopy(source_config["official_args"])
    diagnostic_args = {
        "stage12_validation_diagnostic": stage_identity == "stage12",
        "stage22_validation_diagnostic": stage_identity in {
            "stage22", "stage22_internal", "stage22_crossmask"
        },
        "stage22_internal_features": stage_identity == "stage22_internal",
        "stage22_mcpr_attribution": stage_identity == "stage22_mcpr",
    }
    if stage_identity == "stage12":
        diagnostic_args.update({
            "stage12_checkpoint": str(checkpoint),
            "stage12_metrics_path": str(metrics_path),
            "stage12_samples_path": str(samples_path),
        })
    else:
        diagnostic_args.update({
            "stage22_validation_diagnostic": True,
            "stage22_checkpoint": str(checkpoint),
            "stage22_metrics_path": str(metrics_path),
            "stage22_samples_path": str(samples_path),
            "stage22_nested_target_rate": float(config["diagnostic"]["nested_target_rate"]),
            "stage22_nested_block_length": int(config["diagnostic"]["nested_block_length"]),
            "stage22_nested_seed": int(config["diagnostic"]["nested_seed"]),
        })
    official_args.update(
        {
            "is_training": 0,
            "seed": int(config["experiment"]["seed"]),
            "model_id": run_id,
            "root_path": str(missing_csv.parent),
            "data_path": missing_csv.name,
            "true_data_path": str(complete_csv),
            "checkpoints": str(run_dir / "checkpoints"),
            "results_root": str(run_dir / "results"),
            "test_results_root": str(run_dir / "test_results"),
            "metrics_path": str(metrics_path),
            "max_train_batches": -1,
            "max_eval_batches": int(config["diagnostic"]["max_eval_batches"]),
            "wandb_mode": "disabled",
            "gate_h_diagnostic": False,
            "hcup_diagnostic": False,
            "gpu": 0,
            **diagnostic_args,
        }
    )

    resolved = copy.deepcopy(config)
    resolved["experiment"]["id"] = run_id
    resolved["experiment"]["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    resolved["source"]["resolved_config"] = str(source_config_path)
    resolved["source"]["checkpoint"] = str(checkpoint)
    resolved["source"]["metrics"] = str(source_metrics_path)
    if backbone_metrics_path is not None:
        resolved["source"]["backbone_metrics"] = str(backbone_metrics_path)
    resolved["source"]["checkpoint_sha256"] = source_checkpoint_sha256
    resolved["data"]["mask_asset"] = str(mask_asset)
    resolved["data"]["mask_asset_sha256"] = mask_asset_sha256
    resolved["official_args"] = official_args
    resolved["paths"] = {
        "run_dir": str(run_dir),
        "config_source": str(config_path),
        "baseline_repo": str(BASELINE_ROOT),
        "upstream_repo": str(PROJECT_ROOT / "upstream" / "S4M"),
    }
    atomic_text(
        run_dir / "resolved_config.yaml",
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
    )
    atomic_text(run_dir / "env.txt", safe_environment_text(config))
    atomic_text(
        run_dir / "git_state.txt",
        "[baseline_adapter]\n"
        + git_evidence(BASELINE_ROOT)
        + "\n[upstream_snapshot]\n"
        + git_evidence(PROJECT_ROOT / "upstream" / "S4M"),
    )
    _, code_diff = run_capture(["git", "diff", "--binary"], BASELINE_ROOT)
    atomic_text(run_dir / "code.diff", code_diff)
    (run_dir / "checkpoints").mkdir()
    atomic_json(
        run_dir / "checkpoints" / "source_checkpoint.json",
        {"path": str(checkpoint), "sha256": source_checkpoint_sha256, **checkpoint_verification},
    )

    command = [sys.executable, str(S4M_ROOT / "run.py"), *cli_from_mapping(official_args)]
    python_path = os.pathsep.join(
        [
            str(S4M_ROOT),
            str(S4M_ROOT / "model"),
            str(S4M_ROOT / "model" / "S4" / "s4"),
            str(STRUCTURED_KERNEL_ROOT),
        ]
    )
    gpu_export = ""
    if config["runtime"].get("gpu_uuid"):
        gpu_export = "export CUDA_VISIBLE_DEVICES=" + shlex.quote(str(config["runtime"]["gpu_uuid"])) + "\n"
    command_text = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(S4M_ROOT))}\n"
        f"export PYTHONPATH={shlex.quote(python_path)}\n"
        + gpu_export
        + "export PYTHONDONTWRITEBYTECODE=1\n"
        + "export WANDB_MODE=disabled\n"
        + shlex.join(command)
        + "\n"
    )
    atomic_text(run_dir / "command.sh", command_text)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = python_path
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["WANDB_MODE"] = "disabled"
    environment["MPLCONFIGDIR"] = str(run_dir / ".matplotlib")
    if config["runtime"].get("gpu_uuid"):
        environment["CUDA_VISIBLE_DEVICES"] = str(config["runtime"]["gpu_uuid"])

    status: dict[str, Any] = {
        "experiment_id": run_id,
        "state": "RUNNING",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": str(config_path),
        "method": config["experiment"]["method"],
        "stage_identity": stage_identity,
        "evaluation_seed": int(config["experiment"]["seed"]),
        "device": config["runtime"]["device"],
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": source_checkpoint_sha256,
    }
    atomic_json(run_dir / "status.json", status)
    print(f"run_dir={run_dir}", flush=True)
    started = time.monotonic()
    with (run_dir / "train.log").open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=S4M_ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        status["pid"] = process.pid
        atomic_json(run_dir / "status.json", status)
        return_code = process.wait()
    elapsed = time.monotonic() - started

    metrics: dict[str, Any] = {}
    verification_error: str | None = None
    if return_code == 0 and metrics_path.is_file() and samples_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            with np.load(samples_path) as samples:
                count = int(metrics.get("sample_count", 0))
                if stage_identity in {
                    "stage22", "stage22_internal", "stage22_mcpr",
                    "stage22_crossmask",
                }:
                    vector_keys = (
                        "sample_index", "complete_count", "observed_count",
                        "target_removed_count", "removed_count",
                        "prediction_disagreement_mse", "target_abs_sum",
                        "complete_sse_original", "complete_sae_original",
                        "complete_signed_error_original", "complete_sse_nested",
                        "complete_sae_nested", "complete_signed_error_nested",
                        "complete_sse_ensemble", "complete_sae_ensemble",
                        "complete_signed_error_ensemble", "observed_sse_original",
                        "observed_sae_original",
                    )
                    finite_keys = tuple(key for key in vector_keys if key != "sample_index")
                    if stage_identity == "stage22_mcpr":
                        mcpr_vector_keys = (
                            "complete_sse_backbone", "complete_sae_backbone",
                            "complete_signed_error_backbone", "mcpr_delta_sse",
                            "mcpr_delta_sae", "mcpr_alignment_cross",
                            "mcpr_residual_energy", "mcpr_residual_abs_sum",
                            "mcpr_residual_max_abs", "backbone_abs_error_q50",
                            "backbone_abs_error_q90", "backbone_abs_error_q95",
                            "backbone_abs_error_q99", "backbone_abs_error_max",
                            "candidate_abs_error_q50", "candidate_abs_error_q90",
                            "candidate_abs_error_q95", "candidate_abs_error_q99",
                            "candidate_abs_error_max",
                        )
                        vector_keys = vector_keys + mcpr_vector_keys
                        finite_keys = finite_keys + mcpr_vector_keys
                    recalculated = {
                        "complete_target_mse_original": float(
                            samples["complete_sse_original"].sum() / samples["complete_count"].sum()
                        ),
                        "complete_target_mae_original": float(
                            samples["complete_sae_original"].sum() / samples["complete_count"].sum()
                        ),
                        "complete_target_mse_nested": float(
                            samples["complete_sse_nested"].sum() / samples["complete_count"].sum()
                        ),
                        "complete_target_mae_nested": float(
                            samples["complete_sae_nested"].sum() / samples["complete_count"].sum()
                        ),
                        "complete_target_mse_ensemble": float(
                            samples["complete_sse_ensemble"].sum() / samples["complete_count"].sum()
                        ),
                        "complete_target_mae_ensemble": float(
                            samples["complete_sae_ensemble"].sum() / samples["complete_count"].sum()
                        ),
                        "observed_target_mse_original": float(
                            samples["observed_sse_original"].sum() / samples["observed_count"].sum()
                        ),
                        "observed_target_mae_original": float(
                            samples["observed_sae_original"].sum() / samples["observed_count"].sum()
                        ),
                    }
                    if stage_identity == "stage22_mcpr":
                        recalculated.update({
                            "complete_target_mse_backbone": float(
                                samples["complete_sse_backbone"].sum()
                                / samples["complete_count"].sum()
                            ),
                            "complete_target_mae_backbone": float(
                                samples["complete_sae_backbone"].sum()
                                / samples["complete_count"].sum()
                            ),
                        })
                    history = np.unpackbits(samples["history_mask_packed"], axis=1)[:, :192]
                    nested = np.unpackbits(samples["nested_mask_packed"], axis=1)[:, :192]
                    source_roundtrip = (
                        True
                        if config["experiment"].get("smoke_test")
                        or stage_identity == "stage22_crossmask"
                        else (
                            abs(recalculated["complete_target_mse_original"] - float(source_metrics["mse"])) <= 1e-5
                            and abs(recalculated["complete_target_mae_original"] - float(source_metrics["mae"])) <= 1e-5
                        )
                    )
                    backbone_roundtrip = (
                        True
                        if stage_identity != "stage22_mcpr" or config["experiment"].get("smoke_test")
                        else (
                            abs(
                                recalculated["complete_target_mse_backbone"]
                                - float(backbone_metrics["mse"])
                            ) <= 1e-5
                            and abs(
                                recalculated["complete_target_mae_backbone"]
                                - float(backbone_metrics["mae"])
                            ) <= 1e-5
                        )
                    )
                    checks = {
                        "sample_count_positive": count > 0,
                        "all_vector_counts_match": all(int(samples[key].shape[0]) == count for key in vector_keys),
                        "packed_mask_count_matches": all(
                            int(samples[key].shape[0]) == count
                            for key in ("history_mask_packed", "nested_mask_packed")
                        ),
                        "packed_mask_width_matches": all(
                            int(samples[key].shape[1]) == 24
                            for key in ("history_mask_packed", "nested_mask_packed")
                        ),
                        "sample_index_chronological": bool(np.array_equal(samples["sample_index"], np.arange(count))),
                        "finite_error_arrays": all(np.isfinite(samples[key]).all() for key in finite_keys),
                        "positive_counts": bool((samples["complete_count"] > 0).all() and (samples["observed_count"] > 0).all()),
                        "complete_count_constant": int(np.unique(samples["complete_count"]).size) == 1,
                        "nested_mask_is_strict_subset": bool((nested <= history).all() and (nested < history).any()),
                        "equal_removed_count": bool(np.array_equal(samples["removed_count"], samples["target_removed_count"])),
                        "horizon_aggregate_shapes": all(samples[key].shape == (96,) for key in (
                            "horizon_sse", "horizon_sae", "horizon_signed_error", "horizon_count"
                        )),
                        "variable_aggregate_shapes": all(samples[key].shape == (321,) for key in (
                            "variable_sse", "variable_sae", "variable_signed_error", "variable_count"
                        )),
                        "row_shared_mask": metrics.get("row_shared_mask_all_samples") is True,
                        "chronological_loader": metrics.get("chronological_loader") is True,
                        "validation_only": metrics.get("uses_test_split") is False and metrics.get("source_split") == "val",
                        "frozen_model": metrics.get("updates_model_or_memory") is False,
                        "fixed_response_probe": (
                            float(metrics.get("nested_target_rate", -1.0)) == 0.15
                            and int(metrics.get("nested_block_length", -1)) == 5
                            and int(metrics.get("nested_seed", -1))
                            == (
                                int(config["experiment"]["seed"])
                                if stage_identity == "stage22_crossmask"
                                else 2023
                            )
                        ),
                        "metric_roundtrip": all(abs(recalculated[key] - float(metrics[key])) <= 1e-12 for key in recalculated),
                        "source_validation_roundtrip": source_roundtrip,
                    }
                    if stage_identity == "stage22_mcpr":
                        candidate_delta = (
                            samples["complete_sse_original"]
                            - samples["complete_sse_backbone"]
                        )
                        redundant_delta_abs_error = np.abs(
                            samples["mcpr_delta_sse"] - candidate_delta
                        )
                        checks.update({
                            "mcpr_attribution_declared": metrics.get(
                                "mcpr_attribution_cached"
                            ) is True,
                            "mcpr_checkpoint_contains_parameters": len(
                                checkpoint_verification["checkpoint_mcpr_state_keys"]
                            ) > 0,
                            "mcpr_sample_delta_matches": bool(np.allclose(
                                samples["mcpr_delta_sse"], candidate_delta,
                                rtol=MCPR_REDUCTION_RTOL,
                                atol=MCPR_REDUCTION_ATOL,
                            )),
                            "mcpr_sample_decomposition_closes": bool(np.allclose(
                                samples["mcpr_delta_sse"],
                                samples["mcpr_alignment_cross"]
                                + samples["mcpr_residual_energy"],
                                rtol=2e-5, atol=2e-3,
                            )),
                            "mcpr_horizon_shapes": all(samples[key].shape == (96,) for key in (
                                "horizon_sse_backbone", "horizon_sae_backbone",
                                "horizon_delta_sse", "horizon_delta_sae",
                                "horizon_alignment_cross", "horizon_residual_energy",
                            )),
                            "mcpr_variable_shapes": all(samples[key].shape == (321,) for key in (
                                "variable_sse_backbone", "variable_sae_backbone",
                                "variable_delta_sse", "variable_delta_sae",
                                "variable_alignment_cross", "variable_residual_energy",
                            )),
                            "mcpr_horizon_decomposition_closes": bool(np.allclose(
                                samples["horizon_delta_sse"],
                                samples["horizon_alignment_cross"]
                                + samples["horizon_residual_energy"],
                                rtol=2e-5, atol=2e-3,
                            )),
                            "mcpr_variable_decomposition_closes": bool(np.allclose(
                                samples["variable_delta_sse"],
                                samples["variable_alignment_cross"]
                                + samples["variable_residual_energy"],
                                rtol=2e-5, atol=2e-3,
                            )),
                            "mcpr_backbone_source_roundtrip": backbone_roundtrip,
                        })
                    if stage_identity == "stage22_internal":
                        feature_keys = sorted(
                            key for key in samples.files if key.startswith("feature_")
                        )
                        checks.update({
                            "internal_features_declared": metrics.get(
                                "internal_features_cached"
                            ) is True,
                            "internal_feature_keys_match": feature_keys == metrics.get(
                                "internal_feature_keys"
                            ),
                            "internal_feature_count_positive": len(feature_keys) > 0,
                            "internal_feature_shapes_match": all(
                                samples[key].shape == (count,) for key in feature_keys
                            ),
                            "internal_features_finite": all(
                                np.isfinite(samples[key]).all() for key in feature_keys
                            ),
                        })
                else:
                    vector_keys = (
                        "sample_index", "complete_sse", "complete_sae", "complete_count",
                        "observed_sse", "observed_sae", "observed_count",
                    )
                    recalculated = {
                        "complete_target_mse": float(samples["complete_sse"].sum() / samples["complete_count"].sum()),
                        "complete_target_mae": float(samples["complete_sae"].sum() / samples["complete_count"].sum()),
                        "observed_target_mse": float(samples["observed_sse"].sum() / samples["observed_count"].sum()),
                        "observed_target_mae": float(samples["observed_sae"].sum() / samples["observed_count"].sum()),
                    }
                    checks = {
                        "sample_count_positive": count > 0,
                        "all_vector_counts_match": all(int(samples[key].size) == count for key in vector_keys),
                        "packed_mask_count_matches": int(samples["history_mask_packed"].shape[0]) == count,
                        "packed_mask_width_matches": int(samples["history_mask_packed"].shape[1]) == 24,
                        "sample_index_chronological": bool(np.array_equal(samples["sample_index"], np.arange(count))),
                        "finite_error_arrays": all(np.isfinite(samples[key]).all() for key in vector_keys[1:]),
                        "positive_counts": bool((samples["complete_count"] > 0).all() and (samples["observed_count"] > 0).all()),
                        "complete_count_constant": int(np.unique(samples["complete_count"]).size) == 1,
                        "row_shared_mask": metrics.get("row_shared_mask_all_samples") is True,
                        "chronological_loader": metrics.get("chronological_loader") is True,
                        "validation_only": metrics.get("uses_test_split") is False and metrics.get("source_split") == "val",
                        "frozen_model": metrics.get("updates_model_or_memory") is False,
                        "metric_roundtrip": all(abs(recalculated[key] - float(metrics[key])) <= 1e-12 for key in recalculated),
                    }
            if not all(checks.values()):
                raise RuntimeError(f"{stage_identity} cache artifact checks failed: {checks}")
            metrics.update(
                {
                    "experiment_id": run_id,
                    "dataset": config["experiment"]["dataset"],
                    "method": config["experiment"]["method"],
                    "evaluation_seed": int(config["experiment"]["seed"]),
                    "scope": (
                        "smoke_test_not_formal_result"
                        if config["experiment"].get("smoke_test")
                        else f"{stage_identity}_formal_validation_only_cache"
                    ),
                    "source_validation_mse_reference_only": source_metrics.get("mse"),
                    "source_validation_mae_reference_only": source_metrics.get("mae"),
                    "backbone_validation_mse_reference_only": (
                        backbone_metrics.get("mse") if backbone_metrics else None
                    ),
                    "backbone_validation_mae_reference_only": (
                        backbone_metrics.get("mae") if backbone_metrics else None
                    ),
                    "mcpr_redundant_delta_max_abs_error": (
                        float(redundant_delta_abs_error.max())
                        if stage_identity == "stage22_mcpr" else None
                    ),
                    "mcpr_reduction_check_rtol": (
                        MCPR_REDUCTION_RTOL if stage_identity == "stage22_mcpr" else None
                    ),
                    "mcpr_reduction_check_atol": (
                        MCPR_REDUCTION_ATOL if stage_identity == "stage22_mcpr" else None
                    ),
                    "source_formal_test_mse_reference_only": source_metrics.get("mse"),
                    "source_checkpoint": str(checkpoint),
                    "source_checkpoint_sha256": source_checkpoint_sha256,
                    "mask_asset": str(mask_asset),
                    "mask_asset_sha256": mask_asset_sha256,
                    "artifact_checks": checks,
                    "artifact_status": "PASS",
                    **checkpoint_verification,
                }
            )
            atomic_json(metrics_path, metrics)
        except Exception as exc:
            verification_error = f"{type(exc).__name__}: {exc}"

    log_text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace").lower()
    failure_state = "CODE_ERROR"
    if return_code != 0 and ("out of memory" in log_text or "cuda error: out of memory" in log_text):
        failure_state = "OOM"
    success = (
        return_code == 0
        and metrics_path.is_file()
        and samples_path.is_file()
        and verification_error is None
        and metrics.get("artifact_status") == "PASS"
    )
    atomic_json(
        run_dir / "resource_usage.json",
        {
            "elapsed_seconds": elapsed,
            "device": config["runtime"]["device"],
            "gpu_uuid": config["runtime"].get("gpu_uuid"),
            "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
            "peak_gpu_reserved_mb": metrics.get("peak_gpu_reserved_mb"),
            "return_code": return_code,
        },
    )
    status.update(
        {
            "state": "SUCCESS" if success else failure_state,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "return_code": return_code,
            "metrics_path": str(metrics_path) if metrics_path.is_file() else None,
            "samples_path": str(samples_path) if samples_path.is_file() else None,
            "verification_error": verification_error,
            **checkpoint_verification,
        }
    )
    atomic_json(run_dir / "status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    print(f"run_dir={run_dir}")
    return 0 if success else (return_code or 2)


if __name__ == "__main__":
    raise SystemExit(main())
