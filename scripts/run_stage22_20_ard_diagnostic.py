#!/usr/bin/env python3
"""Run one auditable validation-only Step 22.20 ARD diagnostic."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

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


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines" / "S4M"
S4M_ROOT = BASELINE_ROOT / "s4m"
STRUCTURED_KERNEL_ROOT = S4M_ROOT / "model/S4/s4/extensions/kernels"
EXPECTED_MASK = (
    ROOT / "artifacts/masks/electricity/point/rate_0.06/seed_2025.npz"
).resolve()
EXPECTED_SOURCE_ID = (
    "ard_mcpr_electricity__electricity__point__mr0.06__seed2025"
    "__20260814_182752"
)
EXPECTED_ARRAY_WIDTHS = {
    "global8": 8,
    "structured108": 108,
    "action_logits": 3,
    "action_probabilities": 3,
}


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config: dict[str, Any], source: dict[str, Any]) -> None:
    experiment = config["experiment"]
    diagnostic = config["diagnostic"]
    if experiment.get("method") != "ard_mcpr":
        raise ValueError("Step 22.20 source must be ARD-MCPR")
    if experiment.get("dataset") != "electricity" or int(
        experiment.get("seed", -1)
    ) != 2025:
        raise ValueError("Step 22.20 is frozen to Electricity seed2025")
    if diagnostic.get("split") != "val":
        raise ValueError("Step 22.20 must remain validation-only")
    if project_path(config["data"]["mask_asset"]) != EXPECTED_MASK:
        raise ValueError("Step 22.20 must use the frozen seed2025 mask")
    if int(diagnostic.get("global_feature_dim", -1)) != 8:
        raise ValueError("Step 22.20 global feature width is frozen at 8")
    if int(diagnostic.get("structured_feature_dim", -1)) != 108:
        raise ValueError("Step 22.20 structured feature width is frozen at 108")
    max_batches = int(diagnostic["max_eval_batches"])
    if experiment.get("smoke_test"):
        if max_batches != 3:
            raise ValueError("Step 22.20 smoke is frozen to three batches")
    elif max_batches != -1:
        raise ValueError("Step 22.20 formal must use full validation")

    official = source["official_args"]
    identity_checks = {
        "mipp": bool(official.get("use_mipp", False)),
        "mcpr": bool(official.get("use_mcpr", False)),
        "ard": bool(official.get("use_ard", False)),
        "analytic_objective": (
            official.get("mcpr_objective") == "analytic_decision_distill"
        ),
        "not_ttrr": not bool(official.get("use_ttrr", False)),
        "batch16": int(official.get("batch_size", -1)) == 16,
        "seq192": int(official.get("seq_len", -1)) == 192,
        "pred96": int(official.get("pred_len", -1)) == 96,
        "seed2025": int(official.get("seed", -1)) == 2025,
    }
    if not all(identity_checks.values()):
        raise ValueError(f"source ARD identity failed: {identity_checks}")


def _all_finite(array: np.ndarray) -> bool:
    return bool(np.isfinite(array).all())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-uuid", default=None)
    parser.add_argument("--scheduler-attempt", type=int, default=0)
    args = parser.parse_args()

    config_path = project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_config_path = project_path(config["source"]["resolved_config"])
    source_metrics_path = project_path(config["source"]["metrics"])
    checkpoint = project_path(config["source"]["checkpoint"])
    missing_csv = project_path(config["data"]["missing_csv"])
    complete_csv = project_path(config["data"]["complete_csv"])
    mask_asset = project_path(config["data"]["mask_asset"])
    for required in (
        config_path,
        source_config_path,
        source_metrics_path,
        checkpoint,
        missing_csv,
        complete_csv,
        mask_asset,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"required diagnostic asset is missing: {required}")
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    validate_config(config, source_config)
    if source_metrics.get("experiment_id") != EXPECTED_SOURCE_ID:
        raise ValueError("Step 22.20 source experiment identity changed")
    if Path(source_metrics.get("checkpoint", "")).resolve() != checkpoint:
        raise ValueError("source metrics and diagnostic checkpoint differ")

    if args.gpu_uuid:
        config["runtime"]["gpu_uuid"] = args.gpu_uuid
    if config["runtime"].get("device") != "cuda":
        raise ValueError("Step 22.20 GPU cache requires CUDA")
    config["runtime"].update(
        resolve_gpu_metadata(str(config["runtime"]["gpu_uuid"]))
    )
    config["runtime"]["scheduler_attempt"] = int(args.scheduler_attempt)

    smoke = bool(config["experiment"].get("smoke_test"))
    smoke_suffix = "__smoke" if smoke else ""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = (
        "stage22_20_ard_information__electricity__validation__seed2025"
        f"{smoke_suffix}__{timestamp}"
    )
    run_dir = ROOT / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"experiment ID already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    metrics_path = run_dir / "metrics.json"
    samples_path = run_dir / "ard_information_samples.npz"

    checkpoint_hash = sha256(checkpoint)
    checkpoint_verification = verify_checkpoint(
        checkpoint,
        require_mipp=True,
        require_mcpr=True,
        require_ard=True,
    )
    official_args = copy.deepcopy(source_config["official_args"])
    official_args.update({
        "is_training": 0,
        "seed": 2025,
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
        "stage12_validation_diagnostic": False,
        "stage22_validation_diagnostic": False,
        "stage22_inference_timing": False,
        "stage22_ard_information_diagnostic": True,
        "stage22_ard_diagnostic_checkpoint": str(checkpoint),
        "stage22_ard_diagnostic_metrics_path": str(metrics_path),
        "stage22_ard_diagnostic_samples_path": str(samples_path),
        "stage14_uncertainty_diagnostic": False,
        "stage15_nested_ensemble_test": False,
        "stage18_equal_rate_multiscale_validation": False,
        "stage19_equal_rate_multiscale_test": False,
        "stage21_equal_rate_multiscale_evaluation": False,
        "gpu": 0,
    })

    resolved = copy.deepcopy(config)
    resolved["experiment"]["id"] = run_id
    resolved["experiment"]["created_at"] = dt.datetime.now(
        dt.timezone.utc
    ).isoformat()
    resolved["source"].update({
        "resolved_config": str(source_config_path),
        "metrics": str(source_metrics_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
    })
    resolved["data"].update({
        "missing_csv": str(missing_csv),
        "complete_csv": str(complete_csv),
        "mask_asset": str(mask_asset),
        "mask_asset_sha256": sha256(mask_asset),
    })
    resolved["official_args"] = official_args
    resolved["paths"] = {
        "run_dir": str(run_dir),
        "config_source": str(config_path),
        "baseline_repo": str(BASELINE_ROOT),
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
        + git_evidence(ROOT / "upstream/S4M"),
    )
    _, diff = run_capture(["git", "diff", "--binary"], BASELINE_ROOT)
    atomic_text(run_dir / "code.diff", diff)
    atomic_json(
        run_dir / "checkpoints/source_checkpoint.json",
        {"path": str(checkpoint), "sha256": checkpoint_hash, **checkpoint_verification},
    )

    command = [
        sys.executable,
        str(S4M_ROOT / "run.py"),
        *cli_from_mapping(official_args),
    ]
    python_path = os.pathsep.join([
        str(S4M_ROOT),
        str(S4M_ROOT / "model"),
        str(S4M_ROOT / "model/S4/s4"),
        str(STRUCTURED_KERNEL_ROOT),
    ])
    gpu_uuid = str(config["runtime"]["gpu_uuid"])
    command_text = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(S4M_ROOT))}\n"
        f"export PYTHONPATH={shlex.quote(python_path)}\n"
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(gpu_uuid)}\n"
        "export PYTHONDONTWRITEBYTECODE=1\n"
        "export WANDB_MODE=disabled\n"
        + shlex.join(command)
        + "\n"
    )
    atomic_text(run_dir / "command.sh", command_text)
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": python_path,
        "PYTHONDONTWRITEBYTECODE": "1",
        "WANDB_MODE": "disabled",
        "MPLCONFIGDIR": str(run_dir / ".matplotlib"),
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
    })

    status: dict[str, Any] = {
        "experiment_id": run_id,
        "state": "RUNNING",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": str(config_path),
        "evaluation_seed": 2025,
        "device": "cuda",
        "gpu_uuid": gpu_uuid,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": checkpoint_hash,
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

    verification_error = None
    metrics: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    if return_code == 0 and metrics_path.is_file() and samples_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            with np.load(samples_path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            expected_batches = 3 if smoke else 158
            expected_samples = expected_batches * 16
            expected_keys = {
                "sample_index", "global8", "structured108", "action_logits",
                "action_probabilities", "partial_alpha", "student_gate",
                "complete_teacher_alpha", "complete_teacher_actions",
                "complete_teacher_valid", "observed_teacher_alpha",
                "observed_teacher_actions", "observed_teacher_valid",
                "complete_count", "observed_count", "backbone_sse",
                "backbone_sae", "raw_sse", "raw_sae", "ard_sse", "ard_sae",
                "alignment_cross", "residual_energy",
            }
            probabilities = arrays["action_probabilities"]
            reconstructed_gate = (
                probabilities[:, 2]
                + probabilities[:, 1] * arrays["partial_alpha"]
            )
            quadratic_sse = (
                arrays["backbone_sse"]
                + arrays["student_gate"] * arrays["alignment_cross"]
                + np.square(arrays["student_gate"]) * arrays["residual_energy"]
            )
            valid = arrays["complete_teacher_valid"].astype(bool)
            reconstructed_alpha = np.clip(
                -arrays["alignment_cross"][valid]
                / (2.0 * arrays["residual_energy"][valid]),
                0.0,
                1.0,
            )
            reference_mse = float(source_metrics[config["source"]["mse_key"]])
            reference_mae = float(source_metrics[config["source"]["mae_key"]])
            checks = {
                "validation_only": (
                    metrics.get("test_split_accessed") is False
                    and metrics.get("uses_test_split") is False
                    and metrics.get("source_split") == "val"
                ),
                "chronological_loader": metrics.get("chronological_loader") is True,
                "frozen_model_and_memory": (
                    metrics.get("updates_model_or_memory") is False
                ),
                "expected_batch_count": int(
                    metrics.get("validation_batches_processed", 0)
                ) == expected_batches,
                "expected_sample_count": (
                    int(metrics.get("validation_samples_processed", 0))
                    == expected_samples
                ),
                "exact_array_keys": set(arrays) == expected_keys,
                "all_array_rows_complete": all(
                    value.shape[0] == expected_samples for value in arrays.values()
                ),
                "array_widths": all(
                    arrays[key].shape == (expected_samples, width)
                    for key, width in EXPECTED_ARRAY_WIDTHS.items()
                ),
                "sample_index_chronological": np.array_equal(
                    arrays["sample_index"], np.arange(expected_samples)
                ),
                "all_arrays_finite": all(_all_finite(value) for value in arrays.values()),
                "structured_prefix_exact": np.array_equal(
                    arrays["global8"], arrays["structured108"][:, :8]
                ),
                "action_probabilities_valid": (
                    bool((probabilities >= 0.0).all())
                    and bool((probabilities <= 1.0).all())
                    and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
                ),
                "gate_mixture_identity": np.allclose(
                    reconstructed_gate, arrays["student_gate"], atol=1e-6, rtol=1e-6
                ),
                "teacher_alpha_identity": np.allclose(
                    reconstructed_alpha,
                    arrays["complete_teacher_alpha"][valid],
                    atol=1e-6,
                    rtol=1e-6,
                ),
                "quadratic_residual_identity": np.allclose(
                    quadratic_sse,
                    arrays["ard_sse"],
                    atol=0.008,
                    rtol=2e-5,
                ),
                "metric_recomputed_from_cache": (
                    abs(
                        float(metrics["mse"])
                        - float(arrays["ard_sse"].sum() / arrays["complete_count"].sum())
                    ) <= 1e-12
                    and abs(
                        float(metrics["mae"])
                        - float(arrays["ard_sae"].sum() / arrays["complete_count"].sum())
                    ) <= 1e-12
                ),
                "formal_metric_roundtrip": (
                    smoke
                    or (
                        abs(float(metrics["mse"]) - reference_mse) <= 1e-5
                        and abs(float(metrics["mae"]) - reference_mae) <= 1e-5
                    )
                ),
                "checkpoint_finite": checkpoint_verification[
                    "checkpoint_all_model_tensors_finite"
                ] is True,
                "checkpoint_ard_identity": bool(
                    checkpoint_verification["checkpoint_ard_state_keys"]
                ),
            }
            if not all(checks.values()):
                raise RuntimeError(f"ARD diagnostic artifact checks failed: {checks}")
            metrics.update({
                "experiment_id": run_id,
                "dataset": "electricity",
                "method": "ard_mcpr_information_diagnostic",
                "evaluation_seed": 2025,
                "scope": (
                    "smoke_test_not_formal_result"
                    if smoke
                    else "full_validation_frozen_ard_information_cache"
                ),
                "gpu_uuid": gpu_uuid,
                "source_validation_mse_reference": reference_mse,
                "source_validation_mae_reference": reference_mae,
                "source_checkpoint": str(checkpoint),
                "source_checkpoint_sha256": checkpoint_hash,
                "mask_asset": str(mask_asset),
                "mask_asset_sha256": sha256(mask_asset),
                "samples_path": str(samples_path),
                "artifact_checks": checks,
                "artifact_status": "PASS",
                **checkpoint_verification,
            })
            atomic_json(metrics_path, metrics)
        except Exception as exc:
            verification_error = f"{type(exc).__name__}: {exc}"

    log_text = (run_dir / "train.log").read_text(
        encoding="utf-8", errors="replace"
    ).lower()
    failure_state = "CODE_ERROR"
    if return_code != 0 and "out of memory" in log_text:
        failure_state = "OOM"
    success = (
        return_code == 0
        and metrics_path.is_file()
        and samples_path.is_file()
        and verification_error is None
        and metrics.get("artifact_status") == "PASS"
    )
    atomic_json(run_dir / "resource_usage.json", {
        "elapsed_seconds": float(elapsed),
        "device": "cuda",
        "gpu_uuid": gpu_uuid,
        "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
        "peak_gpu_reserved_mb": metrics.get("peak_gpu_reserved_mb"),
        "return_code": int(return_code),
    })
    status.update({
        "state": "SUCCESS" if success else failure_state,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "return_code": int(return_code),
        "metrics_path": str(metrics_path) if metrics_path.is_file() else None,
        "samples_path": str(samples_path) if samples_path.is_file() else None,
        "verification_error": verification_error,
        "artifact_checks": checks,
        **checkpoint_verification,
    })
    atomic_json(run_dir / "status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    print(f"run_dir={run_dir}")
    return 0 if success else (return_code or 2)


if __name__ == "__main__":
    raise SystemExit(main())
