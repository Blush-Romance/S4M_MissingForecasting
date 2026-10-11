#!/usr/bin/env python3
"""Run one auditable validation-only Step 22.27 BF-CMVD diagnostic."""

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
BASELINE_ROOT = ROOT / "baselines/S4M"
S4M_ROOT = BASELINE_ROOT / "s4m"
STRUCTURED_KERNEL_ROOT = S4M_ROOT / "model/S4/s4/extensions/kernels"
EXPECTED_STUDENT_ID = (
    "bf_cmvd_v2_pg95_traffic__traffic__point__mr0.06__seed2027"
    "__20260820_212927"
)
EXPECTED_BACKBONE_ID = (
    "mipp_pg95_traffic__traffic__point__mr0.06__seed2027"
    "__20260820_174327"
)
EXPECTED_TEACHER_ID = (
    "cnm_se_equal_rate_multiscale_pg95__traffic__validation__seed2027"
    "__20260820_184433"
)
FLOAT32_RESIDUAL_IDENTITY_ATOL = float(8.0 * np.finfo(np.float32).eps)
TOKEN_KEYS = {
    "observed_token", "complete_teacher_beneficial",
    "observed_teacher_beneficial", "complete_student_beneficial",
    "backbone_mse", "teacher_mse", "student_mse",
    "backbone_mae", "teacher_mae", "student_mae",
    "observed_backbone_mse", "observed_teacher_mse",
    "observed_backbone_mae", "observed_teacher_mae",
    "correction_dot", "student_correction_energy",
    "teacher_correction_energy", "correction_mismatch_energy",
    "correction_sign_agreement_count", "correction_sign_eligible_count",
}
VARIABLE_KEYS = {
    f"variable_{label}_{suffix}"
    for label in ("backbone", "teacher", "student")
    for suffix in ("sse", "sae")
} | {
    "variable_correction_dot", "variable_teacher_correction_energy",
    "variable_student_correction_energy",
    "variable_correction_mismatch_energy",
}
STRUCTURAL_KEYS = {
    "sample_index", "history_mask_packed", "l5_mask_packed",
    "l20_mask_packed",
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(config: dict[str, Any], student_config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    diagnostic = config["diagnostic"]
    if experiment.get("method") != "bf_cmvd_v2":
        raise ValueError("Step 22.27 source method must be BF-CMVD-v2")
    if experiment.get("dataset") != "traffic" or int(
        experiment.get("seed", -1)
    ) != 2027:
        raise ValueError("Step 22.27 is frozen to Traffic seed2027")
    if diagnostic.get("split") != "val":
        raise ValueError("Step 22.27 must remain validation-only")
    if float(diagnostic.get("target_rate", -1)) != 0.15:
        raise ValueError("Step 22.27 target rate is frozen at 0.15")
    if diagnostic.get("block_lengths") != [5, 20]:
        raise ValueError("Step 22.27 block lengths are frozen at [5,20]")
    if int(diagnostic.get("nested_seed", -1)) != 2027:
        raise ValueError("Step 22.27 nested seed is frozen at 2027")
    if diagnostic.get("teacher_weights") != {
        "original": 0.5, "l5": 0.25, "l20": 0.25,
    }:
        raise ValueError("Step 22.27 teacher weights changed")
    max_batches = int(diagnostic["max_eval_batches"])
    if bool(experiment.get("smoke_test")):
        if max_batches != 3:
            raise ValueError("Step 22.27 smoke is frozen to three batches")
    elif max_batches != -1:
        raise ValueError("Step 22.27 formal must use full validation")

    official = student_config["official_args"]
    identity = {
        "mipp": bool(official.get("use_mipp", False)),
        "mcpr": bool(official.get("use_mcpr", False)),
        "bf_cmvd": bool(official.get("use_bf_cmvd", False)),
        "token_objective": (
            official.get("mcpr_objective")
            == "token_benefit_filtered_multiview_distill"
        ),
        "not_ttrr": not bool(official.get("use_ttrr", False)),
        "not_ard": not bool(official.get("use_ard", False)),
        "batch16": int(official.get("batch_size", -1)) == 16,
        "seq192": int(official.get("seq_len", -1)) == 192,
        "pred96": int(official.get("pred_len", -1)) == 96,
        "variables862": int(official.get("c_out", -1)) == 862,
        "seed2027": int(official.get("seed", -1)) == 2027,
    }
    if not all(identity.values()):
        raise ValueError(f"BF-CMVD-v2 source identity failed: {identity}")


def all_finite(array: np.ndarray) -> bool:
    return bool(np.isfinite(array).all())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-uuid", default=None)
    parser.add_argument("--scheduler-attempt", type=int, default=0)
    args = parser.parse_args()

    config_path = project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    student_config_path = project_path(config["source"]["student_resolved_config"])
    student_metrics_path = project_path(config["source"]["student_metrics"])
    student_checkpoint = project_path(config["source"]["student_checkpoint"])
    backbone_metrics_path = project_path(config["source"]["backbone_metrics"])
    backbone_checkpoint = project_path(config["source"]["backbone_checkpoint"])
    teacher_metrics_path = project_path(config["source"]["teacher_metrics"])
    teacher_samples_path = project_path(config["source"]["teacher_samples"])
    missing_csv = project_path(config["data"]["missing_csv"])
    complete_csv = project_path(config["data"]["complete_csv"])
    mask_asset = project_path(config["data"]["mask_asset"])
    required_paths = (
        config_path, student_config_path, student_metrics_path,
        student_checkpoint, backbone_metrics_path, backbone_checkpoint,
        teacher_metrics_path, teacher_samples_path, missing_csv, complete_csv,
        mask_asset,
    )
    for required in required_paths:
        if not required.is_file():
            raise FileNotFoundError(f"required Step 22.27 asset is missing: {required}")

    student_config = yaml.safe_load(student_config_path.read_text(encoding="utf-8"))
    student_metrics = load_json(student_metrics_path)
    backbone_metrics = load_json(backbone_metrics_path)
    teacher_metrics = load_json(teacher_metrics_path)
    validate_config(config, student_config)
    if student_metrics.get("experiment_id") != EXPECTED_STUDENT_ID:
        raise ValueError("Step 22.27 student experiment identity changed")
    if backbone_metrics.get("experiment_id") != EXPECTED_BACKBONE_ID:
        raise ValueError("Step 22.27 backbone experiment identity changed")
    if teacher_metrics.get("experiment_id") != EXPECTED_TEACHER_ID:
        raise ValueError("Step 22.27 teacher experiment identity changed")
    if Path(student_metrics.get("checkpoint", "")).resolve() != student_checkpoint:
        raise ValueError("student metrics and checkpoint differ")
    if Path(backbone_metrics.get("checkpoint", "")).resolve() != backbone_checkpoint:
        raise ValueError("backbone metrics and checkpoint differ")
    backbone_hash = sha256(backbone_checkpoint)
    if student_metrics.get("mcpr_source_checkpoint_sha256") != backbone_hash:
        raise ValueError("student checkpoint does not identify the frozen PG95 source")

    if args.gpu_uuid:
        config["runtime"]["gpu_uuid"] = args.gpu_uuid
    if config["runtime"].get("device") != "cuda":
        raise ValueError("Step 22.27 GPU diagnostic requires CUDA")
    config["runtime"].update(
        resolve_gpu_metadata(str(config["runtime"]["gpu_uuid"]))
    )
    config["runtime"]["scheduler_attempt"] = int(args.scheduler_attempt)

    smoke = bool(config["experiment"].get("smoke_test"))
    smoke_suffix = "__smoke" if smoke else ""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = (
        "bf_cmvd_v2_failure_attribution__traffic__validation__seed2027"
        f"{smoke_suffix}__{timestamp}"
    )
    run_dir = ROOT / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"experiment ID already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    metrics_path = run_dir / "metrics.json"
    samples_path = run_dir / "bf_cmvd_failure_tokens.npz"

    student_hash = sha256(student_checkpoint)
    checkpoint_verification = verify_checkpoint(
        student_checkpoint, require_mipp=True, require_mcpr=True
    )
    backbone_verification = verify_checkpoint(backbone_checkpoint, require_mipp=True)
    if checkpoint_verification["checkpoint_ttrr_state_keys"]:
        raise ValueError("Step 22.27 student unexpectedly contains TTRR parameters")
    if checkpoint_verification["checkpoint_ard_state_keys"]:
        raise ValueError("Step 22.27 student unexpectedly contains ARD parameters")

    official_args = copy.deepcopy(student_config["official_args"])
    official_args.update({
        "is_training": 0,
        "seed": 2027,
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
        "stage22_ard_information_diagnostic": False,
        "stage22_27_bf_cmvd_diagnostic": True,
        "stage22_27_checkpoint": str(student_checkpoint),
        "stage22_27_metrics_path": str(metrics_path),
        "stage22_27_samples_path": str(samples_path),
        "stage22_27_target_rate": 0.15,
        "stage22_27_block_lengths": [5, 20],
        "stage22_27_nested_seed": 2027,
        "stage22_27_original_weight": 0.5,
        "stage22_27_l5_weight": 0.25,
        "stage22_27_l20_weight": 0.25,
        "stage14_uncertainty_diagnostic": False,
        "stage15_nested_ensemble_test": False,
        "stage18_equal_rate_multiscale_validation": False,
        "stage19_equal_rate_multiscale_test": False,
        "stage21_equal_rate_multiscale_evaluation": False,
        "stage22_24_batch_parallel_comparison": False,
        "stage22_25_cuda_stream_comparison": False,
        "gpu": 0,
    })

    resolved = copy.deepcopy(config)
    resolved["experiment"]["id"] = run_id
    resolved["experiment"]["created_at"] = dt.datetime.now(
        dt.timezone.utc
    ).isoformat()
    resolved["source"].update({
        "student_resolved_config": str(student_config_path),
        "student_metrics": str(student_metrics_path),
        "student_checkpoint": str(student_checkpoint),
        "student_checkpoint_sha256": student_hash,
        "backbone_metrics": str(backbone_metrics_path),
        "backbone_checkpoint": str(backbone_checkpoint),
        "backbone_checkpoint_sha256": backbone_hash,
        "teacher_metrics": str(teacher_metrics_path),
        "teacher_samples": str(teacher_samples_path),
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
        "[baseline_adapter]\n" + git_evidence(BASELINE_ROOT)
        + "\n[upstream_snapshot]\n" + git_evidence(ROOT / "upstream/S4M"),
    )
    _, diff = run_capture(["git", "diff", "--binary"], BASELINE_ROOT)
    atomic_text(run_dir / "code.diff", diff)
    atomic_json(run_dir / "checkpoints/source_checkpoints.json", {
        "student": {
            "path": str(student_checkpoint), "sha256": student_hash,
            **checkpoint_verification,
        },
        "backbone": {
            "path": str(backbone_checkpoint), "sha256": backbone_hash,
            **backbone_verification,
        },
    })

    command = [sys.executable, str(S4M_ROOT / "run.py"), *cli_from_mapping(official_args)]
    python_path = os.pathsep.join([
        str(S4M_ROOT), str(S4M_ROOT / "model"), str(S4M_ROOT / "model/S4/s4"),
        str(STRUCTURED_KERNEL_ROOT),
    ])
    gpu_uuid = str(config["runtime"]["gpu_uuid"])
    command_text = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(S4M_ROOT))}\n"
        f"export PYTHONPATH={shlex.quote(python_path)}\n"
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(gpu_uuid)}\n"
        "export PYTHONDONTWRITEBYTECODE=1\nexport WANDB_MODE=disabled\n"
        + shlex.join(command) + "\n"
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
        "evaluation_seed": 2027,
        "device": "cuda",
        "gpu_uuid": gpu_uuid,
        "source_checkpoint": str(student_checkpoint),
        "source_checkpoint_sha256": student_hash,
    }
    atomic_json(run_dir / "status.json", status)
    print(f"run_dir={run_dir}", flush=True)
    started = time.monotonic()
    with (run_dir / "train.log").open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command, cwd=S4M_ROOT, env=environment, stdout=log_file,
            stderr=subprocess.STDOUT, text=True,
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
            metrics = load_json(metrics_path)
            with np.load(samples_path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            with np.load(teacher_samples_path, allow_pickle=False) as archive:
                reference_masks = {
                    key: archive[key]
                    for key in (
                        "history_mask_packed", "l5_mask_packed", "l20_mask_packed"
                    )
                }
            expected_batches = 3 if smoke else 103
            expected_samples = expected_batches * 16
            expected_keys = TOKEN_KEYS | VARIABLE_KEYS | STRUCTURAL_KEYS
            observed = arrays["observed_token"].astype(bool)
            complete_benefit = arrays["complete_teacher_beneficial"].astype(bool)
            observed_benefit = arrays["observed_teacher_beneficial"].astype(bool)
            reconstructed_complete = (
                (arrays["teacher_mse"] < arrays["backbone_mse"])
                & (arrays["teacher_mae"] < arrays["backbone_mae"])
            )
            reconstructed_student = (
                (arrays["student_mse"] < arrays["backbone_mse"])
                & (arrays["student_mae"] < arrays["backbone_mae"])
            )
            mismatch_identity = (
                arrays["student_correction_energy"]
                + arrays["teacher_correction_energy"]
                - 2.0 * arrays["correction_dot"]
            )
            checks = {
                "validation_only": (
                    metrics.get("test_split_accessed") is False
                    and metrics.get("uses_test_split") is False
                    and metrics.get("source_split") == "val"
                ),
                "chronological_loader": metrics.get("chronological_loader") is True,
                "frozen_model_and_memory": metrics.get("updates_model_or_memory") is False,
                "expected_batch_count": int(
                    metrics.get("validation_batches_processed", 0)
                ) == expected_batches,
                "expected_sample_count": int(
                    metrics.get("validation_samples_processed", 0)
                ) == expected_samples,
                "exact_array_keys": set(arrays) == expected_keys,
                "token_shapes": all(
                    arrays[key].shape == (expected_samples, 96) for key in TOKEN_KEYS
                ),
                "variable_shapes": all(
                    arrays[key].shape == (862,) for key in VARIABLE_KEYS
                ),
                "packed_shapes": all(
                    arrays[key].shape == (expected_samples, 24)
                    for key in (
                        "history_mask_packed", "l5_mask_packed", "l20_mask_packed"
                    )
                ),
                "sample_index_chronological": np.array_equal(
                    arrays["sample_index"], np.arange(expected_samples)
                ),
                "all_arrays_finite": all(all_finite(value) for value in arrays.values()),
                "teacher_benefit_rule_roundtrip": np.array_equal(
                    complete_benefit, reconstructed_complete
                ),
                "student_benefit_rule_roundtrip": np.array_equal(
                    arrays["complete_student_beneficial"].astype(bool),
                    reconstructed_student,
                ),
                "observed_benefit_is_complete_subset": np.array_equal(
                    observed_benefit, complete_benefit & observed
                ),
                "correction_quadratic_identity": np.allclose(
                    arrays["correction_mismatch_energy"], mismatch_identity,
                    atol=2e-5, rtol=2e-5,
                ),
                "sign_counts_valid": bool(np.all(
                    arrays["correction_sign_agreement_count"]
                    <= arrays["correction_sign_eligible_count"]
                )),
                "fixed_mask_identity": all(np.array_equal(
                    arrays[key], reference_masks[key][:expected_samples]
                ) for key in reference_masks),
                "student_residual_identity": float(
                    metrics.get("student_backbone_residual_identity_max_abs", 1.0)
                ) <= FLOAT32_RESIDUAL_IDENTITY_ATOL,
                "metric_roundtrip": all(
                    abs(float(metrics[metric_key]) - float(arrays[array_key].mean()))
                    <= 1e-7
                    for metric_key, array_key in (
                        ("complete_target_mse_backbone", "backbone_mse"),
                        ("complete_target_mae_backbone", "backbone_mae"),
                        ("complete_target_mse_teacher", "teacher_mse"),
                        ("complete_target_mae_teacher", "teacher_mae"),
                        ("complete_target_mse_student", "student_mse"),
                        ("complete_target_mae_student", "student_mae"),
                    )
                ),
                "formal_reference_roundtrip": smoke or all(
                    abs(float(metrics[key]) - float(reference)) <= 1e-5
                    for key, reference in (
                        ("complete_target_mse_backbone", backbone_metrics["mse"]),
                        ("complete_target_mae_backbone", backbone_metrics["mae"]),
                        ("complete_target_mse_teacher", teacher_metrics["mse"]),
                        ("complete_target_mae_teacher", teacher_metrics["mae"]),
                        ("complete_target_mse_student", student_metrics["mse"]),
                        ("complete_target_mae_student", student_metrics["mae"]),
                    )
                ),
                "checkpoint_finite": checkpoint_verification[
                    "checkpoint_all_model_tensors_finite"
                ] is True,
                "checkpoint_bf_identity": bool(
                    checkpoint_verification["checkpoint_mcpr_state_keys"]
                    and checkpoint_verification["checkpoint_mipp_state_keys"]
                ),
                "compact_cache": (
                    metrics.get("compact_cache_excludes_full_predictions") is True
                    and samples_path.stat().st_size < 64 * 1024 * 1024
                    and not any("prediction" in key for key in arrays)
                ),
            }
            if not all(checks.values()):
                raise RuntimeError(f"Step 22.27 artifact checks failed: {checks}")
            metrics.update({
                "experiment_id": run_id,
                "dataset": "traffic",
                "method": "bf_cmvd_v2_failure_attribution",
                "evaluation_seed": 2027,
                "scope": (
                    "smoke_test_not_formal_result" if smoke
                    else "full_validation_frozen_failure_attribution"
                ),
                "gpu_uuid": gpu_uuid,
                "student_checkpoint": str(student_checkpoint),
                "student_checkpoint_sha256": student_hash,
                "backbone_checkpoint": str(backbone_checkpoint),
                "backbone_checkpoint_sha256": backbone_hash,
                "teacher_metrics": str(teacher_metrics_path),
                "teacher_samples": str(teacher_samples_path),
                "mask_asset": str(mask_asset),
                "mask_asset_sha256": sha256(mask_asset),
                "samples_path": str(samples_path),
                "artifact_checks": checks,
                "artifact_status": "PASS",
                "student_residual_identity_atol": float(
                    FLOAT32_RESIDUAL_IDENTITY_ATOL
                ),
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
        return_code == 0 and metrics_path.is_file() and samples_path.is_file()
        and verification_error is None and metrics.get("artifact_status") == "PASS"
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
