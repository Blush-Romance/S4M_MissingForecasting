#!/usr/bin/env python3
"""Run one auditable validation-only Step 22.29 DMRB diagnostic."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
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
EXPECTED_DMRB_ID = (
    "dmrb_pg95_traffic__traffic__point__mr0.06__seed2027__20260821_233052"
)
EXPECTED_BACKBONE_ID = (
    "mipp_pg95_traffic__traffic__point__mr0.06__seed2027__20260820_174327"
)
EXPECTED_TEACHER_ID = (
    "cnm_se_equal_rate_multiscale_pg95__traffic__validation__seed2027"
    "__20260820_184433"
)
FACTORIZATION_ATOL = float(32.0 * np.finfo(np.float32).eps)
BOOLEAN_KEYS = {
    "teacher_l5_beneficial", "teacher_l20_beneficial",
    "teacher_joint_beneficial", "student_l5_beneficial",
    "student_l20_beneficial", "student_joint_beneficial",
}
LOSS_KEYS = {
    f"{label}_{metric}"
    for label in (
        "backbone", "teacher_l5", "teacher_l20", "teacher_joint",
        "student_l5", "student_l20", "student_joint",
    )
    for metric in ("mse", "mae")
}
GEOMETRY_KEYS = {
    "l5_self_dot", "l5_self_residual_energy", "l5_self_target_energy",
    "l5_self_mismatch_energy", "l20_self_dot",
    "l20_self_residual_energy", "l20_self_target_energy",
    "l20_self_mismatch_energy", "l5_to_l20_target_dot",
    "l20_to_l5_target_dot", "teacher_component_dot",
    "student_branch_dot", "teacher_sum_energy", "student_sum_energy",
}
TOKEN_KEYS = BOOLEAN_KEYS | LOSS_KEYS | GEOMETRY_KEYS
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


def all_finite(array: np.ndarray) -> bool:
    return bool(np.isfinite(array).all())


def dmrb_branch_keys_verified(verification: dict[str, Any]) -> bool:
    keys = verification.get("checkpoint_mcpr_state_keys", [])
    return bool(
        any("mcpr.l5_basis." in key for key in keys)
        and any("mcpr.l20_basis." in key for key in keys)
    )


def validate_config(config: dict[str, Any], source: dict[str, Any]) -> None:
    experiment = config["experiment"]
    diagnostic = config["diagnostic"]
    if experiment.get("method") != "dmrb":
        raise ValueError("Step 22.29 source method must be DMRB")
    if experiment.get("dataset") != "traffic" or int(
        experiment.get("seed", -1)
    ) != 2027:
        raise ValueError("Step 22.29 is frozen to Traffic seed2027")
    if diagnostic.get("split") != "val":
        raise ValueError("Step 22.29 must remain validation-only")
    if float(diagnostic.get("target_rate", -1)) != 0.15:
        raise ValueError("Step 22.29 target rate is frozen at 0.15")
    if diagnostic.get("block_lengths") != [5, 20]:
        raise ValueError("Step 22.29 block lengths are frozen at [5,20]")
    if int(diagnostic.get("nested_seed", -1)) != 2027:
        raise ValueError("Step 22.29 nested seed is frozen at 2027")
    if diagnostic.get("teacher_weights") != {
        "original": 0.5, "l5": 0.25, "l20": 0.25,
    }:
        raise ValueError("Step 22.29 teacher weights changed")
    max_batches = int(diagnostic["max_eval_batches"])
    if bool(experiment.get("smoke_test")):
        if max_batches != 3:
            raise ValueError("Step 22.29 smoke is frozen to three batches")
    elif max_batches != -1:
        raise ValueError("Step 22.29 formal must use full validation")
    official = source["official_args"]
    identity = {
        "mipp": bool(official.get("use_mipp", False)),
        "mcpr": bool(official.get("use_mcpr", False)),
        "bf_cmvd": bool(official.get("use_bf_cmvd", False)),
        "dmrb": bool(official.get("use_dmrb", False)),
        "l5": bool(official.get("dmrb_l5_enabled", False)),
        "l20": bool(official.get("dmrb_l20_enabled", False)),
        "objective": official.get("mcpr_objective")
        == "dual_scale_residual_basis_distill",
        "batch16": int(official.get("batch_size", -1)) == 16,
        "seq192": int(official.get("seq_len", -1)) == 192,
        "pred96": int(official.get("pred_len", -1)) == 96,
        "variables862": int(official.get("c_out", -1)) == 862,
        "seed2027": int(official.get("seed", -1)) == 2027,
    }
    if not all(identity.values()):
        raise ValueError(f"DMRB source identity failed: {identity}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-uuid", default=None)
    parser.add_argument("--scheduler-attempt", type=int, default=0)
    args = parser.parse_args()

    config_path = project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_config_path = project_path(config["source"]["dmrb_resolved_config"])
    dmrb_metrics_path = project_path(config["source"]["dmrb_metrics"])
    dmrb_checkpoint = project_path(config["source"]["dmrb_checkpoint"])
    backbone_metrics_path = project_path(config["source"]["backbone_metrics"])
    backbone_checkpoint = project_path(config["source"]["backbone_checkpoint"])
    teacher_metrics_path = project_path(config["source"]["teacher_metrics"])
    teacher_samples_path = project_path(config["source"]["teacher_samples"])
    missing_csv = project_path(config["data"]["missing_csv"])
    complete_csv = project_path(config["data"]["complete_csv"])
    mask_asset = project_path(config["data"]["mask_asset"])
    required_paths = (
        config_path, source_config_path, dmrb_metrics_path, dmrb_checkpoint,
        backbone_metrics_path, backbone_checkpoint, teacher_metrics_path,
        teacher_samples_path, missing_csv, complete_csv, mask_asset,
    )
    for required in required_paths:
        if not required.is_file():
            raise FileNotFoundError(
                f"required Step 22.29 asset is missing: {required}"
            )

    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    dmrb_metrics = load_json(dmrb_metrics_path)
    backbone_metrics = load_json(backbone_metrics_path)
    teacher_metrics = load_json(teacher_metrics_path)
    validate_config(config, source_config)
    if dmrb_metrics.get("experiment_id") != EXPECTED_DMRB_ID:
        raise ValueError("Step 22.29 DMRB experiment identity changed")
    if backbone_metrics.get("experiment_id") != EXPECTED_BACKBONE_ID:
        raise ValueError("Step 22.29 backbone experiment identity changed")
    if teacher_metrics.get("experiment_id") != EXPECTED_TEACHER_ID:
        raise ValueError("Step 22.29 teacher experiment identity changed")
    if Path(dmrb_metrics.get("checkpoint", "")).resolve() != dmrb_checkpoint:
        raise ValueError("DMRB metrics and checkpoint differ")
    if Path(backbone_metrics.get("checkpoint", "")).resolve() != backbone_checkpoint:
        raise ValueError("backbone metrics and checkpoint differ")
    backbone_hash = sha256(backbone_checkpoint)
    if dmrb_metrics.get("mcpr_source_checkpoint_sha256") != backbone_hash:
        raise ValueError("DMRB checkpoint does not identify the frozen PG95 source")

    if args.gpu_uuid:
        config["runtime"]["gpu_uuid"] = args.gpu_uuid
    if config["runtime"].get("device") != "cuda":
        raise ValueError("Step 22.29 diagnostic requires CUDA")
    config["runtime"].update(
        resolve_gpu_metadata(str(config["runtime"]["gpu_uuid"]))
    )
    config["runtime"]["scheduler_attempt"] = int(args.scheduler_attempt)

    smoke = bool(config["experiment"].get("smoke_test"))
    smoke_suffix = "__smoke" if smoke else ""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = (
        "dmrb_failure_attribution__traffic__validation__seed2027"
        f"{smoke_suffix}__{timestamp}"
    )
    run_dir = ROOT / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"experiment ID already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    metrics_path = run_dir / "metrics.json"
    samples_path = run_dir / "dmrb_branch_failure_tokens.npz"

    dmrb_hash = sha256(dmrb_checkpoint)
    checkpoint_verification = verify_checkpoint(
        dmrb_checkpoint, require_mipp=True, require_mcpr=True
    )
    backbone_verification = verify_checkpoint(backbone_checkpoint, require_mipp=True)
    checkpoint_verification["checkpoint_dmrb_branches_verified"] = (
        dmrb_branch_keys_verified(checkpoint_verification)
    )
    if checkpoint_verification["checkpoint_dmrb_branches_verified"] is not True:
        raise ValueError("Step 22.29 checkpoint lacks complete DMRB branches")
    if checkpoint_verification.get("checkpoint_format") != "mcpr_delta_v1":
        raise ValueError("Step 22.29 requires the audited DMRB delta checkpoint")

    official_args = copy.deepcopy(source_config["official_args"])
    diagnostic_flags = (
        "gate_h_diagnostic", "hcup_diagnostic", "stage12_validation_diagnostic",
        "stage22_validation_diagnostic", "stage22_inference_timing",
        "stage22_ard_information_diagnostic", "stage22_27_bf_cmvd_diagnostic",
        "stage14_uncertainty_diagnostic", "stage15_nested_ensemble_test",
        "stage18_equal_rate_multiscale_validation",
        "stage19_equal_rate_multiscale_test",
        "stage21_equal_rate_multiscale_evaluation",
        "stage22_24_batch_parallel_comparison",
        "stage22_25_cuda_stream_comparison",
    )
    for flag in diagnostic_flags:
        official_args[flag] = False
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
        "stage22_29_dmrb_diagnostic": True,
        "stage22_29_checkpoint": str(dmrb_checkpoint),
        "stage22_29_metrics_path": str(metrics_path),
        "stage22_29_samples_path": str(samples_path),
        "stage22_29_target_rate": 0.15,
        "stage22_29_block_lengths": [5, 20],
        "stage22_29_nested_seed": 2027,
        "stage22_29_original_weight": 0.5,
        "stage22_29_l5_weight": 0.25,
        "stage22_29_l20_weight": 0.25,
        "gpu": 0,
    })

    resolved = copy.deepcopy(config)
    resolved["experiment"]["id"] = run_id
    resolved["experiment"]["created_at"] = dt.datetime.now(
        dt.timezone.utc
    ).isoformat()
    resolved["source"].update({
        "dmrb_resolved_config": str(source_config_path),
        "dmrb_metrics": str(dmrb_metrics_path),
        "dmrb_checkpoint": str(dmrb_checkpoint),
        "dmrb_checkpoint_sha256": dmrb_hash,
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
    critical_sources = {
        "baseline/s4m/model/MCPR.py": S4M_ROOT / "model/MCPR.py",
        "baseline/s4m/model/S4M.py": S4M_ROOT / "model/S4M.py",
        "baseline/s4m/experiments/exp_pretrain1.py": (
            S4M_ROOT / "experiments/exp_pretrain1.py"
        ),
        "baseline/s4m/run.py": S4M_ROOT / "run.py",
        "baseline/s4m/utils/tools.py": S4M_ROOT / "utils/tools.py",
        "project/scripts/run_stage22_29_dmrb_diagnostic.py": Path(__file__).resolve(),
        "project/config_source.yaml": config_path,
    }
    manifest = {}
    for name, source in critical_sources.items():
        destination = run_dir / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manifest[name] = {
            "source": str(source), "sha256": sha256(source),
            "size_bytes": source.stat().st_size,
        }
    atomic_json(run_dir / "source_snapshot_manifest.json", manifest)
    atomic_json(run_dir / "checkpoints/source_checkpoints.json", {
        "dmrb": {
            "path": str(dmrb_checkpoint), "sha256": dmrb_hash,
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
        "source_checkpoint": str(dmrb_checkpoint),
        "source_checkpoint_sha256": dmrb_hash,
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
            expected_keys = TOKEN_KEYS | STRUCTURAL_KEYS
            reconstructed_benefits = {
                f"{label}_beneficial": (
                    (arrays[f"{label}_mse"] < arrays["backbone_mse"])
                    & (arrays[f"{label}_mae"] < arrays["backbone_mae"])
                )
                for label in (
                    "teacher_l5", "teacher_l20", "teacher_joint",
                    "student_l5", "student_l20", "student_joint",
                )
            }
            self_mismatch = {
                branch: (
                    arrays[f"{branch}_self_residual_energy"]
                    + arrays[f"{branch}_self_target_energy"]
                    - 2.0 * arrays[f"{branch}_self_dot"]
                )
                for branch in ("l5", "l20")
            }
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
                "benefit_rules_roundtrip": all(np.array_equal(
                    arrays[key].astype(bool), value
                ) for key, value in reconstructed_benefits.items()),
                "geometry_quadratic_identities": all(np.allclose(
                    arrays[f"{branch}_self_mismatch_energy"], value,
                    atol=2e-5, rtol=2e-5,
                ) for branch, value in self_mismatch.items()),
                "fixed_mask_identity": all(np.array_equal(
                    arrays[key], reference_masks[key][:expected_samples]
                ) for key in reference_masks),
                "student_factorization": float(
                    metrics.get("student_factorization_max_abs", 1.0)
                ) <= FACTORIZATION_ATOL,
                "teacher_factorization": float(
                    metrics.get("teacher_factorization_max_abs", 1.0)
                ) <= FACTORIZATION_ATOL,
                "metric_roundtrip": all(
                    abs(
                        float(metrics[f"complete_target_{metric}_{label}"])
                        - float(arrays[f"{label}_{metric}"].mean())
                    ) <= 1e-7
                    for label in (
                        "backbone", "teacher_l5", "teacher_l20", "teacher_joint",
                        "student_l5", "student_l20", "student_joint",
                    )
                    for metric in ("mse", "mae")
                ),
                "formal_reference_roundtrip": smoke or all(
                    abs(float(metrics[key]) - float(reference)) <= 1e-5
                    for key, reference in (
                        ("complete_target_mse_backbone", backbone_metrics["mse"]),
                        ("complete_target_mae_backbone", backbone_metrics["mae"]),
                        (
                            "complete_target_mse_teacher_joint",
                            teacher_metrics["complete_target_mse_multiscale"],
                        ),
                        (
                            "complete_target_mae_teacher_joint",
                            teacher_metrics["complete_target_mae_multiscale"],
                        ),
                        ("complete_target_mse_student_joint", dmrb_metrics["mse"]),
                        ("complete_target_mae_student_joint", dmrb_metrics["mae"]),
                    )
                ),
                "checkpoint_finite": checkpoint_verification[
                    "checkpoint_all_model_tensors_finite"
                ] is True,
                "checkpoint_dmrb_identity": (
                    checkpoint_verification.get("checkpoint_format") == "mcpr_delta_v1"
                    and checkpoint_verification.get(
                        "checkpoint_dmrb_branches_verified"
                    ) is True
                    and bool(checkpoint_verification["checkpoint_mcpr_state_keys"])
                ),
                "compact_cache": (
                    metrics.get("compact_cache_excludes_full_predictions") is True
                    and samples_path.stat().st_size < 64 * 1024 * 1024
                    and not any("prediction" in key for key in arrays)
                ),
            }
            if not all(checks.values()):
                raise RuntimeError(f"Step 22.29 artifact checks failed: {checks}")
            metrics.update({
                "experiment_id": run_id,
                "dataset": "traffic",
                "method": "dmrb_failure_attribution",
                "evaluation_seed": 2027,
                "scope": (
                    "smoke_test_not_formal_result" if smoke
                    else "full_validation_frozen_dmrb_failure_attribution"
                ),
                "gpu_uuid": gpu_uuid,
                "dmrb_checkpoint": str(dmrb_checkpoint),
                "dmrb_checkpoint_sha256": dmrb_hash,
                "backbone_checkpoint": str(backbone_checkpoint),
                "backbone_checkpoint_sha256": backbone_hash,
                "teacher_metrics": str(teacher_metrics_path),
                "teacher_samples": str(teacher_samples_path),
                "mask_asset": str(mask_asset),
                "mask_asset_sha256": sha256(mask_asset),
                "samples_path": str(samples_path),
                "artifact_checks": checks,
                "artifact_status": "PASS",
                "factorization_atol": FACTORIZATION_ATOL,
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
