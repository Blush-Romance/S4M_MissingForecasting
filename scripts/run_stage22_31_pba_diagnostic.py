#!/usr/bin/env python3
"""Run one auditable validation-only Step 22.31 PBA-CMVD diagnostic."""

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
EXPECTED_PBA_ID = (
    "pba_cmvd_pg95_traffic__traffic__point__mr0.06__seed2027"
    "__20260826_222702"
)
EXPECTED_BACKBONE_ID = (
    "mipp_pg95_traffic__traffic__point__mr0.06__seed2027"
    "__20260820_174327"
)
EXPECTED_BF_DIAGNOSTIC_ID = (
    "bf_cmvd_v2_failure_attribution__traffic__validation__seed2027"
    "__20260821_221024"
)
TOKEN_KEYS = {
    "observed_token", "complete_teacher_beneficial",
    "observed_teacher_beneficial", "complete_student_beneficial",
    "backbone_mse", "teacher_mse", "student_mse",
    "backbone_mae", "teacher_mae", "student_mae",
    "observed_backbone_mse", "observed_teacher_mse",
    "observed_backbone_mae", "observed_teacher_mae",
    "correction_dot", "student_correction_energy",
    "teacher_correction_energy", "correction_mismatch_energy",
    "source_error_correction_dot",
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
FP32_CROSS_PATH_TOL = float(512.0 * np.finfo(np.float32).eps)
MAX_BENEFIT_DISAGREEMENT_RATE = 1e-4


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


def fp32_cross_path_close(current: np.ndarray, reference: np.ndarray) -> bool:
    """Allow only scale-aware FP32 reordering noise between equivalent paths."""
    return bool(np.allclose(
        current, reference,
        atol=FP32_CROSS_PATH_TOL,
        rtol=FP32_CROSS_PATH_TOL,
    ))


def benefit_boundary_stable(
    current: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> tuple[bool, dict[str, Any]]:
    current_benefit = (
        (current["teacher_mse"] < current["backbone_mse"])
        & (current["teacher_mae"] < current["backbone_mae"])
    )
    reference_benefit = (
        (reference["teacher_mse"] < reference["backbone_mse"])
        & (reference["teacher_mae"] < reference["backbone_mae"])
    )
    disagreement = current_benefit ^ reference_benefit
    count = int(disagreement.sum())
    rate = float(disagreement.mean())
    near_boundary = np.ones_like(disagreement, dtype=bool)
    if count:
        margin_current_mse = (
            current["backbone_mse"].astype(np.float64)
            - current["teacher_mse"].astype(np.float64)
        )
        margin_reference_mse = (
            reference["backbone_mse"].astype(np.float64)
            - reference["teacher_mse"].astype(np.float64)
        )
        margin_current_mae = (
            current["backbone_mae"].astype(np.float64)
            - current["teacher_mae"].astype(np.float64)
        )
        margin_reference_mae = (
            reference["backbone_mae"].astype(np.float64)
            - reference["teacher_mae"].astype(np.float64)
        )
        mse_scale = 1.0 + np.maximum.reduce((
            np.abs(current["backbone_mse"]), np.abs(current["teacher_mse"]),
            np.abs(reference["backbone_mse"]), np.abs(reference["teacher_mse"]),
        ))
        mae_scale = 1.0 + np.maximum.reduce((
            np.abs(current["backbone_mae"]), np.abs(current["teacher_mae"]),
            np.abs(reference["backbone_mae"]), np.abs(reference["teacher_mae"]),
        ))
        near_boundary = (
            (np.abs(margin_current_mse) <= FP32_CROSS_PATH_TOL * mse_scale)
            | (np.abs(margin_reference_mse) <= FP32_CROSS_PATH_TOL * mse_scale)
            | (np.abs(margin_current_mae) <= FP32_CROSS_PATH_TOL * mae_scale)
            | (np.abs(margin_reference_mae) <= FP32_CROSS_PATH_TOL * mae_scale)
        )
    stable = bool(
        rate <= MAX_BENEFIT_DISAGREEMENT_RATE
        and (not count or bool(near_boundary[disagreement].all()))
    )
    return stable, {
        "disagreement_count": count,
        "disagreement_rate": rate,
        "all_disagreements_near_boundary": bool(
            not count or near_boundary[disagreement].all()
        ),
        "max_rate": MAX_BENEFIT_DISAGREEMENT_RATE,
        "fp32_cross_path_tolerance": FP32_CROSS_PATH_TOL,
    }


def validate_config(config: dict[str, Any], pba_config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    diagnostic = config["diagnostic"]
    if experiment.get("method") != "pba_cmvd":
        raise ValueError("Step22.31 source method must be PBA-CMVD")
    if experiment.get("dataset") != "traffic" or int(
        experiment.get("seed", -1)
    ) != 2027:
        raise ValueError("Step22.31 is frozen to Traffic seed2027")
    if diagnostic.get("split") != "val":
        raise ValueError("Step22.31 must remain validation-only")
    if float(diagnostic.get("target_rate", -1)) != 0.15:
        raise ValueError("Step22.31 target rate changed")
    if diagnostic.get("block_lengths") != [5, 20]:
        raise ValueError("Step22.31 block lengths changed")
    if int(diagnostic.get("nested_seed", -1)) != 2027:
        raise ValueError("Step22.31 nested seed changed")
    if diagnostic.get("teacher_weights") != {
        "original": 0.5, "l5": 0.25, "l20": 0.25,
    }:
        raise ValueError("Step22.31 teacher weights changed")
    if float(diagnostic.get("bf_tail_quantile", -1)) != 0.90:
        raise ValueError("Step22.31 BF reference tail quantile changed")
    max_batches = int(diagnostic["max_eval_batches"])
    if bool(experiment.get("smoke_test")):
        if max_batches != 3:
            raise ValueError("Step22.31 smoke is frozen to three batches")
    elif max_batches != -1:
        raise ValueError("Step22.31 formal must use full validation")

    official = pba_config["official_args"]
    identity = {
        "mipp": bool(official.get("use_mipp", False)),
        "pba": bool(official.get("use_pba_cmvd", False)),
        "no_mcpr": not bool(official.get("use_mcpr", False)),
        "no_bf": not bool(official.get("use_bf_cmvd", False)),
        "no_dmrb": not bool(official.get("use_dmrb", False)),
        "no_ttrr": not bool(official.get("use_ttrr", False)),
        "no_ard": not bool(official.get("use_ard", False)),
        "tail2": int(official.get("pba_tail_layers", -1)) == 2,
        "batch16": int(official.get("batch_size", -1)) == 16,
        "seq192": int(official.get("seq_len", -1)) == 192,
        "pred96": int(official.get("pred_len", -1)) == 96,
        "variables862": int(official.get("c_out", -1)) == 862,
        "seed2027": int(official.get("seed", -1)) == 2027,
    }
    if not all(identity.values()):
        raise ValueError(f"PBA source identity failed: {identity}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-uuid", default=None)
    parser.add_argument("--scheduler-attempt", type=int, default=0)
    args = parser.parse_args()

    config_path = project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = config["source"]
    pba_config_path = project_path(source["pba_resolved_config"])
    pba_metrics_path = project_path(source["pba_metrics"])
    pba_checkpoint = project_path(source["pba_checkpoint"])
    backbone_metrics_path = project_path(source["backbone_metrics"])
    backbone_checkpoint = project_path(source["backbone_checkpoint"])
    teacher_metrics_path = project_path(source["teacher_metrics"])
    bf_metrics_path = project_path(source["bf_diagnostic_metrics"])
    bf_samples_path = project_path(source["bf_diagnostic_samples"])
    missing_csv = project_path(config["data"]["missing_csv"])
    complete_csv = project_path(config["data"]["complete_csv"])
    mask_asset = project_path(config["data"]["mask_asset"])
    required_paths = (
        config_path, pba_config_path, pba_metrics_path, pba_checkpoint,
        backbone_metrics_path, backbone_checkpoint, teacher_metrics_path,
        bf_metrics_path, bf_samples_path, missing_csv, complete_csv, mask_asset,
    )
    for required in required_paths:
        if not required.is_file():
            raise FileNotFoundError(f"required Step22.31 asset missing: {required}")

    pba_config = yaml.safe_load(pba_config_path.read_text(encoding="utf-8"))
    pba_metrics = load_json(pba_metrics_path)
    backbone_metrics = load_json(backbone_metrics_path)
    teacher_metrics = load_json(teacher_metrics_path)
    bf_metrics = load_json(bf_metrics_path)
    validate_config(config, pba_config)
    if pba_metrics.get("experiment_id") != EXPECTED_PBA_ID:
        raise ValueError("Step22.31 PBA experiment identity changed")
    if backbone_metrics.get("experiment_id") != EXPECTED_BACKBONE_ID:
        raise ValueError("Step22.31 PG95 experiment identity changed")
    if bf_metrics.get("experiment_id") != EXPECTED_BF_DIAGNOSTIC_ID:
        raise ValueError("Step22.31 BF diagnostic identity changed")
    if bf_metrics.get("artifact_status") != "PASS" or any(
        value is not True for value in bf_metrics.get("artifact_checks", {}).values()
    ):
        raise ValueError("Step22.31 requires a fully verified BF compact cache")
    if bf_metrics.get("test_split_accessed") is not False:
        raise ValueError("Step22.31 BF cache must be validation-only")
    if Path(pba_metrics.get("checkpoint", "")).resolve() != pba_checkpoint:
        raise ValueError("PBA metrics and checkpoint differ")
    if Path(backbone_metrics.get("checkpoint", "")).resolve() != backbone_checkpoint:
        raise ValueError("PG95 metrics and checkpoint differ")

    pba_verification = verify_checkpoint(
        pba_checkpoint, require_mipp=True, require_pba=True
    )
    backbone_verification = verify_checkpoint(
        backbone_checkpoint, require_mipp=True
    )
    backbone_hash = sha256(backbone_checkpoint)
    if pba_verification.get("checkpoint_source_checkpoint_sha256") != backbone_hash:
        raise ValueError("PBA delta does not identify the frozen PG95 source")
    if Path(
        pba_verification.get("checkpoint_source_checkpoint", "")
    ).resolve() != backbone_checkpoint:
        raise ValueError("PBA delta source path changed")

    if args.gpu_uuid:
        config["runtime"]["gpu_uuid"] = args.gpu_uuid
    if config["runtime"].get("device") != "cuda":
        raise ValueError("Step22.31 diagnostic requires CUDA")
    config["runtime"].update(
        resolve_gpu_metadata(str(config["runtime"]["gpu_uuid"]))
    )
    config["runtime"]["scheduler_attempt"] = int(args.scheduler_attempt)

    smoke = bool(config["experiment"].get("smoke_test"))
    smoke_suffix = "__smoke" if smoke else ""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = (
        "pba_cmvd_failure_attribution__traffic__validation__seed2027"
        f"{smoke_suffix}__{timestamp}"
    )
    run_dir = ROOT / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"experiment ID already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    metrics_path = run_dir / "metrics.json"
    samples_path = run_dir / "pba_failure_tokens.npz"

    official_args = copy.deepcopy(pba_config["official_args"])
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
        "stage22_27_bf_cmvd_diagnostic": False,
        "stage22_29_dmrb_diagnostic": False,
        "stage22_31_pba_diagnostic": True,
        "stage22_31_checkpoint": str(pba_checkpoint),
        "stage22_31_backbone_checkpoint": str(backbone_checkpoint),
        "stage22_31_metrics_path": str(metrics_path),
        "stage22_31_samples_path": str(samples_path),
        "stage22_31_target_rate": 0.15,
        "stage22_31_block_lengths": [5, 20],
        "stage22_31_nested_seed": 2027,
        "stage22_31_original_weight": 0.5,
        "stage22_31_l5_weight": 0.25,
        "stage22_31_l20_weight": 0.25,
        "stage14_uncertainty_diagnostic": False,
        "stage15_nested_ensemble_test": False,
        "stage18_equal_rate_multiscale_validation": False,
        "stage19_equal_rate_multiscale_test": False,
        "stage21_equal_rate_multiscale_evaluation": False,
        "stage22_24_batch_parallel_comparison": False,
        "stage22_25_cuda_stream_comparison": False,
        "gpu": 0,
    })

    pba_hash = sha256(pba_checkpoint)
    resolved = copy.deepcopy(config)
    resolved["experiment"]["id"] = run_id
    resolved["experiment"]["created_at"] = dt.datetime.now(
        dt.timezone.utc
    ).isoformat()
    resolved["source"].update({
        "pba_resolved_config": str(pba_config_path),
        "pba_metrics": str(pba_metrics_path),
        "pba_checkpoint": str(pba_checkpoint),
        "pba_checkpoint_sha256": pba_hash,
        "backbone_metrics": str(backbone_metrics_path),
        "backbone_checkpoint": str(backbone_checkpoint),
        "backbone_checkpoint_sha256": backbone_hash,
        "teacher_metrics": str(teacher_metrics_path),
        "bf_diagnostic_metrics": str(bf_metrics_path),
        "bf_diagnostic_samples": str(bf_samples_path),
        "bf_diagnostic_samples_sha256": sha256(bf_samples_path),
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
        + "\n[project]\n" + git_evidence(ROOT),
    )
    _, diff = run_capture(["git", "diff", "--binary"], BASELINE_ROOT)
    atomic_text(run_dir / "code.diff", diff)
    atomic_json(run_dir / "checkpoints/source_checkpoints.json", {
        "pba": {
            "path": str(pba_checkpoint), "sha256": pba_hash,
            **pba_verification,
        },
        "backbone": {
            "path": str(backbone_checkpoint), "sha256": backbone_hash,
            **backbone_verification,
        },
    })
    snapshot_dir = run_dir / "source_snapshot"
    snapshot_dir.mkdir()
    snapshot_sources = [
        S4M_ROOT / "run.py",
        S4M_ROOT / "experiments/exp_pretrain1.py",
        Path(__file__).resolve(),
        ROOT / "scripts/analyze_stage22_31_pba_failure.py",
        ROOT / "scripts/run_stage22_31_pba_queue.py",
        ROOT / "docs/STAGE22_31_PBA_FAILURE_ATTRIBUTION_PROTOCOL_ZH.md",
        config_path,
    ]
    snapshot_manifest = {}
    for path in snapshot_sources:
        if path.is_file():
            target = snapshot_dir / path.name
            shutil.copy2(path, target)
            snapshot_manifest[str(path)] = {
                "snapshot": str(target), "sha256": sha256(path),
            }
    atomic_json(snapshot_dir / "manifest.json", snapshot_manifest)

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
        "source_checkpoint": str(pba_checkpoint),
        "source_checkpoint_sha256": pba_hash,
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
            with np.load(bf_samples_path, allow_pickle=False) as archive:
                bf_arrays = {key: archive[key] for key in archive.files}
            expected_batches = 3 if smoke else 103
            expected_samples = expected_batches * 16
            expected_keys = TOKEN_KEYS | VARIABLE_KEYS | STRUCTURAL_KEYS
            observed = arrays["observed_token"].astype(bool)
            benefit = arrays["complete_teacher_beneficial"].astype(bool)
            observed_benefit = arrays["observed_teacher_beneficial"].astype(bool)
            reconstructed_benefit = (
                (arrays["teacher_mse"] < arrays["backbone_mse"])
                & (arrays["teacher_mae"] < arrays["backbone_mae"])
            )
            reconstructed_student = (
                (arrays["student_mse"] < arrays["backbone_mse"])
                & (arrays["student_mae"] < arrays["backbone_mae"])
            )
            bf_reference = {
                key: bf_arrays[key][:expected_samples]
                for key in bf_arrays
            }
            benefit_boundary_pass, benefit_boundary_diagnostics = (
                benefit_boundary_stable(arrays, bf_reference)
            )
            cross_path_diagnostics = {}
            for key in (
                "backbone_mse", "backbone_mae", "teacher_mse", "teacher_mae"
            ):
                difference = np.abs(
                    arrays[key].astype(np.float64)
                    - bf_reference[key].astype(np.float64)
                )
                cross_path_diagnostics[key] = {
                    "max_abs": float(difference.max()),
                    "mean_abs": float(difference.mean()),
                    "q99_abs": float(np.quantile(difference, 0.99)),
                    "scale_aware_close": fp32_cross_path_close(
                        arrays[key], bf_reference[key]
                    ),
                }
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
                "frozen_models_and_memories": (
                    metrics.get("updates_model_or_memory") is False
                    and metrics.get("pba_model_hash_before")
                    == metrics.get("pba_model_hash_after")
                    and metrics.get("pba_memory_hash_before")
                    == metrics.get("pba_memory_hash_after")
                    and metrics.get("source_model_hash_before")
                    == metrics.get("source_model_hash_after")
                    and metrics.get("source_memory_hash_before")
                    == metrics.get("source_memory_hash_after")
                ),
                "expected_batch_count": int(
                    metrics.get("validation_batches_processed", 0)
                ) == expected_batches,
                "expected_sample_count": int(
                    metrics.get("validation_samples_processed", 0)
                ) == expected_samples,
                "exact_array_keys": set(arrays) == expected_keys,
                "token_shapes": all(
                    arrays[key].shape == (expected_samples, 96)
                    for key in TOKEN_KEYS
                ),
                "variable_shapes": all(
                    arrays[key].shape == (862,) for key in VARIABLE_KEYS
                ),
                "packed_shapes": all(
                    arrays[key].shape == (expected_samples, 24)
                    for key in ("history_mask_packed", "l5_mask_packed", "l20_mask_packed")
                ),
                "sample_index_chronological": np.array_equal(
                    arrays["sample_index"], np.arange(expected_samples)
                ),
                "all_arrays_finite": all(all_finite(value) for value in arrays.values()),
                "teacher_benefit_rule_roundtrip": np.array_equal(
                    benefit, reconstructed_benefit
                ),
                "student_benefit_rule_roundtrip": np.array_equal(
                    arrays["complete_student_beneficial"].astype(bool),
                    reconstructed_student,
                ),
                "observed_benefit_is_complete_subset": np.array_equal(
                    observed_benefit, benefit & observed
                ),
                "correction_quadratic_identity": np.allclose(
                    arrays["correction_mismatch_energy"], mismatch_identity,
                    atol=2e-5, rtol=2e-5,
                ),
                "fixed_assets_match_bf_cache": all(np.array_equal(
                    arrays[key], bf_arrays[key][:expected_samples]
                ) for key in STRUCTURAL_KEYS),
                "backbone_teacher_roundtrip_bf_cache": all(np.allclose(
                    arrays[key], bf_reference[key],
                    atol=FP32_CROSS_PATH_TOL,
                    rtol=FP32_CROSS_PATH_TOL,
                ) for key in (
                    "backbone_mse", "backbone_mae", "teacher_mse", "teacher_mae"
                )),
                "teacher_benefit_boundary_stable": benefit_boundary_pass,
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
                        ("complete_target_mse_teacher", teacher_metrics["complete_target_mse_multiscale"]),
                        ("complete_target_mae_teacher", teacher_metrics["complete_target_mae_multiscale"]),
                        ("complete_target_mse_student", pba_metrics["mse"]),
                        ("complete_target_mae_student", pba_metrics["mae"]),
                    )
                ),
                "checkpoint_pba_identity": (
                    pba_verification.get("checkpoint_format") == "pba_delta_v1"
                    and int(pba_verification.get("checkpoint_pba_tail_layers", -1)) == 2
                    and len(pba_verification.get("checkpoint_pba_state_keys", [])) == 60
                ),
                "compact_cache": (
                    metrics.get("compact_cache_excludes_full_predictions") is True
                    and samples_path.stat().st_size < 64 * 1024 * 1024
                    and not any("prediction" in key for key in arrays)
                ),
            }
            if not all(checks.values()):
                raise RuntimeError(f"Step22.31 artifact checks failed: {checks}")
            metrics.update({
                "experiment_id": run_id,
                "dataset": "traffic",
                "method": "pba_cmvd_failure_attribution",
                "evaluation_seed": 2027,
                "scope": (
                    "smoke_test_not_formal_result" if smoke
                    else "full_validation_frozen_failure_attribution"
                ),
                "gpu_uuid": gpu_uuid,
                "pba_checkpoint": str(pba_checkpoint),
                "pba_checkpoint_sha256": pba_hash,
                "backbone_checkpoint": str(backbone_checkpoint),
                "backbone_checkpoint_sha256": backbone_hash,
                "teacher_metrics": str(teacher_metrics_path),
                "bf_diagnostic_metrics": str(bf_metrics_path),
                "bf_diagnostic_samples": str(bf_samples_path),
                "bf_diagnostic_samples_sha256": sha256(bf_samples_path),
                "mask_asset": str(mask_asset),
                "mask_asset_sha256": sha256(mask_asset),
                "samples_path": str(samples_path),
                "artifact_checks": checks,
                "artifact_status": "PASS",
                "fp32_cross_path_tolerance": FP32_CROSS_PATH_TOL,
                "cross_path_diagnostics": cross_path_diagnostics,
                "benefit_boundary_diagnostics": benefit_boundary_diagnostics,
                **pba_verification,
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
        **pba_verification,
    })
    atomic_json(run_dir / "status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    print(f"run_dir={run_dir}")
    return 0 if success else (return_code or 2)


if __name__ == "__main__":
    raise SystemExit(main())
