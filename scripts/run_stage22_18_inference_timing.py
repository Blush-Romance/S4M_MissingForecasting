#!/usr/bin/env python3
"""Run one auditable Step-22.18 same-GPU validation latency measurement."""

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
STRUCTURED_KERNEL_ROOT = S4M_ROOT / "model" / "S4" / "s4" / "extensions" / "kernels"
METHODS = {"pg95", "mcpr_mse", "ttrr_mcpr"}
EXPECTED_MASK = (
    ROOT / "artifacts/masks/electricity/point/rate_0.06/seed_2024.npz"
).resolve()


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
    method = str(experiment["method"])
    if method not in METHODS:
        raise ValueError(f"unsupported Step-22.18 timing method: {method}")
    if experiment["dataset"] != "electricity" or int(experiment["seed"]) != 2024:
        raise ValueError("Step-22.18 timing is frozen to Electricity seed2024")
    if diagnostic.get("split") != "val":
        raise ValueError("Step-22.18 timing must remain validation-only")
    if project_path(config["data"]["mask_asset"]) != EXPECTED_MASK:
        raise ValueError("Step-22.18 timing must use the frozen seed2024 mask")
    max_batches = int(diagnostic["max_eval_batches"])
    warmup = int(diagnostic["timing_warmup_batches"])
    repeats = int(diagnostic["timing_repeats"])
    if max_batches == 0 or max_batches < -1:
        raise ValueError("max_eval_batches must be -1 or positive")
    if warmup < 0 or repeats < 1:
        raise ValueError("timing warmup/repeats are invalid")
    if max_batches > 0 and warmup >= max_batches:
        raise ValueError("timing warmup must leave a measured batch")
    if experiment.get("smoke_test"):
        if max_batches != 3 or warmup != 1 or repeats != 1:
            raise ValueError("timing smoke is frozen to 3 batches/1 warmup/1 repeat")
    elif max_batches != -1 or warmup != 10 or repeats != 2:
        raise ValueError("formal timing is frozen to full validation/10 warmup/2 repeats")

    official = source["official_args"]
    identity = {
        "pg95": (
            not bool(official.get("use_mcpr", False))
            and not bool(official.get("use_ttrr", False))
        ),
        "mcpr_mse": (
            bool(official.get("use_mcpr", False))
            and str(official.get("mcpr_objective")) == "mse_only"
            and not bool(official.get("use_ttrr", False))
        ),
        "ttrr_mcpr": (
            bool(official.get("use_mcpr", False))
            and str(official.get("mcpr_objective")) == "tail_trust_mse"
            and bool(official.get("use_ttrr", False))
        ),
    }
    if not identity[method]:
        raise ValueError(f"source model identity does not match {method}")
    if not bool(official.get("use_mipp", False)):
        raise ValueError("all timing sources must retain the PG95/MIPP backbone")
    if int(official.get("batch_size", -1)) != 16:
        raise ValueError("Step-22.18 timing batch size is frozen at 16")


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
        config_path, source_config_path, source_metrics_path, checkpoint,
        missing_csv, complete_csv, mask_asset,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"required timing asset is missing: {required}")
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    validate_config(config, source_config)

    if args.gpu_uuid:
        config["runtime"]["gpu_uuid"] = args.gpu_uuid
    if config["runtime"].get("device") != "cuda":
        raise ValueError("formal Step-22.18 timing requires CUDA")
    config["runtime"].update(
        resolve_gpu_metadata(str(config["runtime"]["gpu_uuid"]))
    )
    config["runtime"]["scheduler_attempt"] = int(args.scheduler_attempt)

    method = str(config["experiment"]["method"])
    smoke_suffix = "__smoke" if config["experiment"].get("smoke_test") else ""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = (
        f"stage22_18_timing_{method}__electricity__validation__seed2024"
        f"{smoke_suffix}__{timestamp}"
    )
    run_dir = ROOT / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"experiment ID already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    metrics_path = run_dir / "metrics.json"

    checkpoint_hash = sha256(checkpoint)
    checkpoint_verification = verify_checkpoint(
        checkpoint,
        require_mipp=True,
        require_mcpr=method != "pg95",
        require_ttrr=method == "ttrr_mcpr",
    )
    official_args = copy.deepcopy(source_config["official_args"])
    official_args.update({
        "is_training": 0,
        "seed": 2024,
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
        "stage14_uncertainty_diagnostic": False,
        "stage15_nested_ensemble_test": False,
        "stage18_equal_rate_multiscale_validation": False,
        "stage19_equal_rate_multiscale_test": False,
        "stage21_equal_rate_multiscale_evaluation": False,
        "stage22_inference_timing": True,
        "stage22_timing_checkpoint": str(checkpoint),
        "stage22_timing_metrics_path": str(metrics_path),
        "stage22_timing_warmup_batches": int(
            config["diagnostic"]["timing_warmup_batches"]
        ),
        "stage22_timing_repeats": int(config["diagnostic"]["timing_repeats"]),
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
        "[baseline_adapter]\n" + git_evidence(BASELINE_ROOT)
        + "\n[upstream_snapshot]\n" + git_evidence(ROOT / "upstream" / "S4M"),
    )
    _, diff = run_capture(["git", "diff", "--binary"], BASELINE_ROOT)
    atomic_text(run_dir / "code.diff", diff)
    atomic_json(
        run_dir / "checkpoints" / "source_checkpoint.json",
        {"path": str(checkpoint), "sha256": checkpoint_hash, **checkpoint_verification},
    )

    command = [sys.executable, str(S4M_ROOT / "run.py"), *cli_from_mapping(official_args)]
    python_path = os.pathsep.join([
        str(S4M_ROOT),
        str(S4M_ROOT / "model"),
        str(S4M_ROOT / "model" / "S4" / "s4"),
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
        "method": method,
        "evaluation_seed": 2024,
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
    if return_code == 0 and metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            latency = [float(value) for value in metrics["forward_latency_ms"]]
            expected_batches = (
                3 if config["experiment"].get("smoke_test") else 158
            )
            expected_samples = expected_batches * 16
            expected_measured = (
                expected_batches
                - int(config["diagnostic"]["timing_warmup_batches"])
            ) * int(config["diagnostic"]["timing_repeats"])
            reference_mse = float(source_metrics[config["source"]["mse_key"]])
            reference_mae = float(source_metrics[config["source"]["mae_key"]])
            formal = not bool(config["experiment"].get("smoke_test"))
            checks = {
                "validation_only": (
                    metrics.get("test_split_accessed") is False
                    and metrics.get("uses_test_split") is False
                    and metrics.get("source_split") == "val"
                ),
                "chronological_loader": metrics.get("chronological_loader") is True,
                "frozen_model_and_memory": metrics.get("updates_model_or_memory") is False,
                "cuda_event_timing": metrics.get("timing_clock") == "torch.cuda.Event",
                "model_forward_only_scope": str(metrics.get("timing_scope", "")).startswith(
                    "model forward only"
                ),
                "expected_batch_count": int(
                    metrics.get("validation_batches_processed", 0)
                ) == expected_batches,
                "expected_sample_count": int(
                    metrics.get("validation_samples_processed", 0)
                ) == expected_samples,
                "expected_timing_count": (
                    int(metrics.get("timing_measured_batches_total", 0))
                    == expected_measured == len(latency)
                ),
                "latencies_finite_positive": bool(latency) and all(
                    math.isfinite(value) and value > 0.0 for value in latency
                ),
                "metrics_finite": all(
                    math.isfinite(float(metrics[key])) for key in (
                        "mse", "mae", "forward_mean_ms_per_batch",
                        "forward_p50_ms_per_batch", "forward_p95_ms_per_batch",
                    )
                ),
                "formal_metric_roundtrip": (
                    not formal
                    or (
                        abs(float(metrics["mse"]) - reference_mse) <= 1e-5
                        and abs(float(metrics["mae"]) - reference_mae) <= 1e-5
                    )
                ),
                "checkpoint_finite": checkpoint_verification[
                    "checkpoint_all_model_tensors_finite"
                ] is True,
                "checkpoint_identity": (
                    bool(checkpoint_verification["checkpoint_ttrr_state_keys"])
                    if method == "ttrr_mcpr"
                    else not bool(checkpoint_verification["checkpoint_ttrr_state_keys"])
                ),
            }
            if not all(checks.values()):
                raise RuntimeError(f"timing artifact checks failed: {checks}")
            metrics.update({
                "experiment_id": run_id,
                "dataset": "electricity",
                "method": method,
                "evaluation_seed": 2024,
                "scope": (
                    "smoke_test_not_formal_result"
                    if not formal else "full_validation_same_gpu_forward_timing"
                ),
                "gpu_uuid": gpu_uuid,
                "source_validation_mse_reference": reference_mse,
                "source_validation_mae_reference": reference_mae,
                "source_checkpoint": str(checkpoint),
                "source_checkpoint_sha256": checkpoint_hash,
                "mask_asset": str(mask_asset),
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
