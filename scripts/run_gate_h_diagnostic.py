#!/usr/bin/env python3
"""Run one auditable, test-only Gate H prototype intervention diagnostic."""

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

# This parent only orchestrates a child process. Keep it from reserving a CUDA
# context before the selected GPU UUID is exported for the child.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / "baselines" / "S4M"
S4M_ROOT = BASELINE_ROOT / "s4m"
STRUCTURED_KERNEL_ROOT = S4M_ROOT / "model" / "S4" / "s4" / "extensions" / "kernels"


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def diagnostic_id(config: dict[str, Any], timestamp: str) -> str:
    experiment = config["experiment"]
    smoke = "__smoke" if experiment.get("smoke_test") else ""
    return (
        f"{experiment['model']}__{experiment['dataset']}__{experiment['domain']}"
        f"__seed{experiment['seed']}{smoke}__{timestamp}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-uuid", default=None)
    parser.add_argument("--scheduler-attempt", type=int, default=0)
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_config_path = resolve_project_path(config["source"]["baseline_config"])
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    checkpoint = resolve_project_path(config["source"]["checkpoint"])
    source_metrics_path = resolve_project_path(config["source"]["baseline_metrics"])
    missing_csv = resolve_project_path(config["data"]["missing_csv"])
    complete_csv = resolve_project_path(config["data"]["complete_csv"])
    for required in (source_config_path, checkpoint, source_metrics_path, missing_csv, complete_csv):
        if not required.is_file():
            raise FileNotFoundError(f"required Gate H asset is missing: {required}")

    if args.gpu_uuid:
        config["runtime"]["gpu_uuid"] = args.gpu_uuid
    if config["runtime"].get("device") == "cuda":
        config["runtime"].update(resolve_gpu_metadata(str(config["runtime"]["gpu_uuid"])))
    config["runtime"]["scheduler_attempt"] = args.scheduler_attempt

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = diagnostic_id(config, timestamp)
    run_dir = PROJECT_ROOT / "runs" / experiment_id
    if run_dir.exists():
        raise FileExistsError(f"experiment ID already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    metrics_path = run_dir / "metrics.json"
    samples_path = run_dir / "sample_diagnostics.npz"
    source_checkpoint_sha256 = file_sha256(checkpoint)
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    checkpoint_verification = verify_checkpoint(checkpoint)

    official_args = copy.deepcopy(source_config["official_args"])
    official_args.update(
        {
            "is_training": 0,
            "model_id": experiment_id,
            "root_path": str(missing_csv.parent),
            "data_path": missing_csv.name,
            "true_data_path": str(complete_csv),
            "max_train_batches": -1,
            "max_eval_batches": int(config["diagnostic"]["max_eval_batches"]),
            "wandb_mode": "disabled",
            "gate_h_diagnostic": True,
            "gate_h_checkpoint": str(checkpoint),
            "gate_h_metrics_path": str(metrics_path),
            "gate_h_samples_path": str(samples_path),
            "gate_h_delta": float(config["diagnostic"].get("hrr_delta", 0.0)),
            "gpu": 0,
        }
    )

    resolved = copy.deepcopy(config)
    resolved["experiment"]["id"] = experiment_id
    resolved["experiment"]["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    resolved["source"]["baseline_config"] = str(source_config_path)
    resolved["source"]["checkpoint"] = str(checkpoint)
    resolved["source"]["baseline_metrics"] = str(source_metrics_path)
    resolved["source"]["checkpoint_sha256"] = source_checkpoint_sha256
    resolved["official_args"] = official_args
    resolved["paths"] = {
        "run_dir": str(run_dir),
        "config_source": str(config_path),
        "baseline_repo": str(BASELINE_ROOT),
        "upstream_repo": str(PROJECT_ROOT / "upstream" / "S4M"),
    }
    atomic_text(run_dir / "resolved_config.yaml", yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
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
        {
            "path": str(checkpoint),
            "sha256": source_checkpoint_sha256,
            **checkpoint_verification,
        },
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
        gpu_export = f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(config['runtime']['gpu_uuid']))}\n"
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

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    status: dict[str, Any] = {
        "experiment_id": experiment_id,
        "state": "RUNNING",
        "started_at": started_at,
        "config": str(config_path),
        "device": config["runtime"]["device"],
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": source_checkpoint_sha256,
    }
    atomic_json(run_dir / "status.json", status)
    print(f"run_dir={run_dir}", flush=True)
    start = time.monotonic()
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
    elapsed = time.monotonic() - start

    metrics: dict[str, Any] = {}
    verification_error: str | None = None
    if return_code == 0 and metrics_path.is_file() and samples_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if int(metrics.get("sample_count", 0)) < 1:
                raise RuntimeError("Gate H diagnostic produced zero samples")
            expected_on_mse = source_metrics.get("mse")
            observed_on_mse = metrics.get("prototype_on_mse")
            baseline_match_abs_error = None
            if (
                config["experiment"].get("domain_role") == "id"
                and not config["experiment"].get("smoke_test")
                and int(config["diagnostic"]["max_eval_batches"]) < 0
                and expected_on_mse is not None
            ):
                baseline_match_abs_error = abs(float(observed_on_mse) - float(expected_on_mse))
            metrics.update(
                {
                    "experiment_id": experiment_id,
                    "dataset": config["experiment"]["dataset"],
                    "domain": config["experiment"]["domain"],
                    "domain_role": config["experiment"].get("domain_role", "unknown"),
                    "evaluation_seed": config["experiment"]["seed"],
                    "scope": "smoke_test_not_formal_result" if config["experiment"].get("smoke_test") else "formal_gate_h_diagnostic",
                    "source_baseline_mse": expected_on_mse,
                    "prototype_on_baseline_match_abs_error": baseline_match_abs_error,
                    "source_checkpoint": str(checkpoint),
                    "source_checkpoint_sha256": source_checkpoint_sha256,
                    "mask_asset": config["data"].get("mask_asset", "unknown"),
                    "prototype_intervention": "zero_retrieved_vector_only",
                    "paired_same_batch": True,
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
        and int(metrics.get("sample_count", 0)) > 0
    )
    resource_usage = {
        "elapsed_seconds": elapsed,
        "device": config["runtime"]["device"],
        "gpu_uuid": config["runtime"].get("gpu_uuid"),
        "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
        "peak_gpu_reserved_mb": metrics.get("peak_gpu_reserved_mb"),
        "return_code": return_code,
    }
    atomic_json(run_dir / "resource_usage.json", resource_usage)
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
