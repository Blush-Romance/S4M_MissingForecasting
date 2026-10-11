#!/usr/bin/env python3
"""Run one auditable HCUP validation/test diagnostic from a frozen S4M model."""

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

# The parent is orchestration-only and must not reserve a CUDA context before
# the scheduler selects a stable GPU UUID for the child process.
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
    return (
        f"{experiment['model']}__{experiment['dataset']}__{experiment['domain']}"
        f"__seed{experiment['seed']}{smoke}__{timestamp}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    required_files = (
        source_config_path,
        checkpoint,
        source_metrics_path,
        missing_csv,
        complete_csv,
    )
    for required in required_files:
        if not required.is_file():
            raise FileNotFoundError(f"required HCUP asset is missing: {required}")

    if args.gpu_uuid:
        config["runtime"]["gpu_uuid"] = args.gpu_uuid
    if config["runtime"].get("device") == "cuda":
        config["runtime"].update(
            resolve_gpu_metadata(str(config["runtime"]["gpu_uuid"]))
        )
    config["runtime"]["scheduler_attempt"] = int(args.scheduler_attempt)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = experiment_id(config, timestamp)
    run_dir = PROJECT_ROOT / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"experiment ID already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    metrics_path = run_dir / "metrics.json"
    samples_path = run_dir / "hcup_samples.npz"
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    source_checkpoint_sha256 = file_sha256(checkpoint)
    checkpoint_verification = verify_checkpoint(checkpoint)

    official_args = copy.deepcopy(source_config["official_args"])
    official_args.update(
        {
            "is_training": 0,
            "model_id": run_id,
            "root_path": str(missing_csv.parent),
            "data_path": missing_csv.name,
            "true_data_path": str(complete_csv),
            "max_train_batches": -1,
            "max_eval_batches": int(config["diagnostic"]["max_eval_batches"]),
            "wandb_mode": "disabled",
            "hcup_diagnostic": True,
            "hcup_checkpoint": str(checkpoint),
            "hcup_metrics_path": str(metrics_path),
            "hcup_samples_path": str(samples_path),
            "hcup_split": str(config["diagnostic"]["split"]),
            "hcup_probe_lag": int(config["diagnostic"]["probe_lag"]),
            "hcup_probe_horizon": int(
                config["diagnostic"].get(
                    "probe_horizon", config["diagnostic"]["probe_lag"]
                )
            ),
            "hcup_threshold": float(config["diagnostic"].get("threshold", 0.0)),
            "gpu": 0,
        }
    )
    # HCUP uses the original baseline architecture and the explicit operational
    # prototype intervention.  Learned FUSR/MIPP paths remain disabled.
    for flag in (
        "use_fusr",
        "use_utility_read_gate",
        "use_fusr_spectral_uncertainty",
        "fusr_eval_paired_diagnostics",
        "fusr_router_only",
        "use_mipp",
        "use_content_mask_factorization",
        "use_mask_intervention_pair",
        "use_stability_write_gate",
    ):
        official_args[flag] = False

    resolved = copy.deepcopy(config)
    resolved["experiment"]["id"] = run_id
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
        gpu_export = (
            "export CUDA_VISIBLE_DEVICES="
            + shlex.quote(str(config["runtime"]["gpu_uuid"]))
            + "\n"
        )
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
            samples = np.load(samples_path)
            count = int(metrics.get("sample_count", 0))
            checks = {
                "sample_count_positive": count > 0,
                "sample_file_count_matches": int(samples["current_index"].size) == count,
                "probe_valid_positive": int(metrics.get("probe_valid_count", 0)) > 0,
                "probe_history_only": metrics.get("probe_uses_observed_history_only") is True,
                "no_current_or_future_target_in_probe": metrics.get(
                    "probe_uses_current_or_future_target"
                ) is False,
                "split_matches": metrics.get("source_split")
                == str(config["diagnostic"]["split"]),
                "probe_lag_matches": int(metrics.get("probe_lag", -1))
                == int(config["diagnostic"]["probe_lag"]),
                "probe_horizon_matches": int(metrics.get("probe_horizon", -1))
                == int(
                    config["diagnostic"].get(
                        "probe_horizon", config["diagnostic"]["probe_lag"]
                    )
                ),
                "finite_current_mse": all(
                    np.isfinite(samples[key]).all()
                    for key in ("current_mse_on", "current_mse_off")
                ),
            }
            if not all(checks.values()):
                raise RuntimeError(f"HCUP artifact checks failed: {checks}")
            baseline_match_abs_error = None
            if (
                config["diagnostic"]["split"] == "test"
                and not config["experiment"].get("smoke_test")
                and int(config["diagnostic"]["max_eval_batches"]) < 0
            ):
                baseline_match_abs_error = abs(
                    float(metrics["prototype_on_mse"]) - float(source_metrics["mse"])
                )
            metrics.update(
                {
                    "experiment_id": run_id,
                    "dataset": config["experiment"]["dataset"],
                    "evaluation_seed": config["experiment"]["seed"],
                    "scope": (
                        "smoke_test_not_formal_result"
                        if config["experiment"].get("smoke_test")
                        else "formal_validation_hcup_cache"
                        if config["diagnostic"]["split"] == "val"
                        else "formal_test_hcup_evaluation"
                    ),
                    "source_baseline_mse": source_metrics.get("mse"),
                    "prototype_on_baseline_match_abs_error": baseline_match_abs_error,
                    "source_checkpoint": str(checkpoint),
                    "source_checkpoint_sha256": source_checkpoint_sha256,
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
