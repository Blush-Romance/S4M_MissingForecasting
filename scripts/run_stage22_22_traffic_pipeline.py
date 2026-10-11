#!/usr/bin/env python3
"""Dispatch frozen Step-22.22 Traffic training and read-only evaluations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/stage22_22_traffic_pipeline/state.json"
MATERIALIZED_ROOT = ROOT / "runtime/stage22_22_traffic_pipeline/materialized"
TRAIN_RUNNER = ROOT / "scripts/run_s4m_experiment.py"
EVAL_RUNNER = ROOT / "scripts/run_stage14_uncertainty_cache.py"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def materialize_evaluation_config(
    template_path: Path,
    *,
    default_state_path: Path = STATE,
    destination: Path,
) -> Path:
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    source_spec = template.get("source", {})
    source_job_id = str(source_spec.get("job_id", ""))
    if not source_job_id:
        raise ValueError("evaluation config requires source.job_id")
    state_path = resolve_path(
        source_spec.get("queue_state", str(default_state_path))
    )
    state = load_json(state_path)
    source_job = state.get("jobs", {}).get(source_job_id)
    if not source_job:
        raise KeyError(
            f"source job is absent from {state_path}: {source_job_id}"
        )
    if source_job.get("state") != "SUCCESS" or not source_job.get("run_dir"):
        raise RuntimeError(
            f"source job {source_job_id} is not successful: "
            f"state={source_job.get('state')}"
        )
    source_run = resolve_path(source_job["run_dir"])
    paths = {
        "status": source_run / "status.json",
        "resolved_config": source_run / "resolved_config.yaml",
        "metrics": source_run / "metrics.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    status = load_json(paths["status"])
    metrics = load_json(paths["metrics"])
    if status.get("state") != "SUCCESS":
        raise RuntimeError(f"source run is not SUCCESS: {paths['status']}")
    expected_identity = {
        "evaluation_split": "validation",
        "post_train_evaluation": "validation",
        "test_split_accessed": False,
        "chronological_loader": True,
    }
    actual_identity = {key: metrics.get(key) for key in expected_identity}
    if actual_identity != expected_identity:
        raise RuntimeError(
            f"source metrics are not validation-only: {actual_identity}"
        )
    checkpoint = resolve_path(metrics["checkpoint"])
    if not checkpoint.is_file() or not checkpoint.is_relative_to(source_run):
        raise RuntimeError(
            f"source checkpoint is absent or outside formal run: {checkpoint}"
        )
    if bool(template["experiment"].get("smoke_test")) is False:
        source_config = yaml.safe_load(
            paths["resolved_config"].read_text(encoding="utf-8")
        )
        if bool(source_config.get("experiment", {}).get("smoke_test")):
            raise RuntimeError("formal evaluation cannot consume a smoke checkpoint")

    template["source"] = {
        "job_id": source_job_id,
        "queue_state": str(state_path),
        "run_dir": str(source_run),
        "resolved_config": str(paths["resolved_config"]),
        "metrics": str(paths["metrics"]),
        "checkpoint": str(checkpoint),
    }
    template.setdefault("provenance", {})
    template["provenance"].update({
        "materialized_from": str(template_path),
        "queue_state": str(state_path),
        "source_job_id": source_job_id,
        "source_is_validation_only": True,
        "test_precommitted_without_validation_selection": True,
    })
    atomic_yaml(destination, template)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--scheduler-attempt", type=int, default=0)
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runner = TRAIN_RUNNER
    effective_config = config_path
    if config.get("source", {}).get("job_id"):
        destination = MATERIALIZED_ROOT / (
            f"{config_path.stem}__attempt{int(args.scheduler_attempt):02d}.yaml"
        )
        effective_config = materialize_evaluation_config(
            config_path, destination=destination
        )
        runner = EVAL_RUNNER
        print(
            f"materialized_config={effective_config} "
            f"source_job_id={config['source']['job_id']}",
            flush=True,
        )
    command = [
        sys.executable,
        str(runner),
        "--config",
        str(effective_config),
        "--gpu-uuid",
        args.gpu_uuid,
        "--scheduler-attempt",
        str(args.scheduler_attempt),
    ]
    return int(subprocess.run(command, cwd=ROOT, check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
