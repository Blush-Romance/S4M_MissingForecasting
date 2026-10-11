#!/usr/bin/env python3
"""Dispatch Stage-20 PG95 training or materialized frozen evaluation jobs."""

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
STATE = ROOT / "runtime" / "stage20_etth1_pg95_pipeline" / "state.json"
MATERIALIZED_ROOT = (
    ROOT / "runtime" / "stage20_etth1_pg95_pipeline" / "materialized"
)
TRAIN_RUNNER = ROOT / "scripts" / "run_s4m_experiment.py"
EVAL_RUNNER = ROOT / "scripts" / "run_stage14_uncertainty_cache.py"


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
    state_path: Path = STATE,
    destination: Path,
) -> Path:
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    source_job_id = str(template.get("source", {}).get("job_id", ""))
    if not source_job_id:
        raise ValueError("Stage-20 evaluation template must specify source.job_id")
    state = load_json(state_path)
    source_job = state.get("jobs", {}).get(source_job_id)
    if not source_job:
        raise KeyError(f"source job is absent from queue state: {source_job_id}")
    if source_job.get("state") != "SUCCESS" or not source_job.get("run_dir"):
        raise RuntimeError(
            f"source job {source_job_id} is not successful: "
            f"state={source_job.get('state')}"
        )
    source_run = resolve_path(source_job["run_dir"])
    source_status_path = source_run / "status.json"
    source_config_path = source_run / "resolved_config.yaml"
    source_metrics_path = source_run / "metrics.json"
    for path in (source_status_path, source_config_path, source_metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_status = load_json(source_status_path)
    source_metrics = load_json(source_metrics_path)
    if source_status.get("state") != "SUCCESS":
        raise RuntimeError(f"source run is not SUCCESS: {source_status_path}")
    checkpoint = resolve_path(source_metrics["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not checkpoint.is_relative_to(source_run):
        raise RuntimeError(
            f"source checkpoint must belong to the formal training run: {checkpoint}"
        )

    template["source"] = {
        "job_id": source_job_id,
        "run_dir": str(source_run),
        "resolved_config": str(source_config_path),
        "metrics": str(source_metrics_path),
        "checkpoint": str(checkpoint),
    }
    template.setdefault("provenance", {})
    template["provenance"].update(
        {
            "materialized_from": str(template_path),
            "queue_state": str(state_path),
            "source_job_id": source_job_id,
        }
    )
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
    source_job_id = config.get("source", {}).get("job_id")
    runner = TRAIN_RUNNER
    effective_config = config_path
    if source_job_id:
        destination = MATERIALIZED_ROOT / (
            f"{config_path.stem}__attempt{int(args.scheduler_attempt):02d}.yaml"
        )
        effective_config = materialize_evaluation_config(
            config_path,
            destination=destination,
        )
        runner = EVAL_RUNNER
        print(
            f"materialized_config={effective_config} "
            f"source_job_id={source_job_id}",
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
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
