#!/usr/bin/env python3
"""Dispatch Step-22.21 Traffic training, validation, and gated test."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/stage22_21_traffic_pipeline/state.json"
MATERIALIZED_ROOT = ROOT / "runtime/stage22_21_traffic_pipeline/materialized"
GATE_REPORT = ROOT / "reports/stage22_21_traffic_validation_gate.json"
TRAIN_RUNNER = ROOT / "scripts/run_s4m_experiment.py"
EVAL_RUNNER = ROOT / "scripts/run_stage14_uncertainty_cache.py"
GATE_ANALYZER = ROOT / "scripts/analyze_stage22_21_traffic_validation.py"
TEST_VARIANT = "stage22_21_traffic_test"
SOURCE_JOB = "traffic_pg95_seed2026_formal"


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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
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
    if source_job_id != SOURCE_JOB:
        raise ValueError(f"Traffic evaluation source must be {SOURCE_JOB}")
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
    paths = {
        "status": source_run / "status.json",
        "resolved_config": source_run / "resolved_config.yaml",
        "metrics": source_run / "metrics.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    source_status = load_json(paths["status"])
    source_metrics = load_json(paths["metrics"])
    if source_status.get("state") != "SUCCESS":
        raise RuntimeError(f"source run is not SUCCESS: {paths['status']}")
    expected_identity = {
        "evaluation_split": "validation",
        "post_train_evaluation": "validation",
        "test_split_accessed": False,
        "chronological_loader": True,
    }
    actual_identity = {
        key: source_metrics.get(key) for key in expected_identity
    }
    if actual_identity != expected_identity:
        raise RuntimeError(
            f"source metrics are not validation-only: {actual_identity}"
        )
    checkpoint = resolve_path(source_metrics["checkpoint"])
    if not checkpoint.is_file() or not checkpoint.is_relative_to(source_run):
        raise RuntimeError(
            f"source checkpoint is absent or outside formal run: {checkpoint}"
        )
    template["source"] = {
        "job_id": source_job_id,
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
    })
    atomic_yaml(destination, template)
    return destination


def enforce_test_gate(state_path: Path = STATE) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(GATE_ANALYZER)], cwd=ROOT, check=False
    )
    if not GATE_REPORT.is_file():
        raise RuntimeError(
            f"scientific gate produced no report, exit={completed.returncode}"
        )
    gate = load_json(GATE_REPORT)
    state = load_json(state_path)
    expected_run = resolve_path(state["jobs"][SOURCE_JOB]["run_dir"])
    reported_run = resolve_path(gate["pg95"]["run_dir"])
    if expected_run != reported_run:
        raise RuntimeError("scientific gate references a stale PG95 run")
    if gate.get("test_authorized") is True:
        if completed.returncode != 0 or gate.get("status") != "PASS":
            raise RuntimeError("test was authorized despite protocol failure")
        if gate.get("decision") != "TRAFFIC_VALIDATION_PASS_ONE_TEST_AUTHORIZED":
            raise RuntimeError("test authorization decision is inconsistent")
    return gate


def record_skipped_test(
    config_path: Path,
    config: dict[str, Any],
    gate: dict[str, Any],
    scheduler_attempt: int,
) -> Path:
    identifier = (
        "stage22_21_scientific_gate__traffic__test_not_authorized"
        f"__seed2026__{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = ROOT / "runs" / identifier
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved = {
        **config,
        "experiment": {
            **config["experiment"],
            "id": identifier,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
        "runtime": {
            **config["runtime"],
            "scheduler_attempt": int(scheduler_attempt),
        },
        "scientific_gate": gate,
        "paths": {"run_dir": str(run_dir), "config_source": str(config_path)},
    }
    atomic_yaml(run_dir / "resolved_config.yaml", resolved)
    atomic_json(run_dir / "metrics.json", {
        "experiment_id": identifier,
        "dataset": "traffic",
        "method": "cnm_se_equal_rate_multiscale_pg95",
        "scope": "test_skipped_by_frozen_validation_gate",
        "artifact_status": "SKIPPED_BY_VALIDATION_GATE",
        "test_accessed": False,
        "test_authorized": False,
        "scientific_decision": gate.get("decision"),
        "validation_gate_report": str(GATE_REPORT),
    })
    atomic_json(run_dir / "resource_usage.json", {
        "elapsed_seconds": 0.0,
        "peak_gpu_memory_mb": 0.0,
        "return_code": 0,
    })
    atomic_json(run_dir / "status.json", {
        "experiment_id": identifier,
        "state": "SUCCESS",
        "result": "SKIPPED_BY_VALIDATION_GATE",
        "test_accessed": False,
        "test_authorized": False,
        "scientific_decision": gate.get("decision"),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "return_code": 0,
    })
    (run_dir / "command.sh").write_text(
        "# Test intentionally not executed: frozen validation gate failed.\n",
        encoding="utf-8",
    )
    (run_dir / "env.txt").write_text(
        "No model process launched; gate-only record.\n", encoding="utf-8"
    )
    (run_dir / "git_state.txt").write_text(
        "See validation source run artifacts.\n", encoding="utf-8"
    )
    (run_dir / "code.diff").write_text("", encoding="utf-8")
    print(f"run_dir={run_dir}", flush=True)
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)
    return run_dir


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
        variant = config.get("diagnostic", {}).get("variant")
        if variant == TEST_VARIANT:
            gate = enforce_test_gate()
            if gate.get("test_authorized") is not True:
                record_skipped_test(
                    config_path, config, gate, args.scheduler_attempt
                )
                return 0
        destination = MATERIALIZED_ROOT / (
            f"{config_path.stem}__attempt{int(args.scheduler_attempt):02d}.yaml"
        )
        effective_config = materialize_evaluation_config(
            config_path, destination=destination
        )
        runner = EVAL_RUNNER
        print(
            f"materialized_config={effective_config} source_job_id={source_job_id}",
            flush=True,
        )
    command = [
        sys.executable, str(runner), "--config", str(effective_config),
        "--gpu-uuid", args.gpu_uuid,
        "--scheduler-attempt", str(args.scheduler_attempt),
    ]
    return int(subprocess.run(command, cwd=ROOT, check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
