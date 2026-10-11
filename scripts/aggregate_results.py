#!/usr/bin/env python3
"""Aggregate persisted run artifacts without inventing missing metrics."""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"


FIELDS = [
    "experiment_id",
    "dataset",
    "model",
    "mask_type",
    "nominal_missing_rate",
    "realized_missing_rate",
    "seed",
    "mse",
    "mae",
    "rmse",
    "runtime_seconds",
    "peak_gpu_memory_mb",
    "parameter_count",
    "mipp_enabled",
    "mipp_total_loss_mean",
    "mipp_paired_content_cosine_mean",
    "mipp_stable_write_acceptance_rate_mean",
    "status",
    "protocol_scope",
    "same_evaluation_protocol",
    "git_commit",
    "resolved_config_path",
    "checkpoint_path",
    "error_summary",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def baseline_commit(run_dir: Path) -> str:
    path = run_dir / "git_state.txt"
    if not path.is_file():
        return "NA"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line == "[commit] exit=0" and index + 1 < len(lines):
            return lines[index + 1].strip()
    return "NA"


def one_row(run_dir: Path) -> dict[str, Any] | None:
    resolved_path = run_dir / "resolved_config.yaml"
    status_path = run_dir / "status.json"
    if not resolved_path.is_file() or not status_path.is_file():
        return None
    cfg = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    status = load_json(status_path)
    metrics = load_json(run_dir / "metrics.json")
    resources = load_json(run_dir / "resource_usage.json")
    exp = cfg.get("experiment", {})
    protocol_scope = exp.get("protocol_scope", metrics.get("scope", "NA"))
    return {
        "experiment_id": status.get("experiment_id", run_dir.name),
        "dataset": exp.get("dataset", "NA"),
        "model": exp.get("model", "NA"),
        "mask_type": exp.get("mask_type", "NA"),
        "nominal_missing_rate": exp.get("nominal_missing_rate", "NA"),
        "realized_missing_rate": exp.get("realized_missing_rate", "NA"),
        "seed": exp.get("seed", "NA"),
        "mse": metrics.get("mse", "NA"),
        "mae": metrics.get("mae", "NA"),
        "rmse": metrics.get("rmse", "NA"),
        "runtime_seconds": resources.get("elapsed_seconds", "NA"),
        "peak_gpu_memory_mb": resources.get("peak_gpu_memory_mb", "NA"),
        "parameter_count": metrics.get("parameter_count", "NA"),
        "mipp_enabled": metrics.get("mipp_enabled", False),
        "mipp_total_loss_mean": metrics.get("mipp_total_loss_mean", "NA"),
        "mipp_paired_content_cosine_mean": metrics.get("mipp_paired_content_cosine_mean", "NA"),
        "mipp_stable_write_acceptance_rate_mean": metrics.get("mipp_stable_write_acceptance_rate_mean", "NA"),
        "status": status.get("state", "NA"),
        "protocol_scope": protocol_scope,
        "same_evaluation_protocol": (
            "smoke_only"
            if exp.get("smoke_test")
            else exp.get("protocol_match", "unknown")
        ),
        "git_commit": baseline_commit(run_dir),
        "resolved_config_path": str(resolved_path),
        "checkpoint_path": metrics.get("checkpoint", "NA"),
        "error_summary": status.get("verification_error") or ("CUDA OOM" if status.get("state") == "OOM" else ""),
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    columns = ["experiment_id", "model", "dataset", "status", "mse", "mae", "runtime_seconds", "peak_gpu_memory_mb", "protocol_scope"]
    lines = ["# Latest Results", "", "Smoke results are execution evidence, not paper results.", "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = [row for run_dir in sorted(RUNS.iterdir()) if run_dir.is_dir() if (row := one_row(run_dir)) is not None]
    rows.sort(key=lambda row: row["experiment_id"])
    REPORTS.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in (REPORTS / "latest_results.csv", REPORTS / f"results_{timestamp}.csv"):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    content = markdown(rows)
    (REPORTS / "latest_results.md").write_text(content, encoding="utf-8")
    (REPORTS / f"results_{timestamp}.md").write_text(content, encoding="utf-8")
    print(f"aggregated_runs={len(rows)}")
    print(f"csv={REPORTS / 'latest_results.csv'}")
    print(f"markdown={REPORTS / 'latest_results.md'}")


if __name__ == "__main__":
    main()
