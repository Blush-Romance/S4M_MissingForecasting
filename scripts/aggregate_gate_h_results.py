#!/usr/bin/env python3
"""Aggregate Gate H success and failure artifacts without hand-copying metrics."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def value(payload: dict[str, Any], key: str) -> Any:
    item = payload.get(key)
    return "NA" if item is None else item


def main() -> int:
    fields = [
        "experiment_id",
        "dataset",
        "domain",
        "domain_role",
        "evaluation_seed",
        "status",
        "sample_count",
        "prototype_on_mse",
        "prototype_off_mse",
        "hrr",
        "mru",
        "nus",
        "oracle_selector_mse",
        "oracle_relative_gain_vs_prototype_on",
        "similarity_utility_pearson",
        "similarity_utility_spearman",
        "peak_gpu_memory_mb",
        "source_checkpoint_sha256",
        "resolved_config",
        "samples_path",
        "error",
    ]
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(RUNS.glob("s4m_gate_h__*")):
        status = read_json(run_dir / "status.json")
        metrics = read_json(run_dir / "metrics.json")
        resolved = run_dir / "resolved_config.yaml"
        samples = run_dir / "sample_diagnostics.npz"
        row = {
            "experiment_id": metrics.get("experiment_id", status.get("experiment_id", run_dir.name)),
            "dataset": value(metrics, "dataset"),
            "domain": value(metrics, "domain"),
            "domain_role": value(metrics, "domain_role"),
            "evaluation_seed": value(metrics, "evaluation_seed"),
            "status": status.get("state", "UNKNOWN"),
            "sample_count": value(metrics, "sample_count"),
            "prototype_on_mse": value(metrics, "prototype_on_mse"),
            "prototype_off_mse": value(metrics, "prototype_off_mse"),
            "hrr": value(metrics, "hrr"),
            "mru": value(metrics, "mru"),
            "nus": value(metrics, "nus"),
            "oracle_selector_mse": value(metrics, "oracle_selector_mse"),
            "oracle_relative_gain_vs_prototype_on": value(metrics, "oracle_relative_gain_vs_prototype_on"),
            "similarity_utility_pearson": value(metrics, "similarity_utility_pearson"),
            "similarity_utility_spearman": value(metrics, "similarity_utility_spearman"),
            "peak_gpu_memory_mb": value(metrics, "peak_gpu_memory_mb"),
            "source_checkpoint_sha256": value(metrics, "source_checkpoint_sha256"),
            "resolved_config": str(resolved) if resolved.is_file() else "NA",
            "samples_path": str(samples) if samples.is_file() else "NA",
            "error": status.get("verification_error") or status.get("state") if status.get("state") not in {None, "SUCCESS"} else "",
        }
        rows.append(row)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_lines: list[str] = []
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    csv_text = buffer.getvalue()
    header = "| " + " | ".join(fields) + " |\n"
    separator = "| " + " | ".join(["---"] * len(fields)) + " |\n"
    body = "".join(
        "| " + " | ".join(str(row[field]).replace("|", "\\|") for field in fields) + " |\n"
        for row in rows
    )
    markdown = f"# Gate H Results\n\nGenerated: {dt.datetime.now(dt.timezone.utc).isoformat()}\n\n{header}{separator}{body}"

    REPORTS.mkdir(parents=True, exist_ok=True)
    atomic_text(REPORTS / "latest_gate_h_results.csv", csv_text)
    atomic_text(REPORTS / "latest_gate_h_results.md", markdown)
    atomic_text(REPORTS / f"gate_h_results_{timestamp}.csv", csv_text)
    atomic_text(REPORTS / f"gate_h_results_{timestamp}.md", markdown)
    print(f"aggregated_gate_h_runs={len(rows)}")
    print(f"csv={REPORTS / 'latest_gate_h_results.csv'}")
    print(f"markdown={REPORTS / 'latest_gate_h_results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
