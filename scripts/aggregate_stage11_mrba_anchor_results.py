#!/usr/bin/env python3
"""Aggregate the MRBA anchor queue and run the frozen comparison when ready."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime" / "stage11_mrba_anchor_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage11_mrba_anchor_raw_results.json"
REPORT_MD = ROOT / "reports" / "stage11_mrba_anchor_raw_results.md"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    if not STATE.is_file():
        return 0
    state = json.loads(STATE.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    metric_keys = (
        "mse",
        "mae",
        "optimizer_steps",
        "parameter_count",
        "peak_gpu_memory_mb",
        "peak_gpu_reserved_mb",
        "mipp_write_gate_mode",
        "mipp_write_budget",
        "mipp_stable_write_acceptance_rate_mean",
        "mipp_write_eligible_count_mean",
        "mipp_write_sampled_count_mean",
        "mipp_write_updated_count_mean",
        "mipp_write_selected_stability_mean",
        "mipp_write_mask_candidate_pairwise_distance_mean",
        "mipp_write_mask_selected_pairwise_distance_mean",
        "mipp_write_mask_coverage_distance_mean",
        "mipp_write_mask_topk_coverage_distance_mean",
        "mipp_write_mask_coverage_gain_vs_topk_mean",
        "mipp_write_mask_candidate_missing_rate_mean",
        "mipp_write_mask_selected_missing_rate_mean",
        "checkpoint_load_verified",
    )
    for job_id, job in sorted(state.get("jobs", {}).items()):
        record: dict[str, Any] = {
            "job_id": job_id,
            "state": job.get("state"),
            "attempt_count": job.get("attempt_count", 0),
            "run_dir": job.get("run_dir"),
        }
        if job.get("run_dir"):
            metrics_path = Path(job["run_dir"]) / "metrics.json"
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                for key in metric_keys:
                    record[key] = metrics.get(key)
        records.append(record)

    payload = {
        "queue_state": state.get("queue_state"),
        "queue_finished_at": state.get("finished_at"),
        "records": records,
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 步骤11：MRBA锚点实验原始结果",
        "",
        "| job | 状态 | MSE | MAE | coverage gain | coverage distance | run |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['job_id']} | {record.get('state')} | "
            f"{record.get('mse', 'NA')} | {record.get('mae', 'NA')} | "
            f"{record.get('mipp_write_mask_coverage_gain_vs_topk_mean', 'NA')} | "
            f"{record.get('mipp_write_mask_coverage_distance_mean', 'NA')} | "
            f"`{record.get('run_dir') or ''}` |"
        )
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")

    jobs = state.get("jobs", {})
    ready = all(
        jobs.get(f"weather_mrba50_seed{seed}_formal", {}).get("state") == "SUCCESS"
        for seed in (2024, 2025)
    )
    if ready:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "compare_stage11_mrba_anchor_results.py")],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "aggregate_results.py")],
            cwd=ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
