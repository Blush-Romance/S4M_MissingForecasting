#!/usr/bin/env python3
"""Aggregate Step-22.21 Traffic artifacts and validation decision."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/stage22_21_traffic_pipeline/state.json"
REPORT_JSON = ROOT / "reports/stage22_21_traffic_raw_results.json"
REPORT_MD = ROOT / "reports/stage22_21_traffic_raw_results.md"
REQUIRED_VALIDATION_JOBS = (
    "traffic_s4m_seed2026_formal",
    "traffic_pg95_seed2026_formal",
    "traffic_pg95_multiscale_validation_seed2026_formal",
)
CONDITIONAL_TEST_JOB = "traffic_conditional_single_test_seed2026_formal"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def first_present(record: dict, *keys: str):
    for key in keys:
        if record.get(key) is not None:
            return record[key]
    return None


def gain_percent(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100.0


def main() -> int:
    if not STATE.is_file():
        return 0
    state = json.loads(STATE.read_text(encoding="utf-8"))
    metric_keys = (
        "scope", "mse", "mae", "evaluation_split", "source_split",
        "test_split_accessed", "uses_test_split", "sample_count",
        "test_batches_processed",
        "validation_batches_processed", "validation_samples_processed",
        "complete_target_mse_on", "complete_target_mae_on",
        "complete_target_mse_ensemble_l5", "complete_target_mae_ensemble_l5",
        "complete_target_mse_multiscale", "complete_target_mae_multiscale",
        "prediction_difference_to_original_mse_ratio_l5",
        "prediction_difference_to_original_mse_ratio_l20",
        "parameter_count", "peak_gpu_memory_mb", "artifact_status",
        "test_accessed", "test_authorized", "scientific_decision",
    )
    records = []
    for job_id, job in sorted(state.get("jobs", {}).items()):
        record = {
            "job_id": job_id,
            "state": job.get("state"),
            "run_dir": job.get("run_dir"),
        }
        if job.get("run_dir"):
            metrics_path = Path(job["run_dir"]) / "metrics.json"
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                record.update({key: metrics.get(key) for key in metric_keys})
        records.append(record)
    test_summary = None
    test_job = state.get("jobs", {}).get(CONDITIONAL_TEST_JOB, {})
    if test_job.get("state") == "SUCCESS" and test_job.get("run_dir"):
        test_metrics_path = Path(test_job["run_dir"]) / "metrics.json"
        if test_metrics_path.is_file():
            test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
            if test_metrics.get("uses_test_split") is True:
                original = {
                    metric: float(test_metrics[f"complete_target_{metric}_on"])
                    for metric in ("mse", "mae")
                }
                ensemble_l5 = {
                    metric: float(
                        test_metrics[f"complete_target_{metric}_ensemble_l5"]
                    )
                    for metric in ("mse", "mae")
                }
                multiscale = {
                    metric: float(
                        test_metrics[f"complete_target_{metric}_multiscale"]
                    )
                    for metric in ("mse", "mae")
                }
                test_summary = {
                    "run_dir": str(Path(test_job["run_dir"]).resolve()),
                    "artifact_status": test_metrics.get("artifact_status"),
                    "test_batches_processed": int(
                        test_metrics["test_batches_processed"]
                    ),
                    "test_samples_processed": int(test_metrics["sample_count"]),
                    "original": original,
                    "original_plus_l5": ensemble_l5,
                    "multiscale": multiscale,
                    "gains_percent": {
                        "multiscale_vs_original": {
                            metric: gain_percent(original[metric], multiscale[metric])
                            for metric in ("mse", "mae")
                        },
                        "multiscale_vs_original_plus_l5": {
                            metric: gain_percent(
                                ensemble_l5[metric], multiscale[metric]
                            )
                            for metric in ("mse", "mae")
                        },
                    },
                    "decision": (
                        "TRAFFIC_TEST_MULTISCALE_CONFIRMED"
                        if all(
                            multiscale[metric] < original[metric]
                            and multiscale[metric] < ensemble_l5[metric]
                            for metric in ("mse", "mae")
                        )
                        else "TRAFFIC_TEST_MULTISCALE_NOT_CONFIRMED"
                    ),
                }
    atomic_text(
        REPORT_JSON,
        json.dumps(
            {
                "queue_state": state.get("queue_state"),
                "records": records,
                "test_summary": test_summary,
            },
            indent=2, sort_keys=True,
        ) + "\n",
    )
    lines = [
        "# 步骤22.21：Traffic原始结果",
        "",
        "| job | 状态 | split | 原始MSE | 原始MAE | 多尺度MSE | 多尺度MAE | 样本 | batch | test访问 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        split = first_present(record, "evaluation_split", "source_split")
        original_mse = first_present(record, "complete_target_mse_on", "mse")
        original_mae = first_present(record, "complete_target_mae_on", "mae")
        samples = first_present(
            record, "validation_samples_processed", "sample_count"
        )
        batches = first_present(
            record, "validation_batches_processed", "test_batches_processed"
        )
        test_accessed = first_present(
            record, "test_accessed", "test_split_accessed", "uses_test_split"
        )
        lines.append(
            f"| {record['job_id']} | {record.get('state')} | "
            f"{split if split is not None else 'NA'} | "
            f"{original_mse if original_mse is not None else 'NA'} | "
            f"{original_mae if original_mae is not None else 'NA'} | "
            f"{record.get('complete_target_mse_multiscale', 'NA')} | "
            f"{record.get('complete_target_mae_multiscale', 'NA')} | "
            f"{samples if samples is not None else 'NA'} | "
            f"{batches if batches is not None else 'NA'} | "
            f"{test_accessed if test_accessed is not None else 'NA'} |"
        )
    if test_summary is not None:
        lines.extend([
            "",
            "## 唯一Traffic测试结论",
            "",
            f"- 决策：`{test_summary['decision']}`",
            f"- 完整测试：{test_summary['test_batches_processed']} batch / "
            f"{test_summary['test_samples_processed']} 样本",
            f"- 多尺度相对PG95原始路径：MSE提升"
            f"{test_summary['gains_percent']['multiscale_vs_original']['mse']:.6f}%、"
            f"MAE提升{test_summary['gains_percent']['multiscale_vs_original']['mae']:.6f}%",
            f"- 多尺度相对原始+L5：MSE提升"
            f"{test_summary['gains_percent']['multiscale_vs_original_plus_l5']['mse']:.6f}%、"
            f"MAE提升{test_summary['gains_percent']['multiscale_vs_original_plus_l5']['mae']:.6f}%",
        ])
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/aggregate_results.py")],
        cwd=ROOT, check=False,
    )
    if all(
        state.get("jobs", {}).get(job, {}).get("state") == "SUCCESS"
        for job in REQUIRED_VALIDATION_JOBS
    ):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/analyze_stage22_21_traffic_validation.py"),
            ],
            cwd=ROOT, check=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
