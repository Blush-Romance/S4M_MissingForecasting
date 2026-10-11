#!/usr/bin/env python3
"""Incrementally aggregate Step-22.22 and finalize frozen three-seed statistics."""

from __future__ import annotations

import csv
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/stage22_22_traffic_pipeline/state.json"
OLD_STATE = ROOT / "runtime/stage22_21_traffic_pipeline/state.json"
REPORT_JSON = ROOT / "reports/stage22_22_traffic_multiseed_results.json"
REPORT_MD = ROOT / "reports/stage22_22_traffic_multiseed_results.md"
REPORT_CSV = ROOT / "reports/stage22_22_traffic_multiseed_results.csv"
SEEDS = (2026, 2027, 2028)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def job_record(state: dict, job_id: str) -> dict[str, Any] | None:
    job = state.get("jobs", {}).get(job_id)
    if not job or job.get("state") != "SUCCESS" or not job.get("run_dir"):
        return None
    run_dir = Path(job["run_dir"]).resolve()
    paths = {
        "metrics": run_dir / "metrics.json",
        "resource": run_dir / "resource_usage.json",
        "status": run_dir / "status.json",
        "config": run_dir / "resolved_config.yaml",
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    return {
        "job_id": job_id,
        "run_dir": str(run_dir),
        "metrics": load_json(paths["metrics"]),
        "resource": load_json(paths["resource"]),
        "status": load_json(paths["status"]),
        "resolved_config": str(paths["config"]),
    }


def source_records(
    current: dict, old: dict, seed: int
) -> dict[str, dict[str, Any] | None]:
    training_state = old if seed == 2026 else current
    pg_eval_state = old if seed == 2026 else current
    return {
        "s4m_train": job_record(
            training_state, f"traffic_s4m_seed{seed}_formal"
        ),
        "pg95_train": job_record(
            training_state, f"traffic_pg95_seed{seed}_formal"
        ),
        "pg95_validation": job_record(
            pg_eval_state,
            (
                "traffic_pg95_multiscale_validation_seed2026_formal"
                if seed == 2026
                else f"traffic_pg95_multiscale_validation_seed{seed}_formal"
            ),
        ),
        "s4m_test": job_record(
            current, f"traffic_s4m_multiscale_test_seed{seed}_formal"
        ),
        "pg95_test": job_record(
            pg_eval_state,
            (
                "traffic_conditional_single_test_seed2026_formal"
                if seed == 2026
                else f"traffic_pg95_multiscale_test_seed{seed}_formal"
            ),
        ),
    }


def metric_pair(record: dict, mode: str) -> dict[str, float]:
    metrics = record["metrics"]
    if mode == "train":
        return {key: float(metrics[key]) for key in ("mse", "mae")}
    return {
        key: float(metrics[f"complete_target_{key}_{mode}"])
        for key in ("mse", "mae")
    }


def gain(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100.0


def paired_changes(reference: dict, candidate: dict) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "absolute_reduction": reference[metric] - candidate[metric],
            "relative_improvement_percent": gain(
                reference[metric], candidate[metric]
            ),
        }
        for metric in ("mse", "mae")
    }


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
    }


def timing_summary(metrics: dict) -> dict[str, float] | None:
    components = metrics.get("timing_components")
    if not isinstance(components, dict):
        return None
    original = float(components["original_forward_seconds"]["mean_ms"])
    serial_keys = (
        "original_forward_seconds",
        "l5_mask_seconds", "l5_preprocess_seconds", "l5_forward_seconds",
        "l5_fusion_seconds",
        "l20_mask_seconds", "l20_preprocess_seconds", "l20_forward_seconds",
        "l20_fusion_seconds", "multiscale_fusion_seconds",
    )
    serial = sum(float(components[key]["mean_ms"]) for key in serial_keys)
    return {
        "original_forward_mean_ms": original,
        "serial_multiscale_pipeline_mean_ms": serial,
        "serial_over_original_ratio": serial / original,
        "excludes_l1_audit_only_route": True,
        "measured_batches": int(metrics["timing_measured_batches"]),
    }


def finite_pair(pair: dict[str, float]) -> bool:
    return all(math.isfinite(pair[key]) and pair[key] >= 0 for key in pair)


def artifact_checks_pass(record: dict[str, Any]) -> bool:
    checks = record["metrics"].get("artifact_checks")
    return isinstance(checks, dict) and bool(checks) and all(checks.values())


def scientific_decision(
    *,
    complete: bool,
    protocol_pass: bool,
    cnm_all_test_wins: bool,
    pg95_all_test_wins: bool,
) -> str:
    if not complete:
        return "TRAFFIC_MULTISEED_INCOMPLETE"
    if not protocol_pass:
        return "TRAFFIC_MULTISEED_PROTOCOL_FAIL"
    if not cnm_all_test_wins:
        return "TRAFFIC_MULTISEED_UNSTABLE"
    if pg95_all_test_wins:
        return "TRAFFIC_MULTISEED_STRONG_CHAIN_CONFIRMED"
    return "TRAFFIC_MULTISEED_CONFIRMED"


def main() -> int:
    if not STATE.is_file() or not OLD_STATE.is_file():
        return 0
    current = load_json(STATE)
    old = load_json(OLD_STATE)
    queue_records = [
        {
            "job_id": job_id,
            "state": job.get("state"),
            "run_dir": job.get("run_dir"),
            "attempt": job.get("attempt"),
            "last_error_type": job.get("last_error_type"),
        }
        for job_id, job in sorted(current.get("jobs", {}).items())
    ]

    per_seed: list[dict[str, Any]] = []
    missing_sources: list[str] = []
    for seed in SEEDS:
        records = source_records(current, old, seed)
        absent = [name for name, record in records.items() if record is None]
        if absent:
            missing_sources.extend(f"seed{seed}:{name}" for name in absent)
            continue
        s4m_val = metric_pair(records["s4m_train"], "train")
        pg95_val = metric_pair(records["pg95_validation"], "on")
        cnm_val = metric_pair(records["pg95_validation"], "multiscale")
        s4m_test = metric_pair(records["s4m_test"], "on")
        pg95_test = metric_pair(records["pg95_test"], "on")
        cnm_test = metric_pair(records["pg95_test"], "multiscale")
        pairs = (s4m_val, pg95_val, cnm_val, s4m_test, pg95_test, cnm_test)
        protocol_checks = {
            "all_metrics_finite_nonnegative": all(finite_pair(pair) for pair in pairs),
            "validation_samples_1648": (
                int(records["s4m_train"]["metrics"]["validation_samples_processed"])
                == 1648
                and int(records["pg95_validation"]["metrics"]["sample_count"])
                == 1648
            ),
            "test_samples_3408": (
                int(records["s4m_test"]["metrics"]["sample_count"]) == 3408
                and int(records["pg95_test"]["metrics"]["sample_count"]) == 3408
            ),
            "test_split_explicit": (
                records["s4m_test"]["metrics"].get("uses_test_split") is True
                and records["pg95_test"]["metrics"].get("uses_test_split") is True
            ),
            "source_training_validation_only": all(
                record["metrics"].get("test_split_accessed") is False
                for record in (records["s4m_train"], records["pg95_train"])
            ),
            "evaluation_artifact_checks_pass": all(
                artifact_checks_pass(record)
                for record in (
                    records["pg95_validation"], records["s4m_test"],
                    records["pg95_test"],
                )
            ),
        }
        per_seed.append({
            "seed": seed,
            "protocol_checks": protocol_checks,
            "validation": {
                "s4m_original": s4m_val,
                "pg95_original": pg95_val,
                "pg95_cnm_se_multiscale": cnm_val,
                "pg95_vs_s4m": paired_changes(s4m_val, pg95_val),
                "cnm_se_vs_pg95": paired_changes(pg95_val, cnm_val),
                "cnm_se_vs_s4m": paired_changes(s4m_val, cnm_val),
            },
            "test": {
                "s4m_original": s4m_test,
                "pg95_original": pg95_test,
                "pg95_cnm_se_multiscale": cnm_test,
                "pg95_vs_s4m": paired_changes(s4m_test, pg95_test),
                "cnm_se_vs_pg95": paired_changes(pg95_test, cnm_test),
                "cnm_se_vs_s4m": paired_changes(s4m_test, cnm_test),
            },
            "cost": {
                "s4m_parameter_count": int(
                    records["s4m_train"]["metrics"]["parameter_count"]
                ),
                "pg95_parameter_count": int(
                    records["pg95_train"]["metrics"]["parameter_count"]
                ),
                "s4m_training_elapsed_seconds": float(
                    records["s4m_train"]["resource"]["elapsed_seconds"]
                ),
                "pg95_training_elapsed_seconds": float(
                    records["pg95_train"]["resource"]["elapsed_seconds"]
                ),
                "s4m_training_peak_gpu_memory_mb": float(
                    records["s4m_train"]["resource"]["peak_gpu_memory_mb"]
                ),
                "pg95_training_peak_gpu_memory_mb": float(
                    records["pg95_train"]["resource"]["peak_gpu_memory_mb"]
                ),
                "pg95_test_timing": timing_summary(
                    records["pg95_test"]["metrics"]
                ),
            },
            "runs": {
                name: {
                    "job_id": record["job_id"],
                    "run_dir": record["run_dir"],
                    "resolved_config": record["resolved_config"],
                }
                for name, record in records.items()
            },
        })

    complete = len(per_seed) == 3 and not missing_sources
    protocol_pass = complete and all(
        all(item["protocol_checks"].values()) for item in per_seed
    )
    aggregate: dict[str, Any] | None = None
    cnm_all_test_wins = False
    pg95_all_test_wins = False
    if complete:
        aggregate = {"validation": {}, "test": {}}
        for split in ("validation", "test"):
            for method in (
                "s4m_original", "pg95_original", "pg95_cnm_se_multiscale"
            ):
                aggregate[split][method] = {
                    metric: summarize([
                        float(item[split][method][metric]) for item in per_seed
                    ])
                    for metric in ("mse", "mae")
                }
            for comparison in (
                "pg95_vs_s4m", "cnm_se_vs_pg95", "cnm_se_vs_s4m"
            ):
                aggregate[split][comparison] = {
                    metric: {
                        "relative_improvement_percent": summarize([
                            float(item[split][comparison][metric]
                                  ["relative_improvement_percent"])
                            for item in per_seed
                        ]),
                        "win_count": sum(
                            item[split][comparison][metric]
                            ["absolute_reduction"] > 0
                            for item in per_seed
                        ),
                    }
                    for metric in ("mse", "mae")
                }
        cnm_all_test_wins = all(
            item["test"]["cnm_se_vs_pg95"][metric]["absolute_reduction"] > 0
            for item in per_seed for metric in ("mse", "mae")
        )
        pg95_all_test_wins = all(
            item["test"]["pg95_vs_s4m"][metric]["absolute_reduction"] > 0
            for item in per_seed for metric in ("mse", "mae")
        )
    decision = scientific_decision(
        complete=complete,
        protocol_pass=protocol_pass,
        cnm_all_test_wins=cnm_all_test_wins,
        pg95_all_test_wins=pg95_all_test_wins,
    )

    payload = {
        "status": "PASS" if protocol_pass else ("INCOMPLETE" if not complete else "FAIL"),
        "queue_state": current.get("queue_state"),
        "precommitted_seeds": list(SEEDS),
        "all_seeds_included": complete,
        "missing_sources": missing_sources,
        "protocol_pass": protocol_pass,
        "decision": decision,
        "inference_note": (
            "n=3: report mean, sample standard deviation and paired win counts; "
            "do not claim inferential significance. Seed varies training and mask jointly."
        ),
        "queue_records": queue_records,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=[
        "seed", "split", "method", "mse", "mae", "state", "decision",
    ])
    writer.writeheader()
    for item in per_seed:
        for split in ("validation", "test"):
            for method in (
                "s4m_original", "pg95_original", "pg95_cnm_se_multiscale"
            ):
                writer.writerow({
                    "seed": item["seed"], "split": split, "method": method,
                    "mse": item[split][method]["mse"],
                    "mae": item[split][method]["mae"],
                    "state": "SUCCESS", "decision": decision,
                })
    atomic_text(REPORT_CSV, csv_buffer.getvalue())

    counts: dict[str, int] = {}
    for record in queue_records:
        state_name = str(record["state"])
        counts[state_name] = counts.get(state_name, 0) + 1
    lines = [
        "# 步骤22.22：Traffic三seed统计复核",
        "",
        f"- 队列状态：`{current.get('queue_state')}`",
        f"- 任务状态计数：`{json.dumps(counts, sort_keys=True)}`",
        f"- 当前决策：`{decision}`",
        f"- 缺少正式来源：{', '.join(missing_sources) if missing_sources else '无'}",
        "- 统计说明：n=3仅报告均值、样本标准差与逐seed配对胜出数，不声称推断显著性。",
    ]
    if per_seed:
        lines.extend([
            "", "## 已完成seed的主结果", "",
            "| seed | split | S4M MSE/MAE | PG95 MSE/MAE | CNM-SE MSE/MAE | CNM-SE相对PG95 MSE/MAE |",
            "|---:|---|---|---|---|---|",
        ])
        for item in per_seed:
            for split in ("validation", "test"):
                s4m = item[split]["s4m_original"]
                pg = item[split]["pg95_original"]
                cnm = item[split]["pg95_cnm_se_multiscale"]
                change = item[split]["cnm_se_vs_pg95"]
                lines.append(
                    f"| {item['seed']} | {split} | {s4m['mse']:.9f}/{s4m['mae']:.9f} | "
                    f"{pg['mse']:.9f}/{pg['mae']:.9f} | "
                    f"{cnm['mse']:.9f}/{cnm['mae']:.9f} | "
                    f"{change['mse']['relative_improvement_percent']:.6f}%/"
                    f"{change['mae']['relative_improvement_percent']:.6f}% |"
                )
    if aggregate is not None:
        lines.extend(["", "## 三seed均值±样本标准差", ""])
        for split in ("validation", "test"):
            lines.extend([
                f"### {split}", "",
                "| 方法 | MSE | MAE |", "|---|---:|---:|",
            ])
            for method in (
                "s4m_original", "pg95_original", "pg95_cnm_se_multiscale"
            ):
                mse = aggregate[split][method]["mse"]
                mae = aggregate[split][method]["mae"]
                lines.append(
                    f"| {method} | {mse['mean']:.9f} ± {mse['sample_std']:.9f} | "
                    f"{mae['mean']:.9f} ± {mae['sample_std']:.9f} |"
                )
    atomic_text(REPORT_MD, "\n".join(lines) + "\n")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/aggregate_results.py")],
        cwd=ROOT,
        check=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
