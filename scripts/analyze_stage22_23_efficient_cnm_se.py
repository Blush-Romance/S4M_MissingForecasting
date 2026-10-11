#!/usr/bin/env python3
"""Validation-only selection of an efficient two-view Traffic CNM-SE.

Candidate selection is deliberately implemented before and independently from
loading any exposed test artifact.  Test artifacts are only loaded when a
validation candidate has already been frozen.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
CURRENT_STATE = ROOT / "runtime/stage22_22_traffic_pipeline/state.json"
OLD_STATE = ROOT / "runtime/stage22_21_traffic_pipeline/state.json"
REPORT_JSON = ROOT / "reports/stage22_23_efficient_cnm_se.json"
REPORT_MD = ROOT / "reports/stage22_23_efficient_cnm_se.md"
SEEDS = (2026, 2027, 2028)
CANDIDATES = {"E1": 1, "E5": 5, "E20": 20}
TIE_PRIORITY = {"E5": 0, "E20": 1, "E1": 2}
RETENTION_THRESHOLD = 0.80
TIE_TOLERANCE = 1e-12


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def successful_metrics(state: dict[str, Any], job_id: str) -> tuple[dict, str]:
    job = state.get("jobs", {}).get(job_id)
    if not job or job.get("state") != "SUCCESS" or not job.get("run_dir"):
        raise RuntimeError(f"required successful job missing: {job_id}")
    run_dir = Path(job["run_dir"]).resolve()
    metrics_path = run_dir / "metrics.json"
    status_path = run_dir / "status.json"
    config_path = run_dir / "resolved_config.yaml"
    if not all(path.is_file() for path in (metrics_path, status_path, config_path)):
        raise RuntimeError(f"incomplete run artifacts: {run_dir}")
    status = load_json(status_path)
    if status.get("state") != "SUCCESS" or status.get("return_code") != 0:
        raise RuntimeError(f"run status is not successful: {run_dir}")
    return load_json(metrics_path), str(run_dir)


def metric_pair(metrics: dict[str, Any], route: str) -> dict[str, float]:
    return {
        metric: float(metrics[f"complete_target_{metric}_{route}"])
        for metric in ("mse", "mae")
    }


def gain_percent(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100.0


def protocol_checks(metrics: dict[str, Any], split: str) -> dict[str, bool]:
    checks = metrics.get("artifact_checks", {})
    routes = ("on", "ensemble_l1", "ensemble_l5", "ensemble_l20", "multiscale")
    values = [
        float(metrics[f"complete_target_{metric}_{route}"])
        for route in routes for metric in ("mse", "mae")
    ]
    return {
        "artifact_status_pass": metrics.get("artifact_status") == "PASS",
        "all_artifact_checks_pass": (
            isinstance(checks, dict) and bool(checks) and all(checks.values())
        ),
        "traffic_frozen_pg95_multiscale": (
            metrics.get("dataset") == "traffic"
            and metrics.get("method") == "cnm_se_equal_rate_multiscale_pg95"
            and metrics.get("updates_model_or_memory") is False
        ),
        "expected_split_and_count": (
            metrics.get("source_split") == split
            and metrics.get("uses_test_split") is (split == "test")
            and int(metrics.get("sample_count", -1))
            == (1648 if split == "val" else 3408)
        ),
        "equal_count_nested_protocol": (
            metrics.get("block_lengths") == [1, 5, 20]
            and float(metrics.get("target_additional_missing_rate", -1)) == 0.15
            and metrics.get("target_count_rounding") == "numpy_rint_ties_to_even"
            and bool(checks.get("strict_equal_removed_count_per_sample"))
        ),
        "fixed_multiscale_weights": metrics.get("multiscale_weights") == {
            "original": 0.5, "l5": 0.25, "l20": 0.25,
        },
        "all_metrics_finite_nonnegative": all(
            math.isfinite(value) and value >= 0.0 for value in values
        ),
    }


def validation_record(seed: int, metrics: dict[str, Any], run_dir: str) -> dict:
    return {
        "seed": seed,
        "run_dir": run_dir,
        "protocol_checks": protocol_checks(metrics, "val"),
        "original": metric_pair(metrics, "on"),
        "multiscale": metric_pair(metrics, "multiscale"),
        "candidates": {
            name: metric_pair(metrics, f"ensemble_l{block}")
            for name, block in CANDIDATES.items()
        },
        "timing_components": metrics.get("timing_components"),
        "timing_measured_batches": int(metrics.get("timing_measured_batches", 0)),
    }


def load_validation_records(current: dict, old: dict) -> list[dict]:
    records = []
    for seed in SEEDS:
        state = old if seed == 2026 else current
        job_id = f"traffic_pg95_multiscale_validation_seed{seed}_formal"
        metrics, run_dir = successful_metrics(state, job_id)
        records.append(validation_record(seed, metrics, run_dir))
    return records


def route_cost(timing: dict[str, Any], block: int) -> dict[str, float]:
    original = float(timing["original_forward_seconds"]["mean_ms"])
    nested = sum(float(timing[f"l{block}_{part}_seconds"]["mean_ms"])
                 for part in ("mask", "preprocess", "forward", "fusion"))
    three_view = original + sum(
        float(timing[f"l{scale}_{part}_seconds"]["mean_ms"])
        for scale in (5, 20) for part in ("mask", "preprocess", "forward", "fusion")
    ) + float(timing["multiscale_fusion_seconds"]["mean_ms"])
    two_view = original + nested
    return {
        "original_forward_mean_ms": original,
        "two_view_serial_mean_ms": two_view,
        "two_view_over_original_ratio": two_view / original,
        "three_view_serial_mean_ms": three_view,
        "two_view_over_three_view_ratio": two_view / three_view,
        "serial_latency_reduction_vs_three_view_percent":
            (three_view - two_view) / three_view * 100.0,
    }


def summarize_candidate(records: list[dict], name: str) -> dict[str, Any]:
    block = CANDIDATES[name]
    per_seed = []
    for record in records:
        original = record["original"]
        candidate = record["candidates"][name]
        multiscale = record["multiscale"]
        per_seed.append({
            "seed": record["seed"],
            "original": original,
            "candidate": candidate,
            "multiscale": multiscale,
            "gain_vs_original_percent": {
                metric: gain_percent(original[metric], candidate[metric])
                for metric in ("mse", "mae")
            },
            "multiscale_gain_vs_original_percent": {
                metric: gain_percent(original[metric], multiscale[metric])
                for metric in ("mse", "mae")
            },
            "timing": route_cost(record["timing_components"], block),
        })
    average_gain = {
        metric: statistics.mean(
            item["gain_vs_original_percent"][metric] for item in per_seed
        ) for metric in ("mse", "mae")
    }
    average_multiscale_gain = {
        metric: statistics.mean(
            item["multiscale_gain_vs_original_percent"][metric]
            for item in per_seed
        ) for metric in ("mse", "mae")
    }
    retention = {
        metric: average_gain[metric] / average_multiscale_gain[metric]
        if average_multiscale_gain[metric] > 0 else float("-inf")
        for metric in ("mse", "mae")
    }
    gates = {
        "mse_improves_all_three_seeds": all(
            item["gain_vs_original_percent"]["mse"] > 0 for item in per_seed
        ),
        "mae_improves_all_three_seeds": all(
            item["gain_vs_original_percent"]["mae"] > 0 for item in per_seed
        ),
        "mse_retention_at_least_80_percent": retention["mse"] >= 0.80,
        "mae_retention_at_least_80_percent": retention["mae"] >= 0.80,
    }
    return {
        "name": name,
        "block_length": block,
        "fixed_weights": {"original": 0.5, f"l{block}": 0.5},
        "per_seed": per_seed,
        "average_gain_vs_original_percent": average_gain,
        "average_multiscale_gain_vs_original_percent": average_multiscale_gain,
        "gain_retention_fraction": retention,
        "selection_score": average_gain["mse"] + average_gain["mae"],
        "gates": gates,
        "passes_validation_gate": all(gates.values()),
    }


def prefer(left: dict, right: dict) -> bool:
    score_delta = left["selection_score"] - right["selection_score"]
    if abs(score_delta) > TIE_TOLERANCE:
        return score_delta > 0
    mse_delta = (
        left["average_gain_vs_original_percent"]["mse"]
        - right["average_gain_vs_original_percent"]["mse"]
    )
    if abs(mse_delta) > TIE_TOLERANCE:
        return mse_delta > 0
    return TIE_PRIORITY[left["name"]] < TIE_PRIORITY[right["name"]]


def select_candidate(records: list[dict]) -> dict[str, Any]:
    summaries = {name: summarize_candidate(records, name) for name in CANDIDATES}
    eligible = [item for item in summaries.values()
                if item["passes_validation_gate"]]
    selected = None
    for item in eligible:
        if selected is None or prefer(item, selected):
            selected = item
    return {
        "selection_inputs": "validation_only",
        "retention_threshold": RETENTION_THRESHOLD,
        "candidate_summaries": summaries,
        "eligible_candidates": [item["name"] for item in eligible],
        "selected_candidate": None if selected is None else selected["name"],
    }


def load_test_records(
    current: dict, old: dict, selected: str,
) -> list[dict[str, Any]]:
    route = f"ensemble_l{CANDIDATES[selected]}"
    records = []
    for seed in SEEDS:
        state = old if seed == 2026 else current
        job_id = (
            "traffic_conditional_single_test_seed2026_formal"
            if seed == 2026 else f"traffic_pg95_multiscale_test_seed{seed}_formal"
        )
        metrics, run_dir = successful_metrics(state, job_id)
        original = metric_pair(metrics, "on")
        candidate = metric_pair(metrics, route)
        gain = {
            metric: gain_percent(original[metric], candidate[metric])
            for metric in ("mse", "mae")
        }
        records.append({
            "seed": seed,
            "run_dir": run_dir,
            "protocol_checks": protocol_checks(metrics, "test"),
            "original": original,
            "candidate": candidate,
            "gain_vs_original_percent": gain,
            "both_metrics_improve": gain["mse"] > 0 and gain["mae"] > 0,
            "timing": route_cost(metrics["timing_components"], CANDIDATES[selected]),
        })
    return records


def analyze(
    validation_records: list[dict],
    test_loader: Callable[[str], list[dict]],
) -> dict[str, Any]:
    protocol_pass = all(
        all(record["protocol_checks"].values()) for record in validation_records
    )
    selection = select_candidate(validation_records)
    selected = selection["selected_candidate"] if protocol_pass else None
    test_records = None
    test_gate_pass = None
    if selected is not None:
        # This is intentionally the first point at which test artifacts are read.
        test_records = test_loader(selected)
        test_gate_pass = all(
            all(record["protocol_checks"].values())
            and record["both_metrics_improve"] for record in test_records
        )
    if not protocol_pass:
        decision = "EFFICIENT_TWO_VIEW_PROTOCOL_FAIL"
    elif selected is None:
        decision = "EFFICIENT_TWO_VIEW_STOP"
    elif not test_gate_pass:
        decision = "EFFICIENT_TWO_VIEW_VALIDATION_SELECTED_TEST_EXPLORATORY_FAIL"
    else:
        decision = "EFFICIENT_TWO_VIEW_FRESH_SEED_AUTHORIZED"
    return {
        "step": "22.23.2",
        "protocol": "validation-only two-view selection; exposed test is post-hoc",
        "validation_protocol_pass": protocol_pass,
        "validation_records": validation_records,
        "selection": selection,
        "test_sources_accessed": test_records is not None,
        "test_exploratory_records": test_records,
        "test_exploratory_gate_pass": test_gate_pass,
        "decision": decision,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 步骤22.23 高效两路CNM-SE筛选结果", "",
        f"- 决策：`{report['decision']}`",
        f"- 验证协议：{'PASS' if report['validation_protocol_pass'] else 'FAIL'}",
        f"- 冻结候选：`{report['selection']['selected_candidate'] or 'NONE'}`",
        f"- 是否读取已有test：`{report['test_sources_accessed']}`", "",
        "## Validation-only候选比较", "",
        "| 候选 | 平均MSE改善 | 平均MAE改善 | MSE收益保留 | MAE收益保留 | 3seed MSE全胜 | 3seed MAE全胜 | 门禁 |",
        "|---|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for name in CANDIDATES:
        item = report["selection"]["candidate_summaries"][name]
        gains = item["average_gain_vs_original_percent"]
        retention = item["gain_retention_fraction"]
        gates = item["gates"]
        lines.append(
            f"| {name} | {gains['mse']:.6f}% | {gains['mae']:.6f}% | "
            f"{retention['mse'] * 100:.2f}% | {retention['mae'] * 100:.2f}% | "
            f"{gates['mse_improves_all_three_seeds']} | "
            f"{gates['mae_improves_all_three_seeds']} | "
            f"{'PASS' if item['passes_validation_gate'] else 'FAIL'} |"
        )
    lines += ["", "## 结论", ""]
    if report["decision"] == "EFFICIENT_TWO_VIEW_STOP":
        lines.append(
            "没有两路候选同时满足三seed双指标全胜与双指标各保留至少80%三路收益；"
            "按预注册规则停止两路精度压缩，不生成seed2029。"
        )
        lines.append(
            "下一步应保持三路预测数学等价，验证把三个视图合并为批处理并行前向能否降低墙钟时间。"
        )
    elif report["decision"] == "EFFICIENT_TWO_VIEW_FRESH_SEED_AUTHORIZED":
        lines.append("验证门及已有test事后探索门均通过，可生成seed2029作独立确认。")
    else:
        lines.append("按报告决策停止或修复协议，不启动未经授权的seed2029。")
    lines += [
        "", "注：E1/E5/E20分别表示original与L=1/L=5/L=20嵌套视图的0.5/0.5融合。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    current = load_json(CURRENT_STATE)
    old = load_json(OLD_STATE)
    validation_records = load_validation_records(current, old)
    report = analyze(
        validation_records,
        lambda selected: load_test_records(current, old, selected),
    )
    atomic_text(REPORT_JSON, json.dumps(report, indent=2, sort_keys=True) + "\n")
    atomic_text(REPORT_MD, markdown(report))
    print(json.dumps({
        "decision": report["decision"],
        "selected_candidate": report["selection"]["selected_candidate"],
        "eligible_candidates": report["selection"]["eligible_candidates"],
        "test_sources_accessed": report["test_sources_accessed"],
    }, indent=2, sort_keys=True))
    return 0 if report["validation_protocol_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
