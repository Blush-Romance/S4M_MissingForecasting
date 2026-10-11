#!/usr/bin/env python3
"""Select a causal HCUP aggregation using validation artifacts only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from analyze_fusr_v3_temporal_generalization import (
    average_precision,
    atomic_text,
    binary_auc,
    bootstrap_gain_interval,
    calibrate_threshold,
)


ROOT = Path(__file__).resolve().parents[1]
QUEUE_STATE = ROOT / "runtime" / "stage8_hcup_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage8_fusr_v3_hcup_validation.json"
REPORT_MD = ROOT / "reports" / "stage8_fusr_v3_hcup_validation.md"
FORMAL_JOB = "weather_hcup_validation_formal"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_formal_run(queue_state: Path) -> Path:
    state = json.loads(queue_state.read_text(encoding="utf-8"))
    job = state.get("jobs", {}).get(FORMAL_JOB, {})
    if job.get("state") != "SUCCESS" or not job.get("run_dir"):
        raise RuntimeError(f"{FORMAL_JOB} is not SUCCESS in {queue_state}")
    run_dir = Path(job["run_dir"])
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    return run_dir


def causal_ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    state: float | None = None
    for index, value in enumerate(values):
        if np.isfinite(value):
            state = float(value) if state is None else alpha * float(value) + (1.0 - alpha) * state
        if state is not None:
            output[index] = state
    return output


def causal_rolling(values: np.ndarray, window: int, reducer: str) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    history: list[float] = []
    for index, value in enumerate(values):
        if np.isfinite(value):
            history.append(float(value))
        local = history[-window:]
        if local:
            output[index] = float(np.mean(local) if reducer == "mean" else np.median(local))
    return output


def evaluate(
    scores: np.ndarray,
    threshold: float,
    index: np.ndarray,
    mse_on: np.ndarray,
    mse_off: np.ndarray,
    mae_on: np.ndarray,
    mae_off: np.ndarray,
) -> dict[str, Any]:
    local_score = scores[index]
    gate = local_score >= threshold
    routed_mse = np.where(gate, mse_on[index], mse_off[index])
    routed_mae = np.where(gate, mae_on[index], mae_off[index])
    labels = mse_off[index] > mse_on[index]
    on_mse = float(np.mean(mse_on[index]))
    on_mae = float(np.mean(mae_on[index]))
    return {
        "sample_count": int(index.size),
        "gate_rate": float(np.mean(gate)),
        "abstention_rate": float(1.0 - np.mean(gate)),
        "auroc": binary_auc(labels, local_score),
        "auprc": average_precision(labels, local_score),
        "prototype_on_mse": on_mse,
        "routed_mse": float(np.mean(routed_mse)),
        "mse_relative_gain": float((on_mse - np.mean(routed_mse)) / on_mse),
        "prototype_on_mae": on_mae,
        "routed_mae": float(np.mean(routed_mae)),
        "mae_relative_gain": float((on_mae - np.mean(routed_mae)) / on_mae),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-state", type=Path, default=QUEUE_STATE)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()
    run_dir = load_formal_run(args.queue_state)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    samples_path = run_dir / "hcup_samples.npz"
    samples = np.load(samples_path)
    probe_score = samples["probe_score"].astype(np.float64)
    probe_valid = samples["probe_valid"].astype(np.bool_)
    mse_on = samples["current_mse_on"].astype(np.float64)
    mse_off = samples["current_mse_off"].astype(np.float64)
    mae_on = samples["current_mae_on"].astype(np.float64)
    mae_off = samples["current_mae_off"].astype(np.float64)
    if metrics.get("source_split") != "val" or metrics.get(
        "probe_uses_current_or_future_target"
    ) is not False:
        raise RuntimeError("HCUP model selection requires a target-causal validation artifact")
    if probe_score.size != mse_on.size or not np.array_equal(
        samples["current_index"], np.arange(probe_score.size)
    ):
        raise RuntimeError("HCUP samples are not a complete chronological sequence")

    candidates = {
        "direct_last_probe": probe_score.copy(),
        "ewma_alpha_0.50": causal_ewma(probe_score, 0.50),
        "ewma_alpha_0.25": causal_ewma(probe_score, 0.25),
        "ewma_alpha_0.10": causal_ewma(probe_score, 0.10),
        "rolling_mean_32": causal_rolling(probe_score, 32, "mean"),
        "rolling_mean_96": causal_rolling(probe_score, 96, "mean"),
        "rolling_median_32": causal_rolling(probe_score, 32, "median"),
    }
    common_valid = np.flatnonzero(
        probe_valid & np.logical_and.reduce([np.isfinite(score) for score in candidates.values()])
    )
    if common_valid.size < 500:
        raise RuntimeError(f"HCUP validation cache has too few valid probes: {common_valid.size}")
    calibration_count = int(common_valid.size * 0.60)
    embargo = 96
    audit_start = calibration_count + embargo
    if audit_start >= common_valid.size - 100:
        raise RuntimeError("HCUP calibration/embargo split leaves too little audit data")
    calibration_index = common_valid[:calibration_count]
    audit_index = common_valid[audit_start:]

    records = []
    for offset, (name, score) in enumerate(candidates.items()):
        calibration = calibrate_threshold(
            score[calibration_index], mse_on[calibration_index], mse_off[calibration_index]
        )
        threshold = float(calibration["threshold"])
        calibration_metrics = evaluate(
            score,
            threshold,
            calibration_index,
            mse_on,
            mse_off,
            mae_on,
            mae_off,
        )
        audit_metrics = evaluate(
            score,
            threshold,
            audit_index,
            mse_on,
            mse_off,
            mae_on,
            mae_off,
        )
        interval = bootstrap_gain_interval(
            score[audit_index],
            threshold,
            mse_on[audit_index],
            mse_off[audit_index],
            args.seed + offset,
        )
        passes = bool(
            audit_metrics["mse_relative_gain"] > 0.0
            and audit_metrics["mae_relative_gain"] > 0.0
            and 0.0 < audit_metrics["gate_rate"] < 1.0
            and interval["absolute_gain_ci95_low"] > 0.0
        )
        records.append(
            {
                "name": name,
                "causal": True,
                "threshold_source": "early_validation_calibration_only",
                "threshold": threshold,
                "calibration": calibration_metrics,
                "audit": audit_metrics,
                "audit_mse_bootstrap": interval,
                "passes_offline_gate": passes,
            }
        )
    passing = sorted(
        (record for record in records if record["passes_offline_gate"]),
        key=lambda record: record["audit"]["mse_relative_gain"],
        reverse=True,
    )
    if passing:
        selected = passing[0]
        decision = {
            "code": "PROMISING_LOCK_AND_RUN_TEST_ONCE",
            "selected_aggregation": selected["name"],
            "locked_threshold": selected["threshold"],
            "summary_zh": (
                "历史反事实效用在严格后段验证审计上同时改善MSE和MAE，"
                "且MSE bootstrap 95%区间下界为正；可锁定聚合规则与阈值后只运行一次测试。"
            ),
        }
    else:
        best = max(records, key=lambda record: record["audit"]["mse_relative_gain"])
        decision = {
            "code": "HCUP_VALIDATION_NEGATIVE_DO_NOT_TOUCH_TEST",
            "selected_aggregation": None,
            "locked_threshold": None,
            "best_exploratory_aggregation": best["name"],
            "summary_zh": (
                "没有因果历史探针聚合在后段验证审计同时满足MSE、MAE和置信区间门禁；"
                "不应使用测试标签继续调节。"
            ),
        }
    report = {
        "schema_version": 1,
        "step": 8,
        "run_dir": str(run_dir),
        "samples_path": str(samples_path),
        "source_split": "val",
        "probe_uses_observed_history_only": True,
        "test_used_for_selection": False,
        "protocol": {
            "valid_probe_count": int(common_valid.size),
            "calibration_index_range": [
                int(calibration_index[0]),
                int(calibration_index[-1] + 1),
            ],
            "embargo_windows": embargo,
            "audit_index_range": [int(audit_index[0]), int(audit_index[-1] + 1)],
            "candidate_count": len(records),
            "claim_boundary": "single-seed validation screen; test untouched by this selector",
        },
        "raw_probe": {
            "valid_rate": float(np.mean(probe_valid)),
            "score_mean": float(np.mean(probe_score[probe_valid])),
            "score_std": float(np.std(probe_score[probe_valid])),
            "current_utility_pearson": metrics.get("probe_current_utility_pearson"),
        },
        "candidates": records,
        "decision": decision,
    }
    atomic_json(args.report_json, report)

    lines = [
        "# 步骤8：HCUP验证集筛选",
        "",
        f"- run：`{run_dir}`",
        "- 只使用validation；当前/测试未来标签不进入历史探针。",
        f"- 有效历史探针：{common_valid.size}；候选：{len(records)}。",
        "",
        "| 聚合 | calibration门控率 | audit门控率 | audit MSE增益 | audit MAE增益 | 95% CI下界 | 门禁 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['name']} | {record['calibration']['gate_rate']:.4f} | "
            f"{record['audit']['gate_rate']:.4f} | "
            f"{100.0 * record['audit']['mse_relative_gain']:.4f}% | "
            f"{100.0 * record['audit']['mae_relative_gain']:.4f}% | "
            f"{record['audit_mse_bootstrap']['absolute_gain_ci95_low']:.6g} | "
            f"{'PASS' if record['passes_offline_gate'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"- 决策：`{decision['code']}`",
            f"- 锁定聚合：`{decision.get('selected_aggregation') or '无'}`",
            f"- 说明：{decision['summary_zh']}",
            "",
        ]
    )
    atomic_text(args.report_md, "\n".join(lines))
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    print(f"REPORT_JSON={args.report_json}")
    print(f"REPORT_MD={args.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
