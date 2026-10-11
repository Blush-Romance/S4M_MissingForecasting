#!/usr/bin/env python3
"""Compare predeclared short-horizon HCUP lags on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_fusr_v3_temporal_generalization import (
    atomic_text,
    bootstrap_gain_interval,
    calibrate_threshold,
)
from analyze_hcup_validation import causal_ewma, causal_rolling, evaluate


ROOT = Path(__file__).resolve().parents[1]
QUEUE_STATE = ROOT / "runtime" / "stage8_hcup_short_queue" / "state.json"
REPORT_JSON = ROOT / "reports" / "stage8_hcup_short_validation.json"
REPORT_MD = ROOT / "reports" / "stage8_hcup_short_validation.md"
LAGS = (1, 4, 8, 16)
CURRENT_OUTPUT_ATOL = 1e-5
CURRENT_OUTPUT_RTOL = 1e-6


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def formal_run_dirs(queue_state: Path) -> dict[int, Path]:
    state = json.loads(queue_state.read_text(encoding="utf-8"))
    runs: dict[int, Path] = {}
    for lag in LAGS:
        job_id = f"weather_hcup_h{lag}_validation_formal"
        job = state.get("jobs", {}).get(job_id, {})
        if job.get("state") != "SUCCESS" or not job.get("run_dir"):
            raise RuntimeError(f"{job_id} is not SUCCESS in {queue_state}")
        run_dir = Path(job["run_dir"])
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        runs[lag] = run_dir
    return runs


def score_candidates(probe_score: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "direct_last_probe": probe_score.copy(),
        "ewma_alpha_0.50": causal_ewma(probe_score, 0.50),
        "ewma_alpha_0.25": causal_ewma(probe_score, 0.25),
        "ewma_alpha_0.10": causal_ewma(probe_score, 0.10),
        "rolling_mean_32": causal_rolling(probe_score, 32, "mean"),
        "rolling_mean_96": causal_rolling(probe_score, 96, "mean"),
        "rolling_median_32": causal_rolling(probe_score, 32, "median"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-state", type=Path, default=QUEUE_STATE)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()
    runs = formal_run_dirs(args.queue_state)

    records = []
    reference_arrays: dict[str, np.ndarray] | None = None
    cross_run_max_errors: dict[str, float] = {}
    cross_run_allclose: dict[str, bool] = {}
    for lag in LAGS:
        run_dir = runs[lag]
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if (
            metrics.get("artifact_status") != "PASS"
            or metrics.get("source_split") != "val"
            or metrics.get("probe_uses_current_or_future_target") is not False
            or int(metrics.get("probe_lag", -1)) != lag
            or int(metrics.get("probe_horizon", -1)) != lag
        ):
            raise RuntimeError(f"invalid short HCUP artifact: lag={lag}, run={run_dir}")
        samples = np.load(run_dir / "hcup_samples.npz")
        arrays = {
            key: samples[key].astype(np.float64)
            for key in (
                "current_mse_on",
                "current_mse_off",
                "current_mae_on",
                "current_mae_off",
            )
        }
        if reference_arrays is None:
            reference_arrays = arrays
        else:
            for key, values in arrays.items():
                reference = reference_arrays[key]
                if not np.isfinite(values).all() or not np.isfinite(reference).all():
                    raise RuntimeError(
                        f"non-finite current intervention output: {key}, lag={lag}"
                    )
                error = float(np.max(np.abs(values - reference)))
                comparison_key = f"lag{lag}_{key}"
                cross_run_max_errors[comparison_key] = error
                is_close = bool(np.allclose(
                    values,
                    reference,
                    rtol=CURRENT_OUTPUT_RTOL,
                    atol=CURRENT_OUTPUT_ATOL,
                ))
                cross_run_allclose[comparison_key] = is_close
                if not is_close:
                    raise RuntimeError(
                        "current intervention outputs exceed numerical tolerance across lags: "
                        f"{key}, lag={lag}, max_abs={error}, "
                        f"atol={CURRENT_OUTPUT_ATOL}, rtol={CURRENT_OUTPUT_RTOL}"
                    )
        probe_score = samples["probe_score"].astype(np.float64)
        probe_valid = samples["probe_valid"].astype(np.bool_)
        candidates = score_candidates(probe_score)
        common_valid = np.flatnonzero(
            probe_valid
            & np.logical_and.reduce(
                [np.isfinite(score) for score in candidates.values()]
            )
        )
        if common_valid.size < 500:
            raise RuntimeError(f"lag{lag} has too few valid probes: {common_valid.size}")
        calibration_count = int(common_valid.size * 0.60)
        embargo = 96
        audit_start = calibration_count + embargo
        calibration_index = common_valid[:calibration_count]
        audit_index = common_valid[audit_start:]
        if audit_index.size < 500:
            raise RuntimeError(f"lag{lag} leaves too few audit samples")
        for candidate_offset, (aggregation, score) in enumerate(candidates.items()):
            calibration = calibrate_threshold(
                score[calibration_index],
                arrays["current_mse_on"][calibration_index],
                arrays["current_mse_off"][calibration_index],
            )
            threshold = float(calibration["threshold"])
            calibration_metrics = evaluate(
                score,
                threshold,
                calibration_index,
                arrays["current_mse_on"],
                arrays["current_mse_off"],
                arrays["current_mae_on"],
                arrays["current_mae_off"],
            )
            audit_metrics = evaluate(
                score,
                threshold,
                audit_index,
                arrays["current_mse_on"],
                arrays["current_mse_off"],
                arrays["current_mae_on"],
                arrays["current_mae_off"],
            )
            interval = bootstrap_gain_interval(
                score[audit_index],
                threshold,
                arrays["current_mse_on"][audit_index],
                arrays["current_mse_off"][audit_index],
                args.seed + 100 * lag + candidate_offset,
            )
            passes = bool(
                audit_metrics["mse_relative_gain"] > 0.0
                and audit_metrics["mae_relative_gain"] > 0.0
                and 0.0 < audit_metrics["gate_rate"] < 1.0
                and interval["absolute_gain_ci95_low"] > 0.0
            )
            records.append(
                {
                    "lag": lag,
                    "probe_horizon": lag,
                    "aggregation": aggregation,
                    "run_dir": str(run_dir),
                    "threshold": threshold,
                    "threshold_source": "early_validation_calibration_only",
                    "calibration": calibration_metrics,
                    "audit": audit_metrics,
                    "audit_mse_bootstrap": interval,
                    "passes_offline_gate": passes,
                    "valid_probe_count": int(common_valid.size),
                    "calibration_range": [
                        int(calibration_index[0]),
                        int(calibration_index[-1] + 1),
                    ],
                    "audit_range": [int(audit_index[0]), int(audit_index[-1] + 1)],
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
            "code": "SHORT_HCUP_PROMISING_LOCK_FOR_ONE_TEST",
            "selected_lag": int(selected["lag"]),
            "selected_probe_horizon": int(selected["probe_horizon"]),
            "selected_aggregation": selected["aggregation"],
            "locked_threshold": float(selected["threshold"]),
            "summary_zh": (
                "至少一个预声明短跨度HCUP在后段validation同时改善MSE和MAE，"
                "且MSE bootstrap 95%下界为正；可锁定规则后运行一次test。"
            ),
        }
    else:
        best = max(records, key=lambda record: record["audit"]["mse_relative_gain"])
        decision = {
            "code": "SHORT_HCUP_VALIDATION_NEGATIVE_STOP_FEEDBACK_BRANCH",
            "selected_lag": None,
            "selected_probe_horizon": None,
            "selected_aggregation": None,
            "locked_threshold": None,
            "best_exploratory": {
                "lag": int(best["lag"]),
                "aggregation": best["aggregation"],
            },
            "summary_zh": (
                "预声明的1/4/8/16步观测历史探针均未通过后段validation门禁；"
                "应停止历史反馈式FUSR，不能用test继续搜索。"
            ),
        }
    report = {
        "schema_version": 1,
        "step": 8,
        "source_split": "val",
        "test_used_for_selection": False,
        "lags": list(LAGS),
        "aggregation_count_per_lag": 7,
        "cross_run_current_output_max_errors": cross_run_max_errors,
        "cross_run_current_outputs_identical": all(
            value == 0.0 for value in cross_run_max_errors.values()
        ),
        "cross_run_current_outputs_numerically_consistent": all(
            cross_run_allclose.values()
        ),
        "cross_run_current_output_tolerance": {
            "atol": CURRENT_OUTPUT_ATOL,
            "rtol": CURRENT_OUTPUT_RTOL,
        },
        "claim_boundary": "single-seed validation screen; test remains untouched",
        "records": records,
        "decision": decision,
    }
    atomic_json(args.report_json, report)
    lines = [
        "# 步骤8：短跨度HCUP验证筛选",
        "",
        "- 候选lag固定为1/4/8/16；每项7种因果聚合。",
        "- 只使用validation；test未参与选择。",
        f"- 四个run的当前on/off输出逐位完全一致：{report['cross_run_current_outputs_identical']}。",
        f"- 四个run的当前on/off输出在数值容差内一致：{report['cross_run_current_outputs_numerically_consistent']}"
        f"（atol={CURRENT_OUTPUT_ATOL:g}, rtol={CURRENT_OUTPUT_RTOL:g}）。",
        "",
        "| lag | 聚合 | audit门控率 | audit AUROC | audit MSE增益 | audit MAE增益 | 95% CI下界 | 门禁 |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        audit = record["audit"]
        lines.append(
            f"| {record['lag']} | {record['aggregation']} | {audit['gate_rate']:.4f} | "
            f"{audit['auroc'] if audit['auroc'] is not None else 'NA'} | "
            f"{100.0 * audit['mse_relative_gain']:.4f}% | "
            f"{100.0 * audit['mae_relative_gain']:.4f}% | "
            f"{record['audit_mse_bootstrap']['absolute_gain_ci95_low']:.6g} | "
            f"{'PASS' if record['passes_offline_gate'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"- 决策：`{decision['code']}`",
            f"- 选择：lag=`{decision.get('selected_lag')}`, aggregation=`{decision.get('selected_aggregation')}`",
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
