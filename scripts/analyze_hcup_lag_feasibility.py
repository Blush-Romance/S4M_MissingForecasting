#!/usr/bin/env python3
"""Upper-bound audit for shorter causal HCUP feedback lags.

This diagnostic uses complete *historical* validation utility shifted forward
by a lag as an optimistic proxy.  It never reads test data.  Because complete
values at synthetically missing historical positions are not deployable input,
the result can reject a lag family but cannot by itself validate HCUP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_fusr_v3_temporal_generalization import (
    atomic_text,
    binary_auc,
    bootstrap_gain_interval,
    calibrate_threshold,
    safe_corr,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = (
    ROOT
    / "runs"
    / "s4m_fusr_v3_hcup__weather__validation_point_r006__seed2023__20260718_202756"
    / "hcup_samples.npz"
)
DEFAULT_JSON = ROOT / "reports" / "stage8_hcup_lag_feasibility.json"
DEFAULT_MD = ROOT / "reports" / "stage8_hcup_lag_feasibility.md"
LAGS = (1, 4, 8, 16, 32, 48, 64, 96)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def evaluate(
    score: np.ndarray,
    threshold: float,
    index: np.ndarray,
    mse_on: np.ndarray,
    mse_off: np.ndarray,
    mae_on: np.ndarray,
    mae_off: np.ndarray,
) -> dict[str, Any]:
    gate = score[index] >= threshold
    routed_mse = np.where(gate, mse_on[index], mse_off[index])
    routed_mae = np.where(gate, mae_on[index], mae_off[index])
    on_mse = float(np.mean(mse_on[index]))
    on_mae = float(np.mean(mae_on[index]))
    beneficial = mse_off[index] > mse_on[index]
    return {
        "sample_count": int(index.size),
        "auroc": binary_auc(beneficial, score[index]),
        "gate_rate": float(np.mean(gate)),
        "mse_relative_gain": float((on_mse - np.mean(routed_mse)) / on_mse),
        "mae_relative_gain": float((on_mae - np.mean(routed_mae)) / on_mae),
        "routed_mse": float(np.mean(routed_mse)),
        "routed_mae": float(np.mean(routed_mae)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()
    if not args.samples.is_file():
        raise FileNotFoundError(args.samples)
    data = np.load(args.samples)
    mse_on = data["current_mse_on"].astype(np.float64)
    mse_off = data["current_mse_off"].astype(np.float64)
    mae_on = data["current_mae_on"].astype(np.float64)
    mae_off = data["current_mae_off"].astype(np.float64)
    probe_score = data["probe_score"].astype(np.float64)
    sample_count = int(mse_on.size)
    denominator = np.maximum(0.5 * (mse_on + mse_off), 1e-8)
    utility = np.clip((mse_off - mse_on) / denominator, -2.0, 2.0)

    # Use the same chronological samples for every lag so gains are directly
    # comparable.  The 96-window embargo is retained from the HCUP screen.
    common = np.arange(max(LAGS), sample_count, dtype=np.int64)
    calibration_count = int(common.size * 0.60)
    embargo = 96
    calibration_index = common[:calibration_count]
    audit_index = common[calibration_count + embargo :]
    if audit_index.size < 500:
        raise RuntimeError("lag feasibility split leaves too few audit samples")

    records = []
    for offset, lag in enumerate(LAGS):
        score = np.full(sample_count, np.nan, dtype=np.float64)
        score[lag:] = utility[:-lag]
        calibration = calibrate_threshold(
            score[calibration_index],
            mse_on[calibration_index],
            mse_off[calibration_index],
        )
        threshold = float(calibration["threshold"])
        audit = evaluate(
            score,
            threshold,
            audit_index,
            mse_on,
            mse_off,
            mae_on,
            mae_off,
        )
        zero = evaluate(
            score,
            0.0,
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
        record = {
            "lag": lag,
            "proxy": "shifted_complete_historical_full_horizon_utility",
            "utility_autocorrelation_audit": safe_corr(
                utility[audit_index - lag], utility[audit_index]
            ),
            "threshold": threshold,
            "calibration_gate_rate": float(calibration["gate_rate"]),
            "calibration_mse_relative_gain": float(
                calibration["relative_gain_vs_always_on"]
            ),
            "audit": audit,
            "audit_threshold_zero": zero,
            "audit_mse_bootstrap": interval,
        }
        record["passes_optimistic_upper_bound"] = bool(
            audit["mse_relative_gain"] > 0.0
            and audit["mae_relative_gain"] > 0.0
            and 0.0 < audit["gate_rate"] < 1.0
            and interval["absolute_gain_ci95_low"] > 0.0
        )
        records.append(record)

    lag96_complete = np.full(sample_count, np.nan, dtype=np.float64)
    lag96_complete[96:] = utility[:-96]
    valid_observed = np.isfinite(probe_score) & np.isfinite(lag96_complete)
    observed_vs_complete = {
        "sample_count": int(valid_observed.sum()),
        "pearson": safe_corr(probe_score[valid_observed], lag96_complete[valid_observed]),
        "sign_agreement": float(
            np.mean(
                (probe_score[valid_observed] >= 0.0)
                == (lag96_complete[valid_observed] >= 0.0)
            )
        ),
    }
    passing = [record for record in records if record["passes_optimistic_upper_bound"]]
    passing.sort(key=lambda item: item["audit"]["mse_relative_gain"], reverse=True)
    if passing:
        selected = passing[0]
        decision = {
            "code": "SHORT_LAG_HAS_OPTIMISTIC_SIGNAL_BUILD_OBSERVED_SMOKE",
            "selected_lag": int(selected["lag"]),
            "summary_zh": (
                "完整历史效用上界在短滞后后段审计同时改善MSE和MAE，且置信区间下界为正；"
                "可实现相同lag、只用实际观测历史前h步误差的HCUP smoke。"
            ),
        }
    else:
        best = max(records, key=lambda item: item["audit"]["mse_relative_gain"])
        decision = {
            "code": "NO_SHORT_LAG_UPPER_BOUND_STOP_HCUP_BRANCH",
            "selected_lag": None,
            "best_lag": int(best["lag"]),
            "summary_zh": (
                "即使使用不可部署的完整历史效用上界，短滞后也不能稳定改善后段MSE/MAE；"
                "实际观测历史探针不应继续占用GPU。"
            ),
        }
    report = {
        "schema_version": 1,
        "step": 8,
        "samples": str(args.samples.resolve()),
        "test_used": False,
        "claim_boundary": (
            "complete historical utility is an optimistic validation-only rejection proxy, "
            "not a deployable input or positive HCUP result"
        ),
        "protocol": {
            "lags": list(LAGS),
            "calibration_range": [
                int(calibration_index[0]),
                int(calibration_index[-1] + 1),
            ],
            "embargo_windows": embargo,
            "audit_range": [int(audit_index[0]), int(audit_index[-1] + 1)],
        },
        "observed_probe_vs_complete_lag96": observed_vs_complete,
        "records": records,
        "decision": decision,
    }
    atomic_json(args.report_json, report)
    lines = [
        "# 步骤8：HCUP短滞后可行性上界",
        "",
        "- 只使用validation；未读取test。",
        "- score使用完整历史utility，仅作为乐观上界，不能作为可部署结果。",
        "",
        "| lag | utility自相关 | audit AUROC | audit门控率 | audit MSE增益 | audit MAE增益 | 95% CI下界 | 上界门禁 |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        audit = record["audit"]
        corr = record["utility_autocorrelation_audit"]
        lines.append(
            f"| {record['lag']} | {corr if corr is not None else 'NA'} | "
            f"{audit['auroc'] if audit['auroc'] is not None else 'NA'} | "
            f"{audit['gate_rate']:.4f} | {100.0 * audit['mse_relative_gain']:.4f}% | "
            f"{100.0 * audit['mae_relative_gain']:.4f}% | "
            f"{record['audit_mse_bootstrap']['absolute_gain_ci95_low']:.6g} | "
            f"{'PASS' if record['passes_optimistic_upper_bound'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"- lag96观测探针与完整历史效用相关：{observed_vs_complete['pearson']}",
            f"- 决策：`{decision['code']}`",
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
