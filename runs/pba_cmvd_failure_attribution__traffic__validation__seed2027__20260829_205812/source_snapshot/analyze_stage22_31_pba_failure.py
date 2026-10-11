#!/usr/bin/env python3
"""Analyze Step22.31 paired PBA/BF compact validation caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/stage22_31_pba_diagnostic/state.json"
REPORT_JSON = ROOT / "reports/stage22_31_pba_failure.json"
REPORT_MD = ROOT / "reports/stage22_31_pba_failure.md"
FORMAL_JOB = "pba_cmvd_failure_attribution_seed2027_formal"


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(denominator) or denominator <= 0.0:
        return None
    return float(numerator / denominator)


def correction_summary(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    if not bool(mask.any()):
        return {
            "token_count": 0, "cosine": None, "projection_coefficient": None,
            "normalized_mismatch": None, "student_energy": 0.0,
            "teacher_energy": 0.0,
        }
    dot = float(arrays["correction_dot"][mask].sum(dtype=np.float64))
    student_energy = float(
        arrays["student_correction_energy"][mask].sum(dtype=np.float64)
    )
    teacher_energy = float(
        arrays["teacher_correction_energy"][mask].sum(dtype=np.float64)
    )
    mismatch = float(
        arrays["correction_mismatch_energy"][mask].sum(dtype=np.float64)
    )
    return {
        "token_count": int(mask.sum()),
        "cosine": safe_ratio(dot, np.sqrt(student_energy * teacher_energy)),
        "projection_coefficient": safe_ratio(dot, teacher_energy),
        "normalized_mismatch": safe_ratio(mismatch, teacher_energy),
        "student_energy": student_energy,
        "teacher_energy": teacher_energy,
        "mismatch_energy": mismatch,
    }


def gap_summary(
    arrays: dict[str, np.ndarray], benefit: np.ndarray, suffix: str
) -> dict[str, Any]:
    backbone = arrays[f"backbone_{suffix}"].astype(np.float64)
    teacher = arrays[f"teacher_{suffix}"].astype(np.float64)
    student = arrays[f"student_{suffix}"].astype(np.float64)
    positive_gap = np.maximum(student - teacher, 0.0)
    total = float(positive_gap.sum())
    inside = float(positive_gap[benefit].sum())
    outside = float(positive_gap[~benefit].sum())

    def region(mask: np.ndarray) -> dict[str, Any]:
        return {
            "token_count": int(mask.sum()),
            "student_minus_backbone_mean": float((student - backbone)[mask].mean()),
            "teacher_minus_backbone_mean": float((teacher - backbone)[mask].mean()),
            "student_minus_teacher_mean": float((student - teacher)[mask].mean()),
            "student_positive_harm_vs_backbone": float(
                np.maximum(student - backbone, 0.0)[mask].sum()
            ),
        }

    return {
        "positive_teacher_student_gap_total": total,
        "beneficial_positive_gap": inside,
        "nonbeneficial_positive_gap": outside,
        "beneficial_positive_gap_share": safe_ratio(inside, total),
        "nonbeneficial_positive_gap_share": safe_ratio(outside, total),
        "beneficial_region": region(benefit),
        "nonbeneficial_region": region(~benefit),
    }


def tail_summary(
    pba: dict[str, np.ndarray], bf: dict[str, np.ndarray], quantile: float = 0.90
) -> dict[str, Any]:
    bf_mse = bf["student_mse"].astype(np.float64)
    bf_mae = bf["student_mae"].astype(np.float64)
    pba_mse = pba["student_mse"].astype(np.float64)
    pba_mae = pba["student_mae"].astype(np.float64)
    threshold = float(np.quantile(bf_mse, quantile))
    tail = bf_mse >= threshold
    body = ~tail
    positive_mse_excess = np.maximum(pba_mse - bf_mse, 0.0)
    total_positive = float(positive_mse_excess.sum())
    tail_positive = float(positive_mse_excess[tail].sum())
    source_error_energy = pba["backbone_mse"].astype(np.float64) * 862.0
    correction_energy = pba["student_correction_energy"].astype(np.float64)
    correction_dot = pba["source_error_correction_dot"].astype(np.float64)
    tail_alignment = safe_ratio(
        float(correction_dot[tail].sum()),
        float(np.sqrt(
            source_error_energy[tail].sum() * correction_energy[tail].sum()
        )),
    )

    def quantiles(values: np.ndarray) -> dict[str, float]:
        return {
            "q90": float(np.quantile(values, 0.90)),
            "q95": float(np.quantile(values, 0.95)),
            "q99": float(np.quantile(values, 0.99)),
        }

    return {
        "reference": "BF-CMVD-v2 token MSE",
        "quantile": float(quantile),
        "threshold": threshold,
        "tail_token_count": int(tail.sum()),
        "tail_fraction": float(tail.mean()),
        "aggregate_pba_minus_bf_mse": float((pba_mse - bf_mse).mean()),
        "aggregate_pba_minus_bf_mae": float((pba_mae - bf_mae).mean()),
        "tail_positive_mse_excess_share": safe_ratio(tail_positive, total_positive),
        "tail_pba_minus_bf_mse_mean": float((pba_mse - bf_mse)[tail].mean()),
        "tail_pba_minus_bf_mae_mean": float((pba_mae - bf_mae)[tail].mean()),
        "body_pba_minus_bf_mse_mean": float((pba_mse - bf_mse)[body].mean()),
        "body_pba_minus_bf_mae_mean": float((pba_mae - bf_mae)[body].mean()),
        "bf_tail_mse_mean": float(bf_mse[tail].mean()),
        "pba_tail_mse_mean": float(pba_mse[tail].mean()),
        "bf_token_mse_quantiles": quantiles(bf_mse),
        "pba_token_mse_quantiles": quantiles(pba_mse),
        "tail_source_error_correction_cosine": tail_alignment,
        "tail_harmful_first_order_fraction": float((correction_dot[tail] > 0).mean()),
    }


def decide_diagnosis(
    tail: dict[str, Any], gap_mse: dict[str, Any], gap_mae: dict[str, Any],
    geometry: dict[str, Any], nonbenefit_energy_share: float | None,
    protocol_pass: bool,
) -> dict[str, Any]:
    if not protocol_pass:
        return {
            "decision": "PROTOCOL_FAILURE_NO_DECISION",
            "main_cause": None,
            "recommended_next": None,
            "preregistered_triggers": {},
        }

    tail_trigger = bool(
        tail["aggregate_pba_minus_bf_mse"] > 0.0
        and tail["aggregate_pba_minus_bf_mae"] <= 0.0
        and tail["tail_positive_mse_excess_share"] is not None
        and tail["tail_positive_mse_excess_share"] >= 0.60
        and tail["tail_pba_minus_bf_mse_mean"] > 0.0
        and tail["body_pba_minus_bf_mae_mean"] <= 0.0
    )
    nonbenefit_trigger = bool(
        gap_mse["nonbeneficial_positive_gap_share"] is not None
        and gap_mae["nonbeneficial_positive_gap_share"] is not None
        and gap_mse["nonbeneficial_positive_gap_share"] >= 0.60
        and gap_mae["nonbeneficial_positive_gap_share"] >= 0.60
        and gap_mse["nonbeneficial_region"]["student_minus_backbone_mean"] > 0.0
        and gap_mae["nonbeneficial_region"]["student_minus_backbone_mean"] > 0.0
        and nonbenefit_energy_share is not None
        and nonbenefit_energy_share >= 0.25
    )
    low_geometry_count = sum((
        geometry["cosine"] is None or geometry["cosine"] < 0.50,
        geometry["projection_coefficient"] is None
        or geometry["projection_coefficient"] < 0.50,
        geometry["normalized_mismatch"] is None
        or geometry["normalized_mismatch"] > 0.75,
    ))
    direction_trigger = bool(
        gap_mse["beneficial_positive_gap_share"] is not None
        and gap_mae["beneficial_positive_gap_share"] is not None
        and gap_mse["beneficial_positive_gap_share"] >= 0.60
        and gap_mae["beneficial_positive_gap_share"] >= 0.60
        and low_geometry_count >= 2
    )
    triggers = {
        "tail_risk_amplification": tail_trigger,
        "nonbeneficial_drift": nonbenefit_trigger,
        "teacher_direction_under_amortized": direction_trigger,
    }
    if tail_trigger:
        decision = "TAIL_RISK_AMPLIFICATION_FREEZE_CVAR_TR_PBA"
        cause = "tail_risk_amplification"
        candidate = (
            "CVaR-TR-PBA（条件风险价值信赖域部分主干摊销）"
        )
    elif nonbenefit_trigger:
        decision = "NONBENEFICIAL_DRIFT_FREEZE_BTR_PBA"
        cause = "nonbeneficial_drift"
        candidate = "BTR-PBA（受益区域信赖域部分主干摊销）"
    elif direction_trigger:
        decision = "TEACHER_DIRECTION_UNDER_AMORTIZED_FREEZE_HFD_PBA"
        cause = "teacher_direction_under_amortized"
        candidate = "HFD-PBA（层级特征蒸馏部分主干摊销）"
    else:
        decision = "MIXED_OR_INDETERMINATE_STOP_PBA_ROUTE"
        cause = None
        candidate = None
    return {
        "decision": decision,
        "main_cause": cause,
        "recommended_next": candidate,
        "preregistered_triggers": triggers,
        "thresholds": {
            "bf_tail_quantile": 0.90,
            "tail_positive_mse_excess_share_gte": 0.60,
            "regional_positive_gap_share_gte": 0.60,
            "nonbenefit_energy_share_gte": 0.25,
            "beneficial_cosine_lt": 0.50,
            "beneficial_projection_lt": 0.50,
            "beneficial_normalized_mismatch_gt": 0.75,
            "low_geometry_conditions_required": 2,
        },
    }


def analyze(
    metrics: dict[str, Any], pba: dict[str, np.ndarray],
    bf_metrics: dict[str, Any], bf: dict[str, np.ndarray],
) -> dict[str, Any]:
    protocol_pass = bool(
        metrics.get("artifact_status") == "PASS"
        and all(metrics.get("artifact_checks", {}).values())
        and metrics.get("test_split_accessed") is False
        and metrics.get("updates_model_or_memory") is False
        and bf_metrics.get("artifact_status") == "PASS"
        and bf_metrics.get("test_split_accessed") is False
        and np.array_equal(pba["sample_index"], bf["sample_index"])
        and all(np.array_equal(pba[key], bf[key]) for key in (
            "history_mask_packed", "l5_mask_packed", "l20_mask_packed"
        ))
    )
    benefit = pba["complete_teacher_beneficial"].astype(bool)
    geometry_all = correction_summary(
        pba, np.ones_like(benefit, dtype=bool)
    )
    geometry_benefit = correction_summary(pba, benefit)
    geometry_nonbenefit = correction_summary(pba, ~benefit)
    nonbenefit_energy_share = safe_ratio(
        geometry_nonbenefit["student_energy"], geometry_all["student_energy"]
    )
    gap_mse = gap_summary(pba, benefit, "mse")
    gap_mae = gap_summary(pba, benefit, "mae")
    tail = tail_summary(pba, bf)
    decision = decide_diagnosis(
        tail, gap_mse, gap_mae, geometry_benefit,
        nonbenefit_energy_share, protocol_pass,
    )

    horizons = []
    for index in range(pba["student_mse"].shape[1]):
        horizons.append({
            "horizon": index + 1,
            "pg95_mse": float(pba["backbone_mse"][:, index].mean()),
            "teacher_mse": float(pba["teacher_mse"][:, index].mean()),
            "bf_mse": float(bf["student_mse"][:, index].mean()),
            "pba_mse": float(pba["student_mse"][:, index].mean()),
            "pg95_mae": float(pba["backbone_mae"][:, index].mean()),
            "teacher_mae": float(pba["teacher_mae"][:, index].mean()),
            "bf_mae": float(bf["student_mae"][:, index].mean()),
            "pba_mae": float(pba["student_mae"][:, index].mean()),
        })

    report = {
        "protocol": {
            "status": "PASS" if protocol_pass else "FAIL",
            "validation_only": metrics.get("test_split_accessed") is False,
            "sample_count": int(pba["sample_index"].size),
            "horizon": int(pba["student_mse"].shape[1]),
            "variable_count": 862,
            "pba_run": metrics.get("experiment_id"),
            "bf_cache_run": bf_metrics.get("experiment_id"),
        },
        "metrics_roundtrip": {
            "pg95": {
                "mse": float(pba["backbone_mse"].mean()),
                "mae": float(pba["backbone_mae"].mean()),
            },
            "teacher": {
                "mse": float(pba["teacher_mse"].mean()),
                "mae": float(pba["teacher_mae"].mean()),
            },
            "bf_cmvd_v2": {
                "mse": float(bf["student_mse"].mean()),
                "mae": float(bf["student_mae"].mean()),
            },
            "pba_cmvd": {
                "mse": float(pba["student_mse"].mean()),
                "mae": float(pba["student_mae"].mean()),
            },
        },
        "tail_risk": tail,
        "teacher_advantage_gap": {"mse": gap_mse, "mae": gap_mae},
        "correction_geometry": {
            "all": geometry_all,
            "teacher_beneficial": geometry_benefit,
            "teacher_nonbeneficial": geometry_nonbenefit,
            "nonbeneficial_student_energy_share": nonbenefit_energy_share,
        },
        "horizon_diagnostics": horizons,
        **decision,
        "scientific_boundary": (
            "validation-only frozen failure attribution; no new training, no new seed, "
            "no test access"
        ),
    }
    return report


def write_report(report: dict[str, Any]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_JSON.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(REPORT_JSON)
    tail = report["tail_risk"]
    gaps = report["teacher_advantage_gap"]
    geometry = report["correction_geometry"]
    lines = [
        "# Step 22.31 PBA-CMVD 失败归因", "",
        f"- 协议：`{report['protocol']['status']}`；样本：`{report['protocol']['sample_count']}`；test未访问：`{report['protocol']['validation_only']}`",
        f"- 决策：`{report['decision']}`",
        f"- 主因：`{report.get('main_cause')}`",
        f"- 唯一下一候选：`{report.get('recommended_next')}`", "",
        "## 高误差尾部", "",
        f"- BF参考Q90阈值/尾部占比：`{tail['threshold']}` / `{tail['tail_fraction']}`",
        f"- PBA-BF聚合MSE/MAE差：`{tail['aggregate_pba_minus_bf_mse']}` / `{tail['aggregate_pba_minus_bf_mae']}`",
        f"- 尾部正MSE超额份额：`{tail['tail_positive_mse_excess_share']}`",
        f"- 尾部/主体MSE差：`{tail['tail_pba_minus_bf_mse_mean']}` / `{tail['body_pba_minus_bf_mse_mean']}`", "",
        "## 区域与方向", "",
        f"- MSE教师正差距获益区/非获益区份额：`{gaps['mse']['beneficial_positive_gap_share']}` / `{gaps['mse']['nonbeneficial_positive_gap_share']}`",
        f"- MAE教师正差距获益区/非获益区份额：`{gaps['mae']['beneficial_positive_gap_share']}` / `{gaps['mae']['nonbeneficial_positive_gap_share']}`",
        f"- 获益区cosine/projection/mismatch：`{geometry['teacher_beneficial']['cosine']}` / `{geometry['teacher_beneficial']['projection_coefficient']}` / `{geometry['teacher_beneficial']['normalized_mismatch']}`",
        f"- 非获益区PBA修正能量占比：`{geometry['nonbeneficial_student_energy_share']}`", "",
        "本报告只冻结一个下一方向或停止；没有训练新模型、扩seed或访问test。",
    ]
    temporary_md = REPORT_MD.with_suffix(".md.tmp")
    temporary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary_md.replace(REPORT_MD)


def main() -> int:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    job = state["jobs"][FORMAL_JOB]
    if job.get("state") != "SUCCESS" or not job.get("run_dir"):
        raise RuntimeError("Step22.31 formal diagnostic is not successful")
    run_dir = Path(job["run_dir"])
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    with np.load(run_dir / "pba_failure_tokens.npz", allow_pickle=False) as archive:
        pba = {key: archive[key] for key in archive.files}
    bf_metrics_path = Path(metrics["bf_diagnostic_metrics"])
    bf_samples_path = Path(metrics["bf_diagnostic_samples"])
    bf_metrics = json.loads(bf_metrics_path.read_text(encoding="utf-8"))
    with np.load(bf_samples_path, allow_pickle=False) as archive:
        bf = {key: archive[key] for key in archive.files}
    report = analyze(metrics, pba, bf_metrics, bf)
    write_report(report)
    print(json.dumps({
        "decision": report["decision"],
        "main_cause": report["main_cause"],
        "recommended_next": report["recommended_next"],
        "report_json": str(REPORT_JSON),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
