#!/usr/bin/env python3
"""Analyze Step 22.27 compact validation cache using preregistered gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/stage22_27_bf_cmvd_diagnostic/state.json"
REPORT_JSON = ROOT / "reports/stage22_27_bf_cmvd_failure.json"
REPORT_MD = ROOT / "reports/stage22_27_bf_cmvd_failure.md"
FORMAL_JOB = "bf_cmvd_v2_failure_attribution_seed2027_formal"


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(denominator) or denominator <= 0.0:
        return None
    return float(numerator / denominator)


def correction_summary(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    if not bool(mask.any()):
        return {
            "token_count": 0, "cosine": None, "projection_coefficient": None,
            "normalized_mismatch": None, "sign_agreement": None,
            "student_energy": 0.0, "teacher_energy": 0.0,
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
    sign_agree = float(
        arrays["correction_sign_agreement_count"][mask].sum(dtype=np.float64)
    )
    sign_eligible = float(
        arrays["correction_sign_eligible_count"][mask].sum(dtype=np.float64)
    )
    return {
        "token_count": int(mask.sum()),
        "cosine": safe_ratio(dot, np.sqrt(student_energy * teacher_energy)),
        "projection_coefficient": safe_ratio(dot, teacher_energy),
        "normalized_mismatch": safe_ratio(mismatch, teacher_energy),
        "sign_agreement": safe_ratio(sign_agree, sign_eligible),
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
        if not bool(mask.any()):
            return {"token_count": 0}
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


def decide_diagnosis(
    coverage: dict[str, Any],
    gap_mse: dict[str, Any],
    gap_mae: dict[str, Any],
    correction_beneficial: dict[str, Any],
    correction_nonbeneficial: dict[str, Any],
    nonbeneficial_student_energy_share: float | None,
    protocol_pass: bool,
) -> dict[str, Any]:
    if not protocol_pass:
        return {
            "decision": "PROTOCOL_FAILURE_NO_DECISION",
            "main_cause": None,
            "recommended_v3": None,
        }
    coverage_recall = coverage["complete_benefit_observed_support_recall"]
    coverage_gap = coverage["benefit_rate_observed_unobserved_abs_gap"]
    coverage_trigger = bool(
        coverage_recall is not None and coverage_gap is not None
        and coverage_recall < 0.70 and coverage_gap > 0.05
    )
    expression_region = bool(
        gap_mse["beneficial_positive_gap_share"] is not None
        and gap_mae["beneficial_positive_gap_share"] is not None
        and gap_mse["beneficial_positive_gap_share"] >= 0.60
        and gap_mae["beneficial_positive_gap_share"] >= 0.60
    )
    geometry_underfit = bool(
        correction_beneficial["cosine"] is None
        or correction_beneficial["cosine"] < 0.50
        or correction_beneficial["projection_coefficient"] is None
        or correction_beneficial["projection_coefficient"] < 0.50
        or correction_beneficial["normalized_mismatch"] is None
        or correction_beneficial["normalized_mismatch"] > 0.75
    )
    expression_trigger = expression_region and geometry_underfit
    spillover_region = bool(
        gap_mse["nonbeneficial_positive_gap_share"] is not None
        and gap_mae["nonbeneficial_positive_gap_share"] is not None
        and gap_mse["nonbeneficial_positive_gap_share"] >= 0.60
        and gap_mae["nonbeneficial_positive_gap_share"] >= 0.60
    )
    spillover_harm = bool(
        gap_mse["nonbeneficial_region"]["student_minus_backbone_mean"] > 0.0
        and gap_mae["nonbeneficial_region"]["student_minus_backbone_mean"] > 0.0
    )
    spillover_trigger = bool(
        spillover_region and spillover_harm
        and nonbeneficial_student_energy_share is not None
        and nonbeneficial_student_energy_share >= 0.25
    )

    triggers = {
        "coverage_mismatch": coverage_trigger,
        "student_expression_underfit": expression_trigger,
        "nonbeneficial_spillover": spillover_trigger,
    }
    if coverage_trigger:
        decision = "SUPERVISION_COVERAGE_MISMATCH_FREEZE_OBSERVABILITY_BALANCED_BF_CMVD"
        cause = "supervision_coverage_mismatch"
        candidate = "Observability-Balanced BF-CMVD（可观测性均衡收益蒸馏）"
    elif expression_trigger and not spillover_trigger:
        decision = "STUDENT_EXPRESSION_UNDERFIT_FREEZE_DMRB"
        cause = "student_expression_underfit"
        candidate = "DMRB（双尺度掩码条件残差基）"
    elif spillover_trigger and not expression_trigger:
        decision = "NONBENEFICIAL_SPILLOVER_FREEZE_CBR"
        cause = "nonbeneficial_spillover"
        candidate = "CBR（因果收益路由器）"
    elif expression_trigger and spillover_trigger:
        expression_score = min(
            gap_mse["beneficial_positive_gap_share"],
            gap_mae["beneficial_positive_gap_share"],
        )
        spillover_score = min(
            gap_mse["nonbeneficial_positive_gap_share"],
            gap_mae["nonbeneficial_positive_gap_share"],
        )
        if abs(expression_score - spillover_score) < 0.10:
            decision = "MIXED_OR_INDETERMINATE_STOP"
            cause = None
            candidate = None
        elif expression_score > spillover_score:
            decision = "STUDENT_EXPRESSION_UNDERFIT_FREEZE_DMRB"
            cause = "student_expression_underfit"
            candidate = "DMRB（双尺度掩码条件残差基）"
        else:
            decision = "NONBENEFICIAL_SPILLOVER_FREEZE_CBR"
            cause = "nonbeneficial_spillover"
            candidate = "CBR（因果收益路由器）"
    else:
        decision = "MIXED_OR_INDETERMINATE_STOP"
        cause = None
        candidate = None
    return {
        "decision": decision,
        "main_cause": cause,
        "recommended_v3": candidate,
        "preregistered_triggers": triggers,
        "thresholds": {
            "coverage_recall_lt": 0.70,
            "coverage_benefit_rate_gap_gt": 0.05,
            "regional_positive_gap_share_gte": 0.60,
            "beneficial_cosine_lt": 0.50,
            "beneficial_projection_lt": 0.50,
            "beneficial_normalized_mismatch_gt": 0.75,
            "nonbeneficial_student_energy_share_gte": 0.25,
            "mixed_score_separation_gte": 0.10,
        },
    }


def analyze(metrics: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    observed = arrays["observed_token"].astype(bool)
    benefit = arrays["complete_teacher_beneficial"].astype(bool)
    observed_benefit = arrays["observed_teacher_beneficial"].astype(bool)
    intersection = benefit & observed_benefit
    precision = safe_ratio(float(intersection.sum()), float(observed_benefit.sum()))
    recall = safe_ratio(float(intersection.sum()), float(benefit.sum()))
    f1 = (
        safe_ratio(2.0 * precision * recall, precision + recall)
        if precision is not None and recall is not None else None
    )
    union = benefit | observed_benefit
    jaccard = safe_ratio(float(intersection.sum()), float(union.sum()))
    observed_rate = float(benefit[observed].mean())
    unobserved_rate = float(benefit[~observed].mean())
    coverage = {
        "observed_token_fraction": float(observed.mean()),
        "benefit_precision": precision,
        "complete_benefit_observed_support_recall": recall,
        "benefit_f1": f1,
        "benefit_jaccard": jaccard,
        "complete_benefit_rate_observed_tokens": observed_rate,
        "complete_benefit_rate_unobserved_tokens": unobserved_rate,
        "benefit_rate_observed_unobserved_abs_gap": abs(
            observed_rate - unobserved_rate
        ),
    }
    correction_all = correction_summary(arrays, np.ones_like(benefit, dtype=bool))
    correction_beneficial = correction_summary(arrays, benefit)
    correction_nonbeneficial = correction_summary(arrays, ~benefit)
    student_energy_total = correction_all["student_energy"]
    nonbeneficial_student_energy_share = safe_ratio(
        correction_nonbeneficial["student_energy"], student_energy_total
    )
    gap_mse = gap_summary(arrays, benefit, "mse")
    gap_mae = gap_summary(arrays, benefit, "mae")
    protocol_pass = bool(
        metrics.get("artifact_status") == "PASS"
        and all(metrics.get("artifact_checks", {}).values())
        and metrics.get("test_split_accessed") is False
        and metrics.get("updates_model_or_memory") is False
    )
    decision = decide_diagnosis(
        coverage, gap_mse, gap_mae, correction_beneficial,
        correction_nonbeneficial, nonbeneficial_student_energy_share,
        protocol_pass,
    )

    horizon = []
    for index in range(arrays["backbone_mse"].shape[1]):
        horizon_mask = np.zeros_like(benefit, dtype=bool)
        horizon_mask[:, index] = True
        horizon.append({
            "horizon": index + 1,
            "observed_fraction": float(observed[:, index].mean()),
            "teacher_beneficial_fraction": float(benefit[:, index].mean()),
            "student_beneficial_fraction": float(
                arrays["complete_student_beneficial"][:, index].mean()
            ),
            "backbone_mse": float(arrays["backbone_mse"][:, index].mean()),
            "teacher_mse": float(arrays["teacher_mse"][:, index].mean()),
            "student_mse": float(arrays["student_mse"][:, index].mean()),
            "backbone_mae": float(arrays["backbone_mae"][:, index].mean()),
            "teacher_mae": float(arrays["teacher_mae"][:, index].mean()),
            "student_mae": float(arrays["student_mae"][:, index].mean()),
            "correction": correction_summary(arrays, horizon_mask),
        })

    variable_count = arrays["variable_backbone_sse"].shape[0]
    element_count_per_variable = arrays["backbone_mse"].shape[0] * 96
    variable_delta_mse = (
        arrays["variable_student_sse"] - arrays["variable_backbone_sse"]
    ) / element_count_per_variable
    variable_mismatch = arrays["variable_correction_mismatch_energy"]
    top_harm = np.argsort(variable_delta_mse)[-10:][::-1]
    top_mismatch = np.argsort(variable_mismatch)[-10:][::-1]
    report = {
        "protocol": {
            "status": "PASS" if protocol_pass else "FAIL",
            "validation_only": metrics.get("test_split_accessed") is False,
            "sample_count": int(arrays["sample_index"].size),
            "horizon": int(arrays["backbone_mse"].shape[1]),
            "variable_count": int(variable_count),
            "source_run": metrics.get("experiment_id"),
            "samples_path": metrics.get("samples_path"),
        },
        "metrics_roundtrip": {
            "pg95_backbone": {
                "mse": float(arrays["backbone_mse"].mean()),
                "mae": float(arrays["backbone_mae"].mean()),
            },
            "cnm_se_teacher": {
                "mse": float(arrays["teacher_mse"].mean()),
                "mae": float(arrays["teacher_mae"].mean()),
            },
            "bf_cmvd_v2_student": {
                "mse": float(arrays["student_mse"].mean()),
                "mae": float(arrays["student_mae"].mean()),
            },
        },
        "coverage": coverage,
        "correction_geometry": {
            "all": correction_all,
            "teacher_beneficial": correction_beneficial,
            "teacher_nonbeneficial": correction_nonbeneficial,
            "nonbeneficial_student_energy_share": nonbeneficial_student_energy_share,
        },
        "teacher_advantage_gap": {"mse": gap_mse, "mae": gap_mae},
        "horizon_diagnostics": horizon,
        "variable_diagnostics": {
            "top10_student_harm_vs_backbone": [
                {"variable_index": int(i), "delta_mse": float(variable_delta_mse[i])}
                for i in top_harm
            ],
            "top10_correction_mismatch_energy": [
                {"variable_index": int(i), "mismatch_energy": float(variable_mismatch[i])}
                for i in top_mismatch
            ],
        },
        **decision,
        "scientific_boundary": (
            "validation-only frozen failure attribution; no v3 training, no new seed, "
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
    coverage = report["coverage"]
    geometry = report["correction_geometry"]
    gaps = report["teacher_advantage_gap"]
    lines = [
        "# Step 22.27 BF-CMVD-v2失效归因",
        "",
        f"- 协议：`{report['protocol']['status']}`；样本数：`{report['protocol']['sample_count']}`；test未访问：`{report['protocol']['validation_only']}`",
        f"- 决策：`{report['decision']}`",
        f"- 主因：`{report.get('main_cause')}`",
        f"- 唯一v3候选：`{report.get('recommended_v3')}`",
        "",
        "## 监督覆盖",
        "",
        f"- validation目标观测覆盖率：`{coverage['observed_token_fraction']:.6f}`",
        f"- 完整获益的观测支持recall：`{coverage['complete_benefit_observed_support_recall']:.6f}`",
        f"- precision/F1/Jaccard：`{coverage['benefit_precision']:.6f}` / `{coverage['benefit_f1']:.6f}` / `{coverage['benefit_jaccard']:.6f}`",
        f"- 已观测/未观测教师获益率差：`{coverage['benefit_rate_observed_unobserved_abs_gap']:.6f}`",
        "",
        "## 修正几何与差距分解",
        "",
        f"- 获益区cosine/projection/mismatch：`{geometry['teacher_beneficial']['cosine']}` / `{geometry['teacher_beneficial']['projection_coefficient']}` / `{geometry['teacher_beneficial']['normalized_mismatch']}`",
        f"- 非获益区学生残差能量占比：`{geometry['nonbeneficial_student_energy_share']}`",
        f"- MSE正差距获益区/非获益区份额：`{gaps['mse']['beneficial_positive_gap_share']}` / `{gaps['mse']['nonbeneficial_positive_gap_share']}`",
        f"- MAE正差距获益区/非获益区份额：`{gaps['mae']['beneficial_positive_gap_share']}` / `{gaps['mae']['nonbeneficial_positive_gap_share']}`",
        "",
        "本报告只用于冻结下一结构或停止；没有训练v3、扩seed或访问test。",
    ]
    temporary_md = REPORT_MD.with_suffix(".md.tmp")
    temporary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary_md.replace(REPORT_MD)


def main() -> int:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    job = state["jobs"][FORMAL_JOB]
    if job.get("state") != "SUCCESS" or not job.get("run_dir"):
        raise RuntimeError("Step 22.27 formal diagnostic is not successful")
    run_dir = Path(job["run_dir"])
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    with np.load(run_dir / "bf_cmvd_failure_tokens.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    report = analyze(metrics, arrays)
    write_report(report)
    print(json.dumps({
        "decision": report["decision"],
        "main_cause": report["main_cause"],
        "recommended_v3": report["recommended_v3"],
        "report_json": str(REPORT_JSON),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
