#!/usr/bin/env python3
"""Apply the frozen Step 22.29 DMRB failure-attribution decision."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports/stage22_29_dmrb_failure_attribution.json"
REPORT_MD = ROOT / "reports/stage22_29_dmrb_failure_attribution.md"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def conditional_fraction(mask: np.ndarray, condition: np.ndarray) -> float:
    return ratio(float(np.logical_and(mask, condition).sum()), float(condition.sum()))


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    return ratio(
        float(np.logical_and(left, right).sum()),
        float(np.logical_or(left, right).sum()),
    )


def aggregate_geometry(
    arrays: dict[str, np.ndarray], branch: str, region: np.ndarray
) -> dict[str, float]:
    dot = float(arrays[f"{branch}_self_dot"][region].sum(dtype=np.float64))
    residual_energy = float(
        arrays[f"{branch}_self_residual_energy"][region].sum(dtype=np.float64)
    )
    target_energy = float(
        arrays[f"{branch}_self_target_energy"][region].sum(dtype=np.float64)
    )
    mismatch = float(
        arrays[f"{branch}_self_mismatch_energy"][region].sum(dtype=np.float64)
    )
    cosine = ratio(dot, math.sqrt(residual_energy * target_energy))
    projection = ratio(dot, target_energy)
    normalized_mismatch = ratio(mismatch, target_energy)
    return {
        "token_count": int(region.sum()),
        "cosine": cosine,
        "projection": projection,
        "normalized_mismatch": normalized_mismatch,
        "residual_energy": residual_energy,
        "target_energy": target_energy,
        "mismatch_energy": mismatch,
    }


def positive_gap_share(
    arrays: dict[str, np.ndarray], branch: str, region: np.ndarray, metric: str
) -> float:
    gap = np.maximum(
        arrays[f"student_{branch}_{metric}"]
        - arrays[f"teacher_{branch}_{metric}"],
        0.0,
    )
    return ratio(
        float(gap[region].sum(dtype=np.float64)),
        float(gap.sum(dtype=np.float64)),
    )


def analyze_arrays(
    metrics: dict[str, Any], arrays: dict[str, np.ndarray], protocol_pass: bool
) -> dict[str, Any]:
    b5 = arrays["teacher_l5_beneficial"].astype(bool)
    b20 = arrays["teacher_l20_beneficial"].astype(bool)
    joint = arrays["teacher_joint_beneficial"].astype(bool)
    b5_given_joint = conditional_fraction(b5, joint)
    b20_given_joint = conditional_fraction(b20, joint)
    b5_missed_by_joint = ratio(float(np.logical_and(b5, ~joint).sum()), float(b5.sum()))
    b20_missed_by_joint = ratio(
        float(np.logical_and(b20, ~joint).sum()), float(b20.sum())
    )
    partitions = {
        "both_beneficial": conditional_fraction(b5 & b20, joint),
        "only_l5_beneficial": conditional_fraction(b5 & ~b20, joint),
        "only_l20_beneficial": conditional_fraction(~b5 & b20, joint),
        "neither_individually_beneficial": conditional_fraction(~b5 & ~b20, joint),
    }

    target_dot = float(arrays["teacher_component_dot"].sum(dtype=np.float64))
    target_l5_energy = float(
        arrays["l5_self_target_energy"].sum(dtype=np.float64)
    )
    target_l20_energy = float(
        arrays["l20_self_target_energy"].sum(dtype=np.float64)
    )
    target_sum_energy = float(arrays["teacher_sum_energy"].sum(dtype=np.float64))
    teacher_component_cosine = ratio(
        target_dot, math.sqrt(target_l5_energy * target_l20_energy)
    )
    teacher_cancellation = max(
        0.0,
        1.0 - ratio(target_sum_energy, target_l5_energy + target_l20_energy),
    )
    student_dot = float(arrays["student_branch_dot"].sum(dtype=np.float64))
    student_l5_energy = float(
        arrays["l5_self_residual_energy"].sum(dtype=np.float64)
    )
    student_l20_energy = float(
        arrays["l20_self_residual_energy"].sum(dtype=np.float64)
    )
    student_sum_energy = float(arrays["student_sum_energy"].sum(dtype=np.float64))
    student_branch_cosine = ratio(
        student_dot, math.sqrt(student_l5_energy * student_l20_energy)
    )
    student_cancellation = max(
        0.0,
        1.0 - ratio(student_sum_energy, student_l5_energy + student_l20_energy),
    )

    metric_table = {
        label: {
            metric: float(arrays[f"{label}_{metric}"].mean(dtype=np.float64))
            for metric in ("mse", "mae")
        }
        for label in (
            "backbone", "teacher_l5", "teacher_l20", "teacher_joint",
            "student_l5", "student_l20", "student_joint",
        )
    }
    interaction = {
        metric: (
            metric_table["student_joint"][metric]
            - metric_table["student_l5"][metric]
            - metric_table["student_l20"][metric]
            + metric_table["backbone"][metric]
        )
        for metric in ("mse", "mae")
    }
    single_candidates = {}
    for branch in ("l5", "l20"):
        label = f"student_{branch}"
        gains = {
            metric: metric_table["backbone"][metric] - metric_table[label][metric]
            for metric in ("mse", "mae")
        }
        advantages = {
            metric: metric_table["student_joint"][metric] - metric_table[label][metric]
            for metric in ("mse", "mae")
        }
        recovery = {
            metric: ratio(max(0.0, advantages[metric]), max(0.0, gains[metric]))
            for metric in ("mse", "mae")
        }
        single_candidates[branch] = {
            "gain_vs_backbone": gains,
            "advantage_vs_full_dmrb": advantages,
            "lost_gain_recovery_fraction": recovery,
            "beats_full_dmrb_both_metrics": all(
                advantages[metric] > 0.0 for metric in ("mse", "mae")
            ),
        }
    conflict_branches = [
        branch for branch, evidence in single_candidates.items()
        if evidence["beats_full_dmrb_both_metrics"]
        and all(
            evidence["lost_gain_recovery_fraction"][metric] >= 0.25
            for metric in ("mse", "mae")
        )
    ]
    branch_sum_conflict = bool(
        conflict_branches
        and interaction["mse"] > 0.0
        and interaction["mae"] > 0.0
    )

    branch_evidence = {}
    for branch, region in (("l5", b5), ("l20", b20)):
        geometry = aggregate_geometry(arrays, branch, region)
        gap_shares = {
            metric: positive_gap_share(arrays, branch, region, metric)
            for metric in ("mse", "mae")
        }
        geometry_fail_count = sum((
            geometry["cosine"] < 0.50,
            geometry["projection"] < 0.50,
            geometry["normalized_mismatch"] > 0.75,
        ))
        branch_evidence[branch] = {
            **geometry,
            "positive_teacher_student_gap_share": gap_shares,
            "geometry_fail_count": int(geometry_fail_count),
            "underfit_gate_pass": bool(
                gap_shares["mse"] >= 0.60
                and gap_shares["mae"] >= 0.60
                and geometry_fail_count >= 2
            ),
        }
    cross_projection = {
        "r5_on_c20": ratio(
            float(arrays["l5_to_l20_target_dot"].sum(dtype=np.float64)),
            target_l20_energy,
        ),
        "r20_on_c5": ratio(
            float(arrays["l20_to_l5_target_dot"].sum(dtype=np.float64)),
            target_l5_energy,
        ),
    }

    supervision_checks = {
        "joint_precision_below_0_75": min(
            b5_given_joint, b20_given_joint
        ) < 0.75,
        "branch_support_missed_above_0_25": max(
            b5_missed_by_joint, b20_missed_by_joint
        ) > 0.25,
        "component_opposition_or_cancellation": (
            teacher_component_cosine < 0.0 or teacher_cancellation > 0.10
        ),
    }
    supervision_mismatch = all(supervision_checks.values())
    dual_underfit = all(
        evidence["underfit_gate_pass"] for evidence in branch_evidence.values()
    )

    if not protocol_pass:
        decision = "PROTOCOL_FAILURE_NO_DECISION"
    elif supervision_mismatch:
        decision = "SCALE_SUPERVISION_MISMATCH_FREEZE_BRANCH_PARETO_MASKS"
    elif branch_sum_conflict:
        decision = "BRANCH_SUM_CONFLICT_FREEZE_CAUSAL_SCALE_ROUTER"
    elif dual_underfit:
        decision = "DUAL_BRANCH_EXPRESSION_UNDERFIT_STOP_LOW_RANK_ROUTE"
    else:
        decision = "MIXED_OR_INDETERMINATE_STOP"

    return {
        "step": "22.29",
        "decision": decision,
        "protocol_pass": bool(protocol_pass),
        "test_sources_accessed": False,
        "metric_table": metric_table,
        "supervision_alignment": {
            "p_b5_given_bjoint": b5_given_joint,
            "p_b20_given_bjoint": b20_given_joint,
            "p_b5_outside_bjoint_given_b5": b5_missed_by_joint,
            "p_b20_outside_bjoint_given_b20": b20_missed_by_joint,
            "jaccard_b5_b20": jaccard(b5, b20),
            "jaccard_b5_bjoint": jaccard(b5, joint),
            "jaccard_b20_bjoint": jaccard(b20, joint),
            "joint_partition": partitions,
            "teacher_component_cosine": teacher_component_cosine,
            "teacher_cancellation_ratio": teacher_cancellation,
            "checks": supervision_checks,
            "gate_pass": bool(supervision_mismatch),
        },
        "branch_composition": {
            "interaction": interaction,
            "single_candidates": single_candidates,
            "conflict_branches": conflict_branches,
            "student_branch_cosine": student_branch_cosine,
            "student_cancellation_ratio": student_cancellation,
            "gate_pass": branch_sum_conflict,
        },
        "branch_expression": {
            "l5": branch_evidence["l5"],
            "l20": branch_evidence["l20"],
            "cross_projection": cross_projection,
            "gate_pass": dual_underfit,
        },
        "source_metrics_path": metrics.get("samples_path"),
        "validation_samples_processed": metrics.get(
            "validation_samples_processed"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-json", default=str(REPORT_JSON))
    parser.add_argument("--output-md", default=str(REPORT_MD))
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    with np.load(metrics["samples_path"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    checks = metrics.get("artifact_checks", {})
    protocol_pass = bool(
        metrics.get("artifact_status") == "PASS"
        and checks
        and all(checks.values())
        and metrics.get("test_split_accessed") is False
        and int(metrics.get("validation_samples_processed", 0)) == 1648
    )
    report = analyze_arrays(metrics, arrays, protocol_pass)
    report["formal_run"] = str(run_dir)
    atomic_text(
        Path(args.output_json),
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    table = report["metric_table"]
    supervision = report["supervision_alignment"]
    expression = report["branch_expression"]
    lines = [
        "# Step 22.29 DMRB 失效归因", "",
        f"- 决策：`{report['decision']}`",
        f"- 协议门：`{report['protocol_pass']}`",
        f"- test来源访问：`{report['test_sources_accessed']}`",
        f"- p0 MSE/MAE：{table['backbone']['mse']:.9f} / {table['backbone']['mae']:.9f}",
        f"- DMRB MSE/MAE：{table['student_joint']['mse']:.9f} / {table['student_joint']['mae']:.9f}",
        f"- P(B5|Bjoint) / P(B20|Bjoint)：{supervision['p_b5_given_bjoint']:.6f} / {supervision['p_b20_given_bjoint']:.6f}",
        f"- 教师分量余弦 / 抵消率：{supervision['teacher_component_cosine']:.6f} / {supervision['teacher_cancellation_ratio']:.6f}",
        f"- L5自目标余弦/投影/失配：{expression['l5']['cosine']:.6f} / {expression['l5']['projection']:.6f} / {expression['l5']['normalized_mismatch']:.6f}",
        f"- L20自目标余弦/投影/失配：{expression['l20']['cosine']:.6f} / {expression['l20']['projection']:.6f} / {expression['l20']['normalized_mismatch']:.6f}",
    ]
    atomic_text(Path(args.output_md), "\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if protocol_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
