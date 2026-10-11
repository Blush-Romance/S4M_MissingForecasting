#!/usr/bin/env python3
"""Screen frozen MRRA-v3 internal features with time-isolated validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from analyze_stage22_mrra_v3_error import (
    AUDIT,
    DEV,
    FIT,
    aggregate_choice,
    errors,
    latest_formal_run,
    load_run,
    spearman,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "reports" / "stage22_mrra_internal_feature_screen.json"
REPORT_MD = ROOT / "reports" / "stage22_mrra_internal_feature_screen.md"


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def latest_internal_run() -> Path:
    prefix = "stage22_internal_valcache_mrra_v3__electricity__validation__seed2023__"
    candidates: list[Path] = []
    for path in (ROOT / "runs").glob(prefix + "*"):
        status_path = path / "status.json"
        config_path = path / "resolved_config.yaml"
        if not status_path.is_file() or not config_path.is_file():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if status.get("state") != "SUCCESS":
            continue
        if bool(resolved["experiment"].get("smoke_test")):
            continue
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError("no successful formal MRRA internal cache found")
    return sorted(candidates)[-1]


def eligible_fraction(result: dict[str, Any]) -> bool:
    fraction = float(result["candidate_fraction"])
    return 0.05 <= fraction <= 0.95


def screen_features(
    features: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, Any]:
    mse_gain = reference["mse"] - candidate["mse"]
    mae_gain = reference["mae"] - candidate["mae"]
    summaries: dict[str, Any] = {}
    feasible: list[dict[str, Any]] = []
    for name in sorted(features):
        values = np.asarray(features[name], dtype=np.float64)
        finite = bool(np.isfinite(values).all())
        nonconstant_fit = bool(float(np.std(values[FIT])) > 0.0) if finite else False
        summary = {
            "finite": finite,
            "nonconstant_fit": nonconstant_fit,
            "minimum": float(values.min()) if finite else None,
            "maximum": float(values.max()) if finite else None,
            "spearman_with_mse_gain": spearman(values, mse_gain) if finite else None,
            "spearman_with_mae_gain": spearman(values, mae_gain) if finite else None,
            "fit_development_feasible_rule_count": 0,
        }
        summaries[name] = summary
        if not finite or not nonconstant_fit:
            continue
        thresholds = np.unique(np.quantile(values[FIT], np.linspace(0.0, 1.0, 21)))
        for direction in ("low", "high"):
            for threshold in thresholds:
                choose = values <= threshold if direction == "low" else values >= threshold
                fit_result = aggregate_choice(reference, candidate, choose, FIT)
                if not fit_result["gate_pass"] or not eligible_fraction(fit_result):
                    continue
                dev_result = aggregate_choice(reference, candidate, choose, DEV)
                if not dev_result["gate_pass"] or not eligible_fraction(dev_result):
                    continue
                score = (
                    float(dev_result["mse"]) / float(dev_result["reference_mse"])
                    + float(dev_result["mae"]) / float(dev_result["reference_mae"])
                )
                feasible.append({
                    "feature": name,
                    "direction": direction,
                    "threshold": float(threshold),
                    "development_score": float(score),
                    "fit": fit_result,
                    "development": dev_result,
                    "choose": choose,
                })
                summary["fit_development_feasible_rule_count"] += 1

    if not feasible:
        return {
            "status": "NO_FEASIBLE_RULE_ON_FIT_AND_DEVELOPMENT",
            "audit_gate_pass": False,
            "candidate_rule_count": 0,
            "feature_summaries": summaries,
            "selected_rule": None,
        }
    selected = min(
        feasible,
        key=lambda item: (
            item["development_score"],
            item["development"]["candidate_fraction"],
            item["feature"],
            item["threshold"],
            item["direction"],
        ),
    )
    choose = selected.pop("choose")
    selected["audit"] = aggregate_choice(reference, candidate, choose, AUDIT)
    audit_pass = bool(selected["audit"]["gate_pass"])
    return {
        "status": "AUDIT_PASS" if audit_pass else "AUDIT_FAIL",
        "audit_gate_pass": audit_pass,
        "candidate_rule_count": len(feasible),
        "feature_summaries": summaries,
        "selected_rule": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-run", default=None)
    parser.add_argument("--pg95-run", default=None)
    parser.add_argument("--previous-v3-run", default=None)
    args = parser.parse_args()
    internal_run = Path(args.internal_run).resolve() if args.internal_run else latest_internal_run()
    pg95_run = Path(args.pg95_run).resolve() if args.pg95_run else latest_formal_run("pg95")
    previous_v3_run = (
        Path(args.previous_v3_run).resolve()
        if args.previous_v3_run else latest_formal_run("mrra_v3")
    )
    internal_metrics, internal_samples, internal_config = load_run(internal_run)
    pg95_metrics, pg95_samples, pg95_config = load_run(pg95_run)
    previous_metrics, previous_samples, previous_config = load_run(previous_v3_run)
    feature_keys = sorted(key for key in internal_samples if key.startswith("feature_"))
    protocol_checks = {
        "internal_artifact_pass": internal_metrics.get("artifact_status") == "PASS",
        "validation_only": internal_metrics.get("uses_test_split") is False,
        "frozen": internal_metrics.get("updates_model_or_memory") is False,
        "internal_features_declared": internal_metrics.get("internal_features_cached") is True,
        "full_validation": (
            int(internal_metrics.get("sample_count", 0)) == 2528
            and int(internal_metrics.get("validation_batches_processed", 0)) == 158
        ),
        "feature_count_matches": feature_keys == internal_metrics.get("internal_feature_keys"),
        "same_data_as_pg95": internal_config["data"] == pg95_config["data"],
        "same_data_as_previous_v3": internal_config["data"] == previous_config["data"],
        "sample_index_matches_pg95": bool(np.array_equal(
            internal_samples["sample_index"], pg95_samples["sample_index"]
        )),
        "history_mask_matches_pg95": bool(np.array_equal(
            internal_samples["history_mask_packed"], pg95_samples["history_mask_packed"]
        )),
        "instrumentation_preserves_v3_sse": bool(np.allclose(
            internal_samples["complete_sse_original"],
            previous_samples["complete_sse_original"], rtol=0.0, atol=1e-6,
        )),
        "instrumentation_preserves_v3_sae": bool(np.allclose(
            internal_samples["complete_sae_original"],
            previous_samples["complete_sae_original"], rtol=0.0, atol=1e-6,
        )),
        "feature_count_positive": len(feature_keys) > 0,
        "previous_artifacts_pass": (
            pg95_metrics.get("artifact_status") == "PASS"
            and previous_metrics.get("artifact_status") == "PASS"
        ),
    }
    if not all(protocol_checks.values()):
        raise RuntimeError(f"stage22.15 protocol failed: {protocol_checks}")
    features = {
        key.removeprefix("feature_"): internal_samples[key]
        for key in feature_keys
    }
    reference = errors(pg95_samples)
    candidate = errors(internal_samples)
    screening = screen_features(features, reference, candidate)
    decision = (
        "INTERNAL_FEATURE_GATE_SUPPORTED"
        if screening["audit_gate_pass"]
        else "MRRA_ACCURACY_BRANCH_STOP"
    )
    payload = {
        "step": "22.15",
        "status": "PASS",
        "decision": decision,
        "scope": "Electricity seed2023 frozen-checkpoint validation-only internal feature screen",
        "test_accessed": False,
        "runs": {
            "internal": str(internal_run),
            "pg95": str(pg95_run),
            "previous_mrra_v3": str(previous_v3_run),
        },
        "protocol_checks": protocol_checks,
        "split": {
            "fit": [0, 1264], "embargo_1": [1264, 1456],
            "development": [1456, 1888], "embargo_2": [1888, 2080],
            "audit": [2080, 2528],
        },
        "selection_constraints": {
            "thresholds": "fit quantiles 0,5,...,100 percent",
            "directions": ["low", "high"],
            "minimum_candidate_fraction": 0.05,
            "maximum_candidate_fraction": 0.95,
            "fit_and_development_dual_nondegradation_required": True,
            "audit_used_once_after_selection": True,
        },
        "screening": screening,
        "limitations": [
            "single-seed validation diagnosis is not statistical confirmation",
            "a supported rule would still require implementation and new validation-locked training",
            "the test split was not accessed",
        ],
    }
    atomic_text(REPORT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    selected = screening["selected_rule"]
    lines = [
        "# 步骤22.15：MRRA 内部可观测特征筛选", "",
        "- 协议：`PASS`",
        f"- 特征数：`{len(feature_keys)}`",
        f"- fit/development 可行规则数：`{screening['candidate_rule_count']}`",
        f"- 审计状态：`{screening['status']}`",
        f"- 决策：`{decision}`",
        "- test 访问：`false`",
    ]
    if selected is not None:
        lines.extend([
            f"- 选中特征：`{selected['feature']}`",
            f"- 方向/阈值：`{selected['direction']} / {selected['threshold']:.9g}`",
            f"- audit MSE/MAE 收益：`{selected['audit']['mse_gain_percent']:+.6f}% / {selected['audit']['mae_gain_percent']:+.6f}%`",
            f"- audit MRRA-v3 选择比例：`{selected['audit']['candidate_fraction']:.6f}`",
        ])
    lines.extend(["", "完整特征相关性、候选规则和时间隔离结果见 JSON。", ""])
    atomic_text(REPORT_MD, "\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"REPORT_JSON={REPORT_JSON}")
    print(f"REPORT_MD={REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
