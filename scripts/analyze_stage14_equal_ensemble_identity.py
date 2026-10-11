#!/usr/bin/env python3
"""Exact MSE-only audit for the Stage-14 equal-weight two-view ensemble.

The cached Stage-14 artifacts do not contain raw predictions, but squared error
of the arithmetic mean is exactly recoverable:

  ||(e_on + e_nested) / 2||^2
    = (||e_on||^2 + ||e_nested||^2) / 2
      - ||prediction_on - prediction_nested||^2 / 4.

MAE is intentionally not estimated here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/stage14_uncertainty_queue/state.json"
JSON_REPORT = ROOT / "reports/stage14_equal_ensemble_identity.json"
MD_REPORT = ROOT / "reports/stage14_equal_ensemble_identity.md"


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def analyze_seed(samples_path: Path) -> dict[str, float | int | bool]:
    with np.load(samples_path, allow_pickle=False) as loaded:
        sse_on = loaded["complete_sse_on"].astype(np.float64)
        sse_nested = loaded["complete_sse_nested"].astype(np.float64)
        count = loaded["complete_count"].astype(np.float64)
        disagreement_mse = loaded["mask_disagreement_mse"].astype(np.float64)
    if not (sse_on.shape == sse_nested.shape == count.shape == disagreement_mse.shape):
        raise RuntimeError(f"Shape mismatch in {samples_path}")
    if np.any(count <= 0) or not all(
        np.isfinite(array).all()
        for array in (sse_on, sse_nested, count, disagreement_mse)
    ):
        raise RuntimeError(f"Invalid cached values in {samples_path}")

    disagreement_sse = disagreement_mse * count
    ensemble_sse = 0.5 * (sse_on + sse_nested) - 0.25 * disagreement_sse
    numerical_tolerance = 1e-8 * np.maximum(1.0, np.abs(sse_on))
    if np.any(ensemble_sse < -numerical_tolerance):
        raise RuntimeError(f"Negative reconstructed SSE in {samples_path}")
    ensemble_sse = np.maximum(ensemble_sse, 0.0)

    total_count = float(count.sum())
    mse_on = float(sse_on.sum() / total_count)
    mse_nested = float(sse_nested.sum() / total_count)
    mse_ensemble = float(ensemble_sse.sum() / total_count)
    best_single = min(mse_on, mse_nested)
    return {
        "sample_count": int(sse_on.size),
        "element_count": int(total_count),
        "mse_on": mse_on,
        "mse_nested": mse_nested,
        "mse_equal_ensemble": mse_ensemble,
        "relative_gain_vs_on_pct": float((mse_on - mse_ensemble) / mse_on * 100.0),
        "relative_gain_vs_nested_pct": float(
            (mse_nested - mse_ensemble) / mse_nested * 100.0
        ),
        "relative_gain_vs_best_single_pct": float(
            (best_single - mse_ensemble) / best_single * 100.0
        ),
        "improves_both_single_views": bool(mse_ensemble < best_single),
        "mae_available": False,
    }


def main() -> int:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    results = {}
    for seed in (2024, 2025):
        job_name = f"weather_pg95_seed{seed}_uncertainty_formal"
        job = state["jobs"][job_name]
        if job.get("state") != "SUCCESS" or not job.get("run_dir"):
            raise RuntimeError(f"{job_name} is not successful")
        run_dir = Path(job["run_dir"])
        results[str(seed)] = analyze_seed(
            run_dir / "validation_uncertainty_samples.npz"
        )

    both_improve = all(
        result["improves_both_single_views"] for result in results.values()
    )
    report = {
        "status": "PASS" if both_improve else "FAIL",
        "scope": "Post-hoc exact MSE identity on frozen validation caches; no test data",
        "formula": (
            "SSE(mean(on,nested)) = 0.5*(SSE_on+SSE_nested) "
            "- 0.25*SSE(on-nested)"
        ),
        "mae_available": False,
        "results": results,
        "decision": (
            "RERUN_VALIDATION_TO_MEASURE_EQUAL_ENSEMBLE_MAE"
            if both_improve
            else "DO_NOT_PRIORITIZE_EQUAL_ENSEMBLE"
        ),
    }
    atomic_text(JSON_REPORT, json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Stage 14 equal-ensemble identity audit",
        "",
        "- 范围：冻结validation缓存；没有读取test。",
        "- 性质：平方误差恒等式精确重算，不估计MAE。",
        "",
        "| seed | on MSE | nested MSE | equal ensemble MSE | vs on | vs best single |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in (2024, 2025):
        result = results[str(seed)]
        lines.append(
            f"| {seed} | {result['mse_on']:.9f} | "
            f"{result['mse_nested']:.9f} | {result['mse_equal_ensemble']:.9f} | "
            f"{result['relative_gain_vs_on_pct']:.4f}% | "
            f"{result['relative_gain_vs_best_single_pct']:.4f}% |"
        )
    lines.extend(["", f"- 决策：`{report['decision']}`", ""])
    atomic_text(MD_REPORT, "\n".join(lines))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if both_improve else 2


if __name__ == "__main__":
    raise SystemExit(main())
