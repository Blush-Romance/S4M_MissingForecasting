#!/usr/bin/env python3
"""Generate fixed test-only missingness-shift assets for Gate H.

The training portion is copied byte-for-value from the Step-5 missing CSV so
that fitted normalization statistics and the learned checkpoint remain fixed.
Only the region covered by test input windows is replaced by a new missingness
domain sampled from the complete source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from generate_paper_point_masks import contiguous_runs, select_random_blocks


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    slug: str
    complete_csv: Path
    baseline_missing_csv: Path
    data_dir: Path
    id_realized_rate: float


@dataclass(frozen=True)
class ShiftSpec:
    name: str
    block_length: int
    target_rate: float | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def dataset_specs(root: Path) -> dict[str, DatasetSpec]:
    return {
        "weather": DatasetSpec(
            name="Weather",
            slug="weather",
            complete_csv=root / "data/weather/weather.csv",
            baseline_missing_csv=root / "data/weather/weather_point_paper_r0.06_l5_seed2023.csv",
            data_dir=root / "data/weather/gate_h",
            id_realized_rate=0.24698269318354335,
        ),
        "etth1": DatasetSpec(
            name="ETTh1",
            slug="etth1",
            complete_csv=root / "data/ETT-small/ETTh1.csv",
            baseline_missing_csv=root / "data/ETT-small/ETTh1_point_paper_r0.06_l5_seed2023.csv",
            data_dir=root / "data/ETT-small/gate_h",
            id_realized_rate=0.23088404133180251,
        ),
    }


def shift_specs() -> dict[str, ShiftSpec]:
    return {
        "rate_shift": ShiftSpec(name="rate_shift_r045_l5", block_length=5, target_rate=0.45),
        "geometry_shift": ShiftSpec(name="geometry_shift_idrate_l20", block_length=20, target_rate=None),
    }


def generate(spec: DatasetSpec, shift: ShiftSpec, seed: int, seq_len: int) -> dict:
    complete = pd.read_csv(spec.complete_csv)
    baseline = pd.read_csv(spec.baseline_missing_csv)
    if list(complete.columns) != list(baseline.columns) or len(complete) != len(baseline):
        raise ValueError(f"baseline/complete schema mismatch for {spec.name}")
    value_columns = [column for column in complete.columns if column != "date"]
    if complete[value_columns].isna().any().any():
        raise ValueError(f"complete source has NaNs: {spec.complete_csv}")

    rows = len(complete)
    train_rows = int(rows * 0.7)
    test_rows = int(rows * 0.2)
    test_target_start = rows - test_rows
    evaluation_start = test_target_start - seq_len
    evaluation_rows = rows - evaluation_start
    target_rate = spec.id_realized_rate if shift.target_rate is None else shift.target_rate
    local_mask, start_indices, target_missing_rows = select_random_blocks(
        evaluation_rows,
        shift.block_length,
        target_rate,
        seed,
    )

    shifted = complete.copy()
    shifted.loc[: evaluation_start - 1, value_columns] = baseline.loc[
        : evaluation_start - 1, value_columns
    ].to_numpy()
    shifted_indices = np.flatnonzero(local_mask) + evaluation_start
    shifted.loc[shifted_indices, value_columns] = np.nan

    final_nan = shifted[value_columns].isna().to_numpy()
    if np.any(final_nan.any(axis=1) != final_nan.all(axis=1)):
        raise RuntimeError("partial-variable missing rows are not allowed")
    final_row_mask = final_nan.all(axis=1)
    baseline_nan = baseline[value_columns].isna().to_numpy()
    if not np.array_equal(final_nan[:train_rows], baseline_nan[:train_rows]):
        raise RuntimeError("training missingness changed under a test-only shift")
    observed = ~final_nan
    complete_values = complete[value_columns].to_numpy()
    shifted_values = shifted[value_columns].to_numpy()
    observed_abs_error = np.abs(shifted_values[observed] - complete_values[observed])
    max_observed_abs_error = float(observed_abs_error.max(initial=0.0))
    if not np.allclose(
        shifted_values[observed],
        complete_values[observed],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("observed values changed while generating the shift")

    rate_tag = f"r{target_rate:.3f}".replace(".", "p")
    filename = f"{spec.slug}_{shift.name}_{rate_tag}_seed{seed}.csv"
    output_csv = spec.data_dir / filename
    asset_dir = (
        PROJECT_ROOT
        / "artifacts/masks"
        / spec.slug
        / "gate_h"
        / f"{shift.name}_{rate_tag}"
    )
    mask_path = asset_dir / f"seed_{seed}.npz"
    metadata_path = asset_dir / f"seed_{seed}.json"
    atomic_csv(shifted, output_csv)
    atomic_npz(
        mask_path,
        final_row_mask=final_row_mask,
        evaluation_row_mask=local_mask,
        evaluation_start=np.asarray(evaluation_start, dtype=np.int64),
        test_target_start=np.asarray(test_target_start, dtype=np.int64),
        start_indices_local=start_indices,
        block_length=np.asarray(shift.block_length, dtype=np.int64),
        target_evaluation_rate=np.asarray(target_rate, dtype=np.float64),
        seed=np.asarray(seed, dtype=np.int64),
    )

    run_lengths = contiguous_runs(local_mask)
    metadata = {
        "dataset": spec.name,
        "domain": shift.name,
        "protocol_kind": "gate_h_test_only_missingness_shift_v1",
        "seed": seed,
        "rows": rows,
        "train_rows_preserved": train_rows,
        "evaluation_start": evaluation_start,
        "test_target_start": test_target_start,
        "evaluation_rows": evaluation_rows,
        "block_length": shift.block_length,
        "target_evaluation_missing_rate": target_rate,
        "target_evaluation_missing_rows": target_missing_rows,
        "realized_evaluation_missing_rows": int(local_mask.sum()),
        "realized_evaluation_missing_rate": float(local_mask.mean()),
        "realized_full_csv_missing_rate": float(final_row_mask.mean()),
        "selected_start_count": int(start_indices.size),
        "evaluation_run_count": len(run_lengths),
        "evaluation_run_min": min(run_lengths),
        "evaluation_run_max": max(run_lengths),
        "training_mask_identical_to_step5": True,
        "normalization_fit_region_unchanged": True,
        "observed_value_check_rtol": 1e-12,
        "observed_value_check_atol": 1e-12,
        "max_observed_abs_error": max_observed_abs_error,
        "complete_csv": str(spec.complete_csv),
        "baseline_missing_csv": str(spec.baseline_missing_csv),
        "output_csv": str(output_csv),
        "mask_path": str(mask_path),
        "complete_sha256": sha256_file(spec.complete_csv),
        "baseline_missing_sha256": sha256_file(spec.baseline_missing_csv),
        "output_sha256": sha256_file(output_csv),
        "mask_sha256": sha256_file(mask_path),
        "generator": str(Path(__file__).resolve()),
    }
    atomic_json(metadata, metadata_path)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "weather", "etth1"), default="all")
    parser.add_argument("--shift", choices=("all", "rate_shift", "geometry_shift"), default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = dataset_specs(args.project_root.resolve())
    shifts = shift_specs()
    selected_datasets = datasets if args.dataset == "all" else {args.dataset: datasets[args.dataset]}
    selected_shifts = shifts if args.shift == "all" else {args.shift: shifts[args.shift]}
    outputs = [
        generate(dataset, shift, seed, args.seq_len)
        for dataset in selected_datasets.values()
        for shift in selected_shifts.values()
        for seed in args.seeds
    ]
    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
