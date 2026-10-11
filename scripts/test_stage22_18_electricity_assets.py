#!/usr/bin/env python3
"""Audit the unseen cross-mask Electricity asset for Step 22.18."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
COMPLETE = ROOT / "data/electricity/electricity.csv"
MISSING = ROOT / "data/electricity/electricity_point_paper_r0.06_l5_seed2024.csv"
MASK_2024 = ROOT / "artifacts/masks/electricity/point/rate_0.06/seed_2024.npz"
META_2024 = MASK_2024.with_suffix(".json")
MASK_2023 = ROOT / "artifacts/masks/electricity/point/rate_0.06/seed_2023.npz"
MANIFEST = ROOT / "artifacts/manifests/s4m_data_assets.yaml"
REPORT = ROOT / "reports/stage22_18_electricity_asset_test.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contiguous_run_lengths(mask: np.ndarray) -> np.ndarray:
    transitions = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    return np.flatnonzero(transitions == -1) - np.flatnonzero(transitions == 1)


def rebuild_mask(starts: np.ndarray, rows: int, block_length: int) -> np.ndarray:
    rebuilt = np.zeros(rows, dtype=bool)
    for start in starts.astype(np.int64, copy=False):
        rebuilt[int(start) : int(start) + block_length] = True
    return rebuilt


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    required = (COMPLETE, MISSING, MASK_2024, META_2024, MASK_2023, MANIFEST)
    absent = [str(path) for path in required if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"missing Step-22.18 assets: {absent}")

    complete = pd.read_csv(COMPLETE)
    degraded = pd.read_csv(MISSING)
    metadata = json.loads(META_2024.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    new_payload = np.load(MASK_2024)
    old_payload = np.load(MASK_2023)
    new_mask = new_payload["row_mask"].astype(bool, copy=False)
    old_mask = old_payload["row_mask"].astype(bool, copy=False)
    starts = new_payload["start_indices"]
    value_columns = [column for column in complete.columns if column != "date"]

    rows = len(complete)
    num_train = int(rows * 0.7)
    num_test = int(rows * 0.2)
    num_validation = rows - num_train - num_test
    val_start = num_train
    val_stop = num_train + num_validation

    checks = {
        "shape_26304_by_322": complete.shape == (26304, 322),
        "value_columns_321": len(value_columns) == 321,
        "complete_has_no_nan": int(complete.isna().sum().sum()) == 0,
        "missing_shape_matches_complete": degraded.shape == complete.shape,
        "columns_match": list(degraded.columns) == list(complete.columns),
        "dates_match_exactly": degraded["date"].equals(complete["date"]),
        "new_mask_shape_matches_rows": new_mask.shape == (rows,),
        "old_mask_shape_matches_rows": old_mask.shape == (rows,),
        "seed_is_2024": int(new_payload["seed"]) == 2024,
        "block_length_is_5": int(new_payload["block_length"]) == 5,
        "nominal_rate_is_0_06": bool(
            np.isclose(float(new_payload["nominal_start_rate"]), 0.06)
        ),
        "paper_rate_is_0_260": bool(
            np.isclose(float(new_payload["paper_realized_rate"]), 0.260)
        ),
        "mask_rebuilds_exactly_from_starts": bool(
            np.array_equal(new_mask, rebuild_mask(starts, rows, 5))
        ),
        "full_mask_differs_from_seed2023": not np.array_equal(new_mask, old_mask),
        "train_mask_differs_from_seed2023": not np.array_equal(
            new_mask[:num_train], old_mask[:num_train]
        ),
        "validation_mask_differs_from_seed2023": not np.array_equal(
            new_mask[val_start:val_stop], old_mask[val_start:val_stop]
        ),
    }

    degraded_values = degraded[value_columns].to_numpy(dtype=np.float64)
    complete_values = complete[value_columns].to_numpy(dtype=np.float64)
    nan_matrix = np.isnan(degraded_values)
    checks["row_shared_missing_mask"] = bool(
        np.all(nan_matrix == new_mask[:, None])
    )
    observed = ~nan_matrix
    observed_max_abs_error = float(
        np.max(np.abs(degraded_values[observed] - complete_values[observed]))
    )
    checks["observed_values_preserved_with_csv_tolerance"] = bool(
        np.allclose(
            degraded_values[observed],
            complete_values[observed],
            rtol=1e-12,
            atol=1e-12,
        )
    )

    runs = contiguous_run_lengths(new_mask)
    realized_rate = float(new_mask.mean())
    checks["all_missing_runs_at_least_l5"] = bool(
        runs.size > 0 and int(runs.min()) >= 5
    )
    checks["realized_rate_matches_metadata"] = bool(
        np.isclose(realized_rate, float(metadata["realized_missing_rate"]))
    )
    checks["realized_rate_nearest_whole_block_tolerance"] = bool(
        abs(realized_rate - 0.260) <= 5.0 / rows
    )
    checks["source_hash_matches_metadata"] = (
        sha256_file(COMPLETE) == metadata["source_sha256"]
    )
    checks["output_hash_matches_metadata"] = (
        sha256_file(MISSING) == metadata["output_sha256"]
    )
    checks["mask_hash_matches_metadata"] = (
        sha256_file(MASK_2024) == metadata["mask_sha256"]
    )
    entry = manifest["assets"][
        "electricity_point_r006_l5_seed2024_paper_calibrated"
    ]
    checks["manifest_output_hash_matches"] = (
        entry["sha256"] == metadata["output_sha256"]
    )
    checks["manifest_mask_hash_matches"] = (
        entry["mask_sha256"] == metadata["mask_sha256"]
    )
    checks["manifest_protocol_is_approximate"] = (
        entry["protocol_match"] == "approximate_missing_mask_seed_unavailable"
    )
    checks["split_counts_cover_all_rows"] = (
        num_train + num_validation + num_test == rows
    )

    failures = [name for name, passed in checks.items() if not passed]
    disagreement = new_mask != old_mask
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "shape": list(complete.shape),
        "split_counts": {
            "train": num_train,
            "validation": num_validation,
            "test": num_test,
        },
        "realized_missing_rows": int(new_mask.sum()),
        "realized_missing_rate": realized_rate,
        "contiguous_run_count": int(runs.size),
        "contiguous_run_min": int(runs.min()),
        "contiguous_run_max": int(runs.max()),
        "seed2023_seed2024_mask_disagreement": {
            "full_rows": int(disagreement.sum()),
            "full_fraction": float(disagreement.mean()),
            "train_rows": int(disagreement[:num_train].sum()),
            "validation_rows": int(disagreement[val_start:val_stop].sum()),
        },
        "observed_max_abs_error": observed_max_abs_error,
        "source_sha256": sha256_file(COMPLETE),
        "output_sha256": sha256_file(MISSING),
        "mask_sha256": sha256_file(MASK_2024),
        "test_labels_or_model_outputs_accessed": False,
    }
    atomic_json(REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
