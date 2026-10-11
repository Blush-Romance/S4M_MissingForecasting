#!/usr/bin/env python3
"""Audit the fixed Electricity asset before any Step-21 model run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPLETE = PROJECT_ROOT / "data/electricity/electricity.csv"
MISSING = (
    PROJECT_ROOT
    / "data/electricity/electricity_point_paper_r0.06_l5_seed2023.csv"
)
MASK = (
    PROJECT_ROOT
    / "artifacts/masks/electricity/point/rate_0.06/seed_2023.npz"
)
METADATA = MASK.with_suffix(".json")
MANIFEST = PROJECT_ROOT / "artifacts/manifests/s4m_data_assets.yaml"
REPORT = PROJECT_ROOT / "reports/stage21_electricity_asset_test.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contiguous_run_lengths(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return stops - starts


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    required = (COMPLETE, MISSING, MASK, METADATA, MANIFEST)
    missing_paths = [str(path) for path in required if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"missing Step-21 assets: {missing_paths}")

    complete = pd.read_csv(COMPLETE)
    degraded = pd.read_csv(MISSING)
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    mask_payload = np.load(MASK)
    row_mask = mask_payload["row_mask"].astype(bool, copy=False)
    value_columns = [column for column in complete.columns if column != "date"]

    checks = {
        "shape_26304_by_322": complete.shape == (26304, 322),
        "value_columns_321": len(value_columns) == 321,
        "complete_has_no_nan": int(complete.isna().sum().sum()) == 0,
        "missing_shape_matches_complete": degraded.shape == complete.shape,
        "columns_match": list(degraded.columns) == list(complete.columns),
        "dates_match_exactly": degraded["date"].equals(complete["date"]),
        "mask_shape_matches_rows": row_mask.shape == (len(complete),),
        "seed_is_2023": int(mask_payload["seed"]) == 2023,
        "block_length_is_5": int(mask_payload["block_length"]) == 5,
        "paper_rate_is_0_260": bool(
            np.isclose(float(mask_payload["paper_realized_rate"]), 0.260)
        ),
    }

    degraded_values = degraded[value_columns].to_numpy(dtype=np.float64)
    complete_values = complete[value_columns].to_numpy(dtype=np.float64)
    nan_matrix = np.isnan(degraded_values)
    checks["row_shared_missing_mask"] = bool(
        np.all(nan_matrix == row_mask[:, None])
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

    run_lengths = contiguous_run_lengths(row_mask)
    realized_rate = float(row_mask.mean())
    checks["all_missing_runs_at_least_l5"] = bool(
        run_lengths.size > 0 and int(run_lengths.min()) >= 5
    )
    checks["realized_rate_matches_metadata"] = bool(
        np.isclose(realized_rate, float(metadata["realized_missing_rate"]))
    )
    checks["realized_rate_nearest_whole_block_tolerance"] = bool(
        abs(realized_rate - 0.260) <= 5.0 / len(complete)
    )
    checks["source_hash_matches_metadata"] = (
        sha256_file(COMPLETE) == metadata["source_sha256"]
    )
    checks["output_hash_matches_metadata"] = (
        sha256_file(MISSING) == metadata["output_sha256"]
    )
    checks["mask_hash_matches_metadata"] = (
        sha256_file(MASK) == metadata["mask_sha256"]
    )

    manifest_entry = manifest["assets"][
        "electricity_point_r006_l5_seed2023_paper_calibrated"
    ]
    checks["manifest_output_hash_matches"] = (
        manifest_entry["sha256"] == metadata["output_sha256"]
    )
    checks["manifest_mask_hash_matches"] = (
        manifest_entry["mask_sha256"] == metadata["mask_sha256"]
    )
    checks["manifest_protocol_is_approximate"] = (
        manifest_entry["protocol_match"]
        == "approximate_missing_mask_seed_unavailable"
    )

    num_train = int(len(complete) * 0.7)
    num_test = int(len(complete) * 0.2)
    num_validation = len(complete) - num_train - num_test
    checks["split_counts_cover_all_rows"] = (
        num_train + num_validation + num_test == len(complete)
    )

    failures = [name for name, passed in checks.items() if not passed]
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
        "realized_missing_rows": int(row_mask.sum()),
        "realized_missing_rate": realized_rate,
        "paper_reported_realized_rate": 0.260,
        "contiguous_run_count": int(run_lengths.size),
        "contiguous_run_min": int(run_lengths.min()),
        "contiguous_run_max": int(run_lengths.max()),
        "observed_max_abs_error": observed_max_abs_error,
        "source_sha256": sha256_file(COMPLETE),
        "output_sha256": sha256_file(MISSING),
        "mask_sha256": sha256_file(MASK),
    }
    atomic_json(REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
