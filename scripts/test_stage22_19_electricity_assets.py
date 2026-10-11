#!/usr/bin/env python3
"""Audit the preregistered unseen Electricity seed2025 asset for Step 22.19."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_paper_point_masks import select_random_blocks  # noqa: E402


COMPLETE = ROOT / "data/electricity/electricity.csv"
MISSING = ROOT / "data/electricity/electricity_point_paper_r0.06_l5_seed2025.csv"
MASK = ROOT / "artifacts/masks/electricity/point/rate_0.06/seed_2025.npz"
META = MASK.with_suffix(".json")
OLD_MASKS = {
    seed: ROOT / f"artifacts/masks/electricity/point/rate_0.06/seed_{seed}.npz"
    for seed in (2023, 2024)
}
MANIFEST = ROOT / "artifacts/manifests/s4m_data_assets.yaml"
PROTOCOL = ROOT / "docs/STAGE22_19_ARD_MCPR_PROTOCOL_ZH.md"
REPORT = ROOT / "reports/stage22_19_electricity_asset_test.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def contiguous_run_lengths(mask: np.ndarray) -> np.ndarray:
    transitions = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    return np.flatnonzero(transitions == -1) - np.flatnonzero(transitions == 1)


def rebuild_mask(starts: np.ndarray, rows: int, block_length: int) -> np.ndarray:
    rebuilt = np.zeros(rows, dtype=bool)
    for start in starts.astype(np.int64, copy=False):
        rebuilt[int(start) : int(start) + block_length] = True
    return rebuilt


def main() -> int:
    required = (COMPLETE, MISSING, MASK, META, MANIFEST, PROTOCOL, *OLD_MASKS.values())
    absent = [str(path) for path in required if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"missing Step-22.19 assets: {absent}")

    complete = pd.read_csv(COMPLETE)
    degraded = pd.read_csv(MISSING)
    metadata = json.loads(META.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    payload = np.load(MASK)
    mask = payload["row_mask"].astype(bool, copy=False)
    starts = payload["start_indices"]
    old_masks = {
        seed: np.load(path)["row_mask"].astype(bool, copy=False)
        for seed, path in OLD_MASKS.items()
    }
    value_columns = [column for column in complete.columns if column != "date"]
    rows = len(complete)
    num_train = int(rows * 0.7)
    num_test = int(rows * 0.2)
    num_validation = rows - num_train - num_test
    val_start = num_train
    val_stop = num_train + num_validation

    regenerated_mask, regenerated_starts, target_rows = select_random_blocks(
        n_rows=rows,
        block_length=5,
        target_rate=0.260,
        seed=2025,
    )
    degraded_values = degraded[value_columns].to_numpy(dtype=np.float64)
    complete_values = complete[value_columns].to_numpy(dtype=np.float64)
    nan_matrix = np.isnan(degraded_values)
    observed = ~nan_matrix
    observed_max_abs_error = float(
        np.max(np.abs(degraded_values[observed] - complete_values[observed]))
    )
    runs = contiguous_run_lengths(mask)
    realized_rate = float(mask.mean())
    manifest_entry = manifest["assets"][
        "electricity_point_r006_l5_seed2025_paper_calibrated"
    ]

    checks = {
        "protocol_frozen_before_asset": (
            PROTOCOL.stat().st_mtime_ns <= MISSING.stat().st_mtime_ns
        ),
        "shape_26304_by_322": complete.shape == (26304, 322),
        "value_columns_321": len(value_columns) == 321,
        "complete_has_no_nan": int(complete.isna().sum().sum()) == 0,
        "missing_shape_matches_complete": degraded.shape == complete.shape,
        "columns_match": list(degraded.columns) == list(complete.columns),
        "dates_match_exactly": degraded["date"].equals(complete["date"]),
        "mask_shape_matches_rows": mask.shape == (rows,),
        "seed_is_2025": int(payload["seed"]) == 2025,
        "block_length_is_5": int(payload["block_length"]) == 5,
        "nominal_rate_is_0_06": bool(
            np.isclose(float(payload["nominal_start_rate"]), 0.06)
        ),
        "paper_rate_is_0_260": bool(
            np.isclose(float(payload["paper_realized_rate"]), 0.260)
        ),
        "mask_rebuilds_from_persisted_starts": bool(
            np.array_equal(mask, rebuild_mask(starts, rows, 5))
        ),
        "fixed_generator_is_exactly_reproducible": bool(
            np.array_equal(mask, regenerated_mask)
            and np.array_equal(starts, regenerated_starts)
        ),
        "target_rows_match_generator": int(target_rows) == 6839,
        "row_shared_missing_mask": bool(np.all(nan_matrix == mask[:, None])),
        "observed_values_preserved_with_csv_tolerance": bool(
            np.allclose(
                degraded_values[observed],
                complete_values[observed],
                rtol=1e-12,
                atol=1e-12,
            )
        ),
        "all_missing_runs_at_least_l5": bool(
            runs.size > 0 and int(runs.min()) >= 5
        ),
        "realized_rate_matches_metadata": bool(
            np.isclose(realized_rate, float(metadata["realized_missing_rate"]))
        ),
        "realized_rate_nearest_row_tolerance": abs(realized_rate - 0.260) <= 1 / rows,
        "source_hash_matches_metadata": (
            sha256_file(COMPLETE) == metadata["source_sha256"]
        ),
        "output_hash_matches_metadata": (
            sha256_file(MISSING) == metadata["output_sha256"]
        ),
        "mask_hash_matches_metadata": sha256_file(MASK) == metadata["mask_sha256"],
        "manifest_output_hash_matches": (
            manifest_entry["sha256"] == metadata["output_sha256"]
        ),
        "manifest_mask_hash_matches": (
            manifest_entry["mask_sha256"] == metadata["mask_sha256"]
        ),
        "manifest_protocol_is_approximate": (
            manifest_entry["protocol_match"]
            == "approximate_missing_mask_seed_unavailable"
        ),
        "split_counts_cover_all_rows": (
            num_train + num_validation + num_test == rows
        ),
    }
    disagreements = {}
    for seed, old_mask in old_masks.items():
        different = mask != old_mask
        checks[f"full_mask_differs_from_seed{seed}"] = not np.array_equal(mask, old_mask)
        checks[f"train_mask_differs_from_seed{seed}"] = not np.array_equal(
            mask[:num_train], old_mask[:num_train]
        )
        checks[f"validation_mask_differs_from_seed{seed}"] = not np.array_equal(
            mask[val_start:val_stop], old_mask[val_start:val_stop]
        )
        disagreements[str(seed)] = {
            "full_rows": int(different.sum()),
            "full_fraction": float(different.mean()),
            "train_rows": int(different[:num_train].sum()),
            "validation_rows": int(different[val_start:val_stop].sum()),
        }

    failures = [name for name, passed in checks.items() if not bool(passed)]
    report = {
        "step": "22.19.4_asset",
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "shape": list(complete.shape),
        "split_counts": {
            "train": num_train,
            "validation": num_validation,
            "test": num_test,
        },
        "realized_missing_rows": int(mask.sum()),
        "realized_missing_rate": realized_rate,
        "contiguous_run_count": int(runs.size),
        "contiguous_run_min": int(runs.min()),
        "contiguous_run_max": int(runs.max()),
        "cross_seed_mask_disagreement": disagreements,
        "observed_max_abs_error": observed_max_abs_error,
        "source_sha256": sha256_file(COMPLETE),
        "output_sha256": sha256_file(MISSING),
        "mask_sha256": sha256_file(MASK),
        "test_metrics_or_model_outputs_accessed": False,
    }
    atomic_json(REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
