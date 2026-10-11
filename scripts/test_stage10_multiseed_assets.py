#!/usr/bin/env python3
"""Audit the pre-registered Weather masks for the Step-10 paired seed study."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2023, 2024, 2025)
REPORT = ROOT / "reports" / "stage10_multiseed_asset_test.json"
SEED2023_FROZEN_HASHES = {
    "csv": "9b83f16d0ba08495a3433735cee206cf2e9dc25cab5f97ca122f3cd534e0bc5c",
    "mask": "46c9ad3f18e164e3dd0b9efbfed44090918c9338201ce22e64eef6f8bd3415a4",
    "metadata": "d94c5d1791475751f0b88f3871501829e92c10342580354292274303fd6dbb8f",
}


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
        json.dumps(payload, default=json_default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> int:
    complete = pd.read_csv(ROOT / "data" / "weather" / "weather.csv")
    value_columns = [column for column in complete.columns if column != "date"]
    manifest = yaml.safe_load(
        (ROOT / "artifacts" / "manifests" / "s4m_data_assets.yaml").read_text(
            encoding="utf-8"
        )
    )
    seed_reports = {}
    masks = []
    for seed in SEEDS:
        csv_path = ROOT / "data" / "weather" / f"weather_point_paper_r0.06_l5_seed{seed}.csv"
        mask_path = ROOT / "artifacts" / "masks" / "weather" / "point" / "rate_0.06" / f"seed_{seed}.npz"
        metadata_path = mask_path.with_suffix(".json")
        missing = pd.read_csv(csv_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(mask_path) as stored:
            row_mask = stored["row_mask"].astype(bool)
            starts = stored["start_indices"].astype(np.int64)
            block_length = int(stored["block_length"])
            stored_seed = int(stored["seed"])
        rebuilt = np.zeros(len(complete), dtype=bool)
        for start in starts:
            rebuilt[start : start + block_length] = True
        any_missing = missing[value_columns].isna().any(axis=1).to_numpy()
        all_missing = missing[value_columns].isna().all(axis=1).to_numpy()
        observed = ~row_mask
        observed_equal = np.allclose(
            missing.loc[observed, value_columns].to_numpy(dtype=np.float64),
            complete.loc[observed, value_columns].to_numpy(dtype=np.float64),
            rtol=1e-12,
            atol=1e-12,
        )
        csv_hash = sha256_file(csv_path)
        mask_hash = sha256_file(mask_path)
        metadata_hash = sha256_file(metadata_path)
        key = f"weather_point_r006_l5_seed{seed}_paper_calibrated"
        checks = {
            "files_exist": all(path.is_file() for path in (csv_path, mask_path, metadata_path)),
            "shape_matches_complete": missing.shape == complete.shape,
            "dates_match_complete": missing["date"].equals(complete["date"]),
            "all_variables_share_row_mask": np.array_equal(any_missing, all_missing)
            and np.array_equal(all_missing, row_mask),
            "observed_values_match_complete": bool(observed_equal),
            "mask_rebuilds_from_starts": np.array_equal(row_mask, rebuilt),
            "block_length_is_five": block_length == 5,
            "stored_seed_matches": stored_seed == seed,
            "metadata_seed_matches": int(metadata["fixed_generation_seed"]) == seed,
            "metadata_paths_match": Path(metadata["output_path"]).resolve() == csv_path.resolve()
            and Path(metadata["mask_path"]).resolve() == mask_path.resolve(),
            "metadata_hashes_match": metadata["output_sha256"] == csv_hash
            and metadata["mask_sha256"] == mask_hash,
            "realized_rate_matches_metadata": np.isclose(
                row_mask.mean(), float(metadata["realized_missing_rate"]), rtol=0.0, atol=1e-15
            ),
            "realized_rate_near_paper": abs(float(row_mask.mean()) - 0.247)
            <= 5.0 / len(row_mask),
            "manifest_entry_matches": key in manifest["assets"]
            and manifest["assets"][key]["sha256"] == csv_hash
            and manifest["assets"][key]["mask_sha256"] == mask_hash,
        }
        seed_reports[str(seed)] = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "realized_missing_rate": float(row_mask.mean()),
            "csv_sha256": csv_hash,
            "mask_sha256": mask_hash,
            "metadata_sha256": metadata_hash,
        }
        masks.append(row_mask)

    cross_seed_checks = {
        "all_masks_pairwise_distinct": all(
            not np.array_equal(masks[left], masks[right])
            for left in range(len(masks))
            for right in range(left + 1, len(masks))
        ),
        "seed2023_csv_unchanged": seed_reports["2023"]["csv_sha256"]
        == SEED2023_FROZEN_HASHES["csv"],
        "seed2023_mask_unchanged": seed_reports["2023"]["mask_sha256"]
        == SEED2023_FROZEN_HASHES["mask"],
        "seed2023_metadata_unchanged": seed_reports["2023"]["metadata_sha256"]
        == SEED2023_FROZEN_HASHES["metadata"],
    }
    passed = all(item["status"] == "PASS" for item in seed_reports.values()) and all(
        cross_seed_checks.values()
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "pre_registered_seeds": list(SEEDS),
        "seed_reports": seed_reports,
        "cross_seed_checks": cross_seed_checks,
    }
    atomic_json(REPORT, report)
    print(json.dumps(report, default=json_default, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
