#!/usr/bin/env python3
"""Audit Step-22.22 Traffic assets independently before GPU work."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from generate_paper_point_masks import contiguous_runs, select_random_blocks


ROOT = Path(__file__).resolve().parents[1]
COMPLETE = ROOT / "data/traffic/traffic.csv"
MASK_DIRECTORY = ROOT / "artifacts/masks/traffic/point/rate_0.06"
MANIFEST = ROOT / "artifacts/manifests/s4m_data_assets.yaml"
PROTOCOL = ROOT / "docs/STAGE22_22_TRAFFIC_MULTISEED_PROTOCOL_ZH.md"
REPORT = ROOT / "reports/stage22_22_traffic_asset_test.json"
SOURCE_SHA256 = "cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16"
SEEDS = (2026, 2027, 2028)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
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


def main() -> int:
    complete = pd.read_csv(COMPLETE)
    complete_values = complete.iloc[:, 1:].to_numpy(dtype=np.float64)
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["assets"]
    protocol_text = PROTOCOL.read_text(encoding="utf-8")
    checks: dict[str, bool] = {
        "source_hash_frozen": sha256(COMPLETE) == SOURCE_SHA256,
        "source_shape_and_schema": (
            complete.shape == (17544, 863)
            and complete.columns[0] == "date"
            and complete.columns[-1] == "OT"
            and not bool(np.isnan(complete_values).any())
        ),
        "protocol_freezes_all_seeds_before_assets": all(
            token in protocol_text
            for token in (
                "固定纳入seed：2026、2027、2028",
                "不根据seed2027的结果决定是否运行seed2028",
                "0.5*original + 0.25*L5 + 0.25*L20",
            )
        ),
    }
    seed_payloads: dict[int, dict] = {}
    masks: dict[int, np.ndarray] = {}
    num_train = int(len(complete) * 0.7)
    num_test = int(len(complete) * 0.2)
    split_boundaries = (num_train, len(complete) - num_test)

    for seed in SEEDS:
        missing = ROOT / (
            f"data/traffic/traffic_point_external_r0.06_l5_seed{seed}.csv"
        )
        mask_path = MASK_DIRECTORY / f"seed_{seed}.npz"
        metadata_path = mask_path.with_suffix(".json")
        entry_name = f"traffic_point_r006_l5_seed{seed}_external"
        required = (missing, mask_path, metadata_path)
        checks[f"seed{seed}_files_exist"] = all(path.is_file() for path in required)
        if not checks[f"seed{seed}_files_exist"]:
            continue

        degraded = pd.read_csv(missing)
        degraded_values = degraded.iloc[:, 1:].to_numpy(dtype=np.float64)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(mask_path) as loaded:
            payload = {key: loaded[key].copy() for key in loaded.files}
        row_mask = payload["row_mask"].astype(bool, copy=False)
        starts = payload["start_indices"].astype(np.int64, copy=False)
        regenerated_mask, regenerated_starts, target_rows = select_random_blocks(
            n_rows=len(complete), block_length=5, target_rate=0.26, seed=seed
        )
        nan_matrix = np.isnan(degraded_values)
        observed = ~nan_matrix
        run_lengths = np.asarray(contiguous_runs(row_mask), dtype=np.int64)
        entry = manifest.get(entry_name, {})
        output_hash = sha256(missing)
        mask_hash = sha256(mask_path)

        checks.update({
            f"seed{seed}_shape_columns_dates": (
                degraded.shape == complete.shape
                and list(degraded.columns) == list(complete.columns)
                and degraded["date"].equals(complete["date"])
            ),
            f"seed{seed}_mask_keys_scalars": (
                set(payload) == {
                    "row_mask", "start_indices", "block_length", "seed",
                    "nominal_start_rate", "target_realized_rate",
                }
                and row_mask.shape == (17544,)
                and payload["row_mask"].dtype == np.bool_
                and int(payload["block_length"]) == 5
                and int(payload["seed"]) == seed
                and float(payload["nominal_start_rate"]) == 0.06
                and float(payload["target_realized_rate"]) == 0.26
            ),
            f"seed{seed}_shared_missing_and_observed_values": (
                bool(np.all(nan_matrix == row_mask[:, None]))
                and bool(np.allclose(
                    degraded_values[observed], complete_values[observed],
                    rtol=1e-12, atol=1e-12,
                ))
            ),
            f"seed{seed}_deterministic_roundtrip": (
                bool(np.array_equal(row_mask, regenerated_mask))
                and bool(np.array_equal(starts, regenerated_starts))
            ),
            f"seed{seed}_rate_runs_and_split_coverage": (
                int(target_rows) == 4561
                and abs(float(row_mask.mean()) - 0.26) < 5e-4
                and run_lengths.size > 0
                and int(run_lengths.min()) >= 5
                and all(
                    bool(part.any() and (~part).any())
                    for part in np.split(row_mask, split_boundaries)
                )
            ),
            f"seed{seed}_metadata_matches": (
                metadata["source_sha256"] == SOURCE_SHA256
                and metadata["output_sha256"] == output_hash
                and metadata["mask_sha256"] == mask_hash
                and metadata["fixed_generation_seed"] == seed
                and metadata["rows"] == 17544
                and metadata["value_columns"] == 862
                and metadata["realized_missing_rows"] == int(row_mask.sum())
                and np.isclose(metadata["realized_missing_rate"], row_mask.mean())
            ),
            f"seed{seed}_manifest_matches": (
                entry.get("sha256") == output_hash
                and entry.get("mask_sha256") == mask_hash
                and entry.get("shape") == [17544, 863]
                and entry.get("fixed_generation_seed") == seed
                and entry.get("protocol_match")
                == "fixed_fair_external_protocol_official_traffic_hparams_unavailable"
            ),
        })
        masks[seed] = row_mask
        seed_payloads[seed] = {
            "realized_missing_rows": int(row_mask.sum()),
            "realized_missing_rate": float(row_mask.mean()),
            "selected_start_count": int(starts.size),
            "contiguous_run_count": int(run_lengths.size),
            "contiguous_run_min": int(run_lengths.min()),
            "contiguous_run_max": int(run_lengths.max()),
            "output_sha256": output_hash,
            "mask_sha256": mask_hash,
        }

    if set(masks) == set(SEEDS):
        for left_index, left in enumerate(SEEDS):
            for right in SEEDS[left_index + 1:]:
                checks[f"seed{left}_{right}_masks_differ_full"] = not bool(
                    np.array_equal(masks[left], masks[right])
                )
                checks[f"seed{left}_{right}_masks_differ_validation"] = not bool(
                    np.array_equal(
                        masks[left][split_boundaries[0]:split_boundaries[1]],
                        masks[right][split_boundaries[0]:split_boundaries[1]],
                    )
                )
                checks[f"seed{left}_{right}_masks_differ_test"] = not bool(
                    np.array_equal(
                        masks[left][split_boundaries[1]:],
                        masks[right][split_boundaries[1]:],
                    )
                )

    checks = {name: bool(passed) for name, passed in checks.items()}
    failures = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "seed_assets": seed_payloads,
        "split_rows": {
            "train": num_train,
            "validation": len(complete) - num_train - num_test,
            "test": num_test,
        },
        "validation_windows_batches_samples": [1661, 103, 1648],
        "test_windows_batches_samples": [3413, 213, 3408],
        "model_test_split_accessed": False,
    }
    atomic_json(REPORT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
