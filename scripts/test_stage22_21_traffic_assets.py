#!/usr/bin/env python3
"""Audit every frozen Step-22.21 Traffic asset before GPU work."""

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
MISSING = ROOT / "data/traffic/traffic_point_external_r0.06_l5_seed2026.csv"
MASK = ROOT / "artifacts/masks/traffic/point/rate_0.06/seed_2026.npz"
METADATA = MASK.with_suffix(".json")
MANIFEST = ROOT / "artifacts/manifests/s4m_data_assets.yaml"
PROTOCOL = ROOT / "docs/STAGE22_21_TRAFFIC_EXTERNAL_PROTOCOL_ZH.md"
REPORT = ROOT / "reports/stage22_21_traffic_asset_test.json"
SOURCE_SHA256 = "cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16"
OUTPUT_SHA256 = "03912ad8673901220283118d955b3f4a65c7e8b52d8daf2665812c5bd16d5924"
MASK_SHA256 = "92bcc7576c2848484a7059693d54e4c09643208cd1c6314941c1708f13ca96d5"


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
    required = (COMPLETE, MISSING, MASK, METADATA, MANIFEST, PROTOCOL)
    absent = [str(path) for path in required if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"missing Step-22.21 assets: {absent}")

    complete = pd.read_csv(COMPLETE)
    degraded = pd.read_csv(MISSING)
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    with np.load(MASK) as loaded:
        mask_payload = {key: loaded[key].copy() for key in loaded.files}
    row_mask = mask_payload["row_mask"].astype(bool, copy=False)
    starts = mask_payload["start_indices"].astype(np.int64, copy=False)
    value_columns = list(complete.columns[1:])
    complete_values = complete[value_columns].to_numpy(dtype=np.float64)
    degraded_values = degraded[value_columns].to_numpy(dtype=np.float64)
    nan_matrix = np.isnan(degraded_values)

    regenerated_mask, regenerated_starts, target_rows = select_random_blocks(
        n_rows=len(complete), block_length=5, target_rate=0.26, seed=2026
    )
    run_lengths = np.asarray(contiguous_runs(row_mask), dtype=np.int64)

    num_train = int(len(complete) * 0.7)
    num_test = int(len(complete) * 0.2)
    num_validation = len(complete) - num_train - num_test
    partitions = {
        "train": row_mask[:num_train],
        "validation": row_mask[num_train:num_train + num_validation],
        "test": row_mask[num_train + num_validation:],
    }
    validation_windows = num_validation - 96 + 1
    test_windows = num_test - 96 + 1
    manifest_entry = manifest["assets"][
        "traffic_point_r006_l5_seed2026_external"
    ]
    protocol_text = PROTOCOL.read_text(encoding="utf-8")

    observed = ~nan_matrix
    observed_error = float(
        np.max(np.abs(degraded_values[observed] - complete_values[observed]))
    )
    checks = {
        "complete_shape_17544_by_863": complete.shape == (17544, 863),
        "complete_schema_date_plus_862": (
            complete.columns[0] == "date" and len(value_columns) == 862
        ),
        "traffic_target_column_is_final_ot": complete.columns[-1] == "OT",
        "complete_has_no_nan": not bool(np.isnan(complete_values).any()),
        "missing_shape_and_columns_match": (
            degraded.shape == complete.shape
            and list(degraded.columns) == list(complete.columns)
        ),
        "dates_match_exactly": degraded["date"].equals(complete["date"]),
        "mask_keys_exact": set(mask_payload) == {
            "row_mask", "start_indices", "block_length", "seed",
            "nominal_start_rate", "target_realized_rate",
        },
        "mask_shape_and_dtype": (
            row_mask.shape == (17544,)
            and mask_payload["row_mask"].dtype == np.bool_
        ),
        "frozen_mask_scalars": (
            int(mask_payload["block_length"]) == 5
            and int(mask_payload["seed"]) == 2026
            and float(mask_payload["nominal_start_rate"]) == 0.06
            and float(mask_payload["target_realized_rate"]) == 0.26
        ),
        "row_shared_missing_mask": bool(np.all(nan_matrix == row_mask[:, None])),
        "observed_values_preserved": bool(np.allclose(
            degraded_values[observed], complete_values[observed],
            rtol=1e-12, atol=1e-12,
        )),
        "deterministic_row_mask_roundtrip": bool(
            np.array_equal(row_mask, regenerated_mask)
        ),
        "deterministic_start_indices_roundtrip": bool(
            np.array_equal(starts, regenerated_starts)
        ),
        "nearest_reachable_target_count": (
            int(target_rows) == 4561 and int(row_mask.sum()) == 4559
        ),
        "all_missing_runs_at_least_l5": bool(
            run_lengths.size > 0 and int(run_lengths.min()) >= 5
        ),
        "all_partitions_have_observed_and_missing_rows": all(
            bool(part.any() and (~part).any()) for part in partitions.values()
        ),
        "expected_validation_windows_batches_samples": (
            validation_windows == 1661
            and validation_windows // 16 == 103
            and (validation_windows // 16) * 16 == 1648
        ),
        "expected_test_windows_batches_samples": (
            test_windows == 3413
            and test_windows // 16 == 213
            and (test_windows // 16) * 16 == 3408
        ),
        "fixed_hashes": (
            sha256(COMPLETE) == SOURCE_SHA256
            and sha256(MISSING) == OUTPUT_SHA256
            and sha256(MASK) == MASK_SHA256
        ),
        "metadata_hashes_and_schema_match": (
            metadata["source_sha256"] == SOURCE_SHA256
            and metadata["output_sha256"] == OUTPUT_SHA256
            and metadata["mask_sha256"] == MASK_SHA256
            and metadata["rows"] == 17544
            and metadata["value_columns"] == 862
            and metadata["fixed_generation_seed"] == 2026
            and metadata["block_length"] == 5
            and np.isclose(metadata["realized_missing_rate"], row_mask.mean())
        ),
        "manifest_matches_frozen_assets": (
            manifest_entry["sha256"] == OUTPUT_SHA256
            and manifest_entry["mask_sha256"] == MASK_SHA256
            and manifest_entry["shape"] == [17544, 863]
            and manifest_entry["fixed_generation_seed"] == 2026
            and manifest_entry["protocol_match"]
            == "fixed_fair_external_protocol_official_traffic_hparams_unavailable"
        ),
        "protocol_link_and_post_smoke_amendment_are_auditable": (
            Path(metadata["protocol_document"]).resolve() == PROTOCOL.resolve()
            and "首次smoke后的实现性修订（无模型结果）" in protocol_text
            and "不改变变量数量、列顺序、CSV、mask、数据划分、模型、训练预算、多尺度权重或科学门"
            in protocol_text
            and "两个失败run永久保留为`CODE_ERROR`" in protocol_text
        ),
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    failures = [name for name, passed in checks.items() if not passed]
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "shape": list(complete.shape),
        "split_rows": {
            "train": num_train,
            "validation": num_validation,
            "test": num_test,
        },
        "validation_windows_batches_samples": [1661, 103, 1648],
        "test_windows_batches_samples": [3413, 213, 3408],
        "realized_missing_rows": int(row_mask.sum()),
        "realized_missing_rate": float(row_mask.mean()),
        "contiguous_run_count": int(run_lengths.size),
        "contiguous_run_min": int(run_lengths.min()),
        "contiguous_run_max": int(run_lengths.max()),
        "observed_max_abs_error": observed_error,
        "source_sha256": sha256(COMPLETE),
        "output_sha256": sha256(MISSING),
        "mask_sha256": sha256(MASK),
        "model_test_split_accessed": False,
    }
    atomic_json(REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
