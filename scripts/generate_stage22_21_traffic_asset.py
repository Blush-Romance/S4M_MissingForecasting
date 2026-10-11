#!/usr/bin/env python3
"""Generate the preregistered Step 22.21 Traffic missing-data asset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from generate_paper_point_masks import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_npz,
    contiguous_runs,
    select_random_blocks,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/traffic/traffic.csv"
OUTPUT = ROOT / "data/traffic/traffic_point_external_r0.06_l5_seed2026.csv"
MASK_OUTPUT = ROOT / "artifacts/masks/traffic/point/rate_0.06/seed_2026.npz"
METADATA_OUTPUT = ROOT / "artifacts/masks/traffic/point/rate_0.06/seed_2026.json"
EXPECTED_SOURCE_SHA256 = (
    "cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16"
)
SEED = 2026
BLOCK_LENGTH = 5
TARGET_REALIZED_RATE = 0.26


def main() -> int:
    if sha256_file(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Traffic complete asset hash changed")
    source = pd.read_csv(SOURCE)
    if source.shape != (17544, 863):
        raise ValueError(f"unexpected Traffic shape: {source.shape}")
    if source.columns[0] != "date" or source.iloc[:, 1:].isna().any().any():
        raise ValueError("Traffic source must have a complete date+862 schema")
    if source["date"].duplicated().any():
        raise ValueError("Traffic source contains duplicate timestamps")

    row_mask, start_indices, target_rows = select_random_blocks(
        n_rows=len(source),
        block_length=BLOCK_LENGTH,
        target_rate=TARGET_REALIZED_RATE,
        seed=SEED,
    )
    missing = source.copy()
    missing.loc[row_mask, source.columns[1:]] = np.nan
    atomic_write_csv(missing, OUTPUT)
    atomic_write_npz(
        MASK_OUTPUT,
        row_mask=row_mask,
        start_indices=start_indices,
        block_length=np.asarray(BLOCK_LENGTH, dtype=np.int64),
        seed=np.asarray(SEED, dtype=np.int64),
        nominal_start_rate=np.asarray(0.06, dtype=np.float64),
        target_realized_rate=np.asarray(TARGET_REALIZED_RATE, dtype=np.float64),
    )

    run_lengths = contiguous_runs(row_mask)
    metadata = {
        "dataset": "Traffic",
        "protocol_kind": "fixed_fair_external_protocol_v1",
        "protocol_match": "official_traffic_mask_and_hparams_unavailable",
        "paper_nominal_start_rate": 0.06,
        "block_length": BLOCK_LENGTH,
        "target_realized_rate": TARGET_REALIZED_RATE,
        "fixed_generation_seed": SEED,
        "rows": int(len(source)),
        "value_columns": int(source.shape[1] - 1),
        "target_missing_rows": int(target_rows),
        "realized_missing_rows": int(row_mask.sum()),
        "realized_missing_rate": float(row_mask.mean()),
        "selected_start_count": int(start_indices.size),
        "effective_selected_start_rate": float(start_indices.size / len(source)),
        "contiguous_run_count": int(len(run_lengths)),
        "contiguous_run_min": int(min(run_lengths)),
        "contiguous_run_max": int(max(run_lengths)),
        "source_path": str(SOURCE),
        "output_path": str(OUTPUT),
        "mask_path": str(MASK_OUTPUT),
        "source_sha256": sha256_file(SOURCE),
        "output_sha256": sha256_file(OUTPUT),
        "mask_sha256": sha256_file(MASK_OUTPUT),
        "generator": str(Path(__file__).resolve()),
        "protocol_document": str(
            ROOT / "docs/STAGE22_21_TRAFFIC_EXTERNAL_PROTOCOL_ZH.md"
        ),
        "generator_note": (
            "Traffic was not used for method selection. The public S4M code does "
            "not disclose its Traffic mask seed or final hyperparameters; this asset "
            "uses the preregistered seed2026, length-5 blocks and 0.26 target row rate."
        ),
    }
    atomic_write_json(metadata, METADATA_OUTPUT)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
