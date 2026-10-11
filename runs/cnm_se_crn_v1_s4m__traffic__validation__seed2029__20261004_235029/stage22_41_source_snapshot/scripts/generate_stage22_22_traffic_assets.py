#!/usr/bin/env python3
"""Generate the preregistered Step-22.22 Traffic replication assets."""

from __future__ import annotations

import argparse
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
MASK_DIRECTORY = ROOT / "artifacts/masks/traffic/point/rate_0.06"
EXPECTED_SOURCE_SHA256 = (
    "cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16"
)
ALLOWED_SEEDS = (2027, 2028)
BLOCK_LENGTH = 5
TARGET_REALIZED_RATE = 0.26
NOMINAL_START_RATE = 0.06


def output_paths(seed: int) -> tuple[Path, Path, Path]:
    missing = ROOT / (
        f"data/traffic/traffic_point_external_r0.06_l5_seed{seed}.csv"
    )
    mask = MASK_DIRECTORY / f"seed_{seed}.npz"
    return missing, mask, mask.with_suffix(".json")


def generate(source: pd.DataFrame, seed: int) -> dict:
    output, mask_output, metadata_output = output_paths(seed)
    row_mask, start_indices, target_rows = select_random_blocks(
        n_rows=len(source),
        block_length=BLOCK_LENGTH,
        target_rate=TARGET_REALIZED_RATE,
        seed=seed,
    )
    missing = source.copy()
    missing.loc[row_mask, source.columns[1:]] = np.nan
    atomic_write_csv(missing, output)
    atomic_write_npz(
        mask_output,
        row_mask=row_mask,
        start_indices=start_indices,
        block_length=np.asarray(BLOCK_LENGTH, dtype=np.int64),
        seed=np.asarray(seed, dtype=np.int64),
        nominal_start_rate=np.asarray(NOMINAL_START_RATE, dtype=np.float64),
        target_realized_rate=np.asarray(TARGET_REALIZED_RATE, dtype=np.float64),
    )

    run_lengths = contiguous_runs(row_mask)
    metadata = {
        "dataset": "Traffic",
        "protocol_kind": "fixed_fair_external_protocol_multiseed_v1",
        "protocol_match": (
            "fixed_fair_external_protocol_official_traffic_hparams_unavailable"
        ),
        "paper_nominal_start_rate": NOMINAL_START_RATE,
        "block_length": BLOCK_LENGTH,
        "target_realized_rate": TARGET_REALIZED_RATE,
        "fixed_generation_seed": seed,
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
        "output_path": str(output),
        "mask_path": str(mask_output),
        "source_sha256": sha256_file(SOURCE),
        "output_sha256": sha256_file(output),
        "mask_sha256": sha256_file(mask_output),
        "generator": str(Path(__file__).resolve()),
        "protocol_document": str(
            ROOT / "docs/STAGE22_22_TRAFFIC_MULTISEED_PROTOCOL_ZH.md"
        ),
        "generator_note": (
            "The method, training budget, evaluation and seed set were frozen "
            "before this seed-specific asset was generated."
        ),
    }
    atomic_write_json(metadata, metadata_output)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(ALLOWED_SEEDS)
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = tuple(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty unique list")
    unexpected = sorted(set(seeds) - set(ALLOWED_SEEDS))
    if unexpected:
        raise ValueError(f"non-preregistered Step-22.22 seeds: {unexpected}")
    if sha256_file(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Traffic complete asset hash changed")
    source = pd.read_csv(SOURCE)
    if source.shape != (17544, 863):
        raise ValueError(f"unexpected Traffic shape: {source.shape}")
    if (
        source.columns[0] != "date"
        or source.columns[-1] != "OT"
        or source.iloc[:, 1:].isna().any().any()
        or source["date"].duplicated().any()
    ):
        raise ValueError("Traffic source must have a complete date+862 schema")

    generated = [generate(source, seed) for seed in seeds]
    print(json.dumps({"status": "SUCCESS", "assets": generated}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
