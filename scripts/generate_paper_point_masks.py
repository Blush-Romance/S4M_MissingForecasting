#!/usr/bin/env python3
"""Generate fixed point-missing CSVs and masks for S4M experiments.

The S4M paper describes sampling start points and masking the following five
time steps, but the public repository does not publish the generation seed or
the ETTh1 missing CSV.  This script therefore uses a fixed seed and chooses the
prefix of a random start-point permutation whose realised missing-row ratio is
closest to the dataset-specific ratio reported in Table 8 of the paper.

The resulting assets are deliberately labelled as a protocol approximation,
not as the unpublished official masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: Path
    output: Path
    mask_output: Path
    metadata_output: Path
    paper_realized_rate: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contiguous_runs(mask: np.ndarray) -> list[int]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return (ends - starts).astype(int).tolist()


def select_random_blocks(
    n_rows: int,
    block_length: int,
    target_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if n_rows < block_length:
        raise ValueError(f"n_rows={n_rows} is shorter than block_length={block_length}")

    target_rows = int(round(n_rows * target_rate))
    rng = np.random.default_rng(seed)
    candidates = rng.permutation(n_rows - block_length + 1)

    mask = np.zeros(n_rows, dtype=bool)
    best_count = 0
    best_k = 0
    best_error = abs(target_rows)

    for k, start in enumerate(candidates, start=1):
        mask[start : start + block_length] = True
        count = int(mask.sum())
        error = abs(count - target_rows)
        if error < best_error:
            best_error = error
            best_count = count
            best_k = k
        if count >= target_rows + block_length:
            break

    starts = candidates[:best_k].astype(np.int64)
    best_mask = np.zeros(n_rows, dtype=bool)
    for start in starts:
        best_mask[start : start + block_length] = True
    if int(best_mask.sum()) != best_count:
        raise RuntimeError("Internal error while rebuilding the selected point mask")
    return best_mask, starts, target_rows


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, float_format="%.17g")
    os.replace(temp, path)


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp, path)


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def generate(spec: DatasetSpec, seed: int, block_length: int) -> dict:
    source = pd.read_csv(spec.source)
    if "date" not in source.columns:
        raise ValueError(f"{spec.source} does not contain the required date column")
    value_columns = [column for column in source.columns if column != "date"]
    if not value_columns:
        raise ValueError(f"{spec.source} has no value columns")
    if source[value_columns].isna().any().any():
        raise ValueError(f"Complete source unexpectedly contains NaNs: {spec.source}")

    row_mask, start_indices, target_rows = select_random_blocks(
        len(source), block_length, spec.paper_realized_rate, seed
    )
    missing = source.copy()
    missing.loc[row_mask, value_columns] = np.nan
    atomic_write_csv(missing, spec.output)

    realized_rows = int(row_mask.sum())
    run_lengths = contiguous_runs(row_mask)
    atomic_write_npz(
        spec.mask_output,
        row_mask=row_mask,
        start_indices=start_indices,
        block_length=np.asarray(block_length, dtype=np.int64),
        seed=np.asarray(seed, dtype=np.int64),
        nominal_start_rate=np.asarray(0.06, dtype=np.float64),
        paper_realized_rate=np.asarray(spec.paper_realized_rate, dtype=np.float64),
    )

    metadata = {
        "dataset": spec.name,
        "protocol_kind": "paper_reported_realized_rate_calibrated_v1",
        "protocol_match": "approximate_missing_mask_seed_unavailable",
        "paper_nominal_start_rate": 0.06,
        "paper_block_length": block_length,
        "paper_reported_realized_rate": spec.paper_realized_rate,
        "fixed_generation_seed": seed,
        "rows": len(source),
        "value_columns": len(value_columns),
        "target_missing_rows": target_rows,
        "realized_missing_rows": realized_rows,
        "realized_missing_rate": realized_rows / len(source),
        "selected_start_count": int(len(start_indices)),
        "effective_selected_start_rate": len(start_indices) / len(source),
        "contiguous_run_count": len(run_lengths),
        "contiguous_run_min": min(run_lengths),
        "contiguous_run_max": max(run_lengths),
        "source_path": str(spec.source),
        "output_path": str(spec.output),
        "mask_path": str(spec.mask_output),
        "source_sha256": sha256_file(spec.source),
        "output_sha256": sha256_file(spec.output),
        "mask_sha256": sha256_file(spec.mask_output),
        "generator": str(Path(__file__).resolve()),
        "generator_note": (
            "The paper and public code do not publish the original random seed/mask. "
            "This fixed asset preserves length-5 point blocks and matches the paper's "
            "reported realized missing ratio as closely as whole blocks allow."
        ),
    }
    atomic_write_json(metadata, spec.metadata_output)
    return metadata


def specs(project_root: Path, seed: int) -> dict[str, DatasetSpec]:
    return {
        "weather": DatasetSpec(
            name="Weather",
            source=project_root / "data/weather/weather.csv",
            output=project_root
            / f"data/weather/weather_point_paper_r0.06_l5_seed{seed}.csv",
            mask_output=project_root
            / f"artifacts/masks/weather/point/rate_0.06/seed_{seed}.npz",
            metadata_output=project_root
            / f"artifacts/masks/weather/point/rate_0.06/seed_{seed}.json",
            paper_realized_rate=0.247,
        ),
        "etth1": DatasetSpec(
            name="ETTh1",
            source=project_root / "data/ETT-small/ETTh1.csv",
            output=project_root
            / f"data/ETT-small/ETTh1_point_paper_r0.06_l5_seed{seed}.csv",
            mask_output=project_root
            / f"artifacts/masks/etth1/point/rate_0.06/seed_{seed}.npz",
            metadata_output=project_root
            / f"artifacts/masks/etth1/point/rate_0.06/seed_{seed}.json",
            paper_realized_rate=0.231,
        ),
        "electricity": DatasetSpec(
            name="Electricity",
            source=project_root / "data/electricity/electricity.csv",
            output=project_root
            / f"data/electricity/electricity_point_paper_r0.06_l5_seed{seed}.csv",
            mask_output=project_root
            / f"artifacts/masks/electricity/point/rate_0.06/seed_{seed}.npz",
            metadata_output=project_root
            / f"artifacts/masks/electricity/point/rate_0.06/seed_{seed}.json",
            paper_realized_rate=0.260,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", "weather", "etth1", "electricity"),
        default="all",
        help="Asset(s) to generate.",
    )
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected: Iterable[str]
    selected = (
        ("weather", "etth1", "electricity")
        if args.dataset == "all"
        else (args.dataset,)
    )
    definitions = specs(args.project_root.resolve(), args.seed)
    outputs = [generate(definitions[name], args.seed, args.block_length) for name in selected]
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
