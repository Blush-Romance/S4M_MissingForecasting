#!/usr/bin/env python3
"""Rebuild the exact point masks used by the earlier project version (stage 5-22),
so MARS can be compared with its S4M / PG95 / CNM-SE test results on identical data.

Uses the legacy generator `scripts/generate_paper_point_masks.py::select_random_blocks`
and verifies the rebuilt missing CSV against the SHA-256 in
`artifacts/manifests/s4m_data_assets.yaml`.

Output: data/masks/legacy_<dataset>_s<seed>.npy  (1 = observed, shape [T, D])
"""
from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_paper_point_masks import select_random_blocks  # noqa: E402

SPECS = {  # dataset: (raw file, target realised rate, manifest key pattern)
    "etth1": ("ETTh1.csv", 0.231, "etth1_point_r006_l5_seed{}_paper_calibrated"),
    "weather": ("weather.csv", 0.247, "weather_point_r006_l5_seed{}_paper_calibrated"),
    "traffic": ("traffic.csv", 0.26, "traffic_point_r006_l5_seed{}_external"),
}
JOBS = [("etth1", 2023), ("weather", 2023), ("weather", 2024), ("weather", 2025),
        ("traffic", 2026), ("traffic", 2027), ("traffic", 2028)]


def main():
    manifest = yaml.safe_load(open(ROOT / "artifacts/manifests/s4m_data_assets.yaml"))["assets"]
    out_dir = ROOT / "data/masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds, seed in JOBS:
        fname, rate, key = SPECS[ds]
        src = pd.read_csv(ROOT / "data/raw" / fname)
        cols = [c for c in src.columns if c != "date"]
        row_mask, _, _ = select_random_blocks(len(src), 5, rate, seed)
        missing = src.copy()
        missing.loc[row_mask, cols] = np.nan
        buf = io.StringIO()
        missing.to_csv(buf, index=False, float_format="%.17g")
        sha = hashlib.sha256(buf.getvalue().encode()).hexdigest()
        ref = manifest.get(key.format(seed), {}).get("sha256")
        status = "MATCH" if sha == ref else ("NO_REF" if ref is None else "MISMATCH")
        mask = np.repeat((~row_mask)[:, None], len(cols), axis=1).astype(np.float32)
        np.save(out_dir / f"legacy_{ds}_s{seed}.npy", mask)
        print(f"{ds} seed={seed} realized={row_mask.mean():.6f} csv_sha={sha[:12]} ref={str(ref)[:12]} {status}")


if __name__ == "__main__":
    main()
