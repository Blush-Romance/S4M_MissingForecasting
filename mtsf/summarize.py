"""Aggregate results_mtsf/*.json into markdown tables.

    python -m mtsf.summarize [--prefix main] [--out reports_mtsf/main.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from .paper_numbers import PAPER, best_reported


def load(prefix):
    rows = []
    for f in glob.glob("results_mtsf/*.json"):
        r = json.load(open(f))
        name = os.path.basename(f)[:-5]
        if prefix and not name.startswith(prefix):
            continue
        r["name"] = name
        rows.append(r)
    return rows


def model_key(r):
    # name convention: <prefix>__<variant>__...
    parts = r["name"].split("__")
    return parts[1] if len(parts) > 1 else r["model"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = load(a.prefix)
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (r["dataset"], r["pattern"], float(r["rate"]), int(r["seq_len"]))
        groups[k][model_key(r)].append((r["mse"], r["mae"]))
    lines = []
    for k in sorted(groups):
        ds, pat, rate, L = k
        lines.append(f"\n### {ds} | {pat} | r={rate} | L={L}\n")
        lines.append("| model | MSE | MAE | n |\n|---|---|---|---|")
        for m, v in sorted(groups[k].items(), key=lambda kv: np.mean([x[0] for x in kv[1]])):
            v = np.array(v)
            sd = f" ± {v[:, 0].std():.4f}" if len(v) > 1 else ""
            sda = f" ± {v[:, 1].std():.4f}" if len(v) > 1 else ""
            lines.append(f"| {m} | {v[:, 0].mean():.4f}{sd} | {v[:, 1].mean():.4f}{sda} | {len(v)} |")
        if abs(rate - 0.06) < 1e-9 and L in PAPER.get(pat, {}).get(ds, {}):
            s4m = PAPER[pat][ds][L]["S4M"]
            b_mse, b_m = best_reported(pat, ds, L, 0)
            b_mae, b_a = best_reported(pat, ds, L, 1)
            lines.append(f"| *S4M (paper)* | {s4m[0]:.3f} | {s4m[1]:.3f} | – |")
            lines.append(f"| *best in S4M paper* | {b_mse:.3f} ({b_m}) | {b_mae:.3f} ({b_a}) | – |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        open(a.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
