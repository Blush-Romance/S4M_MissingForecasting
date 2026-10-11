"""Evaluate a saved checkpoint on the test split (e.g. an interim best.pt).

    python -m mtsf.eval_ckpt runs_mtsf/<run>/best_copy.pt runs_mtsf/<run>/config.yaml [--device cpu]
"""
from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import torch
import yaml

from .data import MissingForecastData, make_loaders
from .models import build_model
from .train import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("config")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    cfg = SimpleNamespace(**yaml.safe_load(open(a.config)))
    torch.set_num_threads(4)
    data = MissingForecastData(cfg.data_root, cfg.dataset, cfg.seq_len, cfg.pred_len, cfg.pattern, cfg.rate,
                               cfg.block, cfg.mask_seed, cfg.mask_cache, getattr(cfg, "mask_file", None) or None)
    cfg.n_vars = data.n_vars
    loaders = make_loaders(data, cfg.batch_size, 0, a.batch)
    model = build_model(cfg)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    dev = torch.device(a.device)
    model.to(dev)
    _, mse, mae = evaluate(model, loaders["test"], dev, None)
    print(json.dumps({"ckpt": a.ckpt, "test_mse": mse, "test_mae": mae}))


if __name__ == "__main__":
    main()
