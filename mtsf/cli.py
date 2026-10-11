"""python -m mtsf.cli --config configs_mtsf/base.yaml --set model=mars dataset=etth1 ..."""
from __future__ import annotations

import argparse
import os
import time
from types import SimpleNamespace

import yaml


def parse_value(v: str):
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


def load_cfg(paths, overrides):
    cfg = {}
    for p in paths:
        with open(p) as f:
            cfg.update(yaml.safe_load(f) or {})
    for kv in overrides:
        k, v = kv.split("=", 1)
        cfg[k] = parse_value(v)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["configs_mtsf/base.yaml"])
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.set)
    if not cfg.get("out_dir"):
        name = f"{cfg['model']}__{cfg['dataset']}__{cfg['pattern']}_r{cfg['rate']}__L{cfg['seq_len']}__s{cfg['seed']}"
        if cfg.get("tag"):
            name += f"__{cfg['tag']}"
        cfg["out_dir"] = os.path.join(cfg.get("runs_root", "runs_mtsf"), name + time.strftime("__%m%d_%H%M%S"))
    from .train import run
    run(SimpleNamespace(**cfg))


if __name__ == "__main__":
    main()
