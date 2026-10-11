"""Training / evaluation loop shared by every model."""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from .data import MissingForecastData, make_loaders
from .models import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mse(pred, target, mask):
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)


def default_loss(model, batch):
    x, m, y, my = batch[:4]
    pred = model(x, m, batch[5], batch[6])
    loss = masked_mse(pred, y, my)
    return loss, {"fore": loss.item()}


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    """Returns (val_loss on observed targets, mse/mae vs complete truth)."""
    model.eval()
    se_obs = n_obs = 0.0
    se = ae = n = 0.0
    for batch in loader:
        batch = [b.to(device, non_blocking=True) for b in batch]
        x, m, y, my, yt, tx, ty = batch
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            pred = model(x, m, tx, ty)
        pred = pred.float()
        se_obs += (((pred - y) ** 2) * my).sum().item()
        n_obs += my.sum().item()
        d = pred - yt
        se += (d ** 2).sum().item()
        ae += d.abs().sum().item()
        n += d.numel()
    return se_obs / max(n_obs, 1), se / n, ae / n


def run(cfg: SimpleNamespace) -> dict:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = MissingForecastData(cfg.data_root, cfg.dataset, cfg.seq_len, cfg.pred_len,
                               cfg.pattern, cfg.rate, cfg.block, cfg.mask_seed, cfg.mask_cache,
                               getattr(cfg, "mask_file", None) or None)
    cfg.n_vars = data.n_vars
    loaders = make_loaders(data, cfg.batch_size, cfg.num_workers, cfg.eval_batch_size)
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(loaders["train"])
    total = cfg.epochs * steps_per_epoch
    warm = int(cfg.warmup_frac * total)

    def lr_lambda(step):
        if cfg.schedule == "constant":
            return 1.0
        if step < warm:
            return (step + 1) / max(warm, 1)
        p = (step - warm) / max(total - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0))) * 0.99 + 0.01

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    amp_dtype = torch.bfloat16 if (cfg.amp and device.type == "cuda") else None

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(vars(cfg), f, sort_keys=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    log = open(os.path.join(cfg.out_dir, "train.log"), "a")

    def p(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    p(f"params={n_params} realized_missing={data.realized_missing:.4f} mask_sha={data.mask_sha} commit={commit}")
    best, bad, best_ep = float("inf"), 0, -1
    ckpt = os.path.join(cfg.out_dir, "best.pt")
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    loss_fn = model.compute_loss if hasattr(model, "compute_loss") else (lambda b: default_loss(model, b))
    for ep in range(cfg.epochs):
        model.train()
        te = time.time()
        acc = {}
        for it, batch in enumerate(loaders["train"]):
            batch = [b.to(device, non_blocking=True) for b in batch]
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss, logs = loss_fn(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            sched.step()
            for k, v in logs.items():
                acc[k] = acc.get(k, 0.0) + v
        val, vmse, vmae = evaluate(model, loaders["val"], device, amp_dtype)
        msg = " ".join(f"{k}={v / steps_per_epoch:.4f}" for k, v in acc.items())
        p(f"ep {ep + 1} {msg} val_obs={val:.5f} val_true_mse={vmse:.5f} time={time.time() - te:.1f}s")
        if val < best - 1e-7:
            best, bad, best_ep = val, 0, ep + 1
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= cfg.patience:
                p("early stop")
                break
    train_time = time.time() - t0
    model.load_state_dict(torch.load(ckpt, map_location=device))
    t1 = time.time()
    _, mse, mae = evaluate(model, loaders["test"], device, amp_dtype)
    infer = time.time() - t1
    res = {
        "model": cfg.model, "dataset": cfg.dataset, "pattern": cfg.pattern, "rate": cfg.rate,
        "seq_len": cfg.seq_len, "pred_len": cfg.pred_len, "seed": cfg.seed, "mask_seed": cfg.mask_seed,
        "mse": mse, "mae": mae, "best_val": best, "best_epoch": best_ep,
        "params": n_params, "train_time_s": train_time, "test_time_s": infer,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
        "realized_missing": data.realized_missing, "mask_sha": data.mask_sha, "commit": commit,
        "tag": cfg.tag,
    }
    p(f"TEST mse={mse:.5f} mae={mae:.5f}")
    with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    if not cfg.keep_ckpt:
        os.remove(ckpt)
    log.close()
    return res
