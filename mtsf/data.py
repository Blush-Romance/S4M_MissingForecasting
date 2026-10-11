"""Data pipeline that follows the S4M (ICLR'25) evaluation protocol.

Protocol (taken from the S4M paper and its public `Dataset_Custom4` loader):
  * block missingness is injected into the *whole* series before splitting;
    - "point":    each time step starts a block with prob r; the next `block`
                  steps are removed for all variables;
    - "variable": the same procedure, independently per variable;
  * chronological 0.7 / 0.1 / 0.2 split, val/test windows start `seq_len` steps
    before their border;
  * z-score with statistics of the (partially observed) training part;
  * missing inputs are filled with 0 (= training mean after z-scoring);
  * training / validation loss only uses observed future targets;
  * test metrics are computed against the complete ground-truth future.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

DATA_FILES = {
    "etth1": "ETTh1.csv",
    "ettm1": "ETTm1.csv",
    "weather": "weather.csv",
    "electricity": "electricity.csv",
    "traffic": "traffic.csv",
}


def make_mask(shape: tuple[int, int], pattern: str, rate: float, block: int, seed: int) -> np.ndarray:
    """Return an observation mask (1 = observed) with block missingness."""
    t, d = shape
    rng = np.random.default_rng(seed)
    if pattern == "point":
        starts = rng.random(t) < rate
        miss = np.zeros(t, dtype=bool)
        for s in np.flatnonzero(starts):
            miss[s : s + block] = True
        miss = np.repeat(miss[:, None], d, axis=1)
    elif pattern == "variable":
        starts = rng.random((t, d)) < rate
        miss = np.zeros((t, d), dtype=bool)
        for k in range(block):
            miss[k:] |= starts[: t - k]
    else:
        raise ValueError(pattern)
    return (~miss).astype(np.float32)


@dataclass
class Split:
    x: np.ndarray      # zero-filled normalised observed values  [T, D]
    m: np.ndarray      # observation mask                       [T, D]
    y_true: np.ndarray  # complete normalised ground truth       [T, D]
    t: np.ndarray      # time features                          [T, F]


def time_features(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    feats = [
        d.dt.minute / 59.0 - 0.5,
        d.dt.hour / 23.0 - 0.5,
        d.dt.dayofweek / 6.0 - 0.5,
        (d.dt.day - 1) / 30.0 - 0.5,
        (d.dt.dayofyear - 1) / 365.0 - 0.5,
    ]
    return np.stack([f.values for f in feats], axis=1).astype(np.float32)


class MissingForecastData:
    """Loads a dataset, injects missingness and builds the three splits."""

    def __init__(self, root: str, dataset: str, seq_len: int, pred_len: int,
                 pattern: str = "point", rate: float = 0.06, block: int = 5,
                 mask_seed: int = 2024, cache_dir: str | None = None):
        path = os.path.join(root, DATA_FILES[dataset])
        df = pd.read_csv(path)
        cols = [c for c in df.columns if c != "date"]
        # S4M moves the target ('OT') to the last column; order does not affect metrics.
        values = df[cols].values.astype(np.float32)
        tfeat = time_features(df["date"])
        T, D = values.shape
        self.n_vars = D
        self.seq_len, self.pred_len = seq_len, pred_len

        mask = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            mpath = os.path.join(cache_dir, f"{dataset}_{pattern}_r{rate}_b{block}_s{mask_seed}.npy")
            if os.path.exists(mpath):
                mask = np.load(mpath)
        if mask is None:
            mask = make_mask((T, D), pattern, rate, block, mask_seed)
            if cache_dir:
                np.save(mpath, mask)
        self.mask_sha = hashlib.sha256(mask.tobytes()).hexdigest()[:16]
        self.realized_missing = float(1 - mask.mean())

        num_train = int(T * 0.7)
        num_test = int(T * 0.2)
        num_val = T - num_train - num_test
        b1 = [0, num_train - seq_len, T - num_test - seq_len]
        b2 = [num_train, num_train + num_val, T]

        obs = np.where(mask > 0, values, np.nan)
        mean = np.nanmean(obs[: num_train], axis=0)
        std = np.nanstd(obs[: num_train], axis=0)
        std[std < 1e-8] = 1.0
        self.mean, self.std = mean, std
        norm_true = (values - mean) / std
        norm_obs = np.where(mask > 0, norm_true, 0.0).astype(np.float32)

        self.splits = {}
        for name, i in zip(["train", "val", "test"], range(3)):
            s, e = b1[i], b2[i]
            self.splits[name] = Split(norm_obs[s:e], mask[s:e], norm_true[s:e].astype(np.float32), tfeat[s:e])


class WindowDataset(Dataset):
    def __init__(self, split: Split, seq_len: int, pred_len: int):
        self.s, self.L, self.H = split, seq_len, pred_len
        self.n = len(split.x) - seq_len - pred_len + 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s, L, H = self.s, self.L, self.H
        return (
            torch.from_numpy(s.x[i : i + L]),
            torch.from_numpy(s.m[i : i + L]),
            torch.from_numpy(s.x[i + L : i + L + H]),
            torch.from_numpy(s.m[i + L : i + L + H]),
            torch.from_numpy(s.y_true[i + L : i + L + H]),
            torch.from_numpy(s.t[i : i + L]),
            torch.from_numpy(s.t[i + L : i + L + H]),
        )


def make_loaders(data: MissingForecastData, batch_size: int, num_workers: int = 2,
                 eval_batch_size: int | None = None):
    eb = eval_batch_size or batch_size
    out = {}
    for name in ["train", "val", "test"]:
        ds = WindowDataset(data.splits[name], data.seq_len, data.pred_len)
        train = name == "train"
        out[name] = DataLoader(ds, batch_size=batch_size if train else eb, shuffle=train,
                               drop_last=train, num_workers=num_workers, pin_memory=True,
                               persistent_workers=num_workers > 0)
    return out
