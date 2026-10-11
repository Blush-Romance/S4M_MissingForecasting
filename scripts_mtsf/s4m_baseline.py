#!/usr/bin/env python3
"""Run the *official* S4M code on exactly the same masks as mtsf.

  prepare : copy upstream/S4M/s4m -> third_party/s4m and apply a minimal,
            behaviour-preserving patch (paths, wandb/thop/torchstat stubs,
            optimizer switch, metrics dump). The model code is untouched.
  run     : write the NaN-masked csv used by S4M's Dataset_Custom4 and train.

    python scripts_mtsf/s4m_baseline.py prepare
    python scripts_mtsf/s4m_baseline.py run --dataset etth1 --pattern point --rate 0.06 --seq_len 96
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "upstream/S4M/s4m"
DST = ROOT / "third_party/s4m"
sys.path.insert(0, str(ROOT))

# Hyper-parameters from the S4M paper (App. D.3) and the official README command.
HP = {
    "etth1": dict(lr=0.005, d_model=256, en_conv_hidden_size=256, d_ff=256),
    "weather": dict(lr=0.001, d_model=256, en_conv_hidden_size=256, d_ff=256),
    "electricity": dict(lr=0.01, d_model=512, en_conv_hidden_size=512, d_ff=512),
    "traffic": dict(lr=0.01, d_model=512, en_conv_hidden_size=512, d_ff=512),
}
FILE_TAG = {"etth1": "ETTh1", "weather": "weather", "electricity": "electricity", "traffic": "traffic"}

STUBS = {
    "wandb/__init__.py": "def init(*a, **k): pass\ndef log(*a, **k): pass\ndef save(*a, **k): pass\n",
    "thop/__init__.py": "def profile(*a, **k): return 0, 0\n",
    "torchstat/__init__.py": "def stat(*a, **k): pass\n",
    "pytorch_lightning/__init__.py": "",
    "reformer_pytorch/__init__.py": "class LSHSelfAttention:\n    def __init__(self, *a, **k):\n        raise NotImplementedError\n",
    "pytorch_lightning/utilities/__init__.py": "def rank_zero_only(fn):\n    return fn\n",
}


def patch(path: Path, old: str, new: str, count: int = 1):
    text = path.read_text()
    if old not in text:
        raise RuntimeError(f"patch anchor not found in {path}: {old[:60]!r}")
    path.write_text(text.replace(old, new, count if count > 0 else -1))


def prepare():
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    for rel, body in STUBS.items():
        p = DST / "_stubs" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    dl = DST / "data_provider/data_loader.py"
    patch(dl, "df_true = pd.read_csv(true_path)",
          "df_true = pd.read_csv(os.environ.get('S4M_TRUE_PATH', true_path))", count=0)
    exp = DST / "experiments/exp_pretrain1.py"
    patch(exp, "        model_optim = optim.SGD(self.model.parameters(), lr = self.args.learning_rate, momentum=0.9,weight_decay=1e-5)\n",
          "        if os.environ.get('S4M_OPTIM', 'adam') == 'sgd':\n"
          "            model_optim = optim.SGD(self.model.parameters(), lr = self.args.learning_rate, momentum=0.9,weight_decay=1e-5)\n"
          "        else:\n"
          "            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)\n")
    patch(exp, "folder_path = '/home/project/S4M/test_results/' + setting + '/'",
          "folder_path = './test_results/' + setting + '/'")
    patch(exp, "        print('mse:{}, mae:{}'.format(mse, mae))\n",
          "        print('mse:{}, mae:{}'.format(mse, mae))\n"
          "        _P = np.concatenate(preds, 0); _T = np.concatenate(trues, 0)\n"
          "        if os.environ.get('S4M_METRICS_OUT'):\n"
          "            import json as _json\n"
          "            _json.dump({'mse_batchavg': float(mse), 'mae_batchavg': float(mae),\n"
          "                        'mse': float(((_P - _T) ** 2).mean()), 'mae': float(np.abs(_P - _T).mean()),\n"
          "                        'n_windows': int(_P.shape[0])}, open(os.environ['S4M_METRICS_OUT'], 'w'))\n")
    # NumPy-2 compatibility (pure renames, no behavioural change)
    for f in DST.rglob("*.py"):
        t = f.read_text(errors="ignore")
        t2 = t.replace("np.Inf", "np.inf").replace("np.NaN", "np.nan")
        if t2 != t:
            f.write_text(t2)
    print(f"prepared {DST}")


def write_csv(dataset, pattern, rate, block, mask_seed):
    import numpy as np
    import pandas as pd
    from mtsf.data import DATA_FILES
    raw = ROOT / "data/raw" / DATA_FILES[dataset]
    out = ROOT / f"data/s4m/{FILE_TAG[dataset]}_{pattern}_r{rate}_b{block}_s{mask_seed}.csv"
    mpath = ROOT / f"data/masks/{dataset}_{pattern}_r{rate}_b{block}_s{mask_seed}.npy"
    if out.exists():
        return raw, out
    df = pd.read_csv(raw)
    cols = [c for c in df.columns if c != "date"]
    if mpath.exists():
        mask = np.load(mpath)
    else:
        from mtsf.data import make_mask
        mask = make_mask((len(df), len(cols)), pattern, rate, block, mask_seed)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        np.save(mpath, mask)
    vals = df[cols].values.astype(np.float64)
    vals[mask == 0] = np.nan
    df[cols] = vals
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return raw, out


def run(a):
    raw, csv = write_csv(a.dataset, a.pattern, a.rate, a.block, a.mask_seed)
    hp = HP[a.dataset]
    n_vars = {"etth1": 7, "weather": 21, "electricity": 321, "traffic": 862}[a.dataset]
    name = a.name or f"s4m__{a.dataset}__{a.pattern}_r{a.rate}__L{a.seq_len}__s{a.seed}"
    out_dir = ROOT / "runs_mtsf" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "--is_training", "1", "--root_path", str(csv.parent) + "/", "--data_path", csv.name,
        "--model_id", name, "--model", "S4M", "--data", "custom4", "--features", "M",
        "--seq_len", str(a.seq_len), "--pred_len", "96", "--enc_in", str(n_vars), "--dec_in", str(n_vars),
        "--c_out", str(n_vars), "--per_mem_size", "5", "--en_conv_hidden_size", str(hp["en_conv_hidden_size"]),
        "--topK", "20", "--thres1", "0.95", "--thres2", "0.6", "--M", "50", "--memory_size", "200",
        "--momentum", "0.99", "--short_len", "32", "--e_layers", str(a.e_layers), "--des", "Exp",
        "--d_model", str(hp["d_model"]), "--d_ff", str(hp["d_ff"]), "--learning_rate", str(a.lr or hp["lr"]),
        "--train_epochs", str(a.epochs), "--patience", "3", "--num_pattn", "30", "--num_dG", "1",
        "--encoder_type", "3", "--memnet", "13", "--mem_type", "3", "--itr", "1",
        "--mean_type", "global_mean", "--fillna", "mean", "--batch_size", "16", "--seed", str(a.seed),
        "--num_workers", "2", "--checkpoints", str(out_dir / "ckpt") + "/",
    ]
    env = dict(os.environ)
    env["S4M_TRUE_PATH"] = str(raw)
    env["S4M_OPTIM"] = a.optim
    env["S4M_METRICS_OUT"] = str(out_dir / "s4m_metrics.json")
    env["PYTHONPATH"] = os.pathsep.join([str(DST / "_stubs"), str(DST), str(DST / "model"), env.get("PYTHONPATH", "")])
    env["WANDB_MODE"] = "disabled"
    env["PYTHONNOUSERSITE"] = "1"
    with open(out_dir / "train.log", "w") as log:
        rc = subprocess.call([sys.executable, "run.py", *args], cwd=out_dir if False else DST, env=env,
                             stdout=log, stderr=subprocess.STDOUT)
    if rc != 0:
        raise SystemExit(f"S4M failed rc={rc}, see {out_dir/'train.log'}")
    m = json.load(open(out_dir / "s4m_metrics.json"))
    res = {"model": f"s4m-{a.optim}", "dataset": a.dataset, "pattern": a.pattern, "rate": a.rate,
           "seq_len": a.seq_len, "pred_len": 96, "seed": a.seed, "mask_seed": a.mask_seed,
           "mse": m["mse"], "mae": m["mae"], "mse_batchavg": m["mse_batchavg"], "mae_batchavg": m["mae_batchavg"],
           "n_windows": m["n_windows"], "tag": name}
    json.dump(res, open(out_dir / "metrics.json", "w"), indent=2)
    (ROOT / "results_mtsf").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "results_mtsf" / f"{name}.json", "w"), indent=2)
    print(res)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    r = sub.add_parser("run")
    r.add_argument("--dataset", required=True)
    r.add_argument("--pattern", default="point")
    r.add_argument("--rate", type=float, default=0.06)
    r.add_argument("--block", type=int, default=5)
    r.add_argument("--mask_seed", type=int, default=2024)
    r.add_argument("--seq_len", type=int, default=96)
    r.add_argument("--seed", type=int, default=2024)
    r.add_argument("--epochs", type=int, default=15)
    r.add_argument("--e_layers", type=int, default=8)
    r.add_argument("--lr", type=float, default=None)
    r.add_argument("--optim", default="adam", choices=["adam", "sgd"])
    r.add_argument("--name", default=None)
    a = ap.parse_args()
    prepare() if a.cmd == "prepare" else run(a)


if __name__ == "__main__":
    main()
