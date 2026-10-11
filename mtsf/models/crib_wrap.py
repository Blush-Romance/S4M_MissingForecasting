"""Wrapper around the official CRIB implementation (arXiv 2509.23494).

The original code is *not* vendored (no licence file upstream); clone it and
point CRIB_ROOT at it:  git clone https://github.com/Muyiiiii/CRIB ../ext/CRIB

Training objective follows the official `loss_type=123`:
  IB_weight * MAE(pred, y) + Consis_weight * MSE(z1, z2) + KL_weight * KL,
with the MAE restricted to observed future targets (our protocol never uses
unobserved ground truth during training).
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

CRIB_ROOT = os.environ.get("CRIB_ROOT", os.path.expanduser("~/projects/ext/CRIB"))


def _load_crib():
    """Import TSL_models/CRIB.py without running the package __init__
    (which imports every baseline and has a circular import)."""
    import importlib
    import types
    name = "_crib_tsl"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [os.path.join(CRIB_ROOT, "TSL_models")]
        sys.modules[name] = pkg
    return importlib.import_module(f"{name}.CRIB").CRIB


class CRIB(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        _CRIB = _load_crib()
        self.P = getattr(cfg, "crib_patch_len", 8)
        assert cfg.seq_len % self.P == 0
        args = SimpleNamespace(
            patch_len=self.P, patch_num=cfg.seq_len // self.P, model_dim=getattr(cfg, "crib_dim", 32),
            heads_num=4, enc_num=3, dropout=0.1, output_attention=False, activation="relu",
            pred_len=cfg.pred_len, var_num=cfg.n_vars, batch_size=cfg.batch_size,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.net = _CRIB(args)
        self.w_ib, self.w_cons, self.w_kl = 1.0, 1.0, 1e-6

    def _patches(self, x, m):
        B, L, N = x.shape
        x = (x * m).view(B, L // self.P, self.P, N).permute(0, 1, 3, 2)  # [B, P#, N, Plen]
        return x

    def forward(self, x, m, tx=None, ty=None):
        out = self.net(self._patches(x, m), x_mark=None, test_flag=True)
        return out[4]

    def compute_loss(self, batch):
        x, m, y, my = batch[:4]
        e1, _, e2, _, pred, _, kl = self.net(self._patches(x, m), x_mark=None, test_flag=False)
        mae = ((pred - y).abs() * my).sum() / my.sum().clamp(min=1)
        cons = ((e1 - e2) ** 2).mean()
        loss = self.w_ib * mae + self.w_cons * cons + self.w_kl * kl
        return loss, {"fore": mae.item(), "cons": cons.item()}
