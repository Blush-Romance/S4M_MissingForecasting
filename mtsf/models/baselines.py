"""Generic forecasters applied directly to zero-filled partially observed input.

Following CRIB (2025), modern MTSF models are strong baselines for forecasting
with missing values. They receive the zero-filled series (0 = train mean)
and, unless stated otherwise, standard RevIN.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..layers import Block, RevIN, sincos_pe


class DLinear(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        L, H = cfg.seq_len, cfg.pred_len
        self.k = 25
        self.avg = nn.AvgPool1d(self.k, stride=1, padding=0)
        self.trend = nn.Linear(L, H)
        self.seas = nn.Linear(L, H)
        self.rev = RevIN()

    def forward(self, x, m, tx=None, ty=None):
        x, mu, sd = self.rev.norm(x)
        xt = x.transpose(1, 2)
        pad = (self.k - 1) // 2
        xp = torch.cat([xt[..., :1].repeat(1, 1, pad), xt, xt[..., -1:].repeat(1, 1, pad)], -1)
        tr = self.avg(xp)
        y = self.trend(tr) + self.seas(xt - tr)
        return self.rev.denorm(y.transpose(1, 2), mu, sd)


class PatchTST(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.P, self.S = cfg.patch_len, cfg.stride
        self.N = (cfg.seq_len - self.P) // self.S + 2  # with end padding
        d = cfg.d_model
        self.embed = nn.Linear(self.P, d)
        self.pos = nn.Parameter(sincos_pe(self.N, d, "cpu"), requires_grad=False)
        self.blocks = nn.ModuleList([Block(d, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.e_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Flatten(-2), nn.Dropout(cfg.head_dropout), nn.Linear(self.N * d, cfg.pred_len))
        self.rev = RevIN()

    def forward(self, x, m, tx=None, ty=None):
        B, L, D = x.shape
        x, mu, sd = self.rev.norm(x)
        xt = x.transpose(1, 2)
        xt = torch.cat([xt, xt[..., -1:].repeat(1, 1, self.S)], -1)
        p = xt.unfold(-1, self.P, self.S).reshape(B * D, self.N, self.P)
        z = self.embed(p) + self.pos
        for blk in self.blocks:
            z = blk(z)
        y = self.head(self.norm(z)).view(B, D, -1).transpose(1, 2)
        return self.rev.denorm(y, mu, sd)


class iTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.embed = nn.Linear(cfg.seq_len, d)
        self.blocks = nn.ModuleList([Block(d, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.e_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.pred_len)
        self.rev = RevIN()

    def forward(self, x, m, tx=None, ty=None):
        x, mu, sd = self.rev.norm(x)
        z = self.embed(x.transpose(1, 2))
        for blk in self.blocks:
            z = blk(z)
        y = self.head(self.norm(z)).transpose(1, 2)
        return self.rev.denorm(y, mu, sd)
