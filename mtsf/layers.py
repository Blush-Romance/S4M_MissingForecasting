"""Shared building blocks."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObsNorm(nn.Module):
    """Instance normalisation using observed entries only (mask-aware RevIN).

    Plain RevIN on zero-filled inputs is biased towards 0 by the missing
    entries; here the mean/std are computed on observed values only.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def norm(self, x, m):
        cnt = m.sum(1, keepdim=True)
        safe = cnt.clamp(min=1.0)
        mu = (x * m).sum(1, keepdim=True) / safe
        var = (((x - mu) * m) ** 2).sum(1, keepdim=True) / safe
        sd = torch.sqrt(var + self.eps)
        none = cnt < 1
        mu = torch.where(none, torch.zeros_like(mu), mu)
        sd = torch.where(none, torch.ones_like(sd), sd)
        return (x - mu) / sd * m, mu, sd

    @staticmethod
    def denorm(y, mu, sd):
        return y * sd + mu


class RevIN(nn.Module):
    """Standard RevIN (no affine), used by the generic baselines."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def norm(self, x):
        mu = x.mean(1, keepdim=True).detach()
        sd = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps).detach()
        return (x - mu) / sd, mu, sd

    @staticmethod
    def denorm(y, mu, sd):
        return y * sd + mu


def linear_interpolate(x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Within-window linear interpolation along time (dim 1).

    Leading / trailing gaps use nearest observation (forward / backward fill).
    Returns the filled series and the normalised distance to the closest
    observation (0 at observed points).
    """
    B, L, D = x.shape
    idx = torch.arange(L, device=x.device).view(1, L, 1).expand(B, L, D)
    obs = m > 0
    big = L * 4
    prev_idx = torch.where(obs, idx, torch.full_like(idx, -big))
    prev_idx = torch.cummax(prev_idx, dim=1).values
    nxt_idx = torch.where(obs, idx, torch.full_like(idx, 2 * big))
    nxt_idx = torch.flip(torch.cummin(torch.flip(nxt_idx, [1]), dim=1).values, [1])
    has_prev = prev_idx >= 0
    has_next = nxt_idx < L
    pv = torch.gather(x, 1, prev_idx.clamp(0, L - 1))
    nv = torch.gather(x, 1, nxt_idx.clamp(0, L - 1))
    span = (nxt_idx - prev_idx).clamp(min=1).float()
    w = (idx - prev_idx).float() / span
    interp = pv + (nv - pv) * w
    filled = torch.where(has_prev & has_next, interp, torch.where(has_prev, pv, torch.where(has_next, nv, torch.zeros_like(x))))
    filled = torch.where(obs, x, filled)
    dist = torch.minimum((idx - prev_idx).abs(), (nxt_idx - idx).abs()).float()
    dist = torch.where(obs, torch.zeros_like(dist), dist)
    return filled, dist


class MHA(nn.Module):
    """Multi-head self-attention with an optional additive key bias."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = dropout

    def forward(self, x, key_bias=None):
        B, N, d = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        mask = None
        if key_bias is not None:  # [B, h, 1, N] or [B, 1, 1, N]
            mask = key_bias.to(q.dtype)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop if self.training else 0.0)
        return self.out(o.transpose(1, 2).reshape(B, N, d))


class Block(nn.Module):
    def __init__(self, d, heads, d_ff, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = MHA(d, heads, dropout)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_bias=None):
        x = x + self.drop(self.attn(self.n1(x), key_bias))
        return x + self.drop(self.ff(self.n2(x)))


def sincos_pe(n, d, device):
    pos = torch.arange(n, device=device).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe
