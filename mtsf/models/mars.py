"""MARS: Mask-Aware Reliability-weighted patch tranSformer.

Imputation-free forecaster for multivariate series with block missingness.

  1. ObsNorm        instance normalisation from observed entries only;
  2. Gap-aware      patches are built from a non-learned within-window linear
     tokens         interpolation plus the mask and the distance-to-observation,
                    so the encoder sees *how much* of every patch is real;
  3. Reliability    each attention head learns how strongly to down-weight
     attention      keys of poorly observed patches (bias = a_h * log(r + eps));
  4. Variate        per layer, pooled variable summaries exchange information
     mixing         through a light variate-attention block (zero-initialised
                    residual, so the model starts channel-independent);
  5. Missingness-   during training, extra blocks following the *same*
     consistent     missingness pattern are dropped from the input; the model
     masking (MCM)  must still forecast and reconstruct the hidden observed
                    values (auxiliary loss, no ground-truth leakage: only
                    originally observed values are used as targets).

Every component has a switch in the config so it can be ablated.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import Block, ObsNorm, RevIN, linear_interpolate, sincos_pe


def masked_mse(pred, target, mask):
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)


def block_drop(m: torch.Tensor, pattern: str, rate: torch.Tensor, block: int) -> torch.Tensor:
    """Sample a keep-mask with block missingness (per-sample rate [B])."""
    B, L, D = m.shape
    r = rate.view(B, 1, 1)
    if pattern == "point":
        starts = (torch.rand(B, L, 1, device=m.device) < r).float()
    else:
        starts = (torch.rand(B, L, D, device=m.device) < r).float()
    # dilate starts forward by `block` steps
    s = starts.transpose(1, 2)
    s = F.pad(s, (block - 1, 0))
    drop = F.max_pool1d(s.reshape(-1, 1, s.shape[-1]), block, stride=1).view(B, -1, L).transpose(1, 2)
    return 1.0 - drop.expand(B, L, D)


class VarMix(nn.Module):
    def __init__(self, d, heads, d_ff, dropout):
        super().__init__()
        self.block = Block(d, heads, d_ff, dropout)
        self.proj = nn.Linear(d, d)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z, B, D):
        # z: [B*D, N, d]
        v = z.mean(1).view(B, D, -1)
        dv = self.block(v) - v
        return z + self.proj(dv).reshape(B * D, 1, -1)


class MARS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.P, self.S = cfg.patch_len, cfg.stride
        self.N = (cfg.seq_len - self.P) // self.S + 2
        d = cfg.d_model
        in_ch = 3 if cfg.use_gap_tokens else 1
        self.embed = nn.Linear(self.P * in_ch, d)
        self.pos = nn.Parameter(sincos_pe(self.N, d, "cpu"), requires_grad=False)
        self.blocks = nn.ModuleList([Block(d, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.e_layers)])
        self.mixers = nn.ModuleList([VarMix(d, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.e_layers)]) if cfg.use_var_mix else None
        if cfg.use_rel_bias:
            self.rel_alpha = nn.Parameter(torch.zeros(cfg.e_layers, cfg.n_heads))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Flatten(-2), nn.Dropout(cfg.head_dropout), nn.Linear(self.N * d, cfg.pred_len))
        self.rec_head = nn.Linear(d, self.P)
        self.obsnorm = ObsNorm()
        self.revin = RevIN()

    # ------------------------------------------------------------------
    def _patch(self, t):  # t: [B*D, L] -> [B*D, N, P]
        t = torch.cat([t, t[:, -1:].repeat(1, self.S)], -1)
        return t.unfold(-1, self.P, self.S)

    def encode(self, x, m):
        B, L, D = x.shape
        cfg = self.cfg
        if cfg.use_obsnorm:
            xn, mu, sd = self.obsnorm.norm(x, m)
        else:
            xn, mu, sd = self.revin.norm(x)
        if cfg.use_gap_tokens:
            xf, dist = linear_interpolate(xn, m)
            feats = [xf, m, torch.log1p(dist)]
        else:
            feats = [xn]
        patches = [self._patch(f.transpose(1, 2).reshape(B * D, L)) for f in feats]
        z = self.embed(torch.cat(patches, -1)) + self.pos
        bias_base = None
        if cfg.use_rel_bias:
            rel = self._patch(m.transpose(1, 2).reshape(B * D, L)).mean(-1)  # [B*D, N]
            bias_base = torch.log(rel + 0.05).view(B * D, 1, 1, self.N)
        for i, blk in enumerate(self.blocks):
            kb = None if bias_base is None else bias_base * self.rel_alpha[i].view(1, -1, 1, 1)
            z = blk(z, kb)
            if self.mixers is not None:
                z = self.mixers[i](z, B, D)
        return self.norm(z), xn, mu, sd

    def forward(self, x, m, tx=None, ty=None, return_aux=False):
        B, L, D = x.shape
        z, xn, mu, sd = self.encode(x, m)
        y = self.head(z).view(B, D, -1).transpose(1, 2)
        y = ObsNorm.denorm(y, mu, sd)
        if not return_aux:
            return y
        rec = self.rec_head(z)  # [B*D, N, P] -> fold back to length L (+S pad)
        Lp = L + self.S
        rec = F.fold(rec.transpose(1, 2), output_size=(1, Lp), kernel_size=(1, self.P), stride=(1, self.S))
        cnt = F.fold(torch.ones_like(rec.new_empty(B * D, self.P, self.N)), output_size=(1, Lp),
                     kernel_size=(1, self.P), stride=(1, self.S))
        rec = (rec / cnt.clamp(min=1)).view(B, D, Lp)[..., :L].transpose(1, 2)
        return y, rec, mu, sd

    # ------------------------------------------------------------------
    def compute_loss(self, batch):
        x, m, y, my = batch[:4]
        cfg = self.cfg
        if cfg.mcm_rate > 0:
            B = x.shape[0]
            rate = torch.rand(B, device=x.device) * cfg.mcm_rate
            apply = (torch.rand(B, device=x.device) < cfg.mcm_prob).float()
            keep = block_drop(m, cfg.pattern, rate * apply, cfg.block)
            m_in = m * keep
            pred, rec, mu, sd = self(x * m_in, m_in, return_aux=True)
            hidden = m * (1 - m_in)
            target = (x - mu) / sd
            l_rec = masked_mse(rec, target, hidden) if hidden.sum() > 0 else pred.sum() * 0
        else:
            pred = self(x, m)
            l_rec = pred.sum() * 0
        l_fore = masked_mse(pred, y, my)
        loss = l_fore + cfg.rec_weight * l_rec
        return loss, {"fore": l_fore.item(), "rec": l_rec.item()}
