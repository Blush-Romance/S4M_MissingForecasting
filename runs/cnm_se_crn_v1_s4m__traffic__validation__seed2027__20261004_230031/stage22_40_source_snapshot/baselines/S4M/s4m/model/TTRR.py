"""Tail-aware trust-region gate for MCPR forecast residuals."""

from __future__ import annotations

import torch
import torch.nn as nn


class TailAwareTrustRegionGate(nn.Module):
    """Learn one deployable residual-amplitude gate per input window."""

    feature_names = (
        "mask_observed_mean",
        "mask_transition_mean",
        "mask_age_mean",
        "mask_global_observed_mean",
        "hidden_rms",
        "raw_residual_rms",
        "backbone_forecast_rms",
        "backbone_temporal_difference_rms",
    )

    def __init__(self) -> None:
        super().__init__()
        self.feature_dim = len(self.feature_names)
        self.projection = nn.Linear(self.feature_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # A label-free neutral trust prior: sigmoid(0) == 0.5.
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    @staticmethod
    def _rms(value: torch.Tensor) -> torch.Tensor:
        return value.square().mean(dim=(1, 2)).clamp_min(0.0).sqrt()

    def forward(
        self,
        raw_residual: torch.Tensor,
        forecast_hidden: torch.Tensor,
        mask_features: torch.Tensor,
        backbone_forecast: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        expected = raw_residual.shape[:2]
        if forecast_hidden.shape[:2] != expected:
            raise ValueError("TTRR hidden and residual batch/time dimensions must match")
        if mask_features.shape[:2] != expected or mask_features.size(-1) != 4:
            raise ValueError("TTRR mask features must be [batch,time,4]")
        if backbone_forecast.shape != raw_residual.shape:
            raise ValueError("TTRR backbone forecast and residual shapes must match")

        if backbone_forecast.size(1) > 1:
            temporal_difference = (
                backbone_forecast[:, 1:, :] - backbone_forecast[:, :-1, :]
            )
            temporal_difference_rms = self._rms(temporal_difference)
        else:
            temporal_difference_rms = torch.zeros(
                backbone_forecast.size(0),
                dtype=backbone_forecast.dtype,
                device=backbone_forecast.device,
            )

        trust_features = torch.cat(
            (
                mask_features.mean(dim=1),
                self._rms(forecast_hidden).unsqueeze(-1),
                self._rms(raw_residual).unsqueeze(-1),
                self._rms(backbone_forecast).unsqueeze(-1),
                temporal_difference_rms.unsqueeze(-1),
            ),
            dim=-1,
        )
        if trust_features.size(-1) != self.feature_dim:
            raise RuntimeError("TTRR trust feature width changed unexpectedly")
        detached_features = trust_features.detach()
        gate = torch.sigmoid(self.projection(detached_features)).view(-1, 1, 1)
        residual = gate * raw_residual
        return {
            "residual": residual,
            "raw_residual": raw_residual,
            "trust_gate": gate,
            "trust_features": detached_features,
        }
