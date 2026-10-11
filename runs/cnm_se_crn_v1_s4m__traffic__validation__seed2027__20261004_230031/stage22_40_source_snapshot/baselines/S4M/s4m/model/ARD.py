"""Analytic residual decision distillation gate for MCPR."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnalyticResidualDecisionGate(nn.Module):
    """Predict OFF/PARTIAL/FULL residual actions from label-free features."""

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

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        if int(hidden_dim) < 1:
            raise ValueError("ARD hidden dimension must be positive")
        self.feature_dim = len(self.feature_names)
        self.hidden_dim = int(hidden_dim)
        self.feature_projection = nn.Linear(self.feature_dim, self.hidden_dim)
        self.action_head = nn.Linear(self.hidden_dim, 3)
        self.partial_head = nn.Linear(self.hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.feature_projection.weight)
        nn.init.zeros_(self.feature_projection.bias)
        # Zero output heads preserve the neutral initial scale:
        # p=(1/3,1/3,1/3), partial=0.5, expected alpha=0.5.
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        nn.init.zeros_(self.partial_head.weight)
        nn.init.zeros_(self.partial_head.bias)

    @staticmethod
    def _rms(value: torch.Tensor) -> torch.Tensor:
        return value.square().mean(dim=(1, 2)).clamp_min(0.0).sqrt()

    def _features(
        self,
        raw_residual: torch.Tensor,
        forecast_hidden: torch.Tensor,
        mask_features: torch.Tensor,
        backbone_forecast: torch.Tensor,
    ) -> torch.Tensor:
        expected = raw_residual.shape[:2]
        if forecast_hidden.shape[:2] != expected:
            raise ValueError("ARD hidden and residual batch/time dimensions must match")
        if mask_features.shape[:2] != expected or mask_features.size(-1) != 4:
            raise ValueError("ARD mask features must be [batch,time,4]")
        if backbone_forecast.shape != raw_residual.shape:
            raise ValueError("ARD backbone forecast and residual shapes must match")
        if backbone_forecast.size(1) > 1:
            temporal_difference = (
                backbone_forecast[:, 1:, :] - backbone_forecast[:, :-1, :]
            )
            difference_rms = self._rms(temporal_difference)
        else:
            difference_rms = torch.zeros(
                backbone_forecast.size(0),
                dtype=backbone_forecast.dtype,
                device=backbone_forecast.device,
            )
        features = torch.cat((
            mask_features.mean(dim=1),
            self._rms(forecast_hidden).unsqueeze(-1),
            self._rms(raw_residual).unsqueeze(-1),
            self._rms(backbone_forecast).unsqueeze(-1),
            difference_rms.unsqueeze(-1),
        ), dim=-1)
        if features.size(-1) != self.feature_dim:
            raise RuntimeError("ARD feature width changed unexpectedly")
        return features.detach()

    def forward(
        self,
        raw_residual: torch.Tensor,
        forecast_hidden: torch.Tensor,
        mask_features: torch.Tensor,
        backbone_forecast: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self._features(
            raw_residual, forecast_hidden, mask_features, backbone_forecast
        )
        hidden = F.gelu(self.feature_projection(features))
        action_logits = self.action_head(hidden)
        action_probabilities = torch.softmax(action_logits, dim=-1)
        partial_alpha = torch.sigmoid(self.partial_head(hidden)).squeeze(-1)
        alpha = (
            action_probabilities[:, 2]
            + action_probabilities[:, 1] * partial_alpha
        )
        gate = alpha.view(-1, 1, 1)
        return {
            "residual": gate * raw_residual,
            "raw_residual": raw_residual,
            "ard_gate": gate,
            "ard_features": features,
            "ard_action_logits": action_logits,
            "ard_action_probabilities": action_probabilities,
            "ard_partial_alpha": partial_alpha,
        }
