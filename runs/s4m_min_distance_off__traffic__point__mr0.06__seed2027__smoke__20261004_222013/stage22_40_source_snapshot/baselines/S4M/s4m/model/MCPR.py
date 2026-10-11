"""Mask-conditioned post-backbone residual adaptation for S4M."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.TTRR import TailAwareTrustRegionGate
from model.ARD import AnalyticResidualDecisionGate


class MaskConditionedParetoResidual(nn.Module):
    """A zero-initialized low-rank forecast residual with explicit mask input."""

    def __init__(
        self,
        hidden_size: int,
        num_variables: int,
        rank: int = 32,
        kernel_size: int = 5,
        use_ttrr: bool = False,
        use_ard: bool = False,
        ard_hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        if min(hidden_size, num_variables, rank, kernel_size) < 1:
            raise ValueError("MCPR dimensions must be positive")
        self.hidden_size = int(hidden_size)
        self.num_variables = int(num_variables)
        self.rank = int(rank)
        self.kernel_size = int(kernel_size)
        self.use_ttrr = bool(use_ttrr)
        self.use_ard = bool(use_ard)
        if self.use_ttrr and self.use_ard:
            raise ValueError("MCPR cannot enable TTRR and ARD together")
        self.hidden_projection = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.mask_projection = nn.Linear(4, self.rank)
        self.temporal_filter = nn.Conv1d(
            self.rank,
            self.rank,
            kernel_size=self.kernel_size,
            groups=self.rank,
        )
        self.normalization = nn.LayerNorm(self.rank, elementwise_affine=False)
        self.output_projection = nn.Linear(self.rank, self.num_variables)
        self.trust_gate = TailAwareTrustRegionGate() if self.use_ttrr else None
        self.ard_gate = (
            AnalyticResidualDecisionGate(hidden_dim=ard_hidden_dim)
            if self.use_ard else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.hidden_projection.weight)
        nn.init.xavier_uniform_(self.mask_projection.weight)
        nn.init.zeros_(self.mask_projection.bias)
        nn.init.kaiming_uniform_(self.temporal_filter.weight, nonlinearity="linear")
        nn.init.zeros_(self.temporal_filter.bias)
        # The source PG95 prediction is reproduced exactly before adaptation.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def _mask_features(observation_mask: torch.Tensor) -> torch.Tensor:
        if observation_mask.ndim != 3:
            raise ValueError("MCPR observation mask must be [batch,time,variables]")
        observed = observation_mask.to(dtype=torch.float32).mean(dim=-1)
        transition = torch.zeros_like(observed)
        transition[:, 1:] = (observed[:, 1:] - observed[:, :-1]).abs()
        current_age = torch.zeros_like(observed[:, 0])
        age_steps = []
        for time_index in range(observed.size(1)):
            current_age = torch.where(
                observed[:, time_index] > 0.5,
                torch.zeros_like(current_age),
                current_age + 1.0,
            )
            age_steps.append(current_age)
        age = torch.stack(age_steps, dim=1) / float(max(1, observed.size(1)))
        global_observed = observed.mean(dim=1, keepdim=True).expand_as(observed)
        return torch.stack((observed, transition, age, global_observed), dim=-1)

    def forward(
        self,
        forecast_hidden: torch.Tensor,
        observation_mask: torch.Tensor,
        backbone_forecast: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if forecast_hidden.ndim != 3:
            raise ValueError("MCPR hidden input must be [batch,time,hidden]")
        if forecast_hidden.size(-1) != self.hidden_size:
            raise ValueError("MCPR hidden width does not match construction")
        if forecast_hidden.shape[:2] != observation_mask.shape[:2]:
            raise ValueError("MCPR hidden and mask batch/time dimensions must match")
        mask_features = self._mask_features(observation_mask).to(
            dtype=forecast_hidden.dtype, device=forecast_hidden.device
        )
        latent = (
            self.hidden_projection(forecast_hidden.detach())
            + self.mask_projection(mask_features)
        )
        temporal = latent.transpose(1, 2)
        temporal = F.pad(temporal, (self.kernel_size - 1, 0))
        temporal = self.temporal_filter(temporal).transpose(1, 2)
        latent = F.gelu(self.normalization(latent + temporal))
        raw_residual = torch.tanh(self.output_projection(latent))
        outputs = {
            "residual": raw_residual,
            "raw_residual": raw_residual,
            "latent": latent,
            "mask_features": mask_features,
        }
        if self.use_ttrr:
            if backbone_forecast is None:
                raise ValueError("TTRR-enabled MCPR requires the frozen backbone forecast")
            outputs.update(self.trust_gate(
                raw_residual=raw_residual,
                forecast_hidden=forecast_hidden,
                mask_features=mask_features,
                backbone_forecast=backbone_forecast,
            ))
        elif self.use_ard:
            if backbone_forecast is None:
                raise ValueError("ARD-enabled MCPR requires the frozen backbone forecast")
            outputs.update(self.ard_gate(
                raw_residual=raw_residual,
                forecast_hidden=forecast_hidden,
                mask_features=mask_features,
                backbone_forecast=backbone_forecast,
            ))
        return outputs


class DualScaleMaskConditionedResidualBasis(nn.Module):
    """Two independent mask-conditioned bases for L5/L20 teacher corrections."""

    def __init__(
        self,
        hidden_size: int,
        num_variables: int,
        rank: int = 32,
        kernel_size: int = 5,
        l5_enabled: bool = True,
        l20_enabled: bool = True,
    ) -> None:
        super().__init__()
        if not (l5_enabled or l20_enabled):
            raise ValueError("DMRB requires at least one enabled residual basis")
        self.num_variables = int(num_variables)
        self.l5_enabled = bool(l5_enabled)
        self.l20_enabled = bool(l20_enabled)
        self.l5_basis = MaskConditionedParetoResidual(
            hidden_size=hidden_size,
            num_variables=num_variables,
            rank=rank,
            kernel_size=kernel_size,
        )
        self.l20_basis = MaskConditionedParetoResidual(
            hidden_size=hidden_size,
            num_variables=num_variables,
            rank=rank,
            kernel_size=kernel_size,
        )

    def _zero_residual(self, forecast_hidden: torch.Tensor) -> torch.Tensor:
        return forecast_hidden.new_zeros(
            forecast_hidden.size(0), forecast_hidden.size(1), self.num_variables
        )

    def forward(
        self,
        forecast_hidden: torch.Tensor,
        observation_mask: torch.Tensor,
        backbone_forecast: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if backbone_forecast is None:
            raise ValueError("DMRB requires the frozen backbone forecast")
        if self.l5_enabled:
            l5_outputs = self.l5_basis(
                forecast_hidden, observation_mask, backbone_forecast
            )
            residual_l5 = l5_outputs["residual"]
        else:
            l5_outputs = None
            residual_l5 = self._zero_residual(forecast_hidden)
        if self.l20_enabled:
            l20_outputs = self.l20_basis(
                forecast_hidden, observation_mask, backbone_forecast
            )
            residual_l20 = l20_outputs["residual"]
        else:
            l20_outputs = None
            residual_l20 = self._zero_residual(forecast_hidden)
        residual = residual_l5 + residual_l20
        outputs = {
            "residual": residual,
            "raw_residual": residual,
            "residual_l5": residual_l5,
            "residual_l20": residual_l20,
            "dmrb_l5_enabled": self.l5_enabled,
            "dmrb_l20_enabled": self.l20_enabled,
        }
        if l5_outputs is not None:
            outputs.update({
                "latent_l5": l5_outputs["latent"],
                "mask_features_l5": l5_outputs["mask_features"],
            })
        if l20_outputs is not None:
            outputs.update({
                "latent_l20": l20_outputs["latent"],
                "mask_features_l20": l20_outputs["mask_features"],
            })
        return outputs
