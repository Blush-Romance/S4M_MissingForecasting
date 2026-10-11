"""Mask-responsive residual adaptation for S4M.

The module is deliberately lightweight and strictly causal.  It reads only
the decay-imputed history and its observation mask, then produces a residual
for the memory-fused context plus an auxiliary history reconstruction.  The
context projection is zero initialized so enabling MRRA preserves the
baseline forecast at initialization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskResponsiveResidualAdapter(nn.Module):
    """Causal low-rank adapter conditioned on values and missingness."""

    def __init__(
        self,
        num_variables: int,
        context_size: int,
        rank: int = 32,
        kernel_size: int = 5,
        version: str = "v1",
    ) -> None:
        super().__init__()
        if num_variables < 1 or context_size < 1 or rank < 1:
            raise ValueError("MRRA dimensions must be positive")
        if kernel_size < 1:
            raise ValueError("MRRA kernel_size must be positive")
        self.num_variables = int(num_variables)
        self.context_size = int(context_size)
        self.rank = int(rank)
        self.kernel_size = int(kernel_size)
        self.version = str(version)
        if self.version not in {
            "v1",
            "v2_normalized_bounded",
            "v3_gradient_isolated_two_timescale",
        }:
            raise ValueError(f"unsupported MRRA version: {self.version}")
        self.gradient_isolated = (
            self.version == "v3_gradient_isolated_two_timescale"
        )

        self.input_projection = nn.Linear(2 * self.num_variables, self.rank)
        self.temporal_filter = nn.Conv1d(
            self.rank,
            self.rank,
            kernel_size=self.kernel_size,
            groups=self.rank,
            bias=True,
        )
        if self.version == "v2_normalized_bounded":
            self.hidden_normalization = nn.LayerNorm(self.rank)
        elif self.version == "v3_gradient_isolated_two_timescale":
            self.hidden_normalization = nn.LayerNorm(
                self.rank, elementwise_affine=False
            )
        else:
            self.hidden_normalization = None
        self.context_projection = nn.Linear(self.rank, self.context_size)
        self.reconstruction_projection = nn.Linear(self.rank, self.num_variables)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.kaiming_uniform_(self.temporal_filter.weight, nonlinearity="linear")
        nn.init.zeros_(self.temporal_filter.bias)
        # Preserve the exact baseline forecasting path at initialization while
        # the reconstruction head supplies an immediate learning signal.
        nn.init.zeros_(self.context_projection.weight)
        nn.init.zeros_(self.context_projection.bias)
        nn.init.xavier_uniform_(self.reconstruction_projection.weight)
        nn.init.zeros_(self.reconstruction_projection.bias)

    def forward(
        self,
        imputed_history: torch.Tensor,
        observation_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if imputed_history.shape != observation_mask.shape:
            raise ValueError("MRRA history and mask shapes must match")
        if imputed_history.ndim != 3:
            raise ValueError("MRRA expects [batch, time, variables] inputs")
        if imputed_history.size(-1) != self.num_variables:
            raise ValueError("MRRA input variable count does not match construction")

        adapter_history = (
            imputed_history.detach() if self.gradient_isolated else imputed_history
        )
        mask = observation_mask.to(dtype=adapter_history.dtype)
        hidden = F.gelu(
            self.input_projection(torch.cat((adapter_history, mask), dim=-1))
        )
        temporal = hidden.transpose(1, 2)
        temporal = F.pad(temporal, (self.kernel_size - 1, 0))
        temporal = self.temporal_filter(temporal).transpose(1, 2)
        hidden = hidden + temporal
        if self.hidden_normalization is not None:
            hidden = self.hidden_normalization(hidden)
        hidden = F.gelu(hidden)
        context_residual = self.context_projection(hidden)
        reconstruction_correction = self.reconstruction_projection(hidden)
        if self.version in {
            "v2_normalized_bounded",
            "v3_gradient_isolated_two_timescale",
        }:
            # A bounded influence path prevents the auxiliary adapter from
            # injecting arbitrarily large residuals into the S4M backbone.
            context_residual = torch.tanh(context_residual)
            reconstruction_correction = torch.tanh(reconstruction_correction)
        reconstruction = adapter_history + reconstruction_correction
        return {
            "context_residual": context_residual,
            "reconstruction": reconstruction,
            "reconstruction_correction": reconstruction_correction,
            "hidden": hidden,
        }
