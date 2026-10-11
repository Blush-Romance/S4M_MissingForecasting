"""Mask-intervention prototype purification for S4M's prototype bank."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MIPPPurifier(nn.Module):
    """Factor content from missingness and score paired-mask stability.

    The content adapter is zero initialized, so enabling MIPP starts from the
    normalized S4M representation rather than replacing it with a random space.
    """

    def __init__(self, hidden_size: int, num_variables: int, variance_target: float = 1.0):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_variables = int(num_variables)
        self.variance_target = float(variance_target)

        self.content_adapter = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        nn.init.zeros_(self.content_adapter.weight)

        mask_input_size = 2 * self.num_variables
        self.mask_encoder = nn.Sequential(
            nn.Linear(mask_input_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.mask_decoder = nn.Linear(self.hidden_size, mask_input_size)

    @staticmethod
    def mask_statistics(mask: torch.Tensor) -> torch.Tensor:
        """Return per-variable missing rate and transition rate."""
        missing = 1.0 - mask
        missing_rate = missing.mean(dim=1)
        if mask.size(1) > 1:
            transition_rate = torch.abs(mask[:, 1:] - mask[:, :-1]).mean(dim=1)
        else:
            transition_rate = torch.zeros_like(missing_rate)
        return torch.cat([missing_rate, transition_rate], dim=-1)

    def encode_content(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = features + self.content_adapter(features)
        normalized = F.normalize(raw, dim=-1)
        return raw, normalized

    def encode_missingness(self, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        statistics = self.mask_statistics(mask)
        representation = self.mask_encoder(statistics)
        return representation, statistics

    @staticmethod
    def select_write_candidates(
        stability: torch.Tensor,
        mode: str,
        threshold: float,
        budget: int,
        minimum_candidates: int = 0,
        features: torch.Tensor | None = None,
        mask_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
        """Select mask-stable candidates with an explicit admission budget.

        ``absolute`` preserves the original threshold gate. ``budget_topk``
        replaces the ineffective absolute scale with a relative ranking while
        reusing the existing memory-write budget ``M``.

        ``stability_diverse_rank`` greedily fuses two scale-free percentile
        ranks: paired-mask stability and cosine novelty relative to the
        already selected candidates.  It has no learned parameter or tunable
        fusion weight and preserves the same write budget.
        """
        if stability.numel() < 1:
            raise ValueError("MIPP write selection requires at least one candidate")
        if mode not in {
            "absolute",
            "budget_topk",
            "stability_diverse_rank",
            "mask_regime_balanced_rank",
        }:
            raise ValueError(f"unsupported MIPP write gate mode: {mode}")
        if int(budget) < 1:
            raise ValueError("MIPP write budget must be positive")
        if int(minimum_candidates) < 0:
            raise ValueError("minimum_candidates must be non-negative")

        flat = stability.reshape(-1)
        raw_absolute = flat >= float(threshold)
        if mode == "absolute":
            selected = raw_absolute.clone()
            required = min(int(minimum_candidates), flat.numel())
            if int(selected.sum().item()) < required:
                top_ids = torch.topk(flat, k=required, largest=True, sorted=False).indices
                selected = torch.zeros_like(raw_absolute)
                selected[top_ids] = True
        elif mode == "budget_topk":
            selected_count = min(
                flat.numel(),
                max(int(budget), int(minimum_candidates)),
            )
            top_ids = torch.topk(
                flat,
                k=selected_count,
                largest=True,
                sorted=False,
            ).indices
            selected = torch.zeros_like(raw_absolute)
            selected[top_ids] = True
        elif mode == "stability_diverse_rank":
            if features is None:
                raise ValueError(
                    "stability_diverse_rank requires normalized candidate features"
                )
            flattened_features = features.detach().reshape(flat.numel(), -1)
            if flattened_features.size(0) != flat.numel():
                raise ValueError(
                    "candidate feature count must match stability count: "
                    f"features={flattened_features.size(0)} stability={flat.numel()}"
                )
            flattened_features = F.normalize(flattened_features, dim=-1)
            selected_count = min(
                flat.numel(),
                max(int(budget), int(minimum_candidates)),
            )
            stability_rank = MIPPPurifier._fractional_percentile_rank(flat)
            selected = torch.zeros_like(raw_absolute)
            first = torch.argmax(flat)
            selected[first] = True
            maximum_similarity = torch.matmul(
                flattened_features,
                flattened_features[first],
            )
            for _ in range(1, selected_count):
                remaining = torch.nonzero(~selected, as_tuple=False).flatten()
                novelty = 1.0 - maximum_similarity[remaining]
                novelty_rank = MIPPPurifier._fractional_percentile_rank(novelty)
                score = stability_rank[remaining] + novelty_rank
                best_score = score.max()
                tied = remaining[torch.isclose(score, best_score, rtol=0.0, atol=1e-12)]
                if tied.numel() > 1:
                    tied_stability = stability_rank[tied]
                    tied = tied[tied_stability == tied_stability.max()]
                chosen = tied.min()
                selected[chosen] = True
                maximum_similarity = torch.maximum(
                    maximum_similarity,
                    torch.matmul(flattened_features, flattened_features[chosen]),
                )
        else:
            if mask_features is None:
                raise ValueError(
                    "mask_regime_balanced_rank requires historical candidate masks"
                )
            descriptors = MIPPPurifier.mask_regime_descriptor(mask_features)
            descriptors = descriptors.detach().reshape(flat.numel(), -1)
            if descriptors.size(0) != flat.numel():
                raise ValueError(
                    "candidate mask count must match stability count: "
                    f"masks={descriptors.size(0)} stability={flat.numel()}"
                )
            selected_count = min(
                flat.numel(),
                max(int(budget), int(minimum_candidates)),
            )
            # L1 distance is a Hamming distance for the temporal binary part
            # and remains well defined for variable-specific missing rates.
            mask_distance = torch.cdist(descriptors, descriptors, p=1)
            mask_distance = mask_distance / float(descriptors.size(-1))
            stability_rank = MIPPPurifier._fractional_percentile_rank(flat)
            selected = torch.zeros_like(raw_absolute)
            first = torch.argmax(flat)
            selected[first] = True
            nearest_distance = mask_distance[:, first]
            for _ in range(1, selected_count):
                remaining = torch.nonzero(~selected, as_tuple=False).flatten()
                reduced = torch.minimum(
                    nearest_distance.unsqueeze(1),
                    mask_distance[:, remaining],
                )
                coverage_gain = (
                    nearest_distance.unsqueeze(1) - reduced
                ).mean(dim=0)
                coverage_rank = MIPPPurifier._fractional_percentile_rank(
                    coverage_gain
                )
                score = stability_rank[remaining] + coverage_rank
                best_score = score.max()
                tied = remaining[
                    torch.isclose(score, best_score, rtol=0.0, atol=1e-12)
                ]
                if tied.numel() > 1:
                    tied_stability = stability_rank[tied]
                    tied = tied[tied_stability == tied_stability.max()]
                chosen = tied.min()
                selected[chosen] = True
                nearest_distance = torch.minimum(
                    nearest_distance,
                    mask_distance[:, chosen],
                )

        selected_values = flat[selected]
        # An absolute gate is allowed to reject every candidate.  Preserve the
        # empty admission set and report the configured threshold as its
        # effective cutoff instead of reducing an empty tensor.
        effective_cutoff = (
            selected_values.min()
            if selected_values.numel() > 0
            else flat.new_tensor(float(threshold))
        )
        diagnostics: dict[str, torch.Tensor | int] = {
            "write_stability": flat.mean(),
            "write_stability_std": flat.std(unbiased=False),
            "write_stability_min": flat.min(),
            "write_stability_max": flat.max(),
            "write_effective_cutoff": effective_cutoff,
            "write_raw_absolute_acceptance_rate": raw_absolute.float().mean(),
            "write_effective_acceptance_rate": selected.float().mean(),
            "write_candidate_count": int(flat.numel()),
            "write_eligible_count": int(selected.sum().item()),
        }
        if mode == "stability_diverse_rank":
            all_similarity = torch.matmul(flattened_features, flattened_features.T)
            all_upper = all_similarity[
                torch.triu_indices(
                    all_similarity.size(0),
                    all_similarity.size(1),
                    offset=1,
                    device=all_similarity.device,
                ).unbind()
            ]
            selected_features = flattened_features[selected]
            selected_similarity = torch.matmul(selected_features, selected_features.T)
            selected_upper = selected_similarity[
                torch.triu_indices(
                    selected_similarity.size(0),
                    selected_similarity.size(1),
                    offset=1,
                    device=selected_similarity.device,
                ).unbind()
            ]
            candidate_pairwise = (
                all_upper.mean() if all_upper.numel() else flat.new_zeros(())
            )
            selected_pairwise = (
                selected_upper.mean()
                if selected_upper.numel()
                else flat.new_zeros(())
            )
            diagnostics.update({
                "write_candidate_pairwise_cosine": candidate_pairwise,
                "write_selected_pairwise_cosine": selected_pairwise,
                "write_selected_stability": selected_values.mean(),
                "write_diversity_gain": candidate_pairwise - selected_pairwise,
            })
        if mode == "mask_regime_balanced_rank":
            selected_ids = torch.nonzero(selected, as_tuple=False).flatten()
            selected_distance = mask_distance[selected_ids][:, selected_ids]
            selected_upper = selected_distance[
                torch.triu_indices(
                    selected_distance.size(0),
                    selected_distance.size(1),
                    offset=1,
                    device=selected_distance.device,
                ).unbind()
            ]
            all_upper = mask_distance[
                torch.triu_indices(
                    mask_distance.size(0),
                    mask_distance.size(1),
                    offset=1,
                    device=mask_distance.device,
                ).unbind()
            ]
            topk_ids = torch.topk(
                flat,
                k=selected_count,
                largest=True,
                sorted=False,
            ).indices
            selected_coverage = mask_distance[:, selected_ids].min(dim=1).values.mean()
            topk_coverage = mask_distance[:, topk_ids].min(dim=1).values.mean()
            missing = 1.0 - mask_features.detach().reshape(flat.numel(), -1).float()
            diagnostics.update({
                "write_mask_candidate_pairwise_distance": (
                    all_upper.mean() if all_upper.numel() else flat.new_zeros(())
                ),
                "write_mask_selected_pairwise_distance": (
                    selected_upper.mean()
                    if selected_upper.numel()
                    else flat.new_zeros(())
                ),
                "write_mask_coverage_distance": selected_coverage,
                "write_mask_topk_coverage_distance": topk_coverage,
                "write_mask_coverage_gain_vs_topk": (
                    topk_coverage - selected_coverage
                ),
                "write_mask_candidate_missing_rate": missing.mean(),
                "write_mask_selected_missing_rate": missing[selected_ids].mean(),
                "write_selected_stability": selected_values.mean(),
            })
        return selected.reshape_as(stability), diagnostics

    @staticmethod
    def mask_regime_descriptor(mask_blocks: torch.Tensor) -> torch.Tensor:
        """Build a target-free, fixed-width morphology descriptor.

        Input is the observed-value mask ``[B,Q,S,D]``.  The descriptor uses
        per-time missing proportions plus five summary statistics, so its
        width is independent of the number of variables.
        """
        if mask_blocks.ndim != 4:
            raise ValueError(
                "candidate masks must have shape [B,Q,short_len,variables]"
            )
        missing = 1.0 - mask_blocks.float()
        temporal = missing.mean(dim=-1)
        channel_rate = missing.mean(dim=-2)
        overall = missing.mean(dim=(-1, -2), keepdim=False).unsqueeze(-1)
        if temporal.size(-1) > 1:
            transition = torch.abs(
                temporal[..., 1:] - temporal[..., :-1]
            ).mean(dim=-1, keepdim=True)
        else:
            transition = torch.zeros_like(overall)
        channel_std = channel_rate.std(dim=-1, unbiased=False, keepdim=True)
        channel_max = channel_rate.max(dim=-1, keepdim=True).values
        affected = (channel_rate > 0.0).float().mean(dim=-1, keepdim=True)
        return torch.cat(
            [temporal, overall, transition, channel_std, channel_max, affected],
            dim=-1,
        )

    @staticmethod
    def _fractional_percentile_rank(values: torch.Tensor) -> torch.Tensor:
        """Return deterministic [0, 1] fractional ranks with averaged ties."""
        flat = values.reshape(-1)
        if flat.numel() <= 1:
            return torch.ones_like(flat)
        less = (flat.unsqueeze(0) < flat.unsqueeze(1)).sum(dim=1).to(flat.dtype)
        equal = (flat.unsqueeze(0) == flat.unsqueeze(1)).sum(dim=1).to(flat.dtype)
        return (less + 0.5 * (equal - 1.0)) / float(flat.numel() - 1)

    @staticmethod
    def _variance_loss(features: torch.Tensor, target: float) -> torch.Tensor:
        flattened = features.reshape(-1, features.size(-1)) * math.sqrt(features.size(-1))
        std = torch.sqrt(flattened.var(dim=0, unbiased=False) + 1e-4)
        return F.relu(float(target) - std).mean()

    @staticmethod
    def _covariance_loss(features: torch.Tensor) -> torch.Tensor:
        flattened = features.reshape(-1, features.size(-1))
        if flattened.size(0) < 2:
            return flattened.new_zeros(())
        centered = flattened - flattened.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / float(flattened.size(0) - 1)
        diagonal = torch.diagonal(covariance)
        off_diagonal = covariance - torch.diag_embed(diagonal)
        return off_diagonal.pow(2).sum() / float(features.size(-1))

    @staticmethod
    def _cross_covariance_loss(content: torch.Tensor, missingness: torch.Tensor) -> torch.Tensor:
        pooled = content.mean(dim=1)
        if pooled.size(0) < 2:
            return pooled.new_zeros(())
        pooled = pooled - pooled.mean(dim=0, keepdim=True)
        missingness = missingness - missingness.mean(dim=0, keepdim=True)
        cross_covariance = pooled.T @ missingness / float(pooled.size(0) - 1)
        return cross_covariance.pow(2).mean()

    def paired_objective(
        self,
        content_a_raw: torch.Tensor,
        content_a: torch.Tensor,
        mask_a: torch.Tensor,
        content_b_raw: torch.Tensor,
        content_b: torch.Tensor,
        mask_b: torch.Tensor,
        use_factorization: bool,
    ) -> dict[str, torch.Tensor]:
        cosine = F.cosine_similarity(content_a, content_b, dim=-1)
        invariance = (1.0 - cosine).mean()
        variance = 0.5 * (
            self._variance_loss(content_a_raw, self.variance_target)
            + self._variance_loss(content_b_raw, self.variance_target)
        )
        covariance = 0.5 * (
            self._covariance_loss(content_a_raw)
            + self._covariance_loss(content_b_raw)
        )

        zero = invariance.new_zeros(())
        decorrelation = zero
        mask_reconstruction = zero
        if use_factorization:
            missing_a, statistics_a = self.encode_missingness(mask_a)
            missing_b, statistics_b = self.encode_missingness(mask_b)
            decorrelation = 0.5 * (
                self._cross_covariance_loss(content_a, missing_a)
                + self._cross_covariance_loss(content_b, missing_b)
            )
            mask_reconstruction = 0.5 * (
                F.mse_loss(self.mask_decoder(missing_a), statistics_a)
                + F.mse_loss(self.mask_decoder(missing_b), statistics_b)
            )

        return {
            "invariance_loss": invariance,
            "variance_loss": variance,
            "covariance_loss": covariance,
            "decorrelation_loss": decorrelation,
            "mask_reconstruction_loss": mask_reconstruction,
            "paired_content_cosine": cosine.mean(),
        }
