"""Forecast-Utility-Aware Selective Retrieval (FUSR).

FUSR（预测效用感知的选择性原型检索）只使用推理时可获得的历史、
缺失掩码和检索统计估计原型效用。未来目标只能在训练循环中生成
prototype-on/prototype-off 效用标签，不得传入本文件的特征接口。

This module is deliberately independent of S4M's memory implementation.  With
``use_fusr=False`` the caller must not instantiate or invoke it, which keeps the
original S4M parameter set and numerical path unchanged.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorOrScalar = Union[torch.Tensor, float, int]


class MissingnessSpectralFeaturizer(nn.Module):
    """Build the fixed eight-dimensional, target-free FUSR feature vector.

    特征顺序固定为：

    0. Top-1 retrieval score（Top-1 检索得分）；
    1. mean selected score（选中原型平均得分）；
    2. normalized retrieval entropy（归一化检索熵）；
    3. historical missing rate（历史缺失率）；
    4. longest missing-run ratio（最长连续缺失比例）；
    5. cross-variable missing-rate std（变量间缺失率标准差）；
    6. normalized binary mask entropy（归一化掩码二元熵）；
    7. observed/imputed spectral JSD（观测谱与插补谱的 JSD 分歧）。

    ``z_s`` is the already available decay-imputed history, not a new imputer.
    Inputs are detached by default so the utility objective cannot manipulate
    S4M's query representation merely to make routing labels easier.
    """

    FEATURE_NAMES = (
        "top1_similarity",
        "selected_similarity",
        "normalized_retrieval_entropy",
        "missing_rate",
        "longest_missing_run_ratio",
        "variable_missing_rate_std",
        "mask_binary_entropy",
        "spectral_js_divergence",
    )
    FEATURE_DIM = len(FEATURE_NAMES)

    def __init__(self, epsilon: float = 1e-6, detach_inputs: bool = True):
        super().__init__()
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)
        self.detach_inputs = bool(detach_inputs)

    @staticmethod
    def _batch_vector(value: TensorOrScalar, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        """Convert a scalar or per-sample tensor to shape ``[B]``."""
        value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if value.ndim == 0:
            return value.expand(batch_size)
        if value.shape[0] != batch_size:
            raise ValueError(
                f"retrieval statistic has batch {value.shape[0]}, expected {batch_size}"
            )
        if value.ndim > 1:
            value = value.reshape(batch_size, -1).mean(dim=1)
        return value

    @staticmethod
    def _longest_missing_run(mask: torch.Tensor) -> torch.Tensor:
        """Return maximum per-variable missing run divided by sequence length."""
        batch, length, variables = mask.shape
        current = mask.new_zeros((batch, variables))
        longest = mask.new_zeros((batch, variables))
        missing = mask < 0.5
        # L is small for S4M (normally 96); this exact scan avoids approximate
        # pooling rules and is deterministic on CPU and CUDA.
        for time_index in range(length):
            current = (current + 1.0) * missing[:, time_index, :].to(mask.dtype)
            longest = torch.maximum(longest, current)
        return longest.max(dim=1).values / float(max(length, 1))

    def _spectral_js_divergence(self, z_s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Missingness-induced spectral uncertainty in ``[0, 1]``.

        The feature is the Jensen-Shannon divergence between normalized power
        spectra of the observed-only centered history and decay-imputed centered
        history.  Dividing by log(2) gives a bounded, easy-to-monitor signal.
        """
        epsilon = self.epsilon
        observed_count = mask.sum(dim=1).clamp_min(1.0)
        observed_mean = (z_s * mask).sum(dim=1) / observed_count
        observed_centered = (z_s - observed_mean.unsqueeze(1)) * mask
        imputed_centered = z_s - z_s.mean(dim=1, keepdim=True)

        # torch.fft on CPU does not support half precision.  FUSR features are
        # lightweight, so float32 is the safe compute dtype for fp16/bfloat16.
        fft_dtype = (
            torch.float32
            if z_s.dtype in (torch.float16, torch.bfloat16)
            else z_s.dtype
        )
        observed_spectrum = torch.fft.rfft(observed_centered.to(fft_dtype), dim=1)
        imputed_spectrum = torch.fft.rfft(imputed_centered.to(fft_dtype), dim=1)
        observed_power = observed_spectrum.abs().square().mean(dim=2)
        imputed_power = imputed_spectrum.abs().square().mean(dim=2)

        # Centering makes the DC component uninformative; remove it whenever a
        # non-DC bin exists. Degenerate constant histories remain well defined.
        if observed_power.shape[1] > 1:
            observed_power = observed_power[:, 1:]
            imputed_power = imputed_power[:, 1:]

        observed_prob = (observed_power + epsilon) / (
            observed_power.sum(dim=1, keepdim=True)
            + epsilon * observed_power.shape[1]
        )
        imputed_prob = (imputed_power + epsilon) / (
            imputed_power.sum(dim=1, keepdim=True)
            + epsilon * imputed_power.shape[1]
        )
        mixture = 0.5 * (observed_prob + imputed_prob)
        jsd = 0.5 * (
            (observed_prob * (observed_prob.log() - mixture.log())).sum(dim=1)
            + (imputed_prob * (imputed_prob.log() - mixture.log())).sum(dim=1)
        )
        return (jsd / math.log(2.0)).clamp(min=0.0, max=1.0).to(z_s.dtype)

    def forward(
        self,
        z_s: torch.Tensor,
        seq_mask: torch.Tensor,
        top1_similarity: TensorOrScalar,
        selected_similarity: TensorOrScalar,
        retrieval_entropy: TensorOrScalar,
        candidate_count: TensorOrScalar,
        use_spectral: bool = True,
    ) -> torch.Tensor:
        """Return target-free features with shape ``[B, 8]``.

        ``candidate_count`` should be the number of entries over which the
        retrieval softmax was computed (Top-K count, not necessarily full bank
        size); it is used only to normalize entropy.
        """
        if z_s.ndim != 3:
            raise ValueError(f"z_s must have shape [B,L,D], got {tuple(z_s.shape)}")
        if seq_mask.ndim != 3:
            raise ValueError(
                f"seq_mask must have shape [B,L,D] or [B,L,1], got {tuple(seq_mask.shape)}"
            )
        if seq_mask.shape[:2] != z_s.shape[:2]:
            raise ValueError("seq_mask batch/length dimensions must match z_s")
        if seq_mask.shape[2] == 1 and z_s.shape[2] != 1:
            seq_mask = seq_mask.expand(-1, -1, z_s.shape[2])
        elif seq_mask.shape[2] != z_s.shape[2]:
            raise ValueError("seq_mask variable dimension must be 1 or match z_s")

        if self.detach_inputs:
            z_s = z_s.detach()
            seq_mask = seq_mask.detach()
            if isinstance(top1_similarity, torch.Tensor):
                top1_similarity = top1_similarity.detach()
            if isinstance(selected_similarity, torch.Tensor):
                selected_similarity = selected_similarity.detach()
            if isinstance(retrieval_entropy, torch.Tensor):
                retrieval_entropy = retrieval_entropy.detach()
            if isinstance(candidate_count, torch.Tensor):
                candidate_count = candidate_count.detach()

        if not z_s.is_floating_point():
            z_s = z_s.float()
        mask = seq_mask.to(device=z_s.device, dtype=z_s.dtype).clamp(0.0, 1.0)
        batch_size, _, _ = z_s.shape

        top1 = self._batch_vector(top1_similarity, batch_size, z_s)
        selected = self._batch_vector(selected_similarity, batch_size, z_s)
        entropy = self._batch_vector(retrieval_entropy, batch_size, z_s)
        support = self._batch_vector(candidate_count, batch_size, z_s).clamp_min(2.0)
        normalized_entropy = (entropy / support.log()).clamp(0.0, 1.0)

        observation_rate = mask.mean(dim=(1, 2)).clamp(0.0, 1.0)
        missing_rate = 1.0 - observation_rate
        longest_run = self._longest_missing_run(mask)
        variable_missing_rate = 1.0 - mask.mean(dim=1)
        variable_missing_std = variable_missing_rate.std(dim=1, unbiased=False)
        probability = observation_rate.clamp(self.epsilon, 1.0 - self.epsilon)
        mask_entropy = -(
            probability * probability.log()
            + (1.0 - probability) * (1.0 - probability).log()
        ) / math.log(2.0)
        if use_spectral:
            spectral_uncertainty = self._spectral_js_divergence(z_s, mask)
        else:
            # Keeping the eighth coordinate makes spectral ablation compatible
            # with the same router architecture and checkpoint schema.
            spectral_uncertainty = z_s.new_zeros(batch_size)

        features = torch.stack(
            (
                top1,
                selected,
                normalized_entropy,
                missing_rate,
                longest_run,
                variable_missing_std,
                mask_entropy,
                spectral_uncertainty,
            ),
            dim=1,
        )
        return torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)


class RetrievalInterventionFeaturizer(nn.Module):
    """Append six target-free retrieval-intervention features to the base eight.

    The extra features are computed entirely from retrieval/query tensors and
    the prototype-on/prototype-off hidden contexts that are already available
    during inference.  There is deliberately no future target argument.

    8. Top-1 similarity temporal std（Top-1 相似度时间标准差）；
    9. normalized entropy temporal std（归一化检索熵时间标准差）；
    10. retrieved/query norm ratio（检索向量/查询向量范数比）；
    11. retrieved-query cosine mean（检索向量与查询向量平均余弦）；
    12. relative context perturbation norm（上下文相对干预范数）；
    13. prototype-on/off context cosine（开/关原型上下文余弦）。
    """

    EXTRA_FEATURE_NAMES = (
        "top1_similarity_temporal_std",
        "normalized_retrieval_entropy_temporal_std",
        "retrieved_query_norm_ratio",
        "retrieved_query_cosine_mean",
        "context_relative_perturbation_norm",
        "context_on_off_cosine",
    )
    FEATURE_NAMES = MissingnessSpectralFeaturizer.FEATURE_NAMES + EXTRA_FEATURE_NAMES
    FEATURE_DIM = len(FEATURE_NAMES)

    def __init__(self, detach_inputs: bool = True):
        super().__init__()
        self.detach_inputs = bool(detach_inputs)

    def forward(
        self,
        base_features: torch.Tensor,
        top1_temporal_std: TensorOrScalar,
        normalized_entropy_temporal_std: TensorOrScalar,
        retrieved_query_norm_ratio: TensorOrScalar,
        retrieved_query_cosine_mean: TensorOrScalar,
        context_relative_perturbation_norm: TensorOrScalar,
        context_on_off_cosine: TensorOrScalar,
    ) -> torch.Tensor:
        """Return the canonical target-free ``[B,14]`` feature tensor."""
        if base_features.ndim != 2 or base_features.shape[1] != (
            MissingnessSpectralFeaturizer.FEATURE_DIM
        ):
            raise ValueError(
                "base_features must have shape "
                f"[B,{MissingnessSpectralFeaturizer.FEATURE_DIM}], "
                f"got {tuple(base_features.shape)}"
            )
        if not base_features.is_floating_point():
            base_features = base_features.float()
        if self.detach_inputs:
            base_features = base_features.detach()

        batch_size = base_features.shape[0]
        values = (
            top1_temporal_std,
            normalized_entropy_temporal_std,
            retrieved_query_norm_ratio,
            retrieved_query_cosine_mean,
            context_relative_perturbation_norm,
            context_on_off_cosine,
        )
        extras = []
        for value in values:
            if self.detach_inputs and isinstance(value, torch.Tensor):
                value = value.detach()
            extras.append(
                MissingnessSpectralFeaturizer._batch_vector(
                    value,
                    batch_size,
                    base_features,
                )
            )
        features = torch.cat((base_features, torch.stack(extras, dim=1)), dim=1)
        return torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)


class ForecastUtilityRouter(nn.Module):
    """Lightweight utility mean/uncertainty/sign router for selective retrieval."""

    def __init__(
        self,
        feature_dim: Optional[int] = None,
        hidden_dim: int = 32,
        dropout: float = 0.0,
        soft_gate_temperature: float = 0.25,
        gate_threshold: float = 0.0,
        uncertainty_penalty: float = 1.0,
        logvar_min: float = -6.0,
        logvar_max: float = 3.0,
        use_hard_gate_eval: bool = True,
        gate_init_probability: float = 0.9,
        epsilon: float = 1e-6,
        objective_version: str = "v1_joint",
        feature_set: str = "base8",
        gate_score: str = "utility_lcb",
        gate_prior_mode: str = "learned",
    ):
        super().__init__()
        valid_objectives = {"v1_joint", "v2_frozen_balanced_rank"}
        valid_feature_sets = {"base8", "intervention14"}
        valid_gate_scores = {"utility_lcb", "sign_logit"}
        valid_prior_modes = {"learned", "none"}
        if objective_version not in valid_objectives:
            raise ValueError(f"unsupported FUSR objective_version: {objective_version}")
        if feature_set not in valid_feature_sets:
            raise ValueError(f"unsupported FUSR feature_set: {feature_set}")
        if gate_score not in valid_gate_scores:
            raise ValueError(f"unsupported FUSR gate_score: {gate_score}")
        if gate_prior_mode not in valid_prior_modes:
            raise ValueError(f"unsupported FUSR gate_prior_mode: {gate_prior_mode}")
        if feature_set == "intervention14" and objective_version != "v2_frozen_balanced_rank":
            raise ValueError("intervention14 is only defined for v2_frozen_balanced_rank")
        if objective_version == "v2_frozen_balanced_rank" and (
            gate_score != "sign_logit" or gate_prior_mode != "none"
        ):
            raise ValueError(
                "v2_frozen_balanced_rank requires gate_score=sign_logit "
                "and gate_prior_mode=none"
            )

        expected_feature_dim = (
            MissingnessSpectralFeaturizer.FEATURE_DIM
            if feature_set == "base8"
            else RetrievalInterventionFeaturizer.FEATURE_DIM
        )
        if feature_dim is None:
            feature_dim = expected_feature_dim
        elif int(feature_dim) != expected_feature_dim:
            raise ValueError(
                f"feature_dim={feature_dim} does not match {feature_set} "
                f"dimension {expected_feature_dim}"
            )
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if soft_gate_temperature <= 0.0:
            raise ValueError("soft_gate_temperature must be positive")
        if logvar_min >= logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max")
        if not 0.0 < gate_init_probability < 1.0:
            raise ValueError("gate_init_probability must be in (0, 1)")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.soft_gate_temperature = float(soft_gate_temperature)
        self.gate_threshold = float(gate_threshold)
        self.uncertainty_penalty = float(uncertainty_penalty)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.use_hard_gate_eval = bool(use_hard_gate_eval)
        self.epsilon = float(epsilon)
        self.objective_version = str(objective_version)
        self.feature_set = str(feature_set)
        self.gate_score_mode = str(gate_score)
        self.gate_prior_mode = str(gate_prior_mode)

        self.backbone = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(dropout),
        )
        # Outputs are utility mean, log variance, and utility-sign logit.
        self.utility_head = nn.Linear(self.hidden_dim, 3)
        # An untrained uncertainty head starts at logvar=0 (std=1).  Compensate
        # that conservative penalty in the independent gate prior so a newly
        # enabled router starts near the requested prototype-on probability,
        # without corrupting the numerical meaning of predicted utility mean.
        prior_logit = (
            math.log(gate_init_probability / (1.0 - gate_init_probability))
            + (self.uncertainty_penalty + self.gate_threshold)
            / self.soft_gate_temperature
        )
        if self.gate_prior_mode == "learned":
            self.gate_prior_logit = nn.Parameter(
                torch.tensor(prior_logit, dtype=torch.float32)
            )
        else:
            self.register_parameter("gate_prior_logit", None)

        # V2 router-only training fits these statistics on the frozen training
        # cache and calibrates the sign-logit threshold on validation data.
        # V1 ignores both buffers, preserving its original numerical path.
        self.register_buffer("feature_center", torch.zeros(self.feature_dim))
        self.register_buffer("feature_scale", torch.ones(self.feature_dim))
        self.register_buffer("normalization_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("calibrated_threshold", torch.tensor(0.0))
        self.register_buffer("calibration_ready", torch.tensor(False, dtype=torch.bool))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        first_linear = self.backbone[0]
        nn.init.xavier_uniform_(first_linear.weight)
        nn.init.zeros_(first_linear.bias)
        # Zero head weights make the initial behavior deterministic.  logvar=0
        # honestly represents high initial uncertainty; the separate gate prior
        # keeps retrieval enabled until supervised utility evidence is learned.
        nn.init.zeros_(self.utility_head.weight)
        nn.init.zeros_(self.utility_head.bias)

    @staticmethod
    def build_features(
        z_s: torch.Tensor,
        seq_mask: torch.Tensor,
        top1_similarity: TensorOrScalar,
        selected_similarity: TensorOrScalar,
        retrieval_entropy: TensorOrScalar,
        candidate_count: TensorOrScalar,
        use_spectral: bool = True,
        epsilon: float = 1e-6,
        detach_inputs: bool = True,
    ) -> torch.Tensor:
        """Convenience wrapper returning the canonical ``[B,8]`` features."""
        return MissingnessSpectralFeaturizer(
            epsilon=epsilon,
            detach_inputs=detach_inputs,
        )(
            z_s,
            seq_mask,
            top1_similarity,
            selected_similarity,
            retrieval_entropy,
            candidate_count,
            use_spectral=use_spectral,
        )

    @staticmethod
    def build_intervention_features(
        base_features: torch.Tensor,
        top1_temporal_std: TensorOrScalar,
        normalized_entropy_temporal_std: TensorOrScalar,
        retrieved_query_norm_ratio: TensorOrScalar,
        retrieved_query_cosine_mean: TensorOrScalar,
        context_relative_perturbation_norm: TensorOrScalar,
        context_on_off_cosine: TensorOrScalar,
        detach_inputs: bool = True,
    ) -> torch.Tensor:
        """Convenience wrapper for the target-free ``[B,14]`` feature set."""
        return RetrievalInterventionFeaturizer(detach_inputs=detach_inputs)(
            base_features,
            top1_temporal_std,
            normalized_entropy_temporal_std,
            retrieved_query_norm_ratio,
            retrieved_query_cosine_mean,
            context_relative_perturbation_norm,
            context_on_off_cosine,
        )

    @torch.no_grad()
    def fit_feature_normalizer(self, features: torch.Tensor) -> Dict[str, Any]:
        """Fit frozen per-feature center/scale statistics for V2 cache training."""
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must have shape [N,{self.feature_dim}], "
                f"got {tuple(features.shape)}"
            )
        if features.shape[0] == 0:
            raise ValueError("cannot fit feature normalizer on an empty tensor")
        values = features.detach().to(
            device=self.feature_center.device,
            dtype=self.feature_center.dtype,
        )
        if not bool(torch.isfinite(values).all()):
            raise ValueError("feature normalizer requires finite inputs")
        center = values.mean(dim=0)
        scale = values.std(dim=0, unbiased=False).clamp_min(self.epsilon)
        self.feature_center.copy_(center)
        self.feature_scale.copy_(scale)
        self.normalization_ready.fill_(True)
        return {
            "sample_count": int(values.shape[0]),
            "feature_dim": int(values.shape[1]),
            "minimum_scale": float(scale.min().item()),
            "maximum_scale": float(scale.max().item()),
        }

    @torch.no_grad()
    def set_calibrated_threshold(self, threshold: TensorOrScalar) -> float:
        """Lock a finite scalar validation-calibrated deployment threshold."""
        value = torch.as_tensor(
            threshold,
            device=self.calibrated_threshold.device,
            dtype=self.calibrated_threshold.dtype,
        )
        if value.numel() != 1 or not bool(torch.isfinite(value).all()):
            raise ValueError("calibrated threshold must be one finite scalar")
        self.calibrated_threshold.copy_(value.reshape(()))
        self.calibration_ready.fill_(True)
        return float(self.calibrated_threshold.item())

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply frozen V2 normalization; V1 always returns inputs unchanged."""
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must have shape [B,{self.feature_dim}], got {tuple(features.shape)}"
            )
        if self.objective_version != "v2_frozen_balanced_rank" or not bool(
            self.normalization_ready.item()
        ):
            return features
        center = self.feature_center.to(device=features.device, dtype=features.dtype)
        scale = self.feature_scale.to(device=features.device, dtype=features.dtype)
        return torch.nan_to_num(
            (features - center) / scale.clamp_min(self.epsilon),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

    def forward(self, features: torch.Tensor, hard: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        """Predict utility and choose a differentiable or hard retrieval gate.

        Training defaults to ``gate_soft``. Evaluation defaults to a hard gate
        when ``use_hard_gate_eval=True``. If a hard gate is explicitly requested
        during training, a straight-through estimator preserves gradients.
        """
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must have shape [B,{self.feature_dim}], got {tuple(features.shape)}"
            )
        normalized_features = self.normalize_features(features)
        raw = self.utility_head(self.backbone(normalized_features))
        utility_mean = raw[:, 0]
        utility_logvar = raw[:, 1].clamp(self.logvar_min, self.logvar_max)
        sign_logit = raw[:, 2]
        utility_std = torch.exp(0.5 * utility_logvar).clamp_min(self.epsilon)

        conservative_utility = (
            utility_mean
            - self.uncertainty_penalty * utility_std
            - self.gate_threshold
        )
        if self.gate_score_mode == "utility_lcb":
            gate_score = conservative_utility
            # This is the exact V1 expression: gate_threshold is already part
            # of conservative_utility and the learned prior is added below.
            gate_logit = gate_score / self.soft_gate_temperature
        else:
            gate_score = sign_logit
            calibrated_threshold = self.calibrated_threshold.to(
                device=gate_score.device,
                dtype=gate_score.dtype,
            )
            gate_logit = (
                gate_score - calibrated_threshold
            ) / self.soft_gate_temperature
        if self.gate_prior_logit is not None:
            gate_logit = gate_logit + self.gate_prior_logit.to(gate_logit.dtype)
        gate_soft = torch.sigmoid(gate_logit)
        gate_hard = (gate_soft >= 0.5).to(gate_soft.dtype)

        if hard is None:
            hard = self.use_hard_gate_eval and not self.training
        if hard and self.training:
            gate = gate_soft + (gate_hard - gate_soft).detach()
        elif hard:
            gate = gate_hard
        else:
            gate = gate_soft

        return {
            "utility_mean": utility_mean,
            "utility_logvar": utility_logvar,
            "utility_std": utility_std,
            "sign_logit": sign_logit,
            "conservative_utility": conservative_utility,
            "gate_score": gate_score,
            "gate_logit": gate_logit,
            "gate_soft": gate_soft,
            "gate_hard": gate_hard,
            "gate": gate,
        }

    def utility_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        sign_weight: float = 0.5,
        positive_margin: float = 0.0,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Heteroscedastic utility NLL plus positive-utility sign BCE.

        ``target`` must already be detached by the caller from paired on/off
        forecast losses. Non-finite and explicitly invalid samples are excluded.
        The method returns a differentiable zero when no valid label exists.
        """
        if sign_weight < 0.0:
            raise ValueError("sign_weight must be non-negative")
        utility_mean = outputs["utility_mean"]
        utility_logvar = outputs["utility_logvar"]
        sign_logit = outputs["sign_logit"]
        target = target.detach().to(device=utility_mean.device, dtype=utility_mean.dtype).reshape(-1)
        if target.shape != utility_mean.shape:
            raise ValueError(
                f"target shape {tuple(target.shape)} must match utility shape {tuple(utility_mean.shape)}"
            )

        valid = torch.isfinite(target)
        valid = valid & torch.isfinite(utility_mean) & torch.isfinite(utility_logvar)
        valid = valid & torch.isfinite(sign_logit)
        if valid_mask is not None:
            valid_mask = valid_mask.detach().to(device=valid.device).reshape(-1).bool()
            if valid_mask.shape != valid.shape:
                raise ValueError("valid_mask shape must match target")
            valid = valid & valid_mask

        if sample_weight is None:
            weights = torch.ones_like(target)
        else:
            weights = sample_weight.detach().to(target).reshape(-1).clamp_min(0.0)
            if weights.shape != target.shape:
                raise ValueError("sample_weight shape must match target")
            valid = valid & torch.isfinite(weights)

        valid_count = int(valid.sum().item())
        if valid_count == 0:
            zero = (utility_mean.sum() + utility_logvar.sum() + sign_logit.sum()) * 0.0
            return {
                "total_loss": zero,
                "utility_nll": zero,
                "sign_loss": zero,
                "valid_count": 0,
                "sign_accuracy": zero.detach(),
                "positive_rate": zero.detach(),
            }

        mean_valid = utility_mean[valid]
        logvar_valid = utility_logvar[valid]
        sign_valid = sign_logit[valid]
        target_valid = target[valid]
        weight_valid = weights[valid]
        denominator = weight_valid.sum().clamp_min(self.epsilon)

        squared_error = (mean_valid - target_valid).square()
        nll_per_sample = 0.5 * (
            torch.exp(-logvar_valid) * squared_error
            + logvar_valid
            + math.log(2.0 * math.pi)
        )
        utility_nll = (nll_per_sample * weight_valid).sum() / denominator
        positive_target = (target_valid > positive_margin).to(sign_valid.dtype)
        sign_per_sample = F.binary_cross_entropy_with_logits(
            sign_valid,
            positive_target,
            reduction="none",
        )
        sign_loss = (sign_per_sample * weight_valid).sum() / denominator
        total_loss = utility_nll + float(sign_weight) * sign_loss
        with torch.no_grad():
            sign_accuracy = (
                ((sign_valid >= 0.0).to(positive_target.dtype) == positive_target)
                .to(weight_valid.dtype)
                .mul(weight_valid)
                .sum()
                / denominator
            )
            positive_rate = (positive_target * weight_valid).sum() / denominator

        return {
            "total_loss": total_loss,
            "utility_nll": utility_nll,
            "sign_loss": sign_loss,
            "valid_count": valid_count,
            "sign_accuracy": sign_accuracy,
            "positive_rate": positive_rate,
        }


__all__ = [
    "MissingnessSpectralFeaturizer",
    "RetrievalInterventionFeaturizer",
    "ForecastUtilityRouter",
]
