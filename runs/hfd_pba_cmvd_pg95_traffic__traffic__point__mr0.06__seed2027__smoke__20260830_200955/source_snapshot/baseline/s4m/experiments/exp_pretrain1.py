import random
from thop import profile

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric,MAE,MSE,RMSE,MAPE,MSPE
import torch
import torch.nn as nn
from torch import optim
import os
import time
import wandb
import warnings
import numpy as np
import pandas as pd
import json
import copy
import hashlib
import math

from torchstat import stat
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

DMRB_TEACHER_RECONSTRUCTION_ATOL = float(32.0 * np.finfo(np.float32).eps)

def save_model1(i,iter,model,path):
    path = path + '/' + 'checkpoint.pth'
    path = path[:-4]+str(i)+"_"+str(iter)+path[-4:]
    print("save_model1",path)
    dic = {}
    # dic['model'] = model.state_dict()
    # dic['memory'] = model.mem_net.memory.memory
    dic['repre'] = model.mem_net.memory.repre
    dic['num_dic'] = model.mem_net.memory.num_dic
    torch.save(dic,path)    
    
def load_model1(i,iter,model,path):
    path = path[:-4]+str(i)+"_"+str(iter)+path[-4:]
    print(path)
    dic = torch.load(path)
    # model.load_state_dict(dic['model'],strict=False)
    # model.mem_net.memory.memory = dic['memory']
    model.mem_net.memory.repre = dic['repre']
    model.mem_net.memory.num_dic = dic['num_dic']
    return model





class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        parameters = self.model.parameters()
        if (
            bool(getattr(self.args, 'use_mrra', False))
            and str(getattr(self.args, 'mrra_version', 'v1'))
            == 'v3_gradient_isolated_two_timescale'
        ):
            parameters = self._build_mrra_parameter_groups(
                self.model,
                float(self.args.learning_rate),
                float(self.args.mrra_lr_multiplier),
            )
        if self.args.optimizer == 'adam':
            model_optim = optim.Adam(parameters, lr=self.args.learning_rate)
        else:
            model_optim = optim.SGD(parameters, lr = self.args.learning_rate, momentum=0.9,weight_decay=1e-5)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _core_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    @staticmethod
    def _build_mrra_parameter_groups(model, base_lr, mrra_lr_multiplier):
        backbone_parameters = []
        mrra_parameters = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith('mrra.') or '.mrra.' in name:
                mrra_parameters.append(parameter)
            else:
                backbone_parameters.append(parameter)
        if not backbone_parameters or not mrra_parameters:
            raise RuntimeError('MRRA-v3 optimizer requires non-empty disjoint groups')
        if set(map(id, backbone_parameters)) & set(map(id, mrra_parameters)):
            raise RuntimeError('MRRA-v3 optimizer parameter groups overlap')
        return [
            {
                'params': backbone_parameters,
                'lr': float(base_lr),
                'lr_scale': 1.0,
                'group_name': 'backbone',
            },
            {
                'params': mrra_parameters,
                'lr': float(base_lr) * float(mrra_lr_multiplier),
                'lr_scale': float(mrra_lr_multiplier),
                'group_name': 'mrra',
            },
        ]

    def _mrra_parameter_diagnostics(self):
        mrra = self._core_model().mrra
        if mrra is None:
            raise RuntimeError('MRRA diagnostics requested without an adapter')
        gradient_square_sum = 0.0
        parameter_abs_max = 0.0
        gradient_found = False
        for parameter in mrra.parameters():
            if not bool(torch.isfinite(parameter).all().item()):
                raise FloatingPointError(
                    'NAN_OR_DIVERGENCE: non-finite MRRA parameter after optimizer step'
                )
            if parameter.numel():
                parameter_abs_max = max(
                    parameter_abs_max,
                    float(parameter.detach().abs().max().item()),
                )
            if parameter.grad is not None:
                gradient_found = True
                if not bool(torch.isfinite(parameter.grad).all().item()):
                    raise FloatingPointError(
                        'NAN_OR_DIVERGENCE: non-finite MRRA gradient before optimizer step'
                    )
                gradient_square_sum += float(
                    parameter.grad.detach().double().pow(2).sum().item()
                )
        return {
            'gradient_found': gradient_found,
            'gradient_l2': math.sqrt(gradient_square_sum),
            'parameter_abs_max': parameter_abs_max,
        }

    @staticmethod
    def _masked_sample_mse(prediction, target, mask, epsilon=1e-8):
        reduce_dims = tuple(range(1, prediction.ndim))
        observed = mask.sum(dim=reduce_dims)
        squared_error = ((prediction - target) ** 2 * mask).sum(dim=reduce_dims)
        return squared_error / observed.clamp_min(epsilon), observed > 0

    @staticmethod
    def _mrra_reconstruction_objective(
        reconstruction,
        complete_history,
        observation_mask,
        epsilon=1e-8,
    ):
        """Supervise MRRA only where the synthetic history is missing."""
        if not (
            reconstruction.shape
            == complete_history.shape
            == observation_mask.shape
        ):
            raise ValueError('MRRA reconstruction, target, and mask shapes must match')
        missing = 1.0 - observation_mask.to(dtype=reconstruction.dtype)
        missing_count = missing.sum()
        if float(missing_count.detach().item()) <= 0.0:
            raise ValueError('MRRA requires at least one missing history element')
        squared_error = (reconstruction - complete_history).pow(2) * missing
        return squared_error.sum() / missing_count.clamp_min(epsilon), int(
            missing_count.detach().item()
        )

    @staticmethod
    def _rank_adaptive_dual_risk_objective(
        prediction,
        target,
        target_mask,
        epsilon=1e-8,
    ):
        """Balance hard-sample squared risk and body absolute risk.

        Difficulty ranks are detached, so the model cannot manipulate the
        assignment operation.  Both risks are normalized by their detached
        batch means before an equal-weight combination, then returned to the
        MSE scale to preserve the optimizer's approximate gradient magnitude.
        """
        if not prediction.shape == target.shape == target_mask.shape:
            raise ValueError('RADR prediction, target, and mask shapes must match')
        weights = target_mask.to(dtype=prediction.dtype)
        counts = weights.reshape(weights.size(0), -1).sum(dim=1)
        valid = counts > 0
        if not bool(valid.any()):
            raise ValueError('RADR requires at least one observed target per batch')
        counts = counts.clamp_min(1.0)
        sample_mse = (
            (prediction - target).pow(2) * weights
        ).reshape(weights.size(0), -1).sum(dim=1) / counts
        sample_mae = (
            (prediction - target).abs() * weights
        ).reshape(weights.size(0), -1).sum(dim=1) / counts
        sample_mse = sample_mse[valid]
        sample_mae = sample_mae[valid]

        sample_count = sample_mse.numel()
        if sample_count == 1:
            difficulty_rank = sample_mse.new_full((1,), 0.5)
        else:
            order = torch.argsort(sample_mse.detach(), stable=True)
            difficulty_rank = sample_mse.new_empty(sample_count)
            difficulty_rank[order] = torch.linspace(
                0.0, 1.0, sample_count, device=sample_mse.device,
                dtype=sample_mse.dtype,
            )
        difficulty_rank = difficulty_rank.detach()
        tail_weight = difficulty_rank
        body_weight = 1.0 - difficulty_rank
        tail_mse = (tail_weight * sample_mse).sum() / tail_weight.sum().clamp_min(epsilon)
        body_mae = (body_weight * sample_mae).sum() / body_weight.sum().clamp_min(epsilon)
        mse_scale = sample_mse.mean().detach().clamp_min(epsilon)
        mae_scale = sample_mae.mean().detach().clamp_min(epsilon)
        normalized_tail_risk = tail_mse / mse_scale
        normalized_body_risk = body_mae / mae_scale
        total_loss = 0.5 * mse_scale * (
            normalized_tail_risk + normalized_body_risk
        )
        body_mse = (body_weight * sample_mse).sum() / body_weight.sum().clamp_min(epsilon)
        return {
            'total_loss': total_loss,
            'sample_count': int(sample_count),
            'tail_mse': tail_mse,
            'body_mse': body_mse,
            'body_mae': body_mae,
            'normalized_tail_risk': normalized_tail_risk,
            'normalized_body_risk': normalized_body_risk,
            'difficulty_rank_std': difficulty_rank.std(unbiased=False),
            'hard_body_mse_gap': tail_mse - body_mse,
        }

    def _fusr_training_objective(self, fusr_outputs, target, target_mask):
        """Create train-split utility supervision without exposing future targets to the router."""
        prediction_on = fusr_outputs.get('prediction_on')
        prediction_off = fusr_outputs.get('prediction_off')
        if prediction_on is None or prediction_off is None:
            raise RuntimeError('FUSR utility supervision requires on/off predictions')

        mse_on, valid_on = self._masked_sample_mse(prediction_on, target, target_mask)
        mse_off, valid_off = self._masked_sample_mse(prediction_off, target, target_mask)
        valid = valid_on & valid_off
        if not bool(valid.any()):
            return None

        epsilon = 1e-8
        utility_target = ((mse_off - mse_on) / mse_off.clamp_min(epsilon)).clamp(
            -float(self.args.fusr_utility_clip),
            float(self.args.fusr_utility_clip),
        ).detach()
        utility_mean = fusr_outputs['utility_mean']
        utility_logvar = fusr_outputs['utility_logvar'].clamp(-8.0, 4.0)
        sign_logit = fusr_outputs['sign_logit']
        delta = float(self.args.fusr_delta)
        margin = max(float(self.args.fusr_neutral_margin), epsilon)
        sample_weight = ((utility_target - delta).abs() / margin).clamp(max=1.0)
        sample_weight = sample_weight * valid.float()
        weight_sum = sample_weight.sum().clamp_min(epsilon)

        huber = torch.nn.functional.smooth_l1_loss(
            utility_mean, utility_target, reduction='none'
        )
        regression_per_sample = torch.exp(-utility_logvar) * huber + 0.5 * utility_logvar
        regression_loss = (regression_per_sample * sample_weight).sum() / weight_sum
        sign_target = (utility_target > delta).float()
        sign_per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
            sign_logit, sign_target, reduction='none'
        )
        sign_loss = (sign_per_sample * sample_weight).sum() / weight_sum
        total_loss = float(self.args.fusr_util_weight) * (
            regression_loss + float(self.args.fusr_sign_weight) * sign_loss
        )

        valid_target = utility_target[valid]
        valid_mean = utility_mean[valid]
        valid_std = torch.exp(0.5 * utility_logvar[valid])
        predicted_sign = (fusr_outputs['utility_lcb'][valid] > delta)
        true_sign = valid_target > delta
        gate_soft = fusr_outputs['gate_soft'][valid]
        gate_hard = fusr_outputs['gate_hard'][valid]
        return {
            'total_loss': total_loss,
            'regression_loss': regression_loss,
            'sign_loss': sign_loss,
            'utility_pair_count': int(valid.sum().item()),
            'utility_target_mean': valid_target.mean().detach(),
            'utility_target_std': valid_target.std(unbiased=False).detach(),
            'utility_positive_rate': true_sign.float().mean().detach(),
            'utility_prediction_mean': valid_mean.mean().detach(),
            'utility_prediction_std': valid_mean.std(unbiased=False).detach(),
            'utility_uncertainty_mean': valid_std.mean().detach(),
            'utility_sign_accuracy': (predicted_sign == true_sign).float().mean().detach(),
            'gate_mean': gate_soft.mean().detach(),
            'abstention_rate': (1.0 - gate_hard).mean().detach(),
            'spectral_uncertainty': fusr_outputs['features'][valid, 7].mean().detach(),
        }

    @staticmethod
    def _fusr_v2_hash_value(hasher, value):
        """Deterministically hash tensors and nested prototype-memory objects."""
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            hasher.update(b'tensor')
            hasher.update(str(tensor.dtype).encode('utf-8'))
            hasher.update(str(tuple(tensor.shape)).encode('utf-8'))
            hasher.update(tensor.numpy().tobytes())
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            hasher.update(b'ndarray')
            hasher.update(str(array.dtype).encode('utf-8'))
            hasher.update(str(array.shape).encode('utf-8'))
            hasher.update(array.tobytes())
        elif isinstance(value, dict):
            hasher.update(b'dict')
            for key in sorted(value, key=lambda item: str(item)):
                hasher.update(str(key).encode('utf-8'))
                Exp_Long_Term_Forecast._fusr_v2_hash_value(hasher, value[key])
        elif isinstance(value, (list, tuple)):
            hasher.update(type(value).__name__.encode('utf-8'))
            for item in value:
                Exp_Long_Term_Forecast._fusr_v2_hash_value(hasher, item)
        else:
            hasher.update(repr(value).encode('utf-8'))

    def _fusr_v2_backbone_hash(self):
        hasher = hashlib.sha256()
        for key, value in sorted(self._core_model().state_dict().items()):
            if '.fusr.' in key:
                continue
            hasher.update(key.encode('utf-8'))
            self._fusr_v2_hash_value(hasher, value)
        return hasher.hexdigest()

    def _fusr_v2_memory_hash(self):
        memory = self._core_model().mem_net.memory
        hasher = hashlib.sha256()
        for key in ('memory', 'repre', 'num_dic'):
            hasher.update(key.encode('utf-8'))
            self._fusr_v2_hash_value(hasher, getattr(memory, key))
        return hasher.hexdigest()

    @staticmethod
    def _file_sha256(path):
        hasher = hashlib.sha256()
        with open(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _fusr_v2_balanced_accuracy(labels, decisions):
        labels = np.asarray(labels, dtype=np.bool_)
        decisions = np.asarray(decisions, dtype=np.bool_)
        positive = labels
        negative = ~labels
        if not positive.any() or not negative.any():
            return None
        sensitivity = float(np.mean(decisions[positive]))
        specificity = float(np.mean(~decisions[negative]))
        return 0.5 * (sensitivity + specificity)

    @staticmethod
    def _fusr_v2_average_precision(labels, scores):
        labels = np.asarray(labels, dtype=np.bool_)
        scores = np.asarray(scores, dtype=np.float64)
        positives = int(labels.sum())
        if positives == 0:
            return None
        order = np.argsort(-scores, kind='mergesort')
        ranked = labels[order].astype(np.float64)
        precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
        return float(np.sum(precision * ranked) / positives)

    @staticmethod
    def _fusr_v2_split_indices(sample_count, fit_fraction, dev_fraction, requested_embargo):
        """Create chronological fit/dev/calibration blocks with two embargoes."""
        if sample_count < 12:
            raise RuntimeError(
                f'FUSR V2 cache has only {sample_count} valid samples; at least 12 are required'
            )
        fit_count = max(4, int(sample_count * fit_fraction))
        dev_count = max(4, int(sample_count * dev_fraction))
        # Preserve at least four calibration samples. Smoke configurations may
        # request an embargo larger than their deliberately capped cache; only
        # the effective smoke embargo is reduced and is recorded in metrics.
        max_embargo = max(0, (sample_count - fit_count - dev_count - 4) // 2)
        embargo = min(int(requested_embargo), max_embargo)
        dev_start = fit_count + embargo
        dev_end = dev_start + dev_count
        calibration_start = dev_end + embargo
        if calibration_start >= sample_count - 3:
            raise RuntimeError('FUSR V2 chronological split leaves no calibration block')
        return {
            'fit': torch.arange(0, fit_count, dtype=torch.long),
            'dev': torch.arange(dev_start, dev_end, dtype=torch.long),
            'calibration': torch.arange(calibration_start, sample_count, dtype=torch.long),
            'effective_embargo_windows': embargo,
        }

    def _fusr_v2_make_cache(self, data_set):
        """Generate a stationary validation counterfactual cache without test access."""
        if self.args.fusr_cache_split != 'val':
            raise ValueError('FUSR V2 currently permits validation cache only')
        loader = DataLoader(
            data_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False,
        )
        core = self._core_model()
        f_dim = -1 if self.args.features == 'MS' else 0
        features_all = []
        mse_on_all = []
        mse_off_all = []
        mae_on_all = []
        mae_off_all = []
        complete_mse_on_all = []
        complete_mse_off_all = []
        sample_indices = []
        next_index = 0
        feature_repeat_max_error = 0.0
        batches = 0

        self.model.eval()
        self.change_mode(0)
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if self.args.fusr_cache_max_batches >= 0 and batch_index >= self.args.fusr_cache_max_batches:
                    break
                (
                    seq_x, seq_x_mask, seq_y, seq_y_mask, seq_true_x, seq_true_y,
                    max_idx, min_idx, max_value, min_value,
                ) = [item.float().to(self.device) for item in batch]
                batch_size = int(seq_x.shape[0])
                target = seq_y[:, -self.args.pred_len:, f_dim:]
                target_mask = seq_y_mask[:, -self.args.pred_len:, f_dim:]
                complete_target = seq_true_y[:, -self.args.pred_len:, f_dim:]

                core.set_fusr_mode('on')
                prediction_on = self.model(
                    seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                )[:, -self.args.pred_len:, f_dim:]
                on_outputs = core.get_last_fusr_outputs()
                if on_outputs is None:
                    raise RuntimeError('FUSR V2 cache received no on-path router features')
                features_on = on_outputs['features'].detach()

                core.set_fusr_mode('off')
                prediction_off = self.model(
                    seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                )[:, -self.args.pred_len:, f_dim:]
                off_outputs = core.get_last_fusr_outputs()
                if off_outputs is None:
                    raise RuntimeError('FUSR V2 cache received no off-path router features')
                features_off = off_outputs['features'].detach()
                feature_repeat_max_error = max(
                    feature_repeat_max_error,
                    float(torch.max(torch.abs(features_on - features_off)).item()),
                )

                reduce_dims = tuple(range(1, target.ndim))
                observed_count = target_mask.sum(dim=reduce_dims)
                valid = observed_count > 0
                squared_on = (((prediction_on - target) ** 2) * target_mask).sum(dim=reduce_dims)
                squared_off = (((prediction_off - target) ** 2) * target_mask).sum(dim=reduce_dims)
                absolute_on = (torch.abs(prediction_on - target) * target_mask).sum(dim=reduce_dims)
                absolute_off = (torch.abs(prediction_off - target) * target_mask).sum(dim=reduce_dims)
                denominator = observed_count.clamp_min(1.0)
                complete_mse_on = torch.mean(
                    (prediction_on - complete_target) ** 2, dim=reduce_dims
                )
                complete_mse_off = torch.mean(
                    (prediction_off - complete_target) ** 2, dim=reduce_dims
                )

                batch_indices = torch.arange(
                    next_index, next_index + batch_size, dtype=torch.long,
                    device=valid.device,
                )
                next_index += batch_size
                features_all.append(features_on[valid].cpu())
                mse_on_all.append((squared_on / denominator)[valid].cpu())
                mse_off_all.append((squared_off / denominator)[valid].cpu())
                mae_on_all.append((absolute_on / denominator)[valid].cpu())
                mae_off_all.append((absolute_off / denominator)[valid].cpu())
                complete_mse_on_all.append(complete_mse_on[valid].cpu())
                complete_mse_off_all.append(complete_mse_off[valid].cpu())
                sample_indices.append(batch_indices[valid].cpu())
                batches += 1
        core.set_fusr_mode('route')
        if not features_all:
            raise RuntimeError('FUSR V2 cache contains no valid observed-future samples')
        cache = {
            'features': torch.cat(features_all, dim=0),
            'mse_on': torch.cat(mse_on_all, dim=0),
            'mse_off': torch.cat(mse_off_all, dim=0),
            'mae_on': torch.cat(mae_on_all, dim=0),
            'mae_off': torch.cat(mae_off_all, dim=0),
            'complete_mse_on': torch.cat(complete_mse_on_all, dim=0),
            'complete_mse_off': torch.cat(complete_mse_off_all, dim=0),
            'sample_index': torch.cat(sample_indices, dim=0),
            'source_split': 'validation',
            'used_test_targets': False,
            'batches': batches,
            'feature_repeat_max_error': feature_repeat_max_error,
        }
        hasher = hashlib.sha256()
        for key in sorted(cache):
            hasher.update(key.encode('utf-8'))
            self._fusr_v2_hash_value(hasher, cache[key])
        cache['sha256'] = hasher.hexdigest()
        return cache

    @staticmethod
    def _fusr_v2_router_loss(
        router,
        features,
        utility,
        labels,
        rank_weight,
        class_weights=None,
    ):
        outputs = router(features, hard=False)
        scores = outputs['sign_logit']
        positive_count = labels.sum().clamp_min(1.0)
        negative_count = (1.0 - labels).sum().clamp_min(1.0)
        sample_count = labels.numel()
        if class_weights is None:
            positive_weight = float(sample_count) / (2.0 * float(positive_count.item()))
            negative_weight = float(sample_count) / (2.0 * float(negative_count.item()))
        else:
            positive_weight, negative_weight = class_weights
        class_weight = torch.where(
            labels > 0.5,
            labels.new_tensor(float(positive_weight)),
            labels.new_tensor(float(negative_weight)),
        )
        magnitude = utility.abs().clamp_min(1e-4)
        weights = class_weight * magnitude
        decision_each = torch.nn.functional.binary_cross_entropy_with_logits(
            scores, labels, reduction='none'
        )
        decision_loss = (decision_each * weights).sum() / weights.sum().clamp_min(1e-8)

        positive_index = torch.nonzero(labels > 0.5, as_tuple=False).flatten()
        negative_index = torch.nonzero(labels <= 0.5, as_tuple=False).flatten()
        pair_count = min(int(positive_index.numel()), int(negative_index.numel()))
        if pair_count > 0:
            positive_index = positive_index[:pair_count]
            negative_index = negative_index[:pair_count]
            difference = scores[positive_index] - scores[negative_index]
            pair_weight = (
                utility[positive_index] - utility[negative_index]
            ).abs().clamp_min(1e-4)
            ranking_each = torch.nn.functional.softplus(-difference)
            ranking_loss = (ranking_each * pair_weight).sum() / pair_weight.sum()
        else:
            ranking_loss = scores.sum() * 0.0
        total_loss = decision_loss + float(rank_weight) * ranking_loss
        return total_loss, decision_loss, ranking_loss, pair_count, outputs

    def _fusr_v2_evaluate_router(self, router, features, utility, labels):
        router.eval()
        with torch.no_grad():
            outputs = router(features.to(self.device), hard=False)
            scores = outputs['sign_logit'].detach().cpu().numpy()
        labels_np = labels.cpu().numpy().astype(np.bool_)
        decisions = scores >= 0.0
        return {
            'scores': scores,
            'auroc': self._binary_auc(labels_np, scores),
            'auprc': self._fusr_v2_average_precision(labels_np, scores),
            'balanced_accuracy_at_zero': self._fusr_v2_balanced_accuracy(
                labels_np, decisions
            ),
            'score_mean': float(np.mean(scores)),
            'score_std': float(np.std(scores)),
            'utility_mean': float(utility.mean().item()),
            'utility_std': float(utility.std(unbiased=False).item()),
            'positive_rate': float(labels.float().mean().item()),
        }

    @staticmethod
    def _fusr_v2_calibrate_threshold(scores, mse_on, mse_off, tie_break):
        scores = np.asarray(scores, dtype=np.float64)
        mse_on = np.asarray(mse_on, dtype=np.float64)
        mse_off = np.asarray(mse_off, dtype=np.float64)
        order = np.argsort(-scores, kind='mergesort')
        sorted_scores = scores[order]
        sorted_on = mse_on[order]
        sorted_off = mse_off[order]
        prefix_on = np.concatenate(([0.0], np.cumsum(sorted_on)))
        prefix_off = np.concatenate(([0.0], np.cumsum(sorted_off)))
        total_off = float(prefix_off[-1])
        boundaries = [0]
        boundaries.extend(
            index for index in range(1, scores.size)
            if sorted_scores[index] != sorted_scores[index - 1]
        )
        boundaries.append(scores.size)
        risks = {
            int(k): float((prefix_on[k] + total_off - prefix_off[k]) / scores.size)
            for k in boundaries
        }
        best_risk = min(risks.values())
        tolerance = max(1e-12, abs(best_risk) * 1e-12)
        best = [k for k, risk in risks.items() if abs(risk - best_risk) <= tolerance]
        selected = max(best) if tie_break == 'prefer_on' else min(best)
        if selected == 0:
            threshold = float(np.nextafter(sorted_scores[0], np.inf))
        elif selected == scores.size:
            threshold = float(np.nextafter(sorted_scores[-1], -np.inf))
        else:
            threshold = float(0.5 * (
                sorted_scores[selected - 1] + sorted_scores[selected]
            ))
        gate = scores >= threshold
        routed = np.where(gate, mse_on, mse_off)
        return {
            'threshold': threshold,
            'gate': gate,
            'gate_rate': float(np.mean(gate)),
            'risk': float(np.mean(routed)),
            'always_on_risk': float(np.mean(mse_on)),
            'always_off_risk': float(np.mean(mse_off)),
            'selected_count': int(gate.sum()),
        }

    def _train_fusr_v2(self, setting, validation_data, path):
        """Fit a post-hoc router on a frozen Step-5 S4M checkpoint."""
        source_checkpoint = os.path.abspath(self.args.fusr_backbone_checkpoint)
        if not os.path.isfile(source_checkpoint):
            raise FileNotFoundError(f'FUSR V2 backbone checkpoint missing: {source_checkpoint}')
        output_checkpoint = os.path.join(path, 'checkpoint.pth')
        if os.path.abspath(output_checkpoint) == source_checkpoint:
            raise RuntimeError('FUSR V2 must not overwrite its source backbone checkpoint')

        self.load_model(source_checkpoint)
        core = self._core_model()
        router = core.mem_net.fusr
        if router is None:
            raise RuntimeError('FUSR V2 configuration did not instantiate a router')
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for parameter in router.parameters():
            parameter.requires_grad = True
        backbone_frozen = all(
            not parameter.requires_grad
            for name, parameter in self.model.named_parameters()
            if '.fusr.' not in name
        )
        if not backbone_frozen:
            raise RuntimeError('FUSR V2 failed to freeze all backbone parameters')
        self.model.eval()
        self.change_mode(0)

        backbone_hash_before = self._fusr_v2_backbone_hash()
        memory_hash_before = self._fusr_v2_memory_hash()
        cache = self._fusr_v2_make_cache(validation_data)
        backbone_hash_after_cache = self._fusr_v2_backbone_hash()
        memory_hash_after_cache = self._fusr_v2_memory_hash()
        if backbone_hash_after_cache != backbone_hash_before:
            raise RuntimeError('FUSR V2 counterfactual caching changed the frozen backbone')
        memory_unchanged = memory_hash_after_cache == memory_hash_before
        if not memory_unchanged:
            raise RuntimeError('FUSR V2 counterfactual caching changed prototype memory')
        if cache['feature_repeat_max_error'] != 0.0:
            raise RuntimeError(
                'FUSR V2 on/off feature extraction was not deterministic: '
                f"max_error={cache['feature_repeat_max_error']}"
            )

        run_root = os.path.dirname(self.args.metrics_path)
        cache_path = os.path.join(run_root, 'fusr_v2_validation_cache.pt')
        cache_temporary = cache_path + '.tmp'
        torch.save(cache, cache_temporary)
        os.replace(cache_temporary, cache_path)

        features = cache['features'].float()
        mse_on = cache['mse_on'].float()
        mse_off = cache['mse_off'].float()
        symmetric_denominator = (0.5 * (mse_on + mse_off)).clamp_min(1e-8)
        utility = ((mse_off - mse_on) / symmetric_denominator).clamp(
            -float(self.args.fusr_utility_clip), float(self.args.fusr_utility_clip)
        )
        labels = (mse_off - mse_on > float(self.args.fusr_delta)).float()
        split = self._fusr_v2_split_indices(
            int(features.shape[0]),
            float(self.args.fusr_fit_fraction),
            float(self.args.fusr_dev_fraction),
            int(self.args.fusr_embargo_windows),
        )
        fit_index = split['fit']
        dev_index = split['dev']
        calibration_index = split['calibration']
        fit_labels = labels[fit_index]
        positive_count = int(fit_labels.sum().item())
        negative_count = int(fit_labels.numel() - positive_count)
        if positive_count < 1 or negative_count < 1:
            raise RuntimeError(
                'FUSR V2 chronological fit block needs both beneficial and harmful samples; '
                f'positive={positive_count}, negative={negative_count}'
            )
        fit_class_weights = (
            float(fit_labels.numel()) / (2.0 * positive_count),
            float(fit_labels.numel()) / (2.0 * negative_count),
        )

        router.fit_feature_normalizer(features[fit_index].to(self.device))
        optimizer = optim.AdamW(
            router.parameters(),
            lr=float(self.args.fusr_router_lr),
            weight_decay=float(self.args.fusr_router_weight_decay),
        )
        batch_size = int(self.args.fusr_router_batch_size)
        best_state = None
        best_dev_loss = float('inf')
        best_epoch = 0
        stale_epochs = 0
        gradient_steps = 0
        rank_pairs_generated = 0
        training_total_sum = 0.0
        training_decision_sum = 0.0
        training_rank_sum = 0.0
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(self.args.seed) + 8102)

        fit_features = features[fit_index]
        fit_utility = utility[fit_index]
        fit_labels = labels[fit_index]
        for epoch in range(int(self.args.fusr_router_epochs)):
            router.train()
            permutation = torch.randperm(fit_index.numel(), generator=generator)
            for start in range(0, permutation.numel(), batch_size):
                batch_index = permutation[start:start + batch_size]
                batch_features = fit_features[batch_index].to(self.device)
                batch_utility = fit_utility[batch_index].to(self.device)
                batch_labels = fit_labels[batch_index].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                total, decision, ranking, pair_count, _ = self._fusr_v2_router_loss(
                    router, batch_features, batch_utility, batch_labels,
                    float(self.args.fusr_rank_weight),
                    class_weights=fit_class_weights,
                )
                if not torch.isfinite(total):
                    raise RuntimeError('FUSR V2 router loss became non-finite')
                total.backward()
                finite_gradient = any(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all()
                    and float(parameter.grad.detach().abs().sum().item()) > 0.0
                    for parameter in router.parameters()
                )
                if not finite_gradient:
                    raise RuntimeError('FUSR V2 router produced no finite non-zero gradient')
                optimizer.step()
                gradient_steps += 1
                rank_pairs_generated += int(pair_count)
                training_total_sum += float(total.detach().item())
                training_decision_sum += float(decision.detach().item())
                training_rank_sum += float(ranking.detach().item())

            router.eval()
            with torch.no_grad():
                dev_total, _, _, _, _ = self._fusr_v2_router_loss(
                    router,
                    features[dev_index].to(self.device),
                    utility[dev_index].to(self.device),
                    labels[dev_index].to(self.device),
                    float(self.args.fusr_rank_weight),
                    class_weights=fit_class_weights,
                )
            dev_value = float(dev_total.item())
            print(
                f'FUSR_V2_ROUTER epoch={epoch + 1} dev_loss={dev_value:.8f} '
                f'gradient_steps={gradient_steps}',
                flush=True,
            )
            if dev_value < best_dev_loss - 1e-10:
                best_dev_loss = dev_value
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in router.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= int(self.args.fusr_router_patience):
                    break
        if best_state is None:
            raise RuntimeError('FUSR V2 did not produce a router checkpoint')
        router.load_state_dict(best_state)
        router.to(self.device)

        fit_metrics = self._fusr_v2_evaluate_router(
            router, features[fit_index], utility[fit_index], labels[fit_index]
        )
        dev_metrics = self._fusr_v2_evaluate_router(
            router, features[dev_index], utility[dev_index], labels[dev_index]
        )
        calibration_metrics = self._fusr_v2_evaluate_router(
            router,
            features[calibration_index],
            utility[calibration_index],
            labels[calibration_index],
        )
        calibration = self._fusr_v2_calibrate_threshold(
            calibration_metrics['scores'],
            mse_on[calibration_index].numpy(),
            mse_off[calibration_index].numpy(),
            self.args.fusr_calibration_tie_break,
        )
        router.set_calibrated_threshold(calibration['threshold'])
        router.eval()
        self.save_model(output_checkpoint)

        # Reload the exact artifact that test() will use; calibrated buffers and
        # the frozen backbone must survive the round trip.
        self.load_model(output_checkpoint)
        backbone_hash_after_training = self._fusr_v2_backbone_hash()
        memory_hash_after_training = self._fusr_v2_memory_hash()
        if backbone_hash_after_training != backbone_hash_before:
            raise RuntimeError('FUSR V2 router training changed frozen backbone weights')
        if memory_hash_after_training != memory_hash_before:
            raise RuntimeError('FUSR V2 router training changed frozen prototype memory')
        router = self._core_model().mem_net.fusr
        locked_threshold = float(router.calibrated_threshold.detach().cpu().item())
        if not bool(router.calibration_ready.detach().cpu().item()):
            raise RuntimeError('FUSR V2 saved checkpoint is not calibration locked')
        if abs(locked_threshold - calibration['threshold']) > 1e-7:
            raise RuntimeError('FUSR V2 calibrated threshold changed after checkpoint reload')

        complete_utility = cache['complete_mse_off'] - cache['complete_mse_on']
        observed_sign = (mse_off - mse_on) > float(self.args.fusr_delta)
        complete_sign = complete_utility > float(self.args.fusr_delta)
        sign_flip_rate = float((observed_sign != complete_sign).float().mean().item())
        step_denominator = max(1, gradient_steps)
        self.run_evidence = {
            'optimizer_steps': int(gradient_steps),
            'train_batches_processed': int(cache['batches']),
            'validation_batches_processed': int(cache['batches']),
            'memory_bank_warmup_count': 0,
            'mipp_enabled': False,
            'mipp_steps': 0,
            'mipp_pair_views_generated': 0,
            'fusr_enabled': True,
            'fusr_objective_version': 'v2_frozen_balanced_rank',
            'fusr_steps': int(gradient_steps),
            'fusr_router_gradient_steps': int(gradient_steps),
            'fusr_utility_pairs_generated': int(features.shape[0]),
            'fusr_valid_utility_samples': int(features.shape[0]),
            'fusr_cache_samples': int(features.shape[0]),
            'fusr_rank_pairs_generated': int(rank_pairs_generated),
            'fusr_cache_train_samples': int(fit_index.numel()),
            'fusr_cache_validation_samples': int(calibration_index.numel()),
            'fusr_cache_dev_samples': int(dev_index.numel()),
            'fusr_cache_positive_samples': int(positive_count),
            'fusr_cache_negative_samples': int(negative_count),
            'fusr_cache_source_split': 'validation',
            'fusr_cache_scope': 'stationary_validation_diagnostic',
            'fusr_cache_path': cache_path,
            'fusr_cache_sha256': cache['sha256'],
            'fusr_cache_feature_repeat_max_error': float(cache['feature_repeat_max_error']),
            'fusr_effective_embargo_windows': int(split['effective_embargo_windows']),
            'fusr_requested_embargo_windows': int(self.args.fusr_embargo_windows),
            'fusr_calibration_split': 'validation',
            'fusr_calibration_target_source': 'observed_validation_targets_only',
            'fusr_used_test_targets': False,
            'fusr_calibration_used_test_targets': False,
            'fusr_calibration_state': 'CALIBRATION_LOCKED',
            'fusr_calibrated_threshold': float(locked_threshold),
            'fusr_calibration_sample_count': int(calibration_index.numel()),
            'fusr_calibration_gate_rate': float(calibration['gate_rate']),
            'fusr_calibration_abstention_rate': float(1.0 - calibration['gate_rate']),
            'fusr_calibration_objective_before': float(calibration['always_on_risk']),
            'fusr_calibration_objective_after': float(calibration['risk']),
            'fusr_calibration_always_off_mse': float(calibration['always_off_risk']),
            'fusr_gate_mean': float(calibration['gate_rate']),
            'fusr_abstention_rate': float(1.0 - calibration['gate_rate']),
            'fusr_total_loss_mean': float(training_total_sum / step_denominator),
            'fusr_utility_loss_mean': float(training_decision_sum / step_denominator),
            'fusr_utility_nll_mean': None,
            'fusr_sign_loss_mean': float(training_decision_sum / step_denominator),
            'fusr_rank_loss_mean': float(training_rank_sum / step_denominator),
            'fusr_predicted_utility_mean': float(fit_metrics['score_mean']),
            'fusr_utility_std_mean': 0.0,
            'fusr_spectral_uncertainty_mean': float(features[fit_index, 7].mean().item()),
            'fusr_sign_accuracy': float(np.mean(
                (fit_metrics['scores'] >= 0.0) == fit_labels.numpy().astype(np.bool_)
            )),
            'fusr_fit_auroc': fit_metrics['auroc'],
            'fusr_fit_auprc': fit_metrics['auprc'],
            'fusr_fit_balanced_accuracy': fit_metrics['balanced_accuracy_at_zero'],
            'fusr_dev_auroc': dev_metrics['auroc'],
            'fusr_dev_auprc': dev_metrics['auprc'],
            'fusr_calibration_auroc': calibration_metrics['auroc'],
            'fusr_calibration_auprc': calibration_metrics['auprc'],
            'fusr_router_best_epoch': int(best_epoch),
            'fusr_router_best_dev_loss': float(best_dev_loss),
            'fusr_observed_complete_utility_sign_flip_rate': sign_flip_rate,
            'fusr_backbone_frozen': bool(backbone_frozen),
            'fusr_memory_unchanged': True,
            'fusr_source_checkpoint': source_checkpoint,
            'fusr_source_checkpoint_sha256': self._file_sha256(source_checkpoint),
            'fusr_backbone_sha256': backbone_hash_before,
            'fusr_memory_sha256': memory_hash_before,
        }
        print(
            'FUSR_V2_CALIBRATION_LOCKED '
            f"threshold={locked_threshold:.8f} gate_rate={calibration['gate_rate']:.6f} "
            f"mse_before={calibration['always_on_risk']:.8f} "
            f"mse_after={calibration['risk']:.8f}",
            flush=True,
        )
        return self.model

    @staticmethod
    def _binary_auc(labels, scores):
        labels = np.asarray(labels, dtype=np.bool_)
        scores = np.asarray(scores, dtype=np.float64)
        positives = int(labels.sum())
        negatives = int((~labels).sum())
        if positives == 0 or negatives == 0:
            return None
        order = np.argsort(scores, kind='mergesort')
        sorted_scores = scores[order]
        ranks = np.empty(scores.shape[0], dtype=np.float64)
        start = 0
        while start < scores.shape[0]:
            end = start + 1
            while end < scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
            start = end
        positive_rank_sum = float(ranks[labels].sum())
        return float(
            (positive_rank_sum - positives * (positives + 1) / 2.0)
            / (positives * negatives)
        )

    @staticmethod
    def _temporal_extrema(seq_x):
        batch, length, variables = seq_x.shape
        time_index = torch.arange(length, device=seq_x.device).view(1, length, 1)
        max_position = torch.argmax(seq_x, dim=1).view(batch, 1, variables)
        min_position = torch.argmin(seq_x, dim=1).view(batch, 1, variables)
        max_idx = torch.abs(max_position - time_index).expand(batch, length, variables)
        min_idx = torch.abs(min_position - time_index).expand(batch, length, variables)
        max_value = torch.max(seq_x, dim=1).values.view(batch, 1, variables).expand(
            batch, length, variables
        )
        min_value = torch.min(seq_x, dim=1).values.view(batch, 1, variables).expand(
            batch, length, variables
        )
        return max_idx.float(), min_idx.float(), max_value, min_value

    def _build_mipp_pair(self, seq_true_x):
        if not (self.args.use_mipp and self.args.use_mask_intervention_pair):
            return None
        batch, length, variables = seq_true_x.shape
        choices = self.args.mipp_pair_block_lengths
        choice_id = int(torch.randint(len(choices), (1,), device=seq_true_x.device).item())
        block_length = int(choices[choice_id])
        target_rate = torch.empty((), device=seq_true_x.device).uniform_(
            float(self.args.mipp_pair_rate_min), float(self.args.mipp_pair_rate_max)
        )
        # If starts are independent with probability p, max pooling over a
        # block of length k has expected missing rate 1-(1-p)^k.
        start_probability = 1.0 - torch.pow(1.0 - target_rate, 1.0 / block_length)
        starts = (
            torch.rand(batch, 1, length, device=seq_true_x.device)
            < start_probability
        ).float()
        if block_length > 1:
            # Explicit left padding keeps each sampled start active for a
            # causal block without violating max_pool1d's internal-padding
            # constraint (padding must be <= floor(kernel_size / 2)).
            padded_starts = torch.nn.functional.pad(
                starts, (block_length - 1, 0), mode='constant', value=0.0
            )
            missing_time = torch.nn.functional.max_pool1d(
                padded_starts,
                kernel_size=block_length,
                stride=1,
                padding=0,
            )
        else:
            missing_time = starts
        seq_x_mask = (1.0 - missing_time.transpose(1, 2)).expand(
            batch, length, variables
        ).contiguous()
        seq_x = seq_true_x * seq_x_mask
        max_idx, min_idx, max_value, min_value = self._temporal_extrema(seq_x)
        return {
            'seq_x': seq_x,
            'seq_x_mask': seq_x_mask,
            'max_idx': max_idx,
            'min_idx': min_idx,
            'max_value': max_value,
            'min_value': min_value,
            'realized_missing_rate': float((1.0 - seq_x_mask).mean().item()),
            'block_length': block_length,
        }

    @staticmethod
    def _counterfactual_regret_objective(
        prediction_on,
        prediction_off,
        target,
        target_mask,
    ):
        """Penalize only sample-wise harm caused by prototype retrieval.

        The no-prototype prediction is a detached training reference.  This
        prevents the shared model from reducing the penalty by deliberately
        making the reference branch worse.  Counts are computed per sample so
        variable amounts of observed future data do not change the objective.
        """
        if not (
            prediction_on.shape
            == prediction_off.shape
            == target.shape
            == target_mask.shape
        ):
            raise ValueError('CFR prediction, target, and mask shapes must match')
        weights = target_mask.to(dtype=prediction_on.dtype)
        counts = weights.reshape(weights.size(0), -1).sum(dim=1)
        valid = counts > 0
        if not bool(valid.any()):
            raise ValueError('CFR requires at least one observed target per batch')
        counts = counts.clamp_min(1.0)
        squared_on = (prediction_on - target).pow(2) * weights
        squared_off = (prediction_off.detach() - target).pow(2) * weights
        mse_on = squared_on.reshape(squared_on.size(0), -1).sum(dim=1) / counts
        mse_off = squared_off.reshape(squared_off.size(0), -1).sum(dim=1) / counts
        mse_on = mse_on[valid]
        mse_off = mse_off[valid]
        signed_regret = mse_on - mse_off
        positive_regret = torch.relu(signed_regret)
        return {
            'total_loss': positive_regret.mean(),
            'sample_count': int(valid.sum().item()),
            'prototype_on_mse': mse_on.mean(),
            'prototype_off_mse': mse_off.mean(),
            'signed_regret': signed_regret.mean(),
            'positive_regret': positive_regret.mean(),
            'harmful_retrieval_rate': (signed_regret > 0).float().mean(),
        }
    
    def change_mode(self,is_training=1):
        self.model.mem_net.is_training=is_training
        self.model.is_training=is_training
            
    def save_model(self,path):
        dic = {}
        dic['model'] = self.model.state_dict()

        dic['memory'] = self.model.mem_net.memory.memory
        dic['repre'] = self.model.mem_net.memory.repre
        dic['num_dic'] = self.model.mem_net.memory.num_dic   
        torch.save(dic,path)
        
    def load_model(self,path):
        dic = torch.load(path, map_location=self.device)
        if dic.get('checkpoint_format') in {'mcpr_delta_v1', 'pba_delta_v1'}:
            source_checkpoint = os.path.abspath(dic['source_checkpoint'])
            if not os.path.isfile(source_checkpoint):
                raise FileNotFoundError(
                    f'delta checkpoint source missing: {source_checkpoint}'
                )
            source_sha256 = self._file_sha256(source_checkpoint)
            if source_sha256 != dic['source_checkpoint_sha256']:
                raise RuntimeError('delta checkpoint source SHA-256 mismatch')
            source = torch.load(source_checkpoint, map_location=self.device)
            self.model.load_state_dict(source['model'], strict=False)
            self.model.mem_net.memory.memory = source['memory']
            self.model.mem_net.memory.repre = source['repre']
            self.model.mem_net.memory.num_dic = source['num_dic']
            self.model.load_state_dict(dic['model'], strict=False)
        else:
            self.model.load_state_dict(dic['model'],strict=False)
            self.model.mem_net.memory.memory = dic['memory']
            self.model.mem_net.memory.repre = dic['repre']
            self.model.mem_net.memory.num_dic = dic['num_dic']
        return self.model

    def vali(self, vali_data, vali_loader, criterion,epoch=0):
        total_loss = []
        evaluated_batches = 0
        f_dim = -1 if self.args.features == 'MS' else 0
        self.model.eval()
        self.change_mode(0)
        with torch.no_grad():
            for i, (seq_x,seq_x_mask,seq_y,seq_y_mask,seq_true_x,seq_true_y,max_idx,min_idx,max_value,min_value) in enumerate(vali_loader):
                if self.args.max_eval_batches >= 0 and i >= self.args.max_eval_batches:
                    break
                seq_x,seq_x_mask,seq_y,seq_y_mask,seq_true_x,seq_true_y,max_idx,min_idx,max_value,min_value \
                    = seq_x.float().to(self.device),seq_x_mask.float().to(self.device),seq_y.float().to(self.device),seq_y_mask.float().to(self.device),seq_true_x.float().to(self.device),seq_true_y.float().to(self.device),max_idx.float().to(self.device),min_idx.float().to(self.device),max_value.float().to(self.device),min_value.float().to(self.device)
                if (seq_y_mask).sum()==0:
                    continue
                seq_y = seq_y[:, -self.args.pred_len:, f_dim:]
                outputs = self.model(seq_x,seq_x_mask,max_idx,min_idx,max_value,min_value)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                pred = outputs.detach()
                seq_y_mask = seq_y_mask[:, -self.args.pred_len:, f_dim:].detach()
                batch_y = seq_y.detach()
                loss = criterion(pred[seq_y_mask==1.0], batch_y[seq_y_mask==1.0])
                total_loss.append(loss.item())
                evaluated_batches += 1
        total_loss = sum(total_loss)/len(total_loss)
        self.last_val_batches = evaluated_batches
        self.model.train()
        return total_loss

    def _mcpr_backbone_hash(self):
        hasher = hashlib.sha256()
        for key, value in sorted(self._core_model().state_dict().items()):
            if key.startswith('mcpr.') or '.mcpr.' in key:
                continue
            hasher.update(key.encode('utf-8'))
            self._fusr_v2_hash_value(hasher, value)
        return hasher.hexdigest()

    @staticmethod
    def _mcpr_common_descent_gradients(mse_loss, mae_loss, parameters, epsilon=1e-12):
        mse_gradients = torch.autograd.grad(
            mse_loss, parameters, retain_graph=True, allow_unused=False
        )
        mae_gradients = torch.autograd.grad(
            mae_loss, parameters, retain_graph=False, allow_unused=False
        )
        mse_square = sum(
            gradient.double().pow(2).sum() for gradient in mse_gradients
        )
        mae_square = sum(
            gradient.double().pow(2).sum() for gradient in mae_gradients
        )
        mse_norm = mse_square.sqrt()
        mae_norm = mae_square.sqrt()
        if not bool(torch.isfinite(mse_norm).item()) or float(mse_norm.item()) <= epsilon:
            raise FloatingPointError('MCPR MSE gradient norm is non-finite or zero')
        if not bool(torch.isfinite(mae_norm).item()) or float(mae_norm.item()) <= epsilon:
            raise FloatingPointError('MCPR MAE gradient norm is non-finite or zero')
        dot = sum(
            (left.double() * right.double()).sum()
            for left, right in zip(mse_gradients, mae_gradients)
        )
        cosine = dot / (mse_norm * mae_norm).clamp_min(epsilon)
        combined = tuple(
            0.5 * (
                left / mse_norm.to(dtype=left.dtype)
                + right / mae_norm.to(dtype=right.dtype)
            )
            for left, right in zip(mse_gradients, mae_gradients)
        )
        mse_dot = sum(
            (gradient.double() * direction.double()).sum()
            for gradient, direction in zip(mse_gradients, combined)
        )
        mae_dot = sum(
            (gradient.double() * direction.double()).sum()
            for gradient, direction in zip(mae_gradients, combined)
        )
        if float(mse_dot.item()) < -1e-10 or float(mae_dot.item()) < -1e-10:
            raise FloatingPointError('MCPR common direction is not Pareto non-conflicting')
        return combined, {
            'mse_gradient_norm': float(mse_norm.item()),
            'mae_gradient_norm': float(mae_norm.item()),
            'gradient_cosine': float(cosine.item()),
            'common_mse_dot': float(mse_dot.item()),
            'common_mae_dot': float(mae_dot.item()),
        }

    @staticmethod
    def _ttrr_tail_trust_objective(
        prediction,
        backbone_prediction,
        target,
        target_mask,
        tail_fraction,
        tail_weight,
    ):
        """Penalize the largest positive sample harms relative to the backbone."""
        if prediction.shape != backbone_prediction.shape or prediction.shape != target.shape:
            raise ValueError('TTRR prediction/backbone/target shapes must match')
        if target_mask.shape != target.shape:
            raise ValueError('TTRR target mask shape must match the target')
        if not 0.0 < float(tail_fraction) <= 1.0:
            raise ValueError('TTRR tail fraction must lie in (0, 1]')
        if not math.isfinite(float(tail_weight)) or float(tail_weight) <= 0.0:
            raise ValueError('TTRR tail weight must be finite and positive')

        observed = target_mask.to(dtype=torch.bool)
        reduce_dims = tuple(range(1, target.ndim))
        observed_float = observed.to(dtype=prediction.dtype)
        observed_count = observed_float.sum(dim=reduce_dims)
        valid = observed_count > 0
        if not bool(valid.any()):
            raise RuntimeError('TTRR batch contains no sample with observed targets')

        candidate_square = (prediction - target).square()
        backbone_square = (backbone_prediction - target).square()
        candidate_sample_mse = (
            (candidate_square * observed_float).sum(dim=reduce_dims)
            / observed_count.clamp_min(1.0)
        )[valid]
        backbone_sample_mse = (
            (backbone_square * observed_float).sum(dim=reduce_dims)
            / observed_count.clamp_min(1.0)
        )[valid]
        sample_delta = candidate_sample_mse - backbone_sample_mse
        positive_harm = torch.relu(sample_delta)
        tail_count = max(
            1,
            int(math.ceil(float(tail_fraction) * int(positive_harm.numel()))),
        )
        tail_cvar = torch.topk(positive_harm, k=tail_count, largest=True).values.mean()
        candidate_mse = candidate_square[observed].mean()
        total_loss = candidate_mse + float(tail_weight) * tail_cvar
        return {
            'total_loss': total_loss,
            'candidate_mse': candidate_mse,
            'candidate_sample_mse_mean': candidate_sample_mse.mean(),
            'backbone_sample_mse_mean': backbone_sample_mse.mean(),
            'sample_delta_mean': sample_delta.mean(),
            'positive_harm_fraction': (sample_delta > 0.0).to(prediction.dtype).mean(),
            'tail_cvar': tail_cvar,
            'tail_count': tail_count,
            'valid_sample_count': int(positive_harm.numel()),
        }

    @staticmethod
    def _ard_analytic_distillation_objective(
        raw_residual,
        applied_residual,
        deployed_prediction,
        action_logits,
        student_gate,
        target,
        target_mask,
        distill_enabled,
        energy_epsilon,
        action_epsilon,
        action_weight,
        scale_weight,
    ):
        """Train raw MCPR with MSE and distill its analytic sample action."""
        if not (
            raw_residual.shape == applied_residual.shape
            == deployed_prediction.shape == target.shape == target_mask.shape
        ):
            raise ValueError('ARD residual/prediction/target shapes must match')
        if action_logits.shape != (target.size(0), 3):
            raise ValueError('ARD action logits must be [batch,3]')
        if student_gate.numel() != target.size(0):
            raise ValueError('ARD student gate must contain one value per sample')
        if float(energy_epsilon) <= 0.0:
            raise ValueError('ARD energy epsilon must be positive')
        if not 0.0 < float(action_epsilon) < 0.5:
            raise ValueError('ARD action epsilon must lie in (0,0.5)')

        observed = target_mask.to(dtype=torch.bool)
        if not bool(observed.any()):
            raise RuntimeError('ARD batch contains no observed targets')
        backbone_prediction = deployed_prediction - applied_residual
        raw_prediction = backbone_prediction + raw_residual
        raw_mse = (raw_prediction[observed] - target[observed]).square().mean()
        zero = raw_mse.new_zeros(())
        diagnostics = {
            'total_loss': raw_mse,
            'raw_mse': raw_mse,
            'action_loss': zero,
            'scale_loss': zero,
            'teacher_alpha': raw_mse.new_empty((0,)),
            'teacher_actions': torch.empty(
                0, dtype=torch.long, device=raw_mse.device
            ),
            'valid_sample_count': 0,
            'distill_enabled': bool(distill_enabled),
        }
        if not distill_enabled:
            return diagnostics

        reduce_dims = tuple(range(1, target.ndim))
        observed_float = observed.to(dtype=raw_residual.dtype)
        observed_count = observed_float.sum(dim=reduce_dims)
        detached_residual = raw_residual.detach()
        detached_backbone_error = (backbone_prediction - target).detach()
        alignment = (
            2.0 * detached_backbone_error * detached_residual * observed_float
        ).sum(dim=reduce_dims)
        energy = (
            detached_residual.square() * observed_float
        ).sum(dim=reduce_dims)
        valid = (observed_count > 0) & torch.isfinite(alignment) & torch.isfinite(energy)
        valid = valid & (energy > float(energy_epsilon))
        if not bool(valid.any()):
            return diagnostics

        teacher_alpha = torch.clamp(
            -alignment[valid] / (2.0 * energy[valid]), min=0.0, max=1.0
        )
        actions = torch.ones_like(teacher_alpha, dtype=torch.long)
        actions[teacher_alpha <= float(action_epsilon)] = 0
        actions[teacher_alpha >= 1.0 - float(action_epsilon)] = 2
        valid_logits = action_logits[valid]
        valid_gate = student_gate.reshape(-1)[valid]
        action_loss = nn.functional.cross_entropy(valid_logits, actions)
        scale_loss = nn.functional.smooth_l1_loss(valid_gate, teacher_alpha)
        total_loss = (
            raw_mse
            + float(action_weight) * action_loss
            + float(scale_weight) * scale_loss
        )
        return {
            'total_loss': total_loss,
            'raw_mse': raw_mse,
            'action_loss': action_loss,
            'scale_loss': scale_loss,
            'teacher_alpha': teacher_alpha,
            'teacher_actions': actions,
            'valid_sample_count': int(teacher_alpha.numel()),
            'distill_enabled': True,
        }

    def _mcpr_validation_loss(self, validation_loader):
        f_dim = -1 if self.args.features == 'MS' else 0
        self.model.eval()
        self.change_mode(0)
        losses = []
        evaluated_batches = 0
        with torch.no_grad():
            for index, batch in enumerate(validation_loader):
                if self.args.max_eval_batches >= 0 and index >= self.args.max_eval_batches:
                    break
                (
                    seq_x, seq_x_mask, seq_y, seq_y_mask, _seq_true_x,
                    _seq_true_y, max_idx, min_idx, max_value, min_value,
                ) = tuple(item.float().to(self.device) for item in batch)
                prediction = self.model(
                    seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                )[:, -self.args.pred_len:, f_dim:]
                target = seq_y[:, -self.args.pred_len:, f_dim:]
                target_mask = seq_y_mask[:, -self.args.pred_len:, f_dim:]
                if bool((target_mask > 0.5).any()):
                    losses.append(torch.mean(
                        (prediction[target_mask > 0.5] - target[target_mask > 0.5]).pow(2)
                    ).item())
                    evaluated_batches += 1
        if not losses:
            raise RuntimeError('MCPR validation processed zero observed targets')
        self.last_val_batches = evaluated_batches
        return float(np.mean(losses))

    @staticmethod
    def _bf_cmvd_objective(
        student_prediction,
        backbone_prediction,
        teacher_prediction,
        target,
        target_mask,
        filter_granularity='sample',
    ):
        """Direct MSE plus L1 distillation on bi-metric teacher wins only."""
        if not (
            student_prediction.shape == backbone_prediction.shape
            == teacher_prediction.shape == target.shape == target_mask.shape
        ):
            raise ValueError('BF-CMVD prediction/target/mask shapes must match')
        if filter_granularity not in {'sample', 'token'}:
            raise ValueError('BF-CMVD filter granularity must be sample or token')
        observed = target_mask > 0.5
        reduce_dims = (
            tuple(range(1, target.ndim))
            if filter_granularity == 'sample'
            else (target.ndim - 1,)
        )
        raw_counts = observed.sum(dim=reduce_dims)
        eligible = raw_counts > 0
        counts = raw_counts.clamp_min(1).to(target.dtype)

        def sample_losses(prediction):
            error = prediction - target
            mse = (error.square() * observed).sum(dim=reduce_dims) / counts
            mae = (error.abs() * observed).sum(dim=reduce_dims) / counts
            return mse, mae

        student_mse_by_sample, student_mae_by_sample = sample_losses(
            student_prediction
        )
        backbone_mse, backbone_mae = sample_losses(backbone_prediction.detach())
        teacher_mse, teacher_mae = sample_losses(teacher_prediction.detach())
        beneficial = (
            eligible & (teacher_mse < backbone_mse) & (teacher_mae < backbone_mae)
        )
        distill_by_sample = (
            (student_prediction - teacher_prediction.detach()).abs() * observed
        ).sum(dim=reduce_dims) / counts
        if bool(beneficial.any().item()):
            distill_loss = distill_by_sample[beneficial].mean()
        else:
            distill_loss = student_prediction.sum() * 0.0
        observed_student_error = (student_prediction - target)[observed]
        student_mse = observed_student_error.square().mean()
        student_mae = observed_student_error.abs().mean()
        total_loss = student_mse + distill_loss
        return {
            'total_loss': total_loss,
            'student_mse': student_mse,
            'student_mae': student_mae,
            'distill_l1': distill_loss,
            'beneficial_mask': beneficial,
            'eligible_mask': eligible,
            'filter_granularity': filter_granularity,
            'beneficial_fraction': (
                beneficial.sum().to(target.dtype)
                / eligible.sum().clamp_min(1).to(target.dtype)
            ),
            'backbone_mse': backbone_mse[eligible].mean(),
            'backbone_mae': backbone_mae[eligible].mean(),
            'teacher_mse': teacher_mse[eligible].mean(),
            'teacher_mae': teacher_mae[eligible].mean(),
            'teacher_delta_energy': (
                teacher_prediction.detach() - backbone_prediction.detach()
            ).square().mean(),
        }

    @classmethod
    def _dmrb_objective(
        cls,
        student_prediction,
        backbone_prediction,
        teacher_prediction,
        residual_l5,
        residual_l20,
        target_correction_l5,
        target_correction_l20,
        target,
        target_mask,
        l5_enabled=True,
        l20_enabled=True,
    ):
        """Observed-target MSE plus scale-factored L1 on teacher-beneficial tokens."""
        tensors = (
            student_prediction, backbone_prediction, teacher_prediction,
            residual_l5, residual_l20, target_correction_l5,
            target_correction_l20, target, target_mask,
        )
        if not all(tensor.shape == target.shape for tensor in tensors):
            raise ValueError('DMRB prediction/residual/target shapes must match')
        if not (l5_enabled or l20_enabled):
            raise ValueError('DMRB objective requires at least one enabled basis')
        expected_teacher = (
            backbone_prediction.detach()
            + target_correction_l5.detach()
            + target_correction_l20.detach()
        )
        if not torch.allclose(
            teacher_prediction.detach(), expected_teacher,
            rtol=0.0, atol=DMRB_TEACHER_RECONSTRUCTION_ATOL,
        ):
            raise RuntimeError('DMRB component corrections do not reconstruct teacher')
        base = cls._bf_cmvd_objective(
            student_prediction=student_prediction,
            backbone_prediction=backbone_prediction,
            teacher_prediction=teacher_prediction,
            target=target,
            target_mask=target_mask,
            filter_granularity='token',
        )
        observed = target_mask > 0.5
        counts = observed.sum(dim=-1).clamp_min(1).to(target.dtype)
        l5_by_token = (
            (residual_l5 - target_correction_l5.detach()).abs() * observed
        ).sum(dim=-1) / counts
        l20_by_token = (
            (residual_l20 - target_correction_l20.detach()).abs() * observed
        ).sum(dim=-1) / counts
        beneficial = base['beneficial_mask']
        zero = student_prediction.sum() * 0.0
        if bool(beneficial.any().item()):
            l5_loss = l5_by_token[beneficial].mean() if l5_enabled else zero
            l20_loss = l20_by_token[beneficial].mean() if l20_enabled else zero
        else:
            l5_loss = zero
            l20_loss = zero
        basis_distill = l5_loss + l20_loss
        base.update({
            'total_loss': base['student_mse'] + basis_distill,
            'distill_l1': basis_distill,
            'dmrb_l5_distill_l1': l5_loss,
            'dmrb_l20_distill_l1': l20_loss,
            'dmrb_l5_target_energy': target_correction_l5.detach().square().mean(),
            'dmrb_l20_target_energy': target_correction_l20.detach().square().mean(),
            'dmrb_teacher_reconstruction_max_abs': (
                teacher_prediction.detach() - expected_teacher
            ).abs().max(),
        })
        return base

    @staticmethod
    def _pba_state_key_selected(key, e_layers, tail_layers):
        normalized = key[7:] if key.startswith('module.') else key
        if normalized.startswith(('norm.', 'out_proj.')):
            return True
        tail_start = int(e_layers) - int(tail_layers)
        return any(
            normalized.startswith(f's4_layers.{index}.')
            for index in range(tail_start, int(e_layers))
        )

    def _pba_state_hash(self, model, selected=None):
        hasher = hashlib.sha256()
        for key, value in sorted(model.state_dict().items()):
            is_selected = self._pba_state_key_selected(
                key, self.args.e_layers, self.args.pba_tail_layers
            )
            if selected is not None and is_selected is not bool(selected):
                continue
            hasher.update(key.encode('utf-8'))
            self._fusr_v2_hash_value(hasher, value)
        return hasher.hexdigest()

    def _pba_memory_hash(self, model):
        core = model.module if isinstance(model, nn.DataParallel) else model
        memory = core.mem_net.memory
        hasher = hashlib.sha256()
        for key in ('memory', 'repre', 'num_dic'):
            hasher.update(key.encode('utf-8'))
            self._fusr_v2_hash_value(hasher, getattr(memory, key))
        return hasher.hexdigest()

    def _pba_enable_tail_training(self):
        core = self._core_model()
        self.model.eval()
        tail_start = int(self.args.e_layers) - int(self.args.pba_tail_layers)
        for index in range(tail_start, int(self.args.e_layers)):
            core.s4_layers[index].train()
        core.norm.train()
        core.out_proj.train()
        self.change_mode(0)

    @staticmethod
    def _hfd_pba_objective(
        student_features,
        source_features,
        nested_l5_features,
        nested_l20_features,
        beneficial_mask,
        output_distill_loss,
        pred_len,
        layers=(6, 7),
    ):
        """Energy-normalized hidden-correction distillation for HFD-PBA."""
        resolved_layers = tuple(int(layer) for layer in layers)
        if resolved_layers != (6, 7):
            raise ValueError('HFD-PBA layers are frozen at (6, 7)')
        feature_groups = {
            'student': student_features,
            'source': source_features,
            'l5': nested_l5_features,
            'l20': nested_l20_features,
        }
        for name, features in feature_groups.items():
            if set(features) != set(resolved_layers):
                raise ValueError(
                    f'HFD-PBA {name} feature layers differ from {resolved_layers}'
                )
        if beneficial_mask.ndim != 2:
            raise ValueError('HFD-PBA beneficial mask must have shape [B, T]')
        if beneficial_mask.shape[1] != int(pred_len):
            raise ValueError('HFD-PBA beneficial mask length must equal pred_len')
        if output_distill_loss.ndim != 0:
            raise ValueError('HFD-PBA output distillation loss must be scalar')

        beneficial = beneficial_mask.to(dtype=torch.bool)
        zero = sum(
            student_features[layer].sum() * 0.0 for layer in resolved_layers
        )
        layer_raw = {}
        layer_teacher_energy = {}
        layer_student_energy = {}
        layer_cosine = {}
        for layer in resolved_layers:
            tensors = (
                student_features[layer], source_features[layer],
                nested_l5_features[layer], nested_l20_features[layer],
            )
            if not all(tensor.ndim == 3 for tensor in tensors):
                raise ValueError('HFD-PBA features must have shape [B, L, D]')
            if not all(tensor.shape == tensors[0].shape for tensor in tensors):
                raise ValueError(f'HFD-PBA layer {layer} feature shapes do not match')
            if tensors[0].shape[0] != beneficial.shape[0]:
                raise ValueError('HFD-PBA feature and benefit-mask batches differ')
            if tensors[0].shape[1] < int(pred_len):
                raise ValueError('HFD-PBA feature sequence is shorter than pred_len')
            if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
                raise FloatingPointError('HFD-PBA received non-finite features')

            student = tensors[0][:, -int(pred_len):, :]
            source = tensors[1][:, -int(pred_len):, :].detach()
            l5 = tensors[2][:, -int(pred_len):, :].detach()
            l20 = tensors[3][:, -int(pred_len):, :].detach()
            student_delta = student - source
            teacher_delta = 0.25 * (l5 - source) + 0.25 * (l20 - source)
            expanded = beneficial.unsqueeze(-1).to(student.dtype)
            numerator = ((student_delta - teacher_delta).square() * expanded).sum()
            teacher_energy = (teacher_delta.square() * expanded).sum()
            student_energy = (student_delta.square() * expanded).sum()
            if bool(beneficial.any().item()):
                epsilon = torch.finfo(student.dtype).eps
                raw = numerator / (teacher_energy + epsilon)
                dot = (student_delta * teacher_delta * expanded).sum()
                cosine = dot / (
                    torch.sqrt(student_energy * teacher_energy) + epsilon
                )
            else:
                raw = zero
                cosine = zero.detach()
            layer_raw[layer] = raw
            layer_teacher_energy[layer] = teacher_energy.detach()
            layer_student_energy[layer] = student_energy.detach()
            layer_cosine[layer] = cosine.detach()

        raw_mean = sum(layer_raw.values()) / float(len(resolved_layers))
        matched_loss = output_distill_loss.detach() * raw_mean
        if not bool(torch.isfinite(matched_loss).item()):
            raise FloatingPointError('HFD-PBA matched hidden loss is non-finite')
        return {
            'hfd_loss': matched_loss,
            'hfd_raw_mean': raw_mean,
            'hfd_layer_raw': layer_raw,
            'hfd_layer_teacher_energy': layer_teacher_energy,
            'hfd_layer_student_energy': layer_student_energy,
            'hfd_layer_cosine': layer_cosine,
            'hfd_beneficial_units': beneficial.sum(),
        }

    def _train_pba_cmvd(self, setting, train_loader, validation_loader, path):
        """Amortize a frozen three-view teacher into the nonlinear S4 tail."""
        hfd_enabled = bool(getattr(self.args, 'use_hfd_pba', False))
        hfd_layers = tuple(int(layer) for layer in getattr(
            self.args, 'hfd_layers', [6, 7]
        ))
        f_dim = -1 if self.args.features == 'MS' else 0
        source_checkpoint = os.path.abspath(self.args.pba_backbone_checkpoint)
        if not os.path.isfile(source_checkpoint):
            raise FileNotFoundError(
                f'PBA-CMVD backbone checkpoint missing: {source_checkpoint}'
            )
        self.load_model(source_checkpoint)
        core = self._core_model()
        if core.mcpr is not None:
            raise RuntimeError('PBA-CMVD student must not instantiate MCPR/DMRB')

        teacher_model = copy.deepcopy(self.model)
        teacher_core = (
            teacher_model.module
            if isinstance(teacher_model, nn.DataParallel)
            else teacher_model
        )
        teacher_model.eval()
        teacher_core.mem_net.is_training = 0
        teacher_core.is_training = 0
        for parameter in teacher_model.parameters():
            parameter.requires_grad = False
        if hfd_enabled:
            core.set_hfd_feature_capture(True, hfd_layers)
            teacher_core.set_hfd_feature_capture(True, hfd_layers)

        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = self._pba_state_key_selected(
                name, self.args.e_layers, self.args.pba_tail_layers
            )
        trainable_named = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable_named:
            raise RuntimeError('PBA-CMVD has no trainable parameters')
        trainable_names = sorted(name for name, _ in trainable_named)
        unexpected = [
            name for name in trainable_names
            if not self._pba_state_key_selected(
                name, self.args.e_layers, self.args.pba_tail_layers
            )
        ]
        if unexpected:
            raise RuntimeError(f'PBA-CMVD opened unexpected parameters: {unexpected}')
        trainable = [parameter for _, parameter in trainable_named]
        trainable_parameter_count = int(sum(p.numel() for p in trainable))
        if trainable_parameter_count != 4127582:
            raise RuntimeError(
                'PBA-CMVD trainable parameter count differs from the frozen protocol: '
                f'{trainable_parameter_count} != 4127582'
            )

        student_parameters = dict(self.model.named_parameters())
        teacher_parameters = dict(teacher_model.named_parameters())
        shared_parameter_storage = any(
            student_parameters[name].data_ptr() == teacher_parameters[name].data_ptr()
            for name in student_parameters
        )
        if shared_parameter_storage:
            raise RuntimeError('PBA-CMVD teacher and student share parameter storage')

        source_sha256 = self._file_sha256(source_checkpoint)
        frozen_hash_before = self._pba_state_hash(self.model, selected=False)
        trainable_hash_before = self._pba_state_hash(self.model, selected=True)
        memory_hash_before = self._pba_memory_hash(self.model)
        teacher_hash_before = self._pba_state_hash(teacher_model, selected=None)
        teacher_memory_hash_before = self._pba_memory_hash(teacher_model)
        optimizer = optim.Adam([{
            'params': trainable,
            'lr': float(self.args.learning_rate),
            'lr_scale': 1.0,
            'group_name': 'pba_tail',
        }])
        early_stopping = EarlyStopping(
            args=self.args, patience=self.args.patience,
            verbose=True, more_save=True,
        )

        steps = 0
        epochs_completed = 0
        direct_mse_sum = 0.0
        student_mae_sum = 0.0
        distill_l1_sum = 0.0
        total_loss_sum = 0.0
        beneficial_units = 0
        eligible_units = 0
        teacher_mse_sum = 0.0
        teacher_mae_sum = 0.0
        source_mse_sum = 0.0
        source_mae_sum = 0.0
        teacher_delta_energy_sum = 0.0
        teacher_forward_count = 0
        teacher_grad_found = False
        teacher_output_requires_grad = False
        hfd_student_feature_requires_grad = False
        hfd_teacher_feature_requires_grad = False
        hfd_active_steps = 0
        hfd_loss_sum = 0.0
        hfd_raw_sum = 0.0
        hfd_layer_raw_sum = {str(layer): 0.0 for layer in hfd_layers}
        hfd_layer_teacher_energy_sum = {str(layer): 0.0 for layer in hfd_layers}
        hfd_layer_student_energy_sum = {str(layer): 0.0 for layer in hfd_layers}
        hfd_layer_cosine_sum = {str(layer): 0.0 for layer in hfd_layers}
        hfd_layer_feature_gradient_norm_max = {
            str(layer): 0.0 for layer in hfd_layers
        }
        group_gradient_norm_max = {
            f's4_layers_{index}': 0.0
            for index in range(
                int(self.args.e_layers) - int(self.args.pba_tail_layers),
                int(self.args.e_layers),
            )
        }
        group_gradient_norm_max.update({'norm': 0.0, 'out_proj': 0.0})
        gradient_norm_max = 0.0
        parameter_abs_max = 0.0
        time_now = time.time()

        def parameter_group(name):
            normalized = name[7:] if name.startswith('module.') else name
            if normalized.startswith('norm.'):
                return 'norm'
            if normalized.startswith('out_proj.'):
                return 'out_proj'
            for index in range(
                int(self.args.e_layers) - int(self.args.pba_tail_layers),
                int(self.args.e_layers),
            ):
                if normalized.startswith(f's4_layers.{index}.'):
                    return f's4_layers_{index}'
            raise RuntimeError(f'unclassified PBA-CMVD parameter: {name}')

        for epoch in range(self.args.train_epochs):
            epochs_completed = epoch + 1
            self._pba_enable_tail_training()
            if hfd_enabled:
                core.set_hfd_feature_capture(True, hfd_layers)
                teacher_core.set_hfd_feature_capture(True, hfd_layers)
            teacher_model.eval()
            epoch_mse = []
            for index, batch in enumerate(train_loader):
                if self.args.max_train_batches >= 0 and index >= self.args.max_train_batches:
                    break
                (
                    seq_x, seq_x_mask, seq_y, seq_y_mask, seq_true_x,
                    _seq_true_y, max_idx, min_idx, max_value, min_value,
                ) = tuple(item.float().to(self.device) for item in batch)
                optimizer.zero_grad(set_to_none=True)
                student_prediction = self.model(
                    seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                )[:, -self.args.pred_len:, f_dim:]
                student_features = None
                if hfd_enabled:
                    student_features = dict(core.get_last_hfd_features())
                    hfd_student_feature_requires_grad = (
                        hfd_student_feature_requires_grad
                        or all(
                            bool(student_features[layer].requires_grad)
                            for layer in hfd_layers
                        )
                    )
                target = seq_y[:, -self.args.pred_len:, f_dim:]
                target_mask = seq_y_mask[:, -self.args.pred_len:, f_dim:] > 0.5
                if not bool(target_mask.any()):
                    continue

                sample_start = (
                    epoch * len(train_loader) * int(self.args.batch_size)
                    + index * int(self.args.batch_size)
                )
                sample_indices = np.arange(
                    sample_start,
                    sample_start + int(seq_x.size(0)),
                    dtype=np.int64,
                )
                epoch_nested_seed = (
                    int(self.args.pba_nested_seed) + epoch * 1000003
                )
                with torch.no_grad():
                    source_prediction = teacher_model(
                        seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                    )[:, -self.args.pred_len:, f_dim:]
                    source_features = None
                    nested_features = {}
                    if hfd_enabled:
                        source_features = dict(teacher_core.get_last_hfd_features())
                    nested_predictions = {}
                    for block_length in (5, 20):
                        nested_mask = self._stage18_equal_count_observation_mask(
                            seq_x_mask,
                            sample_indices,
                            float(self.args.pba_target_rate),
                            block_length,
                            epoch_nested_seed,
                        )
                        nested_x = seq_true_x * nested_mask
                        nested_extrema = self._temporal_extrema(nested_x)
                        nested_predictions[block_length] = teacher_model(
                            nested_x, nested_mask, *nested_extrema
                        )[:, -self.args.pred_len:, f_dim:]
                        if hfd_enabled:
                            nested_features[block_length] = dict(
                                teacher_core.get_last_hfd_features()
                            )
                    teacher_prediction = (
                        float(self.args.pba_original_weight) * source_prediction
                        + float(self.args.pba_l5_weight) * nested_predictions[5]
                        + float(self.args.pba_l20_weight) * nested_predictions[20]
                    )
                teacher_forward_count += 3
                teacher_output_requires_grad = (
                    teacher_output_requires_grad
                    or bool(teacher_prediction.requires_grad)
                )
                diagnostics = self._bf_cmvd_objective(
                    student_prediction=student_prediction,
                    backbone_prediction=source_prediction,
                    teacher_prediction=teacher_prediction,
                    target=target,
                    target_mask=target_mask,
                    filter_granularity='token',
                )
                hfd_diagnostics = None
                if hfd_enabled:
                    teacher_feature_groups = (
                        source_features, nested_features[5], nested_features[20]
                    )
                    hfd_teacher_feature_requires_grad = (
                        hfd_teacher_feature_requires_grad
                        or any(
                            bool(features[layer].requires_grad)
                            for features in teacher_feature_groups
                            for layer in hfd_layers
                        )
                    )
                    hfd_diagnostics = self._hfd_pba_objective(
                        student_features=student_features,
                        source_features=source_features,
                        nested_l5_features=nested_features[5],
                        nested_l20_features=nested_features[20],
                        beneficial_mask=diagnostics['beneficial_mask'],
                        output_distill_loss=diagnostics['distill_l1'],
                        pred_len=self.args.pred_len,
                        layers=hfd_layers,
                    )
                    hfd_feature_gradients = torch.autograd.grad(
                        hfd_diagnostics['hfd_loss'],
                        [student_features[layer] for layer in hfd_layers],
                        retain_graph=True,
                        allow_unused=True,
                    )
                    for layer, gradient in zip(hfd_layers, hfd_feature_gradients):
                        if gradient is None:
                            continue
                        norm = float(
                            gradient.detach().double().pow(2).sum().sqrt().item()
                        )
                        hfd_layer_feature_gradient_norm_max[str(layer)] = max(
                            hfd_layer_feature_gradient_norm_max[str(layer)], norm
                        )
                    diagnostics['total_loss'] = (
                        diagnostics['total_loss'] + hfd_diagnostics['hfd_loss']
                    )
                if not bool(torch.isfinite(diagnostics['total_loss']).item()):
                    raise FloatingPointError(
                        'NAN_OR_DIVERGENCE: PBA-CMVD objective is non-finite'
                    )
                diagnostics['total_loss'].backward()
                if any(parameter.grad is not None for parameter in teacher_model.parameters()):
                    teacher_grad_found = True
                    raise RuntimeError('PBA-CMVD frozen teacher received gradients')
                if not any(parameter.grad is not None for parameter in trainable):
                    raise RuntimeError('PBA-CMVD produced no student gradients')
                if any(
                    parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all().item())
                    for parameter in trainable
                ):
                    raise FloatingPointError(
                        'NAN_OR_DIVERGENCE: PBA-CMVD gradient is non-finite'
                    )
                total_gradient_square = 0.0
                group_gradient_square = {
                    key: 0.0 for key in group_gradient_norm_max
                }
                for name, parameter in trainable_named:
                    if parameter.grad is None:
                        continue
                    square = float(
                        parameter.grad.detach().double().pow(2).sum().item()
                    )
                    total_gradient_square += square
                    group_gradient_square[parameter_group(name)] += square
                gradient_norm_max = max(
                    gradient_norm_max, math.sqrt(total_gradient_square)
                )
                for key, square in group_gradient_square.items():
                    group_gradient_norm_max[key] = max(
                        group_gradient_norm_max[key], math.sqrt(square)
                    )
                optimizer.step()
                for parameter in trainable:
                    if not bool(torch.isfinite(parameter).all().item()):
                        raise FloatingPointError(
                            'NAN_OR_DIVERGENCE: PBA-CMVD parameter is non-finite'
                        )
                    parameter_abs_max = max(
                        parameter_abs_max,
                        float(parameter.detach().abs().max().item()),
                    )

                beneficial = diagnostics['beneficial_mask'].detach()
                eligible = diagnostics['eligible_mask'].detach()
                beneficial_units += int(beneficial.sum().item())
                eligible_units += int(eligible.sum().item())
                direct_mse_sum += float(diagnostics['student_mse'].detach().item())
                student_mae_sum += float(diagnostics['student_mae'].detach().item())
                distill_l1_sum += float(diagnostics['distill_l1'].detach().item())
                total_loss_sum += float(diagnostics['total_loss'].detach().item())
                if hfd_enabled:
                    hfd_loss_sum += float(
                        hfd_diagnostics['hfd_loss'].detach().item()
                    )
                    hfd_raw_sum += float(
                        hfd_diagnostics['hfd_raw_mean'].detach().item()
                    )
                    if int(hfd_diagnostics['hfd_beneficial_units'].item()) > 0:
                        hfd_active_steps += 1
                    for layer in hfd_layers:
                        key = str(layer)
                        hfd_layer_raw_sum[key] += float(
                            hfd_diagnostics['hfd_layer_raw'][layer].detach().item()
                        )
                        hfd_layer_teacher_energy_sum[key] += float(
                            hfd_diagnostics['hfd_layer_teacher_energy'][layer].item()
                        )
                        hfd_layer_student_energy_sum[key] += float(
                            hfd_diagnostics['hfd_layer_student_energy'][layer].item()
                        )
                        hfd_layer_cosine_sum[key] += float(
                            hfd_diagnostics['hfd_layer_cosine'][layer].item()
                        )
                source_mse_sum += float(diagnostics['backbone_mse'].detach().item())
                source_mae_sum += float(diagnostics['backbone_mae'].detach().item())
                teacher_mse_sum += float(diagnostics['teacher_mse'].detach().item())
                teacher_mae_sum += float(diagnostics['teacher_mae'].detach().item())
                teacher_delta_energy_sum += float(
                    diagnostics['teacher_delta_energy'].detach().item()
                )
                epoch_mse.append(float(diagnostics['student_mse'].detach().item()))
                steps += 1
                if (index + 1) % 50 == 0:
                    elapsed = time.time() - time_now
                    print(
                        f'PBA-CMVD iters={index + 1} epoch={epoch + 1} '
                        f'mse={diagnostics["student_mse"].item():.7f} '
                        f'distill={diagnostics["distill_l1"].item():.7f} '
                        f'elapsed={elapsed:.2f}s', flush=True,
                    )
                    time_now = time.time()
            if not epoch_mse:
                raise RuntimeError('PBA-CMVD epoch completed without optimizer steps')
            if hfd_enabled:
                core.set_hfd_feature_capture(False)
                teacher_core.set_hfd_feature_capture(False)
            validation_loss = self._mcpr_validation_loss(validation_loader)
            print(
                f'PBA-CMVD epoch={epoch + 1} '
                f'train_mse={np.mean(epoch_mse):.7f} '
                f'validation_mse={validation_loss:.7f}', flush=True,
            )
            early_stopping(validation_loss, self.model, path)
            if early_stopping.early_stop:
                print('PBA-CMVD early stopping', flush=True)
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        checkpoint = os.path.join(path, 'checkpoint.pth')
        if hfd_enabled:
            core.set_hfd_feature_capture(False)
            teacher_core.set_hfd_feature_capture(False)
        if not os.path.isfile(checkpoint):
            raise RuntimeError('PBA-CMVD did not produce a checkpoint')
        self.load_model(checkpoint)
        frozen_hash_after = self._pba_state_hash(self.model, selected=False)
        trainable_hash_after = self._pba_state_hash(self.model, selected=True)
        memory_hash_after = self._pba_memory_hash(self.model)
        teacher_hash_after = self._pba_state_hash(teacher_model, selected=None)
        teacher_memory_hash_after = self._pba_memory_hash(teacher_model)
        if frozen_hash_after != frozen_hash_before:
            raise RuntimeError('PBA-CMVD changed a frozen student state')
        if trainable_hash_after == trainable_hash_before:
            raise RuntimeError('PBA-CMVD trainable tail did not change')
        if memory_hash_after != memory_hash_before:
            raise RuntimeError('PBA-CMVD changed the student prototype memory')
        if teacher_hash_after != teacher_hash_before:
            raise RuntimeError('PBA-CMVD changed the frozen teacher state')
        if teacher_memory_hash_after != teacher_memory_hash_before:
            raise RuntimeError('PBA-CMVD changed the frozen teacher memory')
        if teacher_output_requires_grad or teacher_grad_found:
            raise RuntimeError('PBA-CMVD teacher isolation contract failed')
        if not all(value > 0.0 for value in group_gradient_norm_max.values()):
            raise RuntimeError(
                'PBA-CMVD did not activate every trainable module group: '
                f'{group_gradient_norm_max}'
            )
        if beneficial_units < 1 or eligible_units <= beneficial_units:
            raise RuntimeError(
                'PBA-CMVD requires both beneficial and rejected eligible tokens'
            )
        if hfd_enabled:
            if hfd_teacher_feature_requires_grad:
                raise RuntimeError('HFD-PBA frozen teacher features require gradients')
            if not hfd_student_feature_requires_grad:
                raise RuntimeError('HFD-PBA student features did not require gradients')
            if hfd_active_steps < 1:
                raise RuntimeError('HFD-PBA hidden loss never observed a beneficial token')
            if not all(
                value > 0.0
                for value in hfd_layer_feature_gradient_norm_max.values()
            ):
                raise RuntimeError(
                    'HFD-PBA did not activate both hidden feature levels: '
                    f'{hfd_layer_feature_gradient_norm_max}'
                )

        denominator = max(1, steps)
        self.run_evidence = {
            'optimizer_steps': int(steps),
            'train_batches_processed': int(steps),
            'validation_batches_processed': int(getattr(self, 'last_val_batches', 0)),
            'epochs_completed': int(epochs_completed),
            'best_observed_validation_loss': float(early_stopping.val_loss_min),
            'memory_bank_warmup_count': 0,
            'mipp_enabled': True,
            'mipp_frozen_source_only': True,
            'pba_cmvd_enabled': True,
            'pba_tail_layers': int(self.args.pba_tail_layers),
            'pba_tail_start_index': int(
                self.args.e_layers - self.args.pba_tail_layers
            ),
            'pba_trainable_parameter_count': trainable_parameter_count,
            'pba_trainable_parameter_names': trainable_names,
            'pba_trainable_group_gradient_norm_max': group_gradient_norm_max,
            'pba_gradient_norm_max': float(gradient_norm_max),
            'pba_parameter_abs_max': float(parameter_abs_max),
            'pba_learning_rate': float(self.args.learning_rate),
            'pba_source_checkpoint': source_checkpoint,
            'pba_source_checkpoint_sha256': source_sha256,
            'pba_checkpoint_format': 'pba_delta_v1',
            'pba_frozen_hash_before': frozen_hash_before,
            'pba_frozen_hash_after': frozen_hash_after,
            'pba_trainable_hash_before': trainable_hash_before,
            'pba_trainable_hash_after': trainable_hash_after,
            'pba_memory_hash_before': memory_hash_before,
            'pba_memory_hash_after': memory_hash_after,
            'pba_teacher_hash_before': teacher_hash_before,
            'pba_teacher_hash_after': teacher_hash_after,
            'pba_teacher_memory_hash_before': teacher_memory_hash_before,
            'pba_teacher_memory_hash_after': teacher_memory_hash_after,
            'pba_teacher_distinct_object': True,
            'pba_teacher_shared_parameter_storage': False,
            'pba_teacher_parameters_frozen': True,
            'pba_teacher_output_requires_grad': False,
            'pba_teacher_gradient_found': False,
            'pba_teacher_training_forward_count': int(teacher_forward_count),
            'pba_inference_student_forward_count': 1,
            'pba_validation_teacher_forward_count': 0,
            'pba_filter_granularity': 'token',
            'pba_beneficial_units': int(beneficial_units),
            'pba_eligible_units': int(eligible_units),
            'pba_beneficial_fraction': float(
                beneficial_units / max(1, eligible_units)
            ),
            'pba_direct_mse_mean': float(direct_mse_sum / denominator),
            'pba_student_mae_mean': float(student_mae_sum / denominator),
            'pba_distill_l1_mean': float(distill_l1_sum / denominator),
            'pba_total_loss_mean': float(total_loss_sum / denominator),
            'pba_source_training_mse_mean': float(source_mse_sum / denominator),
            'pba_source_training_mae_mean': float(source_mae_sum / denominator),
            'pba_teacher_training_mse_mean': float(teacher_mse_sum / denominator),
            'pba_teacher_training_mae_mean': float(teacher_mae_sum / denominator),
            'pba_teacher_delta_energy_mean': float(
                teacher_delta_energy_sum / denominator
            ),
            'pba_target_rate': float(self.args.pba_target_rate),
            'pba_block_lengths': [int(v) for v in self.args.pba_block_lengths],
            'pba_nested_seed': int(self.args.pba_nested_seed),
            'pba_epoch_seed_stride': 1000003,
            'pba_teacher_weights': {
                'original': float(self.args.pba_original_weight),
                'l5': float(self.args.pba_l5_weight),
                'l20': float(self.args.pba_l20_weight),
            },
            'pba_direct_loss': 'observed_target_mse',
            'pba_distill_loss': 'token_benefit_filtered_l1',
            'pba_benefit_rule': (
                'teacher_mse_lt_source_and_teacher_mae_lt_source'
            ),
            'pba_deployment_path': 'single_original_view_single_student_forward',
            'hfd_pba_enabled': hfd_enabled,
            'hfd_layers': [int(layer) for layer in hfd_layers],
            'hfd_loss_mode': str(getattr(
                self.args, 'hfd_loss_mode', 'correction_energy_normalized_mse'
            )),
            'hfd_scale_mode': str(getattr(
                self.args, 'hfd_scale_mode', 'output_distill_detached'
            )),
            'hfd_feature_capture_default_off': True,
            'hfd_validation_feature_capture_enabled': False,
            'hfd_active_steps': int(hfd_active_steps),
            'hfd_loss_mean': float(hfd_loss_sum / denominator),
            'hfd_raw_mean': float(hfd_raw_sum / denominator),
            'hfd_layer_raw_mean': {
                key: float(value / denominator)
                for key, value in hfd_layer_raw_sum.items()
            },
            'hfd_layer_teacher_energy_mean': {
                key: float(value / denominator)
                for key, value in hfd_layer_teacher_energy_sum.items()
            },
            'hfd_layer_student_energy_mean': {
                key: float(value / denominator)
                for key, value in hfd_layer_student_energy_sum.items()
            },
            'hfd_layer_cosine_mean': {
                key: float(value / denominator)
                for key, value in hfd_layer_cosine_sum.items()
            },
            'hfd_layer_feature_gradient_norm_max': (
                hfd_layer_feature_gradient_norm_max
            ),
            'hfd_student_feature_requires_grad': bool(
                hfd_student_feature_requires_grad
            ),
            'hfd_teacher_feature_requires_grad': bool(
                hfd_teacher_feature_requires_grad
            ),
            'hfd_training_student_feature_forward_count': (
                int(steps) if hfd_enabled else 0
            ),
            'hfd_training_teacher_feature_forward_count': (
                int(teacher_forward_count) if hfd_enabled else 0
            ),
        }
        del teacher_model
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return self.model

    def _train_mcpr(self, setting, train_loader, validation_loader, path):
        f_dim = -1 if self.args.features == 'MS' else 0
        source_checkpoint = os.path.abspath(self.args.mcpr_backbone_checkpoint)
        if not os.path.isfile(source_checkpoint):
            raise FileNotFoundError(f'MCPR backbone checkpoint missing: {source_checkpoint}')
        self.load_model(source_checkpoint)
        core = self._core_model()
        if core.mcpr is None:
            raise RuntimeError('MCPR configuration did not instantiate the adapter')
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for parameter in core.mcpr.parameters():
            parameter.requires_grad = True
        dmrb_enabled = bool(getattr(self.args, 'use_dmrb', False))
        if dmrb_enabled:
            for parameter in core.mcpr.l5_basis.parameters():
                parameter.requires_grad = bool(self.args.dmrb_l5_enabled)
            for parameter in core.mcpr.l20_basis.parameters():
                parameter.requires_grad = bool(self.args.dmrb_l20_enabled)
        trainable = [parameter for parameter in core.mcpr.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError('MCPR has no trainable parameters')
        if any(
            parameter.requires_grad
            for name, parameter in self.model.named_parameters()
            if not (name.startswith('mcpr.') or '.mcpr.' in name)
        ):
            raise RuntimeError('MCPR failed to freeze the PG95 backbone')
        backbone_hash_before = self._mcpr_backbone_hash()
        memory_hash_before = self._fusr_v2_memory_hash()
        source_sha256 = self._file_sha256(source_checkpoint)
        lr_scale = float(self.args.mcpr_learning_rate) / float(self.args.learning_rate)
        optimizer = optim.Adam([{
            'params': trainable,
            'lr': float(self.args.mcpr_learning_rate),
            'lr_scale': lr_scale,
            'group_name': 'mcpr',
        }])
        early_stopping = EarlyStopping(
            args=self.args, patience=self.args.patience, verbose=True, more_save=True
        )
        objective = str(self.args.mcpr_objective)
        steps = 0
        epochs_completed = 0
        residual_energy_sum = 0.0
        mse_sum = 0.0
        mae_sum = 0.0
        cosine_sum = 0.0
        cosine_min = float('inf')
        mse_dot_min = float('inf')
        mae_dot_min = float('inf')
        diagnostic_steps = 0
        gradient_norm_max = 0.0
        parameter_abs_max = 0.0
        ttrr_enabled = bool(getattr(self.args, 'use_ttrr', False))
        ttrr_tail_sum = 0.0
        ttrr_objective_sum = 0.0
        ttrr_candidate_sample_mse_sum = 0.0
        ttrr_backbone_sample_mse_sum = 0.0
        ttrr_delta_sum = 0.0
        ttrr_positive_harm_fraction_sum = 0.0
        ttrr_raw_residual_energy_sum = 0.0
        ttrr_applied_residual_energy_sum = 0.0
        ttrr_gate_mean_sum = 0.0
        ttrr_gate_std_sum = 0.0
        ttrr_gate_min = float('inf')
        ttrr_gate_max = float('-inf')
        ttrr_diagnostic_steps = 0
        ard_enabled = bool(getattr(self.args, 'use_ard', False))
        ard_distillation_steps = 0
        ard_teacher_valid_samples = 0
        ard_teacher_alpha_sum = 0.0
        ard_teacher_alpha_square_sum = 0.0
        ard_teacher_alpha_min = float('inf')
        ard_teacher_alpha_max = float('-inf')
        ard_teacher_action_counts = [0, 0, 0]
        ard_action_loss_sum = 0.0
        ard_scale_loss_sum = 0.0
        ard_gate_mean_sum = 0.0
        ard_gate_std_sum = 0.0
        ard_gate_min = float('inf')
        ard_gate_max = float('-inf')
        ard_action_entropy_sum = 0.0
        ard_raw_residual_energy_sum = 0.0
        ard_applied_residual_energy_sum = 0.0
        ard_gate_stat_steps = 0
        bf_cmvd_enabled = bool(getattr(self.args, 'use_bf_cmvd', False))
        bf_cmvd_diagnostic_steps = 0
        bf_cmvd_beneficial_samples = 0
        bf_cmvd_total_samples = 0
        bf_cmvd_distill_l1_sum = 0.0
        bf_cmvd_backbone_mse_sum = 0.0
        bf_cmvd_backbone_mae_sum = 0.0
        bf_cmvd_teacher_mse_sum = 0.0
        bf_cmvd_teacher_mae_sum = 0.0
        bf_cmvd_teacher_delta_energy_sum = 0.0
        dmrb_l5_distill_l1_sum = 0.0
        dmrb_l20_distill_l1_sum = 0.0
        dmrb_l5_residual_energy_sum = 0.0
        dmrb_l20_residual_energy_sum = 0.0
        dmrb_l5_target_energy_sum = 0.0
        dmrb_l20_target_energy_sum = 0.0
        dmrb_l5_gradient_norm_max = 0.0
        dmrb_l20_gradient_norm_max = 0.0
        dmrb_teacher_reconstruction_max_abs = 0.0
        time_now = time.time()

        for epoch in range(self.args.train_epochs):
            epochs_completed = epoch + 1
            self.model.eval()
            core.mcpr.train()
            self.change_mode(0)
            epoch_mse = []
            for index, batch in enumerate(train_loader):
                if self.args.max_train_batches >= 0 and index >= self.args.max_train_batches:
                    break
                (
                    seq_x, seq_x_mask, seq_y, seq_y_mask, seq_true_x,
                    _seq_true_y, max_idx, min_idx, max_value, min_value,
                ) = tuple(item.float().to(self.device) for item in batch)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.model(
                    seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                )[:, -self.args.pred_len:, f_dim:]
                target = seq_y[:, -self.args.pred_len:, f_dim:]
                target_mask = seq_y_mask[:, -self.args.pred_len:, f_dim:] > 0.5
                if not bool(target_mask.any()):
                    continue
                step_mcpr_outputs = core.get_last_mcpr_outputs()
                if step_mcpr_outputs is None:
                    raise RuntimeError('MCPR training forward produced no adapter outputs')
                error = prediction[target_mask] - target[target_mask]
                mse_loss = error.pow(2).mean()
                mae_loss = error.abs().mean()
                if not bool(torch.isfinite(mse_loss).item()):
                    raise FloatingPointError('NAN_OR_DIVERGENCE: MCPR MSE is non-finite')
                if not bool(torch.isfinite(mae_loss).item()):
                    raise FloatingPointError('NAN_OR_DIVERGENCE: MCPR MAE is non-finite')
                ttrr_diagnostics = None
                ard_diagnostics = None
                bf_cmvd_diagnostics = None
                if objective == 'pareto_common_descent':
                    gradients, diagnostics = self._mcpr_common_descent_gradients(
                        mse_loss, mae_loss, trainable
                    )
                    for parameter, gradient in zip(trainable, gradients):
                        parameter.grad = gradient
                    cosine_sum += diagnostics['gradient_cosine']
                    cosine_min = min(cosine_min, diagnostics['gradient_cosine'])
                    mse_dot_min = min(mse_dot_min, diagnostics['common_mse_dot'])
                    mae_dot_min = min(mae_dot_min, diagnostics['common_mae_dot'])
                    gradient_norm_max = max(
                        gradient_norm_max,
                        math.sqrt(sum(
                            float(gradient.detach().double().pow(2).sum().item())
                            for gradient in gradients
                        )),
                    )
                    diagnostic_steps += 1
                elif objective == 'mse_only':
                    mse_loss.backward()
                    gradient_norm_max = max(
                        gradient_norm_max,
                        math.sqrt(sum(
                            float(parameter.grad.detach().double().pow(2).sum().item())
                            for parameter in trainable if parameter.grad is not None
                        )),
                    )
                elif objective == 'tail_trust_mse':
                    outputs = core.get_last_mcpr_outputs()
                    required_outputs = {
                        'residual', 'raw_residual', 'trust_gate', 'trust_features'
                    }
                    if outputs is None or not required_outputs.issubset(outputs):
                        raise RuntimeError('TTRR forward did not expose all trust diagnostics')
                    applied_residual = outputs['residual'][
                        :, -self.args.pred_len:, f_dim:
                    ]
                    if applied_residual.shape != prediction.shape:
                        raise RuntimeError('TTRR applied residual shape mismatch')
                    backbone_prediction = prediction - applied_residual
                    ttrr_diagnostics = self._ttrr_tail_trust_objective(
                        prediction=prediction,
                        backbone_prediction=backbone_prediction,
                        target=target,
                        target_mask=target_mask,
                        tail_fraction=float(self.args.ttrr_tail_fraction),
                        tail_weight=float(self.args.ttrr_tail_weight),
                    )
                    if not torch.allclose(
                        ttrr_diagnostics['candidate_mse'], mse_loss,
                        rtol=1e-6, atol=1e-7,
                    ):
                        raise RuntimeError('TTRR candidate MSE does not match MCPR MSE')
                    if not bool(torch.isfinite(ttrr_diagnostics['total_loss']).item()):
                        raise FloatingPointError(
                            'NAN_OR_DIVERGENCE: TTRR objective is non-finite'
                        )
                    ttrr_diagnostics['total_loss'].backward()
                    gradient_norm_max = max(
                        gradient_norm_max,
                        math.sqrt(sum(
                            float(parameter.grad.detach().double().pow(2).sum().item())
                            for parameter in trainable if parameter.grad is not None
                        )),
                    )
                elif objective == 'analytic_decision_distill':
                    outputs = core.get_last_mcpr_outputs()
                    required_outputs = {
                        'residual', 'raw_residual', 'ard_gate', 'ard_features',
                        'ard_action_logits', 'ard_action_probabilities',
                        'ard_partial_alpha',
                    }
                    if outputs is None or not required_outputs.issubset(outputs):
                        raise RuntimeError('ARD forward did not expose all diagnostics')
                    raw_residual = outputs['raw_residual'][
                        :, -self.args.pred_len:, f_dim:
                    ]
                    applied_residual = outputs['residual'][
                        :, -self.args.pred_len:, f_dim:
                    ]
                    ard_diagnostics = self._ard_analytic_distillation_objective(
                        raw_residual=raw_residual,
                        applied_residual=applied_residual,
                        deployed_prediction=prediction,
                        action_logits=outputs['ard_action_logits'],
                        student_gate=outputs['ard_gate'],
                        target=target,
                        target_mask=target_mask,
                        distill_enabled=(
                            epoch >= int(self.args.ard_distill_warmup_epochs)
                        ),
                        energy_epsilon=float(self.args.ard_energy_epsilon),
                        action_epsilon=float(self.args.ard_action_epsilon),
                        action_weight=float(self.args.ard_action_weight),
                        scale_weight=float(self.args.ard_scale_weight),
                    )
                    if not bool(torch.isfinite(
                        ard_diagnostics['total_loss']
                    ).item()):
                        raise FloatingPointError(
                            'NAN_OR_DIVERGENCE: ARD objective is non-finite'
                        )
                    ard_diagnostics['total_loss'].backward()
                    mse_loss = ard_diagnostics['raw_mse']
                    gradient_norm_max = max(
                        gradient_norm_max,
                        math.sqrt(sum(
                            float(parameter.grad.detach().double().pow(2).sum().item())
                            for parameter in trainable if parameter.grad is not None
                        )),
                    )
                elif objective in {
                    'benefit_filtered_multiview_distill',
                    'token_benefit_filtered_multiview_distill',
                    'dual_scale_residual_basis_distill',
                }:
                    if not bf_cmvd_enabled:
                        raise RuntimeError('BF-CMVD objective requires its explicit switch')
                    residual = step_mcpr_outputs['residual'][
                        :, -self.args.pred_len:, f_dim:
                    ]
                    if residual.shape != prediction.shape:
                        raise RuntimeError('BF-CMVD residual shape mismatch')
                    backbone_prediction = (prediction - residual).detach()
                    sample_start = (
                        epoch * len(train_loader) * int(self.args.batch_size)
                        + index * int(self.args.batch_size)
                    )
                    sample_indices = np.arange(
                        sample_start,
                        sample_start + int(seq_x.size(0)),
                        dtype=np.int64,
                    )
                    epoch_nested_seed = (
                        int(self.args.bf_cmvd_nested_seed)
                        + epoch * 1000003
                    )
                    nested_predictions = {}
                    previous_mcpr_state = bool(core.use_mcpr)
                    core.use_mcpr = False
                    try:
                        with torch.no_grad():
                            for block_length in (5, 20):
                                nested_mask = self._stage18_equal_count_observation_mask(
                                    seq_x_mask,
                                    sample_indices,
                                    float(self.args.bf_cmvd_target_rate),
                                    block_length,
                                    epoch_nested_seed,
                                )
                                nested_x = seq_true_x * nested_mask
                                nested_extrema = self._temporal_extrema(nested_x)
                                nested_predictions[block_length] = self.model(
                                    nested_x, nested_mask, *nested_extrema
                                )[:, -self.args.pred_len:, f_dim:].detach()
                    finally:
                        core.use_mcpr = previous_mcpr_state
                        core.last_mcpr_outputs = step_mcpr_outputs
                    teacher_prediction = (
                        float(self.args.bf_cmvd_original_weight)
                        * backbone_prediction
                        + float(self.args.bf_cmvd_l5_weight)
                        * nested_predictions[5]
                        + float(self.args.bf_cmvd_l20_weight)
                        * nested_predictions[20]
                    )
                    if objective == 'dual_scale_residual_basis_distill':
                        required_dmrb = {
                            'residual_l5', 'residual_l20', 'residual'
                        }
                        if not required_dmrb.issubset(step_mcpr_outputs):
                            raise RuntimeError('DMRB forward omitted scale residuals')
                        residual_l5 = step_mcpr_outputs['residual_l5'][
                            :, -self.args.pred_len:, f_dim:
                        ]
                        residual_l20 = step_mcpr_outputs['residual_l20'][
                            :, -self.args.pred_len:, f_dim:
                        ]
                        correction_l5 = (
                            float(self.args.bf_cmvd_l5_weight)
                            * (nested_predictions[5] - backbone_prediction)
                        ).detach()
                        correction_l20 = (
                            float(self.args.bf_cmvd_l20_weight)
                            * (nested_predictions[20] - backbone_prediction)
                        ).detach()
                        bf_cmvd_diagnostics = self._dmrb_objective(
                            student_prediction=prediction,
                            backbone_prediction=backbone_prediction,
                            teacher_prediction=teacher_prediction,
                            residual_l5=residual_l5,
                            residual_l20=residual_l20,
                            target_correction_l5=correction_l5,
                            target_correction_l20=correction_l20,
                            target=target,
                            target_mask=target_mask,
                            l5_enabled=bool(self.args.dmrb_l5_enabled),
                            l20_enabled=bool(self.args.dmrb_l20_enabled),
                        )
                    else:
                        bf_cmvd_diagnostics = self._bf_cmvd_objective(
                            student_prediction=prediction,
                            backbone_prediction=backbone_prediction,
                            teacher_prediction=teacher_prediction,
                            target=target,
                            target_mask=target_mask,
                            filter_granularity=(
                                'token'
                                if objective == 'token_benefit_filtered_multiview_distill'
                                else 'sample'
                            ),
                        )
                    if not bool(torch.isfinite(
                        bf_cmvd_diagnostics['total_loss']
                    ).item()):
                        raise FloatingPointError(
                            'NAN_OR_DIVERGENCE: BF-CMVD objective is non-finite'
                        )
                    bf_cmvd_diagnostics['total_loss'].backward()
                    mse_loss = bf_cmvd_diagnostics['student_mse']
                    mae_loss = bf_cmvd_diagnostics['student_mae']
                    if dmrb_enabled:
                        l5_gradient_square = sum(
                            float(parameter.grad.detach().double().pow(2).sum().item())
                            for parameter in core.mcpr.l5_basis.parameters()
                            if parameter.grad is not None
                        )
                        l20_gradient_square = sum(
                            float(parameter.grad.detach().double().pow(2).sum().item())
                            for parameter in core.mcpr.l20_basis.parameters()
                            if parameter.grad is not None
                        )
                        dmrb_l5_gradient_norm_max = max(
                            dmrb_l5_gradient_norm_max, math.sqrt(l5_gradient_square)
                        )
                        dmrb_l20_gradient_norm_max = max(
                            dmrb_l20_gradient_norm_max, math.sqrt(l20_gradient_square)
                        )
                    gradient_norm_max = max(
                        gradient_norm_max,
                        math.sqrt(sum(
                            float(parameter.grad.detach().double().pow(2).sum().item())
                            for parameter in trainable if parameter.grad is not None
                        )),
                    )
                else:
                    raise RuntimeError(f'unsupported MCPR objective: {objective}')
                if not any(parameter.grad is not None for parameter in trainable):
                    raise RuntimeError('MCPR produced no gradients')
                if any(
                    parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all().item())
                    for parameter in trainable
                ):
                    raise FloatingPointError('NAN_OR_DIVERGENCE: MCPR gradient is non-finite')
                optimizer.step()
                for parameter in trainable:
                    if not bool(torch.isfinite(parameter).all().item()):
                        raise FloatingPointError('NAN_OR_DIVERGENCE: MCPR parameter is non-finite')
                    parameter_abs_max = max(
                        parameter_abs_max, float(parameter.detach().abs().max().item())
                    )
                outputs = core.get_last_mcpr_outputs()
                if outputs is None:
                    raise RuntimeError('MCPR training forward produced no adapter outputs')
                residual_energy_sum += float(
                    outputs['residual'].detach().square().mean().item()
                )
                if dmrb_enabled:
                    dmrb_l5_residual_energy_sum += float(
                        outputs['residual_l5'].detach().square().mean().item()
                    )
                    dmrb_l20_residual_energy_sum += float(
                        outputs['residual_l20'].detach().square().mean().item()
                    )
                if ttrr_diagnostics is not None:
                    raw_residual = outputs['raw_residual'].detach()
                    applied_residual = outputs['residual'].detach()
                    gate = outputs['trust_gate'].detach().flatten()
                    trust_features = outputs['trust_features'].detach()
                    if not bool(torch.isfinite(gate).all().item()):
                        raise FloatingPointError('NAN_OR_DIVERGENCE: TTRR gate is non-finite')
                    if not bool(torch.isfinite(trust_features).all().item()):
                        raise FloatingPointError(
                            'NAN_OR_DIVERGENCE: TTRR trust features are non-finite'
                        )
                    if bool(((gate <= 0.0) | (gate >= 1.0)).any().item()):
                        raise RuntimeError('TTRR trust gate left the open interval (0, 1)')
                    ttrr_tail_sum += float(ttrr_diagnostics['tail_cvar'].detach().item())
                    ttrr_objective_sum += float(
                        ttrr_diagnostics['total_loss'].detach().item()
                    )
                    ttrr_candidate_sample_mse_sum += float(
                        ttrr_diagnostics['candidate_sample_mse_mean'].detach().item()
                    )
                    ttrr_backbone_sample_mse_sum += float(
                        ttrr_diagnostics['backbone_sample_mse_mean'].detach().item()
                    )
                    ttrr_delta_sum += float(
                        ttrr_diagnostics['sample_delta_mean'].detach().item()
                    )
                    ttrr_positive_harm_fraction_sum += float(
                        ttrr_diagnostics['positive_harm_fraction'].detach().item()
                    )
                    ttrr_raw_residual_energy_sum += float(
                        raw_residual.square().mean().item()
                    )
                    ttrr_applied_residual_energy_sum += float(
                        applied_residual.square().mean().item()
                    )
                    ttrr_gate_mean_sum += float(gate.mean().item())
                    ttrr_gate_std_sum += float(gate.std(unbiased=False).item())
                    ttrr_gate_min = min(ttrr_gate_min, float(gate.min().item()))
                    ttrr_gate_max = max(ttrr_gate_max, float(gate.max().item()))
                    ttrr_diagnostic_steps += 1
                if ard_diagnostics is not None:
                    gate = outputs['ard_gate'].detach().flatten()
                    features = outputs['ard_features'].detach()
                    probabilities = outputs['ard_action_probabilities'].detach()
                    raw_residual = outputs['raw_residual'].detach()
                    applied_residual = outputs['residual'].detach()
                    if not all(bool(torch.isfinite(value).all().item()) for value in (
                        gate, features, probabilities, raw_residual, applied_residual
                    )):
                        raise FloatingPointError(
                            'NAN_OR_DIVERGENCE: ARD diagnostics are non-finite'
                        )
                    if bool(((gate < 0.0) | (gate > 1.0)).any().item()):
                        raise RuntimeError('ARD gate left the closed interval [0,1]')
                    ard_gate_mean_sum += float(gate.mean().item())
                    ard_gate_std_sum += float(gate.std(unbiased=False).item())
                    ard_gate_min = min(ard_gate_min, float(gate.min().item()))
                    ard_gate_max = max(ard_gate_max, float(gate.max().item()))
                    entropy = -(
                        probabilities.clamp_min(1e-12)
                        * probabilities.clamp_min(1e-12).log()
                    ).sum(dim=-1).mean()
                    ard_action_entropy_sum += float(entropy.item())
                    ard_raw_residual_energy_sum += float(
                        raw_residual.square().mean().item()
                    )
                    ard_applied_residual_energy_sum += float(
                        applied_residual.square().mean().item()
                    )
                    ard_gate_stat_steps += 1
                    if int(ard_diagnostics['valid_sample_count']) > 0:
                        teacher = ard_diagnostics['teacher_alpha'].detach()
                        actions = ard_diagnostics['teacher_actions'].detach()
                        ard_distillation_steps += 1
                        ard_teacher_valid_samples += int(teacher.numel())
                        ard_teacher_alpha_sum += float(teacher.sum().item())
                        ard_teacher_alpha_square_sum += float(
                            teacher.square().sum().item()
                        )
                        ard_teacher_alpha_min = min(
                            ard_teacher_alpha_min, float(teacher.min().item())
                        )
                        ard_teacher_alpha_max = max(
                            ard_teacher_alpha_max, float(teacher.max().item())
                        )
                        for action_index in range(3):
                            ard_teacher_action_counts[action_index] += int(
                                (actions == action_index).sum().item()
                            )
                        ard_action_loss_sum += float(
                            ard_diagnostics['action_loss'].detach().item()
                        )
                        ard_scale_loss_sum += float(
                            ard_diagnostics['scale_loss'].detach().item()
                        )
                if bf_cmvd_diagnostics is not None:
                    beneficial = bf_cmvd_diagnostics['beneficial_mask'].detach()
                    eligible = bf_cmvd_diagnostics['eligible_mask'].detach()
                    diagnostic_values = (
                        bf_cmvd_diagnostics['distill_l1'],
                        bf_cmvd_diagnostics['backbone_mse'],
                        bf_cmvd_diagnostics['backbone_mae'],
                        bf_cmvd_diagnostics['teacher_mse'],
                        bf_cmvd_diagnostics['teacher_mae'],
                        bf_cmvd_diagnostics['teacher_delta_energy'],
                    )
                    if dmrb_enabled:
                        diagnostic_values += (
                            bf_cmvd_diagnostics['dmrb_l5_distill_l1'],
                            bf_cmvd_diagnostics['dmrb_l20_distill_l1'],
                            bf_cmvd_diagnostics['dmrb_l5_target_energy'],
                            bf_cmvd_diagnostics['dmrb_l20_target_energy'],
                            bf_cmvd_diagnostics[
                                'dmrb_teacher_reconstruction_max_abs'
                            ],
                        )
                    if not all(
                        bool(torch.isfinite(value).all().item())
                        for value in diagnostic_values
                    ):
                        raise FloatingPointError(
                            'NAN_OR_DIVERGENCE: BF-CMVD diagnostics are non-finite'
                        )
                    bf_cmvd_diagnostic_steps += 1
                    bf_cmvd_beneficial_samples += int(beneficial.sum().item())
                    bf_cmvd_total_samples += int(eligible.sum().item())
                    bf_cmvd_distill_l1_sum += float(
                        bf_cmvd_diagnostics['distill_l1'].detach().item()
                    )
                    bf_cmvd_backbone_mse_sum += float(
                        bf_cmvd_diagnostics['backbone_mse'].detach().item()
                    )
                    bf_cmvd_backbone_mae_sum += float(
                        bf_cmvd_diagnostics['backbone_mae'].detach().item()
                    )
                    bf_cmvd_teacher_mse_sum += float(
                        bf_cmvd_diagnostics['teacher_mse'].detach().item()
                    )
                    bf_cmvd_teacher_mae_sum += float(
                        bf_cmvd_diagnostics['teacher_mae'].detach().item()
                    )
                    bf_cmvd_teacher_delta_energy_sum += float(
                        bf_cmvd_diagnostics['teacher_delta_energy'].detach().item()
                    )
                    if dmrb_enabled:
                        dmrb_l5_distill_l1_sum += float(
                            bf_cmvd_diagnostics[
                                'dmrb_l5_distill_l1'
                            ].detach().item()
                        )
                        dmrb_l20_distill_l1_sum += float(
                            bf_cmvd_diagnostics[
                                'dmrb_l20_distill_l1'
                            ].detach().item()
                        )
                        dmrb_l5_target_energy_sum += float(
                            bf_cmvd_diagnostics[
                                'dmrb_l5_target_energy'
                            ].detach().item()
                        )
                        dmrb_l20_target_energy_sum += float(
                            bf_cmvd_diagnostics[
                                'dmrb_l20_target_energy'
                            ].detach().item()
                        )
                        dmrb_teacher_reconstruction_max_abs = max(
                            dmrb_teacher_reconstruction_max_abs,
                            float(bf_cmvd_diagnostics[
                                'dmrb_teacher_reconstruction_max_abs'
                            ].detach().item()),
                        )
                mse_sum += float(mse_loss.detach().item())
                mae_sum += float(mae_loss.detach().item())
                epoch_mse.append(float(mse_loss.detach().item()))
                steps += 1
                if (index + 1) % 50 == 0:
                    elapsed = time.time() - time_now
                    print(
                        f'MCPR iters={index + 1} epoch={epoch + 1} '
                        f'mse={mse_loss.item():.7f} mae={mae_loss.item():.7f} '
                        f'elapsed={elapsed:.2f}s', flush=True,
                    )
                    time_now = time.time()
            if not epoch_mse:
                raise RuntimeError('MCPR epoch completed without optimizer steps')
            validation_loss = self._mcpr_validation_loss(validation_loader)
            print(
                f'MCPR epoch={epoch + 1} train_mse={np.mean(epoch_mse):.7f} '
                f'validation_mse={validation_loss:.7f}', flush=True,
            )
            early_stopping(validation_loss, self.model, path)
            if early_stopping.early_stop:
                print('MCPR early stopping', flush=True)
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        checkpoint = os.path.join(path, 'checkpoint.pth')
        if not os.path.isfile(checkpoint):
            raise RuntimeError('MCPR did not produce a checkpoint')
        self.load_model(checkpoint)
        backbone_hash_after = self._mcpr_backbone_hash()
        memory_hash_after = self._fusr_v2_memory_hash()
        if backbone_hash_after != backbone_hash_before:
            raise RuntimeError('MCPR changed the frozen PG95 backbone')
        if memory_hash_after != memory_hash_before:
            raise RuntimeError('MCPR changed the frozen prototype memory')
        self.run_evidence = {
            'optimizer_steps': int(steps),
            'train_batches_processed': int(steps),
            'validation_batches_processed': int(getattr(self, 'last_val_batches', 0)),
            'epochs_completed': int(epochs_completed),
            'best_observed_validation_loss': float(early_stopping.val_loss_min),
            'memory_bank_warmup_count': 0,
            'mipp_enabled': True,
            'mipp_frozen_source_only': True,
            'mcpr_enabled': True,
            'mcpr_objective': objective,
            'mcpr_steps': int(steps),
            'mcpr_rank': int(self.args.mcpr_rank),
            'mcpr_kernel_size': int(self.args.mcpr_kernel_size),
            'mcpr_learning_rate': float(self.args.mcpr_learning_rate),
            'mcpr_source_checkpoint': source_checkpoint,
            'mcpr_source_checkpoint_sha256': source_sha256,
            'mcpr_checkpoint_format': (
                'mcpr_delta_v1'
                if bool(getattr(self.args, 'mcpr_delta_checkpoint', False))
                else 'full_model_v1'
            ),
            'mcpr_backbone_frozen': True,
            'mcpr_backbone_hash_before': backbone_hash_before,
            'mcpr_backbone_hash_after': backbone_hash_after,
            'mcpr_memory_frozen': True,
            'mcpr_memory_hash_before': memory_hash_before,
            'mcpr_memory_hash_after': memory_hash_after,
            'mcpr_trainable_parameter_count': int(sum(p.numel() for p in trainable)),
            'mcpr_training_mse_mean': float(mse_sum / max(1, steps)),
            'mcpr_training_mae_mean': float(mae_sum / max(1, steps)),
            'mcpr_residual_energy_mean': float(residual_energy_sum / max(1, steps)),
            'mcpr_gradient_norm_max': float(gradient_norm_max),
            'mcpr_parameter_abs_max': float(parameter_abs_max),
            'mcpr_pareto_diagnostic_steps': int(diagnostic_steps),
            'mcpr_gradient_cosine_mean': (
                float(cosine_sum / diagnostic_steps) if diagnostic_steps else None
            ),
            'mcpr_gradient_cosine_min': float(cosine_min) if diagnostic_steps else None,
            'mcpr_common_mse_dot_min': float(mse_dot_min) if diagnostic_steps else None,
            'mcpr_common_mae_dot_min': float(mae_dot_min) if diagnostic_steps else None,
            'ttrr_enabled': ttrr_enabled,
            'ttrr_tail_fraction': float(self.args.ttrr_tail_fraction),
            'ttrr_tail_weight': float(self.args.ttrr_tail_weight),
            'ttrr_initial_gate_mean': 0.5 if ttrr_enabled else None,
            'ttrr_diagnostic_steps': int(ttrr_diagnostic_steps),
            'ttrr_total_loss_mean': (
                float(ttrr_objective_sum / ttrr_diagnostic_steps)
                if ttrr_diagnostic_steps else None
            ),
            'ttrr_tail_cvar_mean': (
                float(ttrr_tail_sum / ttrr_diagnostic_steps)
                if ttrr_diagnostic_steps else None
            ),
            'ttrr_candidate_sample_mse_mean': (
                float(ttrr_candidate_sample_mse_sum / ttrr_diagnostic_steps)
                if ttrr_diagnostic_steps else None
            ),
            'ttrr_backbone_sample_mse_mean': (
                float(ttrr_backbone_sample_mse_sum / ttrr_diagnostic_steps)
                if ttrr_diagnostic_steps else None
            ),
            'ttrr_sample_delta_mse_mean': (
                float(ttrr_delta_sum / ttrr_diagnostic_steps)
                if ttrr_diagnostic_steps else None
            ),
            'ttrr_positive_harm_fraction_mean': (
                float(ttrr_positive_harm_fraction_sum / ttrr_diagnostic_steps)
                if ttrr_diagnostic_steps else None
            ),
            'ttrr_raw_residual_energy_mean': (
                float(ttrr_raw_residual_energy_sum / ttrr_diagnostic_steps)
                if ttrr_diagnostic_steps else None
            ),
            'ttrr_applied_residual_energy_mean': (
                float(ttrr_applied_residual_energy_sum / ttrr_diagnostic_steps)
                if ttrr_diagnostic_steps else None
            ),
            'ttrr_gate_mean': (
                float(ttrr_gate_mean_sum / ttrr_diagnostic_steps)
                if ttrr_diagnostic_steps else None
            ),
            'ttrr_gate_std_mean': (
                float(ttrr_gate_std_sum / ttrr_diagnostic_steps)
                if ttrr_diagnostic_steps else None
            ),
            'ttrr_gate_min': float(ttrr_gate_min) if ttrr_diagnostic_steps else None,
            'ttrr_gate_max': float(ttrr_gate_max) if ttrr_diagnostic_steps else None,
            'ard_enabled': ard_enabled,
            'ard_hidden_dim': int(self.args.ard_hidden_dim),
            'ard_distill_warmup_epochs': int(
                self.args.ard_distill_warmup_epochs
            ),
            'ard_energy_epsilon': float(self.args.ard_energy_epsilon),
            'ard_action_epsilon': float(self.args.ard_action_epsilon),
            'ard_action_weight': float(self.args.ard_action_weight),
            'ard_scale_weight': float(self.args.ard_scale_weight),
            'ard_initial_gate_mean': 0.5 if ard_enabled else None,
            'ard_distillation_steps': int(ard_distillation_steps),
            'ard_teacher_valid_samples': int(ard_teacher_valid_samples),
            'ard_teacher_off_count': int(ard_teacher_action_counts[0]),
            'ard_teacher_partial_count': int(ard_teacher_action_counts[1]),
            'ard_teacher_full_count': int(ard_teacher_action_counts[2]),
            'ard_teacher_alpha_mean': (
                float(ard_teacher_alpha_sum / ard_teacher_valid_samples)
                if ard_teacher_valid_samples else None
            ),
            'ard_teacher_alpha_std': (
                float(max(
                    0.0,
                    ard_teacher_alpha_square_sum / ard_teacher_valid_samples
                    - (ard_teacher_alpha_sum / ard_teacher_valid_samples) ** 2,
                ) ** 0.5)
                if ard_teacher_valid_samples else None
            ),
            'ard_teacher_alpha_min': (
                float(ard_teacher_alpha_min) if ard_teacher_valid_samples else None
            ),
            'ard_teacher_alpha_max': (
                float(ard_teacher_alpha_max) if ard_teacher_valid_samples else None
            ),
            'ard_action_loss_mean': (
                float(ard_action_loss_sum / ard_distillation_steps)
                if ard_distillation_steps else None
            ),
            'ard_scale_loss_mean': (
                float(ard_scale_loss_sum / ard_distillation_steps)
                if ard_distillation_steps else None
            ),
            'ard_gate_mean': (
                float(ard_gate_mean_sum / ard_gate_stat_steps)
                if ard_gate_stat_steps else None
            ),
            'ard_gate_std_mean': (
                float(ard_gate_std_sum / ard_gate_stat_steps)
                if ard_gate_stat_steps else None
            ),
            'ard_gate_min': float(ard_gate_min) if ard_gate_stat_steps else None,
            'ard_gate_max': float(ard_gate_max) if ard_gate_stat_steps else None,
            'ard_action_entropy_mean': (
                float(ard_action_entropy_sum / ard_gate_stat_steps)
                if ard_gate_stat_steps else None
            ),
            'ard_raw_residual_energy_mean': (
                float(ard_raw_residual_energy_sum / ard_gate_stat_steps)
                if ard_gate_stat_steps else None
            ),
            'ard_applied_residual_energy_mean': (
                float(ard_applied_residual_energy_sum / ard_gate_stat_steps)
                if ard_gate_stat_steps else None
            ),
            'dmrb_enabled': dmrb_enabled,
            'dmrb_l5_enabled': (
                bool(self.args.dmrb_l5_enabled) if dmrb_enabled else False
            ),
            'dmrb_l20_enabled': (
                bool(self.args.dmrb_l20_enabled) if dmrb_enabled else False
            ),
            'dmrb_inference_backbone_forward_count': 1 if dmrb_enabled else None,
            'dmrb_fixed_residual_fusion': 'sum' if dmrb_enabled else None,
            'dmrb_l5_target_definition': (
                '0.25*(prediction_l5-backbone)' if dmrb_enabled else None
            ),
            'dmrb_l20_target_definition': (
                '0.25*(prediction_l20-backbone)' if dmrb_enabled else None
            ),
            'dmrb_l5_distill_l1_mean': (
                float(dmrb_l5_distill_l1_sum / bf_cmvd_diagnostic_steps)
                if dmrb_enabled and bf_cmvd_diagnostic_steps else None
            ),
            'dmrb_l20_distill_l1_mean': (
                float(dmrb_l20_distill_l1_sum / bf_cmvd_diagnostic_steps)
                if dmrb_enabled and bf_cmvd_diagnostic_steps else None
            ),
            'dmrb_l5_residual_energy_mean': (
                float(dmrb_l5_residual_energy_sum / max(1, steps))
                if dmrb_enabled else None
            ),
            'dmrb_l20_residual_energy_mean': (
                float(dmrb_l20_residual_energy_sum / max(1, steps))
                if dmrb_enabled else None
            ),
            'dmrb_l5_target_energy_mean': (
                float(dmrb_l5_target_energy_sum / bf_cmvd_diagnostic_steps)
                if dmrb_enabled and bf_cmvd_diagnostic_steps else None
            ),
            'dmrb_l20_target_energy_mean': (
                float(dmrb_l20_target_energy_sum / bf_cmvd_diagnostic_steps)
                if dmrb_enabled and bf_cmvd_diagnostic_steps else None
            ),
            'dmrb_l5_gradient_norm_max': (
                float(dmrb_l5_gradient_norm_max) if dmrb_enabled else None
            ),
            'dmrb_l20_gradient_norm_max': (
                float(dmrb_l20_gradient_norm_max) if dmrb_enabled else None
            ),
            'dmrb_teacher_reconstruction_max_abs': (
                float(dmrb_teacher_reconstruction_max_abs)
                if dmrb_enabled else None
            ),
            'bf_cmvd_enabled': bf_cmvd_enabled,
            'bf_cmvd_filter_granularity': (
                'token'
                if objective in {
                    'token_benefit_filtered_multiview_distill',
                    'dual_scale_residual_basis_distill',
                }
                else ('sample' if bf_cmvd_enabled else None)
            ),
            'bf_cmvd_target_rate': (
                float(self.args.bf_cmvd_target_rate)
                if bf_cmvd_enabled else None
            ),
            'bf_cmvd_block_lengths': (
                [int(value) for value in self.args.bf_cmvd_block_lengths]
                if bf_cmvd_enabled else None
            ),
            'bf_cmvd_nested_seed': (
                int(self.args.bf_cmvd_nested_seed)
                if bf_cmvd_enabled else None
            ),
            'bf_cmvd_epoch_seed_stride': 1000003 if bf_cmvd_enabled else None,
            'bf_cmvd_teacher_weights': (
                {
                    'original': float(self.args.bf_cmvd_original_weight),
                    'l5': float(self.args.bf_cmvd_l5_weight),
                    'l20': float(self.args.bf_cmvd_l20_weight),
                }
                if bf_cmvd_enabled else None
            ),
            'bf_cmvd_direct_loss': 'observed_target_mse' if bf_cmvd_enabled else None,
            'bf_cmvd_distill_loss': (
                (
                    'beneficial_token_scale_factored_observed_target_l1'
                    if dmrb_enabled
                    else 'beneficial_sample_observed_target_l1'
                )
                if bf_cmvd_enabled else None
            ),
            'bf_cmvd_benefit_rule': (
                'teacher_mse_lt_backbone_and_teacher_mae_lt_backbone'
                if bf_cmvd_enabled else None
            ),
            'bf_cmvd_diagnostic_steps': int(bf_cmvd_diagnostic_steps),
            'bf_cmvd_beneficial_samples': int(bf_cmvd_beneficial_samples),
            'bf_cmvd_total_samples': int(bf_cmvd_total_samples),
            'bf_cmvd_beneficial_units': int(bf_cmvd_beneficial_samples),
            'bf_cmvd_total_units': int(bf_cmvd_total_samples),
            'bf_cmvd_beneficial_fraction': (
                float(bf_cmvd_beneficial_samples / bf_cmvd_total_samples)
                if bf_cmvd_total_samples else None
            ),
            'bf_cmvd_distill_l1_mean': (
                float(bf_cmvd_distill_l1_sum / bf_cmvd_diagnostic_steps)
                if bf_cmvd_diagnostic_steps else None
            ),
            'bf_cmvd_backbone_training_mse_mean': (
                float(bf_cmvd_backbone_mse_sum / bf_cmvd_diagnostic_steps)
                if bf_cmvd_diagnostic_steps else None
            ),
            'bf_cmvd_backbone_training_mae_mean': (
                float(bf_cmvd_backbone_mae_sum / bf_cmvd_diagnostic_steps)
                if bf_cmvd_diagnostic_steps else None
            ),
            'bf_cmvd_teacher_training_mse_mean': (
                float(bf_cmvd_teacher_mse_sum / bf_cmvd_diagnostic_steps)
                if bf_cmvd_diagnostic_steps else None
            ),
            'bf_cmvd_teacher_training_mae_mean': (
                float(bf_cmvd_teacher_mae_sum / bf_cmvd_diagnostic_steps)
                if bf_cmvd_diagnostic_steps else None
            ),
            'bf_cmvd_teacher_delta_energy_mean': (
                float(
                    bf_cmvd_teacher_delta_energy_sum / bf_cmvd_diagnostic_steps
                )
                if bf_cmvd_diagnostic_steps else None
            ),
        }
        return self.model

    def train(self, setting):
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        train_data, train_loader = self._get_data(flag='train')
        # print(train_data.data_x)
        vali_data, vali_loader = self._get_data(flag='val')
        f_dim = -1 if self.args.features == 'MS' else 0
    
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if bool(getattr(self.args, 'fusr_router_only', False)):
            return self._train_fusr_v2(setting, vali_data, path)
        if bool(getattr(self.args, 'use_pba_cmvd', False)):
            return self._train_pba_cmvd(setting, train_loader, vali_loader, path)
        if bool(getattr(self.args, 'use_mcpr', False)):
            return self._train_mcpr(setting, train_loader, vali_loader, path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(args =self.args,patience=self.args.patience, verbose=True,more_save=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        optimizer_steps = 0
        train_batches_processed = 0
        memory_bank_warmup_count = 0
        mipp_steps = 0
        mipp_pair_views_generated = 0
        mipp_metric_sums = {
            'total_loss': 0.0,
            'invariance_loss': 0.0,
            'variance_loss': 0.0,
            'covariance_loss': 0.0,
            'decorrelation_loss': 0.0,
            'mask_reconstruction_loss': 0.0,
            'paired_content_cosine': 0.0,
            'teacher_pair_cosine': 0.0,
            'stable_write_acceptance_rate': 0.0,
            'write_stability': 0.0,
            'write_stability_std': 0.0,
            'write_effective_cutoff': 0.0,
            'write_raw_absolute_acceptance_rate': 0.0,
            'write_eligible_count': 0.0,
            'write_sampled_count': 0.0,
            'write_updated_count': 0.0,
            'write_candidate_pairwise_cosine': 0.0,
            'write_selected_pairwise_cosine': 0.0,
            'write_selected_stability': 0.0,
            'write_diversity_gain': 0.0,
            'write_mask_candidate_pairwise_distance': 0.0,
            'write_mask_selected_pairwise_distance': 0.0,
            'write_mask_coverage_distance': 0.0,
            'write_mask_topk_coverage_distance': 0.0,
            'write_mask_coverage_gain_vs_topk': 0.0,
            'write_mask_candidate_missing_rate': 0.0,
            'write_mask_selected_missing_rate': 0.0,
            'pair_missing_rate': 0.0,
            'pair_block_length': 0.0,
        }
        mipp_warm_diagnostics = []
        cfr_steps = 0
        cfr_samples = 0
        cfr_metric_sums = {
            'total_loss': 0.0,
            'prototype_on_mse': 0.0,
            'prototype_off_mse': 0.0,
            'signed_regret': 0.0,
            'positive_regret': 0.0,
            'harmful_retrieval_rate': 0.0,
        }
        fusr_steps = 0
        fusr_utility_pairs_generated = 0
        fusr_metric_sums = {
            'total_loss': 0.0,
            'regression_loss': 0.0,
            'sign_loss': 0.0,
            'utility_target_mean': 0.0,
            'utility_target_std': 0.0,
            'utility_positive_rate': 0.0,
            'utility_prediction_mean': 0.0,
            'utility_prediction_std': 0.0,
            'utility_uncertainty_mean': 0.0,
            'utility_sign_accuracy': 0.0,
            'gate_mean': 0.0,
            'abstention_rate': 0.0,
            'spectral_uncertainty': 0.0,
        }
        mrra_steps = 0
        mrra_missing_elements = 0
        mrra_metric_sums = {
            'reconstruction_loss': 0.0,
            'context_residual_energy': 0.0,
            'reconstruction_correction_energy': 0.0,
        }
        mrra_diagnostic_steps = 0
        mrra_gradient_l2_sum = 0.0
        mrra_gradient_l2_max = 0.0
        mrra_parameter_abs_max = 0.0
        initial_optimizer_group_lrs = {
            group.get('group_name', f'group_{index}'): float(group['lr'])
            for index, group in enumerate(model_optim.param_groups)
        }
        radr_steps = 0
        radr_samples = 0
        radr_metric_sums = {
            'total_loss': 0.0,
            'tail_mse': 0.0,
            'body_mse': 0.0,
            'body_mae': 0.0,
            'normalized_tail_risk': 0.0,
            'normalized_body_risk': 0.0,
            'difficulty_rank_std': 0.0,
            'hard_body_mse_gap': 0.0,
        }
        epochs_completed = 0

        for epoch in range(self.args.train_epochs):
            epochs_completed = epoch + 1
            iter_count = 0
            train_loss = []
            pred = []
            self.model.train()
            self.change_mode(1)
            if not self.args.no_renew:
                if "Sea" in self.args.model:
                    self.model.mem_net1.memory.init_bank()
                    self.model.mem_net2.memory.init_bank()
                else:
                    self.model.mem_net.memory.init_bank()                
            epoch_time = time.time()
            self.model.count=0
            print(
                "learning_rates: ",
                {
                    group.get('group_name', f'group_{index}'): group['lr']
                    for index, group in enumerate(model_optim.param_groups)
                },
            )
            for i, (seq_x,seq_x_mask,seq_y,seq_y_mask,seq_true_x,seq_true_y,max_idx,min_idx,max_value,min_value) in enumerate(train_loader):
                if self.args.max_train_batches >= 0 and i >= self.args.max_train_batches:
                    break
                seq_x,seq_x_mask,seq_y,seq_y_mask,seq_true_x,seq_true_y,max_idx,min_idx,max_value,min_value \
                    = seq_x.float().to(self.device),seq_x_mask.float().to(self.device),seq_y.float().to(self.device),seq_y_mask.float().to(self.device),seq_true_x.float().to(self.device),seq_true_y.float().to(self.device),max_idx.float().to(self.device),min_idx.float().to(self.device),max_value.float().to(self.device),min_value.float().to(self.device)
                iter_count += 1
                model_optim.zero_grad()
                mipp_pair = self._build_mipp_pair(seq_true_x)
                if mipp_pair is not None:
                    mipp_pair_views_generated += 1
                # The public code clears the prototype bank at every epoch when
                # no_renew is false, but originally warmed it only in epoch 0.
                # Warm every newly cleared bank before its first retrieval.
                if i == 0 and (epoch == 0 or not self.args.no_renew):
                    print(
                        f"MEMORY_BANK_WARMUP epoch={epoch + 1} "
                        f"reason={'initial' if epoch == 0 else 'renewed'}"
                    )
                    outputs =self.model.warmup(
                        seq_x,seq_x_mask,max_idx,min_idx,max_value,min_value,
                        mipp_pair=mipp_pair,
                    )
                    warm_diagnostics = self._core_model().get_last_mipp_warm_diagnostics()
                    if warm_diagnostics is not None:
                        mipp_warm_diagnostics.append(warm_diagnostics)
                    memory_bank_warmup_count += 1
                    continue
                else:
                    outputs = self.model(
                        seq_x,seq_x_mask,max_idx,min_idx,max_value,min_value,
                        mipp_pair=mipp_pair,
                    )

                seq_y = seq_y[:, -self.args.pred_len:, f_dim:]
                seq_y_mask = seq_y_mask[:, -self.args.pred_len:, f_dim:]
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                
                seq_y_mask = seq_y_mask.detach()
                
                if self.args.use_radr:
                    radr_objective = self._rank_adaptive_dual_risk_objective(
                        outputs, seq_y, seq_y_mask
                    )
                    forecast_loss = radr_objective['total_loss']
                    radr_steps += 1
                    radr_samples += int(radr_objective['sample_count'])
                    for key in radr_metric_sums:
                        value = radr_objective[key]
                        if torch.is_tensor(value):
                            value = value.detach().item()
                        radr_metric_sums[key] += float(value)
                else:
                    forecast_loss = criterion(
                        outputs[seq_y_mask==1.0], seq_y[seq_y_mask==1.0]
                    )
                loss = forecast_loss
                if self.args.use_mrra:
                    mrra_outputs = self._core_model().get_last_mrra_outputs()
                    if mrra_outputs is None:
                        raise RuntimeError('MRRA enabled training step produced no outputs')
                    reconstruction_loss, missing_elements = (
                        self._mrra_reconstruction_objective(
                            mrra_outputs['reconstruction'],
                            seq_true_x,
                            seq_x_mask,
                        )
                    )
                    loss = loss + float(
                        self.args.mrra_reconstruction_weight
                    ) * reconstruction_loss
                    mrra_steps += 1
                    mrra_missing_elements += missing_elements
                    mrra_metric_sums['reconstruction_loss'] += float(
                        reconstruction_loss.detach().item()
                    )
                    mrra_metric_sums['context_residual_energy'] += float(
                        mrra_outputs['context_residual'].detach().pow(2).mean().item()
                    )
                    mrra_metric_sums['reconstruction_correction_energy'] += float(
                        mrra_outputs['reconstruction_correction']
                        .detach().pow(2).mean().item()
                    )
                if self.args.use_cfr:
                    cfr_outputs = self._core_model().get_last_cfr_outputs()
                    if cfr_outputs is None:
                        raise RuntimeError('CFR enabled training step produced no paired outputs')
                    prediction_on = cfr_outputs['prediction_on'][
                        :, -self.args.pred_len:, f_dim:
                    ]
                    prediction_off = cfr_outputs['prediction_off'][
                        :, -self.args.pred_len:, f_dim:
                    ]
                    cfr_loss = self._counterfactual_regret_objective(
                        prediction_on,
                        prediction_off,
                        seq_y,
                        seq_y_mask,
                    )
                    loss = loss + cfr_loss['total_loss']
                    cfr_steps += 1
                    cfr_samples += int(cfr_loss['sample_count'])
                    for key in cfr_metric_sums:
                        value = cfr_loss[key]
                        if torch.is_tensor(value):
                            value = value.detach().item()
                        cfr_metric_sums[key] += float(value)
                if self.args.use_fusr:
                    fusr_outputs = self._core_model().get_last_fusr_outputs()
                    if fusr_outputs is None:
                        raise RuntimeError('FUSR enabled training step produced no router outputs')
                    prediction_on = fusr_outputs['prediction_on'][:, -self.args.pred_len:, f_dim:]
                    prediction_off = fusr_outputs['prediction_off'][:, -self.args.pred_len:, f_dim:]
                    fusr_outputs_for_loss = dict(fusr_outputs)
                    fusr_outputs_for_loss['prediction_on'] = prediction_on
                    fusr_outputs_for_loss['prediction_off'] = prediction_off
                    fusr_loss = self._fusr_training_objective(
                        fusr_outputs_for_loss, seq_y, seq_y_mask
                    )
                    if fusr_loss is not None:
                        loss = loss + fusr_loss['total_loss']
                        fusr_steps += 1
                        fusr_utility_pairs_generated += int(fusr_loss['utility_pair_count'])
                        for key in fusr_metric_sums:
                            value = fusr_loss[key]
                            if torch.is_tensor(value):
                                value = value.detach().item()
                            fusr_metric_sums[key] += float(value)
                mipp_outputs = self._core_model().get_last_mipp_outputs()
                if self.args.use_mipp and mipp_pair is not None:
                    if mipp_outputs is None or 'total_loss' not in mipp_outputs:
                        raise RuntimeError('MIPP enabled training step produced no paired objective')
                    loss = loss + mipp_outputs['total_loss']
                    mipp_steps += 1
                    for key in (
                        'total_loss','invariance_loss','variance_loss','covariance_loss',
                        'decorrelation_loss','mask_reconstruction_loss',
                        'paired_content_cosine','teacher_pair_cosine',
                        'stable_write_acceptance_rate','write_stability',
                        'write_stability_std','write_effective_cutoff',
                        'write_raw_absolute_acceptance_rate',
                        'write_eligible_count','write_sampled_count',
                        'write_updated_count',
                        'write_candidate_pairwise_cosine',
                        'write_selected_pairwise_cosine',
                        'write_selected_stability','write_diversity_gain',
                        'write_mask_candidate_pairwise_distance',
                        'write_mask_selected_pairwise_distance',
                        'write_mask_coverage_distance',
                        'write_mask_topk_coverage_distance',
                        'write_mask_coverage_gain_vs_topk',
                        'write_mask_candidate_missing_rate',
                        'write_mask_selected_missing_rate',
                    ):
                        value = mipp_outputs.get(key)
                        if value is not None:
                            if torch.is_tensor(value):
                                value = value.detach().item()
                            mipp_metric_sums[key] += float(value)
                    mipp_metric_sums['pair_missing_rate'] += float(
                        mipp_pair['realized_missing_rate']
                    )
                    mipp_metric_sums['pair_block_length'] += float(
                        mipp_pair['block_length']
                    )
                if not bool(torch.isfinite(loss).all().item()):
                    raise FloatingPointError(
                        'NAN_OR_DIVERGENCE: non-finite training objective '
                        f'at epoch={epoch + 1}, batch={i + 1}; '
                        f'forecast_loss={forecast_loss.detach().item()}'
                    )
                train_loss.append(forecast_loss.item())
                if (i + 1) % 50 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    if self.args.analysis:
                        save_model1(epoch,i,self.model,path)
                loss.backward()
                if self.args.use_mrra:
                    diagnostics = self._mrra_parameter_diagnostics()
                    if not diagnostics['gradient_found']:
                        raise RuntimeError('MRRA training step produced no adapter gradients')
                    mrra_diagnostic_steps += 1
                    mrra_gradient_l2_sum += diagnostics['gradient_l2']
                    mrra_gradient_l2_max = max(
                        mrra_gradient_l2_max, diagnostics['gradient_l2']
                    )
                    mrra_parameter_abs_max = max(
                        mrra_parameter_abs_max, diagnostics['parameter_abs_max']
                    )
                model_optim.step()
                if self.args.use_mrra:
                    diagnostics = self._mrra_parameter_diagnostics()
                    mrra_parameter_abs_max = max(
                        mrra_parameter_abs_max, diagnostics['parameter_abs_max']
                    )
                optimizer_steps += 1
                train_batches_processed += 1
                
            # save_model1(epoch,self.model,path)
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion,epoch)
            if not np.isfinite(vali_loss):
                raise FloatingPointError(
                    'NAN_OR_DIVERGENCE: non-finite validation objective '
                    f'at epoch={epoch + 1}'
                )
            wandb.log({'valid_loss':vali_loss,'train_loss':train_loss})
            wandb.log({"num_bank1":self.model.mem_net.memory.num_bank()})
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(model_optim, epoch + 1, self.args)
            
        best_model_path = path + '/' + 'checkpoint.pth'
        wandb.save(best_model_path)

        self.load_model(best_model_path)
        self.run_evidence = {
            'optimizer_steps': optimizer_steps,
            'train_batches_processed': train_batches_processed,
            'validation_batches_processed': getattr(self, 'last_val_batches', 0),
            'epochs_completed': int(epochs_completed),
            'best_observed_validation_loss': float(early_stopping.val_loss_min),
            'memory_bank_warmup_count': memory_bank_warmup_count,
            'mipp_enabled': bool(self.args.use_mipp),
            'mipp_steps': int(mipp_steps),
            'mipp_pair_views_generated': int(mipp_pair_views_generated),
            'mipp_write_gate_mode': str(
                getattr(self.args, 'mipp_write_gate_mode', 'absolute')
            ),
            'mipp_write_budget': int(
                getattr(self.args, 'mipp_write_budget', self.args.M)
            ),
            'cfr_enabled': bool(self.args.use_cfr),
            'cfr_steps': int(cfr_steps),
            'cfr_samples': int(cfr_samples),
            'cfr_reference_detached': True,
            'cfr_inference_path': 'prototype_on_single_path',
            'fusr_enabled': bool(self.args.use_fusr),
            'fusr_steps': int(fusr_steps),
            'fusr_utility_pairs_generated': int(fusr_utility_pairs_generated),
            'mrra_enabled': bool(self.args.use_mrra),
            'mrra_version': str(getattr(self.args, 'mrra_version', 'v1')),
            'mrra_steps': int(mrra_steps),
            'mrra_missing_elements': int(mrra_missing_elements),
            'mrra_reconstruction_weight': float(
                self.args.mrra_reconstruction_weight
            ),
            'mrra_gradient_isolated': bool(
                self.args.use_mrra
                and str(getattr(self.args, 'mrra_version', 'v1'))
                == 'v3_gradient_isolated_two_timescale'
            ),
            'mrra_hidden_normalization_affine': bool(
                self.args.use_mrra
                and getattr(
                    getattr(self._core_model().mrra, 'hidden_normalization', None),
                    'elementwise_affine',
                    False,
                )
            ),
            'mrra_learning_rate_multiplier': float(
                getattr(self.args, 'mrra_lr_multiplier', 1.0)
            ),
            'mrra_optimizer_group_count': int(len(model_optim.param_groups)),
            'mrra_initial_optimizer_group_lrs': initial_optimizer_group_lrs,
            'mrra_final_optimizer_group_lrs': {
                group.get('group_name', f'group_{index}'): float(group['lr'])
                for index, group in enumerate(model_optim.param_groups)
            },
            'mrra_diagnostic_steps': int(mrra_diagnostic_steps),
            'mrra_gradient_l2_mean': float(
                mrra_gradient_l2_sum / max(1, mrra_diagnostic_steps)
            ),
            'mrra_gradient_l2_max': float(mrra_gradient_l2_max),
            'mrra_parameter_abs_max': float(mrra_parameter_abs_max),
            'radr_enabled': bool(self.args.use_radr),
            'radr_steps': int(radr_steps),
            'radr_samples': int(radr_samples),
        }
        if self.args.use_mipp:
            denominator = max(1, mipp_steps)
            self.run_evidence.update({
                f'mipp_{key}_mean': float(value / denominator)
                for key, value in mipp_metric_sums.items()
            })
            if mipp_warm_diagnostics:
                self.run_evidence.update({
                    'mipp_warm_pair_cosine_mean': float(np.mean([
                        item['warm_pair_cosine'] for item in mipp_warm_diagnostics
                    ])),
                    'mipp_warm_write_acceptance_rate_mean': float(np.mean([
                        item['warm_write_acceptance_rate'] for item in mipp_warm_diagnostics
                    ])),
                    'mipp_warm_candidates_used_mean': float(np.mean([
                        item['warm_candidates_used'] for item in mipp_warm_diagnostics
                    ])),
                })
                optional_warm_metrics = (
                    'warm_raw_absolute_acceptance_rate',
                    'warm_write_stability_std',
                    'warm_write_effective_cutoff',
                    'warm_write_candidate_pairwise_cosine',
                    'warm_write_selected_pairwise_cosine',
                    'warm_write_selected_stability',
                    'warm_write_diversity_gain',
                    'warm_write_mask_candidate_pairwise_distance',
                    'warm_write_mask_selected_pairwise_distance',
                    'warm_write_mask_coverage_distance',
                    'warm_write_mask_topk_coverage_distance',
                    'warm_write_mask_coverage_gain_vs_topk',
                    'warm_write_mask_candidate_missing_rate',
                    'warm_write_mask_selected_missing_rate',
                )
                for key in optional_warm_metrics:
                    values = [item[key] for item in mipp_warm_diagnostics if key in item]
                    if values:
                        self.run_evidence[f'mipp_{key}_mean'] = float(np.mean(values))
        if self.args.use_cfr:
            denominator = max(1, cfr_steps)
            self.run_evidence.update({
                f'cfr_{key}_mean': float(value / denominator)
                for key, value in cfr_metric_sums.items()
            })
        if self.args.use_fusr:
            denominator = max(1, fusr_steps)
            self.run_evidence.update({
                f'fusr_{key}_mean': float(value / denominator)
                for key, value in fusr_metric_sums.items()
            })
            self.run_evidence.update({
                'fusr_utility_loss_mean': float(
                    fusr_metric_sums['total_loss'] / denominator
                ),
                'fusr_utility_nll_mean': float(
                    fusr_metric_sums['regression_loss'] / denominator
                ),
                'fusr_gate_mean': float(
                    fusr_metric_sums['gate_mean'] / denominator
                ),
                'fusr_abstention_rate': float(
                    fusr_metric_sums['abstention_rate'] / denominator
                ),
                'fusr_predicted_utility_mean': float(
                    fusr_metric_sums['utility_prediction_mean'] / denominator
                ),
                'fusr_utility_std_mean': float(
                    fusr_metric_sums['utility_uncertainty_mean'] / denominator
                ),
                'fusr_sign_accuracy': float(
                    fusr_metric_sums['utility_sign_accuracy'] / denominator
                ),
                'fusr_spectral_uncertainty_mean': float(
                    fusr_metric_sums['spectral_uncertainty'] / denominator
                ),
                'fusr_valid_utility_samples': int(fusr_utility_pairs_generated),
            })
        if self.args.use_mrra:
            denominator = max(1, mrra_steps)
            self.run_evidence.update({
                f'mrra_{key}_mean': float(value / denominator)
                for key, value in mrra_metric_sums.items()
            })
        if self.args.use_radr:
            denominator = max(1, radr_steps)
            self.run_evidence.update({
                f'radr_{key}_mean': float(value / denominator)
                for key, value in radr_metric_sums.items()
            })
        return self.model

    def post_train_validation(self, setting):
        """Evaluate the best training checkpoint on chronological validation only."""
        validation_data, _ = self._get_data(flag='val')
        validation_loader = DataLoader(
            validation_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=True,
        )
        f_dim = -1 if self.args.features == 'MS' else 0
        self.change_mode(0)
        self.model.eval()
        evaluated_batches = 0
        evaluated_samples = 0
        squared_error_sum = 0.0
        absolute_error_sum = 0.0
        element_count = 0
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        validation_started = time.perf_counter()
        with torch.no_grad():
            for batch_index, batch in enumerate(validation_loader):
                if (
                    self.args.max_eval_batches >= 0
                    and batch_index >= self.args.max_eval_batches
                ):
                    break
                (
                    seq_x, seq_x_mask, seq_y, seq_y_mask, seq_true_x, seq_true_y,
                    max_idx, min_idx, max_value, min_value,
                ) = tuple(item.float().to(self.device) for item in batch)
                del seq_y, seq_y_mask, seq_true_x
                prediction = self.model(
                    seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                )[:, -self.args.pred_len:, f_dim:]
                target = seq_true_y[:, -self.args.pred_len:, f_dim:]
                error = prediction - target
                squared_error_sum += float((error ** 2).sum().item())
                absolute_error_sum += float(error.abs().sum().item())
                element_count += int(error.numel())
                evaluated_samples += int(error.shape[0])
                evaluated_batches += 1
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        validation_elapsed_seconds = time.perf_counter() - validation_started
        if evaluated_batches < 1 or element_count < 1:
            raise RuntimeError('post-train validation processed zero complete targets')

        mse = squared_error_sum / element_count
        mae = absolute_error_sum / element_count
        checkpoint = os.path.join(
            self.args.checkpoints, setting, 'checkpoint.pth'
        )
        metrics_payload = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(mse ** 0.5),
            'complete_target_mse': float(mse),
            'complete_target_mae': float(mae),
            'optimizer': self.args.optimizer,
            'parameter_count': int(self.parameter_count),
            'parameter_bytes': int(self.parameter_bytes),
            'buffer_bytes': int(self.buffer_bytes),
            'setting': setting,
            'checkpoint': checkpoint,
            'evaluation_split': 'validation',
            'post_train_evaluation': 'validation',
            'test_split_accessed': False,
            'chronological_loader': True,
            'validation_batches_processed': int(evaluated_batches),
            'validation_samples_processed': int(evaluated_samples),
            'validation_complete_element_count': int(element_count),
            'validation_elapsed_seconds': float(validation_elapsed_seconds),
            'validation_ms_per_batch': float(
                1000.0 * validation_elapsed_seconds / evaluated_batches
            ),
            'peak_gpu_memory_mb': (
                float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
                if self.device.type == 'cuda'
                else None
            ),
            'peak_gpu_reserved_mb': (
                float(torch.cuda.max_memory_reserved(self.device) / (1024 ** 2))
                if self.device.type == 'cuda'
                else None
            ),
            **getattr(self, 'run_evidence', {}),
        }
        if self.args.metrics_path:
            metrics_dir = os.path.dirname(self.args.metrics_path)
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
            temp_path = self.args.metrics_path + '.tmp'
            with open(temp_path, 'w') as metrics_file:
                json.dump(metrics_payload, metrics_file, indent=2, sort_keys=True)
            os.replace(temp_path, self.args.metrics_path)
        print('validation mse:{}, mae:{}'.format(mse, mae))
        return metrics_payload

    @staticmethod
    def _stage22_ard_analytic_targets(
        backbone_prediction,
        raw_residual,
        target,
        target_mask,
        energy_epsilon=1e-8,
        action_epsilon=1e-6,
    ):
        """Return per-sample analytic residual scales and three-way actions."""
        if not (
            backbone_prediction.shape == raw_residual.shape
            == target.shape == target_mask.shape
        ):
            raise ValueError('stage22 ARD analytic target shapes must match')
        observed = target_mask.to(dtype=torch.bool)
        reduce_dims = tuple(range(1, target.ndim))
        observed_float = observed.to(dtype=raw_residual.dtype)
        observed_count = observed_float.sum(dim=reduce_dims)
        error = backbone_prediction - target
        alignment_cross = (
            2.0 * error * raw_residual * observed_float
        ).sum(dim=reduce_dims)
        residual_energy = (
            raw_residual.square() * observed_float
        ).sum(dim=reduce_dims)
        valid = (
            (observed_count > 0)
            & torch.isfinite(alignment_cross)
            & torch.isfinite(residual_energy)
            & (residual_energy > float(energy_epsilon))
        )
        alpha = torch.zeros_like(residual_energy)
        alpha[valid] = torch.clamp(
            -alignment_cross[valid] / (2.0 * residual_energy[valid]),
            min=0.0,
            max=1.0,
        )
        actions = torch.full_like(alpha, -1, dtype=torch.long)
        actions[valid] = 1
        actions[valid & (alpha <= float(action_epsilon))] = 0
        actions[valid & (alpha >= 1.0 - float(action_epsilon))] = 2
        return {
            'alpha': alpha,
            'actions': actions,
            'valid': valid,
            'alignment_cross': alignment_cross,
            'residual_energy': residual_energy,
            'observed_count': observed_count,
        }

    @staticmethod
    def _stage22_ard_structured_features(
        current_features,
        raw_residual,
        backbone_prediction,
        mask_features,
        latent,
    ):
        """Build the preregistered 108-dimensional label-free descriptor."""
        if current_features.ndim != 2 or current_features.size(1) != 8:
            raise ValueError('stage22 ARD current features must be [batch,8]')
        if raw_residual.shape != backbone_prediction.shape:
            raise ValueError('stage22 ARD raw/backbone shapes must match')
        if raw_residual.ndim != 3 or raw_residual.size(1) != 96:
            raise ValueError('stage22 ARD forecast tensors must be [batch,96,variables]')
        if mask_features.ndim != 3 or mask_features.shape[1:] != (192, 4):
            raise ValueError('stage22 ARD mask features must be [batch,192,4]')
        if latent.ndim != 3 or latent.size(1) < 96:
            raise ValueError('stage22 ARD latent must contain the final 96 steps')
        epsilon = 1e-8
        batch_size, _horizon, variables = raw_residual.shape
        raw_blocks = raw_residual.reshape(batch_size, 8, 12, variables)
        backbone_blocks = backbone_prediction.reshape(
            batch_size, 8, 12, variables
        )

        def block_rms(value):
            return value.square().mean(dim=(2, 3)).clamp_min(0.0).sqrt()

        raw_rms = block_rms(raw_blocks)
        backbone_rms = block_rms(backbone_blocks)
        block_dot = (raw_blocks * backbone_blocks).sum(dim=(2, 3))
        block_cosine = block_dot / (
            raw_blocks.square().sum(dim=(2, 3)).sqrt()
            * backbone_blocks.square().sum(dim=(2, 3)).sqrt()
        ).clamp_min(epsilon)
        raw_difference_rms = (
            raw_blocks[:, :, 1:, :] - raw_blocks[:, :, :-1, :]
        ).square().mean(dim=(2, 3)).clamp_min(0.0).sqrt()
        backbone_difference_rms = (
            backbone_blocks[:, :, 1:, :] - backbone_blocks[:, :, :-1, :]
        ).square().mean(dim=(2, 3)).clamp_min(0.0).sqrt()

        history_blocks = mask_features.reshape(batch_size, 8, 24, 4)
        history_summary = history_blocks.mean(dim=2).reshape(batch_size, 32)
        latent_blocks = latent[:, -96:, :].reshape(
            batch_size, 8, 12, latent.size(-1)
        )
        latent_rms = latent_blocks.square().mean(dim=(2, 3)).clamp_min(0.0).sqrt()

        raw_variable_rms = raw_residual.square().mean(dim=1).clamp_min(0.0).sqrt()
        backbone_variable_rms = (
            backbone_prediction.square().mean(dim=1).clamp_min(0.0).sqrt()
        )
        variable_ratio = raw_variable_rms / backbone_variable_rms.clamp_min(epsilon)
        variable_cosine = (raw_residual * backbone_prediction).sum(dim=1) / (
            raw_residual.square().sum(dim=1).sqrt()
            * backbone_prediction.square().sum(dim=1).sqrt()
        ).clamp_min(epsilon)
        quantiles = torch.tensor(
            [0.10, 0.25, 0.50, 0.75, 0.90],
            dtype=raw_residual.dtype,
            device=raw_residual.device,
        )

        def variable_quantiles(value):
            return torch.quantile(value, quantiles, dim=1).transpose(0, 1)

        distribution_summary = torch.cat((
            variable_quantiles(raw_variable_rms),
            variable_quantiles(backbone_variable_rms),
            variable_quantiles(variable_ratio),
            variable_quantiles(variable_cosine),
        ), dim=1)
        structured = torch.cat((
            current_features,
            raw_rms,
            backbone_rms,
            block_cosine,
            raw_difference_rms,
            backbone_difference_rms,
            history_summary,
            latent_rms,
            distribution_summary,
        ), dim=1)
        if structured.shape != (batch_size, 108):
            raise RuntimeError('stage22 ARD structured feature width changed')
        if not bool(torch.isfinite(structured).all().item()):
            raise FloatingPointError('stage22 ARD structured features are non-finite')
        return structured.detach()

    def stage22_ard_information_diagnostic(self, setting):
        """Cache frozen ARD decisions and structured features on validation."""
        validation_data, _ = self._get_data(flag='val')
        validation_loader = DataLoader(
            validation_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=True,
        )
        f_dim = -1 if self.args.features == 'MS' else 0
        self.change_mode(0)
        self.model.eval()
        core = self._core_model()
        if core.mcpr is None or not bool(getattr(core, 'use_ard', False)):
            raise RuntimeError('stage22 ARD diagnostic requires an ARD checkpoint')
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

        arrays = {
            key: [] for key in (
                'sample_index', 'global8', 'structured108', 'action_logits',
                'action_probabilities', 'partial_alpha', 'student_gate',
                'complete_teacher_alpha', 'complete_teacher_actions',
                'complete_teacher_valid', 'observed_teacher_alpha',
                'observed_teacher_actions', 'observed_teacher_valid',
                'complete_count', 'observed_count', 'backbone_sse',
                'backbone_sae', 'raw_sse', 'raw_sae', 'ard_sse', 'ard_sae',
                'alignment_cross', 'residual_energy',
            )
        }
        evaluated_batches = 0
        evaluated_samples = 0
        observed_target_matches_complete = True
        with torch.no_grad():
            for batch_index, batch in enumerate(validation_loader):
                if (
                    self.args.max_eval_batches >= 0
                    and batch_index >= self.args.max_eval_batches
                ):
                    break
                (
                    seq_x, seq_x_mask, seq_y, seq_y_mask, seq_true_x,
                    seq_true_y, max_idx, min_idx, max_value, min_value,
                ) = tuple(item.float().to(self.device) for item in batch)
                del seq_true_x
                prediction = self.model(
                    seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                )[:, -self.args.pred_len:, f_dim:]
                outputs = core.get_last_mcpr_outputs()
                required = {
                    'residual', 'raw_residual', 'latent', 'mask_features',
                    'ard_features', 'ard_action_logits',
                    'ard_action_probabilities', 'ard_partial_alpha', 'ard_gate',
                }
                if outputs is None or not required.issubset(outputs):
                    raise RuntimeError('stage22 ARD diagnostic outputs are incomplete')
                raw_residual = outputs['raw_residual'][
                    :, -self.args.pred_len:, f_dim:
                ]
                applied_residual = outputs['residual'][
                    :, -self.args.pred_len:, f_dim:
                ]
                backbone_prediction = prediction - applied_residual
                complete_target = seq_true_y[:, -self.args.pred_len:, f_dim:]
                observed_target = seq_y[:, -self.args.pred_len:, f_dim:]
                observed_mask = seq_y_mask[
                    :, -self.args.pred_len:, f_dim:
                ] > 0.5
                if bool(observed_mask.any()):
                    observed_target_matches_complete = (
                        observed_target_matches_complete
                        and bool(torch.allclose(
                            observed_target[observed_mask],
                            complete_target[observed_mask],
                            rtol=1e-6,
                            atol=1e-6,
                        ))
                    )
                complete_teacher = self._stage22_ard_analytic_targets(
                    backbone_prediction,
                    raw_residual,
                    complete_target,
                    torch.ones_like(complete_target, dtype=torch.bool),
                    energy_epsilon=float(self.args.ard_energy_epsilon),
                    action_epsilon=float(self.args.ard_action_epsilon),
                )
                observed_teacher = self._stage22_ard_analytic_targets(
                    backbone_prediction,
                    raw_residual,
                    complete_target,
                    observed_mask,
                    energy_epsilon=float(self.args.ard_energy_epsilon),
                    action_epsilon=float(self.args.ard_action_epsilon),
                )
                structured = self._stage22_ard_structured_features(
                    outputs['ard_features'],
                    raw_residual,
                    backbone_prediction,
                    outputs['mask_features'],
                    outputs['latent'],
                )
                raw_prediction = backbone_prediction + raw_residual
                reduce_dims = (1, 2)
                complete_count = torch.full(
                    (prediction.size(0),),
                    prediction[0].numel(),
                    dtype=torch.int64,
                    device=prediction.device,
                )
                indices = torch.arange(
                    evaluated_samples,
                    evaluated_samples + prediction.size(0),
                    dtype=torch.int64,
                    device=prediction.device,
                )
                batch_values = {
                    'sample_index': indices,
                    'global8': outputs['ard_features'],
                    'structured108': structured,
                    'action_logits': outputs['ard_action_logits'],
                    'action_probabilities': outputs['ard_action_probabilities'],
                    'partial_alpha': outputs['ard_partial_alpha'],
                    'student_gate': outputs['ard_gate'].reshape(-1),
                    'complete_teacher_alpha': complete_teacher['alpha'],
                    'complete_teacher_actions': complete_teacher['actions'],
                    'complete_teacher_valid': complete_teacher['valid'],
                    'observed_teacher_alpha': observed_teacher['alpha'],
                    'observed_teacher_actions': observed_teacher['actions'],
                    'observed_teacher_valid': observed_teacher['valid'],
                    'complete_count': complete_count,
                    'observed_count': observed_teacher['observed_count'],
                    'backbone_sse': (
                        backbone_prediction - complete_target
                    ).double().square().sum(dim=reduce_dims),
                    'backbone_sae': (
                        backbone_prediction - complete_target
                    ).double().abs().sum(dim=reduce_dims),
                    'raw_sse': (
                        raw_prediction - complete_target
                    ).double().square().sum(dim=reduce_dims),
                    'raw_sae': (
                        raw_prediction - complete_target
                    ).double().abs().sum(dim=reduce_dims),
                    'ard_sse': (
                        prediction - complete_target
                    ).double().square().sum(dim=reduce_dims),
                    'ard_sae': (
                        prediction - complete_target
                    ).double().abs().sum(dim=reduce_dims),
                    'alignment_cross': complete_teacher['alignment_cross'],
                    'residual_energy': complete_teacher['residual_energy'],
                }
                for key, value in batch_values.items():
                    arrays[key].append(value.detach().cpu().numpy())
                evaluated_samples += int(prediction.size(0))
                evaluated_batches += 1

        if evaluated_batches < 1 or evaluated_samples < 1:
            raise RuntimeError('stage22 ARD diagnostic processed zero samples')
        cached = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
        total_count = float(cached['complete_count'].sum())
        mse = float(cached['ard_sse'].sum() / total_count)
        mae = float(cached['ard_sae'].sum() / total_count)
        complete_valid = cached['complete_teacher_valid'].astype(bool)
        observed_valid = cached['observed_teacher_valid'].astype(bool)
        shared_valid = complete_valid & observed_valid
        teacher_action_agreement = float(
            np.mean(
                cached['complete_teacher_actions'][shared_valid]
                == cached['observed_teacher_actions'][shared_valid]
            )
        ) if bool(shared_valid.any()) else None
        teacher_alpha_mae = float(np.mean(np.abs(
            cached['complete_teacher_alpha'][shared_valid]
            - cached['observed_teacher_alpha'][shared_valid]
        ))) if bool(shared_valid.any()) else None

        samples_path = self.args.stage22_ard_diagnostic_samples_path
        samples_dir = os.path.dirname(samples_path)
        if samples_dir:
            os.makedirs(samples_dir, exist_ok=True)
        temporary_samples = samples_path + '.tmp'
        with open(temporary_samples, 'wb') as handle:
            np.savez_compressed(handle, **cached)
        os.replace(temporary_samples, samples_path)
        payload = {
            'setting': setting,
            'checkpoint': self.args.stage22_ard_diagnostic_checkpoint,
            'source_split': 'val',
            'evaluation_split': 'validation',
            'test_split_accessed': False,
            'uses_test_split': False,
            'chronological_loader': True,
            'updates_model_or_memory': False,
            'sample_count': int(evaluated_samples),
            'validation_samples_processed': int(evaluated_samples),
            'validation_batches_processed': int(evaluated_batches),
            'complete_element_count': int(total_count),
            'mse': mse,
            'mae': mae,
            'global_feature_dim': 8,
            'structured_feature_dim': 108,
            'global8_equals_structured_prefix': bool(np.array_equal(
                cached['global8'], cached['structured108'][:, :8]
            )),
            'all_cached_arrays_finite': all(
                np.isfinite(value).all()
                for key, value in cached.items()
                if key not in {
                    'complete_teacher_actions', 'observed_teacher_actions',
                    'complete_teacher_valid', 'observed_teacher_valid',
                    'sample_index',
                }
            ),
            'observed_target_matches_complete': observed_target_matches_complete,
            'teacher_shared_valid_samples': int(shared_valid.sum()),
            'observed_complete_teacher_action_agreement': teacher_action_agreement,
            'observed_complete_teacher_alpha_mae': teacher_alpha_mae,
            'student_gate_mean': float(cached['student_gate'].mean()),
            'student_gate_std': float(cached['student_gate'].std(ddof=0)),
            'student_gate_min': float(cached['student_gate'].min()),
            'student_gate_max': float(cached['student_gate'].max()),
            'samples_path': samples_path,
            'parameter_count': int(self.parameter_count),
            'parameter_bytes': int(self.parameter_bytes),
            'buffer_bytes': int(self.buffer_bytes),
            'peak_gpu_memory_mb': (
                float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
                if self.device.type == 'cuda' else None
            ),
            'peak_gpu_reserved_mb': (
                float(torch.cuda.max_memory_reserved(self.device) / (1024 ** 2))
                if self.device.type == 'cuda' else None
            ),
        }
        metrics_path = self.args.stage22_ard_diagnostic_metrics_path
        metrics_dir = os.path.dirname(metrics_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        temporary_metrics = metrics_path + '.tmp'
        with open(temporary_metrics, 'w') as metrics_file:
            json.dump(payload, metrics_file, indent=2, sort_keys=True)
        os.replace(temporary_metrics, metrics_path)
        print(
            'stage22 ARD diagnostic mse:{}, mae:{}, samples:{}'.format(
                mse, mae, evaluated_samples
            )
        )
        return payload

    @staticmethod
    def _stage22_27_token_diagnostics(
        backbone_prediction,
        teacher_prediction,
        student_prediction,
        complete_target,
        observed_target,
        observed_mask,
    ):
        """Return compact time-unit evidence for frozen BF-CMVD attribution."""
        tensors = (
            backbone_prediction, teacher_prediction, student_prediction,
            complete_target, observed_target, observed_mask,
        )
        if any(value.shape != tensors[0].shape for value in tensors[1:]):
            raise ValueError('stage22.27 prediction, target and mask shapes must match')
        if backbone_prediction.ndim != 3:
            raise ValueError('stage22.27 tensors must be [batch,horizon,variables]')
        observed = observed_mask > 0.5

        def complete_losses(prediction):
            error = prediction - complete_target
            return error.square().mean(dim=-1), error.abs().mean(dim=-1)

        def observed_losses(prediction):
            error = prediction - observed_target
            counts = observed.sum(dim=-1)
            denominator = counts.clamp_min(1).to(error.dtype)
            mse = (error.square() * observed).sum(dim=-1) / denominator
            mae = (error.abs() * observed).sum(dim=-1) / denominator
            return mse, mae, counts

        backbone_mse, backbone_mae = complete_losses(backbone_prediction)
        teacher_mse, teacher_mae = complete_losses(teacher_prediction)
        student_mse, student_mae = complete_losses(student_prediction)
        observed_backbone_mse, observed_backbone_mae, observed_count = (
            observed_losses(backbone_prediction)
        )
        observed_teacher_mse, observed_teacher_mae, _ = observed_losses(
            teacher_prediction
        )
        complete_teacher_beneficial = (
            (teacher_mse < backbone_mse) & (teacher_mae < backbone_mae)
        )
        observed_eligible = observed_count > 0
        observed_teacher_beneficial = (
            observed_eligible
            & (observed_teacher_mse < observed_backbone_mse)
            & (observed_teacher_mae < observed_backbone_mae)
        )
        complete_student_beneficial = (
            (student_mse < backbone_mse) & (student_mae < backbone_mae)
        )

        teacher_correction = teacher_prediction - backbone_prediction
        student_correction = student_prediction - backbone_prediction
        source_error = backbone_prediction - complete_target
        teacher_nonzero = teacher_correction.abs() > 1e-8
        return {
            'observed_token': observed_eligible,
            'complete_teacher_beneficial': complete_teacher_beneficial,
            'observed_teacher_beneficial': observed_teacher_beneficial,
            'complete_student_beneficial': complete_student_beneficial,
            'backbone_mse': backbone_mse,
            'teacher_mse': teacher_mse,
            'student_mse': student_mse,
            'backbone_mae': backbone_mae,
            'teacher_mae': teacher_mae,
            'student_mae': student_mae,
            'observed_backbone_mse': observed_backbone_mse,
            'observed_teacher_mse': observed_teacher_mse,
            'observed_backbone_mae': observed_backbone_mae,
            'observed_teacher_mae': observed_teacher_mae,
            'correction_dot': (student_correction * teacher_correction).sum(dim=-1),
            'student_correction_energy': student_correction.square().sum(dim=-1),
            'teacher_correction_energy': teacher_correction.square().sum(dim=-1),
            'correction_mismatch_energy': (
                student_correction - teacher_correction
            ).square().sum(dim=-1),
            'source_error_correction_dot': (
                source_error * student_correction
            ).sum(dim=-1),
            'correction_sign_agreement_count': (
                (student_correction * teacher_correction) > 0.0
            ).sum(dim=-1),
            'correction_sign_eligible_count': teacher_nonzero.sum(dim=-1),
        }

    def stage22_27_bf_cmvd_diagnostic(self, setting):
        """Attribute frozen BF-CMVD-v2 gaps on chronological validation only."""
        validation_data, _ = self._get_data(flag='val')
        validation_loader = DataLoader(
            validation_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=True,
        )
        f_dim = -1 if self.args.features == 'MS' else 0
        core = self._core_model()
        if core.mcpr is None or not bool(getattr(core, 'use_mcpr', False)):
            raise RuntimeError('stage22.27 requires an enabled MCPR student')
        if bool(getattr(core, 'use_ard', False)) or bool(
            getattr(core, 'use_ttrr', False)
        ):
            raise RuntimeError('stage22.27 requires the ungated BF-CMVD-v2 residual')
        core.set_prototype_enabled(True)
        self.change_mode(0)
        self.model.eval()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

        token_keys = (
            'observed_token', 'complete_teacher_beneficial',
            'observed_teacher_beneficial', 'complete_student_beneficial',
            'backbone_mse', 'teacher_mse', 'student_mse',
            'backbone_mae', 'teacher_mae', 'student_mae',
            'observed_backbone_mse', 'observed_teacher_mse',
            'observed_backbone_mae', 'observed_teacher_mae',
            'correction_dot', 'student_correction_energy',
            'teacher_correction_energy', 'correction_mismatch_energy',
            'correction_sign_agreement_count',
            'correction_sign_eligible_count',
        )
        arrays = {key: [] for key in token_keys}
        arrays.update({
            'sample_index': [],
            'history_mask_packed': [],
            'l5_mask_packed': [],
            'l20_mask_packed': [],
        })
        variable_totals = {}
        history_row_shared = []
        target_row_shared = []
        evaluated_batches = 0
        residual_identity_max_abs = 0.0
        started = time.monotonic()

        with torch.no_grad():
            for batch_index, batch in enumerate(validation_loader):
                if (
                    self.args.max_eval_batches >= 0
                    and batch_index >= self.args.max_eval_batches
                ):
                    break
                (
                    seq_x, seq_x_mask, seq_y, seq_y_mask, seq_true_x, seq_true_y,
                    max_idx, min_idx, max_value, min_value,
                ) = tuple(item.float().to(self.device) for item in batch)
                batch_size = int(seq_x.shape[0])
                sample_index = np.arange(
                    batch_index * int(self.args.batch_size),
                    batch_index * int(self.args.batch_size) + batch_size,
                    dtype=np.int64,
                )

                student_prediction = self.model(
                    seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                )[:, -self.args.pred_len:, f_dim:]
                mcpr_outputs = core.get_last_mcpr_outputs()
                if mcpr_outputs is None or 'residual' not in mcpr_outputs:
                    raise RuntimeError('stage22.27 student forward returned no residual')
                student_residual = mcpr_outputs['residual'][
                    :, -self.args.pred_len:, f_dim:
                ]
                if student_residual.shape != student_prediction.shape:
                    raise RuntimeError('stage22.27 student residual shape mismatch')
                backbone_prediction = student_prediction - student_residual
                residual_identity_max_abs = max(
                    residual_identity_max_abs,
                    float((
                        student_prediction - backbone_prediction - student_residual
                    ).abs().max().item()),
                )

                previous_mcpr_state = bool(core.use_mcpr)
                nested_predictions = {}
                nested_masks = {}
                core.use_mcpr = False
                try:
                    for block_length in (5, 20):
                        nested_mask = self._stage18_equal_count_observation_mask(
                            seq_x_mask,
                            sample_index,
                            float(self.args.stage22_27_target_rate),
                            int(block_length),
                            int(self.args.stage22_27_nested_seed),
                        )
                        nested_x = seq_true_x * nested_mask
                        nested_prediction = self.model(
                            nested_x,
                            nested_mask,
                            *self._temporal_extrema(nested_x),
                        )[:, -self.args.pred_len:, f_dim:]
                        nested_masks[block_length] = nested_mask
                        nested_predictions[block_length] = nested_prediction
                finally:
                    core.use_mcpr = previous_mcpr_state
                teacher_prediction = (
                    float(self.args.stage22_27_original_weight)
                    * backbone_prediction
                    + float(self.args.stage22_27_l5_weight)
                    * nested_predictions[5]
                    + float(self.args.stage22_27_l20_weight)
                    * nested_predictions[20]
                )

                complete_target = seq_true_y[:, -self.args.pred_len:, f_dim:]
                observed_target = seq_y[:, -self.args.pred_len:, f_dim:]
                observed_mask = seq_y_mask[:, -self.args.pred_len:, f_dim:]
                diagnostics = self._stage22_27_token_diagnostics(
                    backbone_prediction,
                    teacher_prediction,
                    student_prediction,
                    complete_target,
                    observed_target,
                    observed_mask,
                )
                for key in token_keys:
                    value = diagnostics[key]
                    if value.dtype == torch.bool:
                        cached = value.to(torch.uint8).cpu().numpy()
                    elif key.endswith('_count'):
                        cached = value.to(torch.int64).cpu().numpy()
                    else:
                        cached = value.float().cpu().numpy()
                    arrays[key].append(cached)

                original_observed = seq_x_mask[:, :, 0] > 0.5
                arrays['sample_index'].append(sample_index)
                arrays['history_mask_packed'].append(np.packbits(
                    original_observed.to(torch.uint8).cpu().numpy(), axis=1
                ))
                for block_length in (5, 20):
                    nested_observed = nested_masks[block_length][:, :, 0] > 0.5
                    arrays[f'l{block_length}_mask_packed'].append(np.packbits(
                        nested_observed.to(torch.uint8).cpu().numpy(), axis=1
                    ))
                history_row_shared.append((
                    (seq_x_mask > 0.5) == (seq_x_mask[:, :, :1] > 0.5)
                ).reshape(batch_size, -1).all(dim=1).cpu().numpy())
                target_row_shared.append((
                    (observed_mask > 0.5) == (observed_mask[:, :, :1] > 0.5)
                ).reshape(batch_size, -1).all(dim=1).cpu().numpy())

                prediction_map = {
                    'backbone': backbone_prediction,
                    'teacher': teacher_prediction,
                    'student': student_prediction,
                }
                for label, prediction in prediction_map.items():
                    error = prediction - complete_target
                    for suffix, value in (
                        ('sse', error.square().sum(dim=(0, 1))),
                        ('sae', error.abs().sum(dim=(0, 1))),
                    ):
                        key = f'variable_{label}_{suffix}'
                        value_np = value.double().cpu().numpy()
                        variable_totals.setdefault(
                            key, np.zeros_like(value_np, dtype=np.float64)
                        )
                        variable_totals[key] += value_np
                teacher_correction = teacher_prediction - backbone_prediction
                student_correction = student_prediction - backbone_prediction
                for key, value in (
                    ('variable_correction_dot', (
                        teacher_correction * student_correction
                    ).sum(dim=(0, 1))),
                    ('variable_teacher_correction_energy',
                     teacher_correction.square().sum(dim=(0, 1))),
                    ('variable_student_correction_energy',
                     student_correction.square().sum(dim=(0, 1))),
                    ('variable_correction_mismatch_energy', (
                        teacher_correction - student_correction
                    ).square().sum(dim=(0, 1))),
                ):
                    value_np = value.double().cpu().numpy()
                    variable_totals.setdefault(
                        key, np.zeros_like(value_np, dtype=np.float64)
                    )
                    variable_totals[key] += value_np
                evaluated_batches += 1

        if not arrays['sample_index']:
            raise RuntimeError('stage22.27 diagnostic processed zero batches')
        merged = {}
        boolean_keys = {
            'observed_token', 'complete_teacher_beneficial',
            'observed_teacher_beneficial', 'complete_student_beneficial',
        }
        packed_keys = {
            'history_mask_packed', 'l5_mask_packed', 'l20_mask_packed',
        }
        integer_keys = {
            'sample_index', 'correction_sign_agreement_count',
            'correction_sign_eligible_count',
        }
        for key, chunks in arrays.items():
            if key in boolean_keys or key in packed_keys:
                dtype = np.uint8
            elif key in integer_keys:
                dtype = np.int64
            else:
                dtype = np.float32
            merged[key] = np.concatenate(chunks).astype(dtype, copy=False)
        history_row_shared = np.concatenate(history_row_shared).astype(bool)
        target_row_shared = np.concatenate(target_row_shared).astype(bool)
        if not bool(history_row_shared.all()) or not bool(target_row_shared.all()):
            raise RuntimeError('stage22.27 Traffic masks must be row-shared')

        observed = merged['observed_token'].astype(bool)
        complete_beneficial = merged['complete_teacher_beneficial'].astype(bool)
        observed_beneficial = merged['observed_teacher_beneficial'].astype(bool)
        student_beneficial = merged['complete_student_beneficial'].astype(bool)
        token_count = int(merged['backbone_mse'].size)
        variable_count = int(variable_totals['variable_backbone_sse'].size)
        element_count = token_count * variable_count
        elapsed = time.monotonic() - started
        payload = {
            'setting': setting,
            'checkpoint': self.args.stage22_27_checkpoint,
            'source_split': 'val',
            'evaluation_split': 'validation',
            'test_split_accessed': False,
            'uses_test_split': False,
            'chronological_loader': True,
            'updates_model_or_memory': False,
            'compact_cache_excludes_full_predictions': True,
            'sample_count': int(merged['sample_index'].size),
            'validation_samples_processed': int(merged['sample_index'].size),
            'validation_batches_processed': int(evaluated_batches),
            'prediction_horizon': int(self.args.pred_len),
            'variable_count': variable_count,
            'complete_token_count': token_count,
            'complete_element_count': element_count,
            'observed_token_count': int(observed.sum()),
            'observed_token_fraction': float(observed.mean()),
            'complete_teacher_beneficial_count': int(complete_beneficial.sum()),
            'complete_teacher_beneficial_fraction': float(
                complete_beneficial.mean()
            ),
            'observed_teacher_beneficial_count': int(observed_beneficial.sum()),
            'observed_teacher_beneficial_fraction_of_observed': float(
                observed_beneficial.sum() / max(1, observed.sum())
            ),
            'complete_student_beneficial_count': int(student_beneficial.sum()),
            'complete_student_beneficial_fraction': float(student_beneficial.mean()),
            'complete_target_mse_backbone': float(merged['backbone_mse'].mean()),
            'complete_target_mae_backbone': float(merged['backbone_mae'].mean()),
            'complete_target_mse_teacher': float(merged['teacher_mse'].mean()),
            'complete_target_mae_teacher': float(merged['teacher_mae'].mean()),
            'complete_target_mse_student': float(merged['student_mse'].mean()),
            'complete_target_mae_student': float(merged['student_mae'].mean()),
            'mse': float(merged['student_mse'].mean()),
            'mae': float(merged['student_mae'].mean()),
            'target_rate': float(self.args.stage22_27_target_rate),
            'block_lengths': [5, 20],
            'nested_seed': int(self.args.stage22_27_nested_seed),
            'teacher_weights': {
                'original': float(self.args.stage22_27_original_weight),
                'l5': float(self.args.stage22_27_l5_weight),
                'l20': float(self.args.stage22_27_l20_weight),
            },
            'history_row_shared_all_samples': bool(history_row_shared.all()),
            'target_row_shared_all_samples': bool(target_row_shared.all()),
            'student_backbone_residual_identity_max_abs': float(
                residual_identity_max_abs
            ),
            'parameter_count': int(self.parameter_count),
            'parameter_bytes': int(self.parameter_bytes),
            'buffer_bytes': int(self.buffer_bytes),
            'evaluation_elapsed_seconds': float(elapsed),
            'peak_gpu_memory_mb': (
                float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
                if self.device.type == 'cuda' else None
            ),
            'peak_gpu_reserved_mb': (
                float(torch.cuda.max_memory_reserved(self.device) / (1024 ** 2))
                if self.device.type == 'cuda' else None
            ),
        }
        samples_path = self.args.stage22_27_samples_path
        os.makedirs(os.path.dirname(samples_path), exist_ok=True)
        temporary_samples = samples_path + '.tmp'
        with open(temporary_samples, 'wb') as handle:
            np.savez_compressed(handle, **merged, **variable_totals)
        os.replace(temporary_samples, samples_path)
        metrics_path = self.args.stage22_27_metrics_path
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        temporary_metrics = metrics_path + '.tmp'
        with open(temporary_metrics, 'w') as metrics_file:
            json.dump(payload, metrics_file, indent=2, sort_keys=True)
        os.replace(temporary_metrics, metrics_path)
        print('STAGE22_27_BF_CMVD ' + json.dumps(payload, sort_keys=True))
        return payload

    def stage22_31_pba_cmvd_diagnostic(self, setting):
        """Diagnose a frozen PBA student against its frozen PG95 teacher source."""
        validation_data, _ = self._get_data(flag='val')
        validation_loader = DataLoader(
            validation_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=True,
        )
        f_dim = -1 if self.args.features == 'MS' else 0
        pba_core = self._core_model()
        if bool(getattr(pba_core, 'use_mcpr', False)) or pba_core.mcpr is not None:
            raise RuntimeError('stage22.31 PBA student must not contain MCPR')
        if not bool(getattr(self.args, 'use_pba_cmvd', False)):
            raise RuntimeError('stage22.31 requires the frozen PBA identity switch')

        source_checkpoint = os.path.abspath(
            self.args.stage22_31_backbone_checkpoint
        )
        if not os.path.isfile(source_checkpoint):
            raise FileNotFoundError(
                'stage22.31 PG95 source checkpoint is missing: {}'.format(
                    source_checkpoint
                )
            )
        source_payload = torch.load(source_checkpoint, map_location=self.device)
        required_source_keys = {'model', 'memory', 'repre', 'num_dic'}
        if not required_source_keys.issubset(source_payload):
            raise RuntimeError('stage22.31 PG95 source checkpoint is incomplete')
        source_model = copy.deepcopy(self.model)
        source_model.load_state_dict(source_payload['model'], strict=False)
        source_core = (
            source_model.module
            if isinstance(source_model, nn.DataParallel) else source_model
        )
        source_core.mem_net.memory.memory = source_payload['memory']
        source_core.mem_net.memory.repre = source_payload['repre']
        source_core.mem_net.memory.num_dic = source_payload['num_dic']
        for parameter in source_model.parameters():
            parameter.requires_grad = False
        source_core.set_prototype_enabled(True)
        pba_core.set_prototype_enabled(True)
        source_core.mem_net.is_training = 0
        source_core.is_training = 0
        self.change_mode(0)
        source_model.eval()
        self.model.eval()

        pba_hash_before = self._pba_state_hash(self.model, selected=None)
        pba_memory_hash_before = self._pba_memory_hash(self.model)
        source_hash_before = self._pba_state_hash(source_model, selected=None)
        source_memory_hash_before = self._pba_memory_hash(source_model)
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

        token_keys = (
            'observed_token', 'complete_teacher_beneficial',
            'observed_teacher_beneficial', 'complete_student_beneficial',
            'backbone_mse', 'teacher_mse', 'student_mse',
            'backbone_mae', 'teacher_mae', 'student_mae',
            'observed_backbone_mse', 'observed_teacher_mse',
            'observed_backbone_mae', 'observed_teacher_mae',
            'correction_dot', 'student_correction_energy',
            'teacher_correction_energy', 'correction_mismatch_energy',
            'source_error_correction_dot',
            'correction_sign_agreement_count',
            'correction_sign_eligible_count',
        )
        arrays = {key: [] for key in token_keys}
        arrays.update({
            'sample_index': [],
            'history_mask_packed': [],
            'l5_mask_packed': [],
            'l20_mask_packed': [],
        })
        variable_totals = {}
        history_row_shared = []
        target_row_shared = []
        evaluated_batches = 0
        evaluated_samples = 0
        started = time.monotonic()

        with torch.no_grad():
            for batch_index, batch in enumerate(validation_loader):
                if (
                    self.args.max_eval_batches >= 0
                    and batch_index >= self.args.max_eval_batches
                ):
                    break
                (
                    seq_x, seq_x_mask, seq_y, seq_y_mask, seq_true_x,
                    seq_true_y, max_idx, min_idx, max_value, min_value,
                ) = tuple(item.float().to(self.device) for item in batch)
                batch_size = int(seq_x.shape[0])
                sample_index = np.arange(
                    batch_index * int(self.args.batch_size),
                    batch_index * int(self.args.batch_size) + batch_size,
                    dtype=np.int64,
                )

                pba_prediction = self.model(
                    seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                )[:, -self.args.pred_len:, f_dim:]
                backbone_prediction = source_model(
                    seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                )[:, -self.args.pred_len:, f_dim:]
                nested_predictions = {}
                nested_masks = {}
                for block_length in (5, 20):
                    nested_mask = self._stage18_equal_count_observation_mask(
                        seq_x_mask,
                        sample_index,
                        float(self.args.stage22_31_target_rate),
                        int(block_length),
                        int(self.args.stage22_31_nested_seed),
                    )
                    nested_x = seq_true_x * nested_mask
                    nested_prediction = source_model(
                        nested_x,
                        nested_mask,
                        *self._temporal_extrema(nested_x),
                    )[:, -self.args.pred_len:, f_dim:]
                    nested_masks[block_length] = nested_mask
                    nested_predictions[block_length] = nested_prediction
                teacher_prediction = (
                    float(self.args.stage22_31_original_weight)
                    * backbone_prediction
                    + float(self.args.stage22_31_l5_weight)
                    * nested_predictions[5]
                    + float(self.args.stage22_31_l20_weight)
                    * nested_predictions[20]
                )

                complete_target = seq_true_y[:, -self.args.pred_len:, f_dim:]
                observed_target = seq_y[:, -self.args.pred_len:, f_dim:]
                observed_mask = seq_y_mask[:, -self.args.pred_len:, f_dim:]
                diagnostics = self._stage22_27_token_diagnostics(
                    backbone_prediction,
                    teacher_prediction,
                    pba_prediction,
                    complete_target,
                    observed_target,
                    observed_mask,
                )
                for key in token_keys:
                    value = diagnostics[key]
                    if value.dtype == torch.bool:
                        cached = value.to(torch.uint8).cpu().numpy()
                    elif key.endswith('_count'):
                        cached = value.to(torch.int64).cpu().numpy()
                    else:
                        cached = value.float().cpu().numpy()
                    arrays[key].append(cached)

                original_observed = seq_x_mask[:, :, 0] > 0.5
                arrays['sample_index'].append(sample_index)
                arrays['history_mask_packed'].append(np.packbits(
                    original_observed.to(torch.uint8).cpu().numpy(), axis=1
                ))
                for block_length in (5, 20):
                    nested_observed = nested_masks[block_length][:, :, 0] > 0.5
                    arrays['l{}_mask_packed'.format(block_length)].append(
                        np.packbits(
                            nested_observed.to(torch.uint8).cpu().numpy(), axis=1
                        )
                    )
                history_row_shared.append((
                    (seq_x_mask > 0.5) == (seq_x_mask[:, :, :1] > 0.5)
                ).reshape(batch_size, -1).all(dim=1).cpu().numpy())
                target_row_shared.append((
                    (observed_mask > 0.5) == (observed_mask[:, :, :1] > 0.5)
                ).reshape(batch_size, -1).all(dim=1).cpu().numpy())

                prediction_map = {
                    'backbone': backbone_prediction,
                    'teacher': teacher_prediction,
                    'student': pba_prediction,
                }
                for label, prediction in prediction_map.items():
                    error = prediction - complete_target
                    for suffix, value in (
                        ('sse', error.square().sum(dim=(0, 1))),
                        ('sae', error.abs().sum(dim=(0, 1))),
                    ):
                        key = 'variable_{}_{}'.format(label, suffix)
                        value_np = value.double().cpu().numpy()
                        variable_totals.setdefault(
                            key, np.zeros_like(value_np, dtype=np.float64)
                        )
                        variable_totals[key] += value_np
                teacher_correction = teacher_prediction - backbone_prediction
                pba_correction = pba_prediction - backbone_prediction
                for key, value in (
                    ('variable_correction_dot', (
                        teacher_correction * pba_correction
                    ).sum(dim=(0, 1))),
                    ('variable_teacher_correction_energy',
                     teacher_correction.square().sum(dim=(0, 1))),
                    ('variable_student_correction_energy',
                     pba_correction.square().sum(dim=(0, 1))),
                    ('variable_correction_mismatch_energy', (
                        teacher_correction - pba_correction
                    ).square().sum(dim=(0, 1))),
                ):
                    value_np = value.double().cpu().numpy()
                    variable_totals.setdefault(
                        key, np.zeros_like(value_np, dtype=np.float64)
                    )
                    variable_totals[key] += value_np
                evaluated_batches += 1
                evaluated_samples += batch_size

        if not arrays['sample_index']:
            raise RuntimeError('stage22.31 diagnostic processed zero batches')
        merged = {}
        boolean_keys = {
            'observed_token', 'complete_teacher_beneficial',
            'observed_teacher_beneficial', 'complete_student_beneficial',
        }
        packed_keys = {
            'history_mask_packed', 'l5_mask_packed', 'l20_mask_packed',
        }
        integer_keys = {
            'sample_index', 'correction_sign_agreement_count',
            'correction_sign_eligible_count',
        }
        for key, chunks in arrays.items():
            if key in boolean_keys or key in packed_keys:
                dtype = np.uint8
            elif key in integer_keys:
                dtype = np.int64
            else:
                dtype = np.float32
            merged[key] = np.concatenate(chunks).astype(dtype, copy=False)
        history_row_shared = np.concatenate(history_row_shared).astype(bool)
        target_row_shared = np.concatenate(target_row_shared).astype(bool)
        if not bool(history_row_shared.all()) or not bool(target_row_shared.all()):
            raise RuntimeError('stage22.31 Traffic masks must be row-shared')

        pba_hash_after = self._pba_state_hash(self.model, selected=None)
        pba_memory_hash_after = self._pba_memory_hash(self.model)
        source_hash_after = self._pba_state_hash(source_model, selected=None)
        source_memory_hash_after = self._pba_memory_hash(source_model)
        observed = merged['observed_token'].astype(bool)
        complete_beneficial = merged['complete_teacher_beneficial'].astype(bool)
        observed_beneficial = merged['observed_teacher_beneficial'].astype(bool)
        student_beneficial = merged['complete_student_beneficial'].astype(bool)
        token_count = int(merged['backbone_mse'].size)
        variable_count = int(variable_totals['variable_backbone_sse'].size)
        elapsed = time.monotonic() - started
        payload = {
            'setting': setting,
            'checkpoint': self.args.stage22_31_checkpoint,
            'backbone_checkpoint': source_checkpoint,
            'source_split': 'val',
            'evaluation_split': 'validation',
            'test_split_accessed': False,
            'uses_test_split': False,
            'chronological_loader': True,
            'updates_model_or_memory': False,
            'compact_cache_excludes_full_predictions': True,
            'sample_count': int(evaluated_samples),
            'validation_samples_processed': int(evaluated_samples),
            'validation_batches_processed': int(evaluated_batches),
            'prediction_horizon': int(self.args.pred_len),
            'variable_count': variable_count,
            'complete_token_count': token_count,
            'complete_element_count': int(token_count * variable_count),
            'observed_token_count': int(observed.sum()),
            'observed_token_fraction': float(observed.mean()),
            'complete_teacher_beneficial_count': int(complete_beneficial.sum()),
            'complete_teacher_beneficial_fraction': float(
                complete_beneficial.mean()
            ),
            'observed_teacher_beneficial_count': int(observed_beneficial.sum()),
            'observed_teacher_beneficial_fraction_of_observed': float(
                observed_beneficial.sum() / max(1, observed.sum())
            ),
            'complete_student_beneficial_count': int(student_beneficial.sum()),
            'complete_student_beneficial_fraction': float(student_beneficial.mean()),
            'complete_target_mse_backbone': float(merged['backbone_mse'].mean()),
            'complete_target_mae_backbone': float(merged['backbone_mae'].mean()),
            'complete_target_mse_teacher': float(merged['teacher_mse'].mean()),
            'complete_target_mae_teacher': float(merged['teacher_mae'].mean()),
            'complete_target_mse_student': float(merged['student_mse'].mean()),
            'complete_target_mae_student': float(merged['student_mae'].mean()),
            'mse': float(merged['student_mse'].mean()),
            'mae': float(merged['student_mae'].mean()),
            'target_rate': float(self.args.stage22_31_target_rate),
            'block_lengths': [5, 20],
            'nested_seed': int(self.args.stage22_31_nested_seed),
            'teacher_weights': {
                'original': float(self.args.stage22_31_original_weight),
                'l5': float(self.args.stage22_31_l5_weight),
                'l20': float(self.args.stage22_31_l20_weight),
            },
            'history_row_shared_all_samples': bool(history_row_shared.all()),
            'target_row_shared_all_samples': bool(target_row_shared.all()),
            'pba_model_hash_before': pba_hash_before,
            'pba_model_hash_after': pba_hash_after,
            'pba_memory_hash_before': pba_memory_hash_before,
            'pba_memory_hash_after': pba_memory_hash_after,
            'source_model_hash_before': source_hash_before,
            'source_model_hash_after': source_hash_after,
            'source_memory_hash_before': source_memory_hash_before,
            'source_memory_hash_after': source_memory_hash_after,
            'pba_student_forward_count': int(evaluated_batches),
            'source_forward_count': int(evaluated_batches * 3),
            'parameter_count': int(self.parameter_count),
            'parameter_bytes': int(self.parameter_bytes),
            'buffer_bytes': int(self.buffer_bytes),
            'evaluation_elapsed_seconds': float(elapsed),
            'peak_gpu_memory_mb': (
                float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
                if self.device.type == 'cuda' else None
            ),
            'peak_gpu_reserved_mb': (
                float(torch.cuda.max_memory_reserved(self.device) / (1024 ** 2))
                if self.device.type == 'cuda' else None
            ),
        }
        samples_path = self.args.stage22_31_samples_path
        os.makedirs(os.path.dirname(samples_path), exist_ok=True)
        temporary_samples = samples_path + '.tmp'
        with open(temporary_samples, 'wb') as handle:
            np.savez_compressed(handle, **merged, **variable_totals)
        os.replace(temporary_samples, samples_path)
        metrics_path = self.args.stage22_31_metrics_path
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        temporary_metrics = metrics_path + '.tmp'
        with open(temporary_metrics, 'w') as metrics_file:
            json.dump(payload, metrics_file, indent=2, sort_keys=True)
        os.replace(temporary_metrics, metrics_path)
        print('STAGE22_31_PBA ' + json.dumps(payload, sort_keys=True))
        return payload

    @staticmethod
    def _stage22_29_token_diagnostics(
        backbone_prediction,
        residual_l5,
        residual_l20,
        target_correction_l5,
        target_correction_l20,
        complete_target,
    ):
        """Return compact paired evidence for frozen DMRB branch attribution."""
        tensors = (
            backbone_prediction, residual_l5, residual_l20,
            target_correction_l5, target_correction_l20, complete_target,
        )
        if any(value.shape != tensors[0].shape for value in tensors[1:]):
            raise ValueError('stage22.29 prediction, residual and target shapes must match')
        if backbone_prediction.ndim != 3:
            raise ValueError('stage22.29 tensors must be [batch,horizon,variables]')

        student_l5 = backbone_prediction + residual_l5
        student_l20 = backbone_prediction + residual_l20
        student_joint = student_l5 + residual_l20
        teacher_l5 = backbone_prediction + target_correction_l5
        teacher_l20 = backbone_prediction + target_correction_l20
        teacher_joint = teacher_l5 + target_correction_l20

        def losses(prediction):
            error = prediction - complete_target
            return error.square().mean(dim=-1), error.abs().mean(dim=-1)

        predictions = {
            'backbone': backbone_prediction,
            'teacher_l5': teacher_l5,
            'teacher_l20': teacher_l20,
            'teacher_joint': teacher_joint,
            'student_l5': student_l5,
            'student_l20': student_l20,
            'student_joint': student_joint,
        }
        result = {}
        for label, prediction in predictions.items():
            mse, mae = losses(prediction)
            result[f'{label}_mse'] = mse
            result[f'{label}_mae'] = mae

        for label in ('teacher_l5', 'teacher_l20', 'teacher_joint'):
            result[f'{label}_beneficial'] = (
                (result[f'{label}_mse'] < result['backbone_mse'])
                & (result[f'{label}_mae'] < result['backbone_mae'])
            )
        for label in ('student_l5', 'student_l20', 'student_joint'):
            result[f'{label}_beneficial'] = (
                (result[f'{label}_mse'] < result['backbone_mse'])
                & (result[f'{label}_mae'] < result['backbone_mae'])
            )

        def geometry(prefix, residual, target):
            result[f'{prefix}_dot'] = (residual * target).sum(dim=-1)
            result[f'{prefix}_residual_energy'] = residual.square().sum(dim=-1)
            result[f'{prefix}_target_energy'] = target.square().sum(dim=-1)
            result[f'{prefix}_mismatch_energy'] = (
                residual - target
            ).square().sum(dim=-1)

        geometry('l5_self', residual_l5, target_correction_l5)
        geometry('l20_self', residual_l20, target_correction_l20)
        result['l5_to_l20_target_dot'] = (
            residual_l5 * target_correction_l20
        ).sum(dim=-1)
        result['l20_to_l5_target_dot'] = (
            residual_l20 * target_correction_l5
        ).sum(dim=-1)
        result['teacher_component_dot'] = (
            target_correction_l5 * target_correction_l20
        ).sum(dim=-1)
        result['student_branch_dot'] = (
            residual_l5 * residual_l20
        ).sum(dim=-1)
        result['teacher_sum_energy'] = (
            target_correction_l5 + target_correction_l20
        ).square().sum(dim=-1)
        result['student_sum_energy'] = (
            residual_l5 + residual_l20
        ).square().sum(dim=-1)
        return result

    def stage22_29_dmrb_diagnostic(self, setting):
        """Attribute frozen DMRB branch supervision/composition failures on val."""
        validation_data, _ = self._get_data(flag='val')
        validation_loader = DataLoader(
            validation_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=True,
        )
        f_dim = -1 if self.args.features == 'MS' else 0
        core = self._core_model()
        if (
            core.mcpr is None
            or not bool(getattr(core, 'use_mcpr', False))
            or not bool(getattr(core, 'use_dmrb', False))
        ):
            raise RuntimeError('stage22.29 requires an enabled DMRB student')
        if not (
            bool(getattr(core.mcpr, 'l5_enabled', False))
            and bool(getattr(core.mcpr, 'l20_enabled', False))
        ):
            raise RuntimeError('stage22.29 requires both DMRB branches enabled')
        if bool(getattr(core, 'use_ard', False)) or bool(
            getattr(core, 'use_ttrr', False)
        ):
            raise RuntimeError('stage22.29 requires ungated DMRB branches')
        core.set_prototype_enabled(True)
        self.change_mode(0)
        self.model.eval()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

        boolean_keys = {
            'teacher_l5_beneficial', 'teacher_l20_beneficial',
            'teacher_joint_beneficial', 'student_l5_beneficial',
            'student_l20_beneficial', 'student_joint_beneficial',
        }
        loss_keys = {
            f'{label}_{metric}'
            for label in (
                'backbone', 'teacher_l5', 'teacher_l20', 'teacher_joint',
                'student_l5', 'student_l20', 'student_joint',
            )
            for metric in ('mse', 'mae')
        }
        geometry_keys = {
            'l5_self_dot', 'l5_self_residual_energy', 'l5_self_target_energy',
            'l5_self_mismatch_energy', 'l20_self_dot',
            'l20_self_residual_energy', 'l20_self_target_energy',
            'l20_self_mismatch_energy', 'l5_to_l20_target_dot',
            'l20_to_l5_target_dot', 'teacher_component_dot',
            'student_branch_dot', 'teacher_sum_energy', 'student_sum_energy',
        }
        token_keys = boolean_keys | loss_keys | geometry_keys
        arrays = {key: [] for key in token_keys}
        arrays.update({
            'sample_index': [],
            'history_mask_packed': [],
            'l5_mask_packed': [],
            'l20_mask_packed': [],
        })
        history_row_shared = []
        evaluated_batches = 0
        student_identity_max_abs = 0.0
        teacher_identity_max_abs = 0.0
        started = time.monotonic()

        with torch.no_grad():
            for batch_index, batch in enumerate(validation_loader):
                if (
                    self.args.max_eval_batches >= 0
                    and batch_index >= self.args.max_eval_batches
                ):
                    break
                (
                    seq_x, seq_x_mask, seq_y, seq_y_mask, seq_true_x, seq_true_y,
                    max_idx, min_idx, max_value, min_value,
                ) = tuple(item.float().to(self.device) for item in batch)
                batch_size = int(seq_x.shape[0])
                sample_index = np.arange(
                    batch_index * int(self.args.batch_size),
                    batch_index * int(self.args.batch_size) + batch_size,
                    dtype=np.int64,
                )

                student_prediction = self.model(
                    seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                )[:, -self.args.pred_len:, f_dim:]
                dmrb_outputs = core.get_last_mcpr_outputs()
                required = {'residual', 'residual_l5', 'residual_l20'}
                if dmrb_outputs is None or not required.issubset(dmrb_outputs):
                    raise RuntimeError('stage22.29 forward returned incomplete DMRB outputs')
                residual_l5 = dmrb_outputs['residual_l5'][
                    :, -self.args.pred_len:, f_dim:
                ]
                residual_l20 = dmrb_outputs['residual_l20'][
                    :, -self.args.pred_len:, f_dim:
                ]
                residual_sum = dmrb_outputs['residual'][
                    :, -self.args.pred_len:, f_dim:
                ]
                if not (
                    residual_l5.shape == residual_l20.shape
                    == residual_sum.shape == student_prediction.shape
                ):
                    raise RuntimeError('stage22.29 DMRB residual shape mismatch')
                backbone_prediction = student_prediction - residual_l5 - residual_l20
                student_identity_max_abs = max(
                    student_identity_max_abs,
                    float((
                        student_prediction
                        - backbone_prediction - residual_l5 - residual_l20
                    ).abs().max().item()),
                    float((residual_sum - residual_l5 - residual_l20).abs().max().item()),
                )

                previous_mcpr_state = bool(core.use_mcpr)
                nested_predictions = {}
                nested_masks = {}
                core.use_mcpr = False
                try:
                    for block_length in (5, 20):
                        nested_mask = self._stage18_equal_count_observation_mask(
                            seq_x_mask,
                            sample_index,
                            float(self.args.stage22_29_target_rate),
                            int(block_length),
                            int(self.args.stage22_29_nested_seed),
                        )
                        nested_x = seq_true_x * nested_mask
                        nested_prediction = self.model(
                            nested_x,
                            nested_mask,
                            *self._temporal_extrema(nested_x),
                        )[:, -self.args.pred_len:, f_dim:]
                        nested_masks[block_length] = nested_mask
                        nested_predictions[block_length] = nested_prediction
                finally:
                    core.use_mcpr = previous_mcpr_state

                target_correction_l5 = float(
                    self.args.stage22_29_l5_weight
                ) * (nested_predictions[5] - backbone_prediction)
                target_correction_l20 = float(
                    self.args.stage22_29_l20_weight
                ) * (nested_predictions[20] - backbone_prediction)
                teacher_prediction = (
                    float(self.args.stage22_29_original_weight)
                    * backbone_prediction
                    + float(self.args.stage22_29_l5_weight)
                    * nested_predictions[5]
                    + float(self.args.stage22_29_l20_weight)
                    * nested_predictions[20]
                )
                factored_teacher = (
                    backbone_prediction
                    + target_correction_l5 + target_correction_l20
                )
                teacher_identity_max_abs = max(
                    teacher_identity_max_abs,
                    float((teacher_prediction - factored_teacher).abs().max().item()),
                )

                complete_target = seq_true_y[:, -self.args.pred_len:, f_dim:]
                diagnostics = self._stage22_29_token_diagnostics(
                    backbone_prediction=backbone_prediction,
                    residual_l5=residual_l5,
                    residual_l20=residual_l20,
                    target_correction_l5=target_correction_l5,
                    target_correction_l20=target_correction_l20,
                    complete_target=complete_target,
                )
                for key in token_keys:
                    value = diagnostics[key]
                    cached = (
                        value.to(torch.uint8).cpu().numpy()
                        if value.dtype == torch.bool
                        else value.float().cpu().numpy()
                    )
                    arrays[key].append(cached)

                original_observed = seq_x_mask[:, :, 0] > 0.5
                arrays['sample_index'].append(sample_index)
                arrays['history_mask_packed'].append(np.packbits(
                    original_observed.to(torch.uint8).cpu().numpy(), axis=1
                ))
                for block_length in (5, 20):
                    nested_observed = nested_masks[block_length][:, :, 0] > 0.5
                    arrays[f'l{block_length}_mask_packed'].append(np.packbits(
                        nested_observed.to(torch.uint8).cpu().numpy(), axis=1
                    ))
                history_row_shared.append((
                    (seq_x_mask > 0.5) == (seq_x_mask[:, :, :1] > 0.5)
                ).reshape(batch_size, -1).all(dim=1).cpu().numpy())
                evaluated_batches += 1

        if not arrays['sample_index']:
            raise RuntimeError('stage22.29 diagnostic processed zero batches')
        packed_keys = {'history_mask_packed', 'l5_mask_packed', 'l20_mask_packed'}
        merged = {}
        for key, chunks in arrays.items():
            if key in boolean_keys or key in packed_keys:
                dtype = np.uint8
            elif key == 'sample_index':
                dtype = np.int64
            else:
                dtype = np.float32
            merged[key] = np.concatenate(chunks).astype(dtype, copy=False)
        history_row_shared = np.concatenate(history_row_shared).astype(bool)
        if not bool(history_row_shared.all()):
            raise RuntimeError('stage22.29 Traffic history masks must be row-shared')

        token_count = int(merged['backbone_mse'].size)
        sample_count = int(merged['sample_index'].size)
        variable_count = int(complete_target.shape[-1])
        elapsed = time.monotonic() - started
        payload = {
            'setting': setting,
            'checkpoint': self.args.stage22_29_checkpoint,
            'source_split': 'val',
            'evaluation_split': 'validation',
            'test_split_accessed': False,
            'uses_test_split': False,
            'chronological_loader': True,
            'updates_model_or_memory': False,
            'compact_cache_excludes_full_predictions': True,
            'sample_count': sample_count,
            'validation_samples_processed': sample_count,
            'validation_batches_processed': int(evaluated_batches),
            'prediction_horizon': int(self.args.pred_len),
            'variable_count': variable_count,
            'complete_token_count': token_count,
            'complete_element_count': token_count * variable_count,
            'student_factorization_max_abs': float(student_identity_max_abs),
            'teacher_factorization_max_abs': float(teacher_identity_max_abs),
            'history_row_shared_all_samples': bool(history_row_shared.all()),
            'target_rate': float(self.args.stage22_29_target_rate),
            'block_lengths': [5, 20],
            'nested_seed': int(self.args.stage22_29_nested_seed),
            'teacher_weights': {
                'original': float(self.args.stage22_29_original_weight),
                'l5': float(self.args.stage22_29_l5_weight),
                'l20': float(self.args.stage22_29_l20_weight),
            },
            'dmrb_l5_enabled': True,
            'dmrb_l20_enabled': True,
            'parameter_count': int(self.parameter_count),
            'parameter_bytes': int(self.parameter_bytes),
            'buffer_bytes': int(self.buffer_bytes),
            'evaluation_elapsed_seconds': float(elapsed),
            'peak_gpu_memory_mb': (
                float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
                if self.device.type == 'cuda' else None
            ),
            'peak_gpu_reserved_mb': (
                float(torch.cuda.max_memory_reserved(self.device) / (1024 ** 2))
                if self.device.type == 'cuda' else None
            ),
        }
        for label in (
            'backbone', 'teacher_l5', 'teacher_l20', 'teacher_joint',
            'student_l5', 'student_l20', 'student_joint',
        ):
            payload[f'complete_target_mse_{label}'] = float(
                merged[f'{label}_mse'].mean()
            )
            payload[f'complete_target_mae_{label}'] = float(
                merged[f'{label}_mae'].mean()
            )
        for label in (
            'teacher_l5', 'teacher_l20', 'teacher_joint',
            'student_l5', 'student_l20', 'student_joint',
        ):
            payload[f'{label}_beneficial_count'] = int(
                merged[f'{label}_beneficial'].sum()
            )
            payload[f'{label}_beneficial_fraction'] = float(
                merged[f'{label}_beneficial'].mean()
            )
        payload['mse'] = payload['complete_target_mse_student_joint']
        payload['mae'] = payload['complete_target_mae_student_joint']

        samples_path = self.args.stage22_29_samples_path
        os.makedirs(os.path.dirname(samples_path), exist_ok=True)
        temporary_samples = samples_path + '.tmp'
        with open(temporary_samples, 'wb') as handle:
            np.savez_compressed(handle, **merged)
        os.replace(temporary_samples, samples_path)
        metrics_path = self.args.stage22_29_metrics_path
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        temporary_metrics = metrics_path + '.tmp'
        with open(temporary_metrics, 'w') as metrics_file:
            json.dump(payload, metrics_file, indent=2, sort_keys=True)
        os.replace(temporary_metrics, metrics_path)
        print('STAGE22_29_DMRB ' + json.dumps(payload, sort_keys=True))
        return payload

    def stage22_inference_timing_validation(self, setting):
        """Time one frozen model on chronological validation without test access.

        Timing covers the CUDA model forward only. DataLoader work, host-to-device
        transfer and metric reductions are deliberately outside the timed region.
        """
        validation_data, _ = self._get_data(flag='val')
        validation_loader = DataLoader(
            validation_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=True,
        )
        f_dim = -1 if self.args.features == 'MS' else 0
        warmup_batches = int(self.args.stage22_timing_warmup_batches)
        repeats = int(self.args.stage22_timing_repeats)
        self.change_mode(0)
        self.model.eval()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

        repeat_summaries = []
        latency_ms_all = []
        reference_mse = None
        reference_mae = None
        reference_batches = None
        reference_samples = None
        reference_elements = None
        with torch.no_grad():
            for repeat_index in range(repeats):
                squared_error_sum = 0.0
                absolute_error_sum = 0.0
                element_count = 0
                evaluated_batches = 0
                evaluated_samples = 0
                repeat_latency_ms = []
                for batch_index, batch in enumerate(validation_loader):
                    if (
                        self.args.max_eval_batches >= 0
                        and batch_index >= self.args.max_eval_batches
                    ):
                        break
                    (
                        seq_x, seq_x_mask, seq_y, seq_y_mask,
                        seq_true_x, seq_true_y, max_idx, min_idx,
                        max_value, min_value,
                    ) = tuple(item.float().to(self.device) for item in batch)
                    del seq_y, seq_y_mask, seq_true_x
                    if self.device.type == 'cuda':
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                        prediction = self.model(
                            seq_x, seq_x_mask, max_idx, min_idx,
                            max_value, min_value,
                        )[:, -self.args.pred_len:, f_dim:]
                        end_event.record()
                        end_event.synchronize()
                        forward_ms = float(start_event.elapsed_time(end_event))
                    else:
                        forward_started = time.perf_counter()
                        prediction = self.model(
                            seq_x, seq_x_mask, max_idx, min_idx,
                            max_value, min_value,
                        )[:, -self.args.pred_len:, f_dim:]
                        forward_ms = float(
                            (time.perf_counter() - forward_started) * 1000.0
                        )
                    if batch_index >= warmup_batches:
                        repeat_latency_ms.append(forward_ms)
                    target = seq_true_y[:, -self.args.pred_len:, f_dim:]
                    error = prediction - target
                    squared_error_sum += float(error.square().sum().item())
                    absolute_error_sum += float(error.abs().sum().item())
                    element_count += int(error.numel())
                    evaluated_samples += int(error.shape[0])
                    evaluated_batches += 1

                if evaluated_batches < 1 or element_count < 1:
                    raise RuntimeError('stage22 timing processed zero complete targets')
                if not repeat_latency_ms:
                    raise RuntimeError('stage22 timing warmup consumed every batch')
                mse = float(squared_error_sum / element_count)
                mae = float(absolute_error_sum / element_count)
                if reference_mse is None:
                    reference_mse = mse
                    reference_mae = mae
                    reference_batches = evaluated_batches
                    reference_samples = evaluated_samples
                    reference_elements = element_count
                elif not (
                    abs(mse - reference_mse) <= 1e-12
                    and abs(mae - reference_mae) <= 1e-12
                    and evaluated_batches == reference_batches
                    and evaluated_samples == reference_samples
                    and element_count == reference_elements
                ):
                    raise RuntimeError('stage22 timing repeats changed frozen outputs')
                latency_array = np.asarray(repeat_latency_ms, dtype=np.float64)
                repeat_summaries.append({
                    'repeat_index': int(repeat_index),
                    'mse': mse,
                    'mae': mae,
                    'evaluated_batches': int(evaluated_batches),
                    'evaluated_samples': int(evaluated_samples),
                    'measured_batches': int(latency_array.size),
                    'forward_mean_ms_per_batch': float(latency_array.mean()),
                    'forward_std_ms_per_batch': float(latency_array.std(ddof=0)),
                    'forward_p50_ms_per_batch': float(np.percentile(latency_array, 50)),
                    'forward_p95_ms_per_batch': float(np.percentile(latency_array, 95)),
                    'forward_total_seconds': float(latency_array.sum() / 1000.0),
                })
                latency_ms_all.extend(repeat_latency_ms)

        latency_array = np.asarray(latency_ms_all, dtype=np.float64)
        payload = {
            'setting': setting,
            'checkpoint': self.args.stage22_timing_checkpoint,
            'source_split': 'val',
            'evaluation_split': 'validation',
            'test_split_accessed': False,
            'uses_test_split': False,
            'chronological_loader': True,
            'updates_model_or_memory': False,
            'timing_device_serial': True,
            'timing_scope': (
                'model forward only; excludes dataloader, host-to-device transfer, '
                'metric reduction and checkpoint loading'
            ),
            'timing_clock': (
                'torch.cuda.Event' if self.device.type == 'cuda'
                else 'time.perf_counter'
            ),
            'timing_warmup_batches_per_repeat': warmup_batches,
            'timing_repeats': repeats,
            'timing_measured_batches_total': int(latency_array.size),
            'timing_batch_size': int(self.args.batch_size),
            'forward_mean_ms_per_batch': float(latency_array.mean()),
            'forward_std_ms_per_batch': float(latency_array.std(ddof=0)),
            'forward_p50_ms_per_batch': float(np.percentile(latency_array, 50)),
            'forward_p95_ms_per_batch': float(np.percentile(latency_array, 95)),
            'forward_min_ms_per_batch': float(latency_array.min()),
            'forward_max_ms_per_batch': float(latency_array.max()),
            'forward_mean_ms_per_sample': float(
                latency_array.mean() / int(self.args.batch_size)
            ),
            'forward_total_seconds': float(latency_array.sum() / 1000.0),
            'forward_latency_ms': latency_array.tolist(),
            'repeat_summaries': repeat_summaries,
            'mse': float(reference_mse),
            'mae': float(reference_mae),
            'complete_target_mse': float(reference_mse),
            'complete_target_mae': float(reference_mae),
            'validation_batches_processed': int(reference_batches),
            'validation_samples_processed': int(reference_samples),
            'validation_complete_element_count': int(reference_elements),
            'parameter_count': int(self.parameter_count),
            'parameter_bytes': int(self.parameter_bytes),
            'buffer_bytes': int(self.buffer_bytes),
            'peak_gpu_memory_mb': (
                float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
                if self.device.type == 'cuda' else None
            ),
            'peak_gpu_reserved_mb': (
                float(torch.cuda.max_memory_reserved(self.device) / (1024 ** 2))
                if self.device.type == 'cuda' else None
            ),
        }
        metrics_dir = os.path.dirname(self.args.stage22_timing_metrics_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        temporary = self.args.stage22_timing_metrics_path + '.tmp'
        with open(temporary, 'w') as metrics_file:
            json.dump(payload, metrics_file, indent=2, sort_keys=True)
        os.replace(temporary, self.args.stage22_timing_metrics_path)
        print(
            'stage22 timing validation mse:{}, mae:{}, mean_ms_per_batch:{}'.format(
                reference_mse, reference_mae,
                payload['forward_mean_ms_per_batch'],
            )
        )
        return payload

    def test(self, setting, test=0,epoch=0):
        test_data, test_loader = self._get_data(flag='test')
        f_dim = -1 if self.args.features == 'MS' else 0
        criterion = self._select_criterion()
        
        if test:
            print('loading model')
            self.load_model(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        preds = []
        trues = []
        folder_path = os.path.join(self.args.test_results_root, setting) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.change_mode(0)
        total_loss = []
        evaluated_batches = 0
        fusr_gate_all = []
        fusr_utility_all = []
        fusr_uncertainty_all = []
        fusr_spectral_all = []
        fusr_route_mse_all = []
        fusr_on_mse_all = []
        fusr_off_mse_all = []
        core_model = self._core_model()
        if self.args.use_fusr:
            core_model.set_fusr_mode('route')
        self.model.eval()
        with torch.no_grad():
            for i, (seq_x,seq_x_mask,seq_y,seq_y_mask,seq_true_x,seq_true_y,max_idx,min_idx,max_value,min_value) in enumerate(test_loader):
                if self.args.max_eval_batches >= 0 and i >= self.args.max_eval_batches:
                    break
                seq_x,seq_x_mask,seq_y,seq_y_mask,seq_true_x,seq_true_y,max_idx,min_idx,max_value,min_value \
                    = seq_x.float().to(self.device),seq_x_mask.float().to(self.device),seq_y.float().to(self.device),seq_y_mask.float().to(self.device),seq_true_x.float().to(self.device),seq_true_y.float().to(self.device),max_idx.float().to(self.device),min_idx.float().to(self.device),max_value.float().to(self.device),min_value.float().to(self.device)

                seq_y = seq_y[:, -self.args.pred_len:, f_dim:]
                seq_y_mask = seq_y_mask[:, -self.args.pred_len:, f_dim:]
                outputs = self.model(seq_x,seq_x_mask,max_idx,min_idx,max_value,min_value)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if self.args.use_fusr:
                    fusr_outputs = core_model.get_last_fusr_outputs()
                    if fusr_outputs is None:
                        raise RuntimeError('FUSR evaluation produced no router outputs')
                    target_gpu = seq_true_y[:, -self.args.pred_len:, f_dim:]
                    reduce_dims = tuple(range(1, target_gpu.ndim))
                    route_mse = torch.mean((outputs - target_gpu) ** 2, dim=reduce_dims)
                    fusr_route_mse_all.append(route_mse.detach().cpu().numpy())
                    fusr_gate_all.append(
                        fusr_outputs['effective_gate'].detach().cpu().numpy()
                    )
                    fusr_utility_all.append(
                        fusr_outputs.get(
                            'gate_score', fusr_outputs['utility_lcb']
                        ).detach().cpu().numpy()
                    )
                    fusr_uncertainty_all.append(
                        fusr_outputs['utility_std'].detach().cpu().numpy()
                    )
                    fusr_spectral_all.append(
                        fusr_outputs['features'][:, 7].detach().cpu().numpy()
                    )
                    if self.args.fusr_eval_paired_diagnostics:
                        try:
                            core_model.set_fusr_mode('on')
                            outputs_on = self.model(
                                seq_x,seq_x_mask,max_idx,min_idx,max_value,min_value
                            )[:, -self.args.pred_len:, f_dim:]
                            core_model.set_fusr_mode('off')
                            outputs_off = self.model(
                                seq_x,seq_x_mask,max_idx,min_idx,max_value,min_value
                            )[:, -self.args.pred_len:, f_dim:]
                        finally:
                            core_model.set_fusr_mode('route')
                        fusr_on_mse_all.append(
                            torch.mean(
                                (outputs_on - target_gpu) ** 2, dim=reduce_dims
                            ).cpu().numpy()
                        )
                        fusr_off_mse_all.append(
                            torch.mean(
                                (outputs_off - target_gpu) ** 2, dim=reduce_dims
                            ).cpu().numpy()
                        )
                pred = outputs.detach().cpu()
                seq_y = seq_y[:,-self.args.pred_len:, f_dim:].detach().cpu().numpy()
                seq_y_true = seq_true_y[:,-self.args.pred_len:,f_dim:].detach().cpu()

                loss = criterion(pred, seq_y_true)
                total_loss.append(loss.item())
                seq_y_true = seq_y_true.numpy()
                pred = pred.cpu().numpy()
                trues.append(seq_y_true)
                preds.append(pred)
                evaluated_batches += 1
                
                seq_true_x = seq_true_x.detach().cpu().numpy()
                if self.args.plot and i%100==0:
                    if test_data.scale:
                        pred = pred[0,:,:]
                        pred = test_data.inverse_transform1(pred)
                        seq_true_y = seq_y_true[0,:,:]
                        seq_true_y = test_data.inverse_transform5(seq_true_y)
                        seq_true_x = seq_true_x[0,:,:]
                        seq_true_x = test_data.inverse_transform5(seq_true_x)
                    seq_x_mask = seq_x_mask.detach().cpu().numpy()
                    seq_y_mask = seq_y_mask[:, -self.args.pred_len:, f_dim:].detach().cpu().numpy()
                    for j in [-1,0,1]:
                        true_xy = np.concatenate((seq_true_x[:,j],seq_true_y[:,j]),axis=0)
                        pre = pred[:,j]
                        mask = np.concatenate((seq_x_mask[0,:,j],seq_y_mask[0,:,j]),axis=0)
                        visual(true_xy,pre,mask, os.path.join(folder_path, str(i) + '_'+str(j)+'.png'))
                        wandb.save(os.path.join(folder_path, str(i) + '_'+str(j)+'.png'))
                self.args.plot=0
                    
        mae, mse, rmse, mape, mspe = 0.0,0.0,0.0,0.0,0.0
        
        for i in range(len(preds)):
            mae+= MAE(preds[i],trues[i])
            mse+=MSE(preds[i],trues[i])
            rmse+=RMSE(preds[i],trues[i])
            mape+=MAPE(preds[i],trues[i])
            mspe+=MSPE(preds[i],trues[i])
            
        mae,mse,rmse,mape,mspe = mae/len(preds), mse/len(preds), rmse/len(preds), mape/len(preds), mspe/len(preds)
        total_loss = np.average(total_loss)
        
        # result save
        folder_path = os.path.join(self.args.results_root, setting) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open(os.path.join(self.args.results_root, "result_long_term_forecast.txt"), 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        wandb.log({"mse":mse,'mae':mae,'rmse':rmse,'mape':mape,'mspe':mspe,"total_loss":total_loss})

        fusr_test_evidence = {}
        if self.args.use_fusr:
            gate = np.concatenate(fusr_gate_all).astype(np.float64, copy=False)
            predicted_utility = np.concatenate(fusr_utility_all).astype(
                np.float64, copy=False
            )
            predicted_uncertainty = np.concatenate(fusr_uncertainty_all).astype(
                np.float64, copy=False
            )
            spectral_uncertainty = np.concatenate(fusr_spectral_all).astype(
                np.float64, copy=False
            )
            route_sample_mse = np.concatenate(fusr_route_mse_all).astype(
                np.float64, copy=False
            )
            fusr_test_evidence.update({
                'fusr_eval_sample_count': int(gate.size),
                'fusr_eval_gate_mean': float(np.mean(gate)),
                'fusr_eval_abstention_rate': float(np.mean(1.0 - gate)),
                'fusr_eval_predicted_utility_mean': float(np.mean(predicted_utility)),
                'fusr_eval_predicted_utility_std': float(np.std(predicted_utility)),
                'fusr_eval_uncertainty_mean': float(np.mean(predicted_uncertainty)),
                'fusr_eval_spectral_uncertainty_mean': float(
                    np.mean(spectral_uncertainty)
                ),
                'fusr_routed_sample_mse': float(np.mean(route_sample_mse)),
            })
            if fusr_on_mse_all:
                mse_on = np.concatenate(fusr_on_mse_all).astype(np.float64, copy=False)
                mse_off = np.concatenate(fusr_off_mse_all).astype(np.float64, copy=False)
                utility = mse_off - mse_on
                delta = float(self.args.fusr_delta)
                beneficial = utility > delta
                harmful = utility < -delta
                harmful_use = harmful & (gate >= 0.5)
                oracle = np.minimum(mse_on, mse_off)
                on_mean = float(np.mean(mse_on))
                off_mean = float(np.mean(mse_off))
                route_mean = float(np.mean(route_sample_mse))
                oracle_mean = float(np.mean(oracle))
                hrr = float(np.mean(harmful))
                harmful_use_rate = float(np.mean(harmful_use))
                fusr_test_evidence.update({
                    'fusr_always_on_sample_mse': on_mean,
                    'fusr_always_off_sample_mse': off_mean,
                    'fusr_oracle_sample_mse': oracle_mean,
                    'fusr_hrr_always_on': hrr,
                    'fusr_harmful_use_rate': harmful_use_rate,
                    'fusr_hrr_relative_reduction': (
                        float((hrr - harmful_use_rate) / hrr) if hrr > 0.0 else None
                    ),
                    'fusr_beneficial_retrieval_rate': float(np.mean(beneficial)),
                    'fusr_gate_utility_sign_accuracy': float(
                        np.mean((gate >= 0.5) == beneficial)
                    ),
                    'fusr_utility_auroc': self._binary_auc(
                        beneficial, predicted_utility
                    ),
                    'fusr_route_relative_gain_vs_always_on': (
                        float((on_mean - route_mean) / on_mean)
                        if on_mean != 0.0 else None
                    ),
                    'fusr_oracle_relative_gain_vs_always_on': (
                        float((on_mean - oracle_mean) / on_mean)
                        if on_mean != 0.0 else None
                    ),
                    'fusr_oracle_gap': float(route_mean - oracle_mean),
                })

        metrics_payload = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'mspe': float(mspe),
            'complete_target_mse': float(total_loss),
            'optimizer': self.args.optimizer,
            'parameter_count': int(self.parameter_count),
            'parameter_bytes': int(self.parameter_bytes),
            'buffer_bytes': int(self.buffer_bytes),
            'setting': setting,
            'checkpoint': os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'),
            'test_batches_processed': evaluated_batches,
            'peak_gpu_memory_mb': float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)) if self.device.type == 'cuda' else None,
            'peak_gpu_reserved_mb': float(torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)) if self.device.type == 'cuda' else None,
            **getattr(self, 'run_evidence', {}),
            **fusr_test_evidence,
        }
        if self.args.metrics_path:
            metrics_dir = os.path.dirname(self.args.metrics_path)
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
            temp_path = self.args.metrics_path + '.tmp'
            with open(temp_path, 'w') as metrics_file:
                json.dump(metrics_payload, metrics_file, indent=2, sort_keys=True)
            os.replace(temp_path, self.args.metrics_path)

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        return metrics_payload

    @staticmethod
    def _average_ranks(values):
        values = np.asarray(values, dtype=np.float64)
        order = np.argsort(values, kind='mergesort')
        ranks = np.empty(values.shape[0], dtype=np.float64)
        start = 0
        while start < values.shape[0]:
            end = start + 1
            while end < values.shape[0] and values[order[end]] == values[order[start]]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
            start = end
        return ranks

    @staticmethod
    def _safe_correlation(left, right):
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
            return None
        return float(np.corrcoef(left, right)[0, 1])

    def stage12_validation_diagnostic(self, setting):
        """Cache chronological validation errors and historical masks.

        This is a frozen-checkpoint diagnostic.  It never reads the test split,
        never updates model parameters or the prototype memory, and does not
        alter the original validation/early-stopping implementation.
        """
        validation_data, _ = self._get_data(flag='val')
        validation_loader = DataLoader(
            validation_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=True,
        )
        f_dim = -1 if self.args.features == 'MS' else 0
        core_model = self._core_model()
        core_model.set_prototype_enabled(True)
        self.change_mode(0)
        self.model.eval()

        sample_index_all = []
        history_mask_packed_all = []
        complete_sse_all = []
        complete_sae_all = []
        complete_count_all = []
        observed_sse_all = []
        observed_sae_all = []
        observed_count_all = []
        row_shared_all = []
        evaluated_batches = 0

        with torch.no_grad():
            for batch_index, batch in enumerate(validation_loader):
                if self.args.max_eval_batches >= 0 and batch_index >= self.args.max_eval_batches:
                    break
                (
                    seq_x, seq_x_mask, seq_y, seq_y_mask, seq_true_x, seq_true_y,
                    max_idx, min_idx, max_value, min_value,
                ) = tuple(item.float().to(self.device) for item in batch)
                del seq_true_x
                batch_size = int(seq_x.shape[0])
                start = batch_index * int(self.args.batch_size)
                sample_index = np.arange(start, start + batch_size, dtype=np.int64)

                prediction = self.model(
                    seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                )[:, -self.args.pred_len:, f_dim:]
                complete_target = seq_true_y[:, -self.args.pred_len:, f_dim:]
                observed_target = seq_y[:, -self.args.pred_len:, f_dim:]
                observed_mask = seq_y_mask[:, -self.args.pred_len:, f_dim:]
                reduce_dims = tuple(range(1, prediction.ndim))

                complete_error = prediction - complete_target
                observed_error = prediction - observed_target
                complete_sse = (complete_error ** 2).sum(dim=reduce_dims)
                complete_sae = complete_error.abs().sum(dim=reduce_dims)
                complete_count = torch.full_like(
                    complete_sse, float(complete_target[0].numel())
                )
                observed_sse = (
                    (observed_error ** 2) * observed_mask
                ).sum(dim=reduce_dims)
                observed_sae = (
                    observed_error.abs() * observed_mask
                ).sum(dim=reduce_dims)
                observed_count = observed_mask.sum(dim=reduce_dims)

                mask_binary = seq_x_mask > 0.5
                row_shared = (
                    mask_binary == mask_binary[:, :, :1]
                ).reshape(batch_size, -1).all(dim=1)
                history_row_mask = mask_binary[:, :, 0].to(torch.uint8).cpu().numpy()
                history_mask_packed = np.packbits(history_row_mask, axis=1)

                sample_index_all.append(sample_index)
                history_mask_packed_all.append(history_mask_packed)
                complete_sse_all.append(complete_sse.cpu().numpy())
                complete_sae_all.append(complete_sae.cpu().numpy())
                complete_count_all.append(complete_count.cpu().numpy())
                observed_sse_all.append(observed_sse.cpu().numpy())
                observed_sae_all.append(observed_sae.cpu().numpy())
                observed_count_all.append(observed_count.cpu().numpy())
                row_shared_all.append(row_shared.cpu().numpy())
                evaluated_batches += 1

        if not sample_index_all:
            raise RuntimeError('stage12 validation diagnostic processed zero batches')

        sample_index = np.concatenate(sample_index_all).astype(np.int64, copy=False)
        history_mask_packed = np.concatenate(history_mask_packed_all).astype(
            np.uint8, copy=False
        )
        complete_sse = np.concatenate(complete_sse_all).astype(np.float64, copy=False)
        complete_sae = np.concatenate(complete_sae_all).astype(np.float64, copy=False)
        complete_count = np.concatenate(complete_count_all).astype(np.float64, copy=False)
        observed_sse = np.concatenate(observed_sse_all).astype(np.float64, copy=False)
        observed_sae = np.concatenate(observed_sae_all).astype(np.float64, copy=False)
        observed_count = np.concatenate(observed_count_all).astype(np.float64, copy=False)
        row_shared = np.concatenate(row_shared_all).astype(np.bool_, copy=False)
        if not bool(row_shared.all()):
            raise RuntimeError('stage12 Weather diagnostic requires a row-shared input mask')
        if bool((observed_count <= 0).any()):
            raise RuntimeError('stage12 validation diagnostic found a sample without observed targets')

        payload = {
            'setting': setting,
            'checkpoint': self.args.stage12_checkpoint,
            'source_split': 'val',
            'chronological_loader': True,
            'uses_test_split': False,
            'updates_model_or_memory': False,
            'sample_count': int(sample_index.size),
            'validation_batches_processed': int(evaluated_batches),
            'sample_index_start': int(sample_index[0]),
            'sample_index_end': int(sample_index[-1]),
            'history_mask_width': int(self.args.seq_len),
            'history_mask_packed_sha256': hashlib.sha256(
                history_mask_packed.tobytes()
            ).hexdigest(),
            'row_shared_mask_all_samples': bool(row_shared.all()),
            'complete_target_element_count': int(complete_count.sum()),
            'observed_target_element_count': int(observed_count.sum()),
            'complete_target_mse': float(complete_sse.sum() / complete_count.sum()),
            'complete_target_mae': float(complete_sae.sum() / complete_count.sum()),
            'observed_target_mse': float(observed_sse.sum() / observed_count.sum()),
            'observed_target_mae': float(observed_sae.sum() / observed_count.sum()),
            'parameter_count': int(self.parameter_count),
            'parameter_bytes': int(self.parameter_bytes),
            'buffer_bytes': int(self.buffer_bytes),
            'peak_gpu_memory_mb': float(
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            ) if self.device.type == 'cuda' else None,
            'peak_gpu_reserved_mb': float(
                torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
            ) if self.device.type == 'cuda' else None,
        }
        metrics_path = self.args.stage12_metrics_path
        samples_path = self.args.stage12_samples_path
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        os.makedirs(os.path.dirname(samples_path), exist_ok=True)
        metrics_temp = metrics_path + '.tmp'
        with open(metrics_temp, 'w') as metrics_file:
            json.dump(payload, metrics_file, indent=2, sort_keys=True)
        os.replace(metrics_temp, metrics_path)
        samples_temp = samples_path + '.tmp.npz'
        np.savez_compressed(
            samples_temp,
            sample_index=sample_index,
            history_mask_packed=history_mask_packed,
            complete_sse=complete_sse,
            complete_sae=complete_sae,
            complete_count=complete_count,
            observed_sse=observed_sse,
            observed_sae=observed_sae,
            observed_count=observed_count,
        )
        os.replace(samples_temp, samples_path)
        print('STAGE12_VALIDATION ' + json.dumps(payload, sort_keys=True))
        return payload

    def stage22_validation_diagnostic(self, setting):
        """Cache validation risk and mask-response evidence for Stage 22.

        The model and prototype memory are frozen.  The original and one
        deterministic equal-count L=5 nested view use the same chronological
        validation samples; no test loader is constructed.
        """
        validation_data, _ = self._get_data(flag='val')
        validation_loader = DataLoader(
            validation_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=True,
        )
        f_dim = -1 if self.args.features == 'MS' else 0
        core_model = self._core_model()
        core_model.set_prototype_enabled(True)
        collect_internal_features = bool(
            getattr(self.args, 'stage22_internal_features', False)
        )
        collect_mcpr_attribution = bool(
            getattr(self.args, 'stage22_mcpr_attribution', False)
        )
        if collect_internal_features:
            if not bool(getattr(core_model, 'use_mrra', False)):
                raise RuntimeError('stage22 internal features require an MRRA model')
            core_model.set_collect_retrieval_diagnostics(True)
        if collect_mcpr_attribution and not bool(
            getattr(core_model, 'use_mcpr', False)
        ):
            raise RuntimeError('stage22 MCPR attribution requires an MCPR model')
        self.change_mode(0)
        self.model.eval()

        labels = ('original', 'nested', 'ensemble')
        arrays = {
            'sample_index': [],
            'history_mask_packed': [],
            'nested_mask_packed': [],
            'complete_count': [],
            'observed_count': [],
            'target_removed_count': [],
            'removed_count': [],
            'prediction_disagreement_mse': [],
            'target_abs_sum': [],
        }
        for label in labels:
            arrays[f'complete_sse_{label}'] = []
            arrays[f'complete_sae_{label}'] = []
            arrays[f'complete_signed_error_{label}'] = []
        arrays['observed_sse_original'] = []
        arrays['observed_sae_original'] = []
        if collect_mcpr_attribution:
            for key in (
                'complete_sse_backbone', 'complete_sae_backbone',
                'complete_signed_error_backbone', 'mcpr_delta_sse',
                'mcpr_delta_sae', 'mcpr_alignment_cross',
                'mcpr_residual_energy', 'mcpr_residual_abs_sum',
                'mcpr_residual_max_abs', 'backbone_abs_error_q50',
                'backbone_abs_error_q90', 'backbone_abs_error_q95',
                'backbone_abs_error_q99', 'backbone_abs_error_max',
                'candidate_abs_error_q50', 'candidate_abs_error_q90',
                'candidate_abs_error_q95', 'candidate_abs_error_q99',
                'candidate_abs_error_max',
            ):
                arrays[key] = []

        horizon_sse = None
        horizon_sae = None
        horizon_signed_error = None
        horizon_count = None
        variable_sse = None
        variable_sae = None
        variable_signed_error = None
        variable_count = None
        mcpr_axis_totals = {}
        row_shared_all = []
        evaluated_batches = 0

        with torch.no_grad():
            for batch_index, batch in enumerate(validation_loader):
                if self.args.max_eval_batches >= 0 and batch_index >= self.args.max_eval_batches:
                    break
                (
                    seq_x, seq_x_mask, seq_y, seq_y_mask, seq_true_x, seq_true_y,
                    max_idx, min_idx, max_value, min_value,
                ) = tuple(item.float().to(self.device) for item in batch)
                batch_size = int(seq_x.shape[0])
                start = batch_index * int(self.args.batch_size)
                sample_index = np.arange(start, start + batch_size, dtype=np.int64)

                prediction_original = self.model(
                    seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                )[:, -self.args.pred_len:, f_dim:]
                mcpr_residual = None
                backbone_prediction = None
                if collect_mcpr_attribution:
                    mcpr_outputs = core_model.get_last_mcpr_outputs()
                    if mcpr_outputs is None or 'residual' not in mcpr_outputs:
                        raise RuntimeError('stage22 MCPR attribution found no residual')
                    mcpr_residual = mcpr_outputs['residual'][
                        :, -self.args.pred_len:, f_dim:
                    ]
                    if mcpr_residual.shape != prediction_original.shape:
                        raise RuntimeError('stage22 MCPR residual shape mismatch')
                    backbone_prediction = prediction_original - mcpr_residual
                if collect_internal_features:
                    internal_features = self._stage22_mrra_internal_features(
                        core_model.get_last_mrra_outputs(),
                        core_model.get_last_retrieval_diagnostics(),
                        seq_x_mask,
                    )
                    for feature_name, feature_value in internal_features.items():
                        arrays.setdefault(f'feature_{feature_name}', []).append(
                            feature_value.detach().double().cpu().numpy()
                        )
                nested_mask = self._stage18_equal_count_observation_mask(
                    seq_x_mask,
                    sample_index,
                    float(self.args.stage22_nested_target_rate),
                    int(self.args.stage22_nested_block_length),
                    int(self.args.stage22_nested_seed),
                )
                nested_x = seq_true_x * nested_mask
                nested_extrema = self._temporal_extrema(nested_x)
                prediction_nested = self.model(
                    nested_x, nested_mask, *nested_extrema
                )[:, -self.args.pred_len:, f_dim:]
                predictions = {
                    'original': prediction_original,
                    'nested': prediction_nested,
                    'ensemble': 0.5 * (prediction_original + prediction_nested),
                }

                complete_target = seq_true_y[:, -self.args.pred_len:, f_dim:]
                observed_target = seq_y[:, -self.args.pred_len:, f_dim:]
                observed_mask = seq_y_mask[:, -self.args.pred_len:, f_dim:]
                reduce_dims = tuple(range(1, prediction_original.ndim))
                complete_count = torch.full(
                    (batch_size,),
                    float(complete_target[0].numel()),
                    dtype=complete_target.dtype,
                    device=complete_target.device,
                )
                observed_count = observed_mask.sum(dim=reduce_dims)
                if bool((observed_count <= 0).any()):
                    raise RuntimeError('stage22 found a sample without observed targets')

                if collect_mcpr_attribution:
                    backbone_error = backbone_prediction - complete_target
                    candidate_error = prediction_original - complete_target
                    alignment_cross = 2.0 * backbone_error * mcpr_residual
                    residual_energy = mcpr_residual.square()
                    delta_squared_error = (
                        candidate_error.square() - backbone_error.square()
                    )
                    delta_absolute_error = (
                        candidate_error.abs() - backbone_error.abs()
                    )
                    arrays['complete_sse_backbone'].append(
                        backbone_error.square().sum(dim=reduce_dims).cpu().numpy()
                    )
                    arrays['complete_sae_backbone'].append(
                        backbone_error.abs().sum(dim=reduce_dims).cpu().numpy()
                    )
                    arrays['complete_signed_error_backbone'].append(
                        backbone_error.sum(dim=reduce_dims).cpu().numpy()
                    )
                    arrays['mcpr_delta_sse'].append(
                        delta_squared_error.sum(dim=reduce_dims).cpu().numpy()
                    )
                    arrays['mcpr_delta_sae'].append(
                        delta_absolute_error.sum(dim=reduce_dims).cpu().numpy()
                    )
                    arrays['mcpr_alignment_cross'].append(
                        alignment_cross.sum(dim=reduce_dims).cpu().numpy()
                    )
                    arrays['mcpr_residual_energy'].append(
                        residual_energy.sum(dim=reduce_dims).cpu().numpy()
                    )
                    arrays['mcpr_residual_abs_sum'].append(
                        mcpr_residual.abs().sum(dim=reduce_dims).cpu().numpy()
                    )
                    arrays['mcpr_residual_max_abs'].append(
                        mcpr_residual.abs().flatten(1).amax(dim=1).cpu().numpy()
                    )
                    backbone_abs = backbone_error.abs().flatten(1)
                    candidate_abs = candidate_error.abs().flatten(1)
                    for suffix, quantile in (
                        ('q50', 0.50), ('q90', 0.90),
                        ('q95', 0.95), ('q99', 0.99),
                    ):
                        arrays[f'backbone_abs_error_{suffix}'].append(
                            torch.quantile(backbone_abs, quantile, dim=1).cpu().numpy()
                        )
                        arrays[f'candidate_abs_error_{suffix}'].append(
                            torch.quantile(candidate_abs, quantile, dim=1).cpu().numpy()
                        )
                    arrays['backbone_abs_error_max'].append(
                        backbone_abs.amax(dim=1).cpu().numpy()
                    )
                    arrays['candidate_abs_error_max'].append(
                        candidate_abs.amax(dim=1).cpu().numpy()
                    )

                    mcpr_axis_batch = {
                        'horizon_sse_backbone': backbone_error.square().sum(dim=(0, 2)),
                        'horizon_sae_backbone': backbone_error.abs().sum(dim=(0, 2)),
                        'horizon_delta_sse': delta_squared_error.sum(dim=(0, 2)),
                        'horizon_delta_sae': delta_absolute_error.sum(dim=(0, 2)),
                        'horizon_alignment_cross': alignment_cross.sum(dim=(0, 2)),
                        'horizon_residual_energy': residual_energy.sum(dim=(0, 2)),
                        'variable_sse_backbone': backbone_error.square().sum(dim=(0, 1)),
                        'variable_sae_backbone': backbone_error.abs().sum(dim=(0, 1)),
                        'variable_delta_sse': delta_squared_error.sum(dim=(0, 1)),
                        'variable_delta_sae': delta_absolute_error.sum(dim=(0, 1)),
                        'variable_alignment_cross': alignment_cross.sum(dim=(0, 1)),
                        'variable_residual_energy': residual_energy.sum(dim=(0, 1)),
                    }
                    for key, value in mcpr_axis_batch.items():
                        value_np = value.double().cpu().numpy()
                        if key not in mcpr_axis_totals:
                            mcpr_axis_totals[key] = np.zeros_like(value_np)
                        mcpr_axis_totals[key] += value_np

                for label, prediction in predictions.items():
                    error = prediction - complete_target
                    arrays[f'complete_sse_{label}'].append(
                        error.square().sum(dim=reduce_dims).cpu().numpy()
                    )
                    arrays[f'complete_sae_{label}'].append(
                        error.abs().sum(dim=reduce_dims).cpu().numpy()
                    )
                    arrays[f'complete_signed_error_{label}'].append(
                        error.sum(dim=reduce_dims).cpu().numpy()
                    )

                observed_error = prediction_original - observed_target
                arrays['observed_sse_original'].append(
                    (observed_error.square() * observed_mask).sum(dim=reduce_dims).cpu().numpy()
                )
                arrays['observed_sae_original'].append(
                    (observed_error.abs() * observed_mask).sum(dim=reduce_dims).cpu().numpy()
                )
                arrays['prediction_disagreement_mse'].append(
                    (prediction_nested - prediction_original).square()
                    .mean(dim=reduce_dims).cpu().numpy()
                )
                arrays['target_abs_sum'].append(
                    complete_target.abs().sum(dim=reduce_dims).cpu().numpy()
                )

                original_error = prediction_original - complete_target
                batch_horizon_sse = original_error.square().sum(dim=(0, 2)).double().cpu().numpy()
                batch_horizon_sae = original_error.abs().sum(dim=(0, 2)).double().cpu().numpy()
                batch_horizon_signed = original_error.sum(dim=(0, 2)).double().cpu().numpy()
                batch_horizon_count = np.full(
                    self.args.pred_len, batch_size * original_error.shape[2], dtype=np.float64
                )
                batch_variable_sse = original_error.square().sum(dim=(0, 1)).double().cpu().numpy()
                batch_variable_sae = original_error.abs().sum(dim=(0, 1)).double().cpu().numpy()
                batch_variable_signed = original_error.sum(dim=(0, 1)).double().cpu().numpy()
                batch_variable_count = np.full(
                    original_error.shape[2], batch_size * self.args.pred_len, dtype=np.float64
                )
                if horizon_sse is None:
                    horizon_sse = np.zeros_like(batch_horizon_sse)
                    horizon_sae = np.zeros_like(batch_horizon_sae)
                    horizon_signed_error = np.zeros_like(batch_horizon_signed)
                    horizon_count = np.zeros_like(batch_horizon_count)
                    variable_sse = np.zeros_like(batch_variable_sse)
                    variable_sae = np.zeros_like(batch_variable_sae)
                    variable_signed_error = np.zeros_like(batch_variable_signed)
                    variable_count = np.zeros_like(batch_variable_count)
                horizon_sse += batch_horizon_sse
                horizon_sae += batch_horizon_sae
                horizon_signed_error += batch_horizon_signed
                horizon_count += batch_horizon_count
                variable_sse += batch_variable_sse
                variable_sae += batch_variable_sae
                variable_signed_error += batch_variable_signed
                variable_count += batch_variable_count

                original_observed = seq_x_mask[:, :, 0] > 0.5
                nested_observed = nested_mask[:, :, 0] > 0.5
                row_shared = (
                    (seq_x_mask > 0.5) == (seq_x_mask[:, :, :1] > 0.5)
                ).reshape(batch_size, -1).all(dim=1)
                target_removed_count = torch.round(
                    float(self.args.stage22_nested_target_rate)
                    * original_observed.sum(dim=1).float()
                ).to(torch.int64)
                removed_count = (original_observed & (~nested_observed)).sum(dim=1)

                arrays['sample_index'].append(sample_index)
                arrays['history_mask_packed'].append(np.packbits(
                    original_observed.to(torch.uint8).cpu().numpy(), axis=1
                ))
                arrays['nested_mask_packed'].append(np.packbits(
                    nested_observed.to(torch.uint8).cpu().numpy(), axis=1
                ))
                arrays['complete_count'].append(complete_count.cpu().numpy())
                arrays['observed_count'].append(observed_count.cpu().numpy())
                arrays['target_removed_count'].append(target_removed_count.cpu().numpy())
                arrays['removed_count'].append(removed_count.cpu().numpy())
                row_shared_all.append(row_shared.cpu().numpy())
                evaluated_batches += 1

        if not arrays['sample_index']:
            raise RuntimeError('stage22 validation diagnostic processed zero batches')

        merged = {}
        integer_keys = {'sample_index', 'target_removed_count', 'removed_count'}
        packed_keys = {'history_mask_packed', 'nested_mask_packed'}
        for key, chunks in arrays.items():
            dtype = np.int64 if key in integer_keys else np.uint8 if key in packed_keys else np.float64
            merged[key] = np.concatenate(chunks).astype(dtype, copy=False)
        row_shared = np.concatenate(row_shared_all).astype(np.bool_, copy=False)
        if not bool(row_shared.all()):
            raise RuntimeError('stage22 Electricity diagnostic requires row-shared masks')
        if not np.array_equal(merged['removed_count'], merged['target_removed_count']):
            raise RuntimeError('stage22 equal-count mask failed its per-sample count contract')
        feature_keys = sorted(key for key in merged if key.startswith('feature_'))
        if collect_internal_features:
            if not feature_keys:
                raise RuntimeError('stage22 internal diagnostic cached no features')
            for key in feature_keys:
                if merged[key].shape != merged['sample_index'].shape:
                    raise RuntimeError(f'stage22 internal feature has wrong shape: {key}')
                if not bool(np.isfinite(merged[key]).all()):
                    raise RuntimeError(f'stage22 internal feature is non-finite: {key}')

        count_total = merged['complete_count'].sum()
        observed_count_total = merged['observed_count'].sum()
        payload = {
            'setting': setting,
            'checkpoint': self.args.stage22_checkpoint,
            'source_split': 'val',
            'chronological_loader': True,
            'uses_test_split': False,
            'updates_model_or_memory': False,
            'internal_features_cached': collect_internal_features,
            'internal_feature_keys': feature_keys,
            'mcpr_attribution_cached': collect_mcpr_attribution,
            'sample_count': int(merged['sample_index'].size),
            'validation_batches_processed': int(evaluated_batches),
            'history_mask_width': int(self.args.seq_len),
            'row_shared_mask_all_samples': bool(row_shared.all()),
            'nested_target_rate': float(self.args.stage22_nested_target_rate),
            'nested_block_length': int(self.args.stage22_nested_block_length),
            'nested_seed': int(self.args.stage22_nested_seed),
            'equal_removed_count_all_samples': True,
            'complete_target_mse_original': float(
                merged['complete_sse_original'].sum() / count_total
            ),
            'complete_target_mae_original': float(
                merged['complete_sae_original'].sum() / count_total
            ),
            'complete_target_mse_nested': float(
                merged['complete_sse_nested'].sum() / count_total
            ),
            'complete_target_mae_nested': float(
                merged['complete_sae_nested'].sum() / count_total
            ),
            'complete_target_mse_ensemble': float(
                merged['complete_sse_ensemble'].sum() / count_total
            ),
            'complete_target_mae_ensemble': float(
                merged['complete_sae_ensemble'].sum() / count_total
            ),
            'observed_target_mse_original': float(
                merged['observed_sse_original'].sum() / observed_count_total
            ),
            'observed_target_mae_original': float(
                merged['observed_sae_original'].sum() / observed_count_total
            ),
            'prediction_disagreement_mse_mean': float(
                merged['prediction_disagreement_mse'].mean()
            ),
            'prediction_exact_equal_sample_fraction': float(
                (merged['prediction_disagreement_mse'] == 0.0).mean()
            ),
            'parameter_count': int(self.parameter_count),
            'parameter_bytes': int(self.parameter_bytes),
            'buffer_bytes': int(self.buffer_bytes),
            'peak_gpu_memory_mb': float(
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            ) if self.device.type == 'cuda' else None,
            'peak_gpu_reserved_mb': float(
                torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
            ) if self.device.type == 'cuda' else None,
        }
        if collect_mcpr_attribution:
            payload.update({
                'complete_target_mse_backbone': float(
                    merged['complete_sse_backbone'].sum() / count_total
                ),
                'complete_target_mae_backbone': float(
                    merged['complete_sae_backbone'].sum() / count_total
                ),
                'mcpr_delta_mse': float(
                    merged['mcpr_delta_sse'].sum() / count_total
                ),
                'mcpr_delta_mae': float(
                    merged['mcpr_delta_sae'].sum() / count_total
                ),
                'mcpr_alignment_cross_mean': float(
                    merged['mcpr_alignment_cross'].sum() / count_total
                ),
                'mcpr_residual_energy_mean': float(
                    merged['mcpr_residual_energy'].sum() / count_total
                ),
                'mcpr_decomposition_max_abs_error': float(np.max(np.abs(
                    merged['mcpr_delta_sse']
                    - merged['mcpr_alignment_cross']
                    - merged['mcpr_residual_energy']
                ))),
            })
        metrics_path = self.args.stage22_metrics_path
        samples_path = self.args.stage22_samples_path
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        os.makedirs(os.path.dirname(samples_path), exist_ok=True)
        metrics_temp = metrics_path + '.tmp'
        with open(metrics_temp, 'w') as metrics_file:
            json.dump(payload, metrics_file, indent=2, sort_keys=True)
        os.replace(metrics_temp, metrics_path)
        samples_temp = samples_path + '.tmp.npz'
        np.savez_compressed(
            samples_temp,
            **merged,
            horizon_sse=horizon_sse,
            horizon_sae=horizon_sae,
            horizon_signed_error=horizon_signed_error,
            horizon_count=horizon_count,
            variable_sse=variable_sse,
            variable_sae=variable_sae,
            variable_signed_error=variable_signed_error,
            variable_count=variable_count,
            **mcpr_axis_totals,
        )
        os.replace(samples_temp, samples_path)
        print('STAGE22_VALIDATION ' + json.dumps(payload, sort_keys=True))
        return payload

    @staticmethod
    def _stage22_mrra_internal_features(mrra_outputs, retrieval, observation_mask):
        """Return label-free per-sample diagnostics from one MRRA forward."""
        if mrra_outputs is None or retrieval is None:
            raise RuntimeError('MRRA and retrieval diagnostics must both be available')
        required_mrra = (
            'hidden', 'context_residual', 'reconstruction',
            'reconstruction_correction',
        )
        if any(key not in mrra_outputs for key in required_mrra):
            raise RuntimeError('MRRA diagnostic outputs are incomplete')

        def moments(value):
            dims = tuple(range(1, value.ndim))
            return (
                value.abs().mean(dim=dims),
                value.square().mean(dim=dims).sqrt(),
                value.abs().amax(dim=dims),
            )

        hidden_abs, hidden_rms, hidden_max = moments(mrra_outputs['hidden'])
        context_abs, context_rms, context_max = moments(
            mrra_outputs['context_residual']
        )
        correction_abs, correction_rms, correction_max = moments(
            mrra_outputs['reconstruction_correction']
        )
        imputed_history = (
            mrra_outputs['reconstruction']
            - mrra_outputs['reconstruction_correction']
        )
        _, imputed_rms, _ = moments(imputed_history)

        missing = observation_mask[:, :, 0] <= 0.5
        observed_fraction = 1.0 - missing.float().mean(dim=1)
        transition_rate = (
            missing[:, 1:] != missing[:, :-1]
        ).float().mean(dim=1)
        block_starts = missing.clone()
        block_starts[:, 1:] &= ~missing[:, :-1]
        missing_block_count = block_starts.sum(dim=1).to(hidden_rms.dtype)
        current_run = torch.zeros_like(missing_block_count)
        longest_run = torch.zeros_like(missing_block_count)
        for time_index in range(missing.shape[1]):
            current_run = torch.where(
                missing[:, time_index], current_run + 1.0,
                torch.zeros_like(current_run),
            )
            longest_run = torch.maximum(longest_run, current_run)

        features = {
            'mrra_hidden_abs_mean': hidden_abs,
            'mrra_hidden_rms': hidden_rms,
            'mrra_hidden_max_abs': hidden_max,
            'mrra_context_abs_mean': context_abs,
            'mrra_context_rms': context_rms,
            'mrra_context_max_abs': context_max,
            'mrra_correction_abs_mean': correction_abs,
            'mrra_correction_rms': correction_rms,
            'mrra_correction_max_abs': correction_max,
            'mrra_correction_to_input_rms': correction_rms / imputed_rms.clamp_min(1e-6),
            'history_observed_fraction': observed_fraction,
            'history_missing_transition_rate': transition_rate,
            'history_missing_block_count': missing_block_count,
            'history_longest_missing_run': longest_run,
        }
        for key in (
            'top1_similarity', 'selected_similarity', 'retrieval_entropy',
            'top1_temporal_std', 'normalized_entropy_temporal_std',
            'retrieved_query_norm_ratio', 'retrieved_query_cosine_mean',
        ):
            if key not in retrieval:
                raise RuntimeError(f'missing retrieval diagnostic: {key}')
            features[f'retrieval_{key}'] = retrieval[key]
        return features

    @staticmethod
    def _stage14_nested_observation_mask(
        observation_mask,
        sample_indices,
        target_rate,
        block_length,
        seed,
    ):
        """Create deterministic causal block masks nested in observed history."""
        if observation_mask.ndim != 3:
            raise ValueError('stage14 observation mask must have shape [B,L,D]')
        batch_size, length, _ = observation_mask.shape
        if len(sample_indices) != batch_size:
            raise ValueError('stage14 sample index count must match the batch')
        if not 0.0 < float(target_rate) < 1.0:
            raise ValueError('stage14 target rate must lie in (0, 1)')
        if not 1 <= int(block_length) <= length:
            raise ValueError('stage14 block length must lie in [1, history length]')
        row_shared = (
            (observation_mask > 0.5) == (observation_mask[:, :, :1] > 0.5)
        ).reshape(batch_size, -1).all(dim=1)
        if not bool(row_shared.all()):
            raise ValueError('stage14 Weather diagnostic requires row-shared masks')

        start_probability = 1.0 - (1.0 - float(target_rate)) ** (
            1.0 / int(block_length)
        )
        starts = []
        for sample_index in sample_indices:
            generator = torch.Generator(device='cpu')
            mixed_seed = (
                int(seed) * 1000003 + int(sample_index) * 9176 + 104729
            ) % (2 ** 63 - 1)
            generator.manual_seed(mixed_seed)
            starts.append(
                torch.rand(length, generator=generator) < start_probability
            )
        starts = torch.stack(starts, dim=0).float().unsqueeze(1).to(
            observation_mask.device
        )
        if int(block_length) > 1:
            padded = torch.nn.functional.pad(
                starts, (int(block_length) - 1, 0), mode='constant', value=0.0
            )
            additional_missing = torch.nn.functional.max_pool1d(
                padded,
                kernel_size=int(block_length),
                stride=1,
                padding=0,
            )
        else:
            additional_missing = starts
        additional_observed = 1.0 - additional_missing.transpose(1, 2)
        return observation_mask * additional_observed

    def stage14_uncertainty_diagnostic(self, setting):
        """Cache frozen validation risk signals from nested history masks."""
        validation_data, _ = self._get_data(flag='val')
        validation_loader = DataLoader(
            validation_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=True,
        )
        f_dim = -1 if self.args.features == 'MS' else 0
        core_model = self._core_model()
        self.change_mode(0)
        self.model.eval()

        arrays = {
            'sample_index': [],
            'history_mask_packed': [],
            'nested_mask_packed': [],
            'complete_sse_on': [],
            'complete_sae_on': [],
            'complete_sse_off': [],
            'complete_sae_off': [],
            'complete_sse_nested': [],
            'complete_sae_nested': [],
            'complete_sse_ensemble': [],
            'complete_sae_ensemble': [],
            'complete_count': [],
            'mask_disagreement_mse': [],
            'prototype_disagreement_mse': [],
            'additional_missing_rate': [],
        }
        evaluated_batches = 0
        with torch.no_grad():
            for batch_index, batch in enumerate(validation_loader):
                if self.args.max_eval_batches >= 0 and batch_index >= self.args.max_eval_batches:
                    break
                (
                    seq_x, seq_x_mask, seq_y, seq_y_mask, seq_true_x, seq_true_y,
                    max_idx, min_idx, max_value, min_value,
                ) = tuple(item.float().to(self.device) for item in batch)
                del seq_y, seq_y_mask
                batch_size = int(seq_x.shape[0])
                start = batch_index * int(self.args.batch_size)
                sample_index = np.arange(start, start + batch_size, dtype=np.int64)

                core_model.set_prototype_enabled(True)
                prediction_on = self.model(
                    seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                )[:, -self.args.pred_len:, f_dim:]
                core_model.set_prototype_enabled(False)
                prediction_off = self.model(
                    seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                )[:, -self.args.pred_len:, f_dim:]
                core_model.set_prototype_enabled(True)

                nested_mask = self._stage14_nested_observation_mask(
                    seq_x_mask,
                    sample_index,
                    self.args.stage14_nested_target_rate,
                    self.args.stage14_nested_block_length,
                    self.args.stage14_nested_seed,
                )
                nested_x = seq_true_x * nested_mask
                nested_max_idx, nested_min_idx, nested_max_value, nested_min_value = (
                    self._temporal_extrema(nested_x)
                )
                prediction_nested = self.model(
                    nested_x,
                    nested_mask,
                    nested_max_idx,
                    nested_min_idx,
                    nested_max_value,
                    nested_min_value,
                )[:, -self.args.pred_len:, f_dim:]
                prediction_ensemble = 0.5 * (prediction_on + prediction_nested)

                target = seq_true_y[:, -self.args.pred_len:, f_dim:]
                reduce_dims = tuple(range(1, target.ndim))
                count = torch.full(
                    (batch_size,),
                    float(target[0].numel()),
                    device=target.device,
                    dtype=target.dtype,
                )
                for label, prediction in (
                    ('on', prediction_on),
                    ('off', prediction_off),
                    ('nested', prediction_nested),
                    ('ensemble', prediction_ensemble),
                ):
                    error = prediction - target
                    arrays[f'complete_sse_{label}'].append(
                        (error ** 2).sum(dim=reduce_dims).cpu().numpy()
                    )
                    arrays[f'complete_sae_{label}'].append(
                        error.abs().sum(dim=reduce_dims).cpu().numpy()
                    )
                arrays['mask_disagreement_mse'].append(
                    ((prediction_nested - prediction_on) ** 2)
                    .mean(dim=reduce_dims).cpu().numpy()
                )
                arrays['prototype_disagreement_mse'].append(
                    ((prediction_on - prediction_off) ** 2)
                    .mean(dim=reduce_dims).cpu().numpy()
                )
                original_observed = seq_x_mask[:, :, 0] > 0.5
                nested_observed = nested_mask[:, :, 0] > 0.5
                removed = original_observed & (~nested_observed)
                additional_rate = (
                    removed.sum(dim=1).float()
                    / original_observed.sum(dim=1).clamp_min(1).float()
                )
                arrays['sample_index'].append(sample_index)
                arrays['history_mask_packed'].append(
                    np.packbits(original_observed.to(torch.uint8).cpu().numpy(), axis=1)
                )
                arrays['nested_mask_packed'].append(
                    np.packbits(nested_observed.to(torch.uint8).cpu().numpy(), axis=1)
                )
                arrays['complete_count'].append(count.cpu().numpy())
                arrays['additional_missing_rate'].append(additional_rate.cpu().numpy())
                evaluated_batches += 1

        core_model.set_prototype_enabled(True)
        if not arrays['sample_index']:
            raise RuntimeError('stage14 uncertainty diagnostic processed zero batches')
        merged = {}
        for key, parts in arrays.items():
            dtype = np.int64 if key == 'sample_index' else (
                np.uint8 if key.endswith('_packed') else np.float64
            )
            merged[key] = np.concatenate(parts).astype(dtype, copy=False)
        count_total = merged['complete_count'].sum()
        payload = {
            'setting': setting,
            'checkpoint': self.args.stage14_checkpoint,
            'source_split': 'val',
            'chronological_loader': True,
            'uses_test_split': False,
            'updates_model_or_memory': False,
            'sample_count': int(merged['sample_index'].size),
            'validation_batches_processed': int(evaluated_batches),
            'nested_target_rate': float(self.args.stage14_nested_target_rate),
            'nested_block_length': int(self.args.stage14_nested_block_length),
            'nested_seed': int(self.args.stage14_nested_seed),
            'nested_start_probability': float(
                1.0 - (1.0 - float(self.args.stage14_nested_target_rate)) **
                (1.0 / int(self.args.stage14_nested_block_length))
            ),
            'realized_additional_missing_rate': float(
                merged['additional_missing_rate'].mean()
            ),
            'complete_target_mse_on': float(
                merged['complete_sse_on'].sum() / count_total
            ),
            'complete_target_mae_on': float(
                merged['complete_sae_on'].sum() / count_total
            ),
            'complete_target_mse_off': float(
                merged['complete_sse_off'].sum() / count_total
            ),
            'complete_target_mae_off': float(
                merged['complete_sae_off'].sum() / count_total
            ),
            'complete_target_mse_nested': float(
                merged['complete_sse_nested'].sum() / count_total
            ),
            'complete_target_mae_nested': float(
                merged['complete_sae_nested'].sum() / count_total
            ),
            'complete_target_mse_ensemble': float(
                merged['complete_sse_ensemble'].sum() / count_total
            ),
            'complete_target_mae_ensemble': float(
                merged['complete_sae_ensemble'].sum() / count_total
            ),
            'ensemble_weight_on': 0.5,
            'ensemble_weight_nested': 0.5,
            'mask_disagreement_mse_mean': float(
                merged['mask_disagreement_mse'].mean()
            ),
            'prototype_disagreement_mse_mean': float(
                merged['prototype_disagreement_mse'].mean()
            ),
            'parameter_count': int(self.parameter_count),
            'parameter_bytes': int(self.parameter_bytes),
            'buffer_bytes': int(self.buffer_bytes),
            'peak_gpu_memory_mb': float(
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            ) if self.device.type == 'cuda' else None,
            'peak_gpu_reserved_mb': float(
                torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
            ) if self.device.type == 'cuda' else None,
        }
        metrics_path = self.args.stage14_metrics_path
        samples_path = self.args.stage14_samples_path
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        os.makedirs(os.path.dirname(samples_path), exist_ok=True)
        metrics_temp = metrics_path + '.tmp'
        with open(metrics_temp, 'w') as metrics_file:
            json.dump(payload, metrics_file, indent=2, sort_keys=True)
        os.replace(metrics_temp, metrics_path)
        samples_temp = samples_path + '.tmp.npz'
        np.savez_compressed(samples_temp, **merged)
        os.replace(samples_temp, samples_path)
        return payload

    def stage15_nested_ensemble_test(self, setting):
        """Evaluate the fixed causal nested-mask equal ensemble on test."""
        _, test_loader = self._get_data(flag='test')
        f_dim = -1 if self.args.features == 'MS' else 0
        core_model = self._core_model()
        core_model.set_prototype_enabled(True)
        self.change_mode(0)
        self.model.eval()
        arrays = {
            'sample_index': [],
            'history_mask_packed': [],
            'nested_mask_packed': [],
            'complete_sse_on': [],
            'complete_sae_on': [],
            'complete_sse_nested': [],
            'complete_sae_nested': [],
            'complete_sse_ensemble': [],
            'complete_sae_ensemble': [],
            'complete_count': [],
            'mask_disagreement_mse': [],
            'additional_missing_rate': [],
        }
        evaluated_batches = 0
        started = time.monotonic()
        with torch.no_grad():
            for batch_index, batch in enumerate(test_loader):
                if self.args.max_eval_batches >= 0 and batch_index >= self.args.max_eval_batches:
                    break
                (
                    seq_x, seq_x_mask, seq_y, seq_y_mask, seq_true_x, seq_true_y,
                    max_idx, min_idx, max_value, min_value,
                ) = tuple(item.float().to(self.device) for item in batch)
                del seq_y, seq_y_mask
                batch_size = int(seq_x.shape[0])
                start = batch_index * int(self.args.batch_size)
                sample_index = np.arange(start, start + batch_size, dtype=np.int64)
                prediction_on = self.model(
                    seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                )[:, -self.args.pred_len:, f_dim:]
                nested_mask = self._stage14_nested_observation_mask(
                    seq_x_mask,
                    sample_index,
                    self.args.stage15_nested_target_rate,
                    self.args.stage15_nested_block_length,
                    self.args.stage15_nested_seed,
                )
                nested_x = seq_true_x * nested_mask
                nested_max_idx, nested_min_idx, nested_max_value, nested_min_value = (
                    self._temporal_extrema(nested_x)
                )
                prediction_nested = self.model(
                    nested_x,
                    nested_mask,
                    nested_max_idx,
                    nested_min_idx,
                    nested_max_value,
                    nested_min_value,
                )[:, -self.args.pred_len:, f_dim:]
                nested_weight = float(self.args.stage15_ensemble_weight)
                prediction_ensemble = (
                    (1.0 - nested_weight) * prediction_on
                    + nested_weight * prediction_nested
                )
                target = seq_true_y[:, -self.args.pred_len:, f_dim:]
                reduce_dims = tuple(range(1, target.ndim))
                count = torch.full(
                    (batch_size,),
                    float(target[0].numel()),
                    device=target.device,
                    dtype=target.dtype,
                )
                for label, prediction in (
                    ('on', prediction_on),
                    ('nested', prediction_nested),
                    ('ensemble', prediction_ensemble),
                ):
                    error = prediction - target
                    arrays[f'complete_sse_{label}'].append(
                        (error ** 2).sum(dim=reduce_dims).cpu().numpy()
                    )
                    arrays[f'complete_sae_{label}'].append(
                        error.abs().sum(dim=reduce_dims).cpu().numpy()
                    )
                arrays['mask_disagreement_mse'].append(
                    ((prediction_nested - prediction_on) ** 2)
                    .mean(dim=reduce_dims).cpu().numpy()
                )
                original_observed = seq_x_mask[:, :, 0] > 0.5
                nested_observed = nested_mask[:, :, 0] > 0.5
                removed = original_observed & (~nested_observed)
                additional_rate = (
                    removed.sum(dim=1).float()
                    / original_observed.sum(dim=1).clamp_min(1).float()
                )
                arrays['sample_index'].append(sample_index)
                arrays['history_mask_packed'].append(
                    np.packbits(original_observed.to(torch.uint8).cpu().numpy(), axis=1)
                )
                arrays['nested_mask_packed'].append(
                    np.packbits(nested_observed.to(torch.uint8).cpu().numpy(), axis=1)
                )
                arrays['complete_count'].append(count.cpu().numpy())
                arrays['additional_missing_rate'].append(additional_rate.cpu().numpy())
                evaluated_batches += 1
        elapsed = time.monotonic() - started
        if not arrays['sample_index']:
            raise RuntimeError('stage15 nested ensemble test processed zero batches')
        merged = {}
        for key, parts in arrays.items():
            dtype = np.int64 if key == 'sample_index' else (
                np.uint8 if key.endswith('_packed') else np.float64
            )
            merged[key] = np.concatenate(parts).astype(dtype, copy=False)
        count_total = merged['complete_count'].sum()
        payload = {
            'setting': setting,
            'checkpoint': self.args.stage15_checkpoint,
            'source_split': 'test',
            'chronological_loader': True,
            'uses_test_split': True,
            'updates_model_or_memory': False,
            'sample_count': int(merged['sample_index'].size),
            'test_batches_processed': int(evaluated_batches),
            'nested_target_rate': float(self.args.stage15_nested_target_rate),
            'nested_block_length': int(self.args.stage15_nested_block_length),
            'nested_seed': int(self.args.stage15_nested_seed),
            'ensemble_weight_on': 1.0 - float(self.args.stage15_ensemble_weight),
            'ensemble_weight_nested': float(self.args.stage15_ensemble_weight),
            'realized_additional_missing_rate': float(
                merged['additional_missing_rate'].mean()
            ),
            'complete_target_mse_on': float(
                merged['complete_sse_on'].sum() / count_total
            ),
            'complete_target_mae_on': float(
                merged['complete_sae_on'].sum() / count_total
            ),
            'complete_target_mse_nested': float(
                merged['complete_sse_nested'].sum() / count_total
            ),
            'complete_target_mae_nested': float(
                merged['complete_sae_nested'].sum() / count_total
            ),
            'complete_target_mse_ensemble': float(
                merged['complete_sse_ensemble'].sum() / count_total
            ),
            'complete_target_mae_ensemble': float(
                merged['complete_sae_ensemble'].sum() / count_total
            ),
            'mse': float(merged['complete_sse_ensemble'].sum() / count_total),
            'mae': float(merged['complete_sae_ensemble'].sum() / count_total),
            'mask_disagreement_mse_mean': float(
                merged['mask_disagreement_mse'].mean()
            ),
            'inference_forward_multiplier': 2.0,
            'evaluation_elapsed_seconds': float(elapsed),
            'parameter_count': int(self.parameter_count),
            'parameter_bytes': int(self.parameter_bytes),
            'buffer_bytes': int(self.buffer_bytes),
            'peak_gpu_memory_mb': float(
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            ) if self.device.type == 'cuda' else None,
            'peak_gpu_reserved_mb': float(
                torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
            ) if self.device.type == 'cuda' else None,
        }
        metrics_path = self.args.stage15_metrics_path
        samples_path = self.args.stage15_samples_path
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        os.makedirs(os.path.dirname(samples_path), exist_ok=True)
        metrics_temp = metrics_path + '.tmp'
        with open(metrics_temp, 'w') as metrics_file:
            json.dump(payload, metrics_file, indent=2, sort_keys=True)
        os.replace(metrics_temp, metrics_path)
        samples_temp = samples_path + '.tmp.npz'
        np.savez_compressed(samples_temp, **merged)
        os.replace(samples_temp, samples_path)
        return payload

    @staticmethod
    def _stage18_equal_count_observation_mask(
        observation_mask,
        sample_indices,
        target_rate,
        block_length,
        seed,
    ):
        """Remove the same rounded number of observed rows for every block scale."""
        if observation_mask.ndim != 3:
            raise ValueError('stage18 observation mask must have shape [B,L,D]')
        batch_size, length, dimensions = observation_mask.shape
        if len(sample_indices) != batch_size:
            raise ValueError('stage18 sample index count must match the batch')
        if not 0.0 < float(target_rate) < 1.0:
            raise ValueError('stage18 target rate must lie in (0, 1)')
        if not 1 <= int(block_length) <= length:
            raise ValueError('stage18 block length must lie in [1, history length]')
        row_shared = (
            (observation_mask > 0.5) == (observation_mask[:, :, :1] > 0.5)
        ).reshape(batch_size, -1).all(dim=1)
        if not bool(row_shared.all()):
            raise ValueError('equal-rate multiscale evaluation requires row-shared masks')

        original_rows = (
            observation_mask[:, :, 0] > 0.5
        ).detach().cpu().numpy()
        removed_rows = np.zeros((batch_size, length), dtype=np.bool_)
        modulus = 2 ** 63 - 1
        for batch_position, sample_index in enumerate(sample_indices):
            observed_positions = np.flatnonzero(original_rows[batch_position])
            target_count = int(np.rint(float(target_rate) * observed_positions.size))
            target_count = min(max(target_count, 0), int(observed_positions.size))
            if target_count == 0:
                continue
            mixed_seed = (
                int(seed) * 1000003
                + int(sample_index) * 9176
                + int(block_length) * 104729
                + 271828
            ) % modulus
            generator = np.random.default_rng(mixed_seed)
            starts = generator.permutation(length)
            selected = removed_rows[batch_position]
            selected_count = 0
            for start in starts:
                stop = min(length, int(start) + int(block_length))
                candidates = np.flatnonzero(
                    original_rows[batch_position, int(start):stop]
                    & (~selected[int(start):stop])
                ) + int(start)
                if candidates.size == 0:
                    continue
                remaining = target_count - selected_count
                chosen = candidates[:remaining]
                selected[chosen] = True
                selected_count += int(chosen.size)
                if selected_count == target_count:
                    break
            if selected_count < target_count:
                remaining_positions = observed_positions[~selected[observed_positions]]
                remaining_positions = generator.permutation(remaining_positions)
                needed = target_count - selected_count
                selected[remaining_positions[:needed]] = True
                selected_count += min(needed, int(remaining_positions.size))
            if selected_count != target_count:
                raise RuntimeError(
                    'stage18 failed to construct the exact requested removal count'
                )

        keep_rows = torch.from_numpy(~removed_rows).to(
            device=observation_mask.device
        ).unsqueeze(-1)
        keep_rows = keep_rows.expand(batch_size, length, dimensions)
        return observation_mask * keep_rows.to(observation_mask.dtype)

    def _equal_rate_multiscale_evaluation(self, setting, split_name, argument_prefix):
        """Evaluate exact-rate L1/L5/L20 views and frozen multiscale fusion."""
        if split_name not in ('val', 'test'):
            raise ValueError('equal-rate multiscale split must be val or test')
        if argument_prefix not in ('stage18', 'stage19', 'stage21'):
            raise ValueError('equal-rate multiscale argument prefix is unsupported')

        def argument(name):
            return getattr(self.args, f'{argument_prefix}_{name}')

        evaluation_data, _ = self._get_data(flag=split_name)
        evaluation_loader = DataLoader(
            evaluation_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=True,
        )
        f_dim = -1 if self.args.features == 'MS' else 0
        core_model = self._core_model()
        core_model.set_prototype_enabled(True)
        self.change_mode(0)
        self.model.eval()
        labels = (
            'on', 'l1', 'l5', 'l20',
            'ensemble_l1', 'ensemble_l5', 'ensemble_l20', 'multiscale',
        )
        arrays = {
            'sample_index': [],
            'history_mask_packed': [],
            'l1_mask_packed': [],
            'l5_mask_packed': [],
            'l20_mask_packed': [],
            'complete_count': [],
            'target_removed_count': [],
        }
        for label in labels:
            arrays[f'complete_sse_{label}'] = []
            arrays[f'complete_sae_{label}'] = []
        for block_length in (1, 5, 20):
            arrays[f'removed_count_l{block_length}'] = []
            arrays[f'additional_missing_rate_l{block_length}'] = []
            arrays[f'prediction_difference_mse_l{block_length}'] = []

        timing_keys = [
            'original_forward_seconds',
            *[
                f'l{block_length}_{component}_seconds'
                for block_length in (1, 5, 20)
                for component in ('mask', 'preprocess', 'forward', 'fusion')
            ],
            'multiscale_fusion_seconds',
        ]
        timings = {key: [] for key in timing_keys}
        timing_warmup = int(argument('timing_warmup_batches'))
        evaluated_batches = 0
        started = time.monotonic()

        def synchronize():
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)

        def timed_model_forward(model_args):
            synchronize()
            forward_started = time.perf_counter()
            prediction = self.model(*model_args)[:, -self.args.pred_len:, f_dim:]
            synchronize()
            return prediction, time.perf_counter() - forward_started

        with torch.no_grad():
            for batch_index, batch in enumerate(evaluation_loader):
                if self.args.max_eval_batches >= 0 and batch_index >= self.args.max_eval_batches:
                    break
                (
                    seq_x, seq_x_mask, seq_y, seq_y_mask, seq_true_x, seq_true_y,
                    max_idx, min_idx, max_value, min_value,
                ) = tuple(item.float().to(self.device) for item in batch)
                del seq_y, seq_y_mask
                batch_size = int(seq_x.shape[0])
                start = batch_index * int(self.args.batch_size)
                sample_index = np.arange(start, start + batch_size, dtype=np.int64)
                record_timing = batch_index >= timing_warmup

                prediction_on, original_forward_seconds = timed_model_forward((
                    seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                ))
                predictions = {'on': prediction_on}
                masks = {}
                route_timings = {}
                for block_length in (1, 5, 20):
                    synchronize()
                    mask_started = time.perf_counter()
                    nested_mask = self._stage18_equal_count_observation_mask(
                        seq_x_mask,
                        sample_index,
                        argument('target_rate'),
                        block_length,
                        argument('nested_seed'),
                    )
                    synchronize()
                    mask_seconds = time.perf_counter() - mask_started

                    synchronize()
                    preprocess_started = time.perf_counter()
                    nested_x = seq_true_x * nested_mask
                    nested_extrema = self._temporal_extrema(nested_x)
                    synchronize()
                    preprocess_seconds = time.perf_counter() - preprocess_started
                    prediction_nested, forward_seconds = timed_model_forward((
                        nested_x, nested_mask, *nested_extrema
                    ))
                    masks[block_length] = nested_mask
                    predictions[f'l{block_length}'] = prediction_nested
                    route_timings[block_length] = {
                        'mask': mask_seconds,
                        'preprocess': preprocess_seconds,
                        'forward': forward_seconds,
                    }

                for block_length in (1, 5, 20):
                    synchronize()
                    fusion_started = time.perf_counter()
                    predictions[f'ensemble_l{block_length}'] = (
                        0.5 * prediction_on + 0.5 * predictions[f'l{block_length}']
                    )
                    synchronize()
                    route_timings[block_length]['fusion'] = (
                        time.perf_counter() - fusion_started
                    )
                synchronize()
                multiscale_started = time.perf_counter()
                predictions['multiscale'] = (
                    float(argument('original_weight')) * prediction_on
                    + float(argument('l5_weight')) * predictions['l5']
                    + float(argument('l20_weight')) * predictions['l20']
                )
                synchronize()
                multiscale_fusion_seconds = (
                    time.perf_counter() - multiscale_started
                )

                target = seq_true_y[:, -self.args.pred_len:, f_dim:]
                reduce_dims = tuple(range(1, target.ndim))
                count = torch.full(
                    (batch_size,),
                    float(target[0].numel()),
                    device=target.device,
                    dtype=target.dtype,
                )
                for label in labels:
                    error = predictions[label] - target
                    arrays[f'complete_sse_{label}'].append(
                        (error ** 2).sum(dim=reduce_dims).cpu().numpy()
                    )
                    arrays[f'complete_sae_{label}'].append(
                        error.abs().sum(dim=reduce_dims).cpu().numpy()
                    )

                original_observed = seq_x_mask[:, :, 0] > 0.5
                target_removed_count = torch.round(
                    float(argument('target_rate'))
                    * original_observed.sum(dim=1).float()
                ).to(torch.int64)
                arrays['sample_index'].append(sample_index)
                arrays['history_mask_packed'].append(
                    np.packbits(original_observed.to(torch.uint8).cpu().numpy(), axis=1)
                )
                arrays['complete_count'].append(count.cpu().numpy())
                arrays['target_removed_count'].append(
                    target_removed_count.cpu().numpy()
                )
                for block_length in (1, 5, 20):
                    arrays[f'prediction_difference_mse_l{block_length}'].append(
                        (
                            (predictions[f'l{block_length}'] - prediction_on) ** 2
                        ).mean(dim=reduce_dims).cpu().numpy()
                    )
                    nested_observed = masks[block_length][:, :, 0] > 0.5
                    removed_count = (
                        original_observed & (~nested_observed)
                    ).sum(dim=1).to(torch.int64)
                    additional_rate = (
                        removed_count.float()
                        / original_observed.sum(dim=1).clamp_min(1).float()
                    )
                    arrays[f'l{block_length}_mask_packed'].append(
                        np.packbits(
                            nested_observed.to(torch.uint8).cpu().numpy(), axis=1
                        )
                    )
                    arrays[f'removed_count_l{block_length}'].append(
                        removed_count.cpu().numpy()
                    )
                    arrays[f'additional_missing_rate_l{block_length}'].append(
                        additional_rate.cpu().numpy()
                    )

                if record_timing:
                    timings['original_forward_seconds'].append(
                        original_forward_seconds
                    )
                    for block_length in (1, 5, 20):
                        for component in ('mask', 'preprocess', 'forward', 'fusion'):
                            timings[
                                f'l{block_length}_{component}_seconds'
                            ].append(route_timings[block_length][component])
                    timings['multiscale_fusion_seconds'].append(
                        multiscale_fusion_seconds
                    )
                evaluated_batches += 1

        elapsed = time.monotonic() - started
        if not arrays['sample_index']:
            raise RuntimeError(
                f'{argument_prefix} {split_name} processed zero batches'
            )
        merged = {}
        integer_keys = {
            'sample_index', 'target_removed_count',
            'removed_count_l1', 'removed_count_l5', 'removed_count_l20',
        }
        for key, parts in arrays.items():
            if key.endswith('_packed'):
                dtype = np.uint8
            elif key in integer_keys:
                dtype = np.int64
            else:
                dtype = np.float64
            merged[key] = np.concatenate(parts).astype(dtype, copy=False)
        if not timings['original_forward_seconds']:
            raise RuntimeError(
                f'{argument_prefix} timing warmup consumed every evaluated batch'
            )
        timing_arrays = {
            f'timing_{key}': np.asarray(values, dtype=np.float64)
            for key, values in timings.items()
        }

        def timing_summary(values):
            values = np.asarray(values, dtype=np.float64)
            return {
                'mean_ms': float(values.mean() * 1000.0),
                'std_ms': float(values.std(ddof=0) * 1000.0),
                'p50_ms': float(np.percentile(values, 50) * 1000.0),
                'p95_ms': float(np.percentile(values, 95) * 1000.0),
                'total_seconds': float(values.sum()),
            }

        timing = {
            key: timing_summary(values)
            for key, values in timings.items()
        }
        original_seconds = np.asarray(
            timings['original_forward_seconds'], dtype=np.float64
        )
        estimated_latency = {'original': original_seconds}
        for block_length in (1, 5, 20):
            estimated_latency[f'ensemble_l{block_length}'] = sum(
                np.asarray(
                    timings[f'l{block_length}_{component}_seconds'],
                    dtype=np.float64,
                )
                for component in ('mask', 'preprocess', 'forward', 'fusion')
            ) + original_seconds
        estimated_latency['multiscale'] = (
            original_seconds
            + sum(
                np.asarray(timings[f'l5_{component}_seconds'], dtype=np.float64)
                for component in ('mask', 'preprocess', 'forward')
            )
            + sum(
                np.asarray(timings[f'l20_{component}_seconds'], dtype=np.float64)
                for component in ('mask', 'preprocess', 'forward')
            )
            + np.asarray(
                timings['multiscale_fusion_seconds'], dtype=np.float64
            )
        )
        latency_summary = {
            key: timing_summary(values)
            for key, values in estimated_latency.items()
        }
        latency_summary['multiscale']['measured_multiplier_vs_original'] = float(
            estimated_latency['multiscale'].mean() / original_seconds.mean()
        )

        count_total = merged['complete_count'].sum()
        payload = {
            'setting': setting,
            'checkpoint': argument('checkpoint'),
            'source_split': split_name,
            'chronological_loader': True,
            'uses_test_split': split_name == 'test',
            'updates_model_or_memory': False,
            'sample_count': int(merged['sample_index'].size),
            (
                'validation_batches_processed'
                if split_name == 'val'
                else 'test_batches_processed'
            ): int(evaluated_batches),
            'target_additional_missing_rate': float(argument('target_rate')),
            'target_count_rounding': 'numpy_rint_ties_to_even',
            'block_lengths': [1, 5, 20],
            'nested_seed': int(argument('nested_seed')),
            'multiscale_weights': {
                'original': float(argument('original_weight')),
                'l5': float(argument('l5_weight')),
                'l20': float(argument('l20_weight')),
            },
            'timing_device_serial': True,
            'timing_fixed_forward_order': ['original', 'l1', 'l5', 'l20'],
            'timing_scope': (
                'batch latency excluding dataloader/host-to-device; nested routes '
                'include mask construction, extrema preprocessing and model forward'
            ),
            'timing_warmup_batches': timing_warmup,
            'timing_measured_batches': len(timings['original_forward_seconds']),
            'timing_batch_size': int(self.args.batch_size),
            'timing_components': timing,
            'estimated_serial_latency': latency_summary,
            'evaluation_elapsed_seconds': float(elapsed),
            'parameter_count': int(self.parameter_count),
            'parameter_bytes': int(self.parameter_bytes),
            'buffer_bytes': int(self.buffer_bytes),
            'peak_gpu_memory_mb': float(
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            ) if self.device.type == 'cuda' else None,
            'peak_gpu_reserved_mb': float(
                torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
            ) if self.device.type == 'cuda' else None,
        }
        for block_length in (1, 5, 20):
            payload[f'realized_additional_missing_rate_l{block_length}'] = float(
                merged[f'additional_missing_rate_l{block_length}'].mean()
            )
        for label in labels:
            payload[f'complete_target_mse_{label}'] = float(
                merged[f'complete_sse_{label}'].sum() / count_total
            )
            payload[f'complete_target_mae_{label}'] = float(
                merged[f'complete_sae_{label}'].sum() / count_total
            )
        for block_length in (1, 5, 20):
            prediction_difference = float(
                merged[f'prediction_difference_mse_l{block_length}'].mean()
            )
            payload[
                f'prediction_difference_mse_l{block_length}_mean'
            ] = prediction_difference
            payload[
                f'prediction_difference_to_original_mse_ratio_l{block_length}'
            ] = float(
                prediction_difference / payload['complete_target_mse_on']
            )
        payload['mse'] = payload['complete_target_mse_multiscale']
        payload['mae'] = payload['complete_target_mae_multiscale']

        metrics_path = argument('metrics_path')
        samples_path = argument('samples_path')
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        os.makedirs(os.path.dirname(samples_path), exist_ok=True)
        metrics_temp = metrics_path + '.tmp'
        with open(metrics_temp, 'w') as metrics_file:
            json.dump(payload, metrics_file, indent=2, sort_keys=True)
        os.replace(metrics_temp, metrics_path)
        samples_temp = samples_path + '.tmp.npz'
        np.savez_compressed(samples_temp, **merged, **timing_arrays)
        os.replace(samples_temp, samples_path)
        return payload

    def stage18_equal_rate_multiscale_validation(self, setting):
        """Evaluate the Stage-18 frozen multiscale protocol on validation."""
        return self._equal_rate_multiscale_evaluation(
            setting,
            split_name='val',
            argument_prefix='stage18',
        )

    def stage19_equal_rate_multiscale_test(self, setting):
        """Evaluate the frozen multiscale protocol on an external test split."""
        return self._equal_rate_multiscale_evaluation(
            setting,
            split_name='test',
            argument_prefix='stage19',
        )

    def stage21_equal_rate_multiscale_evaluation(self, setting, split_name):
        """Evaluate the frozen multiscale protocol on Electricity val or test."""
        if split_name not in ('val', 'test'):
            raise ValueError('stage21 split must be val or test')
        return self._equal_rate_multiscale_evaluation(
            setting,
            split_name=split_name,
            argument_prefix='stage21',
        )

    def stage22_24_batch_parallel_comparison(self, setting, split_name):
        """Compare mathematically identical serial and batch-parallel CNM-SE."""
        if split_name != 'val':
            raise ValueError('stage22.24 is frozen to validation only')
        parallel_mode = (
            'cuda_streams'
            if getattr(self, '_stage22_25_cuda_stream_mode', False)
            else 'batch_concat'
        )
        parallel_timing_key = (
            'parallel_stream_forward_seconds'
            if parallel_mode == 'cuda_streams'
            else 'parallel_concat_forward_split_seconds'
        )

        def argument(name):
            return getattr(self.args, f'stage21_{name}')

        evaluation_data, _ = self._get_data(flag=split_name)
        evaluation_loader = DataLoader(
            evaluation_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=True,
        )
        f_dim = -1 if self.args.features == 'MS' else 0
        core_model = self._core_model()
        core_model.set_prototype_enabled(True)
        self.change_mode(0)
        self.model.eval()

        arrays = {
            'sample_index': [],
            'history_mask_packed': [],
            'l5_mask_packed': [],
            'l20_mask_packed': [],
            'complete_count': [],
            'target_removed_count': [],
            'removed_count_l5': [],
            'removed_count_l20': [],
            'additional_missing_rate_l5': [],
            'additional_missing_rate_l20': [],
            'complete_sse_serial_multiscale': [],
            'complete_sae_serial_multiscale': [],
            'complete_sse_parallel_multiscale': [],
            'complete_sae_parallel_multiscale': [],
        }
        comparison_routes = ('original', 'l5', 'l20', 'multiscale')
        for route in comparison_routes:
            arrays[f'serial_parallel_mse_{route}'] = []
            arrays[f'serial_parallel_max_abs_{route}'] = []

        timing_keys = (
            'l5_mask_seconds', 'l5_preprocess_seconds',
            'l20_mask_seconds', 'l20_preprocess_seconds',
            'serial_forward_seconds', 'serial_fusion_seconds',
            parallel_timing_key, 'parallel_fusion_seconds',
        )
        timings = {key: [] for key in timing_keys}
        execution_order = []
        timing_warmup = int(argument('timing_warmup_batches'))
        evaluated_batches = 0
        started = time.monotonic()
        route_streams = None
        if parallel_mode == 'cuda_streams':
            if self.device.type != 'cuda':
                raise RuntimeError('stage22.25 CUDA streams require a CUDA device')
            route_streams = {
                route: torch.cuda.Stream(device=self.device)
                for route in ('original', 'l5', 'l20')
            }

        def synchronize():
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)

        def predict(model_args):
            return self.model(*model_args)[:, -self.args.pred_len:, f_dim:]

        def timed_serial(model_args_by_route):
            synchronize()
            forward_started = time.perf_counter()
            outputs = {
                route: predict(model_args_by_route[route])
                for route in ('original', 'l5', 'l20')
            }
            synchronize()
            return outputs, time.perf_counter() - forward_started

        def timed_parallel(model_args_by_route, batch_size):
            synchronize()
            forward_started = time.perf_counter()
            if parallel_mode == 'batch_concat':
                combined_args = tuple(
                    torch.cat([
                        model_args_by_route[route][argument_index]
                        for route in ('original', 'l5', 'l20')
                    ], dim=0)
                    for argument_index in range(6)
                )
                combined = predict(combined_args)
                original, l5, l20 = torch.split(combined, batch_size, dim=0)
                outputs = {
                    'original': original, 'l5': l5, 'l20': l20,
                }
            else:
                default_stream = torch.cuda.current_stream(self.device)
                outputs = {}
                for route in ('original', 'l5', 'l20'):
                    stream = route_streams[route]
                    stream.wait_stream(default_stream)
                    with torch.cuda.stream(stream):
                        outputs[route] = predict(model_args_by_route[route])
                for stream in route_streams.values():
                    default_stream.wait_stream(stream)
            synchronize()
            return outputs, time.perf_counter() - forward_started

        def timed_fusion(predictions):
            synchronize()
            fusion_started = time.perf_counter()
            fused = (
                float(argument('original_weight')) * predictions['original']
                + float(argument('l5_weight')) * predictions['l5']
                + float(argument('l20_weight')) * predictions['l20']
            )
            synchronize()
            return fused, time.perf_counter() - fusion_started

        with torch.no_grad():
            for batch_index, batch in enumerate(evaluation_loader):
                if self.args.max_eval_batches >= 0 and batch_index >= self.args.max_eval_batches:
                    break
                (
                    seq_x, seq_x_mask, seq_y, seq_y_mask, seq_true_x, seq_true_y,
                    max_idx, min_idx, max_value, min_value,
                ) = tuple(item.float().to(self.device) for item in batch)
                del seq_y, seq_y_mask
                batch_size = int(seq_x.shape[0])
                start = batch_index * int(self.args.batch_size)
                sample_index = np.arange(start, start + batch_size, dtype=np.int64)
                record_timing = batch_index >= timing_warmup

                masks = {}
                nested_inputs = {}
                common_timing = {}
                for block_length in (5, 20):
                    synchronize()
                    mask_started = time.perf_counter()
                    nested_mask = self._stage18_equal_count_observation_mask(
                        seq_x_mask,
                        sample_index,
                        argument('target_rate'),
                        block_length,
                        argument('nested_seed'),
                    )
                    synchronize()
                    common_timing[f'l{block_length}_mask_seconds'] = (
                        time.perf_counter() - mask_started
                    )
                    synchronize()
                    preprocess_started = time.perf_counter()
                    nested_x = seq_true_x * nested_mask
                    nested_extrema = self._temporal_extrema(nested_x)
                    synchronize()
                    common_timing[f'l{block_length}_preprocess_seconds'] = (
                        time.perf_counter() - preprocess_started
                    )
                    masks[block_length] = nested_mask
                    nested_inputs[block_length] = (
                        nested_x, nested_mask, *nested_extrema
                    )

                model_args_by_route = {
                    'original': (
                        seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                    ),
                    'l5': nested_inputs[5],
                    'l20': nested_inputs[20],
                }
                parallel_first = bool(batch_index % 2)
                if parallel_first:
                    parallel, parallel_seconds = timed_parallel(
                        model_args_by_route, batch_size
                    )
                    serial, serial_seconds = timed_serial(model_args_by_route)
                else:
                    serial, serial_seconds = timed_serial(model_args_by_route)
                    parallel, parallel_seconds = timed_parallel(
                        model_args_by_route, batch_size
                    )
                serial_fused, serial_fusion_seconds = timed_fusion(serial)
                parallel_fused, parallel_fusion_seconds = timed_fusion(parallel)
                serial['multiscale'] = serial_fused
                parallel['multiscale'] = parallel_fused

                target = seq_true_y[:, -self.args.pred_len:, f_dim:]
                reduce_dims = tuple(range(1, target.ndim))
                count = torch.full(
                    (batch_size,), float(target[0].numel()),
                    device=target.device, dtype=target.dtype,
                )
                for label, predictions in (
                    ('serial', serial_fused), ('parallel', parallel_fused),
                ):
                    error = predictions - target
                    arrays[f'complete_sse_{label}_multiscale'].append(
                        (error ** 2).sum(dim=reduce_dims).cpu().numpy()
                    )
                    arrays[f'complete_sae_{label}_multiscale'].append(
                        error.abs().sum(dim=reduce_dims).cpu().numpy()
                    )
                for route in comparison_routes:
                    difference = parallel[route] - serial[route]
                    arrays[f'serial_parallel_mse_{route}'].append(
                        (difference ** 2).mean(dim=reduce_dims).cpu().numpy()
                    )
                    arrays[f'serial_parallel_max_abs_{route}'].append(
                        difference.abs().amax(dim=reduce_dims).cpu().numpy()
                    )

                original_observed = seq_x_mask[:, :, 0] > 0.5
                target_removed_count = torch.round(
                    float(argument('target_rate'))
                    * original_observed.sum(dim=1).float()
                ).to(torch.int64)
                arrays['sample_index'].append(sample_index)
                arrays['history_mask_packed'].append(
                    np.packbits(original_observed.to(torch.uint8).cpu().numpy(), axis=1)
                )
                arrays['complete_count'].append(count.cpu().numpy())
                arrays['target_removed_count'].append(target_removed_count.cpu().numpy())
                for block_length in (5, 20):
                    nested_observed = masks[block_length][:, :, 0] > 0.5
                    removed_count = (
                        original_observed & (~nested_observed)
                    ).sum(dim=1).to(torch.int64)
                    arrays[f'l{block_length}_mask_packed'].append(
                        np.packbits(nested_observed.to(torch.uint8).cpu().numpy(), axis=1)
                    )
                    arrays[f'removed_count_l{block_length}'].append(
                        removed_count.cpu().numpy()
                    )
                    arrays[f'additional_missing_rate_l{block_length}'].append(
                        (
                            removed_count.float()
                            / original_observed.sum(dim=1).clamp_min(1).float()
                        ).cpu().numpy()
                    )

                if record_timing:
                    for key, value in common_timing.items():
                        timings[key].append(value)
                    timings['serial_forward_seconds'].append(serial_seconds)
                    timings['serial_fusion_seconds'].append(serial_fusion_seconds)
                    timings[parallel_timing_key].append(parallel_seconds)
                    timings['parallel_fusion_seconds'].append(
                        parallel_fusion_seconds
                    )
                    execution_order.append(1 if parallel_first else 0)
                evaluated_batches += 1

        elapsed = time.monotonic() - started
        if not arrays['sample_index']:
            raise RuntimeError('stage22.24 processed zero batches')
        if not timings['serial_forward_seconds']:
            raise RuntimeError('stage22.24 timing warmup consumed every batch')

        integer_keys = {
            'sample_index', 'target_removed_count',
            'removed_count_l5', 'removed_count_l20',
        }
        merged = {}
        for key, parts in arrays.items():
            if key.endswith('_packed'):
                dtype = np.uint8
            elif key in integer_keys:
                dtype = np.int64
            else:
                dtype = np.float64
            merged[key] = np.concatenate(parts).astype(dtype, copy=False)
        timing_arrays = {
            f'timing_{key}': np.asarray(values, dtype=np.float64)
            for key, values in timings.items()
        }
        timing_arrays['timing_execution_order_parallel_first'] = np.asarray(
            execution_order, dtype=np.int8
        )

        common_pipeline = sum(
            timing_arrays[f'timing_l{block}_{component}_seconds']
            for block in (5, 20) for component in ('mask', 'preprocess')
        )
        serial_pipeline = (
            common_pipeline + timing_arrays['timing_serial_forward_seconds']
            + timing_arrays['timing_serial_fusion_seconds']
        )
        parallel_pipeline = (
            common_pipeline
            + timing_arrays[f'timing_{parallel_timing_key}']
            + timing_arrays['timing_parallel_fusion_seconds']
        )
        timing_arrays['timing_serial_pipeline_seconds'] = serial_pipeline
        timing_arrays['timing_parallel_pipeline_seconds'] = parallel_pipeline
        paired_reduction = (serial_pipeline - parallel_pipeline) / serial_pipeline * 100.0
        timing_arrays['timing_paired_reduction_percent'] = paired_reduction

        def timing_summary(values):
            values = np.asarray(values, dtype=np.float64)
            return {
                'mean_ms': float(values.mean() * 1000.0),
                'std_ms': float(values.std(ddof=0) * 1000.0),
                'p50_ms': float(np.percentile(values, 50) * 1000.0),
                'p95_ms': float(np.percentile(values, 95) * 1000.0),
                'total_seconds': float(values.sum()),
            }

        rng = np.random.default_rng(20260820)
        bootstrap_indices = rng.integers(
            0, paired_reduction.size, size=(10000, paired_reduction.size)
        )
        bootstrap_means = paired_reduction[bootstrap_indices].mean(axis=1)
        reduction_ci = np.percentile(bootstrap_means, [2.5, 97.5])
        count_total = merged['complete_count'].sum()
        serial_mse = float(
            merged['complete_sse_serial_multiscale'].sum() / count_total
        )
        serial_mae = float(
            merged['complete_sae_serial_multiscale'].sum() / count_total
        )
        parallel_mse = float(
            merged['complete_sse_parallel_multiscale'].sum() / count_total
        )
        parallel_mae = float(
            merged['complete_sae_parallel_multiscale'].sum() / count_total
        )
        max_abs = {
            route: float(merged[f'serial_parallel_max_abs_{route}'].max())
            for route in comparison_routes
        }
        prediction_tolerance = 1e-4
        metric_tolerance = 1e-6
        equivalence_pass = (
            all(value <= prediction_tolerance for value in max_abs.values())
            and abs(serial_mse - parallel_mse) <= metric_tolerance
            and abs(serial_mae - parallel_mae) <= metric_tolerance
        )
        mean_reduction = float(paired_reduction.mean())
        speed_gate_pass = mean_reduction >= 20.0 and float(reduction_ci[0]) > 0.0

        payload = {
            'setting': setting,
            'checkpoint': argument('checkpoint'),
            'source_split': split_name,
            'chronological_loader': True,
            'uses_test_split': False,
            'updates_model_or_memory': False,
            'execution_optimization_only': True,
            'parallel_execution_mode': parallel_mode,
            'parallel_model_call_count': (
                3 if parallel_mode == 'cuda_streams' else 1
            ),
            'parallel_stream_count': (
                3 if parallel_mode == 'cuda_streams' else 1
            ),
            'sample_count': int(merged['sample_index'].size),
            'validation_batches_processed': int(evaluated_batches),
            'target_additional_missing_rate': float(argument('target_rate')),
            'target_count_rounding': 'numpy_rint_ties_to_even',
            'block_lengths': [5, 20],
            'nested_seed': int(argument('nested_seed')),
            'multiscale_weights': {
                'original': float(argument('original_weight')),
                'l5': float(argument('l5_weight')),
                'l20': float(argument('l20_weight')),
            },
            'timing_device_same_gpu': True,
            'timing_execution_order_alternates': True,
            'timing_order_encoding': {'serial_first': 0, 'parallel_first': 1},
            'timing_scope': (
                'paired batch latency excluding dataloader/host-to-device; both '
                'pipelines include shared L5/L20 mask construction and preprocessing'
            ),
            'timing_warmup_batches': timing_warmup,
            'timing_measured_batches': int(serial_pipeline.size),
            'timing_batch_size': int(self.args.batch_size),
            'parallel_effective_model_batch_size': (
                int(self.args.batch_size)
                if parallel_mode == 'cuda_streams'
                else int(self.args.batch_size) * 3
            ),
            'parallel_aggregate_inflight_batch_size': int(self.args.batch_size) * 3,
            'timing_components': {
                key: timing_summary(values) for key, values in timings.items()
            },
            'serial_pipeline_timing': timing_summary(serial_pipeline),
            'parallel_pipeline_timing': timing_summary(parallel_pipeline),
            'parallel_over_serial_mean_ratio': float(
                parallel_pipeline.mean() / serial_pipeline.mean()
            ),
            'paired_latency_reduction_percent_mean': mean_reduction,
            'paired_latency_reduction_percent_bootstrap_ci95': [
                float(reduction_ci[0]), float(reduction_ci[1])
            ],
            'bootstrap_resamples': 10000,
            'bootstrap_seed': 20260820,
            'complete_target_mse_serial_multiscale': serial_mse,
            'complete_target_mae_serial_multiscale': serial_mae,
            'complete_target_mse_parallel_multiscale': parallel_mse,
            'complete_target_mae_parallel_multiscale': parallel_mae,
            'maximum_absolute_prediction_difference': max_abs,
            'mean_squared_prediction_difference': {
                route: float(merged[f'serial_parallel_mse_{route}'].mean())
                for route in comparison_routes
            },
            'prediction_equivalence_tolerance': prediction_tolerance,
            'metric_equivalence_tolerance': metric_tolerance,
            'prediction_equivalence_pass': bool(equivalence_pass),
            'speed_gate_reduction_threshold_percent': 20.0,
            'speed_gate_pass': bool(speed_gate_pass),
            'evaluation_elapsed_seconds': float(elapsed),
            'parameter_count': int(self.parameter_count),
            'parameter_bytes': int(self.parameter_bytes),
            'buffer_bytes': int(self.buffer_bytes),
            'peak_gpu_memory_mb': float(
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            ) if self.device.type == 'cuda' else None,
            'peak_gpu_reserved_mb': float(
                torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
            ) if self.device.type == 'cuda' else None,
            'mse': parallel_mse,
            'mae': parallel_mae,
        }
        for block_length in (5, 20):
            payload[f'realized_additional_missing_rate_l{block_length}'] = float(
                merged[f'additional_missing_rate_l{block_length}'].mean()
            )

        metrics_path = argument('metrics_path')
        samples_path = argument('samples_path')
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        os.makedirs(os.path.dirname(samples_path), exist_ok=True)
        metrics_temp = metrics_path + '.tmp'
        with open(metrics_temp, 'w') as metrics_file:
            json.dump(payload, metrics_file, indent=2, sort_keys=True)
        os.replace(metrics_temp, metrics_path)
        samples_temp = samples_path + '.tmp.npz'
        np.savez_compressed(samples_temp, **merged, **timing_arrays)
        os.replace(samples_temp, samples_path)
        return payload

    def stage22_25_cuda_stream_comparison(self, setting, split_name):
        """Compare serial CNM-SE with three same-shape concurrent CUDA streams."""
        self._stage22_25_cuda_stream_mode = True
        try:
            return self.stage22_24_batch_parallel_comparison(setting, split_name)
        finally:
            del self._stage22_25_cuda_stream_mode

    def hcup_diagnostic(self, setting):
        """Evaluate a past-only counterfactual probe without test-time labels.

        For a current forecast origin ``t``, the probe shifts the S4M input by
        ``lag`` points and compares prototype-on/off errors only on its leading
        ``horizon <= lag`` predictions.  Those targets end no later than ``t``.
        Only observed entries from the missing-data stream enter the probe
        loss. Complete targets are retained solely for offline evaluation of
        the current forecast and never enter the gate.
        """
        split_name = str(self.args.hcup_split)
        data_set, _ = self._get_data(flag=split_name)
        data_loader = DataLoader(
            data_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=True,
        )
        lag = int(self.args.hcup_probe_lag)
        horizon = int(self.args.hcup_probe_horizon)
        if horizon <= 0 or horizon > lag or horizon > self.args.pred_len:
            raise ValueError(
                'HCUP requires 0 < probe_horizon <= probe_lag and pred_len'
            )
        f_dim = -1 if self.args.features == 'MS' else 0
        core_model = self._core_model()
        core_model.set_prototype_enabled(True)
        self.change_mode(0)
        self.model.eval()

        current_index_all = []
        current_mse_on_all = []
        current_mse_off_all = []
        current_mae_on_all = []
        current_mae_off_all = []
        probe_mse_on_all = []
        probe_mse_off_all = []
        probe_score_all = []
        probe_valid_all = []
        evaluated_batches = 0

        def to_device(batch):
            return tuple(item.float().to(self.device) for item in batch)

        try:
            with torch.no_grad():
                for batch_index, batch in enumerate(data_loader):
                    if self.args.max_eval_batches >= 0 and batch_index >= self.args.max_eval_batches:
                        break
                    (
                        seq_x, seq_x_mask, seq_y, seq_y_mask, seq_true_x, seq_true_y,
                        max_idx, min_idx, max_value, min_value,
                    ) = to_device(batch)
                    batch_size = int(seq_x.shape[0])
                    current_start = batch_index * int(self.args.batch_size)
                    current_index = np.arange(
                        current_start, current_start + batch_size, dtype=np.int64
                    )
                    current_target = seq_true_y[:, -self.args.pred_len:, f_dim:]

                    core_model.set_prototype_enabled(True)
                    current_on = self.model(
                        seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                    )[:, -self.args.pred_len:, f_dim:]
                    core_model.set_prototype_enabled(False)
                    current_off = self.model(
                        seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
                    )[:, -self.args.pred_len:, f_dim:]

                    reduce_dims = tuple(range(1, current_target.ndim))
                    current_mse_on = torch.mean(
                        (current_on - current_target) ** 2, dim=reduce_dims
                    )
                    current_mse_off = torch.mean(
                        (current_off - current_target) ** 2, dim=reduce_dims
                    )
                    current_mae_on = torch.mean(
                        torch.abs(current_on - current_target), dim=reduce_dims
                    )
                    current_mae_off = torch.mean(
                        torch.abs(current_off - current_target), dim=reduce_dims
                    )

                    valid_positions = np.flatnonzero(current_index >= lag)
                    batch_probe_on = np.full(batch_size, np.nan, dtype=np.float64)
                    batch_probe_off = np.full(batch_size, np.nan, dtype=np.float64)
                    batch_probe_score = np.full(batch_size, np.nan, dtype=np.float64)
                    batch_probe_valid = np.zeros(batch_size, dtype=np.bool_)
                    if valid_positions.size:
                        probe_samples = [
                            data_set[int(current_index[position] - lag)]
                            for position in valid_positions
                        ]
                        probe_batch = tuple(
                            torch.from_numpy(np.stack([sample[field] for sample in probe_samples]))
                            .float().to(self.device)
                            for field in range(len(probe_samples[0]))
                        )
                        (
                            probe_x, probe_x_mask, probe_y, probe_y_mask,
                            probe_true_x, probe_true_y, probe_max_idx, probe_min_idx,
                            probe_max_value, probe_min_value,
                        ) = probe_batch
                        del probe_true_x, probe_true_y
                        probe_target = probe_y[:, :horizon, f_dim:]
                        probe_target_mask = probe_y_mask[:, :horizon, f_dim:]
                        core_model.set_prototype_enabled(True)
                        probe_on = self.model(
                            probe_x, probe_x_mask, probe_max_idx, probe_min_idx,
                            probe_max_value, probe_min_value,
                        )[:, -self.args.pred_len:, f_dim:][:, :horizon]
                        core_model.set_prototype_enabled(False)
                        probe_off = self.model(
                            probe_x, probe_x_mask, probe_max_idx, probe_min_idx,
                            probe_max_value, probe_min_value,
                        )[:, -self.args.pred_len:, f_dim:][:, :horizon]
                        probe_reduce_dims = tuple(range(1, probe_target.ndim))
                        observed = probe_target_mask.sum(dim=probe_reduce_dims)
                        denominator = observed.clamp_min(1.0)
                        probe_mse_on = (
                            ((probe_on - probe_target) ** 2 * probe_target_mask)
                            .sum(dim=probe_reduce_dims) / denominator
                        )
                        probe_mse_off = (
                            ((probe_off - probe_target) ** 2 * probe_target_mask)
                            .sum(dim=probe_reduce_dims) / denominator
                        )
                        valid_probe = observed > 0
                        symmetric = (0.5 * (probe_mse_on + probe_mse_off)).clamp_min(1e-8)
                        probe_score = (probe_mse_off - probe_mse_on) / symmetric
                        positions = valid_positions[valid_probe.cpu().numpy()]
                        batch_probe_on[positions] = probe_mse_on[valid_probe].cpu().numpy()
                        batch_probe_off[positions] = probe_mse_off[valid_probe].cpu().numpy()
                        batch_probe_score[positions] = probe_score[valid_probe].cpu().numpy()
                        batch_probe_valid[positions] = True

                    current_index_all.append(current_index)
                    current_mse_on_all.append(current_mse_on.cpu().numpy())
                    current_mse_off_all.append(current_mse_off.cpu().numpy())
                    current_mae_on_all.append(current_mae_on.cpu().numpy())
                    current_mae_off_all.append(current_mae_off.cpu().numpy())
                    probe_mse_on_all.append(batch_probe_on)
                    probe_mse_off_all.append(batch_probe_off)
                    probe_score_all.append(batch_probe_score)
                    probe_valid_all.append(batch_probe_valid)
                    evaluated_batches += 1
        finally:
            core_model.set_prototype_enabled(True)

        if not current_mse_on_all:
            raise RuntimeError('HCUP diagnostic processed zero batches')
        current_index = np.concatenate(current_index_all)
        current_mse_on = np.concatenate(current_mse_on_all).astype(np.float64, copy=False)
        current_mse_off = np.concatenate(current_mse_off_all).astype(np.float64, copy=False)
        current_mae_on = np.concatenate(current_mae_on_all).astype(np.float64, copy=False)
        current_mae_off = np.concatenate(current_mae_off_all).astype(np.float64, copy=False)
        probe_mse_on = np.concatenate(probe_mse_on_all).astype(np.float64, copy=False)
        probe_mse_off = np.concatenate(probe_mse_off_all).astype(np.float64, copy=False)
        probe_score = np.concatenate(probe_score_all).astype(np.float64, copy=False)
        probe_valid = np.concatenate(probe_valid_all).astype(np.bool_, copy=False)
        threshold = float(self.args.hcup_threshold)
        gate = np.ones(current_index.size, dtype=np.bool_)
        gate[probe_valid] = probe_score[probe_valid] >= threshold
        routed_mse = np.where(gate, current_mse_on, current_mse_off)
        routed_mae = np.where(gate, current_mae_on, current_mae_off)
        current_utility = current_mse_off - current_mse_on
        oracle_mse = np.minimum(current_mse_on, current_mse_off)
        valid_correlation = self._safe_correlation(
            probe_score[probe_valid], current_utility[probe_valid]
        )
        payload = {
            'setting': setting,
            'checkpoint': self.args.hcup_checkpoint,
            'source_split': split_name,
            'sample_count': int(current_index.size),
            'test_batches_processed': int(evaluated_batches),
            'probe_lag': lag,
            'probe_horizon': horizon,
            'probe_uses_observed_history_only': True,
            'probe_uses_current_or_future_target': False,
            'probe_valid_count': int(probe_valid.sum()),
            'probe_valid_rate': float(probe_valid.mean()),
            'probe_score_mean': float(np.mean(probe_score[probe_valid])),
            'probe_score_std': float(np.std(probe_score[probe_valid])),
            'probe_current_utility_pearson': valid_correlation,
            'threshold': threshold,
            'invalid_probe_fallback': 'prototype_on',
            'gate_rate': float(gate.mean()),
            'abstention_rate': float(1.0 - gate.mean()),
            'prototype_on_mse': float(np.mean(current_mse_on)),
            'prototype_off_mse': float(np.mean(current_mse_off)),
            'routed_mse': float(np.mean(routed_mse)),
            'prototype_on_mae': float(np.mean(current_mae_on)),
            'prototype_off_mae': float(np.mean(current_mae_off)),
            'routed_mae': float(np.mean(routed_mae)),
            'route_relative_gain_vs_prototype_on': float(
                (np.mean(current_mse_on) - np.mean(routed_mse)) / np.mean(current_mse_on)
            ),
            'oracle_mse': float(np.mean(oracle_mse)),
            'oracle_relative_gain_vs_prototype_on': float(
                (np.mean(current_mse_on) - np.mean(oracle_mse)) / np.mean(current_mse_on)
            ),
            'parameter_count': int(self.parameter_count),
            'parameter_bytes': int(self.parameter_bytes),
            'buffer_bytes': int(self.buffer_bytes),
            'peak_gpu_memory_mb': float(
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            ) if self.device.type == 'cuda' else None,
            'peak_gpu_reserved_mb': float(
                torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
            ) if self.device.type == 'cuda' else None,
        }
        metrics_path = self.args.hcup_metrics_path
        samples_path = self.args.hcup_samples_path
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        os.makedirs(os.path.dirname(samples_path), exist_ok=True)
        metrics_temp = metrics_path + '.tmp'
        with open(metrics_temp, 'w') as metrics_file:
            json.dump(payload, metrics_file, indent=2, sort_keys=True)
        os.replace(metrics_temp, metrics_path)
        samples_temp = samples_path + '.tmp.npz'
        np.savez_compressed(
            samples_temp,
            current_index=current_index,
            current_mse_on=current_mse_on,
            current_mse_off=current_mse_off,
            current_mae_on=current_mae_on,
            current_mae_off=current_mae_off,
            current_utility=current_utility,
            probe_mse_on=probe_mse_on,
            probe_mse_off=probe_mse_off,
            probe_score=probe_score,
            probe_valid=probe_valid.astype(np.uint8),
            gate=gate.astype(np.uint8),
        )
        os.replace(samples_temp, samples_path)
        print('HCUP ' + json.dumps(payload, sort_keys=True))
        return payload

    def gate_h_diagnostic(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        f_dim = -1 if self.args.features == 'MS' else 0
        core_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        core_model.set_collect_retrieval_diagnostics(True)
        core_model.set_prototype_enabled(True)
        self.change_mode(0)
        self.model.eval()

        mse_on_all = []
        mse_off_all = []
        mae_on_all = []
        mae_off_all = []
        top1_all = []
        selected_all = []
        entropy_all = []
        candidate_counts = []
        evaluated_batches = 0

        try:
            with torch.no_grad():
                for i, batch in enumerate(test_loader):
                    if self.args.max_eval_batches >= 0 and i >= self.args.max_eval_batches:
                        break
                    (seq_x,seq_x_mask,seq_y,seq_y_mask,seq_true_x,seq_true_y,
                     max_idx,min_idx,max_value,min_value) = batch
                    (seq_x,seq_x_mask,seq_y,seq_y_mask,seq_true_x,seq_true_y,
                     max_idx,min_idx,max_value,min_value) = (
                        seq_x.float().to(self.device),seq_x_mask.float().to(self.device),
                        seq_y.float().to(self.device),seq_y_mask.float().to(self.device),
                        seq_true_x.float().to(self.device),seq_true_y.float().to(self.device),
                        max_idx.float().to(self.device),min_idx.float().to(self.device),
                        max_value.float().to(self.device),min_value.float().to(self.device)
                    )
                    target = seq_true_y[:, -self.args.pred_len:, f_dim:]

                    core_model.set_prototype_enabled(True)
                    outputs_on = self.model(seq_x,seq_x_mask,max_idx,min_idx,max_value,min_value)
                    outputs_on = outputs_on[:, -self.args.pred_len:, f_dim:]
                    retrieval = core_model.get_last_retrieval_diagnostics()
                    if retrieval is None:
                        raise RuntimeError('prototype-on forward did not expose retrieval diagnostics')

                    core_model.set_prototype_enabled(False)
                    outputs_off = self.model(seq_x,seq_x_mask,max_idx,min_idx,max_value,min_value)
                    outputs_off = outputs_off[:, -self.args.pred_len:, f_dim:]

                    reduce_dims = tuple(range(1, target.ndim))
                    mse_on_all.append(torch.mean((outputs_on - target) ** 2, dim=reduce_dims).cpu().numpy())
                    mse_off_all.append(torch.mean((outputs_off - target) ** 2, dim=reduce_dims).cpu().numpy())
                    mae_on_all.append(torch.mean(torch.abs(outputs_on - target), dim=reduce_dims).cpu().numpy())
                    mae_off_all.append(torch.mean(torch.abs(outputs_off - target), dim=reduce_dims).cpu().numpy())
                    top1_all.append(retrieval['top1_similarity'].cpu().numpy())
                    selected_all.append(retrieval['selected_similarity'].cpu().numpy())
                    entropy_all.append(retrieval['retrieval_entropy'].cpu().numpy())
                    candidate_counts.append(int(retrieval['candidate_count']))
                    evaluated_batches += 1
        finally:
            core_model.set_prototype_enabled(True)
            core_model.set_collect_retrieval_diagnostics(False)

        if not mse_on_all:
            raise RuntimeError('Gate H diagnostic processed zero test batches')

        mse_on = np.concatenate(mse_on_all).astype(np.float64, copy=False)
        mse_off = np.concatenate(mse_off_all).astype(np.float64, copy=False)
        mae_on = np.concatenate(mae_on_all).astype(np.float64, copy=False)
        mae_off = np.concatenate(mae_off_all).astype(np.float64, copy=False)
        top1 = np.concatenate(top1_all).astype(np.float64, copy=False)
        selected = np.concatenate(selected_all).astype(np.float64, copy=False)
        entropy = np.concatenate(entropy_all).astype(np.float64, copy=False)
        utility = mse_off - mse_on
        delta = float(self.args.gate_h_delta)
        harmful = mse_on > (mse_off + delta)
        oracle_mse = np.minimum(mse_on, mse_off)

        pearson = self._safe_correlation(top1, utility)
        spearman = self._safe_correlation(self._average_ranks(top1), self._average_ranks(utility))
        payload = {
            'setting': setting,
            'checkpoint': self.args.gate_h_checkpoint,
            'sample_count': int(mse_on.size),
            'test_batches_processed': int(evaluated_batches),
            'prototype_on_mse': float(np.mean(mse_on)),
            'prototype_off_mse': float(np.mean(mse_off)),
            'prototype_on_mae': float(np.mean(mae_on)),
            'prototype_off_mae': float(np.mean(mae_off)),
            'hrr_delta': delta,
            'hrr': float(np.mean(harmful)),
            'mru': float(np.mean(utility)),
            'nus': float(np.mean(np.maximum(-utility, 0.0))),
            'beneficial_retrieval_rate': float(np.mean(utility > delta)),
            'neutral_retrieval_rate': float(np.mean(np.abs(utility) <= delta)),
            'oracle_selector_mse': float(np.mean(oracle_mse)),
            'oracle_relative_gain_vs_prototype_on': float(
                (np.mean(mse_on) - np.mean(oracle_mse)) / np.mean(mse_on)
            ),
            'top1_similarity_mean': float(np.mean(top1)),
            'selected_similarity_mean': float(np.mean(selected)),
            'retrieval_entropy_mean': float(np.mean(entropy)),
            'similarity_utility_pearson': pearson,
            'similarity_utility_spearman': spearman,
            'prototype_candidate_counts': sorted(set(candidate_counts)),
            'parameter_count': int(self.parameter_count),
            'parameter_bytes': int(self.parameter_bytes),
            'buffer_bytes': int(self.buffer_bytes),
            'peak_gpu_memory_mb': float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)) if self.device.type == 'cuda' else None,
            'peak_gpu_reserved_mb': float(torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)) if self.device.type == 'cuda' else None,
        }

        metrics_path = self.args.gate_h_metrics_path
        samples_path = self.args.gate_h_samples_path
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        os.makedirs(os.path.dirname(samples_path), exist_ok=True)
        metrics_temp = metrics_path + '.tmp'
        with open(metrics_temp, 'w') as metrics_file:
            json.dump(payload, metrics_file, indent=2, sort_keys=True)
        os.replace(metrics_temp, metrics_path)
        samples_temp = samples_path + '.tmp.npz'
        np.savez_compressed(
            samples_temp,
            mse_on=mse_on,
            mse_off=mse_off,
            mae_on=mae_on,
            mae_off=mae_off,
            utility=utility,
            top1_similarity=top1,
            selected_similarity=selected,
            retrieval_entropy=entropy,
            harmful=harmful.astype(np.uint8),
        )
        os.replace(samples_temp, samples_path)
        print('GATE_H ' + json.dumps(payload, sort_keys=True))
        return payload
    
    def get_input(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        inputs = []
        for i, data_i in enumerate(test_loader):
            input = data_i[0].detach().cpu().numpy()
            inputs.append((input))
        folder_path = './results/' + setting + '/'
        np.save(folder_path + 'input.npy', inputs)
