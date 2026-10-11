#!/usr/bin/env python3
"""Frozen S4M path interventions. No optimizer, no training, no test loader."""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from stage22_38_common import ROOT, read, write_json, digest, namespace_values

FILLS = ('zero', 'prefix_mean', 'ffill')
PATHS = ('full', 'query', 'direct', 'retrieval')


def observed_fills(x, mask):
    """Use only observed history. Zero means zero in standardized coordinates."""
    observed = mask.bool()
    clean = torch.where(observed, x, torch.zeros_like(x))
    count = observed.to(x.dtype).cumsum(dim=1)
    prefix = clean.cumsum(dim=1) / count.clamp_min(1)
    positions = torch.arange(x.shape[1], device=x.device).view(1, -1, 1).expand_as(x)
    last = torch.where(observed, positions, torch.full_like(positions, -1)).cummax(dim=1).values
    previous = clean.gather(1, last.clamp_min(0))
    previous = torch.where(last >= 0, previous, torch.zeros_like(previous))
    return {name: torch.where(observed, x, value) for name, value in
            (('zero', torch.zeros_like(x)), ('prefix_mean', prefix), ('ffill', previous))}


def mix_paths(base, alt, kind):
    """Separate query+retrieval, raw direct path and pure retrieval path."""
    if kind == 'full': return alt['retrieved'], alt['query'], alt['raw']
    if kind == 'query': return alt['retrieved'], alt['query'], base['raw']
    if kind == 'direct': return base['retrieved'], base['query'], alt['raw']
    if kind == 'retrieval': return alt['retrieved'], base['query'], base['raw']
    raise ValueError(kind)


def tensor_hash(value):
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            t = x.detach().cpu().contiguous()
            h.update(str((t.dtype, tuple(t.shape))).encode()); h.update(t.numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str): h.update(str(k).encode()); visit(x[k])
        elif isinstance(x, (tuple, list)):
            h.update(type(x).__name__.encode())
            for item in x: visit(item)
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode()); h.update(x.tobytes())
        else: h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def memory_hash(model):
    memory = model.mem_net.memory
    return tensor_hash({k: getattr(memory, k, None) for k in ('memory', 'repre', 'num_dic', 'count', 'idx_arr')})


def errors(prediction, target):
    difference = prediction.double() - target.double()
    return difference.square().sum((1, 2)), difference.abs().sum((1, 2))


def vector_drift(current, base):
    cosine = 1 - F.cosine_similarity(current.double(), base.double(), dim=-1, eps=1e-12)
    relative = (current.double() - base.double()).norm(dim=-1) / base.double().norm(dim=-1).clamp_min(1e-12)
    return cosine.mean(1), relative.mean(1)


def input_groups(mask):
    # Traffic row masks apply jointly to all variables; assert that assumption.
    assert torch.equal(mask, mask[:, :, :1].expand_as(mask))
    missing = (mask[:, :, 0] < .5)
    running = torch.zeros(len(mask), dtype=torch.long, device=mask.device)
    maximum = running.clone()
    for i in range(mask.shape[1]):
        running = torch.where(missing[:, i], running + 1, torch.zeros_like(running))
        maximum = torch.maximum(maximum, running)
    return missing.double().mean(1), maximum, running


def run(config, out):
    sys.path[:0] = [str(ROOT / p) for p in ('baselines/S4M/s4m', 'baselines/S4M/s4m/model',
        'baselines/S4M/s4m/model/S4/s4', 'baselines/S4M/s4m/model/S4/s4/extensions/kernels')]
    from model.S4M import Model
    from data_provider.data_loader import Dataset_Custom4
    from torch.utils.data import DataLoader
    from utils.paired_eval_rng import paired_eval_rng

    official = config['official_args']
    args = SimpleNamespace(**official)
    device = torch.device('cuda:0')
    args.device = device; args.use_gpu = True; args.gpu = 0
    torch.set_num_threads(2)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    assert torch.cuda.is_available()
    environment = dict(python=sys.version, pytorch=torch.__version__, cuda_runtime=torch.version.cuda,
        conda_environment='s4m_missing_forecasting', cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu_name=torch.cuda.get_device_name(device))
    print('ENVIRONMENT '+json.dumps(environment), flush=True)
    assert args.use_mipp is False
    model = Model(args).float()
    checkpoint = torch.load(config['source']['checkpoint'], map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    memory = model.mem_net.memory
    for key in ('memory', 'repre', 'num_dic'): setattr(memory, key, checkpoint[key])
    del checkpoint
    model = model.to(device)
    memory.repre = memory.repre.to(device)
    model.eval(); model.is_training = 0; model.mem_net.is_training = 0
    model.set_prototype_enabled(True)
    assert not any(getattr(model, key, False) for key in ('use_mrra', 'use_mcpr'))
    parameter_count = sum(p.numel() for p in model.parameters())
    dataset = Dataset_Custom4(root_path=args.root_path, args=args, flag='val',
        size=[args.seq_len, args.label_len, args.pred_len], features=args.features,
        data_path=args.data_path, target=args.target, timeenc=1 if args.embed == 'timeF' else 0,
        freq=args.freq, mask=args.mask)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
        drop_last=True, num_workers=args.num_workers)
    reference = np.load(config['source']['cache'], allow_pickle=False)
    expected_samples = 48 if config['experiment']['smoke_test'] else 1648
    assert len(loader) * args.batch_size == 1648
    scalar_cache = {k: reference[k].copy() for k in ('sample_index', 'complete_count',
        'complete_sse_on', 'complete_sae_on', 'history_mask_packed')}
    reference.close()
    checks = dict(observed_values_preserved=True, finite=True, repeat_identical=True,
                  pre_imputation_missing_changes_ignored=True, recomposition_identical=True,
                  historical_baseline_paired=True, memory_immutable=True, weights_immutable=True)
    captured = {}; alternative = None
    original_fuse = model.mem_net._fuse_context
    def before_memory(module, positional, keyword):
        if alternative is None: return None
        return (alternative, *positional[1:]), keyword
    def capture_fuse(retrieved, query, raw_history, batch, length, hidden):
        captured.update(retrieved=retrieved.reshape(batch, length, hidden).detach(),
                        query=query.reshape(batch, length, hidden).detach(), raw=raw_history.detach())
        return original_fuse(retrieved, query, raw_history, batch, length, hidden)
    handle = model.mem_net.register_forward_pre_hook(before_memory, with_kwargs=True)
    model.mem_net._fuse_context = capture_fuse
    records = []
    routes = ('original',) + tuple(f'{p}_{f}' for f in FILLS for p in PATHS)
    arrays = {k: [] for k in ('sample_index', 'complete_count', 'missing_rate', 'longest_gap', 'trailing_gap')}
    fields = ('sse', 'sae', 'prediction_difference_mse', 'query_cosine_drift',
              'query_relative_drift', 'retrieval_cosine_drift', 'retrieval_relative_drift',
              'input_difference_mse', 'sse_h1', 'sse_h2', 'sse_h3', 'sse_h4')
    for route in routes:
        for field in fields: arrays[f'{route}__{field}'] = []
    warmup_state_changes = []
    freeze_hash = freeze_memory = None
    max_cache_difference = max_repeat = 0.
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if config['diagnostic']['max_eval_batches'] > 0 and batch_index >= config['diagnostic']['max_eval_batches']: break
                # Never transfer or inspect seq_true_x: hidden historical truth is not an input.
                x, mask = [batch[i].float().to(device) for i in (0, 1)]
                target = batch[5].float().to(device)[:, -args.pred_len:, :]
                extrema = [batch[i].float().to(device) for i in (6, 7, 8, 9)]
                model_args = (x, mask, *extrema)
                if batch_index == 0:
                    pre_state = {k: tensor_hash(v) for k, v in model.state_dict().items()}
                    pre_memory = memory_hash(model)
                    with paired_eval_rng(True, args.seed, 0, device): model(*model_args)
                    after_state = {k: tensor_hash(v) for k, v in model.state_dict().items()}
                    warmup_state_changes = [k for k in pre_state if pre_state[k] != after_state[k]]
                    assert all(k.endswith('.C') or k.endswith('.l_kernel') for k in warmup_state_changes), 'CODE_ERROR non-kernel warmup mutation'
                    assert pre_memory == memory_hash(model), 'CODE_ERROR memory warmup mutation'
                    freeze_hash = tensor_hash(model.state_dict()); freeze_memory = memory_hash(model)
                    print(f'FROZEN_MODEL warmup_changes={warmup_state_changes} prototypes={len(memory.repre)} parameters={parameter_count}', flush=True)
                alternative = None
                with paired_eval_rng(True, args.seed, batch_index, device):
                    base_prediction = model(*model_args)[:, -args.pred_len:, :]
                base = dict(captured)
                assert torch.equal(base['raw'][mask.bool()], x[mask.bool()])
                with paired_eval_rng(True, args.seed, batch_index, device):
                    repeated = model(*model_args)[:, -args.pred_len:, :]
                diff = float((repeated - base_prediction).abs().max())
                max_repeat = max(max_repeat, diff)
                assert diff == 0., 'CODE_ERROR identity repeat mismatch'
                # Negative control: Model.forward discards values at masked seq_x locations.
                changed_x = torch.where(mask.bool(), x, x.new_full((), 7.))
                with paired_eval_rng(True, args.seed, batch_index, device):
                    negative = model(changed_x, mask, *extrema)[:, -args.pred_len:, :]
                assert torch.equal(negative, base_prediction), 'CODE_ERROR wrong intervention entry'
                projected_mask = model.mask_trans(mask)
                B, L, H = base['query'].shape
                def forecast(r, u, z):
                    c = original_fuse(r.reshape(-1, H), u.reshape(-1, H), z, B, L, H)
                    with paired_eval_rng(True, args.seed, batch_index, device):
                        return model._forecast_from_context(c, projected_mask, observation_mask=mask)[:, -args.pred_len:, :]
                recomposed = forecast(base['retrieved'], base['query'], base['raw'])
                assert torch.equal(recomposed, base_prediction), 'CODE_ERROR context recomposition mismatch'
                count = target.shape[1] * target.shape[2]
                index = np.arange(batch_index * B, (batch_index + 1) * B)
                sse, sae = errors(base_prediction, target)
                packed = np.packbits(mask[:, :, 0].byte().cpu().numpy(), axis=1)
                np.testing.assert_array_equal(packed, scalar_cache['history_mask_packed'][index])
                np.testing.assert_array_equal(index, scalar_cache['sample_index'][index])
                np.testing.assert_array_equal(np.full(B, count), scalar_cache['complete_count'][index])
                for current, key in ((sse, 'complete_sse_on'), (sae, 'complete_sae_on')):
                    error = float(np.max(np.abs(current.cpu().numpy() - scalar_cache[key][index]) / count))
                    max_cache_difference = max(max_cache_difference, error)
                    assert error <= config['diagnostic']['baseline_metric_atol'], 'CODE_ERROR historical baseline mismatch'
                missing_rate, longest, trailing = input_groups(mask)
                for key, value in (('sample_index', index), ('complete_count', np.full(B, count)),
                                   ('missing_rate', missing_rate), ('longest_gap', longest), ('trailing_gap', trailing)):
                    arrays[key].append(value.cpu().numpy() if torch.is_tensor(value) else value)
                def record(route, prediction, values):
                    r, u, z = values
                    assert torch.equal(z[mask.bool()], x[mask.bool()]), 'CODE_ERROR observed values changed'
                    assert all(bool(torch.isfinite(v).all()) for v in (prediction, r, u, z)), 'NAN_OR_DIVERGENCE'
                    a, b = errors(prediction, target)
                    qc, qr = vector_drift(u, base['query']); rc, rr = vector_drift(r, base['retrieved'])
                    values_dict = dict(sse=a, sae=b,
                        prediction_difference_mse=(prediction.double()-base_prediction.double()).square().mean((1, 2)),
                        query_cosine_drift=qc, query_relative_drift=qr, retrieval_cosine_drift=rc,
                        retrieval_relative_drift=rr, input_difference_mse=(z.double()-base['raw'].double()).square().mean((1, 2)))
                    for i in range(4):
                        lo, hi = i * 24, (i + 1) * 24
                        values_dict[f'sse_h{i+1}'] = (prediction[:, lo:hi].double()-target[:, lo:hi].double()).square().sum((1, 2))
                    for field, value in values_dict.items(): arrays[f'{route}__{field}'].append(value.cpu().numpy())
                record('original', base_prediction, (base['retrieved'], base['query'], base['raw']))
                fills = observed_fills(x, mask)
                for name, fill in fills.items():
                    alternative = fill
                    with paired_eval_rng(True, args.seed, batch_index, device):
                        full = model(*model_args)[:, -args.pred_len:, :]
                    alt = dict(captured)
                    alternative = None
                    for kind in PATHS:
                        values = mix_paths(base, alt, kind)
                        prediction = full if kind == 'full' else forecast(*values)
                        record(f'{kind}_{name}', prediction, values)
                print(f'DIAGNOSTIC batch={batch_index+1}/{min(len(loader),config["diagnostic"]["max_eval_batches"]) if config["diagnostic"]["max_eval_batches"]>0 else len(loader)} windows={(batch_index+1)*B}', flush=True)
        assert tensor_hash(model.state_dict()) == freeze_hash, 'CODE_ERROR frozen weights mutated'
        assert memory_hash(model) == freeze_memory, 'CODE_ERROR frozen memory mutated'
    finally:
        handle.remove(); model.mem_net._fuse_context = original_fuse
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    assert len(arrays['sample_index']) == expected_samples
    tmp = out / 'samples.tmp.npz'
    np.savez_compressed(tmp, **arrays); tmp.replace(out / 'samples.npz')
    summaries = {}
    for route in routes:
        summaries[route] = dict(mse=float(arrays[f'{route}__sse'].sum()/arrays['complete_count'].sum()),
            mae=float(arrays[f'{route}__sae'].sum()/arrays['complete_count'].sum()),
            **{field: float(arrays[f'{route}__{field}'].mean()) for field in fields if field not in ('sse', 'sae')})
    for route, value in summaries.items():
        value['mse_gain_percent'] = (summaries['original']['mse'] - value['mse']) / summaries['original']['mse'] * 100
        value['mae_gain_percent'] = (summaries['original']['mae'] - value['mae']) / summaries['original']['mae'] * 100
    groups = {}
    for feature in ('longest_gap', 'trailing_gap'):
        groups[feature] = {}
        for name, chosen in (('0', arrays[feature] == 0), ('1_to_5', (arrays[feature]>=1)&(arrays[feature]<=5)),
            ('6_to_20', (arrays[feature]>=6)&(arrays[feature]<=20)), ('above_20', arrays[feature]>20)):
            if not chosen.any(): continue
            groups[feature][name] = dict(sample_count=int(chosen.sum()),
                routes={r: dict(mse=float(arrays[f'{r}__sse'][chosen].sum()/arrays['complete_count'][chosen].sum()),
                                mae=float(arrays[f'{r}__sae'][chosen].sum()/arrays['complete_count'][chosen].sum())) for r in routes})
    elapsed = time.monotonic() - started
    metrics = dict(stage='22.38', evaluation_split='validation', test_split_accessed=False,
        uses_test_split=False, updates_model_or_memory=False, scope='frozen_checkpoint_intervention_not_new_model',
        sample_count=expected_samples, mse=summaries['original']['mse'], mae=summaries['original']['mae'],
        rmse=summaries['original']['mse']**.5, parameter_count=parameter_count,
        checkpoint=config['source']['checkpoint'], source_checkpoint_sha256=digest(config['source']['checkpoint']),
        checks=checks, baseline_max_window_metric_difference=max_cache_difference,
        repeat_max_prediction_difference=max_repeat, warmup_state_changes=warmup_state_changes,
        prototype_count=int(len(memory.repre)), frozen_weights_sha256=freeze_hash,
        frozen_memory_sha256=freeze_memory, routes=summaries, groups=groups,
        peak_gpu_reserved_mb=torch.cuda.max_memory_reserved(device)/1024**2,
        peak_gpu_allocated_mb=torch.cuda.max_memory_allocated(device)/1024**2,
        elapsed_seconds=elapsed, environment=environment, artifact_status='PASS', samples_sha256=digest(out/'samples.npz'))
    write_json(out/'metrics.json', metrics)
    print('DIAGNOSTIC_COMPLETE '+json.dumps({k:metrics[k] for k in ('sample_count','mse','mae','peak_gpu_reserved_mb','elapsed_seconds')}), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--run-dir', required=True)
    a = p.parse_args()
    from pathlib import Path
    run(yaml.safe_load(Path(a.config).read_text()), Path(a.run_dir))


if __name__ == '__main__': main()
