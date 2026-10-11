#!/usr/bin/env python3
"""Read-only CPU analysis of frozen validation caches, not a model experiment."""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import subprocess
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/STAGE22_42_DIAGNOSTIC_ZH.md'
MANIFEST = ROOT / 'runtime/stage22_42_diagnostic/source_manifest.json'
SOURCE_REPORT = ROOT / 'reports/stage22_41_distance_fix.json'
IDENTITY = ('sample_index', 'complete_count', 'history_mask_packed', 'l1_mask_packed',
            'l5_mask_packed', 'l20_mask_packed', 'target_removed_count',
            'removed_count_l1', 'removed_count_l5', 'removed_count_l20')
ROUTES = ('on', 'multiscale')
CODE_FILES = ('scripts/stage22_42_diagnostic.py', 'scripts/test_stage22_42_diagnostic.py',
              'scripts/stage22_42_bank_audit.py', 'scripts/test_stage22_42_bank_audit.py',
              'docs/STAGE22_42_DIAGNOSTIC_ZH.md')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(value)
    os.replace(temporary, path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def inside_run(value):
    path = Path(value).resolve()
    assert path.is_relative_to(ROOT / 'runs'), 'DATA_ERROR source outside trusted project runs'
    assert '__smoke__' not in path.name, 'DATA_ERROR smoke source is not formal'
    return path


def prepare():
    r = read(SOURCE_REPORT)
    assert r['completed'] and r['queue_state'] == 'SUCCESS'
    assert [x['seed'] for x in r['records']] == [2027, 2028, 2029]
    hashes = {str(SOURCE_REPORT): digest(SOURCE_REPORT), str(PROTOCOL): digest(PROTOCOL)}
    records = []
    for row in r['records']:
        pair = {'seed': row['seed']}
        for family, key in (('original', 'baseline_eval'), ('fixed', 'fixed_eval')):
            ev = inside_run(row['comparison'][key])
            cfg = yaml.safe_load((ev / 'resolved_config.yaml').read_text())
            m = read(ev / 'metrics.json')
            assert m['source_split'] == 'val' and m['uses_test_split'] is False
            assert m['updates_model_or_memory'] is False and all(m['artifact_checks'].values())
            assert m['evaluation_rng_protocol'] == 'stage22_34_crn_v1'
            tr = inside_run(cfg['source']['run_dir'])
            tm = read(tr / 'metrics.json')
            assert tm['test_split_accessed'] is False and tm['evaluation_split'] == 'validation'
            cp = Path(tm['checkpoint']).resolve()
            assert cp.is_relative_to(tr) and digest(cp) == m['source_checkpoint_sha256']
            for run in (ev, tr):
                assert read(run / 'status.json')['state'] == 'SUCCESS'
                for filename in ('resolved_config.yaml', 'metrics.json', 'status.json', 'resource_usage.json'):
                    p = run / filename
                    hashes[str(p)] = digest(p)
            cache = ev / 'validation_equal_rate_multiscale_samples.npz'
            for p in (cache, tr / 'train.log', cp):
                hashes[str(p)] = digest(p)
            pair[family] = dict(eval_run=str(ev), train_run=str(tr), cache=str(cache), checkpoint=str(cp))
        records.append(pair)
    for filename in CODE_FILES:
        path = ROOT / filename
        assert path.is_file(), f'CONFIG_ERROR missing diagnostic source {path}'
        hashes[str(path)] = digest(path)
    prior = read(ROOT / 'reports/stage22_41_tests.json')
    for path, expected in prior['code_hashes'].items():
        assert digest(path) == expected, f'CODE_ERROR frozen model/source changed {path}'
    value = dict(stage='22.42', device='cpu', records=records, source_hashes=hashes,
                 prior_code_hashes=prior['code_hashes'], prior_exploratory_findings_seen=True,
                 no_model_forward=True, no_new_training=True, uses_test_split=False,
                 unavailable=['horizon_errors', 'variable_errors', 'observed_target_errors',
                              'missing_target_errors', 'full_training_bank_trace'])
    if MANIFEST.exists():
        assert read(MANIFEST) == value, 'CONFIG_ERROR refusing to overwrite frozen source manifest'
    else:
        write_json(MANIFEST, value)
    print('PREPARED frozen three-seed CPU inputs', flush=True)


def verify_frozen():
    manifest = read(MANIFEST)
    for path, expected in {**manifest['source_hashes'], **manifest['prior_code_hashes']}.items():
        assert digest(path) == expected, f'DATA_OR_CODE_ERROR frozen input changed {path}'
    return manifest


def mask_descriptors(packed, length=192):
    packed = np.asarray(packed)
    assert packed.dtype == np.uint8 and packed.ndim == 2
    assert packed.shape[1] == (length + 7) // 8
    observed = np.unpackbits(packed, axis=1, count=length, bitorder='big').astype(bool)
    missing = ~observed
    current = np.zeros(len(missing), dtype=np.int64)
    longest = current.copy()
    for col in missing.T:
        current = np.where(col, current + 1, 0)
        longest = np.maximum(longest, current)
    starts = missing & ~np.concatenate((np.zeros((len(missing), 1), dtype=bool), missing[:, :-1]), axis=1)
    return dict(missing_fraction=missing.mean(axis=1), tail_missing_length=current,
                longest_missing_length=longest, missing_run_count=starts.sum(axis=1))


def group_masks(features):
    n = len(features['missing_fraction'])
    groups = {'all': np.ones(n, dtype=bool)}
    for i, indices in enumerate(np.array_split(np.arange(n), 4), 1):
        group = np.zeros(n, dtype=bool)
        group[indices] = True
        groups[f'time_quarter_{i}'] = group
    rate = features['missing_fraction']
    for name, lo, hi in (('lt20', 0., .2), ('20to30', .2, .3), ('ge30', .3, 1.000001)):
        groups[f'missing_{name}'] = (rate >= lo) & (rate < hi)
    for feature, prefix in (('tail_missing_length', 'tail'), ('longest_missing_length', 'longest')):
        x = features[feature]
        for label, lo, hi in (('0', 0, 0), ('1to5', 1, 5), ('6to20', 6, 20), ('gt20', 21, 192)):
            groups[f'{prefix}_{label}'] = (x >= lo) & (x <= hi)
    for i in range(1, 5):
        for label in ('lt20', '20to30', 'ge30'):
            groups[f'quarter_{i}__missing_{label}'] = groups[f'time_quarter_{i}'] & groups[f'missing_{label}']
    return groups


def summarize_group(a, b, route, selected, full_count=None):
    selected = np.asarray(selected, dtype=bool)
    count = float(a['complete_count'][selected].sum())
    out = dict(windows=int(selected.sum()), target_elements=count)
    for metric, accumulator in (('mse', 'sse'), ('mae', 'sae')):
        x = a[f'complete_{accumulator}_{route}'][selected].astype(np.float64)
        y = b[f'complete_{accumulator}_{route}'][selected].astype(np.float64)
        delta = float(y.sum() - x.sum())
        out.update({f'original_{metric}': float(x.sum() / count) if count else None,
                    f'fixed_{metric}': float(y.sum() / count) if count else None,
                    f'{metric}_absolute_change': delta / count if count else None,
                    f'{metric}_change_percent': delta / float(x.sum()) * 100 if count and x.sum() > 0 else None,
                    f'{metric}_error_sum_delta': delta,
                    f'{metric}_contribution_to_global_change': delta / full_count if full_count else None,
                    f'{metric}_worse_window_fraction': float((y > x).mean()) if len(x) else None})
    return out


def average_ranks(x):
    x = np.asarray(x)
    order = np.argsort(x, kind='stable')
    sorted_x = x[order]
    edges = np.concatenate(([0], np.flatnonzero(sorted_x[1:] != sorted_x[:-1]) + 1, [len(x)]))
    ranks = np.empty(len(x), dtype=float)
    for start, end in zip(edges[:-1], edges[1:]):
        ranks[order[start:end]] = (start + end - 1) / 2.
    return ranks


def spearman(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])


def concentration(delta):
    delta = np.asarray(delta, dtype=np.float64)
    positive = delta[delta > 0]
    negative = delta[delta < 0]
    k = max(1, int(np.ceil(.05 * len(delta))))
    return dict(worse_windows=int(len(positive)), improved_windows=int(len(negative)),
                unchanged_windows=int((delta == 0).sum()), positive_error_increment=float(positive.sum()),
                negative_error_increment=float(negative.sum()), net_error_increment=float(delta.sum()),
                top_five_percent_windows=k,
                top_five_percent_share_of_positive_increment=float(np.sort(positive)[-k:].sum() / positive.sum()) if len(positive) else None)


def load_pair(pair):
    arrays = []
    for family in ('original', 'fixed'):
        with np.load(pair[family]['cache'], allow_pickle=False) as cache:
            arrays.append({k: cache[k] for k in (*IDENTITY, *[f'complete_{acc}_{r}' for acc in ('sse', 'sae') for r in ROUTES])})
    a, b = arrays
    for key in IDENTITY:
        assert np.array_equal(a[key], b[key]), f'DATA_ERROR unpaired {key}'
    assert np.array_equal(a['sample_index'], np.arange(1648))
    assert (a['complete_count'] > 0).all()
    for family, arr in zip(('original', 'fixed'), arrays):
        metrics = read(Path(pair[family]['eval_run']) / 'metrics.json')
        for route in ROUTES:
            for metric, acc in (('mse', 'sse'), ('mae', 'sae')):
                vector = arr[f'complete_{acc}_{route}']
                assert np.isfinite(vector).all() and (vector >= 0).all()
                value = float(vector.sum() / arr['complete_count'].sum())
                assert abs(value - metrics[f'complete_target_{metric}_{route}']) < 1e-12
    return a, b


def analyze_pair(pair, include_bank=True):
    a, b = load_pair(pair)
    features = mask_descriptors(a['history_mask_packed'])
    groups = group_masks(features)
    count = float(a['complete_count'].sum())
    strata = {route: {name: summarize_group(a, b, route, mask, count) for name, mask in groups.items()} for route in ROUTES}
    for route in ROUTES:
        for metric in ('mse', 'mae'):
            total = sum(strata[route][f'time_quarter_{i}'][f'{metric}_error_sum_delta'] for i in range(1, 5))
            assert np.isclose(total, strata[route]['all'][f'{metric}_error_sum_delta'], atol=1e-7, rtol=1e-12)
    correlations, tails = {}, {}
    for metric, acc in (('mse', 'sse'), ('mae', 'sae')):
        delta = b[f'complete_{acc}_on'] - a[f'complete_{acc}_on']
        correlations[metric] = {name: spearman(value, delta / a['complete_count']) for name, value in features.items()}
        tails[metric] = concentration(delta)
    stats = {name: {'min': float(value.min()), 'mean': float(value.mean()), 'max': float(value.max())} for name, value in features.items()}
    quarters = {str(i): {name: float(values[groups[f'time_quarter_{i}']].mean()) for name, values in features.items()} for i in range(1, 5)}
    result = dict(seed=pair['seed'], sources=pair, sample_count=len(a['sample_index']),
                  fixed_assets='PASS', metric_roundtrip='PASS', additive_time_decomposition='PASS',
                  mask_feature_stats=stats, time_quarter_mask_feature_means=quarters,
                  strata=strata, feature_delta_spearman=correlations, error_increment_concentration=tails)
    if include_bank:
        from stage22_42_bank_audit import audit_checkpoint_pair
        result['training_and_bank'] = audit_checkpoint_pair(pair['original']['train_run'], pair['fixed']['train_run'])
    return result


def markdown(payload):
    lines = ['# 步骤22.42：三seed收益反转离线诊断', '', f'状态：{payload["decision"]}', '',
             '仅已完成validation（验证集）缓存与CPU检查点；不是新精度实验。变化%=修复相对原始，负值表示改善。', '',
             '| Seed | 分组 | 窗口 | MSE变化% | MAE变化% | MSE对全局变化贡献 | MAE对全局变化贡献 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in payload.get('records', []):
        for name, g in row['strata']['on'].items():
            fmt = lambda x: 'NA' if x is None else f'{x:.6f}'
            lines.append(f'| {row["seed"]} | {name} | {g["windows"]} | {fmt(g["mse_change_percent"])} | {fmt(g["mae_change_percent"])} | {fmt(g["mse_contribution_to_global_change"])} | {fmt(g["mae_contribution_to_global_change"])} |')
    lines += ['', '已知边界：1648缓存窗口不是全部1661理论验证窗口；drop_last丢13尾窗，原配对仍公平。',
              '时间/缺失分组与原型几何仅描述关联，不证明因果；三seed联合变化初始化和缺失布局。',
              '逐预测跨度/变量/观测与缺失目标误差无缓存，全部NA；不以内部观测目标Vali Loss替代完整目标MSE/MAE。',
              '建议下一步：冻结每个checkpoint的权重/银行，先在2028做原/修复公式2×2交叉切换，只诊断直接填补效应与训练状态交互；本轮未启动。',
              '', '全部次要路径、相关性、银行审计、曲线与原始来源见JSON报告。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return 0
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized(), 'CPU diagnostic must not initialize CUDA'
    manifest = verify_frozen()
    if not args.smoke:
        assert read(ROOT / 'reports/stage22_42_tests.json')['passed']
        assert read(ROOT / 'reports/stage22_42_smoke.json')['status'] == 'SUCCESS'
    stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    name = f's4m_distance_diagnostic__traffic__validation__seed2242{"__smoke" if args.smoke else ""}__{stamp}'
    run = ROOT / 'runs' / name
    run.mkdir(exist_ok=False)
    start = time.monotonic()
    cfg = dict(experiment=dict(id=name, model='s4m_distance_diagnostic', dataset='traffic',
        mask_type='point', seed=2242, smoke_test=args.smoke, protocol_match='offline_same_fixed_validation_assets',
        protocol_scope='descriptive_analysis_not_new_predictive_result'), analysis_seeds=[2027, 2028, 2029],
        device='cpu', manifest=str(MANIFEST), manifest_sha256=digest(MANIFEST),
        rules=dict(quarters=4, missing_rate_edges=[0, .2, .3, 1], missing_length_groups=['0', '1to5', '6to20', 'gt20']),
        no_model_forward=True, no_new_training=True, uses_test_split=False, checkpoint_saved=False)
    atomic_text(run / 'resolved_config.yaml', yaml.safe_dump(cfg, sort_keys=False))
    atomic_text(run / 'command.sh', f'cd {shlex.quote(str(ROOT))}\nCUDA_VISIBLE_DEVICES="" {shlex.quote(sys.executable)} scripts/stage22_42_diagnostic.py'+(' --smoke' if args.smoke else '')+'\n')
    atomic_text(run / 'env.txt', f'python={sys.version}\nnumpy={np.__version__}\nplatform={platform.platform()}\nCUDA_VISIBLE_DEVICES=\nconda_prefix={sys.prefix}\n')
    gitlog = []
    for tag, command in [('commit', ['git', 'rev-parse', 'HEAD']), ('status', ['git', 'status', '--short']), ('diff_stat', ['git', 'diff', '--stat'])]:
        result = subprocess.run(command, cwd=ROOT / 'baselines/S4M', capture_output=True, text=True)
        assert result.returncode == 0
        gitlog.append(f'[{tag}] exit=0\n{result.stdout}')
    atomic_text(run / 'git_state.txt', '\n'.join(gitlog))
    binary_diff = subprocess.run(['git', 'diff', '--binary'], cwd=ROOT / 'baselines/S4M', capture_output=True, text=True, check=True).stdout
    atomic_text(run / 'code.diff', binary_diff)
    for filename in CODE_FILES:
        atomic_text(run / 'source_snapshot' / filename, (ROOT / filename).read_text())
    write_json(run / 'source_manifest.json', manifest)
    write_json(run / 'status.json', dict(state='RUNNING', experiment_id=name, pid=os.getpid()))
    code = 1
    try:
        with (run / 'train.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            print('CPU-only frozen cache diagnosis; no model forward/no training/no test', flush=True)
            records = []
            pairs = manifest['records'][1:2] if args.smoke else manifest['records']
            for pair in pairs:
                records.append(analyze_pair(pair, include_bank=not args.smoke))
                print(f'PASS seed={pair["seed"]} fixed assets/metrics/time additive checks', flush=True)
            assert not torch.cuda.is_initialized(), 'CUDA unexpectedly initialized'
            verify_frozen()
            payload = dict(stage='22.42', status='SUCCESS', experiment_id=name, run_dir=str(run),
                smoke_test=args.smoke, uses_test_split=False, uses_new_training=False, no_model_forward=True,
                dataset='traffic', evaluation_split='validation', records=records,
                decision='SMOKE_EXECUTION_ONLY' if args.smoke else 'TEMPORAL_SIGN_REVERSAL_LOCALIZED_CAUSAL_MECHANISM_UNIDENTIFIED',
                information_gaps=manifest['unavailable'], source_manifest_sha256=digest(MANIFEST),
                source_unchanged_after_analysis=True, new_predictive_metrics=False)
            write_json(run / 'metrics.json', payload)
            suffix = 'smoke' if args.smoke else 'diagnostic'
            write_json(ROOT / f'reports/stage22_42_{suffix}.json', payload)
            if not args.smoke:
                atomic_text(ROOT / 'reports/stage22_42_diagnostic.md', markdown(payload))
                with (ROOT / 'reports/stage22_42_diagnostic.csv').open('w', newline='') as f:
                    columns = ['seed', 'route', 'group', *next(iter(records[0]['strata']['on'].values())).keys()]
                    writer = csv.DictWriter(f, fieldnames=columns)
                    writer.writeheader()
                    for record in records:
                        for route, groups in record['strata'].items():
                            for group, values in groups.items():
                                writer.writerow(dict(seed=record['seed'], route=route, group=group, **values))
            print(payload['decision'], flush=True)
        code = 0
        write_json(run / 'status.json', dict(state='SUCCESS', experiment_id=name, return_code=0, finished_at=dt.datetime.now(dt.timezone.utc).isoformat()))
    except Exception as error:
        with (run / 'train.log').open('a') as log:
            traceback.print_exc(file=log)
        write_json(run / 'status.json', dict(state='CODE_ERROR', experiment_id=name, return_code=1, verification_error=repr(error)))
        raise
    finally:
        write_json(run / 'resource_usage.json', dict(device='cpu', elapsed_seconds=time.monotonic()-start,
            peak_cpu_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024., peak_gpu_memory_mb=0,
            return_code=code, checkpoint_saved=False))
    subprocess.run([sys.executable, str(ROOT / 'scripts/aggregate_results.py')], cwd=ROOT, check=True)
    print(f'run_dir={run}', flush=True)
    print(read(run / 'metrics.json')['decision'], flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
