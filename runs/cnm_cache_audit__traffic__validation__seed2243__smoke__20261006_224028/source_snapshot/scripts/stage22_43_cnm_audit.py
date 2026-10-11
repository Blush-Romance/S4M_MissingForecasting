#!/usr/bin/env python3
"""CPU reaggregation of existing CNM validation caches; no model execution."""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
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
import numpy as np
import yaml
import stage22_42_diagnostic as old

ROOT = old.ROOT
MANIFEST = ROOT / 'runtime/stage22_43_cnm_audit/source_manifest.json'
CODE = ('scripts/stage22_43_cnm_audit.py', 'scripts/test_stage22_43_cnm_audit.py',
        'scripts/stage22_42_diagnostic.py', 'scripts/aggregate_results.py')
WEIGHTS = {'original': .5, 'l5': .25, 'l20': .25}
SEEDS = [2027, 2028, 2029]


def compare_routes(a):
    """Sum error accumulators / sum target counts, never average batch means."""
    count = np.asarray(a['complete_count'], dtype=np.float64)
    indices = np.asarray(a['sample_index'])
    assert count.ndim == 1 and count.size, 'DATA_ERROR empty/invalid counts'
    assert indices.shape == count.shape and np.array_equal(indices, np.arange(count.size)), 'DATA_ERROR sample order'
    assert np.isfinite(count).all() and (count > 0).all(), 'DATA_ERROR invalid counts'
    out = {'sample_count': int(count.size), 'target_elements': float(count.sum()),
           'baseline': {}, 'cnm': {}, 'absolute_change': {}, 'relative_gain_percent': {},
           'worse_window_fraction': {}}
    for metric, acc in (('mse', 'sse'), ('mae', 'sae')):
        x = np.asarray(a[f'complete_{acc}_on'], dtype=np.float64)
        y = np.asarray(a[f'complete_{acc}_multiscale'], dtype=np.float64)
        for v in (x, y):
            assert v.shape == count.shape and np.isfinite(v).all() and (v >= 0).all(), 'DATA_ERROR invalid errors'
        base, cnm = float(x.sum() / count.sum()), float(y.sum() / count.sum())
        out['baseline'][metric], out['cnm'][metric] = base, cnm
        out['absolute_change'][metric] = cnm - base
        out['relative_gain_percent'][metric] = (base - cnm) / base * 100 if base > 0 else None
        out['worse_window_fraction'][metric] = float((y > x).mean())
    out['dual_win'] = all(out['cnm'][k] < out['baseline'][k] for k in ('mse', 'mae'))
    return out


def summarize(rows):
    out = {'seed_count': len(rows), 'dual_win_count': sum(x['dual_win'] for x in rows)}
    for kind in ('baseline', 'cnm', 'relative_gain_percent'):
        out[kind] = {}
        for metric in ('mse', 'mae'):
            values = [x[kind][metric] for x in rows]
            out[kind][metric] = {'values': values,
                'mean': float(np.mean(values)) if all(v is not None for v in values) else None,
                'sample_std': float(np.std(values, ddof=1)) if len(values) > 1 and all(v is not None for v in values) else None}
    out['relative_gain_between_metric_means_percent'] = {
        k: (out['baseline'][k]['mean'] - out['cnm'][k]['mean']) / out['baseline'][k]['mean'] * 100
        if out['baseline'][k]['mean'] > 0 else None for k in ('mse', 'mae')}
    return out


def validate_metadata(pair, family):
    src = pair[family]
    ev, tr = old.inside_run(src['eval_run']), old.inside_run(src['train_run'])
    m = old.read(ev / 'metrics.json')
    cfg = yaml.safe_load((ev / 'resolved_config.yaml').read_text())
    train_cfg = yaml.safe_load((tr / 'resolved_config.yaml').read_text())
    if family == 'fixed':
        args = train_cfg['effective_parser_args']
    else:
        # Legacy baseline runs predate full parser snapshots and the fix switch.
        # Do not manufacture a full parser record for the historical reference.
        args = train_cfg.get('effective_parser_args', train_cfg['official_args'])
    assert m['source_split'] == 'val' and m['uses_test_split'] is False, 'DATA_ERROR validation only'
    assert m['updates_model_or_memory'] is False, 'CONFIG_ERROR writable evaluation'
    assert m['evaluation_rng_protocol'] == 'stage22_34_crn_v1', 'CONFIG_ERROR CRN protocol'
    assert m['artifact_checks'] and all(m['artifact_checks'].values()), 'DATA_ERROR prior checks'
    assert m['multiscale_weights'] == WEIGHTS == cfg['diagnostic']['multiscale_weights'], 'CONFIG_ERROR weights'
    assert cfg['diagnostic']['max_eval_batches'] == -1, 'CONFIG_ERROR incomplete source'
    assert cfg['experiment']['seed'] == args['seed'] == pair['seed'], 'CONFIG_ERROR seed'
    assert cfg['experiment']['dataset'] == m['dataset'] == 'traffic', 'DATA_ERROR dataset'
    assert args.get('fix_min_distance_normalization', False) is (family == 'fixed'), 'CONFIG_ERROR formula identity'
    for flag in ('use_mipp', 'use_fusr', 'use_mrra', 'use_mcpr', 'use_ttrr', 'use_ard'):
        assert args.get(flag, False) is False, f'CONFIG_ERROR unexpected module {flag}'
        assert not m[f'checkpoint_{flag[4:]}_state_keys'], f'CONFIG_ERROR unexpected checkpoint module {flag}'
    assert Path(cfg['source']['run_dir']).resolve() == tr
    assert Path(m['source_checkpoint']).resolve() == Path(src['checkpoint']).resolve()
    assert m['source_checkpoint_sha256'] == old.digest(src['checkpoint']), 'DATA_ERROR checkpoint hash'
    assert m['mask_asset_sha256'] == old.digest(m['mask_asset']), 'DATA_ERROR mask hash'
    assert m['evaluation_samples_sha256'] == old.digest(src['cache']), 'DATA_ERROR cache hash'
    for run in (ev, tr):
        assert old.read(run / 'status.json')['state'] == 'SUCCESS', 'DATA_ERROR unfinished source'
    with np.load(src['cache'], allow_pickle=False) as cache:
        for acc in ('sse', 'sae'):
            assert np.array_equal(cache[f'reference_{acc}_on'], cache[f'complete_{acc}_on']), 'DATA_ERROR CRN reference mismatch'
    return m, cfg


def prepare():
    inherited = old.verify_frozen()
    assert [p['seed'] for p in inherited['records']] == SEEDS
    hashes = {**inherited['source_hashes'], **inherited['prior_code_hashes']}
    for pair in inherited['records']:
        old.load_pair(pair)
        for family in ('original', 'fixed'):
            m, _ = validate_metadata(pair, family)
            hashes[m['mask_asset']] = old.digest(m['mask_asset'])
    hashes[str(old.MANIFEST)] = old.digest(old.MANIFEST)
    for relative in CODE:
        hashes[str(ROOT / relative)] = old.digest(ROOT / relative)
    value = {'stage': '22.43', 'records': inherited['records'], 'source_hashes': hashes,
             'inherited_manifest': str(old.MANIFEST), 'weights': WEIGHTS,
             'primary_comparison': 'fixed_formula_same_checkpoint_original_vs_multiscale',
             'comparison_previously_observable': True, 'no_model_forward': True,
             'no_new_training': True, 'uses_test_split': False,
             'formula_cross_diagnostic': 'DEFERRED_NOT_STARTED',
             'memory_index_bug': 'CONFIRMED_NOT_FIXED_NOT_ISOLATED'}
    if MANIFEST.exists():
        assert old.read(MANIFEST) == value, 'CONFIG_ERROR refusing frozen manifest overwrite'
    else:
        old.write_json(MANIFEST, value)
    print(f'PREPARED seeds={SEEDS} hashes={len(hashes)} validation_only', flush=True)


def verify():
    value = old.read(MANIFEST)
    for path, expected in value['source_hashes'].items():
        assert old.digest(path) == expected, f'DATA_OR_CODE_ERROR changed source {path}'
    return value


def analyze(pair):
    arrays = old.load_pair(pair)
    out = {'seed': pair['seed'], 'sources': pair}
    for family, a in zip(('original', 'fixed'), arrays):
        m, cfg = validate_metadata(pair, family)
        value = compare_routes(a)
        for kind, route in (('baseline', 'on'), ('cnm', 'multiscale')):
            for metric in ('mse', 'mae'):
                assert abs(value[kind][metric] - m[f'complete_target_{metric}_{route}']) < 1e-12, 'DATA_ERROR metric roundtrip'
        value.update({'checks': 'PASS', 'checkpoint_sha256': m['source_checkpoint_sha256'],
            'training_parser_evidence': 'full_effective_parser' if family == 'fixed' else 'legacy_official_args_and_checkpoint_not_full_parser',
            'parameter_count': m['parameter_count'], 'additional_parameters': 0,
            'forward_calls_baseline': 1, 'forward_calls_cnm': 3,
            'nominal_missing_rate': cfg['experiment']['nominal_missing_rate'],
            'realized_missing_rate': cfg['experiment']['realized_missing_rate'],
            'timing_is_isolated_benchmark': m['timing_is_isolated_benchmark'],
            'existing_component_timing_reference_only': m['estimated_serial_latency']['multiscale'],
            'isolated_latency_seconds': None, 'route_specific_peak_gpu_memory_mb': None})
        out[family] = value
    return out


def markdown(p):
    rows = p['records']
    lines = ['# 步骤22.43：距离公式修复后CNM-SE增益核查', '',
        'CNM-SE：反事实嵌套掩码自集成。只复算已有Traffic验证集缓存，不训练、不前向、不读测试集。', '',
        '主比较为同一修复后S4M检查点：原始单路径 vs 固定多尺度；不是修复前后对比。', '',
        '| Seed（随机种子） | 修复基线MSE | +CNM MSE | MSE改善% | 修复基线MAE | +CNM MAE | MAE改善% |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        x = row['fixed']
        lines.append(f'| {row["seed"]} | {x["baseline"]["mse"]:.9f} | {x["cnm"]["mse"]:.9f} | {x["relative_gain_percent"]["mse"]:.6f} | {x["baseline"]["mae"]:.9f} | {x["cnm"]["mae"]:.9f} | {x["relative_gain_percent"]["mae"]:.6f} |')
    s = p['summary']['fixed']
    lines += ['', f'双指标改善 {s["dual_win_count"]}/{s["seed_count"]}；逐seed改善率算术平均：MSE {s["relative_gain_percent"]["mse"]["mean"]:.6f}%，MAE {s["relative_gain_percent"]["mae"]["mean"]:.6f}%。',
        'MSE为均方误差，MAE为平均绝对误差，越低越好；正改善率表示降低。', '',
        '| 指标均值±种子样本标准差 | 修复基线 | +CNM |', '| --- | ---: | ---: |']
    for k in ('mse', 'mae'):
        a, b = s['baseline'][k], s['cnm'][k]
        lines.append(f'| {k.upper()} | {a["mean"]:.9f} ± {a["sample_std"]:.9f} | {b["mean"]:.9f} ± {b["sample_std"]:.9f} |')
    lines += ['', '样本标准差不是置信区间或统计显著性。逐seed改善率平均与均值指标相对降幅分开保存。', '',
        '## 历史参照与归因', '',
        '原公式基线上的CNM收益也复算，见JSON/CSV，不与修复收益相加。旧22.41距离修复主比较仍2/3双胜、2028退化，旧判定不变。',
        '当前支持CNM小幅正收益并非完全依赖距离公式错误，不证明PG95有效、两个独立创新成立或所有实现风险已排除。', '',
        '## 代价与证据边界', '',
        '- 每seed同一检查点、1648窗口、固定历史/嵌套掩码，CRN（共同随机数）参照逐窗一致。理论1661窗口中原流程丢13尾窗，本步不改评估。',
        '- 原始预测0.5 + L5预测0.25 + L20预测0.25；L表示额外遮挡块长。L1仅诊断、不参与主多尺度。',
        '- 可训练参数28,371,290，新增参数0，共享训练预算；推理由1次变3次前向，不能声称同推理预算、加速或轻量化。',
        '- 旧串行组件时序仅作参考、非隔离端到端测速；没有两条路线独立峰值显存，缺项保留NA。',
        '- 缓存只有逐窗口误差和，没有完整预测/真值；本步核验汇总，不能从预测重新计算误差或重演融合。',
        '- 文件/检查点哈希前后不变；旧评估记忆只读来自代码写保护与工件记录，没有运行时记忆前后内容哈希。',
        '- 标称生成参数0.06不是实际缺失6%；固定全数据资产实际约26%。三个seed同时改变初始化和缺失布局。',
        '- 已多次开发使用的验证集，不是新独立测试；三seed方向一致不证明跨数据集泛化、显著性或SCI二区水平。',
        '- 上游新增原型代表索引错误仍未修复，不与本步混改。公式×冻结状态交叉诊断暂列备选、未启动。', '',
        '- 历史原公式训练配置未保存完整有效parser（参数解析器）快照；历史参照使用official_args与检查点/冻结源码证据，不伪造缺省参数记录。主比较的修复版有完整有效parser。', '',
        '## 决策与最短下一步', '',
        '保留CNM方向性正信号，不扩大量实验、不写论文。建议22.44先独立实现可关闭的原型代表索引修复并做CPU/基线回归及单seed同预算验证；不同时加创新或改训练协议。其收益归为错误修复。基线确认后再公平复核CNM，第二个创新尚未成立。', '',
        f'本步run：`{p["experiment_id"]}`。全部来源、全精度指标与哈希见同run及JSON报告。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return 0
    manifest = verify()
    tests = old.read(ROOT / 'reports/stage22_43_tests.json')
    assert tests['passed'] and tests['source_hashes'] == {str(ROOT / x): old.digest(ROOT / x) for x in CODE[:2]}, 'CODE_ERROR tests stale'
    if not args.smoke:
        smoke = old.read(ROOT / 'reports/stage22_43_smoke.json')
        assert smoke['status'] == 'SUCCESS' and smoke['source_manifest_sha256'] == old.digest(MANIFEST), 'CONFIG_ERROR smoke required'
    stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    name = f'cnm_cache_audit__traffic__validation__seed2243{"__smoke" if args.smoke else ""}__{stamp}'
    run = ROOT / 'runs' / name
    run.mkdir(exist_ok=False)
    start, code = time.monotonic(), 1
    config = {'experiment': {'id': name, 'model': 'cnm_cache_audit', 'dataset': 'traffic',
        'mask_type': 'point', 'seed': 2243, 'smoke_test': args.smoke,
        'protocol_match': 'reaggregation_of_same_checkpoint_fixed_validation_assets',
        'protocol_scope': 'descriptive_cache_audit_not_new_predictive_experiment'},
        'analysis_seeds': [2028] if args.smoke else SEEDS, 'device': 'cpu',
        'primary': manifest['primary_comparison'], 'weights': WEIGHTS,
        'manifest': str(MANIFEST), 'manifest_sha256': old.digest(MANIFEST),
        'no_model_forward': True, 'no_new_training': True, 'uses_test_split': False,
        'formula_cross_diagnostic': 'DEFERRED_NOT_STARTED',
        'configuration_changes': ['analysis identity and comparison only; model/data/training/evaluation settings unchanged']}
    old.atomic_text(run / 'resolved_config.yaml', yaml.safe_dump(config, sort_keys=False))
    command = f'CUDA_VISIBLE_DEVICES="" {shlex.quote(sys.executable)} scripts/stage22_43_cnm_audit.py' + (' --smoke' if args.smoke else '')
    old.atomic_text(run / 'command.sh', f'cd {shlex.quote(str(ROOT))}\n{command}\n')
    old.atomic_text(run / 'env.txt', f'python={sys.version}\nnumpy={np.__version__}\nplatform={platform.platform()}\nconda_prefix={sys.prefix}\nCUDA_VISIBLE_DEVICES=\nGPU=no model execution\n')
    gitlog = []
    for tag, command in [('commit', ['git', 'rev-parse', 'HEAD']), ('status', ['git', 'status', '--short']), ('diff_stat', ['git', 'diff', '--stat'])]:
        proc = subprocess.run(command, cwd=ROOT / 'baselines/S4M', capture_output=True, text=True, check=True)
        gitlog.append(f'[{tag}] exit=0\n{proc.stdout}')
    old.atomic_text(run / 'git_state.txt', '\n'.join(gitlog))
    diff = subprocess.run(['git', 'diff', '--binary'], cwd=ROOT / 'baselines/S4M', capture_output=True, text=True, check=True).stdout
    old.atomic_text(run / 'code.diff', diff)
    for relative in CODE:
        old.atomic_text(run / 'source_snapshot' / relative, (ROOT / relative).read_text())
    old.write_json(run / 'source_manifest.json', manifest)
    old.write_json(run / 'status.json', {'state': 'RUNNING', 'experiment_id': name, 'pid': os.getpid()})
    try:
        with (run / 'train.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            print('CPU cached-error aggregation only: no training, forward or test', flush=True)
            pairs = manifest['records'][1:2] if args.smoke else manifest['records']
            rows = []
            for pair in pairs:
                rows.append(analyze(pair))
                print(f'PASS seed={pair["seed"]} same-checkpoint/fixed-assets/weighted-metrics', flush=True)
            assert verify() == manifest, 'DATA_ERROR source mutation'
            result = {'stage': '22.43', 'status': 'SUCCESS', 'experiment_id': name, 'run_dir': str(run),
                'smoke_test': args.smoke, 'dataset': 'traffic', 'evaluation_split': 'validation',
                'uses_test_split': False, 'no_new_training': True, 'no_model_forward': True,
                'new_predictive_experiment': False, 'source_unchanged_after_analysis': True,
                'source_manifest_sha256': old.digest(MANIFEST), 'records': rows,
                'summary': {family: summarize([r[family] for r in rows]) for family in ('original', 'fixed')},
                'prior_distance_fix_decision_unchanged': old.read(old.SOURCE_REPORT)['decision'],
                'formula_cross_diagnostic': 'DEFERRED_NOT_STARTED',
                'decision': 'SMOKE_EXECUTION_ONLY' if args.smoke else 'CNM_DIRECTIONAL_SIGNAL_RETAINED_NOT_NEW_INDEPENDENT_EVIDENCE',
                'original_paper_same_protocol_metrics': None}
            old.write_json(run / 'metrics.json', result)
            if args.smoke:
                old.write_json(ROOT / 'reports/stage22_43_smoke.json', result)
            else:
                old.write_json(ROOT / 'reports/stage22_43_cnm_audit.json', result)
                old.atomic_text(ROOT / 'reports/stage22_43_cnm_audit.md', markdown(result))
                with (ROOT / 'reports/stage22_43_cnm_audit.csv').open('w', newline='') as f:
                    columns = ['seed', 'formula_family', 'sample_count', 'baseline_mse', 'cnm_mse',
                               'mse_absolute_change', 'mse_gain_percent', 'baseline_mae', 'cnm_mae',
                               'mae_absolute_change', 'mae_gain_percent', 'dual_win', 'eval_run', 'checkpoint']
                    writer = csv.DictWriter(f, fieldnames=columns)
                    writer.writeheader()
                    for r in rows:
                        for family in ('original', 'fixed'):
                            v = r[family]
                            row = {'seed': r['seed'], 'formula_family': family, 'sample_count': v['sample_count'],
                                   'dual_win': v['dual_win'], 'eval_run': r['sources'][family]['eval_run'],
                                   'checkpoint': r['sources'][family]['checkpoint']}
                            for k in ('mse', 'mae'):
                                row.update({f'baseline_{k}': v['baseline'][k], f'cnm_{k}': v['cnm'][k],
                                            f'{k}_absolute_change': v['absolute_change'][k],
                                            f'{k}_gain_percent': v['relative_gain_percent'][k]})
                            writer.writerow(row)
            print(result['decision'], flush=True)
        code = 0
        old.write_json(run / 'status.json', {'state': 'SUCCESS', 'experiment_id': name, 'return_code': 0,
            'finished_at': dt.datetime.now(dt.timezone.utc).isoformat()})
    except Exception as error:
        with (run / 'train.log').open('a') as log:
            traceback.print_exc(file=log)
        old.write_json(run / 'status.json', {'state': 'CODE_ERROR', 'experiment_id': name,
            'return_code': 1, 'verification_error': repr(error)})
        raise
    finally:
        old.write_json(run / 'resource_usage.json', {'device': 'cpu', 'elapsed_seconds': time.monotonic() - start,
            'peak_cpu_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.,
            'peak_gpu_memory_mb': 0, 'return_code': code, 'checkpoint_saved': False})
    subprocess.run([sys.executable, str(ROOT / 'scripts/aggregate_results.py')], cwd=ROOT, check=True)
    if not args.smoke:
        archive = ROOT / 'reports/archive' / f'stage22_43_final__{stamp}'
        for filename in ('stage22_43_cnm_audit.json', 'stage22_43_cnm_audit.md', 'stage22_43_cnm_audit.csv', 'latest_results.csv', 'latest_results.md'):
            old.atomic_text(archive / filename, (ROOT / 'reports' / filename).read_text())
    print(f'run_dir={run}', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
