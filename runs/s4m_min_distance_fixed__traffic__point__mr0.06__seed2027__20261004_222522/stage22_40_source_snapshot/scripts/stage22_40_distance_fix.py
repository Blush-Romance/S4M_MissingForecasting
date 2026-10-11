#!/usr/bin/env python3
"""Isolated minimum-distance bug-fix control; unchanged training and CRN validation."""
import argparse
import ast
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

import stage22_34_factorial as h
from run_s4m_experiment import validate_experiment_config as validate_train
from run_stage14_uncertainty_cache import validate_config as validate_eval
from run_stage22_22_traffic_pipeline import materialize_evaluation_config

ROOT = h.ROOT
RT = ROOT / 'runtime/stage22_40_distance_fix'
STATE = RT / 'state.json'
CONFIG = ROOT / 'configs/stage22_40_distance_fix'
MANIFEST = RT / 'config_manifest.json'
TESTS = ROOT / 'reports/stage22_40_distance_fix_tests.json'
BASELINE_TRAIN = ROOT / 'runs/s4m_paired_rng__traffic__point__mr0.06__seed2027__20260908_144251'
BASELINE_EVAL = ROOT / 'runs/cnm_se_crn_v1_s4m__traffic__validation__seed2027__20260908_151505'
FIX_FLAG = 'fix_min_distance_normalization'
JOBS = ('off_train_smoke', 'fixed_train_smoke', 'off_eval_smoke', 'fixed_eval_smoke',
        'fixed_train_formal', 'fixed_eval_formal')
IDENTITY_KEYS = ('model_id', 'des')


def frozen(path, value):
    path = Path(path)
    if path.exists() and path.read_text() != value:
        raise ValueError(f'CONFIG_ERROR refusing to overwrite frozen file: {path}')
    if not path.exists():
        h.atomic_text(path, value)


def template_path(kind, tag):
    return ROOT / f'configs/stage22_36_control/s4m_{kind}_{tag}.yaml'


def validate_delta(c):
    """Reject any undeclared scientific setting change before execution."""
    kind = 'eval' if 'diagnostic' in c else 'train'
    tag = 'smoke' if c['experiment']['smoke_test'] else 'formal'
    template = template_path(kind, tag)
    old = yaml.safe_load(template.read_text())
    assert c['provenance']['template_sha256'] == h.digest(template), 'CONFIG_ERROR template changed'
    assert c['data'] == old['data'], 'CONFIG_ERROR changed data protocol'
    assert c['runtime'] == old['runtime'], 'CONFIG_ERROR changed runtime template'
    assert c['experiment']['seed'] == 2027
    assert c['experiment']['smoke_test'] == old['experiment']['smoke_test']
    expected_experiment = copy.deepcopy(old['experiment'])
    identity_fields = ('model', 'purpose', 'protocol_scope') if kind == 'train' else ('model', 'purpose')
    for key in identity_fields:
        expected_experiment[key] = c['experiment'][key]
    assert c['experiment'] == expected_experiment, 'CONFIG_ERROR changed scientific experiment metadata'
    enabled = c['provenance']['fix_enabled']
    assert isinstance(enabled, bool)
    if kind == 'train':
        args = copy.deepcopy(c['official_args'])
        assert args.pop(FIX_FLAG) is enabled
        for key in IDENTITY_KEYS:
            args[key] = old['official_args'][key]
        assert args == old['official_args'], 'CONFIG_ERROR altered training settings'
        validate_train(c)
        assert args['post_train_evaluation'] == 'validation'
        assert args['stage22_36_paired_training_rng'] is True
    else:
        assert c['diagnostic'] == old['diagnostic'], 'CONFIG_ERROR changed CRN evaluator'
        assert c['experiment']['method'] == old['experiment']['method'] == 's4m_baseline'
        assert c['experiment']['protocol_scope'] == old['experiment']['protocol_scope']
        assert c['diagnostic']['split'] == 'val'
        assert c['source']['queue_state'] == str(STATE)
    return c


def audit_baseline():
    for path in (BASELINE_TRAIN, BASELINE_EVAL):
        status = h.read(path / 'status.json')
        assert status['state'] == 'SUCCESS' and status['return_code'] == 0
    train = h.read(BASELINE_TRAIN / 'metrics.json')
    evaluation = h.read(BASELINE_EVAL / 'metrics.json')
    assert train['test_split_accessed'] is False
    assert train['paired_training_rng_protocol'] == 'stage22_36_v1'
    assert train['parameter_count'] == 28371290
    assert evaluation['source_split'] == 'val' and evaluation['uses_test_split'] is False
    assert evaluation['evaluation_rng_protocol'] == 'stage22_34_crn_v1'
    assert evaluation['source_checkpoint_sha256'] == h.digest(train['checkpoint'])
    template = yaml.safe_load(template_path('train', 'formal').read_text())
    actual = yaml.safe_load((BASELINE_TRAIN / 'resolved_config.yaml').read_text())
    assert actual['data'] == template['data']
    assert all(actual['official_args'][k] == v for k, v in template['official_args'].items())
    paths = [BASELINE_TRAIN / n for n in ('resolved_config.yaml', 'metrics.json', 'status.json')]
    paths += [BASELINE_EVAL / n for n in ('resolved_config.yaml', 'metrics.json', 'status.json',
                                       'validation_equal_rate_multiscale_samples.npz')]
    paths.append(Path(train['checkpoint']))
    paths.extend(ROOT / template['data'][n] for n in
                 ('missing_csv', 'complete_csv', 'mask_asset', 'mask_metadata'))
    return dict(baseline_train=str(BASELINE_TRAIN), baseline_eval=str(BASELINE_EVAL),
                frozen_artifacts={str(p): h.digest(p) for p in paths},
                expected_common_sha256=train['paired_initial_common_sha256'],
                parameter_count=train['parameter_count'],
                historical_reserved_peak_mib=train['peak_gpu_reserved_mb'],
                baseline_actual_epochs=train['epochs_completed'],
                primary_metric_keys=['complete_target_mse_on', 'complete_target_mae_on'],
                no_test_access=True, baseline_retrained=False,
                baseline_reuse_requires='default_off_legacy_equivalence_and_initialization_and_sampler_checks')


def prepare():
    if shutil.disk_usage(ROOT).free < 1024**3:
        raise RuntimeError('DATA_ERROR disk below 1GiB; no deletion authorized')
    baseline = audit_baseline()
    frozen(RT / 'baseline_manifest.json', json.dumps(baseline, indent=2) + '\n')
    records = []
    for name in JOBS:
        family, kind, tag = name.split('_')
        enabled = family == 'fixed'
        template = template_path(kind, tag)
        old = yaml.safe_load(template.read_text())
        c = copy.deepcopy(old)
        c['experiment'].update(model=f's4m_min_distance_{family}',
            purpose='isolated_bug_fix_not_research_innovation')
        if kind == 'train':
            c['experiment']['protocol_scope'] = f'stage22_40_distance_bug_fix_{tag}'
            c['official_args'][FIX_FLAG] = enabled
            c['official_args']['model_id'] = f't2240_{family}_2027_{tag[0]}'
            c['official_args']['des'] = f'T2240{family.upper()}2027{tag[0].upper()}'
        else:
            c['source'] = dict(job_id=f'{family}_train_{tag}', queue_state=str(STATE))
        changes = {key: dict(old=old.get(key), new=value,
                    reason='approved isolated minimum-distance correction; identity/provenance only otherwise')
                   for key, value in c.items() if key != 'provenance' and old.get(key) != value}
        c['provenance'] = dict(stage='22.40', user_approved=True, fix_enabled=enabled,
            template=str(template), template_sha256=h.digest(template),
            configuration_changes=changes, baseline_manifest=str(RT / 'baseline_manifest.json'),
            scientific_classification='baseline_bug_fix_not_innovation',
            primary_comparison='original_on_path_CRN_validation_only',
            unchanged_evaluation_routes='inherited_frozen_L1_L5_L20_CNM_secondary_not_new_mechanism')
        validate_delta(c)
        path = CONFIG / f'{name}.yaml'
        frozen(path, yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
        records.append(dict(config=str(path), sha256=h.digest(path)))
    frozen(MANIFEST, json.dumps(records, indent=2) + '\n')
    print('prepared 6 frozen jobs; seed2027, unchanged budgets, validation only', flush=True)


def verify_frozen():
    manifest = h.read(MANIFEST)
    assert len(manifest) == 6
    for row in manifest:
        assert h.digest(row['config']) == row['sha256'], 'CONFIG_ERROR frozen config changed'
        validate_delta(yaml.safe_load(Path(row['config']).read_text()))
    baseline = h.read(RT / 'baseline_manifest.json')
    for path, digest in baseline['frozen_artifacts'].items():
        assert h.digest(path) == digest, f'DATA_ERROR changed baseline/data artifact {path}'
    return manifest


def verify_training_pair(metrics, smoke=False):
    original = h.read(BASELINE_TRAIN / 'metrics.json')
    assert metrics['paired_initial_common_sha256'] == original['paired_initial_common_sha256']
    assert metrics['paired_initial_common_entries'] == original['paired_initial_common_entries'] == 314
    assert metrics['parameter_count'] == original['parameter_count']
    assert metrics['paired_training_rng_protocol'] == original['paired_training_rng_protocol']
    assert metrics['test_split_accessed'] is False
    assert metrics['checkpoint_load_verified'] is True
    assert metrics['checkpoint_all_model_tensors_finite'] is True
    for key in ('paired_train_order', 'paired_validation_order'):
        assert len(metrics[key]) == metrics['epochs_completed']
        if not smoke:
            n = min(len(metrics[key]), len(original[key]))
            assert n > 0 and metrics[key][:n] == original[key][:n], f'CODE_ERROR unpaired sampler {key}'
    return dict(initialization='PASS', parameter_count='PASS', training_rng='PASS',
                sampler_prefix='SMOKE_TRUNCATED_NOT_FORMAL_PREFIX' if smoke else 'PASS',
                original_epochs=original['epochs_completed'], fixed_epochs=metrics['epochs_completed'],
                identical_max_budget_and_early_stopping=True, identical_actual_epochs_or_time_not_required=True)


def namespace_values(log):
    line = next(line for line in log.splitlines() if line.startswith('Namespace('))
    node = ast.parse(line, mode='eval').body
    assert isinstance(node, ast.Call) and node.func.id == 'Namespace'
    return {item.arg: ast.unparse(item.value) if item.arg == 'device' else ast.literal_eval(item.value)
            for item in node.keywords}


def save_effective_args(run, config):
    """Persist real parser defaults as well as sparse launcher arguments, even after OOM."""
    log_path = run / 'train.log'
    if not log_path.exists() or not any(line.startswith('Namespace(') for line in log_path.read_text().splitlines()):
        h.write_json(run / 'effective_parser_args.json', dict(status='NO_NAMESPACE_EMITTED'))
        return
    actual = namespace_values(log_path.read_text())
    kind = 'eval' if 'diagnostic' in config else 'train'
    tag = 'smoke' if config['experiment']['smoke_test'] else 'formal'
    historical = h.read(ROOT / 'runtime/stage22_36_control/state.json')['jobs'][f's4m_{kind}_{tag}']['run_dir']
    previous = namespace_values((Path(historical) / 'train.log').read_text())
    # The prior parser had no explicit flag; its absent value means the legacy false branch.
    previous.setdefault(FIX_FLAG, False)
    differences = {key: dict(old=previous.get(key), new=actual.get(key))
                   for key in previous.keys() | actual.keys() if previous.get(key) != actual.get(key)}
    identity_paths = {'model_id', 'des', 'checkpoints', 'results_root', 'test_results_root',
                      'metrics_path', 'gpu', 'device'}
    allowed = lambda key: key in identity_paths or key == FIX_FLAG or (
        kind == 'eval' and key in ('stage21_checkpoint', 'stage21_metrics_path', 'stage21_samples_path'))
    unexpected = {key: value for key, value in differences.items() if not allowed(key)}
    evidence = dict(status='PASS' if not unexpected else 'CONFIG_ERROR',
        historical_run=historical, actual=actual, differences=differences,
        unexpected_differences=unexpected, actual_namespace_device=actual.get('device'))
    h.write_json(run / 'effective_parser_args.json', evidence)
    resolved_path = run / 'resolved_config.yaml'
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved['effective_parser_args'] = actual
    resolved['effective_parser_comparison'] = {key: value for key, value in evidence.items() if key != 'actual'}
    h.atomic_text(resolved_path, yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
    assert not unexpected, f'CONFIG_ERROR effective parser protocol changed: {unexpected}'
    assert actual[FIX_FLAG] is config['provenance']['fix_enabled']


def execute(path, gpu, attempt):
    manifest = verify_frozen()
    path = (ROOT / path).resolve()
    assert any(row['config'] == str(path) for row in manifest), 'CONFIG_ERROR unknown config'
    tests = h.read(TESTS)
    assert tests['passed'] and tests['manifest_sha256'] == h.digest(MANIFEST), 'CODE_ERROR tests missing/stale'
    baseline = h.read(RT / 'baseline_manifest.json')
    assert tests['expected_common_sha256'] == baseline['expected_common_sha256']
    for source, digest in tests['code_hashes'].items():
        assert h.digest(source) == digest, f'CODE_ERROR source changed after tests: {source}'
    config = validate_delta(yaml.safe_load(path.read_text()))
    is_eval = 'diagnostic' in config
    if is_eval:
        path = materialize_evaluation_config(path, default_state_path=STATE,
            destination=RT / 'materialized' / f'{path.stem}_a{attempt}_{time.strftime("%Y%m%d_%H%M%S")}.yaml')
        effective = yaml.safe_load(path.read_text())
        source = yaml.safe_load(Path(effective['source']['resolved_config']).read_text())
        validate_eval(effective, source)
        assert source['official_args'][FIX_FLAG] is config['provenance']['fix_enabled']
    if shutil.disk_usage(ROOT).free < 1024**3:
        raise RuntimeError('DATA_ERROR disk below 1GiB; stop without deleting artifacts')
    runner = ROOT / ('scripts/run_stage14_uncertainty_cache.py' if is_eval else 'scripts/run_s4m_experiment.py')
    snapshot = {source: Path(source).read_text() for source in tests['code_hashes']}
    run = None
    with subprocess.Popen([sys.executable, '-u', str(runner), '--config', str(path), '--gpu-uuid', gpu,
                           '--scheduler-attempt', str(attempt)], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as child:
        for line in child.stdout:
            print(line, end='', flush=True)
            if line.startswith('run_dir='):
                run = Path(line.strip().split('=', 1)[1])
                for source, content in snapshot.items():
                    h.atomic_text(run / 'stage22_40_source_snapshot' / Path(source).relative_to(ROOT), content)
                h.write_json(run / 'stage22_40_code_manifest.json', tests['code_hashes'])
                h.write_json(run / 'stage22_40_control.json', dict(stage='22.40',
                    scientific_classification='baseline_bug_fix_not_innovation',
                    fix_enabled=config['provenance']['fix_enabled'], frozen_config_sha256=h.digest(path),
                    cpu_tests_sha256=h.digest(TESTS), baseline_manifest_sha256=h.digest(RT / 'baseline_manifest.json'),
                    restart_policy='OOM_from_scratch_no_complete_training_resume_available'))
        code = child.wait()
    if run is not None:
        save_effective_args(run, config)
    if code == 0:
        assert run is not None, 'CODE_ERROR child produced no run_dir'
        metrics = h.read(run / 'metrics.json')
        if not is_eval:
            evidence = verify_training_pair(metrics, smoke=config['experiment']['smoke_test'])
            actual = yaml.safe_load((run / 'resolved_config.yaml').read_text())
            assert actual['official_args'][FIX_FLAG] is config['provenance']['fix_enabled']
            h.write_json(run / 'stage22_40_pairing_checks.json', evidence)
        else:
            assert metrics['uses_test_split'] is False and metrics['updates_model_or_memory'] is False
            assert metrics['artifact_status'] == 'PASS' and all(metrics['artifact_checks'].values())
            assert metrics['evaluation_rng_protocol'] == 'stage22_34_crn_v1'
    return code


def compare_evaluations(original_run, fixed_run):
    runs = [Path(original_run), Path(fixed_run)]
    metrics = [h.read(run / 'metrics.json') for run in runs]
    for metric in metrics:
        assert metric['source_split'] == 'val' and metric['uses_test_split'] is False
        assert metric['updates_model_or_memory'] is False and metric['sample_count'] == 1648
        assert metric['evaluation_rng_protocol'] == 'stage22_34_crn_v1'
        assert metric['artifact_status'] == 'PASS' and all(metric['artifact_checks'].values())
    assert metrics[0]['mask_asset_sha256'] == metrics[1]['mask_asset_sha256']
    with np.load(runs[0] / 'validation_equal_rate_multiscale_samples.npz', allow_pickle=False) as a, \
            np.load(runs[1] / 'validation_equal_rate_multiscale_samples.npz', allow_pickle=False) as b:
        for key in ('sample_index', 'complete_count', 'history_mask_packed', 'l1_mask_packed',
                    'l5_mask_packed', 'l20_mask_packed', 'target_removed_count',
                    'removed_count_l1', 'removed_count_l5', 'removed_count_l20'):
            assert np.array_equal(a[key], b[key]), f'DATA_ERROR unpaired evaluation asset: {key}'
        assert np.array_equal(a['sample_index'], np.arange(1648))
        for arrays, metric in zip((a, b), metrics):
            for route in ('on', 'multiscale'):
                values = np.array([arrays[f'complete_sse_{route}'].sum(), arrays[f'complete_sae_{route}'].sum()])
                values /= arrays['complete_count'].sum()
                np.testing.assert_allclose(values, [metric[f'complete_target_{m}_{route}'] for m in ('mse', 'mae')],
                                           rtol=0, atol=1e-12)
    results = {}
    for label, route in (('primary_original_path', 'on'), ('secondary_inherited_CNM', 'multiscale')):
        before = {m: metrics[0][f'complete_target_{m}_{route}'] for m in ('mse', 'mae')}
        after = {m: metrics[1][f'complete_target_{m}_{route}'] for m in ('mse', 'mae')}
        results[label] = dict(baseline=before, fixed=after,
            absolute_change={m: after[m] - before[m] for m in before},
            relative_gain_percent={m: (before[m] - after[m]) / before[m] * 100 for m in before},
            dual_win=all(after[m] < before[m] for m in before))
    return dict(fixed_evaluation_assets='PASS', metrics_roundtrip='PASS', sample_count=1648,
                baseline_eval=str(runs[0]), fixed_eval=str(runs[1]), **results)


def analyze():
    state = h.read(STATE) if STATE.exists() else dict(queue_state='NOT_STARTED', jobs={})
    result = dict(stage='22.40', queue_state=state['queue_state'], completed=False,
        decision='WAITING_FOR_BUG_FIX_CONTROL', new_research_innovation=False,
        test_accessed=False, original_paper_same_protocol=False,
        original_paper_note='Traffic official hyperparameters unavailable; compare fixed local baseline, not claim paper reproduction',
        jobs=[{key: job.get(key) for key in ('id', 'state', 'run_dir', 'attempt_count', 'oom_retry_count',
              'required_memory_estimate_mib', 'failure_summary')} for job in state['jobs'].values()])
    if state['queue_state'] == 'FAILED':
        result['decision'] = 'EXECUTION_STOP'
    if all(state['jobs'].get(name, {}).get('state') == 'SUCCESS' for name in JOBS):
        try:
            training = Path(state['jobs']['fixed_train_formal']['run_dir'])
            result['pairing_checks'] = verify_training_pair(h.read(training / 'metrics.json'))
            result['comparison'] = compare_evaluations(BASELINE_EVAL,
                state['jobs']['fixed_eval_formal']['run_dir'])
            old, new = h.read(BASELINE_TRAIN / 'metrics.json'), h.read(training / 'metrics.json')
            result['cost'] = {family: {key: data.get(key) for key in ('parameter_count', 'epochs_completed',
                'optimizer_steps', 'validation_elapsed_seconds', 'peak_gpu_memory_mb',
                'peak_gpu_reserved_mb')} for family, data in (('original', old), ('fixed', new))}
            for family, run in (('original', BASELINE_TRAIN), ('fixed', training)):
                result['cost'][family]['resource_usage'] = h.read(run / 'resource_usage.json')
            result['completed'] = True
            result['decision'] = ('BUG_FIX_DIRECTIONAL_GAIN_REQUIRES_REPLICATION_NOT_INNOVATION'
                if result['comparison']['primary_original_path']['dual_win'] else
                'BUG_FIX_NO_DUAL_GAIN_ANALYZE_DYNAMICS_NO_AUTO_EXPANSION')
        except Exception as error:
            result.update(decision='PAIRING_OR_ARTIFACT_ERROR', error=repr(error))
    report_json = ROOT / 'reports/stage22_40_distance_fix.json'
    report_md = ROOT / 'reports/stage22_40_distance_fix.md'
    h.write_json(report_json, result)
    lines = ['# 步骤22.40：最小距离标准化修复', '', f'状态：{result["decision"]}', '',
             '仅validation（验证集）；这是基线错误修复，不是研究创新。',
             '主要指标只取固定CRN（共同随机数）原始单路径；不混用普通训练后验证或CNM集成指标。', '']
    if 'comparison' in result:
        lines += ['| 路径 | 方法 | MSE（均方误差） | MAE（平均绝对误差） |',
                  '| --- | --- | ---: | ---: |']
        for route in ('primary_original_path', 'secondary_inherited_CNM'):
            for family in ('baseline', 'fixed'):
                metric = result['comparison'][route][family]
                lines.append(f'| {route} | {family} | {metric["mse"]:.9f} | {metric["mae"]:.9f} |')
    lines += ['', '```json', json.dumps(result, ensure_ascii=False, indent=2), '```', '']
    h.atomic_text(report_md, '\n'.join(lines))
    subprocess.run([sys.executable, str(ROOT / 'scripts/aggregate_results.py')], cwd=ROOT, check=False)
    if result['completed']:
        archive = ROOT / 'reports/archive' / f'stage22_40_final__{time.strftime("%Y%m%d_%H%M%S")}'
        for path in (report_json, report_md, TESTS, ROOT / 'reports/stage22_40_queue_tests.json',
                     ROOT / 'reports/latest_results.csv', ROOT / 'reports/latest_results.md'):
            if path.exists():
                frozen(archive / path.name, path.read_text())
    print(result['decision'], flush=True)
    return int(result['decision'] == 'PAIRING_OR_ARTIFACT_ERROR')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--scheduler-attempt', type=int, default=0)
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return 0
    if args.config:
        return execute(args.config, args.gpu_uuid, args.scheduler_attempt)
    return analyze()


if __name__ == '__main__':
    raise SystemExit(main())
