#!/usr/bin/env python3
"""Two additional, frozen seeds for the isolated minimum-distance bug fix."""
import argparse
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
import stage22_40_distance_fix as prior
from run_s4m_experiment import validate_experiment_config as validate_train
from run_stage14_uncertainty_cache import validate_config as validate_eval
from run_stage22_22_traffic_pipeline import materialize_evaluation_config

ROOT = h.ROOT
RT = ROOT / 'runtime/stage22_41_distance_fix'
STATE = RT / 'state.json'
CONFIG = ROOT / 'configs/stage22_41_distance_fix'
MANIFEST = RT / 'config_manifest.json'
TESTS = ROOT / 'reports/stage22_41_tests.json'
BASELINE_STATE = ROOT / 'runtime/stage22_37_control/state.json'
SEEDS = (2028, 2029)
FIX_FLAG = prior.FIX_FLAG
JOBS = tuple(f'fixed_{seed}_{kind}_{tag}' for seed in SEEDS
             for tag in ('smoke', 'formal') for kind in ('train', 'eval'))
frozen = prior.frozen
namespace_values = prior.namespace_values
compare_evaluations = prior.compare_evaluations


def template_path(seed, kind, tag):
    return ROOT / f'configs/stage22_37_control/s4m_{seed}_{kind}_{tag}.yaml'


def baseline_run(seed, kind, tag='formal'):
    job = h.read(BASELINE_STATE)['jobs'][f's4m_{seed}_{kind}_{tag}']
    assert job['state'] == 'SUCCESS', 'DATA_ERROR incomplete existing baseline'
    return Path(job['run_dir'])


def validate_delta(c):
    """One changed scientific factor; all per-seed settings remain frozen."""
    seed = c['experiment']['seed']
    assert seed in SEEDS, 'CONFIG_ERROR unsupported seed'
    kind = 'eval' if 'diagnostic' in c else 'train'
    tag = 'smoke' if c['experiment']['smoke_test'] else 'formal'
    template = template_path(seed, kind, tag)
    old = yaml.safe_load(template.read_text())
    assert c['provenance']['template_sha256'] == h.digest(template), 'CONFIG_ERROR changed template'
    assert c['provenance']['fix_enabled'] is True
    assert c['data'] == old['data'], 'CONFIG_ERROR changed data protocol'
    assert c['runtime'] == old['runtime'], 'CONFIG_ERROR changed runtime template'
    expected = copy.deepcopy(old['experiment'])
    for key in ('model', 'purpose') + (('protocol_scope',) if kind == 'train' else ()):
        expected[key] = c['experiment'][key]
    assert c['experiment'] == expected, 'CONFIG_ERROR changed scientific metadata'
    if kind == 'train':
        args = copy.deepcopy(c['official_args'])
        assert args.pop(FIX_FLAG) is True
        for key in ('model_id', 'des'):
            args[key] = old['official_args'][key]
        assert args == old['official_args'], 'CONFIG_ERROR changed training settings'
        validate_train(c)
        assert args['post_train_evaluation'] == 'validation'
        assert args['stage22_36_paired_training_rng'] is True
    else:
        assert c['diagnostic'] == old['diagnostic'], 'CONFIG_ERROR changed CRN evaluation'
        assert c['experiment']['method'] == 's4m_baseline'
        assert c['diagnostic']['split'] == 'val'
        assert c['source'] == dict(job_id=f'fixed_{seed}_train_{tag}', queue_state=str(STATE))
    return c


def audit_baseline(seed):
    training, evaluation = baseline_run(seed, 'train'), baseline_run(seed, 'eval')
    for run in (training, evaluation):
        status = h.read(run / 'status.json')
        assert status['state'] == 'SUCCESS' and status['return_code'] == 0
    a, b = h.read(training / 'metrics.json'), h.read(evaluation / 'metrics.json')
    assert a['test_split_accessed'] is False and a['paired_training_rng_protocol'] == 'stage22_36_v1'
    assert a['parameter_count'] == 28371290
    assert b['source_split'] == 'val' and b['uses_test_split'] is False
    assert b['evaluation_rng_protocol'] == 'stage22_34_crn_v1'
    assert b['source_checkpoint_sha256'] == h.digest(a['checkpoint'])
    template = yaml.safe_load(template_path(seed, 'train', 'formal').read_text())
    actual = yaml.safe_load((training / 'resolved_config.yaml').read_text())
    assert actual['data'] == template['data']
    assert all(actual['official_args'][key] == value for key, value in template['official_args'].items())
    paths = [training / name for name in ('resolved_config.yaml', 'metrics.json', 'status.json')]
    paths += [evaluation / name for name in ('resolved_config.yaml', 'metrics.json', 'status.json',
                                            'validation_equal_rate_multiscale_samples.npz')]
    paths.append(Path(a['checkpoint']))
    paths.extend(ROOT / template['data'][name] for name in
                 ('missing_csv', 'complete_csv', 'mask_asset', 'mask_metadata'))
    return dict(seed=seed, baseline_train=str(training), baseline_eval=str(evaluation),
        frozen_artifacts={str(path): h.digest(path) for path in paths},
        expected_common_sha256=a['paired_initial_common_sha256'],
        parameter_count=a['parameter_count'], historical_reserved_peak_mib=a['peak_gpu_reserved_mb'],
        baseline_actual_epochs=a['epochs_completed'], baseline_retrained=False,
        no_test_access=True, primary_metric_keys=['complete_target_mse_on', 'complete_target_mae_on'],
        baseline_reuse_requires='22.40_default_off_equivalence_plus_per_seed_initialization_and_sampler_checks')


def prepare():
    if shutil.disk_usage(ROOT).free < 1024**3:
        raise RuntimeError('DATA_ERROR disk below 1GiB; no deletion authorized')
    previous = h.read(ROOT / 'reports/stage22_40_distance_fix.json')
    assert previous['completed'] and previous['queue_state'] == 'SUCCESS'
    assert previous['comparison']['primary_original_path']['dual_win']
    baseline = {str(seed): audit_baseline(seed) for seed in SEEDS}
    baseline['2027'] = dict(previous_report=str(ROOT / 'reports/stage22_40_distance_fix.json'),
        frozen_artifacts={str(ROOT / 'reports/stage22_40_distance_fix.json'):
                          h.digest(ROOT / 'reports/stage22_40_distance_fix.json')},
        baseline_eval=previous['comparison']['baseline_eval'],
        fixed_eval=previous['comparison']['fixed_eval'])
    frozen(RT / 'baseline_manifest.json', json.dumps(baseline, indent=2) + '\n')
    records = []
    for name in JOBS:
        _, seed_string, kind, tag = name.split('_')
        seed = int(seed_string)
        template = template_path(seed, kind, tag)
        old = yaml.safe_load(template.read_text())
        c = copy.deepcopy(old)
        c['experiment'].update(model='s4m_min_distance_fixed',
            purpose='multiseed_isolated_bug_fix_replication_not_research_innovation')
        if kind == 'train':
            c['experiment']['protocol_scope'] = f'stage22_41_distance_bug_fix_{tag}'
            c['official_args'][FIX_FLAG] = True
            c['official_args']['model_id'] = f't2241_fixed_{seed}_{tag[0]}'
            c['official_args']['des'] = f'T2241FIXED{seed}{tag[0].upper()}'
        else:
            c['source'] = dict(job_id=f'fixed_{seed}_train_{tag}', queue_state=str(STATE))
        changes = {key: dict(old=old.get(key), new=value,
                    reason='approved two-seed replication of isolated minimum-distance bug fix; identity only otherwise')
                   for key, value in c.items() if key != 'provenance' and old.get(key) != value}
        c['provenance'] = dict(stage='22.41', user_approved=True, fix_enabled=True,
            template=str(template), template_sha256=h.digest(template), configuration_changes=changes,
            baseline_manifest=str(RT / 'baseline_manifest.json'),
            scientific_classification='baseline_bug_fix_not_innovation',
            primary_comparison='original_on_path_CRN_validation_only',
            unchanged_evaluation_routes='inherited_frozen_L1_L5_L20_CNM_secondary_not_new_mechanism')
        validate_delta(c)
        path = CONFIG / f'{name}.yaml'
        frozen(path, yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
        records.append(dict(config=str(path), sha256=h.digest(path)))
    frozen(MANIFEST, json.dumps(records, indent=2) + '\n')
    print('prepared 8 frozen jobs; seeds2028/2029, unchanged budgets, validation only', flush=True)


def verify_frozen():
    manifest = h.read(MANIFEST)
    assert len(manifest) == 8
    for row in manifest:
        assert h.digest(row['config']) == row['sha256'], 'CONFIG_ERROR changed frozen config'
        validate_delta(yaml.safe_load(Path(row['config']).read_text()))
    for baseline in h.read(RT / 'baseline_manifest.json').values():
        for path, digest in baseline['frozen_artifacts'].items():
            assert h.digest(path) == digest, f'DATA_ERROR changed baseline/data artifact {path}'
    return manifest


def verify_training_pair(metrics, seed, smoke=False):
    original = h.read(baseline_run(seed, 'train') / 'metrics.json')
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
            count = min(len(metrics[key]), len(original[key]))
            assert count > 0 and metrics[key][:count] == original[key][:count], f'CODE_ERROR unpaired sampler {key}'
    return dict(initialization='PASS', parameter_count='PASS', training_rng='PASS',
        sampler_prefix='SMOKE_TRUNCATED_NOT_FORMAL_PREFIX' if smoke else 'PASS',
        original_epochs=original['epochs_completed'], fixed_epochs=metrics['epochs_completed'],
        identical_max_budget_and_early_stopping=True, identical_actual_epochs_or_time_not_required=True)


def save_effective_args(run, config):
    log_path = run / 'train.log'
    if not log_path.exists() or not any(line.startswith('Namespace(') for line in log_path.read_text().splitlines()):
        h.write_json(run / 'effective_parser_args.json', dict(status='NO_NAMESPACE_EMITTED'))
        return
    actual = namespace_values(log_path.read_text())
    kind = 'eval' if 'diagnostic' in config else 'train'
    tag = 'smoke' if config['experiment']['smoke_test'] else 'formal'
    historical = baseline_run(config['experiment']['seed'], kind, tag)
    previous = namespace_values((historical / 'train.log').read_text())
    previous.setdefault(FIX_FLAG, False)
    differences = {key: dict(old=previous.get(key), new=actual.get(key))
                   for key in previous.keys() | actual.keys() if previous.get(key) != actual.get(key)}
    identities = {'model_id', 'des', 'checkpoints', 'results_root', 'test_results_root',
                  'metrics_path', 'gpu', 'device', FIX_FLAG}
    if kind == 'eval':
        identities.update(('stage21_checkpoint', 'stage21_metrics_path', 'stage21_samples_path'))
    unexpected = {key: value for key, value in differences.items() if key not in identities}
    evidence = dict(status='PASS' if not unexpected else 'CONFIG_ERROR', historical_run=str(historical),
        actual=actual, differences=differences, unexpected_differences=unexpected)
    h.write_json(run / 'effective_parser_args.json', evidence)
    resolved_path = run / 'resolved_config.yaml'
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved['effective_parser_args'] = actual
    resolved['effective_parser_comparison'] = {key: value for key, value in evidence.items() if key != 'actual'}
    h.atomic_text(resolved_path, yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
    assert not unexpected, f'CONFIG_ERROR effective parser protocol changed: {unexpected}'
    assert actual[FIX_FLAG] is True


def execute(path, gpu, attempt):
    manifest = verify_frozen()
    path = (ROOT / path).resolve()
    assert any(row['config'] == str(path) for row in manifest), 'CONFIG_ERROR unknown config'
    tests = h.read(TESTS)
    assert tests['passed'] and tests['manifest_sha256'] == h.digest(MANIFEST), 'CODE_ERROR tests missing/stale'
    baselines = h.read(RT / 'baseline_manifest.json')
    for seed in SEEDS:
        assert tests['initial_hashes'][str(seed)] == baselines[str(seed)]['expected_common_sha256']
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
        assert source['official_args'][FIX_FLAG] is True
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
                    h.atomic_text(run / 'stage22_41_source_snapshot' / Path(source).relative_to(ROOT), content)
                h.write_json(run / 'stage22_41_code_manifest.json', tests['code_hashes'])
                h.write_json(run / 'stage22_41_control.json', dict(stage='22.41',
                    scientific_classification='baseline_bug_fix_not_innovation',
                    fix_enabled=True, frozen_config_sha256=h.digest(path), cpu_tests_sha256=h.digest(TESTS),
                    baseline_manifest_sha256=h.digest(RT / 'baseline_manifest.json'),
                    restart_policy='OOM_from_scratch_no_complete_training_resume_available'))
        code = child.wait()
    if run is not None:
        save_effective_args(run, config)
    if code == 0:
        assert run is not None, 'CODE_ERROR child produced no run_dir'
        metrics = h.read(run / 'metrics.json')
        if not is_eval:
            evidence = verify_training_pair(metrics, config['experiment']['seed'],
                                            smoke=config['experiment']['smoke_test'])
            h.write_json(run / 'stage22_41_pairing_checks.json', evidence)
        else:
            assert metrics['uses_test_split'] is False and metrics['updates_model_or_memory'] is False
            assert metrics['artifact_status'] == 'PASS' and all(metrics['artifact_checks'].values())
            assert metrics['evaluation_rng_protocol'] == 'stage22_34_crn_v1'
    return code


def summarize(records):
    result = dict(seed_count=len(records), seeds=[record['seed'] for record in records],
                  statistical_claim='descriptive_three_seeds_not_significance_or_journal_guarantee')
    for label in ('primary_original_path', 'secondary_inherited_CNM'):
        result[label] = dict(dual_win_count=sum(record['comparison'][label]['dual_win'] for record in records))
        for family in ('baseline', 'fixed', 'relative_gain_percent'):
            result[label][family] = {}
            for metric in ('mse', 'mae'):
                values = np.array([record['comparison'][label][family][metric] for record in records])
                result[label][family][metric] = dict(mean=float(values.mean()),
                    sample_std=float(values.std(ddof=1)) if len(values) > 1 else None,
                    values=values.tolist())
    return result


def analyze():
    state = h.read(STATE) if STATE.exists() else dict(queue_state='NOT_STARTED', jobs={})
    result = dict(stage='22.41', queue_state=state['queue_state'], completed=False,
        decision='WAITING_FOR_TWO_SEED_BUG_FIX_REPLICATION', new_research_innovation=False,
        test_accessed=False, original_paper_same_protocol=False,
        original_paper_note='Traffic official hyperparameters unavailable; fixed local baseline comparison only',
        jobs=[{key: job.get(key) for key in ('id', 'state', 'run_dir', 'attempt_count', 'oom_retry_count',
              'required_memory_estimate_mib', 'failure_summary')} for job in state['jobs'].values()])
    if state['queue_state'] == 'FAILED':
        result['decision'] = 'EXECUTION_STOP'
    if all(state['jobs'].get(name, {}).get('state') == 'SUCCESS' for name in JOBS):
        try:
            baselines = h.read(RT / 'baseline_manifest.json')
            previous = h.read(baselines['2027']['previous_report'])
            records = [dict(seed=2027, comparison=compare_evaluations(
                baselines['2027']['baseline_eval'], baselines['2027']['fixed_eval']),
                pairing_checks=previous['pairing_checks'], cost=previous['cost'])]
            for seed in SEEDS:
                training = Path(state['jobs'][f'fixed_{seed}_train_formal']['run_dir'])
                original = Path(baselines[str(seed)]['baseline_train'])
                a, b = h.read(original / 'metrics.json'), h.read(training / 'metrics.json')
                costs = {family: {key: metrics.get(key) for key in ('parameter_count', 'epochs_completed',
                    'optimizer_steps', 'validation_elapsed_seconds', 'peak_gpu_memory_mb', 'peak_gpu_reserved_mb')}
                    for family, metrics in (('original', a), ('fixed', b))}
                for family, run in (('original', original), ('fixed', training)):
                    costs[family]['resource_usage'] = h.read(run / 'resource_usage.json')
                records.append(dict(seed=seed, pairing_checks=verify_training_pair(b, seed), cost=costs,
                    comparison=compare_evaluations(baselines[str(seed)]['baseline_eval'],
                        state['jobs'][f'fixed_{seed}_eval_formal']['run_dir'])))
            result.update(completed=True, records=records, summary=summarize(records))
            result['decision'] = ('BUG_FIX_THREE_SEED_DUAL_GAIN_BASELINE_CANDIDATE_NOT_INNOVATION'
                if result['summary']['primary_original_path']['dual_win_count'] == 3 else
                'BUG_FIX_NOT_THREE_SEED_DUAL_GAIN_ANALYZE_NO_AUTO_EXPANSION')
        except Exception as error:
            result.update(decision='PAIRING_OR_ARTIFACT_ERROR', error=repr(error))
    report_json = ROOT / 'reports/stage22_41_distance_fix.json'
    report_md = ROOT / 'reports/stage22_41_distance_fix.md'
    h.write_json(report_json, result)
    lines = ['# 步骤22.41：最小距离修复三seed复核', '', f'状态：{result["decision"]}', '',
             '仅validation（验证集）；基线错误修复，不是创新模块。CRN（共同随机数）固定原始单路径为主要比较。',
             '先固定2028/2029方案；不根据结果选seed、调整模型、训练预算或评估口径。', '']
    if 'records' in result:
        lines += ['| Seed（随机种子） | 原始MSE | 修复MSE | MSE降低% | 原始MAE | 修复MAE | MAE降低% |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for record in result['records']:
            c = record['comparison']['primary_original_path']
            lines.append(f'| {record["seed"]} | {c["baseline"]["mse"]:.9f} | {c["fixed"]["mse"]:.9f} | '
                f'{c["relative_gain_percent"]["mse"]:.4f} | {c["baseline"]["mae"]:.9f} | '
                f'{c["fixed"]["mae"]:.9f} | {c["relative_gain_percent"]["mae"]:.4f} |')
    lines += ['', 'MSE：均方误差；MAE：平均绝对误差。均值与样本标准差仅作三seed描述，不作显著性或分区保证。',
              '', '```json', json.dumps(result, ensure_ascii=False, indent=2), '```', '']
    h.atomic_text(report_md, '\n'.join(lines))
    subprocess.run([sys.executable, str(ROOT / 'scripts/aggregate_results.py')], cwd=ROOT, check=False)
    if result['completed']:
        archive = ROOT / 'reports/archive' / f'stage22_41_final__{time.strftime("%Y%m%d_%H%M%S")}'
        for path in (report_json, report_md, TESTS, ROOT / 'reports/stage22_41_queue_tests.json',
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
