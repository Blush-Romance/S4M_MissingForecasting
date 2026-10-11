#!/usr/bin/env python3
"""Frozen two-seed extension of 22.36; validation only, no model changes."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml
import stage22_34_factorial as h
import generate_stage22_22_traffic_assets as assets
from run_s4m_experiment import validate_experiment_config as validate_train
from run_stage14_uncertainty_cache import validate_config as validate_eval
from run_stage22_22_traffic_pipeline import materialize_evaluation_config

ROOT = h.ROOT
RT = ROOT / 'runtime/stage22_37_control'
STATE = RT / 'state.json'
CONFIG = ROOT / 'configs/stage22_37_control'
MANIFEST = RT / 'config_manifest.json'
TESTS = ROOT / 'reports/stage22_37_tests.json'
SEEDS = (2028, 2029)
SCOPE = 'stage22_37_paired_training_CRN_validation'


def frozen(path, text):
    if path.exists() and path.read_text() != text:
        raise ValueError(f'CONFIG_ERROR frozen file differs: {path}')
    if not path.exists():
        h.atomic_text(path, text)


def audit_assets():
    assert h.digest(assets.SOURCE) == assets.EXPECTED_SOURCE_SHA256
    full = pd.read_csv(assets.SOURCE)
    assert full.shape == (17544, 863) and not full.iloc[:, 1:].isna().any().any()
    assert full.columns[0] == 'date' and full.columns[-1] == 'OT'
    results = {}
    for seed in SEEDS:
        missing, mask, meta = assets.output_paths(seed)
        exists = [p.exists() for p in (missing, mask, meta)]
        if not any(exists) and seed == 2029:
            assets.generate(full, seed)
        elif not all(exists):
            raise ValueError(f'DATA_ERROR partial/missing assets seed{seed}; do not overwrite')
        m = h.read(meta)
        assert h.digest(missing) == m['output_sha256']
        assert h.digest(mask) == m['mask_sha256']
        assert m['source_sha256'] == assets.EXPECTED_SOURCE_SHA256
        expected, starts, _ = assets.select_random_blocks(
            n_rows=len(full), block_length=5, target_rate=.26, seed=seed)
        with np.load(mask, allow_pickle=False) as a:
            np.testing.assert_array_equal(a['row_mask'], expected)
            np.testing.assert_array_equal(a['start_indices'], starts)
            assert int(a['seed']) == seed
        data = pd.read_csv(missing)
        assert data.columns.equals(full.columns) and data['date'].equals(full['date'])
        values = data.iloc[:, 1:].to_numpy()
        np.testing.assert_array_equal(np.isnan(values), np.broadcast_to(expected[:, None], values.shape))
        np.testing.assert_allclose(values[~expected], full.iloc[:, 1:].to_numpy()[~expected], rtol=1e-12, atol=1e-12)
        assert m['realized_missing_rate'] == float(expected.mean())
        results[str(seed)] = dict(metadata=m, audited=True, assets={
            str(p): h.digest(p) for p in (missing, mask, meta)})
    assert results['2028']['metadata']['mask_sha256'] != results['2029']['metadata']['mask_sha256']
    h.write_json(RT / 'asset_audit.json', results)
    return results


def prepare():
    if shutil.disk_usage(ROOT).free < 3 * 1024**3:
        raise RuntimeError('DATA_ERROR less than 3GiB disk available; no deletion authorized')
    audited = audit_assets()
    records = []
    for seed in SEEDS:
        for family in ('s4m', 'pg95'):
            for smoke in (True, False):
                tag = 'smoke' if smoke else 'formal'
                for kind in ('train', 'eval'):
                    template = ROOT / f'configs/stage22_36_control/{family}_{kind}_{tag}.yaml'
                    old = yaml.safe_load(template.read_text())
                    c = copy.deepcopy(old)
                    c['experiment'].update(seed=seed, purpose='frozen_paired_multiseed_replication',
                        protocol_scope=SCOPE if kind == 'eval' else f'stage22_37_paired_training_rng_v1_{tag}')
                    if 'realized_missing_rate' in c['experiment']:
                        c['experiment']['realized_missing_rate'] = audited[str(seed)]['metadata']['realized_missing_rate']
                    for key, value in c['data'].items():
                        if isinstance(value, str):
                            c['data'][key] = value.replace('seed2027', f'seed{seed}').replace('seed_2027', f'seed_{seed}')
                    if 'generation_seed' in c['data']: c['data']['generation_seed'] = seed
                    if 'mask_asset_sha256' in c['data']:
                        c['data']['mask_asset_sha256'] = audited[str(seed)]['metadata']['mask_sha256']
                    if kind == 'train':
                        c['official_args']['seed'] = seed
                        for key in ('model_id', 'des'):
                            c['official_args'][key] = c['official_args'][key].replace('2027', str(seed))
                        validate_train(c)
                    else:
                        c['diagnostic']['nested_seed'] = seed
                        c['source'] = dict(job_id=f'{family}_{seed}_train_{tag}', queue_state=str(STATE))
                    # Explicit before/after record of every changed top-level section.
                    changes = {k: dict(old=old.get(k), new=v, reason='user-approved frozen seed replication')
                               for k, v in c.items() if k != 'provenance' and old.get(k) != v}
                    c['provenance'] = dict(stage='22.37', user_approved=True, template=str(template),
                        template_sha256=h.digest(template), configuration_changes=changes)
                    p = CONFIG / f'{family}_{seed}_{kind}_{tag}.yaml'
                    frozen(p, yaml.safe_dump(c, sort_keys=False))
                    records.append(dict(config=str(p), sha256=h.digest(p)))
    frozen(MANIFEST, json.dumps(records, indent=2) + '\n')
    print('prepared 16 frozen configs; both assets independently audited')


def verify_frozen():
    manifest = h.read(MANIFEST)
    assert len(manifest) == 16
    for row in manifest:
        assert h.digest(row['config']) == row['sha256'], 'CONFIG_ERROR changed frozen config'
    return manifest


def execute(path, gpu, attempt):
    manifest = verify_frozen()
    path = (ROOT / path).resolve()
    assert any(r['config'] == str(path) for r in manifest)
    tests = h.read(TESTS)
    assert tests['passed'] and tests['manifest_sha256'] == h.digest(MANIFEST), 'CODE_ERROR tests missing/stale'
    for p, digest in tests['code_hashes'].items():
        assert h.digest(p) == digest, f'CODE_ERROR code changed since tests: {p}'
    c = yaml.safe_load(path.read_text())
    seed = c['experiment']['seed']
    asset = h.read(RT / 'asset_audit.json')[str(seed)]
    for p, digest in asset['assets'].items():
        assert h.digest(p) == digest, f'DATA_ERROR changed asset {p}'
    is_eval = 'diagnostic' in c
    if is_eval:
        path = materialize_evaluation_config(path, default_state_path=STATE,
            destination=RT / 'materialized' / f'{path.stem}_a{attempt}_{time.strftime("%Y%m%d_%H%M%S")}.yaml')
        effective = yaml.safe_load(path.read_text())
        source = yaml.safe_load(Path(effective['source']['resolved_config']).read_text())
        validate_eval(effective, source)
        assert effective['diagnostic']['split'] == 'val'
    else:
        validate_train(c)
        assert c['official_args']['post_train_evaluation'] == 'validation'
    if shutil.disk_usage(ROOT).free < 1024**3:
        raise RuntimeError('DATA_ERROR disk space below 1GiB; stop without deleting files')
    runner = ROOT / ('scripts/run_stage14_uncertainty_cache.py' if is_eval else 'scripts/run_s4m_experiment.py')
    run = None
    with subprocess.Popen([sys.executable, str(runner), '--config', str(path), '--gpu-uuid', gpu,
                           '--scheduler-attempt', str(attempt)], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as child:
        for line in child.stdout:
            print(line, end='', flush=True)
            if line.startswith('run_dir='):
                run = Path(line.strip().split('=', 1)[1])
                for p in tests['code_hashes']:
                    rel = Path(p).relative_to(ROOT)
                    h.atomic_text(run / 'stage22_37_source_snapshot' / rel, Path(p).read_text())
                h.write_json(run / 'stage22_37_code_manifest.json', tests['code_hashes'])
        code = child.wait()
    if code == 0 and not is_eval:
        m = h.read(run / 'metrics.json')
        assert m['paired_initial_common_sha256'] == tests['initial_hashes'][str(seed)], 'CODE_ERROR initialization hash'
        assert m['paired_training_rng_protocol'] == 'stage22_36_v1'
        assert m['test_split_accessed'] is False
        for key in ('paired_train_order', 'paired_validation_order'):
            assert len(m[key]) == m['epochs_completed']
    return code


def paired_record(seed, state):
    prefix = lambda family, kind: f'{family}_{kind}_formal' if seed == 2027 else f'{family}_{seed}_{kind}_formal'
    jobs = state['jobs']
    ids = [prefix(f, k) for k in ('train', 'eval') for f in ('s4m', 'pg95')]
    if not all(jobs.get(i, {}).get('state') == 'SUCCESS' for i in ids): return None
    a, b = [h.read(Path(jobs[i]['run_dir']) / 'metrics.json') for i in ids[:2]]
    assert a['paired_initial_common_sha256'] == b['paired_initial_common_sha256']
    for key in ('paired_train_order', 'paired_validation_order'):
        n = min(len(a[key]), len(b[key]))
        assert n > 0 and a[key][:n] == b[key][:n]
    h.CRN_RECOVERY = True
    r = h.compare_seed(seed, *[Path(jobs[i]['run_dir']) for i in ids[2:]])
    r.update(pairing_check='PASS', actual_epochs=[a['epochs_completed'], b['epochs_completed']],
             training_sources={f: jobs[ids[j]]['run_dir'] for j, f in enumerate(('s4m', 'pg95'))})
    return r


def summarize(records):
    cells = {}
    for cell in ('A_s4m', 'B_pg95', 'C_s4m_cnm', 'D_pg95_cnm'):
        cells[cell] = {metric: dict(mean=float(np.mean([r['cells'][cell][metric] for r in records])),
            sample_std=float(np.std([r['cells'][cell][metric] for r in records], ddof=1)))
            for metric in ('mse', 'mae')}
    gains = {}
    for key in ('pg95_gain_without_cnm_percent', 'pg95_gain_with_cnm_percent',
                'cnm_gain_on_s4m_percent', 'cnm_gain_on_pg95_percent'):
        x = np.asarray([r[key] for r in records])
        gains[key] = dict(mean=x.mean(0).tolist(), sample_std=x.std(0, ddof=1).tolist(),
                         dual_wins=int((x > 0).all(1).sum()))
    return dict(cells=cells, gains=gains, metric_order=['mse', 'mae'])


def analyze():
    state = h.read(STATE) if STATE.exists() else dict(queue_state='NOT_STARTED', jobs={})
    records, errors = [], []
    for seed in (2027, *SEEDS):
        try:
            s = h.read(ROOT / 'runtime/stage22_36_control/state.json') if seed == 2027 else state
            record = paired_record(seed, s)
            if record: records.append(record)
        except Exception as e: errors.append(dict(seed=seed, error=repr(e)))
    complete = len(records) == 3 and not errors and state['queue_state'] == 'SUCCESS'
    result = dict(stage='22.37', queue_state=state['queue_state'], completed=complete,
        decision='PAIRED_THREE_SEED_COMPLETE' if complete else 'WAITING_FOR_PAIRED_REPLICATION',
        test_accessed=False, new_module=False, randomness='joint_training_and_missing_mask',
        per_seed=records, errors=errors, jobs=list(state['jobs'].values()))
    if errors: result['decision'] = 'PAIRING_OR_ARTIFACT_ERROR'
    elif state['queue_state'] == 'FAILED': result['decision'] = 'EXECUTION_STOP'
    if complete:
        result['summary'] = summarize(records)
        wins = result['summary']['gains']['pg95_gain_with_cnm_percent']['dual_wins']
        result['scientific_conclusion'] = ('PG95_INCREMENT_CONSISTENT_DEVELOPMENT_ONLY' if wins == 3
                                           else 'PG95_INCREMENT_NOT_CONSISTENT')
    h.write_json(ROOT / 'reports/stage22_37_control.json', result)
    lines = ['# 步骤22.37 配对三seed复核', '', f'状态：{result["decision"]}；完整结束：{complete}',
             '', '仅validation；联合训练/掩码随机性；不访问test。', '',
             '| seed | 方法 | MSE | MAE |', '| --- | --- | ---: | ---: |']
    for r in records:
        for name, value in r['cells'].items():
            lines.append(f'| {r["seed"]} | {name} | {value["mse"]:.9f} | {value["mae"]:.9f} |')
    lines += ['', '```json', json.dumps({k: v for k, v in result.items() if k not in ('jobs', 'per_seed')},
                                       ensure_ascii=False, indent=2), '```', '']
    h.atomic_text(ROOT / 'reports/stage22_37_control.md', '\n'.join(lines))
    subprocess.run([sys.executable, str(ROOT / 'scripts/aggregate_results.py')], cwd=ROOT, check=False)
    print(result['decision'], flush=True)
    return int(bool(errors))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare', action='store_true'); p.add_argument('--config')
    p.add_argument('--gpu-uuid'); p.add_argument('--scheduler-attempt', type=int, default=0)
    a = p.parse_args()
    if a.prepare: prepare(); return 0
    if a.config: return execute(a.config, a.gpu_uuid, a.scheduler_attempt)
    return analyze()


if __name__ == '__main__': raise SystemExit(main())
