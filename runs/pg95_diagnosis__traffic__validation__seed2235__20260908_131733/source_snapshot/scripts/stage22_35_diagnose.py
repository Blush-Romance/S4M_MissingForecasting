#!/usr/bin/env python3
"""CPU-only descriptive PG95 diagnosis; never load test arrays or train."""
import argparse
import ast
import copy
import datetime as dt
import json
import os
from pathlib import Path
import random
import re
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

import stage22_34_crn as recovery
stage = recovery.stage
ROOT = stage.ROOT
PROTOCOL = ROOT / 'docs/STAGE22_35_DIAGNOSTIC_PROTOCOL_ZH.md'


def parsed_namespace(log):
    line = next(x for x in log.splitlines() if x.startswith('Namespace('))
    node = ast.parse(line, mode='eval').body
    assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Namespace'
    values = {}
    for item in node.keywords:
        if item.arg == 'device':
            continue
        values[item.arg] = ast.literal_eval(item.value)
    values['device'] = torch.device('cpu')
    return SimpleNamespace(**values)


def geometry(representations):
    x = representations.detach().cpu().double().reshape(representations.shape[0], -1)
    assert bool(torch.isfinite(x).all()) and len(x) > 0
    normalized = torch.nn.functional.normalize(x, dim=1)
    cosine = normalized @ normalized.T
    pairs = cosine[torch.triu(torch.ones_like(cosine, dtype=torch.bool), diagonal=1)]
    spectrum = torch.linalg.svdvals(x - x.mean(0)).square()
    total = float(spectrum.sum())
    if total > 1e-20:
        p = spectrum[spectrum > 0] / spectrum.sum()
        rank = float(torch.exp(-(p * p.log()).sum()))
    else:
        rank = 0.
    return dict(count=len(x), centered_effective_rank=rank,
                pairwise_cosine_mean=float(pairs.mean()) if len(pairs) else None,
                pairwise_cosine_quantiles=torch.quantile(pairs, torch.tensor([.1,.5,.9],dtype=torch.float64)).tolist() if len(pairs) else [])


def summarize_group(a, b, route, selected):
    count = a['complete_count'][selected].sum()
    if count == 0:
        return dict(windows=0, mse_change_percent=None, mae_change_percent=None)
    result = dict(windows=int(selected.sum()))
    for m, accumulator in (('mse','sse'), ('mae','sae')):
        x = a[f'complete_{accumulator}_{route}'][selected]
        y = b[f'complete_{accumulator}_{route}'][selected]
        result[f's4m_{m}'] = float(x.sum()/count)
        result[f'pg95_{m}'] = float(y.sum()/count)
        result[f'{m}_change_percent'] = float((y.sum()-x.sum()) / x.sum()*100)
        result[f'{m}_worse_window_fraction'] = float(np.mean(y>x))
    return result


def initial_state_probe(seed, a_args, b_args):
    paths = ['baselines/S4M/s4m', 'baselines/S4M/s4m/model',
             'baselines/S4M/s4m/model/S4/s4', 'baselines/S4M/s4m/model/S4/s4/extensions/kernels']
    sys.path[:0] = [str(ROOT / p) for p in paths]
    from model.S4M import Model
    states = []
    for args in (a_args, b_args):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        model = Model(copy.deepcopy(args)).cpu()
        states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
        del model
    common = sorted(states[0].keys() & states[1].keys())
    changed = [k for k in common if not torch.equal(states[0][k], states[1][k])]
    return dict(common_state_entries=len(common), changed_common_entries=len(changed), changed_keys=changed,
                identical_keys=[k for k in common if k not in changed],
                scope='current_constructor_replay_not_archived_historical_initial_weights')


def analyze():
    report_path = ROOT / 'reports/stage22_34_crn_factorial.json'
    report = stage.read(report_path)
    assert report['queue_state'] == 'SUCCESS' and not report['artifact_errors']
    records, hashes = [], {str(report_path):stage.digest(report_path)}
    for r in report['per_seed']:
        seed = r['seed']
        assert stage.compare_seed(seed, Path(r['sources']['s4m']), Path(r['sources']['pg95'])) == r
        family_rows, arrays, args_list = {}, [], []
        official = []
        for family in ('s4m','pg95'):
            run = Path(r['sources'][family])
            cfg = yaml.safe_load((run/'resolved_config.yaml').read_text())
            source = Path(cfg['source']['run_dir'])
            m = stage.read(source/'metrics.json')
            assert m['test_split_accessed'] is False and m['evaluation_split'] == 'validation'
            training = yaml.safe_load((source/'resolved_config.yaml').read_text())
            official.append(training['official_args'])
            log = (source/'train.log').read_text()
            args_list.append(parsed_namespace(log))
            curve = [dict(epoch=int(e), steps=int(s), train_loss=float(t), observed_validation_loss=float(v))
                     for e,s,t,v in re.findall(r'Epoch: (\d+), Steps: (\d+) \| Train Loss: ([\d.eE+-]+) Vali Loss: ([\d.eE+-]+)',log)]
            assert len(curve) == m['epochs_completed']
            checkpoint = torch.load(cfg['source']['checkpoint'], map_location='cpu')
            g = geometry(checkpoint['repre'])
            del checkpoint
            evidence_keys = ('epochs_completed','optimizer_steps','parameter_count','best_observed_validation_loss',
                'mipp_steps','mipp_total_loss_mean','mipp_invariance_loss_mean','mipp_paired_content_cosine_mean',
                'mipp_stable_write_acceptance_rate_mean','mipp_warm_write_acceptance_rate_mean','mipp_write_updated_count_mean')
            family_rows[family] = dict(source=str(source), training_curve=curve,
                best_logged_epoch=min(curve,key=lambda x:x['observed_validation_loss'])['epoch'],
                training_evidence={k:m.get(k) for k in evidence_keys}, prototype_geometry=g)
            cache = run/'validation_equal_rate_multiscale_samples.npz'
            with np.load(cache,allow_pickle=False) as z:
                arrays.append({k:z[k] for k in ('sample_index','history_mask_packed','complete_count',
                    'complete_sse_on','complete_sae_on','complete_sse_multiscale','complete_sae_multiscale')})
            for path in (cache, run/'metrics.json', run/'resolved_config.yaml', source/'train.log',
                         source/'metrics.json', source/'resolved_config.yaml', Path(cfg['source']['checkpoint'])):
                hashes[str(path)] = stage.digest(path)
        skip = {'model_id','des','checkpoints','results_root','test_results_root','metrics_path'}
        differences = {k:[official[0].get(k),official[1].get(k)] for k in official[0].keys()|official[1].keys()
                       if official[0].get(k)!=official[1].get(k) and k not in skip}
        unexpected = [k for k in differences if not (k.startswith('mipp_') or k in
                      ('use_mipp','use_mask_intervention_pair','use_content_mask_factorization','use_stability_write_gate'))]
        assert not unexpected, unexpected
        n = len(arrays[0]['sample_index'])
        missing = 1-np.unpackbits(arrays[0]['history_mask_packed'],axis=1)[:,:192].mean(1)
        groups = {'all':np.ones(n,dtype=bool)}
        for i,indices in enumerate(np.array_split(np.arange(n),4)):
            group = np.zeros(n,dtype=bool); group[indices]=True
            groups[f'time_quarter_{i+1}']=group
        for lo,hi in ((0.,.2),(.2,.3),(.3,1.000001)):
            groups[f'missing_{lo}_{min(hi,1.)}']=(missing>=lo)&(missing<hi)
        strata = {route:{name:summarize_group(*arrays,route,sel) for name,sel in groups.items()}
                  for route in ('on','multiscale')}
        probe = initial_state_probe(seed,*args_list)
        records.append(dict(seed=seed, sources=family_rows, training_configuration_differences=differences,
                            unexpected_training_differences=unexpected, error_strata=strata, initialization_probe=probe))
        print(f'diagnosed seed={seed} common_initial_entries_changed={probe["changed_common_entries"]}',flush=True)
    return dict(stage='22.35', status='SUCCESS', uses_test_split=False, uses_new_training=False,
        decision='MECHANISM_CAUSALITY_NOT_IDENTIFIED_PAIRED_CONTROL_NEEDED', records=records, source_hashes=hashes,
        limitations=['overlapping windows are not independent observations',
                     'training diagnostics are averages over all epochs, not only best checkpoint',
                     'same seed does not imply same common initialization or training RNG stream',
                     'current constructor replay is not archived historical initialization',
                     'prototype geometry and grouped errors do not establish causal mechanism'])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    torch.set_num_threads(2)
    stamp=dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    name=f'pg95_diagnosis__traffic__validation__seed2235{"__smoke" if args.smoke else ""}__{stamp}'
    run=ROOT/'runs'/name; run.mkdir(parents=True,exist_ok=False)
    start=time.time()
    stage.write_json(run/'status.json',dict(state='RUNNING',pid=os.getpid()))
    config=dict(experiment=dict(id=name,method='pg95_diagnosis',dataset='traffic',seed=2235,smoke_test=args.smoke),
                analysis_seeds=[2026,2027,2028],device='cpu',protocol=str(PROTOCOL),protocol_sha256=stage.digest(PROTOCOL))
    stage.atomic_text(run/'resolved_config.yaml',yaml.safe_dump(config))
    stage.atomic_text(run/'command.sh',f'cd {ROOT}\n{sys.executable} scripts/stage22_35_diagnose.py'+(' --smoke' if args.smoke else '')+'\n')
    stage.atomic_text(run/'env.txt',f'python={sys.version}\ntorch={torch.__version__}\ncuda_allocated=False\n')
    source_files=[Path(__file__),ROOT/'scripts/test_stage22_35_diagnose.py',PROTOCOL,
        ROOT/'scripts/stage22_34_crn.py', ROOT/'scripts/stage22_34_factorial.py',
        ROOT/'baselines/S4M/s4m/model/S4M.py',ROOT/'baselines/S4M/s4m/model/Bank.py',
        ROOT/'baselines/S4M/s4m/model/MIPP.py',ROOT/'baselines/S4M/s4m/model/Memory.py']
    for p in source_files:
        stage.atomic_text(run/'source_snapshot'/p.relative_to(ROOT),p.read_text())
    stage.write_json(run/'code_manifest.json',{str(p.relative_to(ROOT)):stage.digest(p) for p in source_files})
    try:
        if args.smoke:
            z={'complete_count':np.ones(4),'complete_sse_on':np.ones(4),'complete_sae_on':np.ones(4)}
            result=summarize_group(z,z,'on',np.ones(4,dtype=bool))
            assert result['mse_change_percent']==result['mae_change_percent']==0
            payload=dict(status='SUCCESS',smoke_test=True,synthetic_result=result,uses_test_split=False)
        else:
            assert stage.read(ROOT/'reports/stage22_35_test.json')['passed']
            assert stage.read(ROOT/'reports/stage22_35_smoke.json')['status']=='SUCCESS'
            payload=analyze()
        payload.update(experiment_id=name,method='pg95_diagnosis',dataset='traffic',evaluation_split='validation',
                       smoke_test=args.smoke,run_dir=str(run))
        stage.write_json(run/'metrics.json',payload)
        stage.write_json(ROOT/f'reports/stage22_35_{"smoke" if args.smoke else "diagnosis"}.json',payload)
        stage.write_json(run/'status.json',dict(state='SUCCESS',return_code=0))
        stage.atomic_text(run/'train.log',f'CPU descriptive diagnosis SUCCESS; elapsed={time.time()-start:.3f}s\n')
        print(f'run_dir={run}',flush=True)
    except Exception as e:
        stage.write_json(run/'status.json',dict(state='CODE_ERROR',return_code=1,error=repr(e)))
        stage.atomic_text(run/'train.log',repr(e)+'\n')
        raise
    finally:
        stage.write_json(run/'resource_usage.json',dict(elapsed_seconds=time.time()-start,
            peak_cpu_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,peak_gpu_memory_mb=0))


if __name__=='__main__':
    main()
