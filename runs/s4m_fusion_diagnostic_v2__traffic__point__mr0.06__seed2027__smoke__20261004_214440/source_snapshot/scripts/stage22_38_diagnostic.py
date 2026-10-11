#!/usr/bin/env python3
"""Prepare, dispatch and report frozen validation-only path interventions."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml
from stage22_38_common import ROOT, RT, STATE, CONFIG, MANIFEST, read, atomic_text, write_json, digest, namespace_values

TEST_REPORT = ROOT / 'reports/stage22_38_tests.json'
FILES = ['scripts/stage22_38_common.py', 'scripts/stage22_38_worker.py', 'scripts/stage22_38_diagnostic.py',
         'scripts/run_stage22_38_queue.py', 'scripts/start_stage22_38.sh',
         'scripts/test_stage22_38_diagnostic.py', 'docs/STAGE22_38_IMPUTATION_DIAGNOSTIC_ZH.md',
         'scripts/run_stage5_baseline_queue.py', 'baselines/S4M/s4m/model/S4M.py',
         'baselines/S4M/s4m/model/Bank.py', 'baselines/S4M/s4m/model/Memory.py',
         'baselines/S4M/s4m/data_provider/data_loader.py', 'baselines/S4M/s4m/utils/paired_eval_rng.py',
         'baselines/S4M/s4m/model/S4/s4/models/s4/s4.py',
         'baselines/S4M/s4m/model/S4/s4/models/s4/s4_mask1.py']


def frozen(path, text):
    if path.exists() and path.read_text() != text: raise ValueError(f'CONFIG_ERROR frozen file changed: {path}')
    if not path.exists(): atomic_text(path, text)


def prepare():
    report = read(ROOT / 'reports/stage22_37_control.json')
    assert report['completed'] and report['scientific_conclusion'] == 'PG95_INCREMENT_NOT_CONSISTENT'
    row = next(r for r in report['per_seed'] if r['seed'] == 2027)
    source_run = Path(row['training_sources']['s4m'])
    evaluation = Path(row['sources']['s4m'])
    source = yaml.safe_load((source_run/'resolved_config.yaml').read_text())
    training_metrics = read(source_run/'metrics.json')
    assert training_metrics['test_split_accessed'] is False
    assert source['official_args']['use_mipp'] is False
    for key, value in read(source_run/'source_snapshot_manifest.json').items():
        if key.startswith('baseline/'): assert digest(value['source']) == value['sha256'], f'CODE_ERROR changed baseline {key}'
    actual = namespace_values((source_run/'train.log').read_text())
    actual.update(root_path=str(ROOT/'data/traffic'), data_path='traffic_point_external_r0.06_l5_seed2027.csv',
                  true_data_path=str(ROOT/'data/traffic/traffic.csv'))
    assert (actual['batch_size'],actual['seq_len'],actual['pred_len'],actual['d_var']) == (16,192,96,862)
    source_paths = dict(run_dir=str(source_run), resolved_config=str(source_run/'resolved_config.yaml'),
        train_log=str(source_run/'train.log'), checkpoint=training_metrics['checkpoint'],
        evaluation_run=str(evaluation), cache=str(evaluation/'validation_equal_rate_multiscale_samples.npz'))
    hashes = {str(p): digest(p) for p in source_paths.values() if Path(p).is_file()}
    data = {k: str((ROOT/source['data'][k]).resolve()) for k in ('missing_csv','complete_csv','mask_asset')}
    hashes.update({p:digest(p) for p in data.values()})
    manifest = []
    for smoke in (True, False):
        tag = 'smoke' if smoke else 'formal'
        cfg = dict(experiment=dict(model='s4m_imputation_diagnostic',dataset='traffic',mask_type='point',
            nominal_missing_rate=.06,realized_missing_rate=source['experiment']['realized_missing_rate'],seed=2027,
            smoke_test=smoke,protocol_scope='stage22_38_frozen_path_intervention_'+tag,
            protocol_match='paired_CRN_validation_frozen_checkpoint_intervention_not_new_model'),
            source=source_paths, input_hashes=hashes, data=data, official_args=actual,
            diagnostic=dict(split='val',max_eval_batches=3 if smoke else -1,fills=['zero','prefix_mean','ffill'],
                paths=['full','query','direct','retrieval'],fixed_eval_rng=True,rng_protocol='stage22_34_crn_v1',
                baseline_metric_atol=1e-5),runtime=dict(device='cuda'),
            provenance=dict(stage='22.38',client_date='2026-10-04',user_approved=True,
                purpose='query_sensitivity_mechanism_diagnosis_no_training',
                unchanged_model_loss_and_checkpoint=True,complete_actual_defaults_from='source train.log Namespace',
                changes={'new_diagnostic_paths':dict(old='original',new='observed_only_fill_interventions',
                    reason='locate query/retrieval versus direct value sensitivity')}))
        p = CONFIG / f'seed2027_{tag}.yaml'
        frozen(p,yaml.safe_dump(cfg,sort_keys=False)); manifest.append(dict(config=str(p),sha256=digest(p)))
    frozen(MANIFEST,json.dumps(manifest,indent=2)+'\n')
    print('prepared 2 frozen configs; historical checkpoint and model source verified',flush=True)


def check_configs():
    manifest = read(MANIFEST)
    assert len(manifest)==2
    for row in manifest:
        assert digest(row['config']) == row['sha256'], 'CONFIG_ERROR manifest'
        c = yaml.safe_load(Path(row['config']).read_text())
        assert c['diagnostic']['split']=='val' and c['experiment']['seed']==2027
        assert c['official_args']['use_mipp'] is False
        assert c['diagnostic']['fills']==['zero','prefix_mean','ffill']
        assert c['diagnostic']['paths']==['full','query','direct','retrieval']
    return manifest


def attest():
    check_configs()
    assert read(ROOT/'reports/stage22_38_queue_tests.json')['self_test']=='PASS'
    tests=read(TEST_REPORT); assert tests['passed']
    tests.update(manifest_sha256=digest(MANIFEST),code_hashes={str(ROOT/p):digest(ROOT/p) for p in FILES})
    write_json(TEST_REPORT,tests); print('CPU tests and configuration/code hashes attested',flush=True)


def command_output(command,cwd=ROOT):
    r=subprocess.run(command,cwd=cwd,text=True,capture_output=True)
    return f'command={shlex.join(command)} exit={r.returncode}\n'+r.stdout+r.stderr


def execute(path,gpu,attempt):
    path=(ROOT/path).resolve(); check_configs()
    tests=read(TEST_REPORT)
    assert tests['passed'] and tests['manifest_sha256']==digest(MANIFEST),'CODE_ERROR stale tests'
    for p,value in tests['code_hashes'].items(): assert digest(p)==value, f'CODE_ERROR changed code {p}'
    cfg=yaml.safe_load(path.read_text())
    for p,value in cfg['input_hashes'].items(): assert digest(p)==value, f'DATA_ERROR changed input {p}'
    if shutil.disk_usage(ROOT).free<1024**3: raise RuntimeError('DATA_ERROR disk below 1GiB')
    stamp=dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    smoke='__smoke' if cfg['experiment']['smoke_test'] else ''
    identifier=f's4m_imputation_diagnostic__traffic__point__mr0.06__seed2027{smoke}__{stamp}'
    out=ROOT/'runs'/identifier
    out.mkdir()
    cfg['experiment'].update(id=identifier,created_at=dt.datetime.now(dt.timezone.utc).isoformat())
    cfg['runtime'].update(gpu_uuid=gpu,scheduler_attempt=attempt)
    snapshot=command_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free','--format=csv'])
    write_json(out/'status.json',dict(experiment_id=identifier,state='RUNNING',started_at=cfg['experiment']['created_at']))
    atomic_text(out/'resolved_config.yaml',yaml.safe_dump(cfg,sort_keys=False))
    atomic_text(out/'env.txt',f'Python={sys.executable}\nCUDA_VISIBLE_DEVICES={gpu}\nConda=s4m_missing_forecasting\n'+snapshot)
    repo=ROOT/'baselines/S4M'
    git='[baseline_adapter]\n'+''.join(command_output(c,repo) for c in
        (['git','rev-parse','HEAD'],['git','status','--short'],['git','diff','--stat']))
    atomic_text(out/'git_state.txt',git)
    diff=subprocess.run(['git','diff','--binary'],cwd=repo,capture_output=True,text=True,check=True).stdout
    atomic_text(out/'code.diff',diff)
    for p in FILES: atomic_text(out/'source_snapshot'/p,(ROOT/p).read_text())
    write_json(out/'source_snapshot_manifest.json',tests['code_hashes'])
    command=[sys.executable,'-u',str(ROOT/'scripts/stage22_38_worker.py'),
             '--config',str(out/'resolved_config.yaml'),'--run-dir',str(out)]
    atomic_text(out/'command.sh',f'CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} '+shlex.join(command)+'\n')
    environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',
        WANDB_MODE='disabled',MPLCONFIGDIR=str(out/'.matplotlib'),OMP_NUM_THREADS='2')
    print(f'run_dir={out}',flush=True)
    started=time.monotonic()
    with (out/'train.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        status=read(out/'status.json');status['pid']=child.pid;write_json(out/'status.json',status)
        code=child.wait()
    metrics=read(out/'metrics.json') if (out/'metrics.json').exists() else {}
    if metrics.get('environment'):
        atomic_text(out/'env.txt',(out/'env.txt').read_text()+json.dumps(metrics['environment'],indent=2)+'\n')
    log=(out/'train.log').read_text(errors='replace')
    verification=None
    if code==0:
        if not metrics or not all(metrics.get('checks',{}).values()): code=1;verification='CODE_ERROR incomplete diagnostic'
    state='SUCCESS' if code==0 else ('OOM' if 'out of memory' in log.lower() else 'CODE_ERROR')
    status.update(state=state,return_code=code,finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),verification_error=verification)
    write_json(out/'status.json',status)
    write_json(out/'resource_usage.json',dict(elapsed_seconds=time.monotonic()-started,return_code=code,
        peak_gpu_memory_mb=metrics.get('peak_gpu_allocated_mb'),peak_gpu_reserved_mb=metrics.get('peak_gpu_reserved_mb'),
        gpu_uuid=gpu,pre_run_snapshot=snapshot,post_run_snapshot=command_output(['nvidia-smi',
            '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free','--format=csv'])))
    print(f'finished state={state} return_code={code}',flush=True)
    return code


def analyze():
    s=read(STATE) if STATE.exists() else dict(queue_state='NOT_STARTED',jobs={})
    r=dict(stage='22.38',queue_state=s['queue_state'],completed=False,decision='WAITING_FOR_DIAGNOSIS',
           test_accessed=False,new_model=False,jobs=list(s['jobs'].values()))
    job=s['jobs'].get('seed2027_formal',{})
    if job.get('state')=='SUCCESS':
        p=Path(job['run_dir']);m=read(p/'metrics.json')
        assert m['sample_count']==1648 and m['artifact_status']=='PASS' and all(m['checks'].values())
        assert digest(p/'samples.npz')==m['samples_sha256']
        r.update(completed=True,decision='DIAGNOSTIC_COMPLETE_REVIEW_PATHS',run_dir=str(p),metrics=m)
    elif s['queue_state']=='FAILED': r['decision']='EXECUTION_STOP'
    write_json(ROOT/'reports/stage22_38_diagnostic.json',r)
    lines=['# 步骤22.38：冻结S4M填补敏感性诊断','',f'状态：{r["decision"]}',
        '', '仅seed2027 validation；干预结果不是重训模型收益。']
    if r['completed']:
        lines+=['','| 路径 | MSE | MAE | MSE变化收益% | MAE变化收益% | 预测差异MSE |',
                '| --- | ---: | ---: | ---: | ---: | ---: |']
        for name,value in r['metrics']['routes'].items():
            lines.append(f'| {name} | {value["mse"]:.9f} | {value["mae"]:.9f} | {value["mse_gain_percent"]:.4f} | {value["mae_gain_percent"]:.4f} | {value["prediction_difference_mse"]:.9f} |')
        lines+=['','正收益表示误差降低。query-only包含查询残差；retrieval-only固定查询及历史直达。',
                '完整分组、检查及表示漂移见同名JSON。重叠窗口不作为独立统计重复。']
    atomic_text(ROOT/'reports/stage22_38_diagnostic.md','\n'.join(lines)+'\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/aggregate_results.py')],cwd=ROOT,check=False)
    print(r['decision'],flush=True)
    return 0


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--attest',action='store_true')
    p.add_argument('--queue-test',action='store_true')
    p.add_argument('--config');p.add_argument('--gpu-uuid');p.add_argument('--scheduler-attempt',type=int,default=0)
    a=p.parse_args()
    if a.queue_test:
        result=subprocess.run([sys.executable,str(ROOT/'scripts/run_stage22_38_queue.py'),'--self-test'],
            cwd=ROOT,text=True,capture_output=True)
        print(result.stdout,flush=True)
        if result.returncode: print(result.stderr,flush=True);return result.returncode
        report=json.loads(result.stdout)
        write_json(ROOT/'reports/stage22_38_queue_tests.json',report)
        return 0 if report['self_test']=='PASS' else 1
    if a.prepare: prepare();return 0
    if a.attest: attest();return 0
    if a.config: return execute(a.config,a.gpu_uuid,a.scheduler_attempt)
    return analyze()


if __name__=='__main__': raise SystemExit(main())
