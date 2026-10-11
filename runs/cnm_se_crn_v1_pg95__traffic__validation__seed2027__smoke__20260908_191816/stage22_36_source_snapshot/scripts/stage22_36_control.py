#!/usr/bin/env python3
"""Freeze/dispatch paired RNG control and report new-protocol validation."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml
import stage22_34_factorial as helpers
from run_s4m_experiment import validate_experiment_config as validate_training
from run_stage14_uncertainty_cache import validate_config as validate_eval
from run_stage22_22_traffic_pipeline import materialize_evaluation_config

ROOT=helpers.ROOT
STATE=ROOT/'runtime/stage22_36_control/state.json'
CONFIG=ROOT/'configs/stage22_36_control'
MANIFEST=STATE.parent/'config_manifest.json'


def prepare():
    manifest=[]
    for family in ('s4m','pg95'):
        for smoke in (True,False):
            template=ROOT/f'configs/stage22_22_traffic/{family}_traffic_seed2027{"_smoke" if smoke else ""}.yaml'
            cfg=yaml.safe_load(template.read_text())
            cfg['experiment'].update(model=f'{family}_paired_rng',
                protocol_scope='stage22_36_paired_training_rng_v1'+('_smoke' if smoke else '_formal'),
                purpose='single_seed_mechanism_control_not_new_module')
            cfg['official_args']['stage22_36_paired_training_rng']=True
            cfg['provenance']=dict(source_config=str(template),source_config_sha256=helpers.digest(template),
                user_approved=True,configuration_changes={'stage22_36_paired_training_rng':
                    dict(old=False,new=True,reason='approved matched initialization and separated RNG streams')})
            validate_training(cfg)
            files=[(CONFIG/f'{family}_train_{"smoke" if smoke else "formal"}.yaml',cfg)]
            old=ROOT/f'configs/stage22_34_crn/{family}_traffic_validation_seed2027{"_smoke" if smoke else ""}.yaml'
            ecfg=yaml.safe_load(old.read_text())
            ecfg['experiment'].update(protocol_scope='stage22_36_paired_training_CRN_validation',
                purpose='read_only_new_training_checkpoint_control')
            ecfg['source']=dict(job_id=f'{family}_train_{"smoke" if smoke else "formal"}',queue_state=str(STATE))
            ecfg['provenance']=dict(stage='22.36',template=str(old),template_sha256=helpers.digest(old),
                user_approved=True,configuration_changes={'source':dict(old='historical_checkpoint',
                    new='paired_training_checkpoint_same_family',reason='approved paired control')})
            files.append((CONFIG/f'{family}_eval_{"smoke" if smoke else "formal"}.yaml',ecfg))
            for p,c in files:
                text=yaml.safe_dump(c,sort_keys=False)
                if p.exists() and p.read_text()!=text: raise ValueError(f'CONFIG_ERROR frozen {p}')
                helpers.atomic_text(p,text);manifest.append(dict(config=str(p),sha256=helpers.digest(p)))
    helpers.write_json(MANIFEST,manifest);print('prepared 8 frozen configs')


def execute(path,gpu,attempt):
    path=(ROOT/path).resolve()
    manifest=helpers.read(MANIFEST)
    assert any(r['config']==str(path) and r['sha256']==helpers.digest(path) for r in manifest),'CONFIG_ERROR manifest'
    tests=helpers.read(ROOT/'reports/stage22_36_tests.json')
    assert tests['passed'],'CODE_ERROR CPU controls not passed'
    config=yaml.safe_load(path.read_text())
    is_eval='diagnostic' in config
    if is_eval:
        target=STATE.parent/'materialized'/f'{path.stem}_attempt{attempt}_{time.strftime("%Y%m%d_%H%M%S")}.yaml'
        path=materialize_evaluation_config(path,default_state_path=STATE,destination=target)
        effective=yaml.safe_load(path.read_text())
        source=yaml.safe_load(Path(effective['source']['resolved_config']).read_text())
        assert source['official_args']['stage22_36_paired_training_rng'] is True
        validate_eval(effective,source)
    else: validate_training(config)
    runner=ROOT/('scripts/run_stage14_uncertainty_cache.py' if is_eval else 'scripts/run_s4m_experiment.py')
    files=['scripts/stage22_36_control.py','scripts/run_stage22_36_queue.py','scripts/test_stage22_36_control.py',
           'scripts/start_stage22_36.sh','docs/STAGE22_36_PAIRED_CONTROL_ZH.md',
           'scripts/run_stage14_uncertainty_cache.py','baselines/S4M/s4m/utils/paired_training_rng.py',
           'baselines/S4M/s4m/utils/paired_eval_rng.py']
    snapshot={p:(ROOT/p).read_text() for p in files}; run=None
    with subprocess.Popen([sys.executable,str(runner),'--config',str(path),'--gpu-uuid',gpu,
        '--scheduler-attempt',str(attempt)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True) as child:
        for line in child.stdout:
            print(line,end='',flush=True)
            if line.startswith('run_dir='):
                run=Path(line.strip().split('=',1)[1])
                for p,c in snapshot.items(): helpers.atomic_text(run/'stage22_36_source_snapshot'/p,c)
                helpers.write_json(run/'stage22_36_code_manifest.json',
                    {p:helpers.digest(run/'stage22_36_source_snapshot'/p) for p in snapshot})
        code=child.wait()
    if code==0 and not is_eval:
        m=helpers.read(run/'metrics.json')
        assert m.get('paired_initial_common_sha256')==tests['expected_common_sha256'],'CODE_ERROR initial hash mismatch'
        assert m['paired_training_rng_protocol']=='stage22_36_v1'
        assert len(m['paired_train_order'])==m['epochs_completed']
        assert len(m['paired_validation_order'])==m['epochs_completed']
    return code


def analyze():
    s=helpers.read(STATE) if STATE.exists() else dict(queue_state='NOT_STARTED',jobs={})
    result=dict(stage='22.36',queue_state=s['queue_state'],decision='WAITING_FOR_PAIRED_CONTROL',
                test_accessed=False,new_method=False,jobs=[{k:j.get(k) for k in ('id','state','run_dir','failure_summary')} for j in s['jobs'].values()])
    ids=['s4m_train_formal','pg95_train_formal','s4m_eval_formal','pg95_eval_formal']
    if all(s['jobs'].get(i,{}).get('state')=='SUCCESS' for i in ids):
        try:
            a,b=[helpers.read(Path(s['jobs'][i]['run_dir'])/'metrics.json') for i in ids[:2]]
            assert a['paired_initial_common_sha256']==b['paired_initial_common_sha256']
            for key in ('paired_train_order','paired_validation_order'):
                n=min(len(a[key]),len(b[key])); assert n>0 and a[key][:n]==b[key][:n]
            helpers.CRN_RECOVERY=True
            four=helpers.compare_seed(2027,*[Path(s['jobs'][i]['run_dir']) for i in ids[2:]])
            result.update(pairing_check='PASS',four_cells=four,
                actual_epochs=[a['epochs_completed'],b['epochs_completed']],decision='PAIRED_SINGLE_SEED_COMPLETE_NO_AUTO_EXPANSION')
        except Exception as e: result.update(decision='PAIRING_OR_ARTIFACT_ERROR',error=repr(e))
    elif s['queue_state']=='FAILED': result['decision']='EXECUTION_STOP'
    helpers.write_json(ROOT/'reports/stage22_36_control.json',result)
    helpers.atomic_text(ROOT/'reports/stage22_36_control.md','# 步骤22.36配对控制\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/aggregate_results.py')],cwd=ROOT,check=False)
    print(result['decision'])
    return int(result['decision']=='PAIRING_OR_ARTIFACT_ERROR')


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--config');p.add_argument('--gpu-uuid');p.add_argument('--scheduler-attempt',type=int,default=0);a=p.parse_args()
    if a.prepare: prepare();return 0
    if a.config:return execute(a.config,a.gpu_uuid,a.scheduler_attempt)
    return analyze()


if __name__=='__main__':raise SystemExit(main())
