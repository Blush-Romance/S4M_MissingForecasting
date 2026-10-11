#!/usr/bin/env python3
"""Frozen config preparation, run provenance and validation-only result reporting."""
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
from stage22_38_common import ROOT, read, write_json, atomic_text, digest
from stage22_38_diagnostic import frozen, command_output, FILES as PREVIOUS_FILES

RT=ROOT/'runtime/stage22_39_fusion_v2';STATE=RT/'state.json';MANIFEST=RT/'config_manifest.json'
CONFIG=ROOT/'configs/stage22_39_fusion_v2';TEST_REPORT=ROOT/'reports/stage22_39_v2_tests.json'
BRANCH_SCALES={'retrieval':[0.,.9,1.1],'query':[0.,.9,1.1],'history':[0.,.9,1.1],
    'observed_history':[.9,1.1],'imputed_history':[0.,.9,1.1]}
FILES=PREVIOUS_FILES+['scripts/stage22_39_worker.py','scripts/stage22_39_fusion.py',
    'scripts/run_stage22_39_queue.py','scripts/start_stage22_39.sh','scripts/test_stage22_39_fusion.py',
    'scripts/stage22_39_distance_audit.py','docs/STAGE22_39_FUSION_AUDIT_ZH.md',
    'scripts/test_stage22_39_fusion_v2.py','docs/STAGE22_39_FUSION_V2_ZH.md']


def prepare():
    legacy_archive=ROOT/'reports/archive/stage22_39_v1_failure'
    for name in ('stage22_39_fusion.json','stage22_39_fusion.md'):
        if (ROOT/'reports'/name).exists() and not (legacy_archive/name).exists():
            frozen(legacy_archive/name,(ROOT/'reports'/name).read_text())
    prior=read(ROOT/'reports/stage22_38_diagnostic.json');assert prior['completed']
    old_run=Path(prior['run_dir']);base=yaml.safe_load((old_run/'resolved_config.yaml').read_text())
    for p,value in read(ROOT/'reports/stage22_38_tests.json')['code_hashes'].items():
        assert digest(p)==value,f'CODE_ERROR changed preceding model/diagnostic {p}'
    assert read(ROOT/'reports/stage22_39_distance_audit.json')['passed']
    manifest=[]
    for smoke in (True,False):
        cfg=json.loads(json.dumps(base));tag='smoke' if smoke else 'formal'
        cfg['experiment'].pop('id',None);cfg['experiment'].pop('created_at',None)
        cfg['experiment'].update(model='s4m_fusion_diagnostic_v2',smoke_test=smoke,
            protocol_scope='stage22_39_frozen_operational_fusion_v2_'+tag,
            protocol_match='paired_CRN_validation_frozen_checkpoint_intervention_not_new_model')
        cfg['source']['preceding_diagnostic_run']=str(old_run)
        cfg['input_hashes'].update({str(p):digest(p) for p in (old_run/'metrics.json',old_run/'samples.npz',
            old_run/'resolved_config.yaml',ROOT/'reports/stage22_39_distance_audit.json')})
        extension=ROOT/'baselines/S4M/s4m/model/S4/s4/extensions/kernels/structured_kernels.cpython-310-x86_64-linux-gnu.so'
        cfg['input_hashes'][str(extension)]=digest(extension)
        cfg['diagnostic']=dict(split='val',max_eval_batches=3 if smoke else -1,branch_scales=BRANCH_SCALES,
            fusion_evaluation_order='operational_original_fuse_v2',
            fixed_eval_rng=True,rng_protocol='stage22_34_crn_v1',baseline_metric_atol=1e-5,
            decomposition_relative_atol=1e-5,prototype_off_prediction_atol=1e-5,
            correct_distance_is_not_fed_to_model=True)
        cfg['runtime']=dict(device='cuda')
        cfg['provenance']=dict(stage='22.39',client_date='2026-10-04',user_approved=True,
            purpose='fusion_contribution_and_distance_diagnosis_no_training',unchanged_model_loss_and_checkpoint=True,
            changes={'diagnostic':dict(old='v1_context_plus_projected_component_delta',new='v2_original_fuse_scaled_branch_inputs_same_15_probes',
                reason='v1 FP32 subtraction mismatched prototype-off; exact original fused evaluation restores operational equivalence'),
                'model_and_training':dict(old='original_checkpoint',new='unchanged',reason='no bugfix or retraining in this stage')})
        path=CONFIG/f'seed2027_{tag}.yaml';frozen(path,yaml.safe_dump(cfg,sort_keys=False))
        manifest.append(dict(config=str(path),sha256=digest(path)))
    frozen(MANIFEST,json.dumps(manifest,indent=2)+'\n')
    print('prepared 2 frozen configs, 15 paths, unchanged checkpoint/batch16/validation',flush=True)


def check_configs():
    manifest=read(MANIFEST);assert len(manifest)==2
    for row in manifest:
        assert digest(row['config'])==row['sha256'],'CONFIG_ERROR manifest'
        cfg=yaml.safe_load(Path(row['config']).read_text())
        assert cfg['diagnostic']['split']=='val' and cfg['experiment']['seed']==2027
        assert cfg['diagnostic']['branch_scales']==BRANCH_SCALES
        assert cfg['diagnostic']['fusion_evaluation_order']=='operational_original_fuse_v2'
        assert (cfg['official_args']['batch_size'],cfg['official_args']['seq_len'],cfg['official_args']['pred_len'])==(16,192,96)
        assert cfg['official_args']['use_mipp'] is False
    return manifest


def attest():
    check_configs();tests=read(TEST_REPORT)
    assert tests['passed'] and all(tests['checks'].values()) and read(ROOT/'reports/stage22_39_v2_queue_tests.json')['self_test']=='PASS'
    tests.update(manifest_sha256=digest(MANIFEST),code_hashes={str(ROOT/p):digest(ROOT/p) for p in FILES})
    write_json(TEST_REPORT,tests);print('CPU/code/config/queue gates attested',flush=True)


def execute(path,gpu,attempt):
    check_configs();tests=read(TEST_REPORT)
    assert tests['passed'] and tests['manifest_sha256']==digest(MANIFEST)
    for p,value in tests['code_hashes'].items():assert digest(p)==value,f'CODE_ERROR source changed {p}'
    cfg=yaml.safe_load((ROOT/path).resolve().read_text())
    for p,value in cfg['input_hashes'].items():assert digest(p)==value,f'DATA_ERROR input changed {p}'
    if shutil.disk_usage(ROOT).free<1024**3:raise RuntimeError('DATA_ERROR disk below 1GiB')
    smoke='__smoke' if cfg['experiment']['smoke_test'] else ''
    identifier=f's4m_fusion_diagnostic_v2__traffic__point__mr0.06__seed2027{smoke}__{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    out=ROOT/'runs'/identifier;out.mkdir()
    cfg['experiment'].update(id=identifier,created_at=dt.datetime.now(dt.timezone.utc).isoformat())
    cfg['runtime'].update(gpu_uuid=gpu,scheduler_attempt=attempt)
    snapshot=command_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free','--format=csv'])
    atomic_text(out/'resolved_config.yaml',yaml.safe_dump(cfg,sort_keys=False))
    atomic_text(out/'env.txt',f'Python={sys.executable}\nCUDA_VISIBLE_DEVICES={gpu}\nConda=s4m_missing_forecasting\n'+snapshot)
    repo=ROOT/'baselines/S4M'
    atomic_text(out/'git_state.txt','[baseline_adapter]\n'+''.join(command_output(c,repo) for c in
        (['git','rev-parse','HEAD'],['git','status','--short'],['git','diff','--stat'])))
    atomic_text(out/'code.diff',subprocess.run(['git','diff','--binary'],cwd=repo,text=True,capture_output=True,check=True).stdout)
    for p in FILES:atomic_text(out/'source_snapshot'/p,(ROOT/p).read_text())
    write_json(out/'source_snapshot_manifest.json',tests['code_hashes'])
    command=[sys.executable,'-u',str(ROOT/'scripts/stage22_39_worker.py'),'--config',str(out/'resolved_config.yaml'),'--run-dir',str(out)]
    atomic_text(out/'command.sh',f'CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} '+shlex.join(command)+'\n')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='disabled',
        MPLCONFIGDIR=str(out/'.matplotlib'),OMP_NUM_THREADS='2')
    status=dict(experiment_id=identifier,state='RUNNING',started_at=cfg['experiment']['created_at']);write_json(out/'status.json',status)
    print(f'run_dir={out}',flush=True);started=time.monotonic()
    with (out/'train.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        status['pid']=child.pid;write_json(out/'status.json',status);code=child.wait()
    metrics=read(out/'metrics.json') if (out/'metrics.json').exists() else {}
    verification=None
    if code==0 and (not metrics or not all(metrics.get('checks',{}).values())):
        code=1;verification='CODE_ERROR incomplete diagnostic'
    log=(out/'train.log').read_text(errors='replace')
    state='SUCCESS' if code==0 else ('OOM' if 'out of memory' in log.lower() else
        'NAN_OR_DIVERGENCE' if 'NAN_OR_DIVERGENCE' in log else 'CODE_ERROR')
    status.update(state=state,return_code=code,finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),verification_error=verification)
    write_json(out/'status.json',status)
    if metrics.get('environment'):atomic_text(out/'env.txt',(out/'env.txt').read_text()+json.dumps(metrics['environment'],indent=2)+'\n')
    write_json(out/'resource_usage.json',dict(elapsed_seconds=time.monotonic()-started,return_code=code,gpu_uuid=gpu,
        peak_gpu_memory_mb=metrics.get('peak_gpu_allocated_mb'),peak_gpu_reserved_mb=metrics.get('peak_gpu_reserved_mb'),
        pre_run_snapshot=snapshot,post_run_snapshot=command_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free','--format=csv'])))
    print(f'finished state={state} return_code={code}',flush=True);return code


def analyze():
    state=read(STATE) if STATE.exists() else dict(queue_state='NOT_STARTED',jobs={})
    result=dict(stage='22.39',diagnostic_version='v2_operational_fusion',queue_state=state['queue_state'],completed=False,decision='WAITING_FOR_FUSION_DIAGNOSIS',
        test_accessed=False,new_model=False,jobs=list(state['jobs'].values()),distance_audit='reports/stage22_39_distance_audit.json')
    job=state['jobs'].get('seed2027_formal',{})
    if job.get('state')=='SUCCESS':
        out=Path(job['run_dir']);metrics=read(out/'metrics.json')
        assert metrics['sample_count']==1648 and all(metrics['checks'].values())
        assert digest(out/'samples.npz')==metrics['samples_sha256']
        with np.load(out/'samples.npz',allow_pickle=False) as data:
            for r,value in metrics['routes'].items():
                for key in ('mse','mae'):
                    np.testing.assert_allclose(data[r+('__sse' if key=='mse' else '__sae')].sum()/data['complete_count'].sum(),value[key],atol=1e-12,rtol=0)
        result.update(completed=True,decision='FUSION_DIAGNOSIS_COMPLETE_REVIEW_EVIDENCE',run_dir=str(out),metrics=metrics)
    elif state['queue_state']=='FAILED':result['decision']='EXECUTION_STOP'
    write_json(ROOT/'reports/stage22_39_fusion.json',result)
    lines=['# 步骤22.39：融合贡献与距离审计','',f'状态：{result["decision"]}',
        '','固定checkpoint的validation诊断，不是重训或新模型结果。原始数据/权重/主模型未改。']
    if result['completed']:
        lines+=['','| 路径 | MSE | MAE | MSE收益% | MAE收益% | 预测差异MSE |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for name,v in metrics['routes'].items():lines.append(f'| {name} | {v["mse"]:.9f} | {v["mae"]:.9f} | {v["mse_gain_percent"]:.5f} | {v["mae_gain_percent"]:.5f} | {v["prediction_difference_mse"]:.9f} |')
        lines+=['','正收益表示误差降低。query包含投影与残差；imputed_history仅干预缺失值直达，不修改真实观测。',
            '', '| 投影分量 | 平均RMS | 平均相对context RMS | 平均对齐系数 |',
            '| --- | ---: | ---: | ---: |']
        for name in ('retrieval','query','history','observed_history','imputed_history','bias'):
            a=metrics['activations'];lines.append(f'| {name} | {a[name+"__rms"]["mean"]:.6f} | {a[name+"__rms_ratio_to_context"]["mean"]:.6f} | {a[name+"__alignment_to_context"]["mean"]:.6f} |')
        lines+=['','RMS比不是贡献百分比；范数大小需结合预测干预解释。完整距离/logits/末端组见JSON。',
            '上游实现风险见stage22_39_distance_audit.md；本轮未修复，不将固定ckpt局部收益作为创新证据。']
    atomic_text(ROOT/'reports/stage22_39_fusion.md','\n'.join(lines)+'\n')
    subprocess.run([sys.executable,str(ROOT/'scripts/aggregate_results.py')],cwd=ROOT,check=False)
    if result['completed']:
        archive=ROOT/'reports/archive'/('stage22_39__'+dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
        for name in ('stage22_39_fusion.json','stage22_39_fusion.md','stage22_39_distance_audit.json','stage22_39_distance_audit.md','latest_results.csv','latest_results.md'):
            atomic_text(archive/name,(ROOT/'reports'/name).read_text())
    print(result['decision'],flush=True);return 0


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--attest',action='store_true')
    p.add_argument('--queue-test',action='store_true');p.add_argument('--config');p.add_argument('--gpu-uuid');p.add_argument('--scheduler-attempt',type=int,default=0)
    args=p.parse_args()
    if args.queue_test:
        r=subprocess.run([sys.executable,str(ROOT/'scripts/run_stage22_39_queue.py'),'--self-test'],cwd=ROOT,text=True,capture_output=True)
        print(r.stdout,flush=True)
        if r.returncode:print(r.stderr,flush=True);return r.returncode
        report=json.loads(r.stdout);write_json(ROOT/'reports/stage22_39_v2_queue_tests.json',report)
        return 0 if report['self_test']=='PASS' else 1
    if args.prepare:prepare();return 0
    if args.attest:attest();return 0
    if args.config:return execute(args.config,args.gpu_uuid,args.scheduler_attempt)
    return analyze()


if __name__=='__main__':raise SystemExit(main())
