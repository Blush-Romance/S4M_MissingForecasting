#!/usr/bin/env python3
import math
from pathlib import Path
import run_stage5_baseline_queue as scheduler
ROOT=scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT=ROOT/'runtime/stage22_36_control'
scheduler.STATE_PATH=scheduler.RUNTIME_ROOT/'state.json'
scheduler.EVENTS_PATH=scheduler.RUNTIME_ROOT/'events.jsonl'
scheduler.STOP_PATH=scheduler.RUNTIME_ROOT/'stop.requested'
scheduler.SCHEDULER_LOCK_PATH=scheduler.RUNTIME_ROOT/'scheduler.lock'
scheduler.LOG_ROOT=ROOT/'logs/stage22_36_control'
scheduler.RUNNER=ROOT/'scripts/stage22_36_control.py'
scheduler.AGGREGATOR=scheduler.RUNNER
scheduler.QUEUE_NAME='stage22_36_paired_initialization_training_control'
scheduler.JOB_DEFINITIONS=[dict(id='legacy_s4m_smoke',
    config='configs/stage22_22_traffic/s4m_traffic_seed2027_smoke.yaml',
    runner=str(ROOT/'scripts/run_s4m_experiment.py'),depends_on=[],required_memory_estimate_mib=13312)]
for family,estimate in (('s4m',13312),('pg95',18432)):
    for kind,smoke,parents in (
        ('train',True,['legacy_s4m_smoke']),
        ('eval',True,[f'{family}_train_smoke']),
        ('train',False,[f'{family}_eval_smoke']),
        ('eval',False,[f'{family}_train_formal'])):
        name=f'{family}_{kind}_{"smoke" if smoke else "formal"}'
        scheduler.JOB_DEFINITIONS.append(dict(id=name,config=f'configs/stage22_36_control/{name}.yaml',
            depends_on=parents,required_memory_estimate_mib=estimate if kind=='train' else 4608))


def refresh(job,jobs):
    if '_eval_' in job['id']: peak=3936.
    else: peak=16448. if job['id'].startswith('pg95') else 11874.
    for other in jobs.values():
        if other.get('state')=='SUCCESS' and other.get('run_dir') and other['id'].split('_')[:2]==job['id'].split('_')[:2]:
            r=scheduler.read_json(Path(other['run_dir'])/'resource_usage.json');peak=max(peak,float(r.get('peak_gpu_reserved_mb') or 0))
    margin=max(512.,.1*peak)
    job['required_memory_estimate_mib']=max(job['required_memory_estimate_mib'],int(math.ceil((peak+margin)/256)*256))
    job['memory_estimate_source']=dict(peak_reserved_mib=peak,safety_margin_mib=margin,configuration_unchanged=True)
scheduler.refresh_formal_memory_estimate=refresh
if __name__=='__main__':raise SystemExit(scheduler.main())
