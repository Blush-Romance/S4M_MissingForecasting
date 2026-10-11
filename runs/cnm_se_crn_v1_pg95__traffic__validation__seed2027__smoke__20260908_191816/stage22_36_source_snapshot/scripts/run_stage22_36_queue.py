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
    # User-authorized 2026-09-08 first-attempt trial. This only changes
    # admission headroom, never training configuration or OOM backoff.
    trial=(job['id'] in ('pg95_train_smoke','pg95_train_formal')
           and job.get('attempt_count',0)==0 and job.get('oom_retry_count',0)==0
           and job['required_memory_estimate_mib'] in (17408,18432)
           and not any(j.get('oom_retry_count',0) for j in jobs.values()
                       if j['id'].startswith('pg95_train')))
    margin=max(512.,(.05 if trial else .1)*peak)
    estimate=int(math.ceil((peak+margin)/256)*256)
    job['required_memory_estimate_mib']=(estimate if trial else
        max(job['required_memory_estimate_mib'],estimate))
    job['memory_estimate_source']=dict(peak_reserved_mib=peak,safety_margin_mib=margin,
        configuration_unchanged=True,policy='user_first_attempt_5pct' if trial else 'conservative_10pct',
        original_pg95_estimate_mib=18432)
scheduler.refresh_formal_memory_estimate=refresh
if __name__=='__main__':raise SystemExit(scheduler.main())
