#!/usr/bin/env python3
"""Independent queue; first admission <=6GiB, backoff/requeue without config changes."""
import run_stage5_baseline_queue as scheduler

ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / 'runtime/stage22_40_distance_fix'
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / 'state.json'
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / 'events.jsonl'
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / 'stop.requested'
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / 'scheduler.lock'
scheduler.LOG_ROOT = ROOT / 'logs/stage22_40_distance_fix'
scheduler.RUNNER = ROOT / 'scripts/stage22_40_distance_fix.py'
scheduler.AGGREGATOR = scheduler.RUNNER
scheduler.QUEUE_NAME = 'stage22_40_isolated_min_distance_bug_fix'
scheduler.MAX_CONCURRENT_JOBS = 3
scheduler.JOB_DEFINITIONS = []
smokes = ['off_train_smoke', 'fixed_train_smoke', 'off_eval_smoke', 'fixed_eval_smoke']
previous = []
for name in smokes + ['fixed_train_formal', 'fixed_eval_formal']:
    dependencies = list(smokes) if name == 'fixed_train_formal' else list(previous)
    scheduler.JOB_DEFINITIONS.append(dict(id=name,
        config=f'configs/stage22_40_distance_fix/{name}.yaml', depends_on=dependencies,
        required_memory_estimate_mib=6144 if '_train_' in name else 4608))
    previous = [name]


def refresh(job, jobs):
    # Preserve OOM-learned requirements. Do not reset them to first-admission policy.
    initial = 6144 if '_train_' in job['id'] else 4608
    job['required_memory_estimate_mib'] = max(initial, job['required_memory_estimate_mib'])
    job['memory_estimate_source'] = dict(policy='first_admission_max_6GiB_no_protocol_shrink',
        historical_peak_reserved_mib=11870 if '_train_' in job['id'] else 3936,
        admission_mib=initial, oom_learned_requirements_never_lowered=True,
        training_smokes_serial_to_protect_actual_historical_peak=True,
        source='22.36 actual peak; initial admission follows updated global rule')


scheduler.refresh_formal_memory_estimate = refresh
if __name__ == '__main__':
    raise SystemExit(scheduler.main())
