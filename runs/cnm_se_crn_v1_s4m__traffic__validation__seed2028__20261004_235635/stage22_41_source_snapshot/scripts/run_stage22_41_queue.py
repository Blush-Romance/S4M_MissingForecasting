#!/usr/bin/env python3
"""Frozen bug-fix replication queue; one active experiment per project GPU."""
import run_stage5_baseline_queue as scheduler

ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / 'runtime/stage22_41_distance_fix'
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / 'state.json'
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / 'events.jsonl'
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / 'stop.requested'
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / 'scheduler.lock'
scheduler.LOG_ROOT = ROOT / 'logs/stage22_41_distance_fix'
scheduler.RUNNER = ROOT / 'scripts/stage22_41_distance_fix.py'
scheduler.AGGREGATOR = scheduler.RUNNER
scheduler.QUEUE_NAME = 'stage22_41_isolated_min_distance_three_seed_replication'
scheduler.MAX_CONCURRENT_JOBS = 3
scheduler.JOB_DEFINITIONS = []
for seed in (2028, 2029):
    prefix = f'fixed_{seed}'
    for tag in ('smoke', 'formal'):
        for kind in ('train', 'eval'):
            name = f'{prefix}_{kind}_{tag}'
            if kind == 'eval':
                dependencies = [f'{prefix}_train_{tag}']
            elif tag == 'formal':
                dependencies = [f'{prefix}_eval_smoke']
            else:
                dependencies = []
            scheduler.JOB_DEFINITIONS.append(dict(id=name,
                config=f'configs/stage22_41_distance_fix/{name}.yaml', depends_on=dependencies,
                required_memory_estimate_mib=6144 if kind == 'train' else 4608))

_active_gpus = set()
_choose_gpu = scheduler.choose_gpu


def refresh(job, jobs):
    global _active_gpus
    _active_gpus = {candidate['gpu']['uuid'] for candidate in jobs.values()
                   if candidate.get('state') == 'RUNNING' and candidate.get('gpu')}
    initial = 6144 if '_train_' in job['id'] else 4608
    job['required_memory_estimate_mib'] = max(initial, job['required_memory_estimate_mib'])
    job['memory_estimate_source'] = dict(policy='first_admission_max_6GiB_no_protocol_shrink',
        historical_peak_reserved_mib=11870 if '_train_' in job['id'] else 3936,
        admission_mib=initial, oom_learned_requirements_never_lowered=True,
        seed_chains_independent=True, same_gpu_overlap_forbidden=True,
        source='22.36/22.37 historical peaks; first admission follows updated global rule')


def choose_gpu(job, gpus, candidate_uuids, launched_this_poll, reservations=None):
    # Keep the original atomic lock/admission/OOM policy, but never stack two
    # jobs on a GPU: a low first-admission estimate is not a peak guarantee.
    return _choose_gpu(job, [gpu for gpu in gpus if gpu['uuid'] not in _active_gpus],
                       candidate_uuids, launched_this_poll, reservations)


scheduler.refresh_formal_memory_estimate = refresh
scheduler.choose_gpu = choose_gpu
if __name__ == '__main__':
    raise SystemExit(scheduler.main())
