#!/usr/bin/env python3
"""Reuse the tested resource queue with independent 22.37 state."""
import math
from pathlib import Path
import run_stage5_baseline_queue as scheduler

ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / 'runtime/stage22_37_control'
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / 'state.json'
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / 'events.jsonl'
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / 'stop.requested'
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / 'scheduler.lock'
scheduler.LOG_ROOT = ROOT / 'logs/stage22_37_control'
scheduler.RUNNER = ROOT / 'scripts/stage22_37_control.py'
scheduler.AGGREGATOR = scheduler.RUNNER
scheduler.QUEUE_NAME = 'stage22_37_frozen_paired_multiseed'
scheduler.MAX_CONCURRENT_JOBS = 3
scheduler.JOB_DEFINITIONS = []
for seed in (2028, 2029):
    for family, estimate in (('s4m', 13312), ('pg95', 18432)):
        previous = []
        for kind, tag in (('train', 'smoke'), ('eval', 'smoke'), ('train', 'formal'), ('eval', 'formal')):
            name = f'{family}_{seed}_{kind}_{tag}'
            scheduler.JOB_DEFINITIONS.append(dict(id=name,
                config=f'configs/stage22_37_control/{name}.yaml', depends_on=previous,
                required_memory_estimate_mib=estimate if kind == 'train' else 4608))
            previous = [name]


def refresh(job, jobs):
    family, _, kind, _ = job['id'].split('_')
    peak = 3936. if kind == 'eval' else (16448. if family == 'pg95' else 11874.)
    for other in jobs.values():
        parts = other['id'].split('_')
        if (other.get('state') == 'SUCCESS' and other.get('run_dir')
                and parts[0] == family and parts[2] == kind):
            resource = scheduler.read_json(Path(other['run_dir']) / 'resource_usage.json')
            peak = max(peak, float(resource.get('peak_gpu_reserved_mb') or 0))
    margin = max(512., .1 * peak)
    job['required_memory_estimate_mib'] = max(job['required_memory_estimate_mib'],
        int(math.ceil((peak + margin) / 256) * 256))
    job['memory_estimate_source'] = dict(peak_reserved_mib=peak, safety_margin_mib=margin,
        configuration_unchanged=True, policy='historical_peak_plus_10pct')


scheduler.refresh_formal_memory_estimate = refresh
if __name__ == '__main__': raise SystemExit(scheduler.main())
