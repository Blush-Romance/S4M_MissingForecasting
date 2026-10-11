#!/usr/bin/env python3
"""Two model-family smoke gates then six CRN validation evaluations."""
from run_stage22_34_factorial_queue import scheduler

ROOT = scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT = ROOT / 'runtime/stage22_34_crn'
scheduler.STATE_PATH = scheduler.RUNTIME_ROOT / 'state.json'
scheduler.EVENTS_PATH = scheduler.RUNTIME_ROOT / 'events.jsonl'
scheduler.STOP_PATH = scheduler.RUNTIME_ROOT / 'stop.requested'
scheduler.SCHEDULER_LOCK_PATH = scheduler.RUNTIME_ROOT / 'scheduler.lock'
scheduler.LOG_ROOT = ROOT / 'logs/stage22_34_crn'
scheduler.RUNNER = ROOT / 'scripts/stage22_34_crn.py'
scheduler.AGGREGATOR = scheduler.RUNNER
scheduler.QUEUE_NAME = 'stage22_34_approved_crn_validation_recovery'
scheduler.JOB_DEFINITIONS = []
smokes = [f'traffic_{family}_validation_seed2027_smoke' for family in ('s4m', 'pg95')]
smokes.append('traffic_legacy_rng_seed2027_smoke')
scheduler.JOB_DEFINITIONS.append(dict(id=smokes[-1],
    config='configs/stage22_34_factorial/s4m_traffic_validation_seed2027_smoke.yaml',
    runner=str(ROOT / 'scripts/stage22_34_factorial.py'), depends_on=[], required_memory_estimate_mib=4608))
for seed, smoke in ((2027, True), (2027, False), (2026, False), (2028, False)):
    for family in ('s4m', 'pg95'):
        scheduler.JOB_DEFINITIONS.append(dict(
            id=f'traffic_{family}_validation_seed{seed}_{"smoke" if smoke else "formal"}',
            config=f'configs/stage22_34_crn/{family}_traffic_validation_seed{seed}{"_smoke" if smoke else ""}.yaml',
            depends_on=[] if smoke else smokes,
            required_memory_estimate_mib=4608))

if __name__ == '__main__':
    raise SystemExit(scheduler.main())
