#!/usr/bin/env python3
"""Reuse tested scheduler with independent artifacts and smoke dependency."""
import math
from pathlib import Path
import run_stage5_baseline_queue as scheduler
ROOT=scheduler.PROJECT_ROOT
scheduler.RUNTIME_ROOT=ROOT/'runtime/stage22_39_fusion_v2'
scheduler.STATE_PATH=scheduler.RUNTIME_ROOT/'state.json'
scheduler.EVENTS_PATH=scheduler.RUNTIME_ROOT/'events.jsonl'
scheduler.STOP_PATH=scheduler.RUNTIME_ROOT/'stop.requested'
scheduler.SCHEDULER_LOCK_PATH=scheduler.RUNTIME_ROOT/'scheduler.lock'
scheduler.LOG_ROOT=ROOT/'logs/stage22_39_fusion_v2'
scheduler.RUNNER=ROOT/'scripts/stage22_39_fusion.py';scheduler.AGGREGATOR=scheduler.RUNNER
scheduler.QUEUE_NAME='stage22_39_frozen_fusion_diagnostic_v2';scheduler.MAX_CONCURRENT_JOBS=3
scheduler.JOB_DEFINITIONS=[dict(id='seed2027_smoke',config='configs/stage22_39_fusion_v2/seed2027_smoke.yaml',depends_on=[],required_memory_estimate_mib=4608),
    dict(id='seed2027_formal',config='configs/stage22_39_fusion_v2/seed2027_formal.yaml',depends_on=['seed2027_smoke'],required_memory_estimate_mib=4608)]


def refresh(job,jobs):
    peak=3874.
    for other in jobs.values():
        if other.get('state')=='SUCCESS' and other.get('run_dir'):
            resource=scheduler.read_json(Path(other['run_dir'])/'resource_usage.json')
            peak=max(peak,float(resource.get('peak_gpu_reserved_mb') or 0))
    margin=max(512.,.1*peak)
    job['required_memory_estimate_mib']=max(job['required_memory_estimate_mib'],int(math.ceil((peak+margin)/256)*256))
    job['memory_estimate_source']=dict(peak_reserved_mib=peak,safety_margin_mib=margin,configuration_unchanged=True)
scheduler.refresh_formal_memory_estimate=refresh
if __name__=='__main__':raise SystemExit(scheduler.main())
