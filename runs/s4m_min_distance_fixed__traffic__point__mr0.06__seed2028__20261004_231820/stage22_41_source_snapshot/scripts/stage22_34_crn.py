#!/usr/bin/env python3
"""Approved fixed-RNG recovery; preserve all historical queue/results."""
import copy
from pathlib import Path
import yaml
import stage22_34_factorial as stage

stage.STATE = stage.ROOT / 'runtime/stage22_34_crn/state.json'
stage.CONFIG_ROOT = stage.ROOT / 'configs/stage22_34_crn'
stage.REPORT_STEM = 'stage22_34_crn_factorial'
stage.CRN_RECOVERY = True


def prepare():
    manifest = []
    for family in ('s4m', 'pg95'):
        for seed, smoke in ((2027, True), (2027, False), (2026, False), (2028, False)):
            template = (stage.ROOT / f'configs/stage22_34_factorial/s4m_traffic_validation_seed{seed}.yaml'
                        if family == 's4m' else stage.ROOT / stage.PG95_RUNS[seed] / 'resolved_config.yaml')
            old = yaml.safe_load(template.read_text())
            cfg = {key: copy.deepcopy(old[key]) for key in ('experiment', 'source', 'data', 'diagnostic')}
            cfg['experiment'].pop('id', None)
            cfg['experiment'].pop('created_at', None)
            cfg['experiment'].update(smoke_test=smoke, model=f'cnm_se_crn_v1_{family}',
                protocol_scope='stage22_34_crn_v1_validation' + ('_smoke' if smoke else ''),
                protocol_match='new_fixed_rng_paired_validation_not_historical_rng',
                purpose='approved_recovery_four_cells_no_training_no_test')
            cfg['runtime'] = dict(device='cuda', gpu_uuid='GPU-c88b779d-77c7-92b9-e031-943bae990121')
            cfg['diagnostic'].update(fixed_eval_rng=True, rng_protocol='stage22_34_crn_v1',
                max_eval_batches=3 if smoke else -1, timing_warmup_batches=1 if smoke else 10)
            cfg['provenance'] = dict(stage='22.34.4', user_approved=True, template=str(template),
                source_config_sha256=stage.digest(cfg['source']['resolved_config']),
                source_checkpoint_sha256=stage.digest(cfg['source']['checkpoint']),
                configuration_changes={
                    'fixed_eval_rng': dict(old=False, new=True, reason='uncontrolled eval memory sampling'),
                    'baseline_reference': dict(old='training_end_historical_rng', new='separate_original_only_CRN_pass',
                                               reason='old checkpoint contains no RNG; user-approved repair'),
                    'max_eval_batches': dict(old=-1, new=3 if smoke else -1, reason='isolated smoke only'),
                    'timing_warmup_batches': dict(old=10, new=1 if smoke else 10, reason='isolated smoke only')},
                consistency_tolerance=1e-5, historical_results_preserved=True)
            source = yaml.safe_load(Path(cfg['source']['resolved_config']).read_text())
            stage.validate_config(cfg, source)
            path = stage.CONFIG_ROOT / f'{family}_traffic_validation_seed{seed}{"_smoke" if smoke else ""}.yaml'
            content = yaml.safe_dump(cfg, sort_keys=False)
            if path.exists() and path.read_text() != content:
                raise ValueError(f'CONFIG_ERROR: refusing to replace frozen {path}')
            stage.atomic_text(path, content)
            manifest.append(dict(config=str(path), sha256=stage.digest(path)))
    stage.write_json(stage.STATE.parent / 'config_manifest.json', manifest)
    print(f'prepared {len(manifest)} approved immutable CRN configs')


stage.prepare = prepare
if __name__ == '__main__':
    raise SystemExit(stage.main())
