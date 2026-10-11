#!/usr/bin/env python3
"""CPU gates for frozen budgets, validation boundary and real initial states."""
import copy
import json
import random
import subprocess
import sys
import unittest
import numpy as np
import torch
import yaml
import stage22_37_control as control
import run_stage22_37_queue as queue

ROOT = control.ROOT
sys.path[:0] = [str(ROOT / p) for p in ('baselines/S4M/s4m', 'baselines/S4M/s4m/model',
    'baselines/S4M/s4m/model/S4/s4', 'baselines/S4M/s4m/model/S4/s4/extensions/kernels')]
from utils.paired_training_rng import common_state_hash, AuditedRandomSampler, stream
from stage22_35_diagnose import parsed_namespace

EVIDENCE = {}


def cfg(family, seed, kind, tag='formal'):
    return yaml.safe_load((control.CONFIG / f'{family}_{seed}_{kind}_{tag}.yaml').read_text())


class Tests(unittest.TestCase):
    def test_baseline_source_unchanged_since_22_36(self):
        state = control.h.read(ROOT / 'runtime/stage22_36_control/state.json')
        for family in ('s4m', 'pg95'):
            run = ROOT / state['jobs'][f'{family}_train_formal']['run_dir']
            manifest = control.h.read(run / 'source_snapshot_manifest.json')
            for key, value in manifest.items():
                if key.startswith('baseline/'):
                    self.assertEqual(control.h.digest(value['source']), value['sha256'], key)

    def test_queue_simulated_workers(self):
        run = subprocess.run([sys.executable, str(ROOT / 'scripts/run_stage22_37_queue.py'), '--self-test'],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        report = json.loads(run.stdout)
        self.assertEqual(report['self_test'], 'PASS')
        self.assertEqual(report['max_observed_running'], 3)
        control.h.write_json(ROOT / 'reports/stage22_37_queue_tests.json', report)

    def test_frozen_budget_and_config_delta(self):
        manifest = control.verify_frozen()
        self.assertEqual(len(manifest), 16)
        for r in manifest:
            c = yaml.safe_load(open(r['config']).read())
            p = c['provenance']['template']
            self.assertEqual(control.h.digest(p), c['provenance']['template_sha256'])
            old = yaml.safe_load(open(p).read())
            seed = c['experiment']['seed']
            if 'official_args' in c:
                expected = copy.deepcopy(old['official_args'])
                expected['seed'] = seed
                for k in ('des', 'model_id'): expected[k] = expected[k].replace('2027', str(seed))
                self.assertEqual(c['official_args'], expected)
                self.assertEqual(c['runtime'], old['runtime'])
                control.validate_train(c)
            else:
                expected = copy.deepcopy(old['diagnostic']); expected['nested_seed'] = seed
                self.assertEqual(c['diagnostic'], expected)
                family = 's4m' if c['experiment']['method'] == 's4m_baseline' else 'pg95'
                source = cfg(family, seed, 'train', 'smoke' if c['experiment']['smoke_test'] else 'formal')
                control.validate_eval(c, source)

    def test_2029_validation_only_and_legacy_regression(self):
        for family in ('s4m', 'pg95'):
            source = cfg(family, 2029, 'train')
            c = cfg(family, 2029, 'eval')
            control.validate_eval(c, source)
            for change in ('scope', 'split', 'rng', 'variant', 'training'):
                bad, s = copy.deepcopy(c), copy.deepcopy(source)
                if change == 'scope': bad['experiment']['protocol_scope'] = 'old'
                if change == 'split': bad['diagnostic']['split'] = 'test'
                if change == 'rng': bad['diagnostic']['fixed_eval_rng'] = False
                if change == 'variant': bad['diagnostic']['variant'] = 'stage22_22_traffic_s4m_test'
                if change == 'training': s['official_args']['stage22_36_paired_training_rng'] = False
                with self.assertRaises(ValueError): control.validate_eval(bad, s)
            old = ROOT / 'configs/stage22_36_control'
            control.validate_eval(yaml.safe_load((old / f'{family}_eval_formal.yaml').read_text()),
                                  yaml.safe_load((old / f'{family}_train_formal.yaml').read_text()))

    def test_dependencies_and_oom_estimate(self):
        jobs = {j['id']: j for j in queue.scheduler.JOB_DEFINITIONS}
        self.assertEqual(len(jobs), 16)
        for seed in (2028, 2029):
            for family in ('s4m', 'pg95'):
                self.assertEqual(jobs[f'{family}_{seed}_train_formal']['depends_on'],
                                 [f'{family}_{seed}_eval_smoke'])
                self.assertEqual(jobs[f'{family}_{seed}_eval_formal']['depends_on'],
                                 [f'{family}_{seed}_train_formal'])
                j = copy.deepcopy(jobs[f'{family}_{seed}_train_formal'])
                j.update(oom_retry_count=1, required_memory_estimate_mib=23040)
                queue.refresh(j, {})
                self.assertGreaterEqual(j['required_memory_estimate_mib'], 23040)

    def test_actual_model_initial_hashes_and_rng(self):
        from model.S4M import Model
        oldstate = control.h.read(ROOT / 'runtime/stage22_36_control/state.json')
        hashes = {}
        for seed in (2028, 2029):
            states = []
            for family in ('s4m', 'pg95'):
                run = oldstate['jobs'][f'{family}_train_formal']['run_dir']
                args = parsed_namespace((ROOT / run / 'train.log').read_text())
                args.seed = seed; args.stage22_36_paired_training_rng = True
                torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
                model = Model(args).float()
                states.append(common_state_hash(model)); del model
                before = torch.get_rng_state().clone()
                with stream(args, 'aux'): torch.rand(50)
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(states[0], states[1]); self.assertEqual(states[0][1], 314)
            hashes[str(seed)] = states[0][0]
            a = AuditedRandomSampler(range(77), seed, 'train')
            b = AuditedRandomSampler(range(77), seed, 'train')
            for _ in range(3):
                left = list(a); torch.rand(9); self.assertEqual(left, list(b))
        self.assertNotEqual(hashes['2028'], hashes['2029'])
        EVIDENCE['initial_hashes'] = hashes

    def test_summary_keeps_negative_results(self):
        rows = []
        for gain in (1., -2., 3.):
            rows.append(dict(cells={k: dict(mse=1., mae=2.) for k in
                ('A_s4m', 'B_pg95', 'C_s4m_cnm', 'D_pg95_cnm')},
                **{k: [gain, gain] for k in ('pg95_gain_without_cnm_percent', 'pg95_gain_with_cnm_percent',
                    'cnm_gain_on_s4m_percent', 'cnm_gain_on_pg95_percent')}))
        summary = control.summarize(rows)
        self.assertEqual(summary['gains']['pg95_gain_with_cnm_percent']['dual_wins'], 2)
        self.assertAlmostEqual(summary['gains']['pg95_gain_with_cnm_percent']['mean'][0], 2/3)


if __name__ == '__main__':
    torch.set_num_threads(2)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    files = ['scripts/stage22_37_control.py', 'scripts/run_stage22_37_queue.py',
             'scripts/test_stage22_37_control.py', 'scripts/start_stage22_37.sh',
             'scripts/run_stage14_uncertainty_cache.py', 'scripts/run_s4m_experiment.py',
             'scripts/run_stage5_baseline_queue.py', 'scripts/generate_stage22_22_traffic_assets.py',
             'scripts/generate_paper_point_masks.py', 'scripts/stage22_34_factorial.py',
             'scripts/run_stage22_22_traffic_pipeline.py', 'docs/STAGE22_37_PAIRED_MULTISEED_ZH.md']
    files += [str(p.relative_to(ROOT)) for sub in ('model', 'experiments', 'data_provider', 'utils')
              for p in (ROOT / 'baselines/S4M/s4m' / sub).glob('*.py')]
    files.append('baselines/S4M/s4m/run.py')
    control.h.write_json(control.TESTS, dict(passed=result.wasSuccessful(), tests=result.testsRun,
        manifest_sha256=control.h.digest(control.MANIFEST),
        code_hashes={str(ROOT / p): control.h.digest(ROOT / p) for p in files}, **EVIDENCE))
    raise SystemExit(0 if result.wasSuccessful() else 1)
