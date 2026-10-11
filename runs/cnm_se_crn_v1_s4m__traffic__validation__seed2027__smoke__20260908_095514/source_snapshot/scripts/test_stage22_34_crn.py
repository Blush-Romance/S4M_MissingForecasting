#!/usr/bin/env python3
import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'baselines/S4M/s4m'))
from utils.paired_eval_rng import paired_eval_rng
from run_stage14_uncertainty_cache import validate_config, fixed_rng_reference_checks, run_id
import run_stage22_34_crn_queue as queue


class RecoveryTests(unittest.TestCase):
    def test_disabled_is_original_rng_sequence(self):
        torch.manual_seed(42)
        expected = torch.randperm(200)
        torch.manual_seed(42)
        with paired_eval_rng(False, 2027, 0, 'cpu'):
            actual = torch.randperm(200)
        self.assertTrue(torch.equal(expected, actual))

    def test_views_share_draws_and_do_not_affect_other_batches(self):
        def draw(batch):
            with paired_eval_rng(True, 2027, batch, 'cpu'):
                return torch.randperm(200)[:100]
        state = torch.get_rng_state().clone()
        reference = [draw(i) for i in range(4)]
        for i in range(4):
            for _ in range(4):
                self.assertTrue(torch.equal(reference[i], draw(i)))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertFalse(torch.equal(reference[0], reference[1]))

    def test_exception_restores_rng(self):
        state = torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            with paired_eval_rng(True, 2027, 0, 'cpu'):
                torch.randperm(200)
                raise RuntimeError('simulated failure')
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_references_cannot_pass_on_fabricated_metric(self):
        samples = {'complete_count': np.array([100., 100.])}
        metrics = dict(evaluation_rng_protocol='stage22_34_crn_v1',
            reference_pass='separate_original_only_chronological_loader', reference_consistency_tolerance=1e-5)
        for m, a in (('mse', 'sse'), ('mae', 'sae')):
            samples[f'reference_{a}_on'] = np.array([20., 30.])
            samples[f'complete_{a}_on'] = np.array([20., 30.])
            metrics[f'fixed_rng_reference_{m}'] = .25
            metrics[f'fixed_rng_max_window_{m}_difference'] = 0.
        self.assertTrue(all(fixed_rng_reference_checks(metrics, samples).values()))
        samples['reference_sae_on'][0] += .01
        self.assertFalse(all(fixed_rng_reference_checks(metrics, samples).values()))

    def test_all_configs_and_off_switch(self):
        paths = list((ROOT / 'configs/stage22_34_crn').glob('*.yaml'))
        self.assertEqual(len(paths), 8)
        for path in paths:
            cfg = yaml.safe_load(path.read_text())
            source = yaml.safe_load(Path(cfg['source']['resolved_config']).read_text())
            validate_config(cfg, source)
            self.assertEqual(source['official_args']['batch_size'], 16)
            self.assertEqual(cfg['diagnostic']['max_eval_batches'], 3 if cfg['experiment']['smoke_test'] else -1)
            self.assertIn('crn_v1', run_id(cfg, 'test_timestamp'))
            self.assertEqual(cfg['diagnostic']['nested_seed'], cfg['experiment']['seed'])
            old = copy.deepcopy(cfg)
            old['diagnostic']['fixed_eval_rng'] = False
            validate_config(old, source)
            self.assertNotIn('crn_v1', run_id(old, 'test_timestamp'))

    def test_test_split_and_unknown_rng_rejected(self):
        cfg = yaml.safe_load(next((ROOT / 'configs/stage22_34_crn').glob('*.yaml')).read_text())
        source = yaml.safe_load(Path(cfg['source']['resolved_config']).read_text())
        for key, value in (('split', 'test'), ('rng_protocol', 'anything')):
            bad = copy.deepcopy(cfg)
            bad['diagnostic'][key] = value
            with self.assertRaises(ValueError):
                validate_config(bad, source)

    def test_formal_jobs_require_both_new_and_legacy_smokes(self):
        jobs = queue.scheduler.JOB_DEFINITIONS
        self.assertEqual(len(jobs), 9)
        for job in jobs:
            if job['id'].endswith('formal'):
                self.assertEqual(len(job['depends_on']), 3)
                self.assertGreaterEqual(job['required_memory_estimate_mib'], 4608)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (ROOT / 'reports/stage22_34_crn_test.json').write_text(json.dumps(
        dict(tests=result.testsRun, passed=result.wasSuccessful(), failures=len(result.failures), errors=len(result.errors)), indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
