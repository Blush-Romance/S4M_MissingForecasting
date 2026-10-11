#!/usr/bin/env python3
"""CPU-only V2 gates: intervene through the original fusion computation."""
import json
from datetime import datetime, timezone
from pathlib import Path
import unittest

import torch

from stage22_39_worker import decompose, operational_intervene
from test_stage22_39_fusion import FusionTests as _OriginalFixture


ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ('retrieval', 'query', 'history', 'observed_history', 'imputed_history')


class OperationalFusionTests(unittest.TestCase):
    def setUp(self):
        _OriginalFixture.setUp(self)
        self.calls = []
        self.base = dict(retrieved=self.r, query=self.u, raw=self.z,
                         context=self.original_fuse(self.r.flatten(0, 1), self.u.flatten(0, 1),
                                                    self.z, 2, 5, 7))
        self.calls.clear()

    def original_fuse(self, r, u, z, b, l, h):
        self.assertEqual(r.shape, (b * l, h))
        self.assertEqual(u.shape, (b * l, h))
        self.assertEqual(z.shape, (b, l, 3))
        self.calls.append(dict(retrieved=r.clone(), query=u.clone(), raw=z.clone()))
        joined = torch.cat((r, u, z.reshape(b * l, -1)), dim=-1)
        return (joined @ self.w + self.b + u).reshape(b, l, h)

    def operation(self, branch, scale):
        return operational_intervene(self.base, self.m, branch, scale, self.original_fuse)

    def assertClose(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_all_five_branches_match_mathematical_interventions(self):
        parts = decompose(self.r, self.u, self.z, self.m, self.w, self.b)
        for branch in BRANCHES:
            for scale in (0., .9, 1., 1.1):
                with self.subTest(branch=branch, scale=scale):
                    actual = self.operation(branch, scale)
                    expected = self.base['context'] + (scale - 1.) * parts[branch]
                    self.assertClose(actual, expected)

    def test_scale_one_returns_original_context_without_fuse(self):
        for branch in BRANCHES:
            with self.subTest(branch=branch):
                actual = self.operation(branch, 1.)
                self.assertIs(actual, self.base['context'])
                self.assertTrue(torch.equal(actual, self.base['context']))
        self.assertEqual(self.calls, [])

    def test_retrieval_off_is_exact_original_fuse_with_zero_retrieval(self):
        actual = self.operation('retrieval', 0.)
        expected = self.original_fuse(torch.zeros_like(self.r).flatten(0, 1),
                                      self.u.flatten(0, 1), self.z, 2, 5, 7)
        self.assertTrue(torch.equal(actual, expected))

    def test_retrieval_off_is_exact_in_float32_without_reordering(self):
        r, u, z, w, bias = (v.float() for v in (self.r, self.u, self.z, self.w, self.b))
        def fuse(retrieved, query, raw, b, l, h):
            joined = torch.cat((retrieved, query, raw.reshape(b * l, -1)), dim=-1)
            return (joined @ w + bias + query).reshape(b, l, h)
        base = dict(retrieved=r, query=u, raw=z,
                    context=fuse(r.flatten(0, 1), u.flatten(0, 1), z, 2, 5, 7))
        actual = operational_intervene(base, self.m.float(), 'retrieval', 0., fuse)
        expected = fuse(torch.zeros_like(r).flatten(0, 1), u.flatten(0, 1), z, 2, 5, 7)
        self.assertTrue(torch.equal(actual, expected))

    def test_query_scaling_covers_projection_and_residual(self):
        parts = decompose(self.r, self.u, self.z, self.m, self.w, self.b)
        for scale in (0., .9, 1.1):
            with self.subTest(scale=scale):
                actual = self.operation('query', scale)
                expected = self.base['context'] + (scale - 1.) * (
                    parts['query_projection'] + self.u)
                self.assertClose(actual, expected)
                self.assertTrue(torch.equal(self.calls[-1]['query'], (scale * self.u).flatten(0, 1)))

    def test_history_scaling_only_changes_raw_input(self):
        for scale in (0., .9, 1.1):
            with self.subTest(scale=scale):
                self.operation('history', scale)
                call = self.calls[-1]
                self.assertTrue(torch.equal(call['raw'], scale * self.z))
                self.assertTrue(torch.equal(call['retrieved'], self.r.flatten(0, 1)))
                self.assertTrue(torch.equal(call['query'], self.u.flatten(0, 1)))

    def test_observed_and_imputed_scaling_use_complementary_variable_masks(self):
        for scale in (0., .9, 1.1):
            with self.subTest(scale=scale):
                self.operation('observed_history', scale)
                observed = self.calls[-1]['raw']
                self.assertTrue(torch.equal(observed, torch.where(self.m.bool(), self.z * scale, self.z)))
                self.operation('imputed_history', scale)
                imputed = self.calls[-1]['raw']
                self.assertTrue(torch.equal(imputed, torch.where(self.m.bool(), self.z, self.z * scale)))
                self.assertClose(observed + imputed, (1. + scale) * self.z)

    def test_all_observed_and_all_missing_boundaries(self):
        for mask, untouched_branch in ((torch.ones_like(self.m), 'imputed_history'),
                                       (torch.zeros_like(self.m), 'observed_history')):
            for scale in (0., .9, 1.1):
                with self.subTest(untouched_branch=untouched_branch, scale=scale):
                    actual = operational_intervene(self.base, mask, untouched_branch, scale, self.original_fuse)
                    self.assertTrue(torch.equal(actual, self.base['context']))

    def test_local_imputed_scaling_preserves_every_observed_raw_value(self):
        observed = self.m.bool()
        for scale in (0., .9, 1.1):
            with self.subTest(scale=scale):
                self.operation('imputed_history', scale)
                actual = self.calls[-1]['raw']
                self.assertTrue(torch.equal(actual[observed], self.z[observed]))
                self.assertTrue(torch.equal(actual[~observed], (self.z * scale)[~observed]))

    def test_no_retrieval_or_query_recomputation_for_history_interventions(self):
        for branch in ('history', 'observed_history', 'imputed_history'):
            for scale in (0., .9, 1.1):
                with self.subTest(branch=branch, scale=scale):
                    self.calls.clear()
                    self.operation(branch, scale)
                    self.assertEqual(len(self.calls), 1)
                    self.assertTrue(torch.equal(self.calls[0]['retrieved'], self.r.flatten(0, 1)))
                    self.assertTrue(torch.equal(self.calls[0]['query'], self.u.flatten(0, 1)))

    def test_no_in_place_changes_to_inputs_context_mask_or_parameters(self):
        tensors = dict(self.base, mask=self.m, weight=self.w, bias=self.b)
        snapshots = {name: value.clone() for name, value in tensors.items()}
        for branch in BRANCHES:
            for scale in (0., .9, 1., 1.1):
                self.operation(branch, scale)
        for name, value in tensors.items():
            self.assertTrue(torch.equal(value, snapshots[name]))

    def test_operational_intervention_preserves_autograd_without_training(self):
        self.r, self.u, self.z, self.w, self.b = (
            value.detach().clone().requires_grad_(True)
            for value in (self.r, self.u, self.z, self.w, self.b))
        self.base = dict(retrieved=self.r, query=self.u, raw=self.z,
                         context=self.original_fuse(self.r.flatten(0, 1), self.u.flatten(0, 1),
                                                    self.z, 2, 5, 7))
        actual = self.operation('imputed_history', .9)
        actual.square().mean().backward()
        for value in (self.r, self.u, self.z, self.w, self.b):
            self.assertIsNotNone(value.grad)
            self.assertTrue(bool(torch.isfinite(value.grad).all()))


if __name__ == '__main__':
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OperationalFusionTests)
    identifiers = [test.id().rsplit('.', 1)[-1] for test in suite]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    unsuccessful = {getattr(test, 'test_case', test).id().rsplit('.', 1)[-1]
                    for test, _ in result.failures + result.errors}
    report = ROOT / 'reports/stage22_39_v2_tests.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    prior = json.loads(report.read_text()) if report.exists() else None
    history = prior.get('history', []) if prior else []
    if prior:
        history.append({key: value for key, value in prior.items() if key != 'history'})
    temporary = report.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({
        'passed': result.wasSuccessful(),
        'checks': {name: name not in unsuccessful for name in identifiers},
        'tests': result.testsRun,
        'device': 'cpu',
        'dtype': 'float64, plus exact float32 R0 original-fuse gate',
        'tested_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'V2 original-fusion operational interventions; no GPU model or training',
        'history': history,
    }, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
