#!/usr/bin/env python3
"""CPU-only correctness gates for frozen S4M fusion-path diagnostics."""
import json
from datetime import datetime, timezone
from pathlib import Path
import unittest

import torch

from stage22_39_worker import decompose, intervene


ROOT = Path(__file__).resolve().parents[1]


class FusionTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(2239)
        # Deliberately unequal input/output widths and variable-specific masks:
        # an incorrectly broadcast hidden-coordinate mask cannot pass these.
        self.r = torch.randn(2, 5, 7, generator=generator, dtype=torch.float64)
        self.u = torch.randn(2, 5, 7, generator=generator, dtype=torch.float64)
        self.z = torch.randn(2, 5, 3, generator=generator, dtype=torch.float64)
        self.m = torch.tensor(
            [[[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0]],
             [[0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 0, 0]]],
            dtype=torch.float64,
        )
        # Bank.pred_w is a right-multiplication matrix, not nn.Linear.weight.
        self.w = torch.randn(17, 7, generator=generator, dtype=torch.float64)
        self.b = torch.randn(7, generator=generator, dtype=torch.float64)

    def parts(self, mask=None):
        return decompose(self.r, self.u, self.z, self.m if mask is None else mask,
                         self.w, self.b)

    def context(self):
        return torch.cat((self.r, self.u, self.z), dim=-1) @ self.w + self.b + self.u

    def assertClose(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_decomposition_matches_original_concatenation_fusion(self):
        parts = self.parts()
        actual = parts['retrieval'] + parts['query'] + parts['history'] + parts['bias']
        self.assertClose(actual, self.context())

    def test_each_projected_component_uses_the_correct_weight_slice(self):
        parts = self.parts()
        self.assertClose(parts['retrieval'], self.r @ self.w[:7])
        self.assertClose(parts['query_projection'], self.u @ self.w[7:14])
        self.assertClose(parts['history'], self.z @ self.w[14:])
        self.assertEqual(parts['bias'].shape, self.r.shape)
        self.assertClose(parts['bias'], self.b.expand_as(self.r))

    def test_query_component_includes_unprojected_residual(self):
        parts = self.parts()
        self.assertTrue(torch.equal(parts['query_residual'], self.u))
        self.assertClose(parts['query'], parts['query_projection'] + self.u)
        self.assertFalse(torch.equal(parts['query'], parts['query_projection']))

    def test_observed_and_imputed_history_sum_to_history(self):
        parts = self.parts()
        self.assertClose(parts['observed_history'] + parts['imputed_history'], parts['history'])

    def test_all_observed_boundary_has_no_imputed_history(self):
        parts = self.parts(torch.ones_like(self.m))
        self.assertTrue(torch.equal(parts['imputed_history'], torch.zeros_like(self.r)))
        self.assertClose(parts['observed_history'], parts['history'])

    def test_all_missing_boundary_has_no_observed_history(self):
        parts = self.parts(torch.zeros_like(self.m))
        self.assertTrue(torch.equal(parts['observed_history'], torch.zeros_like(self.r)))
        self.assertClose(parts['imputed_history'], parts['history'])

    def test_identity_intervention_is_bitwise_and_object_identity(self):
        context = self.context()
        for name, component in self.parts().items():
            with self.subTest(component=name):
                actual = intervene(context, component, 1.)
                self.assertIs(actual, context)
                self.assertTrue(torch.equal(actual, context))

    def test_zero_scale_removes_only_the_selected_component(self):
        context = self.context()
        for name in ('retrieval', 'query', 'history', 'imputed_history', 'observed_history'):
            with self.subTest(component=name):
                component = self.parts()[name]
                self.assertClose(intervene(context, component, 0.), context - component)

    def test_local_imputed_scaling_preserves_observed_history_projection(self):
        parts = self.parts()
        observed = parts['observed_history'].clone()
        context = self.context()
        for scale in (.9, 1.1):
            with self.subTest(scale=scale):
                expected = (parts['retrieval'] + parts['query'] + parts['bias']
                            + observed + scale * parts['imputed_history'])
                actual = intervene(context, parts['imputed_history'], scale)
                self.assertClose(actual, expected)
                self.assertTrue(torch.equal(parts['observed_history'], observed))

    def test_masks_are_applied_before_variable_to_hidden_projection(self):
        parts = self.parts()
        expected_observed = (self.z * self.m) @ self.w[14:]
        expected_imputed = (self.z * (1. - self.m)) @ self.w[14:]
        self.assertClose(parts['observed_history'], expected_observed)
        self.assertClose(parts['imputed_history'], expected_imputed)
        self.assertEqual(parts['observed_history'].shape, (2, 5, 7))
        # Coordinate 1 is missing at [0,0] yet legitimately contributes to all
        # seven hidden dimensions; those dimensions must not be mask-filtered.
        self.assertTrue(bool((parts['imputed_history'][0, 0] != 0.).all()))

    def test_inputs_and_parameters_are_not_mutated(self):
        values = (self.r, self.u, self.z, self.m, self.w, self.b)
        snapshots = tuple(value.clone() for value in values)
        context = self.context()
        before_context = context.clone()
        parts = self.parts()
        snapshots_parts = {key: value.clone() for key, value in parts.items()}
        for scale in (0., .9, 1., 1.1):
            for component in parts.values():
                intervene(context, component, scale)
        for actual, expected in zip(values, snapshots):
            self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(context, before_context))
        for name, value in parts.items():
            self.assertTrue(torch.equal(value, snapshots_parts[name]))

    def test_helpers_preserve_autograd_without_training(self):
        r, u, z, w, b = (value.detach().clone().requires_grad_(True)
                         for value in (self.r, self.u, self.z, self.w, self.b))
        parts = decompose(r, u, z, self.m, w, b)
        context = parts['retrieval'] + parts['query'] + parts['history'] + parts['bias']
        actual = intervene(context, parts['imputed_history'], .9)
        actual.square().mean().backward()
        for value in (r, u, z, w, b):
            self.assertIsNotNone(value.grad)
            self.assertTrue(bool(torch.isfinite(value.grad).all()))
        self.assertTrue(bool((z.grad[self.m.bool()] != 0.).any()))
        self.assertTrue(bool((z.grad[~self.m.bool()] != 0.).any()))


if __name__ == '__main__':
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FusionTests)
    identifiers = [test.id().rsplit('.', 1)[-1] for test in suite]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    unsuccessful = {getattr(test, 'test_case', test).id().rsplit('.', 1)[-1]
                    for test, _ in result.failures + result.errors}
    report = ROOT / 'reports/stage22_39_tests.json'
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
        'dtype': 'float64',
        'tested_at': datetime.now(timezone.utc).isoformat(),
        'weight_convention': 'Bank.pred_w: [2H+C,H], right matrix multiplication',
        'scope': 'fusion algebra and intervention correctness; no GPU model or training',
        'history': history,
    }, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
