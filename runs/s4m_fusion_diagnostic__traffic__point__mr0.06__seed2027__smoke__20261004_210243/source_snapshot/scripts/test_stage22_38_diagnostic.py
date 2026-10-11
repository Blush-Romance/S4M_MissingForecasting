#!/usr/bin/env python3
"""CPU gates for observed-only fills and frozen-path interventions."""
import json
from pathlib import Path
import unittest

import torch

from stage22_38_worker import mix_paths, observed_fills


ROOT = Path(__file__).resolve().parents[1]


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor(
            [[[99., 3., -91.], [2., -81., -71.], [77., 9., -61.],
              [8., -51., -41.], [66., 15., -31.]]], dtype=torch.float64
        )
        self.mask = torch.tensor(
            [[[0., 1., 0.], [1., 0., 0.], [0., 1., 0.],
              [1., 0., 0.], [0., 1., 0.]]], dtype=torch.float64
        )

    def test_observed_values_are_preserved_exactly(self):
        fills = observed_fills(self.x, self.mask)
        self.assertEqual(set(fills), {'zero', 'prefix_mean', 'ffill'})
        observed = self.mask.bool()
        for name, filled in fills.items():
            with self.subTest(fill=name):
                self.assertEqual(filled.shape, self.x.shape)
                self.assertEqual(filled.dtype, self.x.dtype)
                self.assertTrue(torch.equal(filled[observed], self.x[observed]))

    def test_known_observed_only_fill_values(self):
        expected = {
            'zero': [[[0., 3., 0.], [2., 0., 0.], [0., 9., 0.],
                      [8., 0., 0.], [0., 15., 0.]]],
            'prefix_mean': [[[0., 3., 0.], [2., 3., 0.], [2., 9., 0.],
                             [8., 6., 0.], [5., 15., 0.]]],
            'ffill': [[[0., 3., 0.], [2., 3., 0.], [2., 9., 0.],
                       [8., 9., 0.], [8., 15., 0.]]],
        }
        for name, actual in observed_fills(self.x, self.mask).items():
            with self.subTest(fill=name):
                self.assertTrue(torch.equal(actual, torch.tensor(expected[name], dtype=self.x.dtype)))

    def test_leading_and_all_missing_columns_are_finite(self):
        for name, filled in observed_fills(self.x, self.mask).items():
            with self.subTest(fill=name):
                self.assertTrue(bool(torch.isfinite(filled).all()))
                self.assertEqual(float(filled[0, 0, 0]), 0.)
                self.assertTrue(torch.equal(filled[:, :, 2], torch.zeros_like(filled[:, :, 2])))
        all_missing = torch.zeros_like(self.mask)
        for name, filled in observed_fills(self.x, all_missing).items():
            with self.subTest(all_missing_fill=name):
                self.assertTrue(bool(torch.isfinite(filled).all()))
                self.assertTrue(torch.equal(filled, torch.zeros_like(filled)))

    def test_hidden_values_never_enter_observed_fills(self):
        changed = torch.where(self.mask.bool(), self.x, self.x * -123. + 9000.)
        before = observed_fills(self.x, self.mask)
        after = observed_fills(changed, self.mask)
        for name in before:
            with self.subTest(fill=name):
                self.assertTrue(torch.equal(before[name], after[name]))

    def test_prefix_mean_and_ffill_do_not_read_future(self):
        # Change both values and availability after the cut; the earlier fill
        # must be identical even when the future contains a different regime.
        cut = 3
        changed_x, changed_mask = self.x.clone(), self.mask.clone()
        changed_x[:, cut:, :] = -1000.
        changed_mask[:, cut:, :] = 1.
        before = observed_fills(self.x, self.mask)
        after = observed_fills(changed_x, changed_mask)
        for name in ('prefix_mean', 'ffill'):
            with self.subTest(fill=name):
                self.assertTrue(torch.equal(before[name][:, :cut], after[name][:, :cut]))
                self.assertFalse(torch.equal(before[name][:, cut:], after[name][:, cut:]))

    def test_full_observation_is_identity_for_every_fill(self):
        x = torch.linspace(-3., 7., 42).reshape(2, 7, 3)
        for name, filled in observed_fills(x, torch.ones_like(x)).items():
            with self.subTest(fill=name):
                self.assertTrue(torch.equal(filled, x))

    def test_filling_is_independent_across_batch_and_variable(self):
        x = torch.cat((self.x, self.x + 100.), dim=0)
        mask = torch.cat((self.mask, self.mask), dim=0)
        full = observed_fills(x, mask)
        for batch in range(2):
            for variable in range(3):
                sliced = observed_fills(x[batch:batch + 1, :, variable:variable + 1],
                                        mask[batch:batch + 1, :, variable:variable + 1])
                for name in full:
                    with self.subTest(fill=name, batch=batch, variable=variable):
                        self.assertTrue(torch.equal(full[name][batch:batch + 1, :, variable:variable + 1],
                                                    sliced[name]))

    def test_mix_paths_isolates_the_requested_branches(self):
        base = {key: torch.arange(6.).reshape(2, 3) + i * 10.
                for i, key in enumerate(('retrieved', 'query', 'raw'))}
        alt = {key: value + 100. for key, value in base.items()}
        selections = {
            'full': ('alt', 'alt', 'alt'),
            'query': ('alt', 'alt', 'base'),
            'direct': ('base', 'base', 'alt'),
            'retrieval': ('alt', 'base', 'base'),
        }
        sources = {'base': base, 'alt': alt}
        for kind, selected in selections.items():
            with self.subTest(kind=kind):
                actual = mix_paths(base, alt, kind)
                self.assertEqual(len(actual), 3)
                for tensor, key, source in zip(actual, ('retrieved', 'query', 'raw'), selected):
                    self.assertTrue(torch.equal(tensor, sources[source][key]))

    def test_helpers_do_not_mutate_their_inputs(self):
        original_x, original_mask = self.x.clone(), self.mask.clone()
        observed_fills(self.x, self.mask)
        self.assertTrue(torch.equal(self.x, original_x))
        self.assertTrue(torch.equal(self.mask, original_mask))
        base = {key: torch.full((2, 3), float(i))
                for i, key in enumerate(('retrieved', 'query', 'raw'))}
        alt = {key: value + 10. for key, value in base.items()}
        before = {name: {key: value.clone() for key, value in source.items()}
                  for name, source in (('base', base), ('alt', alt))}
        for kind in ('full', 'query', 'direct', 'retrieval'):
            mix_paths(base, alt, kind)
        for name, source in (('base', base), ('alt', alt)):
            for key, value in source.items():
                self.assertTrue(torch.equal(value, before[name][key]))


if __name__ == '__main__':
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DiagnosticTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = ROOT / 'reports/stage22_38_tests.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({'passed': result.wasSuccessful(), 'tests': result.testsRun}, indent=2) + '\n')
    temporary.replace(report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
