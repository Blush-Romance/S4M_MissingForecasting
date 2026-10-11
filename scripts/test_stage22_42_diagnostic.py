#!/usr/bin/env python3
"""Small CPU regression fixtures for weighted frozen-cache diagnosis."""
import json
from pathlib import Path
import unittest

import numpy as np

import stage22_42_diagnostic as d


class DiagnosticTests(unittest.TestCase):
    def test_big_bitorder(self):
        p = np.array([[128]], dtype=np.uint8)
        f = d.mask_descriptors(p, 8)
        self.assertEqual(f['tail_missing_length'][0], 7)
        self.assertEqual(f['missing_run_count'][0], 1)

    def test_runs_and_empty_tail(self):
        masks = np.array([[1, 0, 0, 1, 0, 1, 0, 0], [1]*8], dtype=np.uint8)
        f = d.mask_descriptors(np.packbits(masks, axis=1), 8)
        np.testing.assert_array_equal(f['longest_missing_length'], [2, 0])
        np.testing.assert_array_equal(f['tail_missing_length'], [2, 0])
        np.testing.assert_array_equal(f['missing_run_count'], [3, 0])
        self.assertEqual(f['missing_fraction'][0], .625)

    def test_invalid_packed_dimensions_rejected(self):
        with self.assertRaises(AssertionError):
            d.mask_descriptors(np.zeros((1, 23), dtype=np.uint8))

    def test_weighted_aggregation_and_empty(self):
        a = {'complete_count': np.array([1., 9.]), 'complete_sse_on': np.array([1., 27.]), 'complete_sae_on': np.array([1., 18.])}
        b = {**a, 'complete_sse_on': np.array([2., 18.]), 'complete_sae_on': np.array([2., 9.])}
        g = d.summarize_group(a, b, 'on', [True, True], 10)
        self.assertEqual(g['original_mse'], 2.8)
        self.assertEqual(g['fixed_mse'], 2.)
        self.assertAlmostEqual(g['mse_contribution_to_global_change'], -.8)
        self.assertIsNone(d.summarize_group(a, b, 'on', [False, False])['original_mse'])

    def test_concentration_preserves_cancellation(self):
        c = d.concentration(np.array([10., 2., -11., 0.]))
        self.assertEqual(c['net_error_increment'], 1.)
        self.assertEqual(c['positive_error_increment'], 12.)
        self.assertEqual(c['negative_error_increment'], -11.)
        self.assertAlmostEqual(c['top_five_percent_share_of_positive_increment'], 10/12)

    def test_rank_correlation_ties_and_constant(self):
        self.assertIsNone(d.spearman(np.ones(4), np.arange(4)))
        self.assertAlmostEqual(d.spearman(np.array([1, 1, 3, 4]), np.array([4, 4, 2, 1])), -1.)

    def test_fixed_groups_partition_and_preserve_empty(self):
        p = np.ones((8, 192), dtype=np.uint8)
        p[1:, -5:] = 0
        g = d.group_masks(d.mask_descriptors(np.packbits(p, axis=1)))
        self.assertEqual(sum(g[f'time_quarter_{i}'].sum() for i in range(1, 5)), 8)
        self.assertEqual(sum(g[k].sum() for k in ('tail_0', 'tail_1to5', 'tail_6to20', 'tail_gt20')), 8)
        self.assertEqual(g['tail_gt20'].sum(), 0)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DiagnosticTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    d.write_json(d.ROOT / 'reports/stage22_42_tests.json', dict(stage='22.42', passed=result.wasSuccessful(), tests=result.testsRun,
        failed=len(result.failures), errors=len(result.errors), device='cpu', no_model_forward=True,
        source_hashes={str(Path(__file__).resolve()): d.digest(__file__), str(Path(d.__file__).resolve()): d.digest(d.__file__)}))
    raise SystemExit(not result.wasSuccessful())
