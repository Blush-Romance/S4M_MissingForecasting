#!/usr/bin/env python3
"""Small CPU fixtures; do not execute models or mutate source experiments."""
import unittest
import numpy as np
import stage22_43_cnm_audit as audit


def fixture():
    return {'sample_index': np.arange(2), 'complete_count': np.array([1., 9.]),
            'complete_sse_on': np.array([1., 27.]), 'complete_sse_multiscale': np.array([2., 18.]),
            'complete_sae_on': np.array([1., 18.]), 'complete_sae_multiscale': np.array([2., 9.])}


class CacheAuditTests(unittest.TestCase):
    def test_weighted_sum_not_window_mean(self):
        r = audit.compare_routes(fixture())
        self.assertEqual(r['baseline']['mse'], 2.8)
        self.assertEqual(r['cnm']['mse'], 2.)
        self.assertTrue(r['dual_win'])
        self.assertAlmostEqual(r['relative_gain_percent']['mse'], .8 / 2.8 * 100)
        self.assertEqual(r['worse_window_fraction']['mse'], .5)

    def test_counts_zero_nan_rejected(self):
        for value in (0., -1., np.nan, np.inf):
            a = fixture()
            a['complete_count'][0] = value
            with self.assertRaises(AssertionError):
                audit.compare_routes(a)

    def test_error_nonfinite_negative_rejected(self):
        for value in (-1., np.nan, np.inf):
            a = fixture()
            a['complete_sse_multiscale'][0] = value
            with self.assertRaises(AssertionError):
                audit.compare_routes(a)

    def test_shape_and_order_rejected(self):
        for key in ('sample_index', 'complete_sae_on'):
            a = fixture()
            a[key] = np.arange(3)
            with self.assertRaises(AssertionError):
                audit.compare_routes(a)
        a = fixture()
        a['sample_index'] = np.array([1, 0])
        with self.assertRaises(AssertionError):
            audit.compare_routes(a)

    def test_zero_error_no_division_by_zero(self):
        a = fixture()
        a['complete_sse_on'][:] = 0
        r = audit.compare_routes(a)
        self.assertIsNone(r['relative_gain_percent']['mse'])
        self.assertFalse(r['dual_win'])

    def test_means_and_mean_relative_gain_distinguished(self):
        r = audit.compare_routes(fixture())
        a = fixture()
        a['complete_sse_on'] *= 2
        q = audit.compare_routes(a)
        s = audit.summarize([r, q])
        self.assertNotEqual(s['relative_gain_percent']['mse']['mean'], s['relative_gain_between_metric_means_percent']['mse'])
        self.assertAlmostEqual(s['baseline']['mse']['sample_std'], np.std([2.8, 5.6], ddof=1))

    def test_smoke_single_seed_std_is_na(self):
        s = audit.summarize([audit.compare_routes(fixture())])
        self.assertIsNone(s['relative_gain_percent']['mse']['sample_std'])

    def test_ties_are_not_dual_win(self):
        a = fixture()
        a['complete_sse_multiscale'] = a['complete_sse_on'].copy()
        self.assertFalse(audit.compare_routes(a)['dual_win'])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CacheAuditTests))
    audit.old.write_json(audit.ROOT / 'reports/stage22_43_tests.json', {
        'stage': '22.43', 'passed': result.wasSuccessful(), 'tests': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'device': 'cpu',
        'source_hashes': {str(audit.ROOT / x): audit.old.digest(audit.ROOT / x) for x in audit.CODE[:2]}})
    raise SystemExit(not result.wasSuccessful())
