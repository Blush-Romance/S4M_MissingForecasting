#!/usr/bin/env python3
"""Small deterministic tests; no real experiment cache or test split is read."""
import copy
import inspect
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import yaml

import run_stage22_33_psavs as p


class PSAVSTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(p.CONFIG.read_text())

    def test_frozen_configuration_and_temporal_support(self):
        p.validate_config(self.cfg)
        splits = self.cfg["splits"]
        for first, second in (("fit", "development"), ("development", "audit")):
            last_first_input = splits[first][1] - 1
            last_first_target = last_first_input + 192 + 96 - 1
            self.assertLess(last_first_target, splits[second][0])

    def test_bad_protocol_rejected(self):
        for key, value in (("actions", list(reversed(p.ACTIONS))),
                           ("history_length", 96), ("source_runs", {2026: "test/path"}),
                           ("margin_candidates", [0.]),
                           ("splits", dict(fit=[0, 576], development=[577, 900], audit=[1000, 1648]))):
            cfg = copy.deepcopy(self.cfg)
            cfg[key] = value
            with self.assertRaisesRegex(ValueError, "CONFIG_ERROR"):
                p.validate_config(cfg)

    def test_features_exact_mask_extremes(self):
        x = p.mask_features(np.array([[1] * 192, [0] * 192, [1] * 160 + [0] * 32]))
        self.assertEqual(x.shape, (3, 14))
        np.testing.assert_array_equal(x[0], np.zeros(14))
        np.testing.assert_allclose(x[1, :8], [1, 0, 1, 1, 1 / 192, .5, 1, 1])
        np.testing.assert_allclose(x[2, :5], [32 / 192, 1 / 191, 32 / 192, 32 / 192, 1 / 192])
        self.assertEqual(x[2, 7], 32 / 96)
        np.testing.assert_array_equal(x[2, 8:], [0, 0, 0, 0, 0, 1])

    def test_disagreement_features_and_invalid_values(self):
        x = p.make_features(np.ones((2, 192)), np.array([0., 1.]), np.array([0., 2.]))
        self.assertEqual(x["disagreement_aware"].shape, (2, 17))
        np.testing.assert_allclose(x["disagreement_aware"][:, 14:16], [[0, 0], [np.log(2), np.log(3)]])
        self.assertTrue(np.isfinite(x["disagreement_aware"]).all())
        for bad in (np.array([np.nan, 1]), np.array([-1., 0])):
            with self.assertRaisesRegex(ValueError, "DATA_ERROR"):
                p.make_features(np.ones((2, 192)), bad, np.ones(2))

    def test_policy_inputs_do_not_include_labels(self):
        self.assertEqual(list(inspect.signature(p.make_features).parameters), ["observed", "d5", "d20"])
        self.assertEqual(list(inspect.signature(p.select_actions).parameters), ["predicted_gains", "margin", "enabled"])
        for names in p.FEATURE_NAMES.values():
            self.assertFalse(set(names) & {"seed", "sample_index", "losses", "target", "oracle"})

    def test_strict_margin_ties_and_off_switch(self):
        gains = np.array([[[.1, .1], [.1, .1], [-.1, .3]],
                          [[.1, -.01], [.01, .02], [.01, .5]]])
        np.testing.assert_array_equal(p.select_actions(gains, 0), [0, 1])
        np.testing.assert_array_equal(p.select_actions(gains, .1), [3, 3])
        np.testing.assert_array_equal(p.select_actions(gains, .01), [0, 3])
        np.testing.assert_array_equal(p.select_actions(gains, 0, enabled=False), [3, 3])
        np.testing.assert_array_equal(p.select_actions(gains, None), [3, 3])

    def test_targets_fit_scaling_and_checkpoint_roundtrip(self):
        losses = np.tile(np.array([[2., 4.], [3., 6.], [5., 10.], [4., 8.]])[None], (128, 1, 1))
        model, scale = p.fit_selector(np.ones((128, 1)), losses, self.cfg["tree"])
        np.testing.assert_allclose(scale, [4., 8.])
        expected = np.tile([.5, .5, .25, .25, -.25, -.25], (128, 1))
        np.testing.assert_allclose(model.predict(np.ones((128, 1))), expected)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "local_checkpoint.joblib"
            joblib.dump(model, file)
            np.testing.assert_array_equal(joblib.load(file).predict([[1.]]), model.predict([[1.]]))

    def test_calibration_eligibility_and_conservative_tie(self):
        pred = [np.tile([[.006, .006], [-.1, -.1], [-.1, -.1]], (8, 1, 1))] * 3
        good = np.ones((8, 4, 2))
        good[:, 0] = [.9, .9]
        margin, _ = p.calibrate(pred, [good] * 3, self.cfg["margin_candidates"])
        self.assertEqual(margin, .005)
        bad = np.ones((8, 4, 2))
        bad[:, 0] = [.9, 1.1]
        margin, records = p.calibrate(pred, [bad] * 3, self.cfg["margin_candidates"])
        self.assertIsNone(margin)
        self.assertTrue(all(not r["eligible"] for r in records))

    def test_audit_labels_cannot_change_fit_or_calibration(self):
        sources = p.synthetic_sources()
        fit = slice(*self.cfg["splits"]["fit"])
        dev = slice(*self.cfg["splits"]["development"])
        audit = slice(*self.cfg["splits"]["audit"])
        def freeze():
            model, scale = p.fit_selector(
                np.concatenate([s["features"]["disagreement_aware"][fit] for s in sources]),
                np.concatenate([s["losses"][fit] for s in sources]), self.cfg["tree"])
            pred = [model.predict(s["features"]["disagreement_aware"][dev]).reshape(-1, 3, 2) for s in sources]
            margin, records = p.calibrate(pred, [s["losses"][dev] for s in sources], self.cfg["margin_candidates"])
            return model.tree_.value.copy(), scale, margin, records
        first = freeze()
        for source in sources:
            source["losses"][audit] *= 1000
        second = freeze()
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2:], second[2:])

    def test_oracle_never_worse_than_original_and_disabled_exact(self):
        rng = np.random.default_rng(7)
        loss = rng.uniform(.1, 2, (100, 4, 2))
        oracle = p.selected_losses(loss, p.safe_oracle(loss))
        self.assertTrue((oracle <= loss[:, 0]).all())
        off = p.select_actions(np.ones((100, 3, 2)), 0, enabled=False)
        np.testing.assert_array_equal(p.selected_losses(loss, off), loss[:, 3])

    def test_nonallowlisted_source_rejected_before_io(self):
        with self.assertRaisesRegex(ValueError, "non-allowlisted"):
            p.load_source(2026, "runs/test/metrics.json")

    def test_paired_bootstrap_identity_and_known_gain(self):
        base = np.ones((3, 306, 2))
        identity = p.paired_bootstrap(base, base, self.cfg["bootstrap"])
        np.testing.assert_array_equal(identity["mean_gain_percent_ci95"], [[0, 0], [0, 0]])
        gain = p.paired_bootstrap(base, base * .99, self.cfg["bootstrap"])
        np.testing.assert_allclose(gain["mean_gain_percent_ci95"], [[1, 1], [1, 1]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
