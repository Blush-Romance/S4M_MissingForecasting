#!/usr/bin/env python3
"""Configuration, source, queue-memory and factorial-analysis regression tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

import stage22_34_factorial as stage
from run_stage14_uncertainty_cache import validate_config, run_id
import run_stage22_34_factorial_queue as queue


class FactorialTests(unittest.TestCase):
    def setUp(self):
        self.configs = [yaml.safe_load(p.read_text()) for p in sorted(stage.CONFIG_ROOT.glob("*.yaml"))]

    def test_all_four_configs_preserve_frozen_source(self):
        self.assertEqual(len(self.configs), 4)
        for cfg in self.configs:
            source = yaml.safe_load(Path(cfg["source"]["resolved_config"]).read_text())
            validate_config(cfg, source)
            args = source["official_args"]
            self.assertEqual((args["batch_size"], args["seq_len"], args["pred_len"], args["enc_in"]), (16, 192, 96, 862))
            self.assertFalse(args.get("use_mipp", False))
            self.assertIn("__validation__", run_id(cfg, "frozen"))
            self.assertNotIn("__test__", run_id(cfg, "frozen"))
            self.assertEqual(cfg["diagnostic"]["max_eval_batches"], 3 if cfg["experiment"]["smoke_test"] else -1)

    def test_test_and_changed_method_rejected(self):
        for field, value in (("split", "test"), ("target_rate", .2), ("nested_seed", 2030),
                             ("block_lengths", [1, 5, 10]), ("variant", "unsupported_stage")):
            cfg = copy.deepcopy(self.configs[0])
            source = yaml.safe_load(Path(cfg["source"]["resolved_config"]).read_text())
            cfg["diagnostic"][field] = value
            with self.assertRaises(ValueError):
                validate_config(cfg, source)
        cfg = copy.deepcopy(self.configs[0])
        source = yaml.safe_load(Path(cfg["source"]["resolved_config"]).read_text())
        cfg["experiment"]["method"] = "pg95"
        with self.assertRaises(ValueError):
            validate_config(cfg, source)

    def test_wrong_checkpoint_family_rejected(self):
        cfg = self.configs[0]
        source = yaml.safe_load(Path(cfg["source"]["resolved_config"]).read_text())
        source["official_args"]["use_mipp"] = True
        with self.assertRaisesRegex(ValueError, "original baseline"):
            validate_config(cfg, source)

    def test_smoke_is_first_dependency_and_estimate_not_reduced_after_oom(self):
        jobs = queue.scheduler.JOB_DEFINITIONS
        self.assertEqual(len(jobs), 4)
        self.assertTrue(jobs[0]["id"].endswith("smoke"))
        self.assertEqual(jobs[1]["depends_on"], [jobs[0]["id"]])
        self.assertEqual(jobs[2]["depends_on"], [jobs[1]["id"]])
        job = copy.deepcopy(jobs[1])
        queue.refresh_memory(job, {})
        self.assertEqual(job["required_memory_estimate_mib"], 4608)
        job["required_memory_estimate_mib"] = 8192
        queue.refresh_memory(job, {})
        self.assertEqual(job["required_memory_estimate_mib"], 8192)

    def test_old_pg95_validation_remains_accepted(self):
        for seed, relative in stage.PG95_RUNS.items():
            pg95 = stage.ROOT / relative
            cfg = yaml.safe_load((pg95 / "resolved_config.yaml").read_text())
            source = yaml.safe_load(Path(cfg["source"]["resolved_config"]).read_text())
            validate_config(cfg, source)
            self.assertIn("cnm_se_equal_rate_multiscale_pg95__traffic__validation", run_id(cfg, "frozen"))

    def test_factorial_identity_and_unpaired_masks_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "s4m", Path(directory) / "pg95"]
            for i, path in enumerate(paths):
                path.mkdir()
                on, ms = ((1., .95) if i == 0 else (.9, .8))
                metrics = dict(source_split="val", uses_test_split=False, updates_model_or_memory=False,
                    sample_count=1648, artifact_status="PASS", artifact_checks={"fixture": True},
                    multiscale_weights=dict(original=.5, l5=.25, l20=.25), mask_asset_sha256="fixed", source_checkpoint_sha256="fixed")
                for metric in ("mse", "mae"):
                    metrics[f"complete_target_{metric}_on"] = on
                    metrics[f"complete_target_{metric}_multiscale"] = ms
                (path / "metrics.json").write_text(json.dumps(metrics))
                (path / "status.json").write_text(json.dumps(dict(state="SUCCESS", return_code=0)))
                cfg = dict(experiment=dict(seed=2027), data=dict(mask_asset="fixture"), source=dict(checkpoint="fixture"),
                    official_args=dict(seed=2027, use_mipp=bool(i), batch_size=16, seq_len=192, pred_len=96,
                        enc_in=862, e_layers=8, d_model=512, fillna="mean", mean_type="global_mean"))
                (path / "resolved_config.yaml").write_text(yaml.safe_dump(cfg))
                arrays = dict(sample_index=np.arange(1648), complete_count=np.ones(1648))
                for key in ("history_mask_packed", "l1_mask_packed", "l5_mask_packed", "l20_mask_packed"):
                    arrays[key] = np.zeros((1648, 24), dtype=np.uint8)
                for key in ("target_removed_count", "removed_count_l1", "removed_count_l5", "removed_count_l20"):
                    arrays[key] = np.zeros(1648)
                for route, value in (("on", on), ("multiscale", ms)):
                    for kind in ("sse", "sae"):
                        arrays[f"complete_{kind}_{route}"] = np.full(1648, value)
                np.savez_compressed(path / "validation_equal_rate_multiscale_samples.npz", **arrays)
            with patch.object(stage, "digest", return_value="fixed"):
                result = stage.compare_seed(2027, *paths)
                np.testing.assert_allclose(result["interaction_absolute"], [.05, .05])
                np.testing.assert_allclose(result["pg95_gain_with_cnm_percent"], [(.95 - .8) / .95 * 100] * 2)
                arrays["history_mask_packed"][0, 0] = 1
                np.savez_compressed(paths[1] / "validation_equal_rate_multiscale_samples.npz", **arrays)
                with self.assertRaisesRegex(AssertionError, "unpaired"):
                    stage.compare_seed(2027, *paths)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FactorialTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    stage.write_json(stage.ROOT / "reports/stage22_34_factorial_test.json", dict(
        status="PASS" if result.wasSuccessful() else "FAIL", tests_run=result.testsRun,
        failures=[str(test) for test, _ in result.failures], errors=[str(test) for test, _ in result.errors]))
    raise SystemExit(0 if result.wasSuccessful() else 1)
