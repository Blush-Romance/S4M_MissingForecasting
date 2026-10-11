"""Small synthetic CPU tests for the offline checkpoint audit, not experiments."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from stage22_42_bank_audit import (
    _bank_summary,
    _geometry,
    _state_changes,
    audit_checkpoint_pair,
    audit_training_log,
)


class BankAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def _checkpoint(self, n=3):
        memory = [torch.tensor([[float(i), 1.0], [float(i) + 1.0, 1.0]]) for i in range(n)]
        return {
            "model": {"decay1.weight": torch.ones(2, 2), "decay1.bias": torch.ones(2), "counter": torch.tensor(2)},
            "memory": memory,
            "repre": torch.stack([item.mean(0) for item in memory]),
            "num_dic": [7] * n + [0] * 4,
        }

    def test_resident_and_tracked_counts_distinct(self):
        result = _bank_summary(self._checkpoint(), 100, 20)
        self.assertEqual(result["resident_sample_counts"]["sum"], 6)
        self.assertEqual(result["num_dic_active_tracked_counts"]["sum"], 21)
        self.assertEqual(result["resident_vs_num_dic_mismatching_centers"], 3)
        self.assertEqual(result["centroid_vs_resident_mean_abs_max"], 0.0)

    def test_all_center_branch(self):
        result = _bank_summary(self._checkpoint(), 100, 20)["retrieval_branch"]
        self.assertEqual(result["weighted_centers"], 3)
        self.assertFalse(result["random_center_subsampling"])
        self.assertEqual(result["mode"], "all_candidate_centers_softmax")

    def test_subsampling_then_topk(self):
        result = _bank_summary(self._checkpoint(8), 5, 2)["retrieval_branch"]
        self.assertTrue(result["random_center_subsampling"])
        self.assertEqual(result["candidate_centers"], 5)
        self.assertEqual(result["weighted_centers"], 2)
        self.assertTrue(result["top_k_selection"])

    def test_rank_cap_and_effective_rank(self):
        result = _geometry(torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
        self.assertEqual(result["centered_rank_cap"], 2)
        self.assertEqual(result["centered_numerical_rank"], 1)
        self.assertAlmostEqual(result["centered_effective_rank"], 1.0)

    def test_single_center_rank_zero(self):
        result = _geometry(torch.ones(1, 2))
        self.assertEqual(result["centered_rank_cap"], 0)
        self.assertEqual(result["centered_effective_rank"], 0)
        self.assertIsNone(result["effective_rank_fraction_of_count_cap"])

    def test_nonfinite_geometry_safe(self):
        result = _geometry(torch.tensor([[float("nan"), 0.0]]))
        self.assertFalse(result["finite"])
        json.dumps(result, allow_nan=False)

    def test_relative_parameter_l2(self):
        result = _state_changes({"decay1.weight": torch.ones(2)}, {"decay1.weight": torch.full((2,), 2.0)})
        self.assertAlmostEqual(result["groups"]["decay1"]["relative_l2"], 1.0)

    def test_shape_mismatches_not_compared(self):
        result = _state_changes({"a": torch.ones(2)}, {"a": torch.ones(3)})
        self.assertEqual(result["shape_mismatches_excluded"], ["a"])
        self.assertEqual(result["groups"], {})

    def test_zero_denominator_not_infinite(self):
        result = _state_changes({"a": torch.zeros(2)}, {"a": torch.ones(2)})
        self.assertIsNone(result["groups"]["a"]["relative_l2"])
        json.dumps(result, allow_nan=False)

    def test_unverified_epoch_is_unknown(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "train.log"
            path.write_text("Epoch: 1, Steps: 2 | Train Loss: 0.1 Vali Loss: 0.2\n")
            result = audit_training_log(path, {"best_observed_validation_loss": 0.2})
            self.assertIsNone(result["best_epoch"])

    def test_pair_cpu_json_and_hash(self):
        with tempfile.TemporaryDirectory() as root:
            runs = []
            for name, n in (("original", 3), ("fixed", 4)):
                run = Path(root) / name
                run.mkdir()
                checkpoint = run / "checkpoint.pth"
                torch.save(self._checkpoint(n), checkpoint)
                (run / "metrics.json").write_text(json.dumps({"checkpoint": str(checkpoint), "seed": 2028, "test_split_accessed": False, "evaluation_split": "validation", "best_observed_validation_loss": 0.2}))
                (run / "train.log").write_text("Epoch: 1, Steps: 2 | Train Loss: 0.1 Vali Loss: 0.2000000\nValidation loss decreased (inf --> 0.200000).  Saving model ...\nEpoch: 2, Steps: 2 | Train Loss: 0.05 Vali Loss: 0.3\n")
                runs.append(run)
            result = audit_checkpoint_pair(*runs)
            self.assertTrue(result["cpu_only"])
            self.assertFalse(result["model_instantiated"])
            self.assertFalse(torch.cuda.is_initialized())
            self.assertEqual(result["original"]["training_log"]["best_epoch"], 1)
            self.assertEqual(result["fixed"]["bank"]["center_count"], 4)
            self.assertEqual(result["original"]["checkpoint_sha256_before"], result["original"]["checkpoint_sha256_after"])
            json.dumps(result, allow_nan=False)

    def test_validation_only_guard(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root)
            (run / "metrics.json").write_text(json.dumps({"evaluation_split": "test", "test_split_accessed": True}))
            with self.assertRaises(ValueError):
                audit_checkpoint_pair(run, run)


if __name__ == "__main__":
    unittest.main(verbosity=2)
