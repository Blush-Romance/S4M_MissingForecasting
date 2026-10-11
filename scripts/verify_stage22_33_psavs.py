#!/usr/bin/env python3
"""Post-run, read-only numerical verification; does not alter policies or reports."""
import json
from pathlib import Path

import joblib
import numpy as np
import yaml

from run_stage22_33_psavs import ROOT, calibrate, relative_gain, select_actions, sha256


def main():
    state = json.loads((ROOT / "runtime/stage22_33_psavs/state.json").read_text())
    run = Path(state["formal_run_dir"])
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    frozen = json.loads((run / "frozen_selection.json").read_text())
    assert state["state"] == "SUCCESS" and state["return_code"] == 0
    assert metrics["uses_test_split"] is False
    assert sha256(run / "frozen_selection.json") == metrics["frozen_selection_sha256"]
    for source in json.loads((run / "source_manifest.json").read_text()):
        for path, digest in source["input_sha256"].items():
            assert sha256(path) == digest, path
    for relative, record in json.loads((run / "code_manifest.json").read_text()).items():
        assert sha256(ROOT / relative) == sha256(record["snapshot"]) == record["sha256"], relative
    with np.load(run / "offline_validation_inputs.npz", allow_pickle=False) as inputs, np.load(
            run / "audit_choices.npz", allow_pickle=False) as choices:
        dev, audit = (np.arange(*cfg["splits"][name]) for name in ("development", "audit"))
        for policy in cfg["policies"]:
            checkpoint = Path(frozen["calibration"][policy]["checkpoint"])
            assert sha256(checkpoint) == frozen["calibration"][policy]["checkpoint_sha256"]
            package = joblib.load(checkpoint)
            dev_predictions, dev_losses = [], []
            for seed in cfg["experiment"]["source_seeds"]:
                x = inputs[f"seed{seed}_{policy}"]
                pred = package["model"].predict(x[audit]).reshape(-1, 3, 2)
                np.testing.assert_array_equal(pred, choices[f"seed{seed}_{policy}_predicted_gains"])
                dev_predictions.append(package["model"].predict(x[dev]).reshape(-1, 3, 2))
                dev_losses.append(inputs[f"seed{seed}_losses"][dev])
                for variant, margin in (("calibrated", package["margin"]), ("raw", 0.)):
                    action = select_actions(pred, margin)
                    np.testing.assert_array_equal(action, choices[f"seed{seed}_{policy}_{variant}"])
            margin, records = calibrate(dev_predictions, dev_losses, cfg["margin_candidates"])
            assert margin == package["margin"] == frozen["calibration"][policy]["chosen_margin"]
            assert records == frozen["calibration"][policy]["records"]
        for row in metrics["rows"]:
            losses = inputs[f"seed{row['seed']}_losses"][audit]
            if row["policy"].startswith("separate_metric"):
                pair = losses.min(axis=1).mean(0)
            else:
                action = choices[f"seed{row['seed']}_{row['policy']}"]
                pair = losses[np.arange(len(audit)), action].mean(0)
                assert np.bincount(action, minlength=4).tolist() == list(row["action_counts"].values())
            np.testing.assert_allclose(pair, [row["mse"], row["mae"]], rtol=0, atol=1e-14)
            gain = relative_gain(losses[:, 3].mean(0), pair)
            np.testing.assert_allclose(gain, [row["mse_gain_percent"], row["mae_gain_percent"]], rtol=0, atol=1e-12)
        primary = [row for row in metrics["rows"] if row["policy"] == "disagreement_aware_calibrated"]
        gains = np.array([[row["mse_gain_percent"], row["mae_gain_percent"]] for row in primary])
        gate = ((gains > 0).all(1).sum() >= 2) and (gains.mean(0) >= .15).all()
        assert bool(gate) == metrics["scientific_gate_pass"]
        assert np.allclose([metrics["mse"], metrics["mae"]],
                           np.mean([[row["mse"], row["mae"]] for row in primary], axis=0))
    print(json.dumps(dict(verification="PASS", run_dir=str(run), checked_rows=len(metrics["rows"]),
        checks=["source_hashes", "frozen_source_snapshot", "frozen_checkpoint_hashes", "reloaded_predictions",
                "dev_only_calibration", "saved_action_choices", "metrics_recomputed", "scientific_gate"],
        scientific_decision=metrics["decision"], test_accessed=False), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
