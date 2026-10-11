#!/usr/bin/env python3
"""CPU gates for the two-seed, same-budget distance bug-fix extension.

No dataset split is loaded and no trained checkpoint is deserialized. Real
model initialization and sampler permutations are reproduced on CPU; their
hashes are checked against persisted original S4M metadata for each seed.
"""
from __future__ import annotations

import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "baselines/S4M/s4m"
REPORT = ROOT / "reports/stage22_41_tests.json"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/s4m_stage22_41_matplotlib")
sys.path[:0] = [str(BASE), str(BASE / "model"), str(BASE / "model/S4/s4"),
                str(BASE / "model/S4/s4/extensions/kernels")]
from model.S4M import Model  # noqa: E402
from utils.paired_training_rng import common_state_hash, AuditedRandomSampler  # noqa: E402
import stage22_41_distance_fix as control  # noqa: E402
import run_stage22_41_queue as queue  # noqa: E402
import test_stage22_40_distance_fix as previous_tests  # noqa: E402

EVIDENCE = {}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def old_run(seed, kind="train", tag="formal"):
    state = read(ROOT / "runtime/stage22_37_control/state.json")
    return Path(state["jobs"][f"s4m_{seed}_{kind}_{tag}"]["run_dir"])


def cfg(seed, kind, tag="formal"):
    return yaml.safe_load((control.CONFIG / f"fixed_{seed}_{kind}_{tag}.yaml").read_text())


def namespace(path):
    line = next(line for line in path.read_text().splitlines() if line.startswith("Namespace("))
    call = ast.parse(line, mode="eval").body
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "Namespace"
    values = {item.arg: ast.literal_eval(item.value) for item in call.keywords if item.arg != "device"}
    values["device"] = torch.device("cpu")
    return SimpleNamespace(**values)


class MultiSeedDistanceFixTests(unittest.TestCase):
    def test_eight_frozen_configurations_only_change_declared_bugfix_and_identity(self):
        manifest = control.verify_frozen()
        self.assertEqual(len(manifest), 8)
        self.assertEqual({Path(row["config"]).stem for row in manifest}, set(control.JOBS))
        for row in manifest:
            current = control.validate_delta(yaml.safe_load(Path(row["config"]).read_text()))
            seed = current["experiment"]["seed"]
            self.assertIn(seed, (2028, 2029))
            kind = "eval" if "diagnostic" in current else "train"
            tag = "smoke" if current["experiment"]["smoke_test"] else "formal"
            old = yaml.safe_load(control.template_path(seed, kind, tag).read_text())
            self.assertEqual(current["data"], old["data"])
            self.assertEqual(current["runtime"], old["runtime"])
            self.assertIs(current["provenance"]["fix_enabled"], True)
            if kind == "train":
                args = copy.deepcopy(current["official_args"])
                self.assertIs(args.pop(control.FIX_FLAG), True)
                for key in ("model_id", "des"):
                    args[key] = old["official_args"][key]
                self.assertEqual(args, old["official_args"])
                self.assertEqual(args["seed"], seed)
                self.assertEqual(args["post_train_evaluation"], "validation")
                if tag == "formal":
                    expected = {"batch_size": 16, "train_epochs": 15, "patience": 3,
                                "learning_rate": .001, "optimizer": "adam", "lradj": "type1",
                                "max_train_batches": -1, "max_eval_batches": -1,
                                "seq_len": 192, "pred_len": 96, "e_layers": 8, "d_model": 512}
                    for key, value in expected.items():
                        self.assertEqual(args[key], value, key)
            else:
                self.assertEqual(current["diagnostic"], old["diagnostic"])
                self.assertEqual(current["diagnostic"]["split"], "val")
                self.assertIs(current["diagnostic"]["fixed_eval_rng"], True)
                self.assertEqual(current["diagnostic"]["nested_seed"], seed)

    def test_seed_assets_remain_historic_and_distinct(self):
        records = {}
        for seed in (2028, 2029):
            current = cfg(seed, "train")
            original = yaml.safe_load((old_run(seed) / "resolved_config.yaml").read_text())
            self.assertEqual(current["data"], original["data"])
            self.assertEqual(current["experiment"]["nominal_missing_rate"], .06)
            self.assertEqual(current["experiment"]["block_length"], 5)
            for key in ("nominal_missing_rate", "realized_missing_rate", "mask_type",
                        "block_length", "protocol_match", "training_stability_amendment"):
                self.assertEqual(current["experiment"][key], original["experiment"][key])
            metadata = read(ROOT / current["data"]["mask_metadata"])
            self.assertEqual(metadata["mask_sha256"], sha(ROOT / current["data"]["mask_asset"]))
            self.assertEqual(metadata["output_sha256"], sha(ROOT / current["data"]["missing_csv"]))
            records[str(seed)] = {key: sha(ROOT / current["data"][key])
                                 for key in ("mask_asset", "mask_metadata", "missing_csv")}
        self.assertNotEqual(records["2028"]["mask_asset"], records["2029"]["mask_asset"])
        EVIDENCE["asset_hashes"] = records

    def test_undeclared_training_budget_and_protocol_changes_rejected(self):
        changes = {"batch_size": 8, "learning_rate": .002, "train_epochs": 3,
                   "patience": 1, "seq_len": 96, "pred_len": 48, "seed": 2027,
                   "lradj": "constant", "optimizer": "adamw", "e_layers": 4,
                   "max_train_batches": 1, "max_eval_batches": 1,
                   "post_train_evaluation": "test", "stage22_36_paired_training_rng": False,
                   "use_mipp": True, "use_fusr": True, "fix_min_distance_normalization": False,
                   "use_amp": True, "gradient_accumulation": 2}
        for seed in (2028, 2029):
            for key, value in changes.items():
                bad = copy.deepcopy(cfg(seed, "train"))
                bad["official_args"][key] = value
                with self.subTest(seed=seed, argument=key), self.assertRaises((AssertionError, ValueError)):
                    control.validate_delta(bad)
            for section, key, value in (("runtime", "max_train_batches", 1),
                    ("experiment", "seed", 2027), ("experiment", "nominal_missing_rate", .15),
                    ("experiment", "mask_type", "block"), ("experiment", "block_length", 20),
                    ("experiment", "training_stability_amendment", "undeclared_change"),
                    ("data", "missing_csv", "different.csv"), ("data", "split", [.6, .2, .2]),
                    ("provenance", "fix_enabled", False)):
                bad = copy.deepcopy(cfg(seed, "train"))
                bad[section][key] = value
                with self.subTest(seed=seed, section=section, key=key), self.assertRaises((AssertionError, ValueError)):
                    control.validate_delta(bad)

    def test_undeclared_evaluation_and_test_access_rejected(self):
        changes = {"split": "test", "fixed_eval_rng": False, "nested_seed": 2027,
                   "target_rate": .3, "block_lengths": [1, 20], "max_eval_batches": 1,
                   "variant": "stage22_22_traffic_s4m_test",
                   "multiscale_weights": {"original": 1., "l5": 0., "l20": 0.}}
        for seed in (2028, 2029):
            for key, value in changes.items():
                bad = copy.deepcopy(cfg(seed, "eval"))
                bad["diagnostic"][key] = value
                with self.subTest(seed=seed, diagnostic=key), self.assertRaises((AssertionError, ValueError)):
                    control.validate_delta(bad)
            for section, key, value in (("experiment", "method", "pg95"),
                    ("experiment", "protocol_scope", "historic_unmatched"),
                    ("source", "queue_state", "wrong_state.json"),
                    ("source", "job_id", f"fixed_{seed}_train_smoke")):
                bad = copy.deepcopy(cfg(seed, "eval"))
                bad[section][key] = value
                with self.subTest(seed=seed, section=section, key=key), self.assertRaises((AssertionError, ValueError)):
                    control.validate_delta(bad)

    def test_real_models_switch_preserves_historic_init_parameter_count_and_all_rngs(self):
        hashes = {}
        for seed in (2028, 2029):
            args = namespace(old_run(seed) / "train.log")
            records = []
            for enabled in (False, True):
                config = copy.deepcopy(args)
                config.fix_min_distance_normalization = enabled
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                model = Model(config).float().cpu()
                state = np.random.get_state()
                records.append((common_state_hash(model), sum(p.numel() for p in model.parameters()),
                                random.getstate(), (state[0], state[1].tobytes(), *state[2:]),
                                torch.get_rng_state().clone()))
                self.assertIs(model.fix_min_distance_normalization, enabled)
                del model
            self.assertEqual(records[0][:-1], records[1][:-1])
            self.assertTrue(torch.equal(records[0][-1], records[1][-1]))
            old = read(old_run(seed) / "metrics.json")
            self.assertEqual(records[0][0], (old["paired_initial_common_sha256"], 314))
            self.assertEqual(records[0][1], old["parameter_count"])
            self.assertEqual(records[0][1], 28371290)
            hashes[str(seed)] = records[0][0][0]
        self.assertNotEqual(hashes["2028"], hashes["2029"])
        EVIDENCE["initial_hashes"] = hashes
        EVIDENCE["parameter_count"] = 28371290
        EVIDENCE["state_entries"] = 314

    def test_cpu_sampler_permutations_reproduce_historic_order_hashes_without_training(self):
        audits = {}
        for seed in (2028, 2029):
            old = read(old_run(seed) / "metrics.json")
            self.assertEqual(old["paired_training_rng_protocol"], "stage22_36_v1")
            self.assertIs(old["test_split_accessed"], False)
            audits[str(seed)] = {}
            for key, split in (("paired_train_order", "train"), ("paired_validation_order", "val")):
                recorded = old[key]
                self.assertEqual(len(recorded), old["epochs_completed"])
                self.assertTrue(recorded)
                sampler = AuditedRandomSampler(range(recorded[0]["samples"]), seed, split)
                before = torch.get_rng_state().clone()
                for _ in recorded:
                    list(sampler)
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assertEqual(sampler.history, recorded)
                audits[str(seed)][key] = recorded
        EVIDENCE["historic_sampler_order_cpu_recomputed"] = audits

    def test_old_source_snapshots_match_pre_edit_baseline_except_declared_switch(self):
        evidence = {}
        for seed in (2028, 2029):
            original = old_run(seed)
            manifest = read(original / "source_snapshot_manifest.json")
            verified = {}
            for key, record in manifest.items():
                if not key.startswith("baseline/"):
                    continue
                archived = original / "source_snapshot" / key
                self.assertEqual(sha(archived), record["sha256"], key)
                if key == "baseline/s4m/model/S4M.py":
                    current = previous_tests.PRE_EDIT / "S4M.py"
                elif key == "baseline/s4m/run.py":
                    current = previous_tests.PRE_EDIT / "run.py"
                else:
                    current = Path(record["source"])
                self.assertEqual(sha(current), record["sha256"], key)
                verified[key] = record["sha256"]
            self.assertIn("baseline/s4m/experiments/exp_pretrain1.py", verified)
            self.assertIn("baseline/s4m/model/S4M.py", verified)
            self.assertIn("baseline/s4m/run.py", verified)
            evidence[str(seed)] = verified
        EVIDENCE["historical_baseline_snapshot_hashes"] = evidence

    def test_existing_default_off_model_and_cli_remain_unchanged(self):
        prior = read(ROOT / "reports/stage22_40_distance_fix_tests.json")
        self.assertIs(prior["passed"], True)
        self.assertEqual(sha(BASE / "model/S4M.py"), prior["model_source_sha256"])
        self.assertEqual(sha(ROOT / "scripts/test_stage22_40_distance_fix.py"), prior["test_source_sha256"])
        for name in ("test_declared_default_off_changes_recover_entire_pre_edit_model_ast",
                     "test_cli_has_one_default_false_switch_and_no_unrelated_ast_change",
                     "test_false_fp32_paths_are_bit_exact_immutable_upstream",
                     "test_fix_is_not_an_epsilon_or_other_decay_formula_change",
                     "test_baseline_warmup_keeps_raw_sequence_in_both_states"):
            test = previous_tests.DistanceFixTests(name)
            test.setUp()
            getattr(test, name)()
        EVIDENCE["inherited_default_off_checks"] = "PASS_actual_AST_and_FP32_paths"

    def test_formal_training_gated_by_smoke_and_fixed_validation_depends_on_training(self):
        jobs = {job["id"]: job for job in queue.scheduler.JOB_DEFINITIONS}
        self.assertEqual(len(jobs), 8)
        self.assertEqual(set(jobs), set(control.JOBS))
        self.assertEqual(queue.scheduler.MAX_CONCURRENT_JOBS, 3)
        for seed in (2028, 2029):
            self.assertEqual(jobs[f"fixed_{seed}_eval_smoke"]["depends_on"], [f"fixed_{seed}_train_smoke"])
            self.assertIn(f"fixed_{seed}_eval_smoke", jobs[f"fixed_{seed}_train_formal"]["depends_on"])
            self.assertEqual(jobs[f"fixed_{seed}_eval_formal"]["depends_on"], [f"fixed_{seed}_train_formal"])
            for kind in ("train", "eval"):
                self.assertLessEqual(jobs[f"fixed_{seed}_{kind}_smoke"]["required_memory_estimate_mib"], 6144)

    def test_oom_memory_estimates_are_never_reduced_by_queue_refresh(self):
        definitions = {job["id"]: job for job in queue.scheduler.JOB_DEFINITIONS}
        for job in definitions.values():
            retried = copy.deepcopy(job)
            retried.update(required_memory_estimate_mib=23040, oom_retry_count=2)
            queue.refresh(retried, definitions)
            self.assertGreaterEqual(retried["required_memory_estimate_mib"], 23040)

    def test_actual_gpu_selection_never_stacks_running_jobs_and_refills_after_success(self):
        # These synthetic GPU snapshots never query nvidia-smi or launch work.
        saved = set(queue._active_gpus)
        gpu_a = dict(uuid="GPU-fake-a", total_mib=24564, free_mib=21000,
                     used_mib=3564, name="fake-a", physical_index=0)
        gpu_b = dict(uuid="GPU-fake-b", total_mib=16376, free_mib=12000,
                     used_mib=4376, name="fake-b", physical_index=1)
        probe = dict(id="fixed_2029_train_formal", required_memory_estimate_mib=6144,
                     oom_retry_count=0)
        jobs = {"running": dict(state="RUNNING", gpu=dict(uuid=gpu_a["uuid"]))}
        try:
            queue.refresh(probe, jobs)
            selected = queue.choose_gpu(probe, [gpu_a, gpu_b], {gpu_a["uuid"], gpu_b["uuid"]}, set())
            self.assertIsNotNone(selected)
            self.assertEqual(selected["uuid"], gpu_b["uuid"])
            self.assertIsNone(queue.choose_gpu(probe, [gpu_a], {gpu_a["uuid"]}, set()))
            self.assertIsNone(queue.choose_gpu(probe, [gpu_a, gpu_b],
                                              {gpu_a["uuid"], gpu_b["uuid"]}, {gpu_b["uuid"]}))
            jobs["running"]["state"] = "SUCCESS"
            queue.refresh(probe, jobs)
            selected = queue.choose_gpu(probe, [gpu_a], {gpu_a["uuid"]}, set())
            self.assertIsNotNone(selected)
            self.assertEqual(selected["uuid"], gpu_a["uuid"])
            EVIDENCE["gpu_admission_cpu_fixture"] = dict(running_gpu_excluded=True,
                second_gpu_can_refill=True, single_busy_gpu_waits=True,
                same_poll_double_start_blocked=True, success_releases_gpu=True)
        finally:
            queue._active_gpus = saved

    def test_actual_gpu_selection_keeps_oom_more_free_or_larger_card_requirement(self):
        saved = set(queue._active_gpus)
        job = dict(id="fixed_2028_train_formal", required_memory_estimate_mib=6144,
                   oom_retry_count=1, last_launch_free_mib=16000, last_launch_total_mib=24564)
        gpu = dict(uuid="GPU-fake", total_mib=24564, free_mib=16000,
                   used_mib=8564, name="fake", physical_index=0)
        try:
            queue.refresh(job, {})
            self.assertIsNone(queue.choose_gpu(job, [gpu], {gpu["uuid"]}, set()))
            more_free = {**gpu, "free_mib": 16001}
            self.assertIsNotNone(queue.choose_gpu(job, [more_free], {gpu["uuid"]}, set()))
            larger = {**gpu, "free_mib": 15000, "total_mib": 32768}
            self.assertIsNotNone(queue.choose_gpu(job, [larger], {gpu["uuid"]}, set()))
        finally:
            queue._active_gpus = saved

    def test_required_code_and_documentation_sources_exist_before_binding(self):
        for relative in ("scripts/stage22_41_distance_fix.py", "scripts/run_stage22_41_queue.py",
                         "scripts/start_stage22_41.sh", "docs/STAGE22_41_DISTANCE_FIX_ZH.md"):
            self.assertTrue((ROOT / relative).is_file(), relative)


def source_paths():
    files = [Path(__file__), BASE / "run.py", previous_tests.PRE_EDIT / "S4M.py",
             previous_tests.PRE_EDIT / "run.py", ROOT / "reports/stage22_40_distance_fix_tests.json"]
    files.extend(ROOT / "scripts" / name for name in
                 ("stage22_41_distance_fix.py", "run_stage22_41_queue.py", "start_stage22_41.sh",
                  "stage22_40_distance_fix.py", "test_stage22_40_distance_fix.py",
                  "run_s4m_experiment.py", "run_stage14_uncertainty_cache.py", "run_stage5_baseline_queue.py",
                  "run_stage22_22_traffic_pipeline.py", "stage22_34_factorial.py", "stage22_34_crn.py",
                  "generate_stage22_22_traffic_assets.py", "generate_paper_point_masks.py"))
    for directory in ("model", "experiments", "data_provider", "utils"):
        files.extend((BASE / directory).glob("*.py"))
    for seed in (2028, 2029):
        files.extend((old_run(seed) / "source_snapshot/baseline/s4m").rglob("*.py"))
    doc = ROOT / "docs/STAGE22_41_DISTANCE_FIX_ZH.md"
    if doc.is_file():
        files.append(doc)
    return sorted(set(path.resolve() for path in files if path.is_file()))


if __name__ == "__main__":
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MultiSeedDistanceFixTests)
    identifiers = [test.id().rsplit(".", 1)[-1] for test in suite]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    unsuccessful = {getattr(test, "test_case", test).id().rsplit(".", 1)[-1]
                    for test, _ in result.failures + result.errors}
    prior = read(REPORT) if REPORT.exists() else None
    history = prior.get("history", []) if prior else []
    if prior:
        history.append({key: value for key, value in prior.items() if key != "history"})
    report = dict(stage="22.41", passed=result.wasSuccessful(), tests=result.testsRun,
                  checks={name: name not in unsuccessful for name in identifiers}, device="cpu",
                  dataset_splits_accessed=False, trained_checkpoints_loaded=False,
                  tested_at=datetime.now(timezone.utc).isoformat(), history=history,
                  manifest_sha256=sha(control.MANIFEST),
                  code_hashes={str(path): sha(path) for path in source_paths()}, **EVIDENCE)
    report["code_sources_bound"] = len(report["code_hashes"])
    control.h.write_json(REPORT, report)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"history", "code_hashes", "historical_baseline_snapshot_hashes",
                                     "historic_sampler_order_cpu_recomputed"}}, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
