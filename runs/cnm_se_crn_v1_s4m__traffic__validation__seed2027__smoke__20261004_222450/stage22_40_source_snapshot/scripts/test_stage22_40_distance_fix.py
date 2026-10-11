#!/usr/bin/env python3
"""CPU contracts for the isolated, default-off minimum-distance bug fix.

The tests execute the actual Model methods, not a second implementation of the
fix. The tiny memory/head fixture isolates the distance path; real S4M model
construction additionally checks that the switch adds no state or RNG draws.
No dataset split or trained checkpoint is opened by this test program.
"""
from __future__ import annotations

import ast
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "baselines/S4M/s4m"
MODEL_PATH = BASE / "model/S4M.py"
SOURCE_LOG = ROOT / "runs/s4m_paired_rng__traffic__point__mr0.06__seed2027__20260908_144251/train.log"
REPORT = ROOT / "reports/stage22_40_distance_fix_tests.json"
PRE_EDIT = ROOT / "runtime/stage22_40_distance_fix/pre_edit"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/s4m_stage22_40_matplotlib")
sys.path[:0] = [str(BASE), str(BASE / "model"), str(BASE / "model/S4/s4"),
                str(BASE / "model/S4/s4/extensions/kernels")]
from model.S4M import Model  # noqa: E402
from utils.paired_training_rng import common_state_hash  # noqa: E402

EVIDENCE = {}


def own_zscore(value):
    return (value - torch.mean(value, dim=1, keepdim=True)) / torch.std(value, dim=1, keepdim=True)


def method_ast(path, method):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method)


def upstream_normalize(maximum, minimum):
    """Execute the immutable upstream's two actual normalization assignments."""
    method = method_ast(ROOT / "upstream/S4M/s4m/model/S4M.py", "forward")
    nodes = [node for node in method.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id in {"max_idx", "min_idx"}
                     for target in node.targets)][:2]
    assert len(nodes) == 2
    namespace = {"torch": torch, "max_idx": maximum, "min_idx": minimum}
    compiled = compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[])),
                       "<immutable-upstream-distance-path>", "exec")
    exec(compiled, namespace)
    return namespace["max_idx"], namespace["min_idx"]


class Memory(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_mipp = False
        self.use_fusr = False
        self.use_cfr = False
        self.last_call = None
        self.warm_call = None

    def forward(self, value, **kwargs):
        self.last_call = (value, kwargs)
        return value

    def warm(self, *args, **kwargs):
        self.warm_call = (args, kwargs)

    def get_last_fusr_outputs(self):
        return None


class TinyDistanceModel(torch.nn.Module):
    _decay_impute = Model._decay_impute
    forward = Model.forward
    warmup = Model.warmup

    def __init__(self, enabled=False):
        super().__init__()
        self.configs = SimpleNamespace(stage22_36_paired_training_rng=False, seed=2027)
        self.fix_min_distance_normalization = enabled
        self.decay1 = torch.nn.Linear(3, 3, dtype=torch.float64)
        self.decay2 = torch.nn.Linear(3, 3, dtype=torch.float64)
        self.mem_net = Memory()
        self.mask_trans = torch.nn.Identity()
        self.use_mrra = False
        self.use_mcpr = False
        self.last_mrra_outputs = None
        self.last_mcpr_outputs = None
        self.decay_inputs = {}
        for name in ("decay1", "decay2"):
            getattr(self, name).register_forward_pre_hook(
                lambda module, args, key=name: self.decay_inputs.update({key: args[0].detach().clone()}))

    def _forecast_from_context(self, context, mask, observation_mask=None):
        return context


def representative_inputs():
    # Real-style L=192 absolute distances with a different extremum per variable.
    time = torch.arange(192, dtype=torch.float64).view(1, 192, 1)
    maximum = (time - torch.tensor([16., 93., 170.]).view(1, 1, 3)).abs().repeat(2, 1, 1)
    minimum = (time - torch.tensor([140., 37., 111.]).view(1, 1, 3)).abs().repeat(2, 1, 1)
    values = torch.linspace(-1.3, 1.7, 2 * 192 * 3, dtype=torch.float64).reshape(2, 192, 3)
    mask = ((torch.arange(values.numel()).reshape_as(values) % 4) != 0).to(torch.float64)
    upper = torch.full_like(values, 1.8)
    lower = torch.full_like(values, -1.6)
    return values, mask, maximum, minimum, upper, lower


def source_namespace():
    line = next(line for line in SOURCE_LOG.read_text().splitlines() if line.startswith("Namespace("))
    node = ast.parse(line, mode="eval").body
    assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Namespace"
    values = {item.arg: ast.literal_eval(item.value) for item in node.keywords if item.arg != "device"}
    values["device"] = torch.device("cpu")
    return SimpleNamespace(**values)


class DistanceFixTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2040)
        self.inputs = representative_inputs()
        self.model = TinyDistanceModel(False).eval()

    def run_path(self, enabled, path="helper", inputs=None):
        self.model.fix_min_distance_normalization = enabled
        args = self.inputs if inputs is None else inputs
        method = self.model._decay_impute if path == "helper" else self.model.forward
        return method(*args)

    def reference(self):
        values, mask, maximum, minimum, upper, lower = self.inputs
        maximum, minimum = upstream_normalize(maximum, minimum)
        gamma2 = torch.exp(-1 * self.model.decay1(maximum))
        gamma1 = torch.exp(-1 * self.model.decay2(minimum))
        total = gamma1 + gamma2
        return (gamma1 / total * upper + gamma2 / total * lower) * (1 - mask) + values * mask

    def test_false_helper_and_forward_are_bit_exact_immutable_upstream(self):
        expected = self.reference()
        for path in ("helper", "forward"):
            with self.subTest(path=path):
                self.assertTrue(torch.equal(self.run_path(False, path), expected))

    def test_false_fp32_paths_are_bit_exact_immutable_upstream(self):
        self.model.float()
        self.inputs = tuple(value.float() for value in self.inputs)
        expected = self.reference()
        for path in ("helper", "forward"):
            self.assertTrue(torch.equal(self.run_path(False, path), expected))

    def test_true_helper_and_forward_use_own_raw_min_zscore(self):
        expected_maximum = own_zscore(self.inputs[2])
        expected_minimum = own_zscore(self.inputs[3])
        for path in ("helper", "forward"):
            with self.subTest(path=path):
                self.run_path(True, path)
                self.assertTrue(torch.equal(self.model.decay_inputs["decay1"], expected_maximum))
                self.assertTrue(torch.equal(self.model.decay_inputs["decay2"], expected_minimum))
                torch.testing.assert_close(self.model.decay_inputs["decay2"].mean(1),
                                           torch.zeros(2, 3, dtype=torch.float64), atol=1e-14, rtol=0)
                torch.testing.assert_close(self.model.decay_inputs["decay2"].std(1),
                                           torch.ones(2, 3, dtype=torch.float64), atol=1e-14, rtol=0)

    def test_max_branch_is_unchanged_by_fix(self):
        for path in ("helper", "forward"):
            self.run_path(False, path)
            original = self.model.decay_inputs["decay1"].clone()
            self.run_path(True, path)
            self.assertTrue(torch.equal(original, self.model.decay_inputs["decay1"]))

    def test_fixed_min_branch_does_not_read_max_distance(self):
        changed = list(self.inputs)
        changed[2] = changed[2].flip(1)
        for path in ("helper", "forward"):
            self.run_path(True, path)
            fixed = self.model.decay_inputs["decay2"].clone()
            self.run_path(True, path, changed)
            self.assertTrue(torch.equal(fixed, self.model.decay_inputs["decay2"]))
            self.run_path(False, path)
            original = self.model.decay_inputs["decay2"].clone()
            self.run_path(False, path, changed)
            self.assertFalse(torch.equal(original, self.model.decay_inputs["decay2"]))

    def test_fixed_min_branch_preserves_own_temporal_shape(self):
        changed = list(self.inputs)
        changed[3] = changed[3].flip(1)  # Same own mean/std, different temporal shape.
        for path in ("helper", "forward"):
            self.run_path(True, path)
            fixed = self.model.decay_inputs["decay2"].clone()
            self.run_path(True, path, changed)
            torch.testing.assert_close(self.model.decay_inputs["decay2"], fixed.flip(1), atol=1e-14, rtol=0)
            self.assertFalse(torch.equal(fixed, self.model.decay_inputs["decay2"]))
            self.run_path(False, path)
            original = self.model.decay_inputs["decay2"].clone()
            self.run_path(False, path, changed)
            torch.testing.assert_close(original, self.model.decay_inputs["decay2"], atol=1e-14, rtol=0)

    def test_fixed_min_branch_translation_and_positive_scale_invariant(self):
        changed = list(self.inputs)
        changed[3] = changed[3] * 3.5 + 17.
        for path in ("helper", "forward"):
            self.run_path(True, path)
            fixed = self.model.decay_inputs["decay2"].clone()
            self.run_path(True, path, changed)
            torch.testing.assert_close(fixed, self.model.decay_inputs["decay2"], atol=1e-14, rtol=0)

    def test_forward_and_helper_agree_in_both_switch_states(self):
        for enabled in (False, True):
            self.assertTrue(torch.equal(self.run_path(enabled), self.run_path(enabled, "forward")))

    def test_observed_values_unchanged_and_missing_mixture_bounded(self):
        values, mask, _, _, upper, lower = self.inputs
        for enabled in (False, True):
            for path in ("helper", "forward"):
                result = self.run_path(enabled, path)
                self.assertTrue(torch.equal(result[mask.bool()], values[mask.bool()]))
                self.assertTrue(bool(torch.isfinite(result).all()))
                self.assertTrue(bool((result[~mask.bool()] <= upper[~mask.bool()]).all()))
                self.assertTrue(bool((result[~mask.bool()] >= lower[~mask.bool()]).all()))

    def test_actual_mipp_warmup_calls_fixed_helper_for_both_views(self):
        self.model.mem_net.use_mipp = True
        self.model.fix_min_distance_normalization = True
        pair = dict(zip(("seq_x", "seq_x_mask", "max_idx", "min_idx", "max_value", "min_value"),
                        self.inputs))
        pair["min_idx"] = self.inputs[3].flip(1)
        self.model.warmup(*self.inputs, mipp_pair=pair)
        calls, kwargs = self.model.mem_net.warm_call
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(calls[0], self.run_path(True)))
        self.assertTrue(torch.equal(calls[1], self.run_path(True, inputs=list(pair.values()))))
        self.assertIs(kwargs["seq_mask"], self.inputs[1])

    def test_baseline_warmup_keeps_raw_sequence_in_both_states(self):
        for enabled in (False, True):
            self.model.fix_min_distance_normalization = enabled
            before = torch.get_rng_state().clone()
            self.assertIsNone(self.model.warmup(*self.inputs))
            args, kwargs = self.model.mem_net.warm_call
            self.assertEqual(len(args), 1)
            self.assertIs(args[0], self.inputs[0])
            self.assertEqual(kwargs, {})
            self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_actual_forward_pair_uses_same_fixed_helper(self):
        self.model.mem_net.use_mipp = True
        pair = dict(zip(("seq_x", "seq_x_mask", "max_idx", "min_idx", "max_value", "min_value"),
                        self.inputs))
        pair["min_idx"] = self.inputs[3].flip(1)
        self.model.fix_min_distance_normalization = True
        output = self.model(*self.inputs, mipp_pair=pair)
        primary, kwargs = self.model.mem_net.last_call
        self.assertTrue(torch.equal(output, primary))
        self.assertTrue(torch.equal(kwargs["pair_Q"], self.run_path(True, inputs=list(pair.values()))))

    def test_gradients_finite_for_representative_distances_and_decay_parameters(self):
        for path in ("helper", "forward"):
            self.model.zero_grad(set_to_none=True)
            inputs = [value.detach().clone().requires_grad_(True) if index != 1 else value
                      for index, value in enumerate(self.inputs)]
            output = self.run_path(True, path, inputs)
            output.square().mean().backward()
            for index in (0, 2, 3, 4, 5):
                self.assertIsNotNone(inputs[index].grad)
                self.assertTrue(bool(torch.isfinite(inputs[index].grad).all()))
            for parameter in self.model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))

    def test_switch_adds_no_fixture_state_and_eval_consumes_no_rng(self):
        before_keys = tuple(self.model.state_dict())
        for enabled in (False, True):
            before = torch.get_rng_state().clone()
            self.run_path(enabled, "forward")
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(before_keys, tuple(self.model.state_dict()))

    def test_both_states_preserve_training_stream_global_rng(self):
        self.model.train()
        for paired_rng in (False, True):
            self.model.configs.stage22_36_paired_training_rng = paired_rng
            for enabled in (False, True):
                before = torch.get_rng_state().clone()
                self.run_path(enabled, "forward")
                self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_declared_default_off_changes_recover_entire_pre_edit_model_ast(self):
        current = ast.parse(MODEL_PATH.read_text())
        original = ast.parse((PRE_EDIT / "S4M.py").read_text())
        cls = next(node for node in current.body if isinstance(node, ast.ClassDef) and node.name == "Model")
        initializer = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        additions = [node for node in initializer.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Attribute) and target.attr == "fix_min_distance_normalization"
                             for target in node.targets)]
        self.assertEqual(len(additions), 1)
        expression = ast.parse("self.fix_min_distance_normalization = bool(getattr(configs, 'fix_min_distance_normalization', False))").body[0]
        self.assertEqual(ast.dump(additions[0], include_attributes=False), ast.dump(expression, include_attributes=False))
        initializer.body.remove(additions[0])
        for name in ("_decay_impute", "forward"):
            method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
            branches = [node for node in method.body if isinstance(node, ast.If)
                        and isinstance(node.test, ast.Attribute)
                        and node.test.attr == "fix_min_distance_normalization"]
            self.assertEqual(len(branches), 1)
            index = method.body.index(branches[0])
            method.body[index:index + 1] = branches[0].orelse
        self.assertEqual(ast.dump(current, include_attributes=False), ast.dump(original, include_attributes=False))

    def test_cli_has_one_default_false_switch_and_no_unrelated_ast_change(self):
        current = ast.parse((BASE / "run.py").read_text())
        original = ast.parse((PRE_EDIT / "run.py").read_text())
        added = []
        for parent in ast.walk(current):
            for field, value in ast.iter_fields(parent):
                if not isinstance(value, list):
                    continue
                for index, node in enumerate(value):
                    if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                            and any(isinstance(argument, ast.Constant)
                                    and argument.value == "--fix_min_distance_normalization"
                                    for argument in node.value.args)):
                        added.append(node.value)
                        del value[index]
                        break
        self.assertEqual(len(added), 1)
        parser = argparse.ArgumentParser(add_help=False)
        isolated = ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=copy.deepcopy(added[0]))], type_ignores=[]))
        exec(compile(isolated, "<actual-cli-switch>", "exec"), {"parser": parser})
        self.assertFalse(parser.parse_args([]).fix_min_distance_normalization)
        import run_s4m_experiment as launcher
        for enabled in (False, True):
            command = launcher.cli_from_mapping({"fix_min_distance_normalization": enabled})
            self.assertEqual(command, ["--fix_min_distance_normalization"] if enabled else [])
            self.assertEqual(parser.parse_args(command).fix_min_distance_normalization, enabled)
        self.assertEqual(ast.dump(current, include_attributes=False), ast.dump(original, include_attributes=False))

    def test_pre_edit_and_unchanged_training_code_match_historical_seed2027_snapshot(self):
        original_run = SOURCE_LOG.parent
        manifest = json.loads((original_run / "source_snapshot_manifest.json").read_text())
        verified = {}
        for key, record in manifest.items():
            if not key.startswith("baseline/"):
                continue
            archived = original_run / "source_snapshot" / key
            self.assertEqual(hashlib.sha256(archived.read_bytes()).hexdigest(), record["sha256"], key)
            if key == "baseline/s4m/model/S4M.py":
                current = PRE_EDIT / "S4M.py"
            elif key == "baseline/s4m/run.py":
                current = PRE_EDIT / "run.py"
            else:
                current = Path(record["source"])
            self.assertEqual(hashlib.sha256(current.read_bytes()).hexdigest(), record["sha256"], key)
            verified[key] = record["sha256"]
        self.assertIn("baseline/s4m/experiments/exp_pretrain1.py", verified)
        self.assertIn("baseline/s4m/model/S4M.py", verified)
        self.assertIn("baseline/s4m/run.py", verified)
        EVIDENCE["historical_baseline_snapshot_hashes"] = verified

    def test_fix_is_not_an_epsilon_or_other_decay_formula_change(self):
        for name in ("_decay_impute", "forward"):
            method = method_ast(MODEL_PATH, name)
            branch = next(node for node in method.body if isinstance(node, ast.If)
                          and isinstance(node.test, ast.Attribute)
                          and node.test.attr == "fix_min_distance_normalization")
            assignment = branch.body[0]
            expression = ast.unparse(assignment.value)
            self.assertNotIn("max_idx", expression)
            self.assertNotIn("eps", expression)
            self.assertNotIn("clamp", expression)
            expected = ast.parse("min_idx = (min_idx-torch.mean(min_idx,dim=1,keepdims=True))/torch.std(min_idx,dim=1,keepdims=True)").body[0]
            self.assertEqual(ast.dump(assignment, include_attributes=False),
                             ast.dump(expected, include_attributes=False))

    def test_real_model_absent_false_true_identical_initial_state_and_rng(self):
        hashes, states, python_states, numpy_states, parameter_counts = [], [], [], [], []
        original = source_namespace()
        for flag in (None, False, True):
            args = copy.deepcopy(original)
            if flag is None:
                vars(args).pop("fix_min_distance_normalization", None)
            else:
                args.fix_min_distance_normalization = flag
            random.seed(2027)
            np.random.seed(2027)
            torch.manual_seed(2027)
            model = Model(args).float().cpu()
            self.assertEqual(model.fix_min_distance_normalization, flag is True)
            hashes.append(common_state_hash(model))
            states.append(torch.get_rng_state().clone())
            python_states.append(random.getstate())
            numpy_state = np.random.get_state()
            numpy_states.append((numpy_state[0], numpy_state[1].tobytes(), *numpy_state[2:]))
            parameter_counts.append(sum(parameter.numel() for parameter in model.parameters()))
            del model
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(hashes[1], hashes[2])
        self.assertTrue(torch.equal(states[0], states[1]))
        self.assertTrue(torch.equal(states[1], states[2]))
        self.assertEqual(python_states[0], python_states[1])
        self.assertEqual(python_states[1], python_states[2])
        self.assertEqual(numpy_states[0], numpy_states[1])
        self.assertEqual(numpy_states[1], numpy_states[2])
        self.assertEqual(len(set(parameter_counts)), 1)
        self.assertEqual(hashes[0], ("0aea2be2145809f6f3ade6f5680e084a05aae5eaff106ef1554f8346efe76db6", 314))
        self.assertEqual(parameter_counts[0], 28371290)
        EVIDENCE["initial_state_sha256"] = hashes[0][0]
        EVIDENCE["state_entries"] = hashes[0][1]
        EVIDENCE["parameter_count"] = parameter_counts[0]
        EVIDENCE["configuration_source"] = str(SOURCE_LOG.relative_to(ROOT))

    def test_six_frozen_configurations_have_only_declared_delta(self):
        import stage22_40_distance_fix as control
        rows = control.verify_frozen()
        self.assertEqual(len(rows), 6)
        self.assertEqual({Path(row["config"]).stem for row in rows}, set(control.JOBS))
        for row in rows:
            control.validate_delta(yaml.safe_load(Path(row["config"]).read_text()))
        baseline = json.loads((control.RT / "baseline_manifest.json").read_text())
        self.assertEqual(baseline["expected_common_sha256"],
                         "0aea2be2145809f6f3ade6f5680e084a05aae5eaff106ef1554f8346efe76db6")

    def test_unapproved_training_budget_and_protocol_changes_are_rejected(self):
        import stage22_40_distance_fix as control
        original = yaml.safe_load((control.CONFIG / "fixed_train_formal.yaml").read_text())
        changes = {
            "batch_size": 8, "learning_rate": .002, "train_epochs": 3,
            "patience": 1, "seq_len": 96, "pred_len": 48, "seed": 2028,
            "lradj": "constant", "optimizer": "adamw", "e_layers": 4,
            "max_train_batches": 1, "max_eval_batches": 1,
            "post_train_evaluation": "test", "stage22_36_paired_training_rng": False,
            "use_mipp": True, "fix_min_distance_normalization": False,
        }
        for key, value in changes.items():
            with self.subTest(changed_argument=key):
                bad = copy.deepcopy(original)
                bad["official_args"][key] = value
                with self.assertRaises((AssertionError, ValueError)):
                    control.validate_delta(bad)
        for section, key, value in (("runtime", "max_train_batches", 1),
                                    ("experiment", "seed", 2028),
                                    ("experiment", "nominal_missing_rate", .15),
                                    ("experiment", "mask_type", "block"),
                                    ("experiment", "block_length", 20),
                                    ("experiment", "training_stability_amendment", "undeclared_change"),
                                    ("data", "missing_csv", "different.csv"),
                                    ("provenance", "fix_enabled", 1)):
            with self.subTest(changed_section=section, changed_key=key):
                bad = copy.deepcopy(original)
                bad[section][key] = value
                with self.assertRaises((AssertionError, ValueError)):
                    control.validate_delta(bad)

    def test_unapproved_evaluation_changes_and_test_access_are_rejected(self):
        import stage22_40_distance_fix as control
        original = yaml.safe_load((control.CONFIG / "fixed_eval_formal.yaml").read_text())
        changes = {"split": "test", "fixed_eval_rng": False, "nested_seed": 2028,
                   "target_rate": .3, "block_lengths": [1, 20], "max_eval_batches": 1,
                   "variant": "stage22_22_traffic_s4m_test"}
        for key, value in changes.items():
            with self.subTest(changed_diagnostic=key):
                bad = copy.deepcopy(original)
                bad["diagnostic"][key] = value
                with self.assertRaises((AssertionError, ValueError)):
                    control.validate_delta(bad)
        for section, key, value in (("experiment", "method", "pg95"),
                                    ("experiment", "protocol_scope", "historic_unmatched"),
                                    ("source", "queue_state", "wrong_state.json")):
            bad = copy.deepcopy(original)
            bad[section][key] = value
            with self.assertRaises((AssertionError, ValueError)):
                control.validate_delta(bad)

    def test_queue_dependencies_and_oom_memory_estimate_never_decrease(self):
        import run_stage22_40_queue as queue
        jobs = {job["id"]: job for job in queue.scheduler.JOB_DEFINITIONS}
        self.assertEqual(len(jobs), 6)
        self.assertEqual(queue.scheduler.MAX_CONCURRENT_JOBS, 3)
        self.assertEqual(set(jobs["fixed_train_formal"]["depends_on"]),
                         {"off_train_smoke", "fixed_train_smoke", "off_eval_smoke", "fixed_eval_smoke"})
        self.assertEqual(jobs["fixed_eval_formal"]["depends_on"], ["fixed_train_formal"])
        for name, job in jobs.items():
            self.assertLessEqual(job["required_memory_estimate_mib"], 6144)
            retried = copy.deepcopy(job)
            retried.update(required_memory_estimate_mib=23040, oom_retry_count=2)
            queue.refresh(retried, jobs)
            self.assertGreaterEqual(retried["required_memory_estimate_mib"], 23040)

    def test_queue_nonblocking_oom_resume_simulation(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/run_stage22_40_queue.py"), "--self-test"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["self_test"], "PASS")
        destination = ROOT / "reports/stage22_40_queue_tests.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(destination)


if __name__ == "__main__":
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DistanceFixTests)
    identifiers = [test.id().rsplit(".", 1)[-1] for test in suite]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    unsuccessful = {getattr(test, "test_case", test).id().rsplit(".", 1)[-1]
                    for test, _ in result.failures + result.errors}
    prior = json.loads(REPORT.read_text()) if REPORT.exists() else None
    history = prior.get("history", []) if prior else []
    if prior:
        history.append({key: value for key, value in prior.items() if key != "history"})
    report = {
        "stage": "22.40", "passed": result.wasSuccessful(), "tests": result.testsRun,
        "checks": {name: name not in unsuccessful for name in identifiers},
        "device": "cpu", "dataset_splits_accessed": False,
        "tested_at": datetime.now(timezone.utc).isoformat(),
        "model_source_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "history": history, **EVIDENCE,
    }
    manifest = ROOT / "runtime/stage22_40_distance_fix/config_manifest.json"
    report["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None
    report["expected_common_sha256"] = EVIDENCE.get("initial_state_sha256")
    sources = [Path(__file__), BASE / "run.py", PRE_EDIT / "S4M.py", PRE_EDIT / "run.py",
               ROOT / "docs/STAGE22_40_DISTANCE_FIX_ZH.md"]
    sources.extend(ROOT / "scripts" / name for name in
                   ("stage22_40_distance_fix.py", "run_stage22_40_queue.py", "start_stage22_40.sh",
                    "run_s4m_experiment.py", "run_stage14_uncertainty_cache.py",
                    "run_stage5_baseline_queue.py", "run_stage22_22_traffic_pipeline.py",
                    "stage22_34_factorial.py", "stage22_34_crn.py"))
    for directory in ("model", "experiments", "data_provider", "utils"):
        sources.extend((BASE / directory).glob("*.py"))
    old_snapshots = SOURCE_LOG.parent / "source_snapshot" / "baseline/s4m"
    sources.extend(old_snapshots.rglob("*.py"))
    report["code_hashes"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(set(sources)) if path.is_file()}
    report["code_sources_bound"] = len(report["code_hashes"])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(REPORT)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"history", "code_hashes", "historical_baseline_snapshot_hashes"}}, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
