"""Reproducible CPU-only source and synthetic audit; never opens a dataset split.

Actual assignment ASTs are executed in isolated namespaces. No original model
is instantiated, no checkpoint is loaded, and no baseline source is changed.
"""
import ast
import copy
import datetime
import hashlib
import json
import os
import re
import subprocess
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "baselines/S4M"
UPSTREAM = ROOT / "upstream/S4M"
REPORT = ROOT / "reports/stage22_39_distance_audit.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(path):
    text = Path(path).read_text()
    return text, ast.parse(text)


def method(path, cls, name):
    text, tree = parse(path)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fun = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return text, fun


def assignments(fun, targets):
    result = []
    for node in fun.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in targets for t in node.targets):
            result.append(node)
        if len(result) == len(targets):
            break
    assert len(result) == len(targets), (fun.name, targets)
    return result


def evidence(path, nodes):
    text = Path(path).read_text()
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
            "lines": [{"line": n.lineno, "source": ast.get_source_segment(text, n)} for n in nodes]}


def execute(nodes, namespace):
    module = ast.Module(body=copy.deepcopy(nodes), type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<source-audit>", "exec"), namespace)
    return namespace


def distances(length, position):
    return (torch.arange(length, dtype=torch.float64) - position).abs()[None, :, None]


def normalize(nodes, maximum, minimum):
    ns = execute(nodes, {"torch": torch, "max_idx": maximum.clone(), "min_idx": minimum.clone()})
    return ns["max_idx"], ns["min_idx"]


def finite_summary(value):
    finite = value[torch.isfinite(value)]
    return {"shape": list(value.shape), "nonfinite": int((~torch.isfinite(value)).sum()),
            "mean_finite": float(finite.mean()) if finite.numel() else None,
            "std_finite": float(finite.std()) if finite.numel() > 1 else None,
            "min_finite": float(finite.min()) if finite.numel() else None,
            "max_finite": float(finite.max()) if finite.numel() else None}


def audit():
    torch.set_num_threads(1)
    paths = {
        "baseline_model": BASE / "s4m/model/S4M.py",
        "upstream_model": UPSTREAM / "s4m/model/S4M.py",
        "baseline_bank": BASE / "s4m/model/Bank.py",
        "upstream_bank": UPSTREAM / "s4m/model/Bank.py",
        "baseline_loader": BASE / "s4m/data_provider/data_loader.py",
        "upstream_loader": UPSTREAM / "s4m/data_provider/data_loader.py",
    }
    source_before = {key: sha(path) for key, path in paths.items()}
    _, forward = method(paths["baseline_model"], "Model", "forward")
    _, decay = method(paths["baseline_model"], "Model", "_decay_impute")
    _, original_forward = method(paths["upstream_model"], "Model", "forward")
    norms = assignments(forward, {"max_idx", "min_idx"})
    decay_norms = assignments(decay, {"max_idx", "min_idx"})
    original_norms = assignments(original_forward, {"max_idx", "min_idx"})
    ast_equal = lambda left, right: ast.dump(ast.Module(body=left, type_ignores=[]), include_attributes=False) == ast.dump(ast.Module(body=right, type_ignores=[]), include_attributes=False)
    tests = {}
    tests["forward_normalization_ast_matches_upstream"] = ast_equal(norms, original_norms)
    tests["helper_normalization_ast_matches_forward"] = ast_equal(norms, decay_norms)
    min_names = {n.id for n in ast.walk(norms[1].value) if isinstance(n, ast.Name)}
    tests["min_normalization_reads_overwritten_max"] = "max_idx" in min_names

    raw_max, raw_min = distances(192, 16), distances(192, 140)
    actual_max, actual_min = normalize(norms, raw_max, raw_min)
    intended_min = (raw_min - raw_min.mean(1, keepdim=True)) / raw_min.std(1, keepdim=True)
    tests["actual_min_is_not_standardized_own_distance"] = not torch.allclose(actual_min, intended_min)
    shifted_actual = normalize(norms, raw_max, raw_min + 10)[1]
    shifted_intended = ((raw_min + 10) - (raw_min + 10).mean(1, keepdim=True)) / (raw_min + 10).std(1, keepdim=True)
    tests["own_standardization_is_translation_invariant"] = torch.allclose(intended_min, shifted_intended)
    tests["actual_min_is_not_translation_invariant"] = not torch.allclose(actual_min, shifted_actual)
    min_synthetic = {
        "case": "L=192, max position=16, min position=140, one variable; no dataset accessed",
        "raw_max": finite_summary(raw_max), "raw_min": finite_summary(raw_min),
        "actual_max": finite_summary(actual_max), "actual_min": finite_summary(actual_min),
        "own_standardized_min": finite_summary(intended_min),
        "actual_vs_own_standardized_max_abs_difference": float((actual_min-intended_min).abs().max()),
        "actual_min_translation_shift_max_abs": float((shifted_actual-actual_min).abs().max()),
        "temporal_correlation_actual_min_to_actual_max": float(torch.corrcoef(torch.stack([actual_min.flatten(), actual_max.flatten()]))[0, 1]),
        "interpretation": "min branch temporal shape is affine to standardized max branch; raw min contributes its mean and std, not its per-time shape. This demonstrates formula mismatch, not an empirical forecast benefit from repair.",
    }

    stds = torch.stack([distances(192, p).std(1) for p in range(192)])
    tests["all_valid_extremum_positions_L192_have_positive_std"] = bool((stds > 0).all())
    all_missing_norm = normalize(norms, distances(192, 0), distances(192, 0))
    tests["all_missing_L192_distance_normalization_is_finite"] = all(bool(torch.isfinite(x).all()) for x in all_missing_norm)
    zero_norm = normalize(norms, torch.ones(1, 5, 1), torch.ones(1, 5, 1))
    tests["external_constant_distances_produce_nonfinite"] = any(not bool(torch.isfinite(x).all()) for x in zero_norm)
    one_norm = normalize(norms, distances(1, 0), distances(1, 0))
    tests["L1_distance_normalization_produces_nonfinite"] = any(not bool(torch.isfinite(x).all()) for x in one_norm)

    # Execute original Dataset_Custom4.__getitem__ only on a constructed CPU
    # object. __read_data__ is never invoked; no train, validation or test data.
    _, getitem = method(paths["baseline_loader"], "Dataset_Custom4", "__getitem__")
    _, original_getitem = method(paths["upstream_loader"], "Dataset_Custom4", "__getitem__")
    tests["dataset_getitem_ast_matches_upstream"] = ast_equal([getitem], [original_getitem])
    namespace = execute([getitem], {"np": np})
    synthetic = types.SimpleNamespace(
        seq_len=4, pred_len=1, args=types.SimpleNamespace(mean_type="global_mean", model="S4M"),
        data_x=np.array([[1., -1.], [0., 0.], [2., -2.], [3., -3.], [4., -4.]]),
        mask_data=np.array([[1., 1.], [0., 0.], [1., 1.], [1., 1.], [1., 1.]]),
    )
    for name in ("global_mean", "em_mean", "true_data", "data_y", "data_f"):
        setattr(synthetic, name, synthetic.data_x.copy())
    sample = namespace["__getitem__"](synthetic, 0)
    maximum_values, minimum_values = sample[8], sample[9]
    tests["missing_zero_can_be_minimum_of_positive_observations"] = float(minimum_values[0, 0]) == 0
    tests["missing_zero_can_be_maximum_of_negative_observations"] = float(maximum_values[0, 1]) == 0
    tests["missing_zero_extremum_distance_points_at_missing_index"] = bool(sample[7][1, 0] == 0 and sample[6][1, 1] == 0)

    # The source reshape retains shape while changing the meaning of the axes.
    _, encoder = method(paths["baseline_bank"], "Encoder2", "forward")
    _, original_encoder = method(paths["upstream_bank"], "Encoder2", "forward")
    reshape = assignments(encoder, {"input"})
    original_reshape = assignments(original_encoder, {"input"})
    tests["encoder_conv_reshape_ast_matches_upstream"] = ast_equal(reshape, original_reshape)
    values = torch.arange(6, dtype=torch.float64).reshape(1, 2, 3, 1)
    self_stub = types.SimpleNamespace(Tc=3, config=types.SimpleNamespace(en_conv_hidden_size=2))
    reshaped = execute(reshape, {"h_conv1": values, "self": self_stub})["input"]
    permuted = values.squeeze(-1).transpose(1, 2).contiguous()
    tests["reshape_preserves_shape_but_not_time_channel_axis_mapping"] = reshaped.shape == permuted.shape and not torch.equal(reshaped, permuted)

    # Actual normalization for L192 is safe against zero-std. The following is
    # a separate IEEE float32 stress demonstration, not a current-run failure.
    float32_logits = torch.tensor([-100., 1000.], dtype=torch.float32)
    raw_gamma = torch.exp(-float32_logits)
    ratios = raw_gamma / (raw_gamma + raw_gamma)
    stable_ratios = torch.sigmoid(float32_logits - float32_logits)
    tests["float32_extreme_exp_ratios_can_be_nonfinite"] = bool((~torch.isfinite(ratios)).all())
    tests["mathematically_equivalent_sigmoid_ratios_stay_finite"] = bool(torch.isfinite(stable_ratios).all())

    tests["all_audited_source_files_unchanged"] = source_before == {key: sha(path) for key, path in paths.items()}
    frozen_config = ROOT / "configs/stage22_38_diagnostic/seed2027_formal.yaml"
    config_text = frozen_config.read_text()
    dimensions = {}
    for name in ("short_len", "W", "en_conv_hidden_size", "seq_len"):
        matches = re.findall(r"^  " + name + r": ([0-9]+)$", config_text, flags=re.MULTILINE)
        assert len(matches) == 1, (name, matches)
        dimensions[name] = int(matches[0])
    dimensions["Tc"] = dimensions["short_len"] - dimensions["W"] + 1
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE, text=True, capture_output=True, check=False)
    findings = [
        {"id": "distance_min_wrong_source", "status": "CONFIRMED_UPSTREAM_IMPLEMENTATION_MISMATCH", "priority": 1,
         "description_zh": "min_idx 分子误用已标准化 max_idx，再减原始 min 距离均值，混合标准化尺度与原始步数尺度；不是真正的 min 距离标准化。",
         "evidence": [evidence(paths["baseline_model"], norms), evidence(paths["upstream_model"], original_norms), evidence(paths["baseline_model"], decay_norms)],
         "causal_boundary_zh": "数学/源码错误已证实，但当前 checkpoint 已适应该错误；直接修正后的好坏需 validation 诊断，不能声称必然提升，也不计作创新。"},
        {"id": "extrema_include_imputed_zero", "status": "CONFIRMED_UPSTREAM_MASK_SEMANTICS_RISK", "priority": 2,
         "description_zh": "Dataset_Custom4 在归一化并将缺失置 0 后求窗口极值，未排除缺失位置；合成正/负观测例证实缺失 0 可成为最小/最大值。",
         "evidence": [evidence(paths["baseline_loader"], [getitem]), evidence(paths["upstream_loader"], [original_getitem])],
         "causal_boundary_zh": "不访问真实数据，因此未量化当前 Traffic 上比例或性能伤害；全窗历史极值可使用历史内较晚点，不等于使用预测未来标签。"},
        {"id": "encoder_axis_reshape", "status": "CONFIRMED_UPSTREAM_AXIS_MAPPING_MISMATCH", "priority": 3,
         "description_zh": "Conv2d 输出 [N,H,Tc,1] 直接 reshape 为 [N,Tc,H] 并非交换轴；时间/通道含义发生重新打包。",
         "evidence": [evidence(paths["baseline_bank"], reshape), evidence(paths["upstream_bank"], original_reshape)],
         "causal_boundary_zh": "不一定造成报错，也不能证明损害当前已训练模型。应单列 bug-fix baseline，不能把修正后重训收益混同新模块创新。"},
        {"id": "normalization_zero_std", "status": "LATENT_NOT_L192_DATASET_FAILURE", "priority": 4,
         "description_zh": "公式无 epsilon，常数外部距离与 L=1 会非有限；但当前 Dataset_Custom4 L=192 的 |t-k| 对全部合法 k 标准差均为正，完全缺失历史也不自动零 std。",
         "causal_boundary_zh": "不应将一般数值风险误报为当前实验故障。"},
        {"id": "exponential_ratio_numeric_range", "status": "LATENT_STRESS_CASE_NOT_OBSERVED_FAILURE", "priority": 5,
         "description_zh": "exp(-logit) 比值在极端 float32 范围可出现 inf/inf 或 0/0；实际 logits 尚未采样，不能声称当前发生溢出。",
         "causal_boundary_zh": "稳定 sigmoid 为代数等价建议，不在本步骤修改基线。"},
    ]
    result = {
        "stage": "22.39", "kind": "CPU_SOURCE_AND_SYNTHETIC_AUDIT",
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "passed": all(tests.values()), "tests": tests, "test_count": len(tests),
        "environment": {"python": os.sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "device": "CPU"},
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "source_hashes": source_before, "script_sha256": sha(__file__),
        "scope": {"real_dataset_opened": False, "test_split_opened": False, "checkpoint_loaded": False, "baseline_modified": False, "training_performed": False},
        "findings": findings,
        "normalization_example": min_synthetic,
        "std_audit": {"L192_min_std_over_extremum_positions": float(stds.min()), "L192_max_std": float(stds.max()), "constant_distance_nonfinite": [finite_summary(x) for x in zero_norm], "L1_nonfinite": [finite_summary(x) for x in one_norm]},
        "extrema_example": {"history": synthetic.data_x[:4].tolist(), "mask": synthetic.mask_data[:4].tolist(), "max_value": maximum_values[0].tolist(), "min_value": minimum_values[0].tolist(), "expected_observed_only_max_value": [3., -1.], "expected_observed_only_min_value": [1., -3.]},
        "encoder_mapping_example": {"input_NHT1": values.tolist(), "source_reshape_NTH": reshaped.tolist(), "axis_permutation_NTH": permuted.tolist(), "active_Traffic_dimensions": dimensions, "source_config": str(frozen_config.relative_to(ROOT)), "source_config_sha256": sha(frozen_config), "config_note": "Active dimensions read from frozen Step 22.38 seed2027 formal YAML; not model changes."},
        "limitations_zh": ["所有数值均为 CPU 合成单元算例，不是预测性能指标。", "不读取 test，不加载 checkpoint，不更改原始源码。", "问题源于上游已有实现；修复必须保留原 baseline 并独立报告，不能计为学术创新。", "当前 checkpoint 是否因修正变好，只能由冻结 validation 诊断和后续公平重训验证。"],
    }
    return result


def main():
    result = audit()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, REPORT)
    lines = ["# 步骤22.39：距离填补与编码轴 CPU 审计", "", f"状态：{'通过' if result['passed'] else '失败'}；{result['test_count']} 个断言；CPU 合成算例，无真实数据、checkpoint 或 test 读取。", "", "## 确认与边界", ""]
    for item in result["findings"]:
        lines += [f"{item['priority']}. {item['description_zh']}", f"   分类：`{item['status']}`。{item['causal_boundary_zh']}", ""]
        for entry in item.get("evidence", []):
            for source in entry["lines"]:
                lines.append(f"   证据：`{entry['path']}:{source['line']}`；SHA256 `{entry['sha256']}`。")
        lines.append("")
    example = result["normalization_example"]
    lines += ["## 合成单元结果", "", f"L=192 合成距离：实际 min 分支均值 {example['actual_min']['mean_finite']:.6f}，正确自身标准化均值 {example['own_standardized_min']['mean_finite']:.6f}；与 max 分支时间相关系数 {example['temporal_correlation_actual_min_to_actual_max']:.6f}。", f"全部 192 个合法极值位置的最小距离标准差：{result['std_audit']['L192_min_std_over_extremum_positions']:.6f}，故该协议下零 std 没有被支持。", "", "这些不是 MSE/MAE（均方误差/平均绝对误差）实验；不能据此宣称性能提升。", "", "## 可重复命令", "", "```sh", "/mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python scripts/stage22_39_distance_audit.py", "```", "", f"完整断言与源代码哈希：`{REPORT.relative_to(ROOT)}`。", ""]
    md = REPORT.with_suffix(".md")
    temporary = md.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines))
    os.replace(temporary, md)
    print(json.dumps({"passed": result["passed"], "tests": result["test_count"], "report": str(REPORT), "source_unchanged": result["tests"]["all_audited_source_files_unchanged"]}, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
