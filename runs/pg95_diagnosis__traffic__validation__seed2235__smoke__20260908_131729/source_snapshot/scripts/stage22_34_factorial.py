#!/usr/bin/env python3
"""Prepare, execute and incrementally audit the frozen validation-only 2x2 control."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

from run_stage14_uncertainty_cache import validate_config, run_id

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/stage22_34_factorial/state.json"
CONFIG_ROOT = ROOT / "configs/stage22_34_factorial"
REPORT_STEM = 'stage22_34_factorial'
CRN_RECOVERY = False
PG95_RUNS = {
    2026: "runs/cnm_se_equal_rate_multiscale_pg95__traffic__validation__seed2026__20260816_192252",
    2027: "runs/cnm_se_equal_rate_multiscale_pg95__traffic__validation__seed2027__20260820_184433",
    2028: "runs/cnm_se_equal_rate_multiscale_pg95__traffic__validation__seed2028__20260820_194649",
}


def read(path):
    return json.loads(Path(path).read_text())


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_run(seed):
    stage = "22_21" if seed == 2026 else "22_22"
    state_path = ROOT / f"runtime/stage{stage}_traffic_pipeline/state.json"
    job = read(state_path)["jobs"][f"traffic_s4m_seed{seed}_formal"]
    if job["state"] != "SUCCESS":
        raise ValueError("DATA_ERROR: source training is not successful")
    return Path(job["run_dir"])


def prepare():
    manifest = []
    for seed, smoke in ((2027, True), (2027, False), (2026, False), (2028, False)):
        template_path = ROOT / f"configs/stage22_22_traffic/s4m_traffic_multiscale_test_seed{seed}.yaml"
        cfg = yaml.safe_load(template_path.read_text())
        source = source_run(seed)
        source_config = yaml.safe_load((source / "resolved_config.yaml").read_text())
        metrics = read(source / "metrics.json")
        if not (metrics.get("test_split_accessed") is False and metrics.get("evaluation_split") == "validation"
                and metrics.get("chronological_loader") is True):
            raise ValueError("DATA_ERROR: source must be validation-only")
        cfg["experiment"].update(model="s4m_cnm_se_factorial", domain="validation_point_r006", smoke_test=smoke,
            nominal_missing_rate=.06, realized_missing_rate=source_config["experiment"]["realized_missing_rate"],
            protocol_scope="stage22_34_fixed_validation_factorial" + ("_smoke" if smoke else ""),
            protocol_match="same_fixed_validation_assets_with_pg95_cache",
            purpose="missing_s4m_plus_multiscale_validation_control_no_training_no_test")
        cfg["source"] = dict(run_dir=str(source), resolved_config=str(source / "resolved_config.yaml"),
            metrics=str(source / "metrics.json"), checkpoint=metrics["checkpoint"])
        cfg["diagnostic"].update(variant="stage22_34_traffic_s4m_validation", split="val",
            max_eval_batches=3 if smoke else -1, timing_warmup_batches=1 if smoke else 10)
        cfg["provenance"] = dict(template=str(template_path), stage="22.34", uses_test_for_selection=False,
            source_config_sha256=digest(source / "resolved_config.yaml"),
            source_checkpoint_sha256=digest(metrics["checkpoint"]),
            configuration_changes={"split": {"old": "test", "new": "val", "reason": "missing validation control"},
                "variant": {"old": "stage22_22_traffic_s4m_test", "new": cfg["diagnostic"]["variant"],
                            "reason": "explicit validation-only S4M adapter"},
                "max_eval_batches": {"old": -1, "new": 3 if smoke else -1, "reason": "isolated smoke only"}})
        validate_config(cfg, source_config)
        path = CONFIG_ROOT / f"s4m_traffic_validation_seed{seed}{'_smoke' if smoke else ''}.yaml"
        content = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        if path.exists() and path.read_text() != content:
            raise ValueError(f"CONFIG_ERROR: refusing to overwrite existing frozen config {path}")
        atomic_text(path, content)
        manifest.append(dict(config=str(path), sha256=digest(path)))
    write_json(ROOT / "runtime/stage22_34_factorial/config_manifest.json", manifest)
    print(f"prepared {len(manifest)} frozen configurations")


def execute(config_path, gpu_uuid, attempt):
    path = (ROOT / config_path).resolve()
    manifest = read(STATE.parent / "config_manifest.json")
    matching = [row for row in manifest if row["config"] == str(path)]
    if len(matching) != 1 or matching[0]["sha256"] != digest(path):
        raise ValueError("CONFIG_ERROR: configuration is not the frozen prepared asset")
    cfg = yaml.safe_load(path.read_text())
    source = yaml.safe_load(Path(cfg["source"]["resolved_config"]).read_text())
    validate_config(cfg, source)
    allowed_methods = ('s4m_baseline', 'pg95') if CRN_RECOVERY else ('s4m_baseline',)
    if cfg["diagnostic"]["split"] != "val" or cfg["experiment"]["method"] not in allowed_methods:
        raise ValueError("CONFIG_ERROR: this stage only supports S4M validation")
    if CRN_RECOVERY and (cfg['diagnostic'].get('fixed_eval_rng') is not True
                         or cfg['diagnostic'].get('rng_protocol') != 'stage22_34_crn_v1'):
        raise ValueError('CONFIG_ERROR: recovery requires frozen CRN protocol')
    if digest(cfg["source"]["checkpoint"]) != cfg["provenance"]["source_checkpoint_sha256"]:
        raise ValueError("DATA_ERROR: source checkpoint hash changed")
    if digest(cfg["source"]["resolved_config"]) != cfg["provenance"]["source_config_sha256"]:
        raise ValueError("DATA_ERROR: source effective configuration changed")
    files = ["scripts/stage22_34_factorial.py", "scripts/run_stage22_34_factorial_queue.py",
        "scripts/test_stage22_34_factorial.py", "scripts/start_stage22_34_factorial.sh",
        "scripts/run_stage14_uncertainty_cache.py", "scripts/run_stage5_baseline_queue.py",
        "scripts/run_s4m_experiment.py", "docs/STAGE22_34_FACTORIAL_AUDIT_PROTOCOL_ZH.md",
        "baselines/S4M/s4m/utils/paired_eval_rng.py"]
    if CRN_RECOVERY:
        files += ['scripts/stage22_34_crn.py', 'scripts/run_stage22_34_crn_queue.py',
                  'scripts/start_stage22_34_crn.sh', 'scripts/test_stage22_34_crn.py',
                  'docs/STAGE22_34_CRN_RECOVERY_ZH.md']
    snapshot = {name: (ROOT / name).read_text() for name in files}
    command = [sys.executable, str(ROOT / "scripts/run_stage14_uncertainty_cache.py"),
               "--config", str(path), "--gpu-uuid", gpu_uuid, "--scheduler-attempt", str(attempt)]
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith("run_dir="):
                run = Path(line.strip().split("=", 1)[1])
                hashes = {}
                for relative, content in snapshot.items():
                    target = run / "source_snapshot" / relative
                    atomic_text(target, content)
                    hashes[relative] = digest(target)
                write_json(run / "root_code_manifest.json", hashes)
                atomic_text(run / "launch_config.yaml", path.read_text())
        return process.wait()


def gain(a, b):
    return (a - b) / a * 100.


def pair(metrics, route):
    return np.array([metrics[f"complete_target_{metric}_{route}"] for metric in ("mse", "mae")])


def compare_seed(seed, s4m_run, pg95_run):
    configs, metrics = [], []
    for path in (s4m_run, pg95_run):
        state = read(path / "status.json")
        m = read(path / "metrics.json")
        cfg = yaml.safe_load((path / "resolved_config.yaml").read_text())
        assert state["state"] == "SUCCESS" and state["return_code"] == 0
        assert m["source_split"] == "val" and m["uses_test_split"] is False
        assert m["updates_model_or_memory"] is False and m["sample_count"] == 1648
        assert m["artifact_status"] == "PASS" and all(m["artifact_checks"].values())
        assert m["multiscale_weights"] == {"original": .5, "l5": .25, "l20": .25}
        assert cfg["experiment"]["seed"] == seed and cfg["official_args"]["seed"] == seed
        if CRN_RECOVERY:
            assert m.get('evaluation_rng_protocol') == 'stage22_34_crn_v1'
            assert cfg['diagnostic'].get('fixed_eval_rng') is True
        configs.append(cfg)
        metrics.append(m)
    assert not configs[0]["official_args"].get("use_mipp", False)
    assert configs[1]["official_args"]["use_mipp"] is True
    for key in ("batch_size", "seq_len", "pred_len", "enc_in", "e_layers", "d_model", "fillna", "mean_type"):
        assert configs[0]["official_args"][key] == configs[1]["official_args"][key], key
    assert metrics[0]["mask_asset_sha256"] == metrics[1]["mask_asset_sha256"]
    for cfg, m in zip(configs, metrics):
        assert digest(cfg["data"]["mask_asset"]) == m["mask_asset_sha256"]
        assert digest(cfg["source"]["checkpoint"]) == m["source_checkpoint_sha256"]
    with np.load(s4m_run / "validation_equal_rate_multiscale_samples.npz", allow_pickle=False) as a, np.load(
            pg95_run / "validation_equal_rate_multiscale_samples.npz", allow_pickle=False) as b:
        for key in ("sample_index", "complete_count", "history_mask_packed", "l1_mask_packed", "l5_mask_packed",
                    "l20_mask_packed", "target_removed_count", "removed_count_l1", "removed_count_l5", "removed_count_l20"):
            assert np.array_equal(a[key], b[key]), f"unpaired {key} seed{seed}"
        assert np.array_equal(a["sample_index"], np.arange(1648))
        for arrays, m in ((a, metrics[0]), (b, metrics[1])):
            for route in ("on", "multiscale"):
                recomputed = np.array([arrays[f"complete_sse_{route}"].sum(),
                                       arrays[f"complete_sae_{route}"].sum()]) / arrays["complete_count"].sum()
                np.testing.assert_allclose(recomputed, pair(m, route), rtol=0, atol=1e-12)
    A, B = pair(metrics[0], "on"), pair(metrics[1], "on")
    C, D = pair(metrics[0], "multiscale"), pair(metrics[1], "multiscale")
    return dict(seed=seed, sources=dict(s4m=str(s4m_run), pg95=str(pg95_run)),
        cells={name: {"mse": float(x[0]), "mae": float(x[1])} for name, x in zip(("A_s4m", "B_pg95", "C_s4m_cnm", "D_pg95_cnm"), (A, B, C, D))},
        pg95_gain_without_cnm_percent=gain(A, B).tolist(), cnm_gain_on_s4m_percent=gain(A, C).tolist(),
        cnm_gain_on_pg95_percent=gain(B, D).tolist(), pg95_gain_with_cnm_percent=gain(C, D).tolist(),
        interaction_absolute=((B - D) - (A - C)).tolist(),
        interaction_percent_of_s4m=(((B - D) - (A - C)) / A * 100).tolist(),
        fixed_assets_check="PASS", new_test_inference=False)


def analyze():
    state = read(STATE) if STATE.is_file() else {"jobs": {}, "queue_state": "NOT_STARTED"}
    per_seed, missing, errors = [], [], []
    for seed, relative in PG95_RUNS.items():
        job = state["jobs"].get(f"traffic_s4m_validation_seed{seed}_formal", {})
        pg_job = state['jobs'].get(f'traffic_pg95_validation_seed{seed}_formal', {})
        if job.get("state") != "SUCCESS" or (CRN_RECOVERY and pg_job.get('state') != 'SUCCESS'):
            missing.append(seed)
            continue
        try:
            pg_run = Path(pg_job['run_dir']) if CRN_RECOVERY else ROOT / relative
            per_seed.append(compare_seed(seed, Path(job["run_dir"]), pg_run))
        except Exception as error:
            errors.append(dict(seed=seed, type=type(error).__name__, error=str(error)))
    decision = "WAITING_FOR_FIXED_VALIDATION_CONTROLS"
    summary = {}
    if errors:
        decision = "FACTORIAL_ARTIFACT_ERROR"
    elif state.get("queue_state") == "FAILED":
        decision = "FACTORIAL_EXECUTION_STOP"
    elif len(per_seed) == 3:
        marginal = np.array([r["pg95_gain_with_cnm_percent"] for r in per_seed])
        wins = int((marginal > 0).all(1).sum())
        decision = "PG95_INCREMENT_CONSISTENT_DEVELOPMENT_ONLY" if wins == 3 else "PG95_ROLE_NOT_CONSISTENT"
        summary = dict(pg95_with_cnm_dual_winning_seeds=wins,
            mean_pg95_gain_with_cnm_percent=marginal.mean(0).tolist(),
            sample_std_pg95_gain_with_cnm_percent=marginal.std(0, ddof=1).tolist(),
            mean_interaction_absolute=np.mean([r["interaction_absolute"] for r in per_seed], axis=0).tolist(),
            mean_cnm_gain_on_s4m_percent=np.mean([r["cnm_gain_on_s4m_percent"] for r in per_seed], axis=0).tolist())
    output = dict(stage="22.34", decision=decision, queue_state=state.get("queue_state"),
        evaluation_rng_protocol='stage22_34_crn_v1' if CRN_RECOVERY else 'historical_uncontrolled',
        timing_is_isolated_benchmark=False,
        queue_records=[{k: j.get(k) for k in ("id", "state", "run_dir", "required_memory_estimate_mib", "failure_summary")} for j in state["jobs"].values()],
        missing_seeds=missing, artifact_errors=errors, per_seed=per_seed, summary=summary,
        source_reports=["stage7_mipp_comparison", "stage16_cnm_se_three_seed_analysis", "stage18_equal_rate_multiscale_analysis",
                        "stage20_etth1_pg95_full_method", "stage21_electricity_validation_gate", "stage22_22_traffic_multiseed_results"],
        scope="already_developed_validation_factorial_no_new_test", uses_new_training=False,
        limitations=["different historical CNM versions must not be pooled", "negative ETTh1/Electricity results retained",
                     "same 3 timestamps/mask-training seeds are not independent holdout datasets",
                     "no checkpoint or hyperparameter selection is authorized by this audit"])
    lines = ["# Step 22.34 主线验证集四格对照", "", f"决策：`{decision}`；尚缺seed：{missing}。", "",
             "同一Traffic validation/固定掩码；正收益表示误差下降。无新增训练，无test重评。", "",
             "| seed | 方法 | MSE | MAE |", "| --- | --- | ---: | ---: |"]
    rows = []
    for record in per_seed:
        for method, pair_values in record["cells"].items():
            lines.append(f"| {record['seed']} | {method} | {pair_values['mse']:.9f} | {pair_values['mae']:.9f} |")
            rows.append(dict(seed=record["seed"], method=method, **pair_values, scope=output["scope"]))
    for record in per_seed:
        lines += ["", f"seed{record['seed']}：集成下PG95的MSE/MAE增量={record['pg95_gain_with_cnm_percent']}%；"
                  f"绝对交互量={record['interaction_absolute']}。", ""]
    lines += ["", "三seed摘要：", "", json.dumps(summary, ensure_ascii=False), "", "## 边界", "",
        "Weather早期三seed正结果是L5双视图，不能与后来的等计数多尺度混称同一版本。ETTh1/Electricity完整方法负结果保留。",
        "", "这一步不选择新的超参数、主模型或test；交互量只是配对开发指标，不能当作独立显著性、因果协同或二区水平证明。", ""]
    if errors:
        lines += ["工件错误：" + json.dumps(errors, ensure_ascii=False)]
    content = "\n".join(lines)
    if CRN_RECOVERY:
        content += '\n本报告为获批CRN-v1新配对协议，不与旧随机轨迹混比；并发运行的时间仅为运行记录，不是隔离GPU速度结论。\n'
    write_json(ROOT / f"reports/{REPORT_STEM}.json", output)
    atomic_text(ROOT / f"reports/{REPORT_STEM}.md", content)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["seed", "method", "mse", "mae", "scope"])
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(ROOT / f"reports/{REPORT_STEM}.csv", stream.getvalue())
    if len(per_seed) == 3 and not errors:
        archive = ROOT / "reports" / (REPORT_STEM + '_' + time.strftime("%Y%m%d_%H%M%S"))
        write_json(archive.with_suffix(".json"), output)
        atomic_text(archive.with_suffix(".md"), content)
        atomic_text(archive.with_suffix(".csv"), stream.getvalue())
    subprocess.run([sys.executable, str(ROOT / "scripts/aggregate_results.py")], cwd=ROOT, check=False)
    print(json.dumps(dict(decision=decision, missing_seeds=missing, artifact_errors=errors)))
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--scheduler-attempt", type=int, default=0)
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return 0
    if args.config:
        return execute(args.config, args.gpu_uuid, args.scheduler_attempt)
    return analyze()


if __name__ == "__main__":
    raise SystemExit(main())
