#!/usr/bin/env python3
"""Frozen, CPU-only PSAVS screening. No forecasting model or test data is loaded."""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import yaml
from sklearn.tree import DecisionTreeRegressor, export_text

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage22_33_psavs.yaml"
ACTIONS = ("on", "ensemble_l5", "ensemble_l20", "multiscale")
SOURCE_RUNS = {
    2026: "runs/cnm_se_equal_rate_multiscale_pg95__traffic__validation__seed2026__20260816_192252",
    2027: "runs/cnm_se_equal_rate_multiscale_pg95__traffic__validation__seed2027__20260820_184433",
    2028: "runs/cnm_se_equal_rate_multiscale_pg95__traffic__validation__seed2028__20260820_194649",
}
MASK_NAMES = ["missing_fraction", "transition_rate", "longest_missing_run",
              "trailing_missing_run", "missing_run_count", "missing_centroid",
              "last32_missing_fraction", "last96_missing_fraction"] + [
                  f"block{i}_missing_fraction" for i in range(6)]
FEATURE_NAMES = {
    "mask_only": MASK_NAMES,
    "disagreement_aware": MASK_NAMES + ["log1p_d5", "log1p_d20", "log_d5_over_d20"],
}


def now():
    return dt.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(cfg):
    require(tuple(cfg["actions"]) == ACTIONS, "CONFIG_ERROR: action order changed")
    require(cfg["source_runs"] == SOURCE_RUNS, "CONFIG_ERROR: validation source allowlist changed")
    require(cfg["samples_per_seed"] == 1648 and cfg["history_length"] == 192
            and cfg["prediction_length"] == 96, "CONFIG_ERROR: frozen dimensions changed")
    require(cfg["splits"] == {"fit": [0, 576], "development": [863, 1055],
                              "audit": [1342, 1648]}, "CONFIG_ERROR: frozen splits changed")
    require(cfg["tree"] == dict(criterion="squared_error", max_depth=3,
            min_samples_leaf=64, random_state=2233, splitter="best"),
            "CONFIG_ERROR: frozen tree changed")
    require(cfg["margin_candidates"] == [0.0, .001, .002, .005, .01, None],
            "CONFIG_ERROR: frozen margins changed")
    require(cfg["policies"] == ["mask_only", "disagreement_aware"]
            and cfg["primary_policy"] == "disagreement_aware", "CONFIG_ERROR: policy changed")
    require(cfg["gate"] == dict(minimum_dual_winning_seeds=2, minimum_mean_gain_percent=.15),
            "CONFIG_ERROR: frozen gate changed")
    require(cfg["bootstrap"] == dict(replicates=2000, block_length=96, seed=2233),
            "CONFIG_ERROR: frozen bootstrap changed")
    require(cfg["runtime"] == dict(device="cpu", threads=1), "CONFIG_ERROR: CPU-only stage")
    require(cfg["experiment"]["source_seeds"] == [2026, 2027, 2028], "CONFIG_ERROR: seeds")
    previous_stop = None
    for name in ("fit", "development", "audit"):
        start, stop = cfg["splits"][name]
        require(0 <= start < stop <= cfg["samples_per_seed"], "CONFIG_ERROR: invalid split")
        if previous_stop is not None:
            require(start - previous_stop >= 287, "CONFIG_ERROR: overlapping time support")
        previous_stop = stop


def mask_features(observed):
    """Only inference-time history mask; never receives targets, seeds, or indices."""
    observed = np.asarray(observed)
    require(observed.ndim == 2 and observed.shape[1] == 192, "DATA_ERROR: history mask shape")
    require(np.isin(observed, [0, 1]).all(), "DATA_ERROR: nonbinary history mask")
    missing = ~observed.astype(bool)
    n, length = missing.shape
    longest, trailing, runs = np.zeros(n), np.zeros(n), np.zeros(n)
    previous = np.zeros(n, dtype=bool)
    for col in missing.T:
        runs += col & ~previous
        trailing = np.where(col, trailing + 1, 0)
        longest = np.maximum(longest, trailing)
        previous = col
    centroid = (missing * np.linspace(0, 1, length)).sum(1) / np.maximum(missing.sum(1), 1)
    return np.column_stack([
        missing.mean(1), (missing[:, 1:] != missing[:, :-1]).mean(1),
        longest / length, trailing / length, runs / length, centroid,
        missing[:, -32:].mean(1), missing[:, -96:].mean(1),
        *[missing[:, i * 32:(i + 1) * 32].mean(1) for i in range(6)],
    ])


def make_features(observed, d5, d20):
    mask = mask_features(observed)
    d5, d20 = np.asarray(d5), np.asarray(d20)
    require(d5.shape == d20.shape == (len(mask),), "DATA_ERROR: disagreement shape")
    require(np.isfinite(d5).all() and np.isfinite(d20).all()
            and (d5 >= 0).all() and (d20 >= 0).all(), "DATA_ERROR: invalid disagreement")
    return {"mask_only": mask, "disagreement_aware": np.column_stack([
        mask, np.log1p(d5), np.log1p(d20), np.log((d5 + 1e-12) / (d20 + 1e-12))])}


def select_actions(predicted_gains, margin, enabled=True):
    gains = np.asarray(predicted_gains)
    require(gains.ndim == 3 and gains.shape[1:] == (3, 2)
            and np.isfinite(gains).all(), "CODE_ERROR: gain shape or finiteness")
    if not enabled or margin is None:
        return np.full(len(gains), 3, dtype=np.int64)
    score = gains.min(axis=2)
    choice = score.argmax(axis=1)  # fixed action order resolves ties
    return np.where(score[np.arange(len(score)), choice] > margin, choice, 3)


def selected_losses(losses, choices):
    return losses[np.arange(len(losses)), choices]


def relative_gain(reference, candidate):
    return (np.asarray(reference) - candidate) / reference * 100


def fit_selector(features, losses, tree_config):
    """Arguments contain FIT data only. Normalization is never fit on dev/audit."""
    require(losses.shape == (len(features), 4, 2) and np.isfinite(losses).all(),
            "DATA_ERROR: fit loss shape")
    scale = losses[:, 3].mean(axis=0)
    require((scale > 0).all(), "DATA_ERROR: zero normalization")
    target = ((losses[:, 3:4] - losses[:, :3]) / scale).reshape(len(losses), 6)
    model = DecisionTreeRegressor(**tree_config)
    model.fit(features, target)
    return model, scale


def calibrate(dev_gains, dev_losses, margins):
    """Only DEV labels are passed here; audit is not an input."""
    records, eligible = [], []
    for margin in margins:
        gains = []
        for pred, losses in zip(dev_gains, dev_losses):
            choices = select_actions(pred, margin)
            gains.append(relative_gain(losses[:, 3].mean(0), selected_losses(losses, choices).mean(0)))
        gains = np.asarray(gains)
        mean = gains.mean(0)
        wins = int((gains > 0).all(1).sum())
        record = dict(margin=margin, per_seed_gain_percent=gains.tolist(),
                      mean_gain_percent=mean.tolist(), dual_winning_seeds=wins,
                      eligible=bool(wins >= 2 and (mean > 0).all()))
        records.append(record)
        if record["eligible"]:
            eligible.append((float(mean.min()), float("inf") if margin is None else margin, margin))
    chosen = max(eligible)[2] if eligible else None
    return chosen, records


def safe_oracle(losses):
    """Label-aware scalar oracle, NOT a strict individual-metric upper bound."""
    original = losses[:, :1]
    safe = (losses <= original).all(2)
    score = (losses / np.maximum(original, np.finfo(float).tiny)).mean(2)
    return np.where(safe, score, np.inf).argmin(1)


def load_source(seed, relative_path):
    require(SOURCE_RUNS.get(seed) == relative_path, "DATA_ERROR: non-allowlisted source")
    path = (ROOT / relative_path).resolve()
    files = {name: path / name for name in (
        "metrics.json", "status.json", "resolved_config.yaml", "validation_equal_rate_multiscale_samples.npz")}
    for file in files.values():
        require(file.is_file(), f"DATA_ERROR: missing {file}")
    hashes = {str(file): sha256(file) for file in files.values()}
    metrics = json.loads(files["metrics.json"].read_text())
    status = json.loads(files["status.json"].read_text())
    cfg = yaml.safe_load(files["resolved_config.yaml"].read_text())
    require(status.get("state") == "SUCCESS" and status.get("return_code") == 0,
            "DATA_ERROR: source did not succeed")
    require(metrics.get("artifact_status") == "PASS" and metrics.get("artifact_checks")
            and all(metrics["artifact_checks"].values()), "DATA_ERROR: source artifact gates")
    require(metrics.get("dataset") == "traffic" and metrics.get("method") == "cnm_se_equal_rate_multiscale_pg95"
            and metrics.get("source_split") == "val" and metrics.get("uses_test_split") is False
            and metrics.get("updates_model_or_memory") is False, "DATA_ERROR: not frozen validation")
    require(metrics.get("sample_count") == 1648 and metrics.get("block_lengths") == [1, 5, 20]
            and metrics.get("target_additional_missing_rate") == .15
            and metrics.get("target_count_rounding") == "numpy_rint_ties_to_even"
            and metrics.get("multiscale_weights") == {"original": .5, "l5": .25, "l20": .25},
            "DATA_ERROR: source protocol mismatch")
    require(cfg["experiment"]["seed"] == seed and cfg["official_args"]["seq_len"] == 192
            and cfg["official_args"]["pred_len"] == 96 and cfg["official_args"]["batch_size"] == 16,
            "DATA_ERROR: source effective configuration mismatch")
    mask = ROOT / f"artifacts/masks/traffic/point/rate_0.06/seed_{seed}.npz"
    require(Path(cfg["data"]["mask_asset"]).resolve() == mask.resolve(), "DATA_ERROR: mask path")
    # Hash the fixed asset, but do not unpack ANY split of it (especially test).
    hashes[str(mask)] = sha256(mask)
    require(hashes[str(mask)] == cfg["data"]["mask_asset_sha256"] == metrics["mask_asset_sha256"],
            "DATA_ERROR: mask identity mismatch")
    with np.load(files["validation_equal_rate_multiscale_samples.npz"], allow_pickle=False) as z:
        require(np.array_equal(z["sample_index"], np.arange(1648)), "DATA_ERROR: sample order")
        count = z["complete_count"]
        require(count.shape == (1648,) and np.all(count == 96 * 862), "DATA_ERROR: target counts")
        history = np.unpackbits(z["history_mask_packed"], axis=1, count=192).astype(bool)
        require(history.shape == (1648, 192), "DATA_ERROR: packed mask shape")
        removed = np.rint(.15 * history.sum(1)).astype(np.int64)
        require(np.array_equal(z["target_removed_count"], removed), "DATA_ERROR: target mask counts")
        for block in (1, 5, 20):
            nested = np.unpackbits(z[f"l{block}_mask_packed"], axis=1, count=192).astype(bool)
            require(not (nested & ~history).any(), "DATA_ERROR: nested mask is not subset")
            require(np.array_equal((history & ~nested).sum(1), removed)
                    and np.array_equal(z[f"removed_count_l{block}"], removed),
                    "DATA_ERROR: unequal actual removed counts")
        features = make_features(history, z["prediction_difference_mse_l5"], z["prediction_difference_mse_l20"])
        losses = np.stack([np.column_stack([z[f"complete_sse_{a}"] / count,
                                           z[f"complete_sae_{a}"] / count]) for a in ACTIONS], axis=1)
        require(np.isfinite(losses).all() and (losses >= 0).all(), "DATA_ERROR: invalid loss")
        comparison = {}
        for route in (*ACTIONS, "l1", "l5", "l20", "ensemble_l1"):
            comparison[route] = []
            for metric, prefix in (("mse", "sse"), ("mae", "sae")):
                value = float(z[f"complete_{prefix}_{route}"].sum() / count.sum())
                reported = float(metrics[f"complete_target_{metric}_{route}"])
                require(np.isclose(value, reported, rtol=1e-7, atol=1e-9),
                        f"DATA_ERROR: recomputation mismatch seed{seed}/{route}/{metric}")
                comparison[route].append(value)
    require(all(sha256(p) == value for p, value in hashes.items()), "DATA_ERROR: source changed during read")
    return dict(seed=seed, features=features, losses=losses), dict(
        seed=seed, source_run=str(path), input_sha256=hashes, checks="PASS",
        full_validation_recomputed_metrics=comparison,
        source_checkpoint=cfg["source"]["checkpoint"],
        source_checkpoint_sha256=cfg["source"]["checkpoint_sha256"],
        checkpoint_not_loaded=True, test_arrays_not_loaded=True,
        mask_asset_hashed_only=True,
    )


def synthetic_sources():
    rng = np.random.default_rng(2233)
    sources = []
    for seed in SOURCE_RUNS:
        history = rng.random((1648, 192)) > .1
        regime = rng.integers(0, 2, size=1648)
        features = make_features(history, np.where(regime, 1., .01), np.where(regime, .01, 1.))
        losses = np.ones((1648, 4, 2))
        losses[:, 0] = [1.02, 1.02]
        losses[:, 1] = np.where(regime[:, None] == 1, [.97, .98], [1.03, 1.02])
        losses[:, 2] = np.where(regime[:, None] == 0, [.98, .97], [1.02, 1.03])
        sources.append(dict(seed=seed, features=features, losses=losses))
    return sources


def paired_bootstrap(base, chosen, cfg):
    """Same moving-block indices in all seeds; overlapping forecast targets respected."""
    rng = np.random.default_rng(cfg["seed"])
    n, block = base.shape[1], cfg["block_length"]
    samples = np.empty((cfg["replicates"], 2))
    for i in range(len(samples)):
        starts = rng.integers(0, n - block + 1, size=(n + block - 1) // block)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        samples[i] = relative_gain(base[:, idx].mean(1), chosen[:, idx].mean(1)).mean(0)
    return {"mean_gain_percent_ci95": np.quantile(samples, [.025, .975], axis=0).T.tolist(),
            "method": "paired_moving_block_same_indices_across_seeds",
            "replicates": cfg["replicates"], "block_length": block,
            "interpretation": "descriptive_only_about_three_effective_blocks_not_independent_significance"}


def evaluate(sources, cfg, out):
    ranges = {name: np.arange(*bounds) for name, bounds in cfg["splits"].items()}
    np.savez_compressed(out / "split_indices.npz", **ranges)
    fit, dev, audit = (ranges[k] for k in ("fit", "development", "audit"))
    frozen, calibration, costs = {}, {}, {}
    for policy in cfg["policies"]:
        x = np.concatenate([s["features"][policy][fit] for s in sources])
        y = np.concatenate([s["losses"][fit] for s in sources])
        start = time.perf_counter()
        model, scale = fit_selector(x, y, cfg["tree"])
        fit_seconds = time.perf_counter() - start
        dev_gains = [model.predict(s["features"][policy][dev]).reshape(-1, 3, 2) for s in sources]
        margin, records = calibrate(dev_gains, [s["losses"][dev] for s in sources], cfg["margin_candidates"])
        package = dict(model=model, margin=margin, fit_scale=scale, feature_names=FEATURE_NAMES[policy],
                       actions=ACTIONS, config_sha256=sha256(out / "resolved_config.yaml"))
        checkpoint = out / "checkpoints" / f"{policy}.joblib"
        temporary = checkpoint.with_suffix(".tmp")
        joblib.dump(package, temporary)
        os.replace(temporary, checkpoint)
        reloaded = joblib.load(checkpoint)  # only our own newly saved artifact
        require(np.array_equal(model.predict(x), reloaded["model"].predict(x))
                and reloaded["margin"] == margin, "CODE_ERROR: checkpoint roundtrip failed")
        frozen[policy] = reloaded
        calibration[policy] = dict(chosen_margin=margin, records=records,
            fit_scale=scale.tolist(), checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
        costs[policy] = dict(fit_seconds=fit_seconds, tree_nodes=int(model.tree_.node_count),
                            tree_leaves=int(model.get_n_leaves()), feature_count=x.shape[1],
                            trainable_neural_parameters=0, gpu_latency_ms=None)
        atomic_text(out / "checkpoints" / f"{policy}_tree.txt", export_text(model, feature_names=FEATURE_NAMES[policy]))
    # This file and both checkpoints are committed BEFORE any audit-loss scoring.
    write_json(out / "frozen_selection.json", dict(frozen_at=now(), calibration=calibration,
        audit_labels_used_in_fit_or_calibration=False))
    frozen_hash = sha256(out / "frozen_selection.json")
    print("22.33.3 selector checkpoints and dev margins frozen; starting audit", flush=True)
    rows, all_choices, predictions, oracle_full = [], {}, {}, []
    for source in sources:
        seed = source["seed"]
        loss = source["losses"][audit]
        base = loss[:, 3].mean(0)
        choices = {f"fixed_{a}": np.full(len(audit), i, dtype=np.int64) for i, a in enumerate(ACTIONS)}
        choices["label_aware_oracle"] = safe_oracle(loss)
        full_oracle = selected_losses(source["losses"], safe_oracle(source["losses"])).mean(0)
        oracle_full.append(dict(seed=seed, metrics=full_oracle.tolist(),
            gain_vs_fixed_multiscale_percent=relative_gain(source["losses"][:, 3].mean(0), full_oracle).tolist()))
        for policy, package in frozen.items():
            x = source["features"][policy][audit]
            start = time.perf_counter()
            pred = package["model"].predict(x).reshape(-1, 3, 2)
            selected = select_actions(pred, package["margin"])
            elapsed = time.perf_counter() - start
            costs[policy].setdefault("selector_inference_seconds_per_seed", {})[str(seed)] = elapsed
            choices[f"{policy}_calibrated"] = selected
            choices[f"{policy}_raw"] = select_actions(pred, 0.)
            predictions[f"seed{seed}_{policy}_predicted_gains"] = pred
        for policy, action in choices.items():
            chosen = selected_losses(loss, action)
            pair = chosen.mean(0)
            gain = relative_gain(base, pair)
            counts = np.bincount(action, minlength=4)
            calls = 3. if policy.startswith("disagreement") else float(np.array([1, 2, 2, 3])[action].mean())
            if policy == "label_aware_oracle":
                calls = None
            switched = action != 3
            both_harm = (chosen > loss[:, 3]).all(1)
            any_harm = (chosen > loss[:, 3]).any(1)
            rows.append(dict(seed=seed, policy=policy, sample_count=len(audit), mse=float(pair[0]),
                mae=float(pair[1]), mse_gain_percent=float(gain[0]), mae_gain_percent=float(gain[1]),
                action_counts=dict(zip(ACTIONS, counts.tolist())), fallback_fraction=float((action == 3).mean()),
                logical_forward_calls=calls, gpu_latency_ms=None,
                switches=int(switched.sum()), harmful_switches_any_metric=int((switched & any_harm).sum()),
                harmful_switches_both_metrics=int((switched & both_harm).sum()),
                protocol="same_cached_audit_rows_not_paper_test"))
            all_choices[f"seed{seed}_{policy}"] = action
        lower = loss.min(axis=1).mean(0)
        rows.append(dict(seed=seed, policy="separate_metric_ideal_bounds_NOT_one_policy",
                         sample_count=len(audit), mse=float(lower[0]), mae=float(lower[1]),
                         mse_gain_percent=float(relative_gain(base, lower)[0]),
                         mae_gain_percent=float(relative_gain(base, lower)[1]),
                         protocol="label_aware_separate_metric_bound"))
    summaries = {}
    for policy in (f"{p}_{variant}" for p in cfg["policies"] for variant in ("calibrated", "raw")):
        subset = [r for r in rows if r["policy"] == policy]
        gains = np.array([[r["mse_gain_percent"], r["mae_gain_percent"]] for r in subset])
        choices = np.stack([all_choices[f"seed{s['seed']}_{policy}"] for s in sources])
        base = np.stack([s["losses"][audit, 3] for s in sources])
        chosen = np.stack([selected_losses(s["losses"][audit], c) for s, c in zip(sources, choices)])
        pairs = [float((choices[i] == choices[j]).mean()) for i in range(3) for j in range(i + 1, 3)]
        summaries[policy] = dict(mean_gain_percent=gains.mean(0).tolist(),
            dual_winning_seeds=int((gains > 0).all(1).sum()),
            cross_seed_all_three_action_agreement=float((choices == choices[:1]).all(0).mean()),
            cross_seed_pairwise_action_agreement=float(np.mean(pairs)),
            **paired_bootstrap(base, chosen, cfg["bootstrap"]))
    primary_name = cfg["primary_policy"] + "_calibrated"
    primary = summaries[primary_name]
    passed = primary["dual_winning_seeds"] >= cfg["gate"]["minimum_dual_winning_seeds"] and min(
        primary["mean_gain_percent"]) >= cfg["gate"]["minimum_mean_gain_percent"]
    decision = ("PSAVS_DEVELOPMENT_PASS_SEED2029_CONFIRMATION_AUTHORIZED" if passed else "PSAVS_FEASIBILITY_STOP")
    require(sha256(out / "frozen_selection.json") == frozen_hash, "CODE_ERROR: frozen policy changed")
    np.savez_compressed(out / "audit_choices.npz", sample_index=audit, **all_choices, **predictions)
    # Save unified input features and losses with explicit split indices for exact reconstruction.
    np.savez_compressed(out / "offline_validation_inputs.npz", **{
        f"seed{s['seed']}_{k}": value for s in sources
        for k, value in {"losses": s["losses"], **s["features"]}.items()})
    chosen_rows = [r for r in rows if r["policy"] == primary_name]
    result = dict(artifact_status="PASS", decision=decision, scientific_gate_pass=bool(passed),
        uses_test_split=False, updates_forecasting_model=False,
        scope=cfg["experiment"]["protocol_scope"], sample_count_per_seed=len(audit),
        mse=float(np.mean([r["mse"] for r in chosen_rows])), mae=float(np.mean([r["mae"] for r in chosen_rows])),
        metric_scope="three_seed_mean_of_temporally_purged_validation_audit",
        checkpoint=str(out / "checkpoints" / f"{cfg['primary_policy']}.joblib"),
        frozen_selection_sha256=frozen_hash, calibration=calibration, costs=costs,
        summaries=summaries, rows=rows, full_validation_label_aware_oracle=oracle_full,
        paper_comparison="NA: validation subset and inherited PG95; not official paper test protocol",
        backward_optimizer="NA: sklearn tree fit, forecasting backbone not loaded or modified",
        limitations=["previously exposed validation; development evidence only",
            "disagreement features require all three forwards; no demonstrated inference savings",
            "Pareto-safe is predicted-gain gating, not a certified risk guarantee",
            "three overlapping-mask seeds share timestamps; bootstrap is descriptive only"])
    return result


def report(result):
    lines = ["# Step 22.33 PSAVS 离线可行性结果", "", f"决策：`{result['decision']}`。",
        "", "仅Traffic时间隔离validation audit；不是论文test结果，正收益表示优于固定多尺度。",
        "", "| seed | 策略 | MSE | MAE | MSE改善% | MAE改善% |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in result["rows"]:
        lines.append(f"| {row['seed']} | {row['policy']} | {row['mse']:.9f} | {row['mae']:.9f} | "
                     f"{row['mse_gain_percent']:.6f} | {row['mae_gain_percent']:.6f} |")
    lines += ["", "## 主门与开发校准", ""]
    for policy, entry in result["summaries"].items():
        lines.append(f"- `{policy}`：平均MSE/MAE改善={entry['mean_gain_percent']}%，"
                     f"双胜seed={entry['dual_winning_seeds']}/3，描述性95%区间={entry['mean_gain_percent_ci95']}。")
    lines += ["", "`margin=null`表示无穷大裕量，全部回退固定多尺度。", ""]
    for policy, entry in result["calibration"].items():
        lines.append(f"- `{policy}` dev冻结裕量：{entry['chosen_margin']}；树节点={result['costs'][policy]['tree_nodes']}。")
    lines += ["", "## 解释边界", "",
        "分歧感知策略始终需要3次预测前向；选择器CPU时间不等于GPU端到端推理时间。仅掩码策略调用预算是理论值，尚未部署测速。",
        "", "MSE（均方误差）、MAE（平均绝对误差）均越小越好。oracle（知晓真实误差的理想选择器）不可部署；两个指标分别最优的界也不是同一个可实现策略。",
        "", "现有validation已参与旧实验与原模型提前停止，本轮只提供开发证据，不宣称独立泛化、逐样本安全或SCI分区水平。",
        "", "主门通过才进入seed2029确认；失败则停止本配置，不以raw/oracle替代主策略，也不在audit上调参。", ""]
    return "\n".join(lines)


def capture_evidence(out):
    manifest = {}
    for relative in ("scripts/run_stage22_33_psavs.py", "scripts/test_stage22_33_psavs.py",
                     "scripts/start_stage22_33_psavs.sh", "configs/stage22_33_psavs.yaml",
                     "docs/STAGE22_33_PSAVS_PROTOCOL_ZH.md"):
        file = ROOT / relative
        if file.is_file():
            dest = out / "source_snapshot" / relative
            atomic_text(dest, file.read_text())
            manifest[relative] = dict(sha256=sha256(file), mtime_ns=file.stat().st_mtime_ns,
                                      snapshot=str(dest))
    write_json(out / "code_manifest.json", manifest)
    git_lines, patches = ["Project root is not a Git repository; tracked baseline state follows."], []
    for relative in ("baselines/S4M", "upstream/S4M"):
        directory = ROOT / relative
        git_lines.append(f"repository={directory}")
        for label, args in (("commit", ["rev-parse", "HEAD"]), ("status", ["status", "--short"]),
                            ("diffstat", ["diff", "--stat"]), ("diff", ["diff", "--binary"])):
            cp = subprocess.run(["git", "-C", str(directory), *args], text=True, capture_output=True, timeout=15)
            if label == "diff":
                patches.append(f"# repository={relative}\n" + cp.stdout)
            else:
                git_lines.append(f"[{label}] exit={cp.returncode}\n{cp.stdout}{cp.stderr}")
    atomic_text(out / "git_state.txt", "\n".join(git_lines))
    atomic_text(out / "code.diff", "\n".join(patches))
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib", "PyYAML", "torch")}
    env = dict(python=sys.version, executable=sys.executable, platform=platform.platform(), hostname=platform.node(),
        versions=versions, device="cpu", CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        conda_prefix=str(Path(sys.executable).parents[1]), cpu_threads=1,
        torch_cuda_runtime="not_loaded_in_cpu_selector; see inherited source environment",
        pid=os.getpid())
    atomic_text(out / "env.txt", json.dumps(env, indent=2))


def gpu_snapshot(out, phase):
    try:
        cp = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
                             "--format=csv"], text=True, capture_output=True, timeout=10)
        atomic_text(out / f"gpu_{phase}.txt", f"exit={cp.returncode}\n{cp.stdout}{cp.stderr}")
    except (OSError, subprocess.TimeoutExpired) as error:
        atomic_text(out / f"gpu_{phase}.txt", f"unavailable: {error}; CPU-only, no GPU allocated\n")


def run_once(cfg, smoke=False):
    start = time.perf_counter()
    stamp = dt.datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    experiment_id = f"psavs_offline__traffic__point__mr0.06__seed2233__{'smoke__' if smoke else ''}{stamp}"
    out = ROOT / "runs" / experiment_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoints").mkdir()
    resolved = json.loads(json.dumps(cfg))  # YAML int-key source mapping restored below
    resolved["source_runs"] = cfg["source_runs"]
    resolved["experiment"].update(experiment_id=experiment_id, smoke_test=smoke)
    if smoke:
        resolved["experiment"]["protocol_scope"] = "synthetic_cpu_selector_smoke_not_paper_results"
    atomic_text(out / "resolved_config.yaml", yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
    command = [sys.executable, str(Path(__file__).resolve()), "--mode", "smoke" if smoke else "formal"]
    atomic_text(out / "command.sh", "#!/bin/sh\n" + shlex.join(command) + "\n")
    status = dict(experiment_id=experiment_id, state="RUNNING", pid=os.getpid(), started_at=now(),
                  device="cpu", uses_gpu=False, config=str(out / "resolved_config.yaml"))
    write_json(out / "status.json", status)
    print(f"RUN_DIR={out}", flush=True)
    try:
        with (out / "train.log").open("w", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            capture_evidence(out)
            gpu_snapshot(out, "before")
            print(f"{now()} start {'synthetic smoke' if smoke else 'formal offline validation'} pid={os.getpid()}")
            if smoke:
                sources, manifests = synthetic_sources(), []
            else:
                loaded = [load_source(seed, path) for seed, path in cfg["source_runs"].items()]
                sources, manifests = [p[0] for p in loaded], [p[1] for p in loaded]
            write_json(out / "source_manifest.json", manifests)
            print("22.33.2 all source checks PASS; no test arrays or forecasting backbone loaded", flush=True)
            result = evaluate(sources, resolved, out)
            if smoke:
                require(result["scientific_gate_pass"], "CODE_ERROR: known predictive synthetic fixture not recovered")
            for manifest in manifests:
                require(all(sha256(path) == digest for path, digest in manifest["input_sha256"].items()),
                        "DATA_ERROR: input artifact changed during evaluation")
            write_json(out / "metrics.json", result)
            atomic_text(out / "report.md", report(result))
            string = io.StringIO()
            fields = sorted(set().union(*(r.keys() for r in result["rows"])))
            writer = csv.DictWriter(string, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result["rows"])
            atomic_text(out / "audit_results.csv", string.getvalue())
            gpu_snapshot(out, "after")
            status.update(state="SUCCESS", return_code=0, scientific_decision=result["decision"])
            print(json.dumps(dict(decision=result["decision"], primary=result["summaries"]["disagreement_aware_calibrated"])), flush=True)
    except Exception as error:
        with (out / "train.log").open("a") as handle:
            traceback.print_exc(file=handle)
        category = "DATA_ERROR" if "DATA_ERROR" in str(error) else "CODE_ERROR"
        status.update(state=category, return_code=1, verification_error=str(error))
        raise
    finally:
        status["finished_at"] = now()
        write_json(out / "status.json", status)
        write_json(out / "resource_usage.json", dict(elapsed_seconds=time.perf_counter() - start,
            peak_gpu_memory_mb=0, peak_cpu_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            device="cpu", gpu_uuid=None, pid=os.getpid(), started_at=status["started_at"],
            finished_at=status["finished_at"], return_code=status.get("return_code"),
            note="CPU peak RSS is cumulative for pipeline process; no GPU forward run in this stage"))
    if not smoke:
        for suffix, source in (("json", "metrics.json"), ("md", "report.md"), ("csv", "audit_results.csv")):
            atomic_text(ROOT / "reports" / f"stage22_33_psavs.{suffix}", (out / source).read_text())
    print(f"FINISHED={experiment_id} status=SUCCESS decision={result['decision']}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pipeline", "smoke", "formal"), default="pipeline")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    validate_config(cfg)
    runtime = ROOT / "runtime/stage22_33_psavs"
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / "pipeline.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode != "pipeline":
            if args.mode == "formal":
                evidence = json.loads((runtime / "smoke_pass.json").read_text())
                require(evidence["script_sha256"] == sha256(__file__)
                        and evidence["config_sha256"] == sha256(CONFIG), "CONFIG_ERROR: rerun smoke after changes")
            out = run_once(cfg, smoke=args.mode == "smoke")
            if args.mode == "smoke":
                write_json(runtime / "smoke_pass.json", dict(run_dir=str(out), script_sha256=sha256(__file__), config_sha256=sha256(CONFIG)))
            return
        state_path = runtime / "state.json"
        if state_path.is_file():
            old = json.loads(state_path.read_text())
            if old.get("state") == "SUCCESS":
                print(f"Already completed; no duplicate run. {old.get('formal_run_dir')}", flush=True)
                return
            require(old.get("state") not in ("CODE_ERROR", "DATA_ERROR"),
                    "CONFIG_ERROR: deterministic failure must be diagnosed before explicit retry")
        state = dict(state="RUNNING", pid=os.getpid(), started_at=now(), gpu_experiments=0,
            cpu_composite_experiments=1, automatic_oom_retry="NA_CPU_only_no_GPU_allocation")
        write_json(state_path, state)
        try:
            tests = subprocess.run([sys.executable, str(ROOT / "scripts/test_stage22_33_psavs.py")], cwd=ROOT)
            require(tests.returncode == 0, "CODE_ERROR: unit tests failed")
            state["unit_tests_exit_code"] = tests.returncode
            smoke = run_once(cfg, smoke=True)
            state["smoke_run_dir"] = str(smoke)
            write_json(runtime / "smoke_pass.json", dict(run_dir=str(smoke), script_sha256=sha256(__file__), config_sha256=sha256(CONFIG)))
            write_json(state_path, state)
            formal = run_once(cfg, smoke=False)
            state.update(state="SUCCESS", formal_run_dir=str(formal), finished_at=now(), return_code=0)
        except Exception as error:
            state.update(state="DATA_ERROR" if "DATA_ERROR" in str(error) else "CODE_ERROR",
                         error=str(error), return_code=1, finished_at=now())
            raise
        finally:
            write_json(state_path, state)
            subprocess.run([sys.executable, str(ROOT / "scripts/aggregate_results.py")], cwd=ROOT)
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
