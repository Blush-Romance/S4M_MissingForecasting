#!/usr/bin/env python3
"""CPU-only audit of stochastic prototype selection; no data/forecast evaluation."""
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from stage22_34_factorial import PG95_RUNS, digest, write_json


def main():
    records = []
    for seed in (2026, 2027, 2028):
        s4m = yaml.safe_load((ROOT / f"configs/stage22_34_factorial/s4m_traffic_validation_seed{seed}.yaml").read_text())
        pg95 = yaml.safe_load((ROOT / PG95_RUNS[seed] / "resolved_config.yaml").read_text())
        for family, config in (("s4m", s4m), ("pg95", pg95)):
            checkpoint = Path(config["source"]["checkpoint"])
            source = yaml.safe_load(Path(config["source"]["resolved_config"]).read_text())
            state = torch.load(checkpoint, map_location="cpu")
            count = int(state["repre"].shape[0])
            limit = int(source["official_args"]["topM"])
            rng_keys = [key for key in state if any(word in key.lower() for word in ("rng", "random", "sampler"))]
            record = dict(seed=seed, family=family, checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                prototype_count=count, topM=limit, random_sampling_branch_active=count > limit,
                checkpoint_top_level_keys=list(state), saved_rng_keys=rng_keys)
            if count > limit:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    initial = torch.get_rng_state().clone()
                    first = torch.randperm(count)[:limit]
                    next_single_view = torch.randperm(count)[:limit]
                    torch.set_rng_state(initial)
                    replay_first = torch.randperm(count)[:limit]
                    for _ in range(3):
                        torch.randperm(count)[:limit]  # L1/L5/L20 additional forwards
                    next_multiview = torch.randperm(count)[:limit]
                record["cpu_selection_probe"] = dict(
                    same_state_replays_exactly=bool(torch.equal(first, replay_first)),
                    interleaved_views_change_next_original_subset=not bool(torch.equal(next_single_view.sort().values, next_multiview.sort().values)),
                    overlap_of_next_original_subsets=len(set(next_single_view.tolist()) & set(next_multiview.tolist())),
                    description="sampling-only control-flow probe, not forecast accuracy experiment")
                assert record["cpu_selection_probe"]["same_state_replays_exactly"]
                assert record["cpu_selection_probe"]["interleaved_views_change_next_original_subset"]
            records.append(record)
            del state
    failed = ROOT / "runs/cnm_se_equal_rate_multiscale_s4m__traffic__validation__seed2027__20260907_193113"
    failed_config = yaml.safe_load((failed / "resolved_config.yaml").read_text())
    source_config = yaml.safe_load(Path(failed_config["source"]["resolved_config"]).read_text())
    result = dict(status="PASS", scope="cpu_checkpoint_and_rng_control_flow_audit_no_forecasting_no_test",
        historical_and_failed_gpu_identical=source_config["runtime"]["gpu_uuid"] == failed_config["runtime"]["gpu_uuid"],
        gpu_uuid=failed_config["runtime"]["gpu_uuid"], records=records,
        code_location="baselines/S4M/s4m/model/Bank.py:420",
        conclusion="evaluation uses random prototype subsampling even in eval mode; multiview changes the original-view random stream",
        limitation="proves an uncontrolled stochastic path, not that it explains every bit of the observed metric discrepancy",
        recovery="do not blindly retry or relax tolerance; historical RNG is absent; a new fixed-RNG paired reference protocol needs explicit approval")
    write_json(ROOT / "reports/stage22_34_rng_diagnosis.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
