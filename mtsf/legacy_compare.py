"""Compare MARS with the earlier project version on the *identical* legacy masks.

Legacy numbers are the test results recorded in reports/ (stage 5/10/16/20/22.22),
all with seq_len=192, pred_len=96, point missingness, r=0.06, training seed = mask seed.

    python -m mtsf.legacy_compare --out reports_mtsf/legacy_comparison.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

# (MSE, MAE) per mask seed, copied from the legacy reports
LEGACY = {
    "etth1": {
        "S4M (旧版复跑)": {2023: (1.207990943, 0.838923262)},
        "PG95 (旧版)": {2023: (1.211535967, 0.840500396)},
        "PG95+CNM-SE (旧版最好)": {2023: (1.211535970, 0.840500398)},
    },
    "weather": {
        "S4M (旧版复跑)": {2023: (0.2635, 0.3012)},
        "PG95 (旧版)": {2023: (0.261946456, 0.293878906), 2024: (0.271194642, 0.300850301),
                       2025: (0.256450080, 0.295111265)},
        "PG95+CNM-SE (旧版最好)": {2023: (0.254324883, 0.289493222), 2024: (0.264560075, 0.297525038),
                                2025: (0.253108127, 0.294119883)},
    },
    "traffic": {
        "S4M (旧版复跑)": {2026: (0.683681628, 0.366348940), 2027: (0.674754274, 0.365871256),
                        2028: (0.662156749, 0.358811680)},
        "PG95 (旧版)": {2026: (0.672083688, 0.361379550), 2027: (0.673996349, 0.363366556),
                       2028: (0.663565286, 0.361816861)},
        "PG95+CNM-SE (旧版最好)": {2026: (0.665930768, 0.358186681), 2027: (0.668019162, 0.360805960),
                                2028: (0.658421060, 0.359127571)},
    },
}
PAPER_S4M_L192 = {"etth1": (0.593, 0.574), "weather": (0.225, 0.305), "traffic": (0.687, 0.385)}


INTERIM = set()  # (dataset, seed) whose MARS number is an interim checkpoint


def ours():
    out = {}
    # final results first; interim (still-training) results only fill gaps
    files = sorted(glob.glob("results_mtsf/legacy__*.json"), key=lambda f: "__interim" in f)
    for f in files:
        r = json.load(open(f))
        model = os.path.basename(f).split("__")[1]
        d = out.setdefault(r["dataset"], {}).setdefault(model, {})
        if "__interim" in f:
            if int(r["seed"]) in d:
                continue
            INTERIM.add((r["dataset"], int(r["seed"])))
        d[int(r["seed"])] = (r["mse"], r["mae"])
    return out


def fmt(vals):
    v = np.array(list(vals))
    if len(v) == 1:
        return f"{v[0, 0]:.4f}", f"{v[0, 1]:.4f}"
    return f"{v[:, 0].mean():.4f} ± {v[:, 0].std(ddof=1):.4f}", f"{v[:, 1].mean():.4f} ± {v[:, 1].std(ddof=1):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    new = ours()
    lines = ["# MARS 与旧版本对比（同一份 legacy mask）", "",
             "协议：seq_len=192，pred_len=96，point 缺失（长度 5 的块），r=0.06，训练 seed = mask seed；",
             "mask 由旧版生成脚本重建，缺失 CSV 的 SHA-256 与 `artifacts/manifests/s4m_data_assets.yaml` 逐一一致。",
             "指标为测试集上对完整真值的 MSE/MAE（归一化空间）。多 seed 时给出 均值 ± 样本标准差，",
             "只在本方法已完成的 seed 上对比，种子不全时会注明。",
             "“MARS 相对”列为 MARS 相对该行方法的变化，负数表示 MARS 更好。", "",
             "> 注意：旧版在 ETTh1 上的 S4M/PG95 结果（MSE≈1.21）明显是训练失败（论文为 0.593，",
             "> 不处理缺失的 PatchTST 也有 0.45），所以 ETTh1 上的 -63% 主要不是 MARS 的贡献，",
             "> 应以 PatchTST 和论文数字作为保守参照。Weather/Traffic 旧版训练正常，对比更可信。", ""]
    for ds in ["etth1", "weather", "traffic"]:
        mine = new.get(ds, {})
        if not mine:
            lines += [f"## {ds}", "", "（尚未完成）", ""]
            continue
        seeds = sorted(mine.get("mars", {}))
        lines += [f"## {ds}（已完成 seed：{seeds}）", "", "| 方法 | MSE | MAE | MARS 相对 MSE | MARS 相对 MAE |", "|---|---|---|---|---|"]
        mars = mine.get("mars", {})
        m_mse = np.mean([mars[s][0] for s in seeds]) if seeds else None
        m_mae = np.mean([mars[s][1] for s in seeds]) if seeds else None
        rows = []
        for name, d in mine.items():
            ss = [s for s in seeds if s in d]
            if ss:
                rows.append((f"**{name.upper()}（新）**" if name == "mars" else f"{name}（新代码参照）", [d[s] for s in ss], ss))
        for name, d in LEGACY[ds].items():
            ss = [s for s in seeds if s in d]
            if ss:
                rows.append((name, [d[s] for s in ss], ss))
        for name, vals, ss in rows:
            mse, mae = fmt(vals)
            v = np.array(vals)
            rel = "—" if name.startswith("**") else f"{(m_mse_ss(mars, ss)[0] / v[:, 0].mean() - 1) * 100:+.1f}%"
            rela = "—" if name.startswith("**") else f"{(m_mse_ss(mars, ss)[1] / v[:, 1].mean() - 1) * 100:+.1f}%"
            note = "" if ss == seeds else f"（仅 seed {ss}）"
            if name.startswith("**") and any((ds, s) in INTERIM for s in ss):
                note += "（中途 checkpoint，训练未结束）"
            lines.append(f"| {name}{note} | {mse} | {mae} | {rel} | {rela} |")
        p = PAPER_S4M_L192[ds]
        lines.append(f"| S4M 论文 Table 1（官方未公开 mask，仅供参考） | {p[0]:.3f} | {p[1]:.3f} | "
                     f"{(m_mse / p[0] - 1) * 100:+.1f}% | {(m_mae / p[1] - 1) * 100:+.1f}% |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        open(a.out, "w").write(text + "\n")


def m_mse_ss(mars, ss):
    return np.mean([mars[s][0] for s in ss]), np.mean([mars[s][1] for s in ss])


if __name__ == "__main__":
    main()
