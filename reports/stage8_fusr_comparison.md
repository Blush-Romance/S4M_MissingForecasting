# 步骤8：Weather FUSR 独立单 seed 比较

- 协议状态：`PASS`；协议差异数：`0`。
- 固定 mask：`artifacts/masks/weather/point/rate_0.06/seed_2023.npz`。
- 基线 MSE / MAE：`0.263525745` / `0.301159963`。
- FUSR MSE / MAE：`0.263705272` / `0.297522085`。
- MSE 相对收益：`-0.0681%`。
- MAE 相对收益：`+1.2080%`。
- 参数增量：`+452`；峰值显存：`3070.80419921875` MB。
- 实际优化器 steps / early stopping（早停）epoch：`18288` / `8`。
- 同时改善 MSE 与 MAE：`False`。

项目内基线与 FUSR 沿用同一固定 mask，但原论文 mask seed 未公开，因此不声称与原论文具有完全相同的随机 mask。
本报告仅用于 Weather、seed=2023 的方向筛选，不构成统计稳定性或 SCI 分区结论。
