# 步骤8：Weather FUSR V2 冻结路由器单 seed 比较

- 产物状态：`PASS`；主干、数据与测试协议：`PASS`。
- 固定测试掩码：`artifacts/masks/weather/point/rate_0.06/seed_2023.npz`。
- 基线 MSE（均方误差）/ MAE（平均绝对误差）：`0.263525745` / `0.301159963`。
- FUSR V1 MSE / MAE：`0.263705272` / `0.297522085`；路由状态：`COLLAPSED_ALL_ON`。
- FUSR V2 MSE / MAE：`0.263565329` / `0.301181243`；路由状态：`SELECTIVE`。
- V2 相对基线 MSE 收益：`-0.0150%`。
- V2 相对基线 MAE 收益：`-0.0071%`。
- 研究判断：`NEGATIVE_NO_MSE_LIFT_ITERATE_WITHIN_STEP8`。

`artifact_status=PASS` 仅表示代码、检查点与指标产物完整，不等于科学结论为正。
本报告只用于 Weather、seed=2023 的快速方向筛选；原论文 mask seed 未公开，因此不声称与原论文随机 mask 完全一致。
