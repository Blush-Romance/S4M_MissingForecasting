# 步骤22.17：MCPR失败归因

- 协议：`PASS`
- 决策：`MIXED_MCPR_FAILURE_MECHANISMS`
- test访问：`false`

## mcpr_mse

- 主干MSE/MAE：`0.872809589/0.754507652`
- 候选MSE/MAE：`0.876108911/0.752985774`
- alignment cross / residual energy / delta MSE：`-0.002574105 / 0.005873427 / 0.003299322`
- 机制归因：`AMPLITUDE_OVERSHOOT`

## mcpr_pcd

- 主干MSE/MAE：`0.872809589/0.754507652`
- 候选MSE/MAE：`0.915839322/0.744958994`
- alignment cross / residual energy / delta MSE：`0.018289357 / 0.024740376 / 0.043029734`
- 机制归因：`DIRECTION_ALIGNMENT_FAILURE`

完整样本四象限、难度分位、尾部集中、缺失结构相关、预测步/变量及Oracle上界见JSON。
