# 步骤22.16：MCPR快速筛选协议

## 模块定义

MCPR（Mask-Conditioned Pareto Residual，掩码条件帕累托残差）加载步骤21的PG95正式检查点，冻结PG95全部参数和原型记忆，只训练一个位于S4预测主干之后的低秩残差模块。

残差输入包括冻结主干隐藏状态，以及历史掩码的逐时刻观测状态、缺失切换、距最近观测的时间和全局观测比例。输出投影零初始化，因此训练开始前严格复现PG95预测。

## 两个首轮身份

1. `MCPR-MSE`：只用原始MSE（均方误差）训练相同适配器，回答“后主干显式掩码注入是否有效”；
2. `MCPR-PCD`：分别计算MSE与MAE（平均绝对误差）对MCPR参数的梯度，将两者按全局L2范数归一化后等权相加。该方向称为Pareto Common Descent（帕累托共同下降），用于减少双风险梯度尺度冲突。

## 固定设置

- 数据：Electricity，point-r0.06/L5，seed2023固定资产；
- 来源：步骤21 PG95正式最佳检查点；
- batch size：16；
- optimizer：Adam；
- MCPR学习率：0.001；
- 最大epoch：15；early stopping patience：3；
- rank：32；因果卷积核：5；
- checkpoint选择仍使用原验证MSE；
- post-train evaluation只读取chronological validation，不访问test；
- MSE-only与PCD除目标梯度组合外完全相同。

## 继续门与公平性

首轮属于post-hoc方向筛选，因为它在已训练PG95之后增加适配训练预算。任一候选只有在完整2528样本validation上相对冻结PG95满足：MSE和MAE均不退化，且至少一项严格改善，才允许进入下一阶段。

若通过，下一阶段必须补充PG95等额继续训练控制，然后才能把收益归因于MCPR；若不通过，停止该候选，不补多seed、不访问test。
