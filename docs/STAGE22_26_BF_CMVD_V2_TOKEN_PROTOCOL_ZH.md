# 步骤22.26 BF-CMVD-v2 时间单元收益筛选协议

## 1. 失败证据与唯一改动

BF-CMVD-v1 的真实 smoke 完成了 4 个优化步，但训练观测目标上的整样本获益数为 `0/64`；同一冻结教师在 seed2027 完整 validation 的 1648 个样本上有 `1213/1648` 个样本同时改善 MSE 与 MAE。v1 因筛选粒度过粗停止，旧工件不得覆盖。

v2 唯一方法改动是把收益筛选单位从整个样本 `[B]` 改为预测时间单元 `[B,T]`。每个具有观测目标的时间单元，只有教师相对原始 PG95 的跨变量 MSE 和 MAE 同时更低，才对该时间单元施加教师 L1 蒸馏。完整目标只用于上述离线失败诊断，不进入训练或选择。

## 2. 冻结项

- 数据、seed、PG95 checkpoint、batch=16、rank=32、kernel=5、Adam、adapter lr=0.001、最多15 epoch、patience=3不变。
- 教师仍为 `0.5×original + 0.25×L5 + 0.25×L20`；L5/L20均额外移除精确15%已观测历史行。
- 总损失仍为观测目标 MSE 加获益时间单元上的观测目标 L1 蒸馏，两个系数固定为1。
- 主干、prototype memory和教师均冻结；推理只运行 original PG95 与单个低秩残差头。
- 只读取train和validation；本步骤禁止test。

## 3. 门禁

smoke 必须同时证明：获益与非获益时间单元均存在、蒸馏项/梯度/残差/教师差异均有限且非零、checkpoint可保存并重载。否则formal在GPU启动前阻断。

seed2027正式validation必须同时优于原始PG95的MSE和MAE，并对两项分别保留至少50%的CNM-SE收益，才允许seed2028 validation。不得依据本轮结果调权重、目标缺失率或教师组成。
