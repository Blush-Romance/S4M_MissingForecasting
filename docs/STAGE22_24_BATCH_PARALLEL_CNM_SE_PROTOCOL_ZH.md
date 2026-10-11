# 步骤22.24：三视图CNM-SE批处理并行协议

## 1. 研究问题

步骤22.23表明删除L5或L20会损失至少一半的多尺度收益。本步骤不再压缩视图，而是保持三路CNM-SE（Causal Nested-Mask Self-Ensembling，因果嵌套掩码自集成）的预测定义完全不变：

`0.5*original + 0.25*L5 + 0.25*L20`。

唯一变化是把三组形状为`B×T×D`的输入沿batch（批次）维拼成`3B×T×D`，用一次模型调用替代三次串行调用，再按原样拆分与融合。该工作定位为部署效率优化，不冒充新的预测模型创新。

## 2. 固定比较

- 数据与checkpoint：Traffic seed2027固定validation资产与既有PG95 checkpoint；
- 串行路径：original、L5、L20三次模型前向；
- 批处理并行路径：拼接original、L5、L20后一次模型前向；
- 两条路径共享完全相同的L5/L20 mask和预处理结果；
- 两条路径都使用0.5/0.25/0.25融合；
- 不更新模型、memory或任何参数，不访问test；
- batch size保持16，批处理并行路径的有效模型输入批量为48，这属于等价执行而非训练配置变化。

## 3. 同GPU计时

- 先完成3个validation batch的GPU smoke，其中1个warmup batch；
- smoke通过后运行全部103个validation batch，前10个作为warmup，正式计时93个batch；
- 计时排除DataLoader（数据加载器）与host-to-device（主机到GPU）传输；
- mask构造与极值预处理是两条部署路径的共同成本，均纳入总流水线耗时；
- 为控制先后顺序偏差，偶数batch先串行、奇数batch先批处理并行；
- 保存逐batch串行与并行总耗时、顺序、峰值显存和配对速度比。

## 4. 数值等价与速度门

在读取新结果前固定：

1. original、L5、L20及最终融合预测的全样本最大绝对差均不超过`1e-4`；
2. 串行与并行最终MSE、MAE的绝对差均不超过`1e-6`；
3. 正式validation批处理并行平均流水线耗时相对串行至少降低20%；
4. 93个配对batch自助法95%置信区间的耗时降低下界必须大于0；
5. 不发生OOM，且不通过缩batch、减模型、混合精度或减少正式validation规避显存。

若任一数值等价门失败，停止该实现并定位batch依赖来源；不得放宽容差迎合结果。若等价但速度收益不足20%，保留为工程负结果，不作主要效率贡献。

## 5. 复核规则

seed2027正式门全部通过后，才在seed2028完整validation上按相同协议独立复核。seed2028不得修改容差、warmup、batch size或实现。两seed均通过时，决策为`BATCH_PARALLEL_CNM_SE_CONFIRMED`；否则分别记录`EQUIVALENCE_FAIL`、`SPEED_GATE_FAIL`或`REPLICATION_FAIL`。

## 6. 完成判据

- 新入口默认关闭，关闭时原串行评估行为不变；
- CPU组件/配置负测、checkpoint加载与GPU smoke通过；
- 唯一ID、resolved config、代码diff、日志、状态、资源、逐batch计时与自动报告齐全；
- 报告不得把相同模型三视图的批处理优化描述为参数量或预测精度创新。
