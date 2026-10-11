# 步骤22.23：高效CNM-SE协议（两路筛选与全新seed确认）

## 1. 问题与边界

步骤22.22确认多尺度CNM-SE（Causal Nested-Mask Self-Ensembling，因果嵌套掩码自集成）相对PG95在Traffic三个seed上均双指标改善，但正式部署需要original、L5、L20三次前向，串行耗时约为单次PG95的3.02–3.08倍。

本步骤检验：能否用original加一个嵌套视图的两路版本保留大部分收益，把部署前向次数从3次降到2次。

- 不修改PG95训练、MIPP、mask、数据划分或指标；
- 不搜索连续融合权重；两路权重固定为0.5/0.5；
- 现有seed2026/2027/2028 test已在步骤22.22暴露，因此只能作事后探索，不能提供新的确认性证据；
- 最终确认必须使用在候选冻结后生成的全新seed2029资产和PG95 checkpoint；
- 本步骤不试图修复PG95相对S4M的跨seed不稳定，研究主张限定为“压缩CNM-SE相对PG95的稳定增量”。

## 2. 固定候选

候选集合只包含已有评估器已经保存的三种等实际移除数量两路融合：

- `E1 = 0.5*original + 0.5*L1`；
- `E5 = 0.5*original + 0.5*L5`；
- `E20 = 0.5*original + 0.5*L20`。

参照方法仍为：

- PG95 original；
- 三路CNM-SE：`0.5*original + 0.25*L5 + 0.25*L20`。

L1/L5/L20的nested seed、目标额外缺失率0.15、严格等移除数量和掩码生成规则全部保持步骤22.22不变。

## 3. validation-only选择规则

选择器只读取seed2026/2027/2028三个完整validation工件，不读取test字段进行排名。

候选首先必须同时满足：

1. 三个seed上均严格改善PG95 original的MSE；
2. 三个seed上均严格改善PG95 original的MAE；
3. 三seed平均MSE相对改善至少保留三路CNM-SE平均MSE改善的80%；
4. 三seed平均MAE相对改善至少保留三路CNM-SE平均MAE改善的80%。

通过门槛的候选按以下固定顺序选择：

1. 最大化三seed平均的`MSE相对改善率 + MAE相对改善率`；
2. 若总分在`1e-12`内相同，优先平均MSE改善更高者；
3. 再相同则按L5、L20、L1顺序打破平局。

若没有候选通过，决策为`EFFICIENT_TWO_VIEW_STOP`，不生成seed2029资产；下一方向改为保持三路预测完全等价的批处理并行化，不能放宽80%门槛。

## 4. 已暴露test的使用限制

validation选出候选后，可以从已有三个test工件读取该候选指标，用于回答“是否值得花费新seed训练成本”。该结果必须标记为post-hoc exploratory（事后探索）。只有候选在已有test三个seed上也全部双指标改善PG95 original，才允许进入seed2029确认。

已有test无论结果如何都不得：

- 改变候选块长；
- 修改0.5/0.5权重；
- 修改80%门槛；
- 在E1/E5/E20之间重新选择；
- 作为最终独立确认结果。

## 5. seed2029确认协议

若前述门通过：

- 使用与步骤22.22完全相同的Traffic point-r0.06、L5、目标实际缺失行率0.26算法，固定生成seed2029；
- 只训练一个PG95 checkpoint，以降低磁盘和GPU开销；无需重复训练S4M，因为本步骤主比较是高效两路候选相对同checkpoint PG95 original；
- PG95结构、batch size 16、Adam lr0.001、最多15 epoch、patience3和MIPP全部参数保持不变；
- 先2 train-batch/3 validation-batch smoke，再完整正式训练；
- 正式checkpoint后先做3-batch评估smoke，再完整validation与test；test在本协议中预先授权，不以seed2029 validation好坏决定是否运行；
- seed2029只计算original与冻结候选所需的单个嵌套视图；若复用通用评估器计算额外视图，正式效率计时和主结果必须只统计部署所需两路。

磁盘约束：启动时项目盘仅约2.1 GiB可用。seed2029只允许新增一份约194 MiB缺失CSV和一个PG95 checkpoint及必要工件；不得生成S4M checkpoint或冗余预测大数组。任何删除历史资产仍需用户明确授权。

## 6. 决策标签

- `EFFICIENT_TWO_VIEW_STOP`：无候选通过validation门；
- `EFFICIENT_TWO_VIEW_VALIDATION_SELECTED_TEST_EXPLORATORY_FAIL`：validation候选已冻结，但已有test探索不是3/3 seed双指标改善；
- `EFFICIENT_TWO_VIEW_FRESH_SEED_AUTHORIZED`：validation门及已有test探索均通过，允许seed2029确认；
- `EFFICIENT_TWO_VIEW_FRESH_SEED_FAIL`：seed2029完整，但候选未同时改善MSE和MAE；
- `EFFICIENT_TWO_VIEW_FRESH_SEED_CONFIRMED`：seed2029候选相对PG95 original双指标严格改善，且工件/成本审计通过。

任何标签都不等于保证SCI二区；它只补足当前方法的准确率—效率证据。

## 7. 完成判据

- 本协议早于三seed两路候选汇总和seed2029资产；
- validation选择器通过合成正/负例，且源码不读取test进行候选排名；
- test探索明确标记非独立；
- seed2029资产、训练和评估均有唯一ID、resolved config、代码diff、日志、状态、资源与自动报告；
- 报告同时给出两路候选相对PG95与三路CNM-SE的精度保留率、参数量、峰值显存和推理耗时。
