# 步骤22.17：MCPR失败归因预注册协议

## 1. 研究问题

步骤22.16已预注册停止MCPR（Mask-Conditioned Pareto Residual，掩码条件帕累托残差）精度分支：MCPR-MSE和MCPR-PCD都改善MAE（Mean Absolute Error，平均绝对误差），但退化MSE（Mean Squared Error，均方误差）。本步只回答退化来自哪里，不将归因包装成新精度结论。

## 2. 冻结对象

- 数据：Electricity（电力）数据集的chronological validation（时间顺序验证集）。
- seed（随机种子）：2023。
- 缺失协议：point-r0.06（点缺失，名义缺失率6%）及已固定L=5训练/评估资产。
- 参考模型：步骤21的冻结PG95。
- 候选检查点：步骤22.16的MCPR-MSE与MCPR-PCD（Pareto Common Descent，帕累托共同下降）。
- 评估范围：只读、完整validation；不更新模型、optimizer（优化器）或memory（原型记忆库）；不构造或访问test loader（测试集加载器）。

## 3. 精确MSE残差分解

对每个完整目标元素，记PG95冻结主干预测为 `b`，MCPR残差为 `r`，真值为 `y`：

```text
候选预测 = b + r
候选平方误差 - 主干平方误差
= 2 * (b - y) * r + r^2
```

固定名称：

- `alignment_cross`：`2 * (b - y) * r`，负值表示残差方向在一阶上修正主干误差，正值表示方向错误。
- `residual_energy`：`r^2`，恒非负，表示残差幅度代价。
- `delta_sse`：候选SSE（Sum of Squared Errors，平方误差和）减主干SSE，必须与前两项之和数值闭环。

分解同时在四个层级保存：全局、逐样本、96个预测步、321个变量。MAE不做伪加性分解，只保存严格配对的SAE（Sum of Absolute Errors，绝对误差和）变化。

## 4. 预注册诊断视图

1. **总体闭环**：缓存重算的MSE/MAE必须在 `1e-5` 内闭环到各步骤22.16源指标；MCPR检查点内部的冻结主干必须在 `1e-5` 内闭环到PG95。
2. **样本四象限**：统计MSE和MAE同时改善、仅MSE改善、仅MAE改善、两者均退化的样本比例。
3. **PG95难度分组**：按PG95逐样本MSE排序后固定分10组，不根据MCPR结果改分组。
4. **尾部风险**：报告逐样本绝对误差的50/90/95/99分位数与最大值，以及最差1%/5%/10%样本对总`delta_sse`的贡献；不根据结果改百分比。
5. **预测步与变量**：报告完整的逐轴MSE/MAE变化和残差分解，仅在Markdown中摘要最好/最差10个索引。
6. **缺失结构关联**：从固定历史mask（掩码）导出实际缺失率、缺失段数和最长缺失段，计算它们与`delta_sse`、`delta_sae`、`alignment_cross`及`residual_energy`的Spearman（斯皮尔曼秩相关）。
7. **时间不确定性**：对逐样本MSE/MAE变化使用移动块bootstrap（自助法），固定seed=2023、block length=288、repeats=2000，报告95%置信区间。
8. **Oracle上界**：可报告标签知情的逐样本选择或最优缩放上界，但必须标注`not_deployable_uses_validation_labels=true`，不得用于后续调参或精度主张。

## 5. 协议门禁

每个缓存必须同时满足：

- checkpoint包含MCPR参数且全部模型张量有限；
- `use_mcpr=true`，方法身份与`mcpr_objective`一致；
- `use_mipp=true`且原型写入模式为PG95的`absolute`；
- `updates_model_or_memory=false`，评估过程无反向传播、无optimizer step（优化步）、无memory写入；
- 完整运行为2528样本/158 batch（批次），smoke单独标记且不当作正式结果；
- 两个MCPR缓存与既有PG95缓存的sample index、历史mask、嵌套mask和元素计数完全一致；
- `delta_sse = alignment_cross + residual_energy`在全局、逐样本、预测步和变量层级都通过数值容差检查。

## 6. 解释规则与停止规则

- 若`alignment_cross`全局为非负，主要问题定义为**方向对齐失败**。
- 若`alignment_cross`全局为负，但正的`residual_energy`绝对值更大并使`delta_sse>0`，主要问题定义为**幅度过冲**。
- 若总体主要由固定尾部样本贡献，则记录为**尾部集中失败**，但不就地选阈值。
- 诊断结果只决定下一个机制级假设类型，不恢复MCPR分支，不授权多seed或test。
- 由于Electricity seed2023 validation已用于发现，后续新模块的确证性首轮必须改用未参与该归因的固定资产，不能只在同一validation上宣称改善。

## 7. 执行顺序

1. 扩展严格MCPR方法身份、checkpoint验证和诊断工件schema。
2. 运行CPU合成分解单元测试、配置负测和静态编译。
3. 对MCPR-MSE和MCPR-PCD各运行2-batch GPU smoke，验证checkpoint加载、主干重建、数值闭环、NPZ保存和JSON指标。
4. smoke通过后，使用UUID动态队列运行两个完整validation缓存，最大并发5但按实时显存动态启动。
5. 两个正式缓存成功后自动运行冻结分析器，输出JSON与中文Markdown报告。
