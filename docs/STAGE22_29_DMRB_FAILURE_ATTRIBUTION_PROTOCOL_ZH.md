# Step 22.29：DMRB 分支监督与组合失效归因冻结协议

冻结时间：2026-08-26。本文档在运行或计算任何新的 Step 22.29 分支级诊断之前写入。

## 1. 研究问题与禁止事项

DMRB（Dual-scale Mask-conditioned Residual Basis，双尺度掩码条件残差基）在 Traffic seed2027 validation（验证集）上工程合约全部通过，但 MSE（Mean Squared Error，均方误差）/MAE（Mean Absolute Error，平均绝对误差）为 `0.521787612/0.317986334`，相对 BF-CMVD-v2 退化 `0.199201%/0.109647%`，仅保留固定 CNM-SE 教师收益的 `29.7496%/30.7087%`。

本步只回答：负结果主要来自以下哪一项。

1. 联合教师获益掩码不适合两个分支的独立监督；
2. 两个已经学到的分支在固定相加时产生冲突；
3. 两个低秩分支对各自尺度目标仍然表达不足。

本步不得训练或更新模型，不得访问 test（测试集），不得调 DMRB rank（秩）、蒸馏权重、教师权重、融合权重或早停规则，不得依据结果同时选择多个后续模块。

## 2. 冻结资产与执行边界

- 数据：Traffic seed2027 固定 point-r0.06-L5 缺失资产、完整 Traffic CSV 和冻结归一化。
- 划分：chronological validation（按时间顺序验证集），batch size 16、drop-last；formal 固定完整 103 batch/1648 samples，smoke 固定前 3 batch/48 samples。
- 学生：Step22.28 DMRB formal 增量 checkpoint；必须校验来源 PG95 checkpoint 的绝对路径和 SHA-256，并验证 L5/L20 两分支键完整。
- 主干：同一次 DMRB 前向的 `p_student-r5-r20`，不得单独重新调用主干来制造跨调用差异。
- 教师：在同一恢复出的冻结 PG95 主干上生成固定等计数 L5/L20 视图；additional missing rate=0.15、block lengths=[5,20]、nested seed=2027、权重=`0.5/0.25/0.25`。
- model、prototype memory（原型记忆）和所有参数保持 eval/no-grad；不得构造 test loader。

## 3. 配对预测与修正

每个样本、每个 96 步预测时间单元跨 862 个变量定义：

- `p0 = p_student-r5-r20`：PG95 主干预测；
- `s5 = p0+r5`、`s20 = p0+r20`：两个单分支学生；
- `s = p0+r5+r20`：完整 DMRB 学生；
- `c5 = 0.25*(p5-p0)`、`c20 = 0.25*(p20-p0)`：冻结尺度教师贡献；
- `t5 = p0+c5`、`t20 = p0+c20`：两个尺度分量教师；
- `t = p0+c5+c20`：固定 CNM-SE 教师。

必须验证：`s=p0+r5+r20`、`t=p0+c5+c20`，工程容差固定为 `32×float32 epsilon≈3.8147e-6`。所有路径使用同一 complete target（完整目标）计算分支级 MSE/MAE。

## 4. 获益掩码与紧凑工件

严格双指标获益掩码定义为：

- `B5 = (MSE(t5)<MSE(p0)) AND (MAE(t5)<MAE(p0))`；
- `B20 = (MSE(t20)<MSE(p0)) AND (MAE(t20)<MAE(p0))`；
- `Bjoint = (MSE(t)<MSE(p0)) AND (MAE(t)<MAE(p0))`。

小于号保持严格，不改成单指标或非劣。必须保存逐样本×预测步的以下量：七路 MSE/MAE、三个教师获益掩码、三个学生获益掩码、L5/L20 自目标与交叉目标 dot/energy/mismatch、`c5·c20`、`r5·r20`、固定历史/L5/L20 mask 和 sample index。不得保存完整 `1648×96×862` 预测张量。

## 5. 必须报告的诊断量

### 5.1 联合监督与分支监督一致性

- `P(B5|Bjoint)`、`P(B20|Bjoint)`；
- `P(B5\\Bjoint|B5)`、`P(B20\\Bjoint|B20)`；
- B5/B20/Bjoint 两两 Jaccard（交并比）；
- Bjoint 内“两个分支都获益、仅 L5、仅 L20、都不单独获益”的比例；
- 教师分量目标 cosine（余弦）及 cancellation ratio（抵消率）：
  `max(0, 1-||c5+c20||²/(||c5||²+||c20||²+eps))`。

### 5.2 学生分支的单独效果与组合交互

- p0、s5、s20、s、t5、t20、t 的完整 validation MSE/MAE；
- 每个单分支和完整 DMRB 相对 p0 的改善；
- MSE/MAE interaction（交互项）：`L(s)-L(s5)-L(s20)+L(p0)`；
- 最佳单分支相对完整 DMRB 的净优势及其占最佳单分支主干收益的比例；
- 学生分支之间的 correction cosine 和 energy cancellation ratio。

### 5.3 分支表达与串扰

在 B5 内汇总 r5 对 c5，在 B20 内汇总 r20 对 c20：

- cosine；
- projection coefficient（投影系数）`sum(r·c)/sum(||c||²)`；
- normalized mismatch（归一化失配）`sum(||r-c||²)/sum(||c||²)`；
- 分支教师—学生正 MSE/MAE 差距位于各自获益区的比例。

同时报告 cross projection（交叉投影）：r5 对 c20、r20 对 c5，用于区分自目标欠拟合和尺度串扰。

## 6. 结果前冻结的主因判定顺序

判定按以下顺序执行，后项不得覆盖先满足的前项。

1. 任一资产、checkpoint、预测恒等式、固定 mask、指标回算、validation-only、样本数或有限值检查失败：`PROTOCOL_FAILURE_NO_DECISION`。
2. **尺度监督错配**：同时满足：
   - `min(P(B5|Bjoint), P(B20|Bjoint)) < 0.75`；
   - `max(P(B5\\Bjoint|B5), P(B20\\Bjoint|B20)) > 0.25`；
   - 教师分量 correction cosine `<0` 或 cancellation ratio `>0.10`。
   决策：`SCALE_SUPERVISION_MISMATCH_FREEZE_BRANCH_PARETO_MASKS`。
3. **固定相加冲突**：最佳单分支在 MSE 和 MAE 上都严格优于完整 DMRB；MSE 和 MAE interaction 均为正；完整 DMRB 相对最佳单分支损失的收益比例在两指标上都至少 25%。决策：`BRANCH_SUM_CONFLICT_FREEZE_CAUSAL_SCALE_ROUTER`。
4. **双分支表达不足**：对 L5 和 L20 两个分支分别要求：其教师—学生正 MSE/MAE 差距至少 60% 位于自己的 B5/B20 区；且自目标 cosine `<0.50`、projection `<0.50`、normalized mismatch `>0.75` 三项中至少两项成立。决策：`DUAL_BRANCH_EXPRESSION_UNDERFIT_STOP_LOW_RANK_ROUTE`。
5. 若不满足任一单一主因，或仅一个分支满足表达不足：`MIXED_OR_INDETERMINATE_STOP`。不得移动阈值以得到想要的标签。

## 7. 唯一后续映射

- 尺度监督错配：只冻结 Branch-Pareto DMRB（分支帕累托 DMRB）。L5 分支只在 B5 的训练期观测双指标获益 token 蒸馏，L20 分支只在 B20 蒸馏；直接目标 MSE、两个分支结构、固定求和和教师权重不变，不增加推理模块。
- 固定相加冲突：只冻结 Causal Scale Router（因果尺度路由器）。候选只能使用历史 mask、主干隐藏表示和分支残差等推理时可观测量，并先单独验证路由可学性；不得同时改分支监督。
- 双分支表达不足：停止当前低秩残差蒸馏路线，不在同一 seed2027 validation 上继续加 rank/深度追门。
- 混合证据：停止，不堆叠 Branch-Pareto 与路由器。

## 8. 工件与完成判据

- 每个 run 使用唯一 ID，并保存 resolved config、command、env、git state、code diff、source snapshot、status、resource usage、metrics 和 compact NPZ。
- smoke 必须验证 DMRB 增量 checkpoint 可恢复、双分支键、同前向分解、固定嵌套 mask、指标回算、有限值、chronological loader、样本数和 test 禁用，全部通过后才允许 formal。
- formal 必须闭环到 Step22.28 已保存的 p0/完整 DMRB/t 指标容差内，完整处理 103 batch/1648 samples，并由自动分析器输出唯一主因或停止标签。
