# Step 22.27：BF-CMVD-v2 失效归因冻结协议

冻结时间：2026-08-21。本文档在运行任何 Step 22.27 新逐单元诊断之前写入。

## 1. 研究问题

BF-CMVD-v2（Benefit-Filtered Causal Multiview Distillation，收益筛选的因果多视图蒸馏）在 Traffic seed2027 validation 上相对 PG95 同时改善 MSE（Mean Squared Error，均方误差）0.5199% 和 MAE（Mean Absolute Error，平均绝对误差）0.3587%，但只保留固定三视图 CNM-SE（Causal Nested-Mask Self-Ensemble，因果嵌套掩码自集成）教师收益的 48.07% 和 44.16%。

本步只回答：剩余教师—学生差距主要来自监督覆盖错配、学生表达不足，还是学生在教师非获益区域产生副作用。不得用本步访问 test、补 seed、训练新结构、放宽 50% 门或调整现有模型权重。

## 2. 冻结资产与模型路径

- 数据：Traffic seed2027 固定 point-r0.06-L5 缺失资产。
- 划分：chronological validation（按时间顺序的验证集），batch size 16，drop-last，完整 103 batch / 1648 samples；smoke 固定 3 batch / 48 samples。
- 完整目标：`data/traffic/traffic.csv`；观测目标和历史输入来自固定缺失 CSV。
- 学生：Step 22.26 BF-CMVD-v2 formal checkpoint。
- 主干：同一学生 checkpoint 内关闭 MCPR（Mask-Conditioned Pareto Residual，掩码条件帕累托残差）后的冻结 PG95 路径；其参数必须与来源 PG95 checkpoint 身份一致。
- 教师：在关闭 MCPR 的同一冻结 PG95 上构造严格等移除 L5/L20 视图，固定为 `0.5×original + 0.25×L5 + 0.25×L20`。
- nested seed=2027，target additional missing rate=0.15，掩码生成必须逐样本与 Step 22.22 已保存 validation 工件一致。
- test loader 不得构造，model 和 prototype memory（原型记忆）不得更新。

## 3. 配对张量和筛选单位

每个样本、每个 96 步预测时间单元跨 862 个变量计算：

- `E_b^2 / E_b^1`：PG95 主干的跨变量 MSE / MAE；
- `E_t^2 / E_t^1`：固定多视图教师的跨变量 MSE / MAE；
- `E_s^2 / E_s^1`：BF-CMVD-v2 学生的跨变量 MSE / MAE；
- `δ_t = prediction_teacher - prediction_backbone`：教师修正；
- `δ_s = prediction_student - prediction_backbone`：学生修正。

完整目标教师获益单元固定为：

`B_complete = (E_t^2 < E_b^2) AND (E_t^1 < E_b^1)`。

观测目标教师获益单元沿用训练时 token（时间单元）规则，只在 target mask 有观测时定义：

`B_observed = eligible AND (E_t,obs^2 < E_b,obs^2) AND (E_t,obs^1 < E_b,obs^1)`。

所有小于号保持严格，不改成非劣或单指标筛选。

## 4. 必须输出的诊断量

### 4.1 监督覆盖

- validation 目标观测覆盖率；
- `B_observed` 对 `B_complete` 的 precision（精确率）、recall（召回率）、F1 和 Jaccard（交并比）；
- 完整教师获益率在目标已观测与未观测时间单元之间的绝对差；
- 逐预测步的观测覆盖率和教师获益率。

### 4.2 学生—教师修正表达

- 全体、`B_complete` 内、`B_complete` 外的 correction cosine（修正余弦相似度）；
- projection coefficient（投影系数）`sum(δ_s·δ_t) / sum(||δ_t||²)`；
- normalized mismatch（归一化失配）`sum(||δ_s-δ_t||²) / sum(||δ_t||²)`；
- sign agreement（符号一致率）和学生残差能量在获益/非获益区的占比；
- 上述量的逐预测步汇总。

### 4.3 教师优势损失分解

对 MSE 与 MAE 分别定义正教师—学生差距：

`G+ = max(E_s - E_t, 0)`。

将其严格分成 `B_complete` 内与外两部分，报告两者占全部 `G+` 的比例。同时报告：

- 获益区内学生相对主干/教师的净变化；
- 非获益区内学生相对主干/教师的净变化；
- 学生相对主干的正伤害质量 `max(E_s-E_b,0)`；
- 教师与学生各自双指标获益时间单元比例。

## 5. 结果前冻结的主因判定

判定顺序固定如下。

1. 任一资产、checkpoint、掩码、指标回算、validation-only 或冻结性检查失败：`PROTOCOL_FAILURE_NO_DECISION`。
2. 监督覆盖错配：`B_complete` 的观测支持召回率 `<0.70`，且完整教师获益率在已观测/未观测单元间绝对差 `>0.05`。两条件须同时满足。
3. 学生表达不足：MSE 和 MAE 的正教师—学生差距均至少 60% 位于 `B_complete` 内，并且获益区至少满足一项：余弦 `<0.50`、投影系数 `<0.50`、归一化失配 `>0.75`。
4. 非获益区域副作用：MSE 和 MAE 的正教师—学生差距均至少 60% 位于 `B_complete` 外；学生在该区域相对主干的 MSE/MAE 均为正伤害；且该区域承载至少 25% 的学生残差能量。
5. 若第 3 和第 4 条同时满足，只在两者较小的双指标份额仍相差至少 0.10 时选择份额更高者；否则记为 `MIXED_OR_INDETERMINATE_STOP`。
6. 不满足以上任一清晰主因同样停止，不为得到结论而移动阈值。

## 6. 主因到唯一 BF-CMVD-v3 的映射

- 若为监督覆盖错配：只冻结 Observability-Balanced BF-CMVD（可观测性均衡收益蒸馏）。按训练时可观测的预测步/目标掩码状态估计支持概率，对获益监督做裁剪的逆概率加权；不填造未观测标签。
- 若为学生表达不足：只冻结 DMRB（Dual-scale Mask-conditioned Residual Basis，双尺度掩码条件残差基）。保持单次 PG95 前向，用两个可关闭低秩残差基分别蒸馏 L5 与 L20 教师修正，再固定相加；不得同时增加可靠性门。
- 若为非获益区域副作用：只冻结 CBR（Causal Benefit Router，因果收益路由器）。用历史 mask、主干预测和残差等推理时可观测量学习训练期获益标签，并门控现有残差；不得同时扩大学生残差容量。
- 若证据混合，或所需决策只能依赖未来完整真值：停止 BF-CMVD-v3。

## 7. 工件与完成判据

- 每次 run 有唯一 ID、resolved config、command、env、git state、code diff、status、resource usage、metrics 和 compact NPZ。
- NPZ 只保存逐样本×预测步的误差/掩码/修正聚合与逐变量总量，不保存四路完整预测张量。
- smoke 通过 checkpoint 身份、预测分解、固定掩码、指标回算、有限值、样本数和 test 禁用检查后，才允许完整 validation。
- 完整诊断必须复算到既有 PG95、CNM-SE 和 BF-CMVD-v2 validation 指标容差内，并由自动分析器给出唯一标签或停止标签。

