# Step 22.31：PBA-CMVD validation-only 失败归因协议

## 1. 研究问题

Step 22.30 的 PBA-CMVD（Partial-Backbone Amortized Causal Multi-View Distillation，部分主干摊销因果多视图蒸馏）相对 PG95 同时改善 MSE（均方误差）和 MAE（平均绝对误差），但相对 BF-CMVD-v2 的 MSE 退化、MAE 略有改善，且仅保留约 34.30%/47.66% 的 CNM-SE 教师收益。本步只回答一个问题：这种失败主要来自高误差尾部放大、教师非受益区域漂移，还是教师修正方向仍未被摊销。

本步不训练模型、不修改 checkpoint、不调超参数、不扩 seed、不访问 test。所有结论均为开发集机制归因，不是泛化结论。

## 2. 冻结数据与模型资产

- 数据集：Traffic。
- seed：2027。
- split：chronological validation（按时间顺序划分的验证集）。
- batch size：16；完整 formal 必须处理 103 batch、1648 个样本。
- 输入/预测长度：192/96；变量数：862。
- 缺失数据、完整目标、固定 mask、归一化和 loader 顺序必须与 Step 22.27/22.30 相同。
- PG95 来源：`mipp_pg95_traffic__traffic__point__mr0.06__seed2027__20260820_174327`。
- BF-CMVD-v2：`bf_cmvd_v2_pg95_traffic__traffic__point__mr0.06__seed2027__20260820_212927`，复用 Step 22.27 已通过工件门的逐 token 缓存，不重新训练。
- PBA-CMVD：`pba_cmvd_pg95_traffic__traffic__point__mr0.06__seed2027__20260826_222702`。
- CNM-SE 教师：固定为 `0.5×original + 0.25×L5 + 0.25×L20`；额外移除率 0.15、block length 为 5/20、nested seed 为 2027。

PBA 诊断进程必须同时保留两个相互独立的模型对象：只读 PBA 学生与只读 PG95 来源模型。CNM-SE 的 original/L5/L20 均由来源模型生成，禁止用 PBA 学生生成会随学生变化的教师。

## 3. 配对张量与基本定义

对每个样本、预测时间步和全部 862 个变量，固定：

- `p0`：PG95 original-view 预测；
- `pBF`：BF-CMVD-v2 单视图学生预测；
- `pPBA`：PBA-CMVD 单视图学生预测；
- `p5/p20`：PG95 在严格等实际移除率 L5/L20 输入上的预测；
- `t = 0.5*p0 + 0.25*p5 + 0.25*p20`：CNM-SE 教师；
- `y`：完整 validation 目标，仅用于离线诊断；
- `BT`：教师相对 `p0` 在该预测时间 token 上同时降低 MSE 和 MAE 的集合。

修正向量定义为 `dPBA=pPBA-p0`、`dT=t-p0`。所有 token 损失先在 862 个变量上取均值，随后才进行区域、分位数和总指标统计。BF 与 PBA 缓存必须在 sample index、history/L5/L20 mask、`p0` 损失和教师损失上闭环对齐。

## 4. 三类预注册诊断

### 4.1 高误差尾部放大

用 BF-CMVD-v2 的 token MSE 分布定义固定参考尾部，阈值为其全 validation 90% 分位数；高于或等于阈值的 token 记为 `TailBF10`。不允许用 PBA 自身结果重新挑选尾部。

记录：

- PBA-BF 的聚合 MSE/MAE 差；
- `max(MSE_PBA-MSE_BF,0)` 在 `TailBF10` 内的份额；
- 尾部与主体区的 PBA-BF 平均 MSE/MAE 差；
- 两模型 token MSE 的 Q90/Q95/Q99 与固定尾部均值；
- 尾部内 PBA 修正与来源误差的有害同向程度。

只有同时满足以下条件才触发尾部放大：

1. PBA 聚合 MSE 严格差于 BF，而聚合 MAE 不差于 BF；
2. `TailBF10` 承担至少 60% 的正 MSE 超额；
3. `TailBF10` 内 PBA-BF 平均 MSE 差大于 0；
4. 非尾部主体区 PBA-BF 平均 MAE 差不大于 0。

### 4.2 教师非受益区域漂移

分别在 `BT` 与 `~BT` 统计 PBA 相对教师的正 MSE/MAE 差距、PBA 相对来源的平均损失变化，以及 PBA 修正能量。

只有同时满足以下条件才触发非受益区域漂移：

1. `~BT` 承担至少 60% 的 PBA—教师正 MSE 差距；
2. `~BT` 承担至少 60% 的 PBA—教师正 MAE 差距；
3. PBA 在 `~BT` 内相对来源的平均 MSE 和 MAE 均变差；
4. `~BT` 至少承载 25% 的 PBA 总修正能量。

### 4.3 教师方向欠摊销

在 `BT` 内统计 `dPBA` 与 `dT` 的 cosine similarity（余弦相似度）、projection coefficient（投影系数）与 normalized mismatch（归一化失配）。

只有同时满足以下条件才触发教师方向欠摊销：

1. `BT` 承担至少 60% 的 PBA—教师正 MSE 差距；
2. `BT` 承担至少 60% 的 PBA—教师正 MAE 差距；
3. 以下三项至少两项成立：余弦 `<0.50`、投影 `<0.50`、归一化失配 `>0.75`。

## 5. 冻结判定顺序与唯一后续映射

判定顺序固定，不能在看到结果后改变：

1. 任一协议/工件门失败：`PROTOCOL_FAILURE_NO_DECISION`，修复工程后重跑相同诊断。
2. 尾部放大门成立：`TAIL_RISK_AMPLIFICATION_FREEZE_CVAR_TR_PBA`。唯一候选为 CVaR-TR-PBA（Conditional Value-at-Risk Trust-Region PBA，条件风险价值信赖域部分主干摊销）：仅在训练期增加基于已观测目标、相对冻结来源的正 regret（后悔值）CVaR 尾部约束；不改解冻层、教师、主损失预算或部署路径。
3. 尾部门不成立且非受益漂移门成立：`NONBENEFICIAL_DRIFT_FREEZE_BTR_PBA`。唯一候选为 BTR-PBA（Benefit-region Trust-Region PBA，受益区域信赖域部分主干摊销）：只对训练期教师非受益 token 约束学生偏离来源；不加推理路由器。
4. 前两门不成立且教师方向欠摊销门成立：`TEACHER_DIRECTION_UNDER_AMORTIZED_FREEZE_HFD_PBA`。唯一候选为 HFD-PBA（Hierarchical Feature Distillation PBA，层级特征蒸馏部分主干摊销）：在相同末端层范围增加教师隐藏表征蒸馏；不增加解冻深度。
5. 其余或多个机制证据无法按上述优先级形成闭环：`MIXED_OR_INDETERMINATE_STOP_PBA_ROUTE`，停止 PBA 路线，不堆叠模块。

## 6. 工程与科学边界

- smoke 固定 3 batch；formal 固定完整 103 batch。
- 两模型全程 `eval`、`no_grad`，参数和 memory 哈希前后必须一致。
- 只保存逐 token 损失、修正内积/能量和 packed mask，不保存完整预测，避免不必要的大型工件。
- PBA/来源 checkpoint、SHA-256、resolved config、命令、环境、Git 状态、源码 diff、日志、资源记录必须落盘。
- Step 22.27 BF 缓存只在它自身 `artifact_status=PASS`、test 未访问且与本轮资产完全对齐时可复用。
- 本步不能宣布达到 SCI 二区水平；它只为下一次单因素验证提供机制证据。
