# 步骤22.20：ARD信息瓶颈只读诊断协议

## 1. 目标与边界

本步骤只诊断ARD-MCPR（Analytic Residual Decision Distillation，解析残差决策蒸馏）为什么从高方差三动作教师退化为近常数学生门。诊断不更新模型、优化器或原型记忆，不访问test，不补ARD-v1多seed，也不把同一seed2025 validation上的离线探针收益作为正式模型提升。

需要区分两个互斥性不强的故障源：

1. `FEATURE_INFORMATION_BOTTLENECK`（特征信息瓶颈）：现有8个全局均值/RMS特征丢失了时序和变量结构，无法预测OFF/PARTIAL/FULL动作；
2. `MIXTURE_CANCELLATION`（混合抵消）：动作概率或partial系数已经随样本变化，但`p(FULL)+p(PARTIAL)×partial_alpha`把变化抵消为近常数门。

## 2. 固定模型与数据

- 模型：步骤22.19唯一ARD-MCPR seed2025 formal checkpoint；
- 数据：Electricity point-r0.06-L5 seed2025固定缺失资产；
- split：chronological validation only；
- batch size：16，`shuffle=false`，`drop_last=true`；
- smoke：前3 batch，共48样本；
- formal：完整158 batch，共2528样本；
- 不进行第二次嵌套mask前向；
- 使用冻结checkpoint单次前向，不更新模型或memory。

## 3. 固定缓存字段

每个样本缓存：

- `sample_index`：validation时间顺序索引；
- 当前ARD的8维`global8`特征；
- 108维`structured108`无标签结构特征；
- 三动作logits、概率、partial系数和最终gate；
- 完整目标解析alpha/action；
- 训练式“仅观测目标”解析alpha/action及有效标记；
- 冻结PG95主干、raw residual预测和ARD预测的逐样本SSE/SAE；
- 解析反事实所需的`alignment_cross=2Σ(error_backbone×raw_residual)`、`residual_energy=Σ(raw_residual²)`和元素数。

所有结构特征只能来自推理时可观测量，禁止使用目标、误差或解析教师作为输入。

## 4. `structured108`定义

前8维严格复用当前ARD特征。其余100维固定如下：

1. 8个预测时间块的raw residual RMS：8维；
2. 8个预测时间块的冻结主干预测RMS：8维；
3. 8个预测时间块的raw residual—主干预测余弦：8维；
4. 8个预测时间块的raw residual一阶差分RMS：8维；
5. 8个预测时间块的主干预测一阶差分RMS：8维；
6. 历史mask的8时间块×4个MCPR mask统计：32维；
7. MCPR低秩latent的8时间块RMS：8维；
8. 按变量计算的raw residual RMS、主干预测RMS、二者比值和二者时间余弦，各取q10/q25/q50/q75/q90：20维。

总维度：`8+40+32+8+20=108`。分块必须整除固定预测长度96和历史长度192；数值分母统一加`1e-8`。

## 5. 解析教师

对每个样本：

```text
alpha* = clip(-alignment_cross / (2 × residual_energy), 0, 1)
alignment_cross = 2 × Σ[(backbone-target) × raw_residual]
```

完整目标教师使用全部完整validation目标，只用于失败归因。训练式教师只在原缺失资产中可观测的预测目标位置计算，用于量化“训练教师—完整目标教师”偏差。能量不超过`1e-8`的样本标为无效。

动作阈值固定为：`alpha≤1e-6`为OFF，`alpha≥1-1e-6`为FULL，其余为PARTIAL。

## 6. 时间隔离分析

完整2528样本按时间固定划分：

- fit：`[0,1264)`；
- embargo-1：`[1264,1456)`，不使用；
- development：`[1456,1888)`；
- embargo-2：`[1888,2080)`，不使用；
- audit：`[2080,2528)`，只作一次冻结复核。

固定探针：

- 三动作：标准化后多项Logistic Regression（逻辑回归），`C=1`、无class weight、`max_iter=2000`、seed 2220；
- 连续alpha：标准化后Ridge（岭回归），`alpha=1`，预测裁剪到`[0,1]`；
- 比较`global8`和`structured108`，不搜索超参数。

报告balanced accuracy（平衡准确率）、cross-entropy（交叉熵）、alpha Spearman相关、alpha MAE，以及由预测alpha和解析二次式计算的反事实MSE。

## 7. 结果前冻结的判据

### 7.1 特征信息瓶颈

当且仅当`structured108`相对`global8`同时满足：

- development和audit动作balanced accuracy均至少提高0.05；
- development和audit alpha Spearman均至少提高0.10；
- development和audit反事实MSE均不劣于`global8`；

判定`FEATURE_INFORMATION_BOTTLENECK_CONFIRMED`。

### 7.2 混合抵消

当且仅当完整validation同时满足：

- gate标准差不超过0.01；
- `p(FULL)`或`p(PARTIAL)×partial_alpha`至少一项标准差不低于0.01；
- 两项贡献的Pearson相关不高于-0.5；

判定`MIXTURE_CANCELLATION_CONFIRMED`。

### 7.3 下一结构

- 两项均确认：ARD-v2采用结构化token编码器和离散动作执行，不再使用连续概率期望混合；
- 仅特征瓶颈：ARD-v2只替换结构化特征编码器，保留混合形式；
- 仅混合抵消：ARD-v2保留特征，改为离散/直通动作执行；
- 两项均未确认：停止ARD路线，不凭同一validation增加网络容量。

ARD-v2如被支持，必须在实现前冻结结构和训练协议，并使用新的seed2026固定缺失资产做首次正式确认。

## 8. 完成判据

- CPU静态/公式/字段测试通过；
- smoke通过后才运行formal；
- formal为2528样本/158 batch且与步骤22.19 ARD MSE/MAE回算一致；
- checkpoint含ARD参数且所有张量有限；
- 模型、主干和memory无更新；
- 自动报告明确给出两个故障判定和下一结构；
- `test_accessed=false`。
