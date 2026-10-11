# 步骤22.19：ARD-MCPR解析残差决策蒸馏协议

## 1. 研究问题

步骤22.17和22.18共同显示，MCPR-MSE（均方误差训练的掩码条件后主干残差）不是始终无效，而是对缺失掩码和时间区间高度敏感：seed2023上MSE退化，seed2024上MSE与MAE同时改善。TTRR-MCPR（尾部感知信赖域残差）使用单个连续Sigmoid门和尾部伤害损失，最终把平均门压到0.194，仅保留约2.2%的原始残差能量，几乎退回PG95。

步骤22.19不再继续调整TTRR尾部比例、损失权重或门初始化。新问题是：能否从训练样本中解析得到“应关闭、部分保留还是完整保留残差”的最优动作，并把这个动作蒸馏到只依赖输入和模型内部量的推理模块？

## 2. 既有证据与设计边界

只使用步骤22.17已经落盘的seed2023 validation缓存做发现性诊断。对冻结PG95预测`b`、MCPR原始残差`r`和目标`y`，缩放后预测为`b + αr`。逐样本MSE关于`α`的解析最优解为：

```text
alignment = 2 * sum((b - y) * r)
energy = sum(r^2)
alpha_teacher = clip(-alignment / (2 * energy + epsilon), 0, 1)
```

既有缓存中教师动作明显两极化：约41.50%为`α=0`，约47.51%为`α=1`，只有约11.00%位于中间。该结构否定“所有样本使用相近连续收缩”的假设，支持离散动作与连续缩放联合建模。

发现性诊断还显示，额外反事实嵌套视图虽然与教师动作相关，但在时间隔离audit上没有优于单视图混合门，且动作分类准确率很低。因此首轮确证模型不增加第二次前向，不把额外计算包装成创新收益。

## 3. 方法定义

方法命名为`ARD-MCPR（Analytic Residual Decision Distillation，解析残差决策蒸馏）`。

### 3.1 创新点一：训练期解析教师

教师系数只使用训练批次中可用的观测目标和目标掩码计算，不读取validation或test标签。教师、主干预测、原始残差和输入特征在教师分支全部停止梯度。

教师把每个有效样本分成三个动作：

- `OFF（关闭）`：`α <= 1e-6`；
- `PARTIAL（部分保留）`：`1e-6 < α < 1 - 1e-6`；
- `FULL（完整保留）`：`α >= 1 - 1e-6`。

解析教师不是validation oracle（验证集神谕系数）：它在训练过程中逐批计算，推理时不需要目标。

### 3.2 创新点二：离散—连续混合推理门

ARD不再使用TTRR的单层线性Sigmoid门。推理模块接收与TTRR相同的八个无标签特征：

1. 历史观测比例；
2. 掩码转移比例；
3. 缺失年龄；
4. 全局观测比例；
5. 预测隐藏状态RMS（均方根）；
6. 原始残差RMS；
7. PG95主干预测RMS；
8. 主干预测时间差RMS。

特征进入固定宽度16的两层MLP（多层感知机），输出三个动作logit和一个部分缩放logit。最终可微缩放为：

```text
p_off, p_partial, p_full = softmax(action_logits)
alpha_partial = sigmoid(partial_logit)
alpha_student = p_full + p_partial * alpha_partial
prediction = backbone + alpha_student * raw_residual
```

八维到16维的特征投影使用Xavier初始化，两个输出头零初始化；因此初始三个动作概率均为1/3、部分系数为0.5，`alpha_student=0.5`。模块关闭时不得实例化ARD参数，必须恢复原MCPR行为。

### 3.3 训练目标

PG95主干和原型memory保持冻结。MCPR原始残差仍用与控制组相同的观测目标MSE训练：

```text
L_raw = MSE(backbone + raw_residual, observed_target)
L_action = CrossEntropy(action_logits, teacher_action)
L_scale = SmoothL1(alpha_student, alpha_teacher)
L_total = L_raw + L_action + L_scale
```

`L_raw`只更新原MCPR残差参数；`L_action + L_scale`只更新ARD参数。教师和八维特征停止梯度，避免蒸馏损失反向改变原始残差以迎合教师。

原始残差从零初始化开始，因此第1个epoch只训练`L_raw`，ARD保持初始状态；从第2个epoch开始，只有残差能量有限且大于`1e-8`的样本参与教师蒸馏。这是启动稳定性规则，不根据结果调整。

## 4. 固定资产与公平比较

在本协议写入后生成Electricity point-r0.06-L5 seed2025固定缺失资产。seed2025不得参与上述设计、步骤22.19.1诊断或阈值选择。

首轮包含三条方法：

1. 冻结PG95锚点，只读评估seed2025 validation；
2. MCPR-MSE控制，从同一PG95 checkpoint warm start，在seed2025上训练；
3. ARD-MCPR候选，从同一PG95 checkpoint warm start，在seed2025上训练。

MCPR-MSE与ARD-MCPR保持相同：

- 数据划分、固定缺失资产和评估mask；
- seed=2025；
- batch size=16；
- 15 epoch、patience=3；
- Adam优化器；
- MCPR rank=32、kernel size=5、learning rate=0.001；
- 完整训练和完整chronological validation；
- 同一checkpoint选择规则和评价代码。

ARD只增加固定的小型决策头和蒸馏损失，不增加第二个输入视图，不改变主干、输入长度、预测长度、batch、epoch或数据量。

## 5. Smoke与正式实验

PG95、MCPR-MSE、ARD-MCPR分别先运行smoke。ARD smoke必须额外验证：

- epoch1不执行蒸馏更新；
- 后续epoch存在有效解析教师样本；
- 三类动作计数和学生门均有限；
- 学生门位于`[0,1]`；
- 原始残差和应用残差均执行；
- checkpoint含ARD参数且可重载；
- PG95主干和memory哈希前后一致；
- 默认关闭模式不含ARD参数并与MCPR-MSE前向等价。

smoke通过后，使用UUID动态队列运行三项formal validation。允许最大并发5，但实际并发由显存和安全余量决定；同一训练任务不得缩减。

## 6. 继续门和停止门

ARD必须同时通过两个锚点：

1. 相对同seed MCPR-MSE控制，MSE和MAE均不退化，且至少一项严格改善；
2. 相对同seed冻结PG95，MSE和MAE均不退化，且至少一项严格改善。

机制门同时要求：

- 蒸馏有效step大于0；
- 教师至少出现两个动作类别；
- 学生门标准差大于0.01；
- 主干与memory保持冻结；
- checkpoint全部有限；
- test未访问。

若精度双锚点门失败：停止该固定配置，不在seed2025 validation上搜索MLP宽度、loss权重、动作阈值或warmup长度，不进入test和多seed。

若只胜PG95但未胜MCPR-MSE：解析门没有证明优于原始残差，停止ARD。

若同时通过精度门和机制门：下一步先补seed2023/2024的严格复核和同GPU复杂度，再决定是否进入其他数据集；不能直接访问test。

## 7. 结果解释限制

seed2025仍是同一Electricity时间区间上的新缺失掩码，只能作为cross-mask（跨掩码）快速确认，不是独立时间或独立数据集外部验证。任何二区/三区水平判断必须等待多seed、更多数据集、强基线、统计稳定性和复杂度实验完成。
