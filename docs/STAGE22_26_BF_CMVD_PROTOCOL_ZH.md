# 步骤22.26：BF-CMVD收益筛选因果多视图蒸馏协议

## 1. 方法定位

BF-CMVD全称Benefit-Filtered Causal Multi-View Distillation（收益筛选因果多视图蒸馏）。目标是把训练期CNM-SE（因果嵌套掩码自集成）的稳定增益蒸馏到推理期单视图学生，从而保留PG95级别的一次模型前向。

该方案在S4M+PG95上改进，不能脱离原模型独立运行。它复用既有MCPR（掩码条件帕累托残差）的零初始化低秩残差头作为学生，但训练目标和教师机制是新的；旧MCPR负结果仍保留为架构控制。

## 2. 固定教师

训练期教师使用冻结PG95 backbone（主干）和与Traffic验证完全一致的多尺度定义：

`teacher = 0.5*original_backbone + 0.25*L5_backbone + 0.25*L20_backbone`。

- L5/L20只对历史已观测行继续做因果嵌套遮挡；
- 每个样本严格移除原已观测行数的15%，使用NumPy四舍六入五成双规则；
- nested seed（嵌套随机种子）以`2027 + epoch*1000003`确定，每个epoch可复现但干预不同；
- 教师所有预测停止梯度，不更新backbone、prototype memory（原型记忆）或MIPP；
- 教师只能使用train输入与train已观测预测目标，不读取validation/test标签构造训练信号。

## 3. 收益筛选蒸馏

对每个train样本，只在当前固定教师相对原始PG95同时满足以下条件时启用蒸馏：

1. 教师在已观测预测目标上的样本MSE严格更低；
2. 教师在已观测预测目标上的样本MAE严格更低。

该二指标AND门只在训练期使用真实目标，推理期不存在oracle（先知）门。学生损失冻结为：

`L = MSE(student, target) + mean_beneficial(|student - teacher|)`。

若一个batch没有双指标受益样本，第二项严格为0。两个损失项权重均固定1.0，不搜索连续权重。直接MSE保证学生仍学习真实目标；仅对双指标受益样本的L1蒸馏避免复制教师局部退化，并显式照顾MAE。

## 4. 学生与推理

- PG95 backbone及prototype memory全程冻结；
- 学生为已有MCPR rank32、因果kernel size5的mask条件低秩残差头；
- 残差头零初始化，训练前严格复现PG95；
- optimizer（优化器）为Adam，adapter learning rate（适配器学习率）0.001；
- batch size16、最多15 epoch、patience3、validation MSE选checkpoint，与既有MCPR控制一致；
- 推理时不生成L5/L20、不计算教师、不使用训练期收益门，只执行original PG95和残差头一次前向；
- 新增参数、训练耗时、推理耗时和峰值显存必须落盘。

## 5. 实验顺序与门槛

1. 先做2 train batch、3 validation batch、2 epoch smoke，验证forward/backward/optimizer/checkpoint/reload/评估和教师门诊断；
2. smoke通过后在Traffic seed2027运行完整最多15 epoch适配，validation-only；
3. 主比较为同一source checkpoint的PG95 original、BF-CMVD student和冻结CNM-SE teacher；
4. BF-CMVD必须相对PG95 original严格改善validation MSE和MAE；
5. BF-CMVD必须在MSE与MAE上分别保留至少50%的CNM-SE相对PG95收益；
6. 训练受益样本比例必须位于开区间`(0,1)`，残差能量与梯度必须有限且非零；
7. seed2027门失败则不访问test，先做训练信号归因；门通过后冻结全部定义，在seed2028重新训练并只看validation；
8. seed2027/2028均通过后，才生成候选冻结后全新seed2029资产与checkpoint作预授权validation/test确认。

## 6. 创新性边界

当前创新由两点构成：训练期因果退化多视图作为privileged teacher（特权教师），以及双指标收益筛选蒸馏到单视图mask条件学生。概念上高于纯测试时集成和普通一致性训练，但只有在跨seed、跨数据集和强基线下保持精度—效率优势，才可能支撑SCI二区论证；本协议不预先承诺分区。
