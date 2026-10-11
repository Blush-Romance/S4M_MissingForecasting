# 步骤22：MRRA与RADR验证集快速筛选协议

## 1. 研究动机

步骤21与步骤22的冻结证据显示两个不同但相关的失效：

1. S4M和PG95在Electricity验证集上，对额外移除相同数量历史观测的L=5嵌套掩码几乎没有预测响应。预测分歧MSE与预测误差MSE之比均约为`10^-15`，属于数值舍入量级。
2. PG95相对S4M改善MSE（均方误差）`0.634936%`，但MAE（平均绝对误差）退化`0.199360%`。96个预测步的MSE全部改善，而MAE全部退化；低误差样本主要退化，高误差样本主要改善。

因此步骤22不再搜索推理融合权重或块长，而是在PG95上分别验证“掩码响应表示”和“难度分层风险优化”。

## 2. 创新一：MRRA

MRRA全称为Mask-Responsive Residual Adapter（掩码响应残差适配器）。

给定衰减插补历史`z`与观测掩码`m`，MRRA先对`[z,m]`做低秩投影，再使用只沿历史方向进行左填充的深度可分离时间卷积提取响应表示：

```text
h = GELU(W_in [z,m])
h_temporal = GELU(h + DWConv_causal(h))
r_context = W_context h_temporal
```

`W_context`使用零初始化，因此模块开启后的初始模型严格从PG95行为开始，而不是随机破坏主干。`r_context`加到原型融合后的S4上下文中。

MRRA同时使用共享低秩表示重建历史值：

```text
x_reconstructed = z + W_reconstruct h_temporal
```

重建损失只计算训练历史中人工缺失的位置，以`seq_true_x`作为历史监督；不读取预测未来，更不读取验证或测试标签。首轮固定：低秩维度32、时间卷积核5、历史重建权重0.1，不搜索参数。

## 3. 创新二：RADR

RADR全称为Rank-Adaptive Dual Risk（秩自适应双风险）。它不增加模型参数，只替换训练期主预测损失。

对每个batch（批次）内样本，用停止梯度的逐样本MSE生成`[0,1]`难度百分位秩`q_i`。困难样本更多优化MSE，容易样本更多优化MAE：

```text
L_tail = weighted_mean(q_i, sample_MSE_i / stopgrad(mean_MSE))
L_body = weighted_mean(1-q_i, sample_MAE_i / stopgrad(mean_MAE))
L_RADR = 0.5 * stopgrad(mean_MSE) * (L_tail + L_body)
```

该定义没有人工误差阈值；MSE/MAE先按batch统计量归一化，再恢复到MSE量级，避免单纯因数值尺度改变学习率。难度排序只使用训练目标的已观测位置。

## 4. 首轮实验身份

- 冻结锚点：步骤21 Electricity PG95，seed2023。
- 候选A：`PG95 + MRRA`，RADR关闭。
- 候选B：`PG95 + RADR`，MRRA关闭。
- 两个候选不在首轮组合，确保收益可归因。
- 数据、固定mask、训练/验证划分、batch size 16、Adam、学习率0.01、15 epoch上限、patience 3、模型层数和评价实现均与PG95一致。
- 新配置先运行2 epoch、每epoch 2个训练batch的独立smoke test（冒烟测试），覆盖前向、反向、优化器更新、checkpoint保存/加载、validation评估和metrics写入。
- smoke通过后，两个完整validation-only（仅验证集）formal可按显存并行运行。

## 5. 冻结门控

每个候选相对PG95独立判断：

1. 工件、固定资产和同预算协议全部通过；
2. validation MSE与MAE均不得退化；
3. 至少一个指标严格改善；
4. MRRA必须真实执行历史重建并产生有限非零残差；RADR必须真实执行难度秩与双风险项；
5. 不设置人为数值提升阈值，不挑选seed或验证子区间。

只有单模块通过上述门控，才允许继续做重复seed或与另一模块组合。步骤22不授权访问Electricity test（测试集）。

## 6. 缺陷隔离

官方S4M源码把`min_idx`归一化写成了使用`max_idx`。这是上游实现缺陷，步骤22首轮保持PG95原行为以保证单因素比较；后续若验证该修复，必须使用独立bug-fix（缺陷修复）实验身份，并重新运行基线，不能计入MRRA或RADR创新收益。
