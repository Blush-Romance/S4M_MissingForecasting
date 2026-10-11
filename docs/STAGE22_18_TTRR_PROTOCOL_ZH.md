# 步骤22.18：TTRR-MCPR 尾部感知信赖域残差预注册协议

## 1. 本步目的

步骤22.17表明，`MCPR-MSE（Mask-Conditioned Pareto Residual with Mean Squared Error，采用均方误差训练的掩码条件帕累托残差）`并非整体方向错误：53.006%的验证样本同时改善MSE和MAE，残差与主干误差的总体交叉项也有益；失败来自少量样本的残差幅度过冲，最差5%样本贡献了80.121%的净MSE损害。相反，`MCPR-PCD（Pareto Common Descent，帕累托公共下降）`的残差方向与幅度都失效，因此本步停止PCD路线。

本步提出`TTRR-MCPR（Tail-aware Trust-Region Residual for MCPR，面向MCPR的尾部感知信赖域残差）`。它只沿用MSE残差路线，并同时解决两个问题：

1. 推理时如何仅凭历史可观测信息限制逐样本残差幅度；
2. 训练时如何直接惩罚候选模型相对冻结主干造成的少数极端MSE损害。

## 2. 数据隔离与禁止事项

- 主数据仍为Electricity，完整数据、chronological train/validation/test切分、归一化、输入长度192和预测长度96保持不变。
- 新的确认资产固定为point missing、名义起点率0.06、块长5、mask seed 2024；资产必须在训练前生成并保存CSV、NPZ、metadata和哈希。
- seed2024资产此前没有用于步骤22.17失败归因。MCPR-MSE控制组和TTRR-MCPR必须读取同一份seed2024缺失CSV、固定mask和完整目标。
- 由于seed2024只改变缺失掩码、没有改变Electricity的时间区间，本步属于“交叉掩码快速确认”，不能表述为独立数据集或独立时间外部验证。
- test仍被禁止访问；训练选择和结论只使用train与chronological validation。
- 步骤22.17基于validation标签算出的oracle缩放系数0.219131只作失败机理上界，禁止作为初始化、固定超参数、监督标签或报告中的可部署结果。
- 不复活PCD，不在看到本轮结果后修改尾部比例、损失权重、门特征或训练预算。

## 3. 冻结模型定义

### 3.1 可观测信赖门

MCPR先产生原始残差`r_raw`。TTRR使用只来自历史输入、历史缺失掩码、冻结主干预测和当前隐藏状态的八维描述子：

1. 四个既有mask描述子的时间均值；
2. 隐状态均方根；
3. 原始残差均方根；
4. 冻结主干预测均方根；
5. 冻结主干预测一阶时间差的均方根。

描述子在进入门控层前停止梯度，防止上游网络通过篡改统计量规避约束。一个八维到一维的线性层加Sigmoid函数产生逐样本门值`g_i∈(0,1)`：

```text
g_i = sigmoid(W · stop_gradient(f_i) + b)
r_i = g_i · r_raw_i
y_candidate_i = y_backbone_i + r_i
```

`W`和`b`均初始化为0，因此初始门值严格为0.5；该数值不来自任何validation标签。门仅限制残差幅度，不改变残差方向。关闭`use_ttrr`时，必须恢复原MCPR行为，不能残留额外参数、损失或数据处理副作用。

### 3.2 尾部非劣训练约束

对每个含有效目标的训练样本，计算候选预测与冻结主干预测的样本MSE：

```text
c_i = MSE(y_candidate_i, y_i)
b_i = MSE(y_backbone_i, y_i)
delta_i = c_i - b_i
h_i = ReLU(delta_i)
```

令`q=0.10`，在当前batch有效样本中取`k=max(1, ceil(q·N))`个最大的`h_i`，定义尾部条件风险：

```text
L_tail = mean(top_k(h, k))
L_total = L_mse(candidate, target) + lambda · L_tail
```

其中`lambda=1.0`。两项量纲均为MSE，不引入MAE目标。该约束与此前RADR不同：RADR按样本绝对难度重加权并混合MSE/MAE，而TTRR只惩罚“候选相对冻结主干新增的正向伤害”，目标直接对应步骤22.17暴露的少数样本过冲。

### 3.3 训练参数与开关

- `use_mcpr=true`
- `mcpr_objective=tail_trust_mse`
- `use_ttrr=true`
- `ttrr_tail_fraction=0.10`
- `ttrr_tail_weight=1.0`
- MCPR rank=32、kernel size=5
- 冻结PG95主干、memory和其他既有模块；只更新MCPR及其TTRR门参数
- Adam，MCPR学习率0.001，batch size 16，最多15 epoch，patience 3
- 模型选择仍按完整chronological validation普通MSE，不以尾部损失或test选择checkpoint

非法组合必须在启动前拒绝：TTRR未配MCPR、TTRR未配`tail_trust_mse`、尾部比例越界、权重非正或PCD与TTRR混用。

## 4. 公平对照与执行顺序

冻结来源checkpoint为步骤21的Electricity seed2023 PG95正式checkpoint。新缺失资产只改变输入mask和训练seed；两个可训练候选都从同一PG95 checkpoint warm start。

执行顺序：

1. 生成并审计seed2024固定缺失资产；确认与seed2023在train和validation均不同，且缺失结构、观测值保持、哈希和实际缺失率全部通过。
2. 运行零GPU组件测试、配置负测和基线关闭等价回归。
3. 分别运行MCPR-MSE seed2024控制组与TTRR-MCPR的smoke；smoke必须覆盖forward、backward、optimizer step、checkpoint保存/加载、metrics写入和有限值检查。
4. 只有各自smoke成功，才提交相应15 epoch完整validation正式实验。
5. 另用冻结PG95在seed2024 validation上做只读锚点评估；不训练、不访问test。
6. 所有正式结果自动汇总，不手工抄写指标。

除方法开关与其必要参数外，MCPR-MSE和TTRR-MCPR必须保持同数据、训练seed、batch、epoch、early stopping、优化器、学习率、模型规模、checkpoint选择规则和评估实现。

## 5. 必须记录的机制与成本指标

- validation MSE、MAE、样本数和batch数；
- 门值mean/min/max/std；
- 原始残差与应用后残差的能量；
- 训练期尾部CVaR、候选/主干样本MSE差和正伤害样本比例；
- 总参数量、新增参数量；
- 总训练时间、单epoch时间、推理时间和峰值显存；
- checkpoint全部张量有限、可重载；
- 完整resolved config、命令、环境、Git状态、代码diff和固定资产哈希；
- `test_split_accessed=false`。

## 6. 预注册判断门

### 6.1 工程与协议门

任一项不通过则结论为协议失败，不做科学判断：固定资产不一致、smoke失败、checkpoint不可重载或含非有限值、验证样本不完整、训练预算不公平、测试集被访问、主干未冻结、TTRR未真实执行。

### 6.2 快速科学继续门

在协议全部通过后，TTRR-MCPR相对seed2024 MCPR-MSE控制组必须满足：

- MSE不退化；
- MAE不退化；
- 至少一个指标严格改善。

同时相对seed2024冻结PG95锚点也必须MSE/MAE双非劣且至少一项严格改善，才视为完整快速门通过。

若仅击败MCPR-MSE但未击败PG95，结论只能是“过冲治理机制有作用但完整方法仍失败”，不得进入test或多seed。

### 6.3 停止与后续

- 任一主指标退化：停止当前TTRR配置，不根据本轮validation反复搜索`q`、`lambda`或门初始化。
- 双锚点门通过：下一步先在未参与本轮设计的另一数据集或未见时间协议上进行validation-only外部确认；通过后才讨论多seed、消融和一次性test。
- 本步无论结果如何都不直接声称达到SCI二区水平；只有跨数据集/多seed稳定性、强基线、复杂度和完整消融齐全后才能重新评估投稿水平。

