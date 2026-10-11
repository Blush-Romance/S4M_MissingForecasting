# 步骤22.14：MRRA-v3验证集误差迁移诊断协议

## 1. 诊断目标

MRRA-v3已经解决数值发散，但在Electricity seed2023完整validation（验证集）上表现为MSE（Mean Squared Error，均方误差）退化、MAE（Mean Absolute Error，平均绝对误差）改善。本步骤只回答三个问题：

1. 误差迁移集中在哪些PG95基线难度分位、预测步和变量；
2. MRRA-v3对额外嵌套缺失的prediction response（预测响应）能否区分受益和受损样本；
3. PG95与MRRA-v3是否存在足够的逐样本互补性，值得设计下一版风险约束残差门。

本步骤不训练模型、不读取test（测试集）、不根据结果修改已有checkpoint，也不把oracle（使用真实标签的理想选择器）当作可部署方法。

## 2. 冻结资产

- 数据集：Electricity；
- seed（随机种子）：2023；
- split（数据划分）：chronological validation（按时间顺序的验证集）；
- 样本数：2528，batch size（批大小）16，共158个batch；
- 输入长度192，预测长度96，变量数321；
- 固定缺失CSV、完整CSV和mask（掩码）资产与步骤21/22一致；
- PG95 checkpoint：步骤21冻结正式checkpoint；
- MRRA-v3 checkpoint：步骤22.13冻结正式checkpoint；
- 响应探针：从每个样本现有观测历史中等计数额外移除15%，块长L=5，seed=2023；
- 原始和嵌套视图使用同一模型且不更新模型或prototype memory（原型记忆库）。

## 3. 运行顺序

1. MRRA-v3两batch smoke test（冒烟测试）；
2. smoke通过后运行完整2528样本validation缓存；
3. 与既有PG95缓存按sample index（样本索引）、history mask（历史掩码）、nested mask（嵌套掩码）严格配对；
4. 自动回算MSE/MAE并核对source metrics（来源指标）；
5. 输出逐样本、预测步、变量、难度十分位、moving-block bootstrap（移动块自助法）和响应分层结果。

任何mask、样本顺序、回算指标、checkpoint有限性或validation-only检查失败，都停止科学结论。

## 4. 预注册分析

所有gain（收益）均定义为`PG95误差 - MRRA-v3误差`，正值表示MRRA-v3更好。

### 4.1 无参数配对分析

- 逐样本MSE/MAE同时改善、仅MSE改善、仅MAE改善、均未改善的比例；
- 按PG95逐样本MSE排序的10个等样本数难度分位；
- 96个预测步及321个变量上MSE/MAE改善比例；
- 对逐样本MSE/MAE gain做块长288、重复2000次的moving-block bootstrap；
- 计算MRRA-v3嵌套响应与MSE/MAE gain的Spearman rank correlation（斯皮尔曼秩相关）。

### 4.2 互补性上界

分别报告逐样本MSE oracle与MAE oracle。oracle只量化两个checkpoint是否存在互补空间，不用于声称方法收益。

### 4.3 可观测响应路由审计

唯一可用信号是MRRA-v3在原始与固定嵌套视图间的预测分歧MSE，不使用真实误差作为输入。时间顺序固定为：

- fit（拟合段）：样本0–1263；
- embargo（隔离窗）：1264–1455；
- development（开发段）：1456–1887；
- embargo：1888–2079；
- audit（最终审计段）：2080–2527。

在fit段枚举响应的0%、5%、…、100%分位阈值及“低响应选择MRRA-v3/高响应选择MRRA-v3”两种方向。候选必须在fit段相对PG95双指标均不退化且至少一项严格改善；再用development段同一双指标规则选择归一化MSE+MAE最小的候选。audit段只评估一次，不再改阈值。

只有audit段MSE和MAE相对PG95均不退化且至少一项严格改善，才认为该可观测信号足以支持下一版门控；否则放弃“仅靠嵌套响应门控”的方向。

## 5. 后续决策

- 若可观测响应路由通过audit双指标门：冻结默认关闭的风险约束MRRA门，并先做同预算单seed快速训练对照；
- 若oracle有互补而响应路由失败：说明需要更接近MRRA内部残差/不确定性的可观测特征，先增加只读特征缓存，不直接训练复杂路由器；
- 若oracle也不能双指标支配PG95：停止MRRA精度分支，保留v3作为稳定性消融，转向新的损失或主干机制假设；
- 任一情形都不访问test，也不直接把RADR与MRRA-v3组合。
