# Step 22.28：DMRB 结果前冻结协议

冻结时间：2026-08-21。本文档在实现 DMRB 且运行任何 Step 22.28 GPU 实验之前写入。

## 1. 问题与证据

BF-CMVD-v2（Benefit-Filtered Causal Multiview Distillation，收益筛选的因果多视图蒸馏）在 Traffic seed2027 validation 上相对 PG95 同时改善 MSE（Mean Squared Error，均方误差）0.5199% 和 MAE（Mean Absolute Error，平均绝对误差）0.3587%，但只保留 CNM-SE（Causal Nested-Mask Self-Ensemble，因果嵌套掩码自集成）教师 MSE/MAE 收益的 48.07%/44.16%。

Step22.27 在同一固定 validation 上显示：教师获益区贡献教师—学生正差距的 97.4354%/94.3480%；学生与教师修正余弦仅0.200284，投影系数仅0.072097，归一化失配为0.985388。因此本步只检验一个假设：单一低秩残差头混合了 L5/L20 尺度不同的修正；用尺度分解的残差基可提升单前向学生对教师收益的保留。

## 2. 唯一方法变量

DMRB（Dual-scale Mask-conditioned Residual Basis，双尺度掩码条件残差基）共享一次冻结 PG95/S4 主干前向得到隐表示 `h` 和原始预测 `p0`，然后用两个参数不共享、结构相同的低秩掩码条件残差基输出 `r5` 和 `r20`：

`p_student = p0 + r5 + r20`。

固定教师不变：

`p_teacher = 0.5*p0 + 0.25*p5 + 0.25*p20`。

由此冻结两个分支的精确修正目标：

- `c5 = 0.25*(p5-p0)`；
- `c20 = 0.25*(p20-p0)`。

它们满足 `p_teacher = p0 + c5 + c20`。不使用可学习融合权重，不增加 CBR（Causal Benefit Router，因果收益路由器）、TTRR（Tail-Aware Trust Region Residual，尾部感知信任域残差）、ARD（Analytic Residual Decision，解析残差决策）、可观测性重加权、新教师权重或新训练数据。

## 3. 冻结训练目标

教师获益 token（预测时间单元）继续使用 BF-CMVD-v2 的严格双指标规则：在该 token 存在观测目标，且教师的跨变量 MSE 和 MAE 都严格小于 `p0` 时才获益。

直接损失与 v2 相同：所有观测预测目标上的学生 MSE。尺度分解蒸馏损失固定为：

`L_basis = mean_beneficial_token(mean_observed_variable(|r5-c5| + |r20-c20|))`。

总损失为 `L = L_direct_MSE + L_basis`，两项系数均为1。不得依据 seed2027 结果调整系数。`p0/p5/p20/c5/c20` 全部 detach（切断梯度），只允许 DMRB 参数更新。

## 4. 公平预算和资产

- 数据：Traffic seed2027 固定 point-r0.06-L5 缺失 CSV/掩码/归一化与 chronological train/validation 划分。
- 来源：与 BF-CMVD-v2 完全相同的 PG95 checkpoint 及 SHA-256。
- 教师：additional missing rate=0.15，block lengths=[5,20]，nested seed=2027，epoch seed stride=1000003，权重0.5/0.25/0.25。
- 优化：Adam，adapter learning rate=0.001，batch=16，最多15 epoch，patience=3，相同 scheduler。
- 每个分支rank=32、causal kernel=5，两分支共90,364个可训练参数；相对 v2 的45,182个参数扩容必须显式报告，不得将更大容量隐藏为同参数比较。
- 主干、prototype memory（原型记忆）和教师均冻结；推理只有一次 PG95 主干调用加两个轻量残差基，不调用 L5/L20 教师路径。
- smoke 仅2 epoch、每epoch 2个训练batch、3个validation batch，只用于链路与机制验证；formal 恢复上述全部预算。
- 因启动前文件系统仅余约227 MiB，而旧的每个完整 checkpoint 约118 MiB，DMRB 显式使用 `mcpr_delta_v1` 增量 checkpoint：只保存 DMRB 参数、冻结 PG95 来源路径和 SHA-256；加载时必须先校验并恢复来源，再叠加 adapter。这只改变无损存储形式，不改模型、优化、评估或可还原性。

## 5. 开关、基线和工程门

- `use_dmrb=false` 时不实例化或不调用 DMRB，原 MCPR/PG95 行为不变。
- DMRB 必须有 L5 和 L20 独立开关；本步 formal 两者都开启，单分支只为后续消融预留。
- CPU 合约必须证明：零初始化恢复主干预测；`c5+c20` 在冻结的 `32×float32 epsilon≈3.8147e-6` 工程容差内回算教师修正；两分支参数不共享；分支损失梯度到达对应基；教师/主干无梯度；独立关闭正确；checkpoint 保存加载往返等价。该容差只处理数学等价式的FP32运算顺序，不改任何科学阈值。
- smoke 必须产生两分支非零蒸馏损失、非零梯度和非零残差；证明获益/非获益 token 都存在、checkpoint 可加载、主干和 memory 哈希不变、validation-only 且 test 未访问，才能放行 formal。
- 磁盘必须能同时容纳新 smoke/formal 的必需 checkpoint 和工件。空间不足时进入 `WAITING_STORAGE`，不删旧成果、不禁用 checkpoint 合约。
- 增量 checkpoint 的 CPU 合约必须证明：来源 SHA-256 校验通过，来源主干/memory 和新 adapter 能完整重建，L5/L20 两分支键都存在，且不重复写入完整来源权重。

## 6. seed2027 validation 探索门

以固定指标作四方比较：PG95=`0.523472011/0.318781644`，CNM-SE teacher=`0.517810106/0.316191852`，BF-CMVD-v2=`0.520750225/0.317637980`，指标顺序均为 MSE/MAE。

DMRB 只在以下条件全部满足时通过：

1. 所有协议、资产、冻结哈希、checkpoint、样本数、指标回算和 validation-only 检查通过，test 未访问。
2. DMRB 的 MSE 和 MAE 均严格优于 BF-CMVD-v2，因而也均优于 PG95。
3. DMRB 对 CNM-SE 教师收益的 MSE 和 MAE 保留率都至少60%。等价地，MSE 必须不高于约0.520075，MAE 必须不高于约0.317228。
4. 两分支都真实激活：各至少一次非零梯度、非零残差能量和非零分支蒸馏损失；主干/memory 哈希不变。
5. 参数量、训练总时间和峰值显存完整落盘，推理路径证明只有一次主干前向。

60% 门高于v2的50% 最低门，是因为 DMRB 把 adapter 参数翻倍；新容量必须换来足够大的机制性改善，才值得扩大实验。

## 7. 停止与后续边界

- 任一工程/协议门失败：修复确定性问题后以新 experiment ID 重做 smoke，不改科学配置。
- smoke 机制门失败：formal 在 GPU 前阻断。
- formal 任一精度门失败：决策为 `DMRB_SEED2027_STOP_DIAGNOSE`，不扩 seed，不访问 test，不在同一结果上调参追门。
- formal 全通过：决策为 `DMRB_SEED2028_VALIDATION_CONFIRMATION_AUTHORIZED`。因 seed2027 validation 已被用于 Step22.27 归因与 DMRB 设计，它只是开发证据；必须在固定 seed2028 validation 做独立确认，本步不访问 test。
