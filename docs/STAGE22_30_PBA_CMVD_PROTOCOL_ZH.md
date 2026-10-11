# Step 22.30：PBA-CMVD 结果前冻结协议

冻结时间：2026-08-26。本文档在实现 PBA-CMVD 且运行任何 Step 22.30 GPU 实验之前写入。

## 1. 问题与前置证据

Step22.29 已排除 DMRB（Dual-scale Mask-conditioned Residual Basis，双尺度掩码条件残差基）的两个优先替代解释：联合获益监督与分支获益的条件重叠为 `0.808580/0.778921`，未达到监督错配门；完整双分支组合在 MSE（Mean Squared Error，均方误差）和 MAE（Mean Absolute Error，平均绝对误差）上均优于两个单分支，未发生固定求和冲突。

真正触发的是双分支表达不足：L5/L20 学生残差对各自教师贡献的余弦仅 `0.077425/0.077294`、投影系数仅 `0.034052/0.026085`，归一化失配为 `1.125322/1.061720`。因此本步停止预测输出端的低秩残差路线，不增加 rank（秩）、深度、分支掩码或路由器。

本步只检验一个新假设：CNM-SE（Causal Nested-Mask Self-Ensemble，因果嵌套掩码自集成）教师修正需要改变学生的非线性时序动力表示，而不是由预测后的低秩头近似；训练学生末端两层 S4（Structured State Space Sequence Model，结构化状态空间序列模型）及输出投影，可以把训练期三视图教师摊销为推理期单视图模型。

## 2. 唯一方法变量

方法命名为 PBA-CMVD（Partial-Backbone Amortized Causal Multi-View Distillation，部分主干摊销因果多视图蒸馏）。

学生和教师都从同一个 Traffic seed2027 PG95 checkpoint 恢复。二者必须是两个独立模型对象：

- 教师永久冻结并保持 `eval/no_grad`，用原始视图、L5 与 L20 嵌套缺失视图生成固定 CNM-SE 预测；
- 学生推理只接收原始 PG95 单视图，不实例化或调用 MCPR（Mask-Conditioned Pareto Residual，掩码条件帕累托残差）/DMRB；
- 学生只开放 `s4_layers[6]`、`s4_layers[7]`、最终 `LayerNorm`（层归一化）和 `out_proj`（输出投影）；`e_layers` 固定为8，因此前六层 S4、输入投影、衰减插补、原型记忆及全部其他参数冻结；
- 末端两层加归一化和输出投影预计有 `4,127,582` 个可训练参数，约占来源模型状态规模的13.46%；运行时必须由 `named_parameters()` 重新精确核验。

本轮不加入 latent contrastive learning（潜表示对比学习）、可靠性门、尺度路由、额外 adapter（适配器）、新 teacher 权重或新数据增强。这样结果只回答“把修正位置从低秩输出头移入末端非线性动力层是否有效”。

## 3. 冻结教师与因果边界

训练期每个 batch 的冻结教师定义为：

`p_teacher = 0.5*p0 + 0.25*p5 + 0.25*p20`。

其中 `p0` 是冻结 PG95 教师对学生同一原始输入的预测，`p5/p20` 由完整历史 `seq_true_x` 乘以从当前观测 mask 继续删除得到的等计数嵌套 mask 后预测。额外缺失率固定0.15，block lengths 固定 `[5,20]`，nested seed 固定2027，epoch seed stride 固定1000003。嵌套视图只能删除当前已观测历史，不能恢复当前缺失值、读取未来目标或访问 test。

教师三路输出和获益判定全部 `detach`（切断梯度）。每个优化步前后都必须核验教师参数、教师状态和 prototype memory（原型记忆）未改变；学生与教师不得共享参数存储。

## 4. 冻结训练目标

在训练集中存在观测目标的每个预测时间 token（时间单元）上，计算冻结教师和冻结 `p0` 跨变量的 MSE/MAE。严格获益条件保持 BF-CMVD-v2 不变：

`beneficial = (MSE(p_teacher)<MSE(p0)) AND (MAE(p_teacher)<MAE(p0))`。

学生预测记为 `p_student`。损失固定为：

- `L_direct`：所有观测预测目标上的学生 MSE；
- `L_distill`：严格获益 token 内、观测变量上的 `|p_student-p_teacher|` 均值；若当前 batch 无获益 token，则为带学生计算图的0；
- `L_total = L_direct + L_distill`，两项权重均为1。

损失、筛选粒度和权重与 BF-CMVD-v2 相同；唯一变化是可训练函数族。不得在看到 seed2027 validation 后修改权重、改用单指标筛选或增加训练 epoch。

## 5. 公平预算、评估与 checkpoint

- 数据、训练/validation 划分、固定缺失资产、归一化、PG95 来源 checkpoint 及 SHA-256 与 Step22.26-v2/22.28 相同。
- Adam、learning rate=0.001、batch size=16、最多15 epoch、patience=3、`type1` scheduler（学习率调度器）和 validation MSE 选 checkpoint 保持相同。
- formal 使用全部训练 batch 和103个 chronological validation batch；smoke 只用2 epoch、每epoch 2个训练 batch、3个 validation batch，只验证链路，不能作为精度证据。
- 推理路径只有一个学生 PG95 前向；冻结教师只存在于训练，不进入 validation 部署路径、参数量或推理时间。
- 使用 `pba_delta_v1` 可还原增量 checkpoint：保存预注册末端层的全部参数/必要 buffer（缓冲）、来源绝对路径、来源 SHA-256、层范围和键清单；原子写入。加载时先验证并恢复来源模型与 memory，再叠加学生变化状态。
- 每个实验保存唯一 ID、resolved config、command、env、git state、code diff、关键源码快照、metrics、status、resource usage 和 checkpoint；结果进入自动汇总表。

## 6. 开关与工程门

- `use_pba_cmvd=false` 时走既有训练/评估路径，不能冻结或开放额外参数；PG95、MCPR 和 DMRB 行为保持不变。
- `use_pba_cmvd=true` 必须与 MCPR、DMRB、MRRA、FUSR、CFR、RADR 及所有 test/诊断模式互斥。
- CPU/静态测试必须验证：精确训练参数键、冻结参数键、教师/学生存储不共享、教师输出无梯度、获益规则和损失合约、配置负例、`pba_delta_v1` 正反向往返、来源 SHA-256 错误拒绝、原基线回归。
- GPU smoke 必须验证 forward、backward、optimizer step、非零且有限梯度、checkpoint 保存/加载、validation、metrics 落盘；冻结部分/教师/memory 哈希不变；训练中教师获益和蒸馏项至少各出现一次；validation 不调用教师；test loader 未构造。
- 任一确定性工程错误先保留失败 run，修复后用新 experiment ID 重做 smoke；工程门未全过不得启动 formal。

## 7. seed2027 validation 探索门

固定参考为：PG95=`0.523471994/0.318781597`，BF-CMVD-v2=`0.520750225/0.317637980`，DMRB=`0.521787606/0.317986323`，CNM-SE teacher=`0.517810102/0.316191875`，顺序均为 MSE/MAE。

PBA-CMVD 只在以下条件全部满足时通过：

1. 所有协议、来源、冻结哈希、教师隔离、checkpoint、样本数、指标回算和 validation-only 工件检查通过，test 未访问；
2. MSE 和 MAE 均严格优于当前最佳单视图学生 BF-CMVD-v2；
3. 相对 PG95 到 CNM-SE 教师的 MSE 与 MAE 收益保留率均至少60%，等价阈值约为 MSE≤`0.520075`、MAE≤`0.317228`；
4. 所有预注册末端模块都获得至少一次非零有限梯度，冻结参数、教师与 memory 不变；
5. 参数量、训练时间、单路 validation 推理时间和峰值显存完整落盘。

虽然 PBA-CMVD 的可训练参数显著多于低秩头，但不增加部署模型总参数和推理前向次数；60%门仍要求其明显超过BF-CMVD-v2，而不能仅凭“训练成功”继续扩展。

## 8. 停止与下一步

- formal 未同时满足双指标与60%收益保留：`PBA_CMVD_SEED2027_STOP`，不扩seed、不访问test，也不在同一validation上尝试1层/3层、不同学习率或不同损失权重。
- formal 全通过：`PBA_CMVD_SEED2028_VALIDATION_CONFIRMATION_AUTHORIZED`，下一步使用结果前固定的同一结构在Traffic seed2028 validation独立确认；仍不访问test。
- 只有独立validation确认后，才允许把第二创新点冻结为表示层一致性/对比蒸馏，并必须先证明它不是重复增加训练容量；不得在PBA-CMVD失败时用第二模块掩盖第一模块无效。
