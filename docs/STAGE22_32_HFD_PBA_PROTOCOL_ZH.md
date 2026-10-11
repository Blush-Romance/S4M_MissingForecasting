# Step 22.32 HFD-PBA 结果前冻结协议

## 1. 方法定位与唯一假设

HFD-PBA（Hierarchical Feature Distillation Partial-Backbone Amortization，层级特征蒸馏部分主干摊销）只检验一个由 Step 22.31 触发的假设：PBA-CMVD 已经开放足够的非线性末端容量，但仅在输出层接受教师监督，因而没有把冻结多视图教师的修正方向充分摊销到末端时序表征中。

本轮保留 PBA-CMVD 的数据、来源 PG95、固定 CNM-SE 教师、严格双指标获益 token、可训练范围、优化器、学习率、batch size、epoch、early stopping（提前停止）、checkpoint 选择和单视图部署路径。唯一新增变量是在训练期对 S4 第 6/7 层的隐藏修正进行层级蒸馏。

本轮不增加解冻层数、adapter（适配器）、对比学习、可靠性门、尺度路由、额外数据增强、可学习教师权重或推理分支。若失败，不能在同一 seed2027 validation 上调整层、损失尺度或训练预算。

## 2. 冻结模型与特征层

- 学生仍由 PG95 来源 checkpoint 初始化，只训练 `s4_layers.6`、`s4_layers.7`、`norm` 和 `out_proj`，训练参数必须仍为 `4,127,582`。
- 教师仍是来源 PG95 的独立冻结副本；教师和学生不得共享参数存储，教师参数、完整状态和 prototype memory（原型记忆）在训练前后必须保持哈希不变。
- 仅捕获每个 S4 层输出、LayerNorm（层归一化）之前的第 6/7 层隐藏张量；层编号使用代码的零基索引 `6,7`。
- 隐藏捕获必须默认关闭，不注册参数或 buffer（缓冲），不得改变关闭状态下的前向数值与 state dict（状态字典）。
- validation 和部署关闭隐藏捕获；每个样本仍只有一次原始视图学生前向。冻结教师和三路视图只存在于训练阶段。

## 3. 固定教师与隐藏修正目标

对每个训练 batch，冻结教师分别计算：

- `h0_l`：原始历史视图在第 `l` 层的隐藏表征；
- `h5_l`：等计数、额外缺失率 0.15、因果块长 5 的嵌套视图隐藏表征；
- `h20_l`：相同协议下因果块长 20 的嵌套视图隐藏表征。

输出教师保持：

`p_teacher = 0.5*p0 + 0.25*p5 + 0.25*p20`。

第 `l` 层的教师隐藏表征与隐藏修正固定为：

`h_teacher_l = 0.5*h0_l + 0.25*h5_l + 0.25*h20_l`

`delta_teacher_l = h_teacher_l - h0_l = 0.25*(h5_l-h0_l) + 0.25*(h20_l-h0_l)`。

学生只接收原始视图，其隐藏修正定义为：

`delta_student_l = h_student_l - stopgrad(h0_l)`。

其中 `stopgrad` 表示停止梯度。所有教师输出、隐藏表征和获益判定都必须 detach（切断梯度）。隐藏监督只使用最后 `pred_len=96` 个预测时间 token，不读取未来真值来构造特征。

## 4. 冻结获益集合与 HFD 损失

严格获益集合完全复用 PBA-CMVD/BF-CMVD-v2：在训练集已观测预测目标上，只有当固定 CNM-SE 教师相对冻结 `p0` 的 token 级 MSE 和 MAE 都严格更小时，该 token 才进入蒸馏。

对层 `l in {6,7}`，在获益 token 上定义能量归一化隐藏修正误差：

`R_l = sum ||delta_student_l-delta_teacher_l||^2 / (sum ||delta_teacher_l||^2 + epsilon)`。

求和覆盖获益 token 和全部隐藏通道；`epsilon` 只用于防止零除，固定为该张量 dtype（数据类型）的机器精度。若 batch 没有获益 token，`R_l` 返回带学生计算图的零，不产生额外更新。

两层平均：

`R_hfd = 0.5*(R_6+R_7)`。

为了不引入可调超参数，HFD 使用 output-loss matched scaling（输出损失匹配尺度）：

`L_hfd = stopgrad(L_distill) * R_hfd`。

因此新增项的初始量级由同一 batch 的既有输出蒸馏损失自动决定，但尺度本身不接收梯度，也不改变获益集合。总损失固定为：

`L_total = L_direct + L_distill + L_hfd`。

不得在看到 seed2027 结果后加入系数、clip（裁剪）、逐层权重或改成单指标获益筛选。

## 5. 公平预算与开关

- 新开关 `use_hfd_pba=false` 时必须严格走原 PBA-CMVD 路径；`true` 时要求 `use_pba_cmvd=true`。
- 层集合固定 `[6,7]`，损失模式固定 `correction_energy_normalized_mse`，尺度模式固定 `output_distill_detached`；配置不符必须在启动前报 `CONFIG_ERROR`。
- HFD 与 MCPR、DMRB、MRRA、FUSR、CFR、RADR 和所有诊断/test 模式互斥。
- Adam、learning rate=`0.001`、batch size=`16`、最多 `15` epoch、patience=`3`、`type1` scheduler 与 PBA-CMVD 完全一致。
- smoke 只允许 2 epoch、每 epoch 2 个训练 batch、3 个 validation batch，并明确标记 `__smoke__`；formal 使用全部训练 batch 和完整 103 个 chronological validation batch。
- 仍使用 `pba_delta_v1` 增量 checkpoint，因为可训练键和部署结构完全没有变化；resolved config 和 metrics 必须显式记录 HFD 开关、层、损失模式、特征统计与梯度统计。
- test loader 不得构造，test 指标不得读取。

## 6. 工程门

正式排队前必须全部通过：

1. 默认关闭前向、损失、参数键和 state dict 与原 PBA 回归一致；
2. 第 6/7 层学生、来源和 L5/L20 教师特征形状完整，学生特征有梯度，全部教师特征无梯度；
3. 合成正例能得到正有限 HFD 损失和第 6/7 层梯度；无获益 token 得到带图零；错误层、形状和非有限输入被拒绝；
4. HFD 训练时只有预注册参数变化，冻结学生参数、教师和 memory 哈希不变；第 6/7 层各自至少出现一次非零有限 HFD 梯度；
5. `pba_delta_v1` 保存、加载和来源哈希错误拒绝仍通过，加载后单视图 prediction（预测）往返一致；
6. validation 教师调用数为 0、学生主干调用数为每 batch 1，隐藏捕获关闭；test loader 未构造；
7. smoke 的 forward（前向）、backward（反向传播）、optimizer step（优化器更新）、checkpoint、重载评估、metrics、status 和 resource usage 全部落盘。

任一工程门失败都保留失败 experiment ID，修复后以新 ID 重做 smoke；不得直接放行 formal。

## 7. seed2027 开发验证门

固定参考：PG95=`0.523471994/0.318781597`，BF-CMVD-v2=`0.520750225/0.317637980`，PBA-CMVD=`0.521529998/0.317547226`，CNM-SE teacher=`0.517810102/0.316191875`，顺序均为 MSE/MAE。

HFD-PBA 只有在以下条件全部满足时通过：

1. 所有工程、协议、样本数、固定资产、指标回算和 validation-only 工件检查通过；
2. MSE 和 MAE 均严格优于 BF-CMVD-v2，并且均严格优于原 PBA-CMVD；
3. 相对 PG95 到 CNM-SE 教师的两项收益保留率均至少 60%，即 MSE 不高于约 `0.520075`、MAE 不高于约 `0.317228`；
4. HFD 在两个隐藏层均实际激活，teacher correction（教师修正）能量、HFD loss（损失）和对应训练梯度均为正且有限；
5. 参数量、训练时间、单路 validation 推理时间和峰值显存完整落盘。

seed2027 已参与方法设计，因此它只是开发筛选，不是独立验证。

## 8. 决策与停止边界

- 全部门通过：`HFD_PBA_SEED2028_VALIDATION_CONFIRMATION_AUTHORIZED`。下一步只能原样转移到未用于本轮设计的 Traffic seed2028 validation 独立确认，仍不得访问 test。
- 精度或 60% 门失败但工程门通过：`HFD_PBA_SEED2027_STOP`。停止 HFD/PBA 路线，不在同一 validation 调层、调权重、加对比学习或扩训练预算。
- 工程门失败：`HFD_PBA_PROTOCOL_FAILURE`。只修复工程问题，以新 experiment ID 重跑相同冻结配置，不得据失败精度改方法。

本步即使通过也不能单独支撑 SCI 二区结论；独立 seed、更多缺失率/形态、跨数据集、强基线、消融、复杂度和统计检验必须在开发门通过后再补。
