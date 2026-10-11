# 步骤20：ETTh1完整PG95+多尺度CNM-SE事后外部验证协议

## 1. 研究问题

步骤19已经证明：把冻结的多尺度CNM-SE直接接到步骤5原始S4M的ETTh1 checkpoint上，预测几乎不发生变化。该checkpoint此前也表现出原型分支数值失活，因此步骤19不能回答“经过PG95训练后，完整方法能否在ETTh1上有效”。

步骤20检验一个在步骤19结果之后提出的post-hoc（事后）新假设：

> PG95的跨掩码配对一致性训练可能恢复模型对缺失扰动的有效响应；在此基础上，步骤18冻结的等实际缺失率多尺度CNM-SE可能再次获得互补集成收益。

本步骤不会改写步骤19的`EXTERNAL_TRANSFER_FAIL`结论，也不能被表述为步骤19预注册验证成功。

## 2. 方法身份

完整方法只包含以下两个已经定义的模块：

1. **PG95**：Pair-Gate-only 0.95（仅配对一致性与0.95稳定性写入门）。它属于MIPP（Mask-Intervention Prototype Purification，掩码干预原型净化）的紧凑版本：
   - 开启跨掩码配对视图；
   - 配对不变性损失权重固定为0.05；
   - 关闭内容—缺失因子化及其全部辅助损失；
   - 使用绝对稳定性写入门，阈值固定为0.95；
   - 不启用BMSA、SDRA、MRBA、FUSR或CFR。
2. **多尺度CNM-SE**：Causal Nested-Missing Self-Ensemble（因果嵌套缺失自集成）：
   - 仅从原历史中已经观测的时间行继续移除；
   - L=1、L=5、L=20逐样本移除完全相同数量的观测行；
   - 固定使用原始视图、L5视图和L20视图；
   - 融合权重固定为`0.5×原始 + 0.25×L5 + 0.25×L20`；
   - 不新增可学习参数，不重新训练，不搜索融合权重。

## 3. 数据与训练协议

- 数据集：ETTh1。
- 随机种子：2023。
- 缺失类型：point missing（点缺失），名义起点率0.06，连续块长5。
- 固定缺失资产：
  - `data/ETT-small/ETTh1_point_paper_r0.06_l5_seed2023.csv`
  - `artifacts/masks/etth1/point/rate_0.06/seed_2023.npz`
- 实际落盘缺失率：0.2308840413。
- 划分：0.7/0.1/0.2时序划分。
- 输入长度192，预测长度96，7个变量。
- batch size（批大小）16。
- Adam优化器，学习率0.005。
- 最多15个epoch（训练轮次），patience（早停耐心）3。
- 训练、验证、测试均使用完整batch预算；不继承smoke缩减设置。
- 数据、划分、归一化、seed、优化器、early stopping（早停）、checkpoint选择和评价代码与步骤5 ETTh1基线保持一致。
- PG95模块超参数完全继承Weather上已经冻结的PG95，不依据ETTh1测试结果修改。

原论文没有公开缺失mask随机种子，且步骤5 ETTh1基线MSE与论文表值偏差较大。因此本步骤只能称为“项目内固定资产、同协议的外部方法验证”，不能称为严格复现论文原始mask协议。

## 4. 执行顺序

统一后台队列严格按以下依赖执行：

1. ETTh1 PG95两epoch、每epoch两训练batch的GPU smoke；
2. smoke成功后运行完整PG95正式训练；
3. 从正式run动态读取唯一的resolved config、metrics和最佳checkpoint；
4. 对该checkpoint运行3个test batch的多尺度评估smoke；
5. smoke成功后进行一次211-batch完整test。

评估配置在正式checkpoint产生后自动物化；不能人工选择不同checkpoint或依据中间测试结果修改结构。

## 5. 冻结比较与判定

所有判定均同时考察MSE（Mean Squared Error，均方误差）与MAE（Mean Absolute Error，平均绝对误差），不设置人为提升幅度阈值：

1. **PG95训练收益**：ETTh1 PG95正式MSE和MAE是否都低于步骤5 ETTh1 S4M基线。
2. **CNM-SE同checkpoint收益**：多尺度输出的MSE和MAE是否都低于同一次评估的PG95原始视图。
3. **多尺度增量收益**：多尺度输出的MSE和MAE是否都低于同一次评估的L5双视图`0.5×原始+0.5×L5`。

决策标签：

- `FULL_METHOD_DIRECTIONAL_PASS`：三项比较全部双指标改善。
- `PG95_ONLY_PASS`：PG95优于S4M，但多尺度没有同时优于PG95原始路。
- `CNM_SE_ON_PG95_ONLY_PASS`：多尺度优于同run PG95原始路，但PG95本身没有同时优于步骤5 S4M。
- `MULTISCALE_INCREMENT_FAIL`：PG95和CNM-SE主比较通过，但多尺度没有同时优于L5双视图。
- `FULL_METHOD_FAIL`：PG95与多尺度主比较均未通过。

单seed正结果只表示方向性证据，不能声称统计稳定或已经达到SCI二区要求。若通过，下一阶段应补独立seed、更多外部数据协议、强基线和训练/推理复杂度；若失败，则停止把PG95+CNM-SE包装为跨数据集完整方法，并回到validation提出新假设。

## 6. 工件与公平性门

每个任务必须保存唯一experiment ID、解析后配置、命令、环境、Git状态与diff、日志、metrics、状态、资源使用和checkpoint或source checkpoint哈希。

正式结论还要求：

- PG95 checkpoint包含MIPP参数且不包含FUSR/CFR路径；
- source原始PG95指标能在冻结评估的original视图上以`1e-5`容差复现；
- L=1/L=5/L=20逐样本移除数严格相等；
- 三种nested mask（嵌套掩码）均为原mask子集；
- 固定权重、完整test、样本数、指标roundtrip（回算一致性）和串行计时全部通过；
- 上游`upstream/S4M`保持洁净。
