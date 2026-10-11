# 步骤22.21：Traffic外部验证协议（结果前冻结）

## 1. 研究问题与边界

本步骤检验当前项目唯一获得三seed稳定正证据的主线——PG95（跨掩码原型净化）+ CNM-SE（Causal Nested-Mask Self-Ensembling，因果嵌套掩码自集成）——能否从Weather外推到全新Traffic数据。

边界固定为：

- 不继续ARD/MCPR变体；Step22.20已决定`ARD_ROUTE_STOP_NO_CAPACITY_EXPANSION`；
- Traffic未参与方法、权重、块长或门限选择；
- 先且只使用validation（验证集）决定是否允许一次test（测试集）；
- validation门失败时，test任务只写入`SKIPPED_BY_VALIDATION_GATE`审计工件，禁止加载test数据；
- 本轮是单seed方向性外部验证，不能单独支撑统计稳定性或SCI分区声称。

## 2. 官方可还原性限制

S4M公开README只提供Weather训练命令，没有公布Traffic的缺失mask、随机seed或最终超参。公开代码只明确将文件名中的`traffic`映射为862个变量。

因此本协议标记为：

```text
fixed_fair_external_protocol_official_traffic_hparams_unavailable
```

它是S4M与PG95之间严格公平的外部比较，不冒充论文未公开的Traffic官方结果复现。

## 3. 数据与固定缺失资产

Traffic完整CSV审计：

- 17544个小时时间点；
- 862个数值变量；
- 时间范围：2016-07-01 02:00:00至2018-07-02 01:00:00；
- 数值缺失值0，重复时间0；
- 完整资产SHA256：`cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16`。

固定缺失协议：

- 机制：point-missing（点式起点触发的连续块缺失）；
- 名义起点率：0.06；
- 块长：5；
- 目标实际缺失行率：0.26；
- 固定生成seed：2026；
- 同一缺失行对862变量全部设为缺失，与现有Weather/ETTh1/Electricity协议一致；
- 生成后必须保存CSV、NPZ mask、JSON元数据、哈希和manifest记录；
- 在任何GPU实验前验证观测值与完整CSV一致、缺失数与mask一致、seed重生逐值一致。

## 4. 数据划分和正式样本数

`Dataset_Custom4`的时间划分固定为0.7/0.1/0.2，不打乱。

- 输入长度：192；
- 预测长度：96；
- validation原始滑窗：1661；`batch_size=16, drop_last=true`后为103 batch / 1648样本；
- test原始滑窗：3413；`drop_last=true`后为213 batch / 3408样本；
- 目标列显式固定为数据中真实存在的最后一列`OT`；其余861列为`0`至`860`，总变量数仍为862。

### 4.1 首次smoke后的实现性修订（无模型结果）

2026-08-16首次两个训练smoke都在构造DataLoader（数据加载器）时、任何GPU前向和指标生成之前退出。错误暴露出最初schema文字审计把“862个数值变量”误写成列名`0`至`861`；真实表头是`0`至`860`加最后一列`OT`。本修订只把不存在的target（目标列）`861`改为真实最后列`OT`，不改变变量数量、列顺序、CSV、mask、数据划分、模型、训练预算、多尺度权重或科学门。两个失败run永久保留为`CODE_ERROR`，修复后生成新experiment ID并重跑smoke。

## 5. S4M与PG95公平训练协议

两者共享：

- seed 2026；
- `features=M`、`freq=h`；
- 862输入/输出变量；
- 8层S4M，`d_model=d_ff=en_conv_hidden_size=512`；
- batch size 16；
- Adam，learning rate 0.001（见5.1节的结果后数值稳定性修订）；
- 最多15 epoch，patience 3，同一scheduler（学习率调度器）和checkpoint选择标准；
- 完整训练不设batch/validation上限；
- 训练后只评估validation；
- 同一固定CSV/mask/完整目标/归一化逻辑/指标实现。

S4M关闭全部创新开关。PG95只相对S4M打开已冻结MIPP（Mask-Intervention Prototype Purification，掩码干预原型净化）：

- `use_mipp=true`；
- 嵌套配对掩码块长`[1,5,20]`，额外缺失率区间`[0.15,0.50]`；
- 不变性损失权0.05，其余MIPP辅助损失权0；
- 稳定写入门0.95，absolute（绝对阈值）写入模式；
- 禁止FUSR、CFR、MRRA、MCPR、TTRR、ARD等已停止或不在本主线的模块。

smoke test（冒烟测试）仅用于验证链路：原始smoke为2 epoch、每epoch 2个训练batch、3个validation batch；5.1节修订后改为1 epoch、最多64个训练batch和3个validation batch，以覆盖原NaN失败位置。必须覆盖forward/backward/optimizer step/checkpoint保存与加载/metrics落盘；不作科学结论。

### 5.1 正式训练NaN后的数值稳定性修订

2026-08-16首轮两项formal（正式训练）均使用Adam学习率0.01。S4M在epoch1/batch27、PG95在epoch1/batch33出现`forecast_loss=nan`，后续validation和test均未运行。全量无GPU审计确认完整真值、缺失输入的观测部分、归一化数组、填充值、mask和全部训练目标窗口均有效，排除了数据非有限、空目标集合和零尺度归一化。

公开仓库没有Traffic专属训练超参；其参数解析器默认学习率为0.001。用户于2026-08-16明确授权执行下一步后，本协议进行单因素、双臂对称修订：S4M与PG95的Adam学习率共同由0.01改为0.001。模型结构、862变量、固定缺失资产、batch size 16、正式15 epoch、patience 3、scheduler、checkpoint选择、validation门、多尺度权重和test边界全部不变。这是数值稳定性修订，不根据任一方法的精度选择学习率，也不构成PG95专属调参。

修订后的两项formal必须先依赖各自1 epoch/64 train-batch扩展稳定性smoke成功；该smoke必须越过此前batch27/batch33失败点并完成至少63个optimizer step（第一个batch用于memory warmup）。旧0.01失败run永久保留，新运行使用新experiment ID。

### 5.2 validation后分析器一致性修复

两项lr0.001正式训练和多尺度validation均成功后，首次科学门报告只有`frozen_full_training_budget`为false。只读核验确认训练配置实际仍为batch size 16、Adam、最多15 epoch、patience 3、完整batch和validation预算；失败来自分析器遗留的旧硬编码`learning_rate == 0.01`，而不是实验配置漂移。该错误导致test被安全跳过，测试集未访问。

步骤22.21.7只把分析器预算检查同步为已授权的lr0.001，并同时要求配置显式包含5.1节的修订身份；新增负例必须拒绝旧lr0.01、缺失修订身份和训练轮次漂移。修复不重训模型、不改checkpoint、不改validation结果、多尺度权重或科学精度门；门重新通过后只允许执行原预注册的唯一一次test。

## 6. 冻结多尺度CNM-SE

仅对正式PG95 validation checkpoint做单次冻结评估：

- 每样本目标额外缺失率：0.15；
- 块长：L1/L5/L20；
- 三个嵌套视图必须严格移除相同数量的历史观测元素；
- nested seed：2026；
- 最终多尺度预测：`0.5*original + 0.25*L5 + 0.25*L20`；
- L1只作等实际缺失率结构审计，不进入最终融合；
- 不更新模型、原型memory或任何统计；
- 保存original/L1/L5/L20/三种双路/多尺度的指标、逐样本缓存、实际移除数和推理时间。

## 7. validation结果前门禁

### 7.1 PG95门

PG95相对同预算S4M：

- validation MSE不劣；
- validation MAE不劣；
- MSE/MAE至少一项严格更优。

### 7.2 多尺度门

多尺度CNM-SE必须同时：

- MSE和MAE均严格优于PG95 original；
- MSE和MAE均严格优于`0.5*original+0.5*L5`；
- 视图间预测差异不是纯浮点噪声：至少一个L5/L20的预测差异MSE与original预测MSE之比严格大于`1e-10`。这个数值只用于拒绝ETTh1/Electricity已观察到的逐位等价/数值噪声，不是精度改善阈值。

### 7.3 test授权

只有协议审计、PG95门、多尺度门和视图响应门全部通过，才授权同一seed2026 checkpoint的唯一一次完整test。test不得用于修改权重、块长、训练超参或方法结构。

## 8. 决策标签

- `TRAFFIC_PROTOCOL_FAIL_TEST_FORBIDDEN`：工件或公平性失败；
- `TRAFFIC_PG95_FAIL_TEST_FORBIDDEN`：PG95门失败；
- `TRAFFIC_VIEW_COLLAPSE_TEST_FORBIDDEN`：视图响应仅数值噪声；
- `TRAFFIC_MULTISCALE_FAIL_TEST_FORBIDDEN`：多尺度精度门失败；
- `TRAFFIC_VALIDATION_PASS_ONE_TEST_AUTHORIZED`：所有门通过，允许唯一test；
- 如果唯一test也双指标改善，再冻结seed2027/2028资产并扩展多seed；否则保留单seed负结果，不在Traffic test调参。

## 9. 完成判据

- 协议时间早于Traffic缺失资产和任何模型结果；
- 资产结构/哈希/重生/观测值审计通过；
- S4M/PG95各自smoke后才放行formal；
- formal保存唯一ID、resolved config、代码diff、checkpoint、指标、时间、显存和状态；
- validation多尺度只读评估与源PG95指标回算一致；
- 条件test执行或安全跳过均有工件；
- 自动报告明确`test_accessed`、决策和所有门。
