# 步骤22.41：最小距离标准化修复三seed复核

## 目标与完成判据

22.40 seed2027固定验证原始路径MSE（均方误差）降低2.6387%，MAE（平均绝对误差）降低3.0708%。本步只复核这一**基线错误修复**，不是增加创新模块。固定新增seed（随机种子）2028、2029，形成三seed描述性结果，不选择最好seed，不接触test（测试集）。

完成判据：CPU（中央处理器）配置与初始化检查通过；每seed训练、保存、重载和CRN（共同随机数）固定验证smoke test（极小链路检查）通过；同预算正式训练与1648窗口固定验证完成；逐seed和三seed均值、样本标准差、代价及负结果自动归档。三seed仅用于稳定性初步判断，不保证统计显著性或SCI（科学引文索引）分区。

## 唯一科学变化

直接继承22.37对应seed的S4M配置和既有基线。唯一启用变化为`fix_min_distance_normalization: false → true`：最小值距离改为用自身原始距离进行标准化。不开启其他模块，不修改最大值距离、衰减函数、极值提取、编码器维度、loss（损失函数）或评价实现。不增加参数，预算上限、batch size（批量大小）、optimizer（优化器）、scheduler（学习率调度器）、early stopping（早停）、数据划分和掩码资产均保持原值。实际早停轮数可以不同，必须如实记录。

2028/2029原始基线从`runtime/stage22_37_control/state.json`定位，固定模型、训练随机数、初始化哈希、采样顺序前缀和评估资产进行审核。22.40已通过默认关闭等价性与关闭模式GPU（图形处理器）链路检查，本步不浪费时间重复正式原始基线训练。2027的已完成修复结果完整保留并纳入三seed，不手工复制指标。

## 队列与资源

8个任务：每seed依次`train_smoke → eval_smoke → train_formal → eval_formal`。两个seed链相互独立，两张GPU都有资源时可以各跑一个实验；只有一张可用则自动排队。并发上限3，但同GPU不叠加本队列任务，避免把6GiB首次启动估计当作真实峰值保证，因此本轮两卡实际最多并行2个。该保护只管理本队列，不保证其他用户不会使用显存。训练首次准入6144MiB、验证4608MiB；既有训练历史峰值更大，保存在状态中。显存不足/OOM（显存溢出）由原调度器加间隔退避、增加估计并回队列，不修改批量、预算或协议，不囤积显存。

每个run（运行）保存最终配置、实际解析参数、命令、环境、Git（版本控制）状态/diff（修改差异）、源码快照、状态、日志、指标、checkpoint（模型检查点）和资源记录。当前训练入口未提供完整续训状态，OOM只允许原配置从头重跑，并明确留痕。磁盘不足1GiB停止而不删除任何成果。

启动一次并核验真实GPU进程及日志后，不持续人工监督训练；持久后台调度器自动排队、固定验证、汇总与归档。

## 命令与输出

准备：`/mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python scripts/stage22_41_distance_fix.py --prepare`

启动：`sh scripts/start_stage22_41.sh start`

状态：`sh scripts/start_stage22_41.sh status`

安全停止：`sh scripts/start_stage22_41.sh stop`；停止后恢复：`sh scripts/start_stage22_41.sh resume`。

查看日志：`tail -f logs/stage22_41_distance_fix/scheduler.log`。

结果：`reports/stage22_41_distance_fix.json`和`.md`；总表`reports/latest_results.csv`和`.md`；完整完成后按时间戳归档。主要比较使用固定验证`complete_target_mse_on`与`complete_target_mae_on`，继承的CNM（反事实嵌套掩码）多尺度路径只作次要结果，不把集成收益冒充修复收益。原论文Traffic超参数不可完全核验，本轮是固定本地基线比较而非声称完全同原论文协议。

## 决策

报告全部三seed的原始值、修复值、绝对变化、相对变化、双指标胜出次数、参数量、实际轮数/更新次数、耗时与显存。如果三个seed均双指标改善，列为修正版基线候选；否则先定位不一致原因，不自动增加模型模块或扩大实验。即使稳定改善，也归入bug fix（错误修复），不计入创新点。
