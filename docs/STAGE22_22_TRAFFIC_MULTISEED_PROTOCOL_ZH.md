# 步骤22.22：Traffic三seed统计复核协议（结果前冻结）

## 1. 研究问题

步骤22.21已在Traffic seed2026上得到一次validation（验证集）与一次test（测试集）的同向结果。本步骤只回答：在完全相同的方法、训练预算、缺失机制和评价规则下，PG95以及PG95+多尺度CNM-SE（Causal Nested-Mask Self-Ensembling，因果嵌套掩码自集成）的收益能否在seed2027/2028复现，并形成seed2026/2027/2028的统计稳定性证据。

本步骤不是超参数搜索，不引入新模块，也不依据新结果修改方法。

## 2. 预先冻结的纳入与测试规则

- 固定纳入seed：2026、2027、2028；任何seed不得因结果不佳被删除。
- seed2027/2028的validation和test均在本协议中预先授权，全部运行；test不再作为调参门，而是冻结方法的重复验证。
- 不根据seed2027的结果决定是否运行seed2028，也不根据validation结果选择性访问test。
- seed2026复用步骤22.21的S4M/PG95训练checkpoint、PG95 validation和PG95/CNM-SE test，只新增一个S4M只读test评估。
- seed2027/2028分别从各自固定缺失资产训练S4M与PG95，并执行PG95 validation、S4M test和PG95多尺度test。
- 所有成功或失败实验都进入总表；确定性错误修复后使用新experiment ID，旧run永久保留。

## 3. 固定数据与缺失资产

完整Traffic资产保持不变：

- 17544个时间点、862个数值变量；
- 完整CSV SHA256：`cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16`；
- 目标列`OT`，其余列为`0`至`860`；
- 时间划分固定为0.7/0.1/0.2，输入长度192、预测长度96；
- validation为103 batch / 1648样本，test为213 batch / 3408样本（batch size 16、drop_last=true）。

seed2027/2028资产必须使用与seed2026相同的生成算法：

- point-missing（点式起点触发的连续块缺失）；
- 名义起点率0.06；
- 块长L=5；
- 目标实际缺失行率0.26；
- 同一缺失行对862变量全部设为缺失；
- 生成seed分别为2027和2028；
- 保存缺失CSV、NPZ mask、JSON元数据、SHA256和资产manifest（清单）；
- GPU运行前验证完整CSV哈希、seed逐值重生、观测值不变、缺失值位置、跨seed mask不同、validation/test覆盖和实际缺失率。

## 4. 冻结训练协议

S4M与PG95共享：

- 8层、`d_model=d_ff=en_conv_hidden_size=512`、862输入/输出变量；
- batch size 16；
- Adam优化器，learning rate（学习率）0.001；
- 最多15 epoch（训练轮次）、patience（早停耐心）3；
- 相同scheduler（学习率调度器）、checkpoint选择、归一化、训练数据和指标实现；
- 正式训练不限制batch或validation batch；
- seed同时作为训练随机seed和该次固定资产seed。

S4M关闭全部创新开关。PG95只打开步骤22.21已冻结的MIPP（Mask-Intervention Prototype Purification，掩码干预原型净化）：

- 嵌套配对块长`[1,5,20]`；
- 额外缺失率区间`[0.15,0.50]`；
- 不变性损失权0.05，其余MIPP辅助损失权0；
- 稳定写入门0.95，absolute（绝对阈值）模式；
- FUSR、CFR、MRRA、MCPR、TTRR、ARD等模块全部关闭。

每个新训练配置必须先通过独立smoke test（冒烟测试）：1 epoch、最多2个训练batch、3个validation batch，仅验证forward（前向）、backward（反向）、optimizer step（优化器更新）、checkpoint保存/加载和metrics（指标）落盘。正式配置不得继承smoke的batch上限或缩短预算。

步骤22.21已用64-batch稳定性smoke证明lr0.001跨越旧NaN位置；步骤22.22不重复把每个seed的smoke扩大到64 batch，因为训练代码和数值协议未变。若新seed的正式训练出现NaN，按`NAN_OR_DIVERGENCE`停止并诊断，不自动修改学习率或预算。

## 5. 冻结多尺度评估

PG95 validation/test都使用同一等实际移除数量规则：

- 每样本目标额外缺失率0.15；
- L1/L5/L20严格移除相同数量的原历史观测元素；
- nested seed等于对应实验seed；
- 最终多尺度预测固定为`0.5*original + 0.25*L5 + 0.25*L20`；
- L1只作结构审计，不进入最终融合；
- 不更新模型、原型memory或任何统计；
- 保存original、L1/L5/L20、双路、多尺度指标、实际移除数量、预测差异和串行推理时间。

S4M test使用显式的只读评估身份：关闭MIPP并核验S4M checkpoint，不把S4M伪装为PG95。为复用同一指标实现，它可以计算多尺度视图，但正式三模型表只读取S4M original；S4M上的额外融合结果不用于选择、比较或论文主张。

新评估配置也必须先通过3 batch smoke，再运行完整validation/test。smoke指标不进入科学统计。

## 6. 完整任务集合与依赖

seed2026：

1. S4M test smoke；
2. S4M完整test。

seed2027/2028各自固定为：

1. S4M训练smoke → S4M正式训练；
2. PG95训练smoke → PG95正式训练；
3. PG95正式checkpoint → PG95 validation smoke → 完整validation；
4. S4M正式checkpoint → S4M test smoke → 完整test；
5. PG95正式checkpoint → PG95多尺度test smoke → 完整test。

合计22个任务。队列最大并发5，但只在实时显存、历史峰值、安全余量和GPU文件锁允许时动态补位；不足时等待，不修改正式batch size、模型或预算。OOM按递增退避回队列，代码/配置/数据错误不无限重试。

## 7. 冻结统计与主比较

validation和test分别报告：

- 每个seed的S4M original MSE/MAE；
- 每个seed的PG95 original MSE/MAE；
- 每个seed的PG95+多尺度CNM-SE MSE/MAE；
- PG95相对S4M的绝对变化与相对改善率；
- CNM-SE相对PG95的绝对变化与相对改善率；
- CNM-SE相对S4M的绝对变化与相对改善率；
- 三seed均值、样本标准差和双指标胜出seed数；
- 参数量、峰值显存、训练时间、original与多尺度串行推理时间。

其中S4M validation直接取训练formal保存的完整validation指标；S4M test取新增只读test；PG95/CNM-SE取同一个冻结checkpoint、同一个评估run中的original和multiscale字段，保证配对。

n=3仅用于复现稳定性描述。可以报告逐seed配对差值和胜出数，但不以低功效p值声称统计显著性。seed同时改变训练初始化和固定缺失资产，因此结论表述为“联合seed复现稳定性”，不能解释为纯初始化方差。

## 8. 预注册决策标签

- `TRAFFIC_MULTISEED_PROTOCOL_FAIL`：关键配置、资产、checkpoint身份或评估配对失败；
- `TRAFFIC_MULTISEED_INCOMPLETE`：至少一个预先纳入的正式任务未获得可用结果；
- `TRAFFIC_MULTISEED_UNSTABLE`：任务完整，但PG95→CNM-SE未在三个test seed上全部同时改善MSE与MAE；
- `TRAFFIC_MULTISEED_CONFIRMED`：任务完整、协议审计通过，且PG95→CNM-SE在三个test seed上全部同时改善MSE与MAE；
- `TRAFFIC_MULTISEED_STRONG_CHAIN_CONFIRMED`：在上一条件基础上，PG95相对S4M也在三个test seed上全部同时改善MSE与MAE。

这些标签不等同于“保证SCI二区”。最终投稿水平还取决于更多缺失率/形态、强基线、统计量、效率、消融、相似工作边界和论文叙事。

## 9. 完成判据

- 本文档时间早于seed2027/2028资产和任何新seed模型结果；
- 新资产哈希、可重生性、观测值和跨seed差异审计全部通过；
- 22个预注册任务全部有唯一ID和最终状态；
- 每个正式run保存resolved config、command、环境、Git状态/diff、日志、metrics、资源、checkpoint或只读源checkpoint身份；
- 基线关闭路径、PG95评估和历史seed2026结果未被新增代码破坏；
- 三seed结果由metrics自动汇总到JSON/Markdown/CSV，不手工抄写；
- 报告明确全部失败与成本，不根据结果删seed、不调整方法、不选择性报告。
