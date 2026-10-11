# 内容发现

- 原论文官方出处：S4M，ICLR2025，Jing Peng等；官方代码WINTERWEEL/S4M。只据原论文结构与本地实际实现重绘，不能直接拿旧蓝图中已失败FUSR方案当当前模型。
- Weather旧L5双路test与新版等缺失量多尺度validation不同，Traffic最新为固定验证资产的联合随机种子，不能合并为一个完整跨数据集三seedtest结论。
- 最新修复后CNM三seed增益均为正，但属于已有缓存复算，3次前向开销，不是新独立预测实验。PG95仍不稳定，距离修复和未修索引错误不算创新。
- 论文模型图实际为Figure 2（PDF第4页），不是Figure 1；Table 1位于第8页。所有示意依据实际代码重绘，不假画已关闭的因子分解或历史失败模块。
- 本地实际缺失率随数据/seed变化：Weather早期约24.7%、ETTh1约23.1%、Electricity/Traffic约26%，名义r=.06不等于只缺6%。
- 字体验收必须检查Latin与数字：Droid正文首轮字形缺失，WQY正文修复；上标负号改ASCII e-4。最终0缺字NUL，PDF中文/数字可搜索。
