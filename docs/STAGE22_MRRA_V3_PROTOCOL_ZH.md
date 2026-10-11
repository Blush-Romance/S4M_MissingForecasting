# 步骤22 MRRA-v3 冻结协议

## 1. 研究问题

MRRA-v1/v2均在Electricity训练中数值发散。参数审计与源码表明，辅助重建的恒等支路会把梯度从`reconstruction = z_s + correction`直接传回衰减插补层；这些层位于指数计算之前，且与新适配器共用Adam学习率0.01。

## 2. 冻结结构

MRRA-v3（梯度隔离双时间尺度掩码响应残差适配器）固定为：

1. 版本标识：`v3_gradient_isolated_two_timescale`；
2. MRRA读取`z_s.detach()`与观测掩码，辅助重建和上下文残差不得经MRRA支路反传到PG95主干；
3. rank=32、因果深度卷积kernel=5；
4. 使用无仿射参数的LayerNorm（层归一化）；
5. 上下文残差与重建修正继续使用`tanh`有界函数；
6. 上下文投影保持零初始化，启用模块时初始预测严格等于PG95；
7. 重建损失仍为合成缺失位置MSE，权重固定0.1。

## 3. 双时间尺度优化

- PG95主干：Adam，初始学习率0.01；
- MRRA参数组：同一Adam，学习率乘数固定0.1，即初始0.001；
- 两组共同使用原type1 epoch调度，每次更新保持0.1比例；
- batch size、epoch、early stopping、数据和评价资产均不改变。

## 4. 验证顺序

1. CPU组件与负例：梯度隔离、因果性、零初始化、有限输出/梯度、参数组无重叠、学习率比例跨调度保持；
2. 当前代码PG95默认关闭GPU smoke；
3. MRRA-v3稳定性smoke：3 epoch，每epoch最多800 batch，验证覆盖v1/v2故障窗口；
4. 仅稳定性smoke成功后运行15 epoch完整validation formal。

## 5. 科学门

formal相对冻结PG95 Electricity seed2023完整validation，MSE和MAE均不得退化且至少一项严格改善。未通过则不扩多seed、不访问test，并保留负结果。
