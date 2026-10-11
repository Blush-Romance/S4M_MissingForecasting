# 步骤22.25：三视图CNM-SE CUDA多流并发协议

## 1. 动机

步骤22.24把三视图拼成`3B`批次后，MSE/MAE几乎不变，但CUDA在不同batch形状下产生了超过预注册容差的逐点数值差。本步骤保持original、L5、L20每路输入都为原始`B=16`，分别发往三个CUDA stream（CUDA计算流）并发执行，以避免改变单路kernel（计算核）的输入形状。

## 2. 固定执行定义

- 串行对照：同一模型依次执行original、L5、L20；
- 多流候选：同一只读模型在三个独立CUDA stream中分别执行original、L5、L20，主stream等待三路完成后按`0.5/0.25/0.25`融合；
- 三路都保持batch size 16，不拼接、不拆分；
- 参数、checkpoint、输入、mask、预处理、融合、指标和validation顺序不变；
- 模型处于eval模式，prototype memory（原型记忆）只读，不更新任何状态；
- 两种执行路径在同一GPU、同一batch内比较，并按batch奇偶交替先后顺序。

## 3. 门槛

沿用步骤22.24在结果前冻结的门，不因已看到batch拼接结果而放宽：

1. 四路（original/L5/L20/融合）最大绝对预测差均`<=1e-4`；
2. 完整MSE与MAE绝对差均`<=1e-6`；
3. 正式平均流水线耗时至少降低20%；
4. 93个配对batch的bootstrap（自助法）95%置信区间下界大于0；
5. 无OOM，不修改batch size、模型或精度格式。

## 4. 科学硬依赖

- seed2027先跑3-batch smoke（1 warmup、2计时）；
- smoke只决定数值等价与运行安全，不用两batch速度决定正式门；
- 若smoke数值等价失败，formal必须进入`BLOCKED_DEPENDENCY`，不得自动占GPU；
- 只有smoke工件PASS且`prediction_equivalence_pass=true`，才启动103-batch正式validation；
- seed2027正式等价与速度门都通过后，才允许seed2028独立复核。

## 5. 结果定位

该方案仍是部署/系统优化，不是新的预测精度模块。若成功，它解决CNM-SE约3倍串行延迟的工程弱点；若速度门失败，说明单路前向已接近GPU饱和或存在stream不可并行依赖，应作为可信负结果停止，不再以CUDA Graph等只减少少量launch开销的方法包装贡献。
