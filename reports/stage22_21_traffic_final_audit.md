# 步骤22.21 Traffic外部验证终态

## 1. 协议与工件

- 队列终态：`SUCCESS`；S4M、PG95、validation多尺度和唯一一次test均成功。
- validation科学门24项检查全部通过，决策为`TRAFFIC_VALIDATION_PASS_ONE_TEST_AUTHORIZED`。
- 唯一test使用冻结PG95 checkpoint（检查点）、seed2026固定缺失资产、L1/L5/L20严格等实际缺失率视图和固定权重`0.5 original + 0.25 L5 + 0.25 L20`。
- test共213 batch（批次）/3408样本；模型与memory（原型记忆库）均未更新，全部工件检查通过。

## 2. Validation结果

| 方法 | MSE | MAE |
|---|---:|---:|
| S4M | 0.536970827 | 0.316911549 |
| PG95 | 0.528318361 | 0.315015020 |
| PG95 + 多尺度CNM-SE | **0.522464090** | **0.311083244** |

- PG95相对S4M：MSE提升1.61135%，MAE提升0.59844%。
- 多尺度相对PG95：MSE再提升1.10809%，MAE再提升1.24812%。
- 最终方法相对S4M：MSE提升2.70159%，MAE提升1.83910%。

## 3. 唯一Test结果

| 路径 | MSE | MAE |
|---|---:|---:|
| PG95 original（原始路径） | 0.672083688 | 0.361379550 |
| original + L5 | 0.668273131 | 0.359787704 |
| 多尺度CNM-SE | **0.665930768** | **0.358186681** |

- 多尺度相对PG95原始路径：MSE提升0.915499%，MAE提升0.883522%。
- 多尺度相对original+L5：MSE提升0.350510%，MAE提升0.444991%。
- 冻结测试结论：`TRAFFIC_TEST_MULTISCALE_CONFIRMED`，即Traffic测试集双指标延续validation方向。

## 4. 边界与下一步

这是单个Traffic缺失seed的外部方向性确认，不等同于统计稳定性。公开S4M没有披露Traffic最终超参数和mask seed，因此本结果是透明、固定、公平的外部扩展，不冒充官方Traffic复现。

下一步应冻结Traffic seed2027/2028缺失资产并完整复核S4M、PG95和多尺度CNM-SE，不再修改模型、学习率、块长或融合权重；两seed完成后汇总均值、标准差和配对改善率。
