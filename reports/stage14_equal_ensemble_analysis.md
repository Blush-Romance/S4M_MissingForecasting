# 步骤14.5：固定等权双视图validation分析

- 范围：Weather PG95冻结checkpoint；仅validation；未读取test。
- 固定权重：原始视图0.5，嵌套缺失视图0.5。
- 门槛：每个seed MSE改善≥1.5%且MAE改善≥0.5%。

| seed | on MSE | ensemble MSE | MSE gain | on MAE | ensemble MAE | MAE gain | PASS |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2024 | 0.521840304 | 0.509476549 | 2.3693% | 0.367439541 | 0.364180358 | 0.8870% | True |
| 2025 | 0.530318574 | 0.519410617 | 2.0569% | 0.366724006 | 0.361909157 | 1.3129% | True |

- 决策：`STAGE14_EQUAL_ENSEMBLE_GATE_PASS_IMPLEMENT_TEST_MODULE`
