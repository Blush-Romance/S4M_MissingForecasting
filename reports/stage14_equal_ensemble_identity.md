# Stage 14 equal-ensemble identity audit

- 范围：冻结validation缓存；没有读取test。
- 性质：平方误差恒等式精确重算，不估计MAE。

| seed | on MSE | nested MSE | equal ensemble MSE | vs on | vs best single |
|---:|---:|---:|---:|---:|---:|
| 2024 | 0.521840291 | 0.518831910 | 0.509476541 | 2.3693% | 1.8032% |
| 2025 | 0.530318582 | 0.528072508 | 0.519410618 | 2.0569% | 1.6403% |

- 决策：`RERUN_VALIDATION_TO_MEASURE_EQUAL_ENSEMBLE_MAE`
