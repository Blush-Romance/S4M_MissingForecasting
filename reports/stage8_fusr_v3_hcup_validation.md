# 步骤8：HCUP验证集筛选

- run：`/mnt/data/zza/code/S4M_MissingForecasting/runs/s4m_fusr_v3_hcup__weather__validation_point_r006__seed2023__20260718_202756`
- 只使用validation；当前/测试未来标签不进入历史探针。
- 有效历史探针：5072；候选：7。

| 聚合 | calibration门控率 | audit门控率 | audit MSE增益 | audit MAE增益 | 95% CI下界 | 门禁 |
|---|---:|---:|---:|---:|---:|---|
| direct_last_probe | 0.9731 | 0.9979 | -0.0182% | -0.0369% | -0.000210487 | FAIL |
| ewma_alpha_0.50 | 0.9714 | 0.9969 | -0.0264% | -0.0541% | -0.000281645 | FAIL |
| ewma_alpha_0.25 | 0.9714 | 0.9969 | -0.0245% | -0.0516% | -0.00025959 | FAIL |
| ewma_alpha_0.10 | 0.9918 | 1.0000 | 0.0000% | 0.0000% | 0 | FAIL |
| rolling_mean_32 | 0.9885 | 1.0000 | 0.0000% | 0.0000% | 0 | FAIL |
| rolling_mean_96 | 1.0000 | 1.0000 | 0.0000% | 0.0000% | 0 | FAIL |
| rolling_median_32 | 0.9892 | 1.0000 | 0.0000% | 0.0000% | 0 | FAIL |

- 决策：`HCUP_VALIDATION_NEGATIVE_DO_NOT_TOUCH_TEST`
- 锁定聚合：`无`
- 说明：没有因果历史探针聚合在后段验证审计同时满足MSE、MAE和置信区间门禁；不应使用测试标签继续调节。
