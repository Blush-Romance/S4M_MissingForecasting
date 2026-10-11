# 步骤21：Electricity validation科学门控

- 协议审计：`PASS`
- 决策：`FULL_VALIDATION_FAIL_TEST_FORBIDDEN`
- 是否授权唯一一次test：`false`

| 路径 | MSE | MAE |
|---|---:|---:|
| S4M validation | 0.87838678 | 0.75300645 |
| PG95 validation | 0.87280959 | 0.75450765 |
| PG95原始路复核 | 0.87280959 | 0.75450765 |
| 原始+L5 | 0.87280959 | 0.75450765 |
| 多尺度CNM-SE | 0.87280959 | 0.75450765 |

门控仅使用validation；论文的0.274/0.367是test参考值，不参与门控。
