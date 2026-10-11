# 步骤22.21：Traffic validation科学门控

- 协议审计：`PASS`
- 决策：`TRAFFIC_VALIDATION_PASS_ONE_TEST_AUTHORIZED`
- 是否授权唯一一次test：`true`
- L5/L20视图响应比：`0.0490058405689` / `0.05837935126`

| 路径 | MSE | MAE |
|---|---:|---:|
| S4M validation | 0.53697083 | 0.31691155 |
| PG95 validation | 0.52831836 | 0.31501502 |
| PG95原始路复核 | 0.52831836 | 0.31501502 |
| 原始+L5 | 0.52533247 | 0.31307078 |
| 多尺度CNM-SE | 0.52246409 | 0.31108324 |

所有门控只使用validation；未授权时不会构造test loader（测试加载器）。
