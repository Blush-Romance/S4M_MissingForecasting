# 步骤11：MRBA双锚点正式比较

- MRBA：Mask-Regime Balanced Admission（缺失形态均衡准入）。
- 预注册门槛：两个seed均须双指标改善PG95，且均须降低相对稳定性Top-M的mask覆盖距离。
- 工件/协议状态：`PASS`；机制状态：`PASS`。
- 机制状态失败仍必须报告精度负结果；它不等同于协议或工件损坏。

| seed | PG95 MSE | BMSA MSE | MRBA MSE | PG95 MAE | BMSA MAE | MRBA MAE | coverage gain | MRBA双胜PG95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2024 | 0.271194641 | 0.269150735 | 0.269131441 | 0.300850302 | 0.299888049 | 0.299982510 | 0.021153716 | True |

- seed2024 MRBA相对PG95：MSE +0.7608% ，MAE +0.2884% 。
| 2025 | 0.256450072 | 0.260209968 | 0.260133536 | 0.295111260 | 0.299123184 | 0.301007107 | 0.0211500848 | False |

- seed2025 MRBA相对PG95：MSE -1.4363% ，MAE -1.9978% 。

- 决策：`MRBA_TWO_ANCHOR_GATE_FAILED_STOP_CANDIDATE`
- 说明：MRBA未同时通过双锚点精度门与mask覆盖机制门；保留结果并停止该候选，不扩seed或数据集。
