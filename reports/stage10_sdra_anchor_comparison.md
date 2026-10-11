# 步骤10：SDRA双锚点正式比较

- SDRA：Stability-Diversity Rank Admission（稳定性—多样性秩准入）。
- 预注册门槛：seed2024和seed2025都必须同时改善PG95的MSE与MAE。
- 工件/协议状态：`PASS`；机制状态：`FAIL`。
- 机制状态失败仍必须报告精度负结果；它不等同于协议或工件损坏。

| seed | PG95 MSE | BMSA MSE | SDRA MSE | PG95 MAE | BMSA MAE | SDRA MAE | diversity gain | SDRA双胜PG95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2024 | 0.271194641 | 0.269150735 | 0.272600479 | 0.300850302 | 0.299888049 | 0.301668060 | -6.60467961e-06 | False |

- seed2024 SDRA相对PG95：MSE -0.5184% ，MAE -0.2718% 。
| 2025 | 0.256450072 | 0.260209968 | 0.255767497 | 0.295111260 | 0.299123184 | 0.303207312 | 3.33340149e-06 | False |

- seed2025 SDRA相对PG95：MSE +0.2662% ，MAE -2.7434% 。

- 决策：`SDRA_TWO_ANCHOR_GATE_FAILED_STOP_CANDIDATE`
- 说明：SDRA未在两个锚点同时双指标优于PG95；保留结果并停止该候选，不扩seed或数据集。
