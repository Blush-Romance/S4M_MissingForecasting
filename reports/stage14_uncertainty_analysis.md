# 步骤14：跨掩码预测分歧风险诊断

- 范围：冻结PG95 checkpoint，仅使用Weather validation；未读取test样本。
- 主门：两个seed均须Spearman≥0.15、Q4/Q1 MSE≥1.25、MAE≥1.10。

| seed | Spearman | Q1 MSE | Q4 MSE | MSE ratio | Q1 MAE | Q4 MAE | MAE ratio | PASS |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2024 | 0.208782 | 0.477858497 | 0.616060015 | 1.2892 | 0.348402349 | 0.411120247 | 1.1800 | True |
| 2025 | 0.207427 | 0.402836162 | 0.743468917 | 1.8456 | 0.342259995 | 0.414839732 | 1.2121 | True |

- 决策：`STAGE14_UNCERTAINTY_GATE_PASS_FREEZE_PERTURBATION_MODULE`
- 说明：两个seed均通过预注册风险门；允许冻结扰动不确定性感知候选。
