# 步骤11：缺失形态几何诊断

- 范围：仅Weather输入mask的train/validation历史；未加载数值、未来目标或test目标。
- 工件状态：`PASS`。

| seed | split | candidates | unique patterns | entropy | mean Hamming | effective-rank ratio | zero-missing |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2024 | train | 219600 | 7946 | 0.750241 | 0.369441 | 0.760663 | 0.138725 |
| 2024 | validation | 31050 | 2127 | 0.851553 | 0.384253 | 0.763122 | 0.086699 |
| 2025 | train | 219600 | 8201 | 0.756779 | 0.377789 | 0.760875 | 0.129504 |
| 2025 | validation | 31050 | 1961 | 0.836315 | 0.370719 | 0.757873 | 0.103961 |

- 解释边界：mask空间未塌缩只支持进入MRBA快速验证，不能单独证明预测会提升。
