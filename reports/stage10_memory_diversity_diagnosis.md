# 步骤10：原型库多样性诊断

本报告只读取正式checkpoint中的原型库，不加载数据集或测试目标。

| seed | method | centroids | NN cosine↓ | pair≥0.95↓ | centroid rank ratio↑ | vector rank ratio↑ | slot entropy↑ | within cosine |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2023 | pg95 | 3 | 0.999639 | 1.000000 | 0.343800 | 0.080546 | 1.000000 | 0.999963 |
| 2023 | bmsa50 | 3 | 0.999930 | 1.000000 | 0.374342 | 0.118094 | 1.000000 | 0.999992 |
| 2024 | pg95 | 3 | 0.999940 | 1.000000 | 0.498972 | 0.129097 | 1.000000 | 0.999994 |
| 2024 | bmsa50 | 3 | 0.999942 | 1.000000 | 0.551157 | 0.146457 | 1.000000 | 0.999992 |
| 2025 | pg95 | 3 | 0.999941 | 1.000000 | 0.560313 | 0.199816 | 1.000000 | 0.999988 |
| 2025 | bmsa50 | 3 | 0.999876 | 1.000000 | 0.426667 | 0.110244 | 1.000000 | 0.999992 |

## BMSA - PG95 配对差值

- seed 2023:
  - `centroid_nearest_neighbor_cosine_mean` (lower is better): +0.00029147
  - `centroid_redundant_pair_fraction_ge_0_95` (lower is better): +0.00000000
  - `centroid_centered_effective_rank_ratio` (higher is better): +0.03054196
  - `stored_vector_centered_effective_rank_ratio` (higher is better): +0.03754802
  - `slot_occupancy_normalized_entropy` (higher is better): +0.00000000
- seed 2024:
  - `centroid_nearest_neighbor_cosine_mean` (lower is better): +0.00000244
  - `centroid_redundant_pair_fraction_ge_0_95` (lower is better): +0.00000000
  - `centroid_centered_effective_rank_ratio` (higher is better): +0.05218479
  - `stored_vector_centered_effective_rank_ratio` (higher is better): +0.01736037
  - `slot_occupancy_normalized_entropy` (higher is better): +0.00000000
- seed 2025:
  - `centroid_nearest_neighbor_cosine_mean` (lower is better): -0.00006449
  - `centroid_redundant_pair_fraction_ge_0_95` (lower is better): +0.00000000
  - `centroid_centered_effective_rank_ratio` (higher is better): -0.13364580
  - `stored_vector_centered_effective_rank_ratio` (higher is better): -0.08957215
  - `slot_occupancy_normalized_entropy` (higher is better): +0.00000000

这些指标只用于检验原型结构假设；不能单独证明预测因果，也不用于从测试误差反向挑选联合准入权重。
