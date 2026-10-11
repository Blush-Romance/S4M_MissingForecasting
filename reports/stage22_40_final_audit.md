# 步骤22.40独立验收

日期：2026-10-04。只读审计已完成工件，不轮询训练，不修改模型和配置。

6个任务均SUCCESS、退出0、首次成功、无OOM（显存不足）。6份最终实际解析参数与resolved_config（最终配置）一致，无未声明科学设置变化。原基线12份冻结资产和门禁46份源码哈希不变，最终评估checkpoint（模型存档）摘要与对应训练文件一致。

独立重新求和NPZ中的逐窗口SSE（平方误差和）和SAE（绝对误差和），再除共同目标元素计数，得到：

| 固定CRN原始路径 | 原S4M | 最小距离修复 | 相对改善 |
| --- | ---: | ---: | ---: |
| MSE（均方误差） | 0.534761380397 | 0.520650410251 | 2.638741% |
| MAE（平均绝对误差） | 0.326645003141 | 0.316614357334 | 3.070810% |

CRN（Common Random Numbers，共同随机数）评估共1648窗口；10类掩码、计数及索引资产完全配对。双方初始化、完整训练/验证采样摘要列表和训练随机流一致；均4轮、2992次优化更新、28,371,290参数。

原训练耗时1913.27秒，修复2089.97秒，未显示加速。实际分配显存约8007/8005MiB，近似不变；reserved（缓存分配器预留）的变化不能计为结构轻量化。已有CNM（反事实嵌套掩码）次要集成路径改善MSE2.365440%/MAE2.555281%，不作为本轮主要判据或新增创新。

工件来源：

- `runtime/stage22_40_distance_fix/state.json`
- `reports/stage22_40_distance_fix.json`
- 原验证：`runs/cnm_se_crn_v1_s4m__traffic__validation__seed2027__20260908_151505/validation_equal_rate_multiscale_samples.npz`
- 修复验证：`runs/cnm_se_crn_v1_s4m__traffic__validation__seed2027__20261004_230031/validation_equal_rate_multiscale_samples.npz`
- 修复训练：`runs/s4m_min_distance_fixed__traffic__point__mr0.06__seed2027__20261004_222522/`

结论：单seed、单数据集、验证集上的方向性双指标改善，足以补seed2028/2029进行联合重复验证；不是已证明跨数据集稳定，也不是研究创新或SCI分区证据。新seed必须匹配各自22.37历史数据/mask/nested_seed，联合变化缺失资产与训练随机性；不得将重叠窗口当独立统计重复。
