# 步骤22.39：距离填补与编码轴 CPU 审计

状态：通过；19 个断言；CPU 合成算例，无真实数据、checkpoint 或 test 读取。

## 确认与边界

1. min_idx 分子误用已标准化 max_idx，再减原始 min 距离均值，混合标准化尺度与原始步数尺度；不是真正的 min 距离标准化。
   分类：`CONFIRMED_UPSTREAM_IMPLEMENTATION_MISMATCH`。数学/源码错误已证实，但当前 checkpoint 已适应该错误；直接修正后的好坏需 validation 诊断，不能声称必然提升，也不计作创新。

   证据：`baselines/S4M/s4m/model/S4M.py:314`；SHA256 `3f97a483a3c08c31c598a155343797a12b8e42932e79bda22fbd064bd625b526`。
   证据：`baselines/S4M/s4m/model/S4M.py:315`；SHA256 `3f97a483a3c08c31c598a155343797a12b8e42932e79bda22fbd064bd625b526`。
   证据：`upstream/S4M/s4m/model/S4M.py:145`；SHA256 `656131ea5215db76a6d4c134e427d6c290599bba9a97e248c57cfd72a002d1e5`。
   证据：`upstream/S4M/s4m/model/S4M.py:146`；SHA256 `656131ea5215db76a6d4c134e427d6c290599bba9a97e248c57cfd72a002d1e5`。
   证据：`baselines/S4M/s4m/model/S4M.py:262`；SHA256 `3f97a483a3c08c31c598a155343797a12b8e42932e79bda22fbd064bd625b526`。
   证据：`baselines/S4M/s4m/model/S4M.py:263`；SHA256 `3f97a483a3c08c31c598a155343797a12b8e42932e79bda22fbd064bd625b526`。

2. Dataset_Custom4 在归一化并将缺失置 0 后求窗口极值，未排除缺失位置；合成正/负观测例证实缺失 0 可成为最小/最大值。
   分类：`CONFIRMED_UPSTREAM_MASK_SEMANTICS_RISK`。不访问真实数据，因此未量化当前 Traffic 上比例或性能伤害；全窗历史极值可使用历史内较晚点，不等于使用预测未来标签。

   证据：`baselines/S4M/s4m/data_provider/data_loader.py:1249`；SHA256 `abd42e5ae55a1f56c8a5f12eade2fa64413b1c7f75c4e941e5696bbf2f14ee82`。
   证据：`upstream/S4M/s4m/data_provider/data_loader.py:1245`；SHA256 `b907a7fd2d64e2eb8087e6036a6df836b7e2a916bd87baf206161a280f0db1f2`。

3. Conv2d 输出 [N,H,Tc,1] 直接 reshape 为 [N,Tc,H] 并非交换轴；时间/通道含义发生重新打包。
   分类：`CONFIRMED_UPSTREAM_AXIS_MAPPING_MISMATCH`。不一定造成报错，也不能证明损害当前已训练模型。应单列 bug-fix baseline，不能把修正后重训收益混同新模块创新。

   证据：`baselines/S4M/s4m/model/Bank.py:100`；SHA256 `9e2206d4ee94013fc58ab8a290695780e17bb34c62d04839e2ac7db51d108474`。
   证据：`upstream/S4M/s4m/model/Bank.py:97`；SHA256 `50a4aab5a8e80287ff493b4bb469580793ffcc03f4d0cd8a6eb41b3db3f5ed18`。

4. 公式无 epsilon，常数外部距离与 L=1 会非有限；但当前 Dataset_Custom4 L=192 的 |t-k| 对全部合法 k 标准差均为正，完全缺失历史也不自动零 std。
   分类：`LATENT_NOT_L192_DATASET_FAILURE`。不应将一般数值风险误报为当前实验故障。


5. exp(-logit) 比值在极端 float32 范围可出现 inf/inf 或 0/0；实际 logits 尚未采样，不能声称当前发生溢出。
   分类：`LATENT_STRESS_CASE_NOT_OBSERVED_FAILURE`。稳定 sigmoid 为代数等价建议，不在本步骤修改基线。


## 合成单元结果

L=192 合成距离：实际 min 分支均值 -1.431023，正确自身标准化均值 -0.000000；与 max 分支时间相关系数 1.000000。
全部 192 个合法极值位置的最小距离标准差：27.788280，故该协议下零 std 没有被支持。

这些不是 MSE/MAE（均方误差/平均绝对误差）实验；不能据此宣称性能提升。

## 可重复命令

```sh
/mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python scripts/stage22_39_distance_audit.py
```

完整断言与源代码哈希：`reports/stage22_39_distance_audit.json`。
