# 步骤22.43：最终验收与历史来源补审

日期：2026-10-06（北京时间）。状态：执行及复算PASS；不是新训练、独立测试或两个创新成立。

## 执行证据

- 8项合成单测通过；prepare、真实2028缓存smoke、正式CPU分析均退出0。
- smoke：`cnm_cache_audit__traffic__validation__seed2243__smoke__20261006_224028`。
- formal：`cnm_cache_audit__traffic__validation__seed2243__20261006_224121`。
- 正式计算0.906311秒、峰值RSS（常驻内存）38.457MiB、GPU显存0；不执行模型前向/反向、不训练、不读取test。
- 141/141来源哈希相符；独立60项指标/增益/均值与样本标准差复算通过，run内metrics与报告相同。
- 自动总表302项包含新smoke/formal；两者为缓存审计身份，正式行MSE/MAE/RMSE/checkpoint记NA，不重复冒充新模型成绩。
- 三seed修复后同一检查点原路→CNM-SE（反事实嵌套掩码自集成）双指标改善；逐seed相对改善率平均MSE0.9579816080%、MAE0.6834905678%。
- 原22.41公式修复主比较仍2/3双胜，决策字符串不变；本步未切换模型公式、修原型索引、改融合权重或访问test。

## 历史原版身份：额外证据，不回填旧配置

旧原版训练`resolved_config.yaml`没有完整`effective_parser_args`（实际生效参数快照）。主分析兼容`official_args`仅是读取旧记录，不能将缺省直接当作完整解析证据。最终独立补审读取以下三个旧run：

- `runs/s4m_paired_rng__traffic__point__mr0.06__seed2027__20260908_144251`。
- `runs/s4m_paired_rng__traffic__point__mr0.06__seed2028__20260919_194946`。
- `runs/s4m_paired_rng__traffic__point__mr0.06__seed2029__20260919_225902`。

三者各自`train.log`唯一运行时`Namespace(...)`（参数命名空间）经AST（抽象语法树）读取，17个创新/子开关均明确False。包括MIPP/FUSR/CFR/MRRA/MCPR/TTRR/ARD/DMRB/BF-CMVD/PBA/HFD/RADR及`use_content_mask_factorization`、`use_mask_intervention_pair`、`use_stability_write_gate`、`use_utility_read_gate`、`use_fusr_spectral_uncertainty`。实际检查点无对应模块参数，作为旁证而非无参数损失关闭的唯一证据。

三run冻结的`source_snapshot/baseline/s4m/run.py`没有距离修复开关；冻结`model/S4M.py`有两处最小距离公式误用最大距离项的AST结构，确认旧公式来源。三份相应源码哈希一致：

| 冻结文件 | SHA256（文件内容哈希） |
| --- | --- |
| `source_snapshot/baseline/s4m/model/S4M.py` | `3f97a483a3c08c31c598a155343797a12b8e42932e79bda22fbd064bd625b526` |
| `source_snapshot/baseline/s4m/run.py` | `eb32e826c56e476f8c88db4b744be4675d216750cb204f16e866c55038790d6b` |

这是运行时日志+冻结源码的针对性身份佐证，不等于重建所有旧最终参数。修复版主比较拥有完整参数解析记录。补审最初两次正则检查误把子开关名省略`use_`前缀而断言失败，改为对真实Namespace的AST关键词读取后通过；没有修改原实验、参数或冻结分析器。

## 不应越过的证据边界

- 本步已有误差缓存复算而非从完整预测/真值重算；可验证汇总，不能独立重演融合。
- 同检查点推理集成新增参数0，但需3次前向，训练预算共享不等于推理算力相同。
- 历史组件参考推理倍率约3.12/3.15/2.86，非隔离端到端基准；没有路线独立峰值显存证据。
- 已开发验证集和三个联合初始化/人工缺失随机种子，仅支持方向性信号，不支持显著性、跨数据集普遍提升或SCI分区保证。
- 文件哈希不变不是运行时银行前后内容哈希；记忆库只读来自冻结代码条件与工件记录。
- 原型代表索引错误仍未修复/隔离，CNM收益不应被称为完全排除全部实现风险。

结论：保留CNM的局部一致正信号，PG95和第二个独立创新仍未成立。公式交叉诊断暂列备选、未启动。建议22.44独立、可关闭地处理原型代表索引错误，先CPU/旧基线回归与单seed同预算验证；不将错误修复计为创新，本轮未实施。

主要证据：`reports/stage22_43_cnm_audit.{json,md,csv}`、正式run的配置/来源清单/日志/状态/资源/代码快照、`runtime/stage22_43_cnm_audit/source_manifest.json`、`reports/stage22_43_tests.json`、`reports/stage22_43_smoke.json`。
