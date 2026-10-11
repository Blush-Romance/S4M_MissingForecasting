# 研究分支与投稿证据审计（组会只读材料）

审计日期：2026-10-10。范围：项目 `AGENTS.md`、`task_plan.md`、`findings.md` 和已落盘 `reports/`；没有训练、GPU、模型前向或评估，没有读取 test 逐样本缓存。本文件不改任何历史科学判定。

## 组会可直接讲的结论

> 我们经历了几类机制筛选，而不是开展了302次独立精度实验。稳定保留下来的信号是固定因果嵌套视图自集成 CNM-SE；原型效用路由、写入筛选和单视图蒸馏没有通过各自冻结门。近期控制实验表明 PG95 不能作为稳定第二贡献；距离公式修复属于基线纠错，不是创新。下一步先把基线实现风险隔离，再统一方法版本和新测试边界，确认小幅 CNM 收益是否能在强基线和计算代价下成立。

正改善率统一定义为 `(参照误差 - 候选误差) / 参照误差 × 100%`，正数表示更好。下面按11类研究假设压缩历史；类内多个 smoke、缓存、诊断、重试不算独立精度结论。

## 1. 供给侧原型净化：MIPP → SG95 → PG95；PG95整体开关仍不足以证明净化

- 2026-07：Weather seed2023，同固定mask正式测试。S4M为 MSE/MAE `0.263525745/0.301159963`；Main MIPP `0.263811787/0.296138031`（`-0.1085%/+1.6675%`），SG95 `0.266177171/0.297541164`（`-1.0061%/+1.2016%`），PG95 `0.261946457/0.293878906`（`+0.5993%/+2.4177%`）。只有PG95在此单seed双胜。
- 三变体写入接受率均超过99.5%；PG95为99.9142%，其内容/掩码因子分解关闭。新增153,130参数；同最大15epoch/早停patience3，但实际优化步S4M=11,430、PG95=22,860，不是等实际计算量消融。
- 2026-09配对公共初始化、隔离辅助随机流、CRN验证后，Traffic2027/2028/2029 PG95相对S4M逐seed改善率均值为MSE `-0.7209%`、MAE `-0.4616%`，只1/3双胜；已有CNM时PG95仍只1/3双胜。
- 结论：PG95不是完全无效，但不能列为当前稳定第二有效创新；整体开关没有隔离配对损失、adapter、写入门三个子机制。暂停扩展PG95。
- 可核验正式来源：`reports/stage7_mipp_comparison.{md,json}`；`reports/stage22_37_control.{md,json}`。与论文mask seed不完全对应，不把论文表格值用于本地严格胜负。

## 2. 使用侧效用路由：FUSR v1/v2/v3与历史探针

- v1：Weather正式test `0.263705272/0.297522085`，相对S4M `-0.0681%/+1.2080%`；10,432样本全部开门，AUROC=0.4946，拒用率0，因此机制未成立。
- v2：冻结主干和memory、validation-only校准，正式test `0.263565329/0.301181243`，相对S4M `-0.0150%/-0.0071%`。只有17/10,432样本关闭，test AUROC=0.4744，静态14维效用映射不能跨时间稳定外推。
- v3静态离线筛选：8个候选均未过后段validation门；这8项是同缓存可学性诊断，不是8个重新训练的独立模型。
- v3 HCUP：历史探针7种聚合在完整validation均失败；短跨度lag1/4/8/16 × 7种聚合共28项均失败。以lag1直接探针为例，audit MSE/MAE改善 `-0.8223%/-1.0423%`；零变化规则全部或近乎回退always-on，不能算提升。早期32窗口smoke的相关性0.979没有迁移到完整5072有效窗口（约-0.132）。
- 结论：存在label-aware Oracle空间不等于可部署选择器成立；停止静态路由与历史反馈FUSR，不通过test继续挑阈值、lag或容量。
- 正式来源：`reports/stage8_fusr_comparison.{md,json}`；`reports/stage8_fusr_v2_comparison.{md,json}`；validation诊断来源：`reports/stage8_fusr_v3_temporal_diagnosis.{md,json}`、`reports/stage8_fusr_v3_hcup_validation.{md,json}`、`reports/stage8_hcup_short_validation.{md,json}`。v1机制统计见对应JSON与findings.md。

## 3. 写入预算与覆盖：BMSA → SDRA → MRBA

- BMSA：Weather2023/2024/2025相对PG95，双胜2/3；逐seed改善率均值 `-0.0784%/-0.2891%`。PG95 MSE/MAE均值 `0.263197057/0.296613489`，BMSA `0.263352377/0.297461341`，区间跨0。不能只展示前两seed。
- SDRA：2024改善 `-0.5184%/-0.2718%`，2025 `+0.2662%/-2.7434%`，0/2双胜；正式多样性改善只有约1e-6或负，smoke中的执行效果没有稳定保留。
- MRBA：mask覆盖距离确实降低约0.02115，2024改善 `+0.7608%/+0.2884%`，2025 `-1.4363%/-1.9978%`；机制改善未转化为预测精度。后续validation的稀有Q4两个seed均双指标退化，排除“只对稀有形态有效”的解释。
- 结论：固定M=50并未扩大写入预算，失败不是靠容量不公平；稳定性、多样性或mask覆盖不能直接当作预测效用。停止这组写入规则，不继续扩数据集或换mask路由。
- 正式来源：`reports/stage10_weather_multiseed_comparison.{md,json}`、`reports/stage10_sdra_anchor_comparison.{md,json}`、`reports/stage11_mrba_anchor_comparison.{md,json}`；验证诊断：`reports/stage12_validation_mismatch.{md,json}`。

## 4. 反事实风险训练：CFR

- Weather seed2023同预算同mask正式test，PG95+CFR `0.263401212/0.298090090`，相对PG95 `-0.5554%/-1.4330%`。训练期平均正后悔约0.00030373、有害检索率约4.41%，但没有测试增益。
- 结论：无条件逐样本正后悔hinge未形成稳定泛化；停止当前CFR，不把它直接换名为CVaR后在已看test反复调权重。
- 来源：`reports/stage13_cfr_comparison.{md,json}`。

## 5. 响应性与双指标协调：MRRA v1/v2/v3、RADR

- Electricity validation-only。MRRA-v1正式长程训练NaN；v2仅长稳定性smoke仍NaN，formal未启动；这两版没有合法完整精度结果，不能将smoke局部validation数值与正式锚点排名。
- v3用梯度隔离与两时间尺度成功完成训练，正式validation `0.878448046/0.753646903`，相对PG95 `-0.6460%/+0.1141%`。有响应不等于平方/绝对误差协调。21个内部可观测特征在fit/dev无任何双非劣规则，候选数0，停止精度路由。
- RADR validation `0.892448301/0.747824790`，相对PG95 `-2.2501%/+0.8857%`，仅反转原有权衡，未支配锚点。保留数值但工件核验有缺口：`stage22_innovation_gate.json`的 `radr_checkpoint_reload_verified=false`，总体 `PROTOCOL_FAIL_NO_SCIENTIFIC_DECISION`。不将它误写为全部工件PASS。
- 结论：v3分离了数值稳定性与科学有效性；MRRA/RADR均未提供双指标有效贡献，不组合两个未过门模块掩盖独立失败。
- 来源：`reports/stage22_innovation_gate.{md,json}`；后续MRRA复评协议PASS来源：`reports/stage22_mrra_v3_error_diagnostic.{md,json}`、`reports/stage22_mrra_internal_feature_screen.{md,json}`；NaN版本事实见findings.md和持久失败run状态。

## 6. 冻结主干的输出残差与控制：MCPR → TTRR → ARD

- Electricity seed2023：MCPR-MSE `0.876108915/0.752985771`，相对PG95 `-0.3780%/+0.2017%`；MCPR-PCD `0.915839319/0.744958994`，`-4.9300%/+1.2655%`。PCD方向有害，MSE版主要幅度过冲。
- TTRR seed2024新mask同一时间区间：相对MCPR控制 `-0.3469%/-0.2257%`，相对PG95仅 `+0.000217%/+0.019708%`，双锚点门失败。它是新mask快速确认，不是独立时间或跨数据集验证。
- ARD seed2025：相对MCPR `+0.7855%/+0.1068%`，但相对PG95 `-0.1876%/+0.0566%`。门近乎常数，解析教师并不等价于可靠推理策略。108维结构化诊断在audit秩相关-0.1791，比global8的0.1961更差，不支持直接扩容量。
- 结论：幅度治理部分有效，但未稳定支配冻结锚点；停止PCD、TTRR配置和ARD路线。不把额外后处理训练预算的结果当最终公平贡献。
- 正式来源：`reports/stage22_mcpr_comparison.{md,json}`、`reports/stage22_18_ttrr_comparison.{md,json}`、`reports/stage22_19_ard_comparison.{md,json}`；归因来源：`reports/stage22_mcpr_failure_attribution.{md,json}`、`reports/stage22_20_ard_information.{md,json}`。

## 7. 保留下来的方法信号：CNM-SE，外部边界必须一并报告

- Weather早期test正结果是L5等权双路，三seed门通过；等实际新增缺失数量的固定三路 `0.5original+0.25L5+0.25L20` 在Weather只有两个seed的validation正证据。不能直接将两代方法拼成统一三路跨数据集三seed结论。
- ETTh1：S4M/PG95的嵌套视图响应近1e-9；PG95正式MSE/MAE `1.211535967/0.840500396`，相对S4M `-0.2935%/-0.1880%`；固定多尺度无实质收益，约4.09倍前向延迟。该链路与论文差距大，不能按严格官方复现比较。
- Electricity：PG95 validation相对S4M MSE `+0.63494%`、MAE `-0.19936%`，三路近浮点噪声；冻结科学门失败，test明确没有访问。这是失败外部验证，不是缺失数据。
- Traffic旧协议2026/2027/2028 test：同PG95 checkpoint CNM增量均双胜，均值 `+0.859189%/+0.777161%`；完整PG95+CNM相对S4M平均 `+1.386230%/+1.174803%`，MAE仅2/3胜，2028退化0.088038%。
- 新CRN配对2027/2028/2029仅validation：CNM在原S4M上3/3双胜，均值 `+1.0630%/+0.9350%`。距离公式修复后的同checkpoint CNM仍3/3双胜，均值 `+0.957982%/+0.683491%`。这支持小幅收益不完全依赖距离bug，但最近核查只是既有缓存复算，不是新独立泛化。
- 结论：CNM是当前最可信的可重复方向性贡献；适用条件是有效输入扰动能产生预测互补，而非所有数据都有效；固定三路增加推理调用，不能称同推理预算或轻量化。
- 来源：`reports/stage16_cnm_se_three_seed_analysis.{md,json}`、`reports/stage18_equal_rate_multiscale_analysis.{md,json}`、`reports/stage20_etth1_pg95_full_method.{md,json}`、`reports/stage21_electricity_validation_gate.{md,json}`、`reports/stage22_22_traffic_multiseed_results.{md,json,csv}`、`reports/stage22_37_control.{md,json}`、`reports/stage22_43_cnm_audit.{md,json,csv}`。

## 8. 效率路线：删除视图、批拼接与CUDA多流

- 两路方案只有约15%–50%的三路收益保留率，没有候选达到预注册80%门。
- batch拼接253.11→198.75ms/batch，延迟下降21.263%（95%区间20.08%–22.40%），但逐点最大差5.87e-4超过原1e-4等价门；速度有效，严格数值等价失败。
- CUDA多流保持逐位等价，但平均只下降0.5165%，区间跨0、速度门失败；不能解决约3倍模型计算量。
- 结论：这些是执行优化/效率实验，不是新精度贡献；保留成本证据，不能据组件计时声称当前最新修复版本已完成端到端效率评估。
- 来源：`reports/stage22_23_efficient_cnm_se.{md,json}`、`reports/stage22_24_batch_parallel_seed2027.{md,json}`、`reports/stage22_25_cuda_stream_seed2027.{md,json}`。

## 9. 将三路教师摊销为单路学生：BF-CMVD → DMRB → PBA → HFD

- Traffic seed2027开发validation。BF-CMVD-v1的训练整样本筛选无获益者，只有smoke失败，没有formal精度；v2按token筛选真实激活。
- BF-CMVD-v2相对PG95 `+0.519939%/+0.358726%`，教师收益只保留48.0711%/44.1573%，低于冻结50%门；有部分正信号，但不扩seed或test。
- DMRB两残差基 `0.521787612/0.317986334`，相对BF-v2 `-0.199201%/-0.109647%`，教师收益保留29.75%/30.71%，参数翻倍；归因不支持mask错配或分支相加冲突，支持低秩表达不足，停止低秩路线。
- PBA末端主干摊销 `0.521529998/0.317547226`，相对BF-v2 `-0.149732%/+0.028595%`，未过60%门。归因否定“少数灾难尾部导致全部损失”，不能据此直接加CVaR。
- HFD-PBA `0.522073185/0.319194642`，相对BF-v2 `-0.254040%/-0.490051%`；训练损失/隐藏对齐实际执行，但输出Pareto目标未改善，停止HFD/PBA。
- 结论：多视图收益可部分传到单路，但没有达到冻结有效门；各版都是已参与开发seed2027 validation，不是多个独立确认试验。不以最佳单项、事后放宽门或换层继续美化。
- 来源：`reports/stage22_26_bf_cmvd_v2_seed2027.{md,json}`、`reports/stage22_28_dmrb_seed2027.{md,json}`、`reports/stage22_29_dmrb_failure_attribution.{md,json}`、`reports/stage22_30_pba_cmvd_seed2027.{md,json}`、`reports/stage22_31_pba_failure.{md,json}`、`reports/stage22_32_hfd_pba_seed2027.{md,json}`。

## 10. 按样本选视图：PSAVS

- Traffic2026/2027/2028时间隔离validation，fit/dev/audit间清除287个重叠窗口；离线决策树筛选不训练预测主干。
- 分歧感知raw树在audit的平均改善 `-0.143567%/-0.194586%`，0/3双胜；dev校准的主策略全部回退固定三路，正式增益0%/0%。
- label-aware oracle仍有MSE约0.302%–0.427%、MAE约0.574%–0.875%空间；它使用真实误差，不可部署。分歧特征需要三路都先完成，因此没有已证实调用节省。
- 结论：存在动作空间不等于当前特征可可靠预测动作；停止这组浅树/特征配置，不开启预算赌博机或在audit反复换选择器。
- 来源：`reports/stage22_33_psavs.{md,json,csv}`。该audit约0.74的MSE与全validation约0.52属于不同时间子集，不能跨表排名。

## 11. 证据质量回收：随机配对、路径诊断与基线纠错

- eval仍会随机TopM抽样，多视图前向改变后续original随机轨迹；历史checkpoint未保存训练末RNG。后续CRN-v1与公共初始化/独立随机流控制提供更干净归因，不能把随机协议变化算新创新。
- 冻结填补/融合诊断发现历史直达投影量级占优；关闭历史大幅退化，关闭检索只有微小收益；该冻结模型下的路径响应不是普遍因果定律，也不是新增模型结果。
- 最小距离标准化上游公式错误得到独立修复：Traffic2027/2028/2029相对原基线平均 `+1.8276%/+1.8847%`，仅2/3双胜，2028 `-0.1893%/-0.6310%`，全胜门失败。修复不算创新，2028根因尚未因果隔离。
- 新增原型代表索引问题尚未修复，22.44只是建议；不得在组会上报告“已修复全部bug”或“两个有效创新已成立”。
- 来源：`reports/stage22_34_crn_factorial.{md,json,csv}`、`reports/stage22_37_control.{md,json}`、`reports/stage22_38_diagnostic.{md,json}`、`reports/stage22_39_fusion.{md,json}`、`reports/stage22_41_distance_fix.{md,json}`、`reports/stage22_42_summary.md`、`reports/stage22_43_cnm_audit.{md,json,csv}`。

## 工件数与完整附录的正确统计口径

- `reports/latest_results.csv`有302条数据行（303行含表头），包含训练、只读前向、缓存复算、CPU诊断、smoke、失败、重试和条件跳过；302不是正式模型数、独立精度实验数、训练数或可排名结果数。
- 建议附录自动逐条列出persisted formal runs，不手抄指标：`experiment_id / model / dataset / seed / split / protocol_scope / type / status / MSE / MAE / baseline / config / metrics / checkpoint / report`。失败正式run也保留，NaN/缺指标显示NA。
- 分类先读取resolved config和metrics：排除 `__smoke__` 或配置中的smoke/reduced-batch预算；正式训练/完整固定集前向列入“精度或确认”，正式计时列“效率”，CPU缓存分析及标签感知Oracle列“诊断”。不能仅凭run名字、SUCCESS或存在MSE分类。
- 同checkpoint的多个输出视图、同缓存的多个候选和报告再汇总保留可追溯ID，但标注依赖同一来源，不计为新增独立实验证据。
- 仅在同dataset、同split、同固定资产、同目标范围、同指标实现、同checkpoint来源/训练协议的比较组内排序。test、full validation、audit子区间和smoke互不排名；S4M、旧PG95、新CRN与修复版跨协议不合并排名。

## 投稿证据缺口与最短研究路径

当前判断：可进入进一步研究，不足以保证SCI任何分区，也不建议把历史模块堆成论文。已有优势是全流程可核验、负结果透明、CNM在Traffic多个配对seed方向一致；关键缺口如下。

1. 主线统一：选择并冻结一个最终CNM定义（历史Weather双路、后续等计数三路不能混写），在可信基线上复核；PG95没有稳定增量，不能充当第二贡献。
2. 基线可信：先独立隔离已确证索引实现风险，默认可恢复原行为，单因素smoke→单seed同预算validation；修bug收益永远与创新分开。当前不能宣称全部上游风险已消除。
3. 两个有效贡献：现有第二机制均未通过冻结门。先确定是否确需双贡献论文；若需要，应根据CNM成立条件开发一个可关闭、独立有效、可消融的新机制，而非复活旧失败模块或把工程修复改名。
4. 新测试边界：Traffic旧test已经被查看，最近2027/2028/2029验证集也多次开发。新mask seed不等于新时间区间。方法/超参锁定后，用未用于方法选择的新时间区间、未开发的数据集或新缺失机制一次性确认；不能在现有test追调。
5. 跨数据集与缺失机制：ETTh1/Electricity视图塌缩为实质负证据；不能仅展示Weather/Traffic成功。至少确认相同最终方法在可信额外协议的适用条件，并保留失败边界。
6. 强baseline与效率：需在相同资产和预算下比较可信的直接缺失预测强方法；无需当前额外堆大量文献。CNM单路→三路是明确计算开销，补最新代码同卡串行端到端训练/推理/显存与参数证据；旧版本计时和理论前向数不能代替。
7. 稳定性与机制：n=3联合初始化/缺失mask波动只作有限描述；PG95对子机制的因果归因、完整训练历史可用性假设和指标选模差异要披露。正式结论不能依靠最好seed或仅MSE/MAE其中一项。

最短路线：**只修可信基线的单一确证实现问题 → 同版本CNM单seed公平快速复核 → 锁最终方法/资产/新test边界 → 少量强baseline与跨协议确认 → 若收益成立再补独立第二贡献或清晰定位为单一方法论文 → 最后扩统计与写作。** 本次只提出路线，不授权启动任何实验。
