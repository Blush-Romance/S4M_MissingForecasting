# Step 22.33 PSAVS 离线可行性协议

冻结日期：2026-09-07。已有全validation四动作指标及oracle探索已暴露；以下在任何选择器拟合/dev校准/audit结果之前冻结。本轮只做开发筛选，不能称独立检验。

## 1. 问题与四动作

PSAVS（Pareto-Safe Adaptive View Selection，帕累托安全自适应视图选择）检验：固定多尺度融合之外，是否能由输入缺失形态和预测分歧预测每种固定动作的MSE/MAE收益。

动作顺序固定：`on`（原始）、`ensemble_l5`（0.5原始+0.5 L5）、`ensemble_l20`（0.5原始+0.5 L20）、`multiscale`（0.5原始+0.25 L5+0.25 L20）。原始S4M/PG95参数与掩码资产保持冻结；本轮训练的是CPU选择器，不更新预测主干。

主比较与回退动作固定为multiscale。只允许当选择器预测一个替代动作对multiscale的MSE与MAE收益同时超过校准裕量时切换；这是经验风险控制，不是逐样本或分布外安全保证。

## 2. 数据与时间隔离

仅白名单Traffic seed2026/2027/2028的`validation_equal_rate_multiscale_samples.npz`、来源metrics/status/resolved config及固定mask。不得读取test路径、test指标或完整CSV。源cache必须通过原协议工件门、指标回算、样本顺序、掩码嵌套和等移除计数检查。

每seed 1648窗口，输入192步、预测96步，采用保守完整支撑长度288，两个阶段之间清除287窗口。三seed使用完全相同的索引区间（左闭右开）：

- fit（拟合）：`[0,576)`，576窗口；
- gap（隔离）：`[576,863)`；
- development（校准）：`[863,1055)`，192窗口；
- gap（隔离）：`[1055,1342)`；
- audit（留出审计）：`[1342,1648)`，306窗口。

池化三seed的fit拟合一棵共享树；校准仅用三seed的dev；audit仅在选择器、阈值、特征和checkpoint冻结后用于评分。不得跨seed把相同时间段一份用于拟合、另一份用于审计。按时间切分保护的是本轮选择器；原PG95的提前停止曾使用validation，故本轮audit仍非全项目未暴露数据。

## 3. 固定特征与算法

两个预注册版本共用算法、训练样本、目标和复杂度：

- `mask_only`（仅掩码，对照）：14维，包含全窗缺失率、转换率、最长缺失段/192、末端连续缺失长度/192、缺失段数/192、缺失位置归一化质心、末32/96步缺失率、六个连续32步块缺失率。全观测时质心为0。
- `disagreement_aware`（分歧感知，唯一主策略）：上述14维加`log1p(d5)`、`log1p(d20)`、`log((d5+1e-12)/(d20+1e-12))`，其中d5/d20是各嵌套预测相对原始预测的均方分歧，无真实目标输入。

不得将seed、样本索引、真实损失、oracle动作或未来目标作为特征。

使用一棵scikit-learn `DecisionTreeRegressor`（多输出回归树），固定`squared_error, max_depth=3, min_samples_leaf=64, random_state=2233, splitter=best`，不搜索模型结构、不剪枝调参。

预测六个量：三个替代动作各自相对multiscale的MSE/MAE收益。每种指标的归一化分母是**池化fit中multiscale该指标均值**；dev/audit不参与尺度拟合。各替代动作得分为预测的两项归一化收益的较小值，仅得分严格大于裕量的动作可用。取最大得分；并列按固定动作顺序；无合格动作回退multiscale。`enabled=false`严格返回multiscale。

裕量只在dev中选择，候选固定`[0,0.001,0.002,0.005,0.01, infinity]`。候选在dev至少2/3 seed双指标严格改善且平均两项均大于0才合格；合格中最大化两项平均相对收益的较小者，平分选更大裕量。均不合格用infinity（全回退）。同时保留未经校准的裕量0结果作机制诊断，不能根据audit将其改为主策略。

## 4. Oracle与成本

复算此前oracle：仅在真实MSE/MAE都不劣于original的动作中，最小化两项相对original损失的均值。额外报告audit上每项指标单独最小的严格理想界。它们均读取目标，只做参照，不能进入选择器输入或代替主门。

分歧感知策略实际需要全部三个预测，始终计3次前向，加CPU选择器时间；不能用返回on/E5/E20声称节省前向。mask-only在前向前可作决策，报告1/2/2/3逻辑调用预算，GPU端到端延迟本轮为NA。记录训练耗时、selector预测耗时、树节点数、CPU内存；GPU占用为0。

## 5. 工程检查与结果门

先合成smoke：完整fit→dev→冻结checkpoint→重载→audit→metrics链，精确公式、关闭回退、裕量边界、校准仅用dev、教师特征/标签隔离、错误资产拒绝、时间区间不重叠、checkpoint预测往返。CPU树没有反向传播/optimizer，明确记NA，不伪造GPU smoke。

完整实验通过工程检查后，唯一主策略`disagreement_aware`的已校准版本须满足：audit至少2/3 seed MSE和MAE同时严格优于固定multiscale，三seed平均两项相对改善均至少0.15%。不以mask-only/未经校准策略/oracle替代主门。

报告每seed audit绝对指标/相对变化/动作比例及两个策略的dev校准记录；固定四动作均列入对照。报告跨seed同位置动作一致率。附2000次、块长96的配对移动块自助区间，三seed同步采样；区间仅描述有限audit的不确定性，不充当独立样本显著性检验。

通过：`PSAVS_DEVELOPMENT_PASS_SEED2029_CONFIRMATION_AUTHORIZED`。后续需新seed2029及新时间/数据协议确认，随机seed变化本身不等于独立时间样本。失败：`PSAVS_FEASIBILITY_STOP`，本配置停止，不启动预算赌博机或在audit调参。

每次运行唯一ID，保存配置、命令、源码快照/哈希、环境、源工件哈希、split索引、全部选择与预测收益、树checkpoint、日志、资源、状态和结果。更新阶段报告与全局latest_results，全部失败记录保留。
