# S4M 缺失值时序预测项目计划（内部工作记录，非正式项目文档）

## 目标
取得并核验 S4M 官方代码，完成创新方案与相似工作的证据化审查；待用户确认后再生成正式项目文档、搭建环境与实现代码。

## 当前阶段
步骤22.43已完成（2026-10-06）：8单测、真实缓存smoke、正式CPU与独立60数值/141hash复核均PASS/退出0。修复后同一S4M检查点+固定多尺度CNM-SE三seed均双指标改善，逐seed降幅平均MSE0.9579816%/MAE0.6834906%；只是已有验证缓存复算，非新训练/泛化证据，3次推理不是等算力。旧22.41公式修复2/3判定保留；PG95及第二创新未成立，上游原型索引bug未修。原公式交叉诊断暂列备选、未启动。建议22.44独立可关闭索引修复及单seed验证，不同时加模块；本轮未实施。

## 用户可见的全局步骤编号

- [x] **步骤1：** 完成项目目录、专属规范、正式方案文档和最小工程初始化
- [x] **步骤2：** 完成 S4M 官方代码、独立环境、四个数据集和复现协议审计
- [x] **步骤3：** 完成基线适配、Weather CPU smoke、checkpoint 重载和确定性复测
- [x] **步骤4：** 完成官方完整架构 GPU smoke（官方 CUDA 扩展启用；完整训练、评估、checkpoint 保存与重载均通过）
- [x] **步骤5：** 完成 Weather 与 ETTh1 单 seed 快速正式基线（两项正式基线、checkpoint重载、指标落盘和自动汇总均SUCCESS）
- [x] **步骤6：** 完成有/无原型的 Gate H 中心失效诊断（两数据集×ID/缺失率漂移/缺失形态漂移，保留部分成立与负结果）
- [x] **步骤7：** 完成可关闭MIPP模块、三项正式筛选及四模型同协议PASS报告；PG95为唯一同时改善MSE/MAE的候选
- [x] **步骤8：** 完成可关闭FUSR及V1/V2/静态V3/lag96/短跨度HCUP验证；所有预声明方案均为可信负结果，停止反馈式FUSR且不触碰test搜索
- [x] **步骤9：** 基于步骤8负结果重定向创新迭代（completed：BMSA组件门禁、smoke和Weather seed2023同协议正式对照均完成；相对PG95 MSE/MAE分别改善0.4772%/0.1722%，进入复核与多seed门禁）
- [x] **步骤10：** 完成Weather多seed复核、原型有效秩诊断和SDRA双锚点筛选；BMSA均值不稳定，SDRA在seed2024/2025均未双指标优于PG95，按预注册门槛停止且不扩数据集
- [x] **步骤11：** 完成SDRA负结果收口、缺失形态诊断、MRBA实现与双锚点formal；覆盖机制PASS但seed2025精度反转，按预注册门禁停止且不补seed2023
- [x] **步骤12：** 完成validation-only错位诊断；两个seed稀有Q4均双指标退化，已终止mask-space分支
- [x] **步骤13：** 完成PG95+CFR组件、双GPU smoke与Weather seed2023正式对照；工件/协议PASS但MSE与MAE均退化，按预注册门停止且不扩第二seed
- [x] **步骤14：** 完成双seedvalidation嵌套遮挡风险诊断与等权双视图复评；风险门和双指标门均通过
- [x] **步骤15：** 完成默认关闭的CNM-SE正式test模块、GPU smoke及seed2024/2025公平对照；双seed双指标提升并通过扩展门
- [x] **步骤16：** 补Weather seed2023形成三seed正式证据；三seed冻结门通过
- [x] **步骤17：** 完成L=1/L=5/L=20块长与三路融合结构消融；融合机制成立，块长敏感性结论混合
- [x] **步骤18：** 完成严格等实际缺失率复核、同GPU串行推理计时和validation-only多尺度CNM-SE验证；双seed预注册门通过
- [x] **步骤19：** 冻结步骤18结构与权重，在未用于结构选择的新数据集上完成外部验证
- [x] **步骤20：** 在ETTh1按步骤5同协议训练PG95，并对新checkpoint一次性运行冻结的等计数多尺度CNM-SE正式test
- [x] **步骤21：** 建立Electricity固定缺失协议，先训练/评估S4M与PG95，再仅用validation验证预测扰动响应和冻结多尺度互补；只有validation门通过才运行一次正式test
- [ ] **步骤22：** 冻结并验证一个同时治理PG95的MSE/MAE冲突与预测视图塌缩的新模块；先做validation-only诊断、组件测试和smoke，再运行同预算快速正式对照（22.15已停止MRRA分支，22.16转入MCPR）

## 各阶段

### 阶段 1：官方项目获取与仓库审计
- [x] 读取 `/mnt/data/zza/code/AGENTS.md`
- [x] 完整克隆 S4M 官方仓库并记录 commit
- [x] 检查项目内 AGENTS、README、依赖、数据、训练与评估入口
- [x] 判断当前可复现性与环境要求
- **状态：** complete

### 阶段 2：创新想法与相似工作审查
- [x] 精确定义问题、核心假设和两个主创新点
- [x] 检索最相近工作并区分已有模块与真正增量
- [x] 评估创新强度、实现风险、预期收益与投稿水平
- [x] 给出可讨论的取舍方案
- **状态：** complete

### 阶段 3：用户确认节点
- [x] 与用户讨论并修订创新方案
- [x] 获得用户对最终方案的明确确认
- **状态：** complete

### 阶段 4：正式项目文档与初始化
- [x] 在用户确认后生成完整中文项目文档
- [x] 为英文专业名词和缩写提供中文解释
- [x] 创建独立 Conda 环境并取得官方代码
- [x] 创建最小工程目录和参考文献资产
- **状态：** complete

### 阶段 5：基线复现与创新实现
- [x] 完成 Weather 最小 CPU smoke、checkpoint 重载和确定性复测（对应全局步骤3）
- [x] 完成官方完整架构 GPU smoke（对应全局步骤4）
- [x] 完成 Weather/ETTh1 快速正式基线（对应全局步骤5）
- [x] 逐个实现可关闭创新模块并公平对照（对应全局步骤7–9；PG95与BMSA正向，FUSR分支负向并已按验证证据停止）
- [ ] 固化评估资产、实验记录和结果汇总（贯穿步骤5–10）
- **状态：** in_progress

### 阶段 6：步骤10正式扩展
- [x] 冻结BMSA50、训练协议、seed集合和每个seed的固定mask资产
- [x] 为新增seed生成PG95/BMSA成对配置，完成smoke、配置公平性和队列自测
- [x] 后台运行Weather配对多seed复核，自动汇总均值、标准差、胜率和配对统计
- [x] 根据多seed证据决定是否扩展：当前证据不稳定，暂不扩ETTh1、其他缺失率和正式消融
- [x] 诊断Top-M稳定性选择的跨seed方差，设计预算不变的稳定性—多样性联合准入并先做快速对照
- [x] 冻结首个联合准入规则后完成CPU组件测试、GPU smoke、seed2024/2025同协议formal与自动比较；两个formal最终SUCCESS，但均未双指标优于PG95，SDRA候选停止
- **状态：** complete

### 阶段 7：步骤11缺失形态均衡原型记忆
- [x] 修复SDRA负结果报告器，分离协议完整性、机制有效性和精度决策
- [x] 仅使用train/validation输入mask诊断缺失形态几何；内容误差样本未被既有formal持久化，不为追做诊断先改test口径
- [x] 诊断支持后冻结MRBA设施选址覆盖写入规则、独立开关、机制门和双锚点精度门槛
- [x] 完成MRBA组件回归、配置公平性、队列自测和两个GPU smoke；smoke机制指标为正
- [x] 完成seed2024/2025同协议formal与冻结门禁比较；工件、协议、执行和覆盖机制均PASS，但仅seed2024双指标胜出
- [x] 执行预注册停止规则：seed2025双指标退化，因此不补seed2023、不扩数据集、不直接进入MRCR
- **状态：** complete

### 阶段 8：步骤12机制—精度错位诊断
- [x] 使用冻结PG95/MRBA checkpoint，仅在validation上保存逐样本误差与历史mask描述子
- [x] 检查MRBA收益是否稳定集中在高缺失率、稀有形态或高误差尾部，并要求seed2024/2025同向
- [x] 按预注册门执行停止决策：两个seed稀有Q4均双指标退化，终止mask-space分支并回到PG95主线
- **预注册主门：** 训练形态稀有度按每个validation窗口六个32步mask块在train候选中的平均平滑负对数频率定义；主指标固定为完整validation目标的逐样本MSE/MAE。两个seed的最稀有四分位均须同时改善两指标，且其收益均大于最常见四分位。已观测未来误差、缺失率、转换率、最长缺失段和PG95误差尾部只作次级解释，不替代主门。
- **状态：** complete

### 阶段 9：步骤13反事实安全原型学习
- [x] 冻结CFR定义：PG95原型写入和预测协议不变，训练期用同一检索产生prototype-on/off上下文
- [x] 实现默认关闭的CFR开关；off分支停止梯度，模型仍以on分支作为训练输出和推理输出
- [x] 加入逐样本有害检索后悔损失、机制指标和无重复memory更新回归测试
- [x] 完成CPU组件测试、基线关闭等价GPU smoke、CFR GPU smoke和Weather单seed正式公平对照；队列3/3 SUCCESS
- [x] 执行预注册精度门：CFR相对PG95的MSE/MAE均退化，故不运行后续validation机制诊断、不扩第二seed、不依据test调损失权重
- **冻结损失：** `forecast_mse + mean(relu(sample_mse_on - stopgrad(sample_mse_off)))`；只使用训练目标已观测位置，off预测不得接收梯度，附加项无可调权重，测试时完全关闭反事实分支。
- **状态：** complete

### 阶段 10：步骤14跨掩码预测分歧风险诊断
- [x] 审计并复用步骤12冻结validation缓存入口，新增原始PG95、prototype-off和确定性嵌套遮挡PG95三路只读推理
- [x] 固定嵌套遮挡：目标新增缺失率15%、长度5因果块，起点概率按`1-(1-0.15)^(1/5)`推导；只从原历史已观测行继续遮挡，不得读取未来或test
- [x] 保存逐样本完整目标误差、跨掩码预测分歧、prototype-on/off差异、固定mask与样本索引，并完成工件配对门
- [x] 在seed2024/2025上执行预注册风险门：Spearman≥0.15、Q4/Q1 MSE≥1.25、Q4/Q1 MAE≥1.10，两个seed均通过
- [x] 固定0.5/0.5双视图并补validation MSE/MAE；两个seed均超过MSE 1.5%、MAE 0.5%改善门
- **范围：** checkpoint冻结、validation-only、无训练、无test样本；prototype差异仅作次级解释。
- **状态：** complete

### 阶段 11：步骤15 CNM-SE双seed正式测试
- [x] 实现默认关闭的因果嵌套掩码自集成入口；原始/嵌套视图固定0.5/0.5，不新增参数
- [x] 完成配置负测、决策负测、GPU smoke、checkpoint和固定test mask审计
- [x] 完成seed2024/2025同checkpoint同test资产对照；两个seed均同时改善MSE与MAE
- [x] 通过预注册扩展门：两seed平均MSE/MAE改善`1.8748%/0.7206%`
- **状态：** complete

### 阶段 12：步骤16 CNM-SE三seed与结构消融
- [x] 接入现有PG95 seed2023 checkpoint、固定mask与完整test配置
- [x] 冻结三seed门：每个seed双指标提升，平均MSE≥1.0%、MAE≥0.5%
- [x] 完成seed2023资产、配置、编译、队列resume和决策门禁
- [x] 启动并完成seed2023正式CNM-SE test；保留seed2024/2025成功run不重复执行
- [x] 三seed冻结门通过：平均MSE/MAE改善`2.2197%/0.9779%`，三个seed均双指标提升
- [x] 完成预注册L=1/L=5/L=20因果块长及original/nested/equal-ensemble融合结构消融；禁止会加回隐藏真值的泄漏式“非嵌套”对照
- **状态：** complete

### 阶段 13：步骤18等实际缺失率与多尺度CNM-SE
- [x] 冻结并实现逐样本等新增缺失计数的L=1/L=5/L=20因果嵌套mask资产；只在validation验证
- [x] 完成新mask生成器组件测试、配置负测和独立GPU smoke
- [x] 在同一张GPU上串行运行L=1/L=5/L=20，保存warmup后推理延迟、峰值显存和固定顺序
- [x] 预注册并验证L=5+L=20多尺度CNM-SE；固定权重`0.5/0.25/0.25`，未接触test
- [x] 自动汇总公平性、精度、复杂度与是否值得进入新数据集正式验证的结论
- **状态：** complete；决策=`MULTISCALE_VALIDATION_GATE_PASS_EXPAND_ONLY_TO_NEW_PROTOCOLS`

### 阶段 14：步骤19冻结多尺度CNM-SE跨数据集外部验证
- [x] 审计ETTh1/ETTm1/Electricity现有完整数据、缺失资产、基线/PG95 checkpoint与协议可比性
- [x] 在读取新数据集结果前冻结首轮数据集、seed、缺失形态、主比较和继续/停止门
- [x] 复用步骤18等计数多尺度入口并参数化数据集约束，保持Weather行为回归不变
- [x] 完成组件、配置负测、GPU smoke和独立后台队列
- [x] 运行首轮跨数据集正式外部验证并自动汇总精度、稳定性和复杂度
- **状态：** complete；决策=`EXTERNAL_TRANSFER_FAIL`（原始S4M来源上的通用迁移失败）

### 阶段 15：步骤20 ETTh1完整PG95+多尺度CNM-SE事后外部验证
- [x] 恢复全部项目证据链，明确步骤20是步骤19之后提出的post-hoc（事后）新假设，不改写步骤19结论
- [x] 冻结ETTh1 seed2023、步骤5数据/固定mask/训练预算，以及Weather PG95的全部模块超参数
- [x] 完成ETTh1 PG95训练配置、冻结评估配置的依赖交接、静态负测与GPU smoke
- [x] 完成ETTh1 PG95正式训练，并在其最佳checkpoint上一次性运行等计数L5/L20多尺度CNM-SE完整test
- [x] 比较PG95对步骤5 S4M、PG95+多尺度对同run PG95原始路、PG95+多尺度对L5双路的MSE/MAE与资源代价
- **状态：** complete；决策=`FULL_METHOD_FAIL`

### 阶段 16：步骤21 Electricity外部数据validation先行验证
- [x] 审计Electricity完整CSV、论文/代码协议与现有数据适配器，冻结数据划分、模型规模、优化器、学习率、固定mask和评价资产
- [x] 生成Electricity seed2023 point-r0.06、L=5固定缺失CSV/NPZ/metadata并完成哈希、观测值、划分与实际缺失率审计
- [x] 为S4M与PG95新增独立训练smoke和完整训练配置；训练阶段不得访问test指标进行模型选择或模块调整
- [x] 用冻结checkpoint在chronological validation上评估original、等计数L5/L20及固定`0.5/0.25/0.25`多尺度，保存预测分歧、精度和串行延迟
- [x] 预注册继续门：只有PG95相对S4M的validation双指标不退化且至少一项严格改善，并且PG95多尺度相对PG95 original与L5双路均双指标严格改善，才自动生成并执行一次正式test
- [x] 通过独立smoke后使用UUID后台动态队列运行，最大并发5、OOM退避回队列、全部工件与自动汇总落盘
- **状态：** complete；协议审计=`PASS`，科学决策=`FULL_VALIDATION_FAIL_TEST_FORBIDDEN`，条件test按预注册门控未授权且未访问

### 阶段 17：步骤22 验证集驱动的误差协调与视图多样性创新迭代
- [x] 完整恢复既有负结果、步骤21逐样本/逐视图工件和当前可关闭模块接口，避免重复已否定路线
- [x] 在不访问test的前提下，诊断PG95相对S4M的MAE退化来源及L5/L20预测塌缩位置
- [x] 在读取新实验结果前冻结新模块定义、训练/推理预算、主比较、继续门和停止门
- [x] 实现默认关闭的独立模块、完整配置记录和基线关闭等价回归
- [x] 通过MRRA-v3 CPU组件测试、配置负测、checkpoint保存/加载静态回归与调度器自测
- [x] 通过默认关闭PG95 GPU回归smoke及MRRA-v3跨旧故障窗口稳定性GPU smoke
- [x] 使用最大并发5的UUID动态队列运行validation-only同预算快速正式对照并自动汇总
- [x] 执行冻结双指标门：MRRA-v3因MSE退化0.6460%而失败，不进入多seed，保留负结果并迭代新假设
- [x] 步骤22.14复用冻结checkpoint构建MRRA-v3完整chronological validation逐样本/预测步/变量误差与掩码响应缓存
- [x] 将MRRA-v3与既有PG95缓存严格配对，完成指标回算、难度分位、时间块自助法和可观测响应相关诊断
- [x] 步骤22.15缓存21个MRRA/缺失结构/原型检索内部特征并执行fit/development/audit冻结筛选；fit/development候选规则数为0，按预注册停止MRRA精度分支
- [x] 步骤22.16实现MCPR（Mask-Conditioned Pareto Residual，掩码条件帕累托残差）：在PG95预测主干之后加入显式掩码条件低秩残差，并只训练新模块
- [x] 用MSE-only架构对照与Pareto common-descent双目标版本分离“后主干掩码注入”和“冲突梯度治理”贡献，组件/smoke/完整validation全部完成
- [x] 执行预注册双指标门：MSE-only与Pareto版本均未达到MSE/MAE双非劣，停止MCPR分支，不进入多seed、等预算续训对照或test
- [x] 步骤22.17对冻结PG95、MCPR-MSE和MCPR-PCD做严格配对的validation-only失败归因：闭环复算总指标，分解残差对齐项与能量惩罚项，并报告样本/预测步/变量/缺失结构/尾部误差归因
- [x] 步骤22.17通过CPU分析器单测与2-batch GPU smoke，再使用UUID动态队列生成两个完整2528样本MCPR缓存并与既有PG95缓存严格配对；全程不训练、不访问test、不调MCPR参数
- [ ] 步骤22.18在任何新实验结果前预注册新机制：停止PCD路线，对MSE残差引入训练期信赖域缩放与尾部风险约束；确证性首轮使用未参与步骤22.17归因的新固定缺失资产，不能直接使用validation oracle缩放0.219
- [x] 步骤22.18.1–22.18.2完成上下文恢复、失败机制复核与方案冻结；正式命名为TTRR-MCPR，冻结seed2024交叉掩码资产、八维无标签信赖门、最坏10%相对主干正伤害约束、权重1.0和双锚点继续门
- [x] 步骤22.18.3生成并审计Electricity seed2024固定point-r0.06-L5资产，确认其与seed2023的train/validation mask均不同
- [x] 步骤22.18.4实现默认关闭的TTRR模块、训练目标、参数校验、指标和基线关闭等价回归
- [x] 步骤22.18.5完成零GPU测试与PG95/MCPR-MSE/TTRR三项GPU smoke；三项均首次SUCCESS，最大并发5的UUID动态队列已真实启动并自动放行三项formal validation
- [x] 步骤22.18.6自动汇总双锚点精度与机制，并严格执行继续/停止门；TTRR未胜MCPR-MSE控制，决策为`TTRR_CONFIGURATION_STOP`，不进入test
- [x] 步骤22.18.7修复成本汇总字段，并在同一GPU上串行复核PG95/MCPR-MSE/TTRR的完整validation前向延迟与指标回算；6/6任务首次SUCCESS，所有协议检查PASS
- [x] 步骤22.18科学决策为`TTRR_CONFIGURATION_STOP`，复杂度和同卡计时证据已闭环，不进入test
- [x] 步骤22.19.1仅使用既有seed2023/seed2024工件诊断MCPR-MSE跨掩码符号翻转，并检验逐样本解析最优残差缩放的分布、上界与可蒸馏性
- [x] 步骤22.19.2在读取任何新seed逐样本结果前冻结ARD-MCPR（Analytic Residual Decision Distillation，解析残差决策蒸馏）的训练教师、离散—连续混合推理门、公平预算、seed2025新掩码资产与双锚点停止门
- [x] 步骤22.19.3实现默认关闭的ARD模块与训练目标，完成组件测试、配置负测、checkpoint往返和基线关闭等价回归
- [x] 步骤22.19.4生成未参与方法设计的seed2025固定缺失资产，先smoke再以UUID动态队列运行PG95/MCPR-MSE/ARD-MCPR的同预算validation快速确认；6/6任务首次SUCCESS，无OOM/重试/失败
- [x] 步骤22.19.5自动汇总精度、解析教师拟合、门分布、参数/训练耗时/显存，并按双锚点门决定停止、迭代或进入多seed；ARD胜MCPR控制但MSE未胜PG95，决策为`ARD_BEATS_MCPR_CONTROL_BUT_NOT_PG95_STOP`
- [x] 步骤22.20只用既有训练/validation工件诊断“高方差三动作教师→近常数学生门”的信息瓶颈，冻结下一版结构后必须用新seed资产确认；不补ARD-v1多seed、不访问test
- [x] 步骤22.20.1在任何新增逐样本诊断结果前冻结validation-only缓存字段、时间隔离fit/dev/audit、候选结构特征族、判别指标和停止规则
- [x] 步骤22.20.2实现默认只读的ARD逐样本诊断入口，缓存当前8维特征/动作概率/门、训练式与完整目标式解析教师、时序分块和变量分布结构描述；完成CPU/配置/checkpoint/smoke门禁
- [x] 步骤22.20.3以UUID动态队列先smoke再完整缓存2528个validation样本，不更新模型或memory，不访问test
- [x] 步骤22.20.4在冻结的时间隔离切分上比较当前8维特征与结构化特征的动作/alpha可分性及反事实MSE上界，据证据冻结下一版结构或停止ARD路线
- [x] 步骤22.21在全新Traffic数据协议上验证当前唯一稳定正向主线PG95+多尺度CNM-SE；validation与唯一test均双指标确认多尺度收益
- [x] 步骤22.21.1审计Traffic完整数据、上游S4M参数和现有数据加载器，在任何Traffic结果前冻结seed2026缺失资产、训练预算、validation门和test边界
- [x] 步骤22.21.2生成并审计Traffic固定缺失资产，实现/复用smoke→S4M/PG95 formal→validation多尺度→条件test队列
- [x] 步骤22.21.3执行validation门并按冻结结果授权唯一test；最终决策为Traffic多尺度确认并进入多seed扩展
- [x] 步骤22.21.4审计两项formal的NaN终止，排除数据非有限、空目标mask、零尺度归一化、OOM和调度遗漏；保存结构化失败诊断
- [x] 步骤22.21.5将两臂Adam学习率共同从0.01修订为上游默认0.001；两项64-batch扩展稳定性smoke均完成63次optimizer step并SUCCESS，已以新ID重启formal依赖链
- [x] 步骤22.21.6两项formal和完整多尺度validation均SUCCESS且全部精度门通过；唯一test因分析器遗留旧lr0.01常量而被安全跳过，test未访问
- [x] 步骤22.21.7修复分析器预算常量并增加旧学习率/缺失修订身份/epoch漂移负例；唯一test完成213 batch/3408样本并双指标确认多尺度收益
- [x] 步骤22.22冻结Traffic seed2027/2028资产和完全相同协议，运行S4M/PG95/多尺度复核并汇总三seed统计稳定性
- [x] 步骤22.22.1审计现有seed2026资产生成、训练、评估与队列接口；在任何新seed资产/指标前冻结三seed统计协议、测试访问规则和停止条件
- [x] 步骤22.22.2生成seed2027/2028固定缺失CSV/NPZ/JSON并审计哈希、可重生性、跨seed差异、split覆盖和观测值一致性
- [x] 步骤22.22.3生成两seed的S4M/PG95 smoke与formal配置、冻结评估配置、统一最大并发5队列和三seed自动汇总器
- [x] 步骤22.22.4完成零GPU组件/配置/负例/历史回归/调度器自测，记录GPU UUID和显存后启动后台队列
- [x] 步骤22.22.5完成两seed正式训练与冻结validation/test评估，汇总三seed均值、样本标准差、逐seed配对改善率、参数量、显存和推理开销
- [x] 步骤22.23以validation-only规则筛选固定两路高效CNM-SE，并在通过后用全新seed2029确认准确率—效率权衡
- [x] 步骤22.23.1在读取三seed两路候选汇总前冻结候选、80%收益保留门、test事后边界、全新seed确认和磁盘约束
- [x] 步骤22.23.2只读汇总E1/E5/E20的三seedvalidation，冻结唯一候选；随后仅作已有test探索性复核
- [x] 步骤22.23.3按停止门未授权：没有候选通过validation门，因此不生成seed2029资产、不建立训练队列
- [x] 步骤22.23.4按停止门未授权：没有冻结候选，因此无需新seed确认
- [x] 步骤22.24保持三路CNM-SE数学定义不变，将original/L5/L20合并为一次batch并行前向并验证等价性与同卡墙钟收益
- [x] 步骤22.24.1在读取任何新计时结果前冻结数值等价容差、同批次交替顺序、速度门和显存边界
- [x] 步骤22.24.2实现默认关闭的batch并行比较入口、配置负测、CPU静态测试和GPU smoke
- [x] 步骤22.24.3完成Traffic seed2027完整validation同GPU配对正式计时与自动汇总；速度门PASS但严格等价门FAIL
- [x] 步骤22.24.4按等价停止门未授权：不运行seed2028、不放宽数值容差
- [x] 步骤22.25在每路batch size不变的前提下，以三个CUDA stream并发original/L5/L20，验证能否同时满足严格等价与速度门
- [x] 步骤22.25.1在新GPU结果前冻结多流同步、同卡交替计时、等价/速度门和科学硬依赖
- [x] 步骤22.25.2实现默认关闭的CUDA多流入口与“smoke等价失败则formal不启动”队列门
- [x] 步骤22.25.3通过静态/负例/调度测试后运行seed2027 smoke；仅等价PASS才运行完整formal
- [x] 步骤22.25.4按正式速度停止门未授权：不运行seed2028
- [x] 步骤22.26将训练期三视图CNM-SE的稳定增益蒸馏到推理期单视图学生，验证能否兼顾PG95级成本与CNM-SE精度（seed2027双指标改善但未达50%收益保留门，当前版本停止）
- [x] 步骤22.26.1只用既有train/validation证据审计可复用残差头与教师生成接口，在新训练结果前冻结模块、损失和停止门
- [x] 步骤22.26.2实现默认关闭的因果多尺度收益蒸馏模块、组件/配置/checkpoint测试和GPU smoke（v1工程链路通过，但整样本获益0/64，科学门阻断formal）
- [x] 步骤22.26.3尝试Traffic seed2027同checkpoint、同预算validation快速对照；v1因smoke创新项未激活而在formal前停止，不访问test
- [x] 步骤22.26.4基于v1失败证据验证时间单元级BF-CMVD-v2；机制smoke与seed2027完整validation均完成
- [x] 步骤22.26.5按预注册门禁未授权：v2仅保留MSE/MAE教师收益48.07%/44.16%，不运行seed2028
- [x] 步骤22.26.6按双validation前置条件未授权：不生成seed2029、不访问test
- **状态：** complete；BF-CMVD-v2相对PG95双指标改善，但未达到双指标各保留≥50% CNM-SE收益的门禁，当前版本停止并保留完整负/部分正结果。
- [x] 步骤22.27对冻结的BF-CMVD-v2做validation-only失效归因，区分监督覆盖错配、学生表达不足和非获益区域副作用，再只冻结一个证据最强的v3改动
- [x] 步骤22.27.1在读取新逐单元诊断结果前冻结数据边界、配对张量、误差分解、判定顺序和停止规则
- [x] 步骤22.27.2实现冻结checkpoint只读诊断入口、CPU合约/配置负测、checkpoint身份与test禁用检查
- [x] 步骤22.27.3先运行小批量GPU smoke；工件闭环后再运行完整Traffic seed2027 chronological validation诊断
- [x] 步骤22.27.4自动汇总覆盖重合、残差对齐、获益区欠拟合、非获益区伤害与逐预测步证据，只选择一个主因
- [x] 步骤22.27.5仅当主因证据清楚且可由推理时可观测量处理时冻结BF-CMVD-v3；本步不训练v3、不扩seed、不访问test
- **状态：** complete；协议与formal工件PASS，主因=`student_expression_underfit`，唯一v3候选冻结为DMRB（双尺度掩码条件残差基），本步未训练v3、未扩seed、未访问test。

## 关键问题
1. S4M 官方仓库当前是否包含足以还原论文主要表格的代码、依赖和数据说明？
2. 对比学习、频谱最优传输和元适应中，哪两个构成最紧凑且非堆模块的研究主线？
3. 相似工作是否已经覆盖同样的模块位置、损失和评估协议？
4. 在尚无实验结果前，应将目标水平校准到何种区间？

## 已做决策
| 决策 | 理由 |
|------|------|
| 主干优先采用 ICLR 2025 S4M | 与缺失输入下端到端预测直接匹配，且有官方代码 |
| 用户确认前不生成正式项目文档或创新代码 | 避免在研究问题尚未收敛时提前固化工程方案 |
| 官方仓库与本地扩展分层保存 | 保护原始基线，便于复现和公平比较 |
| 最终主线采用 MIPP + FUSR，不以普通对比学习、频谱最优传输或强化学习为核心 | 两个模块分别治理 S4M 原型供给和使用，均直接作用于缺失域漂移下的同一失效链 |
| Gate H后仍先做MIPP，不先扩FUSR多seed | Weather存在约5%稳定Oracle选择空间，但ETTh1原型支路数值失活；先增强跨掩码稳定且可用的原型供给，FUSR再治理逐样本读取更符合两数据集证据 |
| 步骤7新增两个Weather单seed变体，不增加新seed | 用户要求多跑两个；优先测试更严格写入门槛与去除内容-缺失因子化，能直接判断MIPP收益来自净化门控还是辅助损失，且不提前铺开统计实验 |
| 2026-07-18起实验调度最大并发改为5 | 用户当前明确授权；仍以实时显存、历史峰值和安全余量决定实际并发，不通过阉割配置凑满5个 |
| 步骤8先独立验证FUSR，再与PG95组合 | 独立贡献优先可解释；PG95是步骤7唯一同时改善MSE/MAE的MIPP候选，组合实验留到步骤9 |
| 步骤8的FUSR V2先修路由塌缩，不扩数据集或多seed | V1测试gate全1、AUROC 0.4946且训练正类97.98%；先用平衡符号/排序监督、去除永久prior并做validation校准，才能判断选择性读取本身 |
| 步骤8的FUSR V3先做现有缓存的严格时间块诊断，再决定实现 | V2 fit AUROC=0.766但dev/calibration/test降至0.337/0.379/0.474，主故障是时间泛化而非工程或阈值；先用零GPU离线证据筛选时间组稳健方案，优于盲目重跑 |
| 步骤9首候选采用BMSA预算匹配稳定性准入 | PG95每批约96候选、固定0.95门控接受99.91%，随后原始memory仍随机抽M=50；BMSA复用同一M=50直接取稳定性Top-M，不增加写入预算或未来标签，能把无效绝对阈值变成相对尺度自适应选择 |
| 步骤10不直接扩展现有BMSA50 | Weather三seed中BMSA双指标胜出2/3，但一次反转幅度超过两次收益，三seed平均MSE/MAE均略差于PG95且置信区间跨0；先治理Top-M稳定性选择的跨seed方差，比继续铺ETTh1和更多缺失率更符合预注册门槛 |
| 步骤10后续先做checkpoint机制诊断 | 只比较六个已完成PG95/BMSA checkpoint中的原型库多样性、有效秩、槽位覆盖和查询覆盖，不使用test目标来选择规则；诊断用于检验“硬Top-M造成冗余”的机制假设，而不是事后挑指标 |
| 联合准入首候选采用无参数稳定性—多样性秩融合 | 每轮先选稳定性最高候选，随后把稳定性百分位秩与相对已选集合的最小内容距离百分位秩等权相加，贪心选满M=50；两目标先转秩以消除尺度，等权固定且不依据test调节，不新增参数或未来标签 |
| 步骤12后终止mask-space创新分支 | 冻结checkpoint的validation-only预注册诊断中，seed2024/2025最稀有Q4相对PG95的MSE和MAE均退化；这否决了“总体均值掩盖罕见形态收益”的解释，不再继续掩码条件检索或调MRBA |
| 步骤13停止CFR候选且不扩第二seed | CFR工件与公平协议均PASS，但Weather seed2023相对PG95的MSE/MAE分别退化0.5554%/1.4330%；训练有害检索率仅4.41%、正后悔量很小，未形成可支持一次事后调权重的机制证据 |
| 步骤14主信号冻结为嵌套遮挡预测分歧 | 它完全由历史输入产生、无需学习效用路由器，可直接检验“缺失扰动敏感性是否刻画预测风险”；主门在读取任何逐样本结果前固定为双seed Spearman≥0.15、Q4/Q1 MSE≥1.25、MAE≥1.10 |

## 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| 2026-08-21宿主复合状态查询的自动权限审核超时 | 1 | 终止未执行的只读cell，改用沙箱内直接读取已落盘state/report/metrics；未影响实验或后台进程 |
| 先前会话 GitHub TLS 握手失败 | 2 | 本轮更换获取方式并在失败后保存明确阻塞证据 |
| 官方 Conda 环境用默认求解器创建时会话异常结束，未生成环境 | 1 | 改用 libmamba 求解器，仍保持官方 YAML 不变 |
| 原样环境创建后 pip 阶段未完成 | 1 | 用 Conda 自动生成的官方 pip 清单续装，最终退出码 0 |
| 官方 README 命令直接运行出现 `No module named 'S4'` | 1 | 已确认加入 `s4m/model` 到 PYTHONPATH 后可导入；正式修复待用户确认后实施 |
| `nature-polishing` 清单所列 `_shared` 静态核心目录缺失 | 1 | 记录技能包资源缺口，改用该技能现有本地核心和匹配片段继续完成文档 |
| `web.find` 一次包含 4 个 PDF 关键词时返回 `SyntaxError` | 2 | 改为每次最多 3 个关键词，后续查询成功 |
| 项目外层目录不是 Git 仓库，无法生成项目级 `git diff --stat` | 1 | 保留上游 Git commit/洁净状态，并记录新文档 SHA-256；后续初始化本地项目版本库前先确认追踪边界 |
| 默认隔离环境中 `nvidia-smi` 退出码9、PyTorch可见GPU为0 | 3 | 宿主权限下确认CUDA正常；后续GPU核验/启动使用宿主执行和稳定UUID，不能把隔离失败误报为驱动故障 |
| PyTorch 2.1 的 CUDA device properties 没有 `uuid` 属性 | 1 | 用宿主 `nvidia-smi` 获取UUID，再用 `CUDA_VISIBLE_DEVICES=<UUID>` 与设备名称/总显存交叉验证 |
| GPU UUID 导出加入 runner 时字符串拼接漏写 `+`，静态编译失败 | 1 | 立即修正表达式，并在启动任何实验前重新执行语法和 diff 检查 |
| 步骤6 checkpoint形状探测的`python -c`误传字面量换行转义，静态SyntaxError | 1 | 改为单行列表推导式读取，不重复使用失败的转义写法 |
| 读取ETTh1已结束PID的`/proc/.../environ`返回文件不存在 | 1 | 以调度状态、run工件和live NVML快照为准，不把正常进程退出误报为环境读取故障 |
| Gate H漂移生成器用`np.array_equal`检查CSV观测值导致失败 | 1 | 先量化CSV往返误差，再采用有记录的数值容差；训练mask恒等仍保持精确检查 |
| Weather缺失率漂移正式配置的checkpoint文件名误写为漂移数据名 | 1 | 启动前全路径校验准确拦截；改回步骤5原始Weather checkpoint并重新全量校验，不产生实验run |
| Gate H队列组合校验的独立导入未加入`scripts/`搜索路径 | 1 | 核心self-test已PASS；补`PYTHONPATH=scripts`重跑队列定义检查后再启动正式队列 |
| 非宿主`ps`看不到Gate H后台PID，被误判为外部终止 | 1 | 宿主复核确认原调度器和任务始终存活；未重复启动。后续后台真值只用宿主PID+状态+日志，新增EXTERNAL_KILL恢复测试仅作安全兜底 |
| 两个ETTh1漂移文件名为小写，官方大小写敏感逻辑未设置`d_var` | 1 | 在可修改基线中对未知文件名回退到`enc_in`并记录日志；先新增2-batch ETTh1 smoke，再只重排两项失败任务 |
| 修复后nohup重排进程在写入`scheduler_started`前消失 | 1 | 总日志无Traceback且无新事件；不重复同一方式，改用tmux持久会话承载前台调度器并传入相同reset任务/原因 |
| 步骤7读取Weather smoke配置时猜错文件名 | 1 | 未修改任何文件；改用`rg --files configs/baseline`定位真实配置名，不重复猜测路径 |
| MIPP首次GPU smoke的长度20块掩码触发`max_pool1d padding`非法 | 1 | 保留CODE_ERROR run；改为显式左填充与无内部padding，新增长度20组件测试后用新ID重排，第二次smoke SUCCESS |
| S4M内部把外部GPU UUID覆盖为逻辑0，导致调度选4080但实际运行在4090 | 1 | 单标量UUID探针定位到`Exp_Basic._acquire_device`；后续启动保留外部UUID并使用局部`cuda:0`，不打断已运行正式任务 |
| strict-gate新增smoke在创建checkpoint目录时报`OSError 36: File name too long` | 1 | 保留训练前CODE_ERROR run；仅缩短experiment model、model_id和des身份字符串，模型/数据/预算参数不变，重新执行路径长度门禁后再重排 |
| pair-gate-only smoke在配置修复生效前也读取到长身份并报相同`OSError 36` | 1 | 保留第二个训练前CODE_ERROR run；缩短后的四个setting均为240字符，路径长度门禁PASS，两个smoke统一用新ID重排 |
| full-architecture Weather GPU smoke 在约4914 MiB启动空闲显存下 OOM | 1 | 保留失败记录，模型和batch不变；初始估计约9 GiB，后台等待更多显存后再profiling，OOM则递增估计和退避 |
| smoke 审计子 shell 未激活环境，`python: command not found` | 1 | 后续审计使用 `s4m_missing_forecasting` 环境内 Python 绝对路径；正式 runner 内验证未受影响 |
| 相同 seed 的两次 CPU smoke 指标不完全一致 | 2 | W&B 后重新 seed 仍不一致；已定位 Bio.Cluster 隐藏随机多起点，新增默认关闭的确定性 best-of-100 复现开关并待双次验证 |
| 为并发上限新增常量时首次patch猜错现有`QUEUE_NAME`文本 | 1 | 未改动文件；先读取脚本顶部和目标行，随后按真实`stage5_weather_etth1_single_seed_baselines`上下文应用补丁，不重复猜测 |
| 步骤8 V2启动检查再次在外层项目执行`git status`返回非Git仓库 | 1 | 外层本来只作项目容器；后续改用`git -C baselines/S4M`与`git -C upstream/S4M`分别核验可编辑仓库和上游快照 |
| 步骤8 V2源码定位命令的brace路径未被当前shell展开 | 1 | 未修改任何文件；后续直接列出真实路径或在仓库根目录分别执行`rg` |
| 记录上述两项问题的第一次补丁因锚点文本不完全一致而安全失败 | 1 | 先读取文件真实尾部，再以现有最后一行作精确锚点追加，未覆盖既有记录 |
| 步骤8 V2数据加载器定位时再次误用brace路径 | 1 | 命令只读且其他并行检查成功；后续完全停止使用brace路径，逐文件传入`rg` |
| 步骤8检查基线配置时猜错Weather YAML文件名 | 1 | 未修改任何文件；用`rg --files configs/baseline`定位真实`...point_r006_paper_adam...`配置，并直接复用成功run的resolved config/checkpoint |
| 步骤8合并审计第一次从项目外层读取`s4m/model`相对路径失败 | 1 | 命令只读；立即改到`baselines/S4M`工作目录重读，源码未受影响 |
| V2静态检查直接调用系统`python`无执行权限 | 2 | 两个独立子审计均改用项目Conda解释器绝对路径，正式runner也固定使用该环境 |
| V2协议快照首次对未解析源YAML读取runner注入路径时报`KeyError` | 1 | 改用步骤5 resolved config并模拟V2 runner解析后重检，主干/数据/测试协议差异为0 |
| V2 smoke配置初始沿用`required_memory_bank_warmups=2`与较长`des` | 1 | 启动前门禁发现；冻结路由器不重建memory，改为warmups=0，并仅缩短实验描述避免Linux单目录名限制，模型/数据/预算不变 |
| 步骤8 V3恢复上下文时误调用不存在的`session-catchup.sh` | 1 | 技能目录实际提供`session-catchup.py`；记录错误后改用项目Conda Python执行真实脚本，不重复错误后缀 |
| 步骤8 V3首次源码检查引用不存在的`baselines/S4M/exp/exp_pretrain1.py` | 1 | 仅只读`rg`失败；立即用`rg --files`定位真实文件，不猜目录层级 |
| 步骤8 V3指标检查误用无执行权限的系统`python` | 1 | 未写入任何文件；后续固定使用项目Conda解释器绝对路径，并将此前同类教训延续到V3 |
| 步骤8 V3候选运行期间把`write_stdin`会话ID误当作系统PID检查 | 1 | 只读`ps`无匹配且未影响任务；随后用同一会话正确轮询并取得return code 0及完整报告 |
| 步骤8 HCUP源码定位把不存在的外层`experiments/`传给`rg --files` | 1 | 仅只读搜索返回部分有效路径；后续只搜索已确认存在的`scripts/`与`configs/`，不再猜外层目录 |
| 步骤8 HCUP首次后台启动漏传调度器必需的`--gpu-candidates` | 1 | PID立即退出且未创建state/run、未占GPU；补入两张卡的稳定UUID后重启同一smoke，不改实验配置 |
| 步骤8短滞后h8 formal配置首次写入时checkpoint路径漏了run内`checkpoints/setting`层级 | 1 | 启动前人工复核发现，立即改为步骤5同一已验证checkpoint绝对相对路径；未创建run、未启动GPU |
| 步骤8短跨度队列GPU任务全部成功后，最终分析以浮点逐位相等检查跨GPU输出 | 1 | h4在4080运行、其余在4090运行，最大绝对差仅6.20e-6；改为显式atol=1e-5/rtol=1e-6的数值一致门禁，同时保留exact-identical字段，不改变任何实验指标 |
| 步骤8分析缺陷排查时`rg`附带了不存在的`scripts/run_repro_queue.py` | 1 | 只读命令的其他目标仍成功；后续只检索已由`rg --files`确认的队列脚本，不再猜测文件名 |
| 步骤9首轮runner审计尝试读取不存在的`scripts/run_experiment.py` | 1 | 仅该只读子命令失败、未修改文件；改用`rg --files scripts`定位现有MIPP runner/队列入口，不再猜测通用文件名 |
| 步骤9首次直接运行MIPP组件测试报`ModuleNotFoundError: S4` | 1 | 测试未进入断言、未启动GPU；直接命令缺少S4包与structured_kernels路径，后续先定位并使用项目既有环境包装/完整PYTHONPATH，不重复裸调用 |
| 步骤9配置门禁的一行Python命令因字面量`\n`触发SyntaxError | 1 | 其他并行门禁均PASS且未创建run；停止复杂`python -c`，新增独立`test_stage9_bmsa_config.py`持久测试配置与公平性 |
| 步骤9扩展组件测试首次沿用步骤7报告路径 | 1 | 相同seed确定性CPU结果把步骤7JSON覆盖为超集；立即恢复步骤7原schema，并将新测试输出永久迁移到`stage9_bmsa_component_test.json`，不再复用历史工件路径 |
| 步骤9队列自测定位时检查了不存在的`scripts/test_scheduler_queue.py` | 1 | 只读且用回退继续定位到`run_stage5_baseline_queue.py --self-test`；后续直接使用已确认入口，不再猜独立测试文件 |
| 步骤9报告隔离检查与组件测试并行导致短暂`FileNotFoundError` | 1 | 实际输出路径已正确，检查进程早于生成进程完成；改为先运行组件测试、再顺序验证报告隔离，模型/配置无需修改 |
| 步骤10多seed资产测试写JSON时遇到`numpy.bool_ is not JSON serializable` | 1 | 资产断言已执行但报告写入前失败；为测试报告增加仅转换NumPy标量的显式JSON编码器，不放宽断言，随后重新运行完整资产审计 |
| 步骤10 seed2024 PG95 smoke在绝对门控无候选时对空张量调用`min()` | 1 | 保留CODE_ERROR run并阻断formal；空准入仍保持为空，仅把诊断cutoff记录为配置阈值，增加空门控与memory零写入组件回归，修复后只重排该smoke及依赖formal |
| 步骤10后续原型库审计引用不存在的`baselines/S4M/s4m/model/MTNet.py` | 1 | 其余只读搜索正常且未修改文件；改用`rg --files`定位真实网络入口，不再猜模型文件名 |
| 步骤11首次修复SDRA报告器时辅助函数插入决策函数内部 | 1 | `py_compile`通过但决策单测返回`NoneType`并拦截；按函数真实边界移动辅助函数，恢复所有正/负决策断言 |
| SDRA报告器要求历史PG95显式含后来新增的`mipp_write_budget` | 1 | 历史resolved config已保存原始`M=50`且实际行为可还原；从`mipp_write_budget`回退到`M`核验等价预算，数据、模型和训练协议检查不放宽 |
| MRBA启动前尝试向runner传入不存在的`--dry-run`参数 | 1 | argparse在训练/GPU初始化前退出2，未生成run或占用GPU；已有独立配置门禁覆盖解析与公平性，因此不为此扩展正式runner接口，直接按规定先进入GPU smoke队列 |
| 步骤12配置审计发现历史PG95 resolved config没有后来新增的`mipp_write_gate_mode` | 1 | 原run.py默认值和实际行为均为`absolute`；stage12仅对PG95采用缺失字段回退`absolute`，MRBA仍要求显式新模式，其他协议检查不放宽 |
| 步骤12首轮两个GPU smoke在诊断分支报`NameError: exp is not defined` | 1 | 两项均在模型/GPU前向前CODE_ERROR且formal被依赖阻断；补与HCUP一致的`exp = Exp(args)`实例化，保留失败run，静态/组件复测后通过显式reset只重排本队列 |
| 步骤12第二轮smoke在row-shared检查报PyTorch 2.1不支持`torch.all(dim=(1,2))` | 1 | checkpoint已加载但缓存未写、formal仍阻断；改为每样本展平后单维`all(dim=1)`，语义不变，并新增当前环境张量回归后重排 |
| 步骤13定位组件测试时猜测了不存在的`scripts/test_stage9_mipp_components.py` | 1 | 仅只读命令失败；已用`rg --files`确认真实入口为`scripts/test_mipp_components.py`，后续不再猜测文件名 |
| 步骤13首次后台启动发生在隔离PID命名空间，写出PID=11但未创建队列state/run | 1 | 无实验启动、无GPU占用；保留空日志证据，改用宿主持久后台启动同一队列，模型和实验配置均不变 |
| 步骤14分析器合成负例仍越过冻结门槛 | 1 | 仅减弱测试夹具的四分位误差差距，不修改正式门槛、协议或分析器 |
| 步骤14首次修正测试夹具时猜错变量名，补丁安全失败 | 1 | 先读取真实源码，再按`mae_fail_error`精确修改；未覆盖任何文件 |
| 步骤14首次追加findings时假定了不存在的标题，补丁安全失败 | 1 | 读取文件真实尾部，按现有步骤14审计段落精确追加；未覆盖任何文件 |
| 步骤15启动后的宿主状态轮询被平台额度限制拒绝 | 1 | 命令未执行、队列不受影响；不绕过宿主限制，改读项目内持久化state/log |
| 步骤21分析器把run专属输出路径误判为训练预算差异 | 1 | 将路径归入显式run身份白名单，新增路径差异正例与batch-size漂移负例；重算后协议PASS、科学结论不变、test仍禁止 |
| 步骤22恢复时先后猜错可编辑S4M嵌套入口与Stage21配置目录 | 2 | 用`rg --files baselines/S4M configs`定位真实文件后再读取，不重复猜测目录；均未修改代码或实验 |
| 步骤22新建status脚本首次直接执行无可执行位 | 1 | 为Stage22 start/status/stop脚本补齐可执行位并用显式`bash`复核；后台调度器和GPU任务不受影响 |
| 步骤22创新队列启动前受限环境`nvidia-smi`无法连接驱动 | 1 | 在正式服务器权限下重跑同一只读GPU查询并成功核对UUID、名称与显存；随后启动后台队列 |
| 步骤22 MRRA formal在epoch2约iter700出现NaN，epoch3 memory聚类报`data contains NaN values` | 1 | 保留失败run并停止自动重试；按`NAN_OR_DIVERGENCE`根因审计checkpoint/参数/损失，修复后先增加覆盖多epoch的稳定性smoke，再只重排MRRA formal |
| 步骤22.9首次checkpoint诊断的一行Python命令含字面量换行转义 | 1 | 命令在加载checkpoint前即`SyntaxError`退出，未修改工件；改为可复用只读诊断脚本，不再用复杂`python -c` |
| 步骤22.9首次把错误表锚点误写为`progress.md` | 1 | `apply_patch`整体安全拒绝且未产生部分修改；确认错误表实际位于`task_plan.md`后精确重写 |
| 步骤22.10组件测试误用另一项目的`scripts/s4m_py.sh` | 1 | 入口不存在并在执行断言前退出127；改用当前项目Conda解释器直接运行，20项组件检查全部PASS |
| 步骤22.19可行性脚本两处误用NumPy数组`.square()` | 1 | 两次均在报告落盘/GPU调用前由AttributeError停止；统一改为`np.square()`并保留解析恒等式检查 |
| 步骤22.19协议字典把未访问test记录为false后又要求所有检查为true | 1 | 将检查键改为`test_not_accessed=true`，结果payload仍单独保存`test_accessed=false`，不放宽验证集与测试集边界 |
| 步骤22.19.3静态编译误用当前shell不可执行的裸`python` | 1 | 未运行训练且未生成run；改用项目固定Conda解释器绝对路径，6个接入文件随后全部编译PASS |
| 步骤22.18 TTRR回归测试误扫同目录的推理计时YAML | 1 | 模型代码未回归；把测试发现范围收紧到三类训练配置，重跑TTRR 31项与既有Step22配置60项检查均PASS |
| 步骤22.19 ARD专项测试把warmup零梯度限定为必须`None` | 1 | 计算图相消会产生逐元素为0的梯度张量；把正确判据改为`None`或严格全零，训练目标不变，完整检查重跑PASS |
| 步骤22.19控制组模型身份误含`ard`字样 | 1 | 配置门在GPU前正确拒绝；仅将控制组身份改为`mcpr_mse_seed2025_control_electricity`，训练预算/数据/目标不变，随后配置公平性差异为0 |
| 步骤22.21 Traffic资产审计报告包含`numpy.bool_`导致JSON写入失败 | 1 | 不放宽任何断言，将所有检查结果显式转换为Python原生bool；重跑后21项资产审计全部PASS |
| 步骤22.21历史回归命令误把工作目录设为嵌套基线仓库 | 1 | 四个测试均在加载前因路径不存在退出，未启动GPU或修改工件；纠正到项目根目录后四组历史回归全部PASS |
| 步骤22.21首次S4M/PG95 smoke使用了不存在的Traffic目标列`861` | 2 | 两项均在DataLoader构造、模型前向前CODE_ERROR，formal/test被依赖阻断；核对真实表头为`0..860,OT`，透明修订协议与配置为`target=OT`，保留失败run并全量门禁后生成新ID重排 |
| 步骤22.21协议追加透明实现修订后，资产测试的单一mtime前置检查失败 | 1 | 原资产/哈希未变；将不再适用的mtime检查替换为协议路径、冻结哈希、修订边界和失败run保留声明的联合审计，不改科学门或数据断言 |
| 步骤22.21两项Traffic formal在首epoch分别于batch27/batch33出现NaN | 1 | 队列按`NAN_OR_DIVERGENCE`停止且未盲目重试；全量CPU审计排除数据/掩码/归一化异常，建议两臂共同把Adam lr0.01修订为上游默认0.001，等待用户明确授权后先做64-batch稳定性smoke |
| 步骤22.21 validation分析器仍硬编码旧lr0.01 | 1 | 唯一test先被安全跳过且未访问；修复为显式要求授权后的lr0.001与修订身份，新增旧lr/缺失身份/epoch漂移负例，24项门禁全PASS后只重排条件test |
| 步骤22.21原始汇总表未映射评估任务的split/test字段 | 1 | 模型指标不受影响；汇总器改读`source_split/uses_test_split/test_batches_processed/sample_count`并新增自动测试收益和终态决策 |
| 步骤22.28首次源码审计沿用了错误的`exp/exp_pretrain1.py`与外层`run.py`路径 | 1 | 只读`rg`报路径不存在，未修改文件；用`rg --files`定位到`s4m/experiments/exp_pretrain1.py`与`s4m/run.py`，后续不再猜测 |
| 步骤22.28首次GPU smoke的教师因子分解绝对误差`1.9073e-6`高于工程门`1e-6` | 1 | 训练/双分支/checkpoint/validation均成功，formal正确阻断；误差为FP32等价运算顺序，冻结为`32×epsilon≈3.8147e-6`，不改预测、损失或科学门，以新ID重排 |
| 步骤22.29首次smoke预检把`verify_checkpoint`基础返回误当成含runner后置DMRB字段 | 1 | GPU模型前向前由CODE_ERROR拦截、formal未启动；改为直接核验checkpoint的L5/L20参数键并补正负合约，不改模型、数据或科学门，保留失败attempt日志与空run目录 |
| 步骤22.31只读资产检查先后误用裸`python`与缩写错误的Conda环境名 | 2 | 两次均在读取JSON前退出，未修改工件或启动GPU；核对既有start脚本后固定使用`/mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python`。 |
| 步骤22.31历史回归首次猜测不存在的`test_stage5_scheduler_queue.py` | 1 | 该单项在测试加载前退出2；改用已确认的`run_stage5_baseline_queue.py --self-test`，9/9调度能力全部PASS。 |
| 步骤22.31首次smoke被旧PBA训练专用安全门拒绝只读诊断 | 1 | 在模型构造/前向前CODE_ERROR，formal保持BLOCKED且test未访问；只把显式`stage22_31_pba_diagnostic + is_training=0`加入白名单，并为checkpoint、来源一致性、3/full batch和教师协议增加独立门禁，普通PBA评估仍禁止。 |
| 步骤22.31 formal跨run逐token PG95/教师损失闭环未过`2e-5`门 | 1 | formal已完整推理且19/20工件门PASS；PG95 MSE最大差`1.297e-4`、均值差`1.442e-6`，教师获益集合仅5/158208个token不一致。先归因BF缓存的`student-residual`恢复与本轮直接PG95前向的FP32运算顺序，再冻结尺度相关容差或改为同run重算，未直接放宽门。 |
| 步骤22.31新增“非边界翻转”负例仍被边界门接受 | 1 | 测试夹具的参考margin仍为`1e-6`，按定义确属近零边界；只把合成参考/当前margin改为明确`+0.01/-0.01`，不修改正式容差或判定器。 |
| 步骤22.31重排前受限会话`nvidia-smi`无法连接驱动 | 1 | 计划读取成功但同一shell中的GPU查询失败且遮蔽后续磁盘查询；改为宿主权限下独立核对GPU UUID/显存，并把磁盘检查拆成独立命令。 |

## Step 22.28：DMRB 双尺度掩码条件残差基

**本阶段目标**：在一次 PG95 主干前向上，用两个独立、可关闭的低秩残差基分别学习 L5 与 L20 教师修正，验证 Step22.27 定位的“学生表达不足”能否被单一结构因素解决。

**完成判据**：结果前协议、默认关闭等价性、分支目标/梯度、checkpoint 保存加载、配置公平性与调度依赖全部通过；在磁盘可安全落盘时，依次完成 Traffic seed2027 validation smoke 和完整 formal，test 不访问。

- [x] **22.28.1** 写入结果前冻结协议：唯一 DMRB 结构、分解教师目标、同等预算、停止门与 test 禁止边界。
- [x] **22.28.2** 审计 MCPR/S4M/训练入口，实现默认关闭的 DMRB 及 L5/L20 独立开关，保持原基线可运行。
- [x] **22.28.3** 新增组件、目标分解、梯度隔离、配置负例、checkpoint 往返与历史回归测试，完成零 GPU 门禁。
- [x] **22.28.4** 新增唯一可还原 runner、smoke/formal 配置、工件合约、UUID 队列和自动汇总。
- [x] **22.28.5** 在任何 GPU 任务前重查磁盘；若无法容纳 smoke+formal 的必需 checkpoint，标记 `WAITING_STORAGE` 并停止，不删除用户成果。（使用已验证的可还原增量 checkpoint，226 MiB 可安全容纳）
- [x] **22.28.6** smoke 的 forward/backward/optimizer/checkpoint 保存加载/评估/metrics 全通过后，队列自动放行并完成 seed2027 validation formal；两项任务均 `SUCCESS`。
- [x] **22.28.7** 已自动比较 PG95、BF-CMVD-v2、CNM-SE 教师与 DMRB，并汇总参数、时间、峰值显存与分支激活；DMRB 未通过准确率与 60% 收益保留门。
- [x] **22.28.8** 探索门未通过，按预注册停止：不扩 seed2028、不访问 test，转入 validation-only 失败归因。

**步骤22.28终态：** 工程完成、科学门失败。DMRB validation MSE/MAE=`0.521787612/0.317986334`，相对BF-CMVD-v2退化`0.199201%/0.109647%`，仅保留CNM-SE教师收益`29.7496%/30.7087%`；决策=`DMRB_SEED2027_STOP_DIAGNOSE`。

## Step 22.29：DMRB 分支监督与组合失效归因

**本阶段目标**：在不训练新模型、不访问test的前提下，解释DMRB为何在两个尺度分支均真实激活后仍弱于BF-CMVD-v2，并只冻结一个下一版改动方向。

**完成判据**：使用同一Traffic seed2027 chronological validation、PG95源checkpoint、DMRB checkpoint和冻结教师，闭环重算已有指标；分别量化L5/L20单分支效果、分支相加交互、各自教师目标对齐、交叉尺度串扰及“联合获益掩码—分支获益掩码”错配；结果前冻结判定顺序，最终只选择一个主因或停止该残差蒸馏路线。

- [x] **22.29.1** 结果前冻结数据边界、配对张量、分支级指标、主因判定顺序和停止规则；明确禁止test、调权重和新训练。
- [x] **22.29.2** 实现冻结checkpoint只读诊断：`p0+r5`、`p0+r20`、`p0+r5+r20`、L5/L20教师贡献及固定CNM-SE教师在同一validation样本上逐点闭环。
- [x] **22.29.3** 小批量GPU smoke与完整103-batch validation诊断均完成；20/20工件检查通过、test未构造，首次预检CODE_ERROR已保留并修复后重排成功。
- [x] **22.29.4** 自动归因完成：联合监督错配门与固定相加冲突门均未触发，L5/L20两个分支的低秩表达不足门同时触发。
- [x] **22.29.5** 按结果前冻结规则停止当前低秩输出残差路线；不实现分支专属掩码、路由器、加rank或加depth，下一路线必须改变修正发生的位置/可训练函数族。

**步骤22.29终态：** `DUAL_BRANCH_EXPRESSION_UNDERFIT_STOP_LOW_RANK_ROUTE`。L5/L20学生残差对各自教师贡献的余弦仅`0.077425/0.077294`、投影系数仅`0.034052/0.026085`，各自获益区贡献约`97.05%/97.27%`的MSE正差距；分支监督重叠与固定求和均被证据否决。

## Step 22.30：PBA-CMVD 部分主干摊销因果多视图蒸馏

**本阶段目标**：停止在最终预测上叠加低秩残差，改为让单视图学生的末端非线性时序动力层直接摊销冻结CNM-SE多视图教师，在保持单次推理前向的同时验证教师收益能否由更合适的函数族吸收。

**完成判据**：在任何新validation结果前冻结唯一可训练范围、教师快照、损失、训练预算和停止门；创新路径默认关闭时恢复PG95；通过forward/backward/optimizer/checkpoint保存加载/评估/metrics的GPU smoke后，完成Traffic seed2027同协议validation快速对照；不访问test，不同时加入对比学习、可靠性门或新路由器。

- [x] **22.30.1** 审计S4M末端层、冻结教师生成和现有BF-CMVD-v2目标，写入结果前中文协议与唯一PBA-CMVD结构。
- [x] **22.30.2** 实现默认关闭的部分主干摊销训练路径：冻结独立CNM-SE教师，学生仅开放预注册末端层；保存来源身份和全部变化参数的可还原增量checkpoint。
- [x] **22.30.3** 新增参数冻结、教师隔离、单视图推理、配置负例、checkpoint往返、历史基线回归与队列测试。
- [x] **22.30.4** 独立smoke与Traffic seed2027完整validation formal均SUCCESS，全部工程/工件门通过。
- [x] **22.30.5** 自动比较已完成；准确率门失败，停止seed2028与test，转入冻结失败归因。

**步骤22.30终态：** 工程完成、科学门失败。PBA-CMVD validation MSE/MAE=`0.521529998/0.317547226`；相对PG95改善`0.370986%/0.387218%`，但相对BF-CMVD-v2的MSE退化`0.149732%`且MAE仅改善`0.028595%`；决策=`PBA_CMVD_SEED2027_STOP`。

## Step 22.31：PBA-CMVD validation-only 失败归因

**本阶段目标**：不训练新模型、不访问test，在相同Traffic seed2027 chronological validation上解释PBA-CMVD相对BF-CMVD-v2的MSE/MAE分歧和教师收益保留不足。

**完成判据**：固定PG95、BF-CMVD-v2、PBA-CMVD与CNM-SE教师四路预测和评估资产；结果前冻结“高误差尾部放大→非受益区域漂移→教师方向摊销不足”的判定顺序；smoke及完整validation诊断成功落盘后，只冻结一个下一版方向或停止路线。

- [x] **22.31.1** 核验Step22.30队列、checkpoint、日志、训练动态与终态报告。
- [x] **22.31.2** 冻结失败归因协议、指标、阈值、判定顺序和test禁止边界。
- [x] **22.31.3** 实现四路同资产诊断、可还原runner、工件门与自动分析器。
- [x] **22.31.4** CPU/静态门禁、smoke attempt03与完整validation formal attempt02均SUCCESS，21项工件门全通过。
- [x] **22.31.5** seed2027 validation-only正式诊断完成；唯一主因为教师方向欠摊销。
- [x] **22.31.6** 按预注册规则唯一冻结HFD-PBA；否决CVaR尾部约束与非受益区域门控。

**步骤22.31终态：** `TEACHER_DIRECTION_UNDER_AMORTIZED_FREEZE_HFD_PBA`。尾部正MSE超额份额仅`27.4348%`且尾部PBA平均MSE优于BF；教师受益区承担`91.7113%/88.9854%`的MSE/MAE正差距，PBA—教师修正cosine/projection/mismatch=`0.130330/0.138764/1.856097`。

## Step 22.32：HFD-PBA 层级特征蒸馏部分主干摊销

**本阶段目标**：保持PBA的相同末端两层、固定CNM-SE教师、训练预算和单视图部署，只增加结果前冻结的层级教师修正表征监督，直接改善Step22.31确认的教师方向欠摊销。

**完成判据**：默认关闭恢复PBA；HFD只作用于预注册S4第6/7层隐藏修正并可独立关闭；teacher/source/student特征、梯度、checkpoint和单视图推理全部通过smoke；随后完成Traffic seed2027同协议validation，并同时优于BF-CMVD-v2与PBA且保留至少60%教师收益才允许独立seed。

- [x] **22.32.1** 结果前冻结HFD目标、层级、归一化、损失尺度、公平预算和停止门。
- [x] **22.32.2** 实现默认关闭的隐藏特征捕获与HFD-PBA训练路径，保持原PBA/PG95行为不变。
- [x] **22.32.3** 完成组件、配置负例、梯度、checkpoint、历史回归和调度器门禁。
- [x] **22.32.4** 3-batch GPU smoke及Traffic seed2027完整validation均在RTX4090首次SUCCESS，无OOM/重试。
- [x] **22.32.5** 自动比较完成；协议门PASS但精度/60%门FAIL，按预注册规则停止HFD/PBA路线，不扩seed、不访问test。

**步骤22.32终态：** `HFD_PBA_SEED2027_STOP`。HFD-PBA MSE/MAE=`0.522073185/0.319194642`；相对BF-CMVD-v2退化`0.254040%/0.490051%`，相对原PBA退化`0.104153%/0.518794%`。两项教师收益保留率=`24.7059%/-15.9491%`，test未访问。

## Step 22.33：PSAVS 帕累托安全自适应视图选择可行性筛查

**本阶段目标**：停止已失败的单前向蒸馏路线，回到三seed稳定改善的PG95+CNM-SE。利用既有validation逐样本缓存，检验推理时可观测的mask结构与跨视图分歧能否在`original / original+L5 / original+L20 / multiscale`四个固定动作中选择更优路径，同时改善MSE和MAE。

**完成判据**：零GPU、无模型更新、不访问test；固定四动作、特征、时间隔离和停止门后，先验证双指标安全oracle上界，再用固定复杂度策略在chronological audit上比较固定multiscale。只有至少2/3 seed双指标胜出且三seed平均MSE/MAE均改善至少0.15%，才允许实现PSAVS模块；否则停止自适应选择，不训练强化学习策略。

- [x] **22.33.1** 冻结四动作、14/17维特征、共享深度3回归树、dev裕量校准、287窗口时间隔离和主策略audit停止门。
- [x] **22.33.2** 三seed全部源工件门、掩码计数、缓存指标及全validation标签感知oracle回算PASS，统一输入已落盘。
- [x] **22.33.3** 两种预注册策略的fit/dev/冻结checkpoint/audit完成；校准后均100%回退，未经校准分歧感知三seed均退化。
- [x] **22.33.4** 冻结主门FAIL，决策PSAVS_FEASIBILITY_STOP；不运行seed2029、不训练预算策略、不依据audit调参。

**步骤22.33终态（2026-09-07）：** 工程SUCCESS、科学门FAIL。12/12单测、合成smoke、正式CPU筛查与30行指标独立重算全部通过。正式run=`psavs_offline__traffic__point__mr0.06__seed2233__20260907_141643`；主策略平均MSE/MAE改善=`0%/0%`、双胜seed=`0/3`，raw分歧策略平均退化=`0.143567%/0.194586%`。预测主干未更改，test未读取。

**失败后的最短路径（现已进入22.34）：** 先盘点既有PG95+固定CNM-SE主线的跨seed、跨数据集及消融证据，区分“已有贡献还缺验证”与“确实需要新方法”。本轮否决当前特征+树选择器组合，不外推为所有自适应方法不可能有效；但不继续在已暴露audit上挑更深树、更多特征或强化学习。

## Step 22.34：主线证据审计与validation四格对照

**目标与完成判据：** 保留PSAVS负结果，核实历史模型版本/协议/负结果；使用既有S4M检查点补Traffic三seed同资产validation多尺度，对照A=S4M/B=PG95/C=S4M+CNM-SE/D=PG95+CNM-SE，分离模块贡献与交互，不训练、不重评test、不调结构。

- [x] **22.34.1** 审计Weather/Traffic正结果、ETTh1/Electricity负结果和PG95贡献不稳定边界，冻结四格协议。
- [x] **22.34.2** 为原评估器新增S4M-validation显式分支，4份固定配置与4任务队列已生成；6项专项、19项旧流程和9项队列自测通过。
- [x] **22.34.3** 旧冒烟SUCCESS、seed2027正式计算完成但历史基线一致性失败；旧2026/2028未启动。已确认eval随机抽样及历史RNG未保存；2026-09-08用户明确同意固定随机状态/重建配对参照。
- [x] **22.34.4** 获批CRN-v1修复及队列恢复完成：默认关闭开关、独立原始遍历参照、逐窗口1e-5检查；新/旧配置回归与3项GPU smoke均通过，新两模型smoke逐窗口差异0。六项正式任务已自动放行，4080首批3项启动，旧失败记录保留；正式结果尚未完成。
- [x] **22.34.5** 九项任务全部SUCCESS，正式六项各1648窗口、配对差异0。CNM在S4M上三seed双胜；PG95在CNM之上仅1/3双胜，平均MSE/MAE退化0.2544%/0.8956%，判定贡献不稳定。

## Step 22.35：PG95退化诊断（用户已授权）

## Step 22.36：匹配初始化与隔离随机流的配对控制（已授权）

2026-09-08 20:00验收：9/9任务SUCCESS、无OOM，314共同初始化条目及训练/验证顺序配对PASS；单seed组合较S4M降低MSE1.7051%、MAE2.5438%，不作稳定性结论。

## Step 22.37：冻结协议补seed2028/2029（用户已授权）

- [x] **22.37.1** 冻结16份配置、seed2029同规则数据资产；每seed内共享mask，训练/评估/初始化规则完全沿用22.36，保留2027结果。2026-09-19生成及独立审计PASS。
- [x] **22.37.2** CPU初始化/冻结协议/队列测试；四组训练与评估smoke作为对应正式任务的依赖。
  - 2026-09-19：CPU 7/7 PASS，含两个seed的真实314公共初始化一致、源码与22.36一致、validation/test边界、冻结预算、9项队列自测与负结果统计；GPU smoke仍等待资源。
- [x] **22.37.3** 后台启动，核验真实GPU进程；最多3并发、显存不足等待、OOM退避，禁止缩水。
- [x] **22.37.4** 三seed四格配对核验、均值/样本标准差/逐seed增益与双指标胜数；所有负结果保留，不自动新增模块或test评估。
  - 完成验收：16/16 SUCCESS、无OOM、三seed配对PASS。PG95增量仅1/3双胜；CNM在S4M上3/3双胜。暂停扩展PG95。

## Step 22.38：冻结S4M的填补敏感性与路径诊断（2026-10-04用户授权）

- [x] **22.38.1** 冻结seed2027、validation-only协议；固定观测及mask，预定义零/观测均值/前向填充三种干预，区分query-only、direct-only与完整填补替换。另设retrieval-only隔离纯检索。
- [x] **22.38.2** 通过CPU不泄漏/路径隔离测试及GPU独立smoke；独立脚本/hooks，不修改主模型源码，不训练。9/9 CPU及9项队列测试PASS；GPU 48窗口检查全部PASS、退出0、预留峰值3874MiB。
- [x] **22.38.3** 后台完整验证：同batch16/1648窗口/CRN-v1，与历史原始分支逐窗口配对；显存不足等待、OOM退避。2/2 SUCCESS、退出0、无OOM；GPU子进程与调度器均正常退出。
- [x] **22.38.4** 汇总查询、检索、预测敏感性及误差变化；根据已有证据决定下一结构方向，结果不充当新增模型收益。13路径从npz独立复算通过；query最大MSE相对变化0.002490%，主要填补替换效应在历史直达路径。报告与总表按时间戳归档。

## Step22.39：融合分量与衰减实现诊断（2026-10-04用户授权）

- [x] **22.39.1** 比对上游、当前模型和数据流程；独立CPU算例验证距离标准化、极值计算及重排风险，保留原始实现。19项PASS，六份baseline/upstream文件哈希未变；当前L192距离std必为正。
- [x] **22.39.2** 预先冻结同checkpoint/seed2027/validation/CRN协议，量化投影后的检索、查询（含残差）、历史、观测历史、填补历史及bias；固定缩放0/0.9/1.1，不做结果后追加搜索。15路径及两配置、代码/资产哈希已冻结；CPU12项与队列9项PASS。
- [x] **22.39.3** CPU分解/干预测试与GPU smoke后，后台完整1648窗诊断；batch16及固定mask不变，OOM退避，真实运行核验。
  - 第一次GPU smoke退出1：prototype-off与分量相减路径未通过1e-5门禁，formal被依赖阻断。新增仅定位日志，未放宽门禁。宿主核验审批服务因额度拒绝，恢复GPU诊断尚待服务恢复。
  - 用户继续后审批恢复；V1定位确认FP32求值顺序差异。独立V2原融合输入缩放，CPU12项/队列9项/GPU48窗均PASS，保持1e-5。2026-10-04 21:45完整1648窗已在4080真实运行，worker PID2663411，等待结果。
- [x] **22.39.4** 独立复算、归档报告与简明偏差记录；区分实现问题、激活尺度和预测贡献，不把干预收益当作新模型成果。V2两项SUCCESS/退出0，无OOM，15路径独立复算≤1e-12；主模型未改，报告与总表已归档。原距离误用实际std0.02518（正确约1），先隔离基线修复而非堆检索模块。

## Step22.40：隔离最小距离标准化修复（2026-10-04用户授权）

- [x] **22.40.1** 记录原源码、工作区与协议；仅增加默认关闭的最小距离修复开关，保持原参数、初始化、warmup路由和其他公式不变。pre_edit快照与dirty diff已保留；只改S4M.py/run.py。
- [x] **22.40.2** CPU检查关闭路径与原公式逐元素一致、开启公式/梯度/初始化一致；基线和修复各自GPU训练、保存、重载、CRN验证smoke通过。CPU25/25、队列9/9；四GPU smoke均SUCCESS/退出0，无OOM；各训练smoke两批上限含一次warmup，实际optimizer一步。
- [x] **22.40.3** 复用22.36已完成的原S4M基线，仅训练修复版seed2027；相同batch16/15epoch上限/早停3/数据/optimizer/scheduler/选模，不访问test。后台队列、动态安全余量、OOM退避，不缩水。
  - 2026-10-04 22:25：formal已自动启动于4080，wrapper PID2861265、GPU child2861550；当前未完成，正式指标待训练后固定CRN评估。
- [x] **22.40.4** 复核初始化和训练随机流、固定CRN逐窗验证协议，自动汇总指标与代价；归档正/负结果，按证据决定继续或回退。6/6SUCCESS，无OOM，两模型实际4epoch/2992步，原始CRN固定1648窗MSE0.534761→0.520650，MAE0.326645→0.316614；单seed尚不支持稳定结论，原论文同协议数值NA。

## Step22.41：距离修复三seed复核（2026-10-04用户授权，不监督训练）

- [x] **22.41.1** 独立复核22.40结果；冻结2028/2029配置和原22.37基线来源，只改变修复开关与实验身份，其他科学设置保持一致。独立结果复算PASS，8配置/基线清单准备及校验退出0，不改模型/原配置。
- [x] **22.41.2** CPU核验两seed初始化/默认关闭代码/配置与数据资产/队列及越权修改拒绝。13/13CPU、9/9队列检查PASS/退出0，62源码哈希绑定；GPU每seed训练保存及CRN重载smoke是正式任务自动依赖，不提前称GPU门禁已完成。
- [x] **22.41.3** 启动统一后台8任务队列，最多3并发、完整训练资源安全隔离/OOM退避；只进行必要的启动真实性核验，之后不持续监控训练。start退出0，scheduler3039416；2028 GPU child3039702在4080、2029 child3039596在4090，state RUNNING、日志已从新建推进初始化/数据准备。正式任务自动依赖各自smoke通过，尚未完成。
- [x] **22.41.4** 自动汇总及两项独立只读审计完成；8/8SUCCESS、各attempt1/OOM0，三组1648窗及固定资产/指标复算PASS，62源码哈希和8配置不变。平均逐seed相对MSE/MAE改善1.8276%/1.8847%，2/3双胜，2028退化0.1893%/0.6310%；科学全胜门未通过，不自动扩展。三seed同为4epoch/2992步/28371290参数，耗时/显存及跨GPU不确定性已归档；独立推理时间/FLOPs尚无可比测量，不冒充已完成。入口重算退出0，latest总表298条含本阶段全部8任务；审计见reports/stage22_41_final_audit.md。

下一步建议（尚未执行）：**22.42** 使用已有三seed验证逐窗口缓存、训练曲线和动态原型工件，定位2028退化与2029显存差异。先做只读证据分析，不在已暴露validation上反复调参，不自动加模块、扩数据集或重评test。

- [x] **22.41.4补充解释**（用户追问2028差异）：已有缓存/源码/六checkpoint只读核验完成，2028原基线已更优、6中心检索分支不同、第一时间段收益反转、观测目标选模与完整目标评价不一致均已有证据；根因尚未因果隔离。只同步工作记录，不启动22.42新实验或更改任何科学设置。

## Step22.42：三seed修复收益反转离线诊断（2026-10-05）

目标：解释差异定位到哪一层证据，并给出一个最小下一步；不能以相关性冒充因果，也不能在已暴露验证集上继续挑参数。

- [x] **22.42.1** 冻结源报告/六组评估与训练工件、配置/代码/源hash；明确已有四分组属于已知探索，不包装盲预注册。时间×缺失率/末端/最长段规则固定，prepare退出0。独立审计补明验证窗口drop_last丢13尾窗与val-only CSV安全检查边界。
- [x] **22.42.2** 独立CPU缓存7/7、bank12/12合成测试退出0；真实2028全缓存smoke与闭环通过，run后缀20261005_182752、SUCCESS，无模型前向/GPU/test。新增原型索引静态风险4个原方法CPU合成算例PASS，作为独立实现风险，不修复或算创新。
- [x] **22.42.3** 完整三seed两路径168分组成功，run=s4m_distance_diagnostic__traffic__validation__seed2242__20261005_182906；CPU计算2.00秒、RSS744.74MiB/GPU0。配置/命令/源码快照/源manifest/日志/状态/资源全部保存，latest300项，诊断预测指标NA，报告与原始证据全保留。
- [x] **22.42.4** 独立直接NPZ576数值、四段加和、134资产/旧代码hash及latest诊断身份均PASS，无阻断。确定第一段反转而未确定机制因果；仅建议22.43冻结两版完整训练状态（权重+bank）×两公式2×2，不夸称能单独分离权重/银行，不改原baseline、不把修bug当创新。

历史建议（2026-10-06暂列备选，未启动）：**22.43公式交叉诊断** 仅Traffic2028固定1648窗，在原始/修复checkpoint各自冻结权重/银行和CRN随机流下切换公式。用户最新讨论优先已有修正基线+CNM增益核查，故本轮不做此GPU干预。该备选不删除22.42证据，也不变更旧实验的科学判定。

## Step22.43：距离修复后CNM-SE增益核查（2026-10-06）

目标：复用已有Traffic2027/2028/2029验证缓存，明确公式错误修复之外，固定多尺度CNM-SE是否仍有方向一致的增益。本步是已知结果的复算/归因核查，不是新训练或新独立泛化验证。

- [x] **22.43.1** 核验旧结果、同checkpoint、公式开关、固定数据/掩码/CRN资产，冻结只读来源清单。prepare退出0、141来源hash绑定；仅改变本步分析身份与比较对象，不改变训练或评估协议。旧原公式配置无完整parser已明确标注，修复主比较完整参数与模块statekeys通过。
- [x] **22.43.2** 8项最小缓存分析合成测试与真实2028缓存smoke均退出0；weighted指标、非法数据拒绝、来源不变和输出保存通过。smoke run后缀20261006_224028，不运行模型链路、不当精度结果。
- [x] **22.43.3** 正式CPU核查三seed两公式版本下original与固定multiscale，run后缀20261006_224121、SUCCESS/退出0，0.906311秒/RSS38.457MiB/GPU0；配置/命令/环境/代码与来源快照/状态/资源完整，中文报告与6行CSV已保存。
- [x] **22.43.4** 独立60数值/141来源hash及run/report/总表复核PASS，latest302项、诊断指标NA；旧Namespace及冻结源码补审确认17开关关闭/旧公式，不冒充完整parser。结论保留CNM方向性信号、3seed均双胜，非两个创新或新独立证据；旧22.41决策未变，未扩大实验。验收reports/stage22_43_final_audit.md。

边界：不修原型索引、不切换冻结模型公式、不访问test、不调融合权重、不挑seed/时间段。CNM是推理自集成策略，增加前向次数；与单路径只保证训练预算一致，不声称推理算力一致。所有上游失败历史继续保留。

下一最短建议（22.44尚未实施）：将已确证新增原型代表索引错误作为独立bugfix，保持原行为默认可恢复；CPU合成与基线回归后，先单seed同预算训练/验证，再决定CNM对照。修复不算创新，交叉诊断继续备选，不承诺全部root cause已识别。

- [x] **22.36.1** 冻结8份配置，单seed2027、原15epoch/早停3完整预算、同数据/观测MSE选模；只改初始化和训练随机流控制，评估延续CRN-v1。
- [ ] **22.36.2** 默认关闭开关；CPU验证公共权重、采样次序、辅助流隔离及旧开关行为。新两模型训练/保存/加载/验证smoke全部通过才放行正式。
- [ ] **22.36.3** 后台队列两训练+配对验证，实时GPU资源补位、OOM退避、不缩水；核实真实运行后交接。
  - 2026-09-08 19:15进展：S4M链路全部SUCCESS；用户授权首次调度余量10%→5%（最低512MiB），不降实验配置，7/7测试PASS。PG95训练smoke已在4090真实启动，GPU PID2466289，日志初始公共哈希匹配；正式任务仍等待自身smoke通过。
- [ ] **22.36.4** 完成后检查训练证据和四格validation，不把随机协议变化当成创新收益，不自动扩seed或改模型。

- [x] **22.35.1** 六训练源/六CRN缓存核验通过；同预算、最佳epoch1，常规写入门放行>99%，不能单独解释退化。
- [x] **22.35.2** 三项单测、合成smoke、正式CPU诊断通过；三seed48分组全部保留。每seed当前构造314个公共状态条目中124个初始不同，seed2028退化分布广，不是少数异常窗口。
- [x] **22.35.3** 原型几何、训练信息假设与初始化限制已落盘；不宣称唯一根因。下一22.36建议匹配公共初始化/隔离训练随机流的单seed原方法对照，涉及新训练协议，尚未实现/启动。

**结果前后续边界**：若分歧感知策略通过，Step22.34实现PSAVS可关闭模块并在新seed2029 validation确认准确率；只有确认后，Step22.35才引入budgeted contextual bandit（预算约束上下文赌博机）减少平均视图调用数。不得同时训练选择器和预算策略混淆两个创新点。
# Step 22.15：MRRA 内部特征可分性复核

- [x] 审计 MRRA、S4M 与原型记忆库的可观测内部量
- [x] 冻结 validation-only、时间隔离与停止规则
- [x] 实现默认关闭的逐样本内部特征缓存
- [x] 实现并测试预注册 fit/dev/audit 分析器
- [x] 运行 smoke test 并验证缓存字段、checkpoint 加载与 metrics 落盘
- [ ] 后台运行完整验证缓存与冻结分析（已启动）
- [ ] 根据 audit 结果决定构建风险门控或停止 MRRA 精度分支
