# 发现与决策（内部工作记录，非正式项目文档）

## 2026-10-06 步骤22.43：修复后同检查点CNM增益初核

- **正式及独立验收完成**：8单测+真实smoke+formal通过；60数值/141hash验证，三个fixed original→CNM均双胜，平均逐seed降幅MSE0.9579816080%/MAE0.6834905678%。不是距离修复3/3，旧2/3结论保留；不是独立新验证。
- legacy身份不能由缺省推断；进一步用旧train.log唯一Namespace的17项显式False与冻结CLI/模型AST确认无模块/无修复开关及两处旧max分子。三旧模型hash3f97a483…625b526、CLI eb32e826…90d6b一致；主fixed完整有效parser不受legacy缺口影响。
- 科学定位仍为“一个推理集成策略的小幅局部正信号”，不是两个稳定创新。下一优先单因素修已确证原型索引风险；不把修错包装成创新、不在已暴露验证集继续调权或改选模。

- 两项独立只读审计核实三个修复后S4M检查点original→固定multiscale均双指标改善；初算逐seed改善率均值MSE0.9579816%/MAE0.6834906%。这是已有22.41缓存的新比较口径，不是新训练/测试；旧公式修复2/3结论不变。
- 固定权重original0.5/L5 0.25/L20 0.25，3次前向、无新增参数。历史串行组件参考倍率约3.12/3.15/2.86，非隔离端到端基准；模型和银行只读来自冻结源码条件及工件，不是新运行时银行内容哈希。
- NPZ只有逐窗口SSE/SAE及mask/count，不含完整预测/真值；本次能重算加总指标，不能重演融合或从预测重新计算误差。此前用“独立复算”时要明确这个证据层级。
- 当前磁盘约1.1GiB，禁止复制检查点/数据或无授权清理。只新增最小CPU分析/单测和小型run/报告，不接入新模块/模型，不扩test访问。

## 2026-10-05 步骤22.42：离线诊断边界

- 收尾独立复核576数值/134hash及总表均PASS，无阻断；source未改，诊断没有伪造新的MSE/MAE。结论与源报告一致；新增summary.md用于人读，不覆盖原精度。原型norm差异（2028均值≈4.92、2027≈1，2029修复亦有大norm）及raw dot product（原始点积）是潜在混合尺度线索，并非2028独有，不能本轮直接升级为主因或新归一化模块。
- 完整CPU诊断成功退出0（run后缀20261005_182906），168条逐seed×路径×分组统计，源hash前后不变；计算阶段2.00秒/RSS744.74MiB/GPU0，latest300项。主要结论定位时间反转但因果未识别，不能把局部条件相关性当主因。
- 新分组结果：2028Q1的20–30%缺失档313窗及≥30%档99窗都MSE退化5.7928%/5.7133%，MAE4.2877%/3.7406%；四描述量Spearman最大abs0.0913，不支持单一“缺失越多/末端越长导致退化”的解释。末端无缺失1196窗MAE仍+0.5040%，只分析长尾不能解释全部退化。
- 2028MSE793窗（48.12%）/MAE1079窗（65.47%）变差；最坏83窗占正SSE增量62.40%、正SAE40.04%，退化既有集中误差贡献也有广泛MAE变差，不删除极端窗。四段对全局MSE变化贡献+0.005312922/-0.000083925/-0.003855944/-0.000396511，加和为+0.000976541。
- 下一唯一建议22.43先冻结2028两检查点的权重/银行，交叉切换原/修复填补公式（2×2），不训练、不同时修新增簇索引、不变选模。用于隔离即时公式响应与训练状态交互，仍不能分离两checkpoint的权重与银行差异；本轮未启动。
- Memory.py新簇索引已由独立agent及根agent的AST原方法CPU执行互证（upstream/baseline×未满/满FIFO，共4情形）：新resident[-1,0]，新增repre却是前簇[0,1]，二范数偏差sqrt2。错误继承自上游，实际Bank检索直接读repre，保存加载不自动重算；之后该簇高相似更新/重新cluster才可能纠正。记录reports/stage22_42_memory_index_audit.json；未修改任何原模型文件，不能称它已解释2028差异。
- 新CPU缓存单测7/7、bank合成单测12/12均退出0；独立真实三对checkpoint helper smoke均通过，CUDA未初始化/六hash前后一致。新helper无模型实例化、前向或文件写副作用。
- 额外银行风险：六checkpoint代表向量与其驻留均值abs max差异0.217–0.465。保存银行有效秩2027 118.807→125.534、2028 1.588→1.576、2029 25.089→5.788（数量上限199/5/61→83）；2029改善但有效秩下降，反对简单“秩增就收益”的故事。Memory.py新簇索引风险正在独立静态核验，不改源文件，不作因果定论。
- 前序agent被系统打断，未写22.42代码文件；恢复后重新分派独立bank实现及只读协议审计，不复跑训练。原计划/阶段文档已保存，磁盘只余1.8GiB，仅读复用大资产。
- 独立协议审计核验packed mask为(1648,24)、192位big bitorder、1观测0缺失，三seed862变量row-shared成立，val上下文缺失CSV与cache完全匹配。数据N17544、train终点12280、val終点14036、history起点12088；理论1661窗被drop_last保留1648，丢尾13窗，最后预测目标14022未入test。所有“完整1648窗”只指缓存，非全部理论可用val窗口。
- 源cache没有逐horizon/变量/观测与缺失目标分开SSE/SAE，不能从目标mask重建这些误差。固定新诊断只做已有逐窗口统计，缺项NA，不混旧协议缓存或启动GPU补表。
- 诊断文件最初笼统写“不读CSV”，独立审计实际为了核验row-shared安全读取missing CSV header+validation上下文12088–14035；不读取test/完整真值。已明确更新允许的审计范围并记录偏差，主分析仍只用cache。

## 2026-10-05 步骤22.41.4：三seed结果独立核验

- 补充回答用户“2028为什么与另两个区别大”：只读复算已完成缓存，未启动22.42实验。原基线2028 MSE0.515789优于2027/2029的0.534761/0.531545；修复后三组接近0.515421–0.520650。这是“另外两组向更低误差靠近”的数值现象，不证明已找到2028无收益的因果根因。
- 主路径按时间等分4组（各412重叠窗口，描述性分组而非独立统计检验）：2028 MSE变化依次+5.776304%、-0.101007%、-2.377892%、-0.222055%；MAE变化+4.166554%、+0.350591%、-1.476411%、-0.029951%。2027/2029四组均双指标改善。2028不是全段统一退化，而是第一段损失抵消后续收益。
- 逐窗口变差比例：2027 MSE12.0752%/MAE13.7743%，2028 48.1189%/65.4733%，2029 20.6917%/13.8350%；正负窗口均保留，不把重叠窗口当独立seed。
- 独立CPU只读审计发现原/修复最终原型中心数2027为200/200、2028为6/6、2029为62/84；最佳checkpoint均epoch1。2028原型银行显著小于另两组，是潜在机制差异，不代表小银行必然有害或已经证明导致退化。早停内部观测目标与完整预测目标口径不同，2028内部首轮Vali Loss改善而主要完整目标指标小幅退化，需要保留该错位证据。
- 检索分支源码Bank.py:426/434与CPU工件互证：topM100/topK20，2027先随机子采样100再选20，2028对全部6个中心softmax、不随机子采样、不做topK筛选，2029直接top20。因此2028退化不能归咎topM评估随机子采样；不同银行可能对填补变化产生不同响应，因果尚待隔离。2028银行实际驻留样本26、num_dic总和102，二者不混用。
- 内部vali()（exp_pretrain1.py:1108）只对seq_y_mask==1观测目标计算MSE、按batch无权平均；主CRN完整seq_true_y按元素加权且另评分MAE。2028首轮内部Vali Loss0.5056367→0.5013074（改善约0.856%），完整MSE却+0.1893%、MAE+0.6310%；明确是不同口径，不能直接证明缺失目标子集退化或口径错位解释全部差异，亦不能根据已见结果更改选模规则。
- 2028原/修复四轮Train Loss分别[0.1899702,0.0567919,0.0319345,0.0260213]/[0.1950823,0.0614368,0.0359969,0.0293420]，修复四轮均高；其他两seed不呈现同样一致模式。它是优化轨迹响应差异线索，不等于训练失败或因果根因。所有checkpoint均第一轮最佳，不能建议简单加epoch来掩盖无效。

- 只审计已完成任务，不启动训练、不修改模型或冻结配置、不轮询GPU。后台阶段于2026-10-04 23:58:21（北京时间）结束，8/8任务SUCCESS、各一次尝试、OOM重试0。
- 独立只读审计从逐窗口NPZ重新累计平方/绝对误差再除计数；原始单路径及既有多尺度路径全部与报告吻合（容差1e-12）。三组各1648个验证窗口，10类索引/计数/掩码资产逐项一致；六份评估28/28检查通过，六个实际checkpoint SHA256与记录相符。仅validation、CRN共同随机数、冻结模型和记忆。
- 主要比较：seed2027 MSE/MAE降低2.638741%/3.070810%，seed2028降低-0.189330%/-0.631008%（双指标退化），seed2029降低3.033390%/3.214271%。双指标胜数2/3，未通过预先冻结的三seed全胜门，不自动扩展。
- 逐seed相对收益的算术平均：MSE1.827601%、MAE1.884691%；样本标准差分别1.757823、2.179840个百分点。不能与“均值指标之间的相对降低率”混用，也不是置信区间或显著性证据。
- 本次属于最小距离标准化错误修复，不是新创新点。三个seed同时改变初始化与人工缺失资产，不是固定同一掩码的纯训练方差。单个Traffic验证集不能支持论文同协议复现、独立测试泛化或SCI分区判断。
- 下一步仅建议22.42：使用已有验证缓存和训练日志诊断seed2028退化；本轮不实施、不加模块、不访问test。
- 代价独立复核：8冻结配置/基线资产与62份代码哈希均未变，三seed双方均28371290参数、实际4epoch/2992优化步，实际解析参数检查PASS。2027/2028同卡但共享负载/缓存状态未受控，2029旧4090与新4080跨卡；不能声称修复加速。validation_elapsed_seconds是训练中的验证耗时，不是独立串行推理基准，FLOPs未测。
- seed2029训练峰值allocated（实际分配）5559.12→8005.28MiB，未完成归因。只读CPU加载最终checkpoint发现原/修复动态记忆长度62→84、代表向量形状(62,512)→(84,512)、bank_sample_count82→100；没有全训练数量轨迹或峰值时刻，不能声称最终数量差异已解释显存差异。参数不增加也不等于运行态银行或显存不变。
- 自动汇总入口退出0，latest总表298条且本阶段8任务无遗漏。新归档reports/archive/stage22_41_final__20261005_155609/；独立验收摘要reports/stage22_41_final_audit.md。原始门限与协议未改，工程收尾完成、科学全胜门未通过两者明确分开。

## 2026-09-08 步骤22.36启动资源复核

- S4M新随机协议正式训练/CRN验证已SUCCESS；单seed validation原路径MSE/MAE为0.5347613804/0.3266450031，固定多尺度为0.5274178883/0.3226587713；PG95未完成前不推断模块独立收益。
- 4090空闲17561MiB高于PG95历史reserved16448MiB，但低于原含10%余量的18432MiB准入。用户要求立即尝试，采用保留5%且至少512MiB余量的首次受控启动，准入17408MiB；不改实验预算，不调低OOM后的估计，不跳过smoke。

> 2026-07-23 步骤16恢复核对：历史主线仍以冻结PG95 checkpoint上的CNM-SE为唯一有效分支；seed2024/2025已成功且不得重跑，恢复队列只新增seed2023。三seed门控通过前不启动结构消融。
> 早期查新与复现审计再次确认：普通对比学习、频域损失或通用原型路由都不足以单独构成贡献；步骤16的结构消融必须围绕已经出现实证增益的CNM-SE机制，不回退到已否定的模块堆叠。
> 基线与Gate H历史证据确认Weather是当前有效快速验证集，而ETTh1原型分支近乎失活；步骤16先完成Weather三seed机制复核符合既定证据链，后续跨数据集扩展不能用ETTh1弱信号否定或夸大机制。
> 步骤7–12的多seed证据已否定固定写入阈值、稳定性Top-M、内容多样性和mask覆盖作为稳定主贡献；CNM-SE结构消融必须隔离推理视图结构与融合，而不能重新混入这些负分支。

## 需求
- 项目必须位于 `/mnt/data/zza/code` 下的独立目录。
- 获取 S4M 官方项目完整内容并核验可复现性。
- 当前先讨论创新 idea、水平、创新充分性和相似工作。
- 用户明确确认前不生成正式完整项目文档。
- 正式文档需要对英文专业名词和缩写提供中文说明。

## 已知研究事实
- S4M 题为 *S4 for Multivariate Time Series Forecasting with Missing Values*，发表于 ICLR 2025。
- 原方法由 Adaptive Temporal Prototype Mapper（ATPM，自适应时间原型映射器）和 Missing-Aware Dual-Stream S4（MDS-S4，缺失感知双流结构化状态空间模型）组成。
- S4M 正式论文的四个公共数据集统计为 Electricity（321 变量、26,304 时间点、1 小时）、ETTh1（7 变量、17,420 时间点；论文表中误写为 15 min，ETTh1 通常为小时级，需以原始数据时间戳复核）、Weather（21 变量、52,696 时间点、10 分钟）和 Traffic（862 变量、17,544 时间点、1 小时）。
- 原协议按时间顺序以 0.7/0.1/0.2 划分训练、验证和测试；预测长度固定 96，回看长度为 96/192/384/768，batch size 为 16，论文声明使用 Adam 和早停。
- 原论文两种缺失模式均从比例 r={0.03,0.06,0.12,0.24} 的起点开始删除其后连续 5 点；时间点缺失同时删除所有变量，变量缺失则逐变量独立执行。正文一处把 0.24 误写成 0.14，图表和附录一致使用 0.24，正式复现以附录为准。
- 因重叠缺失块，实际缺失率显著高于起点比例。例如 r=0.24 时四数据集实际缺失率约为 0.613–0.705。配置、日志和论文表格必须同时记录 nominal r（名义起点比例）与 realized missing rate（实际缺失率），避免误称。
- 初步候选创新是跨缺失视图的对比原型学习与频谱最优传输一致性；元学习路由器属于待评估的增强项。
- “在 S4M 原文未使用”不等于“全领域没有类似工作”，必须逐项检索并核对方法细节。
- 官方仓库已完整克隆，当前 commit 为 `a718823addd3606e763dfc261174e0135b2535f4`，工作区干净，无 Git submodule 或 Git LFS 资产。
- 仓库只提供一个 Weather 缺失数据 CSV 示例，没有随仓库提供论文中全部数据集，也没有自动下载脚本。
- README 只给出一个 Weather 训练命令；仓库内的 `scripts.sh` 使用作者机器绝对路径，不能原样复现。
- 官方 `environment.yaml` 指定 Python 3.10.13 和 PyTorch 2.1.0，但同时含有明显可疑或不兼容条目，例如 `aeon==0.0.0`、`torchvision==0.2.0`，必须先验证官方环境创建是否成功，不能直接将文件存在等同于环境可复现。
- 源码包含已提交的 Python 3.9/3.10 `__pycache__`，且根目录未发现项目级 LICENSE；这不影响代码可读性，但降低工程整洁性与再分发许可确定性。

## 技术决策
| 决策 | 理由 |
|------|------|
| 不承诺 SCI 二区/三区录用 | 分区、审稿和实验结果均无法在立项阶段保证 |
| 先审查创新之间的因果逻辑，再考虑模块数量 | 两个互补且被消融验证的贡献优于三个松散拼接模块 |

## 遇到的问题
| 问题 | 解决方案 |
|------|---------|
| GitHub 在先前会话出现 TLS 网络故障 | 尝试官方仓库完整克隆；失败则改用论文页和 API 进行仓库资产核验 |
| 本轮 `nvidia-smi` 无法连接驱动 | 当前阶段不启动 GPU 实验；环境建立后再次核验宿主机 GPU 状态 |
| 数据盘仅剩约 191 GiB 且使用率 98% | 控制环境和数据副本数量，避免提前下载全部数据集 |

## 资源
- S4M 论文：https://proceedings.iclr.cc/paper_files/paper/2025/hash/7b2f0758334389b8ad0665a9bd165463-Abstract-Conference.html
- S4M 官方仓库：https://github.com/WINTERWEEL/S4M
- PSW-I 论文：https://proceedings.iclr.cc/paper_files/paper/2025/hash/033394ac055e3c47bfc3c967ffcfee74-Abstract-Conference.html
- PSW-I 官方仓库：https://github.com/FMLYD/PSW-I

## 浏览器发现
- S4M（ICLR 2025）已明确包含 prototype bank（原型库）和 mask dual stream（掩码双流）；因此不能把“引入原型”或“显式使用缺失掩码”作为新贡献。
- PRIME（2024）已把原型记忆、原型辅助循环网络和原型细化用于不规则时序插补；这进一步否定“prototype for missing time series”作为宽泛首创声明。
- CTRL（IJCAI 2024）已将重建、对比学习、hard negative（困难负样本）与 false-negative elimination（假负样本消除）用于对缺失数据鲁棒的通用时序表征；仅在 S4M 上增加普通 InfoNCE 对比损失，创新不足。
- PSW-I（ICLR 2025）已提出 Proximal Spectrum Wasserstein（近端频谱沃瑟斯坦）差异用于时序插补中的周期结构和非平稳分布对齐；直接把其损失搬进 S4M 可能被评价为已有方法拼接。
- SpectraNet（ICLR 2023）已联合处理时序预测、插补、分布漂移和频谱潜变量，因此“频域 + 缺失预测”不是空白。
- MissTSM（NeurIPS 2024）已做无需插补的缺失感知时序建模；CoIFNet（2025 预印本）已联合插补和预测；VIMTS（ICML 2025）已用视觉掩码自编码器处理不规则多变量预测。
- ChannelTokenFormer（ICLR 2026）直接联合建模通道依赖、异步采样和测试时块缺失，是当前最需要正面对比的后续工作。
- 初步判断：原先“对比原型 + 频谱 OT”若只是两个附加 loss，创新度只有弱到中等，不足以支持高把握投稿。必须把研究问题收敛为未见缺失机制下的泛化，并形成与 S4M 原型检索过程耦合的新机制和专门评估协议。
- OpenReview 网页出现浏览器校验，公开 API 请求又遇到临时网络错误；审稿意见尚未成功取得。
- CRIB（2025/ICLR 2026 投稿版本）已经用 consistency regularization（表征/预测一致性正则化）和 information bottleneck（信息瓶颈）直接从部分观测序列预测；普通跨掩码一致性与信息压缩都不应再作为我们的核心新颖性声明。
- S4M 原论文只评估两种固定长度为 5 的合成块缺失：整时间点缺失和逐变量缺失；未验证跨缺失机制、未见块长度、MAR/MNAR、自然缺失或校准不确定性。论文结论也明确把其他 missing types（缺失类型）列为未来工作。
- S4M 的 ATPM 原型库以余弦相似度 top-K 检索、固定阈值写入、FIFO 淘汰和动量编码器更新；没有用预测误差、原型置信度、mask domain（缺失域）或 OOD（分布外）程度控制读写。这是最有主干特异性的切入口。
- 本地源码确认存在复现阻塞：README 数据路径不存在，真实完整数据路径在多个 Dataset 类中硬编码为 `/home/project/S4M/...`；结果图目录同样硬编码；W&B（Weights & Biases）被强制初始化；指标文件保存被注释；原型读取硬编码 `.cuda()`。
- 本地源码还发现疑似实现错误：`S4M.py` 对 `min_idx` 标准化时错误地使用 `max_idx` 作为被减变量。后续必须把 bug fix 与创新收益分开复现实验。
- 候选新主线调整为“未知缺失机制下，可靠性约束的原型泛化”：不是增加通用对比损失，而是诊断并解决 S4M 原型库在缺失域变化下的污染、错误检索和过度自信。
- 专用 Conda 环境 `s4m_missing_forecasting` 已按官方清单完整安装，占用约 5.4 GiB；`pip check` 报告没有损坏依赖。
- 核心版本为 Python 3.10.13、PyTorch 2.1.0+cu121、NumPy 1.26.4、Pandas 2.2.1。PyTorch CUDA runtime 为 12.1，但宿主驱动当前不可见，`torch.cuda.is_available()` 为 False。
- 官方 README 的 `python run.py ...` 直接失败，因为源码把 `S4` 当作顶层包导入；设置 `PYTHONPATH=<repo>/s4m/model` 后核心模块可导入，并回退到较慢的非 CUDA S4 kernel。
- 论文写明使用 Adam 优化器，公开训练入口实际硬编码 SGD；这是协议级复现差异，必须先做 bug/protocol audit，不能直接把公开代码跑出的数值称为论文复现。
- 2026-07-14 补充检索确认，TameR（ICML 2026）通过历史遮挡训练和全局上下文缓解近期数据偏置，其问题包含缺失扰动鲁棒性，但没有针对 S4M 原型库的效用监督、污染治理或选择性读写。
- PIR（NeurIPS 2025）通过预测后误差识别和实例级修正提高时序预测可靠性，与“预测效用估计”有概念邻近性，但工作位置是预测后处理，不是缺失输入下的原型检索控制。
- 精确检索定位到 STABLEMISS+（ICASSP 2024），题目明确涉及 agnostic mask distribution shift（未知掩码分布漂移）。它要求本项目不能泛称“首次研究缺失分布漂移”，而应把新颖性限定为“首次系统研究 S4M/原型增强缺失预测中的原型污染、反事实内容解耦与预测效用驱动的选择性读写”（最终首创措辞仍需全文核验）。
- 截至本轮精确关键词检索，尚未发现把“有/无原型的预测损失差”作为监督，联合控制 S4M 原型库读取、写入和拒用的时序缺失预测工作；这是当前最强的新颖性切口，但只能表述为检索结论，不能替代正式查新。
- StableMiss（IJCAI 2023）假设完整特征与标签的联合分布不变，但训练与测试掩码分布可未知变化；它用双重参数化近似各掩码下的不变最优预测器，并通过去相关降低特征、掩码与特征-掩码相关性的依赖。StableMiss+（ICASSP 2024）进一步保留有用的特征内相关，只去除掩码内和特征-掩码相关。因此，CMPE 不能把“掩码解耦/去相关”本身作为宽泛首创。
- CMPE 与 StableMiss 系列的可辩护边界是：对象从静态不完整特征预测转为多变量时序缺失预测；操作位置从样本重加权和多掩码预测器转为 S4M 原型查询/写入表征；方法从全局去相关转为同一时间窗反事实配对、内容/缺失双因子分解，并保留缺失分支以处理信息性缺失。该边界必须通过针对性消融和跨机制实验实证，而不能只靠措辞。
- RAFT（Retrieval Augmented Time Series Forecasting，检索增强时序预测，ICML 2025）直接检索训练集内历史相似片段及其后续值来辅助预测，说明“检索增强预测”也不是新概念。FUSR 的新颖性必须落在“以有/无原型造成的下游预测损失差作为效用监督，控制缺失感知原型库的选择性读写与拒用”，并把 RAFT 加入检索类相关工作或可行时加入强对比。
- ChannelTokenFormer 已由 ICLR 2026 官方页面确认为正式 Poster，处理通道依赖、异步采样与测试时缺失块；它应作为正式强基线，而不是仅列为背景工作。
- TameR 已由 ICML 2026 官方 OpenReview 元数据标识为 regular paper；其通过随机遮挡和全局上下文提升对缺失、噪声和异常扰动的鲁棒性，应作为“遮挡增强是否足够”的对照。
- 当前创新强度判断从“新问题”调整为“已知问题上的主干特异性新机制”：若 FUSR 不能显著降低 harmful retrieval rate（有害检索率），或 CMPE 只等价于普通遮挡增强/一致性正则，则不足以支撑目标论文。
- STRATA-TS（2025 预印本）已在跨城市少数据预测中通过目标感知检索筛选有益源片段，避免无差别迁移造成负迁移；其使用大语言模型推理与蒸馏，场景不是缺失值预测，也没有 S4M 原型库的反事实效用监督，但会削弱“首次选择性知识迁移”的宽泛表述。
- DPAD（Dual-Prototype Adaptive Disentanglement，双原型自适应解耦，2026 预印本）构造常见/稀有双原型库，并以双路径上下文路由选择性融合原型。这是 FUSR 目前最接近的原型路由工作。FUSR 必须明确区别：DPAD 依据模式相似性和稀有性路由，FUSR 依据原型是否实际降低下游预测损失来学习效用；FUSR 还利用缺失域距离和原型历史效用控制 S4M 的读取、写入与拒用。
- 因 DPAD 已占据“通用时序原型解耦+上下文路由”，本项目不再引入常见/稀有双库，也不把通用 prototype routing（原型路由）作为贡献；保留 S4M 两级队列，专注于 missingness-induced retrieval harm（缺失诱发的检索伤害）和 counterfactual utility supervision（反事实效用监督）。
- 2026 年新增的 Cross-RAG 通过交叉注意力选择查询相关检索样本，SARAF（KDD 2026）指出相似历史并不必然具有相似未来并以平稳性感知的相关性/多样性融合提高鲁棒性，RIDDE 则把检索表征分解为不变与动态成分。这些工作说明“相似度检索有害”“选择性融合”“不变-动态分解”均已成为时序检索预测的活跃方向。
- 因此本项目不声称首次发现检索伤害，也不声称首次做选择性检索或不变分解。可辩护新颖性进一步限定为：在缺失值 S4M 的内部原型库中，利用同一样本、同一主干和原型分支开/关产生的下游预测损失差，监督一个同时管理读取、写入、降权与无原型回退的效用模型；现有精确检索尚未发现这一组合。
- 为避免因果术语滥用，文档将“counterfactual（反事实）”限定为 operational model intervention（操作性模型干预）：固定输入和参数，仅切换原型分支或缺失掩码以测量条件性能差异；不声称识别真实世界 MNAR 缺失机制的因果效应。
- 最终模型故事采用供给侧/使用侧双闭环：MIPP（Mask-Intervention Prototype Purification，掩码干预原型净化）解决写入表征受掩码污染；FUSR 解决检索后是否真正有利于预测。两者围绕同一失效链，优于把对比学习、频谱 OT、强化学习和元学习并列堆叠。

## 当前最终方案
- 项目名：MIRAGE-S4M（Missingness-Intervention and Retrieval-utility-Aware Generalization Enhancement for S4M，面向缺失机制漂移的缺失干预与检索效用感知泛化增强 S4M）。
- 创新一：MIPP（Mask-Intervention Prototype Purification，掩码干预原型净化）。用同一完整窗口的多掩码干预视图分离内容表示和缺失表示，只将跨掩码稳定的内容表示写入原型库，同时保留原始掩码和缺失分支。
- 创新二：FUSR（Forecast-Utility-Aware Selective Retrieval，预测效用感知的选择性原型检索）。以“启用原型相对禁用原型带来的预测损失变化”为训练效用标签，学习可靠性路由器，控制原型读、写、降权和拒用。
- MSS（Missingness Shift Suite，缺失机制漂移评估套件）作为固定跨域训练/评估协议，不包装为第三个网络模块。
- MIPP 取代早期 CMPE 名称，以强调作用位置是 S4M 原型库写入而不是泛化的表征对比。
- 最终论文叙事采用“供给侧 MIPP + 使用侧 FUSR”：MIPP 只写入跨掩码稳定的内容原型并保留独立缺失表示；FUSR 以有/无原型预测损失差监督读取、重排、写入、隔离和无原型回退。
- 理论计划包含效用选择后悔界与掩码干预下的 Top-K 稳定性条件；Oracle selector 仅作为诊断上界，不进入测试推理。
- 当前 idea 水平评估为中高、主干特异性强但仍需 Gate H 先验证中心失效现象。达到二区证据强度的关键不是增加模块，而是跨缺失域最差误差、HRR、污染率、自然数据、强基线和多 seed 同时成立。

## 阶段 5 基线入口复核
- 上游仓库没有更近层级的项目指令文件，当前继续遵循本项目 `AGENTS.md` 并保持 `upstream/S4M` 只读。
- 根 README 与 `s4m/Readme.md` 给出的唯一官方示例均为 Weather、输入长度 192、预测长度 96、15 epoch、batch size 使用代码默认值 64，且没有关闭 W&B 的参数。
- `run.py` 强制执行 `wandb.init()`，并把 `args.device` 预先设为 CUDA；这使离线或无 GPU smoke test 不能直接使用官方入口。
- `Exp_Long_Term_Forecast._select_optimizer()` 实际使用 SGD，而论文声明 Adam；正式结果必须分别标识“公开代码协议”和“论文声明协议”，不能静默选择其一。
- 训练会保存模型与原型库 checkpoint，但测试结果目录包含作者机器的绝对路径，指标仅写到文本和 W&B，`metrics.npy` 保存被注释；本地适配器需补足项目内输出与结构化指标，且保持上游不修改。
- 官方测试中的主 MSE/MAE 来自 `seq_y`（缺失 CSV 中的目标），另有 `total_loss` 对完整 `seq_true_y` 计算；正式适配前必须确认哪个对应论文表格，避免指标口径混淆。
- 2026-07-14 环境版本为 Python 3.10.13、PyTorch 2.1.0+cu121、CUDA runtime 12.1，`pip check` 通过。默认隔离环境中 NVML/驱动不可见，但宿主权限下 CUDA 正常；不能再把隔离环境的退出码9解释为服务器驱动故障。
- `MTNet.forward()` 把原型表示硬编码为 `self.memory.repre.clone().detach().cuda()`，与 `Exp_Basic` 的 CPU 分支不一致；这是设备兼容 bug，需在本地基线适配层修复为跟随查询张量设备，并把该修复与创新收益分开。
- `Dataset_Custom4`/相关自定义数据集除读取缺失 CSV 外，还按数据集名称硬编码读取 `/home/project/S4M/data/...` 下的完整 CSV；仓库仅附带 Weather 缺失示例而没有对应完整 Weather CSV，因此正式训练前必须准备完整原始数据和可复现缺失资产。
- 数据加载器对 train/val/test 均使用 `drop_last=True`，且验证集也 shuffle；这是公开代码协议的一部分，但需要在协议审计中记录其与常见评估实现的差异。
- 服务器已有四个完整标准数据集，可避免联网下载：Weather `(52696, 22)`、Electricity `(26304, 322)`、ETTh1 `(17420, 8)`、Traffic `(17544, 863)`，均含 date 和目标列、无缺失，规模与 S4M 论文表一致。
- 四个完整数据集当前来源分别位于 `timeperceiver_meta_uq/data`（Weather、Electricity、ETTh1）和 `ctldf/.../data/traffic.csv`（Traffic）；应通过项目内数据清单和只读链接复用，避免在磁盘 98% 使用率下额外复制约 242 MiB。
- 上游附带的 `weather_missing.csv` 与完整 Weather 形状/列名一致，含 499,716 个 NaN；需要进一步验证其观测值一致性、实际缺失率和缺失模式，才能决定它可代表哪一项论文协议。
- `Dataset_Custom4` 按 0.7/0.1/0.2 顺序划分，训练 scaler 在含 NaN 的缺失训练段拟合，并从独立完整 CSV 构造 `seq_true_x/seq_true_y`；数据适配器只需把完整 CSV 路径参数化，无需重写其核心划分。
- Weather 缺失示例已完成逐值核验：日期完全一致，606,900 个已观测数值与完整 Weather 最大绝对差为 0；499,716 个缺失全部发生在整时间点（23,796 行的21个变量同时缺失），没有部分缺失行。
- 该示例的整时间点 block start 比例约为 `3269 / 52696 = 0.0620`，与论文 nominal r=0.06、连续长度5的 point-missing 协议吻合；重叠后 realized missing rate 为 0.451571。它可作为首个 Weather 快速复现资产，但仍需保存其内容哈希和口径说明。
- 上游全部内容仅一个初始上传 commit，`scripts.sh` 与 README 只提供 Weather 命令，且 epoch 分别为10和15；不存在可从 Git 历史恢复的完整四数据集命令矩阵。
- 工作区的 `M2F-STNet/private_repro` 中已有一个面向 NYCTAXI 的 S4M 内部适配器和已完成的调度/审计资产；其任务协议（P24-Q12、点缺失率25/40/60）与本项目 ICLR 四数据集协议不同，不能复用其结果，但可只读参考其设备适配、训练链路和记录方式，减少重复踩坑。
- S4M 顶层仓库未声明项目级许可证，既有工作区也将其标记为 private/internal、license unresolved；本项目代码和结果暂按内部科研复现管理，不公开再分发上游源码，除非后续许可状态明确。

## 阶段 5 首次 smoke 结果
- Weather CPU smoke 已两次完整成功：数据加载、原型 warmup、forward、backward、1 次 optimizer step、验证、checkpoint 保存、训练后重载、测试、`metrics.json` 和状态资产均完成；因此最小代码链路已经打通。
- 最新成功实验为 `s4m_public_sgd__weather__point__mr0.06__seed2023__smoke__20260714_154515`，明确标记 `scope=smoke_test_not_formal_result`，其 MSE/MAE 不可与论文值比较。
- 相同 seed 的两次 smoke 指标并不完全相同（MSE 约 0.8674 与 0.8802）。最可能原因是官方代码在设置随机种子后调用 `wandb.init()`，其初始化可能消耗全局随机状态，导致模型初始化或 DataLoader shuffle 不一致；需在 W&B 初始化后重新设置 seed 并复测，作为复现修复而非创新。
- 最新 smoke 的持久化证据包含 `optimizer_steps=1`、train/val/test 各1个实际处理 batch，以及 checkpoint 必需键 `model/memory/repre/num_dic` 的 CPU 重载验证。
- 宿主 GPU 核验：物理 index 0 为 RTX 4080、UUID `GPU-c88b...`、当时仅4 MiB空闲；物理 index 1 为 RTX 4090、UUID `GPU-a3af...`、当时约3691 MiB空闲。两卡均被大量占用，尚未启动正式配置。
- 使用 `CUDA_VISIBLE_DEVICES=GPU-a3af...` 后，项目 PyTorch 环境稳定看到单个逻辑0设备 RTX 4090、总显存约24 GiB；后续启动按 UUID 绑定，并继续在记录中保留用户指定的逻辑/物理反向映射说明。
- W&B 后重新 seed 仍未消除差异；进一步通过本地 `Bio.Cluster.kcluster` 文档确认，官方 `npass=100` 每次使用不同随机初始聚类，且这一路径不受项目设置的 Python/NumPy/PyTorch seed 控制。
- 已将 seed 控制的 best-of-100 `initialid` 聚类实现为默认关闭的 `deterministic_cluster` 开关。关闭时保留公开代码行为；开启时作为独立 reproducibility fix（可复现性修复）身份，不能把其收益计入创新模块。
- 确定性聚类开启后的两次相同 seed CPU smoke（`...154959`、`...155054`）得到完全相同的 MSE/MAE/RMSE，checkpoint SHA-256 也完全相同 `75202d...e10b`；固定 seed 可还原性已验证。
- 官方完整架构 GPU smoke 保持 Weather、seq_len=192、pred_len=96、batch size=64、8层、256维和 SGD，只将 epoch/batch 数作为独立 smoke 限制；在 RTX 4090 启动空闲约4914 MiB时于第一个训练 forward 的 attention 路径 OOM。
- OOM 时本实验已占约4.55 GiB，其中 PyTorch allocated 3.91 GiB、reserved-unallocated 184 MiB，并尝试再申请324 MiB；当时卡上剩余约251 MiB。由于尚未进入 backward，完整一步的真实峰值仍未知，下一次只应在明显更多空闲显存下重试。
- OOM 运行已分类为 `OOM`、退出码1，自身 PID 已消失，未清理或干扰其他用户进程；正式架构和 batch size 均未修改。
- 2026-07-15 用户报告 RTX 4080 释放资源后，宿主实测其空闲 14,349 MiB；但后台队列最近已在 RTX 4090 空闲 18,576 MiB 时发生第3次完整架构 OOM，并将动态需求估计上调到 20,992 MiB。因此 16 GiB RTX 4080 不能在不改变正式 batch size 的前提下安全承载当前 fallback 实现。
- 第3次 OOM 发生在 `s4.py` 的 `torch.fft.rfft`，当时本实验占约4.94 GiB（PyTorch allocated 4.30 GiB）并需再申请672 MiB；日志同时明确报告 S4 structured Cauchy/Vandermonde CUDA extension 未安装，正在退回慢且更耗显存的实现。
- 上游提供 `extensions/kernels/setup.py`、`cauchy.cpp` 和 `cauchy_cuda.cu`，可在项目独立环境本地编译；这是官方建议的性能/显存修复，不改变模型结构、batch size、优化器、epoch 或数据协议。编译前必须核验 nvcc、PyTorch CUDA ABI 和磁盘空间。
- 工具链预检结果：系统 nvcc 为 CUDA 11.5、g++ 11.4；项目 PyTorch 为 2.1.0+cu121、CUDA runtime 12.1、C++11 ABI=False，且磁盘剩余约184 GiB。11.5 编译器与 PyTorch 12.1 ABI 存在高风险不兼容，不能直接运行官方 `setup.py install`；先寻找可用 CUDA 12.1 编译器或在项目 Conda 环境安装匹配 toolkit。
- 已在项目独立 Conda 环境安装 `cuda-nvcc=12.1.105`，其版本与 PyTorch 2.1.0+cu121 完全匹配；官方 `structured_kernels` 已按计算能力 8.6/8.9 编译成功，产物 SHA-256 为 `90c9ee5e1a2086bb57c9dda25857c44814baf92005f0705ab82e8cb8002a0a8b`。
- 官方扩展在 RTX 4080 的小张量校验中被 S4M 实际加载，前向最大绝对误差 `3.07e-05`，梯度最大绝对误差分别为 `3.05e-05` 和 `1.07e-03`，前向与反向断言全部通过。
- 上游 S4 使用 `extensions.kernels` 和 `structured_kernels` 两层绝对导入；仅做 `build_ext --inplace` 后原启动入口仍无法发现动态库。实验启动器现显式记录 `S4/s4` 与 `extensions/kernels` 搜索路径，并保存扩展产物哈希；这是运行路径修复，不改变模型。
- 扩展启用后的 4080 尝试4在启动空闲14,349 MiB时运行52秒后 OOM，精确位置为完整模型 `.to(cuda)`，尚未进入 forward/backward。审计发现父启动器自身建立约1.38 GiB CUDA上下文；现已让父进程隐藏 GPU，仅由子训练进程绑定 UUID。
- 修复后准备重试时，4080空闲显存已被其他作业从14,349 MiB降至约2,951 MiB，4090仅余311 MiB。新版等待器以14,336 MiB为初始估计、以12,969 MiB为上次有效子进程可用量，资源不足时持续等待；若再次 OOM，会提高估计并排除16 GiB 4080，避免死循环。
- 2026-07-16 步骤4第5次尝试在 RTX 4080 启动空闲16,053 MiB时成功：完整8层、256维、batch size 64架构完成1个有效训练/验证/测试 batch和1次optimizer step，checkpoint保存与CPU重载均通过。
- 成功 smoke 的模型参数量为5,821,929（22.21 MiB），峰值allocated显存9,482.35 MiB、reserved显存9,550 MiB，总耗时32.01秒。此前模型迁移OOM主要由资源竞争和父启动器CUDA上下文共同造成，并不代表完整模型必然需要超过16 GiB。
- 成功 smoke 的 MSE=0.6923139、MAE=0.6670732，仅用于链路验证；由于只处理1个测试batch且训练预算被smoke限制，不能与论文数值比较，也不能作为正式基线。
- 成功实验 ID 为 `s4m_public_sgd_reprofix__weather__point__mr0.06__seed2023__smoke__20260716_060345`；checkpoint SHA-256 为 `082c3e5c61d9cd40043d728a9851223a70d21d0ebdba9279186954420f09db6d`，官方扩展 SHA-256 仍为 `90c9ee5e...2a0a8b`。
- 步骤5协议审计确认：论文声明按0.7/0.1/0.2时序划分、预测长度96、batch size 16、Adam与早停；公开README示例则未显式batch size且代码默认64，公开训练实现实际走SGD。首轮可对论文解释的正式基线应使用独立身份`s4m_paper_adam`，不能沿用`s4m_public_sgd_reprofix`名称。
- Weather可直接复用官方附带的point-missing、nominal r=0.06、block length=5资产，但其生成seed未知；ETTh1只有完整CSV，必须先用固定seed2023生成同机制只读缺失CSV，并在清单中保存哈希、名义/实际缺失率和生成算法版本。
- 当前`Dataset_Custom4`已支持显式`true_data_path`和任意同列CSV，可直接用于ETTh1；正式配置需使用`data=custom4`维持0.7/0.1/0.2划分，而不能改用`data=ETTh1`的固定12/4/4月传统ETT划分。
- 对上一条“Weather可直接复用官方附带r=0.06资产”的纠正：逐行审计发现公开`weather_missing.csv`实际缺失率为0.451571，与论文Table 8在nominal r=0.06下报告的0.247明显不符；该文件只能作为来源不明的公开示例，不能用于步骤5正式论文对照。
- 官方仓库、Git历史和脚本均未公开Weather/ETTh1的原始缺失mask或生成seed。步骤5因此使用固定seed2023、连续长度5，并把实际缺失率校准到论文Table 8报告值的可复核资产；协议字段明确标记为`approximate_missing_mask_seed_unavailable`，不宣称完全同mask。
- 固定资产实际缺失率：Weather 0.2469826932（13015/52696，论文0.247），ETTh1 0.2308840413（4022/17420，论文0.231）；均为所有变量同步缺失、最短连续段5、无部分变量缺失。
- 步骤5主配置采用论文声明的Adam、batch size 16和early stopping；学习率Weather=0.001、ETTh1=0.005。公开README只给Weather架构命令，故两者暂用其8层/256维/30模式注意力架构；论文未披露ETTh1在候选{2,4,8}中的最终层数，这是正式比较的不确定性。
- 调度器假任务验证覆盖最大并发2、动态补位、依赖顺序、OOM回队列、确定性数据错误不重试和状态恢复，全部通过。正式后台启动因宿主执行审批通道断开被拒绝；宿主进程核验确认没有调度器或训练进程，不能误报为已启动。
- 2026-07-17用户明确授权后，步骤5队列成功以宿主`nohup+setsid`启动；调度器PID 424621、PPID 1、SID 424621。Weather smoke runner PID 424647，训练子进程PID 426010已在物理4090 UUID上出现，训练日志已进入首批forward，四重证据确认并非假启动。
- 启动时物理4080仅818 MiB空闲，物理4090为6332 MiB；调度器按6144 MiB smoke估计只启动Weather一个任务，ETTh1保持`WAITING_RESOURCE`，没有缩小batch或模型。
- 两项paper-Adam smoke均成功，但首轮正式Weather与ETTh1均在完成epoch 1并保存checkpoint后，于epoch 2首个正常batch报`self.memory.repre is None`。公开源码每个epoch在`no_renew=false`时调用`init_bank()`清空原型库，却仅在`epoch==0 and i==0`执行warmup；这是官方多epoch生命周期bug，不是数据、Adam或OOM问题。
- 最小修复保持每epoch重建原型库的公开语义，只把warmup条件改为“epoch 0或本epoch已renew时的首批数据”；`no_renew=true`时仍只在epoch 0 warmup并跨epoch保留原型库。
- 修复后smoke提升为2个epoch、每epoch最多2个训练batch，并强制`memory_bank_warmup_count>=2`；这是生命周期回归测试，不改变正式15 epoch、batch16或完整数据协议。
- Weather失败run完成epoch 1的2287步、耗时782.5秒；ETTh1完成epoch 1的744步、耗时156.3秒。失败状态和第1个epoch checkpoint均保留，但因checkpoint不含optimizer/scheduler/RNG完整状态，不进行不安全续训，后续生成新experiment ID从头正式运行。
- 修复后ETTh1 2-epoch回归smoke `...20260717_195850`成功：日志明确记录epoch 1 `reason=initial`、epoch 2 `reason=renewed`，`memory_bank_warmup_count=2`、`optimizer_steps=2`、checkpoint CPU重载通过，峰值allocated/reserved显存约3059/3600 MiB；原`repre=None`错误未复现。
- 修复后的ETTh1正式attempt 2 `...20260717_195944`已完成epoch 1的744 steps（Vali Loss 0.9987831）并在epoch 2记录`MEMORY_BANK_WARMUP reason=renewed`，正式完整训练已越过首轮共同失败位置；截至该检查无Traceback、AttributeError或OOM。
- ETTh1正式attempt 2最终`SUCCESS`：MSE=1.2079909、MAE=0.8389233、RMSE=1.0900280，2972个optimizer steps、4次原型库warmup，checkpoint重载通过，峰值allocated/reserved显存约3063/3606 MiB，总耗时579.6秒。
- ETTh1论文参考MSE为0.593，而当前近似mask单seed结果高出0.614991（约+103.7%）；该差距不能忽略。已知主要协议不确定性包括原论文mask seed未公开、ETTh1最终层数未公开，以及当前统一采用公开Weather的8层架构；必须在创新前将其作为Gate H诊断的重要输入，不能声称已数值复现论文表格。
- Weather第2次2-epoch回归smoke成功后自动进入正式attempt 2；宿主PID和持续增长日志确认其已完成epoch 3并进入epoch 4，未出现原型库空值、OOM或Traceback。
- 当前4080约12941 MiB空闲但已有两个其他训练进程且利用率99%；步骤5仅剩Weather正式实验，无额外满足依赖的待跑任务，因此没有为凑并发提前启动步骤6或重复提交相同基线。
- 步骤5最终队列状态为`SUCCESS`且调度器正常退出。Weather正式结果为MSE 0.2635257、MAE 0.3011600，相对论文MSE 0.225高17.1%；ETTh1正式结果为MSE 1.2079909、MAE 0.8389233，相对论文MSE 0.593高103.7%。
- Weather和ETTh1均通过最佳checkpoint CPU重载，固定评估mask、resolved config、日志、资源信息和结构化指标已落盘；它们足以作为同代码同协议的创新快速对照，但由于原论文mask seed及ETTh1最终层数缺失，不能宣称严格数值复现。
- ETTh1偏差远大于Weather，说明步骤6必须把“原型是否在缺失域下造成检索伤害”与“ETTh1架构/协议不确定性”分开报告；Gate H先做同一checkpoint的原型分支开/关配对评估，只有观察到稳定且可解释的伤害空间，才进入MIPP/FUSR实现。
- 步骤5最终工件完整性审计全部通过：两项正式run的配置、命令、环境、代码差异、日志、指标、状态、资源记录和checkpoint均非空，结果总表已更新；官方上游树无修改。
- 步骤6最小接入点已确定：S4M的原型读路径位于`Bank.MTNet.forward`，其输出再进入主预测头；因此可以在不更换checkpoint、不训练新权重的情况下旁路原型贡献，并在测试循环内做严格逐样本配对诊断。
- 无原型操作定义为：把`MTNet`中由memory centroid加权得到的检索向量置零，但保留query encoder、原始插补输入、`pred_w/pred_b`、残差和后续S4层。这是最小操作性模型干预，不等同于重新训练一个无记忆模型。
- 原公开测试在`memory_size > topM`时随机采样候选原型。Gate H逐样本效用必须采用可复核的固定候选选择并记录策略，防止随机候选差异被误判为有害检索。
- 当前两个正式checkpoint的`repre`分别为Weather `(4,256)`、ETTh1 `(3,256)`，都小于`topM=100`，所以现阶段不会进入随机候选采样分支；原型开/关配对可保持步骤5的prototype-on输出完全一致。
- Weather 2-batch Gate H smoke显示强方向性信号：65.6%的样本检索有害、平均效用为负，原型相似度接近1但与真实预测效用相关性接近0；这支持继续完整ID与漂移诊断，但32个样本不足以宣告Gate H成立。
- 完整ID纠正了smoke的局部偏差：Weather全测试集原型总体有益（MRU>0），但仍有41.96%样本受害且相似度无法区分效用；FUSR在Weather有明确Oracle空间。ETTh1原型开关影响约为1e-9量级，说明其大基线误差并非由当前原型检索贡献直接造成。
- Gate H漂移配置必须始终加载步骤5的同一原始checkpoint，漂移只允许作用于测试窗口输入。启动前路径校验已拦截一次把漂移数据名误拼进checkpoint路径的配置错误；修复不会改变任何模型权重或评估协议。
- Weather缺失率漂移2-batch smoke中，原型开MSE为0.21501、原型关MSE为0.17218，32个样本全部显示负效用，且相似度-效用Spearman相关为-0.687。样本量过小但说明漂移评估值得完整运行；正式结论必须以后续全测试集为准。
- Gate H正式评估进程只能在宿主视图中核验；隔离环境的`ps`可返回空结果，即使宿主PID、PPID/SID、runner和DataLoader均真实存活。调度器真值必须联合宿主进程、状态更新时间和日志，不能用隔离空视图判定外部终止。
- Weather完整漂移结果否定了一个过强叙事：HRR从ID的41.96%下降到缺失率漂移38.63%和长块形态漂移38.97%，不能声称缺失域漂移提高有害检索率。更稳健的新叙事是：ID和两类漂移下都持续存在约39%–42%的逐样本负效用，而检索相似度与真实效用的Spearman相关接近0，支持学习效用感知的读取/回退机制。
- Weather的Oracle选择器相对prototype-on潜在收益在ID、缺失率漂移、长块漂移分别约5.08%、4.94%、5.42%，跨域非常稳定；这比“漂移加重伤害”更适合作为FUSR的核心可检验假设。
- S4M公开入口把`d_var`绑定到大小写敏感的数据文件名关键词，而不是显式配置；新ETTh1漂移资产使用小写文件名时模型甚至无法构造。对未知名称回退到已记录的`enc_in`是数据身份兼容修复，不属于创新收益，必须经独立smoke后重新生成正式实验ID。
- ETTh1完整ID、缺失率漂移和长块漂移的prototype-on/off MSE差与Oracle收益均在约1e-9到1e-8量级，85%左右样本被数值容差判为neutral。不能把其7%–8%的原始HRR当作真实有害检索；该分支在当前checkpoint上实质失活。
- Gate H不支持“缺失漂移普遍加剧原型伤害”的跨数据集主张，也不支持直接把FUSR作为两个数据集都有效的第一创新。它支持更准确的两阶段机制：MIPP先提升跨掩码稳定原型的供给与有效使用，FUSR再解决Weather已证实的样本级错误读取/回退。
- 步骤7必须遵循蓝图中的MIPP全称与边界：Mask-Intervention Prototype Purification（掩码干预原型净化）。最小实现也必须直接改变S4M原型查询/写入表示或写入资格，不能仅添加与原型库无关的通用对比损失。
- S4M原型通路是MoCo式双编码器：`encoder_u`有梯度并产生检索query，`encoder_m`冻结并由momentum更新，训练forward末尾用`encoder_m`表示更新原型库。MIPP可复用这套动量教师，不需要再复制完整编码器。
- `MTNet.forward`当前只接收衰减插补后的`z_s`，不知道原始mask，也把写入放在`no_grad`内部。为了实现真正的掩码干预净化，需要显式传入配对视图或预计算其内容稳定性，并将“哪些候选允许写入”传给`FeatureMemory`。
- 数据加载器已经同时返回完整历史`seq_true_x`和缺失历史/掩码；训练时可以只从历史完整值生成第二缺失视图，测试仍只走单视图。未来目标不参与视图构造，避免标签泄漏。
- `seq_x`与`seq_true_x`都在同一训练区间拟合的标准化空间中，缺失位置以0填充；第二视图可以安全构造成`seq_true_x * mask_b`，并按公开数据类同样的argmax/argmin规则重算衰减插补统计。
- 为避免MIPP一开始破坏已验证的S4M空间，内容投影应采用零初始化残差适配器并在检索前归一化；这样MIPP开启的初始表示接近基线，同时允许跨掩码损失逐步学习净化方向。
- Gate M的第一轮机制证据至少应落盘：paired content cosine（配对内容余弦稳定性）、stable write acceptance（稳定写入接受率）、各辅助损失、预测MSE及资源开销。后续正式漂移评估再补原型污染率和Top-K稳定性。
- 正式runner会把YAML中`official_args`逐项转为CLI并自动固化resolved config、代码diff、环境、checkpoint重载和资源指标；步骤7应复用该runner，只扩展参数与smoke验证，不另写一套训练入口。
- Weather生命周期smoke的有效训练步为每epoch跳过首批warmup后仅1步，共2个optimizer step；这足以覆盖MIPP两视图、辅助loss反传、稳定写入和跨epoch重建，但其指标不能用于方向判断。
- `run.py`尚无任何MIPP参数；所有新布尔开关必须用`store_true`且默认false，所有权重/阈值必须有显式默认值。关闭态不应实例化MIPP参数或执行第二视图，确保旧checkpoint与基线行为兼容。

## 2026-07-17 步骤7实施检查（续）

- `/mnt/data/zza/code/S4M_MissingForecasting` 不是 Git 根目录，直接执行 `git status` 会失败；应使用 `git -C baselines/S4M ...`。
- `baselines/S4M` 当前还包含步骤4以来的兼容性与复现修复，不能把整个工作区差异都归因于 MIPP；步骤7必须按文件和功能边界核验，并继续保持 `upstream/S4M` 只读。
- 当前静态编译没有语法错误。剩余高风险点是：配对掩码视图的维度与梯度、原型库稳定性写入门控、MIPP 参数随 checkpoint 保存/加载，以及全部开关关闭时的基线数值闭环；这些必须由组件测试和 GPU 冒烟测试确认。
- CPU 组件证据已排除三个风险：MIPP 初始不会随机破坏基线内容空间；联合目标能向内容适配器和缺失模式编码器反传；原型写入过滤接口能按布尔门控缩小候选集合。
- 公平性脚本确认正式对照没有通过更多 epoch、更大 batch、更换数据或不同测试 mask 获取优势。仍待 GPU 端确认完整 S4M 前向/反向、checkpoint 中存在 MIPP 参数，以及默认关闭的基线闭环。
- 第一次 GPU 冒烟暴露了组件测试此前未覆盖的长度20块扩展边界：`max_pool1d` 不接受 kernel=20、padding=19。该错误属于确定性 `CODE_ERROR`，不能回到 OOM 队列无限重试；修复后必须新增长度20形状/有限缺失率测试并使用新 experiment ID。
- MIPP GPU 冒烟已证明完整链路有效：paired objective（配对目标）被执行2次，平均内容余弦相似度0.9643，教师视图余弦相似度0.8784，稳定写入接受率0.9792，所有指标有限，checkpoint 含7个 MIPP 参数键。
- UUID 路由异常不是 NVML 或调度器选卡错误，而是 S4M 在 `_acquire_device` 中二次覆盖 `CUDA_VISIBLE_DEVICES`。由于 `torch.cuda.is_available()` 尚未固定最终设备映射，后续 `.to(cuda:0)` 落到服务器逻辑0（物理4090）。修复必须保留外部 UUID，并只使用受限可见集合内的局部 `cuda:0`。
- 当前正式任务是在路由修复前启动，实际位于4090且已自然运行，因此按“不主动中断已启动任务”原则继续；路由修复只影响未来新启动或 OOM 重试。当前 run 的 `code.diff` 已冻结修复前状态，后续 run 会记录修复后的新差异，二者不会混淆。
- 主MIPP smoke的稳定写入接受率约0.979，说明阈值0.80几乎没有过滤候选。把一个新增变体设为0.95是高信息量选择：若收益提高，说明真正的prototype purification（原型净化）需要更强选择性；若下降，可识别过度过滤风险。
- 第二个新增变体不改模型主干、数据或训练预算，仅关闭内容-缺失因子化并将variance/covariance/decorrelation/mask reconstruction（方差、协方差、去相关、掩码重建）权重置零，保留配对不变性和0.95写入门控。它不是为了凑消融，而是用于判断复杂辅助目标是否稀释核心原型净化信号。
- 配置审计确认两个正式变体除实验标识和预先声明的机制因素外，共享58项S4M训练参数，并完全复用主MIPP的数据资产、固定测试mask、seed2023、15 epochs与batch16。新增实验不会通过增加训练预算或更换评估资产获得优势。
- S4M的checkpoint setting会把`des`、数据文件名和大量超参数拼进单个目录名；较长的变体身份会触发Linux `NAME_MAX`，即使完整路径整体未超过`PATH_MAX`。修复应只缩短身份字符串，不能改训练参数来绕过。
- 修复后严格门控与简化门控的smoke/formal setting长度均为240，留下15字符文件名余量。当前4080空闲约10.4GiB，但已有约5.6GiB其他占用；两个预计各约4.95GiB reserved的MIPP任务同时共卡只剩约0.5GiB，不足CUDA工作区安全余量，因此不能为表面达到并发3强行同卡启动。
- 短身份修复后的简化门控 smoke 已以新实验 ID 成功：外部 GPU UUID 在 S4M 内部保持为受限可见集合，训练记录显示 `cuda:0` 对应目标 4080 UUID；配对视图、辅助目标反传、跨 epoch memory-bank warmup、MIPP checkpoint 键和 checkpoint CPU 重载均通过。由此可把该 smoke 的 `SUCCESS` 作为其同协议 15-epoch正式任务的唯一依赖解除证据，而非把受限 batch 指标当作研究结论。
- 简化严格门控正式任务已由依赖自动接续并真实运行；其正式命令保留 15 epochs、batch16、完整训练/评估 batch、Adam 和固定 Weather mask/seed，只关闭内容-缺失因子化与其四项辅助损失。当前两张卡的实际可用显存不足以再安全放入严格门控 smoke，故“最多并行3”在此刻表现为2个活跃任务加1个资源等待任务，而不是人为降低并发限制。
- 步骤7单seed筛选中，只有pair-only t=0.95同时改善Weather MSE（约0.60%）和MAE（约2.42%）；完整因子化MIPP在t=0.80/0.95下都只改善MAE而没有改善MSE，且t=0.95的MSE退化约1.01%。这更支持把MIPP收敛到“配对不变性 + 写入控制”的紧凑方案，而非保留全部辅助目标。
- 绝对余弦阈值没有产生持续强净化：主t=0.80正式运行的稳定写入接受率约99.96%，完整t=0.95约99.53%，pair-only t=0.95约99.91%；只有warmup接受率下降到约87%–100%。因此当前pair-only收益更可能来自跨掩码配对不变性，而不是已充分验证的稳定写入门控。后续若强化MIPP，应考虑自适应分位数/容量约束门控，而不是继续手调固定余弦阈值。
- `stage7_mipp_comparison` 的 `PROTOCOL_MISMATCH` 是报告器实现误报，不是训练协议真实不一致：比较器的allowed差异集合没有排除四个run专属输出路径。报告还只读取主队列的main MIPP，漏掉了已经SUCCESS的SG95与最佳PG95。步骤7结束前应修报告而不重跑GPU，并在报告中区分“项目内固定mask同协议”与“原论文mask seed未公开”。
- 项目蓝图中的正式FUSR全称是 `Forecast-Utility-Aware Selective Retrieval`（预测效用感知的选择性原型检索），不是字面上的frequency uncertainty routing；项目规范中的“频率不确定性选择路由”是早期/简写漂移，步骤8实现与后续文档应统一回正式定义，避免论文叙事自相矛盾。
- FUSR的可部署监督必须来自同一训练/验证样本的原型开/关预测损失差，测试目标不能进入路由器。Gate H的Weather测试诊断只能证明约5% Oracle空间，不能直接拿测试utility训练路由器；应复用开/关干预代码在训练/验证split生成标签，测试只输入历史、mask和检索统计。
- 当前最短且仍符合创新定义的步骤8版本应先做`read gate + abstention`（读取门控与无原型回退），不在首轮同时实现单原型Top-K探针、写入隔离和淘汰。先验证效用路由是否能提高预测并降低有害检索；写入治理留作独立后续开关，避免第一次实验无法定位收益来源。
- S4M可在不修改上游快照的情况下实现训练期双分支、推理期单分支：`MTNet`一次检索后构造prototype-on与prototype-off两个上下文，训练时把两者沿batch维拼接后共用下游S4并行计算，路由输出由两预测软组合；测试时先用历史/mask/检索统计得到门控，再只对门控后的检索上下文执行一次下游S4。这样utility标签阶段约2倍下游训练开销，但部署推理不需要原型开/关双预测。
- FUSR最小可部署特征可固定为7类历史可得量：Top-1相似度、选中相似度均值、归一化检索熵、输入缺失率、最长连续缺失比例、变量间缺失率标准差、缺失二元熵。目标Y只用于训练时从detached的on/off预测计算utility回归和正负分类标签，绝不拼入路由器特征。
- FUSR首轮机制指标至少保存：utility regression/sign loss、真实正效用率、预测效用均值/方差、门控均值、abstention rate（拒用率）、utility sign accuracy/AUROC、训练双分支开销；评估再报告路由相对prototype-on的HRR与MSE变化。默认关闭时不实例化路由器、不计算额外统计、不改变现有forward。
- 当前`Exp_Long_Term_Forecast.change_mode(0)`会把`mem_net.is_training`关闭，因此验证/测试双分支不会回写原型库；训练期FUSR在一个`MTNet.forward`内生成on/off上下文也只会执行一次现有memory update。这是避免效用探针污染原型库的关键复用点。
- checkpoint保存路径已经序列化完整`model.state_dict()`，只要FUSR路由器作为`MTNet`子模块就会自动进入checkpoint；需要扩展runner的checkpoint key断言和metrics落盘，但无需另造保存格式。
- 现有`vali()`和`test()`调用同一个默认forward；FUSR推理单分支必须成为该forward在`model.eval()`时的默认行为。训练双分支只在`self.training and use_fusr`时启用，避免验证/测试无意执行utility标签双路径。
- 队列引擎原有`active_gpu_uuids`语义会把任何正在运行的GPU完全排除，因此即使`max_jobs=5`且单卡剩余显存足够，两张GPU的实际并发也永远不超过2。并发5不能只改参数上限；需要允许同卡共存，同时保留启动锁、真实空闲显存检查、安全余量和启动后显存稳定窗口。
- 对当前S4M队列最小安全改造是“每张GPU每个轮询周期最多新启动1个任务”：已运行任务不永久屏蔽该GPU，下一轮询只在nvidia-smi已经反映前一任务显存后才允许继续补位。该策略比同一轮连续启动多个任务更能避免模型尚未完成CUDA分配时的过量提交，并仍可在多轮内达到最多5个并发。
- 修改后的队列自测实际观察到`max_observed_running=5`，证明并发上限5不只是参数校验放宽；同卡补位、OOM退避回队列、确定性错误不重试、依赖和恢复语义仍全部通过。
- FUSR独立审计建议把可关闭的频域不确定性定义为历史观测谱与衰减插补谱之间的Jensen-Shannon divergence（詹森-香农散度）；它只能作为路由特征之一，不能替代“预测效用监督”这一核心创新，也不能接触未来目标。
- FUSR训练期on/off预测必须采用common random numbers（公共随机数）控制dropout随机性，否则两分支误差差会混入随机失活噪声。简单沿batch拼接不能保证两份样本共享相同dropout掩码，因此首版实现应优先使用保存/恢复随机状态的共享主干双前向。
- FUSR路由器首版仅增加452个参数，输入接口只接受衰减插补历史、历史mask和检索统计；utility训练目标在实验循环从on/off预测误差生成后立即detach，因此改变未来目标不会改变同一历史的推理gate。
- 训练采用两个共享权重的下游前向并保存/恢复CPU与CUDA随机状态，使同一batch的on/off路径使用公共dropout随机数；验证/测试使用硬gate后的单下游前向，paired on/off仅作为明确开启的评估诊断，不属于部署推理成本。
- 步骤8正式配置已保持Weather基线的固定mask、seed2023、batch16、Adam、15 epoch上限、patience3和完整数据协议，且`use_mipp=false`；首轮FUSR收益不会混入步骤7 PG95或增加主干训练预算。
- FUSR V1在项目内同协议正式测试中没有实现选择性：hard gate对10432个测试样本全部为1，abstention=0，routed输出等于always-on，HRR降低为0，utility AUROC=0.4946。这比MSE仅变差0.0681%、MAE改善1.2080%更关键，说明当前路由器科研机制未成立。
- 训练utility标签正类率97.98%，而测试原型有益率仅52.01%，存在严重train-test utility分布错位；94.78%的训练sign accuracy主要是类别失衡表象。下一版必须先修标签生成/平衡和validation校准，不能继续手调固定test阈值或直接铺多seed。
- 当前gate并未使用sign head，且高gate prior在正式checkpoint仍约6.149，使utility均值/不确定性即使变化也很难触发拒用。更合理的V2应让平衡后的sign/ranking score参与决策，并移除、冻结退火或通过validation校准抵消该prior。
- V1的Oracle sample MSE比always-on低3.93%，说明选择性读取仍有可检验空间；因此当前结论是“保留FUSR问题、放弃V1训练/门控实现”，而不是放弃整个使用侧创新方向。
- V2有一个比继续joint training更干净的候选：加载步骤5 baseline checkpoint、冻结S4M主干与prototype memory，只生成一次stationary（固定模型）train/val on-off utility cache，再训练数百参数router并用validation校准阈值。这样always-on路径可严格恢复同一baseline，任何测试收益都能归因于selector，而不会再混入+60%主干训练步数。
- 冻结后缓存只需保存历史可得features、observed-future上的on/off SSE/count和detached utility，不保存或使用test标签。router可在小缓存上使用全局类别平衡BCE、正负pairwise ranking和固定尺度Smooth L1，避免V1可学习log-variance负项主导。
- 当前Dataset_Custom4训练/验证返回`seq_y`、`seq_y_mask`和仅供评估/诊断的`seq_true_y`；为保持与baseline监督预算一致，V2训练utility仍必须使用`seq_y_mask`上的观测未来值，不能把完整隐藏`seq_true_y`偷偷加入训练。validation可用于阈值校准，但test只做一次最终评估。
- V1标签审计排除了“相对效用归一化改变符号”这一假设：train/test效用符号定义一致；真正错位来自联合训练时backbone、memory和router同步变化、train memory逐batch更新而validation/test冻结，以及已观测未来与完整未来的监督范围差异。V1 checkpoint的`gate_prior_logit=6.149388`配合temperature=0.25还使关门条件落到裁剪效用范围之外，结构上几乎必然全开。
- 冻结步骤5checkpoint具备精确归因条件：Gate-H always-on与步骤5基线MSE绝对差仅`1.57e-8`，forced-on/off组件闭环误差为0；其测试有益检索率58.04%、Oracle相对空间约5.08%，比联合训练V1被压弱后的treatment effect更适合作为固定两臂。
- 随机窗口交叉验证会制造虚假路由可预测性：现有简单特征随机5折AUC可达0.924，但严格chronological切分仅0.430–0.527。快速V2因此使用validation内fit/dev/calibration时间块并设置embargo；该结果只叫stationary validation diagnostic，不包装成严格无泄露正式协议。若快速筛选有效，必须重训独立base early-stop/router-fit/router-calibration/test嵌套时间块。
- FUSR V2新增的6个intervention features（干预特征）全部只来自历史与模型内部：Top-1相似度时间标准差、归一化检索熵时间标准差、retrieved/query范数比、二者余弦、context on/off相对扰动范数与上下文余弦；与原8维合成14维。它们在路由决策位置直接描述“原型实际改变了多少内部状态”，比V1近常数的4原型均值统计更贴近效用机制。
- V2 smoke工程链路已成立但效果不能外推：640个缓存样本上fit AUROC约0.878、测试40批AUROC约0.613；validation校准全开而测试全关，造成routed MSE相对always-on退化10.45%。该现象说明阈值的跨时间分数稳定性是正式V2首要判据，不能只看训练AUC或HRR下降。
- V2 formal证明工程归因成立但学习假设失败：frozen backbone/memory、validation-only calibration、checkpoint和协议工件全部PASS；MSE/MAE却相对基线分别退化0.0150%/0.0071%，测试只关闭17/10432个样本，AUROC=0.4744，说明“全局静态14维特征到效用符号”的映射不能跨时间稳定泛化。
- V2的fit/dev/calibration AUROC从0.766降到0.337/0.379，且校准风险最优仍是全开；这排除了“只需换一个固定threshold”的解释。V3应把时间块视为environment（环境/任务），先验证效用方向在不同环境是否一致，再考虑group-balanced、distributionally robust或episodic meta-routing，而不是继续加普通MLP容量。
- V3筛选的主判据不能再用fit AUC或随机交叉验证：应在validation内部按时间顺序形成训练、阈值选择和最终审计块，阈值在最终审计块之前锁定；只有最终审计块的counterfactual routed MSE（反事实路由均方误差）优于always-on且拒用率非零，才说明方案值得占用GPU。
- 零GPU V3审计显示8个静态候选全部失败；即使限定为fit环境内方向较稳定的`selected_similarity`、`top1_similarity_temporal_std`和`retrieved_query_cosine_mean`，最终audit块仍退化约0.0134%。因此当前失败不能靠线性化、最坏组优化或时间块集成解决。
- dev与audit块的原型Oracle MSE空间只有约0.705%和0.520%，且所有候选在dev上的最优门控均为全开；少量有害读取的代价结构无法由当前静态聚合特征稳定辨识。继续扩大MLP、调阈值或多seed只会扩大负实验，不应占GPU。
- 更契合FUSR定义的V3是HCUP：对当前预测原点`t`，使用完全位于过去的窗口`[t-288,t-96]`预测`[t-96,t]`，比较同一冻结S4M的prototype-on/off历史误差，再用该局部反事实效用决定当前`[t,t+96]`是否读取原型。它以目标无泄漏的每样本自适应信号替代跨时间不稳定的全局映射，并保持FUSR仍然是对原型读取的改进。
- HCUP必须显式报告额外前向次数、延迟和显存；测试评估只能使用历史窗口中实际可观测mask位置，不能使用`seq_true_y`补全的隐藏值。若test loader非严格时序或数据接口无法提供更早窗口，则不得静默实现近似泄漏版本。
- `Dataset_Custom4`按0.7/0.1/0.2切分时为val/test各保留`seq_len=192`的左侧上下文，test loader明确`shuffle=False`；因此样本index>=96可用dataset index `i-96`构造probe，probe target恰好结束于当前forecast origin，不与当前`[t,t+96]`未来重叠。index<96必须回退prototype-on。
- HCUP smoke的32个有效探针中，历史normalized utility与下一时段真实utility Pearson约0.979，threshold0下相对prototype-on MSE改善约2.77%；这是强可行性信号，但样本来自相邻且高度重叠的早期validation窗口，必须由完整后段时序审计确认，不能直接声称创新有效。
- 完整HCUP validation推翻了早期smoke外推：5072个有效样本上probe-current utility Pearson=-0.132，threshold0使MSE退化约1.52%；7种直接/EMA/rolling聚合在严格后段audit全部失败。早期32样本0.979属于局部状态而非稳定规律。
- 当前HCUP的关键结构问题不是平滑器，而是`lag=pred_len=96`造成反馈陈旧。更合法且更局部的候选是short-horizon retrospective probe（短跨度回顾探针）：在`t-h`发起同一96步模型预测，但只用已在`t`之前发生的前`h`步误差比较on/off；只要`h>0`且probe target严格结束于`t`，就不需要等待完整96步。
- 短跨度HCUP不能凭直觉占GPU：先用现有complete validation utility序列计算不同lag的时间持续性与“前一效用作为gate”的后段风险上界。该上界若在lag 8/16/32也无正收益，则实际observed-history probe更不可能成立，应停止该支路。
- 乐观上界证实短滞后有可检验空间：lag1/4/8/16的后段MSE上界约0.77%/0.73%/0.65%/0.43%，MAE也同步改善；lag32开始MAE转负。该曲线提供了预声明候选集合，而不是事后无限搜索lag。
- lag1的合法时间关系是：当前input为`[i,i+191]`、当前future从`i+192`开始；probe index为`i-1`，其第一预测点是`i+191`，已经包含在当前历史末端。因此用probe前1步observed error路由当前future不接触`i+192`之后目标；lag h同理只允许使用probe前h步。
- 短跨度HCUP的8个GPU任务最终全部SUCCESS且工件PASS；h4在4080、h1/h8/h16在4090，当前on/off数组最大跨硬件绝对差仅`6.20e-6`。这属于浮点舍入级差异，最终分析以显式`atol=1e-5, rtol=1e-6`判定数值一致，同时保留`exact_identical=false`证据。
- lag1/4/8/16各7种因果聚合在严格后段validation共28项全部FAIL。所有具有实质选择性的候选都使MSE和MAE下降；零退化候选实际为gate rate=1的always-on退化解，不能算创新收益。
- 步骤8证据链支持停止历史反馈式FUSR：短滞后complete-utility乐观上界为正，但实际可部署observed-history probe噪声/偏差足以抹去收益。该负结果禁止通过test继续挑lag或阈值，也不应与PG95拼接包装。
- 步骤9起始审计确认PG95的固定稳定性写入门控几乎未工作：22860个训练step的平均接受率为`0.9991415`，10次memory warmup平均接受率为`0.90`；paired content cosine已达`0.9999283`。因此PG95的双指标收益更可能来自跨掩码配对一致性，而不是固定0.95阈值形成了有效容量控制。
- PG95相对基线新增153130参数但关闭content-mask factorization，仅保留权重0.05的pair invariance与稳定性写入筛选；正式MSE/MAE为`0.26194646/0.29387891`。步骤9首个候选应改变“写入选择机制”本身，而不是继续微调固定余弦阈值或恢复已失败的复杂辅助损失。
- BMSA（Budget-Matched Stability Admission，预算匹配稳定性准入）利用一个现有结构事实：Weather正式batch16、seq_len192、short_len32产生约96个候选，而FeatureMemory原本最多随机抽`M=50`。因此按配对稳定性Top-M选择50个可在保持候选/写入计算预算不增加的同时，消除PG95“几乎全通过后随机抽样”的失效链。
- BMSA首版不新增可学习参数、不使用未来目标、不改变pair loss、M/TopK/TopM/memory size或训练预算。接口保留`absolute`默认模式与新的`budget_topk`模式；模块关闭时PG95及基线行为必须恢复。
- BMSA smoke实测把96个候选稳定选为50个：effective acceptance=`0.52083337`、eligible/sample均为50；同批原固定0.95阈值接受率为0，平均稳定性0.8766、有效cutoff 0.8837。这证明相对预算排序能在绝对尺度随训练阶段变化时保持容量，而不是简单放宽/收紧一个固定阈值。

## 2026-07-21：步骤9 BMSA正式结果

- BMSA在Weather point-r0.06、seed2023同协议下取得MSE/MAE `0.260696429/0.293372790`，同时优于PG95的`0.261946457/0.293878906`，相对改善`0.4772%/0.1722%`。
- 参数量和峰值显存与PG95一致；BMSA复用既有M=50预算，从每批96候选中按稳定性Top-M准入，实际有效接受率52.0833%，没有靠扩大容量获得收益。
- 固定0.95门控在正式PG95/BMSA统计中对候选的原始接受率约99.9%，几乎没有选择性；BMSA为“配对掩码一致性提供训练约束 + 相对稳定性预算控制原型供给”的两层机制故事补上了可测的选择环节。
- 当前证据等级仍是单数据集单seed方向性正结果。提升相对PG95较小，下一步必须用预冻结配置做配对多seed复核，避免把随机波动误判为创新收益。

## 2026-07-21：步骤10多seed预注册与资产审计

- 第一轮预注册seed集合为`{2023, 2024, 2025}`；2023沿用已完成结果，2024/2025无条件同时进入PG95/BMSA成对复核，不依据中间test结果挑seed或改BMSA预算。
- 现有`generate_paper_point_masks.py`虽然接受`--seed`，但输出CSV/NPZ/JSON路径硬编码为seed2023。直接以2024/2025运行会覆盖seed2023固定评估资产，因此必须先把输出路径参数化并增加“不覆盖不同seed工件”的回归门禁。
- 每个新增seed同时改变训练随机种子和与其绑定的固定缺失mask；同一seed的PG95/BMSA必须引用完全相同CSV、NPZ和metadata。这样检验的是完整实验重复性；后续如需区分模型随机性与mask随机性，再增加固定mask/多模型seed的方差分解消融。
- 新增seed配置继续保持Weather、point-r0.06、L=5、batch16、Adam、15 epoch上限、patience3、完整batch、pair weight0.05和M=50不变。BMSA相对PG95只允许改变写入模式与预算字段。

## 2026-07-21：步骤10 Weather三seed最终结果

- 队列与8个任务全部`SUCCESS`，所有formal checkpoint均通过CPU重载；seed2024首次PG95 smoke的空候选诊断失败被保留，修复后仅重排该smoke/formal并成功，未重跑其他成功任务。
- BMSA相对PG95在seed2023、2024双指标胜出：MSE分别改善0.4772%和0.7537%，MAE分别改善0.1722%和0.3198%；seed2025则MSE/MAE分别退化1.4661%/1.3595%。
- 三seed均值PG95为MSE/MAE `0.263197057/0.296613489`，BMSA为`0.263352377/0.297461341`；平均逐seed相对增益为MSE `-0.0784%`、MAE `-0.2891%`。两项bootstrap 95%区间均跨0，配对检验不显著，且n=3统计功效有限。
- 协议门禁全部PASS，参数量同为5,975,059，峰值显存约5,036 MiB。seed2025两方法都训练13,716 optimizer steps，排除了训练预算差异；不稳定性来自写入选择机制本身。
- PG95每批约96个绝对阈值合格候选后随机采50个，BMSA固定选择稳定性最高的50个；两者最终实际更新数都约50。现有证据说明“最高跨掩码稳定性”等价于“最有用原型”的假设并不跨seed稳定，硬Top-M还可能牺牲内容覆盖或多样性。
- 当前决策为`BMSA_THREE_SEED_UNSTABLE_REVIEW_BEFORE_EXPANSION`：不选择性删除seed2025、不宣称稳定提升、不直接扩数据集。下一候选应在同一M=50预算内联合稳定性与多样性/覆盖度，并先针对失败seed与一个正向seed做快速公平验证。

## 2026-07-21：联合准入源码审计

- 真实网络入口为`model/Bank.py`中的`MTNet`，不是独立`MTNet.py`。写入候选在teacher encoder输出并归一化、经过MIPP content adapter后形成`[B,Q,H]`特征；Weather每批为96个候选。
- BMSA当前在`MIPPPurifier.select_write_candidates`中只接收标量稳定性，没有候选内容特征，因此硬Top-M无法显式控制内容冗余。要实现联合准入，最小接口扩展是额外传入detached normalized candidate features，且仅新模式读取它们。
- `FeatureMemory.add_features_from_sample_learned`仍执行最终M=50更新；PG95为约96个eligible后在memory层随机抽50，BMSA提前选50后全部传入。联合模式必须继续输出至多50个eligible，不能再让memory层扩大采样或改变写入数量。
- 原型检索使用checkpoint内`memory.repre`，并会在超过`topM`时随机采样候选、随后取TopK相关原型。checkpoint诊断可直接比较`repre`的两两余弦、有效秩和聚类覆盖，而无需加载test标签；但需先确认`repre/num_dic`在state_dict中的真实形状与槽位语义。
- `exp_pretrain1.save_model`确认正式checkpoint同时保存`memory`（每个槽位的原始向量队列）、`repre`（槽位质心）和`num_dic`（槽位占用计数），加载时三者原样恢复。诊断可同时覆盖质心间分离度、原始向量簇内紧致度、有效秩和槽位占用熵。
- 训练默认每个epoch清空并重建memory，checkpoint保存最佳验证epoch对应的最终bank。seed2023/2024的PG95与BMSA早停步数不同，因此跨方法最终bank差异只能作描述性机制线索；seed2025两方法optimizer steps完全相同，是判断写入选择效应的更强配对证据。

## 2026-07-21：checkpoint原型库诊断结果与首候选冻结

- 六checkpoint目标无关诊断`PASS`。所有最终bank都只有3个质心、每槽5个向量，质心余弦均大于0.999；余弦≥0.95冗余率和槽位熵完全饱和，不能作为后续选择规则。
- 有区分力的预注册指标是有效秩：BMSA相对PG95的库存向量有效秩比例在seed2023/2024分别`+0.03755/+0.01736`，在同训练步数的失败seed2025为`-0.08957`；质心有效秩比例同向为`+0.03054/+0.05218/-0.13365`。
- 该三点描述性关联支持“硬稳定性Top-M可能牺牲内容覆盖”的机制假设，但n=3且前两组早停轮次不同，不能作为预测因果证明；新规则仍必须通过冻结后的正式对照。
- 首候选冻结为`stability_diverse_rank`：候选稳定性和相对已选集合的新颖性分别转为百分位秩，首项取最高稳定性，随后等权秩融合贪心选满M。规则无可学习参数、无额外权重超参、不使用未来目标，并保持写入预算50。

## 2026-07-21：SDRA50 GPU smoke与调度证据

- seed2024/2025两个独立GPU smoke均完成forward、backward、optimizer step、checkpoint原子保存、checkpoint重载与完整评估，证明`stability_diverse_rank`不是仅通过静态测试的未执行分支。
- seed2024 smoke的训练写入阶段将已选集合平均两两余弦从候选的`0.8883517`降至`0.8851728`，在预算固定50的前提下产生正多样性增益`0.0031790`；smoke数值不能代表正式精度，但支持模块机制已按设计工作。
- 旧队列“同GPU每轮最多启动一个任务”的保护不足以覆盖超过一个轮询周期的数据加载延迟：seed2024 formal启动30秒后CUDA峰值尚未完全反映，seed2025 formal再次看到约8.3 GiB空闲并同卡启动，随后首batch OOM。该问题属于调度准入竞态，不是SDRA模型错误。
- 通用队列现增加未兑现预留：对每个运行任务以`max(0, launch_reservation - 训练PID实际显存)`计入outstanding reservation，并从实时free中扣除；合成复现中8337 MiB表面空闲、7168 MiB新任务需求、已有1536 MiB未兑现预留时会正确拒绝二次启动。
- seed2025 OOM失败工件被保留，调度器已将其分类为`RETRYING`并提高显存估计/设置退避；研究比较只接受后续完整成功run，不把失败或smoke指标混入正式结果。

## 2026-07-22：SDRA正式负结果收口

- SDRA两个formal最终均SUCCESS、checkpoint重载和协议检查均PASS，但seed2024相对PG95的MSE/MAE退化0.5184%/0.2718%；seed2025仅MSE改善0.2662%，MAE退化2.7434%。两个锚点均未双指标胜出，预注册精度门失败。
- 新模式和M=50预算真实执行，但seed2024选后两两余弦高于候选`6.60e-6`，seed2025仅降低`3.33e-6`；正式内容特征余弦集中在0.9992–0.9998，说明该表示空间中的贪心内容多样性基本没有可用动态范围。
- 报告器现将`artifact/protocol/execution/mechanism/accuracy decision`分离。负机制结论不会再被误写成工件损坏，也不会因抛错而从报告中消失。
- 当前证据否决的是“在归一化内容表示上继续调多样性权重”，不是所有覆盖机制。下一假设必须使用未塌缩且与缺失任务直接相关的空间；历史输入mask的缺失形态是首个待验证候选，但必须先用train/validation诊断，不能依据已查看test结果挑规则。

## 2026-07-22：缺失形态空间诊断与MRBA冻结

- Weather的固定point mask对全部21变量共享同一时间行缺失状态；候选级形态因此可无损表示为长度32的二值时间模式，恰好与每个memory write candidate对齐。
- seed2024/2025训练候选分别包含7946/8201种独特模式，模式熵约0.750/0.757、有效秩比例约0.761、随机对平均Hamming距离约0.369/0.378；这与内容表示0.999+余弦塌缩形成明确对照。
- validation最近训练形态的Hamming均值仅0.0229/0.0197，但validation内部有效秩仍约0.763/0.758；这支持用训练可覆盖的连续mask几何做准入，而不是引入无法外推的新离散标签。
- MRBA首版不采用Sinkhorn，原因是快速筛选阶段不应同时引入温度、迭代次数和regime count。冻结规则使用facility-location边际覆盖：每次候选加入后，以其降低全候选到已选集合最近mask距离的总量作为覆盖增益；增益秩与稳定性秩等权融合，无可调系数。
- mask描述子固定由候选历史mask生成：长度32的逐时刻缺失比例，加整体缺失率、时间转换率、变量缺失率标准差/最大值和受影响变量比例；不使用数值、未来标签或test统计。
- MRBA组件与GPU smoke证实设施选址分支确实改变了写入选择：两个smoke中，对全候选mask的平均最近距离相对纯稳定性Top-M分别下降`0.0203086`和`0.0221439`，同时保持每批写入预算50。该结果证明机制可执行且覆盖度提高，但不能替代完整formal的预测精度门。
- 两个正式锚点已在不同物理GPU上真实运行，单任务显存约5.95–6.12 GiB；未出现SDRA阶段的同卡启动竞态，说明新增未兑现显存预留逻辑在本轮调度中生效。

## 2026-07-22：MRBA双锚点结果

- MRBA的设施选址覆盖目标在正式训练中可靠成立：seed2024/2025相对稳定性Top-M的平均mask覆盖距离均下降约`0.02115`，且写入预算、参数量和峰值显存与BMSA一致。
- 机制成立并不等于预测收益成立。seed2024相对PG95双指标改善`0.7608%/0.2884%`，但seed2025反向退化`1.4363%/1.9978%`；两seed均值MSE/MAE分别差`0.3071%/0.8437%`。
- MRBA相对BMSA在两个seed的MSE均有极小改善（约`0.0072%/0.0294%`），但MAE均恶化（约`0.0315%/0.6298%`）。这提示mask覆盖可能影响少数大误差样本，却没有改善大多数点；目前只是待validation验证的解释，不能据test结果直接据此设计新规则。
- 新候选不能继续只改写入覆盖。合理的证据门是先在冻结checkpoint的validation逐样本误差上检验收益是否与稀有mask regime或误差尾部稳定关联；若无跨seed一致关联，应终止mask-space路线。

## 第12步结论：MRBA 的覆盖机制没有转化为稀有缺失形态上的精度收益

- MRBA 在第11步确实改善 mask-space 覆盖，但第12步冻结 checkpoint 的 validation-only 诊断显示：两个 seed 的最稀有四分位 Q4 均同时劣化 MSE 与 MAE。
- 这排除了“总体指标被常见模式稀释、MRBA 其实只在罕见模式有效”这一关键解释；机制—精度错位是真实的，而非总体均值掩盖。
- 因而不再投入 mask-conditioned retrieval（按缺失掩码条件检索）或继续调 MRBA 超参数。下一创新应转向直接约束预测误差/不确定性的目标空间，而不是继续优化 mask-space 几何覆盖。

## 第13步候选接口审计：从“预测效用”训练目标入手，而非继续学习路由器

- S4M 的原型读出先形成 retrieved context，再与 query 和 raw history 融合；PG95/MIPP 主要改变写入表征和准入，前几轮失败路线都没有直接约束“启用原型不得比禁用原型更差”。
- 已有 FUSR 证明启用/禁用原型的反事实效用标签可以生成，但效用路由器在时间外推时失效；因此下一候选不再预测未来效用，也不依赖 mask-space，而是在训练期直接用 paired prototype-on/off error 约束融合主干。
- 最紧凑的候选是无测试时额外开销的 counterfactual regret regularization（反事实后悔正则）：保留 PG95 预测损失，并只惩罚训练样本中 prototype-on error 高于 detached prototype-off reference 的部分。它直接针对有害检索，推理仍固定单次 prototype-on 前向。
- 实现前还需确认同一主前向能否复用 on/off context 完成两路预测且只更新一次 memory；若必须第二次完整 forward 并导致重复写库，则需在模型内部增加只读 counterfactual head 路径，不能靠外层重复调用破坏公平协议。

## 2026-07-23：CFR结果否决“直接正后悔铰链即可改善泛化”

- CFR工程和机制执行均成立：训练期共享一次memory更新生成prototype-on/off预测，off参考停止梯度，测试保持单路prototype-on；正式13716步无工件或协议异常。
- 训练期prototype-on平均MSE `0.04157`明显低于off的`0.04896`，signed regret为`-0.00739`，有害检索率只有`4.41%`，正后悔均值仅`3.04e-4`。因此PG95的大多数训练样本上原型已占优，CFR实际只给少数样本施加很小的附加梯度。
- 即便如此，CFR测试MSE/MAE仍相对PG95退化`0.5554%/1.4330%`。这说明少数训练期on/off反事实损失差并不是稳定的测试泛化目标；继续调权重会变成依据已看test结果的超参搜索。
- 当前否决的是无条件的逐样本正后悔hinge正则，不等于否决所有风险学习。下一候选若继续目标空间路线，必须先有独立validation证据定义稳定风险组或不确定性目标，不能直接把CFR改成CVaR/加权hinge后重跑。

## 2026-07-23：步骤14冻结诊断接口审计

- 步骤12入口已具备冻结checkpoint加载、严格时序`shuffle=False` validation loader、逐样本完整目标误差、固定历史mask、原子NPZ/JSON和无model/memory更新工件门，可复用其工程框架。
- 步骤14不能直接复用步骤12单路输出schema：需要同一批次依次生成原始prototype-on、原始prototype-off和嵌套遮挡prototype-on三路预测；三路都必须在`change_mode(0)`与`model.eval()`下执行，防止重复写memory。
- 嵌套遮挡必须基于`seq_true_x`重建输入及max/min extrema，且只能在原来已观测的历史行上继续遮挡。目标新增缺失率为15%、因果块长5，对应起点概率`1-(1-0.15)^(1/5)`；使用样本全局index派生确定性随机种子，可保证seed2024/2025各自可复现且不同GPU不改变扰动资产。
- 主风险信号应直接保存逐样本`mean((prediction_nested - prediction_original)^2)`；prototype-on/off差异、off误差和有害检索标记同步保存但只能作为次级解释。
- 步骤14启动前GPU快照：物理0号RTX 4080空闲`16053 MiB`，物理1号RTX 4090空闲`24206 MiB`，两卡利用率均为0%；此前无scheduler PID与state，可安全首次启动。
- 队列最大并发保持5，但本阶段只有1个依赖smoke和2个formal，实际最多同时运行两个正式缓存；formal必须等待smoke成功，不能因GPU空闲而跳过门禁。
- 修正后的分析器负例证明相关性门与效应量门相互独立：即使Spearman为1.0，只要Q4/Q1风险放大不足，整体仍判失败。

## 2026-07-23：步骤14跨掩码预测分歧通过双seed预注册风险门

- seed2024的分歧—完整目标MSE Spearman为`0.20878`，Q4/Q1 MSE与MAE比分别为`1.28921/1.18002`；三项均通过。
- seed2025对应值为Spearman `0.20743`、MSE比`1.84559`、MAE比`1.21206`；三项均通过。两个seed方向和相关强度高度一致，不是单seed偶然现象。
- 原型on/off分歧与误差的Spearman在两个seed为`-0.09127/+0.08387`，方向不稳定；因此证据支持“缺失扰动敏感性是风险信号”，不支持再次设计prototype utility router（原型效用路由器）。
- 嵌套遮挡单路validation MSE还略低于原始输入：seed2024为`0.518832`对`0.521840`，seed2025为`0.528073`对`0.530319`。这提示原始与嵌套预测可能同时具有误差互补和风险信息，下一步应先用冻结缓存验证等权预测集成的精确MSE，再决定采用无训练双视图模块还是训练期一致性模块。
- 现有NPZ逐样本保存`complete_sse_on`、`complete_sse_nested`、共同`complete_count`和`mask_disagreement_mse`，足以用平方误差恒等式无近似重算等权集成MSE；但没有保存逐元素预测或集成SAE，因此不能从缓存推断集成MAE，必须明确标为MSE-only筛选，若信号成立再补一次validation推理。
- 平方误差恒等式精确重算显示等权双视图集成在seed2024/2025的validation MSE为`0.509476541/0.519410618`，相对原PG95改善`2.3693%/2.0569%`，相对两条单路中更优者仍改善`1.8032%/1.6403%`。
- 该结果是看过步骤14单路与分歧缓存后的post-hoc（事后）机制发现，不能冒充预注册主结果；但它在双seed同方向且幅度超过此前多数训练模块，足以支持一次仅补集成SSE/SAE的validation复评。复评前固定等权0.5，不搜索融合权重。
- 现有诊断循环中`prediction_on`和`prediction_nested`已同时驻留，直接增加`prediction_ensemble=0.5*(on+nested)`的SSE/SAE不会新增模型前向、显存模型副本或训练；只扩展缓存schema和审计roundtrip。
- 通用队列支持显式`--reset-jobs`，但为保持原步骤14成功state和run不可变，步骤14.5应使用新的任务ID/独立runtime，而不是把历史formal重排并混淆两代schema。
- 在补测读取集成MAE前固定双seed继续门：每个seed相对原PG95的完整validation MSE改善至少`1.5%`且MAE改善至少`0.5%`，权重恒为`0.5/0.5`；任一seed任一主指标不足即不进入test。

## 2026-07-23：固定等权双视图通过validation双指标门

- seed2024等权集成validation MSE/MAE为`0.509476549/0.364180358`，相对原PG95改善`2.3693%/0.8870%`。
- seed2025对应为`0.519410617/0.361909157`，改善`2.0569%/1.3129%`。两个seed都超过事先冻结的`MSE≥1.5%、MAE≥0.5%`门。
- 集成在两个seed上也同时优于嵌套单路：MSE改善`1.8032%/1.6403%`，MAE改善`1.5298%/1.1248%`；收益不是简单选中了更好的遮挡视图，而来自两路误差互补。
- 固定等权没有新参数、无需未来标签，也不改变PG95 checkpoint；代价是推理时额外一次嵌套视图前向。步骤15应把它作为默认关闭的模型推理模块，在固定test mask与相同checkpoint上对照，并明确报告约2倍前向计算代价。
- 原`Exp_Pretrain1.test()`使用固定`flag=test`、`shuffle=False`、`drop_last=True`，以`seq_true_y`完整目标计算每批MSE/MAE再等权平均；步骤15必须复用该loader、完整目标和批次预算，不能另写不同划分或指标。
- 步骤15在读取新模块test结果前固定第一轮门：seed2024/2025各自MSE和MAE都必须优于同一次运行的原始PG95路；两seed简单平均相对改善还需达到MSE`1.0%`、MAE`0.5%`。任一seed指标反向则先停止test扩展，后续修改只能回到validation选择。

## 2026-07-23：CNM-SE通过Weather双seed正式test门

- seed2024的PG95 MSE/MAE为`0.271194642/0.300850301`，CNM-SE为`0.264560075/0.297525038`，相对改善`2.4464%/1.1053%`。
- seed2025的PG95 MSE/MAE为`0.256450080/0.295111265`，CNM-SE为`0.253108127/0.294119883`，相对改善`1.3032%/0.3359%`。两个seed均双指标提升。
- 两seed相对改善简单均值为MSE`1.8748%`、MAE`0.7206%`，通过事先冻结的`1.0%/0.5%`总体门，自动决策`CNM_SE_TWO_SEED_TEST_GATE_PASS_EXPAND`。
- 嵌套单路并不是稳定强臂：seed2024 nested MSE略差于on，seed2025 nested MSE/MAE均明显差于on；等权集成仍双seed改善，支持“结构化缺失扰动产生可利用的预测误差多样性”而非简单换输入。
- CNM-SE不新增参数，实测峰值allocated显存约`1051.25 MiB`；完整test双前向耗时约`37.1s/60.8s`（后者受外部GPU负载影响），协议记录的前向倍数为2.0。后续必须报告延迟代价，并用seed2023及结构消融验证不是固定两个seed偶然。
- Weather PG95 seed2023已有完整正式checkpoint、resolved config与metrics，run为`mipp_pg95__weather__point__mr0.06__seed2023__20260717_234738`；无需重新训练PG95，只需新增同协议CNM-SE冻结test任务即可补齐三seed。
- 在读取seed2023 CNM-SE结果前，三seed正式继续门固定为：2023/2024/2025各自MSE与MAE都改善，三seed相对改善简单均值MSE至少`1.0%`、MAE至少`0.5%`。seed2023失败不能用已有两seed均值掩盖。
- 步骤16结构消融预注册原则：主CNM-SE的L=5与0.5/0.5权重保持冻结；若三seed门通过，只新增同一嵌套因果mask机制的块长L=1与L=20，并复用每个run已经保存的original-only、nested-only和equal-ensemble三路作为融合消融。不得构造会重新暴露原始缺失位置隐藏真值的“非嵌套”视图，因为那会产生输入泄漏而不是合法消融。
- 三seed门已通过后，结构消融的完成判据冻结为：L=1/L=20各自smoke通过、6个formal全部工件PASS；报告L=1/5/20三种块长在三seed的original/nested/ensemble指标与推理代价。消融不设置“必须优于L=5”的成功阈值，也不依据test结果切换主模型。
- 步骤18在读取新validation结果前冻结：每个样本从原历史观测行中移除完全相同数量的时间点，目标计数为`round(0.15 * observed_rows)`；L=1/L=5/L=20只改变连续块组织方式。若块覆盖产生重复，继续从未移除观测行中按确定性顺序补足，保证三种块长逐样本新增缺失计数严格相等。
- 多尺度CNM-SE预注册为三路预测`0.5*original + 0.25*L5 + 0.25*L20`，不搜索权重；比较对象为original-only、L5双视图0.5/0.5、L20双视图0.5/0.5及三路多尺度。validation主门要求seed2024/2025多尺度MSE与MAE均优于冻结L5双视图，才允许未来在新数据协议上正式验证，不能回到已查看Weather test重选。
- 现有`stage14_uncertainty_diagnostic`已经提供chronological validation loader、冻结PG95 checkpoint、逐样本完整目标误差和packed历史mask；步骤18可复用其数据顺序与模型加载方式。
- 旧`_stage14_nested_observation_mask`按Bernoulli起点和块覆盖概率近似15%，因此L=1/5/20的实际移除数天然不同，不能满足步骤18的逐样本等计数要求；必须新增独立生成器，不能改变旧函数及历史结果。
- 旧步骤15仅记录整个评估过程总耗时，且不同任务曾并发运行；步骤18需要在同一个统一前向循环内分别同步计时original/L1/L5/L20，并由单GPU、`max_jobs=1`队列串行执行两个seed。
- 最小安全实现路径是新增独立`stage18_equal_rate_multiscale_validation`模式：一次生成三份等计数mask，输出original、三种单路、三个0.5双路及固定`0.5/0.25/0.25`多尺度结果；旧stage14/15/16入口与主模型行为保持不变。
- 现有`run_stage14_uncertainty_cache.py`已承担source checkpoint哈希、resolved config、代码diff、NPZ roundtrip与artifact检查，可扩展一个`stage18_validation`变体，避免复制一套容易漂移的runner；新变体必须走独立run前缀、样本文件名和schema校验。
- 现有通用队列支持UUID候选和后台恢复。步骤18只定义`smoke→seed2024 formal→seed2025 formal`的显式依赖链，并在start脚本强制单个GPU UUID与`max_jobs=1`，从调度层保证同卡串行。
- 步骤18双seedvalidation正式结果支持固定多尺度方案：相对等实际缺失率L5双路，seed2024的MSE/MAE改善`0.9169%/0.4526%`，seed2025改善`1.0778%/0.5952%`，预注册四项门全部通过。
- 多尺度相对original-only在seed2024/2025的MSE分别改善`3.3631%/3.0377%`，MAE改善`1.3979%/1.8166%`；收益跨两个validation seed一致，并同时优于L20双路，不是由块长实际缺失率不等造成。
- 代价明确：同RTX 4080串行批延迟由original的`25.3005 ms`、L5双路的`51.3268 ms`增至多尺度`77.1518 ms`，约`3.0494x` original、比L5双路高`50.31%`；峰值allocated显存仍约`1052.60 MiB`且不新增参数。
- 当前证据允许把多尺度CNM-SE作为下一阶段候选，但只允许在新数据集/新缺失协议上做外部验证；不能重新访问已看过的Weather test选择权重或替换历史主结果。
- 步骤19恢复审计首先确认：ETTh1已有完整数据、固定point-r0.06近似mask、步骤5单seedS4M checkpoint和正式测试工件，因此是最快可运行的新数据集；但其MSE相对论文高约103.7%，且原型分支在Gate H中数值失活，必须把“多尺度推理是否泛化”和“PG95原型是否有效”分开验证，不能混为一个结论。
- ETTh1可执行资产已逐项核验：正式run `s4m_paper_adam_reprofix__etth1__point__mr0.06__seed2023__20260717_195944`为`SUCCESS`，checkpoint大小23,207,100 bytes、SHA-256=`d4f3dad6...`且历史CPU重载通过；固定测试mask SHA-256=`a26534a1...`，数据CSV SHA-256=`efb63600...`。
- 该ETTh1协议名义起点率为0.06，但落盘实际缺失率为0.230884；论文官方mask seed未公开，因此只能标记为`approximate_missing_mask_seed_unavailable`。步骤19必须披露这一偏差，不能把新MSE/MAE与论文表值当作严格同协议比较。
- 步骤18评估入口和统一runner目前硬编码`validation + weather + pg95`。步骤19应新增独立外部测试身份，而不是放宽历史步骤18门禁，以保证Weather历史配置和工件回归不变。
- 步骤19正式外部筛选的工程协议全部通过，但ETTh1原始S4M的original/L1/L5/L20/各融合路线MSE与MAE只在约`1e-9`绝对量级波动；多尺度相对original的MSE/MAE变化约`-2.34e-8%/-1.12e-8%`，没有任何有意义的误差响应。
- 这与该ETTh1 checkpoint此前Gate H显示原型干预近乎失活、且基线误差远高于论文的证据一致：当前来源模型可能已形成近乎输入/检索不敏感的退化预测行为。等计数mask本身已由逐样本子集、移除数和尺度组织检查证明真实生效，因此不能把负结果归因于“掩码没加上”。
- `EXTERNAL_TRANSFER_FAIL`只否定CNM-SE在原始S4M ETTh1 checkpoint上的即插即用通用性；它没有检验PG95训练后表示对掩码扰动的敏感性。若继续，必须作为新的、明确post-hoc（事后提出）假设：先训练ETTh1 PG95单seed，再一次性应用完全冻结CNM-SE；不得把该追加实验说成步骤19预注册成功。

## 2026-07-23：步骤20完整PG95+多尺度CNM-SE外部验证负结果

- 四任务依赖队列全部`SUCCESS`，PG95训练smoke、正式训练、冻结多尺度smoke和211-batch完整test均首次attempt成功；checkpoint交接、固定资产、完整预算、MIPP键、等计数mask、指标回算和串行计时检查全部PASS。
- ETTh1 PG95 MSE/MAE=`1.211535967/0.840500396`，相对步骤5 S4M的`1.207990943/0.838923262`分别退化`0.2935%/0.1880%`。因此Weather上的PG95方向性收益没有迁移到ETTh1。
- PG95训练机制真实执行：5944个optimizer step、5952次配对视图、配对内容余弦`0.9999449`、稳定写入接受率`0.9998791`；绝对0.95门仍几乎没有选择性。模型增加138766个参数并把训练峰值显存从约3063 MiB提高到5011 MiB，但没有精度收益。
- 同一PG95 checkpoint的原始路MSE/MAE=`1.211535969805/0.840500398283`，固定多尺度为`1.211535969832/0.840500397799`；MAE仅改善`5.76e-8%`，MSE退化`2.22e-9%`，均为数值噪声量级。多尺度相对L5双路虽两项符号为正，但也只有约`4.4e-9%/4.8e-9%`。
- 正式决策=`FULL_METHOD_FAIL`。这说明原始S4M和重新训练PG95在当前ETTh1复现链路上都对因果嵌套扰动近乎不敏感；问题不是步骤19缺少PG95，而是该数据/复现checkpoint没有形成可被CNM-SE利用的视图多样性。
- 多尺度推理代价明显：同4090串行批延迟original=`62.31 ms`、multiscale=`255.07 ms`，约`4.093x`；在没有精度收益时不值得继续以ETTh1 test调权重或补同一配置多seed。
- 当前不能把PG95+CNM-SE表述为跨数据集完整方法。下一假设必须回到未查看test的validation，优先诊断“预测对输入扰动的响应幅度/视图分歧”是否在其他可信数据协议成立；不能依据本次ETTh1 test继续调PG95阈值、融合权重或块长。

## 2026-07-24：步骤21 Electricity初始资产审计

- 项目内已经登记`data/electricity/`目录，但尚无Electricity固定缺失资产、正式配置或历史checkpoint；因此步骤21是独立的新数据协议，不复用任何已查看的Electricity test结果。
- 可编辑`baselines/S4M`保留步骤4–20的复现/创新修改，官方`upstream/S4M`仍停留在commit `a718823addd3606e763dfc261174e0135b2535f4`且工作区洁净。
- 现有可复用工程资产包括固定point mask生成器、paper-Adam基线runner、步骤18冻结validation等计数多尺度评估和步骤20训练到评估的依赖交接；但它们当前没有Electricity专属协议，必须在启动前增加严格数据身份和321变量门禁。
- `data/electricity/electricity.csv`是只读符号链接，指向`/mnt/data/zza/code/timeperceiver_meta_uq/data/electricity/electricity.csv`；现有清单记录其形状为26304行、date外321个变量且完整无NaN。
- `generate_paper_point_masks.py`目前只支持Weather与ETTh1，并将“名义起点率0.06、块长5、逼近论文报告实际缺失率”的协议固化到CSV/NPZ/JSON。Electricity接入必须先从本地论文核对其Table 8实际缺失率，再扩展choice/spec，不能猜测或直接沿用其他数据集的率。
- 直接读取Electricity完整CSV确认形状`(26304, 322)`、date外321个`float64`变量、NaN总数0，时间范围`2016-07-01 02:00:00`至`2019-07-02 01:00:00`；数据规模与论文统计一致。
- ICLR 2025正式论文Table 8给出Electricity在time-point missing、名义`r=0.06`下的实际缺失率`0.260`；Table 1给出回看长度192时S4M参考MSE/MAE=`0.274/0.367`。
- 论文附录D.3固定Electricity学习率`0.01`、隐藏维度`512`、batch size `16`、Adam与early stopping；层数只说明从`{2,4,8}`中按validation选择，没有公开最终Electricity层数。步骤21不能声称严格同架构复现，必须在结果中披露该不确定性。
- 现有训练入口在`exp.train()`开头无条件构造一次未使用的test loader，并在训练后无条件调用`exp.test()`；这会在步骤21 validation门之前访问test协议。必须新增默认保持旧行为的`post_train_evaluation=validation`显式模式，并让该模式只构造chronological validation loader、写出完整目标MSE/MAE和最佳checkpoint工件。
- 步骤21在看到Electricity任何结果前冻结为8层：它沿用项目所有已复现实验的公开Weather 8层架构，同时采用论文明确的Electricity hidden512/lr0.01。由于论文未公开最终层数，该选择标记为`approximate_architecture_selection_undisclosed`，不宣称严格表格复现，也不根据后续test反向换层。
- 数据清单中的Electricity完整源SHA-256为`7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad`；后续生成资产必须保留该源哈希并向manifest新增独立paper-calibrated条目。
- `run.py`已经对小写`electricity`文件名显式设置`d_var=321`，因此模型变量维入口无需新的兼容性回退；仍需验证`Dataset_Custom4`确实优先使用runner传入的`true_data_path`而不是公开代码中的绝对路径。
- `Dataset_Custom4`已确认优先读取runner传入的`true_data_path`，按0.7/0.1/0.2顺序切分，并在缺失训练段拟合scaler；验证集默认loader会shuffle，但步骤18冻结评估已经显式重建`shuffle=False, drop_last=True`的chronological loader，可安全复用于Electricity。
- 步骤18等计数多尺度核心本身与变量数无关，能够直接处理321变量；历史CLI只接受`stage18/stage19`身份。步骤21应增加独立前缀/dispatch，而不是放宽历史Weather/ETTh1配置门禁。
- 现有冻结评估runner已集中处理source resolved config/checkpoint/metrics哈希、checkpoint键验证、等计数mask/NPZ回算和延迟工件；新增`stage21_electricity_validation`/`stage21_electricity_test`变体比复制runner更安全，但必须把dataset、method、split、seed、PG95冻结超参数及source validation身份分别锁死。
- runner当前仅在历史external-test变体上要求original路与source `metrics.json`匹配；步骤21 source metrics是validation。步骤21 validation应继续做original-source数值闭环，若门通过后的test则必须显式标记该匹配为“不适用（source是validation）”，不能错误比较test与validation或删除source工件。
- Electricity seed2023固定point mask已生成：名义`r=0.06`、L=5、实际缺失`6837/26304=0.2599224453`，逼近论文0.260；完整源/缺失CSV/mask SHA-256分别为`7e45845d...497ad`、`c4fc8e9a...3934`、`bacce090...8406`。
## 步骤21实现发现（2026-07-24）

- `run.py` 与 `experiments/exp_pretrain1.py` 已具备独立 `stage21_equal_rate_multiscale_evaluation`，可显式选择 `val` 或 `test`。
- 当前剩余风险集中在外层审计运行器：它仍把 Stage21 当作 Stage18/19 的旧分支处理，尚未写入 `stage21_*` 参数和来源 validation-only 身份核验。
- 因此正式 smoke 前必须先完成运行器分流；否则即使核心模型选择了验证集，外层产物命名、审计字段和门控语义仍不完整。
- Stage20 已有可靠的依赖到 checkpoint 动态物化机制，可复用于 Stage21；训练与多尺度评估无需硬编码 run 路径。
- Stage21 与 Stage20 的实质差异是：S4M 与 PG95 两条训练必须并行且都只写 validation 指标；PG95 多尺度也只跑 validation；随后由独立科学门控决定是否生成唯一一次 test 作业。
- 正式 Electricity 模型规模为 321 变量、512 隐层、8层，显存需求尚无实测；smoke 应先作为显存 profiling，调度器之后按实际峰值更新估计，不能静默缩模型或 batch。
- Electricity 的固定划分为 train/validation/test = 18412/2632/5260；`custom4` 为验证集附加 192 点历史窗，故可评估窗口数为 `2632-96+1=2537`，`drop_last=True,batch=16` 的正式验证应为 158 batch / 2528 samples。
- `post_train_validation` 使用完整真值目标累计 SSE/SAE 后统一求 MSE/MAE，并明确写入 `test_split_accessed=false`；多尺度验证可用同一 checkpoint 与同一 loader 逐元素回核。
- Stage21 GPU队列尚未启动。平台在命令执行前以Codex宿主提权额度耗尽拒绝了后台命令；本地审计确认无PID、无state、无半启动run，因此不能把步骤21报告为“已运行”或“等待显存中”。
## 2026-08-05 步骤21终态核验

- Stage21队列终态为`SUCCESS`，7/7任务均首次attempt成功；S4M/PG95正式训练和PG95多尺度validation均完成，条件test作业按门控写出`SKIPPED_BY_VALIDATION_GATE/test_accessed=false`。
- 当前门控报告的`PROTOCOL_FAIL_TEST_FORBIDDEN`含一个报告器误报：`same_budget_except_pg95_module=false`只因为比较了每个run必然不同的`checkpoints`、`metrics_path`、`results_root`、`test_results_root`路径；这些不是训练预算差异。其余协议检查全部为true。
- 独立于该误报，科学门明确失败：PG95相对S4M validation的MSE改善0.63494%，但MAE退化0.19936%；多尺度相对PG95原始路与original+L5的MAE仅有约1e-8%数值噪声改善，MSE均有约1e-8%数值噪声退化。因此修正报告器后仍应是`FULL_VALIDATION_FAIL_TEST_FORBIDDEN`，不会授权test。
- 报告器已修正并回归：run身份路径允许不同，`batch_size`真实预算漂移仍被负测拦截；`unexpected_method_differences=[]`、协议审计PASS，最终决策正式修正为`FULL_VALIDATION_FAIL_TEST_FORBIDDEN`。
- 历史条件test run保留当时的旧标签以维持工件不可变性；其关键执行事实`SKIPPED_BY_VALIDATION_GATE/test_authorized=false/test_accessed=false`与修正后的规范决策完全一致，canonical（规范）结论以`reports/stage21_electricity_validation_gate.{json,md}`为准。

## 2026-08-05 步骤22证据恢复（进行中）

- 已确认不能重试的路线包括：静态/时间块FUSR、短跨度HCUP、BMSA/SDRA/MRBA写入选择和CFR反事实训练；这些分支均已有验证集或多seed负证据。
- 当前唯一跨Weather三seed成立的机制是CNM-SE产生的合法预测误差互补，但ETTh1与Electricity的L5/L20预测响应均接近数值噪声；步骤22必须先解释并打破视图塌缩，不能继续搜索固定融合权重或块长。
- Electricity步骤21提供一个未访问test的新协议：PG95相对S4M验证集只改善MSE而损害MAE，适合在同一验证资产上诊断“少量大误差下降、广泛绝对误差上升”的权衡；诊断结果只能用于冻结新训练模块，不能授权test。
- 步骤21多尺度逐样本缓存确认视图塌缩不是汇总误差掩盖：2528个validation样本中，不同L路线相对original只有69–123个样本的累计SSE出现非零浮点差，单样本最大差仅`0.00390625/30816`量级；SAE也同样仅为浮点舍入级变化。继续融合这些预测没有信息增量。
- 首次源码检索误把可编辑仓库入口猜为`baselines/S4M/experiments`，该路径不存在；有效外层runner检索正常。后续先用`rg --files baselines/S4M`定位真实嵌套仓库路径，不重复猜目录。
- 步骤22完整validation诊断把问题进一步限定为“难度分层的MSE/MAE冲突”：PG95在全部96个预测步降低MSE，却在全部96个预测步提高MAE；低至中等误差分位主要退化，高误差分位改善更明显，但最难10%只贡献约47.08%的SSE收益，未通过事前冻结的纯尾部判据。
- L5嵌套视图对S4M/PG95的预测响应分别只有预测误差能量的`1.04e-15/1.26e-15`，因此继续做固定视图融合不可能产生实质增益。新模块必须在训练表示层显式注入掩码响应，而不是再调融合权重、块长或后处理门。
- 冻结的MRRA把`[衰减插补历史, 观测掩码]`送入rank32因果深度卷积适配器，将零初始化残差加到memory-fused上下文，并在合成缺失位置重建完整历史。它直接针对视图塌缩，且初始化时严格保持PG95预测路径不变。
- 冻结的RADR按每批样本detach MSE百分位秩，将困难端加权MSE与主体端加权MAE按各自风险尺度归一，再固定等权组合并回到MSE尺度。它直接针对PG95“降少量大平方误差、升广泛绝对误差”的冲突，不增加参数，也不改validation评价口径。
- 首轮不把MRRA与RADR组合：只有单模块相对冻结PG95验证锚点同时做到MSE/MAE不退化且至少一项严格改善，才有证据进入组合或多seed；任何单模块失败都先保留为负结果并重新分析，不能用另一个模块遮掩其独立贡献。
- 可编辑代码与官方上游均存在`min_idx = (max_idx - mean(min_idx)) / std(min_idx)`实现，疑似应以`min_idx`为分子。由于它是上游缺陷而不是新方法，步骤22不得静默修正；若后续验证，必须使用独立bugfix实验身份并重跑基线/候选。
- 步骤22首轮RADR验证表明其风险重加权确实把优化方向推向绝对误差：MAE相对PG95改善约0.89%，但MSE退化约2.25%。这不是双指标协调，而是把原有“PG95降MSE损MAE”的权衡反向翻转；按冻结门停止RADR单模块扩展，不能凭MAE单项改善进入多seed。
- MRRA短smoke没有暴露长期稳定性风险：正式训练epoch1有限，epoch2中段才开始NaN，之后污染原型memory输入。当前不能把失败解释为方法无效，也不能接受无限重试；必须先定位是辅助重建头、上下文残差还是共享高学习率导致的参数爆炸，并增加跨epoch稳定性smoke覆盖。
- MRRA-v1失败checkpoint审计给出参数级证据：329个模型张量中224个含非有限值，共17,853,822个非有限参数元素；8个MRRA张量的48,257个元素全部非有限，非MRRA主干另有17,805,565个元素被污染，而checkpoint内`repre`仍完全有限。故Bio.Cluster只是下一epoch首先显式报错的位置，根因是联合训练路径先发散。
- MRRA-v1短smoke checkpoint的329个模型张量全部有限，MRRA权重最大绝对值也处于正常范围；2批次/2 epoch smoke不足以覆盖正式训练约第1,830次优化后才出现的延迟发散。新的稳定性smoke必须至少越过旧故障窗口，不能再用2批次成功作为长期稳定证据。
- 原早停实现对`NaN`比较会走“validation下降”分支，从而覆盖epoch1有限最佳checkpoint；这既扩大损失也破坏可恢复性。修复后的训练循环和早停器在非有限训练/验证目标出现时立即抛出`NAN_OR_DIVERGENCE`，检查点改为临时文件后原子替换，外层runner也保留科学错误分类。
- MRRA-v2保留v1可重放分支，新增的仅是rank32隐藏特征LayerNorm（层归一化）及对上下文残差/重建修正的`tanh`有界影响函数；上下文投影继续零初始化，辅助MSE权重、Adam 0.01、batch16、正式15 epoch与固定validation资产不变。该改动的研究含义是限制掩码适配器对共享S4M主干的单步影响，而不是通过降低学习率或缩小模型掩盖不稳定。
- MRRA-v2 CPU极端输入下输出与梯度均有限，默认v1不新增归一化状态；非有限checkpoint负例被拒绝。长稳定性smoke冻结为3 epoch×最多800 batch，覆盖旧故障窗口，仍标记为smoke而非正式结果；只有它及默认关闭PG95回归smoke成功后才允许完整validation正式训练。
- MRRA-v2的归一化与有界输出不足以单独解决长程联合优化不稳定：3×800稳定性smoke仍触发`NAN_OR_DIVERGENCE`，因此不能把v2推进到formal或报告精度提升。新的有限值防护正确工作，方法失败与工件保存缺陷已被分离；下一轮需要依据准确失稳时点判断是否是共享Adam高学习率、梯度回传进入主干或衰减插补指数路径导致，而不是直接重跑。
- 具体失稳发生在v2 epoch2 batch16，约813次成功优化后；v1约在epoch2 batch700附近才显式失稳。v2保存的epoch1模型与原型表示全部有限，但低秩输入投影最大绝对值已从初始化量级增长到2.584，LayerNorm仿射参数也明显漂移。证据更支持“新适配器直接共用基线Adam 0.01造成优化尺度失配并反向污染共享主干”，而不是单纯的前向输出无界。
- v2 epoch1的`0.8787544`只来自smoke限制的3个validation batch，不能与冻结PG95完整validation `0.8728095907/0.7545076541`比较；formal从未启动，故当前唯一合法科学结论是稳定性门失败，而不是精度改善或退化。
- 源码把`reconstruction = z_s + correction(adapter(z_s, mask))`直接用于辅助MSE，因此v1/v2的重建梯度不仅更新MRRA，还会沿`z_s`恒等支路回传到`decay1/decay2`；这两层输出随后进入`exp(-decay)`，是比“输出未限幅”更直接的NaN放大链。MRRA-v3必须使适配器看到`z_s.detach()`，让辅助/残差分支只更新适配器，同时保留PG95原预测支路对衰减插补层的正常forecast梯度。
- 步骤22.13在新结果前冻结MRRA-v3：`v3_gradient_isolated_two_timescale`使用停止梯度的插补输入、无可训练仿射参数的LayerNorm、有界上下文/重建修正，以及`mrra_lr_multiplier=0.1`。主干仍用Adam 0.01和原type1调度，适配器初始0.001并同步乘相同epoch衰减因子；不降低batch、epoch或模型规模。
- MRRA-v3组件证据确认“隔离”没有把模块冻死：插补输入梯度严格为零，但适配器参数梯度有限且非零；主干与适配器参数集合无交集，type1调度前后学习率比例均为0.1。无仿射LayerNorm还把对应适配器参数量从v2的181降到173（组件维度），减少了v2中已观察到的仿射漂移源。
- MRRA-v3把数值稳定性问题与科学有效性问题成功分离：3×800稳定性smoke和8个正式epoch共均未出现非有限值，证明v1/v2的NaN主要可由梯度污染与优化尺度失配解释；但正式validation呈现MSE退化0.6460%、MAE改善0.1141%，说明“让表示对掩码响应”本身仍未协调平方误差与绝对误差目标。
- v3相比RADR把MSE代价从2.2501%压缩到0.6460%，同时MAE收益从0.8857%缩小到0.1141%；两条机制都沿同一Pareto权衡移动而没有支配PG95。下一轮不应直接组合两个未过单模块门的分支，也不应补多seed；应先在固定validation预测/误差缓存上诊断MRRA-v3新增响应究竟集中于哪些样本难度分位和预测步，再冻结一个能够限制主体样本扰动或保留PG95高误差收益的新机制。
- 既有Stage22缓存工件已提供配对所需的固定sample index、历史/嵌套mask、逐样本SSE/SAE/有符号误差、原始—嵌套预测分歧及horizon/variable聚合；因此无需修改模型前向或保存数百MB原始预测即可先回答误差迁移位置与掩码响应相关性。若该低成本诊断无法给出可观测分离信号，再考虑增加内部MRRA响应特征缓存，而不是提前扩大工件和代码面。
# Step 22.15：MRRA 内部可观测特征诊断（2026-08-13）

- Step 22.14 已确认 MRRA-v3 与 PG95 存在明显的逐样本互补性，但原始/嵌套预测响应无法分离受益样本，因此不能继续沿用预测响应路由。
- 代码检查确认不需要重新训练：`MaskResponsiveResidualAdapter` 已返回 `hidden`、`context_residual`、`reconstruction_correction`，`MTNet` 已支持默认关闭的原型检索诊断（top-1 相似度、选中相似度、检索熵及扩展统计）。
- Step 22.15 冻结为“只读验证集诊断”：从 MRRA-v3 冻结检查点导出每个验证样本的 MRRA 内部统计、缺失结构统计和原型检索统计；不得访问测试集，不得更新权重。
- 实现边界：诊断开关默认关闭；关闭时必须保持原模型行为；先 smoke test，再进行完整 validation 缓存；随后仅在时间隔离 fit/dev 上选择单特征方向与阈值，audit 只用于一次冻结复核。
- 停止规则：若预注册的内部特征在 audit 上仍不能同时使 MSE 与 MAE 非劣，则停止 MRRA 精度路由分支，不基于 audit/test 继续调阈值。

## Step 22.15终态与Step 22.16转向（2026-08-13）

- 正式内部缓存2528样本/158 batch首次成功，21个特征、固定历史mask、逐样本索引、PG95/MRRA-v3误差与既有缓存完全对齐；诊断代码没有改变MRRA-v3的SSE/SAE，所有协议检查PASS，test未访问。
- 21个特征虽有中等秩相关（例如重建修正/input RMS与MSE收益Spearman约0.557，最长缺失段约-0.528），但没有任何单特征方向/fit分位阈值能在fit和development同时实现MSE/MAE双非劣；候选规则数=0，因此audit未用于调参，决策=`MRRA_ACCURACY_BRANCH_STOP`。
- 原型检索诊断进一步暴露退化：top1/selected similarity、retrieval entropy、retrieved-query cosine/norm ratio在样本间完全常数或近常数；继续增加检索路由器或把21特征送入更大MLP会违背预注册停止规则，也容易把validation拟合包装成创新。
- 下一候选不再路由MRRA-v3，而是从结构位置与优化冲突同时修正：MCPR把掩码条件残差放在S4预测主干之后，绕开前主干mask抑制；冻结PG95主干，仅训练新低秩残差，避免MRRA训练中主干漂移造成MSE损失。
- MCPR首轮包含一个必要对照：相同架构的MSE-only版本用于判断后主干显式掩码注入本身；正式候选使用MSE与MAE归一化共同下降方向（Pareto common descent，帕累托共同下降）更新新模块，以一阶梯度同时非冲突为目标。两个版本训练预算、数据、seed和评估代码必须一致。
- 训练入口审计确认MRRA-v3此前从随机初始化联合训练PG95主干与MRRA，并没有加载冻结PG95检查点；因此其MSE退化同时包含主干权重漂移，不能用门控修复掩盖。
- MCPR改用专属post-hoc（后处理）适配路径：加载PG95正式checkpoint及原型memory，冻结全部旧参数与memory，只训练新模块；模型保持eval以关闭旧主干dropout和memory写入，仅MCPR进入train。首轮是方向筛选，不把额外训练预算下的结果直接包装成最终公平结论；若通过双指标门，必须补PG95等额继续训练控制。
- 帕累托共同下降固定为对MSE与MAE在MCPR参数上的全局L2归一化梯度等权求和；任一损失梯度范数非有限或为零即失败。对于两个非相反梯度，该方向与两者均有非负一阶内积；记录梯度余弦和共同下降内积作为机制证据。

## Step 22.16终态（2026-08-13）

- 专属队列于17:16:01启动、17:44:37自然结束，两个smoke和两个完整validation共4/4任务均第一次成功，无OOM、无重试；冻结主干、冻结memory、完整validation及机制执行检查全部PASS，test未访问。
- 冻结PG95锚点MSE/MAE=`0.8728095907/0.7545076541`。MCPR-MSE为`0.8761089148/0.7529857714`：MSE退化0.378012%，MAE改善0.201705%；未过双指标门。
- MCPR-PCD为`0.9158393189/0.7449589941`：MSE退化4.930025%，MAE改善1.265548%；平均MSE/MAE梯度余弦仅`0.0351892`，未过双指标门。
- 两个版本均只训练27,329个新参数。MSE-only已显示后主干掩码残差本身不能保住MSE，PCD又放大MSE退化；证据不支持在同一validation上继续调权重、学习率或容量。
- 预注册决策=`MCPR_BRANCH_STOP`，不进入多seed、PG95等额续训控制或test。下一个合理操作是基于已有冻结预测/训练工件做误差分解，区分“大误差样本损害”与“一般样本MAE改善”，再决定新的机制级假设。

## Step 22.17资产审计与分析边界（2026-08-13）

- PG95、MCPR-MSE和MCPR-PCD正式checkpoint、resolved config、固定Electricity point-r0.06/L5缺失资产和完整validation指标都存在；两个MCPR checkpoint已有全模型有限性与重载验证。
- MCPR运行目前只保存总体MSE/MAE、残差能量和PCD梯度统计，没有逐样本/预测步/变量缓存；不能从总体JSON反推尾部误差或配对分解。
- 现有`scripts/run_stage12_validation_cache.py`已实现固定验证数据、sample index、逐样本SSE/SAE及预测步/变量聚合的闭环评估，应扩展严格MCPR方法身份并复用，不新写第二套数据逻辑。
- Step 22.17属于发现性归因：允许使用标签分解误差，但任何oracle缩放、分组或门控只能作为机制证据，不能在同一Electricity validation上转化为新模型的确证性提升。
- S4M的`_forecast_from_context`会先计算主干`out_proj(x)`，再加`last_mcpr_outputs['residual']`；评估时可用最终预测减该残差无损重建PG95主干预测。残差与预测均为`[batch,time,variables]`，因此可直接做逐样本、预测步和变量的精确加性分解。
- 当前`stage22_validation_diagnostic` d在原始视图后才运行嵌套视图；只要在第一次前向后立即保存MCPR残差，就不会被第二次前向的`last_mcpr_outputs`覆盖。无需修改MCPR模块或正常训练路径。

## Step 22.17首次formal工件审计故障（2026-08-13）

- 两个formal都完成2528样本/158 batch前向并return code 0；总指标、PG95主干闭环、冻结性、MCPR checkpoint身份与逐样本/预测步/变量核心分解均PASS。唯一失败项是候选SSE和主干SSE分别float32归约后相减，与先做逐元素差再float32归约不在旧`atol=0.002`内。
- MSE/PCD实测最大绝对差分别为`0.0056763/0.0063477`，超限样本仅`3/2`个；对应总SSE差的相对误差约`1.3e-8/2.6e-8`，核心`delta_sse = alignment_cross + residual_energy`最大误差仅`0.0002594/0.0004883`。这是平行归约顺序导致的浮点消减，不是模型或公式故障。
- 保留冗余审计而非删除：`rtol=2e-5`不变，`atol`从`0.002`调整为`0.008`，等价每个目标元素约`2.6e-7`的最大绝对余量；同时将实际最大差和容差显式写入metrics。0.02量级的实质性差异仍会被拒绝。

## Step 22.17终态与机制结论（2026-08-13）

- 修复后队列终态=`SUCCESS`，两个smoke和两个formal全部用新experiment ID第二次成功；两个formal均完成2528样本/158 batch，27项配对协议检查全部PASS，test未访问。总决策=`MIXED_MCPR_FAILURE_MECHANISMS`。
- MCPR-MSE精确分解：`alignment_cross=-0.0025741053`是有益方向，但`residual_energy=+0.0058734267`更大，导致`delta_mse=+0.0032993216`；定义为`AMPLITUDE_OVERSHOOT`（幅度过冲）。MAE改善0.201705%，MSE退化0.378012%。
- MCPR-PCD精确分解：`alignment_cross=+0.0182893574`已方向有害，再加`residual_energy=+0.0247403761`，导致`delta_mse=+0.0430297336`；定义为`DIRECTION_ALIGNMENT_FAILURE`（方向对齐失败）。其残差能量是MSE版的4.212倍，PCD路线应停止而不是继续调权重。
- MSE版有53.006%样本MSE/MAE同时改善，但最差5%样本贡献了总净MSE退化的80.121%，最差10%的净贡献超过100%（其他样本部分抵消）。所96个预测步均是MAE改善但MSE退化，证据支持“幅度/尾部风险”而非某个预测步故障。
- PCD版只有14.953%样本双改善，47.231%仅MAE改善；移动块自助法的`delta_mse` 95%区间为`[0.017946,0.063200]`，完全大于0，方向失败不是局部时段波动。
- MSE版使用validation标签的不可部署全局最优缩放约`0.219131`，对应MSE上界`0.8725276`略优于PG95 `0.8728096`；这只证明有益方向被过大幅度淹没，不允许直接固定0.219或在本validation上宣称新改进。
- 下一机制应基于MSE-only残差，在训练期通过可还原的信赖域/缩放约束控制残差能量，并对尾部样本引入风险项；不应复活PCD，也不应在同一Electricity seed2023 validation上调缩放系数。

## Step 22.18 TTRR-MCPR机制冻结（2026-08-13）

- 步骤22.17的核心可利用信号是：MCPR-MSE残差总体方向有益，但能量惩罚超过对齐收益；53.006%样本双指标改善，最差5%贡献80.121%净MSE伤害。因此新方法应限制逐样本幅度并直接治理尾部新增伤害，而不是重新学习PCD方向。
- 冻结方法命名为`TTRR-MCPR（Tail-aware Trust-Region Residual，尾部感知信赖域残差）`。其推理模块用四个mask均值、隐藏状态RMS、原始残差RMS、主干预测RMS和主干预测时间差RMS共八个无标签特征学习逐样本Sigmoid门；特征停止梯度，门权重/偏置零初始化，初始门值0.5。
- 冻结训练约束是普通候选MSE加最坏10%`ReLU(sample_mse_candidate - sample_mse_backbone)`均值，权重1.0。该损失只惩罚候选相对冻结主干新增的正伤害，不混合MAE，与此前按绝对难度重加权的RADR有明确区别。
- 快速确认资产冻结为Electricity point-r0.06-L5 seed2024；控制组和候选组从同一PG95 seed2023 checkpoint warm start，用相同seed2024训练/验证资产和同预算。它只提供新缺失掩码，不提供独立时间区间，因此定位为交叉掩码快速确认。
- 科学继续门冻结为：TTRR相对seed2024 MCPR-MSE控制和seed2024冻结PG95锚点都须MSE/MAE双非劣且至少一项严格改善。若只改善控制组而未胜PG95，只能证明过冲治理有效，不进入test或多seed。
- 完整预注册文档为`docs/STAGE22_18_TTRR_PROTOCOL_ZH.md`；协议明确禁止使用validation oracle缩放0.219、禁止复活PCD、禁止结果后搜索尾部比例/权重/门初始化，并要求先smoke后formal。
- Electricity seed2024资产生成与28项审计全部PASS：实际缺失6840/26304行、率0.2600364964，连续段最短5；观测值逐值保持。它与seed2023全段有10165行不同，train/validation分别有7022/1034行不同，因此满足新缺失输入资产门。
- TTRR独立模块位于`baselines/S4M/s4m/model/TTRR.py`，只新增8维线性门的9个参数；关闭时不实例化该模块，MCPR state dict与历史默认逐张量一致。训练器新增`tail_trust_mse`并持久化门统计、原始/应用残差能量、尾部CVaR和正伤害比例。
- TTRR专属31项CPU/静态检查全部PASS；历史MCPR 14项和步骤22既有创新配置60项回归仍PASS。通用调度器9项自测PASS，实测最大并发5、smoke依赖、动态补位、启动预留、OOM回队列与确定性错误停止均正常。
- 步骤22.18队列冻结为6任务：PG95只读锚点、MCPR-MSE控制和TTRR候选各自先smoke后formal；PG95使用validation cache runner，不训练；两候选使用同一seed2024资产、同一PG95 checkpoint和15 epoch完整预算。

## Step 22.18复杂度收尾（2026-08-14）

- 原步骤22.18报告的参数量、总耗时和allocated峰值显存为`null`，根因是汇总器分别读取了不存在的`total_parameters`、`elapsed_seconds`和`peak_gpu_allocated_mb`；真实工件键为`parameter_count`、`resource_usage.json/elapsed_seconds`和`peak_gpu_memory_mb`。修正后MCPR-MSE/TTRR总参数分别为24,145,472/24,145,481，TTRR只增加9个参数；历史formal运行耗时因epoch数、GPU和并发负载不同，只能作为运行证据，不能作为模块速度比较。
- 为获得公平速度证据，新增独立validation-only计时入口：固定seed2024资产、batch size 16、`shuffle=False`，用CUDA Event仅包围模型前向；排除DataLoader、主机到GPU传输、误差归约和checkpoint加载。每个formal完整跑158 batch两遍，每遍前10 batch预热但仍参与精度回算。
- 三方法各自先做3 batch smoke，再按PG95→MCPR-MSE→TTRR-MCPR的严格依赖链在单一GPU UUID上串行formal；队列最大并发固定为1，不能传多个UUID。速度测量只补复杂度证据，不改变`TTRR_CONFIGURATION_STOP`，不重训且不访问test。
- 同卡计时队列最终6/6任务均首次SUCCESS，无OOM、无重试；全部固定RTX 4080 UUID。三项formal均完成2528样本/158 batch的两遍验证，每项有效计时296 batch，完整指标回算误差均在1e-5内，7项计时协议检查全部PASS。
- PG95/MCPR-MSE/TTRR的前向均值分别为100.0245/110.4301/108.0758 ms/batch，P95为142.8773/146.2023/145.0236 ms/batch；相对PG95的均值开销约10.40%/8.05%。TTRR相对MCPR-MSE显示-2.13%仅是测量波动，不能解释为9参数门使模型加速：TTRR两遍均值105.2191/110.9325，PG95两遍93.0080/107.0410，服务器外部负载导致可见波动。
- PG95/MCPR-MSE/TTRR参数量分别24,118,143/24,145,472/24,145,481，allocated峰值显存分别2408.46/2409.88/2410.12 MiB，reserved均3678 MiB；TTRR相对MCPR仅增加9参数和约0.24 MiB实测allocated峰值，部署代价可忽略，但其精度仍未通过双锚点门。

## Step 22.19 ARD-MCPR机制依据（2026-08-14）

- seed2023逐样本解析最优残差系数`clip(-alignment/(2*energy),0,1)`呈明显三动作结构：41.495%为0、47.508%为1、仅10.997%位于中间；总体均值0.5347不能代表单个样本。逐样本解析MSE上界为0.864385，相对冻结PG95理论改善0.9653%。
- 时间隔离结果暴露更强的非平稳性：fit前半段聚合最优常数为0且raw residual退化2.138%，development/audit完整残差反而分别改善1.492%/1.244%。因此TTRR式静态统一收缩并不匹配残差效用的离散和时变结构。
- 固定线性混合动作探针在development/audit相对PG95仍获得0.961%/0.473% MSE改善，说明训练期动作蒸馏具有可行上界；但其audit动作准确率仅0.223，且加入嵌套预测分歧后audit收益降至0.443%，没有支持把第二次反事实前向加入首轮正式模型。
- ARD-MCPR首轮因此冻结为两个紧密机制：训练期解析教师，以及三动作概率+部分系数的离散—连续混合门。它保持单视图推理，不复用TTRR尾部损失，不使用validation oracle缩放，也不增加在线test标签适应。
- 正式协议已在生成seed2025资产前写入`docs/STAGE22_19_ARD_MCPR_PROTOCOL_ZH.md`；固定16宽MLP、三动作阈值1e-6、epoch1教师warmup、能量门1e-8、未加权CE+SmoothL1、同预算三臂比较和双锚点停止门。
- 宿主启动前物理4080空闲16072 MiB、利用率0%，物理4090空闲10793 MiB但利用率89%。调度器按UUID和显存预留动态放置任务，没有缩减任何配置。
- 三项GPU smoke均首次SUCCESS且verification error为空：PG95只读缓存run=`stage22_crossmask_valcache_pg95__electricity__validation__seed2024__smoke__20260813_202719`；MCPR-MSE run=`mcpr_mse_crossmask_electricity__electricity__point__mr0.06__seed2024__smoke__20260813_202719`；TTRR run=`ttrr_mcpr_pg95_electricity__electricity__point__mr0.06__seed2024__smoke__20260813_202735`。
- TTRR smoke checkpoint包含原7个MCPR参数键和2个trust-gate参数键，全部模型张量有限且可重载；控制组只含原7键，证明默认关闭路径未意外创建TTRR参数。
- 三项smoke成功后，队列已自动启动PG95、MCPR-MSE和TTRR三项formal；调度器PID/SID=`761395`、PPID=1，队列最大并发5且test未访问。

## Step 22.19.3实现边界（2026-08-14）

- ARD最终系数严格定义为`p(FULL) + p(PARTIAL) × partial_alpha`，天然位于`[0,1]`；动作头与部分系数头均零初始化，因此初始动作概率各为1/3、部分系数0.5、最终系数精确0.5。
- 解析教师只在训练批次的观测目标上计算，残差能量不超过`1e-8`的样本不参与蒸馏；主干误差、教师系数和8维输入特征全部停止梯度，避免把标签泄漏到推理特征。
- raw MCPR仍由完整残差预测的MSE训练，ARD只由三类交叉熵与最终系数SmoothL1训练。推理只用单次前向和历史mask/隐藏/主干预测/原始残差统计，不需要目标标签或嵌套反事实视图。
- 默认关闭时不实例化ARD参数，TTRR与ARD互斥；历史MCPR/TTRR/Step22配置回归已证明新增入口没有破坏旧分支。下一门禁是专项合成样本的OFF/FULL/PARTIAL解析值、warmup、梯度隔离、checkpoint往返和非法配置测试。

## Step 22.19.3–22.19.4启动前终态（2026-08-14）

- ARD门自身新增212参数，完整候选相对MCPR控制只增加这212个参数；实际总参数、速度和显存仍必须由formal持久工件报告，不能从组件测试代替。
- 合成解析测试精确覆盖OFF/FULL/PARTIAL与低能量剔除；warmup只产生raw残差梯度，ARD路径的梯度为不存在或严格0，从第2 epoch才启动动作与尺度蒸馏。
- seed2025固定资产实际缺失率0.2599984793，并与seed2023/2024在训练/validation缺失行均不相同；它是在方法公式、权重、warmup、继续门冻结后生成的，因此可用于新的交叉掩码快速确认。
- 快速确认仍不是独立时间区间或独立数据集验证：即使通过双锚点门，也只能进入多seed/更强外部验证，不能直接声称达到SCI二区水平或访问test。
- 启动门要求PG95只读锚点、MCPR-MSE控制和ARD候选全部使用同一seed2025资产；MCPR/ARD两formal预算完全一致。ARD必须同时支配控制和PG95（MSE/MAE双非劣且至少一项严格）才继续。
- 17:59 CST动态队列已真实启动。虽然最大并发为5，但启动瞬间只有4080的9541 MiB满足6144 MiB估计，故只启动PG95 smoke；4090空闲4446 MiB未被强行使用，证明调度没有通过阉割配置凑并发。后续由实时显存与锁自动补位。

## Step 22.19终态与ARD-v1停止结论（2026-08-16核查）

- ARD-v1不是完全无效：它相对同架构MCPR-MSE控制同时改善MSE 0.7855%与MAE 0.1068%，证明训练期解析教师确实能抑制幅度过冲；应用/原始残差能量比约0.2842。
- 但ARD-v1没有支配冻结PG95：MSE仍差0.1876%，只有MAE好0.0566%。按结果前冻结的双锚点门，科学结论必须是停止当前配置，而不能选择性强调对控制组的收益。
- 失败机制高度集中在表示瓶颈：教师alpha标准差0.4788且OFF/FULL合计88.36%，学生门平均batch内标准差却只有0.000813，均值0.5305几乎复现了教师总体均值0.5366。当前8个样本级全局RMS/均值特征把时序—变量残差结构压缩得过早，门主要学成全局常数缩放。
- 动作交叉熵0.967低于均匀随机的`ln(3)=1.0986`，说明动作头获得少量信息，但动作概率与partial系数经期望混合后没有形成足够的样本级alpha分离。单纯增加epoch或在seed2025上调损失权重缺乏依据，也会违反新seed确认边界。
- 下一版若继续，应围绕一个明确问题重构：保留raw residual、backbone forecast与mask在时间/变量维的结构化token，再做集合/时序聚合以预测离散动作；同时避免把训练标签教师的信息压成单一连续期望。冻结后必须换新seed资产验证，不用本次validation反复调参。

## Step 22.20结果前冻结的诊断设计（2026-08-16）

> 终态补充：本节下方是结果前冻结的设计；完整分析已于同日完成并决定停止ARD方法线。

- 108维结构特征在development/audit均未超过global8，audit的alpha秩相关从`0.1961`降为`-0.1791`；这是时间外推失败，不支持结构化大网络。
- FULL与PARTIAL贡献虽呈强负相关，但两者标准差均小于0.005；gate接近常数的原因是分量本身已近常数，不是两个强变化信号抵消。改成离散执行也没有充分依据。
- `ARD_ROUTE_STOP_NO_CAPACITY_EXPANSION`是方法线终止，不是另一轮调参起点。现有证据中只有PG95+固定CNM-SE在Weather三seed上均双指标改善，且等实际缺失率validation复核成立。
- Traffic完整CSV已存在但未参与方法选择，适合作为新外部数据。必须在生成缺失资产或读取模型结果前，冻结数据协议、训练预算、多尺度权重和条件test门。

- 现有ARD forward已暴露raw/applied residual、8维特征、action logits/probabilities、partial alpha、MCPR latent和mask features，因此不需要改模型结构或重新训练即可完成信息瓶颈诊断。
- `structured108`把当前8维对照与预测时序分块、历史mask分块、低秩latent分块和变量分布分位数组合；所有输入均为推理时可观测量。目标/误差只用于离线解析教师和评价，不进入特征。
- CPU合约已证明新特征前8维与ARD-v1逐值相等，后100维会对时序反转产生变化，输出有限且无梯度。这只证明结构信息被保留，不预先代表它能预测教师动作。
- 诊断同时缓存完整目标教师和训练式观测目标教师，可以区分“可观测特征不够”与“训练教师本身受缺失目标噪声影响”；该比较此前ARD-v1总体指标无法回答。
- 解析公式CPU负例精确覆盖OFF/FULL/PARTIAL/低能量无效四种情形，观测mask会严格排除未观测位置；后续两种教师差异因此不是公式口径不一致。
- 特征探针超参数与时间切分均在结果前冻结，audit只使用一次。任何支持ARD-v2的结论都只能决定结构类型，正式精度仍需seed2026新资产确认。

## Step 22.21 Traffic启动前证据（2026-08-16）

- Traffic固定资产不是模型结果：完整CSV哈希、seed2026生成规则、目标实际缺失率0.26、L5和全部validation/test边界均先于GPU运行冻结。4559/17544行实际缺失，连续段5–18，完整/缺失CSV除mask位置外在`1.11e-16`内一致。
- S4M与PG95只允许MIPP开关及其冻结参数不同；两者共享862变量、真实最后目标列`OT`、8层/512宽、batch16、Adam 0.01、最多15 epoch、patience3、固定资产、validation loader与checkpoint选择规则。配置差异审计没有发现预算漂移。
- 多尺度评估新增的预测差异缓存是为拒绝ETTh1/Electricity式视图数值塌缩提供直接证据；它对L1/L5/L20分别保存nested与original预测的逐样本MSE，不进入模型输入或融合计算，因此不会改变CNM-SE精度。
- test门的决策优先级已经用合成反例固定：协议失败→PG95失败→视图塌缩→多尺度失败→全部通过；只有最后一种状态会构造一次test loader。其他状态均写`SKIPPED_BY_VALIDATION_GATE`且`test_accessed=false`。
- 当前所有零GPU证据通过，但这仍不能说明Traffic上有提升。下一项高信息量证据是两项2-batch训练smoke；成功后才允许完整S4M/PG95 validation比较。
- 首轮smoke提供的唯一信息是schema实现错误：真实Traffic表头为`date,0,...,860,OT`。S4M和PG95都在DataLoader构造阶段失败，未产生模型预测，因此将target从不存在的`861`修为`OT`不会利用任何精度结果，也不改变预注册科学假设。
- 修订后的两项smoke均完成真实862维forward/backward、optimizer step、checkpoint保存/加载和validation指标落盘；因此数据/模型/显存链路已经成立。smoke指标因只训练4步，明确禁止用于PG95优劣判断。
- 两项同预算formal已分别在RTX 4080/4090并行运行；调度器从smoke的实测reserved峰值加1536 MiB安全余量刷新formal显存估计，没有修改模型或batch。下一次读取结果前不再人工干预训练或门控。

## Step 22.21 Traffic 数值发散诊断（2026-08-16）

- 两项formal均非资源失败：S4M与PG95分别在epoch1的batch27/batch33出现NaN，随后由确定性错误门停止；队列没有漏启任务，也没有OOM回队列任务。
- Traffic完整数据、缺失输入的观测部分、归一化数组、填充值和mask全部有限；全部11993个训练窗口都有大量观测预测目标。因此NaN不是输入脏值、空masked MSE或零方差归一化造成。
- 两种方法在相近步数共同发散把根因收敛到共享训练设置。当前Adam lr0.01来自Electricity外推，并非公开Traffic超参；上游默认lr0.001为最小且有来源的单因素稳定性修订。
- 科学上应把该变化定义为“两臂共享的数值稳定性协议修订”，而不是PG95专属调参。必须保留旧失败run，并在改后先用覆盖失败步数的64-batch扩展smoke验证，不能再依赖只跑2 batch的旧smoke。
- lr0.001的两项64-batch扩展smoke均完成63次实际参数更新且无NaN，随后两项formal分别越过batch100/batch50；这直接支持“0.01是数值不稳定源、0.001恢复训练稳定性”的诊断。smoke总体指标不能用于判断PG95是否优于S4M，正式validation门保持原样。

## Step 22.21 Traffic外部验证终态（2026-08-16）

- PG95在Traffic validation相对S4M同时改善MSE 1.61135%与MAE 0.59844%；固定多尺度CNM-SE进一步相对PG95改善1.10809%与1.24812%。最终相对S4M为2.70159%/1.83910%，三层证据方向一致。
- 唯一Traffic test保持了validation方向：多尺度相对PG95 original改善MSE 0.915499%与MAE 0.883522%，相对original+L5仍改善0.350510%/0.444991%。因此不是只在validation权重选择上成立；权重本身在Traffic结果前已由Weather路线冻结。
- L5/L20预测差异相对original预测MSE分别约4.21%/4.88%，说明Traffic上的嵌套缺失视图包含实质不同响应；这与ETTh1/Electricity的视图塌缩负结果形成清晰数据条件边界。
- 代价是多尺度串行前向约为PG95 original的3.03倍；当前收益不足以忽略效率问题，正式论文需要同时报告准确率—计算量权衡，并考虑并行视图或蒸馏作为后续独立工程方向。
- 当前证据已经从“Weather三seed内验证”推进到“新Traffic数据的validation+test确认”，研究故事明显增强；但Traffic仍只有一个mask seed，下一步必须补seed2027/2028统计复核，不能据单seed外部结果直接声称达到二区水平。

## Step 22.22结果前接口审计（2026-08-20）

- 现有Traffic资产生成器把seed2026和输出路径硬编码，底层`select_random_blocks`已经参数化；多seed应新增独立生成入口或把生成器安全参数化，并保持完整Traffic SHA256、L5、目标行缺失率0.26和原子写入不变。
- 当前`run_stage14_uncertainty_cache.py`把Traffic评估限定为PG95且只允许seed2026；seed2027/2028复核需要显式扩展允许seed，不能通过伪装seed或复用seed2026嵌套掩码绕过门禁。
- PG95多尺度测试工件同时包含original与多尺度指标，因此可直接形成PG95→CNM-SE的配对统计；现有Traffic评估入口拒绝S4M源。若正式三seed主张包含S4M test对照，应新增明确的S4M原始路只读评估身份，而不能把S4M伪装为PG95。
- Step22.22统计协议必须在新资产生成前冻结：所有seed都运行而不按validation选择性纳入；模型、lr0.001、训练预算、nested seed、L1/L5/L20和0.5/0.25/0.25权重不得再改。

### S4M 独立测试路径审计结论

- 现有 `stage19_external_test` 只允许 ETTh1，且会把 source 指标当作 test 指标核验；Traffic 的 source checkpoint 记录的是 validation 指标，不能直接复用该身份检查。
- `exp_pretrain1.py` 的等实际缺失率评估本身可在关闭 MIPP 时运行 S4M，因此无需改变模型数学定义；需要新增一个显式的 Step22.22 Traffic-S4M 只读测试变体，使用 validation-only source 身份核验，并令 checkpoint 核验不要求 MIPP 参数。
- S4M 测试可复用多尺度评估实现，但正式三模型表只读取其 original 指标；S4M 上额外计算出的多尺度视图不进入主张，避免扩展本步骤研究问题。
- seed2027/2028 的 validation 和 test 都预先承诺全部运行，不以 validation 好坏决定是否访问 test；seed2026 复用既有 PG95/CNM-SE test，只补 S4M original test。
- 最终统计固定为 seed2026/2027/2028 的逐 seed 结果、均值±样本标准差、配对相对变化与胜出数。n=3 不足以支持强显著性结论，不能用低功效检验包装结论。
- 每个 replication seed 同时改变训练随机种子与固定缺失资产种子，因此估计的是联合复现实验波动，而不是单独的初始化方差。

### Step 22.22资产证据

- seed2027/2028资产分别缺失4563/4561行，实际率0.2600889193/0.2599749202；均在目标0.26的最近可达邻域，且所有连续缺失段最短为5。
- seed2026/2027/2028在完整、validation和test三个范围内均两两不同，能够排除复用同一mask造成的伪多seed。
- 35项独立审计全部PASS，包括完整资产哈希、CSV/NPZ/JSON/manifest、逐值重生、862变量共享缺失行、观测值保持与split覆盖；资产审计未构造模型或读取test指标。

### Step 22.22配置与队列复用边界

- 步骤22.21 pipeline把PG95 source job和validation gate硬编码为seed2026，不能直接用于无条件多seed复核；Step22.22需要独立pipeline从每份模板的`source.job_id`动态物化checkpoint，且不得再次按validation结果门控test。
- 训练配置可从seed2026正式配置机械派生；新seed smoke按结果前协议缩为2个训练batch/3个validation batch，只作链路检查，正式仍为15 epoch、batch16、lr0.001、patience3和无限batch。
- 评估器需新增一个明确的Traffic S4M test变体；PG95 Traffic原有变体只需把允许seed从2026扩到2026/2027/2028。S4M变体必须关闭MIPP、允许同一冻结架构、使用validation-only source身份核验，并以S4M前缀生成run ID。
- 22任务依赖已结果前固定：seed2026仅S4M test smoke/formal；seed2027/2028各含两训练smoke/formal、PG95 validation smoke/formal、S4M test smoke/formal、PG95 test smoke/formal。
- Step22.21同一多尺度评估器正式test实测reserved峰值3932 MiB，因此Step22.22评估任务以6144 MiB作为可信启动估计；训练任务继续使用S4M 14336 MiB、PG95 20480 MiB，不通过缩小正式配置适配显存。
- 新增Traffic S4M只读test身份已通过真实GPU smoke：关闭MIPP的seed2026 checkpoint完成3个test batch、所有22项评估工件检查PASS，reserved峰值3936 MiB与Step22.21 PG95评估的3932 MiB历史估计一致，证明6144 MiB调度估计有约2.2 GiB余量。

### Step 22.22三seed终态证据

- 22/22任务均首次attempt SUCCESS，无OOM/重试/失败；三seed协议检查、1648个validation样本和3408个test样本均完整。
- CNM-SE相对同checkpoint PG95 original在三个Traffic test seed上均同时改善MSE/MAE，平均相对改善为0.859189%/0.777161%，seed间样本标准差仅0.074102/0.094110个百分点，说明“嵌套视图融合改善PG95”本身具有稳定性。
- PG95相对S4M不稳定：validation平均MSE/MAE反而为-0.221465%/-0.782992%，test平均虽为+0.532000%/+0.401170%，但只有2/3 seed胜出。seed2028上PG95 test相对S4M退化0.212720%/0.837537%。
- 完整PG95+CNM-SE相对S4M的test均值改善1.386230%/1.174803%，MSE为3/3 seed胜出、MAE为2/3；seed2028 MAE仍轻微退化0.088038%。因此只能确认CNM-SE增量，不能声称完整链在所有seed严格支配S4M。
- 初版成本汇总约4.05–4.15倍包含了只作审计、不进入最终融合的L1路，不能作为部署成本。正式CNM-SE只需original/L5/L20三路；修订后的正确串行部署比应约3倍。下一研究优先级仍需同时面对PG95跨seed不稳与多路推理开销，不能只补更多同协议seed回避机制问题。
- 成本口径复算后正式三路串行部署比为3.02355x/3.08197x/3.03841x；这一结果支持把下一步聚焦在固定两路压缩，而不是误以为需要消除四路部署。
- Step22.23是明确的post-hoc效率新假设：候选只能由三seedvalidation选择；已有test即使支持也只是启动全新seed2029确认的理由，不能冒充独立证据。
- Step22.23真实validation显示两路方案只能保留约15%–50%的三路收益：E1最均衡但MSE/MAE保留率仅49.81%/49.99%，E5为50.02%/35.92%，E20为46.95%/15.36%且MAE跨seed不稳。说明L5与L20的互补不是可由任一单尺度替代的冗余，删除视图会损失约一半以上收益。
- 因此效率优化应保持original/L5/L20的预测与固定融合不变，优先利用视图仅在输入mask上不同这一结构，把三个视图沿batch维拼接成一次模型调用；这属于数学等价的执行优化，不应包装成新的预测模块贡献。
- Step22.24 smoke显示batch拼接的聚合指标差极小（MSE约7.9e-9、MAE约1.3e-8），但不同batch形状令FP32 CUDA kernel产生最大3.76e-4的最终逐点差，超过预注册1e-4；因此不能称为严格等价。保持每路batch形状不变的CUDA stream（计算流）并发是更符合失败证据的下一候选。
- Step22.24完整93-batch结果证明batch拼接确实把串行流水线从253.11 ms降到198.75 ms（降低21.26%，95%区间20.08%–22.40%），但最终逐点最大差扩大到5.87e-4，正式决策仍为严格等价失败。这是“速度有效、严格可复现性不达标”的清晰工程负结果。
- Step22.25 smoke验证了机制判断：保持每路B=16并用三CUDA stream时，所有四路预测与串行路径逐位相同，且峰值显存仅约3.17 GiB；这说明Step22.24差异确由batch形状改变而非多尺度逻辑错误。多流是否有实际速度收益仍须完整93-batch正式计时决定。
- Step22.25完整计时表明三CUDA stream虽逐位等价，却只降低0.52%平均延迟且置信区间跨0；单路S4M前向的kernel并行度已基本占满设备，多流不能解决约3倍计算量。后续必须减少推理期模型调用数，最合理路线是把三视图教师的增益训练期蒸馏到单视图学生，而不是继续做launch层优化。
- 现有MCPR低秩头具备零初始化、冻结backbone、mask条件输入和完整checkpoint审计，适合充当单前向学生；但MCPR-MSE/PCD/TTRR/ARD在Electricity上的负结果说明不能把“换训练目标”偷换成“新架构”。BF-CMVD真正的新变量必须是来自固定因果多视图教师的benefit-filtered监督，并需与MCPR-MSE控制分离。
- BF-CMVD的合成合约已精确区分四类样本，包括“教师MSE更好但MAE更差”的冲突样本；后者不会进入蒸馏集合，证明实现不是只按单指标事后筛选。教师与主干张量均detach，只有学生收到梯度。
- 科学门不使用smoke精度作为候选选择依据，只检查机制是否真实执行；seed2027正式validation的预注册门仍是同时优于PG95的MSE/MAE并各保留至少50%的CNM-SE收益，失败不访问test。
- v1失败不是CUDA或优化失败，而是监督稀疏性失败：固定教师在训练观测目标的64个整样本上没有双指标胜者，导致创新项完全未激活；直接放行formal会退化成已经失败过的MCPR-MSE控制。
- 既有validation完整目标缓存显示教师的整样本双胜率实际为73.60%，说明“教师无收益”与“筛选单位不匹配”可区分。v2的时间单元门保留双指标非劣逻辑，同时允许同一样本中有益预测时刻监督学生、伤害时刻被拒绝，比取消筛选或使用完整训练标签更符合任务边界。
- v2 smoke在真实训练观测目标上得到19.10%的时间单元双胜率，直接验证了“局部收益被整样本平均掩盖”的机制解释；这足以授权完整validation，但2-batch smoke的精度不用于模型优劣判断。
- BF-CMVD-v2 formal证明训练时三视图、推理时单路的方向有部分可行性：仅增加45182个可训练参数，便在冻结PG95主干与memory时同时改善validation MSE 0.5199%和MAE 0.3587%。
- 但学生对CNM-SE教师的MSE/MAE收益保留率仅48.07%/44.16%，两项均低于冻结的50%门；这不是统计边界上的一胜一负，而是两个指标一致地略低于最低有效门。因此当前版本不值得直接扩seed或test。
- 机制层面，18.45%的时间单元持续进入蒸馏、总计638222个获益单元，排除了“蒸馏监督再次塌缩为零”的解释。下一轮若继续，应提升学生对局部教师修正的表达/路由能力，而不是增加训练seed、放宽门或继续调固定教师权重。
## Step 22.27启动边界（2026-08-21）

- BF-CMVD-v2已经排除“创新监督完全未激活”：formal训练有638222/3459738个时间单元进入蒸馏（18.4471%），但单前向学生只保留CNM-SE教师MSE/MAE收益的48.07%/44.16%。
- 因此Step22.27不做盲目加epoch、调loss权重、放宽50%门或扩seed；先在冻结validation上解释剩余教师—学生差距来自哪里。
- 诊断只允许读取Traffic seed2027 chronological validation、固定完整目标、固定缺失资产、PG95源checkpoint、BF-CMVD-v2 checkpoint和冻结的`0.5/0.25/0.25`教师；test loader必须保持未构造。
- 只允许从三个候选主因中选择一个：观测目标筛选与完整目标获益错配；学生残差与教师修正方向/幅度不匹配；学生在教师非获益区域产生额外伤害。若证据混合或不可由推理时可观测量处理，则停止v3而不是堆叠多个模块。
- 既有代码允许从一次学生前向取得MCPR residual，并以`student-residual`恢复同checkpoint内主干；关闭MCPR后可复用严格等移除L5/L20教师。该设计避免跨checkpoint数值差异成为失效归因混杂因素。
- 冻结判定不把“低于50%收益保留”本身循环定义为表达不足，而是要求正教师—学生差距的区域分解和修正几何同时支持；这能区分“学生没学到获益修正”和“学生在不该修正处伤害”两种可导致相同总指标的机制。

## Step 22.27终态证据（2026-08-21）

- 完整formal只读诊断处理Traffic seed2027 chronological validation的1648样本、103 batch、158208个预测时间单元和136375296个目标元素；23项工件检查全部PASS，三路指标与既有run在`1e-5`内闭环，test未访问。
- “监督覆盖不足”不是主因：完整获益单元的观测支持recall为0.678290，略低于0.70；但教师完整获益率在目标已观测/未观测单元分别为0.537911/0.536560，只差0.001351，远低于同时要求的0.05。因此低recall主要是固定缺失率的机械覆盖，而不是获益分布错配。
- “非获益区域副作用”也不是主因：学生在教师非获益区放置31.9450%的残差能量，但该区域只贡献2.5646%的MSE正教师—学生差距和5.6520%的MAE正差距；主要损失不发生在错误区域。
- “学生表达不足”得到三层一致证据：教师获益区贡献97.4354%/94.3480%的MSE/MAE正差距；该区学生—教师修正余弦仅0.200284，投影系数仅0.072097，归一化失配0.985388。学生不是单纯幅度略小，而是只投影到约7.2%的教师方向且几乎保留全部失配能量。
- BF-CMVD-v2学生双指标获益单元比例0.590375高于教师的0.537476，说明它并非只在少量位置工作；问题是现有单一低秩残差把L5/L20两种不同尺度修正混成一个弱平均方向，无法复现教师在真正获益区的结构化修正。
- 预注册自动决策为`STUDENT_EXPRESSION_UNDERFIT_FREEZE_DMRB`。下一版只允许DMRB：共享一次PG95主干前向，分别用两个可关闭低秩残差基蒸馏L5与L20修正后固定求和；本轮不得同时加入CBR可靠性门或可观测性重加权，以保持单因素归因。

## Step 22.28 结果前设计边界

- DMRB 的唯一主变量是“单一混合残差头→两个独立尺度残差基+分解监督”；不同时引入 CBR（可靠性门）、可观测性重加权、学习型融合器或新的 teacher 权重。
- 固定 CNM-SE 教师为 `0.5*original + 0.25*L5 + 0.25*L20`；因而两个分支的精确目标分别为 `0.25*(L5-original)` 与 `0.25*(L20-original)`，学生预测为 `original + r5 + r20`，不设可学习融合权重。
- 两分支应共享同一次 PG95/S4 隐表示和同一历史观测掩码，但低秩参数不共享；这直接检验 Step22.27 的尺度修正混合导致表达不足假设。
- seed2027 validation 已用于归因和方法设计，因此 Step22.28 formal 只是开发筛选；即使通过，也必须在 seed2028 validation 独立确认后才能讨论 test。
- 稳定性、公平性和存储合约是科学门的一部分：默认关闭必须恢复基线，两分支必须可分别关闭，checkpoint 必须能保存加载，test loader 不得构造。

## Step 22.28 实现与启动前证据

- DMRB复用MCPR的同一零初始低秩残差结构，但L5/L20分支不共享任何参数；两分支总计90,364个参数，恰为BF-CMVD-v2单头的2倍。
- 训练保留v2的严格双指标token获益集，但将蒸馏目标分解为`0.25*(L5-original)`和`0.25*(L20-original)`；两者之和在CPU合约中精确重建冻结教师修正。
- 引入增量checkpoint是存储形式的无损工程处理：来源路径+SHA-256+新adapter能逐张重建来源主干、prototype memory和双分支；不改变前向、损失、优化、评估或停止门。
- 新runner会额外保存关键源码快照与SHA-256 manifest，解决MCPR.py为Git未追踪文件、普通`git diff`无法单独还原其内容的问题。
- 启动前全部测试通过，4080有16072 MiB空闲显存；当前没有理由缩小batch、epoch或模型规模。

## Step 22.28 终态证据（2026-08-26核验）

- DMRB工程假设被完整执行：L5/L20两个低秩分支均有正蒸馏损失、正梯度与正残差能量，冻结主干和memory哈希不变，单主干前向、固定求和与增量checkpoint往返均通过；因此负结果不能归因于分支未接入或checkpoint失效。
- DMRB validation MSE/MAE为`0.521787612/0.317986334`，虽相对PG95改善`0.321773%/0.249472%`，却相对BF-CMVD-v2退化`0.199201%/0.109647%`。把单一混合头机械拆成两个独立尺度头没有解决Step22.27的表达缺口。
- 教师收益保留率从BF-CMVD-v2的约`48.07%/44.16%`进一步降至`29.7496%/30.7087%`，而参数量从45,182翻倍到90,364；当前证据反对“只需按尺度增加低秩容量”的假设。
- 两分支平均残差能量分别为`3.2846e-4`与`3.2257e-4`，对应教师目标能量为`1.4336e-3`与`1.7075e-3`。这提示分支修正仍明显欠投影，但仅凭能量不能区分监督掩码错配、相加抵消或表达不足，不能直接据此加rank或加权。
- 当前最值得优先验证的机制嫌疑是：DMRB分解了教师修正，却仍给两个分支共用“完整CNM-SE教师联合获益”掩码；联合教师获益不保证L5和L20单独贡献在同一token都获益。该判断目前只是结果驱动的可检验假设，不是结论。
- Step22.29必须用冻结validation分别评估单分支、组合分支、自目标/交叉目标几何及联合/分支获益掩码重叠。只有确认监督掩码错配后，才允许实现训练期分支专属双指标获益掩码；否则应选择证据支持的单一主因或停止低秩残差蒸馏路线。

## Step 22.29 启动边界（2026-08-26）

- 本步复用DMRB checkpoint内一次前向直接返回的L5/L20残差，不重新训练或通过多次开关推理制造跨调用数值差；`p0=p_dmrb-r5-r20`，随后在同一张量上构造两个单分支学生和完整学生。
- L5/L20教师贡献仍严格定义为`0.25*(p5-p0)`与`0.25*(p20-p0)`，固定教师为`p0+c5+c20`；不得在看到分支诊断后调整0.25权重。
- Step22.27的validation-only数据、固定嵌套mask和compact工件框架可复用，但必须新增DMRB checkpoint身份与双分支键验证，不能把BF-CMVD-v2单头诊断冒充DMRB分支结果。

## Step 22.29 终态证据（2026-08-26）

- formal只读诊断完整覆盖Traffic seed2027 chronological validation的103 batch、1648样本和136,375,296个目标元素；20/20工件检查全通过，固定mask、checkpoint身份、七路预测闭环和指标回算成立，test未访问。
- “联合获益掩码不适合两个分支”未获支持：`P(B5|Bjoint)=0.808580`、`P(B20|Bjoint)=0.778921`均高于0.75，两个分支在联合掩码外遗漏的获益都低于19%，没有达到冻结的监督错配门。
- “固定相加导致分支冲突”未获支持：DMRB完整组合比L5-only和L20-only在MSE/MAE上都更好，两个学生分支余弦为正、抵消比为0，组合交互也是小幅正收益。
- “低秩分支表达不足”得到双尺度一致证据：L5/L20学生残差对自身教师贡献的余弦仅`0.077425/0.077294`，投影系数仅`0.034052/0.026085`，归一化失配`1.125322/1.061720`；各自获益区贡献约`97.05%/97.27%`的MSE正教师—学生差距。
- 自动决策为`DUAL_BRANCH_EXPRESSION_UNDERFIT_STOP_LOW_RANK_ROUTE`。继续增加rank/depth、改成分支专属监督或加入组合router都与当前归因不匹配；低秩输出残差路线到此停止。
- 下一候选必须改变可训练函数族或修正位置。优先验证“部分主干摊销”：用独立冻结的CNM-SE教师训练单视图PG95学生的末端S4动力层和输出投影，使非线性时序表征本身吸收多视图教师知识；第一轮只改变这一因素，不同时叠加对比学习或新门控。

## Step 22.30 结果前实现证据（2026-08-26）

- S4M预测路径为记忆上下文→首层mask-aware S4→输入线性映射→其余7层S4→LayerNorm→out_proj。PBA-CMVD只开放最后两层S4、LayerNorm与out_proj，共4,127,582个参数；这比低秩头更有非线性时序表达力，但不改变部署模型结构或总参数。
- 教师必须是来源PG95模型的独立冻结副本，不能复用正在更新的学生生成p0/L5/L20，否则教师会随优化漂移并破坏“固定多视图知识摊销”的因果解释。实现已检查模型对象、参数存储、梯度、全状态和memory五层隔离。
- 为使首轮比较只改变函数族，PBA-CMVD复用BF-CMVD-v2完全相同的严格token双指标获益规则与`观测目标MSE+获益token L1`，而不是同时引入latent loss、对比学习或新权重。
- `pba_delta_v1`只保存末端两层/归一化/输出投影的变化状态和来源SHA-256；CPU正反测试已证明冻结参数从来源恢复、变化参数从增量恢复、memory还原，篡改来源会被拒绝。

## Step 22.30 终态发现（2026-08-29）

- 工程结论可靠：组件与流水线测试均19/19通过，正式运行返回码0；训练后只有预注册的S4第6/7层、LayerNorm和输出投影变化。
- PBA-CMVD相对PG95有效，但没有超过当前最强摊销基线BF-CMVD-v2：MSE更差`0.1497%`，MAE只好`0.0286%`，未达到冻结的双指标科学门。
- MSE/MAE方向分歧且PBA可训练容量显著大于BF-CMVD-v2；仅凭总指标无法判断是少量大误差被放大、非受益token漂移，还是教师修正仍未被充分摊销。
- 因而不能直接调整解冻层数、学习率或损失权重。Step22.31必须先做validation-only归因；任何新方法只能由结果前冻结规则触发。

## Step 22.31 结果前实现证据

- PBA与BF比较不需要重新训练或重新评估BF：Step22.27已持久化同一validation顺序下的逐token BF、PG95和教师损失及固定mask；本轮只需新生成PBA配对缓存，并以PG95/教师损失闭环验证两轮计算完全对齐。
- PBA不是输出残差模型，无法像BF一样用`student-residual`恢复PG95。因此本轮使用独立深拷贝模型加载PG95完整checkpoint，PBA和来源各自做只读前向并检查模型/memory哈希前后不变。
- 尾部定义固定为BF token MSE的Q90，而不是按PBA误差事后挑选；这可区分“PBA只在少量既有困难token放大平方误差”和“PBA在普通区域产生新灾难误差”。
- 若尾部门触发，下一候选只增加训练期相对冻结来源的观测目标CVaR正后悔约束；该设计不改变PBA解冻范围、教师、训练预算或部署单前向，因此仍可做单因素公平验证。
- Step22.31 formal的固定mask、样本顺序、总指标和模型冻结均闭环，跨run逐token差异集中在PG95 MSE：99% token小于`1.37e-5`，但极少数高损失点最大到`1.30e-4`，导致统一`atol=rtol=2e-5`门失败。
- Step22.27的PG95预测由`BF学生预测 - 低秩残差`恢复，本轮则由独立PG95模型直接前向；两条数学等价路径在FP32中运算顺序不同。当前5/158208个教师获益token边界翻转与这一数值路径一致，但仍需检查相对误差和margin后才能决定工程容差。

## Step 22.31 终态证据（2026-08-29）

- 完整formal只读诊断覆盖Traffic seed2027 validation的103 batch、1648样本、158208个预测token和136,375,296个目标元素；21项工件门全PASS，PBA/来源模型与memory哈希不变，test未访问。
- PBA相对BF的MSE/MAE平均差为`+0.000779741/-0.000090823`，但BF参考Q90尾部只承担27.4348%的正MSE超额，且尾部平均差为`-0.000666149`、主体区为`+0.000940398`。因此“MSE差是极少数灾难尾部造成”被证据否决，CVaR方向不匹配。
- 非受益区承载33.9872%的PBA修正能量，但只贡献8.2887%/11.0146%的教师—PBA正MSE/MAE差距，远低于60%门；增加非受益区路由或信赖域不会处理主要误差来源。
- 教师受益区贡献91.7113%/88.9854%的正差距；其中PBA修正对教师修正的cosine仅`0.130330`、projection仅`0.138764`，normalized mismatch为`1.856097`。末端主干虽有容量，却没有沿多视图教师的隐藏修正方向学习。
- 唯一下一方法为HFD-PBA：保持相同可训练层与输出蒸馏，在训练期对S4第6/7层的学生原始视图隐藏修正和冻结来源多视图隐藏修正进行层级对齐；部署仍只保留单视图PBA学生。该候选由结果前规则触发，不是事后任选模块。

## Step 22.32 结果前设计边界（2026-08-29）

- HFD-PBA不增加参数、解冻层或推理视图；唯一变量是在训练期对S4第6/7层的“学生相对冻结来源修正”和“固定CNM-SE教师相对冻结来源修正”做层级对齐。
- 层级误差使用获益token上的教师修正能量归一化MSE；新增损失固定为`stopgrad(L_distill) * mean(R_6,R_7)`，由既有输出蒸馏项匹配尺度，不引入可调权重。
- 获益集合、教师`0.5/0.25/0.25`、可训练参数、Adam、学习率、batch、epoch、patience和validation MSE选点全部继承PBA-CMVD，避免把更多预算冒充结构收益。
- seed2027开发门要求MSE/MAE同时优于BF-CMVD-v2和PBA-CMVD，且两项教师收益保留均至少60%；失败即停止HFD/PBA路线，成功也只能转入seed2028 validation独立确认，不能访问test。

## Step 22.32 启动前工程证据

- 隐藏捕获是普通Python临时状态，默认关闭、不注册参数/buffer；合成前向证明关闭/开启预测逐位一致，state dict键完全不变。
- HFD损失的学生第6/7层特征梯度均为正，冻结来源/L5/L20教师特征和输出蒸馏尺度均不接收梯度；无获益token时返回带学生计算图的严格零。
- 新配置固定同一PG95来源、相同4,127,582个可训练参数、训练预算与validation-only边界；配置负例能拒绝错误层、错误损失/尺度、关闭PBA、访问test和缩水formal。
- 原PBA组件与流水线共38项历史回归保持全过，说明新增默认关闭路径未破坏Step22.30的checkpoint、参数选择与调度门。
- 项目外层不是Git仓库，但总runner固定从`baselines/S4M`嵌套Git记录commit/status/binary diff，并额外快照S4M模型、训练入口、CLI、checkpoint工具、runner和本次配置及SHA-256；因此外层`git status`失败不影响正式run的代码可还原性。
- Step22.32首次排队时两卡都在高利用率，空闲显存低于PBA训练的14336 MiB历史估计；正确行为是`WAITING_RESOURCE`而不是减少batch/epoch或修改模型。队列已常驻并会在资源满足后原子锁定GPU启动smoke。

## Step 22.32 终态证据（2026-08-30复核）

- 队列随后自动在RTX4090上完成smoke与formal，均首次SUCCESS且无OOM/重试；说明隐藏捕获、梯度隔离、增量checkpoint和单视图validation工程链全部真实可运行。
- HFD-PBA相对PG95仅MSE改善`0.267219%`，MAE反而退化`0.129567%`；相对BF-CMVD-v2两项退化`0.254040%/0.490051%`，相对PBA两项退化`0.104153%/0.518794%`。
- HFD层6/7在2996个优化步持续激活，HFD平均损失`0.0261484`、归一化原始误差`0.968707`，因此负结果不是模块未执行。直接强制隐藏修正的欧氏幅度匹配没有改善输出Pareto目标，且明显伤害MAE。
- 冻结决策为`HFD_PBA_SEED2027_STOP`：停止HFD/PBA路线，不在同一validation调层、调损失或扩seed2028；下一研究方向必须改变假设，不能把重复运行包装成独立证据。

## Step 22.33 路线选择前证据（2026-08-30）

- HFD最佳checkpoint仍来自epoch1；其validation MSE=`0.540475`，差于PBA epoch1的`0.539752`。此后训练MSE继续下降但validation不改善，说明不是“多训几轮即可恢复”的问题。
- HFD平均新增损失=`0.026148`，几乎等于既有输出蒸馏损失`0.025964`，使平均总损失从PBA的`0.093070`升到`0.127171`；同时训练直接MSE从`0.066020`恶化到`0.075059`。这构成强烈的多目标梯度竞争迹象，但现有日志没有独立损失梯度余弦，暂不能把“梯度冲突”写成已证实结论。
- 层6/7隐藏修正平均cosine达到`0.413840/0.447480`，但归一化失配仍为`0.903900/1.033515`，且最终MAE恶化；欧氏隐藏接近并不等价于输出双指标改善。继续只调HFD权重不符合预注册停止规则。
- Traffic seed2026/2027/2028均已有完整1648样本validation缓存，包含original、L5双路、L20双路、三路multiscale的逐样本SSE/SAE、历史packed mask和视图预测分歧。无需新GPU即可先量化“按样本选择计算路径”的oracle上界和跨seed可学性；但缓存未保存完整预测张量，所以任意连续可学习融合权重仍需重新前向生成新工件。
- 双指标安全离散oracle相对固定multiscale仍有稳定上界：seed2026 MSE/MAE额外改善`0.5928%/0.4993%`，seed2027为`0.5663%/0.6157%`，seed2028为`0.8132%/0.8056%`。这比继续解释HFD负结果更接近“最快看出新改进是否有提升”的全局目标。
- 四动作oracle分布跨seed相近：original约14.32%–20.27%，E5约17.42%–19.96%，E20约18.08%–19.96%，full multiscale约42.35%–47.63%。这证明最优动作不是永远固定multiscale，但不证明推理时可观测特征足以预测；Step22.33必须先做可学性审计，不能把oracle冒充模型结果。
- 新路线分两层，避免一次堆模块：PSAVS（Pareto-Safe Adaptive View Selection，帕累托安全自适应视图选择）先争取超过固定融合的准确率；其独立确认后才考虑预算约束上下文赌博机减少平均前向次数。两者可形成“可靠性自适应融合+计算预算决策”两个创新点，但当前仅是待验证假设。

## Step 22.33 执行前审计（2026-09-07）

- 源窗口长度192、预测长度96、步长1，随机切分会共享原始时间点。三seed必须采用相同的时间边界，并在fit/dev/audit之间清除287个窗口，以隔离完整288步窗口支撑。
- 分歧感知策略需要original/L5/L20三个预测已经生成；无论最终返回哪个动作，实际前向调用数仍为3。仅掩码策略可在前向前选择，但离线逻辑调用数不等于测量延迟。
- 先前oracle是按双指标相对original非劣约束的固定标量评分选出的理想参照，并非两个指标各自严格的最优上界。不能把真值驱动的oracle安全性或提升当作可部署策略保证。
- 主策略固定为分歧感知，mask-only仅作机制对照；不得根据audit改选主策略。既有三seed和validation均参与过项目开发，本步只能做可行性筛选。

## Step 22.33 正式结果与决策（2026-09-07）

- 正式离线run=`psavs_offline__traffic__point__mr0.06__seed2233__20260907_141643`，工程SUCCESS、冻结科学门FAIL；12项单测和独立合成smoke先通过，不更新预测主干、不加载test。
- 14维mask-only（仅掩码）树未给出双指标正收益切换；17维disagreement-aware（分歧感知）树虽然给出切换，但dev裕量0平均MSE/MAE退化0.759039%/0.557944%，所有非全回退裕量均不合格。dev冻结两者为infinity，audit三个seed均100%回退固定多尺度。
- 主策略audit MSE/MAE平均改善0%/0%，0/3 seed双指标胜出；raw分歧版三seedMSE退化0.150251%/0.122115%/0.158336%，MAE退化0.213518%/0.202629%/0.167610%。因此不能只归因于校准保守；没有证据表明当前特征+固定浅树能将动作差异转化为可泛化收益。
- audit标签感知oracle仍有MSE约0.3020%–0.4271%、MAE约0.5741%–0.8751%的改善空间，但它读取真实误差，不能作为推理模块；存在可选择空间与能够可靠预测选择是两件事。
- 本轮边界只否决预注册特征+树组合，不证明所有自适应视图方法无效。0%回退不是新模型有效，3次前向也没有任何已证实节省；不宣称逐样本安全、独立测试泛化或SCI分区水平。
- 原全validation fixed_multiscale MSE约0.52，本轮audit约0.74是不同时间子集，不可交叉比较。报告中的全部策略均在同一306窗口/seed评分，配对移动块区间仅作描述。
- 最短后续建议为盘点稳定PG95+固定CNM-SE已有证据和贡献缺口，而不是扩seed2029、开启预算赌博机或在现有audit反复换选择器。此建议尚未启动，不改原主门或以其他策略替换主策略。

## Step 22.34 主线证据审计（2026-09-07）

## Step 22.35 诊断完成（2026-09-08）

- 正式CPU诊断run=`pg95_diagnosis__traffic__validation__seed2235__20260908_131733`，耗时79.97s、峰值RSS892.80MiB、GPU0；3项单测与合成smoke先通过。不训练、不读test、不修改S4M主干。
- 六训练均最佳epoch1、epoch4早停、2992优化步；同训练预算。PG95常规写入放行率99.6912%/99.2835%/99.5345%，暖启动约80.47%，不能混称强筛选。训练平均loss小不证明梯度小。
- 当前同seed构造三组均314个公共状态条目中124个不同，包含linear1和后续层，前置decay层相同；新增MIPP在后续层前消耗随机状态。历史初始化未保存，现证据是当前构造复现，不是历史初始权重直接测量；原方法级多seed比较不因此作废，但不能据单seed归因某个子机制。
- seed2028多尺度75.30%窗口MSE更差、79.79%窗口MAE更差，四时间段和三缺失档都退化，末412窗口全部双差；seed2027主要风险在第三时间段。分组是描述，不是独立检验或因果关系。
- S4M原型数200/200/6，PG95为5/8/5；中心化有效秩受原型数量限制。seed2028胜出的S4M自身仅6原型，小库并不单独证明有害坍缩。
- 历史训练code.diff确认辅助视图从seq_true_x生成，可能重新暴露主视图人工缺失位置；该设定依赖完整训练历史可用，不能无条件宣称适用于只有不完整训练输入。不是退化已证实根因，改变它需独立记录协议和重评。
- 下一最短路径为配对初始化和独立随机流下的原S4M/PG95机制对照，不直接改门阈值/加损失权重；仍是归因控制而非创新收益。

### 2026-09-08 失败恢复前诊断

- 2026-09-19 Step22.37：seed2029采用原Traffic生成函数，实际缺失率0.2599749202006384；与2028缺失总量相同但掩码SHA256不同，均通过独立重生及观测值一致性审计。16份配置只改变seed/相应资产和实验身份，不修改模型/损失。新增验证白名单严格限制Traffic串行CRN+配对训练，test不扩展。历史训练run仅约114–119MiB，当前36GiB磁盘暂可容纳本阶段，GPU空闲尚不足完整训练。
- 2026-10-04 Step22.38源码审计：S4M.forward对原seq_x缺失位重新填补，因此直接更改这些位置不构成有效填补干预。独立诊断改在mem_net入口替换已填补历史；原融合含检索、查询、历史投影及查询残差，query-only不是retrieval-only，需四路径区分。只用观测历史生成填补，匹配CRN，model.eval之外须mem_net.is_training=0；S4内核首次初始化需单独记录后冻结。GPU smoke的48窗口已验证所有负控制和缓存一致性，不用于精度结论。
- 2026-10-04 Step22.38完整1648窗13路径：原始MSE0.534761382，查询相对向量漂移约2%但预测差异MSE约1.8e-7，查询/纯检索整体误差变化极小；历史直达替换几乎解释完整替换退化（零/前缀均值/前填MSE分别+8.55%/+2.37%/+18.03%）。仅支持冻结模型对这三类填补变化主要沿历史直达响应，不支持查询污染为主要瓶颈，也不证明原填补劣质或检索不重要。全部最长缺失段6–20，跨尺度证据不足。下一步先审计融合分量尺度和衰减实现，暂停直接堆查询模块。
- 2026-10-04 Step22.39：距离审计19项PASS确认min分子误用已标准化max、极值可选缺失零、encoder reshape轴差异；均来自上游，无当前伤害证明。融合CPU12项PASS（测试曾误用nn.Linear布局，已单独修正并保留失败）。GPU smoke在prototype-off验收失败，formal未运行；c-R与重新零检索fusion可能有浮点差异，但未测到具体差值，不认定根因。补充定位日志且保留原1e-5门禁；审批usage-limit阻止宿主复核。Model.decay实际nn.Linear，未使用TemporalDecay，不能声称双指数可达范围界限。
- 2026-10-04 Step22.39后续更正与最终结论：新增日志证实FP32分块相减与原GEMM求值差是off门禁根因（实际off vs原fusion零r预测差0），独立V2输入分支缩放在原1e-5下通过，2/2 SUCCESS、1648窗15路径复算PASS。关检索收益仅MSE0.03994%/MAE0.11700%，关query指标冲突，关history退化MSE197.07%，关填补直达退化8.54%；历史投影RMS60.63 vs查询1.28/检索0.72。填补0.9微益在长末端缺失组MSE不稳定，不据局部收益加新模块。实际min误用std0.02518 vs1，exp无溢出、极值选缺失位置仅约0.22–0.26%。下一最短先独立distance bugfix baseline（非创新）单seed同预算重训；不改其他风险混入收益，不访问test。
- 2026-10-04 Step22.40授权：仅修改min距离分子变量（max_idx→min_idx），不加epsilon、不改gamma/极值/encoder重排；默认关闭。S4M无MIPP时warmup原本直接使用seq_x，必须保持该路由，不能顺手换成填补历史。原seed2027训练实际4epoch/2992步、历史reserved约11870MiB；修复版沿用15epoch上限/早停3，实际停止轮次可以不同。磁盘余约9.7GiB，无清理授权。
- 2026-10-04 Step22.40最终CPU门禁25项、旧回归7项、调度9项全部PASS；默认关闭FP32位一致、pre_edit与旧源bytehash一致、314初始状态/28371290参数一致。只原始CRN路径作主要比较；真实parser默认参数将在run内持久化并和历史对应train/eval及smoke/formal检查，不允许未声明科学参数漂移。启动器请求PID2810634，GPU真实性待核验。
- 后续实查：四GPU smoke均SUCCESS，formal第1轮日志已到iter200，GPU child2861550在4080占8994MiB；代码/保存加载链路确实可运行，但尚无完整验证集收益。本轮只是隔离已确证公式问题，不能据mini-batch训练loss声称提升或创新。
- 22.40最终结果已自动完成：6/6SUCCESS/无OOM，正式4epoch/2992步，固定1648窗原始CRN路由改善MSE2.63874%/MAE3.07081%，初始化与固定mask/逐窗指标复算PASS。耗时增加约9.23%，allocated峰值近似不变；reserved下降可能来自allocator/机器状态，不当作算法轻量化。单seed方向值得复核，22.41只补2028/2029，不引入第二因素。
- 22.40独立复算agent核实原/修复4轮sampler相同、6份parser与resolved全一致、fixed NPZ1648行和10类评估资产全配对；当前无已发现协议偏差。2028/2029研究重复性联合变化初始化seed与人工缺失资产seed，不是固定同mask纯训练随机性实验；多seed汇总必须明确此限制。
- 用户批准CRN-v1后采用独立original-only整遍参照，而不是用同一次多视图输出自证；额外视图各自重放按seed/batch派生的CPU/CUDA随机状态，退出恢复调用前状态。既不改变topM，也不尝试伪造旧训练末RNG。新S4M/PG95必须同协议重评，旧PG95缓存不能作为新四格的B/D。
- 4080两模型CRN-v1 GPU smoke各48窗口全部通过，独立原始遍历与多视图原始分支逐窗口MSE/MAE差异严格为0；关闭开关的旧S4M smoke也通过。仅证明局部重复性，完整1648窗口仍由正式任务验证；任何修复前后差异不能归为创新增益。

- 源S4M seed2027训练及失败复评均绑定同一4080 UUID；PyTorch2.1.0+cu121、CUDA runtime12.1、结构化核扩展SHA256也相同，不能把跨GPU作为本次差异的解释。
- Bank.py第420–422行在eval模式下仍执行随机topM抽样。实际S4M seed2026/2027记忆数为200、topM=100；seed2028记忆数6，不触发这一分支。多视图额外前向会改变后续original的CPU RNG轨迹。
- 原save_model仅保存model/memory/repre/num_dic；未发现历史RNG工件，不能仅用seed保证复现训练结束时的随机状态。需要先用CPU抽样控制流检查确认这一缺陷，再决定新的共同随机数配对协议；不得擅自改评估规则或放宽容差。

- Weather早期三seedCNM正结果是L5双视图；步骤18等计数多尺度只有Weather两seedvalidation正证据。Traffic三seed结果属于后一冻结多尺度，不能将这两代方法直接拼成一个最终方法的完整跨数据集统计。
- PG95源为MIPP配对掩码干预+不变性损失+0.95稳定写入门，关闭内容/掩码因子分解。步骤7 PG95写入接受率99.9142%，原型强净化机制不应过度表述；三个子机制贡献也不能由整体开关自动证明。
- 既有Traffic主表只有S4M、PG95、PG95+CNM三列；虽存在S4M多尺度test评估工件，但新的机制选择不应依赖未汇报test子结果。因此补已有S4M checkpoint的同协议validation多尺度，是比再添加创新模块更直接的归因实验。
- 代码审计发现旧Traffic S4M-test评估的metrics.method包装可能沿用pg95标签，但其resolved_config/source/use_mipp和run ID明确为S4M。本步骤以最终配置和来源身份识别方法，新S4M-validation分支正确标记s4m，不追改历史数值或旧metadata。
- 标称生成参数r0.06并非实际缺失6%；Traffic三seed固定资产实际约26%，新配置保留标称与实际字段。该实验是透明固定外部协议，公开Traffic超参数/掩码seed不足以支持“官方完全复现”措辞。
- 原S4M与PG95使用同一最大epoch/早停规则，但实现配对视图和实际早停轮数带来计算差异。四格对照用于开发归因，不等于等实际FLOPs的训练消融；若后续科学门有价值，仍需单独补机制和成本隔离。
- GPU启动真实性已核实：nohup+setsid宿主机后台可持久运行，4080 smoke工作进程有实际GPU context；22.33普通nohup失败不应被误诊为服务器驱动损坏。
