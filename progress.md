# 进度日志（内部工作记录，非正式项目文档）

## 2026-10-10 组会材料制作（不执行22.44）

- 后续用户要求五分钟、只重要结论并直接覆盖前版PDF：已压缩30→6页、836字口语稿，两张图重画并用STIX MathText真正数学上/下标；普通部件中文化、训练query loss与独立teacher写门明确分开。六页逐页目视与独立审查PASS，0缺字/越界/字面下划线、7源码hash不变。
- 发布入口`outputs/group_meeting_20261010/_short/publish_short_report.py`退出0，原子覆盖同名PDF（6页、172140字节），Markdown口语稿同步；完整CSV和旧研究结果未修改，研究仍22.43完成、22.44未执行。首版30页验收记为历史，不是当前交付页数。
- 用户授权中文口语化PDF、完整结果/结论/问题/投稿水平及模块位置架构图；执行范围只含汇报材料，不改模型/正式配置、不训练/评估、不访问测试逐样本缓存。
- 使用文件规划、方法型汇报与Python科研图技能，独立计划位于outputs/group_meeting_20261010/。原共享术语资源缺失，使用项目统一台账替代；PDF优先用户要求，不额外强制PPTX。研究状态仍22.43完成、22.44未实施。
- 已完成30页口语化PDF（15正文/15附录）、三页矢量架构PDF及SVG、302工件完整索引和61配对结果；主表来源自动制表，负结果与历史/最新协议分开，发表判断不保证二区。生成/编译/数值与7源码hash校验退出0。
- PDF专用依赖局部安装到汇报目录。真实发现并修正文Latin数字缺字和上标负号缺字；最终30页全渲染，0NUL/0越界/13关键文本PASS、三图均矢量；13关键页联合目视及全文独立内容审计PASS。不声称30页全部逐页目视。研究进度和旧科学判定未改变。

## Step22.43 开始（2026-10-06）：先核查修复后CNM增益，公式交叉暂列备选

- **最终完成**：8单测、prepare、真实缓存smoke（20261006_224028）、正式CPU（20261006_224121）退出0；正式0.906311秒/RSS38.457MiB/GPU0。输出stage22_43_cnm_audit.json/md/csv及唯一run全部来源/配置/日志/状态/资源，归档stage22_43_final__20261006_224121。
- 主比较修复后的同一S4M原始路→固定CNM三seed均双胜，均seed相对MSE/MAE改善0.9579816%/0.6834906%；独立60指标/汇总及141来源hash PASS，latest302项且诊断MSE/MAE等NA。模型/旧配置/缓存/旧判定均未改，没有GPU、训练或test访问。
- 旧原版来源额外只读补审：Namespace中17个创新/子开关False、冻结模型两处旧公式、三seed冻结模型/CLI哈希相同；无完整parser缺口仍保留。初两次子开关正则因省略真实use_前缀断言失败，改用AST实际keyword读取后退出0，与数值/实验无关。正式已冻结分析器不热改，补审另存stage22_43_final_audit.md。
- 结论：CNM局部一致小幅收益值得保留，3前向不是等推理预算、不能宣称两个贡献已成功或二区。公式交叉暂列备选；建议22.44独立可关闭索引bugfix及CPU/基线/单seed验证，未实施。

- 22.43.1冻结准备退出0，141条来源/旧模型代码/新分析代码哈希；同seed两公式原路/多尺度来源核验通过，旧误差工件保持原值。旧schema兼容完成后8项单测再运行全过。真实2028缓存smoke已启动CPU入口，不运行模型。

- 22.43.2初始8项合成测试全部通过、静态编译退出0。读取旧原公式训练配置时发现历史schema没有effective_parser_args，诊断命令KeyError退出1；新主比较的fixed完整parser存在。分析器明确兼容legacy官方参数，并核验模块statekeys，不回填/修改旧配置；该差异只影响历史参照元数据完整性，主数值来源不变。

- 用户最新授权“开始做22.43交叉诊断暂列备选”，按上一轮讨论执行已有修复基线与固定多尺度CNM比较；开工明确解释，不启动公式交叉、训练或test。
- 已读取全局与项目规范并使用planning-with-files-zh恢复规划；嵌套Git存在旧用户修改，全部保留。外层非Git，不操作已有模型。磁盘约1.1GiB，只写小型分析工件，不复制大型检查点。
- 技能恢复首个`python`命令因PATH权限退出127，改用项目Conda解释器后恢复脚本退出0；不安装/更改环境。两次跨文件规划补丁因不匹配标题整体未写入，已改为准确锚点补丁；与实验源码无关。
- 初步独立计算固定修复模型original→multiscale三seed均双指标改善；尚待完整来源/资产核查、单测/真实缓存smoke和正式报告，不将其称为新增训练或已修复所有上游错误。

## Step22.42 完成（2026-10-05）

- 正式CPU入口退出0，run后缀20261005_182906；metrics/status/resource/source均保存，报告stage22_42_diagnostic.json/md/csv及summary.md。source manifest包含三seed两版cache/检查点/训练与评估配置/log/hash，分析前后全部不变，旧62源hash也保持。累计168分组，latest300项，诊断MSE/MAE记NA不当新精度成绩。
- 四时间段加和闭环，2028第一段跨两个实际缺失档均退化；掩码四量相关弱、末端无缺失组MAE也退化，不能把长缺失或少数异常当单独根因。新簇索引问题4CPU情形确认，仅记录上游bug，不改变原源。完整报告独立复核待最后收尾。
- 最终独立复核退出0：576个原始/多尺度分组数值直接从NPZ复算匹配，四段严格加和回全局；134条源manifest/旧源码hash不变，总表诊断MSE/MAE/RMSE/参数/checkpoint为NA、GPU0且scope明确诊断。工程完成，因果未识别，原22.41全胜门未改。
- 银行数值审计初次采用字典逐float严格相等，1线程重算与2线程报告末位归约差（约1e-16–1e-13）触发审计assert；按rel_tol1e-10/abs_tol1e-12重核三对全通过、非float字段严格一致，六checkpoint hash/GPU未初始化检查PASS。不是正式任务失败，未改模型/科学指标容差或原报告。
- 阶段报告/总表/源manifest已非破坏性归档至reports/archive/stage22_42_final__20261005_183350/；mkdir/cp退出0。原模型Git diff --check退出0，已有dirty源码保留。磁盘仍约1.8GiB，无清理。
- 使用文件规划技能保存并同步阶段结论；新增5个CPU分析/测试/静态复现脚本和一份固定边界文档，原模型/训练/评估/原配置全部未改。新报告summary及自动json/md/csv索引完整。下一最短建议22.43冻结checkpoint完整状态×公式2×2，尚未执行；不同时修新增簇索引。
- prepare退出0；真实缓存smoke完成SUCCESS，run=s4m_distance_diagnostic__traffic__validation__seed2242__smoke__20261005_182752，指标与时间分解闭环、源hash不变；总表299项。完整CPU分析已启动，未排GPU队列。
- 新增audit_stage22_42_memory_index.py，以AST执行上游/本地原方法而非手工仿写，4种CPU索引算例通过退出0；原源hash前后不变。该独立实现风险不改变22.41修复协议、不作2028因果推断。
- 新增独立CPU主分析/7项合成测试、bank审计helper/12项合成测试；两组运行退出0。三个真实checkpoint配对helper smoke通过，CUDA未初始化，不改变任何source工件。原型代表与驻留均值不一致属于新静态风险，已明确分开保存，不顺手修复。
- 系统中断后检查新文件：尚无22.42代码落盘，已恢复两个agent，保留已生成计划而不重复训练。新增docs/STAGE22_42_DIAGNOSTIC_ZH.md固定组边界/解释限制。
- 独立协议审计仅读validation上下文CSV证实row-shared与cache；无test内容。发现drop_last丢13理论尾窗、cache仅有逐窗口完整目标总误差。已在文档明确CSV安全审计例外，记录与最初笼统边界的偏差；后续不临时补horizon/目标子集评价。

- 用户授权开始22.42；文件规划技能要求先恢复工作记录，已读三文件并执行session-catchup（退出0）。本步只对已完成验证工件和CPU检查点做独立离线诊断，保留原模型、评估协议和选模规则。
- 磁盘余量约1.8GiB，采用只读复用数据/检查点、输出小型统计和哈希，不复制大型工件、不删除成果。训练队列不启动。修改前已检查baseline Git，既有20tracked修改和untracked创新属于用户历史成果，全部保留。

## Step22.41.4 审计与收尾（2026-10-05）

- 用户追问2028差异：使用文件规划技能保留原因假设边界，只读分析已有NPZ/日志/模型代码，复用成本agent独立CPU检查六个checkpoint，不启训练/不访问test。时间四分组复算退出0：2028首段MSE/MAE退化5.7763%/4.1666%，第三段改善2.3779%/1.4764%，全段收益不一致；2027/2029各段均双指标改善。原型200/200、6/6、62/84，最佳epoch均1。配置/资产已核验通过，不能把差异简单归因错误设置或测量随机性。
- 补充解释核验完成：CPU六checkpoint/日志审计及源码定位退出0。2028小银行走全部6中心softmax、没有topM随机噪声；内部观测目标Vali Loss改善0.856%与完整主指标退化同时成立；修复四轮训练loss更高但未发散。事实与待验证因果分开；仅增量更新工作记录，未开启22.42阶段或更改选模规则。

- 用户明确授权本步；使用文件规划技能保存结果，主agent与两项只读并行审计分别核验指标/固定资产和训练预算/计算代价。仅读取已完成任务，不监督训练、不启动下一阶段。
- 初查队列8/8SUCCESS（各attempt1、OOM0），正式报告已完成；独立NPZ复算与六份验证工件检查退出0。2028双指标退化、三seed仅2/3全胜，保留负结果，不把工程完成等同科学门通过。
- 原有分析入口正在重新复算/刷新自动总表及按时间归档；成本核验与最终状态同步待收尾。仅更新报告/工作记录，不修改基线Git工作区。一次文档补丁锚点错误被工具拒绝，未改文件；已读取真实标题改用准确锚点。
- 收尾：`/mnt/data/zza/anaconda3/envs/s4m_missing_forecasting/bin/python -u scripts/stage22_41_distance_fix.py`退出0，输出`BUG_FIX_NOT_THREE_SEED_DUAL_GAIN_ANALYZE_NO_AUTO_EXPANSION`。最新总表298项、含本阶段全部8项；新归档`reports/archive/stage22_41_final__20261005_155609/`，原归档保留。
- 成本审计退出0：冻结8配置/资产和62源码哈希未变，三seed均4epoch/2992步/28371290参数。2029旧4090/新4080不是同卡测速；2027/2028运行负载不受控，训练中validation elapsed不是独立推理时间。2029旧/新动态记忆长度62/84，但没有峰值数量轨迹，不把它当显存差异根因。
- 新增`reports/stage22_41_final_audit.md`，更新task_plan/findings/progress及badandgood简明记录；不改模型、训练入口、冻结配置或旧实验工件。执行完成但科学门未通过：主要比较2/3双胜，2028双退化；下一建议22.42只读诊断，未自动启动。
- 最终代价摘要将2027时间增幅准确写为9.235888%（四舍五入9.24%），原22.40简报9.23%为近似误差；原始秒数及科学结论不变。额外只读字段/总表核验退出0。

## 会话：2026-07-14

### 阶段 1：官方项目获取与仓库审计
- **状态：** complete
- **开始时间：** 2026-07-14
- 执行的操作：
  - 读取用户提供的全局规范。
  - 完整读取 `/mnt/data/zza/code/AGENTS.md`。
  - 创建独立目录 `/mnt/data/zza/code/S4M_MissingForecasting`。
  - 建立内部规划、发现和进度记录；未生成正式项目文档。
  - 完整克隆官方仓库到 `upstream/S4M` 并记录 commit。
  - 核对仓库文件、README、官方环境、运行入口和示例数据。
  - 检查磁盘、CUDA Toolkit 和 GPU 可见状态。
  - 检索并核对 S4M、PSW-I、PRIME、CTRL、SpectraNet、MissTSM、CoIFNet、VIMTS 和 ChannelTokenFormer 的方法重叠。
  - 发现初版“双 loss 拼接”创新强度不足，开始重构研究问题。
  - 下载并逐节核对 S4M ICLR 2025 正式论文方法、实验和附录。
  - 审计原型库读写、数据路径、日志、指标保存和设备处理代码，定位多个复现阻塞及一个疑似实现错误。
  - 将研究主线收敛到 S4M 特有的原型库污染、错误检索与未知缺失域泛化。
  - 创建独立 Conda 环境 `s4m_missing_forecasting`，完整安装官方依赖，pip 退出码 0。
  - 运行 `pip check`、核心库导入、官方入口导入与修正 PYTHONPATH 后的模型导入检查。
- 创建/修改的文件：
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

## 测试结果
| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| 项目目录唯一性 | `/mnt/data/zza/code/S4M_MissingForecasting` | 不覆盖已有项目 | 新目录已创建 | 通过 |
| 官方代码完整获取 | GitHub 官方仓库 | 完整 Git 仓库且工作区干净 | commit `a718823...`，无本地修改 | 通过 |
| 论文资产完整度 | README、数据、脚本、环境 | 覆盖主要论文表格复现 | 仅 Weather 示例，环境含可疑固定版本 | 部分通过 |
| GPU 可见性 | `nvidia-smi` | 输出 index/UUID/显存 | 无法连接 NVIDIA 驱动 | 失败，当前不训练 |
| 官方依赖安装 | `environment.yaml` | 独立环境全部依赖可安装 | 续装后退出码 0，环境 5.4 GiB | 通过 |
| 依赖一致性 | `pip check` | 无 broken requirements | `No broken requirements found` | 通过 |
| README 入口 | `python run.py --help` | 正常显示帮助 | `No module named 'S4'` | 失败 |
| 模型核心导入 | 增加官方源码所需 PYTHONPATH | S4M 模块可导入 | 成功，使用慢速 S4 fallback | 通过但非原命令 |

## 错误日志
| 时间戳 | 错误 | 尝试次数 | 解决方案 |
|--------|------|---------|---------|
| 先前会话 | GitHub `gnutls_handshake` 失败 | 2 | 本轮采用替代获取与核验路径 |
| 2026-07-14 | `nvidia-smi` 无法与驱动通信 | 1 | 暂停 GPU 工作，稍后从宿主状态重新核验 |
| 2026-07-14 | 官方 `conda env create` 默认求解会话无进程、无环境、无退出状态 | 1 | 停止残留会话，下一次改用 libmamba 求解器 |
| 2026-07-14 | OpenReview 页面触发浏览器校验，API 请求网络失败 | 1 | 以会议正式论文页和 PDF 为主，后续再补公开审稿意见 |
| 2026-07-14 | README 入口缺少 S4 顶层包路径 | 1 | 通过显式 PYTHONPATH 完成诊断；正式包装待用户确认 |

## 五问重启检查
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 3：与用户讨论并确认创新方案 |
| 我要去哪里？ | 用户确认后生成正式文档、修复复现入口并开始 smoke test |
| 目标是什么？ | 形成可复现且创新逻辑充分的 S4M 改进项目 |
| 我学到了什么？ | 见 `findings.md` |
| 我做了什么？ | 见上方记录 |

### 阶段 3：方案确认与阶段 4 启动
- **状态：** complete
- 用户已明确授权由 Codex 选择并推进最有把握的完整创新方案，要求重新生成详细正式项目文档。
- 最终选择 `CMPE + FUSR` 作为两个核心模型创新，`EMG` 作为统一的跨缺失域训练与评估协议，不采用普通对比损失、频谱最优传输、强化学习或完整 MAML 作为核心主线。
- 按技能要求重新读取规划、引用和学术写作规范；发现 `nature-polishing` 的共享静态目录缺失，已记录并采用现有本地片段继续。
- 补充检索 TameR、PIR 和 STABLEMISS+，进一步收紧本项目新颖性表述。
- 阶段 4 已进入进行中，下一步生成并自查正式中文总文档。
- 完成第二轮相似工作精查，确认 StableMiss（IJCAI 2023）、StableMiss+（ICASSP 2024）、RAFT（ICML 2025）、TameR（ICML 2026）和 ChannelTokenFormer（ICLR 2026）的边界。
- 据此撤回“未知缺失机制泛化本身具有首创性”的潜在表述，将论文中心固定为 S4M 原型检索在缺失域漂移下的失效诊断与效用治理。
- 完成第三轮关键词精查，新增 STRATA-TS 和 DPAD；进一步将 FUSR 收窄为“缺失域下游预测效用监督 + 选择性读写 + 拒用”，排除泛化的双原型库和相似度路由叙事。
- 完成第四轮精确检索，新增 Cross-RAG、SARAF 和 RIDDE，确认一般的检索伤害、选择性融合和不变分解已非空白。
- 将创新一从泛化的 CMPE 收紧并重命名为 MIPP，强调它直接净化 S4M 原型库写入；保留内容/缺失双因子，但不宣称因果识别。
- 重新核对 S4M 正式论文附录，确认四数据集规模、0.7/0.1/0.2 划分、96 预测长度、四种回看长度、连续 5 点缺失构造、名义 r 与实际缺失率的区别，以及论文 Adam/代码 SGD 的协议冲突。

### 阶段 4：正式项目文档与初始化
- **状态：** complete
- 创建正式中文总文档 `docs/PROJECT_BLUEPRINT_ZH.md`，共 35 个连续编号章节。
- 最终模型定名为 MIRAGE-S4M，两个核心创新为 MIPP 和 FUSR；增加理论分析、Oracle 上界、固定 MSS 套件、完整实验矩阵、消融、统计和 Gate 停止条件。
- 创建 EndNote 导出 `references/related_work.enw`，包含 16 条参考记录；修正 PSW-I 和 VIMTS 初稿中的作者/题名元数据。
- 创建 `configs/`、`scripts/`、`logs/`、`outputs/`、`results/`、`reports/`、`runs/`、`artifacts/masks/`、`tests/` 和 `src/` 最小目录。
- 自查结果：`ENW_OK records=16`；`FENCES_OK count=14`；`HEADINGS_OK count=35`；官方上游 Git 工作区仍干净。
- 总文档 SHA-256 为 `766a35930548a8b0cec0594d7bc190a61cb3f694ac611fc17e34d777fa053ba3`；参考文献导出 SHA-256 为 `6ff9b7d15264c62339249b480060a5803d062d96a3a6e08d26898ffa33da07f1`。
- 项目外层当前不是 Git 仓库，因此没有项目级 diff；`upstream/S4M` 仍是独立干净 Git 仓库，commit 未变。
- 本阶段未启动训练，因而没有新增模型指标；已明确当前 `nvidia-smi`/CUDA 可见性阻塞只影响下一阶段 GPU 工作。
- 下一步：进入阶段 5，先实现不修改上游的 S4M 适配入口、复现修复和 baseline smoke test，再验证 Gate H。

### 阶段 5：官方基线第一步启动
- **状态：** in_progress
- 已重新读取项目 `AGENTS.md`、内部计划、发现和进度记录，并运行会话恢复检查；未发现未同步上下文。
- 已确认项目专属规范覆盖快速复现、固定评估资产、实验唯一 ID、动态 GPU 调度和 OOM 回队列要求。
- 已核对上游文件清单；`upstream/S4M` 内没有更近层级的 `AGENTS.md` 或 `AGENTS.override.md`。
- 当前子目标：在不修改上游快照的前提下，核验宿主资源与官方入口，建立本地适配入口并跑通 baseline smoke test。
- 已重新读取两份 README、完整训练入口及核心实验类，确认官方命令存在 `PYTHONPATH`、强制 W&B、绝对输出路径和论文 Adam/代码 SGD 协议冲突，不能直接把原命令输出作为论文复现结果。
- 环境复核退出码 0：Python/PyTorch/NumPy/Pandas 与官方环境一致，`pip check` 无损坏依赖。
- 默认隔离环境的 GPU 资源复核退出码9；随后在宿主权限下确认 CUDA 与两张 GPU 正常，因此这是执行隔离差异，不是服务器驱动损坏。
- 已确认官方模型存在 `.cuda()` 硬编码，数据加载器存在完整 CSV 绝对路径硬编码；下一步先复用服务器已有数据，再建立不污染上游的基线适配副本。
- 已在服务器现有项目中找到与论文规模完全一致且无缺失的 Weather、Electricity、ETTh1、Traffic 四个完整 CSV；暂不联网下载或重复复制。
- 磁盘剩余约 186 GiB、使用率 98%；后续优先用只读链接和内容哈希登记数据资产。
- 已确认官方 Weather 缺失示例对应 point-missing、nominal r≈0.06、block length=5，realized missing rate=0.451571；所有已观测值与服务器完整 Weather 完全相同，可作为首个 smoke/快速基线数据资产。
- 上游状态仍干净，commit `a718823addd3606e763dfc261174e0135b2535f4`；官方 Git 历史没有额外复现实验配置可恢复。
- 已发现工作区有不同任务协议下的 S4M 内部适配经验；本项目不会搬用其 NYCTAXI 指标，只将其作为只读工程参考。
- 已创建独立可修改基线副本 `baselines/S4M`；`upstream/S4M` 继续保持只读和干净。
- 已为四个完整数据集和官方 Weather 缺失示例建立项目内只读链接及 SHA-256 数据清单，没有重复复制大文件。
- 已在基线副本中完成最小复现修复：完整数据路径参数化、原型张量跟随当前设备、CPU smoke 开关、W&B 禁用模式、可选 SGD/Adam、smoke batch 上限、项目内 checkpoint/结果路径和原子 `metrics.json`。
- 已新增单实验审计启动器和 Weather point r=0.06 smoke 配置；启动器会保存唯一 ID、resolved config、命令、环境、Git 状态、代码 diff、日志、指标、资源状态和 checkpoint 重载验证。
- 第一次 CPU smoke `...20260714_154318` 退出码0并通过 checkpoint 重载，但审计发现官方残余文本写入基线源码目录；已修复为写入 run 目录并保留第一次记录。
- 修复后第二次 CPU smoke `...20260714_154515` 再次退出码0，明确记录1次 optimizer step、train/val/test 各1个 batch、checkpoint 重载通过，所有必需 run 文件均已落盘。
- 一条辅助审计命令在未激活 Conda 的子 shell 中调用 `python`，出现 `python: command not found`；这不影响启动器内已完成的 checkpoint 验证，后续审计固定使用环境内 Python 绝对路径。
- 相同 seed 两次 smoke 指标存在差异，下一步在 W&B 初始化后重新设置随机种子并做双次确定性复测。
- W&B 后重新 seed 的两次 smoke `...154705` 与 `...154759` 仍不一致；已查明官方 Bio.Cluster 的随机多起点聚类不受现有 seed 控制。
- 已新增默认关闭的 `deterministic_cluster` 复现开关，并将 smoke 身份改为 `s4m_public_sgd_reprofix`；下一步再次双跑验证指标与 checkpoint 哈希。
- 宿主当前 RTX 4080 仅4 MiB空闲、RTX 4090 约3691 MiB空闲；已用 RTX 4090 UUID 验证 PyTorch 映射成功，但尚未在资源不足时启动正式 GPU 配置。
- 新增 full-architecture GPU smoke 配置和 UUID 传递时，runner 一次静态编译发现字符串拼接语法错误；未启动实验，已立即修正并记录。
- 确定性聚类修复后的两次 CPU smoke 指标与 checkpoint 哈希完全一致，固定 seed 可还原性检查通过。
- full-architecture GPU smoke `...20260714_155635` 在 RTX 4090、启动空闲约4914 MiB时 OOM，runner 正确落盘 `state=OOM`；失败发生在首个 forward，尚未到 backward，未修改 batch/model/协议。
- OOM 后已确认本实验 PID 不存在；下一步建立后台资源等待器，初始 required-memory estimate 采用约9 GiB并根据后续真实 profiling 动态更新。
- 已创建并启动后台 GPU 资源等待器，PID `2869590`、SID 与 PID 相同、PPID=1；状态文件已跨轮询更新为 `WAITING_RESOURCE`，初始动态显存估计9000 MiB、OOM计数1。
- 等待器使用两张卡的稳定 UUID、`flock` 原子选择、60秒低频轮询、OOM递增估计与指数退避；不会占用假显存，也不会修改模型/batch或清理其他用户进程。
- 后台启动 shell 因工具会话未立即回收而被终止等待，但等待器已 `setsid` 脱离且持续更新状态；没有重复启动。
- 已生成 `reports/latest_results.csv` 和 `reports/latest_results.md`，自动汇总6个成功 smoke 与1个 OOM 记录；缺失指标保持 `NA`，没有手工填值。
- 最终交接核验：等待器 PID `2869590`、PPID=1、SID=`2869590`、状态 `WAITING_RESOURCE`，状态文件已更新到 `2026-07-14T08:04:39Z`；上游 Git 仍无修改。
- 阶段结论：前期准备完成，第一步的最小 CPU 链路已完成；官方完整架构 GPU smoke 正在后台等待资源，Gate R/正式基线尚未完成。

### 步骤4.1–4.2：4080 资源与 OOM 根因复核
- **状态：** in_progress
- 用户报告 RTX 4080 有空闲显存后，宿主实测空闲 14,349 MiB；4090 仅余311 MiB。
- 等待器真实存活，但其当前动态需求估计为20,992 MiB：此前第3次尝试在4090空闲18,576 MiB时仍 OOM，故不会盲目把同一完整配置投到16 GiB的4080。
- OOM 精确发生在 S4 fallback 的 `torch.fft.rfft`；日志确认官方 structured CUDA extension 未编译。下一步核验工具链后在独立环境补齐该官方扩展，再重试完整架构 GPU smoke。
- 工具链预检：当前系统 nvcc=11.5，而项目 PyTorch=2.1.0+cu121；直接编译官方扩展有 ABI 风险，尚未执行编译。下一步查找可用 CUDA 12.1 编译器或采用项目环境内匹配 toolkit。

### 步骤4.3–4.10：官方 CUDA 扩展、4080 重试与新版等待器
- **状态：** in_progress
- 在独立环境安装匹配 PyTorch cu121 的 `cuda-nvcc=12.1.105`，命令退出码0；未修改系统 CUDA、驱动或 base 环境。
- 编译官方 S4 `structured_kernels` 扩展成功，并新增 `scripts/build_s4_structured_kernels.sh` 一键重建入口；动态库 SHA-256 为 `90c9ee5...2a0a8b`。
- 第一次扩展数值测试因官方双层绝对导入找不到 `structured_kernels` 而退出1；修复实验启动器 PYTHONPATH 后，RTX 4080 上的前向、反向和 S4 加载校验全部通过。
- 新增按 UUID 自动解析 GPU 物理 index、名称、总/已用/空闲显存的审计；本次解析后的设备正确记录为物理 index 0、RTX 4080，而非配置模板中的 RTX 4090。
- 尝试4保持完整架构、batch size 64、SGD 和固定数据协议，在4080启动空闲14,349 MiB时 OOM；状态、resolved config、日志和资源文件位于 `runs/s4m_public_sgd_reprofix__weather__point__mr0.06__seed2023__smoke__20260715_222144/`。
- 尝试4失败发生在模型迁移到 CUDA，训练尚未进入 forward；同时发现父启动器的 CUDA 上下文额外占用约1.38 GiB，已修复为父进程隐藏GPU、子进程独占所选UUID，并新增模型参数/缓冲区占用日志。
- 修复完成后两张卡均被其他任务占用，无法立即重试。已启动新版后台等待器 PID `3477774`（PPID=1、SID=3477774），当前状态 `WAITING_RESOURCE`；需求估计14,336 MiB，OOM后递增退避回队列，不改变任何正式配置。等待器状态已修正并记录尝试4的准确 OOM 时间 `2026-07-15T14:22:36.560961+00:00`。
- 下一步：等待器自动完成尝试5；成功后核验 optimizer step、checkpoint重载、评估、峰值显存并关闭步骤4，失败则依据新模型 footprint 和OOM位置更新资源估计。

## 会话：2026-07-16

### 步骤4.13–4.15：完整架构 GPU smoke 成功与归档
- **状态：** complete
- 用户报告4090完全释放后立即核验宿主状态；未重复提交任务，因为后台等待器已在06:03:45自动抓到启动空闲16,053 MiB的RTX 4080并完成第5次尝试。
- 成功实验 ID：`s4m_public_sgd_reprofix__weather__point__mr0.06__seed2023__smoke__20260716_060345`；runner和训练子进程均退出码0，等待器最终状态`SUCCESS`且没有残留实验PID。
- 保持Weather、point missing nominal r=0.06、seq_len=192、pred_len=96、8层、256维、batch size=64、SGD和seed2023；仅以smoke配置限制batch数，因此不作为论文结果。
- 完成1次optimizer step，训练/验证/测试各处理1个有效batch；MSE=0.6923139、MAE=0.6670732、RMSE=0.8320540，仅用于链路验证。
- 模型参数量5,821,929，参数占用22.21 MiB；峰值GPU allocated显存9,482.35 MiB、reserved显存9,550 MiB；运行耗时32.01秒。
- checkpoint包含`model/memory/repre/num_dic`并通过CPU重载，大小23,803,260字节，SHA-256为`082c3e5c61d9cd40043d728a9851223a70d21d0ebdba9279186954420f09db6d`。
- `resolved_config.yaml`、`command.sh`、`env.txt`、`git_state.txt`、`code.diff`、`train.log`、`metrics.json`、`status.json`、`resource_usage.json`和checkpoint均存在且非空。
- 自动结果表已更新至11条运行记录；上游`upstream/S4M`保持干净。步骤4正式完成。
- 下一步：步骤5，建立Weather与ETTh1同协议单seed快速正式基线配置，先做配置审计和显存预算，再进入统一后台队列。

### 步骤5.1–5.2：正式基线协议与数据缺口审计
- **状态：** in_progress
- 已重新读取项目规范、规划、发现和进度文件，并完成会话恢复检查；步骤5目标保持Weather与ETTh1单seed快速正式基线。
- 现有Weather官方缺失资产可直接复用；ETTh1仅有完整CSV，尚无固定point-missing r=0.06、block length=5缺失资产。
- 论文声明协议为batch size 16、Adam、早停；公开代码实际为SGD且README示例依赖默认batch size 64。首轮主正式身份确定为`s4m_paper_adam`，公开SGD结果继续独立保存。
- ETTh1正式数据继续使用`custom4`数据类，以维持论文0.7/0.1/0.2时序划分和显式完整目标路径；不使用传统ETT固定月份划分。
- 当前GPU实测4080仅64 MiB空闲、4090约11,151 MiB空闲，暂不在协议和smoke未完成前提交正式任务。
- 下一步：从ICLR正式论文核对seq_len=192、point r=0.06下Weather/ETTh1报告值，随后生成固定ETTh1缺失资产、正式配置和统一队列。

### 步骤5.3–5.6：固定mask、正式配置、队列自测与启动阻塞
- **状态：** blocked_at_host_launch_authorization
- 新增`generate_paper_point_masks.py`，用固定seed2023生成Weather/ETTh1连续长度5的point-missing资产，并保存CSV、NPZ mask、起点、元数据和SHA-256；命令退出码0。
- Weather缺失13015/52696行，实际率0.2469826932；ETTh1缺失4022/17420行，实际率0.2308840413。结构、列、全变量同步缺失和完整值保持检查通过。
- 更新数据清单，将公开Weather缺失文件纠正为`weather_point_unknown_public_example`并禁止正式使用；新增两份论文报告实际率校准资产，明确原论文mask seed未公开。
- 新增Weather/ETTh1各一份paper-Adam smoke配置和正式配置；正式配置固定seed2023、seq_len192、pred_len96、batch16、15 epoch、Adam、完整batch，无训练/评估裁剪。
- Weather学习率0.001、ETTh1学习率0.005；两者采用公开Weather命令的8层、256维、30模式注意力架构。ETTh1最终层数未由论文/仓库公开，已作为协议不确定性落盘。
- 新增参数量指标落盘和协议一致性字段；近似mask结果不再被结果汇总错误标记为完全同协议。四份YAML与相关Python编译检查退出码0。
- 新增`run_stage5_baseline_queue.py`及start/status/stop脚本；支持smoke成功后自动提交对应正式实验、最多并发2、UUID选卡、flock、动态显存估计、OOM退避回队列、确定性错误停止、断点恢复和自动结果汇总。
- 调度器极短假任务自测`PASS`：最大并发2、动态补位、依赖顺序、OOM一次回队列后成功、数据错误仅尝试一次、恢复状态保留均通过；退出码0。
- 启动前宿主快照：物理4080空闲64 MiB、物理4090空闲10648 MiB，当前只适合先运行一个batch16 smoke。
- 宿主`nohup+setsid`持久启动请求因审批服务连接中断被安全审查拒绝；随后宿主`ps`确认没有stage5调度器、runner或S4M训练进程。没有进行替代性绕过，也没有虚报启动。
- 下一步：用户明确授权宿主后台启动后，执行`bash scripts/start_stage5_baseline_queue.sh`，再用调度器PID、run状态、训练日志增长和GPU计算进程四重核验真实启动。

## 会话：2026-07-17

### 步骤5.6：宿主后台队列成功启动并完成真实性核验
- **状态：** running
- 用户明确授权继续步骤5后，重新读取项目规划、发现和进度，并执行session catchup；没有发现未同步上下文。
- 启动前确认无旧stage5 PID、状态或训练进程；调度器与shell脚本的Python/语法检查退出码0。
- 启动前GPU快照：物理4080空闲818 MiB、物理4090空闲6332 MiB。当前仅满足一个6144 MiB smoke估算，调度器未强行并发第二项。
- 宿主命令`bash scripts/start_stage5_baseline_queue.sh`退出码0，创建持久调度器PID 424621；宿主核验PPID=1、SID=424621、状态`Ss`。
- `weather_smoke`已分配物理4090 UUID `GPU-a3af...`，runner PID 424647，正式run目录为`runs/s4m_paper_adam_reprofix__weather__point__mr0.06__seed2023__smoke__20260717_155713`。
- run状态为`RUNNING`，训练子进程PID 426010；`nvidia-smi`确认该PID属于`s4m_missing_forecasting`环境并占用物理4090。
- 训练日志持续增长，已打印Adam学习率0.001、batch size16、完整8层模型参数量5821929、`start training`和首批张量形状，证明forward链路真实执行。
- ETTh1 smoke为`WAITING_RESOURCE`；Weather/ETTh1正式任务保持`PENDING`并等待各自smoke成功。调度器会在资源释放后自动补位，无需人工持续监控。
- 下一步：继续沿用步骤5.6，由后台队列等待Weather smoke结果；成功后自动启动后续smoke/正式基线，OOM则按退避规则回队列，确定性错误则停止并保留证据。

### 步骤5：首轮正式失败诊断与epoch边界原型库修复
- **状态：** repair_validated_pending_requeue
- 队列于2026-07-17 16:17结束：Weather与ETTh1 smoke均`SUCCESS`；两项正式任务均为`CODE_ERROR`，调度器正确停止确定性错误且保留失败记录。
- Weather正式run `...weather...20260717_160002`完成epoch 1的2287步并保存checkpoint，耗时782.5秒；ETTh1正式run `...etth1...20260717_161404`完成epoch 1的744步并保存checkpoint，耗时156.3秒。
- 共同错误为epoch 2读取`self.memory.repre.clone()`时`repre=None`。上游源码确认每epoch调用`init_bank()`清空原型库，但warmup条件错误地只覆盖epoch 0。
- 在可修改基线副本中做最小修复：每次renew后的epoch首批都执行memory bank warmup；新增`MEMORY_BANK_WARMUP`日志和`memory_bank_warmup_count`指标。上游`upstream/S4M`不修改。
- Weather/ETTh1 smoke改为2 epoch生命周期回归，仍保持每epoch2个训练batch、1个评估batch；runner要求至少2次warmup，避免1-epoch smoke再次漏检。
- 调度器新增显式`--reset-jobs`修复后回队列功能，保留attempt history和失败run，不覆盖旧结果；标准start脚本通过`RESET_JOBS`环境变量一次性启用。
- Python编译、两份YAML断言、Shell语法和调度器假任务回归均退出码0，self-test仍为`PASS`。
- 不从失败checkpoint续训，因为官方checkpoint未保存optimizer、scheduler和完整随机状态，无法满足安全/等价恢复要求；修复后正式实验将使用新experiment ID从头运行。
- 下一步：以`RESET_JOBS=all`重启stage5队列，先跑两项2-epoch smoke；全部成功后由调度器自动提交Weather/ETTh1正式基线。

### 步骤5：修复后队列重启与ETTh1跨epoch回归通过
- **状态：** running
- 以`RESET_JOBS=all`在宿主重启stage5队列，启动退出码0；新调度器PID 1748452、PPID 1、SID 1748452，旧attempt history完整保留。
- ETTh1第2次smoke run为`runs/s4m_paper_adam_reprofix__etth1__point__mr0.06__seed2023__smoke__20260717_195850`，runner和训练进程均退出码0，状态`SUCCESS`。
- 回归日志跨过epoch边界：epoch 1执行`MEMORY_BANK_WARMUP reason=initial`，epoch 2执行`reason=renewed`；持久指标`memory_bank_warmup_count=2`、`optimizer_steps=2`。
- checkpoint包含model/memory/repre/num_dic并通过CPU重载；参数量5767343，峰值allocated 3059.3 MiB、reserved 3600 MiB。
- smoke的MSE/MAE只来自极少batch，不作为论文结果；本次结论仅是多epoch生命周期bug已在ETTh1链路修复。
- 下一步：等待调度器下次轮询自动补位，继续Weather 2-epoch回归与两项正式基线；确认补位后无需人工持续监控。

### 步骤5：ETTh1正式训练越过原失败点
- **状态：** formal_running_after_regression_gate
- 调度器在ETTh1回归smoke成功后自动启动ETTh1正式attempt 2，runner PID 1753643、训练子PID 1753735，新run为`runs/s4m_paper_adam_reprofix__etth1__point__mr0.06__seed2023__20260717_195944`。
- 正式配置保持Adam、batch16、15 epochs、完整训练/评估batch，无协议裁剪。
- 日志已完成epoch 1的744 steps，Train Loss 1.0332494、Vali Loss 0.9987831，并进入epoch 2执行`MEMORY_BANK_WARMUP reason=renewed`。
- 原首轮失败位置已被正式完整训练越过；截至检查没有`NoneType`、Traceback、AttributeError或CUDA OOM。
- 步骤5仍未完成：需等待ETTh1早停/15 epochs、最佳checkpoint重载和测试，再由队列继续Weather 2-epoch回归与正式基线。
- 下一步：后台调度器继续执行；下次状态检查以ETTh1正式SUCCESS/失败、Weather回归smoke和正式结果为判据，不提前进入步骤6。

### 步骤5：ETTh1正式成功、Weather正式接续运行
- **状态：** weather_formal_running
- ETTh1正式attempt 2于2026-07-17 20:09:44完成，runner退出码0、状态`SUCCESS`；run为`runs/s4m_paper_adam_reprofix__etth1__point__mr0.06__seed2023__20260717_195944`。
- Weather第2次2-epoch回归smoke于20:10:44完成，runner退出码0、状态`SUCCESS`；其历史失败记录仍完整保留。
- 调度器随后自动启动Weather正式attempt 2：runner PID 1815434、训练子PID 1815530，run为`runs/s4m_paper_adam_reprofix__weather__point__mr0.06__seed2023__20260717_201045`。
- 宿主快照显示物理4080空闲12941 MiB，但其上已有两个其他项目训练进程且GPU利用率99%；物理4090上的Weather训练进程占用约4130 MiB并与其他用户任务共享。
- 当前步骤5只有Weather正式任务尚未完成，不存在第二或第三个满足依赖关系的待运行实验；没有为了凑并发跳过smoke门禁，也没有提前启动步骤6创新实验。
- 下一步：核验ETTh1持久化指标、Weather日志增长与进程存活；等待Weather正式成功后关闭步骤5并进入步骤6。
- ETTh1持久化结果核验通过：MSE=1.2079909、MAE=0.8389233、RMSE=1.0900280，checkpoint重载通过，峰值reserved显存3606 MiB，总耗时579.6秒；相较论文参考MSE 0.593明显偏高，后续必须保留协议不确定性并诊断，不能写成完全复现。
- Weather父子PID均存活，日志mtime持续更新；已完成epoch 3并进入epoch 4，当前无Traceback、`repre=None`或OOM。

### 步骤5：Weather与ETTh1单seed快速正式基线完成
- **状态：** complete
- stage5调度器最终`queue_state=SUCCESS`，四项任务（两项2-epoch smoke、两项正式基线）均为`SUCCESS`；调度器已退出，没有残留本项目训练PID。
- Weather正式run `...20260717_201045`在epoch 5触发预设patience=3早停，退出码0；MSE=0.2635257、MAE=0.3011600、RMSE=0.4894362，11430个optimizer steps，checkpoint重载通过，峰值allocated/reserved约3071/3630 MiB，总耗时3053.9秒。
- ETTh1正式run `...20260717_195944`退出码0；MSE=1.2079909、MAE=0.8389233、RMSE=1.0900280，2972个optimizer steps，checkpoint重载通过，峰值allocated/reserved约3063/3606 MiB，总耗时579.6秒。
- 相对论文参考MSE：Weather 0.225→0.263526，绝对高0.038526、相对高17.1%；ETTh1 0.593→1.207991，绝对高0.614991、相对高103.7%。两者均使用固定seed2023近似mask，原论文mask seed未公开，因此只作为可信快速基线，不宣称完全复现论文表格。
- `reports/latest_results.csv`与`reports/latest_results.md`已包含两项正式SUCCESS记录，步骤5完成。
- 下一步：步骤6，仅先做有原型/无原型的Gate H中心失效诊断设计与smoke；用相同checkpoint、固定测试mask和评价脚本衡量原型分支的真实收益或伤害，不提前实现MIPP/FUSR。
- 完成步骤5收口审计：两个正式run的`resolved_config.yaml`、`command.sh`、`env.txt`、`git_state.txt`、`code.diff`、`train.log`、`metrics.json`、`status.json`、`resource_usage.json`与checkpoint均存在且非空；自动结果表存在，上游`upstream/S4M`工作区保持干净。
- 一次Gate H关键词搜索把不存在的根`README.md`列入路径，`rg`报告`No such file or directory`但仍以退出码0返回其他有效匹配；未影响任何代码或实验，后续仅搜索实际存在的`docs/`、配置和规划文件。
- 已复核步骤6的正式Gate H判据：先在不改变模型参数的前提下记录同一样本有/无原型损失、Top-K相似度与原型来源，计算HRR（有害检索率）、MRU（平均检索效用）和NUS（负效用严重度）；快速单seed信号成立后才扩展到两数据集、多seed和两类缺失漂移，否则停止FUSR主线。

## 会话：2026-07-17（步骤6）

### 步骤6：Gate H有/无原型中心失效诊断启动
- **状态：** in_progress
- 用户授权按照既定节奏直接进入下一步，GPU实验无需逐次确认，统一调度最大并发3。
- 本步骤先冻结步骤5的Weather/ETTh1最佳checkpoint、固定测试样本和评价实现，新增可关闭的无原型分支与逐样本配对指标；不训练新权重、不提前实现MIPP/FUSR。
- 快速完成判据：代码开关默认保持基线行为；smoke覆盖有/无原型前向、指标与工件写入；Weather/ETTh1单seed正式诊断完成并输出HRR、MRU、NUS与相似度-效用关系。
- 正式Gate H扩展判据仍为两个数据集、至少两类缺失漂移和多个固定mask seed；仅在快速诊断出现有意义信号后扩展，避免无效大矩阵。
- 新增Weather/ETTh1的缺失率漂移与长块缺失形态漂移seed2023配置；首次全路径校验退出码1，准确发现Weather缺失率漂移正式配置把checkpoint文件名误写成漂移数据名。其他29个必需路径均存在，错误配置尚未进入smoke或正式队列。
- 已把该路径修正为步骤5原始Weather最佳checkpoint；下一步重新执行5份配置的全路径校验与静态编译，全部通过后才启动Weather漂移smoke。
- 修复后5份配置的30个必需路径全部存在，相关Python/Shell静态检查退出码0，上游树干净。
- Weather缺失率漂移smoke `s4m_gate_h__weather__rate_shift_r045_l5__seed2023__smoke__20260717_214238`完整SUCCESS、退出码0；32个样本的原型开/关严格配对、checkpoint加载、样本NPZ、metrics/status/resource工件均通过，峰值allocated/reserved显存约1049.5/1564 MiB。
- smoke方向信号为HRR=1.0、MRU=-0.04284、prototype-on/off MSE=0.21501/0.17218、Oracle相对潜在收益19.9%；只作为2-batch筛选证据，不作为正式Gate H结论。
- Gate H队列已扩展4个seed2023正式漂移诊断任务；最大并发仍为3，实际同时运行数由两张卡的实时资源与UUID锁决定。
- 队列核心self-test全部PASS；同一组合命令最后的独立导入检查因未把`scripts/`加入Python搜索路径而退出1。该问题不影响脚本正常入口，已暂停启动并将用正确`PYTHONPATH`完成最后校验。
- 补`scripts/`搜索路径后队列定义检查与第二次self-test均退出码0：6个任务ID唯一、配置齐全、最大支持并发3。
- `MAX_JOBS=3 bash scripts/start_gate_h_queue.sh`退出码0，宿主持久调度器PID 2400583、PPID 1、SID 2400583；状态文件记录队列名`step6_gate_h_id_and_seed2023_shift_diagnostics`和最大并发3。
- Weather缺失率漂移与长块形态漂移两个正式任务已分别在4080 UUID `GPU-c88...`与4090 UUID `GPU-a3af...`真实运行，runner和评估子进程均存活，run状态为RUNNING且日志已加载正确checkpoint；两个ETTh1任务保持WAITING_RESOURCE并将在卡释放后自动补位。
- 一次非宿主`ps`返回空进程视图，曾被误判为调度器与任务外部终止；宿主复核纠正：调度器PID 2400583自21:45:50持续存活，两项Weather runner/评估/DataLoader进程均持续运行，状态文件每30秒更新。没有第二次启动、没有失败run、没有覆盖结果。
- 为安全兜底，调度器恢复逻辑仍新增“旧PID消失且run仍为RUNNING→EXTERNAL_KILL”分类及自测，另保留tmux恢复入口；本次原队列未触发该逻辑，tmux入口检测到PID存活后退出0并拒绝重复调度器。
- 后续后台真实性判断固定使用宿主PID、状态文件更新时间和run日志三重证据，不能再用隔离环境的空`ps`结果下结论。
- 两项Weather seed2023正式漂移诊断已SUCCESS并由队列自动补位两个ETTh1任务。Weather缺失率漂移：on/off MSE=0.26635/0.28208、HRR=38.63%、MRU=+0.01573、Oracle潜在收益4.94%；长块形态漂移：on/off MSE=0.26420/0.27262、HRR=38.97%、MRU=+0.00842、Oracle潜在收益5.42%。
- Weather两类漂移均显示约39%样本受原型伤害且相似度-效用Spearman仅0.068/0.012，但HRR没有高于ID的41.96%。因此“漂移必然增加有害检索率”当前不成立；“相似度无法识别逐样本效用、选择器仍有约5%Oracle空间”得到跨三种Weather域支持。
- 两个ETTh1漂移任务首次attempt均在模型构造前以CODE_ERROR停止，错误同为`Namespace has no attribute d_var`；原因是公开代码按大小写敏感文件名推断变量维数，新资产采用小写`etth1_`。失败run、日志和状态完整保留，队列未盲目重试。
- 最小修复仅在无法识别文件名时令`d_var=enc_in`并打印`D_VAR_FALLBACK`，已识别的原始Weather/ETTh1路径行为不变；新增ETTh1缺失率漂移2-batch smoke，待通过后再重排两项失败正式任务。
- 修复smoke `s4m_gate_h__etth1__rate_shift_r045_l5__seed2023__smoke__20260717_215502`已SUCCESS、退出码0；日志记录`d_var=7 source=enc_in`，32样本配对、checkpoint加载、metrics/NPZ/status/resource均通过，峰值reserved显存1544 MiB。
- 调度器手工重排接口新增可审计`--reset-reason`；下一步只重置`etth1_rate_shift_seed2023_formal`与`etth1_geometry_shift_seed2023_formal`，理由记录为`etth1_lowercase_filename_d_var_fallback`，不重跑四项SUCCESS任务。
- 首次修复后nohup重排命令虽打印PID 2434603，但宿主随后确认PID已消失、状态未改、attempt未增加、事件流无新`scheduler_started`，总日志也无Traceback；没有任何实验被重复启动。
- tmux持久启动入口现支持同一`RESET_JOBS/RESET_REASON`参数；下一步改用该入口，并以tmux会话、宿主PID和事件流三重证据核验。
- tmux重排成功：会话`s4m_gate_h_step6`、调度器PID 2439334、两项ETTh1 attempt 2及事件流四重证据一致；旧CODE_ERROR attempt和修复原因完整保留，四项既有SUCCESS未重跑。
- 步骤6正式队列最终`SUCCESS`，6/6正式诊断成功；自动报告已更新并同时保留两项首次CODE_ERROR。ETTh1缺失率漂移on/off MSE差约8.1e-10、Oracle收益6.9e-9；长块漂移差约-1.2e-9、Oracle收益8.4e-9，均属于数值噪声量级。
- Gate H最终判定为部分成立：Weather跨ID/两漂移域稳定存在38.6%–42.0%有害样本、4.9%–5.4%Oracle空间且相似度-效用相关接近0；但漂移未提高HRR，ETTh1原型支路基本失活。停止扩展2024/2025 mask seed，不将负结果包装成跨数据集成功。
- **步骤6状态：complete。** 下一步为步骤7：实现默认关闭、可独立消融的MIPP最小版本，先做forward/backward/checkpoint smoke，再在Weather固定seed2023、同预算下与步骤5基线正式比较；FUSR暂不提前实现。

## 会话：2026-07-17（步骤7）

### 步骤7：MIPP最小正式版本启动
- **状态：** in_progress
- 用户授权继续下一步；本步骤只实现MIPP（Mask-Intervention Prototype Purification，掩码干预原型净化），不提前接入FUSR。
- 完成判据：MIPP具备默认关闭开关；关闭态恢复步骤5基线；开启态完成配对掩码视图、稳定内容表示和原型写入接入；smoke覆盖forward/backward/optimizer/checkpoint保存与重载；通过后直接启动Weather seed2023同预算正式对照。
- 已完成文件规划恢复和session catchup，无未同步会话内容；继续保持全局步骤编号为7，不使用“7.x”作为用户可见步骤。
- 现有项目蓝图将MIPP定位在S4M原型库供给侧：同一时序窗口施加不同掩码，稳定内容表示决定写入，缺失表示独立保留；创新不能退化成只在最终预测损失上增加普通一致性正则。
- 已完成蓝图与训练/原型代码路径审计。原型查询来自`encoder_u`，写入来自动量`encoder_m`；每次正常训练forward都会在`no_grad`中把`encoder_m`表示写入`FeatureMemory`。因此MIPP最小接入点必须同时覆盖查询内容投影和写入候选稳定性，而不能只在训练循环额外算loss。
- 当前数据batch已经提供`seq_true_x`完整历史、`seq_x_mask`和缺失输入`seq_x`；人工缺失数据可用同一完整历史构造第二掩码视图，无需泄露未来`seq_true_y`。正式实现必须保证测试阶段不构造第二视图、不访问未来目标。
- 已冻结MIPP最小机制草案：训练时以原缺失视图为A、从`seq_true_x`历史部分生成不同块长度/缺失率的视图B；主预测只走A，B只进入共享查询编码器生成内容稳定约束。测试/验证不生成B。
- MIPP开启时新增零初始化残差内容投影，query检索和momentum写入统一使用净化内容空间；轻量mask统计编码器保留缺失信息，并用内容-缺失去相关与mask统计重建避免“完全不变”删除信息性缺失。
- 写入门直接比较A/B在momentum encoder下的内容余弦稳定性，只允许稳定chunk更新`FeatureMemory`；训练日志和metrics将记录接受率、内容余弦、各项辅助loss和额外显存/时间。MIPP关闭时不创建第二视图、不计算额外loss、不改变写入路径。
- 一次读取smoke配置时猜错文件名并由`sed`返回文件不存在；未修改代码或配置。后续用`rg --files`定位真实名称，不重复猜测。
- 已定位真实Weather paper-Adam smoke配置：2 epochs、每epoch最多2个训练batch、1个验证/测试batch、batch16，并要求两次原型库warmup。MIPP smoke将继承这套生命周期门禁，只新增MIPP开关与机制指标，不改变基线smoke文件。
- 现有正式runner已覆盖唯一experiment ID、resolved config、环境、git/diff、metrics、resource、checkpoint重载；MIPP实验继续复用，runner只需新增MIPP smoke机制指标断言和结果身份，不新建平行训练系统。
- 下一步：完整审计项目局部规范和S4M模型/测试入口，确定最小、可逆的原型旁路位置及诊断工件格式。
- 项目局部`AGENTS.md`已完整复核：创新开关默认关闭时必须恢复基线，多任务使用最大并发3调度器，GPU按UUID审计，smoke先行且上游只读。
- 初步代码定位确认原型检索集中在`baselines/S4M/s4m/model/Bank.py`的`MTNet.forward`，测试循环位于`experiments/exp_pretrain1.py`；最公平路径是在同一加载模型、同一batch和同一目标下执行原型开/关配对前向。
- 一次代码搜索包含不存在的`baselines/S4M/s4m/models`路径，`rg`报告`No such file or directory`；有效匹配来自实际的`model/`目录，未修改文件。后续只使用已确认路径。
- 完整张量路径确认：`MTNet.forward`将检索原型向量、查询编码和原始插补输入拼接后通过已有`pred_w/pred_b`，最后再加查询残差；无原型干预应仅把检索向量置零，保留其余通道和所有已训练权重。
- 测试态仍可能在原型数大于`topM`时调用`torch.randperm`随机抽候选；Gate H必须固定该候选选择或保存/恢复随机状态，否则同一checkpoint的效用标签会混入候选抽样噪声。
- 诊断应使用独立test-only入口加载步骤5最佳checkpoint，避免现有训练入口重新训练；同一batch连续计算prototype-on与prototype-off，并保存逐样本损失及检索相似度。
- 现有`run.py`只在`is_training=1`路径执行S4M train+test，没有可直接加载任意checkpoint并测试的正式入口；步骤6将新增独立审计runner，不复用会触发重训的路径。
- checkpoint形状探测第一次命令把字面量`\n`传给`python -c`，触发`SyntaxError: unexpected character after line continuation character`（退出码1）；这是只读诊断命令转义错误，未触碰实验。下一次改为无换行的一行表达式。
- 修正后的只读探测退出码0：Weather最佳checkpoint仅有4个原型、ETTh1仅有3个原型，均远小于`topM=100`；因此步骤5/6当前checkpoint测试不存在随机topM候选抽样，配对结果可直接复核，无需改变候选策略。
- 已在可修改基线副本加入原型开关、检索诊断缓存、`run.py`的test-only Gate H模式，以及`exp_pretrain1.py`的逐样本配对评估；默认原型开启，不改变原基线入口。
- 逐样本工件将保存prototype-on/off MSE与MAE、utility、top1/selected相似度、检索熵和harmful标记；汇总计算HRR、MRU、NUS、Oracle selector上界及相似度-效用Pearson/Spearman相关性。
- 静态编译和`git diff --check`均退出码0；`upstream/S4M`仍干净。尚未启动GPU实验，下一步先创建可审计runner与Weather smoke配置。
- 新增`scripts/run_gate_h_diagnostic.py`：test-only加载源checkpoint，生成唯一实验ID、resolved config、命令、环境、Git/diff、源checkpoint指针与哈希、逐样本NPZ、汇总JSON、资源和状态文件。
- 新增`configs/gate_h/s4m_weather_id_seed2023_smoke.yaml`，固定Weather步骤5checkpoint与原测试资产，只处理2个测试batch验证配对链路；该smoke不作为Gate H正式结论。
- runner、四个改动源码的静态编译、`--help`、YAML必需路径断言和`git diff --check`全部退出码0；上游仍干净。下一步核验GPU后直接启动Weather Gate H smoke。
- RTX 4080启动前宿主快照：物理index0、UUID`GPU-c88b...`、空闲12941 MiB；首次Gate H smoke run为`runs/s4m_gate_h__weather__id_point_r006__seed2023__smoke__20260717_212748`。
- smoke退出码0、状态`SUCCESS`，处理2个batch/32个样本；源checkpoint重载、逐样本NPZ、汇总JSON及全部审计工件存在，峰值allocated/reserved约1049.5/1564 MiB，总耗时32.7秒。
- 初步链路信号：HRR=0.65625、MRU=-0.0064526、NUS=0.0122299，Oracle selector相对prototype-on潜在改善6.15%；Top-1相似度均值0.997706，但相似度-效用Spearman仅0.09898。该结果仅验证链路和方向，不作为Gate H正式结论。
- smoke仅看前2个batch，却被runner计算了相对完整步骤5 MSE的差值，字段在统计意义上不可比较；已修正为只有ID域、非smoke且完整测试时才写`prototype_on_baseline_match_abs_error`，保留原smoke工件不改写历史。
- 下一步：生成Weather与ETTh1完整ID配对诊断配置，静态验证后交由最大并发3的统一队列运行；完整prototype-on MSE必须与步骤5基线匹配。
- 新增Weather与ETTh1完整ID诊断配置、`run_gate_h_queue.py`、start/status/stop脚本及`aggregate_gate_h_results.py`；调度器配置最大并发3，当前两项任务可在资源允许时并行。
- 为保持已验证调度逻辑，Gate H wrapper复用步骤5调度引擎，仅隔离runtime/log/state、runner、job definitions和结果汇总器；步骤5默认队列名保持不变。
- Python编译、Shell语法、两份formal YAML路径/完整测试断言、`git diff --check`均退出码0；上游干净。
- 假任务调度自测`PASS`：动态补位、依赖顺序、OOM一次回队列后成功、确定性数据错误不重试及恢复状态均通过；自动Gate H结果表已生成并纳入现有smoke。
- 下一步：以`MAX_JOBS=3`宿主后台启动ID诊断队列，并用调度器PID、两个runner状态、训练日志和GPU进程核验真实运行。
- 宿主命令`MAX_JOBS=3 bash scripts/start_gate_h_queue.sh`退出码0；调度器PID 2304968、PPID 1、SID 2304968，状态`RUNNING`。
- Weather ID formal runner PID 2304976分配物理4080 UUID`GPU-c88b...`；ETTh1 ID formal runner PID 2304979分配物理4090 UUID`GPU-a3af...`。两项均为attempt 1，队列记录最大并发3。
- 首次启动后13秒GPU进程快照尚未出现两个runner的训练子进程，符合父runner先做CPU checkpoint哈希/完整性校验的路径；下一步继续核验run目录、子PID、日志增长和GPU上下文后再确认真实前向运行。
- 后续核验确认两个runner和训练子PID均存活、run状态`RUNNING`、日志已完成模型加载；但宿主`nvidia-smi`显示Weather子PID 2305091与ETTh1子PID 2305102都实际位于物理4090，和调度器记录Weather→4080、ETTh1→4090不一致。
- 两个本项目子进程当前各占约2 GiB，尚未OOM；在定位UUID路由前不向队列追加第三任务。下一步只读检查各子进程的`CUDA_VISIBLE_DEVICES`和生成命令，判断是环境导出丢失还是PyTorch初始化顺序问题。
- Weather子进程环境与工件均明确为`CUDA_VISIBLE_DEVICES=GPU-c88b...`、resolved physical index0/RTX4080；ETTh1工件明确为`GPU-a3af...`、physical index1/RTX4090，说明调度器到子进程的UUID导出没有丢失。
- 读取ETTh1 `/proc/2305102/environ`时进程已在检查瞬间结束，命令返回`No such file`（退出码1）；这不是实验错误。下一步读取调度最终状态与指标，并复核当前live GPU进程，判断先前双落4090是否为瞬时NVML快照/进程结束竞争或真实路由问题。
- ID队列于2026-07-17 21:34:03结束，最终`queue_state=SUCCESS`；ETTh1与Weather formal attempt 1均退出码0、run状态`SUCCESS`，无OOM和残留本项目GPU进程。
- Weather的子进程环境、command和env审计三者均固定4080 UUID，ETTh1固定4090 UUID；最终两项正常结束。先前一次NVML把两个瞬时PID都列在4090的现象未伴随环境/工件错误，当前按瞬时快照竞争保留记录，不据此改写成功run的GPU审计。
- 自动Gate H汇总已更新到3条记录（1个smoke、2个正式ID）。下一步读取两项正式指标并核对prototype-on与步骤5 MSE的一致性，再决定漂移扩展。
- 完整ID公平性通过：Weather prototype-on MSE 0.2635257609，与步骤5相差1.57e-8；ETTh1 1.2079909674，与步骤5相差2.47e-8。
- Weather ID：HRR=0.41957、MRU=+0.008842、NUS=0.013400、Oracle相对上界5.08%，相似度-效用Spearman=0.0684；原型总体有益但对约42%样本有害，存在选择空间。
- ETTh1 ID：prototype-on/off MSE差约4.4e-10，85.5%样本为浮点级中性，Oracle上界约7.3e-9；该checkpoint几乎不使用原型分支，不能把7.2%的符号翻转HRR解释成实质伤害。
- 当前Gate H尚未成立：需要验证缺失漂移是否在至少两个数据集提高实质HRR。下一步生成rate shift（缺失强度漂移）与geometry shift（长块形态漂移）固定资产。
- 漂移资产将只替换测试窗口覆盖区，训练段、验证主体和训练归一化统计保持与步骤5完全相同；每类先固定seed2023快速诊断，有信号后再扩展seed2024/2025。
- 新增`scripts/generate_gate_h_shift_assets.py`：从完整CSV重建测试窗口覆盖区，同时逐值保留步骤5训练段mask；支持rate shift（长度5、评估区缺失率0.45）和geometry shift（长度20、保持数据集ID缺失率），固定seed列表并保存CSV/NPZ/JSON/hash。
- 生成器静态编译与`--help`均退出码0；下一步先生成Weather/ETTh1两类漂移的seed2023共4份资产，并审计训练mask恒等、观测值恒等、缺失率和块长度。
- seed2023首次生成命令退出码1，错误为`RuntimeError: observed values changed while generating the shift`；失败发生在写文件前的观测值严格相等断言，没有产生新漂移资产，也未影响训练/GPU任务。
- 初步判断是步骤5缺失CSV与完整CSV经过CSV序列化后存在浮点表示级差异，`np.array_equal`过严；下一步量化非相等元素、最大绝对/相对误差，再改成显式容差断言并保存误差审计值，不能直接删除检查。
- 量化结果：Weather观测值最大绝对差2.27e-13、ETTh1为7.11e-15，两者在`rtol=atol=1e-12`下全部一致；确认是CSV序列化精度而非数据污染。
- 生成器已改为显式`1e-12`相对/绝对容差，并把最大观测值误差和容差写入metadata；训练mask恒等仍使用精确`array_equal`。下一步重新生成seed2023资产。
- 修复后生成器编译退出码0，seed2023四份漂移资产生成退出码0；所有metadata均确认`training_mask_identical_to_step5=true`与`normalization_fit_region_unchanged=true`。
- Weather rate shift评估区实际缺失率0.44991（L=5），geometry shift为0.24779（L=20）；ETTh1分别为0.45022（L=5）与0.23069（L=20）。训练段、完整目标和观测值均通过检查，CSV/NPZ/JSON及SHA-256已落盘。
- 下一步：创建Weather rate-shift的2-batch smoke配置验证新数据链路；通过后把Weather/ETTh1两类seed2023漂移共4项加入最大并发3的正式队列。

## 2026-07-17 步骤7实施检查（续）

- 项目聚合目录本身不是 Git 仓库；可编辑基线仓库为 `baselines/S4M`，后续 Git 状态与差异均以该仓库为准。
- 当前 MIPP 主体改动已落到 `MIPP.py`、`Memory.py`、`Bank.py`、`S4M.py`、`exp_pretrain1.py` 与 `run.py`，尚未宣告通过，正在执行编译、默认关闭闭环和组件测试。
- 六个 MIPP 相关 Python 文件及实验运行/汇总脚本已通过 `py_compile`，退出码为 0；这只证明语法可导入，不代表训练链路已经通过。
- 新增 `scripts/test_mipp_components.py` 并通过：零初始化内容适配器保持恒等映射、所有损失有限、内容与掩码分支均有非零梯度、稳定性写入门控只接收 eligible candidate（合格候选）；报告为 `reports/stage7_mipp_component_test.json`。
- 步骤7调度器假任务自测通过：最大并发、动态补位、OOM 回队列、确定性数据错误不重试、依赖顺序和状态恢复全部为 PASS。
- Weather 正式 MIPP 与步骤5 Weather 基线已完成配置公平性检查：除 MIPP 开关/权重与实验标识外，50 项官方训练参数、数据资产、seed、15 epoch 训练预算完全一致。
- 首次 GPU 冒烟运行 `s4m_mipp__weather__point__mr0.06__seed2023__smoke__20260717_223735` 失败并保留为 `CODE_ERROR`：长度20块掩码使用了 PyTorch 不允许的内部 padding=19。已改为左侧显式补零后执行无 padding 池化，正式协议未改变。
- 修复后冒烟 `s4m_mipp__weather__point__mr0.06__seed2023__smoke__20260717_223927` 成功：2 次优化器更新、2 次原型库暖启动、MIPP 参数完整写入并加载 checkpoint，峰值分配/预留显存约 4541/4946 MiB。
- GPU 路由审计发现 S4M `_acquire_device` 会把调度器传入的 UUID 覆盖为逻辑 `0`，导致请求 4080 实际落到 4090。原生 UUID 单标量探针确认 4080 UUID 可正确识别；已修复为保留外部 `CUDA_VISIBLE_DEVICES`，仅将进程内设备设为局部 `cuda:0`。已经运行的正式任务不被中断。
- Weather 正式同预算 MIPP 实验 `s4m_mipp__weather__point__mr0.06__seed2023__20260717_224027` 已由队列自动启动；已完成至少300个 epoch-1 迭代并持续写日志，当前 `status.json=RUNNING`，不做人工中断或缩水。
- 新增无 GPU 后台比较器：正式队列成功后自动读取持久化 `metrics.json`，生成 `reports/stage7_mipp_comparison.json` 与 `.md`，包含 MSE/MAE 绝对变化、相对收益、参数/耗时/显存与同协议复核；失败则保留 `comparison.log`，不伪造比较结果。
- 用户要求在主MIPP正式实验之外再跑两个实验。步骤7将新增两个Weather seed2023同预算变体：`strict_gate_t095`（稳定写入阈值0.95，其余MIPP机制不变）与`pair_gate_only_t095`（保留配对一致性和严格写入门控，关闭内容-缺失因子化并将其辅助权重置零）。每个变体先独立smoke，再进入15-epoch正式任务。
- 四份变体配置与统一调度器已生成。静态语法、58项共享训练参数、数据/mask/seed/预算一致性、4个任务定义唯一性和路径存在性均PASS；调度器假任务的并发、补位、OOM回队列、确定性错误停止与恢复测试全部PASS，上游`upstream/S4M`保持干净。
- strict-gate首次smoke在创建checkpoint目录前以`OSError 36`停止，根因是S4M拼接后的单个目录名超过文件系统限制；运行`...s4m_mipp_strict_gate_t095...232513`保留为CODE_ERROR。已仅缩短四份配置的身份字段为`mipp_sg95/mipp_pg95`和`SG95*/PG95*`，未修改模型机制或正式协议。
- pair-gate-only首次smoke在调度器下一轮补位时仍读取了修复前配置，同样在训练前以`OSError 36`停止；运行`...s4m_mipp_pair_gate_only_t095...232613`保留。修复后四份配置生成的S4M setting目录名均为240字符（低于NAME_MAX=255），路径门禁PASS。
- 用户再次要求有实验时并行3个。所有队列保持`max_jobs=3`；当前主MIPP占4090、4080只够安全容纳一个约5GiB MIPP任务，另一个变体持续排队。若任一GPU出现足够空间，调度器立即补位到3，不通过牺牲显存安全余量或改batch强行并发。
### 步骤7：并行变体队列恢复（本轮）
- **状态：** preparing_requeue_after_identity_shortening
- 用户再次明确要求：只要存在待运行实验，调度器应保持最多并行 3 个并可随显存变化自动补位。
- 当前只会重排已经通过公平性审计的两条 Weather 单 seed MIPP 机制变体；主 MIPP 正式实验保持不被中断。两个旧 smoke 的 `OSError: [Errno 36] File name too long` 已作为确定性 `CODE_ERROR` 保留，修复后的短身份 setting 通过 240 字符长度门禁。
- 下一动作：先以宿主 PID、队列状态、日志增长和 GPU UUID 复核真实状态；随后用 `RESET_JOBS=all` 且明确 reset 原因重新排队。队列上限仍为 3；若当前实际显存只允许 2 个，则第三个保持 `WAITING_RESOURCE`，资源增加时自动启动。

### 步骤7：短 ID 变体队列已重新启动，进行真实性收口
- **状态：** requeued_then_verifying_runner_liveness
- 宿主启动命令 `RESET_JOBS=all RESET_REASON=shorten_s4m_setting_below_name_max MAX_JOBS=3 bash scripts/start_stage7_mipp_variants_queue.sh` 最终退出码 0，持久调度器 PID 为 `2910219`（PPID 1、SID 2910219），队列状态为 `RUNNING`、并发上限为 3。
- 新队列实际先选中 `pair_gate_only_t095_smoke`，attempt 2 的新短 run 为 `runs/mipp_pg95__weather__point__mr0.06__seed2023__smoke__20260717_234641`；`strict_gate_t095_smoke` 按资源判断保持 `WAITING_RESOURCE`，两个正式任务仍等待各自 smoke 成功。
- 首次复核时队列状态中出现 training PID `2910653`，但 runner PID `2910234` 同时呈 zombie；因此不能把状态文件当成训练已确认。下一动作改为直接核验 training PID、run `status.json` 与 `train.log`，再决定是正常成功、确定性错误还是调度器状态滞后。

### 步骤7：简化门控 smoke 成功，正式依赖已解除
- **状态：** pair_smoke_success_waiting_for_formal_autolaunch
- 直接核验显示 runner zombie 是其已经成功退出后的正常子进程回收窗口，而不是异常：`runs/mipp_pg95__weather__point__mr0.06__seed2023__smoke__20260717_234641/status.json` 为 `SUCCESS`、return code 0、checkpoint CPU 重载通过，且 checkpoint 包含 7 个 MIPP 参数键。
- 训练日志确认外部 UUID 没有再被 S4M 覆盖：`Use GPU: cuda:0 (preserved external CUDA_VISIBLE_DEVICES=GPU-c88...)`；运行的机制开关为 `use_content_mask_factorization=False`、严格稳定写入门控 0.95、只保留配对不变性损失 0.05。
- smoke 的 MSE/MAE=1.07837/0.81708 仅来自受限验证链路，不用于与正式基线比较；该 smoke 的作用是解除 `pair_gate_only_t095_formal` 的依赖。下一次队列轮询会按实时显存尝试启动其 15-epoch 正式任务；`strict_gate_t095_smoke` 保持资源等待而非强行同卡。

### 步骤7：两项正式实验已真实并行，第三项持续资源等待
- **状态：** two_formals_running_third_waiting_resource
- `pair_gate_only_t095_formal` 已由队列自动启动，run 为 `runs/mipp_pg95__weather__point__mr0.06__seed2023__20260717_234738`，runner PID `2914612`、训练 PID `2914733` 均宿主存活；日志已推进到 epoch 1 / iter 200。实际 GPU 是 RTX 4080 UUID `GPU-c88...`，`nvidia-smi` 对应 PID 显示约 5948 MiB。
- 主 MIPP 正式训练 PID `2607077` 仍在运行，实际位于 RTX 4090 UUID `GPU-a3af...`、约 5450 MiB。主队列历史状态中保存的旧 GPU UUID 是路由修复前的元数据，实时 compute-process UUID 和训练日志为此时的路由真值。
- 此刻 4080 仅剩约4477 MiB、4090仅约32 MiB；严格门控完整 MIPP smoke 的可信启动估计为6656 MiB，因此其 `WAITING_RESOURCE` 是正确的安全行为。队列仍以 `MAX_JOBS=3` 后台轮询，显存满足时会自动启动第三项；没有改小 batch、epoch 或模型来凑并发。
- 下一动作：不人工轮询占用 GPU；等待后台队列完成主/简化门控正式任务并自动补位严格门控 smoke，随后由其依赖启动对应正式任务，并自动生成同协议比较表。
### 步骤7：2026-07-18 完成状态实时审计
- **状态：** all_queues_success_pending_result_decision
- 宿主只读核验确认两个步骤7队列都已正常结束：`runtime/stage7_mipp_queue/state.json` 的 `queue_state=SUCCESS`、结束时间 2026-07-18 00:39:10（北京时间）；`runtime/stage7_mipp_variants_queue/state.json` 的 `queue_state=SUCCESS`、结束时间 2026-07-18 03:31:40（北京时间）。两个 scheduler PID 均已退出，当前 GPU 上没有步骤7训练进程残留。
- 三项正式 run 均为 `SUCCESS`：主 MIPP `runs/s4m_mipp__weather__point__mr0.06__seed2023__20260717_224027`；严格门控 `runs/mipp_sg95__weather__point__mr0.06__seed2023__20260718_013328`；简化门控 `runs/mipp_pg95__weather__point__mr0.06__seed2023__20260717_234738`。
- 严格门控 smoke 最终 attempt 3 成功，队列保留了此前一次 OOM 历史并在资源满足后回队列；没有缩小正式 batch、epoch 或模型。下一动作是审计三项正式指标、机制指标、资源代价和工件完整性，依据同协议结果决定步骤7是完成后进入 FUSR，还是在 MIPP 内继续最小迭代。

### 步骤7：正式结果初步比较
- **状态：** positive_but_modest_pending_artifact_gate
- 同一 Weather、固定 point-missing mask、seed2023、15 epoch上限、batch16、Adam、patience3 和相同评估脚本下：基线 MSE/MAE=`0.2635257/0.3011600`；主 MIPP t=0.80=`0.2638118/0.2961380`（MSE +0.109%，MAE -1.668%）；完整 MIPP t=0.95=`0.2661772/0.2975412`（MSE +1.006%，MAE -1.202%）；简化 pair-only t=0.95=`0.2619465/0.2938789`（MSE -0.599%，MAE -2.418%）。
- 当前唯一同时改善 MSE 与 MAE 的是 `pair_gate_only_t095`；这支持“配对不变性有效、复杂内容-缺失因子化辅助目标可能稀释预测信号”的方向，但单 seed 与小幅 MSE 提升还不足以形成强结论。
- 所有配置的最大预算和 early stopping 规则相同，但实际 optimizer steps 不同：基线11430、主/完整MIPP各18288、pair-only 22860。它是同一早停规则下自然产生的训练长度差异，仍需在正式结论中披露；当前不把更长实际训练隐藏起来。
- MIPP 增加153130参数（约+2.63%）；pair-only峰值显存约5036/5622 MiB（allocated/reserved），高于基线3071/3630 MiB。下一动作等待工件/协议独立审计，再决定是否把步骤7标记为“快速验证完成”。

### 步骤7：完成判据审计与唯一剩余收尾
- **状态：** gpu_complete_report_closure_pending
- 三项正式MIPP run及基线的 `resolved_config.yaml`、`command.sh`、`env.txt`、`git_state.txt`、`code.diff`、`train.log`、`metrics.json`、`status.json`、`resource_usage.json` 和 checkpoint 均存在且非空；全部 checkpoint load verification 通过。
- 四项实验共用固定mask `artifacts/masks/weather/point/rate_0.06/seed_2023.npz`、相同CSV、split、seed、输入/预测长度、batch16、Adam、lr0.001、15 epoch上限、patience3及652个测试batch。原论文未公开mask seed，所以只能称项目内严格公平，不能称与论文原mask完全一致。
- 现有 `reports/stage7_mipp_comparison.*` 只比较baseline和主MIPP，没有纳入SG95/PG95；同时比较器未排除 `checkpoints`、`metrics_path`、`results_root`、`test_results_root` 四个必然不同的实验输出目录，导致假性 `PROTOCOL_MISMATCH`。
- 因此步骤7的GPU计算已经全部完成，但项目完成判据还差一个CPU收尾：修复 `scripts/compare_stage7_mipp_results.py`，重新生成baseline+main+SG95+PG95的同协议比较表并明确近似论文mask边界。该动作无需重跑实验；完成后才能正式勾选步骤7并进入步骤8。
### 步骤7收尾与步骤8启动：用户授权最大并发5
- **状态：** report_fix_then_fusr_implementation
- 用户要求开始下一步，并明确允许把最大并发从3改为5。项目专属 `AGENTS.md` 已按当前明确指令更新为最多5个，同时保留动态显存核验、GPU锁、OOM退避和 `WAITING_RESOURCE`；这不是必须同时跑满5个，也不授权缩小batch、epoch或模型。
- 执行顺序保持研究闭环：先修复步骤7四模型比较器并生成PASS报告、正式勾选步骤7；随后进入步骤8，独立实现默认关闭的FUSR，完成组件测试、默认关闭基线闭环、GPU smoke和checkpoint重载后，才提交正式单seed队列。
- 步骤8正式队列会使用 `MAX_JOBS=5`，但实际并行数由实时显存和FUSR smoke/profile得到的峰值决定。

### 步骤8：FUSR设计与代码接入点恢复
- **状态：** design_audit_in_progress
- 已从正式蓝图恢复FUSR定义：以原型开启/关闭造成的下游预测损失差作为utility（效用）监督，学习测试时只依赖历史、mask与检索统计的选择器；拒用只回退到无原型预测，不拒绝输出预测。
- 当前首轮边界收敛为独立的read gate/abstention：不与PG95组合、不提前实现写入隔离/淘汰、不使用Gate H测试标签训练。需要在训练或验证split生成开/关效用标签，并保证测试目标永不进入路由器输入。
- 下一动作：审计`MTNet.forward`、S4M顶层forward和训练循环，确定最低额外前向次数且可默认关闭的接入方式；同时等待步骤7报告修复完成。

### 步骤8：候选训练/推理路径已确定
- **状态：** implementation_design_pending_independent_review
- 候选实现不是每个训练batch盲目执行三次完整模型：`MTNet`共享query和retrieval，产生on/off上下文；训练时在batch维拼接后通过一次双倍batch的下游S4，得到on/off预测、utility监督和软路由预测；测试时根据历史可得特征先缩放retrieved向量，只走一次下游预测。
- 该路径能同时满足“训练效用标签由目标生成”“测试不泄漏目标”“推理单主干路径”“默认关闭恢复基线”。下一动作是结合独立审计结果冻结接口，然后修改可编辑 `baselines/S4M`、参数入口、runner与组件测试。

### 步骤8：训练循环与checkpoint接入审计
- **状态：** ready_to_freeze_interfaces
- 已确认训练循环可以从模型暴露的on/off预测与router输出计算target-derived utility loss；验证/测试仍走无标签单分支。`change_mode(0)`会禁止memory update，防止评估探针改变原型资产。
- 现有checkpoint已保存完整model state，FUSR作为`MTNet`子模块即可自动保存/加载。需要新增的是FUSR参数键验证、训练机制指标累计和评估门控指标，不需更换run目录协议。
- 下一动作：等待两份独立FUSR审计与步骤7报告修复，交叉检查后开始最小跨文件实现。

### 步骤7：四模型报告闭环完成
- **状态：** complete
- `scripts/compare_stage7_mipp_results.py` 已纳入baseline、main MIPP、SG95、PG95，排除run身份/输出目录/GPU与MIPP开关等允许差异，同时继续严格比较数据、固定mask、seed、S4M主干、训练预算、早停和评估协议。
- 新 `reports/stage7_mipp_comparison.json/.md` 状态为 `PASS`、协议差异0、四组共用固定mask；明确原论文mask seed未公开，未把项目内公平对照包装成原论文同随机mask复现。PG95仍为最佳：MSE提升0.5993%、MAE提升2.4177%。
- 步骤7所有GPU计算、checkpoint、机制指标和报告已满足快速创新验证完成判据，正式关闭。

### 步骤8：调度并发硬上限同步为5
- **状态：** scheduler_patch_pending_self_test
- 除项目规范外，通用队列引擎原先仍硬编码 `--max-jobs <= 3`；现已改为允许1–5。默认值仍保持保守，步骤8启动脚本会显式传5；实际任务数仍由实时显存决定。
- 下一动作：运行队列自测与显式`max_jobs=5`参数门禁，再进入FUSR跨文件实现。
- 第一次为上限增加常量的patch因猜错既有`QUEUE_NAME`字面值而安全失败，没有产生部分修改；已读取真实上下文后改用`MAX_CONCURRENT_JOBS=5`常量，并把该条件加入队列self-test，避免仅改错误消息而未验证实际上限。
- 四模型比较脚本重新执行退出码0、报告`PASS`；队列编译与原有OOM/依赖/恢复self-test也通过。但进一步审计发现`active_gpu_uuids`把同卡共存完全禁止，当前“支持参数5”并不等于能实际并发5。
- 下一动作：把调度选择改为每张GPU每轮最多新增1项、下一轮按真实剩余显存再补位；扩展假任务到峰值并发5并重跑自测。此改造只影响任务选择，不改变任何实验配置。

### 步骤8：最大并发5调度闭环与FUSR实现启动
- **状态：** fusr_implementation_in_progress
- 队列引擎已完成“每张GPU每轮最多新增1项、后续轮询依据真实剩余显存继续同卡补位”的改造；`py_compile`与完整假任务自测均退出0，实测峰值并发为5，动态补位、OOM回队列、确定性错误停止、依赖、恢复和外部终止分类全部PASS。
- 独立设计审计把首版边界冻结为FUSR read gate + abstention（读取门控与无原型回退），不与PG95组合、不修改原型写入、不用Gate H测试标签训练。路由器只使用历史、mask、检索统计和可关闭的频谱不确定性；训练效用标签来自训练样本的原型开/关预测损失差。
- 设计审计补充了训练期关键风险：on/off双分支若使用不同dropout随机掩码，会污染效用标签。实现将采用公共随机数配对或等价的受控双前向，确保损失差主要来自原型干预。
- 下一动作：完成`FUSR.py`、`Bank.py`、`S4M.py`、训练循环、CLI/runner和步骤8配置/队列的最小接入，先执行默认关闭闭环和CPU组件测试，再提交GPU smoke。

### 步骤8：FUSR首版跨文件接入与静态门禁
- **状态：** component_and_default_off_tests_in_progress
- 已新增452参数的`ForecastUtilityRouter`与固定8维无目标泄漏特征；已接入`MTNet`原型读取、S4M共享下游on/off训练路径、训练split效用监督、硬门推理、测试期可选paired diagnostics及checkpoint/metrics验证。未来目标只在训练循环计算detached utility标签，不进入router接口。
- 训练期on/off主干使用保存/恢复CPU与CUDA随机状态的公共随机数，两次前向共享dropout掩码；推理仅走一次硬门控后的S4主干。FUSR关闭时不实例化路由器、不计算FFT或双分支。
- Python编译、`git diff --check`、两份FUSR配置严格校验、shell语法与上游只读检查均退出0；步骤8队列重新自测`PASS`，真实观察峰值并发5，未启动GPU。
- 已发现并修正两个接入一致性问题：组合MIPP时FUSR loss不能被覆盖；训练机制指标必须同时输出runner要求的`fusr_gate_mean`与`fusr_abstention_rate`稳定键。
- 下一动作：运行FUSR组件测试和默认关闭/强制on-off数值闭环；通过后查看实时GPU并提交两epoch完整链路smoke，正式15 epoch任务只依赖smoke成功自动接续。

### 步骤8：组件闭环通过，GPU队列已持久启动
- **状态：** waiting_resource
- `scripts/test_fusr_components.py`本轮实跑退出0：核心16/16、MTNet集成9/9全部PASS；force-on/force-off与FUSR关闭基线路径最大绝对误差均为0.0，未来目标变化不影响gate，router参数有非零有限梯度，checkpoint roundtrip一致，新增参数452。
- 宿主持久调度器PID=`671181`、PPID=1、SID=`671181`，命令显式`--max-jobs 5 --resume`；状态文件、日志和进程三方一致，当前`queue_state=WAITING_RESOURCE`，不是假启动。
- 启动快照：物理0/RTX4080/UUID `GPU-c88...`空闲2500 MiB，物理1/RTX4090/UUID `GPU-a3af...`空闲487 MiB；低于FUSR smoke当前9216 MiB历史估计与安全余量，因此smoke正确保持`WAITING_RESOURCE`，正式任务因依赖保持`PENDING`。
- 队列会每30秒非阻塞轮询，两张卡任一满足条件即原子启动smoke；smoke成功后自动解锁同协议15-epoch正式FUSR。OOM只回队列并更新估计，不修改batch16、epoch、模型或评估协议。
- 下一动作：等待队列自动获得GPU并完成smoke；首次真实GPU链路若为CODE/CONFIG/DATA错误则保留失败run并窄修复，若SUCCESS则无需人工确认直接进入正式单seed对照。

### 步骤8：GPU smoke成功，正式Weather对照运行中
- **状态：** formal_running
- FUSR smoke `runs/s4m_fusr__weather__point__mr0.06__seed2023__smoke__20260718_143548` 已于首次attempt成功：2个optimizer steps、32个train-split utility pairs、checkpoint含7个FUSR参数键且CPU重载通过；峰值allocated/reserved约3070/4070 MiB，无OOM。
- smoke仅验证链路，不作为效果结论；其单测试batch硬门仍为always-on且Oracle空间较大，符合仅训练2步尚未学会拒用的预期，不能用来判断正式FUSR是否有效。
- 队列已依据smoke峰值把正式任务需求从9216 MiB校准为6144 MiB（4070+1536 MiB安全余量并取调度粒度），随后自动启动正式run `runs/s4m_fusr__weather__point__mr0.06__seed2023__20260718_143713`。
- 宿主核验runner PID `2333230`与训练PID `2333426`均存活，训练PID位于4090 UUID `GPU-a3af...`、当前占用约4572 MiB；日志已完成epoch 4并进入epoch 5（iter至少450），epoch 4 validation loss从0.663505降至0.658504，checkpoint继续更新，无Traceback或OOM。
- 当前4090实时空闲8197 MiB（用户观察约7 GiB与采样时刻差异一致），但步骤8首轮只有这一项已通过smoke且同协议的正式主实验；按“先看主改进是否提升，再决定迭代/消融”原则不提交无证据重复任务。调度上限仍为5，后续若进入变体筛选会动态补位。
- 下一动作：不人工中断或持续占用前台；等待正式任务自然早停/完成并自动生成baseline-vs-FUSR比较报告。若主结果无提升，再依据门控率、效用AUROC、HRR与Oracle gap设计下一版最小迭代。

### 步骤8：FUSR V1正式结果完成，但科研判据未通过
- **状态：** v1_complete_mechanism_negative_iteration_required
- 正式run `runs/s4m_fusr__weather__point__mr0.06__seed2023__20260718_143713` 已于首次attempt `SUCCESS`结束，return code 0、8 epochs早停、18288 optimizer steps、checkpoint CPU重载和7个FUSR参数键验证通过；队列整体`SUCCESS`并正常退出。
- `reports/stage8_fusr_comparison.json/.md` 已生成：与步骤5基线协议差异0、共用固定mask。基线MSE/MAE=`0.263525745/0.301159963`；FUSR=`0.263705272/0.297522085`，即MSE变差0.0681%、MAE改善1.2080%，没有同时改善两项。
- 机制判据明确失败：测试hard gate均值1.0、abstention=0，routed MSE严格等于always-on；HRR与harmful-use均47.99%，HRR降低0；utility AUROC=0.4946接近随机。训练sign accuracy 94.78%由97.98%正utility标签失衡造成，不能证明选择器有效。
- Oracle sample MSE=`0.253340914`，相对always-on仍有3.93%空间，说明研究问题仍存在，但当前训练标签/门控方法没有利用该空间。FUSR实际训练比基线多60% steps、总耗时4.91倍，不能用MAE单项改善掩盖机制失效和额外训练代价。
- 负辅助loss为有限的异方差log-variance项，不是NaN/发散；但其均值约-0.151，而末期forecast loss约0.001，且gate prior从约6.197仅降到约6.149，构成路由长期always-on的重要风险。
- 当前结论：不进入多seed、跨数据集或FUSR+PG95组合；步骤8保持未完成，下一轮只修复路由塌缩。最短迭代应先对齐train/validation utility分布、类别平衡与validation阈值校准，让sign/ranking score实际控制gate，并移除或退火固定gate prior；新smoke必须先证明abstention非0且路由输出有限，再同固定资产单seed复测。

### 步骤8：FUSR V2迭代启动
- **状态：** v2_design_and_implementation_in_progress
- 用户已授权继续。V2保持同一步编号、同Weather固定mask和单seed筛选，不提前扩展多seed/ETTh1/PG95组合。
- V2目标冻结为：修复V1的97.98%训练正类失衡、永久gate prior和sign head不参与决策三个根因；先审计train/validation utility标签，再实现类别平衡的sign/ranking监督、固定尺度Smooth L1效用回归、可退火prior及validation-only门控校准。
- 完成顺序：独立设计/代码审计 → 最小跨文件实现 → 默认关闭与组件测试 → 新配置smoke → smoke成功后同协议正式V2；调度器上限保持5，实际任务按显存动态补位。
- 下一动作：并行审计utility标签分布、router接口和validation校准路径，冻结最小V2实现后修改可编辑baseline与独立V2配置/队列。
- 初步候选已收敛到post-hoc frozen-router（后置冻结路由器）路径：复用步骤5baseline权重与memory，冻结主干，一次缓存train/val原型开关效用，tiny router离线平衡训练并以validation observed-target误差校准硬门；正在等待独立审计确认后冻结接口。

### 步骤8：FUSR V2路线冻结前检查
- **状态：** frozen_backbone_router_design_selected
- 两项独立审计均推荐放弃V1的joint training（联合训练），改为加载步骤5 Weather基线checkpoint后冻结backbone/memory（主干/原型库），一次性生成train/validation反事实缓存，离线训练tiny router（小型路由器），并只用validation（验证集）已观测未来值校准阈值。
- 标签审计确认V1的97.98%训练正类并非效用归一化符号错误，而是主干、memory和router同步变化造成的非平稳自举标签；其`gate_prior/T`还让关门条件落到效用裁剪范围之外，结构上几乎必然全开。
- V2主结果只用`seq_y_mask`覆盖的已观测目标生成训练和验证效用，隐藏完整`seq_true_y`只允许用于不参与训练/校准的离线分布诊断；测试目标只做一次最终评价。
- 该路线使always-on（始终使用原型）数值闭环到步骤5基线，避免V1额外主干训练步数与路由收益混在一起。外层非Git和一次`rg`路径展开问题均无代码影响，已转为在两个S4M子仓库核验。
- 下一动作：冻结时间块fit/calibration划分与V2接口，分工修改router、训练循环、runner和独立V2实验队列。
- 标签审计补充量化：冻结步骤5基线的prototype-on测试MSE与原始基线只差`1.57e-8`，有益检索率58.04%、Oracle相对空间5.08%；V1 test效用幅度却仅为该冻结基线的9.91%，进一步支持冻结路线。
- 快速V2不采用随机K折：在validation缓存的前段拟合router，留出至少`seq_len+pred_len=288`个时间索引的embargo（隔离区），再用后段校准阈值；该证据明确标记为stationary diagnostic（固定状态快速诊断），有效后再重做严格嵌套时间划分。

### 步骤8：FUSR V2接口与实验资产落盘
- **状态：** merged_static_validation_in_progress
- V2接口已冻结为`intervention14`目标无关特征、单`sign_logit`效用符号分数、无可学习prior、fit-only特征标准化和validation-only校准阈值；V1默认公式和452参数路径保持兼容。
- 冻结训练分支加载步骤5checkpoint后关闭全部backbone/memory梯度，使用按时间顺序的validation on/off缓存训练router，并保存cache/checkpoint/hash/校准来源证据；测试只在`CALIBRATION_LOCKED`后执行。
- 已新增独立V2 smoke→formal队列和三模型比较器；正式主干、数据与测试协议对步骤5解析后差异为0。smoke的`required_memory_bank_warmups`从2改为0，因为V2加载冻结原型库且禁止重新warm；`des`由长标识缩短为`F2V2S/F2V2F`以预防S4M单目录名超过Linux限制，未改变模型、数据或训练预算。
- 下一动作：执行合并后的组件、默认关闭闭环、CLI/配置/runner门禁与smoke dry-run；全部通过后直接启动独立GPU队列。

### 步骤8：FUSR V2静态门禁通过并启动GPU smoke
- **状态：** v2_smoke_running
- 合并后验证全部PASS：V1组件核心16项+MTNet 9项、V2路由/14维特征30项、时间块切分/平衡排序/阈值校准10项；Python编译、`git diff --check`、两份配置校验、Shell语法、动态并发5/OOM/恢复队列自测均退出0，上游快照洁净。
- 独立队列已在宿主持久启动：scheduler PID=`875291`、PPID=1、SID=`875291`，显式`--max-jobs 5 --resume`。smoke runner PID=`875304`，run=`runs/s4m_fusr_v2__weather__point__mr0.06__seed2023__smoke__20260718_192950`，已绑定4090 UUID并出现对应CUDA进程；状态、PID、attempt log三方一致。
- 启动时4080空闲200 MiB、4090空闲6909 MiB；队列按6144 MiB估计只启动一个smoke，正式任务保持依赖PENDING。没有缩小模型、batch或正式预算。
- 下一动作：等待smoke完成cache→router backward→validation calibration→checkpoint reload→paired evaluation全链路；SUCCESS后队列自动解锁同协议formal，若确定性错误则保留run并窄修复后新attempt。

### 步骤8：FUSR V2 smoke成功，正式筛选自动接续
- **状态：** v2_formal_running
- V2 smoke首次attempt完整`SUCCESS`：return code 0、640个固定validation缓存样本、4个router gradient steps（路由器梯度步）、230个rank pairs（排序对）、`CALIBRATION_LOCKED`、checkpoint含11个FUSR V2状态键且CPU重载通过；backbone/memory哈希前后一致，峰值allocated/reserved约1069/1732 MiB。
- smoke只训练2 epochs且只取40批：校准段选择全开（gate rate=1），测试40批却全部低于阈值而全关，测试AUROC约0.613、routed MSE退化约10.45%。这是小缓存分数分布漂移的机制预警，不是工程错误，也不当作正式结论。
- 队列已自动启动完整formal：runner PID=`889321`，run=`runs/s4m_fusr_v2__weather__point__mr0.06__seed2023__20260718_193121`，4090 UUID上对应CUDA进程PID=`889514`；使用完整validation缓存、80 router epochs上限、patience10、288窗口隔离和完整测试，主干/checkpoint/batch/数据均未缩水。
- 下一动作：正式任务完成后生成baseline+V1+V2报告，重点判断整体MSE/MAE、同checkpoint routed-vs-always-on、测试gate状态、AUROC、HRR和validation→test分数漂移；若仍塌缩或退化，步骤8内继续最小V3迭代而不扩seed/数据集。

### 步骤8：FUSR V2正式负结果闭环并启动V3诊断
- **状态：** v3_temporal_generalization_diagnosis_in_progress
- V2 formal首次attempt `SUCCESS`且artifact/protocol均PASS，但MSE/MAE相对步骤5分别退化约0.0150%/0.0071%；测试gate rate=99.837%、AUROC=0.4744、HRR降低仅0.206%，research decision为`NEGATIVE_NO_MSE_LIFT_ITERATE_WITHIN_STEP8`。
- 最关键证据是时间外排序反转：fit/dev/calibration/test AUROC分别约0.766/0.337/0.379/0.474。V2不是简单阈值未调好，而是router学到了只在早期时间块成立、后续反向或消失的关系；Oracle仍保留5.08%空间。
- V3不先占GPU、不扩seed/数据集：直接读取5175条validation固定反事实缓存，做按时间块效用率、特征漂移、单特征方向、滚动训练/验证和分组稳健候选的离线审计；只有严格后段验证改善的候选才接入新smoke→formal队列。
- 本轮继续使用文件化项目规划；首次恢复命令误写为不存在的`session-catchup.sh`，未修改项目，现改用技能包实际的`session-catchup.py`。
- 下一动作：并行完成缓存统计审计、稳健学习候选审计和V3公平性/叙事审计，冻结一个最小可检验方案。

### 步骤8：FUSR V3零GPU时序诊断恢复执行
- 已按项目文件规划恢复上下文，并确认本轮仍属于步骤8，不新增步骤编号、不扩数据集或seed。
- 本轮完成判据固定为：仅使用已生成的validation反事实缓存，量化时间块标签/特征/效用方向漂移；候选方法必须在严格后段validation审计块相对always-on有非平凡改善，才允许提交GPU smoke。
- 公平性边界：不读取test标签选择模型或阈值，不修改步骤5主干、batch、epoch、mask、seed和评价代码；离线诊断失败也作为正式负证据保留。
- 下一动作：核验缓存与router checkpoint结构，生成可复跑的V3时序诊断脚本和JSON/Markdown报告。

### 步骤8：静态时序稳健候选全部未通过，转向因果历史探针
- 新增并实跑`scripts/analyze_fusr_v3_temporal_generalization.py`，生成`reports/stage8_fusr_v3_temporal_diagnosis.json/.md`；脚本仅读取5175条validation固定缓存，fit拟合、dev锁阈值、最终audit块审计，未读取test目标或占用GPU。
- 8个预声明候选均未通过：V2 checkpoint、pooled linear、stable-feature linear、pooled MLP、worst-group MLP及3种时间块ridge集成在audit块均未取得正MSE增益。最好的`temporal_ridge_median`仍退化约0.0033%，bootstrap 95%绝对增益区间跨零。
- 所有候选在dev块的经验风险最优门控率都是1.0；dev/audit的Oracle相对always-on仅约0.705%/0.520%。这说明继续更换静态分类器或阈值没有足够可识别信号，不能把fit AUC包装成可部署进展。
- V3下一假设改为HCUP（Historical Counterfactual Utility Probe，历史反事实效用探针）：在当前预测原点之前，用已发生历史构造同长度伪预测，直接比较原型on/off历史误差并路由当前预测；当前未来和test未来标签均不可见。
- 下一动作：审计`Dataset_Custom4`的原始索引、mask和test loader顺序，证明可以构造`[t-288,t-96] -> [t-96,t]`历史探针后，再实现默认关闭的最小HCUP与GPU smoke。

### 步骤8：HCUP实现与首次调度修复
- 已确认test loader严格时序且从样本index 96起可构造完整过去探针；前96个样本固定回退prototype-on。新增HCUP独立诊断入口、审计runner、validation smoke/formal配置、因果聚合分析器及最大并发5的持久队列；原baseline/FUSR/MIPP开关默认路径不变。
- Python编译、Shell语法、YAML、diff/upstream只读门禁均退出0；队列自测PASS并覆盖动态补位、依赖、OOM回队列、确定性错误、恢复和并发5。
- 首次宿主机后台启动PID立即退出，日志明确为start脚本漏传必需`--gpu-candidates`；未创建state/run、未启动GPU。已补入4080/4090稳定UUID，下一动作是重启并以PID+state+日志+显存四证据确认真实运行。

### 步骤8：HCUP smoke通过，完整验证已自动接续
- 修复启动包装后，调度器宿主PID `1439086`、独立SID、state和run日志一致。smoke使用物理1/RTX4090/UUID `GPU-a3af...`，未依赖逻辑编号；完整验证复用同一UUID并由smoke峰值加安全余量得到6144 MiB启动估计。
- smoke run `runs/s4m_fusr_v3_hcup__weather__validation_point_r006__seed2023__smoke__20260718_202658`首次attempt `SUCCESS`、return code 0、artifact checks全PASS；128样本中32个探针合法，前96个按协议prototype-on回退，checkpoint未含FUSR/MIPP参数。
- smoke峰值allocated/reserved约1052/1564 MiB；历史探针只用observed past，`probe_uses_current_or_future_target=false`。threshold 0下gate rate=0.875、MSE相对prototype-on改善约2.77%，probe与下一时段真实utility Pearson约0.979；这些只证明机制值得完整验证，不作为正式效果结论。
- 完整validation run `runs/s4m_fusr_v3_hcup__weather__validation_point_r006__seed2023__20260718_202756`已真实运行：queue job、runner与训练子PID分别对齐，4090显存从17307 MiB增至约19349 MiB，状态为RUNNING。
- 完整run结束后队列将自动生成原始汇总，并运行`analyze_hcup_validation.py`：早段validation锁定聚合与阈值、隔离96窗口、后段validation审计；仅当MSE/MAE同时改善且MSE bootstrap 95%下界为正，才允许生成一次性test配置。

### 步骤8：次日恢复并核验HCUP完整验证
- 用户要求确认实验并继续下一步；本轮已通过文件规划恢复步骤8上下文，保持全局编号不变。
- 当前完成判据：先以queue state、宿主PID、run status、日志、metrics和自动分析报告确认完整validation真实终态；PASS则锁定validation选择的因果聚合/阈值并只运行一次test，FAIL则继续步骤8最小迭代且不触碰test调参。

### 步骤8：HCUP完整验证为可信负结果，禁止触碰test
- 队列整体`SUCCESS`，smoke/formal均首次attempt成功且宿主进程已自然退出；formal run `runs/s4m_fusr_v3_hcup__weather__validation_point_r006__seed2023__20260718_202756` return code 0、artifact checks全PASS、5072/5168样本有合法历史探针。
- threshold0全validation的gate rate约0.7146，但routed MSE相对prototype-on退化约1.5226%，历史probe与当前utility Pearson从早期smoke的0.979反转为完整区间-0.132；smoke信号未跨时间泛化。
- 自动validation筛选7种因果聚合全部FAIL：后段audit中direct/EMA候选MSE约退化0.0182%–0.0264%，其余平滑退化为全开；没有候选同时改善MSE、MAE并满足bootstrap 95%下界为正。
- 决策严格执行`HCUP_VALIDATION_NEGATIVE_DO_NOT_TOUCH_TEST`，不生成test配置、不用test标签继续调节。
- 下一动作仍属于步骤8：用现有validation工件做utility时间持续性上界诊断，比较1/4/8/16/32/48/96步滞后；若短滞后信号明显，则把HCUP改为只评估过去最近h步的短时探针并先跑smoke，否则放弃历史反馈式FUSR。

### 步骤8：短滞后效用持续性上界通过
- 新增并实跑`scripts/analyze_hcup_lag_feasibility.py`，输出`reports/stage8_hcup_lag_feasibility.json/.md`；只用formal validation样本，不读取test。
- lag1/4/8/16在严格后段audit的完整历史utility乐观上界均PASS：utility自相关约0.998/0.982/0.947/0.849，MSE相对增益约0.766%/0.729%/0.655%/0.432%，MAE也均改善，MSE bootstrap 95%下界为正。
- lag32虽仍有约0.156% MSE上界，但MAE退化；lag48/64/96进一步衰减或反转。后续只筛1/4/8/16，不浪费资源扩更长lag。
- lag96 observed probe与同一历史窗口complete utility相关约0.979、符号一致率约95.76%，说明上一版失败主要来自96步反馈陈旧，而非观测mask把历史效用估计完全破坏。
- 下一动作：把HCUP接口扩为`probe_horizon<=probe_lag`，lag=h时只用过去已发生的前h个预测点计算observed-history utility；为1/4/8/16分别创建smoke→完整validation依赖队列，仍禁止test。

### 步骤8：短跨度HCUP实现与队列门禁通过
- `run.py`、`exp_pretrain1.py`和`run_hcup_diagnostic.py`已新增`hcup_probe_horizon`，启动门禁强制`0<horizon<=lag`且`horizon<=pred_len`；旧lag96未显式配置时仍默认horizon=lag，兼容既有协议。
- 已为lag1/4/8/16各创建独立smoke与完整validation配置，共8项；每个formal只依赖对应smoke，均复用步骤5冻结checkpoint、Weather固定mask、seed2023、batch16和同一评价资产，不启用MIPP或静态FUSR。
- 新增`run_stage8_hcup_short_queue.py`、原始汇总器和`analyze_hcup_short_validation.py`；最终比较会断言四个run的current on/off逐样本完全一致，再在early validation锁阈值、embargo96、late validation审计，不接触test。
- Python编译、Shell语法、8份YAML资产/路径、因果索引契约、`git diff --check`和upstream只读检查全部PASS；队列self-test再次实际观察并发5，OOM回队列、动态补位、依赖和恢复均PASS。
- 下一动作：记录实时GPU UUID/显存并宿主后台启动8任务队列；资源不足时保持WAITING_RESOURCE，不降低模型、batch或评价协议。

### 步骤8：短跨度HCUP队列已宿主持久启动
- 启动前实时资源：物理0/RTX4080/UUID `GPU-c88b...` 已用14260 MiB、空闲1816 MiB；物理1/RTX4090/UUID `GPU-a3af...` 已用23935 MiB、空闲281 MiB，两卡利用率约98%–99%，均不足以安全启动当前任务。
- 调度器已以宿主PID `419344`、PPID=1、SID=`419344` 持久运行，命令显式使用两张卡的稳定UUID、`--max-jobs 5`和30秒非阻塞轮询；PID、state和启动命令三方一致。
- 队列总状态为`WAITING_RESOURCE`：4个smoke均保留为资源等待，4个formal均保留为依赖PENDING，共8个任务无丢失；当前没有训练子进程，也没有把等待误报为实验已运行。
- 队列会在显存释放后动态启动最多5个smoke；各smoke成功后只解锁对应formal。没有修改模型、batch、训练预算、固定mask、seed或评价协议，也仍未创建或运行test配置。
- 下一动作：等待任一GPU出现满足历史峰值加安全余量的空间；队列会自动补位。下一次检查以宿主PID、queue state、attempt/run日志和GPU进程联合判断真实进度。

### 步骤8：短跨度HCUP全部完成并形成最终负结论
- 队列最终`SUCCESS`且调度器PID `419344`已自然退出；lag1/4/8/16各自的smoke和formal共8个任务全部成功，formal均有5168个validation样本，工件检查PASS。h4首次与另一任务并发时OOM，保留失败历史后按14336 MiB估计等待，第二次在空闲4080上成功；没有修改正式配置。
- 自动后处理首次没有生成最终报告：分析脚本要求跨GPU推理结果逐位完全相等，h4/4080与参考4090最大绝对差仅`6.198883e-6`；同时聚合脚本`check=False`吞掉了子分析退出码。已改为显式数值容差`atol=1e-5, rtol=1e-6`，保留exact/numerical两类证据，并把子分析改为`check=True`。
- 修复后的语法检查、独立分析和完整聚合入口退出码均为0；四个run逐位完全一致为False、数值容差内一致为True，最大差异位于`lag4_current_mse_off`。upstream官方快照仍保持洁净。
- `reports/stage8_hcup_short_validation.json/.md`显示28/28个预声明候选FAIL。direct/EMA/median等选择性规则的audit MSE和MAE均退化；MSE零变化候选的gate rate=1，只是always-on，不满足选择性门禁。
- 最终决策为`SHORT_HCUP_VALIDATION_NEGATIVE_STOP_FEEDBACK_BRANCH`：步骤8完成但科研结果为负；不生成test配置、不用test继续搜索，也不把失败FUSR与PG95强行组合。
- 下一动作：进入步骤9前先重定向使用侧创新。基于现有证据，最短合理路线是围绕步骤7唯一双指标改善的PG95，设计可验证的自适应容量/分位数写入控制或其他与原型供给直接相关的第二创新，而不是继续训练效用路由器。

### 步骤9：PG95机制强化启动
- **目标：** 不组合步骤8失败FUSR，围绕步骤7唯一同时改善MSE与MAE的PG95，先量化其配对一致性与写入门控的真实贡献，再实现可关闭的自适应原型写入控制。
- **快速完成判据：** 新机制有独立开关；默认关闭恢复PG95/基线；组件测试、forward/backward、checkpoint保存/加载和metrics落盘通过；Weather seed2023固定mask与相同训练预算下完成PG95对照。无提升则保留负结果并在步骤9内迭代，不扩多seed/数据集。
- **公平性边界：** 固定Weather point-r0.06数据、seed2023、batch16、15 epoch上限、Adam、patience3、评价脚本和测试mask；不使用FUSR、不读取test选择写入规则、不修改上游只读快照。
- 下一动作：读取PG95 resolved config、MIPP实现、训练日志/metrics和原型库状态，生成零GPU写入分布与收益归因诊断，依据证据冻结首个候选。

### 步骤9：PG95首轮机制证据
- PG95正式run工件确认同协议MSE/MAE=`0.26194645696/0.29387890602`，训练22860步、MIPP配对视图22870次、参数量5975059、峰值allocated/reserved约5036/5622 MiB。
- 固定0.95稳定写入门控的训练平均接受率为99.914%，warmup平均90%；paired content cosine约0.999928。固定阈值没有持续筛掉不稳定候选，PG95收益不能归因于已经成立的“严格净化”。
- 当前结论：步骤9不能只把0.95改成另一个固定阈值。首个候选需提供每批/每阶段自适应的相对稳定性排序和显式容量预算，使写入率可控、跨缺失强度尺度自适应，并保持PG95配对损失与训练预算不变。
- 下一动作：读取MIPP在warmup/forward中生成teacher cosine与eligible mask的确切张量形状，以及FeatureMemory采样/更新顺序；随后冻结无未来目标、可关闭的自适应分位数容量写入接口。

### 步骤9：BMSA首候选冻结
- 源码审计确认每批稳定性张量形状为`[B,Q]`，Weather配置中`B=16`、`Q=seq_len/short_len=6`，约96个候选；`FeatureMemory.add_features_from_sample_learned`在eligible过滤后最多随机抽`M=50`。
- 冻结BMSA规则：启用`budget_topk`时在同一批96候选上按paired-mask cosine取Top-M=50，再交给不变的FeatureMemory；warmup同样按同一预算选Top-M且满足聚类至少21候选。该规则不需要test/validation目标，也不增加写入候选数。
- 公平快速对照只改写入准入模式；PG95的pair invariance权重0.05、mask生成、模型、batch、epoch、optimizer、early stopping和固定评估资产全部不变。首轮只做一个BMSA配置，避免多容量事后搜索。
- 下一动作：实现`absolute/budget_topk`可关闭接口、稳定性分布/有效cutoff/候选计数证据，扩展组件测试和runner门禁，再创建独立smoke→formal队列。

### 步骤9：BMSA核心与CPU组件门禁通过
- `MIPP.py`新增`absolute/budget_topk`候选选择；`Bank.py`在warmup和训练写入复用现有M预算，记录稳定性均值/标准差、原绝对阈值接受率、有效cutoff、eligible/sampled/updated计数；`run.py`强制BMSA预算等于原有`M`。
- `exp_pretrain1.py`与`run_s4m_experiment.py`已扩展resolved evidence和工件验证：BMSA必须显式报告模式、预算及候选计数，eligible/sample均不得超过预算；默认`absolute`仍兼容旧PG95配置。
- 首次裸跑组件测试因测试入口缺少S4/structured_kernels搜索路径而在导入阶段失败，未进入测试、未占GPU；补完整项目PYTHONPATH后退出码0。已进一步把这些只读路径固化进测试脚本，后续无需手工环境变量。
- 新增BMSA断言全部PASS：absolute阈值行为不变，budget_topk从4个分数精确选Top-2，接受率0.5、cutoff约0.8；原有梯度、长度20配对掩码、eligible memory写入检查继续PASS。
- 下一动作：创建BMSA smoke/formal两份同协议配置和独立依赖队列，完成runner配置门禁、默认关闭闭环、队列自测后再检查GPU并启动smoke。

### 步骤9：BMSA启动前静态门禁通过
- 新增smoke/formal配置、两任务依赖队列、start/status/stop脚本、原始汇总器和baseline+PG95+BMSA最终比较器；正式配置仅相对PG95改变`model_id/des`及`mipp_write_gate_mode=budget_topk`、`mipp_write_budget=50`。
- `test_stage9_bmsa_config.py`退出码0：旧PG95、新smoke、新formal均通过runner校验；故意把预算改为49会被拒绝；正式batch16、15 epoch上限、完整batch、Adam、patience3和数据/mask资产未缩减，FUSR关闭。
- 组件测试、Python编译、Shell语法、`git diff --check`和upstream只读检查均PASS。队列self-test退出码0，实际观察最大并发5，并覆盖动态补位、依赖、OOM回队列、确定性错误不重试和恢复。
- 工件隔离已修正：步骤7组件JSON恢复原schema，新BMSA结果写入`reports/stage9_bmsa_component_test.json`。一次并行隔离检查早于报告生成而短暂FileNotFound，顺序复核已`REPORT_ISOLATION_PASS`。
- 下一动作：记录两张GPU的UUID/显存，宿主持久启动BMSA smoke→formal队列；资源不足则保持`WAITING_RESOURCE`，不降低配置。

### 步骤9：BMSA smoke成功并自动启动正式实验
- 启动前物理4080/UUID `GPU-c88b...`空闲1375 MiB，物理4090/UUID `GPU-a3af...`空闲10451 MiB；队列按UUID选择4090，不依赖反向逻辑编号。启动脚本退出码0，scheduler PID=`2439848`、PPID=1、独立SID且state/events一致。
- smoke run `runs/mipp_bmsa50__weather__point__mr0.06__seed2023__smoke__20260720_154634`首次attempt `SUCCESS`，return code 0、2个optimizer steps、2次warmup、checkpoint含7个MIPP键且CPU重载通过；峰值allocated/reserved约5036/5620 MiB。
- smoke机制证据：`budget_topk`预算50，96候选中effective acceptance=52.083%，eligible/sample均为50、平均实际memory update=1.5；同批原absolute-0.95接受率为0，稳定性均值/标准差约0.8766/0.0442、有效cutoff约0.8837。smoke MSE/MAE只用于链路检查，不作研究结论。
- 队列已自动启动完整formal run `runs/mipp_bmsa50__weather__point__mr0.06__seed2023__20260720_154704`：runner PID=`2440979`、训练PID=`2441060`，run status=`RUNNING`；训练PID位于4090 UUID并占用约6122 MiB，日志已完成参数解析、模型加载和epoch1 memory warmup，无Traceback/OOM。
- 正式命令保持batch16、15 epoch上限、Adam、patience3、完整train/val/test、固定Weather mask和seed2023；只改变PG95写入准入模式，不启用FUSR、不增加参数。
- 下一动作：等待正式训练自然早停/完成；队列成功后自动生成`stage9_bmsa_raw_results.*`和baseline+PG95+BMSA比较报告。若MSE/MAE均优于PG95才进入复核，否则在步骤9内继续最小迭代。

### 步骤9：BMSA正式实验宿主侧运行复核
- 调度器PID `2439848`、runner PID `2440979`和训练PID `2441060`均在宿主侧存活，队列与run状态均为`RUNNING`；不是仅依赖PID文件或陈旧状态作出的判断。
- 正式训练已进入epoch 1并到达至少第150次迭代；日志无Traceback/OOM，当前loss已正常输出。训练仍使用Weather固定mask、seed2023、batch16、完整15 epoch上限及`budget_topk/M=50`，没有运行中配置修改。
- 4090稳定UUID `GPU-a3af...`复核时显存约19893/24564 MiB、利用率99%；4080空闲约1375 MiB，不足以安全并发同类完整任务，因此不为凑并发而启动额外重复实验。
- 自动后处理链路已复核：formal成功后先生成`stage9_bmsa_raw_results.*`，随后以子进程退出码门禁运行baseline+PG95+BMSA比较器。
- 下一动作：让正式任务后台自然完成。结束后先判断BMSA相对PG95的MSE和MAE是否双提升；在结果落盘前不声称创新有效，也不提前扩展多seed。

### 步骤9：BMSA单seed快速正式验证完成
- 队列、smoke和formal最终均为`SUCCESS`；formal return code 0、checkpoint CPU重载通过，训练在epoch 9按与PG95相同的patience=3正常早停，调度器和训练进程均自然退出。
- Weather point-r0.06、seed2023同协议结果：baseline MSE/MAE=`0.263525745/0.301159963`，PG95=`0.261946457/0.293878906`，BMSA50=`0.260696429/0.293372790`。
- BMSA相对PG95的MSE/MAE分别改善`0.4772%/0.1722%`，相对baseline分别改善`1.0736%/2.5857%`；参数量与PG95完全相同（5,975,059），峰值显存也基本相同（约5035.74 MiB）。
- 写入机制按设计生效：每批96个候选固定选择Top-50，有效接受率52.0833%；原absolute-0.95门控平均接受99.9046%，进一步证明BMSA改变的是候选选择质量而非扩大写入预算。
- 自动工件与协议门禁`PASS`，决策为`BMSA_SINGLE_SEED_DIRECTIONAL_POSITIVE`。当前只支持“方向性正结果、值得复核”，不能据此声称统计稳定、达到二区或优于原论文全表；固定mask来自当前复现资产，原官方mask seed不可获得的边界继续保留。
- 下一最短路径：固定当前BMSA50不再调参，先补Weather额外seed的PG95/BMSA配对复核；若多数seed同向且统计汇总支持，再扩ETTh1和缺失率，并进入正式消融与复杂度实验。

### 步骤10：Weather配对多seed复核启动
- **目标：** 冻结步骤9的BMSA50，不再依据test结果调容量或阈值；在Weather point-r0.06上用固定seed集合和每seed固定mask，成对比较PG95/BMSA。
- **第一门完成判据：** 所有新增配置只允许改变seed、与seed绑定的训练/测试mask资产及实验身份；PG95与BMSA同一seed必须共享完全相同的数据、训练预算和评价代码；形成逐seed结果、均值/标准差、双指标胜率和配对统计。
- **扩大边界：** 只有配对多seed显示方向稳定，才扩ETTh1、其他缺失率、消融和强基线；若结果混合或反转，则保留负证据并先分析BMSA稳定性，不挑seed美化。
- 下一动作：审计现有缺失资产生成器、seed2023 PG95/BMSA resolved config、队列复用接口和实时GPU资源，预注册新增seed集合后再生成配置。
- 文件化上下文恢复确认：历史正/负结果、GPU UUID映射、队列最大并发5、固定mask公平性和`upstream/S4M`只读边界均继续有效；步骤10不得复用FUSR失败分支，也不得依据新增test结果反向改BMSA预算。
- 资产审计发现启动前阻塞：现有mask生成器的`--seed`没有进入输出文件名，直接生成2024/2025会覆盖seed2023资产。尚未执行生成命令、未覆盖任何文件；下一动作先参数化输出路径并对seed2023哈希不变、seed间路径隔离、同seed成对资产一致性做CPU回归。
- 生成器路径已参数化，seed2024/2025 Weather资产均生成成功；首次资产测试的所有数据断言执行后，在写JSON报告时因`numpy.bool_`不可序列化退出1。该错误只影响测试报告，不影响资产；已增加NumPy标量到Python标量的显式转换，待完整重跑确认。

### 步骤10：Weather三seed复核启动前门禁通过
- 预注册集合固定为seed2023/2024/2025；新增2024/2025资产完整审计`PASS`，三份mask两两不同，同步点缺失、观测值恒等、block length5和metadata/hash均通过，seed2023三个历史哈希未变化。
- 新增seed2024/2025的PG95与BMSA smoke/formal配置共8份；配置门禁全部`PASS`。同seed PG95/BMSA共用完全相同CSV/NPZ/metadata，正式训练均保持batch16、15 epoch上限、Adam、patience3和完整batch；BMSA只改变`budget_topk`与预算50。
- 新增步骤10统一队列、start/status/stop入口、原始汇总器和三seed配对统计器；后者将在全部formal成功后自动报告均值、样本标准差、逐seed胜率、配对差值、bootstrap区间、配对t/Wilcoxon/sign检验，并明确三seed低统计功效边界。
- Python编译、Shell语法、配置公平性、`git diff --check`及上游只读检查均退出0；队列self-test实际观察最大并发5，动态补位、OOM回队列、确定性错误不重试、依赖和恢复均PASS。
- 统计器人工数组单元测试退出0；因测试数组的配对差值被刻意设为完全相同，SciPy发出precision-loss警告，但所有输出有限且断言通过，不影响正式数据。
- 下一动作：宿主侧记录两卡UUID/显存及旧进程状态，后台启动8任务队列；每个formal只在对应smoke成功后解锁，资源不足则持续`WAITING_RESOURCE`且不降低配置。

### 步骤10：Weather多seed队列已启动
- 启动前宿主快照：4080 UUID `GPU-c88b...`空闲14349 MiB，4090 UUID `GPU-a3af...`空闲12986 MiB；未发现旧步骤10调度器或重复任务。
- 启动命令退出0，持久调度器PID=`558518`、PPID=1、独立SID、显式`--max-jobs 5 --resume`。队列状态`RUNNING`。
- 首轮按UUID启动seed2024两项smoke：PG95 runner/training PID=`558526/558752`位于4080目标，BMSA runner/training PID=`558528/558749`位于4090目标；两个run目录均已创建，命令核验seed2024、同一CSV/mask、2-epoch smoke及各自写入模式正确。
- seed2025两项smoke当前`WAITING_RESOURCE`，四项formal为依赖`PENDING`；队列将在显存快照反映首批真实占用后动态补位，不通过缩水配置凑并发。
- 下一动作：确认两项训练日志进入forward/backward并出现GPU上下文；确认后停止人工监控，由队列自动执行smoke→formal→统计报告。

### 步骤10：PG95空准入诊断缺陷与窄修复
- seed2024 BMSA smoke首次attempt `SUCCESS`并解锁formal；seed2024 PG95 smoke在首个训练batch以`CODE_ERROR`停止，对应formal正确进入`BLOCKED_DEPENDENCY`。seed2025两项smoke继续运行，未被中断。
- 根因是步骤9新增的诊断字段对absolute-0.95门控的空选择调用`min()`；这是空候选边界处理缺陷，不是数据、模型梯度或GPU错误。FeatureMemory原本已经支持eligible=0并返回零写入。
- 窄修复保持空eligible不变，只在空集合时把`write_effective_cutoff`记录为配置阈值0.95；BMSA Top-M路径和训练配置不变。新增空absolute gate、零接受率、cutoff和memory零写入回归。
- 下一动作：重跑MIPP/BMSA组件门禁和静态检查；通过后保留失败历史，只重排`weather_pg95_seed2024_smoke/formal`。当前其他成功或运行任务不重跑、不打断。
- 空候选、零写入及原有BMSA Top-M组件回归均`PASS`；配置公平性、Python/Shell语法、diff和上游只读门禁再次退出0。
- 为避免终止当前承载其他实验的调度器，新增`wait_and_resume_stage10_after_empty_gate_fix.py`：等待旧调度器自然退出后读取最终state，只重置PG95中未成功的smoke/formal，并以明确修复原因重启；任何已成功任务保持不变。

### 步骤10：宿主侧运行状态最终确认
- 一次沙箱内`ps`/`/proc`检查无法看到宿主持久进程，曾短暂造成“调度器和训练进程已退出”的错误判断；未据此修改状态、重启或杀进程。随后按项目真值规则改用宿主进程命名空间复核，确认这只是PID命名空间可见性差异，不是实验中断或PID复用。
- 宿主侧调度器PID=`558518`、PPID=1、SID=`558518`仍真实运行；自动修复等待器PID=`573826`、PPID=1、SID=`573826`也仍真实运行。启动脚本再次调用时因检测到同一调度器而安全退出，没有产生重复队列。
- 三个完整formal均真实存活且命令未缩水：BMSA seed2024训练PID=`564017`，BMSA seed2025训练PID=`565287`，PG95 seed2025训练PID=`565291`；均保持batch16、15 epoch上限、完整batch、固定seed/mask及原定PG95/BMSA写入模式。
- seed2024 PG95首次smoke的`CODE_ERROR`失败历史继续保留；当前调度器自然结束后，等待器会只重置并补跑该smoke及其formal，不会重跑三个已成功smoke或打断正在运行的formal。
- 当前尚无新的正式MSE/MAE，`stage10_weather_multiseed_raw_results.*`只是队列运行中的阶段快照，不可作为最终研究结论；最终配对统计只在四个新增formal全部`SUCCESS`后生成。
- 下一动作：停止人工持续监控，让后台队列完成三个formal，并由等待器自动接力seed2024 PG95 smoke→formal；全部成功后自动生成三seed均值/标准差、逐seed胜率和配对统计，再据此决定是否扩展ETTh1与更多缺失率。

### 步骤10：Weather三seed配对复核完成
- 最终队列状态`SUCCESS`，8/8任务成功，调度器和训练进程均自然退出；seed2024 PG95修复等待器按设计只重排失败smoke及依赖formal，handoff返回0。
- 逐seed结果：2023与2024的BMSA均同时改善MSE/MAE；2025同时退化。双指标胜率2/3，但2025退化幅度超过前两次收益。
- 三seed均值：PG95 MSE/MAE=`0.263197057/0.296613489`，BMSA=`0.263352377/0.297461341`；平均逐seed相对增益分别为`-0.0784%/-0.2891%`，即均值方向略差。
- MSE配对t检验p=`0.9397`，MAE p=`0.6467`；两指标bootstrap 95%区间均跨0。n=3下这些检验功效有限，但已经足以否定“当前BMSA50稳定提升、可直接全面扩展”的门槛。
- 协议检查全部PASS；seed2025两方法optimizer steps同为13,716、参数量和显存一致，反转不能归因于训练预算或模型规模不公平。
- 已运行统一`aggregate_results.py`，退出0并汇总68个run；步骤10成功、失败历史和checkpoint路径已进入`reports/latest_results.csv/.md`。
- 当前结论：本轮实验完成，科研结果为可信的不稳定/负均值证据。步骤10总体保持`in_progress`，暂停ETTh1和更多缺失率扩展；下一最短路径是诊断硬Top-M是否降低原型多样性，并实现同预算的稳定性—多样性联合准入快速对照。

### 步骤10：稳定性—多样性联合准入迭代启动
- **目标：** 使用六个已完成正式checkpoint做零GPU原型库诊断，判断BMSA硬Top-M是否降低原型多样性/覆盖度；证据成立后实现预算仍为M=50、无未来标签、可独立关闭的联合准入。
- **完成判据：** 诊断报告可从checkpoint复算；新模式默认关闭时恢复PG95/BMSA；CPU组件与完整GPU生命周期smoke通过；冻结规则后同时对seed2024正向锚点和seed2025失败锚点做同协议正式比较。
- **预注册边界：** 不读取test预测误差选择联合分数或权重；首候选在任何新formal前冻结；不能只在seed2025调到提升而忽略seed2024；M、pair loss、数据、mask、batch、epoch、optimizer、early stopping和评估代码保持不变。
- 下一动作：审计六个checkpoint中memory参数结构、MIPP/Bank写入张量与现有诊断入口，先构建不加载数据集目标的checkpoint比较器。
- 首次源码定位误查不存在的`model/MTNet.py`，仅该只读目标失败；已记录并改用仓库文件清单定位，其他`Bank.py/MIPP.py/runner`搜索结果有效。
- checkpoint序列化审计完成：正式工件包含`memory/repre/num_dic`，足以在CPU上复算原型库结构指标；诊断不会加载数据集或未来目标。下一动作是实现六checkpoint统一比较器与JSON/Markdown报告。
- 运行前冻结5项主诊断：质心最近邻余弦、质心对余弦≥0.95冗余率、中心化质心有效秩比例、库存向量有效秩比例、槽位占用归一化熵；原型数和簇内紧致度仅作辅助，不依据test误差增删指标。
- 新增`scripts/diagnose_stage10_memory_diversity.py`，从步骤10比较报告解析六个正式run及checkpoint，并原子写出JSON/Markdown诊断；尚未执行，下一步先静态编译再运行。
- 诊断脚本静态编译与执行均退出0，六checkpoint报告`PASS`。两个正向seed的BMSA有效秩高于PG95，而失败seed2025显著低于PG95；饱和的质心冗余阈值和槽位熵不用于设计。
- 正式实验前冻结新模式`stability_diverse_rank`：稳定性秩与内容新颖性秩等权贪心，固定M=50，无未来标签、无可学习参数、无可调融合权重。下一动作是定位CLI/runner模式校验和诊断聚合接口，再做最小跨文件实现。
- 模式接入点已定位：`run.py`的CLI choices与预算门禁、`run_s4m_experiment.py`的配置/工件校验、`MIPP.py`选择器、`Bank.py` warm/训练写入调用，以及`exp_pretrain1.py`的训练期指标聚合。现有`budget_topk`回归必须全部保留。
- 已实现`stability_diverse_rank`：旧模式接口默认不读取features；新模式以稳定性最高候选为起点，随后等权融合稳定性百分位秩与相对已选集合的新颖性百分位秩，确定性贪心选满预算。新增候选/已选两两余弦、已选稳定性和多样性增益诊断。
- `Bank.py`只在新模式向选择器传detached normalized content；Memory最终M采样/更新逻辑未改。CLI、runner配置门禁、训练指标聚合和工件校验已接入新模式。
- 新增独立`test_stage10_sdca_components.py`，不覆盖步骤9历史报告；下一动作是静态编译、旧组件回归和新组件测试。
- 静态编译、旧MIPP组件回归和新SDCA组件测试全部退出0；新选择在合成候选上固定选2/4、保留最稳定首项并替换高冗余第二项，重复执行mask与诊断逐值一致。
- 新增seed2024/2025的SDRA50 smoke/formal四份配置；正式设置保持batch16、15 epoch上限、Adam、patience3和完整batch。新增`test_stage10_sdra_configs.py`检查逐字段公平性、资产存在、预算50以及错误预算/缺少门控会被拒绝；尚待执行。
- SDRA配置测试编译与执行退出0，四配置相对各自BMSA只改变实验身份与写入模式；错误预算49和关闭稳定门控均被拒绝。
- 性能审计发现初版贪心每轮`.item()`会制造GPU主机同步；已改为设备端0维索引张量，选择规则与tie-break不变。下一动作是重跑新组件逐值回归并做CPU微基准，再决定是否进入GPU smoke。
- 设备端索引优化后的组件编译/逐值回归仍退出0，选择mask与全部机制诊断不变。
- 冻结双锚点决策门槛：seed2024和seed2025都必须同时改善各自PG95的MSE与MAE，才补跑seed2023；否则停止SDRA，不扩ETTh1/缺失率。BMSA只作次级参照。
- 新增4任务统一队列、原始聚合器、PG95/BMSA/SDRA双锚点比较器及start/status/stop入口；formal依赖同seed smoke，最多并发5且按实时显存补位。下一动作是静态门禁、队列自测和比较器合成单元测试。
- 启动前门禁全部退出0：新Python文件编译、3个Shell脚本语法、SDRA配置测试、冻结决策合成测试、`git diff --check`和上游只读洁净检查均PASS。
- 队列self-test为`PASS`，实际最大并发5，并覆盖动态补位、OOM回队列、确定性错误不重试、依赖顺序、resume与陈旧RUNNING分类。下一动作：记录宿主GPU UUID/显存并持久启动smoke→formal队列。
- 宿主启动命令成功，持久调度器PID=`1149232`、PPID=1、独立SID，队列最大并发5。seed2024与seed2025两个GPU smoke均`SUCCESS`，各自完成2个训练batch、optimizer step、checkpoint保存/重载、评估与metrics落盘。
- seed2024 smoke确认新模式真实执行：正式写入候选两两余弦均值`0.8883517`，已选集合`0.8851728`，多样性增益`0.0031790`；写入模式、预算50、候选/已选诊断字段均通过runner工件门禁。
- seed2024 formal以完整batch16、15 epoch、全量batch在4090 UUID `GPU-a3af...`真实运行，宿主父/子PID=`1151642/1151694`；按PID查询实测训练进程显存约6122 MiB。
- seed2025 formal首次启动与seed2024 formal落在同一4090；第二次启动时nvidia-smi尚未体现第一个任务CUDA峰值，导致其在首batch OOM。失败run状态准确保存为`OOM`并转`RETRYING`，配置没有缩减，等待空闲显存比上次更多后重试。
- 已修复通用队列的启动竞态：按GPU UUID和训练PID读取实际显存，只从实时free中额外扣除每个运行任务尚未兑现的launch reservation。静态编译、宿主进程显存查询和含5并发/OOM/依赖/恢复/竞态的新self-test全部通过。当前旧调度进程不会热加载补丁，但现有OOM回队列足以完成仅剩任务；补丁用于后续启动或恢复。
- 下一动作：不持续人工监控。后台先完成seed2024 formal；seed2025在满足OOM重试资源条件后自动重跑。两项均SUCCESS后自动生成PG95/BMSA/SDRA双锚点比较，并严格应用“两个seed都双指标优于PG95才补seed2023”的冻结门槛。

### 步骤11：缺失形态均衡原型记忆迭代启动
- **目标：** 先完整收口SDRA负结果，再仅用训练/验证信息检验“内容特征塌缩后，原型库缺少缺失形态覆盖”这一新假设；证据支持才实现MRBA（Mask-Regime Balanced Admission，缺失形态均衡准入）。
- **完成判据：** SDRA协议、机制和精度结论分别落盘；缺失形态诊断不使用test目标选择规则；MRBA保持M=50及训练协议不变、可独立关闭；组件测试与GPU smoke通过后才进入seed2024/2025 formal。
- SDRA队列最终4/4 `SUCCESS`，seed2025第一次OOM后未降配、自动换至4080第二次成功。SDRA在seed2024相对PG95的MSE/MAE分别退化0.5184%/0.2718%；seed2025的MSE改善0.2662%，但MAE退化2.7434%，两个锚点均未双指标胜出。
- 正式内容特征平均两两余弦达到0.9992–0.9998，SDRA多样性增益仅seed2024 `-6.60e-6`、seed2025 `+3.33e-6`；继续调内容空间多样性权重缺乏可信信号，停止SDRA。
- 自动比较器把机制正增益混入协议/工件门禁，机制失败时在写最终报告前抛错；原始metrics不受影响。下一动作是分离三类门禁并为负结果生成可审计报告。
- 步骤11.1报告器修复完成：协议检查、模式执行检查、多样性机制检查和精度决策已分离；机制失败不再阻止负结果落盘。静态编译与扩展决策单测退出0。
- 正式报告为`artifact_status=PASS`、`protocol_status=PASS`、`execution_status=PASS`、`mechanism_status=FAIL`；失败项仅为seed2024的正多样性增益和选后相似度下降，协议/执行无失败项。
- 首次回归暴露辅助函数插入位置错误和历史PG95缺少新预算字段；均在测试阶段修复。历史PG95以落盘`M=50`作为等价有效预算，没有更改或重跑任何实验。
- 新生成`reports/stage10_sdra_anchor_comparison.json/.md`，决策保持`SDRA_TWO_ANCHOR_GATE_FAILED_STOP_CANDIDATE`。下一动作：审计validation数据接口、固定mask、已有缓存与checkpoint，确定步骤11.2最低成本诊断路径。
- 步骤11.2确认Dataset_Custom4的历史mask语义及写入块映射：Weather掩码按时间行共享到21变量，每个192历史精确对应6个长度32的写入候选块；诊断无需加载完整数值或未来目标。
- 新增并两次运行`diagnose_stage11_mask_regime_geometry.py`，静态编译和工件检查退出0，两次JSON/Markdown SHA-256完全一致。seed2024/2025训练候选各219600个，独特32位形态7946/8201种，有效秩比例约0.761，平均候选Hamming距离约0.369/0.378。
- validation同样保持约0.758–0.763有效秩；到训练候选的最近Hamming均值仅0.0197–0.0229，说明mask形态具有明显动态范围且validation并非不可覆盖的分布外。该证据只支持快速测试mask-space admission，不声称预测必然改善。
- 冻结MRBA首版为`mask_regime_balanced_rank`：首项取最高稳定性；其后等权融合稳定性百分位秩与历史mask描述子的边际facility-location覆盖增益百分位秩，贪心选满M=50。相较Sinkhorn首版不新增温度、迭代次数、簇数或可学习参数。
- 预注册机制门：seed2024/2025都必须使选中集合对候选mask的平均最近覆盖距离低于同批纯稳定性Top-M；精度门仍为两个seed都同时改善各自PG95的MSE与MAE。任一门失败即停止MRBA，不补seed2023。
- 步骤11.3已实现可关闭的`mask_regime_balanced_rank`，贯通MIPP、Bank、S4M、训练聚合、CLI、统一runner和metrics校验；默认关闭时保留原PG95/BMSA行为。组件测试、旧模式回归、配置公平性测试、比较器正负决策测试、静态编译和通用队列self-test均退出0。
- 新增4份冻结配置、MRBA双锚点队列/聚合/比较/status/start/stop脚本。正式配置继续使用Weather point-r0.06、固定seed mask、batch16、15 epoch上限、patience3、Adam、M=50和完整train/eval batches；相对各自PG95只改变写入准入模式，没有改训练预算或测试协议。
- 启动前两次runner `--dry-run`检查因该入口不支持此参数而在argparse阶段退出2，未创建run、未初始化CUDA、未改变配置。独立配置测试已经覆盖同等门禁，因此没有为了测试便利修改正式runner接口。
- 步骤11.4后台队列PID=`3615242`启动成功，最大并发配置5但按资源实际并发2。seed2024/2025两个GPU smoke均`SUCCESS`，各自完成forward/backward/optimizer step、checkpoint保存/加载、评估和metrics写入，随后依赖formal自动补位。
- smoke机制门方向正确：seed2024训练阶段mask覆盖距离由纯稳定性Top-M的`0.0494133`降至MRBA的`0.0291047`，增益`0.0203086`；seed2025由`0.0535705`降至`0.0314265`，增益`0.0221439`。smoke仅证明机制和工程链路，不用于正式精度结论。
- 两个formal现均为`RUNNING`：seed2024 run为`mipp_mrba50__weather__point__mr0.06__seed2024__20260722_143257`，训练PID=`3618162`，4090 UUID `GPU-a3af...`实测约6122 MiB；seed2025 run为`...seed2025__20260722_143257`，训练PID=`3618172`，4080 UUID `GPU-c88b...`实测约5948 MiB。配置为完整batch16/15 epoch/全量batch，未继承smoke缩水参数。
- 下一动作：按用户要求不持续人工监控，等待两个formal结束。队列完成后自动聚合并严格执行“双seed均相对PG95双指标改善且coverage机制均为正”门禁；只有通过才补seed2023，否则停止MRBA并保留负结果。

### 步骤11：MRBA双锚点正式结果与停止决策
- 队列最终4/4 `SUCCESS`，后台调度器正常退出；两个formal的checkpoint、resolved config、metrics、status和总表条目均落盘。工件、协议、模式执行和参数量公平性检查全部`PASS`。
- seed2024：PG95 MSE/MAE=`0.271194641/0.300850302`，MRBA=`0.269131441/0.299982510`，相对改善`0.7608%/0.2884%`，双指标胜出。
- seed2025：PG95 MSE/MAE=`0.256450072/0.295111260`，MRBA=`0.260133536/0.301007107`，相对变化`-1.4363%/-1.9978%`，双指标均退化。
- 两锚点简单均值：PG95 MSE/MAE=`0.263822357/0.297980781`，MRBA=`0.264632489/0.300494808`；MRBA平均相对变化为MSE `-0.3071%`、MAE `-0.8437%`。
- 覆盖机制在两个seed均按设计生效：coverage gain分别为`0.021153716/0.021150085`，机制检查`PASS`；MRBA相对BMSA的MSE在两个seed都仅有极小改善，但MAE均更差，说明“提高mask覆盖”没有稳定转化为整体点误差收益。
- 正式决策为`MRBA_TWO_ANCHOR_GATE_FAILED_STOP_CANDIDATE`：不补seed2023、不扩数据集、不依据test调稳定性/覆盖权重，也不把单个正seed包装成有效创新。
- 步骤11至此完整结束。下一最短路径是步骤12的validation-only逐样本错位诊断：只有两个seed都显示MRBA在特定可解释mask regime稳定获益，才重新预注册掩码条件检索；否则终止mask-space分支并保留PG95作为当前最可信主线。

### 步骤12：机制—精度错位诊断启动
- **目标：** 使用冻结的PG95/MRBA seed2024/2025 checkpoint，只在validation上比较逐样本误差与历史缺失形态；不读取test逐样本结果、不训练模型、不回改步骤11规则。
- **主变量冻结：** 将192步历史切成六个32步候选块；用对应seed训练候选块的频率估计validation样本的平均平滑负对数频率，定义train-derived mask regime rarity（训练派生缺失形态稀有度）。
- **预注册通过门：** 对每个seed独立按rarity四分位分组；最稀有四分位的MRBA相对PG95平均MSE和MAE收益都必须为正，且两个收益都必须大于最常见四分位。seed2024/2025均通过才允许设计掩码条件检索。
- 缺失率、mask转换率、最长连续缺失段、PG95误差尾部和相关系数只作为解释性指标；不能在主门失败后从中挑一个正分组继续该方向。
- **完成判据：** checkpoint与validation资产配对核验PASS；逐样本顺序和有效元素计数一致；聚合误差可由逐样本文件重算；双seed报告与继续/停止决定落盘。
- 步骤12.2源码审计确认`Exp_Pretrain1.vali()`按validation loader顺序执行，但只在`seq_y_mask==1`的已观测未来位置计算early-stopping损失；`seq_true_y`完整验证目标同时由同一Dataset_Custom4返回。
- 主门口径在查看样本结果前进一步冻结为complete-target validation MSE/MAE（完整验证目标均方/绝对误差），因为它与步骤11完整测试指标的科学问题一致；observed-target误差同步保存，只用于检查结论是否依赖验证early-stopping口径，不替代主门。
- 现有HCUP/Gate-H诊断已验证冻结checkpoint加载、顺序validation推理、NPZ临时文件原子替换和工件记录，可复用其工程模式；但其on/off两臂语义不适合PG95/MRBA跨checkpoint配对，因此新增独立默认关闭的stage12诊断入口，不能修改原`vali/test`行为。
- 公平性审计发现通用`data_provider(flag='val')`默认`shuffle=True`；两个独立checkpoint若直接复用该loader，逐样本顺序不可安全配对。新入口必须从同一validation dataset显式重建`shuffle=False, drop_last=True` DataLoader，并保存`sample_index`与每个历史mask的确定性指纹。
- 合并分析器的前置硬门固定为：同seed PG95/MRBA的sample index、mask fingerprint、完整目标元素计数、已观测目标元素计数和样本数逐项相等；任一不一致即工件失败，不计算分组收益。
- 步骤11的mask资产脚本已提供行级固定mask与train/validation边界；步骤12稀有度可直接复用同一seed的`row_mask`和六个32步候选块定义，无需加载test mask或数值。
- `Exp_Pretrain1.load_model()`会同时恢复model state、原始memory向量、`repre`质心和`num_dic`槽位计数；因此冻结复评不会因遗漏原型库状态而改变PG95/MRBA行为。
- Weather完整validation样本的预测目标固定为`96×21`，但已观测未来元素计数随mask变化。缓存schema冻结为每样本`complete/observed SSE、SAE、count`，聚合器以误差和除以元素总数精确重算MSE/MAE，禁止对可变计数的样本均值简单平均。
- 新入口只做一次prototype-on原模型前向，不复用HCUP的on/off干预；保存packed历史row mask、显式sample index和完整/已观测两套误差，不保存原始数值或测试样本。
- 历史PG95 resolved config生成于显式`mipp_write_gate_mode`字段加入之前，实际parser默认与训练行为为`absolute`。stage12配置验证仅对此历史字段缺失回退到`absolute`；MRBA继续要求显式`mask_regime_balanced_rank`与M=50。
- 步骤12.3已新增默认关闭的`stage12_validation_diagnostic`、冻结缓存runner、2份seed2024 smoke配置和4份双seed完整validation配置；训练、原`vali()`和`test()`路径未修改。
- 静态编译、上游洁净检查和可编辑基线`git diff --check`均退出0。配置测试报告`reports/stage12_validation_config_test.json`为`PASS`：6份配置的资产、seed、方法模式、同seed数据、完整validation预算及smoke/formal身份全部通过。
- 配置负向测试确认`split=test`和把MRBA来源伪装成`budget_topk`都会被启动前拒绝；因此步骤12不会静默读test或错配checkpoint。
- 步骤12.4新增固定分析器、正负决策单测、统一队列、聚合器及start/status/stop脚本。分析器先做逐样本index、packed mask、mask资产、complete/observed count和方法身份硬配对，再执行预注册rarity Q1/Q4门。
- 分析器单测`PASS`：等样本数四分位、按元素计数加权、全门通过正例，以及“MSE改善但MAE失败必须停止”的负例均符合冻结规则。
- 通用队列self-test `PASS`，最大观测并发5，并验证依赖、动态补位、OOM单次回队列后成功、确定性数据错误不重试、resume保留成功结果、EXTERNAL_KILL分类和未兑现显存预留。
- 正式队列定义为2个seed2024两批smoke + 4个PG95/MRBA双seed完整validation cache；所有formal共同依赖两个smoke成功，单任务显存估计6144 MiB，不修改batch或模型。
- 步骤12.5首轮两个smoke均在GPU前向前报`NameError: exp is not defined`，状态正确分类为`CODE_ERROR`，四个formal均`BLOCKED_DEPENDENCY`且没有启动。根因是新增run.py诊断dispatch漏写现有HCUP分支已有的`exp = Exp(args)`。
- 修复范围冻结为补一行实验实例化，并让start脚本接受显式`RESET_JOBS/RESET_REASON`；两个失败run继续保留，不覆盖、不冒充smoke成功。修复后必须重跑静态/队列门禁，再reset本队列任务。
- 修复后`py_compile`、Shell语法、6配置门禁、分析器正负单测和`git diff --check`再次全部退出0。启动脚本同时支持位置参数`<reset_jobs> <reset_reason>`，本轮用`all`重置两个CODE_ERROR smoke与四个从未运行的BLOCKED_DEPENDENCY formal；历史失败run不删除。
- 第二轮两个smoke均完成模型构造和checkpoint加载后，在row-shared断言触发PyTorch 2.1兼容错误：当前版本`torch.all`不接受tuple维度。缓存和metrics均未写，formal仍未启动。
- 修复为把每样本比较张量展平后执行单维`all(dim=1)`；不改变row-shared定义、模型前向或任何误差口径。按三次失败协议，补当前环境张量回归与全套静态门禁后再进行第三次重排；若仍出现同一阻塞链则停止并重新设计入口。
- 第三轮两个seed2024 smoke均`SUCCESS`，各处理2批/32个连续validation样本；PG95/MRBA历史mask SHA-256同为`37820ed...`，`chronological_loader=true`、`uses_test_split=false`、metric roundtrip和checkpoint检查全部PASS。
- smoke峰值GPU allocated约1050.32 MiB、reserved约1564 MiB；smoke只证明冻结推理、缓存、原子写入与配对工件链，不用于步骤12科学门禁。
- 两个smoke成功后四个formal自动解锁。队列先在4080/4090启动MRBA seed2024/2025完整validation缓存，PG95两项处于`WAITING_RESOURCE`，资源释放后自动补位；无配置缩减。

## 2026-07-22 第12步完成：冻结验证集错位诊断

- 队列终态：`SUCCESS`，2/2 smoke 与 4/4 formal 全部成功；每个正式缓存含 5168 个按时间顺序排列的 validation 样本。
- 完整 validation 指标：seed2024 PG95/MRBA MSE `0.521840304/0.538541432`、MAE `0.367439541/0.378900841`；seed2025 PG95/MRBA MSE `0.530318582/0.532414822`、MAE `0.366724013/0.370550085`。
- 预注册主门失败：seed2024 稀有 Q4 的 MSE/MAE gain 为 `-0.015126771/-0.013119114`；seed2025 为 `-0.012769293/-0.016037642`，均未改善且未优于常见 Q1。
- 自动判定：`STAGE12_RARITY_GATE_FAILED_TERMINATE_MASK_SPACE`。按预注册规则终止 mask-space（缺失掩码空间）分支，不从次级分组挑选正结果。
- 报告：`reports/stage12_validation_raw_results.{json,md}` 与 `reports/stage12_validation_mismatch.{json,md}`。

## 2026-07-22 第13步启动：CFR反事实后悔正则

- 已冻结首轮方法：保留PG95的MIPP、绝对0.95写入门、M=50预算和全部训练/数据/评估协议；仅在训练期增加prototype-on/off共享主干配对与非负后悔项。
- 损失固定为 `forecast_mse + mean(relu(sample_mse_on - stopgrad(sample_mse_off)))`。off分支只作为不可作弊的参考，不能接收梯度；评估/测试只跑prototype-on单路。
- 首轮锚点固定Weather seed2023，与已有PG95 seed2023正式结果同协议比较；先组件测试和smoke，正式双指标改善且validation有害检索机制下降才扩第二seed。
- 源码审计确认现有FUSR路径已经实现同一次memory更新下的on/off context和共同随机数预测，可抽出独立CFR路径，无需重复调用整个模型或引入FUSR路由器。
- 第13步核心接口已实现并保持默认关闭：`run.py`、`Bank.py`、`S4M.py`、`exp_pretrain1.py`和审计runner静态编译、diff检查全部通过。
- CPU组件报告`reports/stage13_cfr_component_test.json`为`PASS`：精确数值、逐样本mask计数、只惩罚正后悔、on分支梯度和off分支stop-gradient全部成立。
- 配置报告`reports/stage13_cfr_config_test.json`为`PASS`：正式CFR相对历史PG95只改变身份字段和`use_cfr`开关；训练预算、数据资产和固定mask无差异，非法身份/无MIPP/与FUSR叠加均会被拒绝。
- 队列启动前GPU快照：物理0号4080空闲14379 MiB，物理1号4090空闲20354 MiB；两项smoke具备并行条件。
- 第一次在隔离执行上下文调用后台脚本只产生命名空间PID=11，未创建state/run且随后该上下文无法通信GPU；未启动任何实验。保持配置不变，改由宿主持久会话重启。
- 宿主队列PID `589318` 正常运行。PG95默认关闭回归smoke与PG95+CFR smoke均在约30秒内`SUCCESS`，checkpoint/MIPP/CFR工件验证通过。
- 两项smoke成功后，Weather seed2023完整正式CFR实验于`2026-07-22T15:09:14Z`自动解锁并在物理0号4080启动，run为`runs/mipp_pg95_cfr__weather__point__mr0.06__seed2023__20260722_230914`；未缩减配置。

## 2026-07-23 第13步完成：CFR正式负结果收口

- 队列终态`SUCCESS`，PG95默认关闭回归smoke、CFR smoke、CFR formal共3/3成功；正式checkpoint重载、MIPP/CFR执行证据、固定mask和公平协议检查全部PASS。
- Weather seed2023正式结果：PG95 MSE/MAE=`0.261946457/0.293878906`，PG95+CFR=`0.263401212/0.298090090`；CFR相对变化为MSE `-0.5554%`、MAE `-1.4330%`。
- CFR完整执行13716个optimizer step、219456个样本配对；训练期有害检索率`0.0441045`、正后悔均值`0.00030373`、signed regret `-0.00738859`，说明原型总体有益且惩罚仅覆盖少量样本，但该干预仍损害测试泛化。
- 自动决策`CFR_ACCURACY_GATE_FAILED_STOP_OR_MINIMAL_REVIEW`；结合触发稀少且无精度收益的机制证据，本项目将其收口为停止候选：不做validation机制诊断、不扩seed2024/2025、不依据test调整正则权重。
- 报告：`reports/stage13_cfr_comparison.{json,md}`与`reports/stage13_cfr_raw_results.{json,md}`。
- 正式运行耗时`4543.38s`（约75.7分钟），峰值allocated显存`5444.46 MiB`、reserved `6996 MiB`，参数量`5,975,059`，相对PG95未新增推理参数。
- 已重新运行统一结果汇总器，90个run写入`reports/latest_results.{csv,md}`；第13步正式run现已进入总表，指标、耗时、显存、参数量、commit、配置与checkpoint路径均可追溯。

## 2026-07-23 第14步启动：跨掩码预测分歧风险诊断

- 不继续调整CFR。回到PG95 seed2024/2025冻结checkpoint，只使用validation历史与完整validation目标进行逐样本风险诊断；不训练、不加载test样本。
- 主信号固定为：对原历史额外施加目标新增缺失率15%、长度5因果块的确定性嵌套遮挡后，PG95预测相对原预测的逐样本均方差异；起点概率按块覆盖公式推导，不把15%误当起点概率。
- 主门在查看结果前固定：两个seed各自的分歧—完整目标MSE Spearman均≥0.15，最高分歧Q4/最低Q1的MSE比均≥1.25、MAE比均≥1.10。prototype-on/off差异和检索统计只作次级解释，不能救主门。
- 源码审计确认步骤12冻结validation工程可复用，但须新增独立步骤14入口和schema，避免修改或覆盖历史步骤12工件。三路推理将共享同一checkpoint和validation batch，始终关闭memory更新。
- 已实现独立步骤14入口、审计runner、seed2024 smoke及seed2024/2025 formal配置；静态编译、`git diff --check`和上游洁净检查通过。
- CPU组件报告`reports/stage14_uncertainty_component_test.json`为`PASS`：相同seed逐位确定、不同seed改变扰动、nested mask严格为原mask子集、行共享成立，实测新增缺失率15.158%。
- 配置报告`reports/stage14_uncertainty_config_test.json`为`PASS`：validation-only、15%/L5/seed身份、完整formal预算和全部资产通过；test split、20%遮挡、错误nested seed及非PG95身份均被拒绝。
- 步骤14分析器单测首次运行时，预期失败的合成负例仍越过全部冻结门槛（MSE比`1.32`、MAE比`1.1489`、Spearman `0.9188`）。根因是测试夹具不够弱，不是分析器或门槛实现错误；仅把负例误差范围收窄为`0.25–0.29`，不修改预注册阈值、正式配置或分析逻辑。
- 修正测试夹具的第一次补丁因变量名判断错误而安全失败，未修改任何文件；读取精确源码后按真实变量`mae_fail_error`修正，不再猜测上下文。
- 修正后步骤14分析器单测`PASS`：正例三门全过；弱效应负例虽有Spearman=`1.0`，但Q4/Q1 MSE=`1.12085`、MAE=`1.05870`，因此按效应量硬门正确停止。
- 步骤14组件、配置、分析器单测，全部相关Python编译、Shell语法、可编辑基线`git diff --check`与上游洁净性联合检查退出码`0`；允许进入GPU smoke队列。
- 启动前宿主GPU快照显示4080/4090分别空闲`16053/24206 MiB`且利用率0%；步骤14旧scheduler和state均不存在。
- 首次向`findings.md`追加启动证据时因假定了不存在的步骤14标题而安全失败，未改动文件；读取真实尾部后改为在既有“步骤14冻结诊断接口审计”末尾精确追加。
- 步骤14宿主队列PID=`2453990`。seed2024 smoke于`runs/stage14_uncertainty_pg95__weather__validation__seed2024__smoke__20260723_152128`成功，随后两个formal于`2026-07-23 15:21:43`自动并发启动。
- seed2024 formal在物理0号4080运行（PID `2455030`，检查时约`1866 MiB`），seed2025 formal在物理1号4090运行（PID `2455031`，约`2040 MiB`）；未修改配置、batch或validation预算。
- 步骤14队列终态`SUCCESS`：1/1 smoke与2/2 formal全部成功；每个formal保存5168个按时间顺序排列的validation样本，工件完整性、冻结checkpoint、协议身份、有限数值和validation-only检查全部PASS。
- 双seed预注册风险门全部通过：seed2024 Spearman/MSE比/MAE比=`0.20878/1.28921/1.18002`；seed2025=`0.20743/1.84559/1.21206`。自动决定为`STAGE14_UNCERTAINTY_GATE_PASS_FREEZE_PERTURBATION_MODULE`。
- 步骤14.5新增独立MSE恒等式审计`reports/stage14_equal_ensemble_identity.{json,md}`，编译与执行退出码0。等权双视图validation MSE在seed2024/2025相对PG95分别改善`2.3693%/2.0569%`，相对最佳单路仍改善`1.8032%/1.6403%`。
- 现有缓存无法重算等权集成MAE；下一动作冻结为权重0.5的validation-only补测，直接保存ensemble SSE/SAE，不搜索权重、不读取test、不训练。
- 步骤14.5补测实现边界已冻结：复用同一三路诊断前向，在内存中增加固定`0.5*(prediction_on+prediction_nested)`并保存ensemble SSE/SAE；新增独立方法身份、run前缀和runtime队列，不重置或覆盖原步骤14成功state。
- 步骤14.6静态准入全部通过：新旧配置门禁、嵌套mask组件、风险分析器正负例、等权分析器正负例、全部相关Python编译、Shell语法、`git diff --check`与上游洁净检查退出码0。
- 独立等权队列self-test `PASS`，最大观测并发5，并覆盖依赖、动态补位、OOM回队列、确定性错误停止、resume和未兑现显存预留。允许进入新schema GPU smoke。
- 步骤14.7启动前4080空闲`16053 MiB`；4090有外部作业占用约`9519 MiB`、仍空闲`14697 MiB`。独立队列PID=`2478162`。
- seed2024等权schema smoke于`runs/stage14_equal_ensemble_pg95__weather__validation__seed2024__smoke__20260723_153106`成功；两个formal随后自动解锁并发运行，未修改模型或评估预算。
- 步骤14.7队列终态`SUCCESS`，1/1 smoke和2/2 formal全部成功，自动分析决策`STAGE14_EQUAL_ENSEMBLE_GATE_PASS_IMPLEMENT_TEST_MODULE`。
- seed2024等权validation MSE/MAE=`0.509476549/0.364180358`，相对PG95改善`2.3693%/0.8870%`；seed2025=`0.519410617/0.361909157`，改善`2.0569%/1.3129%`。双seed全部工件、固定权重、validation-only和恒等式复核PASS。
- 步骤14至此完成。步骤15开始：实现默认关闭的PG95固定嵌套缺失等权推理模块，并在固定test资产上做seed2024/2025正式对照。
- 步骤15.1已实现默认关闭的`stage15_nested_ensemble_test`入口、固定0.5/0.5 CNM-SE前向、逐样本test审计缓存、三份配置、独立队列/聚合器/决策器。
- 步骤15启动前静态门禁全部PASS：新配置的test-only/固定权重/完整预算及负测、双seed决策正负例、历史步骤14配置回归、Python编译、Shell语法、`git diff --check`和上游洁净检查均退出0。
- 步骤15队列self-test PASS，最大观测并发5；smoke会同时输出原始PG95路和CNM-SE路，可核验默认基线预测未被新模块代码破坏。
- 步骤15.2启动前4080空闲`16053 MiB`，4090有外部作业占用后仍空闲`14657 MiB`；宿主队列PID=`2500930`。
- 启动后首次宿主轮询因平台执行额度限制被审批层拒绝，未执行任何命令，也不是实验失败；不绕过该限制，后续仅读取项目内持久化state/log判断进度，后台队列不受影响。
- 项目内state确认seed2024 smoke在`runs/cnm_se_pg95__weather__test__seed2024__smoke__20260723_154012`成功；两个formal于`15:40:27`自动并发启动，run分别为seed2024/2025同时间戳目录。
- 步骤15队列终态`SUCCESS`，1/1 smoke与2/2 formal全部成功；同run PG95路与历史source metrics在`1e-5`内匹配，固定test资产、checkpoint、0.5/0.5权重、无更新和逐样本metric roundtrip全部PASS。
- seed2024 CNM-SE相对PG95的MSE/MAE改善`2.4464%/1.1053%`；seed2025改善`1.3032%/0.3359%`。两个seed均双指标改善；平均改善`1.8748%/0.7206%`，预注册门通过。
- 步骤15自动决策`CNM_SE_TWO_SEED_TEST_GATE_PASS_EXPAND`。下一步为步骤16：补Weather seed2023形成三seed正式结果，再运行结构消融与复杂度对照。
- 步骤16.1已新增seed2023正式配置与三seed分析器，并把任务作为现有步骤15队列的新依赖后formal；旧seed2024/2025与smoke状态将由resume保留，不重复运行。
- seed2023资产、三seed完整预算、固定test/0.5权重、编译、决策负测、队列resume/OOM自测、diff和上游洁净性全部PASS。
- 统一结果汇总已重跑，`reports/latest_results.{csv,md}`现包含99个run，步骤14/15正式工件已进入总表。
- 步骤16 seed2023启动尚未执行：同一轮宿主工具此前已明确返回平台执行额度限制，并要求不得绕过或重复尝试；代码、配置和resume队列均已就绪，待宿主执行权限可用后只会新增seed2023任务，不会重跑既有成功项。

## 2026-07-23 步骤16续跑

- 用户明确要求完成seed2023三seed复核，并在通过后立即进入CNM-SE结构消融。
- 本轮按`planning-with-files-zh`恢复`task_plan.md`、`progress.md`与`findings.md`；当前完成判据保持不变：seed2023正式工件成功、三seed冻结门判定落盘，通过后启动结构消融。
- 已继续分段读取历史进度，确认步骤8反馈式FUSR已按validation负证据终止，步骤16不得重新启用该分支；当前唯一有效主线仍是冻结PG95上的CNM-SE。
- 已复核步骤9–15后期记录：步骤14风险诊断、步骤15两seed正式结果及步骤16预备修改均已落盘且无配置冲突；当前最短路径是恢复seed2023推理队列，再按冻结门控自动决定是否启动结构消融。
- `planning-with-files-zh`会话恢复完成：三份规划文件已复核，`session-catchup.py`退出码0；没有发现需要回滚或重做的已完成实验。
- 步骤16.2静态恢复检查：现有队列终态仍为`SUCCESS`且只含seed2024/2025；seed2023配置、资产、正式预算与三seed分析器检查均PASS，Python编译退出0。项目根目录不是Git仓库，根级`git diff --check`按预期退出129，后续改查实际工作树`baselines/S4M`。
- 步骤16.3宿主GPU快照：4080/4090分别空闲`16053/24206 MiB`、利用率均为0%；以`MAX_JOBS=5`恢复原CNM-SE队列，PID=`3043543`。状态确认仅新增`weather_cnm_se_seed2023_formal=RUNNING`，seed2024/2025继续保持`SUCCESS`，resume语义正确。
- 步骤16.4 seed2023正式run=`runs/cnm_se_pg95__weather__test__seed2023__20260723_194924`已`SUCCESS`：PG95 MSE/MAE=`0.261946456/0.293878906`，CNM-SE=`0.254324883/0.289493222`，改善`2.9096%/1.4923%`；工件、固定test、0.5/0.5权重、冻结checkpoint和metric roundtrip均PASS。
- 三seed冻结门自动通过，决策=`CNM_SE_THREE_SEED_GATE_PASS_START_ABLATIONS`。三seed平均相对改善MSE=`2.2197%`、MAE=`0.9779%`，且2023/2024/2025每个seed均双指标改善；允许立即进入结构消融。
- 步骤16.5开始实现结构消融：扩展冻结评估runner新增独立`stage16_ablation`身份，只允许块长`{1,20}`，继续强制test-only、15%新增缺失率、seed绑定和0.5/0.5等权。主`stage15_test`仍只允许L=5，避免消融放宽主结果配置；修改后Python编译退出0。
- 已新增L=1/L=20各自的seed2024 smoke配置，以及2023/2024/2025共6个完整formal配置；每个seed复用对应PG95 checkpoint、固定缺失CSV和固定mask资产，只改变预注册的嵌套块长。
- 已新增结构消融独立队列定义：2个smoke分别解锁各自块长的3个formal，共8个任务；继续复用通用后台调度器的动态GPU补位、最大并发5、OOM退避回队列、状态恢复与显存预留逻辑。静态配置测试同时覆盖非法块长、错误缺失率、错误权重和validation误用的负例门禁。
- 结构消融启动前门禁全部PASS：8份配置/资产与负例检查、步骤15主配置回归、相关Python编译、Shell语法、实际Git工作树`diff --check`均退出0；队列self-test观察到最大并发5，并通过依赖、动态补位、OOM回队列、确定性错误停止、resume与未兑现显存预留检查。
- 步骤16.6结构消融后台队列已启动，PID=`3061968`、最大并发5。启动前4080/4090空闲`16053/24206 MiB`且利用率0%；L=1与L=20两个独立smoke均已`RUNNING`，6个formal依赖等待。上游快照`git status --short`为空，适配器工作树保留项目既有修改。
- 两个结构消融smoke均`SUCCESS`并自动解锁formal；当前L1 seed2023/2024/2025与L20 seed2023/2024共5个formal同时`RUNNING`，L20 seed2025为`WAITING_RESOURCE`，将在任一任务完成后自动补位。所有正式任务保持完整652个test batch，不改模型或评估预算。
- 统一结果表已更新至107个run，输出`reports/latest_results.{csv,md}`；结构消融全完成后独立aggregator会再次自动更新总表并生成L1/L5/L20与三路融合分析。

## 2026-07-23 步骤18

- 用户确认执行严格等实际缺失率的L=1/L=5/L=20复核、同GPU串行推理计时，以及validation-only的L=5+L=20多尺度CNM-SE验证。
- 本轮重新启用`planning-with-files-zh`恢复项目证据链；步骤18禁止依据步骤17 test结果直接修改主模型，模型选择与权重冻结只能使用validation。
- 已更新`task_plan.md`：步骤16/17标记完成，新增阶段13/步骤18五项完成判据；完整计划已分段复核。
- 已复核`findings.md`前200行，确认步骤18继续使用冻结PG95 checkpoint和validation完整目标，不回到已否定的FUSR、mask-space写入或CFR分支。
- 已复核后期研究结论，步骤17的关键剩余混杂确为不同块长实际新增缺失率不一致；步骤18冻结了逐样本等计数规则与`0.5/0.25/0.25`多尺度权重，尚未读取任何新validation结果。
- `findings.md`已完整分段复核至文件末尾；步骤18的接口可直接复用步骤14冻结validation三路评估框架，但需要新增严格等计数mask和三视图输出，不能复用步骤17的Bernoulli起点近似。
- `progress.md`恢复读取已覆盖早期初始化、基线复现与步骤6/7起始；历史GPU UUID、独立环境和上游只读约束保持不变。
- `progress.md`第1–300行已完整复核，确认正式评估均由唯一run、resolved config、checkpoint哈希和固定mask审计支撑。
- `progress.md`第301–450行已完整复核，确认通用调度器已支持并发5、未兑现显存预留和OOM回队列；步骤18的串行计时将另设最大并发1。
- `progress.md`第451–750行已复核：固定评估资产、正式配置不缩水、GPU UUID映射、失败历史保留和“先验证再扩大”的约束继续有效；步骤18不会把历史并发运行耗时当作正式推理计时。
- `progress.md`第751行至文件末尾已完整复核：步骤14的validation缓存框架、步骤15/16固定CNM-SE评估链和步骤17块长混杂均已确认；步骤18将新建独立validation-only身份，不覆盖已有成功工件。
- `planning-with-files-zh`会话恢复完成：三份规划文件均已读至末尾，`session-catchup.py`退出码0；未发现需回滚或重复运行的历史实验。
- 步骤18核心评估入口已初步实现：新增独立CLI/dispatch、逐样本等计数L1/L5/L20生成、八路精度缓存与original/L1/L5/L20同步分项计时；旧stage14/15/16入口未改。当前正在补runner专属schema门禁，尚未进入测试或GPU队列。
- runner已接入独立`stage18_validation`身份与专属工件校验：逐样本三尺度移除数必须精确相等、packed mask必须为原mask子集、八路指标必须从NPZ精确回算、计时数组必须经过warmup且全部有限为正。三份核心文件`py_compile`退出码0。
- 步骤18 CPU/静态门禁全部PASS：等计数组件确认L1/L5/L20每样本均精确移除29/192行、实际率完全一致，平均mask转换数随块长由49.65降至11.35/3.77；三份配置资产、validation-only、固定权重与负向拒绝测试全部通过。
- Python编译、Shell语法、适配器`git diff --check`与上游只读洁净检查退出码0；通用调度器self-test为PASS，覆盖动态补位、依赖、OOM回队列、确定性错误停止、resume和启动显存预留。下一门是单GPU三批smoke。
- 启动前宿主GPU快照：RTX 4080 UUID `GPU-c88b...`空闲16053 MiB且利用率0%；RTX 4090虽空闲16378 MiB但利用率99%并有外部任务。为控制计时干扰，步骤18固定选择4080。
- 同GPU串行队列已启动，scheduler PID=`3257377`、`max_jobs=1`；seed2024三批smoke run=`runs/cnm_se_equal_rate_multiscale_pg95__weather__validation__seed2024__smoke__20260723_211139`已进入checkpoint加载与GPU评估，两个formal仍按依赖等待。
- 首次smoke模型子进程退出0，48个样本、八路精度、三尺度packed mask和2个warmup后计时批次均成功落盘；工件检查18项中仅`additional_rates_roundtrip`失败。根因是GPU float32缺失率落盘后被runner以`1e-12`容差重算，不是模型、掩码或计时错误；formal均保持`BLOCKED_DEPENDENCY`且未启动。
- 窄修复仅把该roundtrip绝对容差改为`1e-7`，与保存值的float32精度匹配；失败run与状态保留，科学协议、模型前向、权重和正式配置均不变。
- 修复后完整CPU组件/配置门禁与静态检查再次PASS；队列以审计原因`float32_rate_roundtrip_tolerance_fix`重排，首次失败run未删除。
- 第二次smoke run=`runs/cnm_se_equal_rate_multiscale_pg95__weather__validation__seed2024__smoke__20260723_211405`为`SUCCESS`，48样本、2个计时批次、同卡UUID、八路metric roundtrip和严格等计数检查全部`PASS`。
- seed2024完整validation已在固定RTX 4080上自动启动，seed2025保持依赖`PENDING`，将在seed2024成功后同卡串行接力；按用户要求确认正式实验运行后不持续人工监控。
- 步骤18队列最终`SUCCESS`：1个smoke与seed2024/2025两个完整validation共3/3成功，调度器自然退出；每个formal含5168个样本、313个warmup后计时批次，峰值allocated显存约1052.60 MiB。
- 严格公平性成立：每个seed内L=1/L=5/L=20逐样本实际移除数完全一致；平均实际新增缺失率seed2024=`0.150057425`、seed2025=`0.150136680`，三个块长逐值相同。
- 固定多尺度`0.5*original+0.25*L5+0.25*L20`通过预注册validation门：相对L5双路，seed2024 MSE/MAE改善`0.9169%/0.4526%`，seed2025改善`1.0778%/0.5952%`；两个seed四项均为正。
- 相对original-only，多尺度在seed2024改善MSE/MAE=`3.3631%/1.3979%`，seed2025=`3.0377%/1.8166%`。双seed均值MSE/MAE=`0.509249735/0.361182632`。
- 同一RTX 4080串行批延迟：original=`25.3005 ms`，L5双路=`51.3268 ms`，多尺度=`77.1518 ms`；多尺度约为original的`3.0494x`，较L5双路增加`50.31%`延迟，不新增模型参数。
- 自动决策=`MULTISCALE_VALIDATION_GATE_PASS_EXPAND_ONLY_TO_NEW_PROTOCOLS`；Weather test未重访。下一步只能扩展到新数据集/新缺失协议做正式外部验证，不能回到已看Weather test修改主模型。

## 2026-07-23 步骤19

- 用户要求开始下一步；步骤19定义为冻结步骤18多尺度结构与`0.5/0.25/0.25`权重，在未参与结构选择的新数据集上做外部验证。
- 本轮重新启用`planning-with-files-zh`；已完整读取技能说明与`task_plan.md`，正在分段恢复`findings.md`和`progress.md`。尚未选择数据集、修改评估代码或启动实验。
- `findings.md`已复核至第280行：ETTh1已有步骤5论文协议近似mask、单seed完整S4M checkpoint和固定评估工件，但基线MSE与论文差距约103.7%，且Gate H显示该checkpoint原型分支近乎失活；步骤19不能未经审计就把Weather PG95假定为可直接迁移。
- `findings.md`已完整复核至末尾；`progress.md`已复核至第460行。历史证据要求步骤19优先复用步骤5 ETTh1固定资产和checkpoint做低成本外部筛选，同时严格披露ETTh1协议偏差，不能把一次正/负结果解释为完整跨数据集结论。
- `progress.md`继续复核至第690行：失败的FUSR/HCUP分支不得重新组合，PG95/BMSA的固定mask、公平预算和负结果保留原则继续有效。步骤19只验证步骤18冻结推理结构的跨数据集可迁移性，不借机改变训练或回看Weather test调参。
- `progress.md`已完整复核至当前文件末尾：确认mask-space、CFR等负分支均已按门禁停止，步骤14–18建立的冻结推理、严格等实际缺失率、固定权重、固定资产和单卡计时证据链完整；步骤19无需重做历史实验。
- `planning-with-files-zh`会话恢复脚本退出码0；三份规划文件与磁盘状态一致，步骤19可以从跨数据集资产/代码审计继续。
- 步骤19.1资产审计完成：本地未发现ETTm1/Electricity完整数据、固定mask或可复用checkpoint；ETTh1是唯一立即具备完整闭环的外部数据集。其正式来源run、数据、mask、checkpoint哈希和211个test batch均已核验，upstream快照保持洁净。
- 步骤19.2预注册协议已写入`docs/STAGE19_EXTERNAL_VALIDATION_PROTOCOL_ZH.md`：冻结ETTh1 seed2023、一次性完整test、逐样本15%等计数L1/L5/L20、`0.5/0.25/0.25`权重；主门为多尺度MSE/MAE均优于同run原始路，增量门为均优于L5双视图。结果分为full/partial/fail且不依据ETTh1回改权重。
- 科学解释边界同时冻结：来源是步骤5原始S4M而非PG95，首轮只验证CNM-SE推理迁移；ETTh1 mask seed不可得且基线与论文偏差较大，不能宣称严格论文复现或统计稳定。
- 步骤19.3代码与资产初版已落盘：新增独立`stage19_equal_rate_multiscale_test`开关，步骤18核心评估被抽为split可配置内部函数，但历史步骤18 wrapper仍硬固定`val`；步骤19 wrapper硬固定`test`。新增ETTh1 smoke/formal配置、协议/分析测试、自动分析器、原始汇总器及后台队列start/status/stop入口。
- 步骤19队列为`smoke→formal`依赖链，最大并发参数保持5但本轮最多只有1个可运行任务；调度器继续按实时UUID/显存动态选择、OOM退避回队列和resume，不为并发或显存缩减正式协议。
- 步骤19.4启动前全部静态门禁PASS：步骤18组件/配置回归、步骤19配置与负测、科学决策表、Python/Shell语法、`git diff --check`、上游洁净和通用队列self-test均退出0；self-test实际观察最大并发5并覆盖OOM回队列、依赖、动态补位和resume。
- 宿主队列PID=`3327844`已启动；seed2023 ETTh1三批smoke run=`runs/cnm_se_equal_rate_multiscale_s4m__etth1__test__seed2023__smoke__20260723_213716`为`SUCCESS`，48样本、20项工件检查全部PASS。随后完整211-batch formal run=`runs/cnm_se_equal_rate_multiscale_s4m__etth1__test__seed2023__20260723_213731`已在4090 UUID `GPU-a3af...`自动启动，未改配置。
- 步骤19.5队列终态`SUCCESS`，smoke/formal共2/2成功，调度器自然退出；formal处理211个test batch/3376个样本，来源基线MSE/MAE在`1e-5`内复现，三尺度逐样本等移除计数、固定权重、split、checkpoint和指标回算全部PASS。
- ETTh1结果几乎数值不变：原始S4M MSE/MAE=`1.207990944735/0.838923262624`，多尺度=`1.207990945018/0.838923262718`；相对变化约`-2.34e-8%/-1.12e-8%`。相对L5双视图仅MSE有`1.67e-8%`数值噪声级改善、MAE退化`8.50e-8%`，两个预注册门均未通过。
- 正式决策=`EXTERNAL_TRANSFER_FAIL`。该负结果否定“冻结CNM-SE可在原始S4M ETTh1 checkpoint上直接通用迁移”，但不等于已经检验PG95+CNM-SE完整方法，因为来源checkpoint明确没有MIPP/PG95。
- 推理代价：同4090批延迟original=`40.7098 ms`、L5双路=`84.5693 ms`、多尺度=`127.1349 ms`，多尺度为original的`3.1230x`；参数量5,767,343不变，峰值allocated约1036.11 MiB。统一总表已更新至114个run。

## 2026-07-23 步骤20

- 用户要求开始下一步；步骤20定义为明确标注post-hoc的新假设：先在ETTh1 seed2023上按现有同协议训练PG95，再用完全冻结的等计数多尺度CNM-SE做一次性正式test，区分“原始S4M即插即用失败”与“完整PG95+CNM-SE方法是否跨数据集有效”。
- 本轮重新启用`planning-with-files-zh`；已完整读取技能说明与`task_plan.md`，确认步骤19为工件PASS但科学门失败，步骤20不得改写或包装步骤19预注册结论。正在继续恢复`findings.md`与`progress.md`，尚未创建步骤20配置或启动训练。
- `findings.md`恢复至步骤7早期：ETTh1固定point-r0.06资产、Adam/batch16/15epoch协议和步骤5基线工件可直接复用；历史Gate H已证明原始ETTh1 checkpoint原型支路失活，因此步骤20的关键信息增量只能来自“重新训练PG95是否恢复跨掩码表示响应”，不能把原始S4M负结果重复包装。
- `findings.md`继续恢复至步骤19结论：步骤15–18已经冻结逐样本等实际缺失率、L5/L20多尺度结构和`0.5/0.25/0.25`权重；步骤19只能否定原始S4M上的即插即用迁移，不能代替ETTh1上完整PG95+CNM-SE的外部验证。
- 已无截断复核 `findings.md` 第211–315行：确认PG95的有效部分是跨掩码配对一致性，Weather训练配置与后续失败分支边界清楚；步骤20不能混入BMSA、FUSR、CFR等已否决模块。
- `findings.md`已无截断完整恢复至文件末尾：步骤20的唯一新增科学问题、数据边界和冻结评估结构均已确认，接下来恢复完整执行日志后再落地配置。
- `progress.md`共948行，已从头复核至第120行：项目初始化、独立环境、上游只读快照、基线适配副本与固定资产原则仍有效；步骤20继续只修改 `baselines/S4M` 和项目外层脚本。
- `progress.md`已复核至第360行：步骤5 ETTh1正式基线的完整训练预算、成功工件与论文偏差已确认；Gate H证明原始ETTh1原型分支数值失活，PG95训练是步骤20必要且未做过的区别因素。
- `progress.md`已复核至第600行：Weather PG95的真实run、机制开关、正式资源开销及其后FUSR/HCUP负分支均已确认；步骤20严格复用PG95而不复活已停止的效用路由路线。
- `progress.md`已复核至第840行：BMSA/SDRA/MRBA/CFR各分支的预注册失败与停止边界、步骤14风险证据及队列工程修复均已恢复；这些模块均不进入步骤20。
- `progress.md`已完整复核至当前文件末尾：步骤18/19的实现、正式结果、失败解释和步骤20起始状态均与 `findings.md` 一致，没有需要重做的历史实验。
- 项目专属 `AGENTS.md` 已重新完整核对；步骤20继续遵守独立smoke、固定资产、唯一实验ID、最大并发5动态队列、UUID选卡、OOM退避、工件与自动汇总要求。
- 步骤20资产初查：ETTh1步骤5基线配置与步骤19冻结多尺度配置均存在；Weather PG95正式/冒烟配置可作为模块参数来源。当前4080/4090分别空闲约11.7/15.9 GiB但利用率较高，正式启动仍交由UUID队列动态判断；上游快照洁净，可编辑仓库保留既有项目修改。
- 配置差异已逐项核对：ETTh1必须保留`freq=h`、7变量、Adam学习率0.005及步骤5的完整15 epoch协议；PG95只移植Weather已冻结的MIPP开关、pair loss 0.05、绝对稳定性门0.95和配对mask范围，不移植Weather的数据/学习率。
- 队列实现方案已收敛：使用一个四任务依赖链`PG95训练smoke→PG95正式训练→冻结评估smoke→完整test`。评估source checkpoint由正式训练成功run动态解析并物化配置，避免手工监控或提前猜测带时间戳的checkpoint路径。
- 评估runner扩展边界已确认：新增独立`stage20_pg95_external_test`身份，继续复用步骤19的等计数测试核心，但必须强制source为ETTh1 PG95、绝对写入门、无FUSR/CFR，并要求MIPP checkpoint键；历史步骤18/19验证条件保持不变。
- 步骤20.1协议已落盘至`docs/STAGE20_ETTH1_FULL_METHOD_PROTOCOL_ZH.md`，并新增两份PG95训练配置与两份动态source评估模板；正式比较、解释边界和五类决策标签均在任何新结果出现前冻结。
- 通用调度器接口已核验可直接承载混合任务：统一dispatcher只需把训练配置交给训练runner，把含`source.job_id`的评估模板在依赖成功后物化并交给冻结评估runner；scheduler仍能从透传的`run_dir=`、status和metrics完成PID、OOM与结果跟踪。
- 步骤20首轮静态门禁已全部退出0：新Python/Shell语法、四份配置及4个负例、动态source精确交接与越界checkpoint拒绝、五类科学决策表，以及步骤18/19历史配置回归均PASS。
- 启动前第二轮门禁同样PASS：严格等计数组件、通用队列并发5/OOM/依赖/resume/未兑现显存预留自测、适配器`git diff --check`和上游洁净检查均退出0；新训练setting目录名预计239字符，低于Linux 255字符上限。
- 第一次从隔离进程命名空间启动只写出无效PID=`11`，无state、无run、无GPU进程；未误报为成功。改用宿主持久启动后有效scheduler PID=`3393217`、PPID=1、SID=3393217，队列状态RUNNING。
- ETTh1 PG95训练smoke首次attempt `SUCCESS`：run=`runs/mipp_pg95_etth1__etth1__point__mr0.06__seed2023__smoke__20260723_220331`，return code 0、checkpoint CPU重载及7个MIPP参数键PASS；正式训练依赖已自动解除并启动。
- ETTh1 PG95正式训练已通过四重真实性核验：run=`runs/mipp_pg95_etth1__etth1__point__mr0.06__seed2023__20260723_220401`、run status=`RUNNING`、训练PID=`3394567`、4090 UUID上实占约6084 MiB；日志已进入epoch1至少iter100，batch16/Adam lr0.005/15 epoch完整配置且无Traceback/OOM。
- 按用户规范，确认正式训练真实运行后停止人工持续监控。后台队列会在训练成功后自动物化正式checkpoint，依次运行3-batch冻结评估smoke和211-batch完整test；失败按类别保留工件，OOM退避回队列。
- 步骤20队列终态`SUCCESS`，四任务4/4首次attempt成功，scheduler PID=`3393217`自然退出；正式PG95 run和多尺度run分别为`mipp_pg95_etth1__etth1__point__mr0.06__seed2023__20260723_220401`与`cnm_se_equal_rate_multiscale_pg95__etth1__test__seed2023__20260723_224153`。
- PG95相对步骤5 S4M的MSE/MAE退化`0.2935%/0.1880%`；多尺度相对同run PG95原始路只有MAE数值噪声级改善、MSE数值噪声级退化。正式决策=`FULL_METHOD_FAIL`，协议/工件检查全部PASS。
- PG95正式训练8 epochs早停、5944 optimizer steps、耗时2253.23秒、峰值allocated/reserved约5011/5582 MiB；多尺度完整评估3376样本、耗时93.84秒、峰值allocated约1036.65 MiB。
- 多尺度串行批延迟original=`62.314 ms`、multiscale=`255.072 ms`，约`4.093x`；统一总表和`reports/stage20_etth1_pg95_full_method.{json,md}`已自动生成。步骤20完成但科研结论为负。

## 2026-07-24 步骤21

- 用户要求执行步骤21；目标冻结为Electricity外部数据协议：先建立固定缺失资产与S4M/PG95同协议结果，再在validation验证预测扰动响应和冻结多尺度互补，只有validation证据成立才允许一次正式test。
- 本轮启用`planning-with-files-zh`并完整读取技能说明；`task_plan.md`已恢复至第170行，确认步骤20为工件PASS但`FULL_METHOD_FAIL`，步骤21不能依据ETTh1 test继续调整PG95阈值、块长或融合权重。
- `task_plan.md`已完整恢复至文件末尾；`findings.md`已开始复核，Electricity协议先验为321变量、26304个小时级时间点、按0.7/0.1/0.2划分和预测长度96。下一轮继续完整恢复`findings.md`与`progress.md`后才修改代码或启动实验。
- `findings.md`大段并行读取因工具输出截断，未把截断内容误记为“已完整读取”；已恢复到步骤5/6的核心边界：固定mask、Adam/batch16、同checkpoint原型诊断及Weather/ETTh1基线差异。后续改为小分块逐段读完。
- `findings.md`第111–230行已逐段完整恢复：固定资产生成、S4 CUDA扩展/原型库生命周期修复、步骤5两基线、Gate H、MIPP/PG95与FUSR最小边界均已确认；这些历史修复与负分支不重做。
- `findings.md`第231–350行已完整恢复：FUSR各版/短跨度历史探针均已在validation失败；BMSA三seed不稳定，SDRA/MRBA虽改变写入机制但未形成稳定精度收益；步骤21不再沿这些已否定方向调参。
- `findings.md`第351行至文件末尾已完整恢复：CFR失败；跨掩码预测分歧、固定等权CNM-SE及冻结多尺度在Weather validation/test成立；ETTh1步骤19/20均因预测扰动不敏感而失败。步骤21的关键是先在Electricity validation证明非零视图分歧与多尺度互补。
- `progress.md`第1–200行已完整恢复：项目初始化、步骤4完整架构smoke、步骤5固定mask/基线队列与原型库epoch生命周期修复的执行证据均已核对。
- `progress.md`第201–400行已完整恢复：步骤5收口、Gate H跨域诊断、MIPP/PG95实现与GPU调度路由修复、步骤7正式结果的完整执行链已核对。
- `progress.md`第401–600行已完整恢复：步骤7报告闭环、并发5调度实测、FUSR V1/V2/V3与HCUP各轮负结果及停止边界均已确认；步骤21不得重新试这些已失败选择器。
- `progress.md`第601–800行已完整恢复：BMSA单seed正但三seed均值负，SDRA/MRBA与validation稀有mask分组均失败；对应机制、队列竞态修复和停止决策已核对。
- `progress.md`第801行至文件末尾已完整恢复：CFR、步骤14–18、ETTh1步骤19/20的协议、门禁、正式结果与代价均已核对；三份文件规划状态现已完整恢复。
- 项目专属`AGENTS.md`已完整复核；`session-catchup.py`在项目根目录退出码0。步骤21继续执行固定资产、独立smoke、UUID动态队列、并发上限5、OOM退避和validation先行的本地规范。
- 步骤21.1初查确认项目已有`data/electricity/`登记目录，但没有Electricity固定mask、配置或checkpoint；`upstream/S4M`仍洁净，可编辑基线保留既有修改。后续先核验真实CSV、论文超参数和生成器适配性，尚未启动实验。
- Electricity完整CSV为项目内只读链接，现有mask生成器仅支持Weather/ETTh1；步骤21需要新增Electricity数据规范，且论文实际缺失率必须先从本地PDF/Table 8核对，未凭经验填值。
- Electricity CSV逐值结构审计退出码0：26304行、322列、321个float64变量、无NaN，时间起止与每小时协议一致；该文件可作为固定缺失资产的完整源。
- 已从ICLR 2025官方论文核对Electricity协议：实际缺失率0.260、lr0.01、hidden512、batch16、Adam/early stopping，H=192参考MSE/MAE 0.274/0.367；最终层数未公开，只给候选{2,4,8}，该不确定性将显式保留。
- 步骤21预先冻结8层/hidden512，并识别出训练入口会无条件构造test loader与训练后test。下一项工程修复是新增默认不影响旧实验的validation-only训练后评估模式，确保正式test只可能在validation门通过后发生。
- 数据manifest与模型入口审计：Electricity完整源哈希已确认，`run.py`能识别321变量；下一项只读核验是Dataset_Custom4完整目标路径和验证集顺序，随后开始最小代码修改。
- Dataset_Custom4与步骤18评估核心审计完成：完整目标路径、0.7/0.1/0.2划分和chronological validation可直接复用；步骤21需独立训练后validation模式和独立Electricity多尺度身份，历史入口不放宽。
- 步骤21.2固定资产生成命令退出码0：Electricity seed2023 point-r0.06/L5实际缺失率0.2599224453，CSV/NPZ/JSON与三项哈希已落盘并登记manifest；下一项是独立资产回归审计。
- 资产回归第一次在全部断言后因`numpy.bool_`不可JSON序列化退出1，未改动资产；显式转为Python bool后重跑退出0。22项检查全部PASS，训练/验证/test行数为18412/2632/5260，观测值最大误差0。
- 步骤21.3四份训练配置与validation-only入口已落盘，静态/配置门禁全部PASS；S4M/PG95正式配置除模块开关外同预算，setting名243/244字节均低于255。下一项接入独立Electricity冻结多尺度validation/test身份。
- 冻结评估runner验证路径已审计：步骤21 validation会闭环source validation指标；未来test若获准，只复用checkpoint与协议，不把test原始路错误要求等于validation指标。
## 2026-07-24 步骤21.4：Electricity 多尺度入口实现审计

- 审计了 `scripts/run_stage14_uncertainty_cache.py` 现有 Stage18/19/20 分流。
- 已确认不能复用“external test”分支冒充验证：Stage21 必须拥有独立 validation/test 标志、独立样本文件名及独立来源指标门禁。
- 正在补齐以下硬约束：训练来源必须是 validation-only；Stage21 validation 必须 `uses_test_split=false`；Stage21 test 只能由后续科学门控授权。
- 已完成核心 `run.py`、`exp_pretrain1.py` 与外层审计运行器的 Stage21 独立分流，四个文件 `py_compile` 均为退出码 0。
- 外层运行器现在会在任何 Stage21 评估前核验来源 `metrics.json`：`evaluation_split=validation`、`post_train_evaluation=validation`、`test_split_accessed=false`、`chronological_loader=true`。
- 已复核通用队列：支持依赖、最多5并发、显存动态补位、启动预留、smoke峰值回填正式显存估计、OOM退避回队列与断点恢复。
- 决定 Stage21 第一条后台队列只包含 S4M/PG95 smoke、正式 validation 训练及 PG95 validation 多尺度复核；科学门控报告将由队列聚合器生成。唯一 test 不进入第一条静态队列，避免门控失败时仍被依赖机制误启动。
- 已进一步实现为单一依赖流水线：最后一个“条件测试”作业会先重新生成并核验validation门控；未授权时只写 `SKIPPED_BY_VALIDATION_GATE/test_accessed=false` 审计记录，绝不调用数据加载器；授权时才动态物化PG95 checkpoint并运行唯一一次test。
- 新增3份多尺度配置、Stage21调度/聚合/分析/启停状态脚本及管线负例测试。
- `py_compile` 通过；`test_stage21_electricity_assets.py`、`test_stage21_electricity_training_configs.py`、`test_stage21_electricity_pipeline.py` 均退出码0。
- Stage21调度器假任务自测9项全部PASS，实测最大并发5。
- 启动前GPU快照：RTX4080空闲8537 MiB、RTX4090空闲10537 MiB；两卡均有其他合法作业。由于512维/8层 Electricity 尚无本项目峰值记录，首次smoke保留14336 MiB启动预算，当前应进入非阻塞 `WAITING_RESOURCE`，而不是冒险启动或缩小正式模型。
- 后台启动 `bash scripts/start_stage21_electricity_queue.sh` 在真正执行前被平台拒绝：Codex宿主提权执行额度已耗尽。复核 `scheduler.pid` 与 `state.json` 均不存在，因此没有半启动任务、没有GPU进程、没有test访问。
- 当前唯一外部阻塞是启动权限；用户在项目根目录执行同一启动脚本后，流水线将自动等待显存、smoke profiling、跑正式validation、生成门控，并仅在授权时运行一次test。
## 会话：2026-08-04（步骤21继续）

- 用户确认Codex执行额度已恢复，授权继续步骤21。
- 本轮先按文件规划流程重读项目规范与步骤21持久状态，并运行 `session-catchup.py`；尚未启动新进程或访问test。
- 当前目标保持不变：先启动S4M/PG95 Electricity validation-only smoke与正式训练，再跑冻结validation多尺度；仅当预注册门全部通过时授权唯一一次test。
- 步骤21.1恢复复检完成：旧 `state.json` 与 `scheduler.pid` 均不存在，确认没有重复队列或半启动任务。
- 重新执行核心Python编译、22项Electricity资产审计、训练配置公平性/validation-only审计与Stage21流水线负例测试，全部退出码0、状态PASS。
- 首次队列PID 255207在两项smoke均未启动时安全停止，终态`STOPPED`，事件流含` scheduler_stopped`，没有产生run或GPU进程。
- 仅修正调度显存估计：S4M `14336→9216 MiB`（历史8层/256维峰值3630 MiB，按512维近似翻倍再加1536 MiB安全余量）；PG95 `14336→13312 MiB`（历史峰值5620 MiB，近似翻倍并加安全余量）。模型、batch、epoch、学习率和数据协议均未改变。
- 步骤21队列已在宿主恢复：scheduler PID/SID=`261619`、PPID=1、最大并发5，状态`RUNNING`。
- S4M Electricity smoke attempt1已在物理4080 UUID `GPU-c88b...`真实运行：runner PID=`261627`、训练PID=`261694`、run=`runs/s4m_paper_adam_electricity__electricity__point__mr0.06__seed2023__smoke__20260804_193540`。
- 日志已打印完整321变量、512维、8层、batch16、Adam lr0.01、`post_train_evaluation=validation`、参数量22,934,781、memory warmup和首批forward张量；NVML确认训练PID占用5478 MiB。PG95 smoke保持`WAITING_RESOURCE`，后续任务按依赖排队，test尚未访问。
## 会话：2026-08-05（步骤21终态核验）

- 队列`queue_state=SUCCESS`；S4M smoke/formal、PG95 smoke/formal、多尺度validation smoke/formal、条件test门控共7/7任务均`SUCCESS`，调度器自然结束。
- S4M validation MSE/MAE=`0.8783867813/0.7530064546`；PG95=`0.8728095907/0.7545076541`。PG95只改善MSE，MAE退化，未过双指标门。
- 多尺度validation MSE/MAE=`0.8728095891/0.7545076515`；相对原始路和original+L5均只有浮点噪声量级变化且MSE方向为负，未过多尺度门。
- 条件test工件明确记录`test_authorized=false`、`test_accessed=false`，没有运行Electricity test。
- 发现分析器把run专属输出路径误算为预算差异，导致报告标签为`PROTOCOL_FAIL_TEST_FORBIDDEN`；该报告器问题尚未修复，因此步骤21实验已完成但正式收口仍差一次零GPU报告修正与回归。
- 已修改`scripts/analyze_stage21_electricity_validation.py`，显式分离run身份字段、PG95模块字段与真实协议漂移；新增`audit_method_differences`可测试函数。
- `scripts/test_stage21_electricity_pipeline.py`新增两项回归：四个run路径差异不得触发预算失败，`batch_size 16→8`必须被识别为真实漂移。Python编译、11项检查、5个负例和`git diff --check`均退出码0。
- 零GPU重算结果：`protocol_pass=true`、`unexpected_method_differences=[]`、`pg95_gate_pass=false`、`multiscale_gate_pass=false`、`decision=FULL_VALIDATION_FAIL_TEST_FORBIDDEN`、`test_authorized=false`。
- 步骤21正式完成。规范报告为`reports/stage21_electricity_validation_gate.json/.md`；未重新训练、未修改任何指标、未访问test。

## 2026-08-05 步骤22启动

- 用户授权继续下一步；步骤22冻结为validation-only（仅验证集）创新迭代，目标同时处理PG95相对S4M的MAE退化和L5/L20预测视图近乎相同的问题。
- 已重新读取`planning-with-files-zh`技能与项目`AGENTS.md`，并把`task_plan.md`当前阶段更新为阶段17/用户可见步骤22。
- 当前只恢复证据和执行零GPU诊断；尚未修改模型、尚未启动GPU实验、尚未访问Electricity test。
- 已按规划技能继续恢复`progress.md`前520行：确认既有训练/评估入口的默认关闭兼容性、固定资产、UUID队列和历史失败边界；步骤22不会重做FUSR V1/V2/HCUP，也不会把test用于模块选择。
- 已继续恢复`progress.md`第521行至文件末尾：BMSA/SDRA/MRBA/CFR的停止边界、CNM-SE在Weather成立而在ETTh1/Electricity视图塌缩、以及步骤21的validation-only门控均已核对；历史结论无需重算。
- 步骤22.2逐样本多尺度工件审计完成：Electricity L1/L5/L20相对original只出现浮点舍入级误差差异，视图塌缩成立。两次只读路径猜测错误均已改由`rg --files`定位真实文件，未改代码或实验。
- 已确认可复用步骤12的冻结checkpoint、chronological validation缓存骨架；步骤22将新增独立Electricity身份和配对分析，不放宽或覆盖步骤12 Weather历史协议。
- 步骤22.3已新增默认关闭的`stage22_validation_diagnostic`入口及CLI门禁：同一冻结checkpoint只读评估原始视图、等计数L5嵌套视图和0.5等权视图，保存样本/预测步/变量维误差与预测分歧；尚未通过编译与组件测试，因此尚未启动GPU。
- 通用冻结缓存runner已在保持Stage12默认语义的前提下增加`stage_identity=stage22`分流，并新增S4M/PG95各自smoke与完整Electricity validation配置；正式配置固定2528样本、0.15等新增缺失率、L=5、seed2023和原步骤21 checkpoint。
- 已新增Stage22专属配置负测、配对误差/块自助法/预测步/变量维分析器、分析单测、四任务smoke→formal动态队列、聚合器及start/status/stop入口；下一项是全套静态门禁和队列假任务自测，未通过前不启动GPU。
- 步骤22静态门禁全部退出0：配置27项、分析8项、历史Stage12回归30项、调度器9项均PASS，假任务最大并发5；随后宿主队列PID/SID=`799772`以最大并发5启动。
- 启动快照：物理4080 UUID `GPU-c88...`空闲15643 MiB，物理4090 UUID `GPU-a3af...`空闲3513 MiB；调度器先在4080运行S4M smoke，未冒险把任务投到资源不足的4090。
- S4M Stage22 smoke首次attempt SUCCESS，32样本、19项工件检查PASS、峰值allocated/reserved约2443.62/3694 MiB；预测分歧MSE均值仅`7.68148e-16`，证明原始S4M本身已视图塌缩。
- 新建status脚本初始无可执行位，直接调用报`Permission denied`；已为Stage22 start/status/stop补可执行位并成功复核。该观察脚本问题未影响scheduler或实验。
- 步骤22四个validation缓存任务均首次attempt成功；S4M/PG95完整validation各2528样本、test access均为false。PG95相对S4M的MSE改善`0.63494%`而MAE退化`0.19936%`，块自助置信区间均跨0。
- 逐样本诊断显示`39.83%`样本双指标改善、`18.12%`仅MSE改善、`42.05%`双指标均未改善；全部96个预测步MSE改善但0/96个预测步MAE改善。冲突是难度分层且广泛存在，预注册“top10贡献超过50%”条件未通过，不能事后改写为纯极端尾部问题。
- 预测视图分歧/预测MSE比在S4M与PG95分别只有`1.04e-15/1.26e-15`，确认是架构级视图塌缩。源码还发现官方`min_idx`标准化疑似误用`max_idx`，该上游bug将用独立身份处理，不计入创新收益。
- 步骤22.5已在读取任何新训练结果前冻结`docs/STAGE22_MRRA_RADR_PROTOCOL_ZH.md`：候选A为MRRA（掩码响应残差适配器），候选B为RADR（秩自适应双风险损失），两者首轮独立运行，固定Electricity PG95 seed2023、batch16、Adam 0.01、15 epoch、validation-only。
- MRRA已作为默认关闭的因果低秩模块接入：输入衰减插补历史与观测掩码，零初始化上下文投影保证初始预测等于基线，并用权重0.1的缺失历史重建辅助损失驱动掩码响应。RADR不增加参数，以batch内detach样本MSE秩分别加权难样本MSE和主体样本MAE，归一后0.5/0.5组合；验证指标实现保持不变。
- 新增四份独立smoke/formal配置、runner字段与checkpoint/机制审计、组件和公平性负测。组件12项、配置25项、历史Electricity回归52项、调度器9项均PASS，`py_compile`、shell语法、`git diff --check`退出0，上游快照洁净。
- 步骤22.6五任务总队列已后台启动，scheduler PID/SID=`875586`、PPID=1、最大并发5；依赖为默认关闭PG95回归smoke→MRRA/RADR并行smoke→各自formal，支持UUID选卡、动态补位、显存预留、OOM退避回队列和resume。
- 启动快照：物理4080 UUID `GPU-c88...`空闲10319 MiB，物理4090 UUID `GPU-a3af...`空闲3473 MiB，均低于首个PG95 smoke的13312 MiB安全估计；队列当前`WAITING_RESOURCE`且attempt_count=0，没有冒险启动、没有缩模型、没有访问test，显存释放后会自动补位。
- 已新增步骤22自动科学门与中文报告器，6项门控单测PASS；它会在两项formal结束后自动检查同预算、固定资产、完整validation、checkpoint与机制证据，并按“双指标不退化且至少一项严格改善”决定下一轮，不会授权test。
- 为让已运行调度器加载新聚合器，在所有任务attempt_count仍为0时使用安全stop请求退出旧PID `875586`；状态写为`STOPPED`后以resume恢复，新scheduler PID/SID=`898264`、PPID=1，任务状态和历史完整保留，当前仍为`WAITING_RESOURCE`。

## 2026-08-06 步骤22.8队列终态与首轮结果

- 五任务队列终态`FAILED`：默认关闭PG95回归smoke、MRRA smoke、RADR smoke和RADR formal均首次attempt `SUCCESS`；MRRA formal首次attempt失败，成功任务不重复运行。
- MRRA formal完整保留为`runs/mrra_pg95_electricity__electricity__point__mr0.06__seed2023__20260805_212515`。训练epoch1 validation loss=`0.8844768`；epoch2约iter700后训练loss出现NaN，epoch2 validation也为NaN，epoch3 warmup进入Bio.Cluster时因输入包含NaN退出。runner将其表面分类为`CODE_ERROR`，科学根因应记为`NAN_OR_DIVERGENCE`。
- RADR formal run=`runs/radr_pg95_electricity__electricity__point__mr0.06__seed2023__20260805_215330`完整2528样本validation、checkpoint重载、固定数据和`test_split_accessed=false`均PASS；5655 optimizer step、5 epoch早停、峰值allocated/reserved=`11540.74/12382 MiB`。
- RADR validation MSE/MAE=`0.8924483014/0.7478247903`；相对冻结PG95 `0.8728095907/0.7545076541`，MSE约退化`2.25%`而MAE约改善`0.89%`。RADR机制真实执行但未通过双指标门，不扩多seed、不访问test。
- 下一动作是只读审计MRRA失败checkpoint中的首个非有限参数和优化链，随后冻结最小数值稳定修复；修复必须先通过覆盖多epoch/更多batch的独立smoke，不能直接重跑formal。

## 2026-08-06 步骤22.9–22.10 MRRA根因审计与v2启动前门禁

- 新增只读`scripts/diagnose_stage22_mrra_checkpoint.py`并保存`reports/stage22_mrra_checkpoint_finite_audit.json`。MRRA smoke checkpoint 329/329张量有限；失败formal checkpoint有224/329张量、17,853,822个元素非有限，其中MRRA自身48,257/48,257个元素全部非有限，主干另污染17,805,565个元素，`repre`仍有限。
- 日志定位到epoch2 iter650仍有限、iter700首次打印NaN；epoch2 validation NaN被原`EarlyStopping`错误当成改善并覆盖epoch1有限checkpoint，epoch3 memory warmup才在聚类阶段显式报错。因此失败科学分类修正为`NAN_OR_DIVERGENCE`，不再作为普通代码错误重试。
- `baselines/S4M/s4m/model/MRRA.py`新增显式`mrra_version`：默认`v1`保持原始结构和checkpoint行为，`v2_normalized_bounded`增加低秩隐藏LayerNorm与有界`tanh`上下文/重建残差；零初始化、rank32、kernel5和0.1辅助MSE不变。
- 训练循环新增非有限训练/验证目标即停；早停器拒绝NaN并用临时文件+`os.replace`原子保存；runner把对应失败持久化为`NAN_OR_DIVERGENCE`并逐张量验证checkpoint有限性。上述防护不修改任何有限训练目标或正式超参数。
- 新增MRRA-v2 stability-smoke/formal配置、三任务依赖队列（默认关闭PG95回归smoke→MRRA-v2 3×800长稳定性smoke→完整formal）、独立聚合/状态/启停脚本，并更新validation-only联合科学门读取旧RADR与新MRRA-v2结果。
- 启动前门禁退出0：Python/Shell编译、MRRA/RADR组件20项、六份配置公平性与负例43项、科学门6项、v2队列8项、通用调度器9项以及`git diff --check`全部PASS。长smoke明确不是正式结果；正式配置仍15 epoch、batch16、Adam 0.01、完整158批validation且不访问test。
- 启动前GPU快照：物理4080 UUID `GPU-c88...`空闲16050 MiB且利用率0%；物理4090 UUID `GPU-a3af...`空闲22635 MiB但利用率98%。队列按UUID、真实显存和13312 MiB含安全余量估计自动选择，不强行并发依赖任务。
- MRRA-v2后台队列已真实启动：scheduler PID/SID=`3894441`、PPID=1、最大并发5、`queue_state=RUNNING`。当前代码下PG95默认关闭回归smoke首次attempt `SUCCESS`，run=`runs/mipp_pg95_electricity__electricity__point__mr0.06__seed2023__smoke__20260806_141953`。
- 依赖已自动补位到MRRA-v2长稳定性smoke，runner PID=`3898130`、训练主PID=`3898183`，run=`runs/mrra_v2_pg95_electricity__electricity__point__mr0.06__seed2023__smoke__20260806_142023`，使用4090 UUID `GPU-a3af...`；实际命令已核对为321变量、512维、8层、batch16、Adam lr0.01、3 epoch、每epoch最多800 batch、`mrra_version=v2_normalized_bounded`和validation-only。正式实验仍为PENDING，test未访问。

## 2026-08-10 步骤22.12 MRRA-v2队列终态初核

- `runtime/stage22_mrra_v2/state.json`终态为`FAILED`，scheduler PID=`3894441`已自然退出，当前GPU上没有本项目残留进程。
- 当前代码PG95默认关闭回归smoke为`SUCCESS`；MRRA-v2长稳定性smoke为`NAN_OR_DIVERGENCE`；15 epoch formal为`BLOCKED_DEPENDENCY`且未启动，因此不存在可用于科学比较的MRRA-v2正式MSE/MAE。
- 失败run保持为`runs/mrra_v2_pg95_electricity__electricity__point__mr0.06__seed2023__smoke__20260806_142023`。有限值保护成功阻止非有限目标继续训练和覆盖最佳checkpoint；下一项只读提取具体epoch/batch、有限checkpoint及相对v1的失稳窗口。
- 日志精确定位：MRRA-v2完成epoch1的800个枚举batch（首批memory warmup不优化），局部3-batch validation loss=`0.8787544`并原子保存有限checkpoint；epoch2重新warmup后在batch16被逐batch保护捕获`forecast_loss=nan`，总运行574.15秒。该局部validation不是完整158-batch正式结果，不得与PG95正式锚点比较。
- 保存的epoch1 checkpoint审计PASS：331个模型张量、24,166,484个参数元素全部有限；10个MRRA张量48,321个元素全部有限，`repre`也有限。相较v1约epoch2 batch700才显式NaN，v2约813次成功optimizer step后即失败，稳定窗口反而明显缩短。
- MRRA-v2 epoch1末适配器参数已显著放大：`input_projection.weight`最大绝对值约2.584、`temporal_filter.weight`约2.015、LayerNorm权重/偏置约1.925/1.475；在共享Adam lr=0.01下，下一轮优先验证“适配器与主干优化耦合/学习率尺度不匹配”，而不是继续增加输出限幅。

## 2026-08-10 步骤22.13 MRRA-v3协议冻结

- 用户授权继续下一步；本轮目标不是重跑v2，而是实现梯度隔离双时间尺度MRRA-v3。
- 源码审计确认v1/v2辅助重建存在`z_s`恒等梯度支路，会以主干Adam lr0.01更新`decay1/decay2`并在后续指数插补中放大数值风险。
- 新结果前冻结：版本=`v3_gradient_isolated_two_timescale`；适配器输入detach；LayerNorm无仿射参数；上下文/重建修正继续`tanh`有界；主干lr=0.01不变，MRRA参数组lr乘数固定0.1；正式batch16/15 epoch/完整validation不变。
- 先完成CPU梯度隔离、参数组/调度器、默认v1/v2回归、checkpoint与配置负测，再运行PG95默认关闭回归smoke→3×800长稳定性smoke→formal依赖链；任何smoke失败都阻断formal。
- MRRA-v3核心代码已接入且默认关闭：`MRRA.py`实现detach输入、无仿射LayerNorm与有界输出；训练器实现主干/MRRA双参数组、逐步梯度与参数有限性诊断；学习率调度器按每组`lr_scale`保持比例。
- 首轮CPU/静态门禁退出0：Python编译、`git diff --check`及26项组件检查全部PASS。组件证实v3输入无梯度、适配器参数仍有有限非零梯度、参数组无重叠，学习率从`0.01/0.001`调度到`0.005/0.0005`后仍保持0.1比例；v1/v2历史行为测试继续PASS。
- 步骤22.13启动前工程已补齐：新增v3 stability/formal配置、默认关闭PG95回归→v3长稳定性→formal三任务依赖队列、聚合器及start/status/stop脚本；formal会被任何前置smoke失败自动阻断。
- 完整零GPU门禁全部退出0：8份创新配置公平性/负例检查PASS，科学门单测PASS，v3队列8项PASS，通用调度器9项PASS且假任务实测最大并发5；Shell语法和可编辑S4M仓库`git diff --check`均PASS。Matplotlib仅报告不可写用户缓存并自动改用`/tmp`，不影响测试。
- 启动前宿主GPU快照：物理4080 UUID `GPU-c88...`空闲8494 MiB，物理4090 UUID `GPU-a3af...`空闲16242 MiB；两卡利用率均99%且存在其他合法作业。队列按13312 MiB含安全余量估计选择4090，不干扰其他进程、不缩模型。
- MRRA-v3后台队列已真实启动：scheduler PID/SID=`1243768`、PPID=1、最大并发5。默认关闭PG95回归smoke首次attempt `SUCCESS`，run=`runs/mipp_pg95_electricity__electricity__point__mr0.06__seed2023__smoke__20260810_163423`；解析参数确认`use_mrra=false`且新增默认参数不改变基线。
- 依赖已自动补位到MRRA-v3长稳定性smoke：runner PID=`1247818`、训练PID=`1247986`，run=`runs/mrra_v3_pg95_electricity__electricity__point__mr0.06__seed2023__smoke__20260810_163546`，使用4090 UUID `GPU-a3af...`并占用约12870 MiB。实际配置核对为321变量、512维、8层、batch16、3 epoch×最多800 batch、validation最多3 batch、`v3_gradient_isolated_two_timescale`、主干/MRRA学习率=`0.01/0.001`；formal仍为PENDING且test未访问。

## 2026-08-13 步骤22.13 MRRA-v3终态核验

- 三任务队列于2026-08-10自然结束，`queue_state=SUCCESS`；PG95默认关闭回归、MRRA-v3长稳定性smoke和正式validation均首次attempt成功，无OOM、无重试、调度器PID已退出。
- 长稳定性smoke完成3 epoch、2397个有限optimizer step，越过v1/v2旧失稳窗口；梯度范数最大值`0.0955781`、MRRA参数最大绝对值`1.10305`，checkpoint全部模型张量有限且可重载。因此v3的梯度隔离、无仿射归一化和0.1倍适配器学习率解决了已观察到的NaN稳定性故障。
- 正式配置预算仍为15 epoch并按相同early stopping在epoch8停止，完成9048个optimizer step和完整158 batch/2528样本chronological validation。checkpoint可重载、全部模型张量有限、机制证据和固定数据/同预算/validation-only检查均PASS，`test_split_accessed=false`。
- 正式MRRA-v3 MSE/MAE=`0.8784480463/0.7536469033`；冻结PG95锚点为`0.8728095907/0.7545076541`。MRRA-v3的MSE退化`0.646012%`，MAE改善`0.114081%`，未通过“两项均不退化且至少一项严格改善”的冻结门，不进入多seed。
- MRRA-v3增加48,257参数（`+0.20009%`），峰值allocated显存增加约34.24 MiB（`+0.2967%`）。本次总运行时间7501.68秒、锚点3810.39秒，但两者GPU和同期负载不同且early-stopping epoch为8对6，只能记录观察值，不能归因成纯模块开销。
- 联合自动报告显示`PROTOCOL_FAIL_NO_SCIENTIFIC_DECISION`，唯一失败项来自历史RADR工件缺少后来新增的`checkpoint_all_model_tensors_finite`显式证据；MRRA-v3自身全部协议检查为true。该历史证据缺口不改变MRRA-v3直接比较已明确失败的双指标门，也不授权test。

## 2026-08-13 步骤22.14 MRRA-v3误差迁移诊断启动

- 用户授权继续下一步；本轮只在固定Electricity seed2023 chronological validation上读取冻结PG95/MRRA-v3 checkpoint，不训练、不访问test。
- 现有`run_stage12_validation_cache.py`及`stage22_validation_diagnostic`可直接复用同一数据、L=5等计数嵌套掩码和逐样本/96预测步/321变量误差缓存，并对source validation MSE/MAE做`1e-5`闭环；历史PG95正式缓存已`artifact_status=PASS`且2528样本齐全。
- 当前入口有意只允许`s4m/pg95`方法身份。步骤22.14将新增严格`mrra_v3`身份，要求source显式启用MIPP与MRRA-v3、版本=`v3_gradient_isolated_two_timescale`、学习率乘数0.1、RADR关闭，并保持历史身份行为不变。
- 本轮首要诊断不是用validation误差直接训练oracle门，而是检验MRRA-v3收益/损害在PG95难度分位、预测步、变量及可观测嵌套掩码响应上的分布。只有响应信号能预测误差迁移时，才有依据冻结风险约束残差门。
- 已在读取MRRA-v3新缓存前冻结`docs/STAGE22_MRRA_V3_ERROR_DIAGNOSTIC_PROTOCOL_ZH.md`：无参数分组、oracle上界和可观测响应路由三层证据分开报告；路由固定chronological fit/embargo/development/embargo/audit划分，audit不再调参。
- `run_stage12_validation_cache.py`新增严格`mrra_v3`身份和checkpoint MRRA键检查；新增两batch smoke/完整validation配置、独立两任务依赖队列、聚合器、启停状态脚本及配对分析器。历史`s4m/pg95/mrba50`身份与行为保持不变。
- 启动前门禁全部退出0：六份Stage22缓存配置42项检查PASS，MRRA-v3分析6项合成测试PASS，专属队列8项PASS，历史Stage22分析8项PASS，通用调度器9项PASS且最大并发5；Python编译、Shell语法和`git diff --check`均PASS。
- 启动前宿主快照：物理4080 UUID `GPU-c88...`空闲16050 MiB、利用率0%；物理4090 UUID `GPU-a3af...`空闲9887 MiB、利用率95%。诊断队列按6144 MiB含安全余量估计选择完全空闲的4080，不干扰4090上的其他用户作业。
- 步骤22.14后台队列已真实启动：scheduler PID/SID=`2649509`、PPID=1、最大并发5；smoke runner PID=`2649515`、CUDA子进程PID=`2649602`，run=`runs/stage22_valcache_mrra_v3__electricity__validation__seed2023__smoke__20260813_144705`。
- NVML确认目标进程在物理4080占用3662 MiB；日志确认加载SHA-256=`1ed4736e...a705e2f`的冻结MRRA-v3 checkpoint，参数为`is_training=0`、`stage22_validation_diagnostic=true`、`max_eval_batches=2`、`use_mrra=true`、v3版本和0.1学习率身份。formal仍为PENDING，test未构造或访问。
# Step 22.15（进行中）：MRRA 内部可观测特征诊断

- 已完成代码入口审计：确认 MRRA 与原型记忆库已有足够的无标签内部量，可用冻结检查点进行诊断，无需训练。
- 已冻结约束：validation-only、时间隔离 fit/dev/audit、诊断默认关闭、smoke 先行、禁止根据 test 修改主模型。
- 下一操作：实现诊断缓存、预注册分析器与独立后台队列，然后运行 smoke → 正式缓存 → 冻结分析。

## 2026-08-13 实施与启动记录

- 新增 21 个逐样本、无标签内部特征，覆盖 MRRA 隐状态/上下文残差/重建修正、历史缺失结构和原型检索统计；诊断开关默认关闭，未改变训练或普通推理路径。
- 新增独立配置、冻结分析器、smoke→formal 非阻塞后台队列以及 start/status/stop 脚本；零 GPU 编译、分析器测试、队列契约测试均通过。
- smoke 第一次失败：变量定义被放入相邻 Stage12 函数，报 `NameError`；已修复作用域并保留失败实验目录。
- smoke 第二次已完成实际前向和缓存，但外层工件审计误走 Stage12 字段分支，报 `KeyError: complete_sse`；已将 `stage22_internal` 纳入 Stage22 审计，并加入内部特征键/形状/有限性检查。
- smoke 第三次成功：`runs/stage22_internal_valcache_mrra_v3__electricity__validation__seed2023__smoke__20260813_151518`，32 个样本、2 个 batch、21 个特征，GPU peak allocated/reserved 约 `2462/3698 MiB`，工件状态 `PASS`。
- 正式全验证缓存已由队列自动启动：队列 PID `2691544`，正式任务 PID `2692838`，GPU UUID `GPU-c88b779d-77c7-92b9-e031-943bae990121`（物理 RTX 4080），当前无 OOM。
- 待自动完成：2528 个验证样本缓存 → time-isolated fit/dev 选择 → audit 单次复核 → 生成步骤22.15 JSON/Markdown 报告。

## 2026-08-13 Step 22.15终态与Step 22.16启动

- Step 22.15队列终态`SUCCESS`；formal run=`runs/stage22_internal_valcache_mrra_v3__electricity__validation__seed2023__20260813_151549`，2528样本、158 batch、工件/协议全部PASS。
- 冻结分析报告=`reports/stage22_mrra_internal_feature_screen.json`；fit/development可行规则数0，决策=`MRRA_ACCURACY_BRANCH_STOP`，没有访问audit标签进行二次选择，也没有访问test。
- Step 22.16开始：冻结MCPR主假设、MSE-only架构对照、Pareto共同下降正式候选及双指标继续门；下一操作为审计PG95 warm-start、训练器参数组与forecast输出接口后实施默认关闭模块。
- 接口审计完成：`load_model(strict=False)`可从PG95 checkpoint载入旧权重并保留零初始化新模块；现有FUSR已有冻结主干专属训练范式可复用。MCPR将使用独立训练分支，不进入原PG95 memory更新/MIPP辅助损失链，避免被冻结参数的隐式状态变化。
- MCPR核心实现、四份smoke/formal配置、中文协议、自动比较器和四任务队列已完成；14项组件/配置/队列门禁PASS，通用调度器9项自测PASS且最大并发5。
- CPU证据：输出投影零初始化时严格复现PG95；一次更新后对不同历史mask产生不同残差；合成PCD方向对MSE/MAE的一阶内积分别为`0.206244/0.134130`，均为正。
- 2026-08-13 17:16宿主队列已启动，scheduler PID=`2905657`。MCPR-MSE smoke runner/train PID=`2905662/2905713`；MCPR-PCD smoke runner/train PID=`2906502/2906582`。两项均在物理4090 UUID `GPU-a3af...`真实运行，formal任务按各自smoke依赖保持PENDING；尚无OOM或结果，test未访问。

## 2026-08-13 Step 22.16终态

- 队列`runtime/stage22_mcpr/state.json`终态=`SUCCESS`，完成时间17:44:37；两个smoke和两个formal共4/4任务全部attempt01成功，无OOM、无失败、无重试。
- 自动汇总`reports/stage22_mcpr_comparison.json/.md`已生成，协议状态=`PASS`仅表示工件和执行合规，科学决策为`MCPR_BRANCH_STOP`。
- PG95锚点MSE/MAE=`0.8728095907/0.7545076541`；MCPR-MSE=`0.8761089148/0.7529857714`（MSE -0.378012%收益，MAE +0.201705%收益）；MCPR-PCD=`0.9158393189/0.7449589941`（MSE -4.930025%收益，MAE +1.265548%收益）。两者都未过MSE/MAE双非劣门。
- 所有协议检查通过：主干和memory冻结、完整2528样本/158 batch chronological validation、新机制实际执行且checkpoint有限；`test_accessed=false`。
- 本轮状态检查首次误用系统`python`被权限拒绝，改用项目Conda Python后又发现检查命令假定了错误的`runtime/stage22_mcpr_queue/queue_state.json`路径；实际路径为`runtime/stage22_mcpr/state.json`。两个问题只影响人工查询，不影响队列或训练。
- 下一步最短路径：Step 22.17先复用冻结工件做MCPR误差迁移/尾部风险归因，不训练、不访问test、不基于当前validation调MCPR参数；归因后再预注册新的机制级候选。

## 2026-08-13 Step 22.17启动

- 用户授权开始下一步；本步固定为validation-only的MCPR失败归因，不训练、不访问test、不基于本次结果调MCPR超参数。
- 资产审计确认三套正式checkpoint和固定验评资产完整；现有MCPR `metrics.json`仅有总体指标，没有逐样本缓存，因此必须扩展只读validation-cache入口。
- 下一操作：审计`run_stage12_validation_cache.py`的方法身份、checkpoint加载和工件schema，然后在读取MCPR新逐样本结果之前写入正式中文预注册协议。
- 入口审计确认可在MCPR检查点的同一次前向中用`候选预测-模型.last_mcpr_outputs['residual']`精确重建冻结PG95主干预测，不需要另外训练或用两个进程近似对齐。
- 已在读取任何新MCPR逐样本结果前写入`docs/STAGE22_17_MCPR_FAILURE_ATTRIBUTION_PROTOCOL_ZH.md`，冻结了数据、方法、MSE精确分解、分组、自助法、协议门禁、解释规则和停止规则。
- 下一操作：对只读缓存入口和S4M诊断前向做最小扩展，然后先运行CPU/静态测试。
- 模型路径审计完成：`last_mcpr_outputs['residual']`在原始视图前向后可用，且与最终预测形状一致；可直接重建冻结主干并计算`delta_sse = alignment_cross + residual_energy`。
- 实施边界固定：只增加`stage22_mcpr_attribution`诊断子开关、`mcpr_mse/mcpr_pcd`严格方法身份和工件字段；原有stage12/stage22/MRRA缓存行为保持不变。
- 已实现诊断子模式、两方法×smoke/formal四配置、精确分解分析器、自动聚合器、四任务UUID动态队列和start/status/stop脚本。
- Step 22.17专属零GPU测试14项全部PASS；AST语法、Shell语法和`git diff --check`均退出0。历史Step 22.16组件14项、Step 22配置41项和Step 22.14分析6项回归也全部PASS。
- 组合回归命令末尾误调用不存在的`scripts/test_stage5_scheduler.py`，导致组合命令最终退出2；前三组已显式PASS的结果不受影响。已确认通用调度器正确测试入口是`run_stage5_baseline_queue.py --self-test`，下一次使用正确入口。
- 通用调度器正确`--self-test`入口9项全部PASS：最大并发5、动态补位、OOM回队列、确定性错误停止、启动预留锁和断点恢复等行为符合要求；Step 22创新配置完整回归也PASS。
- 首次在受限沙箱内运行`nvidia-smi`时无法连接驱动，返回的“PID 12”实为宿主内核线程而不是调度器；已立即改用宿主权限重新启动并四重核验，不把沙箱PID当作启动证据。
- Step 22.17宿主队列已真实启动：scheduler PID/SID=`3415850`、PPID=1，最大并发5，GPU候选为两个固定UUID。启动时物理4080空闲6389 MiB/利用率96%，物理4090空闶148 MiB/利用率100%；因此调度器只在4080上串行补位，没有强行占用或修改实验配置。
- MCPR-MSE 2-batch smoke run=`runs/stage22_mcpr_valcache_mcpr_mse__electricity__validation__seed2023__smoke__20260813_180825`已首次SUCCESS，32样本、2 batch、return code 0、verification error为null，checkpoint含7个MCPR state key。候选/内嵌主干MSE=`1.146040963/1.120861723`，`alignment_cross/residual_energy/delta_mse=0.020618272/0.004560976/0.025179249`，逐样本分解最大绝对闭环误差`1.373291e-4`，所有27项工件检查PASS。smoke结果只证明链路正确，不作科学结论。
- 调度器已将MCPR-MSE smoke置为SUCCESS并自动补位MCPR-PCD smoke；MCPR-PCD smoke在物理4080上RUNNING，MCPR-MSE formal因当前显存不足保持WAITING_RESOURCE，PCD formal保持依赖PENDING。当前无OOM、无重试。
- 下一自动操作：PCD smoke成功后，两个formal依据实时显存动态启动；两者都SUCCESS后自动运行步骤22.17冻结分析器。已确认后台持久运行，按用户规则无需人工持续监控。
- 队列首轮终态=`FAILED`；两个smoke均SUCCESS，两个formal均在完成实际前向并return code 0后，因唯一冗余浮点归约审计超出过紧容差而标为CODE_ERROR；无OOM，test未访问，冻结分析器未运行。
- 已量化根因并修复：保留审计，将每样本30,816个float32元素两种归约顺序的绝对容差从0.002调整为0.008，实测最大0.006348；新容差仍拒绝0.02实质性差异，并将实际差与容差写入metrics。下一步先重跑smoke，再用新ID重跑formal。
- 修复后Step 22.17专属测试从14项扩展为16项并全部PASS；新增测试明确接受实测0.00635浮点差、拒绝0.02实质性差，AST语法也PASS。
- 启动脚本已补充`RESET_JOBS/RESET_REASON`透传，用于对确定性代码修复进行可审计重排。第二轮以`RESET_JOBS=all`启动，原因=`float32_reduction_order_tolerance_0.008_after_measured_max_0.006348`，所有旧成功/失败工件保留。
- 修复后宿主队列PID/SID=`4157354`、PPID=1，以最大并发5持久运行。MCPR-MSE smoke为attempt02并已在物理4080真实运行；MCPR-PCD smoke为WAITING_RESOURCE，两个formal依赖PENDING。快照中4080空闲7165 MiB但已有其他计算任务，4090仅空闱42 MiB，因此当前只能串行补位；无新OOM或实验配置变更。

## 2026-08-13 Step 22.17终态

- 修复后队列于19:19:54自然结束，`queue_state=SUCCESS`；两个smoke与两个formal全4/4新实验ID均SUCCESS，formal run分别为`runs/stage22_mcpr_valcache_mcpr_mse__electricity__validation__seed2023__20260813_191651`和`runs/stage22_mcpr_valcache_mcpr_pcd__electricity__validation__seed2023__20260813_191839`。
- 自动报告`reports/stage22_mcpr_failure_attribution.json/.md`已生成，状态=`PASS`、27项协议检查全部true、`test_accessed=false`，总决策=`MIXED_MCPR_FAILURE_MECHANISMS`。
- MCPR-MSE为幅度过冲：有益对齐项`-0.0025741053`被能量项`+0.0058734267`淹没，净`delta_mse=+0.0032993216`；53.006%样本双改善，最差5%样本贡献80.121%净MSE退化。
- MCPR-PCD为方向对齐失败：对齐项`+0.0182893574`和能量项`+0.0247403761`均有害，净`delta_mse=+0.0430297336`；残差能量是MSE版4.212倍。
- 下一步Step 22.18：预注册一个仅沿MSE残差的训练期信赖域缩放+尾部风险约束模块，首轮确证必须使用未参与Step 22.17归因的新固定缺失资产；不复活PCD，不直接使用validation oracle缩放0.219。

## 2026-08-13 Step 22.18启动

- 已完成步骤22.18.1上下文恢复和22.18.2机制冻结；没有启动新训练，也没有读取test。
- 新方法固定为TTRR-MCPR：八维历史可观测描述子驱动的逐样本残差信赖门，加最坏10%候选相对冻结主干正向MSE伤害约束；尾部权重固定1.0，门零初始化对应0.5，不使用步骤22.17的validation oracle系数。
- 已写入`docs/STAGE22_18_TTRR_PROTOCOL_ZH.md`，冻结新资产、模块公式、公平预算、双锚点门、停止条件和对RADR/PCD的区别。
- 下一操作：生成Electricity seed2024 point-r0.06-L5固定缺失资产并做结构/哈希/跨seed差异审计；通过后实现默认关闭模块与训练目标。
- 步骤22.18.3资产生成命令退出0，28项审计全部PASS；seed2024实际缺失率0.2600364964，与seed2023在train/validation均不同，CSV/NPZ/JSON/manifest哈希闭环且未访问测试标签或模型输出。
- 步骤22.18.4已实现TTRR独立模块、`tail_trust_mse`目标、CLI/runner非法配置门、机制指标和seed2024的控制/候选smoke与formal配置。默认关闭时没有门参数且与历史MCPR state dict完全一致。
- 步骤22.18专属31项CPU/静态门禁、历史MCPR 14项回归、步骤22既有配置60项回归、通用调度器9项自测、Python/Shell编译和`git diff --check`均通过。
- 已建立6任务最大并发5动态队列：PG95只读锚点、MCPR-MSE控制、TTRR-MCPR候选均先smoke再放行formal，支持GPU UUID、flock、显存预留、动态补位、OOM退避回队列、断点恢复和自动双锚点报告。
- 下一操作：核对宿主两张GPU的UUID/名称/显存后启动后台队列，并以scheduler PID、run状态、训练日志和NVML进程四重确认真实运行。
- 首次受限会话启动返回PID 11，但没有state、日志、run或GPU进程，已明确判定为未启动；没有虚报，也没有重复实验工件。随后改用宿主持久方式启动相同队列，配置不变。
- 宿主调度器PID/SID=`761395`、PPID=1，真实持久运行。三项smoke均attempt01 SUCCESS、return code 0、checkpoint有限且可重载、verification error为空。
- PG95 smoke完成后先放行PG95 formal；MCPR-MSE与TTRR smoke随后成功，队列也已自动放行两个训练formal。当前三项formal均已创建唯一run目录，test未访问、无OOM、无失败、无重试。
- formal run分别为`runs/stage22_crossmask_valcache_pg95__electricity__validation__seed2024__20260813_202804`、`runs/mcpr_mse_crossmask_electricity__electricity__point__mr0.06__seed2024__20260813_202819`和`runs/ttrr_mcpr_pg95_electricity__electricity__point__mr0.06__seed2024__20260813_202834`。
- 下一操作：按用户规则不持续人工监控，让后台队列自然完成；下次检查读取终态、三项完整validation指标和`reports/stage22_18_ttrr_comparison.json/.md`，再执行双锚点继续/停止门。

## 2026-08-14 Step 22.18.7复杂度与计时收尾

- 已确认原6任务队列终态SUCCESS，双锚点报告协议PASS；TTRR相对MCPR-MSE的MSE/MAE均退化，科学决策保持`TTRR_CONFIGURATION_STOP`，没有访问test。
- 已修正`aggregate_stage22_18_ttrr.py`成本字段：参数量、formal运行总耗时、allocated/reserved峰值显存、epoch、optimizer step和GPU UUID均可从持久化工件读取，不再显示错误的null。
- 新增默认关闭的`stage22_inference_timing`入口、专属runner、PG95/MCPR-MSE/TTRR各自smoke/formal共6配置，以及单UUID最大并发1的严格串行队列。正式协议为完整2528样本/158 batch、2次重复、每次10个预热batch、CUDA Event模型前向计时；指标必须回算到既有formal值1e-5以内。
- 零GPU门禁全部PASS：6配置与资产审计14项、通用队列9项自测、Python编译、Shell语法、CLI解析和`git diff --check`均退出0；当前报告成本字段已非空，推理计时状态为IN_PROGRESS。
- 启动前宿主GPU快照：4080空闲3711 MiB、4090空闲2562 MiB且两卡利用率均100%；均低于6144 MiB估计，因此队列应进入资源等待而不能抢占或缩小实验。
- 15:35 CST已真实启动后台调度器PID `3989185`，固定RTX 4080 UUID、`max_jobs=1`；宿主`ps`确认进程由PID 1托管并持续运行。首项PG95 smoke状态为`WAITING_RESOURCE`，其余5项依赖排队，当前0次启动/0次OOM/0次失败，符合显存释放后自动补位规则。
- Step 22.18.7队列于17:07:26 CST自然结束，`queue_state=SUCCESS`；PG95/MCPR-MSE/TTRR各自smoke和formal共6/6任务均attempt01 SUCCESS，固定同一RTX 4080 UUID，最大并发1，无OOM、无重试、无失败。
- 三项formal均为validation-only，完整2528样本/158 batch跑2遍，每遍排除前10 batch计时后合计296个CUDA Event前向观测；所有checkpoint、时序loader、冻结性、指标回算和同GPU检查PASS，test未访问。
- 前向均值/P95（ms/batch）：PG95=`100.0245/142.8773`，MCPR-MSE=`110.4301/146.2023`，TTRR=`108.0758/145.0236`；TTRR对PG95均值开销8.049%，MCPR对PG95开销10.403%。MCPR与TTRR的2.132%倒挂处于服务器负载波动内，不作为加速结论。
- 参数量：PG95=`24,118,143`，MCPR-MSE=`24,145,472`，TTRR=`24,145,481`；TTRR仅比MCPR多9参数。allocated峰值分别约2408.46/2409.88/2410.12 MiB，reserved均3678 MiB。
- 步骤22.18至此闭环：复杂度证据充分，但TTRR对MCPR-MSE控制的MSE/MAE均退化，科学决策保持`TTRR_CONFIGURATION_STOP`，不进入test或多seed。

## 2026-08-14 Step 22.19启动

- 已用既有Step 22.17 seed2023缓存完成解析残差动作诊断；没有新GPU前向、模型训练或test访问。可重复报告为`reports/stage22_19_ard_feasibility.json/.md`。
- 解析教师动作比例为OFF/PARTIAL/FULL=`0.414953/0.109968/0.475079`；逐样本解析上界相对PG95 MSE改善0.9653%，证明单一全局缩放会掩盖样本级开关结构。
- 时间隔离fit/development/audit显示明显残差效用漂移；单视图混合探针在development/audit获得0.9614%/0.4731% MSE改善，但动作分类audit准确率仅0.2232。加入嵌套分歧未改善audit收益，因此首轮不增加第二前向。
- 可行性脚本前两次分别因NumPy `.square()`和`test_accessed=false`参与全true检查而在报告落盘前停止；均为CPU分析代码错误，未创建实验run。修复后第三次退出0，8项协议检查全部PASS。
- 已在任何seed2025资产/结果出现前冻结`docs/STAGE22_19_ARD_MCPR_PROTOCOL_ZH.md`：ARD使用训练期观测标签解析教师和三动作—部分系数混合门；第1 epoch只训练原始MCPR，随后蒸馏；同seed2025 PG95/MCPR-MSE/ARD三臂同预算比较，失败不在validation上调参。
- 下一操作：实现默认关闭ARD模块、训练目标与诊断指标，先做CPU组件/配置/checkpoint/基线关闭门禁，再生成并审计seed2025固定资产。

## 2026-08-14 Step 22.19.3 ARD代码接入与第一轮回归

- 新增`baselines/S4M/s4m/model/ARD.py`，在MCPR中以默认关闭开关接入8维无标签特征、16宽隐藏层、OFF/PARTIAL/FULL三动作头和连续部分系数头；TTRR与ARD互斥。
- `exp_pretrain1.py`新增解析教师目标：仅用训练批次观测标签计算逐样本解析系数；主干误差、教师和8维特征停止梯度；raw MCPR由MSE训练，ARD由动作交叉熵和尺度SmoothL1训练，epoch1为raw-only warmup。
- `run.py`和`run_s4m_experiment.py`已加入固定协议参数、非法组合拒绝、机制指标检查与ARD checkpoint键验证；正式runner现在也显式强制TTRR/ARD专属参数存在。
- 项目Conda解释器对6个修改文件`py_compile`退出0，`git -C baselines/S4M diff --check`退出0。
- 历史MCPR 14项、修复配置发现范围后的TTRR 31项、既有Step22创新配置60项检查全部PASS；尚未启动新GPU实验，未访问test。
- 第一次静态检查误用裸`python`，后续已统一为项目Conda绝对解释器。TTRR测试首次失败是旧测试误扫后来加入的计时YAML，只修复了测试发现范围，未改历史配置或结果。

## 2026-08-14 Step 22.19.3完成与22.19.4启动门

- ARD专项检查扩展为39项并全部PASS：合成解析系数严格为`[0,1,0.25]`，动作严格为OFF/FULL/PARTIAL，212个ARD参数的动作头、部分系数头与raw残差均有有限非零梯度，默认关闭逐张量等价，checkpoint严格往返成功。
- warmup首轮测试唯一失败是把相消得到的零梯度张量误要求为`None`；修正为“无梯度或严格全零”后通过，训练实现未改。
- seed2025 Electricity资产已生成并纳入manifest，实际缺失6839/26304行、率0.2599984793，连续段5–20，观测值误差0；33项结构/哈希/可复现/跨seed审计全部PASS。
- seed2025 mask相对seed2023/2024分别有10012/10165行不同，训练段与validation段均不同；协议文档时间早于资产生成，未访问test指标或模型输出。
- 已创建PG95只读锚点、MCPR-MSE控制、ARD-MCPR候选各smoke/formal共6配置；两训练formal除模块身份/目标/日志名外官方参数差异为空，均为15 epoch、batch16、相同Adam/学习率/主干checkpoint/validation协议。
- 控制组初始身份名含`ard`而被配置门正确拒绝；只改为不含`ard`的控制身份，未改任何实验参数，重检通过。
- 6任务动态队列、start/status/stop及自动双锚点汇总器已完成；调度器9项假任务自测PASS，最大并发实测5、OOM回队列/动态补位/依赖/断点恢复/确定性错误停止均正常。
- 所有Python编译、Shell语法、ARD 39项、MCPR 14项、TTRR 31项、资产33项和`git diff --check`均退出0；聚合器在尚无formal时正确报告`IN_PROGRESS`。
- 下一操作：查询两张GPU的index/UUID/名称/空闲显存与利用率，随后用稳定UUID后台启动队列；三项smoke可以在资源允许时并行，formal必须各自等待对应smoke成功。
- 启动前宿主快照：物理0 RTX 4080（UUID `GPU-c88b...`）空闲9541 MiB、利用率97%；物理1 RTX 4090（UUID `GPU-a3af...`）空闲4446 MiB、利用率100%；无重复22.19进程。
- 17:59 CST已通过宿主持久方式启动调度器，PID/SID=`1583332`、PPID=1、`max_jobs=5`。首次状态为`RUNNING`，PG95 smoke attempt01已在4080启动，runner PID=`1583369`，对应项目GPU进程已占用424 MiB并继续加载；MCPR/ARD smoke因当时安全余量不足处于`WAITING_RESOURCE`，没有OOM或失败。
- 按用户规则，确认真实运行后不持续人工监控；队列会在资源释放时自动补位，并在各smoke成功后放行对应formal。下次检查直接读取持久化state、run、metrics和自动双锚点报告。

## 2026-08-16 Step 22.19.4–22.19.5终态核查

- 持久化state显示队列于2026-08-14 19:02:26 CST自然结束，`queue_state=SUCCESS`；PG95、MCPR-MSE、ARD-MCPR各smoke/formal共6/6任务均attempt01 SUCCESS，return code 0，无OOM、无重试、无失败，调度器PID已退出。
- 三项formal均完成2528个validation样本/158 batch；相同seed2025固定mask、相同冻结PG95来源、MCPR/ARD同训练预算、checkpoint有限与可重载、主干和memory哈希不变，11项协议检查全部PASS，`test_accessed=false`。
- 完整validation：PG95=`0.879107853/0.756513116`，MCPR-MSE=`0.887730208/0.756893787`，ARD-MCPR=`0.880757112/0.756085279`（MSE/MAE）。
- ARD相对MCPR-MSE的MSE/MAE分别改善`0.785497%/0.106819%`，通过控制组门；相对PG95为MSE退化`0.187606%`、MAE改善`0.056554%`，未通过双指标门。自动决策=`ARD_BEATS_MCPR_CONTROL_BUT_NOT_PG95_STOP`。
- 解析教师仍明显三态：OFF/PARTIAL/FULL=`51527/14756/60501`，alpha均值/标准差=`0.536646/0.478759`；但学生门均值=`0.530497`、平均batch内标准差仅`0.0008127`，范围`0.451692–0.584840`，说明门接近常数缩放，未学到教师的样本级离散结构。
- ARD把raw残差能量`0.00362178`压到应用能量`0.00102910`，因此有效修复了MCPR的幅度过冲；但近常数约0.53缩放不足以保住PG95的MSE优势。
- ARD/MCPR总参数=`24,145,684/24,145,472`，ARD仅增加212参数；峰值allocated=`2476.00/2471.58 MiB`、reserved均3700 MiB。ARD因早停于8 epoch、MCPR于7 epoch，总耗时分别2071.87/1500.95秒，不能当作严格同epoch推理速度比较。
- 步骤22.19按预注册规则停止ARD-v1：不进入多seed、不访问test。下一步应先诊断8个全局标量特征为何无法承载三动作教师，再冻结结构化时序/变量残差表示，并使用新seed做新的确证性验证。

## 2026-08-16 Step 22.20启动

- 用户要求继续下一步；已将22.20拆为协议冻结、只读逐样本缓存、smoke/formal动态队列、时间隔离分析四个有编号子步骤。
- 诊断问题固定为区分两个故障源：现有8个全局标量本身不可分，或三动作概率与partial系数在期望混合时发生抵消。当前不训练新模型、不补ARD-v1多seed、不访问test。
- 下一操作：审计既有validation/timing入口与ARD forward工件，确定最小只读扩展；随后先写协议文档，再实现代码。
- 已在新增逐样本结果前写入`docs/STAGE22_20_ARD_INFORMATION_BOTTLENECK_PROTOCOL_ZH.md`，冻结单次前向、108维无标签结构特征、完整/观测解析教师、时间隔离切分、固定逻辑/岭探针和两个故障判据。
- 下一操作：在现有validation-only入口增加ARD只读缓存模式，先完成CPU公式/形状/配置/checkpoint测试。
- 步骤22.20.2已完成第一段实现：`run.py`增加与其他诊断互斥的ARD只读入口；`exp_pretrain1.py`增加完整/观测两种逐样本解析教师、108维无标签结构特征和原子NPZ/JSON落盘。不训练、不更新memory、不读test。
- 项目Conda解释器对两个修改文件`py_compile`退出0，`git diff --check`退出0。新增`scripts/test_stage22_20_ard_diagnostic.py`，6项CPU公式/掩码/形状/时间信息/异常输入合约全部PASS。
- 下一操作：从既有seed2025 ARD formal工件生成完整可还原的smoke/formal诊断配置和runner，加入checkpoint身份、原指标回算和validation-only硬门禁。
- 步骤22.20.2已完成：新增两份冻结配置、只读runner、时间隔离逻辑/岭探针分析器、smoke依赖formal的UUID动态队列和start/status/stop入口。
- 启动前门禁全部PASS：Step22.20专项11项（包括完整分析决策合成路径）、旧ARD 44项、MCPR 14项、TTRR 31项、调度器9项假任务自测以及Python/Shell静态检查均退出0。
- 物理RTX 4080启动前空闲16072 MiB，RTX 4090空闲5875 MiB；队列按稳定UUID选择4080。smoke run=`runs/stage22_20_ard_information__electricity__validation__seed2025__smoke__20260816_173609`已SUCCESS，48样本/3 batch、return code 0、无OOM，19项工件检查全部true，MSE/MAE=`1.134902368/0.847376235`（smoke不作科学结论）。
- formal run=`runs/stage22_20_ard_information__electricity__validation__seed2025__20260816_173640`已自动放行，状态RUNNING，训练子进程PID=`496524`，GPU UUID为RTX 4080的`GPU-c88b...`。这是单次冻结前向，不更新模型/memory且不访问test。
- 下一自动操作：formal成功后调度器运行`analyze_stage22_20_ard_information.py`，生成`reports/stage22_20_ard_information.json/.md`并按预注册门判定特征瓶颈与混合抵消；已确认真实运行，现按用户规则不持续监控。

## 2026-08-16 Step 22.20终态与Step 22.21启动

- Step22.20队列于17:37:11 CST自然结束，`queue_state=SUCCESS`；smoke/formal 2/2均attempt01 SUCCESS、return code 0，无OOM/重试/失败。formal完成2528样本/158 batch，19项工件和11项分析协议检查全部true，MSE/MAE=`0.880757116/0.756085279`与源ARD formal在`1e-5`内回算，`test_accessed=false`。
- 结构化108维相对global8在development的动作平衡准确率/alpha Spearman/反事实MSE变化=`-0.009379/-0.015985/+0.00008609`，audit为`-0.021088/-0.375179/+0.00022634`；特征瓶颈的三类预注册条件全部失败。
- 学生gate标准差`0.001237`，FULL/PARTIAL贡献标准差仅`0.003245/0.004455`，虽两贡献Pearson=`-0.997746`，但未达任一分量标准差≥0.01条件；混合抵消亦未确认。自动决策=`ARD_ROUTE_STOP_NO_CAPACITY_EXPANSION`，不实现ARD-v2。
- 训练式观测教师与完整目标教师动作一致率`0.798655`、alpha Spearman=`0.847703`，说明教师有噪声但不是完全失真；更关键的是该关系在后段audit强烈漂移，同一validation不支持扩大门网络。
- 下一主线冻结为Step22.21：不再迭代ARD/MCPR，把唯一经Weather三seed双指标支持的PG95+多尺度CNM-SE迁移到全新Traffic协议。先同预算训练S4M/PG95并仅用validation判断跨掩码预测响应与融合互补；预注册门失败则禁止test。
- 当前进入22.21.1：Traffic完整CSV已存在并纳入资产manifest，尚无项目内固定缺失CSV/NPZ和训练配置；下一操作是审计上游Traffic参数与当前加载器，先写结果前协议再生成seed2026资产。
- 上游README只公开Weather命令，没有Traffic专属训练超参；上源和可编辑副本都明确将文件名含`traffic`映射为862变量。因此Step22.21不宣称严格官方Traffic超参复现，而是在两方法间公平的固定外部协议。
- `Dataset_Custom4`继续使用0.7/0.1/0.2时间分割，且已支持`true_data_path`显式完整CSV，不需要修改数据加载逻辑。Traffic方法对照将共享同一资产、Adam、15 epoch、batch16、early stopping、评价代码和checkpoint选择标准。
- Traffic完整数据审计为17544行×862数值变量，时间从2016-07-01 02:00到2018-07-02 01:00，数值缺失0、重复时间0。Step22.21冻结为point-r0.06-L5的项目外部协议，目标实际缺失行率0.26、生成seed2026；这是公开代码规则的近似协议，不冒充论文未公开的Traffic mask。
- Traffic S4M/PG95架构冻结为862变量、输入192/预淋96、8层、`d_model=d_ff=en_conv_hidden_size=512`、Adam lr0.01、batch16、15 epoch、patience3；这是从已冻结的Electricity大型多变量协议迁移，不在Traffic结果上搜索。
- 多尺度保持既有结果前规则：每样本严格等实际新增缺失数，L1/L5/L20，但最终只融合`0.5*original+0.25*L5+0.25*L20`；validation必须同时满足PG95相对S4M双指标不劣且至少一项严格改善、多尺度相对PG95 original和original+L5均双指标严格改善，否则test禁止。
- 已在任何Traffic缺失资产或模型结果前写入`docs/STAGE22_21_TRAFFIC_EXTERNAL_PROTOCOL_ZH.md`，冻结数据、seed2026、862变量架构、同预算训练、严格等移除多尺度和条件test门。validation预期正式评估103 batch/1648样本，test为213 batch/3408样本。首次smoke在DataLoader构造阶段揭示目标列名审计错误，已透明修订为CSV真实最后列`OT`；无GPU前向/指标，其他协议均不变。
- 固定Traffic资产生成命令退出0：seed2026、L5、1021个起点产生4559个缺失行，实际率`0.2598609211`，连续段5–18。CSV/NPZ/JSON哈希已写入manifest；下一项是独立重生/观测值/split审计，尚未启动GPU。
- Traffic独立资产审计已完成：21项全部PASS，mask与起点seed2026逐值重生一致，862变量共享同一缺失行，观测值最大往返误差`1.11e-16`；validation=`103 batch/1648样本`，test=`213 batch/3408样本`，模型test未访问。
- 已冻结并实现7任务链：S4M/PG95各自2-batch训练smoke后放行15-epoch formal；PG95 formal后做3-batch多尺度smoke和完整validation；最终条件test同时依赖S4M formal与多尺度validation formal。
- 为预注册视图塌缩门新增逐样本`prediction_difference_mse_l1/l5/l20`缓存、均值及相对original预测MSE比值；它只增加诊断工件，不改变任何预测、融合权重或训练目标。历史Stage18与Stage21回归全部PASS。
- Step22.21零GPU总门禁已通过：Python/Shell编译、16项Traffic管线检查、5个负例、精确checkpoint交接、历史Stage18/Stage21回归及通用调度器9项自测均PASS；最大并发5、动态补位、启动预留、OOM回队列和确定性错误停止均有效。
- 资产报告首次因NumPy布尔量无法JSON序列化而退出；显式转原生bool后全量重跑PASS。一次历史回归命令误设嵌套仓库工作目录，四项在加载前路径失败；纠正根目录后均PASS，未产生GPU run或结果污染。
- 下一操作：记录GPU index/UUID/名称/总量/已用/空闲快照后启动Step22.21后台队列；先确认两项真实smoke进程/状态，再按全局规则停止持续监控，由队列自动完成formal、validation门和条件test/安全跳过。
- 首次后台队列确实依次启动S4M与PG95 smoke，但两者均在`Dataset_Custom4.cols.remove(target)`处、模型前向前退出`CODE_ERROR`；失败run和日志已保留，所有formal/validation/test依赖均阻断。根因是CSV列为`0..860,OT`而非先前记录的`0..861`，不是模型、显存或数据内容错误。
- target修订后的22项资产审计、16项管线检查、5个负例、Python/Shell编译和基线Git diff检查重新全部PASS；以审计原因`traffic_target_schema_fix_before_model_forward`重置失败/阻断任务，attempt计数与旧run均保留。
- 第二轮S4M/PG95 smoke已并行真实运行并均SUCCESS、return code 0、checkpoint有限且可重载、verification error为空。run分别为`runs/s4m_traffic__traffic__point__mr0.06__seed2026__smoke__20260816_182836`和`runs/mipp_pg95_traffic__traffic__point__mr0.06__seed2026__smoke__20260816_182836`。
- smoke日志已确认862维前向张量`[16,6,32,862]`、两epoch/每epoch两训练batch、三validation batch和checkpoint重载链路；smoke MSE/MAE仅作工程证据，S4M=`1.473396/0.903037`、PG95=`1.607148/0.970648`，不用于科学门。
- 队列已自动并行放行两项正式训练：PG95 run=`runs/mipp_pg95_traffic__traffic__point__mr0.06__seed2026__20260816_182923`固定RTX 4090；S4M run=`runs/s4m_traffic__traffic__point__mr0.06__seed2026__20260816_182923`固定RTX 4080。两者已进入首epoch真实前向，完整15-epoch/early-stopping预算未缩减。
- 下一自动操作：两项formal成功后，PG95依赖链先做多尺度validation smoke再做完整103 batch validation，随后分析器执行预注册PG95/视图响应/多尺度门；只在全部通过时运行一次test，否则写安全跳过工件。已确认真实运行，按用户规则停止持续人工监控。
- 两项formal并未持续运行：S4M在epoch1/batch27、PG95在epoch1/batch33分别出现`forecast_loss=nan`，队列于18:30:39 CST以`FAILED`终止；后续validation/test均为`BLOCKED_DEPENDENCY`，当前没有实验进程。上一轮只确认启动而未及时汇报约一分钟后的终止，已纠正状态。
- 步骤22.21.4无GPU根因审计完成：完整真值、缺失资产观测值、`Dataset_Custom4`全部训练数组和掩码均有限；11993个训练窗口目标观测数最少34480、空目标窗口0；归一化尺度最小0.010674、零尺度变量0，排除数据非有限、空mask和归一化零除。
- 两方法共同的Adam lr0.01是当前最可信发散源；公开代码无Traffic专属超参，README Weather示例为0.005、上游参数默认值为0.001。建议将两方法共同修订为Adam lr0.001，其他协议不变，先做1 epoch/64 train-batch扩展稳定性smoke，再生成新ID重启formal；按禁止静默修改关键配置规则，等待用户明确确认。
- 诊断已保存为`reports/stage22_21_traffic_failure_diagnosis.json/.md`；旧失败run、配置和日志全部保留，未盲目重试，未访问test。
- 用户已明确授权步骤22.21.5：S4M/PG95共同把Adam学习率从0.01修订为上游默认0.001，其他正式协议不变。两项smoke改为1 epoch/64 train batch/3 validation batch，以覆盖原batch27/batch33失败点；下一操作为零GPU配置/负例/队列门禁，通过后以可审计reset原因重启完整依赖链。
- 步骤22.21.5启动前门禁全部PASS：Traffic管线17项、资产22项、历史等实际缺失率多尺度回归、5个负例、通用调度器9项自测、Python/Shell语法和Git diff检查均退出0。启动快照中物理4080空闲16072 MiB、4090空闲24216 MiB。
- 两项扩展稳定性smoke已并行SUCCESS且均return code 0：S4M run=`runs/s4m_traffic__traffic__point__mr0.06__seed2026__smoke__20260816_184813`，PG95 run=`runs/mipp_pg95_traffic__traffic__point__mr0.06__seed2026__smoke__20260816_184813`。两者均完成63个optimizer step、1次memory warmup、checkpoint保存/重载和3-batch validation，无NaN、verification error为空、test未访问。
- smoke工程指标S4M MSE/MAE=`0.571377429/0.373345765`，PG95=`0.557477337/0.375439071`；它们只证明64-batch链路稳定，不作为科学优劣结论。reserved峰值S4M/PG95=`11872/16448 MiB`。
- 调度器已自动补位两项15-epoch formal：S4M run=`runs/s4m_traffic__traffic__point__mr0.06__seed2026__20260816_184930`固定RTX4080，PG95 run=`runs/mipp_pg95_traffic__traffic__point__mr0.06__seed2026__20260816_184930`固定RTX4090。命令和日志均确认Adam lr0.001；S4M已到batch100、PG95已到batch50且loss有限，均越过旧batch27/batch33发散点。
- 后台scheduler PID/SID=`1470677`、PPID=1，queue_state=`RUNNING`，两项训练PID=`1488445/1488435`并有真实GPU显存占用约12196/16944 MiB。下一自动操作为formal完成后启动PG95多尺度validation smoke→完整validation→预注册门→条件test/安全跳过；已确认真实运行，按规则不持续人工监控。
- 步骤22.21.6队列于19:23:54 CST自然结束，7/7任务状态均SUCCESS；S4M/PG95 formal与多尺度validation真实运行成功，最后一个任务是validation门安全跳过test的合规工件，测试集未访问。
- S4M validation MSE/MAE=`0.536970827/0.316911549`；PG95=`0.528318361/0.315015020`，分别提升1.61135%/0.59844%；多尺度=`0.522464090/0.311083244`，相对PG95再提升1.10809%/1.24812%，相对S4M总提升2.70159%/1.83910%。PG95门、视图响应门和多尺度门全部通过。
- 总协议门唯一false为`frozen_full_training_budget`。核验源码确认分析器仍硬编码旧`learning_rate == 0.01`；实际两臂resolved config均为已授权lr0.001，batch16、max15 epoch、patience3、完整2992 optimizer steps和103-batch validation均一致。这是分析器一致性bug，不是训练预算漂移。
- 用户授权步骤22.21.7：只修复分析器预算检查并增加旧lr/缺失修订身份/epoch漂移负例；门通过后仅重排原条件test任务，不重训或修改任何模型与融合配置。
- 步骤22.21.7专项19项、资产22项、历史等实际缺失率回归和静态检查全部PASS；分析器重新计算后24项协议检查全true，决策=`TRAFFIC_VALIDATION_PASS_ONE_TEST_AUTHORIZED`。
- 仅重排条件test任务，scheduler PID/SID=`2265268`、PPID=1；RTX4090上的唯一test run=`runs/cnm_se_equal_rate_multiscale_pg95__traffic__test__seed2026__20260816_194746`于19:49:00 CST SUCCESS，return code 0，无OOM/重试/失败。
- test完整执行213 batch/3408样本，全部artifact check为true，checkpoint精确交接、模型/memory不更新。PG95 original MSE/MAE=`0.672083688/0.361379550`，多尺度=`0.665930768/0.358186681`，分别提升0.915499%/0.883522%；相对original+L5再提升0.350510%/0.444991%。
- 自动终态审计13项全部PASS，决策=`TRAFFIC_TEST_MULTISCALE_CONFIRMED`；历史安全跳过工件保留，真实test run严格只有1个，授权决策已写入attempt日志。
- 修正Traffic原始汇总器对评估任务`source_split/uses_test_split/test_batches_processed/sample_count`的字段映射，并加入自动test收益与决策；`reports/stage22_21_traffic_raw_results.md`现正确显示test访问、213 batch和3408样本。
- Step22.21正式闭环：Weather三seed正向主线在新Traffic数据上获得单seed validation和唯一test双指标确认。下一步Step22.22冻结Traffic seed2027/2028并做同协议多seed复核，不再调整当前模型或融合规则。

## 2026-08-20 Step 22.22启动

- 用户授权开始Traffic三seed统计复核；本步使用文件规划技能维护`task_plan.md/findings.md/progress.md`，技能只影响持久化进度记录，不改变实验协议。
- 已完成会话恢复检查，`session-catchup.py`无未同步输出；Step22.21终态工件、队列SUCCESS和下一阶段计划一致。
- Step22.22拆分为结果前协议冻结、seed2027/2028资产、配置/队列、零GPU门禁与后台启动、三seed统计汇总五个子阶段。
- 当前下一操作：审计seed2026资产生成脚本、Traffic训练/多尺度评估入口和队列依赖，确定是否需要补齐S4M测试原始路，随后在生成任何新seed资产前写入中文预注册协议。
- 接口审计发现：Traffic生成器硬编码seed2026但底层算法可复用；评估入口把Traffic限定为PG95且允许seed仅2026。Step22.22必须显式参数化资产和扩展seed门，禁止伪装seed。
- 为得到论文可用的三方法test表，当前PG95多尺度工件已提供PG95 original和CNM-SE；S4M test尚无合法只读身份。下一操作是在协议中冻结是否补S4M原始路，并审计现有Stage19 S4M只读测试模式能否在不改变模型/指标的情况下安全推广到Traffic。
- 已确认Stage19的source/test一致性检查不适用于Traffic validation checkpoint，不能伪装复用；评估数学实现本身可在关闭MIPP时运行。
- 已冻结实现方向：新增Step22.22 Traffic-S4M显式只读测试身份；PG95 validation/test延展允许seed2027/2028；不改模型、掩码或指标公式。
- 已冻结报告规则：seed2027/2028无条件全部进入validation/test；最终三模型主表仅使用S4M original、PG95 original、PG95+多尺度CNM-SE。
- 下一操作：写入结果前协议文档，然后生成并审计seed2027/2028固定资产。
- 步骤22.22.1已完成：`docs/STAGE22_22_TRAFFIC_MULTISEED_PROTOCOL_ZH.md`在新资产/新模型结果前落盘，冻结22任务集合、无条件test纳入、三模型主表、均值±样本标准差和五类决策标签。
- 下一操作进入22.22.2：实现seed参数化资产生成入口，生成seed2027/2028 CSV/NPZ/JSON后做独立重生与跨seed审计。
- 已新增参数化且只允许预注册seed2027/2028的资产生成器，以及覆盖完整哈希、逐值重生、观测值、split覆盖、manifest和三seed两两差异的独立审计器；Python静态编译通过。
- 资产生成命令已提交执行；首次30秒返回窗口没有带回stdout/退出码，下一操作先核验进程与原子输出是否已完成，禁止盲目重复生成。
- 核验确认生成器已自然完成且无残留进程：seed2027/2028两份约194 MiB CSV、NPZ和JSON均已原子落盘；实际缺失率分别为0.2600889193和0.2599749202。
- 已把两份资产的实际CSV/NPZ SHA256、seed、实际缺失率和生成器写入manifest。下一操作运行独立三seed资产审计；通过后完成22.22.2。
- 步骤22.22.2完成：`scripts/test_stage22_22_traffic_assets.py`退出0，35项检查全部PASS，`model_test_split_accessed=false`；三seed在完整、validation和test区间均两两不同。
- seed2026/2027/2028实际缺失行分别4559/4563/4561，实际率0.2598609211/0.2600889193/0.2599749202；报告为`reports/stage22_22_traffic_asset_test.json`。
- 下一操作进入22.22.3：复用步骤22.21训练与评估实现，新增显式S4M test身份、seed参数化配置物化、22任务依赖队列与三seed自动统计。
- 已完成Step22.21代码复用审计：现有pipeline硬编码seed2026和条件test门，不能用于本次无条件复核；现有评估数学实现可复用，但需新增S4M Traffic test身份并扩展PG95允许seed。
- 下一操作：先修改评估器的最小身份/核验分支，再实现Step22.22配置生成器、pipeline和22任务队列。
- 评估器最小扩展已实现：新增`stage22_22_traffic_s4m_test`，S4M源关闭MIPP且不要求MIPP checkpoint键；Traffic允许seed2026/2027/2028，PG95历史分支语义保持不变。
- 配置物化设计已冻结：seed2026的S4M源从Step22.21旧队列状态精确引用；seed2027/2028从Step22.22当前队列动态交接，不复制checkpoint、不猜路径。
- 下一操作：用可审计生成器机械派生全部配置，再实现无validation门控的动态pipeline和22任务队列。
- 22份配置已由结果前生成器机械派生并落盘；新增pipeline支持从模板的`source.job_id`和可选`source.queue_state`精确交接validation-only checkpoint，formal评估拒绝smoke checkpoint。
- 22任务队列、start/status/stop入口已实现，最大并发默认5；seed2026的2项只读S4M test任务引用旧成功run，seed2027/2028的20项按训练与评估smoke依赖自动补位。
- seed2026成本字段审计确认：S4M/PG95参数量28371290/30663702，训练峰值8007/12174 MiB；多尺度评估峰值3087 MiB，test串行original前向均值58.142 ms，三嵌套前向各约57.84 ms。
- 下一操作：实现可在队列运行中增量更新、在22任务完整后自动给出三seed主表与预注册决策的汇总器。
- 增量汇总器已实现：复用seed2026正式工件，自动读取新队列，按validation/test分别形成S4M、PG95、CNM-SE三方法主表、逐seed配对收益、均值±样本标准差、胜出数、参数/显存/训练时间/推理时间和预注册决策。
- Python/Shell静态与22配置验证命令已提交；首次10秒返回窗口未带回stdout/退出码，下一操作先确认测试进程和结果，避免重复并发执行同一全量资产验证。
- 首次验证返回窗口的问题是临时内联测试未设置`scripts`导入路径，属于测试命令环境错误，无GPU run、无配置修改；修正`PYTHONPATH=scripts`后8份训练配置和14份评估配置全部通过各自正式validator。
- 下一操作：增加Step22.22专项正/负例、checkpoint交接、队列依赖和统计决策测试，再运行历史Step22.21回归与通用调度器假任务自测。
- Step22.22专项19项检查全部PASS：22配置/22队列任务、两seed训练公平性、S4M显式身份、无条件PG95 val/test、validation-only checkpoint精确交接、4个负例和5类决策优先级均通过。
- 历史Step22.21管线19项回归全部PASS，证明Traffic允许seed扩展和S4M新分支没有破坏seed2026条件test语义。
- 已定位通用调度器自测入口为`run_stage5_baseline_queue.py --self-test`；下一操作运行该假任务自测，并补充聚合器合成完整三seed自测/历史等实际缺失率回归。
- 聚合器端到端合成测试8项全部PASS：完整三seed纳入、18行主表、样本标准差、成本比值和strong-chain决策均正确；通用调度器9项假任务自测PASS，实测最大并发5、动态补位、OOM一次回队列成功、确定性错误不重试、resume和启动预留竞争均正常。
- 项目顶层不是Git仓库，因此顶层`git diff --check`不可用；这不影响每个run由runner记录两个实际Git子仓库的commit/diff。后续静态检查改为对实际baseline Git仓库执行并单独检查新增文件语法。
- 下一操作：运行Stage18/19/21等实际缺失率历史回归、资产审计与baseline Git diff检查，然后做启动前GPU快照。
- 步骤22.22.3完成；步骤22.22.4零GPU总门禁完成且总命令退出0：Stage18组件9项/配置11项、Stage19配置13项、Stage21训练53项/管线11项、Step22.22资产35项/管线19项/聚合器8项、调度器9项全部PASS，Python/Shell静态检查和baseline Git diff检查通过。
- 下一操作：打印物理GPU index/UUID/名称/总量/已用/空闲/利用率，核对反向映射，然后以最大并发5启动后台队列并确认首批真实smoke进程、run目录和GPU占用。
- 启动前宿主快照：物理0号RTX4080 UUID=`GPU-c88b...`总16376/已用4/空闲16072 MiB、利用率0%；物理1号RTX4090 UUID=`GPU-a3af...`总24564/已用12739/空闲11477 MiB、利用率75%。编号与项目固定反向逻辑映射一致，队列使用UUID避免混淆。
- 沙箱内首次`nvidia-smi`因设备隔离无法通信；获准的宿主只读查询退出0，确认不是驱动故障。
- 同评估器在Step22.21实测reserved峰值3932 MiB；据此把Step22.22评估任务启动估计从保守16384校准为6144 MiB，仍保留超过2 GiB安全余量。该变化只影响调度可启动性，不改batch、模型、权重、数据或正式协议。
- 下一操作：重跑队列专项依赖/调度自测后启动后台队列，预计4090先容纳S4M只读test smoke，4080并行一个S4M训练smoke；PG95等待4090释放或足够显存。
- 资源估计修订后的专项19项与调度器9项自测再次PASS；后台启动命令退出0，scheduler PID/SID=`3438090`，最大并发5，UUID候选顺序为4080后4090。
- 首次状态为RUNNING：1个任务运行、4个WAITING_RESOURCE、17个PENDING。调度器选择空闲最多的4080运行seed2026 S4M test smoke；4090只余11477 MiB，不够14336/20480 MiB训练任务，因此首批实际并发为1而不是预计2，配置未被缩减。
- 下一操作：确认seed2026 S4M test smoke已创建真实run、模型子进程和GPU显存占用；确认后完成22.22.4并按用户规则不持续监控，等待队列自动补位。
- 已确认精确交接到旧seed2026 S4M formal checkpoint并创建真实run：`runs/cnm_se_equal_rate_multiscale_s4m__traffic__test__seed2026__smoke__20260820_145817`；队列job已记录run_dir，非占位任务。
- 随后的瞬时GPU快照未观察到显存占用，可能处于CPU数据加载或已完成但调度状态尚未在15秒轮询中刷新；下一操作读取run状态/训练日志和队列新一轮状态，必须确认真实forward或明确错误后再结束本轮。
- run状态确认训练子进程PID=`3442494`且为RUNNING；日志已打印完整862维配置、`MODEL_FOOTPRINT parameters=28371290`、使用RTX4080 UUID以及精确checkpoint加载语句，证明已进入真实模型评估入口。
- checkpoint加载后的宿主快照显示4080显存从4 MiB升至474 MiB；此时仍处于模型/数据准备早段，尚不能把smoke记为SUCCESS。下一操作做一次队列状态刷新，若已成功则确认formal自动放行；若仍运行则按规则结束持续监控并由队列继续。
- seed2026 S4M test smoke已真实完成3 test batch/48样本，run状态SUCCESS、return code 0、verification error为空；22项artifact checks全部true，checkpoint有限且可重载，峰值allocated/reserved=`3084.44/3936 MiB`，耗时65.82秒。
- 步骤22.22.4完成，进入22.22.5。队列轮询快照仍可能短暂显示该job为RUNNING，但run终态已落盘；scheduler将在下一15秒轮询中归档SUCCESS并自动放行seed2026 S4M完整test及其他资源可用任务。
- 按用户全局规则，真实smoke成功且后台队列可自动补位后不持续人工监控。下一次状态检查应读取队列终态/失败分类；全部完成后汇总器自动生成JSON/Markdown/CSV三seed报告。

## 2026-08-20 Step 22.22终态检查

- 用户要求检查是否跑完、汇报结果，并在完成时直接继续下一步；本轮重新启用文件规划技能并完成会话恢复检查，未发现未同步上下文。
- 当前下一操作：只读检查scheduler/state、22项job终态、聚合报告和运行进程；若完整则做最终审计并据预注册决策进入Step22.23，否则先按错误分类处理。
- Step22.22队列终态=`SUCCESS`，22/22任务全部首次attempt成功，无OOM、重试或错误，scheduler PID已自然退出；所有三seed正式来源和自动JSON/Markdown/CSV报告齐全。
- 冻结主结论：CNM-SE相对PG95在validation和test均3/3 seed双指标改善；test平均相对改善MSE=`0.859189%±0.074102%`、MAE=`0.777161%±0.094110%`，自动决策=`TRAFFIC_MULTISEED_CONFIRMED`。
- 完整系统相对S4M的test均值改善MSE=`1.386230%`、MAE=`1.174803%`，但MAE仅2/3 seed胜出；PG95相对S4M也仅2/3 test seed双指标方向一致。因此不能升级为`TRAFFIC_MULTISEED_STRONG_CHAIN_CONFIRMED`。
- 串行多尺度test相对单次PG95前向的耗时比为4.052x/4.154x/4.124x，准确率收益稳定但效率代价明确。下一操作：新增独立终态审计，逐run复核状态、配置、样本、统计和决策，然后冻结Step22.23的最短高价值方向。
- 首次新增终态审计脚本的单次大补丁在写入新文件时失败，未留下目标文件、未修改实验工件；当前未获得具体磁盘错误文本。按文件规划规则不原样重试，下一操作先检查目标路径、磁盘/配额和是否存在同名文件，再拆分为较小补丁。
- 路径检查发现目标文件为0字节且项目盘仅余约2.1 GiB（inode充足）；拆分小补丁后审计脚本成功写入并通过Python编译。
- 首次终态审计17/18项PASS，唯一失败为`first_attempt_no_errors`；22项run文件、状态、资源、checkpoint交接、CSV/均值/样本标准差/配对收益和预注册决策均已复算通过。该失败与先前状态摘要显示全部attempt=1矛盾，判断为审计器读取attempt字段名错误，下一操作只读确认state实际字段后修复审计器，不触碰实验结果。
- 已确认scheduler字段为`attempt_count`而非`attempt`，且抽查job均为`attempt_count=1, oom_retry_count=0`。终态审计器仅修复字段映射并显式增加OOM次数为0要求；实验run、metrics和科学决策未改变。
- 字段映射修复后终态审计18/18项PASS，决策=`TRAFFIC_MULTISEED_CONFIRMED`。
- 在规划Step22.23时发现成本汇总器把L1审计路也累加到`serial_multiscale_pipeline_mean_ms`；但冻结最终预测只融合original/L5/L20，L1明确不进入推理。该问题只影响成本标签，不影响任何预测或精度。
- 已修正成本汇总为original+L5+L20+融合，并加入`excludes_l1_audit_only_route=true`及合成/终态回归要求；下一操作重跑聚合器、合成测试和终态审计，得到正确部署开销。
- 成本修订后的聚合器合成测试、实际汇总和终态审计均退出0；正确三seed串行部署比为3.02355x/3.08197x/3.03841x，19项终态审计全部PASS，Step22.22正式完成。
- Step22.23开始：已在读取E1/E5/E20三seed候选汇总前写入`docs/STAGE22_23_EFFICIENT_CNM_SE_PROTOCOL_ZH.md`，冻结validation-only选择、双指标3/3胜出、双指标各保留≥80%多尺度平均收益、固定打破平局规则和已有test事后边界。
- 若validation与已有test探索门通过，唯一确认路径为候选冻结后新生成seed2029，只训练PG95并预先授权完整test；磁盘约束纳入协议，不创建S4M checkpoint或冗余大预测缓存。
- 下一操作进入22.23.2：实现不读取test参与排名的分析器和合成负例，随后从既有三seedvalidation工件选出或否决唯一两路候选。
- 步骤22.23.1完成：协议文件已冻结候选E1/E5/E20、固定0.5/0.5权重、三seed双指标全胜与80%收益保留门；当前磁盘可用空间回升至约3.6 GiB，但仍保持单资产/单checkpoint约束。
- 步骤22.23.2开始：已定位三seed完整validation与test工件；选择器将先独立构造validation记录并完成排名，只有选出候选后才进入单独的test探索函数，避免test反向影响候选。
- 步骤22.23.2完成：编译、七项合成正负例与真实三seed分析均退出0；E1/E5/E20均未达到双指标80%收益保留门，决策=`EFFICIENT_TWO_VIEW_STOP`。
- E1平均MSE/MAE改善0.515095%/0.452280%，仅保留三路收益49.81%/49.99%；E5为0.517269%/0.324958%（50.02%/35.92%）；E20为0.485566%/0.138993%（46.95%/15.36%）且MAE不是3/3胜出。
- 分析器确认`test_sources_accessed=false`，因此没有利用已暴露test调整候选；按协议不生成seed2029、不放宽80%门，步骤22.23正式停止。
- 步骤22.24开始：转向保持三路预测数学等价的batch维合并前向；现有Traffic评估峰值显存约3086.85 MiB，理论上有验证空间，但当前`nvidia-smi`暂时无法连接驱动，先完成结果前协议与零GPU实现。
- 步骤22.24.1完成：在`docs/STAGE22_24_BATCH_PARALLEL_CNM_SE_PROTOCOL_ZH.md`冻结最大预测差1e-4、指标差1e-6、平均耗时至少降低20%、bootstrap 95%下界>0、同batch交替先后顺序和显存不阉割规则。
- 步骤22.24.2实现默认关闭的新入口：同一batch共享L5/L20 mask与预处理；串行三次前向对照和`3B`拼接单次前向交替顺序执行，保存逐样本等价误差、逐batch配对耗时、峰值显存和固定seed自助法区间。
- 已新增seed2027 smoke/formal配置、2任务依赖队列、启动/状态/停止脚本、自动汇总器和12项静态/负例/路由/决策测试；全部PASS，`git diff --check` PASS。
- 已重跑步骤22.22原有19类队列与配置回归，全部PASS，确认新增默认关闭入口未破坏既有串行评估。
- 步骤22.24调度器9项自测全部PASS；GPU探测时4080/4090分别空闲约14.6/23.3 GiB，使用4090真实启动seed2027 smoke。
- 首次沙箱内后台启动PID被当前命名空间误判为消失；沙箱外持久启动实际上仍持有flock并持续更新state，随后确认任务正常执行。这不是实验外部终止，也没有重启或重复任务。
- seed2027 smoke首次SUCCESS、无OOM、峰值显存8805.69 MiB、18项工件检查PASS；但严格数值等价门FAIL：original/L5/L20/multiscale最大绝对差分别7.27e-4/8.24e-4/7.73e-4/3.76e-4，高于冻结1e-4。完整MSE/MAE差仅7.87e-9/1.28e-8，说明聚合指标近似等价但逐点门不满足。
- smoke两计时batch平均降低19.3599%，95%区间[17.9655%,20.7544%]，也略低于冻结20%门；smoke不作正式速度结论。
- 调度设计缺陷：scheduler只把artifact SUCCESS作为formal依赖，没有把`prediction_equivalence_pass`作为科学硬门，导致正式任务在smoke完成后的下一轮自动启动。正式任务已运行，按不主动中断规则允许完成，但不得用于放宽预注册容差或改写smoke失败。
- 下一实现方向已由失败机制限定：batch拼接改变了CUDA kernel的batch形状并产生高于阈值的FP32逐点差；后续应验证保持每路batch size=16的CUDA多流并发，而非调整容差迎合结果。
- 步骤22.24 formal仍在4090首次attempt运行；训练日志已完成模型/checkpoint加载且state持续刷新，未发现OOM或代码异常。
- 步骤22.25.1开始：基于smoke失败冻结“每路仍为B=16、三次模型调用分配到三个CUDA stream并发”的新执行假设；沿用1e-4/1e-6等价门和20%速度门，不使用22.24 formal结果调整规则。
- 步骤22.24 formal首次SUCCESS、无OOM、93个配对batch和全部工件协议PASS；串行/拼接并行平均253.1101/198.7509 ms，耗时降低21.2630%，95%区间[20.0774%,22.4045%]，速度门PASS，峰值显存8806.44 MiB。
- 但formal最终融合最大绝对差5.87e-4（单路最大约1.12e-3）仍超过1e-4，决策=`BATCH_PARALLEL_EQUIVALENCE_FAIL`；MSE/MAE绝对差仅8.74e-10/1.37e-9不能替代逐点门，故不运行seed2028。
- 步骤22.25.2已实现三CUDA stream同shape并发、独立配置/队列/聚合和科学硬依赖；6项配置负测与门禁测试全部PASS，确认失败smoke会在formal占GPU前将其置为`BLOCKED_DEPENDENCY`。
- 步骤22.25启动前再次通过9项调度器自测、12项22.24回归和6项22.25科学门测试；持久队列在4090空闲24216 MiB时真实启动smoke。
- CUDA多流smoke首次SUCCESS、无OOM、18项工件检查PASS、峰值显存3166.89 MiB；original/L5/L20/multiscale的最大绝对差和均方差全部为0，完整MSE/MAE逐位一致，严格等价硬门PASS。
- smoke仅2个计时batch，平均耗时变化-0.5959%、95%区间[-1.3761%,0.1843%]；按预注册规则不以smoke速度阻止formal。科学依赖已记录`passed=true`，formal随即首次attempt启动。
- 步骤22.25 formal首次SUCCESS、无OOM、全部工件协议PASS；串行/多流平均199.0998/197.9592 ms，平均仅降低0.5165%，95%区间[-0.0524%,1.1057%]，速度门FAIL。
- formal四路最大绝对差与MSE差均为0，MSE/MAE完全一致，确认实现严格等价；但收益远低于20%，决策=`CUDA_STREAM_SPEED_GATE_FAIL`，按规则不运行seed2028并停止底层执行优化。
- 步骤22.26开始：转向训练期三视图CNM-SE教师、推理期original单视图学生的收益蒸馏；只使用train/validation设计与筛选，现有2026–2028 test不参与。
- 步骤22.26.1审计确认可复用既有MCPR零初始化低秩残差头、冻结PG95训练路径、checkpoint/哈希审计和等计数mask生成器；旧MCPR失败的核心训练信号是直接目标MSE，本步骤只复用架构与工程，不复用失败目标。
- 已冻结方法名BF-CMVD（Benefit-Filtered Causal Multi-View Distillation，收益筛选因果多视图蒸馏）：训练期固定CNM-SE教师，仅在教师相对原始PG95的逐样本MSE和MAE同时更优时蒸馏；推理期只保留original+低秩残差头单次前向。
- 步骤22.26.2已完成默认关闭实现：新增固定0.15等计数L5/L20训练视图、0.5/0.25/0.25冻结教师、逐样本MSE与MAE双胜筛选，以及“观测目标MSE+有益样本L1蒸馏”目标；PG95主干和prototype memory均冻结。
- 新增seed2027 smoke/formal独立配置、机制级科学硬依赖队列和validation-only聚合器；formal只有在smoke证明获益/非获益样本同时存在、蒸馏与教师差异非零、梯度/残差非零且checkpoint可重载时才允许占GPU。
- BF-CMVD 26项组件/配置/负例/科学门测试、MCPR 14项回归、TTRR 30项回归和通用调度器9项自测全部PASS；三处修改源码均已用项目Conda Python静态编译通过。
- 启动前磁盘剩余3.6 GiB，足够本轮约123 MiB checkpoint与日志但余量偏紧；受限环境内`nvidia-smi`无法连接驱动，需用宿主权限完成UUID/显存核验后再启动。
- BF-CMVD-v1真实smoke首次运行完成训练/验证，4步梯度、残差、教师差异和冻结哈希均正常，峰值allocated/reserved为3186.92/4282 MiB；但训练观测目标上获益样本为0/64、蒸馏项为0，runner按预注册机制门标记CODE_ERROR，formal为BLOCKED_DEPENDENCY且从未启动。
- 离线只读分析既有seed2027 validation样本缓存：同一CNM-SE教师在完整目标上双指标获益1213/1648（73.6044%），MSE/MAE单项胜率82.0995%/75.4248%；说明教师有效而整样本训练筛选过粗。
- 步骤22.26.4冻结BF-CMVD-v2：只把筛选单位从整样本改为预测时间单元，仍只用训练观测目标、仍要求跨变量MSE和MAE同时改善、损失与所有教师/预算参数不变。新增独立配置/队列/工件身份，不覆盖v1。
- v2新增局部收益被样本平均掩盖的合成回归；v1 28项、v2 8项、历史MCPR 14项检查全部PASS，当前环境静态编译通过。
- BF-CMVD-v2真实smoke首次attempt SUCCESS、无OOM：4步覆盖4795个可观测时间单元，其中916个双指标获益，获益率19.1032%；蒸馏L1均值0.019976、最大梯度范数0.035108、残差能量4.094e-05，checkpoint保存/重载与冻结哈希审计均PASS，test_split_accessed=false。
- 科学依赖记录`passed=true`后，调度器于21:29:24自动在RTX 4090启动seed2027完整formal，训练PID 1838563；smoke实测reserved 4282 MiB加1536 MiB安全余量后，formal显存估计自动更新为6144 MiB，未修改batch或正式预算。
- 2026-08-21终态核查：BF-CMVD-v2队列=`SUCCESS`，smoke/formal均首次attempt SUCCESS、OOM重试0；formal运行2266.24秒、4 epoch早停、2996次优化，checkpoint重载、主干哈希和prototype memory哈希全部验证通过，test_split_accessed=false。
- seed2027 validation结果：PG95=`0.523472/0.318782`，CNM-SE教师=`0.517810/0.316192`，BF-CMVD-v2学生=`0.520750/0.317638`（MSE/MAE）。学生相对PG95改善0.519939%/0.358726%，但只保留教师收益48.0711%/44.1573%，未达到两项各≥50%的预注册门。
- 正式训练共筛出638222/3459738个获益时间单元（18.4471%），蒸馏项真实持续激活；失败是收益强度不足而非机制未运行。自动决策=`BF_CMVD_SEED2027_STOP_OR_REDESIGN`，因此不运行seed2028、不生成seed2029、不访问test，步骤22.26完成。
- 本轮一次宿主复合只读状态查询在自动权限审核阶段超时且未执行；改为直接读取持久化JSON/metrics完成核查，实验工件未受影响。
- 终态审计发现专用Step22.26报告已生成但通用latest结果表未刷新；只运行既有`scripts/aggregate_results.py`重新汇总220个run，退出0。BF-CMVD-v2 formal现已进入`reports/latest_results.csv/.md`，原始metrics和训练工件未改写。
- 2026-08-21进入步骤22.27：目标是用冻结BF-CMVD-v2/PG95/CNM-SE在Traffic seed2027 chronological validation上区分三类主因——训练监督覆盖错配、学生修正表达不足、非教师获益区域副作用。
- 已恢复`task_plan.md`、`progress.md`、`findings.md`和文件规划session；Step22.26正式run/报告、48.07%/44.16%收益保留率及“不扩seed、不访问test”的停止边界一致。
- 已把22.27拆成协议冻结、只读实现、GPU smoke、完整validation诊断、单一v3决策五段；本步不训练新模型，也不因诊断结果修改现有test结论。
- 下一操作：审计Step22.20只读诊断入口、Step22.26 formal checkpoint/配置及多尺度预测实现；随后在任何新诊断指标前写入22.27中文冻结协议。
- 接口初审完成：Step22.20已有完整的只读run/NPZ/JSON/UUID队列模板；Step22.26-v2 formal checkpoint约117.6 MiB且包含完整resolved config；`_equal_rate_multiscale_evaluation`已有冻结L5/L20输入构造和教师融合实现，可复用而无需重写掩码数学。
- 诊断将保存逐样本、逐预测步和少量聚合统计，不保存`1648×96×862`的四路完整FP32预测，避免在当前磁盘上产生约2 GiB以上冗余工件。
- 下一操作：读取只读runner的checkpoint审计/落盘约定及多尺度评估的张量口径，形成可实现且结果前冻结的精确误差分解。
- 步骤22.27.1已完成：新增`docs/STAGE22_27_BF_CMVD_FAILURE_ATTRIBUTION_PROTOCOL_ZH.md`，在任何新诊断指标前冻结三类主因、配对张量、误差分解、阈值、判定优先级和主因到唯一v3结构的映射。
- 冻结门明确要求：覆盖错配须同时满足支持召回<0.70与获益率差>0.05；表达不足/非获益副作用分别要求MSE和MAE正教师差距的区域份额均≥60%，并附加残差对齐或伤害/能量条件；混合证据直接停止。
- 下一操作进入22.27.2：实现新的validation-only只读模式、compact NPZ、runner与CPU/配置/checkpoint测试。
- 已在`run.py`加入与其他诊断互斥的Step22.27参数、冻结配置负测和只读dispatch；在`exp_pretrain1.py`加入token诊断公式与完整validation缓存入口。
- 新入口一次学生前向取得学生/主干残差分解，关闭MCPR后依次生成严格等移除L5/L20教师；保存`sample×96`误差/获益/修正几何和逐变量总量，不保存完整预测。
- 两个修改文件使用项目Conda解释器`py_compile`均退出0；`git diff --check`只报告该长期修改文件已有的历史行尾空格，本次新增块未引入语法错误，暂不机械清理无关旧行。
- 下一操作：新增可还原runner、smoke/formal配置、artifact合约测试和自动分析器。
- 已新增`run_stage22_27_bf_cmvd_diagnostic.py`及smoke/formal配置：固定三个来源run身份和checkpoint哈希，保存双checkpoint来源、resolved config、环境、Git状态、代码diff、命令、状态、资源和compact NPZ。
- runner预设22类关键工件检查，包括103/3 batch数量、`sample×96`形状、862变量总量、严格双指标获益回算、观测获益子集、修正二次恒等式、Step22.22固定L5/L20掩码逐值一致以及三路formal指标回算。
- 下一操作：实现预注册阈值分析器、合成CPU合约和smoke依赖formal的UUID队列，再统一跑零GPU门禁。
- Step22.27专项25项CPU/静态/配置/决策/队列门检查全部PASS；Python编译和三个Shell语法检查均退出0。
- 通用调度器`--self-test`的9项检查全部PASS，覆盖最大并发5、动态补位、依赖顺序、OOM一次回队列、确定性错误停止、resume和启动预留竞争。
- 启动前GPU快照：物理0 RTX4080 UUID=`GPU-c88b...`空闲16072 MiB；物理1 RTX4090 UUID=`GPU-a3af...`仅空闲326 MiB。调度估计6144 MiB，当前应由UUID选择4080。
- 启动前文件系统仅余约261 MiB；本诊断不复制checkpoint、不保存完整预测，runner硬限制compact NPZ小于64 MiB，预计两run工件可容纳。若真实落盘失败，将分类为资源/写盘错误并停止，不删除用户成果。
- 下一操作：以最大并发5启动后台队列；只有3-batch smoke全部工件PASS才自动放行103-batch formal。
- 首次普通沙箱后台启动未越过PID namespace：没有state、子进程、GPU占用或run，已明确判定为未启动而非实验失败。
- 改用宿主命名空间后队列真实启动：scheduler PID=`3963636`，smoke attempt01状态RUNNING；runner PID=`3963669`、真实S4M子进程PID=`3964314`，分配RTX4080 UUID=`GPU-c88b...`，命令明确为`max_eval_batches=3`与Step22.27只读模式。
- formal仍为PENDING并依赖smoke；4090仅326 MiB空闲未被强行使用，没有缩小配置或越过门禁。
- 下一自动操作：smoke完成后runner执行22类工件验证；全部PASS才由调度器启动完整formal，最终分析器按预注册门生成唯一主因/停止标签。
- smoke attempt01真实完成3 batch且模型返回码0，23项工件中22项PASS；唯一失败为残差加减恒等式最大误差`2.384185791015625e-7`略高于硬编码`2e-7`，formal被队列正确阻断，队列状态FAILED，无test访问。
- 该误差精确处于FP32的2个机器epsilon量级，固定掩码、严格获益回算、修正二次恒等式、全部有限值和指标回算均PASS；根因是审计容差而非模型/数据错误。
- 将工程恒等式容差改为`8×float32 epsilon≈9.54e-7`，仍小于`1e-6`；不改模型、checkpoint、掩码、误差分解或预注册科学阈值。旧失败run完整保留。
- 下一操作：重跑专项CPU测试后，以明确reset reason重排smoke/formal，生成新experiment ID。
- 工程容差修复后专项26/26 PASS、调度器9/9 PASS；以`float32_residual_identity_two_ulp_tolerance_fix`写入events并重排两个任务，attempt计数保留。
- smoke attempt02已在RTX4080真实启动，旧attempt01仍保留为CODE_ERROR证据；formal保持PENDING，不存在越门重跑。
- attempt02完成模型推理与工件计算后，在最终`status.json`序列化时触发`numpy.bool_ is not JSON serializable`；来源是新容差常量保留为NumPy float导致比较结果为NumPy布尔量，formal再次被阻断。
- 修复为显式Python `float`并增加常量类型合约；这是第二个确定性审计器bug，不改任何预测或诊断值。下一操作重跑测试后以新的reset reason进行第三次smoke。
- 类型修复后专项26/26再次PASS；以`python_scalar_serialization_fix_after_artifact_pass`重排，smoke attempt03已在RTX4080真实启动，events保留前两次失败与两次修复原因。
- smoke attempt03最终SUCCESS、return code 0，23/23 artifact checks全true，run=`runs/bf_cmvd_v2_failure_attribution__traffic__validation__seed2027__smoke__20260821_221008`；48样本/3 batch、固定掩码逐值一致、test未访问。
- artifact gate自动放行formal attempt01；run=`runs/bf_cmvd_v2_failure_attribution__traffic__validation__seed2027__20260821_221024`，S4M子进程PID=`4033737`，RTX4080实占约4252 MiB、GPU利用率84%，`max_eval_batches=-1`确认完整103 batch真实运行。
- 按全局规则已确认正式实验真实运行后停止持续人工监控；队列会在formal成功后自动执行Step22.27分析器并生成`reports/stage22_27_bf_cmvd_failure.json/.md`。
- Step22.27队列于2026-08-21 22:11:08 CST自然结束，queue_state=`SUCCESS`；最终smoke和formal均SUCCESS，formal attempt01 return code 0、无OOM/重试、103 batch/1648样本、23项artifact check全true。
- formal精确回算PG95/CNM-SE/BF-CMVD-v2 validation MSE/MAE=`0.523472011/0.318781644`、`0.517810106/0.316191852`、`0.520750225/0.317637980`；test未访问，模型与memory未更新。
- 覆盖假设未触发：观测支持recall虽为0.678290，但已观测/未观测教师获益率仅差0.001351，未满足结果前冻结的联合门。
- 非获益副作用未触发：非获益区虽承载31.9450%学生残差能量，但只占MSE/MAE正教师差距的2.5646%/5.6520%。
- 学生表达不足强触发：获益区占MSE/MAE正教师差距97.4354%/94.3480%，学生—教师修正cosine=0.200284、projection=0.072097、normalized mismatch=0.985388。
- 自动决策=`STUDENT_EXPRESSION_UNDERFIT_FREEZE_DMRB`；唯一BF-CMVD-v3冻结为DMRB（双尺度掩码条件残差基）。Step22.27本身不训练v3、不扩seed、不访问test。
- 当前下一最短路径是Step22.28：先写DMRB结果前冻结结构与公平门，再实现两个独立低秩残差基分别蒸馏L5/L20教师修正；推理仍保持一次PG95主干前向，不同时增加可靠性门。

## Step 22.28 进度（2026-08-21）

- 已开始 Step22.28，按 `planning-with-files-zh` 文件化科研流程恢复并更新任务状态。
- 当前子步为 **22.28.1 结果前协议冻结**；尚未运行 DMRB 训练，也未访问任何 test 指标。
- 已知资源风险：上一次核验项目所在文件系统仅余约235 MiB；DMRB smoke 与 formal 都必须保存可加载 checkpoint，因此先完成不产生大工件的协议、实现和 CPU 测试，GPU 前必须重查。
- 磁盘不足时不会删除既有 checkpoint、run 或用户成果，也不会通过禁用 checkpoint 保存/加载来伪造 smoke 通过。
- 下一操作：审计 MCPR/S4M 模块、BF-CMVD-v2 训练目标和 Step22.26 队列工件，写入唯一 DMRB 结构与预注册科学门。
- 22.28.1–22.28.4已完成：结果前协议、DMRB模块、双分支尺度分解目标、可关闭开关、配置、runner、smoke→formal队列、自动汇总和操作脚本均已落盘。
- 两组Step22.28专项测试35/35 PASS；历史MCPR 14/14、BF-CMVD-v2 8/8、Step22.27 26/26回归均PASS；调度器9/9自测PASS，Shell语法和8个Python关键文件静态编译PASS。
- DMRB两分支参数不共享，单分支45,182、双分支90,364；零初始化、固定求和、独立梯度、教师因子分解和checkpoint往返全部通过。
- 磁盘门复核为226 MiB。不删旧成果，已实现并验证`mcpr_delta_v1`：保存新adapter+来源SHA-256，加载能完整重建主干/memory/DMRB，不再重复118 MiB冻结权重。
- GPU启动前快照：4080 UUID=`GPU-c88b...`空闱16072 MiB，4090 UUID=`GPU-a3af...`空闱328 MiB；smoke调度估计14336 MiB，应由稳定UUID选择4080。
- 下一操作：以最大并发5启动Step22.28后台队列；smoke机制/工件全通过后自动启动formal。
- Step22.28首次GPU smoke已在4080真实完成训练/评估与359 KiB增量checkpoint；唯一失败是教师因子分解FP32绝对误差`1.9073486328125e-6`高于旧工程容差`1e-6`，formal被正确阻断，旧run已保留。
- 已将工程恒等容差冻结为`32×float32 epsilon=3.814697265625e-6`，不改模型预测、损失、教师权重、训练预算或60%科学门；更新后专项组件20/20和pipeline20/20均PASS。
- 以reset reason=`fp32_teacher_factorization_32epsilon_engineering_gate_fix`保留attempt历史后重排；smoke attempt02已真实启动，run=`runs/dmrb_pg95_traffic__traffic__point__mr0.06__seed2027__smoke__20260821_233022`，scheduler PID=`969045`，runner PID=`969078`，S4M PID=`969327`。
- attempt02使用RTX4080 UUID=`GPU-c88b...`，已进入第2轮训练；formal仍PENDING且只会在smoke全工件通过后自动放行。按全局规则，确认真实运行后不持续人工监控。

## Step 22.28 终态（2026-08-26核验）

- 后台队列已自然结束，queue_state=`SUCCESS`；修复后的smoke attempt02与formal均为`SUCCESS`，formal return code 0、无OOM、无资源等待、无确定性错误。
- formal run=`runs/dmrb_pg95_traffic__traffic__point__mr0.06__seed2027__20260821_233052`；完成7 epoch、5243个训练batch、103个validation batch/1648样本，耗时1715.657秒，峰值显存3223.086 MiB、reserved 4284 MiB。
- 工程与协议门全部通过：L5/L20分支损失、梯度和残差能量均为正；PG95主干与memory冻结；90,364个新增可训练参数；359 KiB增量checkpoint已验证加载与双分支键；`test_split_accessed=false`。
- 科学门失败：DMRB MSE/MAE=`0.521787612/0.317986334`；相对PG95改善`0.321773%/0.249472%`，但相对BF-CMVD-v2退化`0.199201%/0.109647%`。
- DMRB仅保留CNM-SE教师MSE/MAE收益的`29.7496%/30.7087%`，显著低于预注册60%门，也低于BF-CMVD-v2此前的`48.07%/44.16%`。自动决策=`DMRB_SEED2027_STOP_DIAGNOSE`。
- 已按预注册停止seed2028与test；结果已进入`reports/stage22_28_dmrb_seed2027.{json,md}`及`reports/latest_results.{csv,md}`。
- 下一最短路径是Step22.29 validation-only分支归因：先区分联合筛选掩码错配、分支相加冲突和分支表达仍不足，再只允许一个最小改动，不直接继续训练v4。

## Step 22.29 进度（2026-08-26）

- 已开始Step22.29并恢复文件化计划；当前处于22.29.1结果前协议冻结，尚未计算任何新的分支诊断值、未训练新模型、未访问test。
- 接口初审确认：DMRB已提供L5/L20独立basis与开关，Step22.27已有validation-only checkpoint加载、固定L5/L20嵌套视图、compact NPZ、UUID队列和工件审计模板，可在不改模型数学定义的前提下扩展只读诊断。
- 下一操作：读取DMRB返回张量与Step22.27诊断/runner的精确数据口径，在任何新分支结果前写入Step22.29中文冻结协议和自动决策顺序。
- 22.29.1已完成：新增`docs/STAGE22_29_DMRB_FAILURE_ATTRIBUTION_PROTOCOL_ZH.md`，在任何新分支指标前冻结七路预测、B5/B20/Bjoint、分量抵消、学生交互、自/交叉目标几何及三类主因的阈值和优先级。
- 冻结映射只允许一个方向：监督错配→Branch-Pareto分支专属双指标掩码；固定相加冲突→因果尺度路由器；双分支仍欠拟合→停止低秩路线；混合证据同样停止。
- 下一操作进入22.29.2：实现DMRB validation-only诊断入口、可还原runner、compact NPZ与自动分析器，不改训练路径。
- 已在`run.py`增加默认关闭且与所有既有诊断互斥的Step22.29参数、DMRB身份/目标/权重/validation-only负测和只读dispatch。
- 已在`exp_pretrain1.py`实现同前向七路预测诊断：直接缓存r5/r20并恢复p0，关闭adapter只生成冻结L5/L20教师视图；输出分支/联合损失、获益掩码、自/交叉修正几何和抵消量，不保存完整预测。
- 下一操作：静态编译新增入口；随后实现可还原runner、工件合约与结果前自动分析器。
- 22.29.2已完成：新增两份固定配置、唯一run ID的可还原runner、关键源码快照、DMRB增量checkpoint/来源SHA-256审计、compact NPZ、结果前分析器、smoke→formal UUID队列及start/status/stop脚本。
- 新增Step22.29专项20/20检查全部PASS，合成数据分别精确触发监督错配、固定相加冲突、双分支欠拟合与混合停止；Step22.28组件20/20、pipeline20/20和通用调度器9/9回归均PASS。
- 7个Python关键文件静态编译、3个Shell脚本语法和本轮两处baseline源码`git diff --check`全部退出0。
- 下一操作进入22.29.3：核对磁盘与GPU UUID/显存，随后后台启动3-batch smoke；只有全部工件门PASS才自动放行103-batch formal。
- Step22.29首次smoke在checkpoint预检阶段被标记CODE_ERROR，未进入模型前向、formal保持BLOCKED_DEPENDENCY、test未访问。根因是基础`verify_checkpoint`只返回adapter键，`checkpoint_dmrb_branches_verified`原本由通用runner后置生成；新runner错误地提前读取该后置字段。
- 已改为从`checkpoint_mcpr_state_keys`直接验证L5/L20两组参数并把结果写回审计工件，新增“缺任一分支必须失败”的正负测试；不改预测、权重、掩码、阈值或诊断公式。
- 下一操作：重新执行Step22.29专项与静态门禁；通过后以明确reset reason只重排本队列，保留attempt01。
- 修复后Step22.29专项21/21与静态编译再次PASS；以reset reason=`dmrb_checkpoint_branch_key_direct_verification_fix`保留attempt01后重排smoke/formal。
- smoke attempt02已真实运行：scheduler PID=`3942153`、runner PID=`3942158`、S4M PID=`3942603`；run=`runs/dmrb_failure_attribution__traffic__validation__seed2027__smoke__20260826_164922`。
- 调度器按稳定UUID选择RTX4080 `GPU-c88b...`；命令明确为`is_training=0`、`max_eval_batches=3`、Step22.29 validation-only模式，formal仍PENDING且test loader未构造。
- 下一自动操作：smoke完成后验证分支恒等式、固定mask、checkpoint、指标回算与compact工件；全部PASS才自动启动完整103-batch formal。按全局规范，确认真实运行后不持续人工轮询。
- Step22.29队列已自然完成，queue_state=`SUCCESS`；修复后的smoke attempt02与formal attempt01均`SUCCESS`，formal处理103 batch/1648样本/136,375,296个元素，20/20工件检查全true，test未访问。
- formal回算PG95、DMRB与CNM-SE教师MSE/MAE分别为`0.523471994/0.318781597`、`0.521787606/0.317986323`、`0.517810102/0.316191875`；与既有结果闭环，学生/教师因子分解最大绝对误差均小于`1e-6`。
- 联合监督错配门未触发：`P(B5|Bjoint)=0.808580`、`P(B20|Bjoint)=0.778921`，分支获益在联合掩码外遗漏仅约18.8%，均未达到预注册阈值。
- 固定求和冲突门未触发：完整DMRB双指标均优于任一单分支，学生分支余弦`0.364590`、无抵消，交互项为小幅正收益。
- 双分支表达不足门明确触发：L5/L20自目标余弦仅`0.077425/0.077294`，投影系数仅`0.034052/0.026085`，归一化失配`1.125322/1.061720`；两分支各自获益区承担约`97.05%/97.27%`的MSE正差距。
- 自动决策=`DUAL_BRANCH_EXPRESSION_UNDERFIT_STOP_LOW_RANK_ROUTE`；按冻结规则不加rank/depth、不改分支mask、不加router，Step22.29终止低秩输出残差路线。

## Step 22.30 进度（2026-08-26）

- 已开始Step22.30：候选冻结为PBA-CMVD（部分主干摊销因果多视图蒸馏），核心变化是把多视图教师的修正吸收到学生末端非线性时序动力层，而不是继续在输出端拟合低秩残差。
- 当前处于22.30.1结果前协议冻结；尚未训练新模型、未读取新的validation结果、未访问test。
- 下一操作：精确审计S4M层级、现有训练器的教师/学生耦合和checkpoint保存边界，随后写唯一可训练范围、冻结教师、同预算损失与停止门。
- 22.30.1已完成：新增`docs/STAGE22_30_PBA_CMVD_PROTOCOL_ZH.md`，在任何新GPU结果前冻结学生末端S4第6/7层、LayerNorm和out_proj，训练目标沿用BF-CMVD-v2的token双指标获益筛选、观测MSE+获益L1，教师权重/掩码/预算不变。
- 来源checkpoint静态状态审计显示预注册末端范围为4,127,582个可训练参数（约占来源状态规模13.46%）；其余主干与prototype memory冻结，部署模型不增加参数，推理仍为原始单视图一次前向。
- 22.30.2已完成：实现默认关闭的`use_pba_cmvd`训练分支、独立deep-copy冻结教师、精确参数解冻/分组梯度审计、教师与memory哈希、训练期三视图/validation单视图边界及`pba_delta_v1`原子增量checkpoint。
- 新增formal/smoke固定配置、配置负例、唯一run工件、来源源码快照、smoke→formal UUID队列、科学门、自动比较汇总以及start/status/stop脚本。
- 22.30.3零GPU门禁完成：PBA组件19/19、PBA配置/队列19/19、BF-CMVD-v2 8/8、Step22.27 26/26、DMRB组件20/20、DMRB pipeline20/20、Step22.29 21/21、通用调度器9/9全部PASS；关键Python静态编译和Shell语法退出0。
- 下一操作进入22.30.4：核对GPU UUID/显存、磁盘和后台任务后，启动PBA smoke；只有教师隔离、四组梯度、增量checkpoint、validation-only等全部门通过才自动放行formal。
- GPU启动前宿主快照：RTX4080 UUID=`GPU-c88b...`空闲16072 MiB、RTX4090 UUID=`GPU-a3af...`空闲10947 MiB且利用率95%；调度估计14336 MiB，队列原子选择4080。磁盘剩余约96 GiB，无需删除或阉割工件。
- Step22.30后台队列已真实启动：scheduler PID=`383451`、runner PID=`383485`、S4M/CUDA PID=`383729`；smoke run=`runs/pba_cmvd_pg95_traffic__traffic__point__mr0.06__seed2027__smoke__20260826_222632`。
- RTX4080已出现该任务真实CUDA占用约1134 MiB且唯一run的resolved config、命令、环境、git/diff、源码快照、日志和checkpoint目录均已落盘；formal保持`PENDING`，只有smoke全工件PASS才自动放行。
- 按全局规范，已确认实验真实运行后不持续人工监控；调度器将非阻塞轮询并在smoke通过后立即补位formal，OOM则按退避规则回队列且不改正式配置。

## Step 22.30 终态与 Step 22.31 启动（2026-08-29）

- Step22.30队列已自然结束，smoke与formal seed2027均`SUCCESS`，无OOM、重试或隐藏失败；正式run=`runs/pba_cmvd_pg95_traffic__traffic__point__mr0.06__seed2027__20260826_222702`。
- PBA-CMVD验证MSE/MAE=`0.5215299982/0.3175472255`；相对PG95改善`0.3710%/0.3872%`，但相对BF-CMVD-v2的MSE退化`0.1497%`、MAE仅改善`0.0286%`。
- 教师增益保留率仅MSE `34.30%`、MAE `47.66%`；协议门通过但准确率门失败，自动决策=`PBA_CMVD_SEED2027_STOP`，因此不跑seed2028、不访问test。
- 四个epoch后早停且最佳验证MSE在epoch1；冻结哈希、四组正梯度和delta checkpoint验证均通过，实现错误不是当前首要解释。
- 已进入Step22.31：仅在同一validation上区分高误差尾部放大、非受益区域漂移和教师方向摊销不足；下一操作是在任何新诊断值前写入冻结协议。
- 22.31.2已完成：新增`docs/STAGE22_31_PBA_FAILURE_ATTRIBUTION_PROTOCOL_ZH.md`，结果前冻结四路资产、BF参考Q90尾部、三类机制阈值、判定优先级与唯一方法映射；本步仍未生成任何新诊断值。
- 22.31.3已完成：新增PBA+独立PG95双模型validation诊断、PBA/BF逐token配对分析器、formal/smoke配置、唯一run工件、源码快照、UUID smoke→formal队列及start/status/stop入口。
- 新诊断缓存包含PBA/来源/教师逐token损失、教师修正几何、来源误差—PBA修正内积和固定packed mask；BF-CMVD-v2直接复用Step22.27已验证缓存，不重训、不保存完整预测。
- 零GPU门禁：Step22.31专项20/20、PBA组件19/19、PBA pipeline19/19、Step22.27历史26/26、Step22.29历史21/21、通用调度器self-test 9/9均PASS；关键Python编译、Shell语法与baseline diff检查均退出0。
- 下一操作进入22.31.4：启动前核对磁盘、GPU UUID/显存与其他本用户任务；随后后台启动3-batch smoke，全部paired artifact门通过才自动放行103-batch formal。
- Step22.31首次宿主smoke在模型构造前由旧`PBA-CMVD is a training-only adaptation path`安全门标记CODE_ERROR，formal正确保持BLOCKED_DEPENDENCY、test未访问、GPU未产生该任务上下文；失败run完整保留。
- 已只对白名单`stage22_31_pba_diagnostic && is_training=0`开放PBA只读加载，并新增PBA delta存在、PG95来源逐路径一致、max batch、教师权重/seed/rate等门禁；普通`use_pba_cmvd + is_training=0`仍被拒绝。
- 修复后Step22.31专项20/20、历史PBA pipeline19/19、静态编译和diff检查再次PASS；下一操作以明确reset reason保留attempt01后重排smoke/formal。
- 已以reset reason=`pba_readonly_diagnostic_whitelist_fix`保留attempt01并在宿主持久后台重排；scheduler PID=`3167675`，smoke attempt02 run=`runs/pba_cmvd_failure_attribution__traffic__validation__seed2027__smoke__20260829_205711`。
- attempt02已在RTX4080 UUID=`GPU-c88b...`真实建立CUDA上下文（PID=`3168045`）；formal保持PENDING，只会在smoke全部paired artifact门通过后自动放行。按全局规范，确认真实运行后不持续人工监控。
- Step22.31 smoke attempt02已SUCCESS并自动放行formal；formal attempt01完整处理103 batch/1648样本、返回码0，但工件终审为19/20，唯一失败是本轮直接PG95/教师逐token损失与旧Step22.27 BF缓存的`2e-5`跨run闭环。
- 固定sample/mask逐位一致，总指标全部在`1e-5`内闭环；PG95 MSE逐token最大/平均绝对差=`1.296997e-4/1.441593e-6`，99%分位=`1.370907e-5`，教师获益集合仅5/158208 token不一致。formal仍标记CODE_ERROR、自动分析未执行、test未访问。
- 下一操作：检查最坏点相对尺度和获益边界margin，确认差异来自BF的`student-residual`恢复运算顺序；只有证据支持时才把工程门改为尺度感知FP32闭环，并补合成负例防止真实资产错配被放行。
- 已冻结透明工程修订：`atol=rtol=512×float32 epsilon`，同时保留formal总指标`1e-5`、获益翻转率`≤1e-4`且仅限近零margin、固定mask逐位相等四重边界；旧formal不改状态，新ID重跑。
- 首轮新增负例中“非边界翻转”的参考端误设为`1e-6` margin，判定器正确把它识别为边界；测试夹具将改为两端绝对margin均`0.01`，正式逻辑不变。
- 修正测试夹具后Step22.31专项26/26 PASS；历史PBA pipeline19/19、Step22.27 26/26、静态编译和diff检查保持PASS。旧失败formal离线验证四类cross-path均通过，新门报告5个翻转、率`3.1604e-5`且全部为近零边界。
- 重排前重新读取Step22.31计划后，受限会话`nvidia-smi`无法连接驱动；将改用宿主权限独立核对GPU，磁盘另行检查，不据此判断资源不可用。
- 宿主资源复核：RTX4080 UUID=`GPU-c88b...`空闲16072 MiB且利用率0%，RTX4090 UUID=`GPU-a3af...`空闲13159 MiB但利用率93%；磁盘剩余90 GiB。Step22.31估计8192 MiB，优先4080且无需修改任何配置。
- 以reset reason=`fp32_cross_path_scale_aware_artifact_gate_fix`重排后，smoke attempt03 run=`runs/pba_cmvd_failure_attribution__traffic__validation__seed2027__smoke__20260829_213643`已SUCCESS，新增尺度门/边界门及全部21项工件检查通过。
- formal attempt02已由队列自动放行并真实运行，run=`runs/pba_cmvd_failure_attribution__traffic__validation__seed2027__20260829_213729`；当前不访问test、不更新模型，等待完整103 batch结束后自动分析。
- Step22.31队列已自然SUCCESS：smoke attempt03与formal attempt02均SUCCESS，无OOM/资源重试；formal 103 batch/1648样本、21/21工件检查通过，耗时76.944秒，峰值allocated/reserved=`3340.95/4148 MiB`，test未访问。
- 自动归因决策=`TEACHER_DIRECTION_UNDER_AMORTIZED_FREEZE_HFD_PBA`。尾部门不触发：尾部正MSE超额份额`0.274348`，且尾部PBA-BF MSE差为`-0.000666`；非受益漂移门不触发：非受益正差距份额仅MSE/MAE=`0.082887/0.110146`。
- 教师方向欠摊销门强触发：教师受益区正差距份额MSE/MAE=`0.917113/0.889854`，获益区cosine/projection/mismatch=`0.130330/0.138764/1.856097`。
- Step22.31.4–22.31.6完成，唯一下一候选冻结为HFD-PBA（层级特征蒸馏部分主干摊销）；不实现CVaR尾部约束或非受益路由器。已建立Step22.32计划，下一操作为结果前冻结隐藏层目标与损失尺度。
- Step22.32.1完成：已在`docs/STAGE22_32_HFD_PBA_PROTOCOL_ZH.md`结果前冻结唯一变量为S4第6/7层隐藏修正蒸馏；教师/来源/学生特征定义、能量归一化目标、无可调系数的输出损失匹配尺度、PBA同预算、工程门和seed2027停止门均已明确。开发门要求HFD-PBA同时严格优于BF-CMVD-v2与原PBA，且MSE/MAE教师收益保留均不低于60%；本轮仍禁止test。
- Step22.32.2首轮实现已接入模型、PBA训练路径、CLI和总runner；第一次静态编译因误用不存在的历史环境路径`.../envs/s4m_missing/bin/python`以127退出。既有formal的`env.txt`确认真实环境为`.../envs/s4m_missing_forecasting/bin/python`；该问题只影响检查命令，未启动实验或修改配置。
- Step22.32.2–22.32.3完成：新增默认关闭且不进入state dict的第6/7层隐藏捕获、能量归一化HFD目标、停止梯度输出损失尺度、训练指标核验、正式/冒烟配置、smoke→formal动态队列和自动科学门。正确Conda环境静态编译退出0。
- Step22.32专项组件14/14、配置/流水线18/18全部PASS；历史Step22.30组件19/19、流水线19/19继续PASS。合成测试确认两层学生特征均获非零梯度，来源/L5/L20教师特征与尺度项均无梯度，无获益时为带图零，默认关闭前向逐位一致且不增加checkpoint键。
- Step22.32.4队列已真实后台启动，PID=`1431987`，最大并发5、UUID动态选择、OOM退避回队列；初始状态=`WAITING_RESOURCE`。启动快照显示RTX4080/4090空闲显存分别`10352/13145 MiB`，均不足14336 MiB历史训练估计，故smoke尚未产生experiment ID；formal保持PENDING并依赖smoke机制门。没有OOM、代码失败或配置缩水，资源释放后会自动补位。
- 2026-08-30复核发现Step22.32队列已自行获得资源并完整结束：smoke run=`runs/hfd_pba_cmvd_pg95_traffic__traffic__point__mr0.06__seed2027__smoke__20260830_200955`、formal run=`runs/hfd_pba_cmvd_pg95_traffic__traffic__point__mr0.06__seed2027__20260830_201313`，两项均attempt1 SUCCESS、OOM/retry=0。
- 两项实际均绑定RTX4090 UUID=`GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb`；formal耗时`2744.37s`，峰值allocated/reserved=`3464.83/4420 MiB`。用户要求“尝试用4090启动”时该实验已在4090完成，因此没有浪费资源重复启动同配置。
- Step22.32协议门PASS、test未访问，但科学门FAIL：HFD-PBA MSE/MAE=`0.522073185/0.319194642`，相对BF-CMVD-v2退化`0.254040%/0.490051%`，相对PBA退化`0.104153%/0.518794%`；教师收益保留率仅`0.247059/-0.159491`。决策=`HFD_PBA_SEED2027_STOP`，不扩seed2028。
- Step22.33路线规划开始：已对比HFD/PBA训练动态。HFD新增损失与输出蒸馏同量级，并使训练直接MSE明显恶化；当前只能称为“强梯度竞争迹象”，需独立梯度/结构一致性诊断才能归因，不能直接以此调小权重重跑。
- 已审计Traffic三seed既有CNM-SE validation缓存：每个seed均有1648样本的四种计算路径逐样本SSE/SAE及mask/分歧特征，足以先做零GPU的预算化视图选择上界筛查，避免直接训练新的强化学习策略。
- 已完成Step22.33首轮oracle预检：对每个样本仅从同时不劣于original的动作中选择，seed2026/2027/2028相对固定multiscale的MSE/MAE额外上界分别为`0.5928%/0.4993%`、`0.5663%/0.6157%`、`0.8132%/0.8056%`。oracle选择full multiscale约42.35%–47.63%，其余样本分散选择original/E5/E20，存在可观的自适应选择空间。
- 下一最短路径正式调整为Step22.33 PSAVS零GPU可学性筛查，而不是继续消耗GPU解释已停止的HFD。先冻结特征/时间切分/模型复杂度，再检查mask-only和分歧感知策略；未达到2/3 seed双指标胜出且均值至少0.15%时，不进入强化学习。

## Step 22.33 执行（2026-09-07）

- 已恢复计划与项目AGENTS，确认尚无22.33实现或正式run；仅新增项目级离线选择器，不改S4M主干。
- 启动前明确287窗口时间隔离、三seed共用边界、分歧特征必须支付三次前向成本、oracle不等于可部署安全保证。下一操作为结果前冻结唯一树策略、dev校准规则和audit门。
- 22.33.1完成：`configs/stage22_33_psavs.yaml`与中文协议冻结共享深度3多输出树、14/17维特征、fit/dev/audit三段及原0.15%主门；尚未接触新选择器结果。
- 22.33.2–22.33.3实现离线runner、12项单元测试和单CPU后台总入口。首次单元检查发现YAML把未引用`on`解析为布尔值；改为`'on'`后12/12通过（退出0），不改变预注册动作语义，无受影响正式结果。静态Python/shell检查退出0。
- 正式任务是一个CPU复合流程：测试→合成smoke→三seed离线审计，不提交GPU训练、不预留显存，因此不需要GPU OOM队列；保留后台启动锁、独立run日志/状态、可查询入口和重复成功运行保护。
- 首次沙箱内nohup返回PID=10，但随后状态为NOT_STARTED、日志空且进程不存在；确认是短生命周期PID隔离环境未保活，不记录为成功启动，也没有正式run。修正启动器的PID复用判断（必须同时匹配本项目进程命令），随后申请持久后台启动所需执行权限。
- 服务器权限后台启动也返回PID后消失（无状态/空日志）；未把它记为已运行。新增同一总脚本`run`入口，通过工具管理的可续接非阻塞会话真实执行；这是进程保活方式变化，不改科学配置。run状态PID=6是执行会话内PID，不是可供宿主机kill的全局PID；未杀任何进程。
- 执行命令：`sh scripts/start_stage22_33_psavs.sh run`，退出0。流水线内12/12单测通过；合成smoke=`runs/psavs_offline__traffic__point__mr0.06__seed2233__smoke__20260907_141641`（SUCCESS）；正式run=`runs/psavs_offline__traffic__point__mr0.06__seed2233__20260907_141643`（SUCCESS）。正式CPU工作约0.939s、进程累计峰值RSS约106.85MiB、GPU分配0；当前隔离会话nvidia-smi不可用已记录，CPU计算不受影响。
- 22.33.2源门全部PASS：三seed固定四动作/其他缓存路径指标回算、共享掩码资产哈希、嵌套子集、逐窗口严格等实际移除数和原PG95身份记录完整。未加载主干或CSV，未解包固定mask资产的任何split，仅校验文件哈希；未访问test指标或test数组。
- 22.33.3：两组共享树均7节点/4叶，dev无任何合格切换裕量，均冻结infinity。mask-only raw及校准版本都全回退；分歧感知校准版本audit也是918/918全回退。raw分歧版475次切换中348次至少一指标更差、297次双指标更差；只做失败诊断，不替代主门。
- 22.33.4主门FAIL：0/3 seed双胜，平均提升0%/0%；raw MSE/MAE平均退化0.143567%/0.194586%。决定PSAVS_FEASIBILITY_STOP，不扩seed、不启强化学习、不在当前audit调参。合成smoke的PASS仅证明链路能识别已知可预测关系，不是科研正结果。
- 自动报告`reports/stage22_33_psavs.{json,md,csv}`及全局`reports/latest_results.{csv,md}`已生成；30行阶段结果包含固定路径、两组raw/校准策略、标签感知oracle和分别理想界。全局总表240条run，明确标注本轮validation子集协议，不与原论文test混比。
- 追加只读复核`python scripts/verify_stage22_33_psavs.py`退出0：源工件/源码快照/checkpoint哈希、重载预测、dev校准、动作、30行指标和科学门全部PASS；再次执行总脚本返回Already completed，不重复训练。详见`docs/STAGE22_33_PSAVS_HANDOFF_ZH.md`。

## Step 22.34 执行（2026-09-07）

## Step 22.35 执行（2026-09-08）

## Step 22.36 执行（2026-09-08）

## Step22.41 准备（2026-10-04用户授权）

- 最终启动真实性已一次性核验：scheduler3039416 PPID1、seed2028 wrapper3039438/GPU child3039702绑定4080 c88b...；seed2029 wrapper3039477/GPU child3039596绑定4090 a3af...；各实际GPU context348/496MiB。state两项RUNNING，run ID分别`...seed2028__smoke__20261004_231539`及`...seed2029__smoke__20261004_231539`，日志从新建推进到对应初始sha/28371290参数/训练入口，当前数据准备阶段，未声称正式训练或精度结果已完成。
- 按用户要求停止人工监督训练，不继续等待GPU smoke或formal完成；持久后台队列自动负责后续依赖、OOM退避、固定验证、三seed复算/归档及latest总表。状态命令`sh scripts/start_stage22_41.sh status`，阶段报告`reports/stage22_41_distance_fix.*`，详细日志`logs/stage22_41_distance_fix/`及各run/train.log。
- 后台命令`sh scripts/start_stage22_41.sh start`退出0，nohup+setsid调度请求PID3039416；新state为RUNNING并启动seed2028训练smoke wrapper3039438。真实GPU PID/日志增长待一次核验；没有仅凭PID文件宣称训练成功。后续由队列自动依赖推进，不人工监控正式训练。
- 最终CPU检查13/13 PASS、退出0、18.17秒；两个seed的真实Model关闭/开启314初始化状态/28371290参数及随机状态完全一致，与对应历史摘要相符；旧sampler CPU重建hash一致。8配置及62源码/doc/旧快照哈希已绑定到reports/stage22_41_tests.json。新GPU过滤和更多资源/更大卡OOM重试负例均通过，独立脚本复审无阻断。此后不热改被绑定源码。
- 8份冻结配置和对应两seed原基线资产清单已生成，prepare/verify_frozen退出0。新增独立wrapper/queue/start/CPUtests/中文阶段协议，不改Model、CLI、22.40源代码或原数据。两个seed链独立允许两卡并行，但同GPU不叠本队列任务；队列9项模拟PASS并保存reports/stage22_41_queue_tests.json，编译及shell语法退出0。最终CPU两seed构造和数据/配置门禁待完成后才启动。
- 两seed原22.37正式源已核对SUCCESS：2028初始摘要f8ddf474...125e345、2029摘要ea475795...02ca80，均4epoch/2992steps/28371290参数。基线不重训，开启修复构造必须匹配对应初始摘要，不共用2027权重或mask。
- 独立22.40验收结论已保存`reports/stage22_40_final_audit.md`。一次只读配置定位误用了upstream cwd（文件不存在），已改正确项目根读取；upstream状态查询显示无修改。没有实验协议/代码变动。
- 独立agent22.40最终复算PASS：6项退出0/一次尝试/无OOM；6份actual parser与resolved完全一致、无未授权差异，1648窗和10类mask/count/index资产逐项一致，12份原基线资产hash与46项代码hash保持原值。训练四轮sampler历史全部一致，最终checkpoint实sha匹配，主路由SSE/SAE重算与报告一致。
- 2028/2029必须各自沿用22.37对应CSV/mask/nested_seed，不能仅在2027配置改seed。新8任务分两条独立seed链，单GPU禁止重叠本队列任务，资源允许两卡各跑一个；准入与OOM记录保留，不缩budget。一次启动准备快照4080空闲11560MiB/4090空闲13205MiB；这不是训练监控。
- 用户要求“开始下一步，不用监督训练”。只读取一次22.40队列/结果，发现6/6SUCCESS且无OOM；直接转入已出现方向性收益后的最短三seed复核，不持续轮询旧训练。使用文件规划技能与三个子任务分别实现新队列、CPU门禁及22.40独立结果复算。
- 22.40主要固定CRN原始路由：MSE0.5347613803970451→0.5206504102513323（改善2.6387414393%）；MAE0.32664500314070355→0.3166143573335947（改善3.0708095059%）。双方4epoch/2992optimizer steps、28371290参数；实际耗时1913.27→2089.97秒，不宣称加速。原始基线和修复+冻结CNM均双指标改善，但CNM为次要既有路径，不混算创新。
- 新阶段只新增2028/2029相同修复开关、配置、训练/重载smoke与formal，保留22.40/22.37全部源结果。source_model及CLI不再修改；每seed内固定资产、训练预算、初始化和随机流检查。磁盘约7.9GiB，复用已有数据/checkpoint，不删除或复制大文件。

## Step22.40 执行（2026-10-04）

- 22:28最终真实性核验：formal run=`runs/s4m_min_distance_fixed__traffic__point__mr0.06__seed2027__20261004_222522`，scheduler2810634/wrapper2861265/GPU child2861550宿主live；4080实际占8994MiB。train.log从初始化推进至epoch1 iter200、文件10696字节，loss有限；没有正式validation MSE/MAE。后台自动负责后续训练、固定CRN验证和汇总，无需人工持续监控；正式训练尚未完成。
- 22:25实查：4项GPU smoke全部SUCCESS/退出0/无OOM。off/on训练各完成1epoch、一次warmup+一次optimizer step（max_train_batches=2，不把日志Steps749当实际训练749步），两者初始sha/参数量相同，checkpoint有限/加载验证真，allocated7861.13MiB、reserved8666MiB；各自后续独立CRN重载评估通过既定checks。smoke结果不作精度创新结论。
- 正式修复训练自动放行：queue RUNNING、fixed_train_formal wrapper2861265、GPU child2861550于4080，绑定UUID c88b779d...；fixed_eval_formal等待训练。启动时空闲11560MiB，完整batch16/192→96/8层512/15epoch上限/早停3不变。初始数据准备曾CPU较长，历史smoke耗时209.6秒，当前off约同量级；不能把冷/热缓存差异当模型效率收益。
- 核验同GPU近期小进程为另一个TrustGTR项目，未改动/终止；本项目只由独立队列调度自身作业。磁盘降至8.2GiB，保留全部旧成果，现阶段不复制大数据。
- 最终CPU门禁25/25 PASS/退出0，报告绑定六份冻结配置及46份源码/doc/旧快照哈希；单独队列9/9模拟和sh -n通过。更正前述快照尾空白疑虑：实测pre_edit/S4M.py、run.py与旧2027正式训练快照bytehash完全一致，其余旧训练manifest源码亦相同；旧eval仅加入未触发的2028/2029允许条件，不改变本轮2027数值实现。
- 22:20执行`sh scripts/start_stage22_40.sh start`退出0，宿主nohup+setsid调度请求PID2810634；独立runtime/stage22_40_distance_fix与logs同名目录。初始state已创建，真实GPU工作和日志增长仍待核验，不能仅凭启动命令宣称实验运行。
- 一次py_compile用错cwd相对路径返回文件不存在（后接git检查导致整个shell退出0），已拆开两命令并使用绝对路径重查，py_compile和baseline git --check均独立退出0，不用复合退出码掩盖验证错误。
- 最小实现已完成：S4M.py新增plain bool开关与两处if分支，run.py新增store_true入口；False分支原运算顺序不变，不加epsilon、不改gamma/noMIPP warmup。pre_edit原文本/已有Git状态和diff已保存（文本快照末尾空白归一，AST可验证；不宣称其hash为原字节hash）。
- 第一轮新CPU检查14/14 PASS/退出0，真实Model absent/False/True初始公共sha相同0aea2be...76db6、314项、28371290参数；仍待配置清单绑定与GPU门禁。旧22.36回归通过7/7；随后在非Git项目根执行diff --check退出129，已改为baseline仓库执行退出0，不影响测试或实验。Matplotlib缓存权限警告仅CPU导入，无绘图，使用临时目录。
- 只读协议审计确认CRN原始路由基线MSE0.5347613803970451/MAE0.32664500314070355，不能混用顶层CNM结果或训练后非CRN值；新eval将原封复用固定CRN和多尺度路径，仅original作为主要判据。源训练实际1913.27秒/4epoch/2992步，无完整续训状态，OOM必须从头回队列。
- 新队列4项smoke串行避免完整训练显存争抢及旧评估秒级run ID冲突，正式修复训练依赖全部smoke。初试准入按全局规则不高于6GiB，历史训练reserved11870MiB明确记录；OOM后保留资源估计与退避，绝不改batch/预算。
- 用户“继续下一步”授权隔离距离修复；使用文件规划技能与三个子任务分别审计协议、编写数学/初始化测试、复用训练/验证队列。尚未宣称GPU启动或精度提升。
- 已检查baseline HEAD/dirty状态，保留20项既有修改与未跟踪模块；仅计划修改S4M.py/run.py，默认关闭，不改上游副本。原正式基线和CRN验证缓存复用，不复制大数据或checkpoint；磁盘约9.7GiB。
- 定位时猜测的run_stage22_36_train.py不存在、首次rg cwd错误、shell的python不可用；均为只读工具命令错误，无代码/协议影响。已改为实际stage22_36_control.py及独立Conda Python，session-catchup退出0。

## Step22.39 执行（2026-10-04）

- 22.39完成验收：V2 smoke48窗和formal1648窗均SUCCESS/退出0、无OOM，正式计算52.87秒、预留3874MiB；结束后宿主调度/子PID均消失且4080显存释放。formal run=`runs/s4m_fusion_diagnostic_v2__traffic__point__mr0.06__seed2027__20261004_214457`。正式checks七项全真、prototype-off在三个smoke batch差均0，原1e-5不变。15路径由主脚本和独立agent从npz复算，MSE/MAE差≤1e-12；数组有限、哈希一致、基线与22.38完全一致。未访问test或改baseline。
- 结果：原始MSE0.5347613819/MAE0.3266450028；关闭检索改善0.039937%/0.116996%，关闭query改善MSE0.094787%但MAE退化0.159152%，关闭history退化197.0659%/161.9312%，关闭填补直达退化8.53909%/10.93943%。填补×0.9仅微益0.002913%/0.032869%，且尾部缺失6–20组MSE退化；不作为稳定创新提升。RMS历史60.629/检索0.722/查询1.282，尺度与干预共同支持当前历史路径主导，但RMS比不是贡献百分比。
- 实测：错误min距离时间std0.025178 vs正确约1，均值-1.538；原距离std最小27.788，指数前值范围[-27.294,26.581]且exp全有限。极值选缺失位置每窗变量平均比例max0.2194%/min0.2559%，不是总体缺失率，也不支持极值零选中是高频主问题。上游公式错误不等于已证明损害精度，不能在旧ckpt直接换公式后冒充重训收益。
- 新增V2测试/协议/独立配置及`summarize_stage22_39.py`，修改独立worker/runner/queue/start求值路径（未改模型）。报告`reports/stage22_39_fusion.*`、`stage22_39_distance_audit.*`、`stage22_39_mechanism_summary.*`及latest总表自动更新；最终归档`reports/archive/stage22_39_final__20261004_214938`。V1两次失败/配置保留、V1报告另归档。summary与Git diff --check退出0；下一22.40建议只做min距离修复开关和同预算单seed重训，仍未执行。

- V2验收与启动：12/12新CPU测试、9项队列模拟、静态检查/attest均退出0；独立GPU smoke48窗SUCCESS/退出0，原容差下off与V2零r精确一致。后台V2调度PID2662615，formal run=`runs/s4m_fusion_diagnostic_v2__traffic__point__mr0.06__seed2027__20261004_214457`，宿主GPU表确有子PID2663411占4080 4176MiB，日志随batch增长；同batch16完整1648窗正式诊断运行中。旧两次失败记录与V1配置保留，V1报告归档reports/archive/stage22_39_v1_failure/。无模型、数据或评价规则更改。

- V1第二次smoke（21:38 run）仍退出1，但新日志明确根因：prototype-off与原fusion零r预测差0，query/raw差0；分块相减c-R的context差0.0001220703125、prediction差0.000396728515625，独立审计一致认定FP32求值重排，不是模型开关/CRN异常。原容差1e-5保留。
- 结果前V2诊断修复：同15条探针改为原_fuse_context缩放r/u/z后计算，数学目标不变，query同时缩放投影与残差；alpha1直接返回basec。新V2配置/清单/runtime/log/实验ID，不覆盖V1失败资产；V1最新报告另归档。只改独立诊断脚本、新增V2协议与单测，主模型、训练、数据、评价指标未变。GPU通过前不启动正式。

- 用户再次要求继续，审批服务已恢复，宿主nvidia-smi查询成功。4080空闲11560MiB/4090空闲8783MiB，旧调度/工作PID均已退出（ps退出1表示未匹配，并非GPU失败）。保持原冻结配置与1e-5门禁，带新增定位日志重新smoke；启动器增加可选reset-jobs/reset-reason参数，保留原历史。并行只读复核off路径，不再猜测根因。

- 22.39.3第一GPU smoke已真实加载4080，源模型参数28371290/原型200、warmup无状态变化，但第1batch的prototype-off与c-R预测一致性断言失败，退出1，run=`runs/s4m_fusion_diagnostic__traffic__point__mr0.06__seed2027__smoke__20261004_210243`。队列FAILED，formal BLOCKED_DEPENDENCY，未产生正式指标。原版本已保存在run/source_snapshot中；不放宽1e-5、不改冻结配置、不宣称门禁通过。新worker只增加prototype_off_check.json定位真实off、原_fuse_context零检索与分量相减三者的context/prediction差以及query/raw不变性，原断言不变，原因未确定。
- 宿主核验请求被审批服务因usage limit拒绝，提示2026-10-05 01:15后可重试；并非安全拒绝，未执行该请求，不绕过审批或降权限启动GPU。文件读写与CPU仍可完成，重新GPU试验需要服务恢复。用户重复开始22.39按继续当前阶段处理，不重复已成功CPU审计。
- 独立审计agent曾误以为实际decay使用TemporalDecay而提出双指数界限；主agent复核Model.__init__ 97–98实际为nn.Linear，TemporalDecay只是未使用类，故未采纳该说法；既有19项距离报告仍有效。此时不声称实际指数溢出。

- 22.39.1/22.39.2完成：独立距离审计19/19、融合CPU12/12、调度自测9项均PASS。第一次融合测试误用nn.Linear权重布局导致失败，仅测试约定改为Bank.pred_w [2H+C,H]，history保留，worker/模型未因此改变。预定义15路径（R/U/Z取0/.9/1.1，观测直达.9/1.1，填补直达0/.9/1.1），只诊断一个seed，比例不是学习参数。保存投影前后RMS/L2、对齐、抵消、距离/logits和缺失极值选择率；正确min距离不进入模型。
- 两冻结配置准备、attest、shell语法均退出0；后台命令`sh scripts/start_stage22_39.sh start`退出0，宿主调度PID2584184，独立状态`runtime/stage22_39_fusion/state.json`，日志`logs/stage22_39_fusion/`。GPU真实工作及smoke判据尚待核验，不将PID文件当启动成功。初始准入4608MiB来自22.38的3874MiB峰值+动态余量，未缩正式配置。

- 用户明确授权开始22.39。文件规划技能恢复计划/发现/进度并运行session-catchup（退出0无输出）；当前阶段标题由旧22.34同步为22.39。两子agent独立审计上游距离实现与融合诊断设计，主agent实现独立GPU脚本。基线Git已有20项修改/大量未跟踪源，HEAD a718823addd3606e763dfc261174e0135b2535f4，全部保留，本次不改baseline。
- GPU4080 UUID c88b…空闲11560MiB，4090 UUID a3af…空闲8845MiB，磁盘16GiB；仅保存逐窗统计，不复制checkpoint。新增全局简明记录规则已读取，本阶段创建badandgood项目索引，历史仅回填有正式证据的22.38，未声称穷尽历史。
- 距离初审发现上游也有min_idx误用已标准化max_idx，极值计算包括缺失位置填入的零，以及Conv2d后reshape而非transpose风险；独立算例与哈希证据待落盘，未修复或宣称性能伤害。

## Step 22.38 完成（2026-10-04）

- CPU9/9、调度9项、GPU smoke48窗与formal1648窗均PASS。后台两任务2/2 SUCCESS、均退出0、无OOM；正式计算98.3秒、预留3874MiB、参数28371290、原型200。宿主PID与GPU实际使用已验证，结束后进程正常消失、显存释放；原模型源码与评估资产未改变，无训练或test访问。
- 命令：`sh scripts/start_stage22_38.sh start`；正式run为`runs/s4m_imputation_diagnostic__traffic__point__mr0.06__seed2027__20261004_203616`。原始MSE0.5347613818904244/MAE0.3266450028367135，历史总体MSE差约1.5e-9，最大逐窗指标差1.16e-7；重复/重组预测逐点相同、冻结权重和memory SHA256一致。
- 三替代×四路径+原始，共13路径；查询路径预测差异MSE约1.7–1.9e-7，整体MSE最大相对变化0.002490%；纯检索最大相对变化0.000341%。完整替换MSE退化8.5482%/2.3727%/18.0294%，历史直达几乎复现完整替换的效应。独立子agent与`summarize_stage22_38.py`均从samples.npz复算；不支持查询填补污染是当前主瓶颈，但不证明原填补有错，也不意味着检索无用。
- 末端缺失0/1–5/6–20窗数1130/390/128；所有最长缺失段均6–20，不能做最长缺失尺度间推断，重叠窗不能当独立seed。分段sse_h是SSE均值，摘要JSON另除24×862保存horizon_mse。
- 新建独立`stage22_38_common.py`/`stage22_38_worker.py`/`stage22_38_diagnostic.py`/`run_stage22_38_queue.py`/`start_stage22_38.sh`/`test_stage22_38_diagnostic.py`/`summarize_stage22_38.py`及中文协议文档、两配置；未修改原模型。结果`reports/stage22_38_diagnostic.*`、`stage22_38_mechanism_summary.*`、`latest_results.*`（280项），归档`reports/archive/stage22_38__20261004_204221/`。无未处理错误。
- 下一步22.39最短路径：审计融合三分量实际激活尺度、原衰减距离实现；区分查询稳定与分支贡献弱，再选最小缺失可靠性改进。未自动添加ECA/SK/PG95或训练新模型。

## Step 22.37 执行

- 2026-10-04 20:36 Step22.38.1–22.38.2：新增独立diagnostic/common/worker/queue/start/test六类入口及中文冻结协议，生成两份最终配置。CPU 9/9和调度9项PASS，shell语法退出0，配置/源码/固定数据及checkpoint SHA256门禁生效。宿主后台PID2528933、4080 UUID GPU-c88b779d-77c7-92b9-e031-943bae990121，smoke子PID2528996真实在GPU上运行；48窗口全部通过，退出0，峰值预留3874MiB，权重/完整memory无变动，原始预测重复与重组逐点相同、已有缓存逐窗口匹配。无训练、不访问test、不缩batch；完整103batch任务由同队列自动接续。启动：`sh scripts/start_stage22_38.sh start`；状态/安全停止：同脚本status/stop；日志：`tail -f logs/stage22_38_diagnostic/scheduler.log`。

- 2026-10-04用户授权下一步，完成记录同步：22.37最终16/16 SUCCESS，无OOM，配对检查通过，PG95_INCREMENT_NOT_CONSISTENT。进入22.38冻结S4M填补敏感性诊断；使用文件规划技能维护计划，并行委托两项只读审计。项目根非Git，首次误在根调用git退出129；改在baselines/S4M核验既有修改及HEAD，未改变源码。磁盘剩22GiB，诊断仅存聚合和逐窗口统计。目标只用已观测历史构建替代填补，完整历史真值不输入干预；保留既有原始填补实现，疑似标准化/reshape问题另列，不混入本次操作。

- 22.37.3后台核验（2026-09-19 17:39 Asia/Shanghai）：启动退出0；宿主PID2047454、PPID1、状态Ss，独立队列真实存活。状态WAITING_RESOURCE，4训练smoke等待、12后续任务PENDING、全部attempt_count=0。4080空闲6287MiB、4090空闲8604MiB，低于S4M准入13312MiB和PG95准入18432MiB；因此目前没有本阶段GPU实验，不宣称已训练或完成。后台每15秒检查，资源足够自动启动smoke→评估smoke→完整正式训练→正式评估，最终自动汇总三seed四格。查询：`sh scripts/start_stage22_37.sh status`；日志：`tail -f logs/stage22_37_control/scheduler.log`；安全停补位：`sh scripts/start_stage22_37.sh stop`。22.37.2 GPU门禁、22.37.3真实GPU运行与22.37.4结果仍未完成。

- 22.37.2 CPU验收：`test_stage22_37_control.py`最终7/7 PASS（33.631s），两个seed真实314公共状态哈希一致；关键模型源码与22.36两训练快照逐文件一致。9项调度自测PASS、模拟最大并发3。shell语法、Python编译、Git diff --check均通过；分析入口成功复核2027并输出WAITING_FOR_PAIRED_REPLICATION，总表262项。仅Matplotlib不可写默认缓存提示，自动使用/tmp，无计算失败。启动命令`sh scripts/start_stage22_37.sh start`已提交宿主后台；GPU smoke/正式尚未宣称成功。

- 22.37.1实现：新增结果前冻结协议、16任务独立队列、配置/资产审计与三seed报告入口；仅将评估器白名单扩展为配对训练+串行CRN的Traffic validation seed2029，test边界不变。原模型文件未修改。生成seed2029与独立重生审计正在CPU执行；队列沿用历史峰值+10%余量，不移植22.36专属5%试启动政策。调度自测允许有效并发上限1–5（此前自测硬编码必须5，实际调度逻辑不变），本阶段上限3。

- 2026-09-19恢复执行：用户要求完成本阶段；采用文件规划技能维护冻结协议及运行证据，不实现新模块。GPU只读核验：4080空闲6289MiB、4090空闲8604MiB，暂不足原训练历史峰值；正式配置不缩水。磁盘仅约36GiB可用，先核算新增工件空间。未发现Slurm/PBS命令（资源检查末尾退出127，不是GPU故障）。基线仓库既有大量修改全部保留，HEAD=a718823addd3606e763dfc261174e0135b2535f4。

- 用户授权补seed2028/2029。22.36已9/9成功、调度器正常退出，无OOM；新阶段不重跑2027。沿用历次每seed同时决定训练和mask的规则，因此统计是联合随机性而非纯训练方差。2028资产复用；2029需要同生成器新增。旧评估器seed白名单止于2028，仅对22.37的Traffic CRN validation扩展至2029，不改指标/测试边界。新队列采用独立状态目录，先smoke，后对应正式。

- 19:14恢复命令`sh scripts/start_stage22_36.sh resume`退出0；新增检查后7/7测试PASS（26.081s）。旧PID2399608于19:12:38安全退出，新调度器PID2464723/PPID1。PG95训练smoke首次真实启动于4090（UUID GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb），GPU子进程2466289，run=`pg95_paired_rng__traffic__point__mr0.06__seed2027__smoke__20260908_191449`，状态RUNNING，启动时空闲17561MiB。已核验宿主机live PID、GPU进程及日志从设备初始化推进到MODEL_FOOTPRINT；314公共条目初始哈希与已完成S4M一致。当前仅确认smoke启动，不宣称反向/保存已通过或PG95正式训练已启动。训练/评估smoke完成后队列自动继续正式任务。
- 19:11左右用户要求现在尝试启动。核验5/9任务SUCCESS，S4M正式训练与CRN验证已完成；PG95未尝试、无OOM。4090空闲17561MiB，历史PG95 reserved峰值16448MiB。只对本阶段PG95训练首次尝试将调度余量10%→5%（最低512MiB），准入18432→17408MiB；模型、batch、epoch、损失及8份冻结配置不变。若本阶段PG95曾OOM或同任务已有尝试，不再放宽，保留提高的估计和退避。正式训练仍须自己的训练/评估smoke通过。安全停止空闲旧调度器后测试并恢复，不手动重复启动训练。
- 用户明确授权匹配初始化/分离随机流对照，冻结原训练预算和CRN-v1最终评估。新增默认关闭控制、独立sampler/worker种子、前向/记忆/辅助上下文，记录公共状态及每轮样本顺序哈希。
- 初次CPU检查因新入口误用validate_config函数名而在导入时退出1；实际旧入口为validate_experiment_config，已更正。未启动GPU实验、未修改已冻结正式超参；队列9项模拟自测已PASS。
- 当前4080空闲14942MiB，4090约5100MiB；S4M历史reserved11874MiB、PG95为16448MiB，按历史峰值+余量排队，不降batch。Slurm/PBS入口未发现（command -v退出非零），不绕过作业管理。
- 修正接口导入后冻结8份配置；6/6 CPU检查通过（包括真实两模型314公共条目哈希相同、差异0、采样器跨轮顺序、辅助流隔离、异常恢复、关键字前向及反向）。预期公共哈希`0aea2be2145809f6f3ade6f5680e084a05aae5eaff106ef1554f8346efe76db6`。原22.34配置7/7、CRN7/7回归通过；旧队列9/9模拟通过；静态编译/shell检查通过。
- 本轮保留已知PG95完整训练历史辅助视图假设，不同时修改训练信息量。训练上限15/早停3、原损失和门、数据与batch均未改动；独立9任务队列先旧基线smoke，再各自训练/评估smoke与正式对照。

- `test_stage22_35_diagnose.py`3/3通过、独立合成smoke退出0；正式`stage22_35_diagnose.py`退出0，run=`pg95_diagnosis__traffic__validation__seed2235__20260908_131733`。耗时79.97s、CPU峰值892.80MiB、GPU0。导入链Matplotlib提示配置目录不可写并使用临时目录，不影响计算，没有绘图。
- `render_stage22_35_diagnosis.py`保存中文报告和48行分组CSV、运行快照/哈希/补充Git证据，调用全局汇总。没有更改主干或运行GPU实验。结论为机制因果未识别，需要匹配初始化/独立随机流控制，尚未授权或启动新的训练协议。

- 用户授权PG95退化诊断，使用文件规划固定三seed/六训练来源/六CRN缓存，不改模型、不重训、不读test。22.34九任务已全部SUCCESS，配对检查为0，旧失败队列保留。
- 已核对训练日志：六训练都是最佳epoch1、epoch4早停、2992优化步。PG95常规门放行率99.2835%–99.6912%，额外不变性加权损失均值约2.30e-6–1.29e-5；这些是跨全训练均值，不是最佳checkpoint逐轮统计，不能直接声称梯度影响小或门控造成退化。

### 2026-09-08 恢复审计

- 用户已明确同意固定评估随机状态、重建四组配对参照并恢复4080队列。步骤22.34.4开始：新协议/新队列与旧失败记录隔离；原1e-5容差不变，原训练/checkpoint/数据资产不变。先CPU回归和两模型GPU smoke，通过才放行正式任务。
- 已实现默认关闭paired_eval_rng上下文、独立original-only遍历和持久化逐窗口参照；共享runner独立回算总体/逐窗口检查，历史指标保留差异字段。新配置8份冻结，新队列3项smoke（含原基线关闭开关）+6项正式，限定4080、并发最多3。
- 新7项单测、旧7项单测、队列9项自测退出0；静态编译通过。一次文档补丁因整行匹配失败未生效，已按完整行修正，不影响实验配置。磁盘余48GB，仅写误差缓存不复制checkpoint。
- 旧22.22流程19项检查及4个负例也PASS。执行`sh scripts/start_stage22_34_crn.sh start`退出0，宿主机调度器PID2468758、PPID1持久存活；4080启动前14942MiB空闲，UUID核对正确。
- 三项GPU smoke均SUCCESS/退出0，各48窗口；legacy run后缀095458，新S4M后缀095514，新PG95后缀095530（日期20260908）。新两模型独立original-only与多视图original逐窗口MSE/MAE最大差异均0，峰值reserved分别3936/3932MiB；旧开关关闭路径也PASS。smoke只验证链路，48窗口指标不与1648窗口历史指标作科学比较。
- 新正式任务已被自动放行；继续核对正式子进程/日志/GPU一致后即可交由后台队列运行，无需持续人工监控。
- 09:57实查正式首批：S4M2027 runner2497362/GPU child2498022，PG952027 runner2500732/GPU child2503408，S4M2026 runner2503835开始初始化；前两项已有4080 context，三个run配置/日志均落盘并推进。其余3项排队，无需新增授权；科学结果等待22.34.5自动汇总。新总报告明确CRN-v1并显示WAITING_FOR_FIXED_VALIDATION_CONTROLS，旧FACTORIAL_EXECUTION_STOP报告保留。
- 09:58最终核验三个正式GPU child为2498022/2503408/2506353，4080各占约4238/4232/4238MiB；原始参照日志分别推进至batch80/20/0，真实前向已经开始。队列RUNNING、3 smoke SUCCESS、3 formal RUNNING、3 formal等待；本次交接后由后台继续，无人工盯守。

- 用户授权恢复4080队列；先保留失败记录并定位matched_source_validation_baseline唯一失败项，不能盲重跑或放宽1e-5容差。
- 2027正式评估实际完成1648窗口/103批，计算进程退出0，封装校验失败。MSE偏差5.785276e-6、MAE偏差1.104616e-5；后续两seed未启动。当前4080空闲14942MiB，旧队列已退出，无资源阻塞。
- 已恢复文件规划记录，开始对比原post_train_validation与多视图评估的checkpoint加载、前向路径和数值归约；尚未认定跨GPU为根因。
- 后续查证：原历史与失败复评均在同一4080、同环境/扩展哈希。已确认eval随机topM原型抽样未关闭，S4M2026/2027的200→100抽样会受额外视图前向影响；PG95原型数5/8/5、S4M2028为6，不触发该分支。
- `diagnose_stage22_34_rng.py`退出0，六检查点均无RNG状态；CPU抽样复现/视图插入影响检查PASS，诊断已落盘，不是新的精度结果。
- 自动报告器修复FAILED→FACTORIAL_EXECUTION_STOP，并新增回归用例；不再把确定性校验失败显示为等待资源。未改配置、容差或模型，未重置旧失败历史。
- 恢复仍需评估规则修订授权：固定随机抽样并重建四组配对验证参照，原1e-5用于新同协议分支检查，历史值保留作差异报告。该建议已明确标记待确认；未擅自恢复正式GPU队列。见`docs/STAGE22_34_RECOVERY_DIAGNOSIS_ZH.md`。

- 用户继续授权22.34。先恢复Weather/Traffic不同CNM版本的正结果和ETTh1/Electricity完整组合负结果；PG95自身不是稳定三seed全胜，缺少同资产S4M+CNM-SE validation四格对照，不能直接宣称两模块均必要。
- 已在`docs/STAGE22_34_FACTORIAL_AUDIT_PROTOCOL_ZH.md`冻结A=S4M/B=PG95/C=S4M+CNM-SE/D=PG95+CNM-SE归因、三seed和停止边界；不读取新的test逐样本缓存，不训练新模型，不重新调PSAVS。
- 新增`run_stage14_uncertainty_cache.py`显式S4M-validation分支，原模型/数学评估代码不变，旧分支保留；创建`stage22_34_factorial.py`配置准备/执行/自动分析器、专用队列/启动器/测试脚本及四份配置。
- 6/6专项测试、旧22.22的19/19检查和4个负例全部PASS；公共队列9/9自测通过（含假OOM回队列、补位、确定性错误不重试、恢复和预留竞争）。静态Python/shell检查退出0，原PG95 validation仍接受。
- 资源估计仅从旧保守6144MiB改为历史reserved3932MiB+至少15%/512MiB安全余量，初值4608MiB；正式根据成功smoke修正，绝不降低OOM后的估计。没有改batch16、192/96长度、8层512维、数据量或主融合权重。
- 宿主机NVML只读查询正常，说明22.33中的nvidia-smi失败仅为隔离环境限制。用`sh scripts/start_stage22_34_factorial.sh start`在宿主机通过nohup+setsid真实后台启动，返回0，调度器PID3391387已存活、PPID=1。
- 队列共4任务、最大并发3：seed2027 smoke→seed2027 formal→seed2026/2028 formal；自动非阻塞轮询、OOM退避、安全停止和恢复。实际启动smoke时4080空闲6225MiB，绑定UUID=`GPU-c88b779d-77c7-92b9-e031-943bae990121`。
- seed2027 smoke run=`runs/cnm_se_equal_rate_multiscale_s4m__traffic__validation__seed2027__smoke__20260907_192822`，工作进程PID3395535已被宿主机GPU进程表确认，初始化阶段占用468MiB；这不是正式峰值或完成指标。正式任务仍依赖smoke通过。
- 状态在`runtime/stage22_34_factorial/state.json`；日志在`logs/stage22_34_factorial/`和每run的`train.log`；自动四格报告为`reports/stage22_34_factorial.{json,md,csv}`。确认实际启动后交由队列运行，无须持续人工监控；22.34.3/22.34.4尚未标记完成。
