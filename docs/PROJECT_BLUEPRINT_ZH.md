# MIRAGE-S4M 项目总文档

> 文档版本：1.0  
> 日期：2026-07-14  
> 项目目录：`/mnt/data/zza/code/S4M_MissingForecasting`  
> 当前状态：研究方案已确认；官方代码、独立环境与复现审计已完成；正式训练尚未开始。  
> 基础模型：S4M，ICLR 2025。  
> 官方代码：https://github.com/WINTERWEEL/S4M  
> 正式论文：https://proceedings.iclr.cc/paper_files/paper/2025/hash/7b2f0758334389b8ad0665a9bd165463-Abstract-Conference.html

## 1. 最终决策

本项目不采用“普通对比学习 + 频谱最优传输 + 元学习/强化学习”的模块堆叠路线。经过对 2023–2026 年相似工作的多轮核查，最终选择一条围绕 S4M 原型记忆失效机制展开的统一主线：

> 当训练和测试阶段的缺失机制、缺失形状、缺失率或连续缺失长度发生变化时，S4M 的相似度原型检索可能把“掩码相似”误当成“预测有用”，并让受缺失模式污染的表示持续写入原型库。MIRAGE-S4M 通过掩码干预净化原型供给，再通过预测效用监督控制原型读取、写入、降权和拒用。

模型暂定名为：

**MIRAGE-S4M：Missingness-Intervention and Retrieval-utility-Aware Generalization Enhancement for S4M**

中文说明：**面向缺失机制漂移的缺失干预与检索效用感知泛化增强 S4M**。

建议论文题目为：

**When Similar Prototypes Hurt: Utility-Governed Prototype Memory for Multivariate Time-Series Forecasting under Missingness Shift**

中文说明：**当相似原型反而有害：面向缺失机制漂移的效用治理原型记忆多变量时序预测**。

该题目比“在 S4M 上增加两个模块”更能呈现完整故事，因为它先提出可诊断的失败现象，再给出供给侧和使用侧两个对应解法。

## 2. 需要诚实说明的结论

1. 本方案的目标是达到 SCI（Science Citation Index，科学引文索引）二区论文所需的证据强度，但不能预先保证分区或录用。
2. “未知掩码分布漂移”不是新问题。StableMiss（IJCAI 2023）和 StableMiss+（ICASSP 2024）已研究不完整数据在训练/测试掩码分布变化下的稳定预测。
3. “检索可能有害”“选择性检索”“时序原型路由”和“不变/动态表征分解”也不是空白。RAFT、Cross-RAG、SARAF、RIDDE 和 DPAD 已分别覆盖其中部分问题。
4. 本项目当前最可辩护的新颖性，不是首次研究缺失漂移或首次使用原型，而是首次系统研究并治理 **S4M 类缺失感知原型库中的掩码诱发检索伤害**。该主张仍需最终投稿前进行一次系统查新。
5. 如果实验不能证明原型检索在缺失漂移下确实产生稳定、可重复的伤害，论文的中心假设就不成立。项目必须允许出现这一负结论，不能为追求正结果而改变评估协议。

## 3. 为什么选择 S4M 作为基础模型

S4M 的全称是 **S4 for Multivariate Time Series Forecasting with Missing Values**，中文为**面向缺失值多变量时序预测的结构化状态空间序列模型**。它发表于 ICLR 2025，直接从带缺失的历史窗口预测未来，而不是先独立插补再预测。

S4M 包含两个核心组件：

- ATPM（Adaptive Temporal Prototype Mapper，自适应时间原型映射器）：从历史表示中构建两级原型队列，并按余弦相似度检索原型。
- MDS-S4（Missing-Aware Dual-Stream S4，缺失感知双流 S4）：分别建模数据表示与缺失掩码，再生成未来预测。

选择 S4M 的原因不是它容易改，而是它的原型库暴露了一个明确、可测量、尚未被充分解决的内部失效点：

1. 原型查询表示同时受真实时序内容和缺失模式影响。
2. 原型读取只依据表示相似度，没有判断该原型是否降低下游预测误差。
3. 原型写入使用固定相似度阈值和先进先出规则，没有预测可靠性或缺失域信息。
4. 测试时即使检索结果有害，模型也没有回退到无原型预测的机制。
5. 原论文主要评估固定长度为 5 的两种合成块缺失，没有系统评估跨缺失机制、跨块长度和跨缺失形状的泛化。

这些限制刚好形成一个闭环研究问题，而不是任意寻找可插入模块的位置。

## 4. 官方项目可复现性审计

### 4.1 已取得的资产

- 官方仓库位置：`upstream/S4M`
- 官方提交：`a718823addd3606e763dfc261174e0135b2535f4`
- 官方工作区：干净，未修改。
- 独立 Conda 环境：`s4m_missing_forecasting`
- Python：3.10.13
- PyTorch：2.1.0+cu121
- NumPy：1.26.4
- Pandas：2.2.1
- `pip check`：无损坏依赖。

### 4.2 已确认的复现阻塞

1. README（项目说明文件）给出的 Weather 数据路径与仓库实际文件位置不一致。
2. 仓库只附带一个约 4.9 MB 的 `weather_missing.csv`，没有四个主数据集的完整数据和统一下载脚本。
3. 多个数据读取类硬编码作者机器的 `/home/project/S4M/...` 路径。
4. `scripts.sh` 使用 `/fast/qiong/project/S4M/...` 绝对路径。
5. 结果目录硬编码为 `/home/project/S4M/test_results/`。
6. W&B（Weights & Biases，实验跟踪平台）被强制初始化，无法直接离线运行。
7. README 命令直接运行会出现 `No module named 'S4'`，需要正确设置 Python 包路径。
8. 原型读取代码硬编码 `.cuda()`，会破坏 CPU 测试和设备无关性。
9. 论文声明使用 Adam（自适应矩估计优化器），公开训练代码实际硬编码 SGD（Stochastic Gradient Descent，随机梯度下降）。
10. `S4M.py` 中 `min_idx` 标准化疑似错误地使用了 `max_idx`，必须作为独立缺陷修复进行验证。
11. 主要指标持久化代码被注释，现有入口不满足实验完整还原要求。
12. 项目根目录没有明确 LICENSE（开源许可证），本项目不得在未确认许可前重新分发官方代码，只保留本地上游副本和适配层。

### 4.3 基线身份必须分开

后续不能把所有结果都称为“S4M 基线”。至少保存以下四个身份：

| 基线 ID | 优化器 | `min_idx` 修复 | 用途 |
|---|---:|---:|---|
| `s4m_public_sgd` | SGD | 否 | 还原公开代码行为 |
| `s4m_public_sgd_bugfix` | SGD | 是 | 分离疑似缺陷影响 |
| `s4m_paper_adam` | Adam | 否 | 接近论文文字协议 |
| `s4m_canonical` | Adam | 是 | 经验证后作为创新公平比较的规范基线 |

只有当缺陷复现实验证明 `min_idx` 确实有误，`s4m_canonical` 才采用修复版本。缺陷修复收益不能计入创新收益。

## 5. 原论文协议

### 5.1 数据集

| 数据集 | 变量数 | 时间点数 | 原论文标注粒度 | 用途 |
|---|---:|---:|---|---|
| Electricity（电力） | 321 | 26,304 | 1 小时 | 大变量数、周期性负荷 |
| ETTh1（电力变压器温度小时级数据） | 7 | 17,420 | 论文表中写 15 分钟，需按时间戳复核 | 小变量数、分布变化明显 |
| Weather（天气） | 21 | 52,696 | 10 分钟 | 多气象变量、长序列 |
| Traffic（交通） | 862 | 17,544 | 1 小时 | 超高变量数、强空间相关 |
| USHCN（美国历史气候网络） | 10 | 271,728 | 以原始资产为准 | 原论文附录真实缺失数据 |

### 5.2 划分与预测设置

- 训练/验证/测试按时间顺序划分为 0.7/0.1/0.2。
- 预测长度固定为 96。
- 回看长度为 96、192、384 和 768。
- 批大小为 16。
- 论文声明使用 Adam 和早停。
- 核心评价指标为 MSE（Mean Squared Error，均方误差）和 MAE（Mean Absolute Error，平均绝对误差）。

### 5.3 原始缺失构造

原论文包含两种连续块缺失：

- Time-point missing（整时间点缺失）：选中一个时间点后，连续 5 个时间点的全部变量缺失。
- Variable missing（逐变量缺失）：对每个变量独立选择起点，并删除其后连续 5 个时间点。

名义起点比例为 `r={0.03, 0.06, 0.12, 0.24}`。由于缺失块重叠，实际缺失率高于名义比例。例如 `r=0.24` 时，四个数据集的实际缺失率约为 0.613–0.705。

所有配置、日志和结果表必须同时记录：

- nominal missing rate（名义缺失起点比例）；
- realized missing rate（实际缺失项比例）。

## 6. 精确研究问题

给定完整历史窗口

\[
X\in\mathbb{R}^{L\times D},
\]

未来目标

\[
Y\in\mathbb{R}^{H\times D},
\]

以及二值观测掩码

\[
M\in\{0,1\}^{L\times D},
\]

模型只接收部分观测输入 \(X^{obs}\) 和掩码 \(M\)，并预测 \(Y\)。缺失域定义为：

\[
d=(\text{mechanism},\text{pattern},\text{rate},\text{block length}),
\]

中文分别为缺失机制、缺失形状、缺失率和连续缺失长度。

训练时可见一组源缺失域 \(\mathcal{D}_{train}\)，测试时出现未知目标缺失域 \(d^*\notin\mathcal{D}_{train}\)。项目研究的问题是：

> 当完整时间序列生成过程基本不变、但缺失域发生变化时，如何防止 S4M 原型库把掩码伪相关写入记忆，并在每次预测中只使用真正降低预测误差的原型？

项目不试图从观测数据中完整识别真实 MNAR（Missing Not At Random，非随机缺失）的因果机制。这里的“干预”是受控的数据或模型分支切换，用于测量同一样本在不同掩码或原型使用状态下的性能差异。

## 7. 核心科学假设

### H1：掩码诱发的原型污染

S4M 的查询和原型表示同时包含时序内容与缺失模式。训练和测试缺失域变化时，原型相似度可能主要由掩码模式驱动，导致高相似但预测无益的原型被检索。

### H2：原型相似度不等于预测效用

两个历史片段在表示空间相似，不代表其原型在当前缺失条件下能够降低未来预测误差。原型效用应由下游预测损失变化直接定义，而不是由余弦相似度代替。

### H3：供给侧和使用侧必须同时治理

只清洗写入表示不能阻止旧有或偶然错误原型被使用；只增加读取门控也无法防止原型库不断积累掩码污染。原型净化与效用治理应共同构成闭环。

### H4：缺失模式可能具有信息性

直接强制所有表示对掩码完全不变，可能删除 MAR（Missing At Random，随机缺失）或 MNAR 中有用的缺失信号。因此，模型需要把内容表示与缺失表示分开，而不是彻底丢弃缺失信息。

## 8. 整体模型

MIRAGE-S4M 保留 S4M 的主干和 MDS-S4 双流结构，只重构 ATPM 的表示写入和原型使用路径。

```text
部分观测序列 Xobs 与掩码 M
           │
           ▼
S4M 局部统计提取与查询编码器
           │
           ├────────► 缺失表示 zm ─────► MDS-S4 缺失分支
           │
           ▼
内容表示 zc
           │
           ├────────► MIPP 稳定性检查 ─► 净化后原型写入
           │
           ▼
候选原型检索 + 原型可靠性元数据
           │
           ▼
FUSR 预测效用估计
           │
           ├── 有益：重排、加权并融合原型
           └── 无益/不确定：拒用原型，回退到无原型 S4M
           │
           ▼
MDS-S4 数据分支 + 缺失分支
           │
           ▼
未来预测
```

模型包含两个核心创新：

1. MIPP（Mask-Intervention Prototype Purification，掩码干预原型净化）。
2. FUSR（Forecast-Utility-Aware Selective Retrieval，预测效用感知的选择性原型检索）。

跨缺失域训练和固定评估资产构成统一实验协议，不额外包装成第三个松散网络模块。

## 9. 创新一：MIPP 掩码干预原型净化

### 9.1 要解决的问题

原 S4M 直接把由缺失输入生成的表示写入原型库。如果查询编码器把缺失形状编码进原型，原型库就会记住“传感器如何坏掉”，而不只是“序列真实动态是什么”。缺失域发生变化后，这类原型会产生错误匹配。

### 9.2 配对掩码干预

对同一个完整训练窗口 \(X\)，从两个不同缺失域采样掩码 \(M^a\) 和 \(M^b\)，得到：

\[
X^a=M^a\odot X,\qquad X^b=M^b\odot X.
\]

两种视图具有相同的真实历史内容和未来目标，但缺失机制、形状、缺失率或块长度不同。它们用于识别哪些表示成分在掩码变化下仍然稳定。

### 9.3 双因子表示

查询编码器输出 \(h\)，随后通过两个独立投影头得到：

\[
z_c=P_c(h),\qquad z_m=P_m(h,M),
\]

其中：

- \(z_c\) 是 content representation（内容表示），用于原型查询和写入；
- \(z_m\) 是 missingness representation（缺失表示），用于保留缺失率、块长度、变量缺失差异等信息，并增强缺失感知分支。原 MDS-S4 接收的原始掩码 \(M\) 始终保留，\(z_m\) 不能替代或隐藏原始掩码。

### 9.4 内容稳定约束

同一个窗口在不同掩码下的内容表示应接近：

\[
\mathcal{L}_{inv}=\lVert z_c^a-\operatorname{sg}(\bar z_c)\rVert_1+
\lVert z_c^b-\operatorname{sg}(\bar z_c)\rVert_1,
\]

其中 \(\operatorname{sg}\) 表示停止梯度，\(\bar z_c\) 来自较少遮挡视图或指数移动平均教师编码器。

为防止所有内容表示坍缩到常数，还加入方差保持和协方差约束。这里可采用 VICReg（Variance-Invariance-Covariance Regularization，方差-不变性-协方差正则化）的标准组成，但不把该损失本身声称为创新。

### 9.5 内容与缺失解耦

通过交叉协方差或 HSIC（Hilbert-Schmidt Independence Criterion，希尔伯特-施密特独立性准则）降低 \(z_c\) 和 \(z_m\) 的冗余：

\[
\mathcal{L}_{dec}=\lVert\operatorname{Cov}(z_c,z_m)\rVert_F^2.
\]

同时让 \(z_m\) 重建掩码统计或识别缺失域：

\[
\mathcal{L}_{mask}=\operatorname{BCE}(D_m(z_m),M)+
\operatorname{CE}(C_m(z_m),d),
\]

其中 BCE 是二元交叉熵，CE 是多类交叉熵。这样可避免通过“完全掩码不变”删除信息性缺失。

### 9.6 稳定性写入门

对配对视图计算内容稳定性：

\[
s_{stable}=\cos(z_c^a,z_c^b).
\]

只有满足以下条件的候选表示才进入原型库：

1. 内容稳定性高于阈值；
2. 表示方差未坍缩；
3. 不属于重复原型；
4. 后续 FUSR 未把该类原型判定为持续负效用。

原型库只存 \(z_c\) 作为检索内容，但额外保存缺失域元数据，不把缺失表示直接混入内容原型。

### 9.7 自然缺失数据处理

USHCN 等自然缺失数据没有完整历史真值。此时不伪造完整视图，而采用 nested masking（嵌套掩码）：

- 原始可观测部分作为教师视图；
- 只从已观测位置继续遮挡，构造学生视图；
- 内容一致性只在合法样本上计算；
- 未来指标只在真实可观测目标位置计算。

### 9.8 MIPP 的创新边界

MIPP 不声称首次提出多掩码训练、表示一致性或表征解耦。其创新位置是：

> 用同一时序窗口的掩码干预稳定性直接决定 S4M 原型库“写什么”，并通过内容/缺失双因子保留信息性缺失，而不是仅在最终预测器上增加一个通用一致性损失。

## 10. 创新二：FUSR 预测效用感知的选择性原型检索

### 10.1 要解决的问题

原 S4M 的核心假设是“表示相似的原型能够帮助预测”。FUSR 不接受这一替代关系，而直接测量原型是否降低下游预测损失。

### 10.2 操作性原型效用

固定同一个输入、目标和模型参数，分别生成：

- \(\hat Y^0\)：关闭原型分支的无原型预测；
- \(\hat Y^+\)：使用检索原型的预测。

定义聚合原型效用：

\[
u=\frac{\ell(\hat Y^0,Y)-\ell(\hat Y^+,Y)}{\ell(\hat Y^0,Y)+\epsilon}.
\]

- \(u>0\)：使用原型降低了误差；
- \(u<0\)：使用原型产生了负迁移；
- \(|u|\) 很小：原型作用有限。

这里的“反事实效用”只是模型分支开/关的配对干预量，不声称真实世界因果效应。

### 10.3 单原型效用探针

只使用聚合效用无法判断 Top-K 中哪个原型有害。训练阶段对前 \(K_u\) 个候选进行单原型探针：

\[
u_i=\frac{\ell(\hat Y^0,Y)-\ell(\hat Y^{(i)},Y)}{\ell(\hat Y^0,Y)+\epsilon},
\]

其中 \(\hat Y^{(i)}\) 只使用第 \(i\) 个原型。为控制训练成本：

- 探针只在预设间隔和分层抽样批次执行；
- 使用向量化批处理；
- 效用标签由指数移动平均教师模型生成并停止梯度；
- 推理阶段不执行额外预测，只有一个轻量效用路由器。

该采样是算法设计，不是缩减正式实验数据或训练预算。

### 10.4 效用路由器输入

效用路由器接收以下可在推理时获得的特征：

1. 查询与候选原型的余弦相似度及其统计量；
2. 当前输入的缺失率、最长连续缺失、变量间缺失差异和缺失熵；
3. 当前缺失签名与原型历史缺失签名的距离；
4. 原型历史平均效用、效用方差、支持次数和年龄；
5. 查询到原型库分布的 OOD（Out-of-Distribution，分布外）距离；
6. MIPP 输出的内容稳定性；
7. 原型队列拥挤度和重复度。

路由器输出每个候选的预测效用 \(\hat u_i\)、效用不确定性以及整组原型的使用门 \(g\)。

### 10.5 效用监督

效用路由器同时进行连续回归和正负效用分类：

\[
\mathcal{L}_{util}=\operatorname{Huber}(\hat u_i,u_i)+
\lambda_{sign}\operatorname{BCE}(\sigma(\hat u_i),\mathbb{1}[u_i>\delta]).
\]

Huber 损失降低极端效用标签的影响。阈值 \(\delta\) 只在验证集选择，测试集不参与调参。

### 10.6 读取重排与回退

候选权重由相似度和预测效用共同决定：

\[
\alpha_i=\operatorname{softmax}\left((s_i+\beta\hat u_i)/T\right),
\qquad
r=\sum_i\alpha_i p_i.
\]

如果满足以下任一条件，则拒用原型并回退到 \(\hat Y^0\)：

- 所有候选的预测效用均低于阈值；
- 路由器不确定性过高；
- 当前查询严重偏离原型库支持区域；
- 原型历史负效用比例过高。

这里的 abstention（拒用）仅指拒绝原型检索，模型仍输出无原型预测，不是拒绝提供预测。

推理时路由器在进入 MDS-S4 前完成决策，只执行“带原型”或“无原型”中的一条主干路径。训练效用标签阶段才需要配对前向传播，因此不能把训练期双路径成本误报为推理成本。

### 10.7 写入治理

每个原型维护：

- 效用指数移动平均；
- 效用方差；
- 正/负效用次数；
- 支持样本数；
- 最近更新时间；
- 缺失域签名分布；
- 内容稳定性。

持续负效用原型先进入 quarantine（隔离区），在确认后再淘汰。新原型只有通过 MIPP 稳定性门和 FUSR 效用门才写入主库。

### 10.8 数据泄漏防护

真实未来 \(Y\) 只在训练或验证阶段用于生成效用标签。测试时路由器只能访问历史输入、掩码、相似度和原型元数据。必须通过测试确保：

- 测试目标不会进入路由器输入；
- 原型库不会在测试目标公开后回写；
- 阈值不使用测试结果选择；
- 效用标签文件按数据划分隔离。

### 10.9 FUSR 的创新边界

FUSR 不声称首次做选择性检索或时序原型路由。其核心新意是：

> 以“同一样本启用/禁用原型导致的预测损失差”作为监督，在缺失域变化下统一控制 S4M 原型的读取、重排、写入、隔离和无原型回退。

## 11. 总损失与训练流程

完整目标为：

\[
\mathcal{L}=\mathcal{L}_{forecast}
+\lambda_{inv}\mathcal{L}_{inv}
+\lambda_{dec}\mathcal{L}_{dec}
+\lambda_{mask}\mathcal{L}_{mask}
+\lambda_{var}\mathcal{L}_{var}
+\lambda_{util}\mathcal{L}_{util}.
\]

训练分为五个稳定阶段：

1. **规范基线阶段**：先确定 `s4m_canonical`，不启用任何创新。
2. **表示预热阶段**：训练内容/缺失双因子，原型库只读或关闭写入。
3. **MIPP 建库阶段**：只允许稳定内容原型写入，建立可靠的初始原型库。
4. **效用采集阶段**：冻结或使用教师主干生成原型开/关效用标签，训练 FUSR。
5. **联合微调阶段**：软效用门参与预测；硬拒用和写入阈值由验证集确定。

各阶段必须使用独立配置和实验 ID，不得在运行中静默改变状态。

## 12. 理论与诊断分析计划

理论部分的目标不是包装一个不现实的全局泛化保证，而是解释两个机制为何可测、何时有效、何时应失败。

### 12.1 原型选择后悔界

记 \(L_0\) 为无原型损失，\(L_1\) 为使用候选原型的损失，真实效用为 \(u=L_0-L_1\)。Oracle selector（先知选择器）在 \(u>0\) 时使用原型，否则回退。它只用于训练诊断和上界分析，不能用于测试推理。

如果效用路由器输出 \(\hat u\)，采用符号阈值选择，则相对先知选择器的额外损失只发生在 \(u\) 与 \(\hat u\) 异号时，并满足：

\[
\operatorname{Regret}\leq
\mathbb{E}\left[|u|\,\mathbb{1}(u\hat u\leq0)\right]
\leq\mathbb{E}|u-\hat u|.
\]

该界说明效用回归误差能够直接控制选择损失，但前提是效用标签稳定、没有测试泄漏。论文中需给出完整推导和经验校准，不能只引用公式。

### 12.2 掩码干预下的检索稳定性

若 MIPP 使同一内容在两个缺失域下的表示距离不超过 \(\epsilon\)，且正确原型与次近原型的检索间隔大于 \(2\epsilon\)，则最近原型身份在这两个掩码干预下保持不变。该结果可用于解释内容稳定性为何降低掩码诱发的检索翻转。

实验需报告 Top-K overlap（前 K 个候选重合率）、rank flip rate（排序翻转率）和检索间隔。若 MIPP 只提高表示相似度但不提高候选稳定性，该理论动机不成立。

### 12.3 先知上界与可用提升空间

每个数据集都要比较：

1. Always-on retrieval（始终使用原型）；
2. Always-off retrieval（始终禁用原型）；
3. Similarity threshold（相似度阈值）；
4. Oracle selective retrieval（先知选择性检索，仅作诊断）；
5. FUSR（可部署效用路由）。

先知上界用于判断该数据集是否存在足够的选择空间。如果先知策略相对最佳固定策略几乎没有收益，继续优化效用路由器没有研究价值。

## 13. 为什么不选择其他创新路线

| 候选方向 | 不作为主创新的原因 | 可保留用途 |
|---|---|---|
| 普通 InfoNCE 对比学习 | CTRL、CRIB、MTS-LOF 等已有大量缺失鲁棒对比或一致性学习；直接添加损失新意弱 | 作为 MIPP 的对照实现 |
| 频谱最优传输 | PSW-I 已把频谱沃瑟斯坦用于时序插补；RIDDE 等也开始分解检索影响 | 可作为附加分析，不进入主线 |
| 强化学习 | 当前读写动作具有直接可监督的预测损失差，强化学习会增加方差和训练复杂度 | 若未来形成长期在线记忆再讨论 |
| 完整 MAML 元学习 | 计算成本高，容易把故事从原型失效转成通用元学习 | 跨缺失域情景采样保留为训练协议 |
| 通用双原型库 | DPAD 已有常见/稀有双原型库和上下文路由 | 作为相似机制对比 |
| 不确定性预测头 | 不能直接解决原型污染和负效用，容易形成第三条松散主线 | 路由器内部只估计效用不确定性 |

## 14. 相似工作与边界

| 工作 | 已做内容 | 与本项目重叠 | 本项目必须证明的区别 |
|---|---|---|---|
| S4M，ICLR 2025 | 缺失感知双流 S4、原型库 | 基础模型 | 缺失域漂移下原型伤害诊断与治理 |
| StableMiss，IJCAI 2023 | 未知掩码分布漂移、双参数化、去相关 | 缺失漂移与解耦 | 时序原型库、配对掩码干预、效用读写 |
| StableMiss+，ICASSP 2024 | 保留特征内相关，去除掩码相关 | 掩码去相关 | 内容/缺失双因子与原型供给净化 |
| PRIME，2024 | 原型辅助不规则时序插补 | 缺失时序中的原型 | 直接预测、漂移和效用治理 |
| CRIB，2025 | 一致性正则与信息瓶颈 | 跨掩码稳健表示 | 原型库位置、效用监督、读写闭环 |
| ChannelTokenFormer，ICLR 2026 | 通道依赖、异步采样、测试时缺失块 | 强缺失预测基线 | 重点不是新主干，而是原型可靠性 |
| TameR，ICML 2026 | 历史遮挡和全局上下文缓解近期数据偏置 | 遮挡鲁棒训练 | MIPP 必须优于普通遮挡增强 |
| RAFT，ICML 2025 | 检索历史片段及其后续值辅助预测 | 检索增强预测 | 内部原型、缺失域与效用标签 |
| Cross-RAG，2026 | 交叉注意力选择相关检索样本 | 选择性检索 | 相关性不等于下游效用 |
| SARAF，KDD 2026 | 平稳性感知检索、多样性与融合 | 相似度检索可能不可靠 | 样本级效用、原型写入与拒用 |
| RIDDE，2026 | 检索引导的不变/动态分解 | 表示分解 | 分解来源是掩码干预，直接服务原型净化 |
| DPAD，2026 | 常见/稀有双原型库和上下文路由 | 原型路由 | 缺失特异性、预测损失效用、写入治理 |
| STRATA-TS，2025 | 选择性跨域检索以避免负迁移 | 有益知识筛选 | 场景、模型和监督信号不同 |

最安全的贡献表述是：

1. 我们首先系统量化 S4M 原型库在缺失域漂移下的有害检索率和原型污染，而不是声称首次研究缺失漂移。
2. 我们提出 MIPP，用配对掩码干预稳定性净化进入 S4M 原型库的内容表示，同时保留独立缺失表示。
3. 我们提出 FUSR，用原型分支开/关产生的预测损失差监督原型读取、写入和拒用。
4. 我们建立固定掩码、跨缺失域、可复现的评估套件，验证机制而不只报告平均误差。

## 15. 缺失机制漂移评估套件

套件暂称 MSS（Missingness Shift Suite，缺失机制漂移评估套件）。它不是为了增加实验数量，而是为了直接检验论文假设。

### 15.1 缺失维度

1. MCAR（Missing Completely At Random，完全随机缺失）。
2. MAR（Missing At Random，依赖已观测变量的随机缺失）。
3. MNAR（Missing Not At Random，依赖未观测值本身的非随机缺失）。
4. Point missing（随机点缺失）。
5. Time-point block missing（整时间点连续块缺失）。
6. Variable block missing（逐变量连续块缺失）。
7. Periodic outage（周期性掉线）。
8. Mixed outage（混合故障）。

缺失生成器必须避免使用未来目标来决定历史掩码。建议按以下规则实现：

- MCAR 点缺失：对历史窗口内每个位置独立采样伯努利变量，再通过可复现校准达到目标实际缺失率。
- MAR 缺失：只使用当前时刻之前仍可观测的驱动变量 \(v_{t-1}\)，令缺失概率为 \(\sigma(a v_{t-1}+b)\)，其中 \(b\) 通过二分搜索校准目标缺失率。驱动变量的可观测性规则必须预先固定。
- MNAR 自掩码：在生成合成资产时可用完整历史值 \(x_{t,d}\) 决定其自身被隐藏的概率 \(\sigma(a|z(x_{t,d})|+b)\)。隐藏后的值绝不能进入模型输入。该机制只是可控压力测试，不等同于真实 MNAR 因果识别。
- 连续块缺失：先采样起点，再按固定或分布采样的块长度扩展；重叠后记录实际缺失率。
- 周期掉线：按周期和相位生成规则块，并加入少量随机抖动，避免退化为固定时间编码。
- 混合故障：预先定义各生成器权重，不根据测试结果重新选择。

所有归一化统计只从未遮挡的训练集可用值估计。验证和测试阶段复用同一统计量，禁止各自重新拟合。

### 15.2 漂移类型

| 漂移 | 训练 | 测试 |
|---|---|---|
| 缺失率漂移 | `r=0.06` | `r=0.03/0.12/0.24` |
| 块长度漂移 | 长度 5 | 长度 1/12/24 |
| 形状漂移 | 整时间点块 | 逐变量块/随机点/周期掉线 |
| 机制漂移 | MCAR | MAR/MNAR |
| 复合漂移 | 单一训练域或源域混合 | 未见机制 + 未见长度 + 未见缺失率 |
| 自然漂移 | USHCN 较早时间段 | 后续时间段真实掩码 |

为了分离块长度和实际缺失率影响，块长度漂移实验需额外提供 realized-rate-matched（实际缺失率匹配）版本。

### 15.3 固定评估资产

测试掩码保存为：

```text
artifacts/masks/{dataset}/{mechanism}/{pattern}/rate_{rate}/length_{length}/seed_{seed}.npz
```

每个资产包含：

- 掩码矩阵；
- 数据划分哈希；
- 名义缺失率和实际缺失率；
- 生成器版本；
- seed（随机种子）；
- 机制、形状、长度参数；
- 数据源文件哈希；
- 创建时间。

训练掩码可动态生成，但生成规则和随机数状态必须固定；验证和测试掩码禁止评估时临时重采样。

## 16. 完整实验计划

### 16.1 A 组：官方复现与协议审计

目标：建立可信规范基线，并量化论文、公开代码、缺陷修复之间的差异。

- 四个公共数据集。
- 两种原始缺失形状。
- 四个名义缺失比例。
- 四个回看长度。
- 预测长度 96。
- 第一轮固定 seed 用于对照论文完整表格。
- 关键设置补 3 seed；规范基线确认后，主结果补 5 seed。

必须同时报告论文值、公开代码值、修复值和规范基线值。

### 16.2 B 组：中心假设诊断

在不改模型前，记录每个测试样本：

- 有原型预测损失；
- 无原型预测损失；
- Top-K 相似度；
- 当前缺失域；
- 原型来源缺失域；
- 原型年龄和队列位置。

检验缺失漂移是否提高有害检索率，以及相似度与真实效用是否脱钩。

### 16.3 C 组：单模块公平比较

按完全相同预算依次运行：

1. `s4m_canonical`
2. `s4m + random mask augmentation`
3. `s4m + MIPP alignment only`
4. `s4m + full MIPP`
5. `s4m + FUSR read gate only`
6. `s4m + FUSR read/write governance`
7. `MIRAGE-S4M full`

第一轮只改变一个主要因素；只有单模块确实有效，才进入组合实验。

### 16.4 D 组：主跨域矩阵

对四个公共数据集运行：

- 原始域内条件；
- 缺失率漂移；
- 块长度漂移；
- 形状漂移；
- MCAR→MAR；
- MCAR→MNAR；
- 复合漂移。

所有主方法至少 3 seed。最重要的 ETTh1、Weather、Traffic 和自然缺失设置补 5 seed。

### 16.5 E 组：强基线

优先级从高到低：

1. `s4m_canonical`
2. ChannelTokenFormer
3. CRIB
4. CoIFNet
5. TameR
6. BiTGraph
7. GRU-D 与 BRITS
8. S4 Mean/Ffill/Decay/SAITS
9. S4M + 简单相似度阈值门
10. S4M + Cross-RAG 风格交叉注意力门
11. S4M + DPAD 风格相似度/稀有性路由
12. 可公平适配时加入 RAFT 或 SARAF

外部基线必须使用同一数据划分、同一测试掩码和同一评价脚本。若其官方训练预算不同，应保留官方推荐设置并另做同预算版本，不可只选对本项目有利的一个。

### 16.6 F 组：自然缺失

优先恢复 S4M 附录使用的 USHCN 官方资产：

- 按时间顺序划分；
- 使用真实掩码；
- 未来只在实际观测位置计算误差；
- 统计训练/测试时期掩码分布差异；
- 同时运行自然掩码和嵌套受控遮挡；
- 不用插补结果充当预测真值。

如果 USHCN 原论文处理过程无法公开还原，应明确标记 `PARTIAL`，并选择具有公开数据许可和自然缺失的第二数据集，不得伪造相同协议。

## 17. 消融实验

### 17.1 MIPP 消融

- 无配对掩码干预；
- 仅预测一致性；
- 仅内容对齐；
- 移除内容/缺失解耦；
- 移除缺失重建；
- 移除稳定性写入门；
- 内容和缺失重新混合写入；
- 使用普通 InfoNCE 代替当前约束；
- 使用同域掩码对与跨域掩码对比较。

### 17.2 FUSR 消融

- 只使用余弦相似度；
- 使用固定相似度阈值；
- 只做读取门控；
- 只做写入门控；
- 移除原型拒用；
- 移除历史效用统计；
- 移除缺失域距离；
- 移除 OOD 距离；
- 聚合效用标签与单原型效用标签比较；
- 精确效用探针与一阶近似比较；
- 不同效用探针频率和 \(K_u\) 比较。

### 17.3 组合消融

- S4M
- S4M + MIPP
- S4M + FUSR
- S4M + MIPP + FUSR
- 完整模型但禁用原型写入治理
- 完整模型但禁用检索拒用

## 18. 评价指标

### 18.1 预测精度

- MSE：均方误差。
- MAE：平均绝对误差。
- RMSE（Root Mean Squared Error，均方根误差），作为补充。

### 18.2 泛化指标

- ID average（域内平均误差）。
- OOD average（分布外平均误差）。
- Worst-domain error（最差缺失域误差）。
- Relative degradation（相对退化率）。
- Source-to-target transfer matrix（源域到目标域迁移矩阵）。

### 18.3 原型机制指标

有害检索率：

\[
\operatorname{HRR}=\frac{1}{N}\sum_n\mathbb{1}[\ell_n^+>\ell_n^0+\delta].
\]

其中 HRR 是 Harmful Retrieval Rate（有害检索率）。

平均检索效用：

\[
\operatorname{MRU}=\frac{1}{N}\sum_n(\ell_n^0-\ell_n^+).
\]

负效用严重度：

\[
\operatorname{NUS}=\frac{1}{N}\sum_n\max(0,\ell_n^+-\ell_n^0).
\]

还需报告：

- 原型库污染率；
- 跨掩码内容稳定性；
- 相似度与真实效用的 Spearman 等级相关；
- 效用正负分类 AUROC（受试者工作特征曲线下面积）与 AUPRC（精确率-召回率曲线下面积）；
- 效用回归 MAE；
- Brier score（布里尔分数）与 ECE（Expected Calibration Error，期望校准误差）；
- 检索覆盖率-风险曲线与 AURC（Area Under the Risk-Coverage Curve，风险-覆盖曲线下面积）。

### 18.4 代价指标

- 新增参数量和总参数量；
- 单 epoch 时间；
- 总训练时间；
- 推理延迟；
- 峰值显存；
- 原型库容量；
- 效用探针额外计算量；
- 可可靠获得时报告 FLOPs（Floating-Point Operations，浮点运算次数）。

## 19. 统计检验

1. 主矩阵至少 3 个独立随机种子，关键结果 5 个随机种子。
2. 报告均值、标准差和 95% 置信区间。
3. 使用同一固定掩码和相同测试窗口进行配对比较。
4. 对时间有序损失差使用 Diebold-Mariano 预测精度检验，并采用适合重叠预测窗口的异方差自相关稳健方差估计。
5. 使用 moving block bootstrap（移动块自助法）构造时间相关置信区间。
6. 多个数据集/条件同时检验时使用 Holm 方法校正多重比较。
7. 不把单个 seed 的最好结果作为正式结论。

## 20. 预注册式成功与停止标准

以下阈值是项目决策标准，不是对结果的承诺。

### Gate R：复现可信

- 数据、训练、保存、加载和评估链路完整。
- 公开代码与论文协议差异全部落盘。
- 至少 ETTh1 和 Weather 的规范基线可稳定重复。
- 与论文值的偏差可量化并有证据解释。

### Gate H：中心失效现象成立

- 至少两个数据集上，缺失域漂移使 HRR 稳定增加。
- 相似度与真实效用相关性不足以可靠区分有益/有害原型。
- 该现象在多个随机种子和至少两类漂移上出现。

若 Gate H 不成立，应停止 FUSR 主线，不能强行继续完整矩阵。

### Gate M：MIPP 有效

- 跨掩码内容稳定性显著提高；
- 原型污染率下降；
- OOD MSE 有稳定改善；
- 域内 MSE 退化不超过约 1%；
- 优于普通随机掩码增强和普通一致性正则。

### Gate U：FUSR 有效

- 效用正负 AUROC 达到可用水平，建议至少 0.70；
- HRR 相对下降建议至少 25%；
- 最差缺失域 MSE 有稳定改善；
- 简单相似度阈值不能取得相同效果；
- 推理额外成本合理。

### Gate Q2：达到二区证据强度的候选条件

- 四个公共数据集和至少一个自然缺失数据集上均有完整结果；
- 至少三类未见缺失漂移上改善稳定；
- 最差域 MSE 相对规范 S4M 建议提高约 5% 或以上，且多数强基线不占优；
- 域内性能基本不牺牲；
- 关键结果 5 seed、统计显著；
- HRR、效用校准和污染率证明确实解决了声称的机制；
- 参数、时间和显存代价可接受；
- 2026 强基线、消融、失败案例和自然数据齐全。

如果只在少量合成条件上提高 1%–2%，或机制指标没有改善，最多属于弱增量，不应声称达到二区水平。

## 21. 公平比较规则

创新与规范 S4M 的第一轮比较必须保持：

- 相同 epoch；
- 相同早停；
- 相同批大小；
- 相同学习率、优化器和调度器；
- 相同数据划分；
- 相同训练样本；
- 相同验证/测试掩码；
- 相同 seed；
- 相同 checkpoint 选择规则；
- 相同评价脚本。

MIPP 需要额外掩码视图，这是方法本身的训练输入，不得通过减少训练数据或 epoch 抵消成本。必须如实报告其额外时间和显存。

## 22. 配置开关

所有创新都必须可关闭：

```yaml
model:
  use_mipp: false
  use_content_mask_factorization: false
  use_mask_intervention_pair: false
  use_stability_write_gate: false
  use_fusr: false
  use_utility_reranking: false
  use_utility_read_gate: false
  use_utility_write_gate: false
  use_retrieval_abstention: false
  use_prototype_quarantine: false
```

全部关闭时必须恢复规范 S4M 行为。模块关闭后不得残留额外损失、参数更新或数据副作用。

## 23. 工程结构

```text
S4M_MissingForecasting/
├── upstream/S4M/                 # 官方代码，只读保存
├── docs/                         # 正式项目文档
├── references/                   # 文献导出与查新记录
├── src/
│   └── mirage_s4m/
│       ├── adapters/             # S4M 兼容适配层
│       ├── data/                 # 数据与固定划分
│       ├── missingness/          # 缺失生成器和缺失签名
│       ├── models/               # MIPP、FUSR、完整模型
│       ├── memory/               # 原型库、元数据、隔离区
│       ├── training/             # 分阶段训练逻辑
│       ├── evaluation/           # 公平评估和机制指标
│       └── utils/                # 配置、日志、随机状态
├── configs/                      # 基线、模块和正式矩阵配置
├── scripts/                      # 初始化、训练、评估、调度脚本
├── artifacts/
│   └── masks/                    # 固定测试掩码
├── tests/                        # 单元、集成、泄漏和回归测试
├── runs/                         # 唯一实验 ID 的完整资产
├── outputs/                      # 中间输出
├── results/                      # 不可变原始结果
├── reports/                      # 自动汇总和分析
└── logs/                         # 调度器与全局日志
```

不得直接改写 `upstream/S4M`。所有路径修复、设备修复、指标保存和创新代码优先放在本地适配层；若必须修改上游代码，则保存独立补丁并明确区分复现修复与创新修改。

## 24. 每个实验的持久化资产

实验 ID 采用：

```text
{model}__{dataset}__{mask_type}__mr{rate}__seed{seed}__{timestamp}
```

每个正式实验必须包含：

```text
runs/<experiment_id>/
├── resolved_config.yaml
├── command.sh
├── env.txt
├── git_state.txt
├── code.diff
├── train.log
├── metrics.json
├── status.json
├── resource_usage.json
└── checkpoints/
```

必须记录完整配置、命令、环境、代码状态、固定掩码路径、GPU UUID、逻辑/物理 GPU 映射、峰值显存、训练时间、退出码和失败类型。

## 25. Smoke test 与正式实验

任何新数据流程、配置或模块必须先完成独立 smoke test（冒烟测试）：

- 少量样本；
- 1 个 epoch；
- forward（前向传播）；
- backward（反向传播）；
- optimizer step（优化器更新）；
- checkpoint 保存与重新加载；
- 验证和测试；
- `metrics.json` 写入；
- 基线开关回归测试；
- 测试目标泄漏检查。

Smoke test 只证明链路可运行，不作为论文结果，也不会替代完整训练预算。

## 26. GPU 与调度

正式矩阵使用统一后台调度器：

- 最大并发 3，但按实时资源动态决定；
- 使用 `flock` 做 GPU 选择和启动的原子锁；
- 启动前核对 GPU index、UUID、名称、总显存、已用和空闲显存；
- 明确服务器逻辑 0/1 与物理 GPU 的反向映射；
- 未知任务先 profiling（性能与显存剖析）；
- 保留动态安全显存余量；
- OOM（Out of Memory，显存不足）采用退避重试，不能秒级死循环；
- 只清理本项目确认归属的进程；
- 支持断点续训、状态恢复、安全停止和自动结果汇总。

当前会话中 `nvidia-smi` 无法连接驱动，PyTorch 报告 CUDA 不可用，因此尚未启动训练。正式 GPU 工作前必须重新从宿主状态验证设备。

## 27. 自动结果总表

每轮结束自动生成：

```text
reports/latest_results.csv
reports/latest_results.md
```

至少包含：实验 ID、数据集、模型、缺失机制、缺失形状、名义/实际缺失率、块长度、seed、MSE、MAE、HRR、MRU、运行时间、峰值显存、参数量、状态、Git 提交、配置路径和 checkpoint 路径。

失败实验也必须保留状态和错误摘要。

## 28. 主要风险与应对

| 风险 | 后果 | 应对 |
|---|---|---|
| 缺失漂移不增加有害检索 | 中心假设失败 | Gate H 后停止，不扩大矩阵 |
| MIPP 等价于普通遮挡增强 | 创新不足 | 与 TameR 风格遮挡和一致性基线直接比较 |
| FUSR 等价于相似度阈值 | 机制贡献不足 | 加入固定阈值、Cross-RAG/DPAD 风格门控对照 |
| 效用标签随主干快速变化 | 路由器不稳定 | 教师网络、停止梯度、预热和分阶段训练 |
| 单原型探针成本过高 | 训练时间过长 | 分层抽样、向量化、探针间隔和近似消融 |
| 完全掩码不变损害 MNAR | 丢失信息性缺失 | 内容/缺失双因子，保留缺失分支 |
| USHCN 处理无法还原 | 自然数据证据不足 | 保留 PARTIAL，增加公开自然缺失数据集 |
| 官方代码与论文协议冲突 | 基线可信度不足 | 四种基线身份分离，修复收益不计创新 |
| 2026 新工作继续出现 | 新颖性被覆盖 | 实验完成和投稿前再次系统查新 |
| 数据盘空间紧张 | 任务中断 | 单份数据资产、环境复用、checkpoint 保留策略 |

## 29. 论文叙事

### 29.1 引言逻辑

1. 缺失值时序预测需要直接从部分观测历史生成未来，而真实部署中的缺失机制会变化。
2. S4M 通过原型记忆增强缺失输入表示，但默认高相似原型总是有益。
3. 我们发现缺失域漂移会同时污染写入表示并增加有害检索，余弦相似度无法可靠预测下游效用。
4. MIPP 从供给侧净化进入原型库的内容，FUSR 从使用侧按预测效用治理原型。
5. 固定跨缺失域套件和机制指标证明改善来自更可靠的原型使用，而不是更大模型或不同掩码。

### 29.2 结果叙事顺序

1. 先展示 S4M 的有害检索现象。
2. 再展示相似度与真实效用失配。
3. 展示 MIPP 减少原型污染。
4. 展示 FUSR 降低 HRR 并改善最差域误差。
5. 展示完整模型在域内和分布外的权衡。
6. 最后报告代价、失败案例和自然缺失结果。

### 29.3 不能使用的措辞

- “首次研究缺失机制漂移”；
- “首次将原型用于缺失时序”；
- “首次选择性检索时序原型”；
- “保证达到二区”；
- “在所有设置上达到最优”，除非完整结果确实支持；
- 把 smoke test、单 seed 或不同测试掩码结果写成正式提升。

## 30. 项目阶段与交付物

### 阶段 0：正式立项

- 本文档；
- 文献导出；
- 官方代码和环境审计。

### 阶段 1：复现修复

- 不修改上游行为的适配入口；
- 路径、设备、离线日志、指标保存和 checkpoint 修复；
- 四种 S4M 基线身份；
- 基线 smoke test。

### 阶段 2：数据和掩码资产

- 四个公共数据集和 USHCN；
- 固定数据划分与归一化统计；
- 原协议掩码；
- MSS 跨域掩码；
- 资产哈希和生成测试。

### 阶段 3：规范基线与中心诊断

- 官方完整表格复现；
- 有/无原型损失记录；
- HRR、MRU、相似度-效用关系；
- Gate H 决策。

### 阶段 4：MIPP

- 可关闭实现；
- 单元和回归测试；
- 单模块公平实验；
- Gate M 决策。

### 阶段 5：FUSR

- 效用标签、路由器、读写治理和拒用；
- 泄漏测试；
- 单模块与组合实验；
- Gate U 决策。

### 阶段 6：完整正式矩阵

- 多 seed；
- 强基线；
- 消融；
- 自然缺失；
- 复杂度、统计和失败案例。

### 阶段 7：论文与开源

- 论文主文和补充材料；
- 一键环境、数据、训练和评估说明；
- 代码与数据可用性声明；
- 最终查新和投稿定位。

## 31. 项目完成判据

只有同时满足以下条件，才称为“项目完整完成”：

1. 独立环境可重建。
2. 官方和规范 S4M 均可训练、保存、加载和评估。
3. 官方表格主要条件的偏差已量化。
4. 所有固定评估资产可校验。
5. MIPP 和 FUSR 均可独立关闭。
6. 基线没有被创新代码破坏。
7. 主实验至少 3 seed，关键结果 5 seed。
8. 强基线、消融、自然数据、统计和复杂度齐全。
9. 所有结果能由实验 ID、代码版本、配置、环境和数据资产还原。
10. 负结果和失败实验没有从报告中消失。
11. 论文贡献措辞与实验证据一致。
12. 只有达到 Gate Q2 的证据条件，才建议优先冲击二区；否则按证据水平调整目标期刊。

## 32. 当前水平判断

### 仅从想法看

当前方案属于**中高水平、主干特异性较强但尚未被实验验证的研究设计**。它明显强于“给 S4M 加两个通用损失”，也比对比学习、频谱最优传输、强化学习和元学习并列更容易形成单一因果链。

### 达到二区的潜力

如果以下三点全部成立，具备较现实的二区竞争力：

1. 先发现并稳定复现“缺失漂移提高有害原型检索”这一新诊断现象；
2. MIPP 和 FUSR 分别改善污染率与有害检索率，完整模型显著改善最差域预测；
3. 在 2025–2026 强基线、自然缺失、多 seed 和成本约束下仍成立。

如果只有平均 MSE 小幅提高，而没有 HRR、效用监督和原型污染的机制证据，创新会被评价为复杂门控或模块组合，达不到预期水平。

## 33. 英文术语与缩写表

| 术语/缩写 | 中文说明 |
|---|---|
| S4 | Structured State Space Sequence，结构化状态空间序列模型 |
| S4M | S4 for Multivariate Time Series Forecasting with Missing Values，面向缺失值多变量时序预测的 S4 |
| ATPM | Adaptive Temporal Prototype Mapper，自适应时间原型映射器 |
| MDS-S4 | Missing-Aware Dual-Stream S4，缺失感知双流 S4 |
| MIRAGE-S4M | Missingness-Intervention and Retrieval-utility-Aware Generalization Enhancement for S4M，缺失干预与检索效用感知泛化增强 S4M |
| MIPP | Mask-Intervention Prototype Purification，掩码干预原型净化 |
| FUSR | Forecast-Utility-Aware Selective Retrieval，预测效用感知的选择性原型检索 |
| MSS | Missingness Shift Suite，缺失机制漂移评估套件 |
| MTSF | Multivariate Time-Series Forecasting，多变量时序预测 |
| MCAR | Missing Completely At Random，完全随机缺失 |
| MAR | Missing At Random，随机缺失 |
| MNAR | Missing Not At Random，非随机缺失 |
| ID | In-Distribution，分布内/域内 |
| OOD | Out-of-Distribution，分布外 |
| HRR | Harmful Retrieval Rate，有害检索率 |
| MRU | Mean Retrieval Utility，平均检索效用 |
| NUS | Negative Utility Severity，负效用严重度 |
| MSE | Mean Squared Error，均方误差 |
| MAE | Mean Absolute Error，平均绝对误差 |
| RMSE | Root Mean Squared Error，均方根误差 |
| AUROC | Area Under the Receiver Operating Characteristic Curve，受试者工作特征曲线下面积 |
| AUPRC | Area Under the Precision-Recall Curve，精确率-召回率曲线下面积 |
| ECE | Expected Calibration Error，期望校准误差 |
| AURC | Area Under the Risk-Coverage Curve，风险-覆盖曲线下面积 |
| VICReg | Variance-Invariance-Covariance Regularization，方差-不变性-协方差正则化 |
| HSIC | Hilbert-Schmidt Independence Criterion，希尔伯特-施密特独立性准则 |
| BCE | Binary Cross-Entropy，二元交叉熵 |
| CE | Cross-Entropy，交叉熵 |
| EMA | Exponential Moving Average，指数移动平均 |
| OOM | Out of Memory，显存或内存不足 |
| FLOPs | Floating-Point Operations，浮点运算次数 |
| W&B | Weights & Biases，实验跟踪平台 |
| SGD | Stochastic Gradient Descent，随机梯度下降 |
| Adam | Adaptive Moment Estimation，自适应矩估计优化器 |
| SCI | Science Citation Index，科学引文索引 |
| Q2/Q3 | Journal Quartile 2/3，期刊二区/三区；具体口径需按投稿当年的中科院或 JCR 核验 |
| Top-K | 取得分最高的前 K 个候选 |
| FIFO | First-In-First-Out，先进先出 |
| checkpoint | 训练检查点，保存模型和优化状态的文件 |
| smoke test | 冒烟测试，用极小任务验证完整链路可运行 |
| profiling | 性能剖析，用于测量时间、显存和计算热点 |
| baseline | 基线方法，用于公平比较的参照模型 |
| prototype | 原型，代表一类历史时序内容的记忆表示 |
| utility router | 效用路由器，预测候选原型是否有助于下游预测的轻量模型 |
| Oracle selector | 先知选择器，使用真实未来损失选择原型，仅用于诊断上界 |
| GPU | Graphics Processing Unit，图形处理器 |
| CPU | Central Processing Unit，中央处理器 |
| CUDA | Compute Unified Device Architecture，英伟达并行计算平台 |
| UUID | Universally Unique Identifier，通用唯一标识符；本项目用它稳定识别 GPU |
| Git | 分布式版本控制系统 |
| README | 项目说明文件 |
| LICENSE | 软件许可文件 |
| Conda | 环境与软件包管理工具 |
| PyTorch | 深度学习框架 |
| NumPy | Python 数值数组计算库 |
| Pandas | Python 表格与数据处理库 |
| MAML | Model-Agnostic Meta-Learning，与模型无关的元学习 |
| InfoNCE | Information Noise-Contrastive Estimation，信息噪声对比估计损失 |
| CRIB | Consistency-Regularized Information Bottleneck，一致性正则化信息瓶颈 |
| PSW-I | Proximal Spectrum Wasserstein for Imputation，近端频谱沃瑟斯坦插补 |
| VIMTS | Visual Masked Autoencoder for Irregular Multivariate Time Series，面向不规则多变量时序的视觉掩码自编码模型 |
| DPAD | Dual-Prototype Adaptive Disentanglement，双原型自适应解耦 |
| RIDDE | Retrieval-guided Invariant-Dynamic Decomposition，检索引导的不变-动态分解 |
| SARAF | Stationarity-Aware Retrieval-Augmented Forecasting，平稳性感知检索增强预测 |
| RAFT | Retrieval Augmented Time Series Forecasting，检索增强时序预测 |
| GRU-D | Gated Recurrent Unit with Decay，带衰减机制的门控循环单元 |
| BRITS | Bidirectional Recurrent Imputation for Time Series，双向循环时序插补 |
| USHCN | United States Historical Climatology Network，美国历史气候网络 |
| JCR | Journal Citation Reports，期刊引证报告 |
| ICLR | International Conference on Learning Representations，国际学习表征会议 |
| ICML | International Conference on Machine Learning，国际机器学习会议 |
| IJCAI | International Joint Conference on Artificial Intelligence，国际人工智能联合会议 |
| ICASSP | IEEE International Conference on Acoustics, Speech and Signal Processing，IEEE 国际声学、语音与信号处理会议 |
| KDD | ACM SIGKDD Conference on Knowledge Discovery and Data Mining，ACM 数据挖掘与知识发现会议 |
| NeurIPS | Conference on Neural Information Processing Systems，神经信息处理系统大会 |
| StableMiss | 面向未知掩码分布漂移的稳定不完整数据预测方法 |
| PRIME | 使用跨序列原型信息进行不规则时序插补的模型名 |
| CTRL | 基于神经受控微分方程的通用时序表征学习框架名 |
| MTS-LOF | Medical Time-Series Representation Learning via Occlusion-Invariant Features，基于遮挡不变特征的医疗时序表征学习 |
| ChannelTokenFormer | 通道令牌 Transformer，同时处理通道依赖、异步采样与缺失 |
| CoIFNet | Collaborative Imputation-Forecasting Network，协同插补-预测网络 |
| TameR | 通过全局上下文缓解近期数据偏置的鲁棒时序预测方法 |
| Cross-RAG | Cross-Attention Retrieval-Augmented Generation，基于交叉注意力的检索增强方法 |
| STRATA-TS | Selective Transfer via Target-Aware Retrieval for Time Series，基于目标感知检索的时序选择性迁移 |

## 34. 关键参考文献

1. Peng, J. et al. S4M: S4 for Multivariate Time Series Forecasting with Missing Values. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/7b2f0758334389b8ad0665a9bd165463-Abstract-Conference.html
2. Zhu, Y. et al. Prediction with Incomplete Data under Agnostic Mask Distribution Shift. IJCAI 2023. https://www.ijcai.org/proceedings/2023/525
3. Zhu, Y. & Jiang, B. StableMiss+: Prediction with Incomplete Data under Agnostic Mask Distribution Shift. ICASSP 2024. https://doi.org/10.1109/ICASSP48485.2024.10446980
4. Han, S. et al. Retrieval Augmented Time Series Forecasting. ICML 2025. https://openreview.net/forum?id=GUDnecJdJU
5. Jang, J. et al. Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework for Dependency, Asynchrony, and Missingness. ICLR 2026. https://iclr.cc/virtual/2026/poster/10007178
6. Yang, J. et al. Revisiting Multivariate Time Series Forecasting with Missing Values. arXiv:2509.23494. https://arxiv.org/abs/2509.23494
7. Tang, K. et al. CoIFNet: A Unified Framework for Multivariate Time Series Forecasting with Missing Values. arXiv:2506.13064. https://arxiv.org/abs/2506.13064
8. Xu, L. et al. Taming the Recent-Data Bias: Towards Robust Time Series Forecasting with Global Context. ICML 2026. https://openreview.net/forum?id=gDA03Yn3fi
9. Liu, Z. et al. Improving Time Series Forecasting via Instance-aware Post-hoc Revision. NeurIPS 2025. https://openreview.net/forum?id=H7e5RpeIi4
10. Lee, S. et al. Cross-RAG: Zero-Shot Retrieval-Augmented Time Series Forecasting via Cross-Attention. arXiv:2603.14709. https://arxiv.org/abs/2603.14709
11. Zhou, S. et al. Stationarity-Aware Retrieval-Augmented Time Series Forecasting. KDD 2026. https://arxiv.org/abs/2606.04135
12. Chi, J. et al. Factorize to Generalize: Retrieval-Guided Invariant-Dynamic Decomposition for Time Series Forecasting. arXiv:2605.24911. https://arxiv.org/abs/2605.24911
13. Yang, H. et al. Dual-Prototype Disentanglement: A Context-Aware Enhancement Framework for Time Series Forecasting. arXiv:2601.16632. https://arxiv.org/abs/2601.16632
14. Jiang, Y. et al. STRATA-TS: Selective Knowledge Transfer for Urban Time Series Forecasting with Retrieval-Guided Reasoning. arXiv:2508.18635. https://arxiv.org/abs/2508.18635
15. Wang, H. et al. Optimal Transport for Time Series Imputation. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/033394ac055e3c47bfc3c967ffcfee74-Abstract-Conference.html
16. Hu, Z. et al. IMTS is Worth Time × Channel Patches: Visual Masked Autoencoders for Irregular Multivariate Time Series Prediction. ICML 2025. https://proceedings.mlr.press/v267/hu25y.html

参考文献管理器导出文件位于 `references/related_work.enw`。

## 35. 下一步最短路径

1. 创建本地 S4M 适配入口，保持 `upstream/S4M` 不变。
2. 实现路径、设备、离线日志、配置解析、checkpoint 和指标持久化修复。
3. 生成 Weather 最小 smoke 数据链，验证保存、加载和评估。
4. 获取 ETTh1 与 Weather 完整数据，固化原始协议测试掩码。
5. 运行四种 S4M 基线身份的最小诊断，确定规范基线。
6. 在任何创新实现前先完成有/无原型诊断，验证 Gate H。

这一路径不会缩减正式实验，只把完整项目按证据依赖顺序执行，避免在中心假设尚未成立时提前消耗全部计算资源。
