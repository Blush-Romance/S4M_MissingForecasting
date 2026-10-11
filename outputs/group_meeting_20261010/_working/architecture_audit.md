# S4M / PG95 / CNM 实际代码架构审计

审计日期：2026-10-10。范围：当前运行副本 `baselines/S4M/s4m`、正式 resolved config 与 CNM 入口；只读代码审计，未启动模型、GPU、测试或实验，未改模型。项目根非 Git，运行副本 Git HEAD 为 `a718823addd3606e763dfc261174e0135b2535f4`，工作区已有较多用户修改；没有覆盖或清理。本文不是方案图，也不使用计划文档代替实现。

## 1. 最小主干图与边清单

记号：`B` batch，`L` 历史长度，`C` 变量数，`S=short_len`，`H=en_conv_hidden_size`，`D=d_model`，`P=pred_len`。

```text
历史输入 X,M [B,L,C] + 历史窗口 max/min 值及其距离
                 |
       decay imputation → Z [B,L,C] ---------------------------+
                 |                                           |
       左填充、长度 S 的滑窗 [B,L,S,C]                         |
                 |                                           |
       Encoder_u: Conv2d→reshape→SelfAttn→S4→时间线性池化        |
                 |                                           |
       L2-normalized query U [B,L,H]                           |
                 |                                           |
      (PG95保留: h+A h，再归一化)                             |
                 |                                           |
       bank centroid similarity → TopK/softmax → R [B,L,H]     |
                 |                    |                       |
                 +--------------------+-----------------------+
                                      |
                     concat[R,U,Z] @ W + b + U
                                      |
                           context [B,L,H]
                                      |
                 mask-aware S4 layer 0 ← Linear(M):C→H
                                      |
                           linear1:H→D
                                      |
                       ordinary S4 layers 1…E−1
                                      |
                        LayerNorm → out_proj:D→C
                                      |
                    output [B,L,C] → tail P → Ŷ [B,P,C]
```

| 边 | 实际实现与证据 |
|---|---|
| `X,M,extrema → Z` | `S4M.py:319–332`；缺失处是两种历史极值的学习衰减加权，观测处保持 X。不是“最近上一/下一观测值”。极值/距离来源 `data_provider/data_loader.py:1249–1272`，DataLoader 输出 `:1279–1294`。 |
| `Z → sliding histories` | `Bank.py:156` 定义 Unfold，`:368–375` 左右 padding 后取前 L 个位置，输出 `[B,L,S,C]`；沿时间的左侧零填充滑窗。 |
| `sliding → Encoder_u → U` | `Bank.py:64–80,82–110,373–380`；Conv2d 核 `(W,C)`，ReLU，直接 reshape 成 `[B·L,Tc,H]`，attention，encoder 内 S4，`Linear(Tc,1)` 时间池化。U 最后 L2 normalize。 |
| `U → PG95 content adapter → Ũ` | `Bank.py:386–388`；`MIPP.py:25–26,47–50`，`Ũ=normalize(U+A U)`，A 零初始化。此路径在 PG95 推理保留。 |
| `bank centroids + U → R` | `Bank.py:423–443`；若原型数 > topM 随机抽 topM，再 `m @ U.T`，TopK、softmax 加权求和；数量较少则所有原型参与。不是新增交叉注意力编码器。 |
| `R,U,Z → context` | `Bank.py:262–267,624`；确切顺序 `[retrieved,query,raw_history]`，矩阵 W 投影到 H，再加 query 残差。这里 `raw_history` 已是传入 MTNet 的 decay-imputed Z，不是原始完整真值历史。 |
| `M → first S4 mask stream` | `S4M.py:181,362` 的 `mask_trans=Linear(C,H)`；`:278–290` 首层接 mask，其余层不接。首层 S4_mask 内部 mask 卷积与数据卷积相加，见 `model/S4/s4/models/s4/s4_mask1.py:1707–1765`。不要画成简单 mask 乘法门。 |
| `context → S4 stack → head` | `S4M.py:161–179,278–294,411–416`。首层 H 维，linear1 后 D 维，其余 E−1 层，norm 和逐时间点 Linear 输出 C 维；实验取尾 P，`exp_pretrain1.py:3306–3308`。没有额外 horizon decoder 或把 L 投影成 P 的时间全连接头。 |
| `S4 layer residual` | `S4M.py:418–441`：S4 输出残差，LayerNorm，1×1 Conv FFN，残差与 LayerNorm。虽然构造函数实例化 s4_1/s4_2，forward 仅调用 s4_1；不能据此画双向/双分支。 |

检索分数建议标注“相似度 / dot product”，而不是严格的 cosine attention：query 被 normalize，但 `FeatureMemory.bank_centroid` 是成员均值并未重新 normalize（`Memory.py:55–61`）。

填补代码的精确系数对应为 `a=exp(-decay1(dmax))`，`b=exp(-decay2(dmin))`，`Z=M⊙X+(1−M)⊙[(b/(a+b))⊙vmax+(a/(a+b))⊙vmin]`（`S4M.py:327–332`）。这是当前交叉权重对应，不能自行改写成“max 距离系数直接乘 max 值”；两线性层 `C→C`。固定/原始 dmin 分别按下文开关核验。

## 2. PG95 加在哪里，以及训练/推理界线

PG95 的真实启用项是 `use_mipp + use_mask_intervention_pair + use_stability_write_gate`，`use_content_mask_factorization=false`。历史 Weather resolved config `runs/mipp_pg95__weather__point__mr0.06__seed2023__20260717_234738/resolved_config.yaml:93–109`；Traffic `runs/mipp_pg95_traffic__traffic__point__mr0.06__seed2027__20260820_174327/resolved_config.yaml:92–111`；paired-RNG 版本 `configs/stage22_36_control/pg95_train_formal.yaml:84–105` 一致。

```text
完整历史 truth X* （仅训练构造干预，不是未来 target）
            |
    mask intervention Mᵇ: block∈{1,5,20}, rate~U(.15,.50)
            |
         Xᵇ=X*⊙Mᵇ → extrema → decay imputation Zᵇ
            |
    shared Encoder_u + shared content adapter → Uᵇ
            |                               ↑ primary Uᵃ
            +----------- invariance loss ----+

primary Zᵃ → nonoverlap S-blocks → frozen momentum Encoder_m → content A → qᵃ
paired Zᵇ  → same block boundaries → same Encoder_m → content A → qᵇ
                cosine(qᵃ,qᵇ)≥.95 → eligible candidates → bank write
                （no_grad；teacher EMA；原 memory 更新规则继续使用）
```

- 配对掩码生成在训练循环，不是模型内部第二套 encoder：`exp_pretrain1.py:978–1024,3276–3282`。mask 先采起点，因果 max_pool 形成块，沿 C 共享；paired history 从 `seq_true_x` 乘新掩码生成，不是严格嵌套于原始 mask，也不取未来数据。
- primary 和 paired 分别 decay-impute：`S4M.py:339–350`。辅助 query 分支止于共享 Encoder_u/content representation 与不变损失：`Bank.py:389–418`。PG95 不为 paired view 另做检索、完整 S4 预测或 paired forecast loss。
- 可学习 content adapter 位于查询检索之前，并复用到 teacher 写入/初始化：`Bank.py:287–297,386–399,636–645`。factorization 关闭不等于 adapter 关闭。mask encoder/decoder 仍由 `MIPPPurifier.__init__` 实例化（`MIPP.py:28–34`），但本配置不走它们，无额外 decorrelation/mask reconstruction 路径；不可画成已启用的“内容-掩码解耦网络”。
- `Encoder_u` 可训练；`Encoder_m` 在 `Bank.py:146–147` 冻结，`:269–275` 用 `θ_m←.99 θ_m+.01 θ_u` 更新；teacher 编码与 memory 写操作在 no_grad 内（`:627–705`）。adapter 本身被正常 query objective 训练，不属于独立 EMA teacher。
- 写门 `cos(qᵃ,qᵇ)≥0.95` 为 `absolute` 模式，见 `Bank.py:645–667`、`MIPP.py:92–100`。它筛的是非重叠 S 长历史块的写入候选，不筛 forecast token/输出，不是检索读门。筛后 `Memory.py:120–151` 仍随机取最多 M=50 候选，按旧 `thres1/thres2` 规则入簇或新建簇；`.95` stability 阈值和原 `.95` thres1 是不同条件。
- query 配对损失的 representation 为 `[B,L,H]`；写入 teacher candidate 为 `[B,floor(L/S),H]`，对应 stability/gate mask `[B,floor(L/S)]`（`Bank.py:628–645`）。写门并不对 `[B,L,H]` 的每个 query token 做阈值过滤。
- 推理不构造 pair，不执行不变损失或 memory 更新；`exp_pretrain1.py:1071–1073,1108–1126`、CNM `:6940–6942` 设置 memory `is_training=0`。query adapter 与已经学习的原型库仍参与推理。`Bank.forward` 仍会计算 teacher q 的旁路，但 `:700–704` 不写库；不能声称推理完全删除 teacher 计算。

### 训练损失与数值

`MIPP.py:375–418` 与 `Bank.py:411–417` 的完整公式为：

```text
L = Lforecast + λinv Linv + λvar Lvar + λcov Lcov + λdec Ldec + λmask Lmask
Lforecast = MSE(Ŷ, Y)，仅原训练 future mask==1 的 target 元素
Linv = mean_{b,t}[1−cos(Uᵃ_bt,Uᵇ_bt)]
PG95: (λinv,λvar,λcov,λdec,λmask)=(.05,0,0,0,0)
实际有效损失：L = Lforecast + .05 Linv
```

forecast loss 和相加位置分别 `exp_pretrain1.py:3325–3328,3400–3404`。零权重 variance/covariance 仍被 `paired_objective` 计算，不贡献有效损失；factorization=false 使 decorrelation/mask reconstruction 为零。不要使用 run.py 的 MIPP 泛化默认权重/默认 threshold=.80 作为 PG95 配置。

### warmup 路由，不能误画为训练 loss warmup schedule

- 每个新清空 memory bank 的第一 batch 做聚类初始化并 `continue`，不做 forecast/optimizer step；`exp_pretrain1.py:3252–3259,3283–3299`。`no_renew=false` 时每 epoch 清空且重 warm，true 时只有最初 warm。这是修复已有上游“清空后未重 warm”逻辑，不是 λinv 随 epoch ramp-up。
- baseline 的 `Model.warmup` 直接传 seq_x；PG95 pair 存在时传 decay-imputed primary/paired：`S4M.py:192–208`。
- warm 分支是非重叠 S-blocks、frozen Encoder_m、PG95 normalize+adapter，再 paired stability 筛选，随后 `FeatureMemory.cluster`：`Bank.py:277–363`。
- absolute 初始化门包含 `minimum_candidates=min(21,N)` 回退，以满足最多20簇的聚类需求：`Bank.py:308–317`、`MIPP.py:94–100`。因此初始化时不能断言所有写入候选都满足 .95；不足21时改取稳定性最高的至少21候选。常规写门没有这条最低候选回退。

## 3. CNM 是推理 wrapper，不是新模型层

新版 CNM-SE / multiscale：

```text
                    +→ original (X,M,extrema) → same fθ → Ŷorig -- .50 --+
历史 X,M -----------+→ nested L5  (X*,M5)→extrema→ same fθ → ŶL5 -- .25 --+→ weighted sum
                    +→ nested L20 (X*,M20)→extrema→same fθ → ŶL20-- .25 --+
                       M5≤M, M20≤M; each removes round(.15×observed rows)
```

- 只有一个冻结 checkpoint 和同一个 `self.model`；入口先 `load_model`：`run.py:1238–1262`，恢复 model state 与 memory/repre/num_dic：`exp_pretrain1.py:1084–1105`。runner 复制 source official args 并设置 `is_training=0`：`scripts/run_stage14_uncertainty_cache.py:460–519`。
- mask 在现有已观测行上额外删除，逐窗口精确移除 `np.rint(.15×原观测行数)`，L5/L20只改变删除形态，不增加未来信息：`exp_pretrain1.py:6840–6914`。
- “L5/L20”是 causal block length，不是序列长度5/20，也不是两套可学习 encoder。原始历史长度 L 保持不变。每路都重新算极值和填补，必须从输入端分叉，不应在已经编码的 query 上加 mask。
- 三个输出定权融合在 `exp_pretrain1.py:7072–7076`；已冻结 `.50/.25/.25` 于 `scripts/run_stage14_uncertainty_cache.py:192–207` 与对应配置。
- 当前通用评价函数为了单尺度对照还执行 L1，见 `exp_pretrain1.py:7030–7053`；L1不在 multiscale sum 中。图中可以不画 L1，但不可把融合成本虚报为实际审计函数仅跑三路。
- evaluation `.eval()+change_mode(0)+no_grad`，`exp_pretrain1.py:6940–6942,7010`；不训练、不更新 memory、不增参数。成本来自重复 forward/预处理，而不是新层。
- Traffic CRN v1 包装每路相同 seed/batch RNG，确保随机 topM 原型抽样一致，并另做 original-only reference pass：`exp_pretrain1.py:6982–7008`，`utils/paired_eval_rng.py:8–21`。这是公平评价 RNG 控制，不是 CNM 学习模块。

### Weather 历史版必须独立标注

Stage15 Weather 的旧 CNM-SE 是“原始+L5”两路，`.5/.5`；`configs/stage15_cnm_se/s4m_weather_cnm_se_seed2023.yaml:24–32`、`exp_pretrain1.py:6661–6723`。旧 `_stage14_nested_observation_mask` 是按起点概率做 max-pool，再乘原 mask（`:6399–6451`），不是新版精确等计数删除。不可把 Weather 旧 test 结果贴在新版三路等计数图上；图注应写“历史两路 CNM-SE”或分开一小框。

## 4. Active / disabled 矩阵

此矩阵为正式 PG95 配置+当前 opt-in parser 的实际执行路径，未显式出现的 later flags 以 `store_true` 默认 false 核验；不是把所有历史尝试拼成同一个方法。

| 部件 | S4M baseline | PG95 train | PG95 inference | PG95+CNM inference |
|---|---|---|---|---|
| decay imputation、Encoder_u、prototype retrieval/fusion、S4/head | 开 | 开 | 开 | 同checkpoint每路复用 |
| content residual adapter `U+A U` | 关 | 开 | 保留 | 每路保留 |
| paired mask intervention + auxiliary shared query branch | 关 | 开 | 不执行 | 不执行 |
| .05 cosine invariance loss | 关 | 开 | 不执行 | 不执行 |
| .95 absolute teacher write gate | 关 | 开（初始化最低候选回退另记） | 不写库 | 不写库 |
| content-mask factorization、mask reconstruction/decorrelation | 关 | 关 | 关 | 关 |
| variance/covariance 有效损失 | 关 | 权重0 | 不执行 | 不执行 |
| FUSR/utility read gate、CFR | 关 | 关 | 关 | 关 |
| MRRA、RADR、MCPR、BF-CMVD、DMRB、PBA-CMVD、HFD-PBA、TTRR、ARD | 关 | 关 | 关 | 关 |
| CNM input-view wrapper / fixed output fusion | 关 | 关 | 关 | 开，无新增参数 |
| min-distance bugfix | 原始路径关 | 独立开关，不属于PG95 | 继承checkpoint训练配置 | 同路复用；最新修复S4M可开 |

opt-in 定义证据：`run.py:240–291,319–328`；普通 path 分支：`S4M.py:411–414`、`Bank.py:624`。旧 resolved config 并非完整 parser dump，因此不要伪称旧文件显式列出所有 later-disabled flags。

## 5. 两处 bug / 易误画点

1. **已实现且默认可关：min distance 归一化。** 原始错误把已经标准化的 max_idx 放入 min_idx 的分子。correct path 为 `(min_idx−mean(min_idx))/std(min_idx)`；两处 `S4M.py:265–276,321–332` 均已受 `fix_min_distance_normalization` 控制，构造缺省 false（`:99–102`），CLI store_true（`run.py:164–165`）。独立代码正确性修复，不能画成 PG95 新创新模块，也不能把历史 checkpoint 当作该修复已开。
2. **未修：新增 memory index。** `Memory.py:140–150` 中 append(qi) 后仍以 `memory[num_bank−1]` 计算新增 centroid（`:143`），指向旧最后一簇而非刚追加的 q。图里可以画原 memory bank，不应声称“已修复整个 bank”或把修复收益归 PG95。
3. Encoder 卷积输出 `[B·n,H,Tc,1]` 在 `Bank.py:100` 直接 reshape 而非标准 transpose/permute。图宜写“Conv→reshape/reorder→attention”，不要画成标准维度置换，亦不要在制图时顺手改代码。
4. PG95 不等于完整 MIPP factorization，不等于 FUSR read gate，不等于 distillation；只标真正启用项。
5. paired train view 与 CNM nested inference view不同：前者从完整历史重新mask，非原mask嵌套；后者只从已观测部分额外删除。两种分叉应分成训练图/推理wrapper。
6. Mask-aware S4 只在首层；两套 s4_1/s4_2 实例中 forward 只用第一套；不能画每层都有mask stream或双向双分支。
7. CNM不添 encoder、router、可学融合权重或新增参数；增加的是 forward 次数。bank retrieval会随机topM抽样，受CRN评价控制；它不是 mask-only完全确定性前向。

## 6. 给绘图的最短落地建议

主图：只画 `input→decay→query encoder→retrieval→[R,U,Z]融合+U→mask-aware首层S4→其余S4→head`。PG95 retained adapter贴在query输出处；训练-only paired branch与teacher写门用虚线。单独一个推理wrapper小图画三路复用 `fθ` 和固定权重融合，不展开三遍encoder。图注单列 Weather旧两路、warmup最低候选回退、bugfix独立于PG95；负结果/公平性审计结论从专门结果审计引用，本文不推断性能。
