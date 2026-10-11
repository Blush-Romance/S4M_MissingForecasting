# 组会材料独立内容审计

审计日期：2026-10-10。范围仅为本次组会交付材料；不修改模型/训练配置，不启动训练、GPU或前向，不读取逐样本 validation/test NPZ 缓存。本文由独立审计代理形成。

## 当前审计状态

独立内容审计通过。已审阅30页同源口语稿/生成脚本及PDF提取文本，并独立目视关键页面；首轮发现的三项问题均已由主代理修订、读回确认。审计通过限于材料内容与报告质量，不构成新的实验有效性或投稿保证。

## 1. 架构三图核对

已核对 `_working/results_audit.md`、`_working/results_audit.json`、`_working/architecture_audit.md`、`figures/architecture_manifest.json` 与 `_working/draw_architecture.py`。未直接读取测试缓存，也未执行绘图脚本。

- 原始主干：历史输入→衰减填补 Z→滑窗 query encoder（Conv/reshape/attention/S4/时间池化）→原型检索→`Linear([R,U,Z])+U`→首层 mask-aware S4→H→D→其余 S4→LayerNorm/head→取尾 P。不存在新增 horizon decoder、每层 mask 门或第二套预测骨干。
- PG95：查询残差 adapter 在推理保留；有效训练损失为 `Lforecast + .05 Linv`；`.95` 为训练 teacher 非重叠块候选稳定性写门，不是 query token 筛选或推理检索读门；teacher EMA 为 `.99/.01`；初始化低候选回退已注明；factorization 关闭且其他 MIPP 有效损失权重为0。
- CNM：从输入端分原路/L5/L20三路；同一冻结 checkpoint/原型库；每路重算历史极值与填补；`.50/.25/.25` 固定输出融合；新增参数0而非新增 encoder/router；L1 不参与融合。
- 旧 Weather 两路 `.5/.5` 的概率块掩码 test 结果已与新版等计数三路图明确分隔；不能当新版三路三seed test。
- 跨图文字和 manifest 中核心事实一致，未发现主干或模块接入位置错误。

另独立打开三张 `architecture_*_a4_preview.png` 检查，框中文字与箭头均清楚，无溢框、遮挡或数据流分叉误读；`.05/.95`、factorization关闭与旧Weather限定均可读。此项只验证已生成图片，不重新生成图片或运行模型。

首轮图2副标题“推理只保留 query adapter 与学习后的原型库”存在概括歧义，曾建议明确 teacher 旁路仍计算。最终版已在图内直接改成“推理保留 adapter 与原型库；不生成训练 pair，不更新 bank；teacher 旁路仍计算”，且04页caption同样明确。独立打开最终04页PDF预览确认该限定、公式、写门与关闭factorization均清楚；歧义已解决。

## 2. 待正文/PDF复核的高风险事项

- 22.43 已完成且仅 CPU 汇总核验已有 validation 误差和，不是新训练/新前向/新 test；22.44 尚未实施。
- Traffic 配对四格需保留 PG95 1/3 双胜与均值改善率为负；CNM在两个底座均3/3双胜，不证明PG95独立有效。
- 距离公式修复仅2/3双胜、2028双指标退化；收益是 bugfix，不计创新，不与CNM增益相加归因；代表索引错误未修。
- Traffic修复后CNM逐seed改善率平均为 MSE `0.9579816080%`、MAE `0.6834905678%`；均值指标间降幅 `0.9586853774%`、`0.6838729482%`，二者不能混用。
- n=3 是联合 seed 描述性稳定性，不等于显著性；开发 validation 多次使用，不等于锁定独立 test。
- ETTh1完整组合失败与Electricity验证门失败必须保留；Electricity不获得test授权，不能跨split比较论文test值。
- CNM三次前向不等于同推理预算/免费/轻量；Weather早期PG95实际5→10epoch、11430→22860优化步，不等于实际计算等长；独立端到端时延/路线峰值缺项仍为NA。
- 标称 r=.06 不是实际6%缺失；原论文差异只作背景，不冒充完全复现或模块创新增益。
- 不应声称两个独立创新成立、基线全部实现风险已排除或SCI二区/三区投稿证据已达成。

## 3. 正文/PDF最终审计

已读取交付根目录 `build_report.py`、`组会口语稿.md`、`_working/pdf_extracted_text.txt`、`_working/qa_report.json`；文稿实际为15页正文+15页附录共30页。图3/4/5的PDF嵌入为矢量（raster image count均为0）。主要表格从results_audit.json生成，历史压缩表与history_audit.md对照无实质冲突，工件数从artifact_register.json核对为302条（262 SUCCESS、39失败、1历史RUNNING）。

独立打开重点PDF预览第4/6/7/8/10/14/24/25页，数字、英文、负号和表格可读，无明显文字遮挡或越界；并查看全部30页提取文本/生成源内容。不能将上述抽查说成30页逐页目视全检。

### 3.1 科学内容核验：通过

- 04页caption明确“现源码仍计算教师旁路，但不写库”，已消除图2副标题的推理概括歧义；.05有效损失、.95 teacher候选写门、推理adapter保留与factorization关闭均准确。
- 06/09页明确区分Weather旧两路三seed test与新版三路两seed validation；07页最新Traffic主结果仅validation，22.43是旧误差汇总，不被写成新测试。
- 08页保留PG95严格四格仅1/3双胜及均值退化，明确不能成立稳定第二创新；10页保留ETTh1/Electricity完整组合失败；Electricity不访问本轮test。
- 12页披露1→3次前向、Weather约3.05倍串行批延迟，最新Traffic缺独立端到端时延/路线显存；附录D披露早期实际训练5→10epoch、11430→22860step与参数增量。未宣称免费、同推理预算或轻量化。
- 附录C单列距离纠错仅2/3双胜与seed2028退化，不与CNM增益相加；索引错误未修、22.44未启动均明确。
- 14/15页发表判断符合证据：当前不够完整投稿证据，不保证SCI分区，不把两模块数量当发表硬条件；建议仅作为计划，不假报新实验。
- n=3为联合初始化/缺失布局的描述性方向一致，不被写成统计显著；旧test/多次开发validation无法冒充新独立确认。

### 3.2 数值复算：通过

06页六行逐seed改善率平均/单seed组合改善独立复算如下；均与表中两位小数舍入一致。

| 比较 | MSE改善% | MAE改善% | 显示舍入 |
| --- | --- | --- | --- |
| Weather旧两路test | 2.2197245945 | 0.9778559091 | +2.22 / +0.98 |
| Weather新三路validation | 3.2003917793 | 1.6072425315 | +3.20 / +1.61 |
| Traffic修复基线同checkpoint CNM | 0.9579816080 | 0.6834905678 | +0.96 / +0.68 |
| Traffic历史PG95同checkpoint CNM test | 0.8591890723 | 0.7771609691 | +0.86 / +0.78 |
| ETTh1完整组合test | -0.2934647120 | -0.1879952167 | -0.29 / -0.19 |
| Electricity完整组合validation | 约0.634936 | 约-0.199360 | +0.63 / -0.20 |

07页逐seedCNM改善按原始MSE/MAE复算与JSON完全一致（差0），基线/新方法均值与样本SD及三seed方向均准确。08页PG95负增量与胜seed数，09页两代Weather数字，10页负结果、附录C纠错数字均与审计来源一致。没有把均值指标间降幅偷换成逐seed改善率均值。

### 3.3 已反馈且已修正的事项

1. **附录I总体缺失率表达不准确。** 原句“本地固定资产实测约26%”位于四数据集总括下，可能错误涵盖全部数据；应改为Weather约24.7%、ETTh1约23.1%、Electricity/Traffic约26%，名义r=.06仍须保留。
2. **附录H上标负号缺字。** 更换WQY正文后，普通Latin/数字已可搜索，required_text_checks均pass，但提取文本仍在“5.87×10⁻⁴ / 10⁻⁴”处出现2个NUL，对应U+207B上标负号。应改为ASCII `5.87e-4`、`1e-4`（或确认完整字体支持的等价排版），并补NUL检查，不能仅依赖目前十个required_text字段。
3. **06页百分率单位建议。** 表中以百分数显示，定义应显式写`(参照误差−候选误差)/参照误差 × 100%`，避免仅写纯比值。

首轮PDF文字缺字已被主代理发现并更换字体；审计以重建后的文本/图片为准，首轮不作为最终交付。

最终读回复核：06页已加入`×100%`；24页文本已改为`5.87e-4`与`1e-4`；25页已分别写Weather约24.7%、ETTh1约23.1%、Electricity/Traffic约26%，并额外明确论文背景“变化率”正=更差，与正文改善率相反。24/25页最终预览均已独立打开，指数负号、实际缺失率和变化率定义清楚可读。全PDF提取文本不再有NUL（QA `null_glyph_count=0`），13项required_text全部通过，`out_of_page_text=[]`，三图矢量嵌入保持0 raster images；总页数仍30。最后一次重绘/重建后04页subtitle已直接消除teacher旁路歧义，主结果数字未改变。没有剩余必须修订的内容问题。
