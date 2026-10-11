#!/usr/bin/env python3
"""Read persisted reports only; create the meeting PDF, speaking notes and tables.

No model imports, inference, CUDA, cache NPZ access or research config writes.
PDF-only dependencies are installed in the local output folder, not the env.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT / "_deps"))
import fitz
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle

DATA = json.loads((OUT / "_working/results_audit.json").read_text())
T = DATA["tables"]
HISTORY = json.loads((OUT / "_working/history_audit.json").read_text())
REGISTER = json.loads((OUT / "_working/artifact_register.json").read_text())
FIGURES = OUT / "figures"
PAPER_URL = "https://proceedings.iclr.cc/paper_files/paper/2025/file/7b2f0758334389b8ad0665a9bd165463-Paper-Conference.pdf"
GITHUB_URL = "https://github.com/WINTERWEEL/S4M"
FONT_FILE = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
# DroidSansFallback has CJK but lacks Latin/digits in ReportLab (unlike mpl's
# fallback list). WQY embeds both alphabets, so tables cannot silently lose data.
pdfmetrics.registerFont(TTFont("CN", FONT_FILE, subfontIndex=0))
W, HT = landscape(A4)
MARGIN = 38
CW = W - 2 * MARGIN
NAVY = colors.HexColor("#193A52")
BLUE = colors.HexColor("#315D79")
GRAY = colors.HexColor("#5D6974")
BG = colors.HexColor("#EEF3F6")
ORANGE = colors.HexColor("#BA6A27")
STYLES = {
    "body": ParagraphStyle("body", fontName="CN", fontSize=12.4, leading=19.0, wordWrap="CJK", textColor=NAVY),
    "speech": ParagraphStyle("speech", fontName="CN", fontSize=13, leading=20.2, wordWrap="CJK", textColor=NAVY),
    "note": ParagraphStyle("note", fontName="CN", fontSize=10, leading=14.4, wordWrap="CJK", textColor=GRAY),
    "cell": ParagraphStyle("cell", fontName="CN", fontSize=11, leading=15.2, wordWrap="CJK", textColor=NAVY),
    "smallcell": ParagraphStyle("smallcell", fontName="CN", fontSize=9.8, leading=13.8, wordWrap="CJK", textColor=NAVY),
    "headcell": ParagraphStyle("headcell", fontName="CN", fontSize=10.5, leading=14.4, wordWrap="CJK", textColor=colors.white),
}
PAGES = []


def f(v, n=6):
    return "NA" if v is None else f"{v:.{n}f}"


def pct(v, n=2):
    return f"{v:+.{n}f}%"


def pair(r):
    return f(r["mse"]) + " / " + f(r["mae"])


def gain(a, b):
    return 100 * (a - b) / a


def item(text, style="body"):
    return ("p", text, style)


def table(headers, rows, weights=None, small=False):
    return ("table", headers, rows, weights, small)


def page(title, items, speech=None, source=None, appendix=False):
    if speech:
        items += [("sub", "口语稿｜可以直接讲"), item(speech, "speech")]
    if source:
        items += [item("数字／代码来源：" + source, "note")]
    PAGES.append({"title": title, "items": items, "appendix": appendix})


def diagram_page(title, stem, caption, speech, source):
    PAGES.append({"title": title, "diagram": stem, "caption": caption, "speech": speech, "source": source, "appendix": False})


def create_content():
    page("01｜缺失历史条件下的时序预测：当前进展", [
        item("S4M_MissingForecasting 组会汇报 · 2026年10月10日", "note"),
        item("一句话结论：我们找到了可重复的小幅推理收益，但还没有形成跨数据集稳定、可投稿的完整新方法。"),
        table(["已经有的证据", "还没有的证据", "本次汇报边界"], [[
            "CNM-SE在部分协议方向一致；固定资产、配对控制和负结果可追溯。",
            "PG95稳定第二贡献；统一版本跨数据集成功；足够强的公平对比与成本证据。",
            "研究做到22.43；22.44未执行。本次只整理材料，没有新实验。"
        ]], [1, 1, 1]),
        item("使用方式：正文01–15按顺序讲，约12–15分钟；附录用于追问和核对。全部302条运行工件另附CSV，不能当成302次独立正式精度实验。", "note"),
    ], "今天我先汇报结论，再解释模型具体改在哪里。现在最值得保留的是CNM-SE这条推理思路；PG95还没有稳定成立。我们不会把修复代码错误当成创新，也不会只展示成功数据集。最后我会说明，这些证据离一篇完整论文还缺什么。")

    page("02｜任务与版本：预测未来，不是只补历史", [
        item("任务：输入带缺失的多变量历史序列及观测掩码，输出未来多个时间点的各变量预测。补历史只是前处理；主评价是未来预测误差。"),
        item("当前Traffic正式配置示例：历史192步、预测96步、862个变量；batch=16。输入[16,192,862]与同形掩码，预测输出[16,96,862]。", "note"),
        table(["名称", "究竟是什么", "在汇报中的身份"], [
            ["S4M原始行为版", "官方代码主干＋必要复现工程修复，保留原距离公式。", "本地baseline（基线），不是论文数字本身。"],
            ["PG95", "查询残差适配器＋配对不变损失＋0.95教师写入门。", "已经尝试的训练侧候选；不是修复版。"],
            ["CNM-SE", "从历史输入构造嵌套缺失视图，共享检查点，固定融合预测。", "推理包装；不增加可学习参数。"],
            ["距离修复版S4M", "只修最小距离的归一化分子，独立开关。", "基线纠错，不计作创新；最新主比较对象。"],
        ], [1.1, 2.6, 2.0]),
    ], "这个项目不是把插补误差做低就结束了，最终还是要预测未来。这里要把三个概念分开：baseline是S4M，PG95是加了训练机制的版本，修复版是纠正基线错误。它们不是同一个东西。CNM又是在输入和输出外面包了一层多视图推理流程。", "_working/architecture_audit.md；正式resolved_config与reports。")

    diagram_page("03｜实际主干：模块究竟加在哪里", "architecture_01_backbone",
        "依据原论文Figure 2（PDF第4页）与实际代码重新绘制。B=批量，L=历史长度，C=变量数，H=查询维度，D=预测主干维度，P=预测长度。蓝色是共享主干，橙色是PG95增量，虚线是训练分支。",
        "顺着图看，历史先填补，再编码成查询，去原型库取参考。检索结果、查询和填补历史一起融合，再进入S4预测主干。PG95加在查询表示和训练写库的位置，没有另造预测解码器。掩码直接进入预测主干的首层，后面各层不再单独接掩码。",
        "baselines/S4M/s4m/model/{S4M.py,Bank.py,MIPP.py}；原论文Figure 2。")

    diagram_page("04｜PG95：学习稳定表示，而不是输出门控", "architecture_02_pg95",
        "PG95中文：仅配对一致性与0.95稳定写入门的紧凑候选。有效损失=观测未来目标上的MSE＋0.05×配对余弦不变损失。内容／掩码因子分解关闭。推理保留适配器；现源码仍计算教师旁路，但不写库。",
        "训练时，把同一段完整历史重新做一份缺失干预，让两份查询尽量一致。另一条教师分支用0.95相似度筛写库候选。注意，这不是预测输出的开关，也不是检索读取门。推理不再造训练配对、不加辅助损失，但学到的适配器和原型库仍在。",
        "model/MIPP.py；model/Bank.py；experiments/exp_pretrain1.py。")

    diagram_page("05｜CNM-SE：在输入端分叉，在输出端融合", "architecture_03_cnm",
        "CNM-SE中文：因果嵌套缺失自集成。新版为原始、L5、L20三路，权重0.50／0.25／0.25。L5/L20是缺失块长，不是输入长度；两路移除相同数量的已观测历史行。旧Weather测试用两路0.5／0.5，不能套新版图。",
        "这里不是三个模型，而是同一个模型看三份历史。额外视图只能删掉已有观测，不能补进未来或不可见真值。每一路都重新填补和预测，然后按固定权重相加。它不增加参数，但要做三次前向，所以精度收益必须和计算代价一起讨论。",
        "exp_pretrain1.py:6840–7076；run_stage14_uncertainty_cache.py；CRN仅为公共随机数控制。")

    page("06｜四个数据集：哪里有效，哪里没有效果", [
        table(["数据／协议", "比较对象", "平均改善 MSE / MAE", "一致性与结论"], [
            ["Weather：旧L5，test", "同PG95检查点，加两路CNM", "+2.22% / +0.98%", "3/3双指标改善；历史版证据。"],
            ["Weather：新版，validation", "同PG95检查点，加三路CNM", "+3.20% / +1.61%", "2/2双指标改善；不是新版test。"],
            ["Traffic：新版，validation", "距离修复S4M，加三路CNM", "+0.96% / +0.68%", "3/3双指标改善；当前最干净主比较。"],
            ["Traffic：历史协议，test", "同PG95检查点，加三路CNM", "+0.86% / +0.78%", "3/3增量双胜；不能替代新协议test。"],
            ["ETTh1：test，2023", "S4M → PG95＋三路CNM", "−0.29% / −0.19%", "完整组合失败；视图预测几乎不变。"],
            ["Electricity：val，2023", "S4M → PG95＋三路CNM", "+0.63% / −0.20%", "MAE退化；未访问本轮test。"],
        ], [1.3, 1.6, 1.2, 2.0]),
        item("改善率=(参照误差−候选误差)/参照误差×100%；正数为更好。跨seed汇总采用逐seed改善率平均；不同数据／版本／split（划分）不横向排名，不合算总平均。", "note"),
    ], "先看全景，Weather和Traffic确实有信号，但不能说四个数据集都成功。ETTh1基本没有视图响应，完整方法还比基线差；Electricity只有MSE略好，MAE变差。也就是说，当前故事是有条件的有效性，还不是通用解决缺失预测问题。", "stage16、18、20、21、22.22、22.43持久化结果。")

    rows = []
    for r in T["traffic22_43_cnm_per_seed_both_formulas"]:
        x = r["fixed"]
        rows.append([r["seed"], pair(x["baseline"]), pair(x["cnm"]), pct(x["relative_gain_percent"]["mse"]), pct(x["relative_gain_percent"]["mae"])])
    s = T["traffic22_43_cnm_summary_both_formulas"]["fixed"]
    means = [[label, f(x["mse"]["mean"]) + " ± " + f(x["mse"]["sample_std"]), f(x["mae"]["mean"]) + " ± " + f(x["mae"]["sample_std"])] for label, x in [("修复S4M", s["baseline"]), ("同检查点＋CNM", s["cnm"])]]
    page("07｜最新主结果：修复基线上的CNM增量", [
        item("Traffic · 固定validation · 同检查点 · 三路权重已冻结 · seed 2027/2028/2029"),
        table(["seed", "修复S4M：MSE / MAE", "＋CNM：MSE / MAE", "MSE改善", "MAE改善"], rows, [0.65, 1.85, 1.85, 0.9, 0.9]),
        table(["三seed汇总", "MSE均值 ± 样本标准差", "MAE均值 ± 样本标准差"], means, [1.2, 1.8, 1.8]),
        item("逐seed改善率平均：MSE +0.957982%，MAE +0.683491%。3/3是方向一致，不等于统计显著。22.43只复算既有验证误差缓存，未新增训练／前向／独立测试。", "note"),
    ], "这是我现在最有把握汇报的一组结果。同一个修复后的S4M检查点，只改变有没有CNM，三个seed都是双指标更好。平均大概是MSE降0.96%、MAE降0.68%。这个提升真实可核对，但幅度不大，而且是已有验证结果的重新归因，不是一次新的独立测试。", "reports/stage22_43_cnm_audit.json；final_audit.md。")

    fs = T["traffic22_37_factorial_summary"]
    labels = {"A_s4m": "A：S4M", "B_pg95": "B：PG95", "C_s4m_cnm": "C：S4M＋CNM", "D_pg95_cnm": "D：PG95＋CNM"}
    cellrows = [[labels[k], f(v["mse"]["mean"]), f(v["mae"]["mean"])] for k,v in fs["cells"].items()]
    gs = fs["gains"]
    grows = [[lab, pct(gs[k]["mean"][0]), pct(gs[k]["mean"][1]), f'{gs[k]["dual_wins"]}/3'] for lab,k in [
        ["PG95单独增量 A→B", "pg95_gain_without_cnm_percent"],
        ["CNM在S4M上 A→C", "cnm_gain_on_s4m_percent"],
        ["PG95在CNM下 C→D", "pg95_gain_with_cnm_percent"],
    ]]
    page("08｜为什么不能把PG95当成稳定第二贡献", [
        item("Traffic · 22.37严格配对三seed validation · 以下是原距离公式版，不是最新修复版。", "note"),
        table(["组合", "MSE均值", "MAE均值"], cellrows, [2,1,1]),
        table(["独立增量", "平均MSE改善", "平均MAE改善", "双指标胜seed"], grows, [2,1,1,1]),
    ], "我们特地做了四个格子的控制。结果表明，CNM不依赖PG95也能有收益；反过来，加了CNM以后，PG95仍然只在一个seed上双指标更好，平均还退化。这不是说PG95完全没用，而是现在的证据不足以把它列为稳定第二创新，更不能把组合的全部收益都归给它。", "reports/stage22_37_control.json。")

    wr = [[r["seed"], f(r["mse_on"]), f(r["mse_cnm_se"]), pct(r["mse_gain_pct"]), f(r["mae_on"]), f(r["mae_cnm_se"]), pct(r["mae_gain_pct"])] for r in T["weather16_l5_test"]]
    wn = []
    for r in T["weather18_multiscale_validation"]:
        a,b=r["routes"]["on"],r["routes"]["multiscale"]
        wn.append([r["seed"],pair(a),pair(b),pct(gain(a["mse"],b["mse"]))+" / "+pct(gain(a["mae"],b["mae"]))])
    page("09｜Weather：旧版测试与新版验证分开看", [
        item("历史两路CNM（原路＋L5，0.5／0.5）· test（测试集）· 同PG95检查点", "note"),
        table(["seed", "原MSE", "CNM MSE", "改善", "原MAE", "CNM MAE", "改善"], wr, [0.55,1,1,0.8,1,1,0.8], small=True),
        item("新版严格等新增缺失数三路CNM · validation（验证集）· 同PG95检查点", "note"),
        table(["seed", "原 MSE / MAE", "三路 MSE / MAE", "MSE / MAE改善"], wn, [0.55,1.9,1.9,1.5]),
    ], "Weather是最早看到效果的数据。旧两路版三个seed都改善，但它不能证明新版三路在测试集也一样有效。后来我们严格控制新增缺失数量，在验证集上看到了三路的进一步收益。汇报里要分开说，不能把两份结果拼成一个更强、但实际没做过的结论。", "stage16_cnm_se_three_seed_analysis.json；stage18_equal_rate_multiscale_analysis.json。")

    erows=[]
    for dataset,key,scope in [("ETTh1", "etth1_stage20", "test"),("Electricity","electricity_stage21","validation")]:
        x=T[key]
        for lab,r in [("S4M",x["baseline"]),("PG95",x["pg95"]),("PG95＋三路CNM",x["routes"]["multiscale"])]:
            erows.append([dataset+" / "+scope,lab,f(r["mse"]),f(r["mae"])])
    page("10｜必须保留的负结果：视图有时不产生互补", [
        table(["数据／划分；seed=2023", "方法", "MSE", "MAE"], erows,[1.6,1.7,1,1]),
        item("ETTh1与Electricity的原路／L5／L20预测误差差异接近浮点量级，CNM没有实质增量。Electricity验证门未过，未访问本轮测试集。", "note"),
    ], "这两组负结果很重要。它们说明，多造几份缺失视图不一定有互补；如果模型输出几乎没变，集成自然也帮不上忙。这里我们不能继续硬说方法通用。可能涉及填补、历史直达路径或检索分支响应不足，但这些只是待隔离的解释，还不是已经证明的根因。", "stage20_etth1_pg95_full_method.json；stage21_electricity_validation_gate.json。")

    page("11｜实验过程只讲四个转折", [
        table(["转折", "我们做了什么", "保留下来的认识"], [
            ["1：先建立基线", "跑通Weather/ETTh1，固定缺失资产和训练链路。", "本地可比较；不等于已完整复现论文数字。"],
            ["2：筛训练侧机制", "配对表示、原型写入／读取规则、残差控制。", "很多代理目标改善，却未形成稳定预测提升。"],
            ["3：保留推理信号", "CNM由两路发展为严格等计数三路。", "Weather/Traffic有互补；另外两个数据失败。"],
            ["4：回收归因证据", "控制随机数与公共初始化，分离bugfix和创新。", "CNM有小幅独立增量，PG95不稳定；基线仍有风险。"],
        ], [0.8,1.8,2]),
        item("失败路线不是当前模型的组成部分；历史机制与数字在附录，302条全量工件另附索引。", "note"),
    ], "过程我不按几十个步骤展开。核心只有四件事：建立基线，试训练机制，发现多视图推理有效，再用更严的控制重新核对。现在回头看，真正有价值的不只是一次好结果，而是知道哪些想法没成立、哪些提升来自纠错、哪些才是方法本身的增量。")

    latency=T["weather18_serial_latency_mean_ms"]
    page("12｜代价：不增加参数，不等于免费", [
        table(["Weather同GPU串行测量", "模型调用数", "平均批延迟", "相对原路"], [
            ["原始路", "1", f(latency["original"],2)+" ms", "1.00×"],
            ["严格L5两路", "2", f(latency["ensemble_l5"],2)+" ms", f(latency["ensemble_l5"]/latency["original"],2)+"×"],
            ["三路多尺度", "3", f(latency["multiscale"],2)+" ms", f(latency["multiscale"]/latency["original"],2)+"×"],
        ], [2,0.9,1.2,1]),
        table(["当前Traffic主比较", "已知", "仍需补齐"], [[
            "修复S4M → 同检查点＋CNM", "总参数28,371,290；CNM新增0。单路→三路。", "新版隔离端到端延迟、路线独立显存、等计算预算强基线。"
        ]],[1.5,2,2]),
        item("通用审计程序另跑L1与original-only参考；三次是部署融合所需路径，不等于审计程序总共只有三次调用。旧GPU时序不能替代最新隔离测试。", "note"),
    ], "CNM的优点是不用再训练一个模型，也没有新增参数。但不能把这说成轻量化，因为它需要重复预测。Weather实测大概三倍批延迟，Traffic最新版本还没有完整的隔离端到端计时。小幅提升换这份代价是否划算，是论文里必须正面回答的问题。", "stage18_equal_rate_multiscale_analysis.json；stage22_43_cnm_audit.json。")

    page("13｜当前问题：分清已证实与未证实", [
        table(["问题", "证据", "目前能下的结论"], [
            ["训练侧第二贡献不稳", "PG95配对三seed仅1/3双胜；多个替代方向失败。", "不能把已失败模块叠加成两个成功创新。"],
            ["泛化与视图响应不够", "ETTh1/Electricity三路输出接近相同。", "存在适用边界；具体因果来源待隔离。"],
            ["基线仍有实现风险", "距离公式已修；新增原型代表索引错误尚未修。", "不能声称所有上游bug已排除。"],
            ["选择与评价协议不同", "训练未来观测元素MSE；外部全目标MSE/MAE。", "需披露选模目标差异；不能事后改指标。"],
            ["确认数据不够新", "Traffic验证集多次用于开发；旧test已查看。", "换mask不等于换时间区间；需冻结新确认边界。"],
        ],[1.2,2,2]),
    ], "我把问题分成证据确定的和只是怀疑的。确定的是第二贡献不稳、两个外部数据没有效果、基线还有一个索引风险。怀疑的是历史直达路径可能压住检索，但当前冻结路径诊断不能证明普遍因果。我们接下来应先排掉已确认的实现风险，再决定结构创新，而不是先堆模块。", "architecture_audit.md；stage22_37、38、39、41、42、43。")

    page("14｜能达到发论文的水平吗？目前还不够", [
        table(["审稿时会问什么", "当前证据状态", "缺口"], [
            ["是否有清晰、非平凡的方法贡献？", "有条件的CNM信号；它仍是输入扰动＋固定集成。", "现有记录不构成完整相关工作／新颖性审查。"],
            ["是否有两个独立有效创新？", "没有。PG95不稳，纠错不计创新。", "两个模块不是发表硬条件；贡献质量更重要。"],
            ["是否统一方法、跨数据集可靠？", "旧Weather两路和新版三路不同；两个数据失败。", "锁定同一版本与适用假设，再确认外推。"],
            ["是否优于强基线且代价合理？", "当前主要与本地S4M比较；三路成本明显。", "补同协议强方法与精度—成本对照。"],
            ["是否有独立、稳定的确认？", "n=3方向性结果；seed同时变初始化与mask。", "未开发确认集、训练／mask波动分离、统计稳定性。"],
        ],[1.6,2,2]),
    ], "如果现在问能不能直接写一篇方法论文，我的判断是还不够，更不能保证二区。问题不只是提升小，而是最终方法、泛化、独立贡献和成本还没有闭环。它已经是一个可核验的研究原型，有继续投入的理由，但不是可以把所有模块包装起来就投稿。这里的判断是根据现有证据，不是期刊录用保证。")

    page("15｜下一步：先可信，再统一，最后扩展", [
        table(["优先级", "最短动作", "它要回答的问题"], [
            ["1", "独立、可关闭地处理原型代表索引错误；先回归测试，再单seed同预算验证。", "基线结果是否还受已确认实现错误影响？修复收益单列。"],
            ["2", "在可信基线上复核一个冻结的CNM版本。", "约1%的收益是否仍成立，且不是版本／随机轨迹混杂？"],
            ["3", "冻结方法与新确认边界，补少量强基线和可信外部协议。", "收益能否外推，计算代价是否合理？"],
            ["4", "只有正信号保留，才开发独立训练侧机制或选择单贡献定位。", "是否真需要第二创新，它能否单独有效、开关消融？"],
        ],[0.5,2.5,2]),
        item("以上是计划，不是已执行结果。本次材料制作未开始22.44，没有新增GPU任务。", "note"),
    ], "我的建议不是马上再接一个注意力模块。先把已经确定的实现问题独立处理，然后用同一个CNM定义做快速复核。如果收益还在，就锁定方法和新确认数据，补强基线与代价。如果还是无法形成稳定贡献，我们就调整研究假设，不把继续跑实验当作一定能发论文的承诺。")

    # Appendix: every primary result family and negative branch remains traceable.
    fr=[]
    for r in T["traffic22_37_factorial_per_seed"]:
        fr.append([r["seed"]]+[pair(r["cells"][k]) for k in labels])
    page("附录A｜Traffic配对四格：完整逐seed", [
        table(["seed", "A S4M", "B PG95", "C S4M＋CNM", "D PG95＋CNM"],fr,[0.5,1.3,1.3,1.3,1.3],small=True),
        item("表中每格为MSE / MAE。原距离公式版、validation、公共初始化配对、辅助随机流隔离、公共随机数CRN-v1。三seed2027/2028/2029同时影响训练初始化与缺失布局，不能解释为只改变训练seed。"),
        item("三个seed均实际训练4 epoch（训练轮），每个检查点CNM同原始路共享训练。PG95辅助机制计算仍增加训练成本；配对epoch相同不等于算力完全相同。", "note"),
    ],source="reports/stage22_37_control.json；源train/eval run在配套JSON。",appendix=True)

    hr=[]
    for r in T["traffic22_22_historical_per_seed"]:
        for split in ["validation","test"]:
            x=r[split]
            hr.append([r["seed"],split,pair(x["s4m_original"]),pair(x["pg95_original"]),pair(x["pg95_cnm_se_multiscale"])])
    page("附录B｜Traffic历史validation / test，不与新版合并", [
        table(["seed", "划分", "S4M MSE / MAE", "PG95 MSE / MAE", "PG95＋CNM MSE / MAE"],hr,[0.5,0.9,1.7,1.7,1.7],small=True),
        item("同PG95检查点的CNM历史test增量三seed均双胜，平均MSE +0.859189%、MAE +0.777161%。完整PG95＋CNM相对S4M的test MAE仅2/3胜，2028 MAE退化0.088038%。"),
        item("这些是旧训练来源／旧评估流程下的历史证据，不替代22.37配对validation或22.43修复后的主比较；旧test已经查看，不能再充当全新最终确认。", "note"),
    ],source="reports/stage22_22_traffic_multiseed_results.json。",appendix=True)

    br=[]; sr=[]
    for r in T["traffic22_41_bugfix_per_seed"]:
        for key,arr in [("primary_original_path",br),("secondary_inherited_CNM",sr)]:
            x=r["comparison"][key]
            arr.append([r["seed"],pair(x["baseline"]),pair(x["fixed"]),pct(x["relative_gain_percent"]["mse"])+" / "+pct(x["relative_gain_percent"]["mae"])])
    page("附录C｜纠错收益单列：不是CNM创新收益", [
        item("主对照：原距离公式S4M → 最小距离公式修复S4M；均为原始输入路。", "note"),
        table(["seed", "修复前 MSE / MAE", "修复后 MSE / MAE", "MSE / MAE改善"],br,[0.5,1.8,1.8,1.6],small=True),
        item("次要对照：继承同一个CNM策略，比较距离修复前后。", "note"),
        table(["seed", "原公式＋CNM", "修复＋CNM", "MSE / MAE改善"],sr,[0.5,1.8,1.8,1.6],small=True),
        item("主对照平均改善+1.827601% / +1.884691%，仅2/3双胜。2028双指标退化，与最新同检查点CNM的3/3双胜不是一个比较，不能相加归因。", "note"),
    ],source="reports/stage22_41_distance_fix.json。",appendix=True)

    early=[]
    for r in T["weather7_early_training"]:
        x=r["performance"];cost=r["efficiency_and_realized_training"]
        early.append([r["label"],pair(x),str(cost["actual_epochs_completed"])+" / "+str(cost["actual_optimizer_steps"]),str(cost["parameter_count"]),f(cost["peak_gpu_memory_mb"],1),f(cost["elapsed_seconds"],1)])
    page("附录D｜早期训练侧候选与真实训练代价", [
        table(["Weather test 2023", "MSE / MAE", "实际epoch / step", "参数量", "峰值MiB", "耗时s"],early,[1.1,1.65,1.3,1,0.8,0.85],small=True),
        item("最大15 epoch、早停耐心3相同，但实际早停不同：PG95比S4M多153,130参数，实际训练5→10 epoch、11,430→22,860步。运行GPU／公共负载不同，历史耗时不作隔离速度结论。"),
        item("Main MIPP=完整掩码干预原型净化；SG95=完整净化机制配0.95门；PG95=关闭内容／掩码因子分解的紧凑配对候选。PG95历史写入接受率99.9142%，不能仅凭0.95阈值宣称强净化。", "note"),
    ],source="reports/stage7_mipp_comparison.json。",appendix=True)

    a=T["weather16_block_ablation"]["means"]
    ab=[]
    for k,v in a.items():
        ab.append(["L"+k,pair(v["routes"]["ensemble"]),f(v["realized_additional_missing_rate"]*100,3)+"%",pct(v["ensemble_mse_gain_vs_on_pct"])+" / "+pct(v["ensemble_mae_gain_vs_on_pct"])])
    page("附录E｜Weather块长：旧消融与严格等计数分开", [
        item("旧版test三seed均值：每个块长均做原路＋该块长视图两路0.5／0.5。", "note"),
        table(["块长", "集成MSE / MAE均值", "实际额外删除率", "均值指标间MSE / MAE改善"],ab,[0.7,1.5,1.1,2],small=True),
        item("旧L1/L5/L20实际删除率并不相等，不能把L20的MSE更低全归因于块长；没有据旧test重新挑主模型。"),
        item("随后新版validation：同seed同窗口额外删除数完全相同。L1仅诊断；新版多尺度只融合原始、L5、L20。下页给全部路由结果。", "note"),
    ],source="stage16_cnm_se_ablation_analysis.json；stage18_equal_rate_multiscale_analysis.json。",appendix=True)

    wn=T["weather18_multiscale_validation"]
    routes=["on","l1","l5","l20","ensemble_l1","ensemble_l5","ensemble_l20","multiscale"]
    names=["原始单路","L1单路","L5单路","L20单路","原始＋L1等权","原始＋L5等权","原始＋L20等权","三路0.5／0.25／0.25"]
    rn=[[name]+[pair(r["routes"][key]) for r in wn] for key,name in zip(routes,names)]
    page("附录F｜Weather严格等计数：所有路由结果", [
        table(["validation路由", "seed2024：MSE / MAE", "seed2025：MSE / MAE"],rn,[1.6,1.8,1.8],small=True),
        item("额外删除率各块长一致：2024为15.0057425%，2025为15.0136680%；这是原观测行的比例，不是完整输入的总体缺失率。每seed共5168验证窗口。", "note"),
        item("三路相对严格L5两路的额外改善：2024 MSE +0.916908%／MAE +0.452648%；2025 +1.077797%／+0.595166%。收益正向，但增加一路调用。", "note"),
    ],source="reports/stage18_equal_rate_multiscale_analysis.json。",appendix=True)

    # Negative history kept in Chinese; values are audited persisted-report summaries.
    history_rows = [
        ["原型净化 MIPP/SG95/PG95", "Weather：PG95 +0.60%/+2.42%；Traffic严格配对 −0.72%/−0.46%。", "仅Weather早期单seed支持；不是稳定第二贡献。"],
        ["效用读取 FUSR／历史探针HCUP", "Weather v1 −0.0681%/+1.2080%；v2 −0.0150%/−0.0071%；28种短跨度探针均失败。", "静态／历史可观测特征未能稳定预测检索效用；停止。"],
        ["写入预算／覆盖 BMSA/SDRA/MRBA", "BMSA三seed −0.0784%/−0.2891%；SDRA 0/2双胜；MRBA 1/2。", "覆盖、多样性、稳定性代理不保证预测效用。"],
        ["反事实风险 CFR", "Weather2023 MSE/MAE 0.263401/0.298090；相对PG95 −0.5554%/−1.4330%。", "当前逐样本后悔损失未泛化；停止。"],
        ["响应增强 MRRA／风险残差 RADR", "Electricity：v1/v2 NaN；v3 −0.6460%/+0.1141%；RADR −2.2501%/+0.8857%。", "稳定不等于有效；RADR重载核验缺口，不能当协议全PASS。"],
        ["输出残差 MCPR／TTRR／ARD", "Electricity：MCPR −0.3780%/+0.2017%；PCD −4.9300%/+1.2655%；ARD相对PG95 −0.1876%/+0.0566%。", "幅度控制部分奏效，但双锚点门失败；停止当前路线。"],
        ["CNM-SE主线", "Weather/Traffic局部有效；ETTh1/Electricity视图近乎不变。", "保留有条件信号，不宣称通用跨数据集方法。"],
        ["执行优化：少视图／批合并／CUDA多流", "两路保留约15%–50%三路收益；批合并延迟降21.263%但逐点容差失败；多流降0.5165%不可靠。", "没有已证实的等价高效替代；执行优化不是第二精度创新。"],
        ["单视图蒸馏 BF-CMVD／DMRB／PBA／HFD", "Traffic2027 BF-v2 +0.5199%/+0.3587%；保留教师收益48.07%/44.16%，低于冻结50%门。后续版未过门。", "可部分摊销，尚不够；都是开发validation，不算多次独立确认。"],
        ["自适应视图选择 PSAVS", "Traffic audit raw −0.1436%/−0.1946%；校准后100%回退固定三路，增益0。", "有标签oracle空间，不等于可部署选择器或调用节省。"],
        ["控制与基线纠错", "公共随机数、公共初始化、路径诊断；距离修复仅2/3胜，索引问题未修。", "增强证据可信度；不计作模型创新。"],
    ]
    for ix, group in enumerate([history_rows[:6],history_rows[6:]], start=1):
        page(f"附录G{ix}｜完整研究分支压缩表（非当前模块清单）",[
            table(["问题／分支（中文定位）","关键结果：MSE / MAE改善率","结论"],group,[1.35,2.5,1.8],small=True),
            item("正数=相对各自锚点更好；不同数据／split／版本不横向排名。NaN版没有完整正式精度。标签oracle使用真值误差，只是上限分析。失败门未事后放宽。", "note"),
        ],source="_working/history_audit.md/json列出11类及全部对应报告。",appendix=True)

    distill=[
        ["BF-CMVD-v2","+0.519939% / +0.358726%","48.07% / 44.16%","部分正信号；未过50%收益保留门。"],
        ["DMRB","−0.199201% / −0.109647%（相对BF-v2）","29.75% / 30.71%","双残差基增加参数但退化。"],
        ["PBA","−0.149732% / +0.028595%（相对BF-v2）","未过60%门","末端主干摊销未有效。"],
        ["HFD-PBA","−0.254040% / −0.490051%（相对BF-v2）","未过冻结门","隐藏对齐已执行，输出目标没改善。"],
    ]
    page("附录H｜效率与蒸馏：有部分信号，但未形成贡献",[
        table(["Traffic2027开发validation","改善／退化","教师收益保留","判断"],distill,[1,2.2,1.2,1.6],small=True),
        item("效率实验：批拼接253.11→198.75 ms/batch，下降21.263%，但最大逐点误差5.87e-4超过预设1e-4等价门。CUDA多流逐位等价，但平均降0.5165%，区间跨0。"),
        item("这些结果不能用于宣称当前修复版轻量／加速。单路学生有小幅增益也不能冒充保留了全部三路收益；同开发缓存的多候选没有独立性。", "note"),
    ],source="stage22_23–26、28、30、32报告；具体锚点见history_audit.md。",appendix=True)

    paperrefs=[["Weather","test", "0.225 / 0.305",pair(T["weather7_early_training"][0]["performance"]),"MSE +17.12%；MAE −1.26%"],
               ["ETTh1","test","0.593 / 0.574",pair(T["etth1_stage20"]["baseline"]),"MSE +103.71%；MAE +46.15%"],
               ["Electricity","test参考", "0.274 / 0.367","项目本轮仅validation","NA：不跨split比较"],
               ["Traffic","test参考","0.687 / 0.385","见历史test；非同协议","NA：不宣称严格复现"]]
    page("附录I｜与原论文相比：参考差距，不是创新增益",[
        table(["数据","论文划分","论文MSE / MAE","本地seed2023／说明","本地相对论文变化"],paperrefs,[0.8,0.9,1.3,1.65,1.6],small=True),
        item("本表变化率=(本地−论文)/论文×100%，正数表示误差更大、负数更小；与前文“改善率”的符号相反。这里只量化背景差距，不作同协议胜负。", "note"),
        item("论文参考：ICLR 2025 S4M，Table 1的时间点缺失r=0.06、输入192／预测96。当前本地结果不能认定完全同协议：原mask seed及部分最终超参数不可精确还原。"),
        item("原生成器的名义r=0.06不是实际最终缺失率；本地早期资产Weather约24.7%、ETTh1约23.1%、Electricity与Traffic约26%。具体值随数据与seed记录。必须同时披露名义／实际比例，不能说只缺6%。", "note"),
        item("本地基线已具备可运行可比较链路，不等于论文结果完全复现。创新增益只从同协议的本地基线计算。", "note"),
    ],source="官方论文Table 1；stage7_mipp_comparison.json；stage20/21协议限制。",appendix=True)

    page("附录J｜评价与可复现边界",[
        table(["维度","本报告披露的条件"],[
            ["训练loss与外部metrics","loss（损失）为未来可观测目标的MSE；外部评估为冻结范围全目标MSE/MAE。两个指标均越小越好。"],
            ["Traffic最新资产","按时间70%／10%／20%划分，历史192、预测96、batch16；Adam学习率0.001、type1调度、最多15轮、耐心3，实际4轮。固定资产1648窗口，旧drop_last丢13尾窗；本次未改评价。"],
            ["CNM证据与限制","同检查点、同冻结原型库、固定权重、公共随机数；缓存保存每窗平方／绝对误差和，22.43独立重汇总而非从原预测重算。"],
            ["实现与状态证据","有代码写门和工件检查，但缺少运行时原型库前后hash；索引bug仍待隔离，不声称全部风险排除。"],
            ["完整工件口径",f'CSV共{REGISTER["counts"]["source_data_rows"]}条，262 SUCCESS、39失败状态、1条历史RUNNING。是持久化状态快照，未核查实时进程。'],
            ["失败与创新边界","smoke（极小链路测试）不算正式精度；资源重试不算独立实验；bug fix（错误修复）不算创新。"],
        ],[1.0,4.6],small=True),
        item("302条索引包括smoke、训练、评估、计时、CPU缓存分析和失败；SUCCESS不等于科学门通过。配套完整CSV保留全部原字段、原行序和缺失值，没有手填指标。", "note"),
    ],source="artifact_register.json；stage22_43_final_audit.md；resolved_config。",appendix=True)

    glossary=[
        ["Baseline","基线：用来公平比较的原始／可信参照模型。"],
        ["S4 / S4M","S4为结构化状态空间序列模型；S4M是其面向带缺失多变量预测的基线。"],
        ["ATPM / MDS-S4","自适应时间原型映射器／缺失感知双流S4：原论文两大部件。"],
        ["Encoder / Head / Adapter","编码器／输出头／适配器；分别提取表示、产生预测、对查询做残差调整。"],
        ["Query / Prototype bank","查询表示／原型库：用当前历史表示检索过去学到的模式。"],
        ["Mask / Nested view / L5、L20","观测掩码（1可见、0缺失）／嵌套视图（只再删除）／缺失块长5、20。"],
        ["PG95 / MIPP","PG95是仅配对一致性与0.95写门候选；MIPP是掩码干预原型净化框架。"],
        ["CNM-SE","因果嵌套缺失自集成：同模型看不同删减历史，固定融合输出。项目早期英文有Mask/Missing写法差异，本报告统一中文。"],
        ["MSE / MAE / SD","均方误差／平均绝对误差／标准差。MSE较敏感于大误差；MAE衡量绝对误差。"],
        ["EMA / CRN / RNG","指数移动平均（教师更新）／公共随机数（配对评估）／随机数生成器。"],
        ["Train / Validation / Test / Audit","训练／验证／测试／审计子集；用于开发选择的验证不能冒充独立最终测试。"],
        ["Seed / Epoch / Checkpoint / Smoke","随机种子／训练轮／模型检查点／极小链路测试。"],
    ]
    page("附录K｜英文专业名词和缩写：中文说明",[
        table(["英文／缩写","中文和项目中的含义"],glossary,[1.2,4.4],small=True),
    ],appendix=True)

    hist_glossary=[
        ["FUSR / HCUP","原型效用读取／历史可观测效用探针；判断检索会不会帮助预测。"],
        ["BMSA / SDRA / MRBA","写入预算、稳定性、多样性与掩码覆盖的候选规则；都属于原型写入侧尝试。"],
        ["CFR","反事实后悔风险：比较使用／不使用检索后的预测损失。"],
        ["MRRA / RADR","缺失响应增强／风险协调残差；试图使模型对掩码有响应并协调MSE、MAE。"],
        ["MCPR / PCD / TTRR / ARD","掩码条件残差、方向／幅度约束、残差治理、解析教师蒸馏等历史候选；不是当前共同开启的模块。"],
        ["BF-CMVD / DMRB / PBA / HFD","收益筛选多视图蒸馏／双残差基／主干摊销／隐藏特征蒸馏；把三路教师收益转给单路学生。"],
        ["PSAVS / Pareto-safe","帕累托安全自适应视图选择／不牺牲一个目标改善另一个目标的安全约束。"],
        ["Oracle / AUROC / NaN / OOM","使用标签的理想上限／ROC曲线下面积／数值非法／显存不足。oracle不等于可部署模型。"],
        ["Distillation / Ablation / FLOPs","蒸馏（学生学教师）／消融（关闭部件核验作用）／浮点计算量。"],
    ]
    page("附录L｜历史缩写只作索引，不作模块堆叠",[
        table(["缩写","中文定位"],hist_glossary,[1.3,4.3],small=True),
        item("历史候选名称较多，本报告按机制给中文解释，不据名称推断未实现功能。主讲只需要S4M、PG95、CNM-SE三个名字。", "note"),
    ],appendix=True)

    page("附录M｜老师可能追问：口语化回答",[
        ("sub","问：为什么2028与另外两个seed差很多？"),
        item("答：这些seed同时换了初始化和缺失布局，不只是训练随机数。距离修复在2028上双指标小幅退化，诊断不支持只是几个异常点导致；具体根因还没完成因果隔离。CNM在同一个修复检查点上仍有独立正增量，这两件事并不矛盾。"),
        ("sub","问：约1%的提升，要三次前向，值得发论文吗？"),
        item("答：目前只能说有可重复信号，不能说性价比足够。需要与同计算预算的强基线比较，补统一版本和独立确认；否则很容易被质疑只是常规集成换取小幅收益。"),
        ("sub","问：两个创新是不是把PG95和CNM写成两点就行？"),
        item("答：不行。PG95严格配对实验不稳定，修bug也不能算第二点。两个模块数量不是发表硬要求，独立有效、问题明确、证据完整才重要。"),
        ("sub","问：为什么不直接加一个新Attention或Mamba？"),
        item("答：原编码器已有注意力，预测主干已有状态空间建模。当前问题更像有效视图响应、路径贡献和基线可信度，不是单凭模块名字就能解决。先验证机制，再决定是否需要结构修改。"),
    ],appendix=True)

    page("附录N｜来源与交付文件",[
        item("原论文：Jing Peng等，S4M: S4 for multivariate time series forecasting with Missing values，ICLR 2025。模型参考Figure 2（PDF第4页），原文Table 1用于论文数值核对。", "note"),
        item("<a href='"+PAPER_URL+"'>打开官方论文PDF</a>　　<a href='"+GITHUB_URL+"'>打开官方GitHub</a>", "body"),
        table(["本次交付","用途"],[
            ["S4M_组会汇报_20261010.pdf","主报告：口语稿、架构、结果、负结果和证据判断。"],
            ["组会口语稿.md","可编辑正文＋附录，与PDF同源生成。"],
            ["完整运行工件索引.csv","302条原字段全保留；包括失败、smoke和诊断。"],
            ["完整配对结果.csv","61条全精度配对比较，含来源与协议标签；不跨表排名。"],
            ["figures/architecture_*.pdf / *.svg","三张矢量架构图，可单独插到后续汇报。"],
            ["_working/*_audit.json / *.md","数字／代码／历史来源核验、完整run路径和限制。"],
        ],[1.8,3.7],small=True),
        item("本次未改模型、训练入口、实验配置、指标实现和历史科学判定；未训练、未做模型前向、未启动GPU、未打开逐样本test缓存。报告制作依赖只安装在本输出目录_deps。", "note"),
    ],appendix=True)


def text_paragraph(text, style="body"):
    # Permit authored links only; all other strings are plain text, not HTML.
    if "<a href=" not in text:
        text = escape(str(text))
    return Paragraph(text.replace("\n", "<br/>"), STYLES[style])


def write_outputs():
    raw_pdf = OUT / "_working/report_layout.pdf"
    final_pdf = OUT / "S4M_组会汇报_20261010.pdf"
    c = canvas.Canvas(str(raw_pdf), pagesize=(W, HT))
    c.setTitle("S4M缺失条件时序预测：组会汇报与当前证据")
    c.setAuthor("S4M_MissingForecasting项目")
    overlays=[]; md=[]; page_qa=[]
    for idx,p in enumerate(PAGES):
        c.setFillColor(NAVY);c.setFont("CN",22);c.drawString(MARGIN, HT-42,p["title"])
        c.setStrokeColor(colors.HexColor("#C9D6DF"));c.line(MARGIN, HT-57, W-MARGIN, HT-57)
        c.setFillColor(GRAY);c.setFont("CN",9)
        c.drawString(MARGIN,23,"S4M_MissingForecasting · 2026-10-10 · "+("附录查阅" if p["appendix"] else "组会正文"))
        c.drawRightString(W-MARGIN,23,f"{idx+1} / {len(PAGES)}")
        md += ["## "+p["title"],""]
        y=HT-76
        if "diagram" in p:
            rect=(MARGIN,76,W-MARGIN,76+CW*540/1152)
            overlays.append((idx,rect,FIGURES/(p["diagram"]+".pdf")))
            y=HT-rect[3]-9
            items=[item(p["caption"],"note"), item("口语稿："+p["speech"],"body")]
            md += [f'![架构图](figures/{p["diagram"]}.svg)',"",p["caption"],"","口语稿："+p["speech"],"","来源："+p["source"],""]
        else:
            items=p["items"]
        for it in items:
            if it[0]=="sub":
                y-=5;c.setFillColor(BLUE);c.setFont("CN",12.8);c.drawString(MARGIN,y-13,it[1]);y-=26
                md += ["### "+it[1],""]
            elif it[0]=="p":
                q=text_paragraph(it[1],it[2]);qw,qh=q.wrap(CW,y-43)
                if y-qh<40:
                    raise RuntimeError(f'Page {idx+1}: paragraph exceeds page by {40-(y-qh):.1f} pt')
                q.drawOn(c,MARGIN,y-qh);y-=qh+9
                if "diagram" not in p:md += [it[1],""]
            elif it[0]=="table":
                _,heads,rows,weights,small=it
                weights=weights or [1]*len(heads);colw=[CW*v/sum(weights) for v in weights]
                content=[[text_paragraph(h,"headcell") for h in heads]]+[[text_paragraph(str(v),"smallcell" if small else "cell") for v in r] for r in rows]
                tb=Table(content,colWidths=colw,repeatRows=1,hAlign="LEFT")
                tb.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),NAVY),("VALIGN",(0,0),(-1,-1),"TOP"),
                    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,BG]),
                    ("LINEBELOW",(0,0),(-1,0),0.7,NAVY),
                    ("LINEBELOW",(0,1),(-1,-1),0.3,colors.HexColor("#DCE4E9")),
                    ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
                    ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
                ]))
                tw,th=tb.wrap(CW,HT)
                if y-th<40:
                    raise RuntimeError(f'Page {idx+1}: table exceeds page by {40-(y-th):.1f} pt')
                tb.drawOn(c,MARGIN,y-th);y-=th+11
                md += ["| "+" | ".join(heads)+" |","| "+" | ".join(["---"]*len(heads))+" |"]
                md += ["| "+" | ".join(map(str,r))+" |" for r in rows]+[""]
        page_qa.append({"page":idx+1,"title":p["title"],"lowest_body_y_pt":round(y,2),"appendix":p["appendix"]})
        c.showPage()
    c.save()
    doc=fitz.open(raw_pdf)
    for idx,rect,pdfpath in overlays:
        src=fitz.open(pdfpath);doc[idx].show_pdf_page(fitz.Rect(rect),src,0);src.close()
    # Bookmarks make the long appendix usable without burdening the spoken talk.
    doc.set_toc([[1 if not p["appendix"] else 2,p["title"],i+1] for i,p in enumerate(PAGES)])
    doc.save(final_pdf,garbage=4,deflate=True)
    architecture_pdf=fitz.open()
    for stem in ["architecture_01_backbone","architecture_02_pg95","architecture_03_cnm"]:
        figure=fitz.open(FIGURES/(stem+".pdf"));architecture_pdf.insert_pdf(figure);figure.close()
    architecture_pdf.save(OUT/"模型架构图_三页.pdf",garbage=4,deflate=True);architecture_pdf.close()
    extracted="\n".join(p.get_text() for p in doc)
    (OUT/"_working/pdf_extracted_text.txt").write_text(extracted)
    (OUT/"组会口语稿.md").write_text("# S4M缺失条件时序预测：组会口语稿\n\n日期：2026-10-10。正文01–15约12–15分钟，附录查阅；PDF同源。\n\n"+"\n".join(md))
    with (OUT/"完整配对结果.csv").open("w",encoding="utf-8-sig",newline="") as fp:
        cols=["table","dataset","split","version","seed","baseline_method","new_method","baseline_mse","baseline_mae","new_mse","new_mae","delta_mse_new_minus_baseline","delta_mae_new_minus_baseline","mse_gain_percent","mae_gain_percent","dual_win","source","same_protocol_internally","same_protocol_as_paper"]
        wr=csv.DictWriter(fp,fieldnames=cols,extrasaction="ignore");wr.writeheader();wr.writerows(DATA["comparison_rows"])
    selected=list(range(len(doc)))
    previews=OUT/"_working/previews";previews.mkdir(exist_ok=True)
    for i in selected:
        if i<len(doc):doc[i].get_pixmap(matrix=fitz.Matrix(1.45,1.45)).save(previews/f"page_{i+1:02d}.png")
    text_issues=[]
    for i,p in enumerate(doc):
        for b in p.get_text("dict")["blocks"]:
            if b.get("type")!=0:continue
            for ln in b["lines"]:
                for s in ln["spans"]:
                    x0,y0,x1,y1=s["bbox"]
                    if x0<0 or y0<0 or x1>W+1 or y1>HT+1:text_issues.append({"page":i+1,"text":s["text"],"bbox":s["bbox"]})
    required=["0.517612","0.512650","0.315987","0.313826","22.44未执行","0.957982","0.683491","PG95","CNM","0.95","5.87e-4","1e-4"]
    checks={x:x in extracted for x in required}
    provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/"_working/results_audit.json",OUT/"_working/architecture_audit.json",OUT/"_working/history_audit.json",OUT/"_working/artifact_register.json",OUT/"_working/s4m_iclr2025.pdf"]}
    qa={"report_date":"2026-10-10","pdf":str(final_pdf),"pages":len(doc),"main_pages":15,"searchable_text_characters":len(extracted),"null_glyph_count":extracted.count(chr(0)),"out_of_page_text":text_issues,"required_text_checks":checks,"all_required_text_present":all(checks.values()),"page_layout":page_qa,"architecture_vector_embedding":True,"architecture_raster_image_counts":[len(doc[i].get_images(full=True)) for i in [2,3,4]],"new_model_runs":0,"training_code_or_config_changes":0,"npz_cache_reads":0,"source_sha256":provenance,"visual_review_status":"pending_root_view_image_and_independent_content_review"}
    (OUT/"_working/qa_report.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2))
    print(json.dumps({"pdf":str(final_pdf),"pages":len(doc),"searchable_chars":len(extracted),"out_of_page_spans":len(text_issues),"required_text_pass":all(checks.values()),"diagram_raster_images":qa["architecture_raster_image_counts"]},ensure_ascii=False))
    assert all(checks.values()) and not text_issues and not qa["null_glyph_count"], "PDF QA failed"
    doc.close()


if __name__ == "__main__":
    create_content()
    write_outputs()
