#!/usr/bin/env python3
"""Six-page meeting brief. Produces a candidate; publishing is a separate step.

This report reads audited JSON only: no model imports, no CUDA, no NPZ access.
The user explicitly authorized replacing the previous main PDF after review.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SHORT=OUT/"_short"
SHORT.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR",str(SHORT/".mplconfig"))
sys.path.insert(0,str(OUT/"_deps"))
import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph,Table,TableStyle

CNFONT=Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
pdfmetrics.registerFont(TTFont("CN",str(CNFONT),subfontIndex=0))
matplotlib.rcParams.update({"mathtext.fontset":"stix","font.family":"DejaVu Sans","pdf.fonttype":42,"svg.fonttype":"none","text.usetex":False})
TABLES=json.loads((OUT/"_working/results_audit.json").read_text())["tables"]
W,H=960,600
M=44
CW=W-2*M
COL={"ink":"#1D3445","muted":"#647789","blue":"#3C6485","teal":"#267E82","orange":"#B97135","pale":"#F0F5F8","line":"#DBE5EB"}
OVERLAYS=[]
TEXTBOXES=[]

SPEECH=[
    "今天我用五分钟汇报这个项目。我们的任务是用带缺失的历史数据预测未来，不是只把历史补完整。基线是S4M。我们主要尝试了训练侧的PG95和推理侧的CNM。目前最可信的信号来自CNM，但提升不大，也不是四个数据集都有效。下面我先说明模块加在哪里，再讲结果、问题和下一步。",
    "先看主干。历史数据填补以后编码成查询，检索原型库，再把检索结果、查询和填补历史一起融合，交给S4预测。PG95加在查询适配和训练写库的位置，没有替换预测主干。训练时增加配对一致性损失，并用零点九五的稳定门筛写入候选。推理仍保留适配器。它是训练候选，不是距离公式的修复版。",
    "CNM更简单：同一个冻结模型看三份历史。原始路不变，另外两路只从已观测历史中再删除相同数量的数据，用两种缺失块长构造视图，然后固定融合三份预测。没有加入未来真值，也没有新增参数。这个设计希望不同视图的误差能够互补，但需要三次前向，不是免费的提升。",
    "这是现在最可靠的一组结果。我们用Traffic上距离公式修复后的S4M作为基线，只比较同一个检查点加不加CNM，三个seed都是均方误差和绝对误差同时更好。逐seed平均，MSE下降约百分之零点九六，MAE下降约百分之零点六八。这只是已有验证结果的复核，不能说成新测试或统计显著。",
    "看跨数据集边界。新版三路在Weather验证集有收益，在Traffic也有小幅收益。但ETTh1完整组合反而退化，Electricity只有MSE略好、MAE变差，而且它没有进入本轮测试。另外，PG95在严格配对的Traffic实验里只在一个seed上双胜，平均还退化，所以它现在不能算稳定的第二贡献。不同对照之间不能直接排名。",
    "因此，我的结论是，这个项目值得继续验证，但现在还不够形成完整方法论文，更不能认定达到二区要求。主要缺口是收益小却要三次前向，跨数据集不稳定，第二贡献没成立，基线还存在一个待处理的原型索引错误。下一步先独立纠错，再冻结一个CNM版本，补新的确认边界和强基线。纠错收益单独记录，不算创新。",
]


def para(c,text,x,top,width=CW,size=17,leading=None,color="ink",bottom=570):
    p=Paragraph(escape(text),ParagraphStyle("p",fontName="CN",fontSize=size,leading=leading or size*1.45,wordWrap="CJK",textColor=colors.HexColor(COL[color])))
    _,h=p.wrap(width,H)
    assert top+h<=bottom, f"text overflow: {text[:40]} at {top+h:.1f}"
    p.drawOn(c,x,H-top-h)
    TEXTBOXES.append({"text":text,"box":[x,top,x+width,top+h]})
    return top+h


def math_asset(name,expr):
    """MathText consumes LaTeX syntax and exports real vector math glyphs."""
    fig=plt.figure(figsize=(12,0.8))
    fig.text(0.01,.48,"$"+expr+"$",fontsize=24,color=COL["ink"],va="center",ha="left")
    path=SHORT/(name+".pdf")
    fig.savefig(path,bbox_inches="tight",pad_inches=.03)
    fig.savefig(SHORT/(name+".svg"),bbox_inches="tight",pad_inches=.03)
    plt.close(fig)
    return path


def insert_math(page,path,rect):
    OVERLAYS.append((page,rect,path))


def header(c,num,title,subtitle):
    c.setFillColor(colors.HexColor(COL["blue"]));c.setFont("CN",12);c.drawString(M, H-25,f"{num:02d} / 06    S4M · 五分钟组会")
    c.setFillColor(colors.HexColor(COL["ink"]));c.setFont("CN",25);c.drawString(M,H-62,title)
    para(c,subtitle,M,76,size=12,color="muted")


def notes(c,i,source):
    c.setStrokeColor(colors.HexColor(COL["line"]));c.line(M,H-469,W-M,H-469)
    para(c,"口语稿",M,478,size=11,color="blue")
    para(c,SPEECH[i],M,498,size=13,leading=19,bottom=570)
    c.setFont("CN",8.5);c.setFillColor(colors.HexColor(COL["muted"]));c.drawString(M,14,source)
    c.drawRightString(W-M,14,str(i+1)+" / 6")


def tab(c,heads,rows,top,widths,size=15):
    style=ParagraphStyle("cell",fontName="CN",fontSize=size,leading=size*1.35,wordWrap="CJK",textColor=colors.HexColor(COL["ink"]))
    headstyle=ParagraphStyle("head",parent=style,textColor=colors.white)
    content=[[Paragraph(escape(str(h)),headstyle) for h in heads]]+[[Paragraph(escape(str(x)),style) for x in r] for r in rows]
    t=Table(content,colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor(COL["blue"])),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor(COL["pale"])]),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LINEBELOW",(0,1),(-1,-1),.4,colors.HexColor(COL["line"])),
        ("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12),
        ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
    ]))
    tw,th=t.wrap(CW,H)
    assert top+th<460,"table overflow"
    t.drawOn(c,M,H-top-th)
    return top+th


def pct(x):return f"{x:+.2f}%"
def relative(a,b):return (a-b)/a*100


def build():
    for p in [SHORT/"backbone_short.pdf",SHORT/"cnm_short.pdf"]:
        if not p.exists():raise FileNotFoundError(f"waiting for reviewed diagram: {p}")
    fm={
        "task":math_asset("equation_task",r"f_{\theta}(X,M)\longrightarrow\widehat{Y},\qquad X,M\in\mathbb{R}^{B\times L\times C},\quad\widehat{Y}\in\mathbb{R}^{B\times P\times C}"),
        "fusion":math_asset("equation_fusion",r"O=\mathrm{Linear}([\,R,\widetilde{U},Z\,])+\widetilde{U}"),
        "loss":math_asset("equation_loss",r"\mathcal{L}=\mathcal{L}_{\mathrm{pred}}+0.05\,\mathcal{L}_{\mathrm{inv}}"),
        "cnm":math_asset("equation_cnm",r"\widehat{Y}=0.50\,\widehat{Y}_{\mathrm{orig}}+0.25\,\widehat{Y}_{5}+0.25\,\widehat{Y}_{20}"),
    }
    raw=SHORT/"short_layout.pdf";candidate=SHORT/"short_candidate.pdf"
    c=canvas.Canvas(str(raw),pagesize=(W,H));c.setTitle("S4M缺失条件预测｜五分钟组会简报")
    titles=["当前结论：局部有效，还没有形成完整方法","PG95改在查询表示与训练写入，不替换主干","CNM在输入端分视图，在输出端固定融合","Traffic：同一修复基线，三seed方向一致","跨数据集边界：不是四个数据集都成功","值得继续，但暂不足以支撑完整方法论文"]
    # 1: claim-led. No cover-only page and no historical process list.
    header(c,1,titles[0],"任务：用带缺失的多变量历史预测未来。基线：S4M（ICLR 2025）。")
    para(c,"我们保留下来的信号，是同模型多视图推理的误差互补。",M,132,size=24)
    insert_math(0,fm["task"],(M,189,W-M,239))
    para(c,"两个候选方向",M,270,size=14,color="muted")
    para(c,"PG95：配对一致性训练＋稳定写入门。独立效果尚不稳定。",M,302,size=19)
    para(c,"CNM-SE：因果嵌套缺失自集成。部分数据上有可重复小幅收益。",M,349,size=19)
    para(c,"研究做到22.43；22.44未执行。此次只压缩汇报，没有新增实验。",M,415,size=12,color="muted")
    notes(c,0,"基线来源：S4M，ICLR 2025；内容依据已完成实验审计。")
    c.showPage()
    # 2: backbone hero figure and two vector equations.
    header(c,2,titles[1],"蓝色：原S4M共享路径；橙色：PG95增量；虚线：只用于训练的机制。")
    OVERLAYS.append((1,(M,104,W-M,399),SHORT/"backbone_short.pdf"))
    insert_math(1,fm["fusion"],(M,400,490,430))
    insert_math(1,fm["loss"],(516,400,W-M,430))
    para(c,"配对损失为余弦不变损失；预测损失只用未来观测目标。0.95门筛教师写入候选，不筛预测输出。",M,441,size=11.5,color="muted")
    notes(c,1,"按实际代码重绘，参考S4M Figure 2；PG95因子分解关闭，推理适配器保留。")
    c.showPage()
    # 3: input wrapper hero, clean typography.
    header(c,3,titles[2],"每路重新填补；三路共用冻结模型与原型库，只删已观测历史，不使用未来真值。")
    OVERLAYS.append((2,(M,104,W-M,399),SHORT/"cnm_short.pdf"))
    insert_math(2,fm["cnm"],(172,400,790,432))
    para(c,"L5／L20是缺失块长；两路删除数相同。新增参数为0，融合需3次前向，不等于3套模型。",M,442,size=11.5,color="muted")
    notes(c,2,"新版：等计数三路固定权重。Weather历史两路测试证据不套用此图。")
    c.showPage()
    # 4: all three latest matched seeds; statistics generated from audited JSON.
    header(c,4,titles[3],"固定验证集、同检查点、同掩码和评价规则。MSE（均方误差）／MAE（平均绝对误差）越小越好。")
    sr=[]
    for r in TABLES["traffic22_43_cnm_per_seed_both_formulas"]:
        x=r["fixed"];a,b=x["baseline"],x["cnm"];g=x["relative_gain_percent"]
        sr.append([r["seed"],f'{a["mse"]:.6f} → {b["mse"]:.6f}',f'{a["mae"]:.6f} → {b["mae"]:.6f}',pct(g["mse"])+" / "+pct(g["mae"])])
    y=tab(c,["随机种子","MSE：修复S4M → ＋CNM","MAE：修复S4M → ＋CNM","MSE / MAE改善"],sr,123,[82,282,282,226],size=14)
    s=TABLES["traffic22_43_cnm_summary_both_formulas"]["fixed"]
    gm=s["relative_gain_percent"]["mse"]["mean"];ga=s["relative_gain_percent"]["mae"]["mean"]
    c.setFont("CN",31);c.setFillColor(colors.HexColor(COL["teal"]));c.drawString(M,H-y-65,f"平均MSE ↓ {gm:.2f}%     MAE ↓ {ga:.2f}%")
    para(c,"3/3双指标改善，仅表示方向一致，不代表统计显著。",M,y+92,size=16)
    para(c,"22.43是既有验证误差缓存复核，不是新训练、新前向或独立测试。改善率取逐seed平均。",M,429,size=11.5,color="muted")
    notes(c,3,"结果来源：22.43修复后同检查点CNM审计；纠错收益与创新增量分开。")
    c.showPage()
    # 5: same method generation, clearly labeled heterogeneous comparisons.
    header(c,5,titles[4],"正数表示误差下降，负数表示退化。以下对照对象不同，只说明适用边界，不作跨行排名。")
    w=TABLES["weather18_multiscale_validation"]
    wg=[sum(relative(r["routes"]["on"][k],r["routes"]["multiscale"][k]) for r in w)/len(w) for k in ["mse","mae"]]
    eg=TABLES["etth1_stage20"]["gains"]["full_method_vs_s4m_pct"]
    e=TABLES["electricity_stage21"];em=e["routes"]["multiscale"]
    ee=[relative(e["baseline"][k],em[k]) for k in ["mse","mae"]]
    rows=[
        ["Weather","验证／2seed","同PG95加CNM",pct(wg[0]),pct(wg[1])],
        ["Traffic","验证／3seed","同修复S4M加CNM",pct(gm),pct(ga)],
        ["ETTh1","测试／1seed","S4M→PG95＋CNM",pct(eg["mse"]),pct(eg["mae"])],
        ["Electricity","验证／1seed","S4M→PG95＋CNM",pct(ee[0]),pct(ee[1])],
    ]
    y=tab(c,["数据集","划分／数量","对照对象","MSE改善","MAE改善"],rows,120,[124,132,294,161,161],size=14)
    pg=TABLES["traffic22_37_factorial_summary"]["gains"]["pg95_gain_without_cnm_percent"]
    para(c,f'PG95严格配对（Traffic原公式版）：仅{pg["dual_wins"]}/3双胜，平均MSE / MAE退化{abs(pg["mean"][0]):.2f}% / {abs(pg["mean"][1]):.2f}%。',M,y+21,size=15,color="orange")
    para(c,"ETTh1与Electricity视图响应近乎不变。Electricity本轮验证未过，不访问测试集。",M,y+58,size=12.5,color="muted")
    notes(c,4,"结果来源：Weather 18；Traffic 22.43；ETTh1 20；Electricity 21；PG95配对22.37。")
    c.showPage()
    # 6: open discussion layout, exactly three problems and one route.
    header(c,6,titles[5],"不把模块数量当创新质量，不把错误修复当方法贡献，也不预先保证SCI分区。")
    lines=[
        ("精度—代价未闭环","小幅收益换3次前向；最新版本缺隔离端到端效率与强基线对照。"),
        ("通用性和第二贡献不足","两个外部数据未成功；PG95尚不稳定，不能说两项创新已成立。"),
        ("基线与确认边界仍需可信","距离公式已纠错，原型代表索引问题待处理；验证集多次参与开发。"),
    ]
    for j,(h,body) in enumerate(lines):
        top=125+j*76
        para(c,str(j+1)+"  "+h,M,top,size=19,color="blue")
        para(c,body,M+26,top+30,size=15)
    para(c,"下一步：独立纠错 → 冻结CNM版本 → 新确认数据与强基线",M,379,size=19,color="teal")
    para(c,"已有局部方向性证据；尚不能认定达到二区投稿要求。先把实验闭环，再决定写作定位。",M,430,size=12.5,color="muted")
    notes(c,5,"计划而非已执行：22.44未启动；本次未改变任何模型、训练或评价协议。")
    c.showPage();c.save()

    doc=fitz.open(raw)
    for i,rect,p in OVERLAYS:
        src=fitz.open(p);doc[i].show_pdf_page(fitz.Rect(rect),src,0);src.close()
    doc.set_toc([[1,t,i+1] for i,t in enumerate(titles)])
    doc.save(candidate,garbage=4,deflate=True)
    text="\n".join(p.get_text() for p in doc)
    (SHORT/"short_extracted_text.txt").write_text(text)
    preview=SHORT/"previews";preview.mkdir(exist_ok=True)
    for i,p in enumerate(doc):p.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(preview/f"page_{i+1:02d}.png")
    outside=[]
    for i,p in enumerate(doc):
        for b in p.get_text("dict")["blocks"]:
            if b.get("type")!=0:continue
            for ln in b["lines"]:
                for sp in ln["spans"]:
                    x0,y0,x1,y1=sp["bbox"]
                    if x0<0 or y0<0 or x1>W+.1 or y1>H+.1:outside.append({"page":i+1,"text":sp["text"],"bbox":sp["bbox"]})
    core_source=json.loads((OUT/"figures/architecture_manifest.json").read_text())["source_sha256"]
    unchanged={k:hashlib.sha256((ROOT/"baselines/S4M/s4m"/k).read_bytes()).hexdigest()==v for k,v in core_source.items()}
    english_pattern=re.findall(r"[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9]+",text)
    qa={"pages":len(doc),"canvas":[W,H],"candidate":str(candidate),"target":str(OUT/"S4M_组会汇报_20261010.pdf"),"null_glyph_count":text.count(chr(0)),"literal_code_subscripts":english_pattern,"out_of_page_text":outside,"raster_image_count":sum(len(p.get_images(full=True)) for p in doc),"spoken_characters":sum(len(s) for s in SPEECH),"planned_page_seconds":[35,55,50,55,60,45],"planned_total_seconds":300,"core_source_hashes_unchanged":unchanged,"formulas_backend":"matplotlib MathText: STIX vector mathematical glyphs; LaTeX-style syntax, no external LaTeX engine","source_json_sha256":hashlib.sha256((OUT/"_working/results_audit.json").read_bytes()).hexdigest(),"review_status":"PENDING_VISUAL_AND_INDEPENDENT_REVIEW","new_training_or_inference":False,"test_cache_access":False}
    (SHORT/"short_qa.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2))
    md=["# 五分钟组会口语稿", "", "6页，计划35／55／50／55／60／45秒；总计约5分钟。可按个人语速调整。", "", "公式在PDF中为LaTeX风格矢量数学字形，不是字面下划线。", ""]
    for i,(title,speech) in enumerate(zip(titles,SPEECH)):
        md += ["## "+str(i+1)+". "+title,"",speech,""]
    (SHORT/"五分钟口语稿.md").write_text("\n".join(md))
    print(json.dumps({"pages":len(doc),"candidate":str(candidate),"nul":qa["null_glyph_count"],"code_subscripts":english_pattern,"bbox_issues":len(outside),"raster_images":qa["raster_image_count"],"speech_chars":qa["spoken_characters"],"core_code_unchanged":all(unchanged.values())},ensure_ascii=False))
    assert len(doc)==6 and not outside and not english_pattern and qa["null_glyph_count"]==0 and all(unchanged.values())
    doc.close()


if __name__=="__main__":build()
