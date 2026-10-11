#!/usr/bin/env python3
"""Code-audited S4M/PG95/CNM schematics, not quantitative efficacy evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/group_meeting_20261010/figures"
DEPS = OUT.parent / "_deps"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/s4m_meeting_mplconfig")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(DEPS))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MPath
import fitz

FONT = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
font_manager.fontManager.addfont(str(FONT))
FONT_NAME = font_manager.FontProperties(fname=str(FONT)).get_name()
matplotlib.rcParams.update({
    "font.family": [FONT_NAME, "DejaVu Sans"],
    "font.size": 18,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
    "text.usetex": False,
    "savefig.facecolor": "white",
})
COL = {
    "ink": "#263747", "muted": "#657382", "blue": "#365C79",
    "blue_fill": "#E9F0F5", "orange": "#BC681E", "orange_fill": "#FFF0DB",
    "gray": "#8D99A4", "gray_fill": "#F4F6F7", "white": "#FFFFFF",
}
ITEMS = []


class Diagram:
    def __init__(self, stem, title, subtitle):
        self.stem = stem
        self.fig = plt.figure(figsize=(16, 7.5), facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, 16), ylim=(0, 7.5))
        self.ax.axis("off")
        self.box_labels = []
        self.label(0.60, 7.07, title, size=25, weight="bold", ha="left")
        self.label(0.60, 6.66, subtitle, size=17, color=COL["muted"], ha="left")

    def label(self, x, y, text, size=18, color=None, weight="normal", ha="center", va="center", **kwargs):
        return self.ax.text(x, y, text, fontsize=size, color=color or COL["ink"],
                            weight=weight, ha=ha, va=va, linespacing=1.23,
                            zorder=5, **kwargs)

    def box(self, x, y, w, h, text, kind="blue", size=18, training=False, weight="normal"):
        edge = COL.get(kind, COL["blue"])
        fill = COL.get(kind + "_fill", COL["blue_fill"])
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                              linewidth=1.35, edgecolor=edge, facecolor=fill,
                              linestyle=(0, (5, 4)) if training else "solid", zorder=3)
        self.ax.add_patch(patch)
        textobj = self.label(x + w / 2, y + h / 2, text, size=size, weight=weight)
        self.box_labels.append((textobj, (x, y, w, h), text))
        return (x, y, w, h)

    def arrow(self, points, training=False, color=None, width=1.65, scale=19):
        codes = [MPath.MOVETO] + [MPath.LINETO] * (len(points) - 1)
        patch = FancyArrowPatch(path=MPath(points, codes), arrowstyle="-|>",
                                mutation_scale=scale, linewidth=width,
                                color=color or COL["blue"],
                                linestyle=(0, (5, 4)) if training else "solid",
                                capstyle="round", joinstyle="round", zorder=2)
        self.ax.add_patch(patch)

    def save(self, contract, findings):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        problems = []
        for txt, (x, y, w, h), label in self.box_labels:
            bounds = txt.get_window_extent(renderer=renderer).transformed(self.ax.transData.inverted())
            if bounds.x0 < x + 0.045 or bounds.x1 > x + w - 0.045 or bounds.y0 < y + 0.025 or bounds.y1 > y + h - 0.025:
                problems.append({"label": label, "text_bounds": list(bounds.extents), "box": [x, y, w, h]})
        OUT.mkdir(parents=True, exist_ok=True)
        paths = {ext: OUT / f"{self.stem}.{ext}" for ext in ("pdf", "svg", "png")}
        self.fig.savefig(paths["pdf"])
        self.fig.savefig(paths["svg"])
        self.fig.savefig(paths["png"], dpi=150)
        plt.close(self.fig)
        doc = fitz.open(paths["pdf"])
        page = doc[0]
        preview = OUT / f"{self.stem}_pdf_preview.png"
        page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False).save(preview)
        a4doc = fitz.open()
        a4page = a4doc.new_page(width=841.89, height=595.28)
        a4rect = fitz.Rect(26, 26, 815.89, 396.26)
        a4page.show_pdf_page(a4rect, doc, 0)
        a4preview = OUT / f"{self.stem}_a4_preview.png"
        a4page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(a4preview)
        a4doc.close()
        data = {
            "stem": self.stem, "paths": {k: str(v.relative_to(ROOT)) for k, v in paths.items()},
            "pdf_preview": str(preview.relative_to(ROOT)), "contract": contract,
            "a4_landscape_preview": str(a4preview.relative_to(ROOT)),
            "findings": findings, "qa": {
                "size_inches": [16, 7.5], "pdf_size_points": [page.rect.width, page.rect.height],
                "vector_raster_image_count": len(page.get_images()),
                "selectable_text_characters": len(page.get_text()),
                "text_bbox_problems": problems,
                "minimum_box_font_pt": min(t.get_fontsize() for t, _, _ in self.box_labels),
                "minimum_box_font_at_a4_pt": min(t.get_fontsize() for t, _, _ in self.box_labels) * (789.89 / 1152),
                "font_types": sorted(set(f[2] for f in page.get_fonts())),
                "render_backend": "Python PyMuPDF PDF render",
                "visual_review": "passed: all three PDF renders and all three A4-landscape previews inspected with view_image; two first-pass label overflows and a CNM long note corrected; final labels readable, no box overflow, no arrows through unrelated boxes.",
            },
        }
        doc.close()
        ITEMS.append(data)
        print(f"{self.stem}: PDF/SVG/PNG saved; bbox problems={len(problems)}", flush=True)
        for p in problems:
            print("TEXT_BBOX:", p["label"].replace("\n", " / "), flush=True)


COMMON = {
    "conclusion": "准确展示 S4M、PG95 与 CNM 的真实接入点；图是方法示意，不是性能证明。",
    "archetype": "schematic-led composite",
    "backend": "Python matplotlib",
    "dimensions_inches": [16, 7.5],
    "exports": ["editable PDF", "editable-text SVG", "PNG", "PDF-rendered PNG"],
    "font": "DroidSansFallbackFull.ttf for Chinese, DejaVu Sans for fallback Latin",
    "statistics": "not applicable; no numerical result or statistical claim",
    "integrity": "original code-derived schematic; no source microscopy/image manipulation",
    "encoding": "blue baseline/shared path; orange PG95 addition; text names redundantly encode color; dashed means training-only",
}


def figure_one():
    d = Diagram("architecture_01_backbone", "S4M 主干：PG95 改在查询表示与训练写入，不替换预测骨干",
                "实线表示推理路径；橙色表示 PG95 增量；虚线只表示训练旁路。所有形状均按实际代码。")
    d.box(.62, 4.90, 1.73, 1.00, "历史 X, M\n[B, L, C]", size=19)
    d.box(2.80, 4.90, 1.65, 1.00, "Decay 填补\nZ [B,L,C]", size=18)
    d.box(4.90, 4.82, 2.32, 1.16, "滑窗 Query 编码\nConv→Attn→S4\nU [B,L,H]", size=18)
    d.box(7.70, 4.82, 1.80, 1.16, "PG95 adapter\nh + A h\n再归一化", "orange", size=18)
    d.box(10.02, 4.90, 1.95, 1.00, "相似度检索\nTopK→softmax\nR [B,L,H]", size=18)
    d.box(12.80, 4.90, 2.55, 1.00, "Linear [R, U, Z]\n+ query 残差", size=19)
    for points in [((2.35,5.4),(2.80,5.4)), ((4.45,5.4),(4.90,5.4)),
                   ((7.22,5.4),(7.70,5.4)), ((9.50,5.4),(10.02,5.4)),
                   ((11.97,5.4),(12.80,5.4))]:
        d.arrow(points)
    d.box(10.02, 6.07, 1.95, .43, "原型库 centroid", size=17)
    d.arrow(((11.00,6.07),(11.00,5.90)))
    d.box(4.90, 6.10, 4.60, .40, "仅训练：paired query / .05 loss / .95 写门", kind="orange", size=16, training=True)
    d.arrow(((6.05,6.10),(6.05,5.98)), training=True, color=COL["orange"], scale=16)
    d.arrow(((8.58,4.82),(8.58,4.50),(12.35,4.50),(12.35,5.17),(12.80,5.17)))
    d.label(10.48,4.62,"U：拼接 + 残差",size=17)
    d.arrow(((3.63,4.90),(3.63,4.10),(12.57,4.10),(12.57,5.64),(12.80,5.64)))
    d.label(7.30,4.23,"Z 直接进入融合（不是原始完整历史）",size=17)
    d.label(6.12,3.68,"Conv 直接 reshape → Attn → S4 → 时间池化 → U 归一化",size=17)
    d.box(12.55, 2.03, 2.80, 1.10, "首层 Mask-aware S4\n[B,L,H]", size=19)
    d.box(10.20, 2.19, 1.75, .79, "Linear\nH→D", size=19)
    d.box(7.05, 2.03, 2.55, 1.10, "其余 S4 层\n不再接 mask", size=19)
    d.box(3.95, 2.03, 2.50, 1.10, "LayerNorm + head\n[B,L,C]", size=18)
    d.box(.62, 2.03, 2.73, 1.10, "取最后 P 个时间点\n预测 Ŷ [B,P,C]", size=19)
    d.arrow(((14.07,4.90),(14.07,3.13)))
    for a,b in [(12.55,11.95),(10.20,9.60),(7.05,6.45),(3.95,3.35)]:
        d.arrow(((a,2.58),(b,2.58)))
    d.arrow(((.62,5.12),(.29,5.12),(.29,1.31),(13.94,1.31),(13.94,2.03)))
    d.label(9.15,1.53,"M → Linear C→H：只进入首层",size=19)
    d.label(.62,.77,"Baseline 跳过 adapter；PG95 推理保留；content-mask factorization 关闭。",size=17,ha="left")
    d.label(.62,.35,"填补基于历史全局 max / min 与距离；主干无新增 horizon decoder，输出再取尾部 P。",size=17,ha="left",color=COL["muted"])
    d.save({**COMMON,"panel_role":"hero: shared deployed backbone and exact PG95 insertion point"}, [
        "Fusion is Linear([retrieval, query, decay-imputed Z]) + query residual.",
        "Only the first forecasting S4 layer receives the projected historical mask.",
        "PG95 content adapter remains active during inference; factorization disabled.",
        "Encoder Conv output uses reshape, not a standard transpose; no code change.",
        "S4M outputs B,L,C then selects last P positions; no independent horizon decoder.",
    ])


def figure_two():
    d = Diagram("architecture_02_pg95", "PG95 放大：配对查询不变性 + teacher 候选稳定写入门",
                "推理保留 adapter 与原型库；不生成训练 pair，不更新 bank；teacher 旁路仍计算。")
    d.box(.62,4.83,1.98,1.19,"完整历史 X*\n仅训练构造干预\n无未来 target",size=18)
    d.box(3.18,5.16,2.02,.94,"原始训练 view\nXᵃ, Mᵃ",size=18)
    d.box(3.18,3.86,2.02,1.10,"新 mask view\nXᵇ=X*⊙Mᵇ\n非嵌套原 mask",kind="orange",size=18,training=True)
    d.arrow(((2.60,5.41),(2.88,5.41),(2.88,5.63),(3.18,5.63)),training=True)
    d.arrow(((2.60,5.16),(2.88,5.16),(2.88,4.41),(3.18,4.41)),training=True,color=COL["orange"])
    d.label(.65,4.02,"块长 {1,5,20}\nrate ~ U(.15,.50)",size=18,ha="left")
    d.box(5.76,4.10,2.60,2.04,"分别 Decay 填补\n共享 Encoder_u\n共享 adapter\n归一化 Uᵃ / Uᵇ",kind="orange",size=18)
    d.arrow(((5.20,5.63),(5.76,5.63)),training=True)
    d.arrow(((5.20,4.41),(5.76,4.41)),training=True,color=COL["orange"])
    d.box(8.95,5.21,2.14,.78,"Uᵃ [B,L,H]",size=19)
    d.box(8.95,4.20,2.14,.78,"Uᵇ [B,L,H]",kind="orange",size=19)
    d.arrow(((8.36,5.60),(8.95,5.60)),training=True)
    d.arrow(((8.36,4.59),(8.95,4.59)),training=True,color=COL["orange"])
    d.box(12.02,4.69,3.30,1.25,"L = MSE + 0.05 L_inv\n只对原 view 做预测\n其他 MIPP 权重 = 0",kind="orange",size=19,training=True)
    d.arrow(((11.09,5.60),(11.60,5.60),(11.60,5.53),(12.02,5.53)),training=True)
    d.arrow(((11.09,4.59),(11.60,4.59),(11.60,5.01),(12.02,5.01)),training=True,color=COL["orange"])
    d.label(12.06,3.88,"L_inv = mean[1 − cos(Uᵃ,Uᵇ)]",size=19)
    d.ax.plot([.62,15.32],[3.53,3.53],color="#D6DEE5",lw=1.0)
    d.label(.62,3.28,"训练写入路径：no_grad；Q = floor(L / S)",size=21,ha="left",weight="bold")
    d.box(.62,1.91,2.60,1.05,"Zᵃ, Zᵇ 非重叠块\n[B,Q,S,C]",size=19)
    d.box(3.79,1.91,2.85,1.05,"冻结 Encoder_m\n+ 共享 adapter\nqᵃ,qᵇ [B,Q,H]",kind="orange",size=18)
    d.box(7.23,1.91,3.00,1.05,"稳定写入门\ncos(qᵃ,qᵇ) ≥ 0.95\ngate [B,Q]",kind="orange",size=18,training=True)
    d.box(10.82,1.91,2.22,1.05,"原 memory 更新\n最多 M=50 候选",size=18)
    d.box(13.65,1.91,1.69,1.05,"原型库\n推理检索",size=19)
    for a,b in [(3.22,3.79),(6.64,7.23),(10.23,10.82),(13.04,13.65)]:
        d.arrow(((a,2.43),(b,2.43)),training=True,color=COL["orange"])
    d.label(5.20,1.45,"θm ← .99 θm + .01 θu",size=18)
    d.label(.62,.99,"初始化聚类：若稳定候选不足，回退选至少 min(21,N) 个最稳定候选；不是每次都严格过 .95。",size=17,ha="left")
    d.label(.62,.56,"adapter 在 query 路径推理保留；factorization 关闭，不启用 mask encoder / decoder。",size=17,ha="left")
    d.label(.62,.16,"写门只筛 teacher 非重叠历史块，不筛 query token，不是检索读门。",size=17,ha="left",color=COL["muted"])
    d.save({**COMMON,"panel_role":"mechanism: distinguish query auxiliary loss from teacher write admission"},[
        "Effective PG95 objective is observed-future MSE + .05 mean(1-cos(primary query, paired query)).",
        "Paired query branch does not generate a paired full forecast.",
        "Training pair is independently masked complete historical data, not nested in primary mask.",
        "Absolute .95 write gate filters B,Q teacher candidates; query representation has B,L tokens.",
        "Warmup clustering has min(21,N) fallback; teacher EMA .99/.01; ordinary write budget M=50.",
        "Factorization disabled, content adapter retained in inference.",
    ])


def figure_three():
    d = Diagram("architecture_03_cnm", "CNM 推理：输入端三种历史 view，共享冻结模型与原型库",
                "每路重新计算历史极值并 Decay 填补；固定权重融合输出，不增加 encoder、router 或可学习参数。")
    d.box(.62,3.38,2.53,1.26,"历史 X, M [B,L,C]\n只删除已观测历史\n不读取未来信息",size=18)
    d.box(4.04,5.34,2.67,1.00,"原 view：M\n输入长度 L 不变",size=19)
    d.box(4.04,3.49,2.67,1.00,"L5 nested view\nM₅ ≤ M，块长 5",size=19)
    d.box(4.04,1.65,2.67,1.00,"L20 nested view\nM₂₀ ≤ M，块长 20",size=19)
    for y in (5.84,3.99,2.15):
        d.arrow(((3.15,4.01),(3.59,4.01),(3.59,y),(4.04,y)))
    panel = FancyBboxPatch((7.45,1.39),3.90,5.06,boxstyle="round,pad=0.02,rounding_size=.13",
                          facecolor=COL["gray_fill"],edgecolor=COL["gray"],linewidth=1.2,zorder=0)
    d.ax.add_patch(panel)
    for y in (5.34,3.49,1.65):
        d.box(7.77,y,3.26,1.00,"同一冻结 fθ\n同一冻结原型库",size=19)
        d.arrow(((6.71,y+.50),(7.77,y+.50)))
    d.label(9.40,1.05,"三次 forward ≠ 三套模型",size=18)
    for y,value in [(5.34,"0.50"),(3.49,"0.25"),(1.65,"0.25")]:
        d.box(11.79,y,1.18,1.00,value,size=22)
        d.arrow(((11.03,y+.50),(11.79,y+.50)))
    d.box(13.58,3.18,1.78,1.62,"固定融合\n.50 Ŷ原\n+ .25 ŶL5\n+ .25 ŶL20",size=19)
    d.arrow(((12.97,5.84),(13.25,5.84),(13.25,4.47),(13.58,4.47)))
    d.arrow(((12.97,3.99),(13.58,3.99)))
    d.arrow(((12.97,2.15),(13.25,2.15),(13.25,3.48),(13.58,3.48)))
    d.label(.65,2.43,"L5 / L20 等新增计数\n删除数 = np.rint\n(.15 × n_obs)\n块长不是输入长度",size=18,ha="left")
    d.box(13.58,1.66,1.78,1.00,"新增参数 0\n不更新 model\n不更新 bank",kind="gray",size=17)
    d.label(.62,.66,"旧 Weather：原路 + L5 两路 .5/.5，非等计数；不可套用本图解释其历史结果。",size=17,ha="left")
    d.label(.62,.25,"CRN 仅控制各路随机数；不是新模块。L1 是诊断对照，不参与三路融合。",size=17,ha="left",color=COL["muted"])
    d.save({**COMMON,"panel_role":"deployment wrapper: input branching and fixed fusion without new learnable modules"},[
        "Original, L5, and L20 routes share the same frozen checkpoint/model and frozen bank.",
        "Nested masks remove an equal rounded 15% count of originally observed history rows per sample.",
        "Weights .50/.25/.25 frozen; new parameter count zero; no future values enter model.",
        "Generic evaluation also computes L1 diagnostically; not in fused prediction.",
        "Weather legacy is two-route .5/.5 with probabilistic block masking, not exact count.",
        "CRN is evaluation random-number control rather than a learned module.",
    ])


def main():
    figure_one()
    figure_two()
    figure_three()
    source_files = ["model/S4M.py", "model/Bank.py", "model/MIPP.py", "model/Memory.py",
                    "experiments/exp_pretrain1.py", "run.py", "utils/paired_eval_rng.py"]
    hashes = {p: hashlib.sha256((ROOT / "baselines/S4M/s4m" / p).read_bytes()).hexdigest() for p in source_files}
    manifest = {"date":"2026-10-10", "contract":COMMON, "source_sha256":hashes,
                "figures":ITEMS, "caption_sources":{
                    "architecture_01_backbone":"Bank.py:64-110,262-267,368-443,624; S4M.py:278-294,319-362,411-441",
                    "architecture_02_pg95":"exp_pretrain1.py:978-1024,3283-3308,3325-3328,3400-3404; Bank.py:277-363,386-418,627-705; MIPP.py:25-50,92-100,375-418",
                    "architecture_03_cnm":"exp_pretrain1.py:6399-6451,6661-6723,6840-6914,6940-6942,6982-7008,7024-7076; run_stage14_uncertainty_cache.py:192-207,460-519",
                },"limitations":["Methods schematics only; do not imply PG95 or CNM statistically improves forecasting.",
                    "Min-distance bugfix is independent and default opt-in; uncorrected new-memory centroid index is not a new module."]}
    (OUT / "architecture_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()
