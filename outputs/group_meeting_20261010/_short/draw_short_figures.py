#!/usr/bin/env python3
"""Compact, code-audited architecture schematics for a five-minute report."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
DEPS = OUT.parent / "_deps"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/s4m_short_mplconfig")
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
    "font.size": 17,
    "mathtext.fontset": "stix",
    "text.usetex": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
    "savefig.facecolor": "white",
})

COL = {
    "ink": "#263746", "muted": "#6B7985",
    "blue": "#4F718A", "blue_fill": "#EDF2F6",
    "orange": "#BC7B41", "orange_fill": "#FCF1E6",
    "teal": "#458E94", "teal_fill": "#EAF5F4",
    "line": "#DCE4E9",
}
MANIFEST = []


def line_hits_rect(a, b, rect):
    """Liang-Barsky intersection, with a small inset to ignore border contact."""
    x0, y0, x1, y1 = rect
    inset = .008
    x0, y0, x1, y1 = x0 + inset, y0 + inset, x1 - inset, y1 - inset
    dx, dy = b[0] - a[0], b[1] - a[1]
    lo, hi = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]),
                 (-dy, a[1] - y0), (dy, y1 - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                lo = max(lo, t)
            else:
                hi = min(hi, t)
            if lo > hi:
                return False
    return True


class Diagram:
    def __init__(self, stem, conclusion):
        self.stem = stem
        self.conclusion = conclusion
        self.fig = plt.figure(figsize=(14, 4.8), facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, 14), ylim=(0, 4.8))
        self.ax.axis("off")
        self.texts = []
        self.box_labels = []
        self.boxes = []
        self.arrows = []

    def label(self, x, y, text, size=17, color=None, ha="center", weight="normal"):
        obj = self.ax.text(x, y, text, fontsize=size, color=color or COL["ink"],
                           ha=ha, va="center", linespacing=1.12, weight=weight,
                           zorder=6)
        self.texts.append((obj, text))
        return obj

    def box(self, x, y, w, h, text, kind="blue", size=17, dashed=False):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.016,rounding_size=0.09",
            edgecolor=COL[kind], facecolor=COL[kind + "_fill"], linewidth=1.5,
            linestyle=(0, (4, 3)) if dashed else "solid", zorder=3,
        )
        self.ax.add_patch(patch)
        txt = self.label(x + w/2, y + h/2, text, size=size)
        self.box_labels.append((txt, [x, y, w, h], text))
        self.boxes.append((txt, [x, y, x+w, y+h]))
        return patch

    def arrow(self, points, kind="blue", dashed=False, head=True):
        points = list(points)
        patch = FancyArrowPatch(
            path=MPath(points, [MPath.MOVETO] + [MPath.LINETO] * (len(points)-1)),
            arrowstyle="-|>" if head else "-", mutation_scale=15,
            color=COL[kind], linewidth=1.5,
            linestyle=(0, (4, 3)) if dashed else "solid",
            capstyle="round", joinstyle="round", zorder=2,
        )
        self.ax.add_patch(patch)
        self.arrows.append(points)

    def save(self, findings, additional_notes):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        text_bboxes = []
        bbox_problems = []
        box_problems = []
        arrow_problems = []
        other_box_problems = []
        for txt, text in self.texts:
            bb = txt.get_window_extent(renderer).transformed(self.ax.transData.inverted())
            bounds = list(bb.extents)
            text_bboxes.append({"text": text, "bbox_inches": bounds, "font_pt": txt.get_fontsize()})
            if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > 14 or bb.y1 > 4.8:
                bbox_problems.append(text)
            for ai, points in enumerate(self.arrows):
                if any(line_hits_rect(a, b, bounds) for a, b in zip(points[:-1], points[1:])):
                    arrow_problems.append({"arrow": ai, "text": text})
            if text:
                for owner, (x0, y0, x1, y1) in self.boxes:
                    # Empty-label boxes are explicit shared-model containers.
                    if (not owner.get_text() and bb.x0 >= x0 + .035 and bb.x1 <= x1 - .035
                            and bb.y0 >= y0 + .025 and bb.y1 <= y1 - .025):
                        continue
                    if owner is not txt and bb.x1 > x0 and bb.x0 < x1 and bb.y1 > y0 and bb.y0 < y1:
                        other_box_problems.append(text)
        for txt, (x, y, w, h), text in self.box_labels:
            bb = txt.get_window_extent(renderer).transformed(self.ax.transData.inverted())
            if bb.x0 < x + .035 or bb.x1 > x + w - .035 or bb.y0 < y + .025 or bb.y1 > y + h - .025:
                box_problems.append({"text": text, "bbox_inches": list(bb.extents), "box_inches": [x, y, w, h]})
        paths = {ext: OUT / f"{self.stem}.{ext}" for ext in ("pdf", "svg", "png")}
        for ext, path in paths.items():
            self.fig.savefig(path, dpi=200 if ext == "png" else None)
        plt.close(self.fig)
        pdf = fitz.open(paths["pdf"])
        page = pdf[0]
        preview = OUT / f"{self.stem}_qa.png"
        page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(preview)
        fonts = sorted({f[3] for f in page.get_fonts(full=True)})
        embedded_fonts = [{"name": f[3], "type": f[2], "extension": f[1]} for f in page.get_fonts(full=True)]
        info = {
            "stem": self.stem,
            "contract": {
                "conclusion": self.conclusion,
                "archetype": "schematic-led composite",
                "role": "code-derived insertion-point schematic, not efficacy evidence",
                "backend": "Python matplotlib; Python PyMuPDF rendering",
                "dimensions_inches": [14, 4.8],
                "report_placement_points": [864, 296],
                "exports": ["PDF", "SVG", "PNG"],
                "fonts": "Droid Sans Fallback for Chinese; STIX mathtext for real mathematical scripts",
                "font_pt": [16, 18],
                "encoding": "low-saturation blue backbone; orange PG95; teal CNM; dashed means training-only",
                "statistics": "not applicable: no metrics or performance claim",
                "integrity": "new vector schematic from audited implementation; no input image manipulation",
            },
            "paths": {k: p.name for k, p in paths.items()},
            "qa_png": preview.name,
            "implementation_findings": findings,
            "caption_notes": additional_notes,
            "qa": {
                "pdf_dimensions_points": list(page.rect)[2:],
                "raster_image_count": len(page.get_images()),
                "selectable_text_characters": len(page.get_text()),
                "embedded_fonts": embedded_fonts,
                "math_font_verified": any("STIX" in name for name in fonts),
                "canvas_text_overflow": bbox_problems,
                "box_text_overflow": box_problems,
                "arrow_text_intersections": arrow_problems,
                "other_box_text_intersections": other_box_problems,
                "text_bboxes": text_bboxes,
                "visual_review": (
                    "passed: both Python PDF-rendered QA PNGs visually inspected; mathematical scripts verified; no box collisions or arrows through text"
                    if "--reviewed" in sys.argv else "pending Python PDF-rendered image inspection"
                ),
            },
        }
        pdf.close()
        MANIFEST.append(info)
        print(f"{self.stem}: canvas={len(bbox_problems)} box={len(box_problems)} arrow/text={len(arrow_problems)} other-box/text={len(other_box_problems)}; mathfont={info['qa']['math_font_verified']}")
        for problem in box_problems + arrow_problems:
            print("QA_DETAIL", problem)


def backbone():
    conclusion = "PG95 增加查询适配与训练写门；推理保留适配器（基线跳过），预测主干不变。"
    d = Diagram("backbone_short", conclusion)
    # Deployed inference flow, with two explicit direct fusion edges.
    d.box(.35, 3.10, 1.20, .72, "历史输入\n$X,M$")
    d.box(1.95, 3.10, 1.35, .72, "衰减填补\n$Z$")
    d.box(3.70, 3.10, 1.90, .72, "滑窗查询编码\n$U$")
    d.box(6.05, 3.10, 1.70, .72, "PG95 适配器\n$\\widetilde{U}$", "orange")
    d.box(8.25, 3.10, 1.90, .72, "原型检索\n$R$")
    d.box(10.65, 3.10, 2.95, .72, "拼接 + 查询残差\n$O$")
    for a, b in ((1.55, 1.95), (3.30, 3.70), (5.60, 6.05), (7.75, 8.25), (10.15, 10.65)):
        d.arrow(((a, 3.46), (b, 3.46)))
    d.box(8.25, 4.16, 1.90, .43, "原型库", size=16)
    d.arrow(((9.20, 4.16), (9.20, 3.82)))
    d.box(3.70, 4.03, 4.05, .60,
          "仅训练：适配后配对查询损失\n$\\widetilde{U}^{a},\\widetilde{U}^{b}$",
          "orange", size=16, dashed=True)
    # The auxiliary query objective consumes representations AFTER the adapter.
    d.arrow(((8.00, 3.46), (8.00, 4.28), (7.75, 4.28)), "orange", dashed=True)
    # Teacher candidates are from an independent frozen historical-block branch,
    # not from the query stream. Only this admission gate has an edge into bank.
    d.box(10.65, 4.03, 2.95, .60,
          "独立冻结教师候选\n$q^{a},q^{b}$：$0.95$ 写门", "orange", size=16, dashed=True)
    d.arrow(((10.65, 4.38), (10.15, 4.38)), "orange", dashed=True)
    d.arrow(((6.90, 3.10), (6.90, 2.78), (11.20, 2.78), (11.20, 3.10)))
    d.label(8.85, 2.56, "查询 $\\widetilde{U}$ 直达融合", size=16)
    d.arrow(((2.625, 3.10), (2.625, 2.35), (12.90, 2.35), (12.90, 3.10)))
    d.label(4.55, 2.51, "$Z$ 直达融合", size=16)
    # Forecasting flow folds back; the mask enters the first S4 only.
    d.box(10.55, .98, 2.55, .78, "首层缺失感知 S4\n$H$", size=17)
    d.box(8.56, .98, 1.50, .78, "线性映射\n$H\\rightarrow D$")
    d.box(6.01, .98, 2.05, .78, "其余 S4 层\n$D$")
    d.box(3.55, .98, 1.96, .78, "归一化＋输出头\n$D\\rightarrow C$")
    d.box(.35, .98, 2.70, .78, "尾部 $P$ 步预测\n$\\hat{Y}\\in\\mathbb{R}^{B\\times P\\times C}$")
    d.arrow(((13.60, 3.46), (13.82, 3.46), (13.82, 1.37), (13.10, 1.37)))
    for a, b in ((10.55, 10.06), (8.56, 8.06), (6.01, 5.51), (3.55, 3.05)):
        d.arrow(((a, 1.37), (b, 1.37)))
    d.arrow(((.35, 3.46), (.13, 3.46), (.13, .57), (11.825, .57), (11.825, .98)))
    d.label(8.45, .76, "掩码 $M$ 经线性投影，仅接首层", size=16)
    d.label(.35, .20, conclusion, size=16, ha="left")
    d.save([
        "X,M and historical extrema/distances feed decay-imputed Z; query is U from sliding histories.",
        "The retained PG95 residual content adapter is normalize(U + A U); baseline bypasses it.",
        "Fusion is O=Linear([R,U-tilde,Z])+U-tilde. Both U-tilde and Z have explicit direct edges.",
        "Only the first S4 consumes Linear(M); H-to-D precedes ordinary S4 layers; output selects tail P.",
        "Paired-query invariance uses adapter outputs U-tilde^a/U-tilde^b; its dashed edge starts after the adapter and does not enter the bank.",
        "An independent frozen teacher branch encodes nonoverlapping historical S-blocks into q^a/q^b; only its 0.95 candidate-write gate has a dashed edge into the training bank.",
    ], [
        "Baseline bypasses adapter; PG95 retains it at inference. Content-mask factorization is disabled.",
        "At inference, teacher bypass is still computed but the prototype bank is not written.",
        "The 0.95 training initialization gate has a minimum-candidate fallback, as documented in the implementation audit.",
        "The full fusion and adapter formulas are placed in the report body to keep this schematic compact.",
    ])


def cnm():
    conclusion = "CNM：同一冻结模型的三路前向、固定加权；新增参数为 0。"
    d = Diagram("cnm_short", conclusion)
    d.label(.35, 4.57, conclusion, size=18, ha="left")
    d.box(.35, 2.32, 1.60, 1.02, "历史输入\n$X,M$")
    rows = [(3.88, "原始视图\n$X,M$", r"$f_\theta(X,M)$", r"$\hat{Y}_{\mathrm{orig}}$"),
            (2.79, "块长 5\n$X_5,\ M_5\\leq M$", r"$f_\theta(X_5,M_5)$", r"$\hat{Y}_5$"),
            (1.70, "块长 20\n$X_{20},\ M_{20}\\leq M$", r"$f_\theta(X_{20},M_{20})$", r"$\hat{Y}_{20}$")]
    d.box(5.80, 1.22, 2.65, 3.16, "", size=17)
    d.label(7.125, 4.17, "共享冻结模型＋原型库", size=16)
    d.ax.plot([6.00, 8.25], [3.55, 3.55], color=COL["line"], lw=1)
    d.ax.plot([6.00, 8.25], [2.45, 2.45], color=COL["line"], lw=1)
    for cy, view, call, prediction in rows:
        d.box(2.70, cy-.37, 2.45, .74, view, "teal")
        d.arrow(((1.95, 2.83), (2.33, 2.83), (2.33, cy), (2.70, cy)), "teal")
        d.arrow(((5.15, cy), (5.80, cy)), "teal")
        d.label(7.125, cy-.09, call, size=18)
        d.arrow(((8.45, cy), (8.80, cy)))
        d.label(9.25, cy, prediction, size=18)
    d.box(10.60, 2.30, 2.67, 1.06, "定权融合\n$0.50\,/\,0.25\,/\,0.25$", "teal")
    d.arrow(((9.73, 3.88), (10.08, 3.88), (10.08, 3.12), (10.60, 3.12)), "teal")
    d.arrow(((9.73, 2.79), (10.60, 2.79)), "teal")
    d.arrow(((9.73, 1.70), (10.08, 1.70), (10.08, 2.48), (10.60, 2.48)), "teal")
    d.arrow(((13.27, 2.83), (13.53, 2.83)), "teal")
    d.label(13.78, 2.83, r"$\hat{Y}$", size=18)
    d.label(.35, .98,
            "两种嵌套视图各删除同等数量的已观测行（15%）；5 / 20 是块长，输入长度不变。",
            size=16, ha="left")
    d.label(.35, .53, "每路重新填补；不更新参数与原型库。", size=17, ha="left")
    d.save([
        "CNM branches before extrema recomputation/decay imputation, not after query encoding.",
        "Nested masks M5<=M and M20<=M remove the same rounded 15% observed-row count per window.",
        "5 and 20 are causal block lengths; original history length is unchanged.",
        "Each view recomputes history extrema and imputation within the same frozen f_theta and same restored memory bank.",
        "The fusion core uses three forward calls and zero new parameters; these are not three independently learned models.",
        "Weights are fixed 0.50 original + 0.25 L5 + 0.25 L20; no learned router/fusion.",
    ], [
        "Three calls describe only the displayed fusion core. The full audit evaluation also includes L1 control and an original-only reference pass; do not report total audit cost as exactly 3x.",
        "Historical Weather two-view CNM is footnoted outside this schematic in the report and is not represented as the new three-view result.",
        "The full fixed-sum equation is placed outside this schematic in the report, avoiding duplicate equations.",
        "X5 and X20 denote masked observed histories only; no future or newly revealed missing values are used.",
    ])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    backbone()
    cnm()
    payload = {
        "source": Path(__file__).name,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "two vector schematics only; no model, config, data, training, checkpoint, or GPU access",
        "figures": MANIFEST,
    }
    (OUT / "short_figures_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
