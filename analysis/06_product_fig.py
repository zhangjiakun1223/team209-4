# -*- coding: utf-8 -*-
"""
06_product_fig.py
绘制产品化落地流程示意图（图6）：数据接入 → 特征计算 → 风险评分 → 分级处置 → 贷后闭环
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from viz_style import C

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "report", "figures")

STEPS = [
    ("数据接入", ["订单流水明细", "资金收付流水", "工商/司法/征信"], C["normal"]),
    ("特征计算", ["43 项量化指标", "客户结构·金额形态", "时间节律·资金往来"], C["normal"]),
    ("风险评分", ["逻辑回归评分卡", "LightGBM 交叉验证", "业务规则兜底仲裁"], C["accent"]),
    ("分级处置", ["低风险：自动通过", "中风险：加强审核", "高风险：复核降额"], C["brush"]),
    ("贷后闭环", ["额度动态调整", "用信与经营监控", "结论回流与迭代"], C["flow"]),
]


def main():
    fig, ax = plt.subplots(figsize=(7.6, 2.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")
    w, gap = 17.4, 3.2
    for i, (title, lines, color) in enumerate(STEPS):
        x = 1 + i * (w + gap)
        box = FancyBboxPatch((x, 6), w, 30, boxstyle="round,pad=0.6,rounding_size=1.6",
                             linewidth=1.2, edgecolor=color, facecolor=color + "14")
        ax.add_patch(box)
        ax.text(x + w / 2, 31.5, title, ha="center", va="center", fontsize=9.6,
                color=color, fontweight="bold")
        for j, t in enumerate(lines):
            ax.text(x + 1.3, 24.5 - j * 6.2, "· " + t, ha="left", va="center", fontsize=7.2, color="#333333")
        if i < len(STEPS) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.4, 21), (x + w + gap - 0.5, 21),
                                         arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color="#9A9A9A"))
    ax.text(50, 2.2, "评分与处置结果同步输出至审批页面，并保留可解释依据（特征贡献、同类商户对比）",
            ha="center", va="center", fontsize=7.8, color="#666666")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_product_flow.png"))
    plt.close(fig)
    print("[saved] fig6_product_flow.png")


if __name__ == "__main__":
    main()