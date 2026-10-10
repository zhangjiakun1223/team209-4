# -*- coding: utf-8 -*-
"""viz_style.py —— 报告图表统一风格（中文字体、配色、导出规范）"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "figure.dpi": 110,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "axes.edgecolor": "#BFBFBF",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#E8E8E8",
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
})

C = {
    "normal": "#4C78A8",     # 合规经营
    "fake": "#E45756",       # 非规范行为
    "brush": "#F58518",      # 刷单
    "fab": "#B279A2",        # 虚构交易
    "flow": "#54A24B",       # 伪造流水
    "grey": "#8C8C8C",
    "accent": "#1F4E79",
    "warn": "#C00000",
}


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


MODE_GROUPS = [
    ("合规经营", "normal"),
    ("刷单", "brush"),
    ("虚构交易", "fab"),
    ("伪造流水", "flow"),
]