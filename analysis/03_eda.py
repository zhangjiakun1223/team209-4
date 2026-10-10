# -*- coding: utf-8 -*-
"""
03_eda.py
描述性分析：
1) 真实公开数据的分布校准结果（图 1）
2) 合规经营与三类非规范行为的量化特征画像（图 2）
3) 关键统计结果输出至 report/eda_stats.json，供报告撰写引用
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from viz_style import C, despine, MODE_GROUPS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
FIG = os.path.join(BASE, "report", "figures")
os.makedirs(FIG, exist_ok=True)

STATS = json.load(open(os.path.join(PROC, "official_baseline.json"), encoding="utf-8"))


def fig1_real_calibration():
    """图 1：官方口径基线（2026）——月度零售节律与笔均金额锚点。"""
    monthly = STATS["nbs_retail_monthly_亿元"]
    season = STATS["season_factor_by_calendar_month"]
    anchors = STATS["industry_anchors"]
    pbc = STATS["pbc_payment"]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))

    # (a) 月度社零序列与季节因子
    ax = axes[0]
    keys = list(monthly.keys())
    vals = [monthly[k] / 10000 for k in keys]
    bars = ax.bar(range(len(keys)), vals, color=C["normal"], width=0.68)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}", ha="center", fontsize=6.4, color="#333333")
    mean_v = STATS["nbs_retail_mean_亿元"] / 10000
    ax.axhline(mean_v, ls="--", lw=0.9, color=C["fake"], label=f"12个月均值 {mean_v:.2f} 万亿")
    ax.legend(frameon=False, fontsize=7.2, loc="upper right", handlelength=1.6)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k[2:].replace("-", "/") for k in keys], rotation=60, fontsize=6.6)
    ax.set_ylabel("社会消费品零售总额（万亿元）")
    ax.set_title("(a) 官方月度零售节律（国家统计局，2025/09—2026/08）", loc="left", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.20)
    ax.text(0.02, 0.98, f"季节因子区间 {min(season.values()):.2f}—{max(season.values()):.2f}",
            transform=ax.transAxes, fontsize=7.4, va="top")
    despine(ax)

    # (b) 笔均金额锚点（央行支付体系 + 行业派生）
    ax = axes[1]
    labels = ["餐饮服务", "生活服务", "电商零售", "批发供货"]
    means = [anchors[k]["mean_amt"] for k in labels]
    colors = [C["flow"], C["fab"], C["brush"], C["normal"]]
    bars = ax.bar(range(len(labels)), means, color=colors, width=0.6)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + max(means) * 0.03, f"{v:,.0f} 元",
                ha="center", fontsize=7.8, color="#333333")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{k}\n(官方增速 {anchors[k]['growth']*100:.1f}%)" for k in labels], fontsize=7.6)
    ax.set_ylabel("订单笔均金额（元）")
    ax.set_title("(b) 笔均金额与自然增速锚点（央行2026Q2 / 统计局）", loc="left", fontsize=9)
    ax.set_ylim(0, max(means) * 1.22)
    ax.text(0.02, 0.98, f"银行卡消费笔均 {pbc['card_consumption']['笔均_元']:.0f} 元｜"
                        f"非银网络支付笔均 {pbc['nonbank_network_payment']['笔均_元']:.0f} 元",
            transform=ax.transAxes, fontsize=7.0, va="top")
    despine(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_official_baseline.png"))
    plt.close(fig)


def mode_label(row):
    if row["is_noncompliant"] == 0:
        return "合规经营" if row["hard_negative"] == 0 else None
    return row["fraud_modes"].split(";")[0] if row["n_fraud_modes"] == 1 else "组合模式"


def fig2_pattern_profile(feat: pd.DataFrame):
    """图 2：合规经营与三类非规范行为的量化特征画像（小倍数图，取窗口期中位数）。"""
    metrics = [
        ("night_share", "凌晨订单占比", 100.0, "%"),
        ("burst_share", "秒级批量订单占比", 100.0, "%"),
        ("dup_amt_share", "重复金额订单占比", 100.0, "%"),
        ("round1000_share", "千元整数金额占比", 100.0, "%"),
        ("big_round_count", "整数大额订单笔数", 1.0, "笔"),
        ("flow_to_gmv_ratio", "进账/订单收入比", 1.0, "倍"),
        ("same_day_pair_rate", "当日进出配对率", 100.0, "%"),
        ("win_gmv_share_of_year", "窗口GMV占全年比重", 100.0, "%"),
    ]
    groups = [("合规经营", feat[(feat.is_noncompliant == 0) & (feat.hard_negative == 0)], "normal"),
              ("刷单", feat[feat.fraud_modes.fillna("").str.contains("刷单")], "brush"),
              ("虚构交易", feat[feat.fraud_modes.fillna("").str.contains("虚构交易")], "fab"),
              ("伪造流水", feat[feat.fraud_modes.fillna("").str.contains("伪造流水")], "flow")]

    fig, axes = plt.subplots(2, 4, figsize=(7.6, 3.5))
    for ax, (col, name, scale, unit) in zip(axes.ravel(), metrics):
        vals = [g[1][col].median() * scale for g in groups]
        bars = ax.bar(range(len(groups)), vals, color=[C[g[2]] for g in groups], width=0.62)
        ax.set_title(name, fontsize=9)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(["合规", "刷单", "虚构", "伪造"], fontsize=8.2)
        top = max(vals) if max(vals) > 0 else 1
        ax.set_ylim(0, top * 1.30)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.05, f"{v:.1f}",
                    ha="center", fontsize=7.4, color="#333333")
        despine(ax)
    fig.tight_layout(h_pad=1.4, w_pad=1.0)
    fig.savefig(os.path.join(FIG, "fig2_pattern_profile.png"))
    plt.close(fig)


def univariate_auc(x: np.ndarray, y: np.ndarray) -> float:
    """单特征 AUC（秩统计）。"""
    r = pd.Series(x).rank().values
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    feat = pd.read_csv(os.path.join(PROC, "features.csv"))
    tx = pd.read_csv(os.path.join(PROC, "transactions.csv.gz"), usecols=["merchant_id", "is_fake"])
    fl = pd.read_csv(os.path.join(PROC, "fund_flows.csv.gz"), usecols=["merchant_id", "is_fake"])
    mc = pd.read_csv(os.path.join(PROC, "merchants.csv"))
    tx_n, fl_n = len(tx), len(fl)

    feat_cols = [c for c in feat.columns if c not in mc.columns and c != "merchant_id"]
    y = feat["is_noncompliant"].values
    uni = {c: univariate_auc(feat[c].fillna(feat[c].median()).values, y) for c in feat_cols}
    uni_s = pd.Series(uni).dropna()
    uni_s = pd.Series(np.where(uni_s < 0.5, 1 - uni_s, uni_s), index=uni_s.index).sort_values(ascending=False)

    groups = {
        "合规经营": feat[(feat.is_noncompliant == 0) & (feat.hard_negative == 0)],
        "刷单": feat[feat.fraud_modes.fillna("").str.contains("刷单")],
        "虚构交易": feat[feat.fraud_modes.fillna("").str.contains("虚构交易")],
        "伪造流水": feat[feat.fraud_modes.fillna("").str.contains("伪造流水")],
        "非规范合计": feat[feat.is_noncompliant == 1],
    }
    show = ["night_share", "burst_share", "top5_cust_share", "cust_hhi", "dup_amt_share", "round1000_share",
            "big_round_count", "max_order_amount", "big_order_share", "oneshot_new_share", "new_cust_big_share",
            "flow_to_gmv_ratio", "same_day_pair_rate", "win_gmv_share_of_year", "order_growth", "win_gmv", "win_orders"]
    median_tbl = {k: {c: round(float(v[c].median()), 4) for c in show} for k, v in groups.items()}

    fake_vis = feat[(feat.is_noncompliant == 1) & (feat.visible_fraud == 1)]
    out = {
        "scale": {
            "merchants": int(len(feat)), "transactions": int(tx_n), "fund_flows": int(fl_n),
            "train": int((feat.data_split == "train").sum()), "test": int((feat.data_split == "test").sum()),
            "noncompliant": int(feat.is_noncompliant.sum()),
            "noncompliant_ratio": round(float(feat.is_noncompliant.mean()), 4),
            "visible_fraud": int(feat.loc[feat.is_noncompliant == 1, "visible_fraud"].sum()),
            "invisible_fraud": int(feat.loc[feat.is_noncompliant == 1, "visible_fraud"].eq(0).sum()),
            "invisibility_rate": round(float(feat.loc[feat.is_noncompliant == 1, "visible_fraud"].eq(0).mean()), 4),
            "fake_order_ratio": round(float(tx.is_fake.mean()), 4),
            "fake_flow_ratio": round(float(fl.is_fake.mean()), 4),
            "hard_negative": int(feat.hard_negative.sum()),
        },
        "inflation": {   # 虚增幅度仅对「窗口内可见」的非规范商户统计（其余行为不可观测）
            "visible_fraud_n": int(len(fake_vis)),
            "median_gmv_inflation_fake": round(float(fake_vis.gmv_inflation.median()), 4),
            "p25": round(float(fake_vis.gmv_inflation.quantile(.25)), 4),
            "p75": round(float(fake_vis.gmv_inflation.quantile(.75)), 4),
            "median_order_inflation_fake": round(float(fake_vis.order_inflation.median()), 4),
            "by_mode": {k: round(float(fake_vis[fake_vis.fraud_modes.fillna("").str.contains(k)].gmv_inflation.median()), 4)
                        for k in ["刷单", "虚构交易", "伪造流水"]},
            "by_intensity": {k: round(float(fake_vis[fake_vis.fraud_intensity == k].gmv_inflation.median()), 4)
                             for k in ["低", "中", "高"]},
        },
        "mode_counts": {k: int(feat.fraud_modes.fillna("").str.contains(k).sum())
                        for k in ["刷单", "虚构交易", "伪造流水"]},
        "median_by_group": median_tbl,
        "top_univariate": {k: round(float(v), 4) for k, v in uni_s.head(15).items()},
        "hard_negative_gmv": {"合规普通": round(float(groups["合规经营"].win_gmv.median()), 2),
                              "困难负样本": round(float(feat[feat.hard_negative == 1].win_gmv.median()), 2)},
    }
    json.dump(out, open(os.path.join(BASE, "report", "eda_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    fig1_real_calibration()
    fig2_pattern_profile(feat)
    print(json.dumps({k: out[k] for k in ["scale", "inflation", "mode_counts"]}, ensure_ascii=False, indent=2))
    print("单特征 AUC Top10:", json.dumps({k: round(v, 3) for k, v in list(out["top_univariate"].items())[:10]}, ensure_ascii=False))


if __name__ == "__main__":
    main()