# -*- coding: utf-8 -*-
"""
05_validation.py
应用验证：授信额度情景模拟
- 基准情形：未部署识别模型，按「含虚假数据的经营流水」核定授信额度
- 策略 A：分数阈值（训练集 FPR≈10%）触发人工复核，复核后按真实经营数据核额
- 策略 B：排名制（固定 25% 复核比例）触发人工复核
- 输出：额度虚高规模、拦截效果、误伤情况、预期损失敞口下降、跨期稳定性和额度模拟图
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from viz_style import C, despine
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
FIG = os.path.join(BASE, "report", "figures")

LIMIT_FACTOR = 1.5        # 授信额度 = 月均经营流水 × 1.5
LIMIT_CAP = 1_000_000     # 单户额度上限 100 万元
PD, LGD = 0.08, 0.60      # 小微客群违约率与违约损失率假设（用于估算预期损失敞口）

# 复核后核额用的真实经营流水占比（复核通过者可完全按真实数据核定）
RECOVER_TRUE = 1.0


def main():
    pred = pd.read_csv(os.path.join(PROC, "predictions.csv"), parse_dates=["apply_date"])
    pred["limit_true"] = np.minimum(LIMIT_FACTOR * pred["true_window_gmv"] / 3, LIMIT_CAP)
    pred["limit_obs"] = np.minimum(LIMIT_FACTOR * pred["obs_window_gmv"] / 3, LIMIT_CAP)
    pred["limit_obs"] = pred[["limit_obs", "limit_true"]].max(axis=1)   # 核额取观测数据（含虚假数据）
    pred["excess"] = (pred["limit_obs"] - pred["limit_true"]).clip(lower=0)   # 额度虚高部分

    def simulate(flag_col):
        """按复核名单模拟：命中者按真实经营数据核额，未命中者按观测数据核额。"""
        flag = pred[flag_col].astype(bool)
        limit_after = np.where(flag, LIMIT_FACTOR * pred["true_window_gmv"] / 3 * RECOVER_TRUE, pred["limit_obs"])
        limit_after = np.minimum(limit_after, LIMIT_CAP)
        fake = pred["is_noncompliant"] == 1
        excess_before = float(pred.loc[fake, "excess"].sum())
        excess_after = float(np.clip(limit_after[fake.values] - pred.loc[fake, "limit_true"], 0, None).sum())
        compliant_limit_change = float(np.abs(limit_after[~fake.values] - pred.loc[~fake, "limit_obs"]).sum())
        return {
            "total_limit_before": round(float(pred["limit_obs"].sum()) / 1e4, 1),
            "total_limit_after": round(float(limit_after.sum()) / 1e4, 1),
            "review_share": round(float(flag.mean()), 4),
            "review_n": int(flag.sum()),
            "flagged_precision": round(float(pred.loc[flag.values, "is_noncompliant"].mean()), 4),
            "flagged_recall": round(float(flag[fake.values].mean()), 4),
            "excess_before_w": round(excess_before / 1e4, 1),
            "excess_after_w": round(excess_after / 1e4, 1),
            "excess_intercept_rate": round(float(1 - excess_after / excess_before), 4) if excess_before else 0.0,
            "compliant_limit_change_w": round(compliant_limit_change / 1e4, 1),
            "exposure_reduction_w": round((excess_before - excess_after) * PD * LGD / 1e4, 1),
        }

    res = {"baseline": {"total_limit_before": round(float(pred["limit_obs"].sum()) / 1e4, 1),
                        "excess_before_w": round(float(pred.loc[pred.is_noncompliant == 1, "excess"].sum()) / 1e4, 1),
                        "review_share": 1.0, "review_n": int(len(pred)), "excess_intercept_rate": 0.0,
                        "exposure_reduction_w": 0.0, "compliant_limit_change_w": 0.0}}
    res["policy_A_score_threshold"] = simulate("flag_score")
    res["policy_B_top25_rank"] = simulate("flag_rank")

    # 按额度分层的虚高集中度
    pred["limit_bin"] = pd.cut(pred["limit_obs"], [0, 5e4, 1e5, 3e5, 1e6],
                               labels=["5万以下", "5-10万", "10-30万", "30-100万"])
    layer = pred.groupby("limit_bin", observed=True).apply(
        lambda d: pd.Series({"商户数": len(d), "非规范商户数": int(d.is_noncompliant.sum()),
                             "虚高额度(万元)": round(float(d["excess"].sum()) / 1e4, 1)}), include_groups=False)

    # 跨期稳定性：测试期内各申请月份 AUC
    pred["apply_month"] = pred["apply_date"].dt.to_period("M").astype(str)
    by_month = pred.groupby("apply_month").apply(
        lambda d: pd.Series({"商户数": len(d),
                             "AUC": round(float(roc_auc_score(d.is_noncompliant, d.score_main)), 4)
                             if d.is_noncompliant.nunique() > 1 else np.nan}), include_groups=False)

    # ---- 图5：额度虚高（按行为模式）与识别策略效果 ----
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    vis = pred[pred.is_noncompliant == 1].copy()
    vis["excess_ratio"] = (vis["limit_obs"] / vis["limit_true"].replace(0, np.nan) - 1).clip(lower=0) * 100
    modes = [("刷单", C["brush"]), ("虚构交易", C["fab"]), ("伪造流水", C["flow"])]
    data, colors, labels = [], [], []
    for name, col in modes:
        sub = vis[vis.fraud_modes.fillna("").str.contains(name)]["excess_ratio"].dropna()
        data.append(sub.values); colors.append(col); labels.append(f"{name}\n(n={len(sub)})")
    bp = axes[0].boxplot(data, patch_artist=True, showfliers=False, widths=0.5,
                         medianprops=dict(color="#333333", lw=1.3))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.75); patch.set_edgecolor(col)
    axes[0].set_xticklabels(labels, fontsize=8.2)
    axes[0].set_ylabel("授信额度虚高幅度（%）")
    axes[0].set_title("(a) 不同行为模式造成的额度虚高", loc="left")
    for i, d in enumerate(data, start=1):
        axes[0].text(i, np.median(d) + 3, f"中位 {np.median(d):.0f}%", ha="center", fontsize=7.4, color=C["warn"])
    axes[0].set_ylim(0, min(max(np.percentile(d, 95) for d in data) * 1.35, 220))
    despine(axes[0])

    names = ["未识别\n（全量人工）", "策略A\n分数阈值", "策略B\n排名前25%"]
    keys = ["baseline", "policy_A_score_threshold", "policy_B_top25_rank"]
    real_part = [pred["limit_true"].sum() / 1e4] * 3
    excess_part = [res["baseline"]["excess_before_w"]] + [res[k]["excess_after_w"] for k in keys[1:]]
    x = np.arange(3)
    axes[1].bar(x, real_part, width=0.55, color=C["normal"], label="真实经营对应额度")
    axes[1].bar(x, excess_part, width=0.55, bottom=real_part, color=C["fake"], label="虚假行为造成的虚高额度")
    axes[1].text(0, real_part[0] + excess_part[0] * 0.5, f"{excess_part[0]:,.0f} 万元",
                 ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    axes[1].annotate(f"仅剩 {excess_part[1]:,.1f} 万元", xy=(1, real_part[1] + 26), xytext=(1.05, real_part[1] + 190),
                     fontsize=8, color=C["warn"], arrowprops=dict(arrowstyle="->", color=C["warn"], lw=0.9))
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, fontsize=8.2)
    axes[1].set_ylabel("授信额度合计（万元）")
    axes[1].set_ylim(0, pred["limit_true"].sum() / 1e4 * 1.22)
    axes[1].set_title("(b) 不同识别策略下的授信额度构成", loc="left")
    axes[1].legend(frameon=False, fontsize=7.6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    despine(axes[1])
    fig.tight_layout(h_pad=1.0)
    fig.savefig(os.path.join(FIG, "fig5_credit_simulation.png"))
    plt.close(fig)

    out = {"assumptions": {"limit_factor": LIMIT_FACTOR, "limit_cap": LIMIT_CAP, "PD": PD, "LGD": LGD},
           "test_sample": int(len(pred)), "results": res,
           "layer_table": layer.reset_index().to_dict("records"),
           "monthly_auc": by_month.reset_index().to_dict("records"),
           "median_excess_ratio_fake": round(float(vis["excess_ratio"].median() / 100), 4),
           "median_excess_ratio_visible_fake": round(float(
               vis.loc[vis.visible_fraud == 1, "excess_ratio"].median() / 100), 4)}
    json.dump(out, open(os.path.join(BASE, "report", "validation_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("\n额度分层:\n", layer.to_string())
    print("\n按月 AUC:\n", by_month.to_string())
    print(f"\n非规范商户额度虚高中位数: {vis['excess_ratio'].median():.1f}%"
          f"（窗口内可见者: {vis.loc[vis.visible_fraud == 1, 'excess_ratio'].median():.1f}%）")


if __name__ == "__main__":
    main()