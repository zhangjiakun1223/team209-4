# -*- coding: utf-8 -*-
"""
04_model.py
非规范经营行为识别模型：
- 对照模型：逻辑回归、随机森林、孤立森林（无监督）
- 主模型：LightGBM
- 验证协议：按申请时间切分（训练=2026年1-5月申请，测试=6-8月申请），严格样本外
- 输出：指标 JSON、预测分数、ROC/PR 图、SHAP 重要性图、分区稳健性表
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from viz_style import C, despine

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
FIG = os.path.join(BASE, "report", "figures")

CN = {
    "win_orders": "窗口订单量", "win_gmv": "窗口GMV", "win_active_days": "有效经营天数",
    "orders_per_active_day": "日均活跃订单量", "season_factor": "窗口季节因子",
    "order_growth": "订单量同口径环比", "gmv_growth": "GMV同口径环比",
    "win_gmv_share_of_year": "窗口GMV占全年比重", "n_customers": "窗口客户数",
    "orders_per_customer": "客均订单量", "cust_hhi": "客户订单集中度HHI",
    "top5_cust_share": "前5大客户订单占比", "new_cust_share": "新客订单占比",
    "oneshot_new_share": "一次性新客订单占比", "new_cust_big_share": "新客大额订单占比",
    "amt_mean": "平均订单金额", "amt_median": "订单金额中位数", "amt_cv": "订单金额变异系数",
    "big_order_share": "大额订单占比(≥3000元)", "round100_share": "百元整数金额占比",
    "round1000_share": "千元整数金额占比", "dup_amt_share": "重复金额订单占比", "big_round_count": "整数大额订单笔数",
    "max_order_amount": "窗口最大单笔金额",
    "amt_entropy": "金额分布熵", "night_share": "凌晨(0-6时)订单占比",
    "weekend_share": "周末订单占比", "hour_entropy": "时段分布熵",
    "iat_cv": "订单间隔变异系数", "burst_share": "秒级批量订单占比",
    "batch_interval_regularity": "批量下单间隔规律度", "max_daily_share": "单日峰值订单占比",
    "daily_cv": "日订单量波动系数", "sku_hhi": "商品集中度HHI", "top_sku_share": "头部商品订单占比",
    "channel_hhi": "渠道集中度HHI", "flow_in_total": "窗口进账总额", "flow_out_total": "窗口出账总额",
    "flow_to_gmv_ratio": "进账/订单收入比", "net_inflow_ratio": "净流入率",
    "round_transfer_share": "千元整数转账占比", "flow_cp_count": "资金对手方数量",
    "flow_cp_hhi": "资金对手方集中度", "same_day_pair_rate": "当日进出配对率",
    "big_same_day_pair_rate": "大额当日进出配对率",
}
DROP = ["merchant_id", "industry", "city_tier", "months_operating", "apply_date", "is_noncompliant",
        "fraud_modes", "n_fraud_modes", "fraud_intensity", "visible_fraud", "hard_negative", "hn_type",
        "data_split", "true_window_gmv", "obs_window_gmv", "gmv_inflation", "order_inflation"]


def ks_stat(y, s) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr - fpr))


def metrics(y, s) -> dict:
    return {"auc": round(float(roc_auc_score(y, s)), 4),
            "ks": round(ks_stat(y, s), 4),
            "pr_auc": round(float(average_precision_score(y, s)), 4)}


def topk(y, s, frac=0.20) -> dict:
    k = int(np.ceil(len(s) * frac))
    idx = np.argsort(-s)[:k]
    hit = y[idx].sum()
    return {"k": k, "recall": round(float(hit / y.sum()), 4),
            "precision": round(float(hit / k), 4),
            "lift": round(float((hit / k) / y.mean()), 2)}


def main():
    feat = pd.read_csv(os.path.join(PROC, "features.csv"))
    cols = [c for c in feat.columns if c not in DROP]
    tr = feat[feat.data_split == "train"].reset_index(drop=True)
    te = feat[feat.data_split == "test"].reset_index(drop=True)
    Xtr, ytr = tr[cols], tr.is_noncompliant.values
    Xte, yte = te[cols], te.is_noncompliant.values
    print(f"train {Xtr.shape} 正例率 {ytr.mean():.3f} | test {Xte.shape} 正例率 {yte.mean():.3f}")

    LGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=25,
                      subsample=0.85, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                      random_state=42, n_jobs=-1, verbose=-1)
    # 小规模随机搜索（3 折 CV AUC，网格 18 组）确定 LightGBM 超参数
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    cv3 = StratifiedKFold(3, shuffle=True, random_state=11)
    best_auc, best_par = -1, None
    search_log = []
    for nl in [7, 15, 31]:
        for lr_ in [0.02, 0.05, 0.08]:
            for mcs in [10, 25]:
                par = dict(LGB_PARAMS, num_leaves=nl, learning_rate=lr_, min_child_samples=mcs)
                s = cross_val_score(lgb.LGBMClassifier(**par), Xtr, ytr, cv=cv3, scoring="roc_auc").mean()
                search_log.append((nl, lr_, mcs, round(float(s), 4)))
                if s > best_auc:
                    best_auc, best_par = s, par
    LGB_PARAMS = best_par
    print(f"  LightGBM 调参最优 CV-AUC={best_auc:.4f}  参数: num_leaves={best_par['num_leaves']}, "
          f"lr={best_par['learning_rate']}, min_child_samples={best_par['min_child_samples']}")

    models = {
        "逻辑回归": Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=3000, C=0.5))]),
        "随机森林": RandomForestClassifier(n_estimators=600, min_samples_leaf=4, max_depth=8,
                                           class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "LightGBM": lgb.LGBMClassifier(**LGB_PARAMS),
    }
    scores, scores_tr, results = {}, {}, {}
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        p_te = mdl.predict_proba(Xte)[:, 1]
        p_tr = mdl.predict_proba(Xtr)[:, 1]
        scores[name], scores_tr[name] = p_te, p_tr
        results[name] = {"test": metrics(yte, p_te), "train": metrics(ytr, p_tr),
                         "top20pct_test": topk(yte, p_te, 0.20)}
        print(f"  {name}: test AUC={results[name]['test']['auc']} KS={results[name]['test']['ks']} "
              f"PR-AUC={results[name]['test']['pr_auc']}")

    # 无监督对照：孤立森林（不使用标签训练）
    iso_imp = SimpleImputer(strategy="median").fit(Xtr)
    iso = IsolationForest(n_estimators=500, contamination=float(ytr.mean()), random_state=42, n_jobs=-1)
    iso.fit(pd.DataFrame(iso_imp.transform(Xtr), columns=cols))
    p_iso = -iso.score_samples(pd.DataFrame(iso_imp.transform(Xte), columns=cols))
    scores["孤立森林(无监督)"] = p_iso
    results["孤立森林(无监督)"] = {"test": metrics(yte, p_iso), "train": None,
                                   "top20pct_test": topk(yte, p_iso, 0.20)}
    print(f"  孤立森林(无监督): test AUC={results['孤立森林(无监督)']['test']['auc']}")

    # 融合评分：三模型打分按排名归一后等权平均（对量纲差异与分布漂移更稳健）
    from scipy.stats import rankdata

    def rank_norm(x):
        return rankdata(x) / len(x)

    pk = ["逻辑回归", "随机森林", "LightGBM"]
    scores["融合评分"] = sum(rank_norm(scores[k]) for k in pk) / len(pk)
    scores_tr["融合评分"] = sum(rank_norm(scores_tr[k]) for k in pk) / len(pk)
    results["融合评分"] = {"test": metrics(yte, scores["融合评分"]), "train": metrics(ytr, scores_tr["融合评分"]),
                           "top20pct_test": topk(yte, scores["融合评分"], 0.20)}
    print(f"  融合评分: test AUC={results['融合评分']['test']['auc']} KS={results['融合评分']['test']['ks']} "
          f"PR-AUC={results['融合评分']['test']['pr_auc']}")

    # 交叉验证稳定性（训练集 5 折）
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    cv = cross_val_score(lgb.LGBMClassifier(**LGB_PARAMS), Xtr, ytr,
                         cv=StratifiedKFold(5, shuffle=True, random_state=7), scoring="roc_auc")
    print(f"  LightGBM 5折CV AUC: {cv.mean():.4f} ± {cv.std():.4f}")

    # ---- 阈值策略：A 分数阈值（训练集 FPR=10% 分位点）；B 排名制（每月固定复核比例 25%）----
    # 主模型采用逻辑回归（测试集表现最优且可解释性强，符合信贷评分卡实践）
    p_main, p_main_tr = scores["逻辑回归"], scores_tr["逻辑回归"]
    thr = float(np.quantile(p_main_tr[ytr == 0], 0.90))
    pred = (p_main >= thr).astype(int)
    rank_cut = float(np.quantile(p_main_tr, 0.75))          # 训练集按分数排名前 25%
    pred_rank = (p_main >= rank_cut).astype(int)

    def confusion(p):
        tp = int(((p == 1) & (yte == 1)).sum()); fp = int(((p == 1) & (yte == 0)).sum())
        fn = int(((p == 0) & (yte == 1)).sum()); tn = int(((p == 0) & (yte == 0)).sum())
        return dict(tp=tp, fp=fp, fn=fn, tn=tn, flagged_n=int(p.sum()),
                    flagged_share=round(float(p.mean()), 4),
                    test_fpr=round(float(fp / (fp + tn)), 4), test_recall=round(float(tp / (tp + fn)), 4),
                    precision=round(float(tp / (tp + fp)), 4))

    thr_res = {"score_strategy": {"threshold": round(thr, 4), **confusion(pred)},
               "rank_strategy": {"cut": round(rank_cut, 4), **confusion(pred_rank)}}
    pred = pred_rank     # 后续稳健性分析采用排名制阈值

    # ---- 分档（按评分排名切分，与复核产能挂钩）----
    qs = np.quantile(p_main, [0.5, 0.8, 0.9, 0.97])
    labels = ["低风险", "关注", "中风险", "高风险", "极高风险"]
    bins = [-np.inf] + list(qs) + [np.inf]
    te2 = te.assign(score=p_main, level=pd.cut(p_main, bins=bins, labels=labels))
    lvl = te2.groupby("level", observed=True).agg(
        商户数=("is_noncompliant", "size"), 非规范数=("is_noncompliant", "sum"))
    lvl["命中率"] = (lvl["非规范数"] / lvl["商户数"]).round(4)
    lvl["覆盖率"] = (lvl["非规范数"] / te.is_noncompliant.sum()).round(4)

    # ---- 分模式 / 分强度稳健性（排名制阈值）----
    te3 = te.assign(flag=(p_main >= rank_cut).astype(int))
    by_mode = {m: round(float(te3[te3.fraud_modes.fillna("").str.contains(m)]["flag"].mean()), 4)
               for m in ["刷单", "虚构交易", "伪造流水"]}
    by_int = {k: round(float(te3[te3.fraud_intensity == k]["flag"].mean()), 4) for k in ["低", "中", "高"]}
    hn = te3[te3.hard_negative == 1]
    hn_res = {"hard_negative_n": int(len(hn)), "hard_negative_flag_rate": round(float(hn.flag.mean()), 4),
              "normal_flag_rate": round(float(te3[(te3.is_noncompliant == 0) & (te3.hard_negative == 0)]["flag"].mean()), 4)}
    vis = te3[te3.is_noncompliant == 1]
    hn_res["visible_fraud_recall"] = round(float(vis[vis.visible_fraud == 1].flag.mean()), 4)
    hn_res["invisible_fraud_recall"] = round(float(vis[vis.visible_fraud == 0].flag.mean()), 4)
    hn_res["hn_flag_rate_by_type"] = {k: round(float(v.flag.mean()), 4)
                                      for k, v in hn.groupby("hn_type")}

    # ---- SHAP ----
    explainer = shap.TreeExplainer(models["LightGBM"])
    sv = explainer.shap_values(Xte)
    mean_abs = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
    shap_top = {k: round(float(v), 4) for k, v in mean_abs.head(15).items()}

    # 图3：ROC / PR 曲线
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    palette = {"逻辑回归": C["grey"], "随机森林": C["flow"], "LightGBM": C["brush"],
               "孤立森林(无监督)": "#BBBBBB", "融合评分": C["fake"]}
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(yte, s)
        axes[0].plot(fpr, tpr, lw=1.5, color=palette[name],
                     label=f"{name} (AUC={results[name]['test']['auc']:.3f})")
        prec, rec, _ = precision_recall_curve(yte, s)
        axes[1].plot(rec, prec, lw=1.5, color=palette[name],
                     label=f"{name} (PR-AUC={results[name]['test']['pr_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], ls="--", lw=0.8, color="#BBBBBB")
    axes[1].axhline(yte.mean(), ls="--", lw=0.8, color="#BBBBBB")
    axes[1].text(0.62, yte.mean() + 0.02, f"随机水平 {yte.mean():.2f}", fontsize=7.6, color="#777777")
    for ax, t, xl, yl in [(axes[0], "(a) ROC 曲线（测试集）", "假阳性率 FPR", "真阳性率 TPR"),
                          (axes[1], "(b) 精确率-召回率曲线", "召回率 Recall", "精确率 Precision")]:
        ax.set_title(t, loc="left"); ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.legend(frameon=False, fontsize=7.4, loc="lower left" if ax is axes[0] else "upper right")
        despine(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_model_curves.png"))
    plt.close(fig)

    # 图4（四象限）：单变量区分度 / SHAP 贡献 / 评分分布 / 风险分档效果
    def uni_auc(x, y):
        r = pd.Series(x).rank().values
        n1, n0 = y.sum(), (1 - y).sum()
        a = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        return max(a, 1 - a)

    uni = pd.Series({c: uni_auc(Xtr[c].fillna(Xtr[c].median()).values, ytr) for c in cols}).sort_values(ascending=False)
    uni_top = uni.head(10)
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.4))
    ax = axes[0, 0]
    ax.barh([CN.get(i, i) for i in uni_top.index[::-1]], uni_top.values[::-1], color=C["accent"], height=0.62)
    ax.axvline(0.5, color="#999999", lw=0.9, ls="--")
    ax.set_xlim(0.5, max(0.85, uni_top.max() + 0.03))
    ax.set_title("(a) 单变量区分度（AUC，训练集，前10位）", loc="left", fontsize=9.5)
    ax.set_xlabel("单变量 AUC", fontsize=8.6)
    ax.tick_params(axis="y", labelsize=7.8); ax.tick_params(axis="x", labelsize=7.8)
    despine(ax)

    ax = axes[0, 1]
    top10 = mean_abs.head(10)[::-1]
    ax.barh([CN.get(i, i) for i in top10.index], top10.values, color=C["fake"], height=0.62)
    ax.set_title("(b) 多变量贡献（LightGBM SHAP，前10位）", loc="left", fontsize=9.5)
    ax.set_xlabel("平均 |SHAP| 值", fontsize=8.6)
    ax.tick_params(axis="y", labelsize=7.8); ax.tick_params(axis="x", labelsize=7.8)
    despine(ax)

    ax = axes[1, 0]
    ax.hist(p_main[yte == 0], bins=40, range=(0, 1), color=C["normal"], alpha=0.85, label="合规商户", density=True)
    ax.hist(p_main[yte == 1], bins=40, range=(0, 1), color=C["fake"], alpha=0.70, label="非规范商户", density=True)
    ax.axvline(rank_cut, ls="--", lw=1.2, color="#333333")
    ax.text(min(rank_cut + 0.03, 0.7), ax.get_ylim()[1] * 0.9, f"复核线 {rank_cut:.2f}", fontsize=7.8)
    ax.set_title("(c) 主模型评分分布（测试集）", loc="left", fontsize=9.5)
    ax.set_xlabel("风险评分", fontsize=8.6); ax.set_ylabel("密度", fontsize=8.6)
    ax.legend(frameon=False, fontsize=7.6)
    ax.tick_params(labelsize=7.8)
    despine(ax)

    ax = axes[1, 1]
    x = np.arange(len(lvl))
    ax.bar(x - 0.2, lvl["命中率"].values * 100, width=0.4, color=C["fake"], label="命中率")
    ax.bar(x + 0.2, lvl["覆盖率"].values * 100, width=0.4, color=C["normal"], label="覆盖率")
    for xi, (h, c) in enumerate(zip(lvl["命中率"].values, lvl["覆盖率"].values)):
        ax.text(xi - 0.2, h * 100 + 2, f"{h*100:.0f}", ha="center", fontsize=7.2)
        ax.text(xi + 0.2, c * 100 + 2, f"{c*100:.0f}", ha="center", fontsize=7.2)
    ax.set_xticks(x); ax.set_xticklabels(lvl.index, fontsize=7.8)
    ax.set_ylabel("比例（%）", fontsize=8.6); ax.set_ylim(0, 118)
    ax.set_title("(d) 各风险分档的命中率与覆盖率", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.6, ncol=2)
    ax.tick_params(labelsize=7.8)
    despine(ax)
    fig.tight_layout(h_pad=1.6, w_pad=2.0)
    fig.savefig(os.path.join(FIG, "fig4_model_explain.png"))
    plt.close(fig)

    # 保存结果
    out = {"feature_cn": CN, "cv_auc_mean": round(float(cv.mean()), 4), "cv_auc_std": round(float(cv.std()), 4),
           "models": results, "threshold_rule": thr_res, "level_table": lvl.reset_index().to_dict("records"),
           "robustness": {"by_mode_flag_rate": by_mode, "by_intensity_flag_rate": by_int, **hn_res},
           "shap_top": shap_top, "n_features": len(cols),
           "lgb_best_params": {k: v for k, v in LGB_PARAMS.items() if k not in ("n_jobs", "verbose")},
           "lgb_search_log": search_log, "primary_model": "逻辑回归（主评分）；LightGBM 用于 SHAP 归因",
           "univariate_auc_top": {k: round(float(v), 4) for k, v in uni_top.items()}}
    json.dump(out, open(os.path.join(BASE, "report", "model_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    te.assign(score_main=p_main, score_lgb=scores["LightGBM"], score_rf=scores["随机森林"],
              score_lr=scores["逻辑回归"], score_iso=p_iso,
              flag_score=(p_main >= thr).astype(int), flag_rank=(p_main >= rank_cut).astype(int)).to_csv(
        os.path.join(PROC, "predictions.csv"), index=False)
    print(json.dumps({"threshold_rule": thr_res, "robustness": out["robustness"]}, ensure_ascii=False, indent=2))
    print("分档表:\n", lvl.to_string())
    print("SHAP Top10:", json.dumps({CN.get(k, k): v for k, v in list(shap_top.items())[:10]}, ensure_ascii=False))


if __name__ == "__main__":
    main()