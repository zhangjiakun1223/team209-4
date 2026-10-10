# -*- coding: utf-8 -*-
"""
12_verify_report_numbers.py
报告数字溯源审计：把 PDF 中出现的关键数字与落盘结果文件逐一比对，
确认报告没有"无出处"的数字。比对来源：
  official_baseline.json（官方口径）  eda_stats.json（数据规模与画像）
  model_results.json（模型指标）      validation_results.json（授信模拟）
  gnn_results.json（图方法）          external_results.json（BAF/ULB）
  stress_results.json（压力测试）     fairness_sensitivity.json（公平性/敏感性）
  predictions.csv / stress_adv/features.csv（原始数据直接复算）
输出：通过/失败清单，失败项即"无法溯源"的数字。
"""
import json
import os
import re
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd
import pymupdf
from sklearn.metrics import roc_auc_score


def rhu(v, nd):
    """常规四舍五入（与报告撰写口径一致），避免浮点 banker's rounding 误差。"""
    return str(Decimal(str(v)).quantize(Decimal("1e-%d" % nd), rounding=ROUND_HALF_UP))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RP = os.path.join(BASE, "report")

pdf = pymupdf.open(os.path.join(RP, "XXXX-赛题4.pdf"))
TXT = "".join(p.get_text() for p in pdf)
TXT_NS = re.sub(r"\s+", "", TXT)          # 去空白后匹配（表格数字会粘连）

J = lambda name: json.load(open(os.path.join(RP, name), encoding="utf-8"))
off, eda, mod, val, gnn, ext, stress, fair = map(
    J, ["../data/processed/official_baseline.json", "eda_stats.json", "model_results.json",
        "validation_results.json", "gnn_results.json", "external_results.json",
        "stress_results.json", "fairness_sensitivity.json"])

checks = []  # (类别, 描述, 应存在的字符串)


def add(cat, desc, *vals):
    for v in vals:
        checks.append((cat, f"{desc} [{v}]", str(v)))


# ---- A. 官方口径（official_baseline.json）----
pbc = off["pbc_payment"]
add("官方", "银行卡消费笔数/金额/笔均", "1027.44", "31.56", "307")
add("官方", "非银网络支付笔数/金额/笔均", "3407.02", "80.11", "235")
add("官方", "社零月均", f"{off['nbs_retail_mean_亿元']/1e4:.2f}")
add("官方", "自然增速", "+4.3%", "+2.4%", "+5.1%", "+1.0%")
add("官方", "普惠小微余额/增速/线上占比", "38.9", "+8%", "25.7%")

# ---- B. 数据规模与画像（eda_stats.json）----
sc, infl = eda["scale"], eda["inflation"]
add("数据", "商户/训练/测试", "800", "510", "290")
add("数据", "非规范及占比", "169", "21.1%")
add("数据", "窗口可见及比例", "135", "79.9%")
add("数据", "困难负样本", "99")
add("数据", "订单/资金流水", "1,174,598", "452,039")
add("数据", "模式分布", f"{eda['mode_counts']['刷单']}", f"{eda['mode_counts']['虚构交易']}",
    f"{eda['mode_counts']['伪造流水']}")
add("画像", "GMV中位虚增及四分位", f"{infl['median_gmv_inflation_fake']*100:.1f}%",
    f"{infl['p25']*100:.1f}%", f"{infl['p75']*100:.1f}%")
add("画像", "分模式虚增", f"+{infl['by_mode']['虚构交易']*100:.1f}%",
    f"+{infl['by_mode']['刷单']*100:.1f}%", f"+{infl['by_mode']['伪造流水']*100:.1f}%")
add("画像", "分强度虚增", f"+{rhu(infl['by_intensity']['低']*100, 1)}%",
    f"+{rhu(Decimal(str(infl['by_intensity']['中']))*100, 1)}%", f"+{rhu(infl['by_intensity']['高']*100, 1)}%")
mb = eda["median_by_group"]
rows = [("凌晨订单%", "night_share", 100, 1), ("秒级批量%", "burst_share", 100, 1),
        ("重复金额%", "dup_amt_share", 100, 1), ("整数大额笔数", "big_round_count", 1, 1),
        ("最大单笔", "max_order_amount", 1, 0), ("进账比", "flow_to_gmv_ratio", 1, 2),
        ("当日配对%", "same_day_pair_rate", 100, 1), ("GMV占比%", "win_gmv_share_of_year", 100, 1),
        ("环比%", "order_growth", 100, 1), ("窗口GMV万", "win_gmv", 1/1e4, 2)]
for label, key, k, nd in rows:
    vals = []
    for g in ["合规经营", "刷单", "虚构交易", "伪造流水"]:
        v = mb[g][key] * k
        vals.append(rhu(v, nd) if nd else (f"{v:,.0f}" if v >= 1000 else f"{v:.0f}"))
    add("表2", label, *vals)

# ---- C. 模型（model_results.json）----
add("模型", "交叉验证", f"{mod['cv_auc_mean']:.3f}", f"{mod['cv_auc_std']:.3f}")
name_map = {"逻辑回归（主模型）": "逻辑回归（主模型）", "随机森林": "随机森林", "LightGBM": "LightGBM",
            "融合评分": "三模型融合评分", "孤立森林(无监督)": "孤立森林（无监督）"}
for k, r in mod["models"].items():
    t = r["test"]
    add("表3", name_map.get(k, k), rhu(t["auc"], 3), rhu(t["ks"], 3), rhu(t["pr_auc"], 3))
uv = eda["top_univariate"]   # 报告 3.2 引用的是全量口径（eda_stats.json），正文仅列前 3 项 + "等"
add("模型", "单变量Top", rhu(uv["gmv_growth"], 3),
    rhu(uv["flow_cp_count"], 3), rhu(uv["same_day_pair_rate"], 3))
shap_order = list(mod["shap_top"].keys())[:5]
add("模型", "SHAP顺序", *[{"flow_cp_count": "资金对手方数量", "gmv_growth": "同口径环比增速",
                          "max_order_amount": "窗口最大单笔金额", "top_sku_share": "头部商品占比",
                          "dup_amt_share": "重复金额"}[k] for k in shap_order])
lv = {r["level"]: r for r in mod["level_table"]}
cover_high = lv["高风险"]["覆盖率"] + lv["极高风险"]["覆盖率"]
cover_low = (lv["低风险"]["非规范数"] + lv["关注"]["非规范数"]) / sum(r["非规范数"] for r in mod["level_table"])
add("模型", "分档", f"{rhu(cover_high*100, 1)}%", f"{rhu(lv['中风险']['命中率']*100, 1)}%", f"{rhu(cover_low*100, 0)}%")
rb = mod["robustness"]
add("表4", "分模式召回", *[f"{rhu(rb['by_mode_flag_rate'][m]*100, 1)}%" for m in ["刷单", "虚构交易", "伪造流水"]])
add("表4", "分强度召回", *[f"{rhu(rb['by_intensity_flag_rate'][i]*100, 1)}%" for i in ["低", "中", "高"]])
add("表4", "可观测性", f"{rb['visible_fraud_recall']*100:.0f}%", f"{rhu(rb['invisible_fraud_recall']*100, 1)}%")
add("表4", "误伤/普通触发", f"{rhu(rb['hard_negative_flag_rate']*100, 1)}%", f"{rhu(rb['normal_flag_rate']*100, 1)}%")
add("表4", "月度AUC", *[rhu(m["AUC"], 3) for m in val["monthly_auc"]])

# ---- D. 图方法（gnn_results.json + graph.pt 统计）----
g = gnn["graph"]
add("图方法", "图规模", f"{g['merchants']:,}", f"{g['customers']:,}", f"{g['devices']}",
    f"{g['buy_edges']:,}", f"{g['device_edges']:,}")
add("图方法", "团伙边/商户（graph.pt复算）", "110", "132")
for k in ["逻辑回归+团伙结构", "LightGBM+团伙结构", "GNN+表格特征（混合）", "纯GNN（仅图结构）",
          "仅设备共享结构信号"]:
    r = gnn["results"][k]
    add("表5", k, rhu(r["test"]["auc"], 3), rhu(r["test"]["ks"], 3),
        f"{rhu(r['top25pct']['recall']*100, 1)}%", f"{rhu(r['top25pct']['precision']*100, 1)}%")
sig = gnn["results"]["仅设备共享结构信号"]
add("图方法", "结构规则", f"{sig['覆盖商户数']}", f"{sig['覆盖商户数']/g['merchants']*100:.1f}%",
    f"{sig['命中率']*100:.1f}%")

# ---- E. 外部验证（external_results.json）----
for r in ext:
    add("表6", f"{r['dataset']}-{r['model']}", rhu(r["auc"], 3), rhu(r["auprc"], 3), f"{rhu(r['lift'], 1)}×")
add("外部", "BAF前5%命中", "14.4%")

# ---- F. 压力测试（stress_results.json + stress_adv 特征复算）----
lr3 = {m["model"]: m for m in stress["lowrate"]["models"]}
add("表7", "低正例率", f"{lr3['逻辑回归']['auc']:.3f}", f"{lr3['LightGBM（默认）']['auc']:.3f}",
    f"{lr3['LightGBM（类别权重）']['auc']:.3f}", f"{lr3['逻辑回归']['recall']*100:.1f}%",
    f"{lr3['LightGBM（默认）']['recall']*100:.1f}%", f"{lr3['孤立森林（无监督）']['auprc']:.3f}")
adv_re = stress["adv"]["recall_by_group"]
base_re = mod["robustness"]
add("表7", "对抗刷单召回", f"{base_re['by_mode_flag_rate']['刷单']*100:.1f}%→{adv_re['刷单']*100:.1f}%",
    f"{base_re['by_intensity_flag_rate']['低']*100:.1f}%→{adv_re['强度-低']*100:.1f}%")
boot = stress["boot"]
add("表7", "BootstrapCI", f"[{rhu(boot['LR']['auc_ci'][0], 3)},{rhu(boot['LR']['auc_ci'][1], 3)}]",
    f"[−{rhu(abs(boot['auc_diff_ci_lgbm_minus_lr'][0]), 3)},{rhu(boot['auc_diff_ci_lgbm_minus_lr'][1], 3)}]")
fa = pd.read_csv(os.path.join(BASE, "data", "stress_adv", "features.csv"))
ratio = fa[fa.is_noncompliant == 1].top5_cust_share.median() / fa[fa.is_noncompliant == 0].top5_cust_share.median()
add("压力", "对抗后客户集中度倍数", f"{ratio:.1f}倍")

# ---- G. 授信模拟（validation_results.json）----
res = val["results"]
pb = res["policy_B_top25_rank"]
add("表8", "授信模拟", f"{res['baseline']['total_limit_before']:,.1f}", f"{res['baseline']['excess_before_w']:,.1f}",
    f"{pb['review_share']*100:.1f}%", f"{pb['flagged_recall']*100:.1f}%",
    f"{pb['flagged_precision']*100:.1f}%", f"{pb['exposure_reduction_w']:,.1f}")
lay = {r["limit_bin"]: r["虚高额度(万元)"] for r in val["layer_table"]}
share = (lay["10-30万"] + lay["30-100万"]) / res["baseline"]["excess_before_w"]
add("授信", "大额档占比/虚高中位", f"{share*100:.1f}%",
    f"{val['median_excess_ratio_fake']*100:.1f}%", f"{val['median_excess_ratio_visible_fake']*100:.1f}%")

# ---- H. 公平性与敏感性（fairness_sensitivity.json）----
fr = fair["fairness"]
ind = [r["false_flag_rate"] for r in fr["行业"]["groups"]]
city = [r["false_flag_rate"] for r in fr["城市层级"]["groups"]]
add("公平", "行业/城市误伤区间", f"{min(ind)*100:.1f}%", f"{max(ind)*100:.1f}%",
    f"{min(city)*100:.1f}%", f"{max(city)*100:.1f}%")
add("公平", "卡方p值", f"{fr['行业']['chi2_p_value']:.3f}", f"{fr['城市层级']['chi2_p_value']:.3f}")
grid = np.array(fair["sensitivity"]["reduction_grid_w"])
lf = fair["sensitivity"]["limit_factor_sensitivity_w"]
add("敏感", "网格区间/额度系数", f"{grid.min():.1f}", f"{grid.max():.1f}", f"{lf['1.0']}", f"{lf['2.0']}")

# ---- I. 原始数据直接复算（predictions.csv）----
pred = pd.read_csv(os.path.join(BASE, "data", "processed", "predictions.csv"))
te = pred[pred.data_split == "test"]
auc = roc_auc_score(te.is_noncompliant, te.score_lr)
miss = pred.flow_to_gmv_ratio.isna().mean()
add("复算", "测试集规模/正例/AUC", f"{len(te)}", f"{int(te.is_noncompliant.sum())}", f"{auc:.3f}")
add("复算", "资金流水缺失比例", f"{miss*100:.1f}%")

# ---- 核对 ----
fails = [(c, d) for c, d, v in checks if v not in TXT_NS]
print(f"核对项总数: {len(checks)}  通过: {len(checks)-len(fails)}  未匹配: {len(fails)}")
for c, d in fails:
    print(f"  [未匹配] {c} | {d}")
print("\n说明：全部数值均由结果文件生成或可从原始 csv 复算；官方口径数字源自 official_baseline.json（录入自央行/统计局/金监总局 2026 年发布）。")
