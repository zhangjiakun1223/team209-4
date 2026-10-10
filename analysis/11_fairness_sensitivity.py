# -*- coding: utf-8 -*-
"""
11_fairness_sensitivity.py
报告交付前的两项补充核验（基于现有 predictions.csv，不重新生成数据）：

① 公平性快查：按行业与城市层级统计合规商户的复核触发率（误伤率）与
   非规范商户召回率（排名制前 25%），并用卡方检验判断组间差异是否显著；
② 敞口敏感性：对违约率 PD（5%—12%）× 违约损失率 LGD（40%—80%）网格
   重算预期损失敞口下降，并对额度系数 limit_factor（1.0—2.0）重算虚高规模。

输出：report/fairness_sensitivity.json
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "report", "fairness_sensitivity.json")

LIMIT_FACTOR = 1.5
LIMIT_CAP = 1_000_000


def fairness(pred: pd.DataFrame) -> dict:
    pred = pred.copy()
    pred["flag"] = pred["flag_rank"].astype(bool)
    out = {}
    for dim, dim_cn in [("industry", "行业"), ("city_tier", "城市层级")]:
        rows = []
        for g, d in pred.groupby(dim):
            comp, fraud = d[d.is_noncompliant == 0], d[d.is_noncompliant == 1]
            rows.append({
                "group": str(g), "n": int(len(d)),
                "n_compliant": int(len(comp)), "n_fraud": int(len(fraud)),
                # 合规商户被触发复核的比例 = 误伤率
                "false_flag_rate": round(float(comp.flag.mean()), 4) if len(comp) else None,
                # 非规范商户被触发的比例 = 召回
                "fraud_recall": round(float(fraud.flag.mean()), 4) if len(fraud) else None,
            })
        # 卡方检验：合规商户「是否被触发」× 分组维度（检验误伤率组间差异）
        comp = pred[pred.is_noncompliant == 0]
        ct = pd.crosstab(comp[dim], comp.flag)
        chi2, pval, dof, _ = chi2_contingency(ct)
        out[dim_cn] = {"groups": rows,
                       "chi2_p_value": round(float(pval), 4),
                       "contingency": ct.values.tolist()}
        print(f"\n[{dim_cn}] 合规商户误伤率（排名制前 25% 复核）:")
        for r in rows:
            print(f"  {r['group']:6s} n={r['n']:3d}（合规 {r['n_compliant']} / 非规范 {r['n_fraud']}）"
                  f"  误伤率 {r['false_flag_rate']:.1%}  召回 {r['fraud_recall']:.1%}")
        print(f"  组间差异卡方检验 p = {pval:.3f}"
              f" -> {'显著' if pval < 0.05 else '不显著（未见系统性偏倚）'}")
    return out


def sensitivity(pred: pd.DataFrame) -> dict:
    fraud = pred.is_noncompliant == 1

    def excess_total(factor: float) -> float:
        lt = np.minimum(factor * pred["true_window_gmv"] / 3, LIMIT_CAP)
        lo = np.minimum(factor * pred["obs_window_gmv"] / 3, LIMIT_CAP)
        lo = np.maximum(lo, lt)                       # 核额取观测数据（含虚假数据）
        return float((lo - lt).clip(lower=0)[fraud].sum())

    base_excess = excess_total(LIMIT_FACTOR)          # 对应报告的 256.3 万元

    # PD × LGD 网格：策略拦截率 100%（excess_after=0），敞口下降 = 虚高总额 × PD × LGD
    pds, lgds = [0.05, 0.06, 0.08, 0.10, 0.12], [0.40, 0.50, 0.60, 0.70, 0.80]
    grid = [[round(base_excess * pd_ * lgd / 1e4, 1) for lgd in lgds] for pd_ in pds]

    # 额度系数敏感性：虚高规模本身随 limit_factor 变化
    lf_sens = {str(k): round(excess_total(k) / 1e4, 1) for k in (1.0, 1.5, 2.0)}

    print(f"\n[敞口敏感性] 额度系数 {LIMIT_FACTOR} 下虚高总额 {base_excess/1e4:.1f} 万元")
    print("  PD\\LGD  " + "  ".join(f"{l:.0%}" for l in lgds))
    for pd_, row in zip(pds, grid):
        print(f"  {pd_:>5.0%}  " + "  ".join(f"{v:5.1f}" for v in row))
    print(f"  额度系数 1.0 / 1.5 / 2.0 对应虚高总额: {lf_sens} 万元")
    return {"excess_total_w": round(base_excess / 1e4, 1),
            "pd_grid": pds, "lgd_grid": lgds, "reduction_grid_w": grid,
            "limit_factor_sensitivity_w": lf_sens}


if __name__ == "__main__":
    pred = pd.read_csv(os.path.join(BASE, "data", "processed", "predictions.csv"))
    print(f"[data] 测试集 {len(pred)} 户，非规范 {int(pred.is_noncompliant.sum())} 户")
    res = {"fairness": fairness(pred), "sensitivity": sensitivity(pred)}
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[save] {OUT}")
