# -*- coding: utf-8 -*-
"""
10_stress_test.py
方案健壮性压力测试（回应「自构数据偏理想」的核心质疑），三个实验：

  ① 低正例率重跑（lowrate）：非规范商户占比从 21% 降到 3%（贴近真实客群），
     商户池扩到 3000 户保证测试集有足够正例，重跑「数据生成→特征→建模」全流程，
     并对照 LightGBM 默认 vs 类别权重（呼应 ULB 数据集上的发现）；
  ② 对抗性隐蔽刷单（adv）：保持原规模与正例率，但把注入参数改为「拟人化」操作
     ——避开凌晨时段、金额随机化、下单间隔不规则、金额非整数，
     检验模型对「进化骗术」的召回衰减（按行为模式分组对比基线）；
  ③ 统计置信区间（boot）：对现有测试集做 Bootstrap 重采样，
     给出主模型 AUC / PR-AUC 的 95% 置信区间，并检验 LR 与 LightGBM 差异是否显著。

输出：report/stress_results.json（增量合并，不覆盖任何现有数据与结果文件）

用法：python 10_stress_test.py [boot|lowrate|adv|all]
"""
import importlib.util
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "report", "stress_results.json")
SPLIT = pd.Timestamp("2026-06-01")
SEED = 42

LABEL_COLS = ["merchant_id", "industry", "city_tier", "apply_date", "is_noncompliant",
              "fraud_modes", "n_fraud_modes", "fraud_intensity", "visible_fraud",
              "hard_negative", "hn_type", "data_split"]

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=7, min_child_samples=10,
                  subsample=0.85, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                  random_state=SEED, n_jobs=-1, verbose=-1)


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ks_stat(y, s) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr - fpr))


def topk(y, s, frac=0.25) -> dict:
    k = max(int(np.ceil(len(s) * frac)), 1)
    idx = np.argsort(-s)[:k]
    hit = float(y[idx].sum())
    return {"k": k, "recall": round(hit / max(y.sum(), 1), 4),
            "precision": round(hit / k, 4)}


def evaluate(name, y, s) -> dict:
    r = {"auc": round(float(roc_auc_score(y, s)), 4),
         "auprc": round(float(average_precision_score(y, s)), 4),
         "ks": round(ks_stat(y, s), 4)}
    r.update(topk(y, s))
    r["model"] = name
    return r


# ---------------- ① / ② 复用 01+02 的完整生成管线，输出到独立目录 ----------------

def run_pipeline(tag: str, out_dir: str, overrides01: dict) -> pd.DataFrame:
    """按覆盖参数重跑数据生成与特征计算（不触碰 data/processed 原始结果）。"""
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{tag}] 数据生成 -> {out_dir}", flush=True)
    m1 = load_module(os.path.join(BASE, "analysis", "01_build_dataset.py"), f"ds_{tag}")
    m1.PROC = out_dir                     # 模块加载时已用原目录读完官方基线，改路径只影响输出
    for k, v in overrides01.items():
        setattr(m1, k, v)
    m1.main()

    print(f"[{tag}] 特征计算", flush=True)
    m2 = load_module(os.path.join(BASE, "analysis", "02_features.py"), f"ft_{tag}")
    m2.PROC = out_dir
    m2.main()
    return pd.read_csv(os.path.join(out_dir, "features.csv"), parse_dates=["apply_date"])


def fit_models(f: pd.DataFrame, with_weight: bool = False) -> tuple:
    tr, te = f[f.apply_date < SPLIT], f[f.apply_date >= SPLIT]
    feats = [c for c in f.columns if c not in LABEL_COLS
             and pd.api.types.is_numeric_dtype(f[c])]
    Xtr, ytr = tr[feats], tr.is_noncompliant.values.astype(int)
    Xte, yte = te[feats], te.is_noncompliant.values.astype(int)
    print(f"  训练 {len(tr)}（正例 {int(ytr.sum())}）｜测试 {len(te)}（正例 {int(yte.sum())}）", flush=True)

    models = {
        "逻辑回归": Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=3000, C=0.5))]),
        "LightGBM（默认）": Pipeline([("imp", SimpleImputer(strategy="median")),
                                    ("clf", lgb.LGBMClassifier(**LGB_PARAMS))]),
        "孤立森林（无监督）": Pipeline([("imp", SimpleImputer(strategy="median")),
                                      ("clf", IsolationForest(n_estimators=300, random_state=SEED,
                                                              n_jobs=-1))]),
    }
    if with_weight:
        pw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        models["LightGBM（类别权重）"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", lgb.LGBMClassifier(scale_pos_weight=pw, **LGB_PARAMS))])

    results, main_scores = [], None
    for nm, mdl in models.items():
        if "孤立森林" in nm:
            mdl.fit(Xtr)
            s = -mdl.score_samples(Xte)          # 异常分越高越可疑
        else:
            mdl.fit(Xtr, ytr)
            s = mdl.predict_proba(Xte)[:, 1]
        r = evaluate(nm, yte, s)
        results.append(r)
        print(f"  {nm}: AUC={r['auc']} AUPRC={r['auprc']} KS={r['ks']} "
              f"前25%召回={r['recall']:.1%} 命中={r['precision']:.1%}", flush=True)
        if nm == "逻辑回归":
            main_scores = (te[["merchant_id", "fraud_modes", "fraud_intensity", "visible_fraud"]].copy(),
                           yte, s)
    return results, main_scores


def recall_by_group(te_info, yte, s) -> dict:
    """主模型按评分排名前 25% 触发复核时，分组召回率。"""
    k = max(int(np.ceil(len(s) * 0.25)), 1)
    flag = np.zeros(len(s), dtype=bool)
    flag[np.argsort(-s)[:k]] = True
    d = te_info.assign(y=yte, flag=flag)
    out = {"总召回": round(float(d.loc[d.y == 1, "flag"].mean()), 4)}
    for m in ["刷单", "虚构交易", "伪造流水"]:
        sub = d[d.fraud_modes.fillna("").str.contains(m) & (d.y == 1)]
        if len(sub):
            out[m] = round(float(sub.flag.mean()), 4)
    for it in ["低", "中", "高"]:
        sub = d[(d.fraud_intensity == it) & (d.y == 1)]
        if len(sub):
            out[f"强度-{it}"] = round(float(sub.flag.mean()), 4)
    return out


def device_share_rule(out_dir: str) -> dict:
    """轻量版结构信号：窗口内订单里，商户的客户是否使用了「跨商户共用设备」。"""
    tx = pd.read_csv(os.path.join(out_dir, "transactions.csv.gz"),
                     usecols=["merchant_id", "ts", "device_id"], parse_dates=["ts"])
    mc = pd.read_csv(os.path.join(out_dir, "merchants.csv"), parse_dates=["window_start"])
    tx = tx.merge(mc[["merchant_id", "window_start"]], on="merchant_id")
    win = tx[tx.ts >= tx.window_start]
    dev_m = win.groupby("device_id")["merchant_id"].nunique()
    shared = dev_m[dev_m > 1].index
    hit_merchants = win.loc[win.device_id.isin(shared), "merchant_id"].unique()
    y = mc.set_index("merchant_id").is_noncompliant
    covered = y.reindex(hit_merchants).fillna(0)
    return {"覆盖商户数": int(len(hit_merchants)),
            "覆盖率": round(len(hit_merchants) / len(y), 4),
            "命中率": round(float(covered.mean()), 4) if len(covered) else None,
            "规则召回": round(float(covered.sum() / max(y.sum(), 1)), 4)}


# ---------------- ③ Bootstrap 置信区间 ----------------

def run_boot(B: int = 2000) -> dict:
    p = pd.read_csv(os.path.join(BASE, "data", "processed", "predictions.csv"))
    te = p[p.data_split == "test"].reset_index(drop=True)
    y = te.is_noncompliant.values.astype(int)
    lr, lgbm, diff, lr_p, lgbm_p = [], [], [], [], []
    rng = np.random.default_rng(SEED)
    n = len(te)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() in (0, n):
            continue
        s1, s2 = te.score_lr.values[idx], te.score_lgb.values[idx]
        a1, a2 = roc_auc_score(yy, s1), roc_auc_score(yy, s2)
        lr.append(a1); lgbm.append(a2); diff.append(a2 - a1)
        lr_p.append(average_precision_score(yy, s1)); lgbm_p.append(average_precision_score(yy, s2))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    d_ci = ci(diff)
    print(f"[boot] B={B}")
    print(f"  逻辑回归  AUC {roc_auc_score(y, te.score_lr):.4f} 95%CI {ci(lr)} | "
          f"PR-AUC {average_precision_score(y, te.score_lr):.4f} 95%CI {ci(lr_p)}")
    print(f"  LightGBM  AUC {roc_auc_score(y, te.score_lgb):.4f} 95%CI {ci(lgbm)} | "
          f"PR-AUC {average_precision_score(y, te.score_lgb):.4f} 95%CI {ci(lgbm_p)}")
    print(f"  AUC差(LGBM-LR) 95%CI {d_ci} -> {'显著' if d_ci[0] > 0 or d_ci[1] < 0 else '不显著（差异在随机波动范围内）'}")
    return {"B": B, "n_test": int(n),
            "LR": {"auc_ci": ci(lr), "auprc_ci": ci(lr_p)},
            "LGBM": {"auc_ci": ci(lgbm), "auprc_ci": ci(lgbm_p)},
            "auc_diff_ci_lgbm_minus_lr": d_ci,
            "diff_significant": bool(d_ci[0] > 0 or d_ci[1] < 0)}


# ---------------- 实验入口 ----------------

def run_lowrate() -> dict:
    out_dir = os.path.join(BASE, "data", "stress_lowrate")
    fpath = os.path.join(out_dir, "features.csv")
    if os.path.exists(fpath):      # 数据已生成（此前运行），直接复用避免重复等待
        f = pd.read_csv(fpath, parse_dates=["apply_date"])
        print("[lowrate] 复用已生成数据", flush=True)
    else:
        f = run_pipeline("lowrate", out_dir,
                         dict(N_MERCHANTS=3000, FAKE_RATIO=0.03,
                              N_POOL_CUSTOMERS=225000))   # 客户池同比扩大，保持网络密度一致
    print("[lowrate] 正例率", round(f.is_noncompliant.mean() * 100, 2), "%", flush=True)
    results, _ = fit_models(f, with_weight=True)
    rule = device_share_rule(out_dir)
    print(f"  设备共享规则: {rule}", flush=True)
    return {"config": {"n_merchants": 3000, "fake_ratio": 0.03},
            "pos_rate": round(float(f.is_noncompliant.mean()), 4),
            "models": results, "device_share_rule": rule}


def run_adv() -> dict:
    out_dir = os.path.join(BASE, "data", "stress_adv")
    # 拟人化对抗参数：避开凌晨 + 金额随机化 ±45% + 间隔抖动 160% + 非整数金额网格
    # （原参数：夜间 0.30-1.00、金额抖动 0.005-0.06、间隔抖动 0.07-0.45）
    stealth = dict(mult=None, amt_jit=0.45, gap_jit=1.60, night_ratio=0.03, round_lvl=0.55)
    intensity = {}
    for lvl, base in [("低", 0.12), ("中", 0.45), ("高", 1.15)]:   # 仅保留虚增幅度差异
        intensity[lvl] = dict(stealth, mult=base)
    f = run_pipeline("adv", out_dir, dict(INTENSITY=intensity))
    results, (te_info, yte, s) = fit_models(f)
    by_group = recall_by_group(te_info, yte, s)
    print(f"[adv] 分组召回: {by_group}", flush=True)
    return {"config": "刷单拟人化：避开凌晨、金额随机化、间隔不规则、非整数",
            "models": results, "recall_by_group": by_group}


def merge_save(key: str, value: dict):
    data = {}
    if os.path.exists(OUT):
        data = json.load(open(OUT, encoding="utf-8"))
    data[key] = value
    json.dump(data, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[save] {OUT} ({key})", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("boot", "all"):
        merge_save("boot", run_boot())
    if which in ("lowrate", "all"):
        merge_save("lowrate", run_lowrate())
    if which in ("adv", "all"):
        merge_save("adv", run_adv())
