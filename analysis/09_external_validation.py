# -*- coding: utf-8 -*-
"""
09_external_validation.py
在权威公开数据集上做外部验证。

数据来源
--------
1) BAF（Bank Account Fraud Dataset Suite, NeurIPS 2022）
   Jesus S, Pombal J, Alves D, et al. Turning the Tables: Biased, Imbalanced,
   Dynamic Tabular Datasets for ML Evaluation. NeurIPS 2022.
   100 万条真实银行开户欺诈数据（经隐私保护的生成式扩增），含时序字段与多种偏差变体。
   Kaggle: sgpjesus/bank-account-fraud-dataset-neurips-2022

2) IEEE-CIS Fraud Detection（Kaggle 2019 竞赛）
   59 万条真实电商交易，含设备/身份信息，是交易欺诈检测的经典基准。
   Kaggle: ieee-fraud-detection

方法（借鉴上述论文与相关基准的做法）
------------------------------------
- 极端不平衡处理：类别权重、焦点损失（Focal Loss）、保守合成少数类
- 时序验证：按时间切分（BAF 的 month 字段 / IEEE-CIS 的 TransactionDT），
  而非随机切分，复现论文强调的「时序动态」评测
- 评估指标：以 AUPRC 为主（极端不平衡下的标准指标），辅以 AUC 与 KS
- 对照：逻辑回归 / 随机森林 / LightGBM 三档模型

用法
----
先配置 Kaggle 凭据（任选其一）：
  A. 保存 token 到 ~/.kaggle/access_token
  B. 设置环境变量 KAGGLE_API_TOKEN
  C. 执行 kaggle auth login
然后运行：python 09_external_validation.py
"""
import glob
import json
import os
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "external")
OUT = os.path.join(BASE, "report", "external_results.json")
os.makedirs(RAW, exist_ok=True)
SEED = 42


def ks_stat(y, s) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr - fpr))


def evaluate(name, y, s, frac=0.05):
    k = int(np.ceil(len(s) * frac))
    idx = np.argsort(-s)[:k]
    hit = y[idx].sum()
    return {"dataset": name,
            "auc": round(float(roc_auc_score(y, s)), 4),
            "auprc": round(float(average_precision_score(y, s)), 4),
            "ks": round(ks_stat(y, s), 4),
            "top5pct_precision": round(float(hit / k), 4),
            "top5pct_recall": round(float(hit / y.sum()), 4),
            "lift": round(float((hit / k) / y.mean()), 2)}


def kaggle_download(kind: str, ref: str, dest: str) -> bool:
    """调用 kaggle CLI 下载数据集或竞赛数据。"""
    cmd = (["kaggle", "datasets", "download", "-d", ref, "-p", dest, "--unzip"]
           if kind == "dataset" else
           ["kaggle", "competitions", "download", "-c", ref, "-p", dest])
    print(f"[kaggle] {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            print(f"[kaggle] 失败：{r.stderr.strip()[:300]}")
            return False
    except FileNotFoundError:
        print("[kaggle] 未找到 kaggle 命令，请先 pip install kaggle 并配置凭据")
        return False
    except subprocess.TimeoutExpired:
        print("[kaggle] 下载超时")
        return False
    # 解压竞赛数据（datasets download 已带 --unzip）
    for z in glob.glob(os.path.join(dest, "*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        os.remove(z)
    return True


def run_baf():
    """BAF：银行开户欺诈，按时序（month）切分。"""
    d = os.path.join(RAW, "baf")
    if not glob.glob(os.path.join(d, "Base.csv")):
        if not kaggle_download("dataset", "sgpjesus/bank-account-fraud-dataset-neurips-2022", d):
            return None
    files = sorted(glob.glob(os.path.join(d, "*.csv")))
    base = [f for f in files if os.path.basename(f).lower() == "base.csv"] or files
    if not base:
        print("[BAF] 未找到 csv 文件")
        return None
    df = pd.read_csv(base[0])
    print(f"[BAF] {os.path.basename(base[0])}: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    ycol = "fraud_bool"
    if ycol not in df.columns:
        print(f"[BAF] 未找到标签列 {ycol}，现有列：{list(df.columns)[:12]}")
        return None
    print(f"[BAF] 正例率 {df[ycol].mean():.4%}")
    # 时序切分：month 字段
    if "month" in df.columns:
        cut = df["month"].quantile(0.7)
        tr, te = df[df["month"] <= cut], df[df["month"] > cut]
    else:
        n = int(len(df) * 0.7); tr, te = df.iloc[:n], df.iloc[n:]
    drop = [ycol, "month"]
    cols = [c for c in df.columns if c not in drop]
    Xtr = pd.get_dummies(tr[cols], drop_first=True)
    Xte = pd.get_dummies(te[cols], drop_first=True).reindex(columns=Xtr.columns, fill_value=0)
    return Xtr, tr[ycol].values, Xte, te[ycol].values


def run_ieee():
    """IEEE-CIS：电商交易欺诈，按 TransactionDT 时序切分。"""
    d = os.path.join(RAW, "ieee")
    if not glob.glob(os.path.join(d, "train_transaction.csv")):
        if not kaggle_download("competition", "ieee-fraud-detection", d):
            return None
    txp = os.path.join(d, "train_transaction.csv")
    idp = os.path.join(d, "train_identity.csv")
    if not os.path.exists(txp):
        print("[IEEE] 未找到 train_transaction.csv")
        return None
    tx = pd.read_csv(txp)
    if os.path.exists(idp):
        tx = tx.merge(pd.read_csv(idp), on="TransactionID", how="left")
    print(f"[IEEE] {tx.shape[0]:,} 行 × {tx.shape[1]} 列，正例率 {tx.isFraud.mean():.4%}")
    tx = tx.sort_values("TransactionDT").reset_index(drop=True)
    n = int(len(tx) * 0.7)
    tr, te = tx.iloc[:n], tx.iloc[n:]
    drop = ["isFraud", "TransactionID", "TransactionDT"]
    cols = [c for c in tx.columns if c not in drop]
    Xtr = pd.get_dummies(tr[cols], drop_first=True)
    Xte = pd.get_dummies(te[cols], drop_first=True).reindex(columns=Xtr.columns, fill_value=0)
    return Xtr, tr["isFraud"].values, Xte, te["isFraud"].values


def run_ulb():
    """ULB 信用卡欺诈（机器学习组，布鲁塞尔自由大学）：欺诈检测最常用基准。

    28.5 万笔欧洲持卡人交易、492 笔欺诈（0.172%），特征为 PCA 降维后的 V1—V28。
    按 Time 字段（交易相对秒数）时序切分。
    """
    d = os.path.join(RAW, "ulb")
    if not glob.glob(os.path.join(d, "*.csv")):
        if not kaggle_download("dataset", "mlg-ulb/creditcardfraud", d):
            return None
    files = glob.glob(os.path.join(d, "*.csv"))
    if not files:
        print("[ULB] 未找到 csv 文件")
        return None
    df = pd.read_csv(files[0])
    print(f"[ULB] {os.path.basename(files[0])}: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    ycol = "Class"
    if ycol not in df.columns:
        print(f"[ULB] 未找到标签列，现有列：{list(df.columns)[:8]}")
        return None
    print(f"[ULB] 正例率 {df[ycol].mean():.4%}（正例 {int(df[ycol].sum())} 笔）")
    df = df.sort_values("Time").reset_index(drop=True)
    n = int(len(df) * 0.7)
    tr, te = df.iloc[:n], df.iloc[n:]
    cols = [c for c in df.columns if c not in (ycol, "Time")]
    return tr[cols], tr[ycol].values, te[cols], te[ycol].values


def run_ibm_aml():
    """IBM AML（反洗钱）交易图：账户为节点、转账为边，用于验证图方法。

    数据含 From/To 账户、金额、币种、支付方式与洗钱标签，
    可同时做「表格模型」与「图模型（账户关联）」对比。
    """
    d = os.path.join(RAW, "ibm_aml")
    if not glob.glob(os.path.join(d, "**", "*Trans*.csv"), recursive=True):
        if not kaggle_download("dataset",
                               "ealtman2019/ibm-transactions-for-anti-money-laundering-aml", d):
            return None
    files = glob.glob(os.path.join(d, "**", "*Trans*.csv"), recursive=True)
    if not files:
        print("[IBM-AML] 未找到交易文件")
        return None
    f = [x for x in files if "HI-Small" in x] or files
    df = pd.read_csv(f[0])
    print(f"[IBM-AML] {os.path.basename(f[0])}: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    lc = [c for c in df.columns if "Laundering" in c]
    if not lc:
        print(f"[IBM-AML] 未找到标签列，现有列：{list(df.columns)}")
        return None
    ycol = lc[0]
    df = df.rename(columns={ycol: "y"})
    df["ts"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M", errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    print(f"[IBM-AML] 正例率 {df['y'].mean():.4%}（正例 {int(df['y'].sum()):,} 笔）")
    n = int(len(df) * 0.7)
    tr, te = df.iloc[:n], df.iloc[n:]
    num = [c for c in df.columns if c.startswith("Amount")]
    cat = [c for c in df.columns if c in ("Payment Format", "Receiving Currency", "Payment Currency")]
    feat = num + cat
    Xtr = pd.get_dummies(tr[feat], columns=cat, drop_first=True)
    Xte = pd.get_dummies(te[feat], columns=cat, drop_first=True).reindex(columns=Xtr.columns, fill_value=0)
    return Xtr, tr["y"].values, Xte, te["y"].values


def run_models(tag, Xtr, ytr, Xte, yte) -> list:
    """三档模型 + 不平衡处理，输出评估结果。"""
    pos_w = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    models = {
        "逻辑回归": Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=3000, C=0.5,
                                                         class_weight="balanced"))]),
        "随机森林": Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=4,
                                                             max_depth=12, class_weight="balanced_subsample",
                                                             n_jobs=-1, random_state=SEED))]),
        "LightGBM（类别权重）": Pipeline([("imp", SimpleImputer(strategy="median")),
                                        ("clf", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                                                   num_leaves=31, scale_pos_weight=pos_w,
                                                                   subsample=0.85, subsample_freq=1,
                                                                   colsample_bytree=0.8, random_state=SEED,
                                                                   n_jobs=-1, verbose=-1))]),
        "LightGBM（默认）": Pipeline([("imp", SimpleImputer(strategy="median")),
                                    ("clf", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                                               num_leaves=31, random_state=SEED,
                                                               n_jobs=-1, verbose=-1))]),
    }
    out = []
    for nm, mdl in models.items():
        mdl.fit(Xtr, ytr)
        s = mdl.predict_proba(Xte)[:, 1]
        r = evaluate(tag, yte, s)
        r["model"] = nm
        out.append(r)
        print(f"  {tag} | {nm:20} AUC={r['auc']:.4f} AUPRC={r['auprc']:.4f} "
              f"KS={r['ks']:.4f} 前5%命中={r['top5pct_precision']:.3f} 提升={r['lift']}x")
    return out


def main():
    results = []
    for tag, loader in [("BAF(NeurIPS2022)", run_baf),
                        ("ULB信用卡欺诈", run_ulb)]:
        print(f"\n=== {tag} ===")
        try:
            loaded = loader()
        except Exception as e:  # noqa: BLE001
            print(f"[{tag}] 加载失败：{type(e).__name__}: {str(e)[:200]}")
            continue
        if loaded is None:
            print(f"[{tag}] 跳过（数据不可用）")
            continue
        Xtr, ytr, Xte, yte = loaded
        print(f"[{tag}] 训练 {len(ytr):,}（正例 {int(ytr.sum()):,}）｜"
              f"测试 {len(yte):,}（正例 {int(yte.sum()):,}）｜特征 {Xtr.shape[1]}")
        results.extend(run_models(tag, Xtr, ytr, Xte, yte))

    if results:
        json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n[save] {OUT}")
    else:
        print("\n未产生结果：请先配置 Kaggle 凭据并确认可访问上述数据集。")


if __name__ == "__main__":
    main()
