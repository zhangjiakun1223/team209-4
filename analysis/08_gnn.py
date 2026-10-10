# -*- coding: utf-8 -*-
"""
08_gnn.py
图神经网络团伙识别模型，与表格模型在同一时间切分协议下对比。

方法借鉴
--------
1) 关系型图卷积（R-GCN / HeteroConv）：在「客户—商户」「客户—设备」多种关系上
   分别做消息传递，捕获跨实体的关联结构。
2) 注意力邻域选择（GAT，对应 CARE-GNN 的相似度感知邻居筛选思路）：
   对「客户→商户」聚合边引入注意力权重，使模型能自动降低伪装邻居（正常客户）
   的贡献、突出可疑邻居（共用设备的团伙账号）。
3) 二跳团伙信号：商户 A ← 客户 → 设备 ← 客户 → 商户 B，
   该路径只有图模型能表达，是表格特征的结构性补充。

对比设置（三组，同一时间切分）
  A. 表格模型（逻辑回归 / 随机森林 / LightGBM）—— 仅用 45 项商户特征
  B. 纯 GNN —— 仅用图结构与节点/边特征
  C. GNN + 表格特征（混合）—— 图嵌入与表格特征拼接

输出：report/gnn_results.json、report/figures/fig7_gnn_compare.png
"""
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv
import matplotlib.pyplot as plt
from viz_style import C, despine

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
FIG = os.path.join(BASE, "report", "figures")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def ks_stat(y, s) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr - fpr))


def metrics(y, s) -> dict:
    return {"auc": round(float(roc_auc_score(y, s)), 4),
            "ks": round(ks_stat(y, s), 4),
            "pr_auc": round(float(average_precision_score(y, s)), 4)}


def topk(y, s, frac=0.25) -> dict:
    k = int(np.ceil(len(s) * frac))
    idx = np.argsort(-s)[:k]
    hit = y[idx].sum()
    return {"k": k, "recall": round(float(hit / y.sum()), 4),
            "precision": round(float(hit / k), 4),
            "lift": round(float((hit / k) / y.mean()), 2)}


class GangDetector(nn.Module):
    """异构图团伙识别模型：关系卷积 + 注意力邻域选择。"""

    def __init__(self, dims: dict, hidden=32, layers=2, dropout=0.35, heads=4, tab_dim=0):
        super().__init__()
        self.dropout = dropout
        self.proj = nn.ModuleDict({k: nn.Linear(v, hidden) for k, v in dims.items()})
        self.tab_proj = nn.Linear(tab_dim, hidden) if tab_dim else None
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            conv = HeteroConv({
                # 注意力聚合：可自动压低伪装邻居的权重（CARE-GNN 思路）
                ("customer", "buys", "merchant"): GATConv(hidden, hidden // heads, heads=heads,
                                                          edge_dim=5, add_self_loops=False),
                ("merchant", "rev_buys", "customer"): GATConv(hidden, hidden // heads, heads=heads,
                                                              edge_dim=5, add_self_loops=False),
                ("customer", "uses", "device"): GATConv(hidden, hidden // heads, heads=heads,
                                                        edge_dim=1, add_self_loops=False),
                ("device", "rev_uses", "customer"): GATConv(hidden, hidden // heads, heads=heads,
                                                            edge_dim=1, add_self_loops=False),
                # 多关系融合：团伙关联与业务相似关系直接在商户层传递（PromoGuardian 思路）
                ("merchant", "gang_link", "merchant"): GATConv(hidden, hidden // heads, heads=heads,
                                                               edge_dim=2, add_self_loops=False),
                ("merchant", "co_customer", "merchant"): GATConv(hidden, hidden // heads, heads=heads,
                                                                 edge_dim=1, add_self_loops=False),
            }, aggr="sum")
            self.convs.append(conv)
            self.norms.append(nn.ModuleDict({k: nn.LayerNorm(hidden) for k in dims}))
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def encode(self, x_dict, edge_index_dict, edge_attr_dict):
        h = {k: F.relu(self.proj[k](v)) for k, v in x_dict.items()}
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, edge_index_dict, edge_attr_dict)
            h = {k: F.dropout(F.relu(norm[k](out[k])), self.dropout, self.training) if k in out else h[k]
                 for k in h}
        return h

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, tab=None, use_tab=True):
        h = self.encode(x_dict, edge_index_dict, edge_attr_dict)
        m = h["merchant"]
        if use_tab and tab is not None and self.tab_proj is not None:
            m = torch.cat([m, F.relu(self.tab_proj(tab))], dim=1)
        else:
            m = torch.cat([m, torch.zeros_like(m)], dim=1)
        return self.head(m).squeeze(-1)


def run_gnn(data: HeteroData, tab: np.ndarray, use_tab: bool, seed=SEED, epochs=300):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dims = {k: data[k].x.shape[1] for k in ["merchant", "customer", "device"]}
    tab_dim = tab.shape[1]
    # 表格特征先做标准化并降维（避免维度压制图嵌入）
    from sklearn.preprocessing import StandardScaler
    tab_s = StandardScaler().fit_transform(tab)
    tab_t = torch.tensor(tab_s, dtype=torch.float32)

    model = GangDetector(dims, hidden=32, layers=2, dropout=0.35, tab_dim=tab_dim)

    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-3)
    y = data["merchant"].y
    split = data["merchant"].split
    tr = split == 0
    te = split == 1
    pos_weight = torch.tensor([(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    xd = {k: data[k].x for k in ["merchant", "customer", "device"]}
    ei = {et: data[et].edge_index for et in data.edge_types}
    ea = {et: data[et].edge_attr for et in data.edge_types}

    best = {"auc": -1, "state": None, "epoch": 0}
    patience, bad = 40, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        logit = model(xd, ei, ea, tab_t, use_tab)
        loss = lossf(logit[tr], y[tr])
        loss.backward(); opt.step()
        if (ep + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(xd, ei, ea, tab_t, use_tab))
            a = roc_auc_score(y[tr].numpy(), p[tr].numpy())
            if a > best["auc"]:
                best = {"auc": a, "state": {k: v.clone() for k, v in model.state_dict().items()},
                        "epoch": ep}
                bad = 0
            else:
                bad += 1
                if bad >= patience // 5:
                    break
    model.load_state_dict(best["state"]); model.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(model(xd, ei, ea, tab_t, use_tab))[te].numpy()
    return p_te, best["epoch"], best["auc"]


def gang_structural_features(data: HeteroData) -> np.ndarray:
    """从图中抽取商户层的团伙结构特征（供表格模型对照，量化图结构的增量价值）。"""
    n = data["merchant"].x.shape[0]
    ei = data["merchant", "gang_link", "merchant"].edge_index.numpy()
    ea = data["merchant", "gang_link", "merchant"].edge_attr.numpy()
    deg = np.zeros(n); devs = np.zeros(n); accs = np.zeros(n)
    np.add.at(deg, ei[1], 1)
    np.add.at(devs, ei[1], ea[:, 0])
    np.add.at(accs, ei[1], ea[:, 1])
    return np.column_stack([deg, devs, accs]).astype(np.float32)


def main():
    data = HeteroData()
    data = torch.load(os.path.join(PROC, "graph.pt"), weights_only=False)
    feat = pd.read_csv(os.path.join(PROC, "features.csv"))
    mc = pd.read_csv(os.path.join(PROC, "merchants.csv"))
    drop = ["merchant_id", "industry", "city_tier", "geo_city", "apply_date", "is_noncompliant",
            "fraud_modes", "n_fraud_modes", "fraud_intensity", "visible_fraud", "hard_negative",
            "hn_type", "data_split", "true_window_gmv", "obs_window_gmv", "gmv_inflation",
            "order_inflation", "window_start"]
    tab_cols = [c for c in feat.columns if c not in drop]
    feat = feat.set_index("merchant_id").reindex(mc["merchant_id"])
    tab = np.nan_to_num(feat[tab_cols].astype(np.float32).values, nan=0.0, posinf=0.0, neginf=0.0)

    y = data["merchant"].y.numpy()
    split = data["merchant"].split.numpy()
    yte, ytr = y[split == 1], y[split == 0]
    print(f"图: 商户 {data['merchant'].x.shape[0]}｜客户 {data['customer'].x.shape[0]:,}"
          f"｜设备 {data['device'].x.shape[0]}｜客户-商户边 "
          f"{data['customer', 'buys', 'merchant'].edge_index.shape[1]:,}")
    print(f"训练 {int((split == 0).sum())} 户（正例 {int(ytr.sum())}）｜"
          f"测试 {int((split == 1).sum())} 户（正例 {int(yte.sum())}）")

    results, scores = {}, {}
    for name, use_tab in [("纯GNN（仅图结构）", False), ("GNN+表格特征（混合）", True)]:
        p, ep, auc_tr = run_gnn(data, tab, use_tab)
        scores[name] = p
        results[name] = {"test": metrics(yte, p), "top25pct": topk(yte, p, 0.25),
                         "best_epoch": ep, "train_auc": round(float(auc_tr), 4)}
        print(f"  {name}: test AUC={results[name]['test']['auc']} "
              f"KS={results[name]['test']['ks']} PR-AUC={results[name]['test']['pr_auc']}")

    # 表格模型基线（从 04_model.py 的结果复用）
    tab_res = json.load(open(os.path.join(BASE, "report", "model_results.json"), encoding="utf-8"))
    pred = pd.read_csv(os.path.join(PROC, "predictions.csv"))
    for nm, col in [("逻辑回归（表格基线）", "score_main"), ("LightGBM（表格基线）", "score_lgb"),
                    ("孤立森林（无监督基线）", "score_iso")]:
        if col in pred.columns:
            s = pred[col].values
            scores[nm] = s
            results[nm] = {"test": metrics(yte, s), "top25pct": topk(yte, s, 0.25)}

    # 表格 + 团伙结构特征：量化「图结构」相对纯表格的增量价值
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    import lightgbm as lgb
    struct = gang_structural_features(data)
    tab_aug = np.hstack([tab, struct])
    tr_m = split == 0
    for nm, mdl in [
        ("逻辑回归+团伙结构", Pipeline([("imp", SimpleImputer(strategy="median")),
                                        ("sc", StandardScaler()),
                                        ("clf", LogisticRegression(max_iter=3000, C=0.5))])),
        ("LightGBM+团伙结构", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7,
                                                 min_child_samples=10, subsample=0.85, subsample_freq=1,
                                                 colsample_bytree=0.7, reg_lambda=2.0, random_state=SEED,
                                                 n_jobs=-1, verbose=-1)),
    ]:
        mdl.fit(tab_aug[tr_m], y[tr_m])
        s = mdl.predict_proba(tab_aug[split == 1])[:, 1]
        scores[nm] = s
        results[nm] = {"test": metrics(yte, s), "top25pct": topk(yte, s, 0.25)}
        print(f"  {nm}: test AUC={results[nm]['test']['auc']} KS={results[nm]['test']['ks']}")

    # 图结构价值：仅用「设备共享」这一结构信号能识别多少团伙
    dev_multi = data["device"].x[:, 0].numpy()
    cust_dev = data["customer", "uses", "device"].edge_index
    gang_cust = set(cust_dev[0].numpy().tolist())
    cb = data["customer", "buys", "merchant"].edge_index
    gang_merch = set(cb[1][np.isin(cb[0].numpy(), list(gang_cust))].numpy().tolist())
    is_gang_merch = np.zeros(len(y), dtype=bool)
    is_gang_merch[list(gang_merch)] = True
    struct_signal = is_gang_merch.astype(float)
    results["仅设备共享结构信号"] = {"test": metrics(yte, struct_signal[split == 1]),
                                     "top25pct": topk(yte, struct_signal[split == 1], 0.25),
                                     "覆盖商户数": int(is_gang_merch.sum()),
                                     "命中率": round(float(y[is_gang_merch].mean()), 4)}
    print(f"  仅设备共享结构信号: 覆盖 {int(is_gang_merch.sum())} 户，"
          f"命中率 {results['仅设备共享结构信号']['命中率']:.1%}，"
          f"AUC={results['仅设备共享结构信号']['test']['auc']}")

    # ---- 图 7：模型对比 ----
    order = ["逻辑回归（表格基线）", "LightGBM（表格基线）", "孤立森林（无监督基线）",
             "逻辑回归+团伙结构", "LightGBM+团伙结构", "纯GNN（仅图结构）", "GNN+表格特征（混合）"]
    order = [o for o in order if o in results]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.1))

    ax = axes[0]
    vals = [results[o]["test"]["auc"] for o in order]
    cmap = {"逻辑回归（表格基线）": C["grey"], "LightGBM（表格基线）": C["normal"],
            "孤立森林（无监督基线）": "#BBBBBB", "逻辑回归+团伙结构": C["accent"],
            "LightGBM+团伙结构": C["flow"], "纯GNN（仅图结构）": C["brush"],
            "GNN+表格特征（混合）": C["fake"]}
    cols = [cmap[o] for o in order]
    ax.barh(range(len(order)), vals, color=cols, height=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6.8)
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("测试集 AUC", fontsize=8.4)
    ax.axvline(0.5, ls="--", lw=0.8, color="#999999")
    for i, v in enumerate(vals):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=7.0)
    ax.set_title("(a) 模型区分度对比", loc="left", fontsize=9.4)
    despine(ax)

    ax = axes[1]
    for o in order:
        if o not in scores:
            continue
        s = scores[o][split == 1] if len(scores[o]) == len(y) else scores[o]
        fpr, tpr, _ = roc_curve(yte, s)
        ax.plot(fpr, tpr, lw=1.4, label=f"{o.split('（')[0]} ({results[o]['test']['auc']:.3f})",
                color=cols[order.index(o)])
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#CCCCCC")
    ax.set_xlabel("假阳性率", fontsize=8.4); ax.set_ylabel("真阳性率", fontsize=8.4)
    ax.set_title("(b) ROC 曲线", loc="left", fontsize=9.4)
    ax.legend(frameon=False, fontsize=6.6, loc="lower right")
    ax.tick_params(labelsize=7.6)
    despine(ax)

    short = {"逻辑回归（表格基线）": "LR-表格", "LightGBM（表格基线）": "LGBM-表格",
             "孤立森林（无监督基线）": "孤立森林", "逻辑回归+团伙结构": "LR+结构",
             "LightGBM+团伙结构": "LGBM+结构", "纯GNN（仅图结构）": "纯GNN",
             "GNN+表格特征（混合）": "GNN+表格"}
    ax = axes[2]
    x = np.arange(len(order))
    rec = [results[o]["top25pct"]["recall"] * 100 for o in order]
    pre = [results[o]["top25pct"]["precision"] * 100 for o in order]
    ax.bar(x - 0.2, rec, width=0.4, color=C["accent"], label="召回率")
    ax.bar(x + 0.2, pre, width=0.4, color=C["fake"], label="命中率")
    for xi, (r, p) in enumerate(zip(rec, pre)):
        ax.text(xi - 0.2, r + 2, f"{r:.0f}", ha="center", fontsize=6.4)
        ax.text(xi + 0.2, p + 2, f"{p:.0f}", ha="center", fontsize=6.4)
    ax.set_xticks(x)
    ax.set_xticklabels([short[o] for o in order], fontsize=6.4, rotation=40, ha="right")
    ax.set_ylabel("比例（%）", fontsize=8.4); ax.set_ylim(0, 120)
    ax.set_title("(c) 前 25% 复核效果", loc="left", fontsize=9.4)
    ax.legend(frameon=False, fontsize=6.8, ncol=2, loc="upper center")
    despine(ax)
    fig.tight_layout(w_pad=1.8)
    fig.savefig(os.path.join(FIG, "fig7_gnn_compare.png"))
    plt.close(fig)

    out = {"graph": {"merchants": int(data["merchant"].x.shape[0]),
                     "customers": int(data["customer"].x.shape[0]),
                     "devices": int(data["device"].x.shape[0]),
                     "buy_edges": int(data["customer", "buys", "merchant"].edge_index.shape[1]),
                     "device_edges": int(data["customer", "uses", "device"].edge_index.shape[1])},
           "results": results,
           "tab_baseline_cv": {"auc_mean": tab_res["cv_auc_mean"], "auc_std": tab_res["cv_auc_std"]}}
    json.dump(out, open(os.path.join(BASE, "report", "gnn_results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps({k: v["test"] for k, v in results.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
