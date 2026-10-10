# -*- coding: utf-8 -*-
"""
07_build_graph.py
从订单流水构建异构图（商户—客户—设备），供图神经网络使用。

图结构设计
----------
节点：
  merchant  商户（分类目标，800 个）
  customer  客户（保留有复购或跨商户行为的客户，承载商户之间的连通）
  device    设备（保留被多个账号共用的设备，承载团伙信号）

边：
  (customer) -[buys]-> (merchant)   交易关系，边特征刻画该客户在该商户的行为
  (customer) -[uses]-> (device)     设备使用关系

关键点：商户 A 与商户 B 若存在「客户共用设备」的二跳路径，
       即为团伙协同的强信号——这是表格特征无法表达的结构信息。

输出：data/processed/graph.pt（PyG HeteroData）
"""
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")

WINDOW_DAYS = 90
MIN_CUST_ORDERS = 2      # 保留复购客户（度为 1 的客户对消息传递无贡献）
MIN_DEV_ACCOUNTS = 2     # 保留被多账号共用的设备（团伙信号）


def _entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def build_graph():
    tx = pd.read_csv(os.path.join(PROC, "transactions.csv.gz"), parse_dates=["ts"])
    mc = pd.read_csv(os.path.join(PROC, "merchants.csv"), parse_dates=["apply_date", "window_start"])
    feat = pd.read_csv(os.path.join(PROC, "features.csv"))

    tx = tx.merge(mc[["merchant_id", "window_start"]], on="merchant_id", how="left")
    tx["in_window"] = tx["ts"] >= tx["window_start"]
    win = tx[tx["in_window"]].copy()          # 图只基于申贷前 90 天窗口（与特征口径一致）
    print(f"[data] 窗口内订单 {len(win):,} 条")

    # ---------- 客户节点：保留有复购行为的客户 ----------
    cust_stat = win.groupby("customer_id").agg(
        n_orders=("ts", "size"),
        n_merchants=("merchant_id", "nunique"),
        n_devices=("device_id", "nunique"),
        amount_sum=("amount", "sum"),
        amount_mean=("amount", "mean"),
        night=("ts", lambda s: float((s.dt.hour < 6).mean())),
        is_fake_any=("is_fake", "max"),
    )
    # ---------- 设备节点：先识别共享设备（团伙信号），再据此补充客户节点 ----------
    dev_stat = win.groupby("device_id").agg(
        n_accounts=("customer_id", "nunique"),
        n_orders=("ts", "size"),
        n_merchants=("merchant_id", "nunique"),
    )
    dev_keep = dev_stat[dev_stat["n_accounts"] >= MIN_DEV_ACCOUNTS].index
    dev_stat = dev_stat.loc[dev_keep]
    dev_index = {d: i for i, d in enumerate(dev_stat.index)}
    print(f"[node] 设备节点 {len(dev_index):,}（共享设备，原 {win.device_id.nunique():,}）")

    # 客户保留规则：有复购行为，或使用了被多个账号共用的设备
    # （后者往往只在该商户下过 1 单，但正是团伙识别的关键证据）
    gang_cust = set(win.loc[win["device_id"].isin(dev_keep), "customer_id"].unique())
    cust_keep = set(cust_stat[cust_stat["n_orders"] >= MIN_CUST_ORDERS].index) | gang_cust
    cust_stat = cust_stat.loc[sorted(cust_keep)]
    cust_index = {c: i for i, c in enumerate(cust_stat.index)}
    print(f"[node] 客户节点 {len(cust_index):,}"
          f"（复购 {int((cust_stat['n_orders'] >= MIN_CUST_ORDERS).sum()):,}"
          f" + 团伙设备关联 {len(gang_cust):,}，原 {win.customer_id.nunique():,}）")

    # ---------- 商户节点 ----------
    merch_ids = mc["merchant_id"].tolist()
    m_index = {m: i for i, m in enumerate(merch_ids)}

    # 重复金额占比：该客户在同一商户反复用同一金额下单的比例（刷单的典型痕迹）
    cust_amt = (win[win["customer_id"].isin(cust_keep)]
                .groupby(["customer_id", "merchant_id"])["amount"]
                .agg(["size", lambda s: float((s.value_counts() >= 2).mean())])
                .rename(columns={"<lambda_0>": "dup_ratio"}))
    dup_by_cust = cust_amt.groupby("customer_id")["dup_ratio"].mean()

    # ---------- 商户节点 ----------
    merch_ids = mc["merchant_id"].tolist()
    m_index = {m: i for i, m in enumerate(merch_ids)}

    # ---------- 客户特征 ----------
    cust_feat = np.column_stack([
        np.log1p(cust_stat["n_orders"].values),
        np.log1p(cust_stat["n_merchants"].values),
        np.log1p(cust_stat["n_devices"].values),
        np.log1p(cust_stat["amount_sum"].values),
        np.log1p(cust_stat["amount_mean"].values),
        cust_stat["night"].values,
        dup_by_cust.reindex(cust_stat.index).fillna(0.0).values,
    ]).astype(np.float32)

    # ---------- 设备特征 ----------
    dev_feat = np.column_stack([
        np.log1p(dev_stat["n_accounts"].values),
        np.log1p(dev_stat["n_orders"].values),
        np.log1p(dev_stat["n_merchants"].values),
        dev_stat["n_accounts"].values / max(dev_stat["n_accounts"].max(), 1),
    ]).astype(np.float32)

    # ---------- 商户特征：沿用表格特征 ----------
    drop = ["merchant_id", "industry", "city_tier", "geo_city", "apply_date", "is_noncompliant",
            "fraud_modes", "n_fraud_modes", "fraud_intensity", "visible_fraud", "hard_negative",
            "hn_type", "data_split", "true_window_gmv", "obs_window_gmv", "gmv_inflation",
            "order_inflation", "window_start"]
    mcols = [c for c in feat.columns if c not in drop]
    feat = feat.set_index("merchant_id").reindex(merch_ids)
    merch_feat = feat[mcols].astype(np.float32).values
    merch_feat = np.nan_to_num(merch_feat, nan=0.0, posinf=0.0, neginf=0.0)
    # 标签与切分以 merchants.csv 为准（features.csv 可能滞后）
    label = mc.set_index("merchant_id").reindex(merch_ids)

    # ---------- 边：客户—商户 ----------
    ce = win[win["customer_id"].isin(cust_keep)].groupby(["customer_id", "merchant_id"]).agg(
        n_orders=("ts", "size"),
        amount_sum=("amount", "sum"),
        night=("ts", lambda s: float((s.dt.hour < 6).mean())),
        dup_ratio=("amount", lambda s: float((s.value_counts() >= 2).mean())),
        span_days=("ts", lambda s: float((s.max() - s.min()).days)),
    ).reset_index()
    ce["cust_i"] = ce["customer_id"].map(cust_index)
    ce["merch_i"] = ce["merchant_id"].map(m_index)
    ce = ce.dropna(subset=["cust_i", "merch_i"])
    buy_edge_index = torch.tensor(ce[["cust_i", "merch_i"]].values.T.astype(np.int64))
    buy_edge_attr = np.column_stack([
        np.log1p(ce["n_orders"].values),
        np.log1p(ce["amount_sum"].values),
        ce["night"].values,
        ce["dup_ratio"].values,
        np.log1p(ce["span_days"].values),
    ]).astype(np.float32)

    # ---------- 边：客户—设备 ----------
    de = win[win["device_id"].isin(dev_keep) & win["customer_id"].isin(cust_keep)] \
        .groupby(["customer_id", "device_id"]).size().reset_index(name="n")
    de["cust_i"] = de["customer_id"].map(cust_index)
    de["dev_i"] = de["device_id"].map(dev_index)
    de = de.dropna(subset=["cust_i", "dev_i"])
    use_edge_index = torch.tensor(de[["cust_i", "dev_i"]].values.T.astype(np.int64))
    use_edge_attr = np.log1p(de["n"].values).astype(np.float32).reshape(-1, 1)

    # ---------- 边：商户—商户（多关系融合）----------
    # 关系 1「团伙关联」：两商户的客户共用同一设备（欺诈协同的强信号）
    # 关系 2「业务相似」：两商户共享较多客户（正常的经营相似性，用作对照）
    merch_of = ce.set_index("cust_i")["merch_i"].groupby(level=0).apply(list).to_dict()
    gang_pairs = {}
    for dev_i, grp in de.groupby("dev_i"):
        custs = grp["cust_i"].values
        ms = sorted({int(m) for c in custs for m in merch_of.get(int(c), [])})
        for a in range(len(ms)):
            for b in range(a + 1, len(ms)):
                gang_pairs.setdefault((ms[a], ms[b]), []).append(int(dev_i))
    if gang_pairs:
        gi = np.array(list(gang_pairs.keys()), dtype=np.int64)
        gdev = np.array([len(set(v)) for v in gang_pairs.values()], dtype=np.float32)
        gacc = np.array([len(v) for v in gang_pairs.values()], dtype=np.float32)
        gang_edge_index = torch.tensor(np.concatenate([gi, gi[:, ::-1]], axis=0).T)
        gang_edge_attr = torch.tensor(np.concatenate([
            np.column_stack([np.log1p(gdev), np.log1p(gacc)]),
            np.column_stack([np.log1p(gdev), np.log1p(gacc)])]).astype(np.float32))
    else:
        gang_edge_index = torch.zeros((2, 0), dtype=torch.long)
        gang_edge_attr = torch.zeros((0, 2), dtype=torch.float32)

    co = ce.groupby("merch_i")["cust_i"].apply(set).to_dict()
    pairs = {}
    mids = sorted(co.keys())
    for i, a in enumerate(mids):
        for b in mids[i + 1:]:
            n = len(co[a] & co[b])
            if n >= 30:
                pairs[(a, b)] = n
    if pairs:
        pi = np.array(list(pairs.keys()), dtype=np.int64)
        pn = np.array([pairs[k] for k in pairs], dtype=np.float32)
        co_edge_index = torch.tensor(np.concatenate([pi, pi[:, ::-1]], axis=0).T)
        co_edge_attr = torch.tensor(np.concatenate([
            np.log1p(pn).reshape(-1, 1), np.log1p(pn).reshape(-1, 1)]).astype(np.float32))
    else:
        co_edge_index = torch.zeros((2, 0), dtype=torch.long)
        co_edge_attr = torch.zeros((0, 1), dtype=torch.float32)
    print(f"[edge] 商户—商户 团伙关联 {gang_edge_index.shape[1] // 2:,} 对"
          f"｜业务相似 {co_edge_index.shape[1] // 2:,} 对")

    # ---------- 组装 HeteroData ----------
    data = HeteroData()
    data["merchant"].x = torch.tensor(merch_feat)
    data["customer"].x = torch.tensor(cust_feat)
    data["device"].x = torch.tensor(dev_feat)

    data["customer", "buys", "merchant"].edge_index = buy_edge_index
    data["customer", "buys", "merchant"].edge_attr = torch.tensor(buy_edge_attr)
    data["merchant", "rev_buys", "customer"].edge_index = buy_edge_index.flip(0)
    data["merchant", "rev_buys", "customer"].edge_attr = torch.tensor(buy_edge_attr)

    data["customer", "uses", "device"].edge_index = use_edge_index
    data["customer", "uses", "device"].edge_attr = torch.tensor(use_edge_attr)
    data["device", "rev_uses", "customer"].edge_index = use_edge_index.flip(0)
    data["device", "rev_uses", "customer"].edge_attr = torch.tensor(use_edge_attr)

    data["merchant", "gang_link", "merchant"].edge_index = gang_edge_index
    data["merchant", "gang_link", "merchant"].edge_attr = gang_edge_attr
    data["merchant", "co_customer", "merchant"].edge_index = co_edge_index
    data["merchant", "co_customer", "merchant"].edge_attr = co_edge_attr

    data["merchant"].y = torch.tensor(label["is_noncompliant"].values.astype(np.float32))
    data["merchant"].split = torch.tensor(
        (label["data_split"] == "test").values.astype(np.int64))
    data["merchant"].visible_fraud = torch.tensor(label["visible_fraud"].values.astype(np.float32))

    print(f"[edge] 客户—商户 {buy_edge_index.shape[1]:,} 条｜客户—设备 {use_edge_index.shape[1]:,} 条")
    print(f"[graph] {data}")
    torch.save(data, os.path.join(PROC, "graph.pt"))
    print(f"[save] {os.path.join(PROC, 'graph.pt')}")
    return data


if __name__ == "__main__":
    build_graph()
