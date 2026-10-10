# -*- coding: utf-8 -*-
"""
02_features.py
从订单流水与资金流水中构建商户级特征（申贷前 90 天美化窗口为主视角），
覆盖客户结构、金额形态、时间节律、商品集中度、资金流水五类共 30 余项指标。

输出：data/processed/features.csv
"""
import json
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
WINDOW_DAYS = 90
MF_MAP = {int(k): v for k, v in json.load(open(os.path.join(PROC, "official_baseline.json"),
                                               encoding="utf-8"))["season_factor_by_calendar_month"].items()}


def entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main():
    tx = pd.read_csv(os.path.join(PROC, "transactions.csv.gz"), parse_dates=["ts"])
    fl = pd.read_csv(os.path.join(PROC, "fund_flows.csv.gz"), parse_dates=["ts"])
    mc = pd.read_csv(os.path.join(PROC, "merchants.csv"), parse_dates=["apply_date", "window_start"])

    tx = tx.merge(mc[["merchant_id", "window_start", "apply_date"]], on="merchant_id", how="left")
    fl = fl.merge(mc[["merchant_id", "window_start"]], on="merchant_id", how="left")
    tx["in_window"] = tx["ts"] >= tx["window_start"]
    fl["in_window"] = fl["ts"] >= fl["window_start"]
    # 季节因子（由公开数据拟合的月度季节性指数），用于「同口径」增速计算
    tx["mf"] = tx["ts"].dt.month.map(MF_MAP)

    # 客户首次出现时间（全量历史），用于识别窗口内「新客」
    first_seen = tx.groupby(["merchant_id", "customer_id"])["ts"].min().rename("first_ts").reset_index()
    tx = tx.merge(first_seen, on=["merchant_id", "customer_id"], how="left")
    tx["is_new_cust"] = tx["first_ts"] >= tx["window_start"]

    win = tx[tx["in_window"]].copy()
    tmin = mc.set_index("merchant_id")["window_start"]
    tx["prior_window"] = (tx["ts"] >= tx["window_start"] - pd.Timedelta(days=WINDOW_DAYS)) & (~tx["in_window"])
    prior = tx[tx["prior_window"]]
    fwin = fl[fl["in_window"]].copy()

    g = win.groupby("merchant_id")
    f = pd.DataFrame(index=mc["merchant_id"])

    # ---- 1. 规模与增长 ----
    f["win_orders"] = g.size()
    f["win_gmv"] = g["amount"].sum()
    f["win_active_days"] = g["ts"].apply(lambda s: s.dt.normalize().nunique())
    f["orders_per_active_day"] = (f["win_orders"] / f["win_active_days"]).round(4)
    pr = prior.groupby("merchant_id")
    # 增速与占比均采用「同口径」：先除以各自季节因子，剔除季节性造成的不可比
    win_sf = win.groupby("merchant_id")["mf"].mean()
    pr_sf = prior.groupby("merchant_id")["mf"].mean()
    all_sf = tx.groupby("merchant_id")["mf"].mean()
    f["season_factor"] = win_sf
    f["order_growth"] = ((f["win_orders"] / win_sf) / (pr.size() / pr_sf) - 1)
    f["gmv_growth"] = ((f["win_gmv"] / win_sf) / (pr["amount"].sum() / pr_sf) - 1)
    tot = tx.groupby("merchant_id")["amount"].agg(["sum", "size"])
    f["win_gmv_share_of_year"] = (f["win_gmv"] / win_sf) / (tot["sum"] / all_sf)

    # ---- 2. 客户结构 ----
    cust_orders = g["customer_id"].agg(["nunique", "size"])
    f["n_customers"] = cust_orders["nunique"]
    f["orders_per_customer"] = f["win_orders"] / f["n_customers"]
    per_cust = win.groupby(["merchant_id", "customer_id"]).size()
    hhi = per_cust.groupby("merchant_id").apply(lambda s: float(((s / s.sum()) ** 2).sum()))
    f["cust_hhi"] = hhi
    f["top5_cust_share"] = per_cust.groupby("merchant_id").apply(
        lambda s: float(np.sort(s.values)[::-1][:5].sum() / s.sum()))
    f["new_cust_share"] = g["is_new_cust"].mean()
    f["oneshot_new_share"] = win.assign(oneshot=win["is_new_cust"] & (win.groupby(
        ["merchant_id", "customer_id"])["customer_id"].transform("size") == 1)).groupby("merchant_id")["oneshot"].mean()
    f["new_cust_big_share"] = win.assign(big=win["is_new_cust"] & (win["amount"] >= 3000)).groupby("merchant_id")["big"].mean()

    # ---- 3. 金额形态 ----
    f["amt_mean"] = g["amount"].mean()
    f["amt_median"] = g["amount"].median()
    f["amt_cv"] = (g["amount"].std() / g["amount"].mean())
    f["big_order_share"] = win.assign(big=win["amount"] >= 3000).groupby("merchant_id")["big"].mean()
    f["round100_share"] = win.assign(r=(win["amount"] % 100 == 0)).groupby("merchant_id")["r"].mean()
    f["round1000_share"] = win.assign(r=(win["amount"] % 1000 == 0)).groupby("merchant_id")["r"].mean()
    cnt = win.groupby(["merchant_id", "amount"]).size().rename("c").reset_index()
    cnt["dup"] = cnt["c"] >= 3
    f["dup_amt_share"] = cnt.groupby("merchant_id").apply(
        lambda d: float(d.loc[d["dup"], "c"].sum() / d["c"].sum()))
    f["big_round_count"] = win.assign(br=(win["amount"] >= 1000) & (win["amount"] % 1000 == 0)).groupby("merchant_id")["br"].sum()
    f["max_order_amount"] = g["amount"].max()
    f["amt_entropy"] = g["amount"].apply(lambda s: entropy(np.histogram(
        np.log10(np.clip(s, 1, None)), bins=24, range=(0, 5))[0].astype(float)))

    # ---- 4. 时间节律 ----
    f["night_share"] = win.assign(n=(win["ts"].dt.hour < 6)).groupby("merchant_id")["n"].mean()
    f["weekend_share"] = win.assign(w=(win["ts"].dt.dayofweek >= 5)).groupby("merchant_id")["w"].mean()
    f["hour_entropy"] = g["ts"].apply(lambda s: entropy(np.bincount(s.dt.hour, minlength=24).astype(float)))
    win_sorted = win.sort_values(["merchant_id", "ts"])
    iat = win_sorted.groupby("merchant_id")["ts"].diff().dt.total_seconds()
    win_sorted = win_sorted.assign(iat=iat)
    gg = win_sorted.groupby("merchant_id")["iat"]
    f["iat_cv"] = gg.apply(lambda s: float(s.std() / s.mean()) if s.mean() and s.notna().sum() > 2 else np.nan)
    short = win_sorted[(win_sorted["iat"].notna()) & (win_sorted["iat"] < 600)]
    f["burst_share"] = (short.groupby("merchant_id").size() / f["win_orders"]).fillna(0.0)
    f["batch_interval_regularity"] = short.assign(bucket=(short["iat"] / 30).round()).groupby(
        "merchant_id")["bucket"].apply(lambda s: float(s.value_counts().iloc[0] / len(s)) if len(s) >= 5 else np.nan)
    daily = win.groupby(["merchant_id", win["ts"].dt.normalize()]).size()
    f["max_daily_share"] = daily.groupby("merchant_id").max() / f["win_orders"]
    f["daily_cv"] = daily.groupby("merchant_id").apply(lambda s: float(s.std() / s.mean()) if s.mean() else np.nan)

    # ---- 5. 商品集中度 ----
    sku = win.groupby(["merchant_id", "stock_code"]).size()
    f["sku_hhi"] = sku.groupby("merchant_id").apply(lambda s: float(((s / s.sum()) ** 2).sum()))
    f["top_sku_share"] = sku.groupby("merchant_id").apply(lambda s: float(s.max() / s.sum()))
    f["channel_hhi"] = win.groupby(["merchant_id", "channel"]).size().groupby("merchant_id").apply(
        lambda s: float(((s / s.sum()) ** 2).sum()))

    # ---- 6. 资金流水 ----
    fg = fwin.groupby("merchant_id")
    inflow = fwin[fwin["direction"] == "进"].groupby("merchant_id")["amount"].sum()
    outflow = fwin[fwin["direction"] == "出"].groupby("merchant_id")["amount"].sum()
    f["flow_in_total"] = inflow
    f["flow_out_total"] = outflow
    f["flow_to_gmv_ratio"] = inflow / f["win_gmv"]
    f["net_inflow_ratio"] = ((inflow - outflow) / inflow).replace([np.inf, -np.inf], np.nan)
    f["round_transfer_share"] = fwin.assign(r=(fwin["amount"] % 1000 == 0)).groupby("merchant_id")["r"].mean()
    f["flow_cp_count"] = fwin.groupby("merchant_id")["counterparty_id"].nunique()
    f["flow_cp_hhi"] = fwin.groupby(["merchant_id", "counterparty_id"])["amount"].sum().groupby(
        "merchant_id").apply(lambda s: float(((s / s.sum()) ** 2).sum()))

    # 当日进出配对率：进账笔中，能在同日找到金额接近（±5%）的出账
    ins = fwin[fwin["direction"] == "进"][["merchant_id", "ts", "amount"]].rename(columns={"amount": "amt_in"})
    outs = fwin[fwin["direction"] == "出"][["merchant_id", "ts", "amount"]].rename(columns={"amount": "amt_out"})
    ins["d"] = ins["ts"].dt.normalize(); outs["d"] = outs["ts"].dt.normalize()
    ins["in_id"] = np.arange(len(ins))
    m = ins.merge(outs, on=["merchant_id", "d"], how="left", suffixes=("", "_o"))
    m["matched"] = (np.abs(m["amt_out"] - m["amt_in"]) / m["amt_in"] <= 0.05).fillna(False)
    matched_in = m.groupby(["merchant_id", "in_id"])["matched"].any().reset_index()
    f["same_day_pair_rate"] = matched_in.groupby("merchant_id")["matched"].mean()
    big_in = ins[ins["amt_in"] >= 10000]
    m2 = big_in.merge(outs, on=["merchant_id", "d"], how="left")
    m2["matched"] = (np.abs(m2["amt_out"] - m2["amt_in"]) / m2["amt_in"] <= 0.03).fillna(False)
    matched_big = m2.groupby(["merchant_id", "in_id"])["matched"].any().reset_index()
    f["big_same_day_pair_rate"] = matched_big.groupby("merchant_id")["matched"].mean()

    feat = f.reset_index().rename(columns={"index": "merchant_id"})
    feat = mc[["merchant_id", "industry", "city_tier", "months_operating", "apply_date",
               "is_noncompliant", "fraud_modes", "n_fraud_modes", "fraud_intensity",
               "visible_fraud", "hard_negative", "hn_type", "data_split", "true_window_gmv",
               "obs_window_gmv", "gmv_inflation", "order_inflation"]].merge(feat, on="merchant_id", how="left")
    feat = feat.replace([np.inf, -np.inf], np.nan)
    # 未授权资金流水数据的商户：流水类特征整体缺失（模型需具备缺失稳健性）
    flow_cols = ["flow_in_total", "flow_out_total", "flow_to_gmv_ratio", "net_inflow_ratio",
                 "round_transfer_share", "flow_cp_count", "flow_cp_hhi", "same_day_pair_rate",
                 "big_same_day_pair_rate"]
    miss_ids = mc.loc[mc["flow_missing"] == 1, "merchant_id"]
    miss_mask = feat["merchant_id"].isin(miss_ids)
    feat.loc[miss_mask, flow_cols] = np.nan
    # 「事件型占比」特征的缺失代表窗口内未出现该类事件，统一记为 0
    zero_fill = ["dup_amt_share", "same_day_pair_rate", "big_same_day_pair_rate",
                 "burst_share", "batch_interval_regularity", "top5_cust_share",
                 "flow_cp_hhi", "round_transfer_share", "net_inflow_ratio"]
    feat[zero_fill] = feat[zero_fill].fillna(0.0)
    # 流水缺失商户保持缺失；其余商户的事件型特征按 0 处理
    feat.loc[miss_mask, flow_cols] = np.nan
    feat["order_growth"] = feat["order_growth"].clip(-1, 30)
    feat["gmv_growth"] = feat["gmv_growth"].clip(-1, 30)
    feat.to_csv(os.path.join(PROC, "features.csv"), index=False)
    print(f"[done] features: {feat.shape[0]} 行 x {feat.shape[1]} 列")
    feat_cols = [c for c in feat.columns if c not in mc.columns and c != "merchant_id"]
    print(f"  特征数: {len(feat_cols)}")
    print(feat.groupby("is_noncompliant")[["cust_hhi", "night_share", "round1000_share",
                                           "flow_to_gmv_ratio", "same_day_pair_rate",
                                           "dup_amt_share", "top5_cust_share"]].median().round(3).to_string())
    print("  缺失值 Top10:", feat[feat_cols].isna().sum().sort_values(ascending=False).head(10).to_dict())


if __name__ == "__main__":
    main()