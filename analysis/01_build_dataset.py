# -*- coding: utf-8 -*-
"""
01_build_dataset.py
构造「小微经营主体授信流水」场景数据集。

设计思路
--------
1) 官方口径基线（2026 年，详见 00_official_baseline.py）：
   - 笔均金额锚定《2026年第二季度支付体系运行总体情况》（银行卡消费 307 元、
     非银行支付机构网络支付 235 元），并按行业结构系数派生行业笔均；
   - 月度季节性锚定国家统计局 2025年9月—2026年8月社会消费品零售总额月度序列；
   - 自然增长率锚定国家统计局 2026年1—8月分品类零售增速。
   生成时按对数正态参数换算，使各行业订单金额的均值与官方笔均一致。
2) 业务真实性细节：
   - 价目表：商户按价目表成交（餐饮菜单价、零售标价），自然形成重复金额；
   - 资金流水：平台结算、供应商货款、房租工资，含整数化结算习惯；
   - 备付金/退款等当日进出：合规商户亦会自然出现。
3) 非规范行为注入（申贷前 90 天美化窗口，按强度分低/中/高三档）：
   - 刷单：少量固定账号、爆款价位重复下单、夜间批量、间隔规律
   - 虚构交易：整数大额、窗口内新建一次性客户
   - 伪造流水：整数进出、当日配对回流、流水收入比异常
4) 困难负样本（合规但形态可疑）：批发大额整数型、夜间经营型、整数结算型、直播促销型。

输出（data/processed/）：
  merchants.csv    商户画像与标签（含真实值/观测值对照，用于应用验证）
  transactions.csv.gz  订单级流水（is_fake 仅用于效果评估，不进入模型）
  fund_flows.csv.gz    资金流水
  dataset_config.json  生成参数与随机种子（可复现）
"""
import json
import math
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
OFFICIAL = json.load(open(os.path.join(PROC, "official_baseline.json"), encoding="utf-8"))
ANCHORS = OFFICIAL["industry_anchors"]

SEED = 42
rng = np.random.default_rng(SEED)

N_MERCHANTS = 800          # 经营主体数量（约 120 万条订单，适配普通笔记本运行）
FAKE_RATIO = 0.20          # 非规范商户占比
HARD_NEG_RATIO = 0.18      # 合规但形态可疑的困难负样本占比
WINDOW_DAYS = 90           # 申贷前美化窗口 = 90 天
HISTORY_DAYS = 365         # 观测历史 12 个月
APPLY_START = pd.Timestamp("2026-01-01")
APPLY_END = pd.Timestamp("2026-08-31")
SPLIT_DATE = pd.Timestamp("2026-06-01")   # 之前为训练集（申请较早），之后为测试集
MISSING_FLOW_RATIO = 0.04  # 未授权资金流水数据的商户占比（真实数据接入缺失）


def to_median(mean: float, sigma: float) -> float:
    """对数正态：由均值换算中位数（mean = exp(mu + sigma^2/2)）。"""
    return mean / math.exp(sigma ** 2 / 2)


# 行业画像：笔均（官方锚点派生）、日均订单量、复购客群规模、金额离散度、夜间倾向、周末倾向、自然增速
INDUSTRIES = {
    "电商零售": dict(lam=3.5, pool=160, night=0.010, wkend=1.18, sku=90),
    "批发供货": dict(lam=1.2, pool=35,  night=0.004, wkend=0.45, sku=45),
    "餐饮服务": dict(lam=6.0, pool=120, night=0.020, wkend=1.35, sku=60),
    "生活服务": dict(lam=2.4, pool=90,  night=0.008, wkend=1.25, sku=40),
}
for _k, _v in INDUSTRIES.items():
    _a = ANCHORS[_k]
    _v["sigma"] = _a["sigma"]
    _v["mean_amt"] = _a["mean_amt"]
    _v["med_amt"] = to_median(_a["mean_amt"], _a["sigma"])
    _v["growth"] = _a["growth"]

IND_WEIGHTS = [0.30, 0.18, 0.32, 0.20]
CITY_TIERS = ["一线", "新一线", "二线", "三线及以下"]
CITY_W = [0.18, 0.26, 0.31, 0.25]

# ---------------- 图结构相关参数 ----------------
N_POOL_CUSTOMERS = 60000          # 全局真实消费者池规模（支撑客户—商户—设备网络）
CITY_NAMES = ["A", "B", "C", "D", "E", "F"]   # 地理分区（简化命名，避免暴露真实城市）
GEO_W = [0.28, 0.24, 0.18, 0.14, 0.10, 0.06]
LOCALITY_LEAK = 0.10              # 异地消费的权重折减（模拟就近消费）
FRAUD_ACC_PER_MERCHANT = (3, 18)  # 单个非规范商户的刷单账号数量区间
FRAUD_DEVICES_PER_GROUP = (2, 6)  # 单个刷单团伙共用设备数量区间
CAMOUFLAGE_RATIO = 0.35           # 刷单账号在其他商户做正常消费的比例（伪装）
GANG_RATIO = 0.45                 # 非规范商户中属于「跨商户团伙」的比例
GANG_SIZE = (2, 5)                # 单个团伙包含的商户数区间
CHANNELS = {
    "电商零售": ["App", "小程序", "PC商城", "短视频平台"],
    "批发供货": ["微信订货", "电话订单", "ERP对接"],
    "餐饮服务": ["扫码点餐", "外卖平台", "收银POS"],
    "生活服务": ["门店小程序", "到店POS", "电话预约"],
}


def _expand_hour_weights() -> np.ndarray:
    """把官方未公开的日内分布假设（分组权重）展开为 24 小时权重。"""
    groups = {
        "0-6": (range(0, 6), 0.006), "7-8": (range(7, 9), 0.030), "9-11": (range(9, 12), 0.250),
        "12-13": (range(12, 14), 0.200), "14-17": (range(14, 18), 0.250),
        "18-19": (range(18, 20), 0.160), "20-22": (range(20, 23), 0.100), "23": ([23], 0.004),
    }
    w = np.zeros(24, dtype=float)
    for hours, weight in groups.values():
        hours = list(hours)
        for h in hours:
            w[h] = weight / len(hours)
    return w / w.sum()


hour_p = _expand_hour_weights()
# 星期节律（官方未公开，采用零售业通行的周内分布假设：周末略高于工作日）
_DOW_BASE = np.array([1.00, 1.00, 1.02, 1.03, 1.06, 1.08, 1.00])
dow_p = _DOW_BASE / _DOW_BASE.sum()
month_f = {int(k): v for k, v in OFFICIAL["season_factor_by_calendar_month"].items()}

FAKE_TYPES = ["刷单", "虚构交易", "伪造流水"]
# 注入强度：倍数、金额抖动、时间抖动、夜间占比、整数化程度
INTENSITY = {
    "低": dict(mult=0.12, amt_jit=0.06, gap_jit=0.45, night_ratio=0.30, round_lvl=0.55),
    "中": dict(mult=0.45, amt_jit=0.02, gap_jit=0.18, night_ratio=0.75, round_lvl=0.85),
    "高": dict(mult=1.15, amt_jit=0.005, gap_jit=0.07, night_ratio=1.00, round_lvl=1.00),
}


def industry_hour_profile(ind: str) -> np.ndarray:
    p = hour_p.copy()
    prof = INDUSTRIES[ind]
    if ind in ("餐饮服务", "生活服务"):
        for h, w in [(11, 1.6), (12, 1.5), (17, 2.0), (18, 2.2), (19, 2.2), (20, 1.8), (21, 1.4)]:
            p[h] *= w
    elif ind == "电商零售":
        for h, w in [(20, 2.5), (21, 2.6), (22, 2.2), (23, 1.6)]:
            p[h] *= w
    if prof["night"] > 0.015:   # 夜宵/便利店等夜间经营
        for h in list(range(0, 6)) + [23]:
            p[h] = max(p[h], 1e-4) + prof["night"] / 7
    p = np.where(np.arange(24) < 6, p * max(prof["night"] / 0.010, 0.2), p)
    return p / p.sum()


def industry_dow_profile(ind: str) -> np.ndarray:
    base = dow_p.copy()
    base = np.where(np.arange(7) >= 5, base.mean(), base)
    tilt = np.array([1.0, 1.0, 1.0, 1.0, 1.05, INDUSTRIES[ind]["wkend"], INDUSTRIES[ind]["wkend"]])
    p = base * tilt
    return p / p.sum()


def month_factor_of(dates: pd.DatetimeIndex) -> np.ndarray:
    return np.array([month_f[d.month] for d in dates], dtype=float)


def build_price_list(med: float, n: int, round_lvl: float = 0.0) -> np.ndarray:
    """商户价目表：餐饮菜单价、零售标价等，形成自然重复金额；round_lvl 控制整数化程度。"""
    prices = med * np.exp(rng.normal(0, 0.55, n))
    if round_lvl > 0:   # 部分商户习惯以整数定价（如批发报价）
        mask = rng.random(n) < round_lvl
        prices[mask] = np.round(prices[mask], -int(rng.choice([0, 1, 2])))
    return np.round(np.clip(prices, 1.0, med * 60), 2)


def build_merchant_population() -> pd.DataFrame:
    inds = rng.choice(list(INDUSTRIES), N_MERCHANTS, p=IND_WEIGHTS)
    city = rng.choice(CITY_TIERS, N_MERCHANTS, p=CITY_W)
    geo_city = rng.choice(len(CITY_NAMES), N_MERCHANTS, p=GEO_W)   # 商户所在地理分区
    months_op = rng.integers(12, 241, N_MERCHANTS)
    offs = rng.integers(0, (APPLY_END - APPLY_START).days + 1, N_MERCHANTS)
    apply_dates = (APPLY_START + pd.to_timedelta(offs, unit="D")).normalize()

    is_fake = rng.random(N_MERCHANTS) < FAKE_RATIO
    is_hn = (~is_fake) & (rng.random(N_MERCHANTS) < HARD_NEG_RATIO)
    hn_type = np.where(is_hn, rng.choice(["批发大额整数型", "夜间经营型", "整数结算型", "直播促销型"],
                                         N_MERCHANTS, p=[0.22, 0.24, 0.22, 0.32]), "")
    intensity = np.where(is_fake, rng.choice(["低", "中", "高"], N_MERCHANTS, p=[0.32, 0.43, 0.25]), "")
    # 约 1/4 非规范行为发生在观察窗口之外或幅度低于可观测水平（不可见 → 检出率上界）
    visible = np.where(is_fake, rng.random(N_MERCHANTS) > 0.25, True)

    modes, n_modes = [], []
    for i in range(N_MERCHANTS):
        if not is_fake[i]:
            modes.append(""); n_modes.append(0); continue
        k = int(rng.choice([1, 1, 2, 3], p=[0.34, 0.18, 0.28, 0.20]))
        modes.append(";".join(sorted(rng.choice(FAKE_TYPES, size=k, replace=False)))); n_modes.append(k)

    # 团伙编组：部分非规范商户组成「跨商户刷单团伙」，共用同一批账号与设备。
    # 这是图方法真正有价值的场景——单个商户看只是少量可疑账号（表格特征难以察觉），
    # 但多个商户共享账号/设备的结构只有图能刻画。
    gang_id = np.full(N_MERCHANTS, -1, dtype=int)
    cand = np.where(is_fake)[0].copy()
    rng.shuffle(cand)
    gi, i = 0, 0
    while i < len(cand):
        size = int(rng.integers(*GANG_SIZE))
        if rng.random() < GANG_RATIO:
            gang_id[cand[i:i + size]] = gi
            gi += 1
        i += size
    # 团伙成员必然存在刷单行为（否则团伙账号池无法在多个商户间形成重叠）
    for i in np.where(gang_id >= 0)[0]:
        ms = set(modes[i].split(";")) if modes[i] else set()
        ms.add("刷单")
        modes[i] = ";".join(sorted(ms))
        n_modes[i] = len(ms)

    df = pd.DataFrame(dict(
        merchant_id=[f"M{i+1:05d}" for i in range(N_MERCHANTS)],
        industry=inds, city_tier=city, geo_city=geo_city,
        months_operating=months_op, apply_date=apply_dates,
        is_noncompliant=is_fake.astype(int), fraud_modes=modes, n_fraud_modes=n_modes,
        fraud_intensity=intensity, visible_fraud=visible.astype(int),
        fraud_gang=gang_id,
        hard_negative=is_hn.astype(int), hn_type=hn_type,
        scale=np.exp(rng.normal(0, 0.38, N_MERCHANTS)),
        growth=np.array([rng.normal(INDUSTRIES[i]["growth"], 0.18) for i in inds]),
        new_cust_ratio=rng.uniform(0.18, 0.40, N_MERCHANTS),
        flow_missing=(rng.random(N_MERCHANTS) < MISSING_FLOW_RATIO).astype(int),
    ))
    df["data_split"] = np.where(df["apply_date"] < SPLIT_DATE, "train", "test")
    return df


def build_customer_universe():
    """构建全局真实消费者池：活跃度（幂律）、行业亲和、地理分区、设备绑定。

    这是让图结构成立的关键：真实消费者会在多个商户重复消费，
    从而把不同商户连成一张有意义的网络；设备实体则承载团伙欺诈信号。
    """
    n = N_POOL_CUSTOMERS
    activity = rng.pareto(1.35, n) + 0.35                    # 少数高活跃客户贡献多数订单
    affinity = rng.random((n, len(INDUSTRIES))) ** 2.5       # 行业偏好稀疏（多数行业低亲和）
    for j in range(n):                                        # 每个客户强化 1—3 个偏好行业
        k = int(rng.integers(1, 4))
        idx = rng.choice(len(INDUSTRIES), k, replace=False)
        affinity[j, idx] += rng.uniform(1.5, 3.5)
    city = rng.choice(len(CITY_NAMES), n, p=GEO_W)
    n_dev = rng.choice([1, 1, 1, 2], n)                      # 多数人 1 台设备
    dev_ptr = np.concatenate([[0], np.cumsum(n_dev)]).astype(np.int64)
    device_flat = np.array([f"D{i:07d}" for i in range(int(n_dev.sum()))], dtype=object)
    return dict(n=n, activity=activity, affinity=affinity, city=city,
                device_ptr=dev_ptr, device_flat=device_flat,
                customer_ids=np.array([f"C{i:06d}" for i in range(n)], dtype=object))


UNIVERSE = None
DEVICE_COUNTER = None
GANG_POOLS = {}


def build_gang_pools(pop: pd.DataFrame):
    """为每个跨商户团伙建立共享的账号池与设备池。"""
    global GANG_POOLS
    GANG_POOLS = {}
    for gid in sorted(set(pop.loc[pop["fraud_gang"] >= 0, "fraud_gang"])):
        n_acc = int(rng.integers(10, 26))      # 团伙共用账号
        n_dev = int(rng.integers(3, 8))        # 团伙共用设备
        GANG_POOLS[int(gid)] = {
            "accounts": [f"GT{gid:03d}A{i:03d}" for i in range(n_acc)],
            "devices": _new_device_ids(n_dev),
        }
    n_gang_merch = int((pop["fraud_gang"] >= 0).sum())
    print(f"[gang] 跨商户团伙 {len(GANG_POOLS)} 个，涉及 {n_gang_merch} 户非规范商户")


def _new_device_ids(k: int) -> np.ndarray:
    """团伙专用的共享设备 ID（独立命名空间，避免与真实设备混淆）。"""
    global DEVICE_COUNTER
    ids = np.array([f"G{DEVICE_COUNTER + i:06d}" for i in range(k)])
    DEVICE_COUNTER += k
    return ids


def _pool_weights(industry: str, city: int) -> np.ndarray:
    """给定商户的行业与所在城市，计算客户池中各客户的到店权重。"""
    ind_idx = list(INDUSTRIES).index(industry)
    w = UNIVERSE["activity"] * UNIVERSE["affinity"][:, ind_idx]
    w = w * np.where(UNIVERSE["city"] == city, 1.0, LOCALITY_LEAK)   # 地理就近消费
    return w / w.sum()


def pick_customers(n: int, industry: str, city: int, new_ratio: float):
    """返回 (客户ID, 设备ID)。池内客户按到店权重抽样并绑定其名下设备；
    少量一次性新客使用临时 ID 与临时设备（无历史、不复购）。"""
    w = _pool_weights(industry, city)
    is_new = rng.random(n) < new_ratio
    cids = np.empty(n, dtype=object)
    dids = np.empty(n, dtype=object)
    n_rep = int((~is_new).sum())
    if n_rep:
        pick = rng.choice(UNIVERSE["n"], n_rep, p=w)
        cids[~is_new] = UNIVERSE["customer_ids"][pick]
        ptr = UNIVERSE["device_ptr"]
        off = rng.integers(0, np.maximum(ptr[pick + 1] - ptr[pick], 1))
        dids[~is_new] = UNIVERSE["device_flat"][ptr[pick] + off]
    for j in np.where(is_new)[0]:
        cids[j] = f"X{int(rng.integers(1, 10**9)):09d}"
        dids[j] = f"U{int(rng.integers(1, 10**9)):09d}"
    return cids, dids


def gen_normal_orders(m: pd.Series) -> pd.DataFrame:
    """生成商户 12 个月的自然经营订单（按价目表成交 + 组合购买）。"""
    prof = INDUSTRIES[m["industry"]]
    days = pd.date_range(m["apply_date"] - pd.Timedelta(days=HISTORY_DAYS),
                         m["apply_date"] - pd.Timedelta(days=1), freq="D")
    mf = month_factor_of(days)
    dp = industry_dow_profile(m["industry"])[days.dayofweek]
    trend = (1 + m["growth"]) ** (np.arange(len(days)) / 365.0)
    lam = prof["lam"] * m["scale"] * mf * dp / (dp.mean() * mf.mean()) * trend * rng.lognormal(0, 0.12, len(days))
    counts = rng.poisson(np.clip(lam, 0.01, 80))
    n = int(counts.sum())
    if n == 0:
        return pd.DataFrame(name="", dtype=float)

    ts_day = days[np.repeat(np.arange(len(days)), counts)]
    hp = industry_hour_profile(m["industry"])
    ts = (ts_day + pd.to_timedelta(rng.choice(24, n, p=hp), unit="h")
          + pd.to_timedelta(rng.integers(0, 60, n), unit="m") + pd.to_timedelta(rng.integers(0, 60, n), unit="s"))

    # 金额：约 45% 按价目表成交（叠加会员价/促销折扣形成自然金额分布），其余为组合购买
    price_list = build_price_list(prof["med_amt"] * m["scale"], max(int(prof["sku"] // 2), 10),
                                  round_lvl=0.5 if m["industry"] == "批发供货" else 0.15)
    idx = rng.integers(0, len(price_list), n)
    on_list = rng.random(n) < 0.45
    qty = rng.choice([1, 1, 1, 2, 2, 3], n)
    discount = 1.0 if m["industry"] == "批发供货" else np.where(
        rng.random(n) < 0.5, 1 - rng.choice([0.05, 0.10, 0.15, 0.20], n), 1.0)
    amt = np.where(on_list, price_list[idx] * qty * discount,
                   np.round(np.clip(prof["med_amt"] * m["scale"] * rng.lognormal(0, prof["sigma"]), 1, None) * qty, 2))
    amt = np.round(np.clip(amt, 1.0, prof["med_amt"] * m["scale"] * 300), 2)
    skus = [f"S{price_list.tolist().index(price_list[i]) + 1:04d}" if on_list[j] else
            f"S{int(rng.integers(1, prof['sku'] + 1)):04d}" for j, i in enumerate(idx)]

    cust, dev = pick_customers(n, m["industry"], int(m["geo_city"]), m["new_cust_ratio"])
    return pd.DataFrame(dict(
        merchant_id=m["merchant_id"], ts=ts, amount=amt,
        customer_id=cust, device_id=dev,
        channel=rng.choice(CHANNELS[m["industry"]], n), stock_code=skus,
        direction="收", is_fake=0, fake_type=""))


def inject_fake_orders(m: pd.Series, win_orders_true: int) -> pd.DataFrame:
    """注入刷单与虚构交易订单（按强度分层控制隐蔽性）。"""
    prof = INDUSTRIES[m["industry"]]
    modes = m["fraud_modes"].split(";")
    P = INTENSITY[m["fraud_intensity"]]
    inten = P["mult"]
    # 中/高强度直接使用完全重复的成交价（同款同价反复下单 / 整数大额）；
    # 低强度混入价格扰动，模拟更隐蔽的操作
    jit = P["amt_jit"] if P["round_lvl"] < 0.7 else 0.0
    w_start = m["apply_date"] - pd.Timedelta(days=WINDOW_DAYS)
    rows = []

    if "刷单" in modes:
        base = max(int(win_orders_true * 0.55), 60)
        n_brush = int(base * inten * rng.uniform(0.7, 1.3))
        # 团伙模式：使用团伙共享的账号池与设备池（跨商户协作，图结构才能识别）
        gid = int(m.get("fraud_gang", -1))
        if gid >= 0 and gid in GANG_POOLS:
            pool = GANG_POOLS[gid]
            k_use = max(4, int(len(pool["accounts"]) * rng.uniform(0.45, 0.80)))
            accounts = list(rng.choice(pool["accounts"], k_use, replace=False))
            n_acc = len(accounts)
            group_devices = pool["devices"]
        else:
            n_acc = int(np.clip(rng.integers(*FRAUD_ACC_PER_MERCHANT) * (0.6 + inten),
                                FRAUD_ACC_PER_MERCHANT[0], FRAUD_ACC_PER_MERCHANT[1]))
            accounts = [f"CB{m['merchant_id'][-5:]}{i:02d}" for i in range(n_acc)]
            n_gdev = int(rng.integers(*FRAUD_DEVICES_PER_GROUP))
            n_gdev = max(2, min(n_gdev, n_acc // 2))     # 保证设备数少于账号数，必然形成共享
            group_devices = _new_device_ids(n_gdev)
        # 每个账号固定使用团伙设备池中的一台（同一设备承载多个账号）
        acc_device = {a: group_devices[i % len(group_devices)] for i, a in enumerate(accounts)}
        levels = max(2, int(rng.integers(2, 5)))
        base_p = prof["med_amt"] * m["scale"] * rng.uniform(0.75, 1.25)
        price_levels = np.round(base_p * rng.uniform(0.9, 1.1, levels), 2)
        hot_sku = f"S{int(rng.integers(1, max(prof['sku'] // 3, 2))):04d}"
        n_sess = int(np.clip(n_brush / rng.integers(4, 12), 6, 90))
        sess_days = rng.integers(0, WINDOW_DAYS, n_sess)
        night_hours = [0, 1, 2, 3, 4, 5, 22, 23]
        day_hours = list(range(9, 22))
        brush = []
        for d in sess_days:
            if rng.random() < P["night_ratio"]:
                hour = int(rng.choice(night_hours))
            else:
                hour = int(rng.choice(day_hours))
            t0 = pd.Timestamp(w_start + pd.Timedelta(days=int(d), hours=hour, minutes=int(rng.integers(0, 60))))
            k = int(rng.integers(3, 16))
            gap = int(rng.integers(30, 240))
            acc = accounts[int(rng.integers(0, n_acc))]
            for j in range(k):
                jitter = int(gap * P["gap_jit"] * rng.uniform(-1, 1))
                brush.append((t0 + pd.Timedelta(seconds=max(gap * j + jitter, 0)),
                              round(float(rng.choice(price_levels)) * (1 + jit * rng.uniform(-1, 1)), 2),
                              acc, hot_sku, acc_device[acc]))
        if brush:
            bts, bamt, bacc, bsku, bdev = map(list, zip(*brush))
            # 刷单集中在商户本就存在的 1-2 个渠道上（不得引入行业外的渠道，避免失真）
            chan_pool = CHANNELS[m["industry"]][:2]
            bchan = (rng.choice(chan_pool, len(bts), p=[0.78, 0.22]) if len(chan_pool) > 1
                     else np.array(chan_pool * len(bts)))
            rows.append(pd.DataFrame(dict(
                merchant_id=m["merchant_id"], ts=bts, amount=np.round(bamt, 2), customer_id=bacc,
                device_id=bdev, channel=bchan, stock_code=bsku, direction="收", is_fake=1,
                fake_type="刷单")))

    if "虚构交易" in modes:
        n_fab = int(np.clip(rng.integers(4, 22) * max(inten, 0.35), 2, 30))
        scale_k = {"电商零售": 1.0, "批发供货": 3.0, "餐饮服务": 0.4, "生活服务": 1.0}[m["industry"]]
        # 低强度：金额更接近自然区间、整数化程度低；高强度：整数大额
        if P["round_lvl"] < 0.7:
            grid = np.array([1200, 2600, 4800, 6800]) * scale_k * rng.uniform(0.9, 1.1)
        else:
            grid = np.array([1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000]) * scale_k
        days_fab = rng.integers(0, WINDOW_DAYS, n_fab)
        # 虚构交易同样由少量账号完成，并共用设备（与刷单一致，形成可观测的团伙痕迹）
        fab_devices = _new_device_ids(max(2, n_fab // 3))
        for i in range(n_fab):
            ts = pd.Timestamp(w_start + pd.Timedelta(days=int(days_fab[i]),
                                                     hours=int(rng.choice([10, 11, 14, 15, 16, 17])),
                                                     minutes=int(rng.integers(0, 60))))
            amt = float(np.round(rng.choice(grid) * (1 + jit * rng.uniform(-1, 1)), 2))
            # 虚构交易同样使用行业内自然存在的渠道（批发类商户以对公/ERP 为主）
            fab_chan = rng.choice(CHANNELS[m["industry"]])
            rows.append(pd.DataFrame(dict(
                merchant_id=[m["merchant_id"]], ts=[ts], amount=[amt],
                customer_id=[f"CF{m['merchant_id'][-5:]}{i:02d}"],
                device_id=[fab_devices[i % len(fab_devices)]],
                channel=[fab_chan], stock_code=[f"S{int(rng.integers(1, 6)):04d}"],
                direction=["收"], is_fake=[1], fake_type=["虚构交易"])))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def gen_normal_flows(m: pd.Series, daily_gmv: pd.Series) -> pd.DataFrame:
    """平台结算入账 + 供应商货款 + 房租工资 + 备付金/退款等当日进出。"""
    rows = []
    for d, v in daily_gmv.items():
        if v <= 0:
            continue
        n_in = int(rng.integers(1, 3))
        parts = rng.dirichlet(np.ones(n_in)) * v * rng.uniform(0.90, 1.06)
        if rng.random() < 0.25:      # 部分商户的结算习惯取整
            parts = np.round(parts, -int(rng.choice([1, 2])))
        for p in parts:
            ts = d + pd.Timedelta(days=int(rng.integers(0, 2)), hours=int(rng.integers(9, 18)),
                                  minutes=int(rng.integers(0, 60)))
            rows.append((m["merchant_id"], ts, round(float(p), 2), "进",
                         rng.choice(["支付平台结算", "收单机构清算"]), "PF001", 0, ""))
    weekly = daily_gmv.resample("W").sum()
    for w, v in weekly.items():
        if v <= 0 or rng.random() > 0.85:
            continue
        pay = v * rng.uniform(0.32, 0.55)
        if rng.random() < 0.5:
            pay = np.round(pay, -int(rng.choice([2, 3])))
        ts = w + pd.Timedelta(hours=int(rng.integers(10, 17)), minutes=int(rng.integers(0, 60)))
        rows.append((m["merchant_id"], ts, round(float(pay), 2), "出", "供应商货款",
                     f"SP{int(rng.integers(1, 6)):03d}", 0, ""))
    for p in pd.Series(daily_gmv.index).dt.to_period("M").unique():
        d0 = p.to_timestamp()
        rows.append((m["merchant_id"], d0 + pd.Timedelta(days=int(rng.integers(1, 6)), hours=11),
                     float(np.round(rng.uniform(3000, 25000) * m["scale"], -3)), "出", "房租水电", "LD001", 0, ""))
        rows.append((m["merchant_id"], d0 + pd.Timedelta(days=int(rng.integers(8, 13)), hours=15),
                     float(np.round(rng.uniform(8000, 60000) * m["scale"], -3)), "出", "工资代发", "PY001", 0, ""))
    # 备付金/退款等自然当日进出（合规商户亦常见）
    for _ in range(int(rng.integers(0, 5))):
        d = daily_gmv.index[int(rng.integers(0, len(daily_gmv)))]
        a = float(np.round(rng.uniform(0.3, 3.0) * 10_000, -2))
        t = d + pd.Timedelta(hours=int(rng.integers(9, 12)), minutes=int(rng.integers(0, 60)))
        rows.append((m["merchant_id"], t, a, "进", "备付金退回", f"SP{int(rng.integers(1, 6)):03d}", 0, ""))
        rows.append((m["merchant_id"], t + pd.Timedelta(hours=int(rng.integers(1, 6))),
                     float(np.round(a * rng.uniform(0.97, 1.0), -2)), "出", "保证金划出",
                     f"SP{int(rng.integers(1, 6)):03d}", 0, ""))
    return pd.DataFrame(rows, columns=["merchant_id", "ts", "amount", "direction", "cp_type",
                                       "counterparty_id", "is_fake", "fake_type"])


def inject_fake_flows(m: pd.Series, base_gmv: float) -> pd.DataFrame:
    """注入伪造流水：以商户自身规模为参照夸大进账，整数进出、当日配对回流、关联对手方。"""
    P = INTENSITY[m["fraud_intensity"]]
    w_start = m["apply_date"] - pd.Timedelta(days=WINDOW_DAYS)
    # 伪造流水总额 = 窗口真实销售额 × 强度系数（低强度仅小幅夸大，具备隐蔽性）
    lo, hi = {"低": (0.15, 0.40), "中": (0.60, 1.20), "高": (1.50, 2.80)}[m["fraud_intensity"]]
    total_target = max(base_gmv, 5_000) * rng.uniform(lo, hi)
    n_cyc = int(np.clip(rng.integers(3, 13) * (0.6 + P["mult"]), 2, 30))
    n_cp = {"低": 2, "中": 4, "高": 6}[m["fraud_intensity"]]
    n_cp = int(rng.integers(1, n_cp + 1))
    cps_in = [f"RF{m['merchant_id'][-5:]}{i:02d}A" for i in range(n_cp)]
    cps_out = [f"RF{m['merchant_id'][-5:]}{i:02d}B" for i in range(n_cp)]
    rounding = 3 if P["round_lvl"] >= 0.85 else 2       # 千元 / 百元整数
    dev = 0.03 if P["round_lvl"] >= 0.85 else 0.08      # 回流金额偏差（影响配对识别）
    rows = []
    for _ in range(n_cyc):
        d = int(rng.integers(0, WINDOW_DAYS))
        amt = float(np.round(total_target / n_cyc * rng.uniform(0.7, 1.3), -rounding))
        j = int(rng.integers(0, n_cp))
        t_in = pd.Timestamp(w_start + pd.Timedelta(days=d, hours=int(rng.integers(9, 12)),
                                                  minutes=int(rng.integers(0, 60))))
        t_out = t_in + pd.Timedelta(hours=int(rng.integers(2, 9)), minutes=int(rng.integers(0, 60)))
        rows.append((m["merchant_id"], t_in, amt, "进", "转账汇入", cps_in[j], 1, "伪造流水"))
        rows.append((m["merchant_id"], t_out,
                     float(np.round(amt * (1 - dev * rng.random()), -rounding)),
                     "出", "转账汇出", cps_out[j], 1, "伪造流水"))
    return pd.DataFrame(rows, columns=["merchant_id", "ts", "amount", "direction", "cp_type",
                                       "counterparty_id", "is_fake", "fake_type"])


def apply_hn_orders(m: pd.Series, orders: pd.DataFrame, w_start):
    """困难负样本（合规）在订单侧的「真实经营特点」。"""
    if m["hard_negative"] != 1 or len(orders) == 0:
        return orders
    in_win = orders["ts"] >= w_start
    if m["hn_type"] == "夜间经营型":
        hp = industry_hour_profile(m["industry"]).copy()
        night = rng.uniform(0.18, 0.35)
        for h in range(6):
            hp[h] += night / 6
        hp = hp / hp.sum()
        orders["ts"] = orders["ts"].dt.normalize() + pd.to_timedelta(rng.choice(24, len(orders), p=hp), unit="h")
        orders["ts"] += pd.to_timedelta(rng.integers(0, 3600, len(orders)), unit="s")
    elif m["hn_type"] == "批发大额整数型":
        orders["amount"] = np.round(orders["amount"] * rng.uniform(3, 7), -2)
        orders = orders[rng.random(len(orders)) < 0.45].copy()
        # 合规的大客户集中采购：少量固定客户（非团伙，使用独立设备，避免与刷单混淆）
        hn_cust = f"HB{m['merchant_id'][-5:]}00"
        mask = rng.random(len(orders)) < 0.35
        orders.loc[mask, "customer_id"] = hn_cust
        orders.loc[mask, "device_id"] = f"HD{m['merchant_id'][-5:]}00"
    elif m["hn_type"] == "直播促销型":
        win = orders[in_win].copy()
        base = orders[~in_win].copy()
        if len(win):
            n_days = win["ts"].dt.normalize().nunique()
            promo_days = pd.to_datetime(rng.choice(win["ts"].dt.normalize().unique(),
                                                   size=min(int(rng.integers(6, 15)), n_days), replace=False))
            n_promo = int(len(win) * rng.uniform(0.35, 0.65))
            idx_promo = rng.choice(len(win), n_promo, replace=False)
            d_sel = rng.choice(len(promo_days), n_promo)
            hour = rng.choice([19, 20, 20, 21], n_promo)
            win.iloc[idx_promo, win.columns.get_loc("ts")] = (
                pd.Series(promo_days[d_sel]) + pd.to_timedelta(hour, unit="h")
                + pd.to_timedelta(rng.integers(0, 3600, n_promo), unit="s")).values
            win.iloc[idx_promo, win.columns.get_loc("channel")] = CHANNELS[m["industry"]][-1]
            levels = win.iloc[idx_promo]["amount"].values[[0, -1]]
            win.iloc[idx_promo, win.columns.get_loc("amount")] = np.round(
                rng.choice(levels * rng.uniform(0.98, 1.02, 2), n_promo), 2)
            new_mask = rng.random(len(win)) < 0.45
            n_new = int(new_mask.sum())
            win.loc[new_mask, "customer_id"] = [f"HN{int(rng.integers(1, 10**8)):08d}" for _ in range(n_new)]
            # 直播新客为真实消费者，各自使用独立设备（合规行为，不应呈现团伙特征）
            win.loc[new_mask, "device_id"] = [f"HV{int(rng.integers(1, 10**8)):08d}" for _ in range(n_new)]
            orders = pd.concat([base, win], ignore_index=True)
    return orders


def apply_hn_flows(m: pd.Series, flows: pd.DataFrame):
    """困难负样本（合规）在资金流水侧的「真实经营特点」。"""
    if m["hard_negative"] != 1 or m["hn_type"] != "整数结算型" or len(flows) == 0:
        return flows
    mask_in = flows["direction"] == "进"
    flows.loc[mask_in, "amount"] = np.round(flows.loc[mask_in, "amount"] * rng.uniform(1.1, 1.8), -3)
    for _ in range(int(rng.integers(4, 11))):   # 整数化当日进出
        d = flows["ts"].dt.normalize().sample(1, random_state=int(rng.integers(0, 10**6))).iloc[0]
        a = float(np.round(rng.uniform(5, 40) * 10_000, -3))
        t = d + pd.Timedelta(hours=int(rng.integers(9, 11)))
        flows = pd.concat([flows, pd.DataFrame([
            (m["merchant_id"], t, a, "进", "货款预收", f"SP{int(rng.integers(1, 6)):03d}", 0, ""),
            (m["merchant_id"], t + pd.Timedelta(hours=3), float(np.round(a * 0.98, -3)), "出",
             "货款退回", f"SP{int(rng.integers(1, 6)):03d}", 0, "")],
            columns=flows.columns)], ignore_index=True)
    return flows


def gen_camouflage_orders(tx: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """伪装行为：刷单/虚构账号在「其他商户」的正常消费。

    这是让图方法有意义的难点所在——欺诈账号不只是窝在一家店里，
    它们会到别的商户正常消费来稀释单点异常信号；但设备仍与团伙共用，
    因此只有建模「账号—设备—商户」的关联结构才能识别。
    """
    acc = (tx.loc[tx["is_fake"] == 1, ["customer_id", "device_id"]]
             .drop_duplicates("customer_id"))
    acc = acc[acc["customer_id"].str.match(r"^C[BF]")]
    if not len(acc):
        return pd.DataFrame()
    sel = acc.sample(frac=CAMOUFLAGE_RATIO, random_state=SEED)
    mid = pop["merchant_id"].values
    ind_of = dict(zip(pop["merchant_id"], pop["industry"]))
    rows = []
    for _, r in sel.iterrows():
        n_other = int(rng.integers(1, 4))
        others = rng.choice(mid, n_other, replace=False)
        for om in others:
            prof = INDUSTRIES[ind_of[om]]
            k = int(rng.integers(1, 5))
            days = rng.integers(0, HISTORY_DAYS, k)
            hp = industry_hour_profile(ind_of[om])
            ts = (pd.Timestamp("2026-01-01") - pd.Timedelta(days=HISTORY_DAYS)
                  + pd.to_timedelta(days, unit="D")
                  + pd.to_timedelta(rng.choice(24, k, p=hp), unit="h")
                  + pd.to_timedelta(rng.integers(0, 60, k), unit="m"))
            amt = np.round(np.clip(prof["med_amt"] * rng.lognormal(0, prof["sigma"] * 0.8, k), 1, None), 2)
            rows.append(pd.DataFrame(dict(
                merchant_id=om, ts=ts, amount=amt, customer_id=r["customer_id"],
                device_id=r["device_id"], channel=rng.choice(CHANNELS[ind_of[om]], k),
                stock_code=[f"S{int(rng.integers(1, prof['sku'] + 1)):04d}" for _ in range(k)],
                direction="收", is_fake=0, fake_type="")))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    global UNIVERSE, DEVICE_COUNTER
    UNIVERSE = build_customer_universe()
    DEVICE_COUNTER = 0
    print(f"[universe] 客户池 {UNIVERSE['n']:,} 人，设备 "
          f"{len(UNIVERSE['device_flat']):,} 台")

    pop = build_merchant_population()
    build_gang_pools(pop)
    tx_parts, flow_parts, meta_rows = [], [], []

    for _, m in pop.iterrows():
        w_start = m["apply_date"] - pd.Timedelta(days=WINDOW_DAYS)
        orders = gen_normal_orders(m)
        # 困难负样本的形态特征属于其「真实经营特点」，不计为虚增
        orders = apply_hn_orders(m, orders, w_start)

        daily = orders.groupby(orders["ts"].dt.normalize())["amount"].sum() if len(orders) else pd.Series(dtype=float)
        daily.index = pd.to_datetime(daily.index)
        flows = apply_hn_flows(m, gen_normal_flows(m, daily))

        # 「真实经营口径」：注入虚假行为之前的经营数据，用于对照评估额度虚高
        true_orders_all, true_gmv_all = len(orders), float(orders["amount"].sum()) if len(orders) else 0.0
        true_win = orders[orders["ts"] >= w_start] if len(orders) else orders
        true_win_gmv, true_win_n = float(true_win["amount"].sum()), len(true_win)

        if m["is_noncompliant"] == 1 and m["visible_fraud"] == 1:
            extra = inject_fake_orders(m, true_win_n)
            if len(extra):
                orders = pd.concat([orders, extra], ignore_index=True)
            if "伪造流水" in m["fraud_modes"]:
                flows = pd.concat([flows, inject_fake_flows(m, true_win_gmv)], ignore_index=True)

        obs_win = orders[orders["ts"] >= w_start] if len(orders) else orders
        obs_win_gmv, obs_win_n = float(obs_win["amount"].sum()), len(obs_win)

        tx_parts.append(orders)
        flow_parts.append(flows)
        meta_rows.append(dict(
            merchant_id=m["merchant_id"], window_start=w_start,
            true_window_orders=true_win_n, obs_window_orders=obs_win_n,
            true_window_gmv=round(true_win_gmv, 2), obs_window_gmv=round(obs_win_gmv, 2),
            true_orders_all=true_orders_all, true_gmv_all=round(true_gmv_all, 2),
        ))

    tx = pd.concat(tx_parts, ignore_index=True)
    # 追加伪装订单：欺诈账号在其他商户的正常消费（稀释单点异常，考验图方法）
    camo = gen_camouflage_orders(tx, pop)
    if len(camo):
        tx = pd.concat([tx, camo], ignore_index=True)
    tx = tx.sort_values(["merchant_id", "ts"]).reset_index(drop=True)
    flows = pd.concat(flow_parts, ignore_index=True).sort_values(["merchant_id", "ts"]).reset_index(drop=True)
    merch = pop.merge(pd.DataFrame(meta_rows), on="merchant_id")
    merch["gmv_inflation"] = (merch["obs_window_gmv"] / merch["true_window_gmv"].replace(0, np.nan) - 1).fillna(0.0).round(4)
    merch["order_inflation"] = (merch["obs_window_orders"] / merch["true_window_orders"].replace(0, np.nan) - 1).fillna(0.0).round(4)
    merch = merch.drop(columns=["scale", "growth", "new_cust_ratio"])

    tx.to_csv(os.path.join(PROC, "transactions.csv.gz"), index=False, compression="gzip")
    flows.to_csv(os.path.join(PROC, "fund_flows.csv.gz"), index=False, compression="gzip")
    merch.to_csv(os.path.join(PROC, "merchants.csv"), index=False)
    json.dump(dict(seed=SEED, n_merchants=N_MERCHANTS, fake_ratio=FAKE_RATIO, hard_neg_ratio=HARD_NEG_RATIO,
                   window_days=WINDOW_DAYS, history_days=HISTORY_DAYS, split_date=str(SPLIT_DATE.date()),
                   missing_flow_ratio=MISSING_FLOW_RATIO, intensity=INTENSITY, industries=INDUSTRIES),
              open(os.path.join(PROC, "dataset_config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"[done] merchants={len(merch):,}  tx={len(tx):,}  flows={len(flows):,}")
    print(f"  非规范商户: {int(merch.is_noncompliant.sum())} ({merch.is_noncompliant.mean():.1%})"
          f"  其中窗口内可见: {int(merch.loc[merch.is_noncompliant == 1, 'visible_fraud'].sum())}"
          f"  困难负样本: {int(merch.hard_negative.sum())}")
    print(f"  虚假订单占比: {tx.is_fake.mean():.2%}   虚假流水占比: {flows.is_fake.mean():.2%}")
    if len(camo):
        print(f"  伪装订单（欺诈账号在他店正常消费）: {len(camo):,} 条")
    # 图结构诊断：客户节点连通性
    cg = tx.groupby("customer_id").agg(n=("ts", "size"), nm=("merchant_id", "nunique"))
    print(f"  图结构: 客户节点 {len(cg):,}，其中度数≥2 占 {(cg['n'] >= 2).mean():.1%}，"
          f"跨商户占 {(cg['nm'] >= 2).mean():.1%}，设备 {tx.device_id.nunique():,} 台")
    dev_multi = tx.groupby("device_id")["customer_id"].nunique()
    print(f"  共享设备（承载≥2个账号）: {int((dev_multi >= 2).sum()):,} 台")
    print("  分强度 GMV 虚增中位数:", merch[merch.is_noncompliant == 1].groupby("fraud_intensity").gmv_inflation.median().round(3).to_dict())
    print("  分模式 GMV 虚增中位数:", {k: round(float(merch[merch.fraud_modes.fillna('').str.contains(k)].gmv_inflation.median()), 3)
                                       for k in FAKE_TYPES})
    print(merch.groupby("data_split")["is_noncompliant"].agg(["size", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()