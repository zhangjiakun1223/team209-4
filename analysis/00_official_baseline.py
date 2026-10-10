# -*- coding: utf-8 -*-
"""
00_official_baseline.py
采集并结构化管理 2026 年官方口径基线数据，替代原公开数据集（UCI Online Retail II，2009—2011）
作为场景化数据集的校准依据。

数据来源（均为官方发布，人工核录，含原文出处）：
  1) 国家统计局：社会消费品零售总额月度数据（2025年9月—2026年8月）
     —— 用于校准月度经营季节性（双11、春节等节律）
  2) 中国人民银行：《2026年第二季度支付体系运行总体情况》（2026-09-02）
     —— 用于校准交易笔均金额（银行卡消费、移动支付、非银行支付机构网络支付）
  3) 国家统计局：2026年1—8月分品类零售增速
     —— 用于校准各类经营主体的自然增长率
  4) 国家金融监督管理总局：2026年二季度银行业主要监管指标（2026-08-14）
     —— 用于佐证小微客群规模与业务背景

输出：data/processed/official_baseline.json
"""
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")

# ---------------- 官方数据（人工核录，含出处） ----------------
# 1) 社会消费品零售总额（单月值，亿元）；统计局对 1—2 月合并发布，此处按均值拆分
NBS_RETAIL_MONTHLY = {
    "2025-09": 41971,   # 统计局：9月份社零总额 41971 亿元（2025-10-20 发布）
    "2025-10": 46292,   # 由 1—11月 456067 与 1—9月 365877 差分并对 11 月扣除得到
    "2025-11": 43898,   # 统计局：11月份社零总额 43898 亿元（2025-12-15 发布）
    "2025-12": 45136,   # 统计局：12月份社零总额 45136 亿元（2026-01-19 发布）
    "2026-01": 43039.5,  # 统计局：1—2月合计 86079 亿元，按均值拆分（2026-03-16 发布）
    "2026-02": 43039.5,
    "2026-03": 41616,   # 统计局：3月份社零总额 41616 亿元（2026-04-16 发布）
    "2026-04": 37246,   # 由 1—5月 206031、1—3月 127695 与 5 月 41090 差分得到
    "2026-05": 41090,   # 统计局：5月份社零总额 41090 亿元（2026-06-16 发布）
    "2026-06": 42691,   # 由 1—7月 287744、1—5月 206031 与 7 月 39022 差分得到
    "2026-07": 39022,   # 统计局：7月份社零总额 39022 亿元（2026-08-17 发布）
    "2026-08": 39824,   # 统计局：8月份社零总额 39824 亿元（2026-09-15 发布）
}

# 2) 支付体系（2026 年第二季度）
PBC_PAYMENT_2026Q2 = {
    "source": "中国人民银行《2026年第二季度支付体系运行总体情况》（2026-09-02 发布）",
    "card_consumption": {"笔数_亿": 1027.44, "金额_万亿": 31.56, "笔均_元": round(31.56e12 / (1027.44e8), 1)},
    "mobile_payment": {"笔数_亿": 608.24, "金额_万亿": 144.07, "笔均_元": round(144.07e12 / (608.24e8), 1)},
    "nonbank_network_payment": {"笔数_亿": 3407.02, "金额_万亿": 80.11, "笔均_元": round(80.11e12 / (3407.02e8), 1)},
    "bank_card_total": {"笔数_亿": 1650.37, "金额_万亿": 250.51},
    "unit_accounts_亿户": 1.25,      # 单位银行账户（小微经营主体规模参照）
    "note": "银行卡消费笔均反映线下消费典型金额；非银行支付机构网络支付笔均反映线上小额高频特征",
}

# 3) 分品类零售增速（2026 年 1—8 月，国家统计局）
NBS_GROWTH_2026 = {
    "source": "国家统计局 2026年1—8月份社会消费品零售总额数据（2026-09-15 发布）",
    "社会消费品零售总额": 0.011,
    "网上商品零售额": 0.043,
    "网上服务零售额": 0.051,
    "餐饮收入": 0.024,
    "商品零售额": 0.010,
    "粮油食品类": 0.068,
    "服装鞋帽针纺织品类": 0.051,
    "日用品类": 0.038,
    "网上商品零售额占社零比重": 0.257,
}

# 4) 小微金融（2026 年二季度）
NFRA_MICRO_2026Q2 = {
    "source": "国家金融监督管理总局《2026年二季度银行业保险业主要监管指标数据情况》（2026-08-14 发布）",
    "普惠型小微企业贷款余额_万亿元": 38.9,
    "同比增速": 0.08,
}

# ---------------- 行业锚点（由官方数据派生） ----------------
# 笔均金额：线上小微商户以「非银行支付机构网络支付笔均」为基准，
# 线下行业以「银行卡消费笔均」为基准，再按行业结构系数调整
CARD_MEAN = PBC_PAYMENT_2026Q2["card_consumption"]["笔均_元"]        # 307 元
NONBANK_MEAN = PBC_PAYMENT_2026Q2["nonbank_network_payment"]["笔均_元"]  # 235 元

INDUSTRY_ANCHORS = {
    # 行业: 笔均锚点来源、行业系数、自然增速（对应官方分项增速）、经营时段假设
    "电商零售": dict(base="线上", mean_amt=round(NONBANK_MEAN * 1.0, 1), sigma=0.75,
                     growth=NBS_GROWTH_2026["网上商品零售额"], sku=90, pool=160, lam=3.5, wkend=1.18),
    "批发供货": dict(base="线下", mean_amt=round(CARD_MEAN * 2.6, 1), sigma=0.90,
                     growth=NBS_GROWTH_2026["商品零售额"] + 0.01, sku=45, pool=35, lam=1.2, wkend=0.45),
    "餐饮服务": dict(base="线下", mean_amt=round(CARD_MEAN * 0.20, 1), sigma=0.55,
                     growth=NBS_GROWTH_2026["餐饮收入"], sku=60, pool=120, lam=6.0, wkend=1.35),
    "生活服务": dict(base="线下", mean_amt=round(CARD_MEAN * 0.55, 1), sigma=0.70,
                     growth=NBS_GROWTH_2026["网上服务零售额"] - 0.02, sku=40, pool=90, lam=2.4, wkend=1.25),
}

# 经营时段假设（官方未公开日内分布，采用小微商户经营时段的通行假设）
HOUR_ASSUMPTION = {
    "说明": "官方公开数据未包含日内交易分布，采用小微商户经营时段通行假设：经营时段集中于 9—22 时，凌晨 0—6 时订单占比极低",
    "时段权重": {"0-6": 0.006, "7-8": 0.030, "9-11": 0.250, "12-13": 0.200,
                 "14-17": 0.250, "18-19": 0.160, "20-22": 0.100, "23": 0.004},
}


def main():
    vals = NBS_RETAIL_MONTHLY
    mean_v = sum(vals.values()) / len(vals)
    # 日历月季节因子（1—2 月按拆分值计算）
    seasonal = {}
    for k, v in vals.items():
        m = int(k.split("-")[1])
        seasonal.setdefault(m, []).append(v / mean_v)
    season_factor = {str(m): round(sum(xs) / len(xs), 4) for m, xs in sorted(seasonal.items())}

    out = {
        "data_year": 2026,
        "sources": {
            "nbs_retail": NBS_GROWTH_2026["source"],
            "pbc_payment": PBC_PAYMENT_2026Q2["source"],
            "nfra_micro": NFRA_MICRO_2026Q2["source"],
            "nbs_monthly_series": "国家统计局月度社会消费品零售总额（2025年9月—2026年8月，逐月发布）",
        },
        "nbs_retail_monthly_亿元": vals,
        "nbs_retail_mean_亿元": round(mean_v, 1),
        "season_factor_by_calendar_month": season_factor,
        "pbc_payment": PBC_PAYMENT_2026Q2,
        "nbs_growth": NBS_GROWTH_2026,
        "nfra_micro": NFRA_MICRO_2026Q2,
        "industry_anchors": INDUSTRY_ANCHORS,
        "hour_assumption": HOUR_ASSUMPTION,
    }
    os.makedirs(PROC, exist_ok=True)
    path = os.path.join(PROC, "official_baseline.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[save] {path}")
    print("官方口径基线：")
    print(f"  月度社零均值 {mean_v:,.1f} 亿元；季节因子 {json.dumps(season_factor, ensure_ascii=False)}")
    print(f"  笔均（元）：银行卡消费 {CARD_MEAN}｜移动支付 "
          f"{PBC_PAYMENT_2026Q2['mobile_payment']['笔均_元']}｜非银网络支付 {NONBANK_MEAN}")
    print(f"  行业笔均锚点：" + "，".join(f"{k} {v['mean_amt']}元(g={v['growth']:.1%})"
                                          for k, v in INDUSTRY_ANCHORS.items()))


if __name__ == "__main__":
    main()