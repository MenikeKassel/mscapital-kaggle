# -*- coding: utf-8 -*-
"""
Exp C1: 增强窗口特征 — 深挖窗口统计方向 (B1发现窗口组贡献最大)
Control: 90特征 (CV1 = 0.130204)
Treatment: 90 + 18个新特征 (窗口5/30/300s + 偏度/分位数/比率/窗口收益)
协议: CV1, 官方参数。若CV提升显著且特征数<120, 候选P2特征集。
"""
import time
import numpy as np
import polars as pl
import lightgbm as lgb

RAW = r"D:\mscapital-forecasting\data\raw"
FEAT = r"D:\mscapital-forecasting\data\processed\train_features.parquet"
N_THREADS = 12

def build_enhanced_market():
    lf = pl.scan_ipc(f"{RAW}/train/market.feather", memory_map=False)
    lf = lf.sort(["sample_id", "seconds_before_predict"], descending=[False, True])
    lf = lf.with_columns([
        ((pl.col("ask_price_1") + pl.col("bid_price_1")) * 0.5).alias("_mid"),
        (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("_sp"),
        ((pl.col("ask_volume_1") - pl.col("bid_volume_1"))
         / (pl.col("ask_volume_1") + pl.col("bid_volume_1") + 1.0)).alias("_imb"),
    ])
    lf = lf.with_columns([
        pl.col("_mid").diff().over("sample_id").fill_null(0.0).alias("_dmid"),
        (pl.col("ask_volume_1").diff().over("sample_id").fill_null(0.0)
         - pl.col("bid_volume_1").diff().over("sample_id").fill_null(0.0)).alias("_ofi"),
    ])
    exprs = []
    # 新窗口 5/30/300: rv / ofi / txv
    for w in [5, 30, 300]:
        cond = pl.col("seconds_before_predict") <= w
        exprs += [
            ((pl.col("_dmid").filter(cond)) ** 2).sum().sqrt().alias(f"m_rv_{w}"),
            pl.col("_ofi").filter(cond).sum().alias(f"m_ofi_sum_{w}"),
            pl.col("transaction_volume").filter(cond).sum().alias(f"m_txv_sum_{w}"),
        ]
    # 60s 增强统计: skew / kurt / 分位数 / spread std / imb range
    c60 = pl.col("seconds_before_predict") <= 60
    exprs += [
        pl.col("_mid").filter(c60).skew().alias("m_mid_skew_60"),
        pl.col("_mid").filter(c60).kurtosis().alias("m_mid_kurt_60"),
        pl.col("_mid").filter(c60).quantile(0.25).alias("m_mid_p25_60"),
        pl.col("_mid").filter(c60).quantile(0.75).alias("m_mid_p75_60"),
        pl.col("_sp").filter(c60).std().alias("m_sp_std_60"),
        pl.col("_imb").filter(c60).max().sub(pl.col("_imb").filter(c60).min()).alias("m_imb_range_60"),
        pl.col("_mid").filter(c60).last().sub(pl.col("_mid").filter(c60).first()).alias("m_mid_ret_60"),
    ]
    # 比率特征
    c5 = pl.col("seconds_before_predict") <= 5
    cfull = pl.col("seconds_before_predict") <= 600
    exprs += [
        (pl.col("_ofi").filter(c5).sum() / (pl.col("_ofi").filter(cfull).sum().abs() + 1e-8)).alias("x_ofi_5_over_full"),
        ((pl.col("_dmid").filter(c5)) ** 2).sum().sqrt() / ((pl.col("_dmid").filter(cfull)) ** 2).sum().sqrt().alias("x_rv_5_over_full"),
        (pl.col("_mid").filter(c60).std() / pl.col("_mid").filter(cfull).std()).alias("x_mid_std_60_over_full"),
    ]
    return lf.group_by("sample_id").agg(exprs).collect(streaming=True)

t0 = time.time()
enh = build_enhanced_market()
print(f"增强特征构建: {enh.shape} ({time.time()-t0:.1f}s)", flush=True)

tr = pl.read_parquet(FEAT)
tr = tr.join(enh, on="sample_id", how="left")
new_cols = [c for c in enh.columns if c != "sample_id"]
print(f"新特征 {len(new_cols)} 个: {new_cols}", flush=True)
for c in new_cols:
    n_null = tr[c].is_null().sum()
    if n_null > 0:
        print(f"  {c}: {n_null} nulls -> fill 0", flush=True)
        tr = tr.with_columns(pl.col(c).fill_null(0.0))

all_feats = [c for c in tr.columns if c not in ("sample_id", "month", "target")]
print(f"总特征数: {len(all_feats)}", flush=True)

PARAMS = dict(
    objective="regression", metric="rmse",
    learning_rate=0.02, num_leaves=32, min_data_in_leaf=300,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    lambda_l2=5.0, max_bin=255, verbose=-1, num_threads=N_THREADS, seed=0)

def cos_uncenter(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def run(name, feats):
    tr_df = tr.filter(pl.col("month") <= 50)
    va_df = tr.filter((pl.col("month") > 50) & (pl.col("month") <= 70))
    X_tr = tr_df.select(feats).to_numpy().astype(np.float32)
    y_tr = tr_df["target"].to_numpy().astype(np.float32)
    X_va = va_df.select(feats).to_numpy().astype(np.float32)
    y_va = va_df["target"].to_numpy().astype(np.float32)
    t0 = time.time()
    dtr = lgb.Dataset(X_tr, y_tr)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)
    model = lgb.train(PARAMS, dtr, num_boost_round=10000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200)])
    p_va = model.predict(X_va, num_iteration=model.best_iteration)
    c = cos_uncenter(p_va, y_va)
    print(f"{name}: n={len(feats)} cos={c:.6f} iter={model.best_iteration} ({time.time()-t0:.1f}s)", flush=True)
    return c

BASE = 0.130204
r = run("C1-108feat", all_feats)
print(f"\n=== Exp C1 汇总 ===\n90特征: {BASE:.6f}\n108特征: {r:.6f} ({r-BASE:+.6f})")
