# -*- coding: utf-8 -*-
"""
P2: 3 folds 融合验证 — 表格(R2+micro CatBoost) × TCN 增强版
关键: 融合增益是否在 T3/T4/PSEUDO 全部为正 (多 fold 同号 = 晋级标准)
"""
import time
import numpy as np
import polars as pl
from catboost import CatBoostRegressor

FEAT = r"D:\mscapital-forecasting\data\processed\train_features.parquet"
MICRO = r"D:\mscapital-forecasting\data\processed\micro_features_train.parquet"
P12 = r"D:\mscapital-forecasting\data\processed\p12_out"
N_THREADS = 12

tr = pl.read_parquet(FEAT)
mi = pl.read_parquet(MICRO)
all_feats = [c for c in tr.columns if c not in ("sample_id", "month", "target")]
micro_cols = [c for c in mi.columns if c != "sample_id"]

def make_R2(df):
    return df.with_columns([
        (pl.col("m_sp_mean") / (pl.col("m_mid_mean") + 1e-8)).alias("m_sp_mean"),
        (pl.col("m_depth_mean") / (pl.col("m_txv_sum_60") + 1.0)).alias("m_depth_mean"),
        (pl.col("m_rv") / (pl.col("m_mid_std") + 1e-8)).alias("m_rv"),
        (pl.col("o_vol_sum") / (pl.col("t_vol_sum") + 1.0)).alias("o_vol_sum"),
        (pl.col("o_n_120") / (pl.col("t_n_120") + 1.0)).alias("o_n_120"),
        (pl.col("m_txv_sum_180") / (pl.col("m_txv_sum_60") + 1.0)).alias("m_txv_sum_180"),
        (pl.col("m_sp_mean_60") / (pl.col("m_mid_mean_60") + 1e-8)).alias("m_sp_mean_60"),
        (pl.col("m_sp_mean_180") / (pl.col("m_mid_mean_180") + 1e-8)).alias("m_sp_mean_180"),
    ])

tr_all = make_R2(tr.select(["sample_id"] + all_feats)).join(mi, on="sample_id", how="left")
tr_all = tr_all.with_columns([pl.col(c).fill_null(0.0) for c in micro_cols])
cols = all_feats + micro_cols
X_all = tr_all.select(cols).to_numpy().astype(np.float32)
y_all = tr["target"].to_numpy().astype(np.float32)
m_all = tr["month"].to_numpy().astype(np.int32)

def cos_uncenter(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

FOLDS = {"T3": (0, 50, 51, 60), "T4": (0, 50, 61, 70), "PSEUDO": (0, 32, 33, 70)}
print(f"{'fold':<8} {'表格':>9} {'TCN':>9} {'corr':>7} {'融合w':>8} {'融合':>9} {'Δ':>8}")
results = {}
T0 = time.time()
for fname, (a, b, c0, d0) in FOLDS.items():
    sel_t = (m_all >= a) & (m_all <= b); sel_v = (m_all >= c0) & (m_all <= d0)
    X, y, Xv, yv = X_all[sel_t], y_all[sel_t], X_all[sel_v], y_all[sel_v]
    cb = CatBoostRegressor(iterations=10000, learning_rate=0.02, depth=6, l2_leaf_reg=5.0,
                           subsample=0.8, colsample_bylevel=0.8, loss_function="RMSE",
                           early_stopping_rounds=200, verbose=0, thread_count=N_THREADS, random_seed=0)
    cb.fit(X, y, eval_set=(Xv, yv))
    p_tab = cb.predict(Xv)
    c_tab = cos_uncenter(p_tab, yv)
    d = np.load(f"{P12}/p12_{fname}.npz")
    p_tcn = d["pred"]
    c_tcn = float(d["cos"])
    corr = float(np.corrcoef(p_tab, p_tcn)[0, 1])
    # 融合网格 (细)
    best = (-1, 0.0)
    for wt in np.arange(0, 0.31, 0.02):
        cc = cos_uncenter((1 - wt) * p_tab + wt * p_tcn, yv)
        if cc > best[0]:
            best = (cc, wt)
    results[fname] = (c_tab, c_tcn, corr, best)
    print(f"{fname:<8} {c_tab:.5f} {c_tcn:.5f} {corr:.3f} {best[1]:.2f} {best[0]:.5f} {best[0]-c_tab:+.5f}", flush=True)

print(f"\n=== P2 汇总 (总 {time.time()-T0:.0f}s) ===")
gains = [results[f][3][0] - results[f][0] for f in FOLDS]
print(f"融合增益: {[f'{g:+.4f}' for g in gains]}")
print(f"全部为正: {all(g > 0 for g in gains)} | mean Δ = {np.mean(gains):+.5f}")
