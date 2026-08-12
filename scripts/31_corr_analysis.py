# -*- coding: utf-8 -*-
"""
P1-2b: TCN corr + 融合分析 (PSEUDO fold)
TCN pred vs CatBoost R2+micro pred: corr → 融合网格 → cos
"""
import time
import numpy as np
import polars as pl
from catboost import CatBoostRegressor

FEAT = r"D:\mscapital-forecasting\data\processed\train_features.parquet"
MICRO = r"D:\mscapital-forecasting\data\processed\micro_features_train.parquet"
P12 = r"D:\mscapital-forecasting\data\processed\p12_out\p12_valid_pred.npz"
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

# PSEUDO fold: train m0-32 / valid m33-70
sel_t = (m_all <= 32); sel_v = (m_all > 32) & (m_all <= 70)
X, y, Xv, yv = X_all[sel_t], y_all[sel_t], X_all[sel_v], y_all[sel_v]
print(f"train {X.shape[0]:,} valid {Xv.shape[0]:,}", flush=True)

t0 = time.time()
cb = CatBoostRegressor(iterations=10000, learning_rate=0.02, depth=6, l2_leaf_reg=5.0,
                       subsample=0.8, colsample_bylevel=0.8, loss_function="RMSE",
                       early_stopping_rounds=200, verbose=0, thread_count=N_THREADS, random_seed=0)
cb.fit(X, y, eval_set=(Xv, yv))
p_tab = cb.predict(Xv)
c_tab = cos_uncenter(p_tab, yv)
print(f"tabular (CatBoost R2+micro): cos={c_tab:.6f} ({time.time()-t0:.0f}s)", flush=True)

d = np.load(P12)
p_tcn = d["pred"]
y_p12 = d["y"]
# 验证样本对齐 (TCN 用同样 PSEUDO valid, 顺序相同: label 里 month 33-70 的顺序)
print(f"TCN pred: {p_tcn.shape}, y match: {np.allclose(y_p12, yv)}", flush=True)

# corr
c = np.corrcoef(p_tcn, p_tab)[0, 1]
print(f"\ncorr(TCN, tabular) = {c:.4f}")
c_tcn = cos_uncenter(p_tcn, yv)
print(f"TCN cos = {c_tcn:.6f} | tabular cos = {c_tab:.6f}")

# 融合网格
print("\n=== 融合搜索 (w_tab, w_tcn) ===")
best = (-1, None)
for wt in np.arange(0, 1.01, 0.1):
    wc = 1 - wt
    p = wc * p_tab + wt * p_tcn
    cc = cos_uncenter(p, yv)
    if cc > best[0]:
        best = (cc, (round(wc, 1), round(wt, 1)))
    print(f"  w_tab={wc:.1f} w_tcn={wt:.1f}: {cc:.6f}")
print(f"\nbest: w={best[1]} cos={best[0]:.6f} (vs tabular-only {c_tab:.6f}, Δ={best[0]-c_tab:+.5f})")
