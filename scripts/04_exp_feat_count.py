# -*- coding: utf-8 -*-
"""
Exp A2: 特征数量实验 — 基础17特征 vs 完整90特征 (One-Question-One-Experiment)
RQ2: 90特征甜点位背后的机理 (社区: 17特征 CV 0.1032/LB 0.093; 90特征 CV 0.1389/LB 0.119)
协议: CV1 (train m0-50 / valid m51-70), 官方参数, 仅变更特征子集
"""
import time
import numpy as np
import polars as pl
import lightgbm as lgb

FEAT = r"D:\mscapital-forecasting\data\processed\train_features.parquet"
N_THREADS = 12

tr = pl.read_parquet(FEAT)
all_feats = [c for c in tr.columns if c not in ("sample_id", "month", "target")]

# 基础17特征: 仅盘口基础统计 + 成交量 (无窗口/EWM/交叉特征)
BASE17 = [
    "m_mid_last", "m_mid_mean", "m_mid_std", "m_mid_range",
    "m_sp_last", "m_sp_mean",
    "m_imb_last", "m_imb_mean", "m_imb_std",
    "m_depth_mean", "m_rv", "m_ofi_sum",
    "t_vol_sum", "t_sv_sum", "t_px_last",
    "o_vol_sum", "o_sv_sum",
]
assert all(f in all_feats for f in BASE17), "BASE17 有列不在特征集"

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
    print(f"{name}: n_feat={len(feats)} valid_cos={c:.6f} best_iter={model.best_iteration} ({time.time()-t0:.1f}s)", flush=True)
    return c

r17 = run("A2-17feat", BASE17)
r90 = run("A2-90feat", all_feats)
print(f"\n=== Exp A2 汇总 ===\n17特征: {r17:.6f}\n90特征: {r90:.6f}\n增量: {r90-r17:+.6f}")
