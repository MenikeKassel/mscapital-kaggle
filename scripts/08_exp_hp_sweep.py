# -*- coding: utf-8 -*-
"""
Exp D1: 超参数敏感性 (One-Question-One-Experiment: 每次只变一个参数)
Control: 官方参数 (CV1 = 0.130204, best_iter=700)
Treatment: num_leaves 64/128, lr 0.05, lambda_l2 20, min_data_in_leaf 100
"""
import time
import numpy as np
import polars as pl
import lightgbm as lgb

FEAT = r"D:\mscapital-forecasting\data\processed\train_features.parquet"
N_THREADS = 12

tr = pl.read_parquet(FEAT)
all_feats = [c for c in tr.columns if c not in ("sample_id", "month", "target")]
tr_df = tr.filter(pl.col("month") <= 50)
va_df = tr.filter((pl.col("month") > 50) & (pl.col("month") <= 70))
X_tr = tr_df.select(all_feats).to_numpy().astype(np.float32)
y_tr = tr_df["target"].to_numpy().astype(np.float32)
X_va = va_df.select(all_feats).to_numpy().astype(np.float32)
y_va = va_df["target"].to_numpy().astype(np.float32)

BASE = dict(
    objective="regression", metric="rmse",
    learning_rate=0.02, num_leaves=32, min_data_in_leaf=300,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    lambda_l2=5.0, max_bin=255, verbose=-1, num_threads=N_THREADS, seed=0)

VARIANTS = {
    "D1a_leaves64": {"num_leaves": 64},
    "D1b_leaves128": {"num_leaves": 128},
    "D1c_lr005": {"learning_rate": 0.05},
    "D1d_l2_20": {"lambda_l2": 20.0},
    "D1e_minleaf100": {"min_data_in_leaf": 100},
}

def cos_uncenter(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def run(name, params):
    dtr = lgb.Dataset(X_tr, y_tr)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)
    t0 = time.time()
    model = lgb.train(params, dtr, num_boost_round=10000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200)])
    p_va = model.predict(X_va, num_iteration=model.best_iteration)
    c = cos_uncenter(p_va, y_va)
    print(f"{name}: cos={c:.6f} iter={model.best_iteration} ({time.time()-t0:.1f}s)", flush=True)
    return c

BASE_CV = 0.130204
results = {}
for name, override in VARIANTS.items():
    p = dict(BASE)
    p.update(override)
    results[name] = run(name, p)

print("\n=== Exp D1 汇总 (baseline=%.6f) ===" % BASE_CV)
for name, c in results.items():
    print(f"{name}: {c:.6f} ({c-BASE_CV:+.6f})")
