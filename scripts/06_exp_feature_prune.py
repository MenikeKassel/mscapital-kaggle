# -*- coding: utf-8 -*-
"""
Exp B2: 特征精简 — 删除无用组后 CV 是否提升?
Control: 90特征 (CV1 = 0.130204)
Treatment: V1=77特征 (删EWM6+t基础7), V2=84特征 (只删EWM6)
结论预期: 若 V1/V2 CV >= 0.1302, 采纳精简集 (更少特征=更少过拟合=更好泛化)
"""
import re
import time
import numpy as np
import polars as pl
import lightgbm as lgb

FEAT = r"D:\mscapital-forecasting\data\processed\train_features.parquet"
N_THREADS = 12

tr = pl.read_parquet(FEAT)
all_feats = [c for c in tr.columns if c not in ("sample_id", "month", "target")]

def is_ewm(f):
    return "_ewm_" in f and not f.startswith("x_")

def is_tx_base(f):
    return f.startswith("t_") and not re.search(r"_(15|45|60|120|180)$", f)

V1 = [f for f in all_feats if not is_ewm(f) and not is_tx_base(f)]   # 77
V2 = [f for f in all_feats if not is_ewm(f)]                          # 84
print(f"V1(删EWM+成交基础): {len(V1)} | V2(只删EWM): {len(V2)} | 全集: {len(all_feats)}", flush=True)

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
r1 = run("V1-77feat", V1)
r2 = run("V2-84feat", V2)
print(f"\n=== Exp B2 汇总 ===\n90特征: {BASE:.6f}\nV1(77): {r1:.6f} ({r1-BASE:+.6f})\nV2(84): {r2:.6f} ({r2-BASE:+.6f})")
