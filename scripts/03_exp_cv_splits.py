# -*- coding: utf-8 -*-
"""
Exp A1: CV 协议敏感性 — 同特征(90)/同参数, 不同 month split。
RQ1: 可靠 CV 能否建立? CV 对切分有多敏感?
对照: 社区 bestwater 数据 (同协议): CV1 0.1237 / CV2 0.1269 / CV3 0.1266 / CV4 0.1306 / CV5 0.1217 (35特征版)
本实验用 90特征官方特征集。
"""
import time
import numpy as np
import polars as pl
import lightgbm as lgb

FEAT = r"D:\mscapital-forecasting\data\processed\train_features.parquet"
N_THREADS = 12

tr = pl.read_parquet(FEAT)
feat_cols = [c for c in tr.columns if c not in ("sample_id", "month", "target")]
print(f"features: {len(feat_cols)}, rows: {tr.height:,}", flush=True)

PARAMS = dict(
    objective="regression", metric="rmse",
    learning_rate=0.02, num_leaves=32, min_data_in_leaf=300,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    lambda_l2=5.0, max_bin=255, verbose=-1, num_threads=N_THREADS, seed=0)

def cos_uncenter(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

splits = [
    ("CV1", 50),   # train 0-50 / valid 51-70  (官方B1)
    ("CV2", 55),
    ("CV3", 60),
    ("CV4", 65),
]
results = []
for name, cut in splits:
    tr_df = tr.filter(pl.col("month") <= cut)
    va_df = tr.filter((pl.col("month") > cut) & (pl.col("month") <= 70))
    X_tr = tr_df.select(feat_cols).to_numpy().astype(np.float32)
    y_tr = tr_df["target"].to_numpy().astype(np.float32)
    X_va = va_df.select(feat_cols).to_numpy().astype(np.float32)
    y_va = va_df["target"].to_numpy().astype(np.float32)
    t0 = time.time()
    dtr = lgb.Dataset(X_tr, y_tr)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)
    model = lgb.train(PARAMS, dtr, num_boost_round=10000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200)])
    p_va = model.predict(X_va, num_iteration=model.best_iteration)
    c = cos_uncenter(p_va, y_va)
    results.append((name, cut, tr_df.height, va_df.height, c, model.best_iteration, time.time()-t0))
    print(f"{name} (cut={cut}): train={tr_df.height:,} valid={va_df.height:,} "
          f"valid_cos={c:.6f} best_iter={model.best_iteration} ({time.time()-t0:.1f}s)", flush=True)

print("\n=== Exp A1 汇总 ===")
for r in results:
    print(f"{r[0]} cut={r[1]}: cos={r[4]:.6f} iter={r[5]}")
cos_vals = [r[4] for r in results]
print(f"mean={np.mean(cos_vals):.6f} std={np.std(cos_vals):.6f} range=[{min(cos_vals):.6f},{max(cos_vals):.6f}]")
