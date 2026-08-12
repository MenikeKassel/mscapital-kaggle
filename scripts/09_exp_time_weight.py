# -*- coding: utf-8 -*-
"""
Exp E1: 时间衰减样本加权 — 分布漂移应对 (RQ7, 文献 P010/P012)
假设: 近期训练样本与测试期(71-108月)更相似, 加权可改善验证/泛化。
Control: 官方等权 (CV1 = 0.130204, leaves=32)
Treatment: 相同参数 + sample_weight (线性/指数/平方, 按 month)
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
m_tr = tr_df["month"].to_numpy().astype(np.float32)
X_va = va_df.select(all_feats).to_numpy().astype(np.float32)
y_va = va_df["target"].to_numpy().astype(np.float32)

PARAMS = dict(
    objective="regression", metric="rmse",
    learning_rate=0.02, num_leaves=32, min_data_in_leaf=300,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    lambda_l2=5.0, max_bin=255, verbose=-1, num_threads=N_THREADS, seed=0)

def cos_uncenter(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def run(name, w=None):
    dtr = lgb.Dataset(X_tr, y_tr, weight=w)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)
    t0 = time.time()
    model = lgb.train(PARAMS, dtr, num_boost_round=10000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200)])
    p_va = model.predict(X_va, num_iteration=model.best_iteration)
    c = cos_uncenter(p_va, y_va)
    print(f"{name}: cos={c:.6f} iter={model.best_iteration} ({time.time()-t0:.1f}s)", flush=True)
    return c

BASE = 0.130204
results = {}

# 线性: 权重从 month0 的 1 到 month50 的 2
w_lin = 1.0 + m_tr / 50.0
results["E1a_linear_2x"] = run("E1a_linear_2x", w_lin)

# 线性: 1 -> 3
w_lin3 = 1.0 + 2.0 * m_tr / 50.0
results["E1b_linear_3x"] = run("E1b_linear_3x", w_lin3)

# 指数: exp(month/25)  -> 月0:1, 月50: e^2≈7.4
w_exp = np.exp(m_tr / 25.0)
results["E1c_exp"] = run("E1c_exp", w_exp)

# 温和指数: exp(month/50) -> 月50: e≈2.7
w_exp2 = np.exp(m_tr / 50.0)
results["E1d_exp_mild"] = run("E1d_exp_mild", w_exp2)

print("\n=== Exp E1 汇总 (baseline=%.6f) ===" % BASE)
for name, c in results.items():
    print(f"{name}: {c:.6f} ({c-BASE:+.6f})")
