# -*- coding: utf-8 -*-
"""
P1-1b: 双轴筛选 — 新微观结构特征 × (Alpha轴: temporal cos 增量) × (Drift轴: domain AUC 增量)
基线: R2 特征 (90, 归一化替换) CatBoost
folds: PSEUDO / T3 / T4 (远期为主)
"""
import time
import numpy as np
import polars as pl
from catboost import CatBoostRegressor
import lightgbm as lgb

FEAT = r"D:\mscapital-forecasting\data\processed\train_features.parquet"
MICRO = r"D:\mscapital-forecasting\data\processed\micro_features_train.parquet"
MICRO_TE = r"D:\mscapital-forecasting\data\processed\micro_features_test.parquet"
N_THREADS = 12

tr = pl.read_parquet(FEAT)
mi = pl.read_parquet(MICRO)
mi_te = pl.read_parquet(MICRO_TE)
all_feats = [c for c in tr.columns if c not in ("sample_id", "month", "target")]
micro_cols = [c for c in mi.columns if c != "sample_id"]

# R2 归一化 (polars)
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
print(f"train+micro: {tr_all.shape} (90 + {len(micro_cols)} 新特征)", flush=True)

def cos_uncenter(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def run_cat(X, y, Xv, yv):
    cb = CatBoostRegressor(iterations=8000, learning_rate=0.02, depth=6, l2_leaf_reg=5.0,
                           subsample=0.8, colsample_bylevel=0.8, loss_function="RMSE",
                           early_stopping_rounds=200, verbose=0, thread_count=N_THREADS, random_seed=0)
    cb.fit(X, y, eval_set=(Xv, yv))
    return cb.predict(Xv)

FOLDS = {"T3": (0, 50, 51, 60), "T4": (0, 50, 61, 70), "PSEUDO": (0, 32, 33, 70)}
m_all = tr["month"].to_numpy().astype(np.int32)

# 特征矩阵 (R2 列 + micro 列)
base_cols = all_feats  # 已 R2 替换
cols = base_cols + micro_cols
X_all = tr_all.select(cols).to_numpy().astype(np.float32)
y_all = tr["target"].to_numpy().astype(np.float32)

# 1) Alpha 轴: 基线 vs +micro (PSEUDO/T3/T4)
print("\n=== Alpha 轴: CatBoost R2 vs R2+micro ===", flush=True)
T0 = time.time()
base_scores, micro_scores = {}, {}
for fname, (a, b, c0, d0) in FOLDS.items():
    sel_t = (m_all >= a) & (m_all <= b); sel_v = (m_all >= c0) & (m_all <= d0)
    # 基线 (90 R2)
    Xb = X_all[sel_t][:, :len(base_cols)]; Xbv = X_all[sel_v][:, :len(base_cols)]
    t0 = time.time()
    p = run_cat(Xb, y_all[sel_t], Xbv, y_all[sel_v])
    c0s = cos_uncenter(p, y_all[sel_v])
    base_scores[fname] = c0s
    print(f"  {fname} base: {c0s:.6f} ({time.time()-t0:.0f}s)", flush=True)
    # +micro
    Xm = X_all[sel_t]; Xmv = X_all[sel_v]
    t0 = time.time()
    p = run_cat(Xm, y_all[sel_t], Xmv, y_all[sel_v])
    cm = cos_uncenter(p, y_all[sel_v])
    micro_scores[fname] = cm
    print(f"  {fname} +micro: {cm:.6f} ({cm-c0s:+.5f}) ({time.time()-t0:.0f}s)", flush=True)

print("\nAlpha 轴汇总:")
for fname in FOLDS:
    print(f"  {fname}: {base_scores[fname]:.5f} → {micro_scores[fname]:.5f} ({micro_scores[fname]-base_scores[fname]:+.5f})")
print(f"  mean Δ: {np.mean([micro_scores[f]-base_scores[f] for f in FOLDS]):+.5f}")

# 2) Drift 轴: domain AUC (full train vs test, 基线 vs +micro)
print("\n=== Drift 轴: domain AUC (full train vs test) ===", flush=True)
te_all = make_R2(pl.read_parquet(FEAT.replace("train_features", "test_features")).select(["sample_id"] + all_feats))
te_all = te_all.join(mi_te, on="sample_id", how="left")
te_all = te_all.with_columns([pl.col(c).fill_null(0.0) for c in micro_cols])
X_te = te_all.select(cols).to_numpy().astype(np.float32)

from scipy import stats
def auc(y_true, y_score):
    n_pos = int(y_true.sum()); n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    r = stats.rankdata(y_score)
    return float((r[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def domain_auc(Xa, Xb):
    X = np.vstack([Xa, Xb])
    y = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])
    idx = np.random.RandomState(0).permutation(len(X))
    cut = int(len(X) * 0.8)
    p = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=31,
             min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
             lambda_l2=1.0, verbose=-1, num_threads=N_THREADS, seed=0)
    m = lgb.train(p, lgb.Dataset(X[idx[:cut]], y[idx[:cut]]), 1000,
                  valid_sets=[lgb.Dataset(X[idx[cut:]], y[idx[cut:]])], callbacks=[lgb.early_stopping(30)])
    return auc(y, m.predict(X, num_iteration=m.best_iteration))

t0 = time.time()
a_base = domain_auc(X_all[:, :len(base_cols)], X_te[:, :len(base_cols)])
print(f"  base (90 R2): AUC={a_base:.4f} ({time.time()-t0:.0f}s)", flush=True)
t0 = time.time()
a_micro = domain_auc(X_all, X_te)
print(f"  +micro: AUC={a_micro:.4f} ({a_micro-a_base:+.4f}) ({time.time()-t0:.0f}s)", flush=True)

print(f"\n=== P1-1b 汇总 (总 {time.time()-T0:.0f}s) ===")
print(f"Alpha: mean Δcos = {np.mean([micro_scores[f]-base_scores[f] for f in FOLDS]):+.5f}")
print(f"Drift: ΔAUC = {a_micro-a_base:+.4f}")
print("判定: Δcos>0 且 ΔAUC 不大幅上升 → 新特征是良性增量")
