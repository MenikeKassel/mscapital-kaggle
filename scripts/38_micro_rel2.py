# -*- coding: utf-8 -*-
"""
P1-1e: 微观特征第二轮 — 高漂移新特征归一化/相对化
P1-1b 发现 +micro 后 ΔAUC +0.0168 (新特征携带漂移) → 对到达率/强度类做相对化
相对化: o_arrival_rate→/t_arrival_rate, o_iat_mean_log→/t_iat, o_large_share→相对, 等
验证: PSEUDO/T4 CatBoost Alpha轴 + domain AUC Drift轴
"""
import time
import numpy as np
import polars as pl
from catboost import CatBoostRegressor
import lightgbm as lgb
from scipy import stats

FEAT = r"D:\mscapital-forecasting\data\processed\train_features.parquet"
MICRO = r"D:\mscapital-forecasting\data\processed\micro_features_train.parquet"
MICRO_TE = r"D:\mscapital-forecasting\data\processed\micro_features_test.parquet"
N_THREADS = 12

tr = pl.read_parquet(FEAT)
mi = pl.read_parquet(MICRO)
mi_te = pl.read_parquet(MICRO_TE)
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

def make_R3(mi_df):
    """第二轮: 对高漂移新特征做相对化"""
    return mi_df.with_columns([
        (pl.col("o_arrival_rate") / (pl.col("t_arrival_rate") + 1e-4)).alias("o_arrival_rel"),
        (pl.col("o_iat_cv") - pl.col("t_iat_cv")).alias("o_iat_cv_diff"),
        (pl.col("o_large_share") / (pl.col("t_large_share") + 1e-4)).alias("o_large_share_rel"),
        (pl.col("o_burst_ratio") / (pl.col("t_iat_cv") + 1e-4)).alias("o_burst_rel"),
        (pl.col("t_arrival_rate") / (pl.col("o_arrival_rate") + 1e-4)).alias("t_arrival_rel"),
        (pl.col("o_add_cancel_ratio") / (pl.col("t_arrival_rate") + 1e-4)).alias("o_ac_ratio_rel"),
    ])

tr_all = make_R2(tr.select(["sample_id"] + all_feats)).join(make_R3(mi), on="sample_id", how="left")
tr_all = tr_all.with_columns([pl.col(c).fill_null(0.0) for c in micro_cols])
cols = all_feats + micro_cols
X_all = tr_all.select(cols).to_numpy().astype(np.float32)
y_all = tr["target"].to_numpy().astype(np.float32)
m_all = tr["month"].to_numpy().astype(np.int32)
print(f"X {X_all.shape}", flush=True)

te_all = make_R2(pl.read_parquet(FEAT.replace("train_features", "test_features")).select(["sample_id"] + all_feats))
te_all = te_all.join(make_R3(mi_te), on="sample_id", how="left")
te_all = te_all.with_columns([pl.col(c).fill_null(0.0) for c in micro_cols])
X_te = te_all.select(cols).to_numpy().astype(np.float32)

def cos_uncenter(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def run_cat(X, y, Xv, yv):
    cb = CatBoostRegressor(iterations=8000, learning_rate=0.02, depth=6, l2_leaf_reg=5.0,
                           subsample=0.8, colsample_bylevel=0.8, loss_function="RMSE",
                           early_stopping_rounds=200, verbose=0, thread_count=N_THREADS, random_seed=0)
    cb.fit(X, y, eval_set=(Xv, yv))
    return cb.predict(Xv)

FOLDS = {"T4": (0, 50, 61, 70), "PSEUDO": (0, 32, 33, 70)}
print("\n=== Alpha 轴: R2+micro(旧) vs R2+micro+rel(新) ===", flush=True)
T0 = time.time()
scores = {"old": {}, "new": {}}
for fname, (a, b, c0, d0) in FOLDS.items():
    sel_t = (m_all >= a) & (m_all <= b); sel_v = (m_all >= c0) & (m_all <= d0)
    X, y, Xv, yv = X_all[sel_t], y_all[sel_t], X_all[sel_v], y_all[sel_v]
    p = run_cat(X, y, Xv, yv)
    scores["old"][fname] = cos_uncenter(p, yv)
    print(f"  {fname}: R2+micro = {scores['old'][fname]:.6f} ({time.time()-T0:.0f}s)", flush=True)
r3_cols = ["o_arrival_rel", "o_iat_cv_diff", "o_large_share_rel", "o_burst_rel", "t_arrival_rel", "o_ac_ratio_rel"]
drop_cols = ["o_arrival_rate", "o_iat_cv", "o_large_share", "o_burst_ratio", "t_arrival_rate", "o_add_cancel_ratio"]
cols2 = cols + r3_cols
X_all2 = tr_all.select(cols2).to_numpy().astype(np.float32)
X_te2 = te_all.select(cols2).to_numpy().astype(np.float32)
print(f"X2 {X_all2.shape} (含 R3 相对化列)", flush=True)
new_cols = [c for c in cols2 if c not in drop_cols]
new_idx = [cols2.index(c) for c in new_cols]
for fname, (a, b, c0, d0) in FOLDS.items():
    sel_t = (m_all >= a) & (m_all <= b); sel_v = (m_all >= c0) & (m_all <= d0)
    X, y, Xv, yv = X_all2[sel_t][:, new_idx], y_all[sel_t], X_all2[sel_v][:, new_idx], y_all[sel_v]
    p = run_cat(X, y, Xv, yv)
    scores["new"][fname] = cos_uncenter(p, yv)
    d = scores["new"][fname] - scores["old"][fname]
    print(f"  {fname}: R2+micro+rel = {scores['new'][fname]:.6f} ({d:+.5f}) ({time.time()-T0:.0f}s)", flush=True)

print("\n=== Drift 轴: domain AUC (full train vs test) ===")
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

a_old = domain_auc(X_all, X_te)
a_new = domain_auc(X_all2[:, new_idx], X_te2[:, new_idx])
print(f"  R2+micro: AUC={a_old:.4f} | R2+micro+rel: AUC={a_new:.4f} ({a_new-a_old:+.4f})")

print(f"\n=== P1-1e 汇总 (总 {time.time()-T0:.0f}s) ===")
print(f"Alpha mean Δ: {np.mean([scores['new'][f]-scores['old'][f] for f in FOLDS]):+.5f}")
print(f"Drift ΔAUC: {a_new-a_old:+.4f}")
