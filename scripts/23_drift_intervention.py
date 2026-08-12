# -*- coding: utf-8 -*-
"""
P0.5-C: Drift Intervention Ablation (GPT 双评审拍板)
R0 = 原90特征 | R1 = 删top10漂移特征 | R2 = 归一化替换 (relative spread/depth/rv/flow)
folds: T3/T4/H2/PSEUDO (Long-Horizon Stress Fold), 模型 CatBoost
同时输出 monthly_target_diagnostics.csv (target 形状诊断)
"""
import time
import numpy as np
import polars as pl
import csv
from catboost import CatBoostRegressor

FEAT = r"D:\mscapital-forecasting\data\processed\train_features.parquet"
N_THREADS = 12

tr = pl.read_parquet(FEAT)
all_feats = [c for c in tr.columns if c not in ("sample_id", "month", "target")]
X_all = tr.select(all_feats).to_numpy().astype(np.float32)
y_all = tr["target"].to_numpy().astype(np.float32)
m_all = tr["month"].to_numpy().astype(np.int32)

# ===== target 月度诊断 =====
print("=== monthly target diagnostics ===", flush=True)
with open(r"D:\mscapital-kaggle\output\monthly_target_diagnostics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["month", "n", "mean", "std", "mad", "p50_abs", "p90_abs", "p99_abs", "skew", "kurt", "pos_ratio", "top1pct_y2_share"])
    for mo in range(71):
        sel = m_all == mo
        y = y_all[sel]
        if len(y) < 100:
            continue
        absy = np.abs(y)
        y2 = y ** 2
        top1 = np.quantile(y2, 0.99)
        row = [mo, len(y), float(y.mean()), float(y.std()), float(np.median(absy - np.median(absy))),
               float(np.quantile(absy, 0.5)), float(np.quantile(absy, 0.9)), float(np.quantile(absy, 0.99)),
               float(np.mean(((y - y.mean()) / (y.std() + 1e-9)) ** 3)),
               float(np.mean(((y - y.mean()) / (y.std() + 1e-9)) ** 4)),
               float((y > 0).mean()), float(y2[y2 >= top1].sum() / (y2.sum() + 1e-12))]
        w.writerow(row)
print("saved output/monthly_target_diagnostics.csv", flush=True)

# ===== drift 特征集合 =====
DRIFT_TOP10 = ["m_sp_mean", "m_depth_mean", "m_rv", "o_vol_sum", "o_n_120",
               "m_txv_sum_180", "t_lv_mean", "o_add_ratio", "m_sp_mean_60", "m_sp_mean_180"]

def build_R2_cols():
    """归一化替换: raw -> relative"""
    out = []
    d = {}
    for c in all_feats:
        d[c] = c
    d["m_sp_mean"] = "m_sp_mean / m_mid_mean"
    d["m_depth_mean"] = "m_depth_mean / (m_txv_sum + 1)"
    d["m_rv"] = "m_rv / (m_mid_std + 1e-8)"
    d["o_vol_sum"] = "o_vol_sum / (t_vol_sum + 1)"
    d["o_n_120"] = "o_n_120 / (t_n_120 + 1)"
    d["m_txv_sum_180"] = "m_txv_sum_180 / (m_txv_sum + 1)"
    d["m_sp_mean_60"] = "m_sp_mean_60 / (m_mid_mean_60 + 1e-8)"
    d["m_sp_mean_180"] = "m_sp_mean_180 / (m_mid_mean_180 + 1e-8)"
    return d

def make_R2_X(X):
    """对 numpy 特征矩阵做归一化替换 (按 all_feats 顺序)"""
    d = build_R2_cols()
    idx = {c: i for i, c in enumerate(all_feats)}
    Xr = X.copy()
    def col(name):
        return X[:, idx[name]]
    # m_sp_mean / m_mid_mean
    Xr[:, idx["m_sp_mean"]] = col("m_sp_mean") / (col("m_mid_mean") + 1e-8)
    Xr[:, idx["m_depth_mean"]] = col("m_depth_mean") / (col("m_txv_sum_60") + 1.0)
    Xr[:, idx["m_rv"]] = col("m_rv") / (col("m_mid_std") + 1e-8)
    Xr[:, idx["o_vol_sum"]] = col("o_vol_sum") / (col("t_vol_sum") + 1.0)
    Xr[:, idx["o_n_120"]] = col("o_n_120") / (col("t_n_120") + 1.0)
    Xr[:, idx["m_txv_sum_180"]] = col("m_txv_sum_180") / (col("m_txv_sum_60") + 1.0)
    Xr[:, idx["m_sp_mean_60"]] = col("m_sp_mean_60") / (col("m_mid_mean_60") + 1e-8)
    Xr[:, idx["m_sp_mean_180"]] = col("m_sp_mean_180") / (col("m_mid_mean_180") + 1e-8)
    return Xr

def cos_uncenter(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def run_cat(X, y, Xv, yv):
    cb = CatBoostRegressor(iterations=10000, learning_rate=0.02, depth=6, l2_leaf_reg=5.0,
                           subsample=0.8, colsample_bylevel=0.8, loss_function="RMSE",
                           early_stopping_rounds=200, verbose=0, thread_count=N_THREADS, random_seed=0)
    cb.fit(X, y, eval_set=(Xv, yv))
    return cb.predict(Xv)

FOLDS = {"T3": (0, 50, 51, 60), "T4": (0, 50, 61, 70),
         "H2": (0, 40, 51, 60), "PSEUDO": (0, 32, 33, 70)}
R1_COLS = [c for c in all_feats if c not in DRIFT_TOP10]
print(f"R0: {len(all_feats)} 特征 | R1(删top10): {len(R1_COLS)} 特征", flush=True)

T0 = time.time()
results = {}
for fname, (a, b, c0, d0) in FOLDS.items():
    sel_t = (m_all >= a) & (m_all <= b); sel_v = (m_all >= c0) & (m_all <= d0)
    X, y, Xv, yv = X_all[sel_t], y_all[sel_t], X_all[sel_v], y_all[sel_v]
    print(f"\n=== {fname} (train m{a}-{b}, valid m{c0}-{d0}) ===", flush=True)
    # R0
    t0 = time.time()
    p = run_cat(X, y, Xv, yv)
    c_r0 = cos_uncenter(p, yv)
    print(f"  R0 (90): {c_r0:.6f} ({time.time()-t0:.0f}s)", flush=True)
    # R1
    t0 = time.time()
    Xr1 = X[:, [all_feats.index(c) for c in R1_COLS]]
    Xr1v = Xv[:, [all_feats.index(c) for c in R1_COLS]]
    p = run_cat(Xr1, y, Xr1v, yv)
    c_r1 = cos_uncenter(p, yv)
    print(f"  R1 (删10): {c_r1:.6f} ({c_r1-c_r0:+.6f}) ({time.time()-t0:.0f}s)", flush=True)
    # R2
    t0 = time.time()
    p = run_cat(make_R2_X(X), y, make_R2_X(Xv), yv)
    c_r2 = cos_uncenter(p, yv)
    print(f"  R2 (归一): {c_r2:.6f} ({c_r2-c_r0:+.6f}) ({time.time()-t0:.0f}s)", flush=True)
    results[fname] = (c_r0, c_r1, c_r2)

print("\n\n========== P0.5-C 汇总 (CatBoost) ==========")
print(f"{'fold':<8} {'R0(90)':>9} {'R1(删10)':>9} {'R2(归一)':>9}")
for fname in FOLDS:
    r0, r1, r2 = results[fname]
    print(f"{fname:<8} {r0:.5f} {r1:.5f} ({r1-r0:+.4f}) {r2:.5f} ({r2-r0:+.4f})")
r0s = [results[f][0] for f in FOLDS]; r2s = [results[f][2] for f in FOLDS]
print(f"\nR2-R0 mean: {np.mean(r2s)-np.mean(r0s):+.5f} | per-fold: {['%+.4f' % (r2-r0) for r0, r2 in zip(r0s, r2s)]}")
