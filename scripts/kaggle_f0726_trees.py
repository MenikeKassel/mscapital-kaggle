# -*- coding: utf-8 -*-
"""
P1-1h: 0726 特征 × 树模型 (Kaggle 云端)
CatBoost + LGBM 在 152 事件动力学特征上训练
验证: PSEUDO (m0-32/33-70, 诚实) + 80万切分 (作者协议, 对比)
全量训练 → test 预测 → 输出 npz (下载回本地融合)
"""
import time
import numpy as np
import polars as pl
import lightgbm as lgb
from catboost import CatBoostRegressor

DATA = "/kaggle/input/msc-f0726-data"
OUT = "/kaggle/working"
N_THREADS = 4

import os, glob
print("=== input listing ===", flush=True)
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files[:20]:
        print(os.path.join(root, f), flush=True)
parqs = glob.glob("/kaggle/input/**/*.parquet", recursive=True)
print("parquet files:", parqs, flush=True)
if not parqs:
    raise RuntimeError("no parquet mounted")

print("loading features...", flush=True)
train = pl.read_parquet(f"{DATA}/f0726_train_f32.parquet")
test = pl.read_parquet(f"{DATA}/f0726_test_f32.parquet")
label = pl.read_ipc("/kaggle/input/competitions/ms-capital-real-financial-market-forecasting/train/label.feather", memory_map=False)
feats = [c for c in train.columns if c not in ("sample_id", "target")]
train = train.join(label.select(["sample_id", "month"]), on="sample_id", how="left")
print(f"train {train.shape} test {test.shape} feats {len(feats)}", flush=True)

def cos_uncenter(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def fit_model(X, y, Xv, yv, kind):
    if kind == "cat":
        m = CatBoostRegressor(iterations=10000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                              subsample=0.8, colsample_bylevel=0.8, loss_function="RMSE",
                              early_stopping_rounds=200, verbose=0, thread_count=N_THREADS, random_seed=0)
        m.fit(X, y, eval_set=(Xv, yv))
        return m.predict(Xv)
    m = lgb.train(dict(objective="regression", metric="rmse", learning_rate=0.02, num_leaves=64,
                       min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                       lambda_l2=5.0, max_bin=255, verbose=-1, num_threads=N_THREADS, seed=0),
                  lgb.Dataset(X, y), 10000,
                  valid_sets=[lgb.Dataset(Xv, yv, reference=lgb.Dataset(X, y))],
                  callbacks=[lgb.early_stopping(200)])
    return m.predict(Xv, num_iteration=m.best_iteration)

# ===== CV 验证 =====
m_all = train["month"].to_numpy().astype(np.int32)
y_all = train["target"].to_numpy().astype(np.float32)
X_all = train.select(feats).to_numpy().astype(np.float32)
X_te = test.select(feats).to_numpy().astype(np.float32)

print("\n=== PSEUDO (m0-32/33-70, 诚实协议) ===", flush=True)
sel_t = m_all <= 32; sel_v = (m_all > 32) & (m_all <= 70)
for kind in ["cat", "lgb"]:
    t0 = time.time()
    p = fit_model(X_all[sel_t], y_all[sel_t], X_all[sel_v], y_all[sel_v], kind)
    print(f"  {kind} PSEUDO: {cos_uncenter(p, y_all[sel_v]):.6f} ({time.time()-t0:.0f}s)", flush=True)

print("\n=== 80万切分 (作者协议) ===", flush=True)
for kind in ["cat", "lgb"]:
    t0 = time.time()
    p = fit_model(X_all[:800000], y_all[:800000], X_all[800000:], y_all[800000:], kind)
    print(f"  {kind} 80万: {cos_uncenter(p, y_all[800000:]):.6f} ({time.time()-t0:.0f}s)", flush=True)

# ===== 全量训练 + test 预测 =====
print("\n=== full train + test predict ===", flush=True)
preds = {}
for kind in ["cat", "lgb"]:
    t0 = time.time()
    if kind == "cat":
        m = CatBoostRegressor(iterations=8000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                              subsample=0.8, colsample_bylevel=0.8, loss_function="RMSE",
                              verbose=0, thread_count=N_THREADS, random_seed=0)
        m.fit(X_all, y_all)
        preds[kind] = m.predict(X_te)
    else:
        m = lgb.train(dict(objective="regression", metric="rmse", learning_rate=0.02, num_leaves=64,
                           min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                           lambda_l2=5.0, max_bin=255, verbose=-1, num_threads=N_THREADS, seed=0),
                      lgb.Dataset(X_all, y_all), 2000)
        preds[kind] = m.predict(X_te)
    print(f"  {kind} full done ({time.time()-t0:.0f}s)", flush=True)

p_final = 0.5 * preds["cat"] + 0.5 * preds["lgb"]
np.savez(f"{OUT}/f0726_tree_test_pred.npz", pred=p_final, cat=preds["cat"], lgb=preds["lgb"],
         test_ids=test["sample_id"].to_numpy())
print(f"saved {OUT}/f0726_tree_test_pred.npz ({len(p_final):,})")
print("DONE", flush=True)
