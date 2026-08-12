# -*- coding: utf-8 -*-
"""
Exp G3: 三模型融合 v2 — LGBM + XGB + MLP-ens(3seed×30ep)
"""
import time
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import lightgbm as lgb
import xgboost as xgb

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

def cos_uncenter(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

preds = {}
t0 = time.time()
params_l = dict(objective="regression", metric="rmse", learning_rate=0.02, num_leaves=64,
                min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                lambda_l2=5.0, max_bin=255, verbose=-1, num_threads=N_THREADS, seed=0)
dtr_l = lgb.Dataset(X_tr, y_tr); dva_l = lgb.Dataset(X_va, y_va, reference=dtr_l)
m_l = lgb.train(params_l, dtr_l, 10000, valid_sets=[dva_l], callbacks=[lgb.early_stopping(200)])
preds["lgb"] = m_l.predict(X_va, num_iteration=m_l.best_iteration)
print(f"LGBM: {cos_uncenter(preds['lgb'], y_va):.6f} ({time.time()-t0:.0f}s)", flush=True)

t0 = time.time()
params_x = dict(objective="reg:squarederror", eval_metric="rmse", eta=0.02, max_depth=5,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, nthread=N_THREADS, seed=0)
m_x = xgb.train(params_x, xgb.DMatrix(X_tr, label=y_tr), 10000,
                evals=[(xgb.DMatrix(X_va, label=y_va), "va")], early_stopping_rounds=200, verbose_eval=False)
preds["xgb"] = m_x.predict(xgb.DMatrix(X_va), iteration_range=(0, m_x.best_iteration + 1))
print(f"XGB: {cos_uncenter(preds['xgb'], y_va):.6f} ({time.time()-t0:.0f}s)", flush=True)

class MLP(nn.Module):
    def __init__(self, n_in, h=256, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.GELU(), nn.Dropout(p),
            nn.Linear(h, h), nn.GELU(), nn.Dropout(p),
            nn.Linear(h, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tr_c = np.nan_to_num(X_tr, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
X_va_c = np.nan_to_num(X_va, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
mu = X_tr_c.mean(axis=0, keepdims=True); sd = X_tr_c.std(axis=0, keepdims=True) + 1e-6
X_va_s = ((X_va_c - mu) / sd).clip(-10, 10)
y_tr_mean = y_tr.mean(); y_tr_std = y_tr.std() + 1e-6
mlp_preds = []
for seed in [2026, 7, 123]:
    model = MLP(X_tr.shape[1]).to(device)
    model.load_state_dict(torch.load(rf"D:\mscapital-kaggle\output\mlp_seed{seed}.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        mlp_preds.append(model(torch.from_numpy(X_va_s).to(device)).cpu().numpy() * y_tr_std + y_tr_mean)
preds["mlp"] = np.mean(mlp_preds, axis=0)
print(f"MLP-ens: {cos_uncenter(preds['mlp'], y_va):.6f}", flush=True)

names = list(preds)
print("相关性矩阵:")
for a in names:
    print(f"  {a}: " + "  ".join(f"{np.corrcoef(preds[a], preds[b])[0,1]:.3f}" for b in names))

best = (-1, None)
for wl in np.arange(0, 1.01, 0.1):
    for wx in np.arange(0, 1.01 - wl, 0.1):
        wm = 1 - wl - wx
        c = cos_uncenter(wl * preds["lgb"] + wx * preds["xgb"] + wm * preds["mlp"], y_va)
        if c > best[0]:
            best = (c, (round(wl, 2), round(wx, 2), round(wm, 2)))
print(f"\n=== Exp G3 汇总 ===")
for k in names:
    print(f"{k}: {cos_uncenter(preds[k], y_va):.6f}")
print(f"blend w={best[1]}: {best[0]:.6f} ({(best[0]-max(cos_uncenter(preds[k], y_va) for k in names))*1000:+.3f}k vs best single)")
