# -*- coding: utf-8 -*-
"""
Exp G2: 三模型融合 — LGBM + XGBoost + MLP
1) XGB单模型 CV1; 2) 相关性矩阵; 3) 3维权重网格搜索
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

# --- LGBM leaves64 ---
t0 = time.time()
params_l = dict(objective="regression", metric="rmse", learning_rate=0.02, num_leaves=64,
                min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                lambda_l2=5.0, max_bin=255, verbose=-1, num_threads=N_THREADS, seed=0)
m_l = lgb.train(params_l, lgb.Dataset(X_tr, y_tr), 10000,
                valid_sets=[lgb.Dataset(X_va, y_va, reference=lgb.Dataset(X_tr, y_tr))],
                callbacks=[lgb.early_stopping(200)])
preds["lgb"] = m_l.predict(X_va, num_iteration=m_l.best_iteration)
print(f"LGBM: cos={cos_uncenter(preds['lgb'], y_va):.6f} ({time.time()-t0:.1f}s)", flush=True)

# --- XGBoost ---
t0 = time.time()
params_x = dict(objective="reg:squarederror", eval_metric="rmse", eta=0.02, max_depth=5,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, nthread=N_THREADS, seed=0)
dtr_x = xgb.DMatrix(X_tr, label=y_tr)
dva_x = xgb.DMatrix(X_va, label=y_va)
m_x = xgb.train(params_x, dtr_x, 10000, evals=[(dva_x, "va")], early_stopping_rounds=200, verbose_eval=False)
preds["xgb"] = m_x.predict(dva_x, iteration_range=(0, m_x.best_iteration + 1))
print(f"XGB:  cos={cos_uncenter(preds['xgb'], y_va):.6f} iter={m_x.best_iteration} ({time.time()-t0:.1f}s)", flush=True)

# --- MLP ---
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
mu = X_tr_c.mean(axis=0, keepdims=True)
sd = X_tr_c.std(axis=0, keepdims=True) + 1e-6
X_va_s = ((X_va_c - mu) / sd).clip(-10, 10)
y_tr_mean = y_tr.mean(); y_tr_std = y_tr.std() + 1e-6
mlp = MLP(X_tr.shape[1]).to(device)
mlp.load_state_dict(torch.load(r"D:\mscapital-kaggle\output\mlp_best.pt", map_location=device))
mlp.eval()
with torch.no_grad():
    preds["mlp"] = mlp(torch.from_numpy(X_va_s).to(device)).cpu().numpy() * y_tr_std + y_tr_mean
print(f"MLP:  cos={cos_uncenter(preds['mlp'], y_va):.6f}", flush=True)

# --- 相关性矩阵 ---
print("\n预测相关性矩阵:")
names = list(preds)
for a in names:
    row = "  ".join(f"{np.corrcoef(preds[a], preds[b])[0,1]:.3f}" for b in names)
    print(f"  {a}: {row}")

# --- 3维权重网格 ---
best = (-1, None)
for wl in np.arange(0, 1.01, 0.1):
    for wx in np.arange(0, 1.01 - wl, 0.1):
        wm = 1 - wl - wx
        p = wl * preds["lgb"] + wx * preds["xgb"] + wm * preds["mlp"]
        c = cos_uncenter(p, y_va)
        if c > best[0]:
            best = (c, (round(wl, 2), round(wx, 2), round(wm, 2)))
print(f"\n=== Exp G2 汇总 ===")
for k in names:
    print(f"{k} alone: {cos_uncenter(preds[k], y_va):.6f}")
print(f"3-model blend w={best[1]}: {best[0]:.6f} ({(best[0]-max(cos_uncenter(preds[k], y_va) for k in names))*1000:+.3f}k vs best single)")
