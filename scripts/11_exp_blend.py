# -*- coding: utf-8 -*-
"""
Exp G1: LGBM + MLP 融合 — 错误相关性分析 + 权重搜索
前提: 需要 mlp_best.pt 已训练 (10_exp_mlp.py)
协议: CV1。先算两者预测相关性, 再网格搜索融合权重。
"""
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import lightgbm as lgb
import time

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

# --- LGBM (leaves64 采纳参数) ---
PARAMS = dict(
    objective="regression", metric="rmse",
    learning_rate=0.02, num_leaves=64, min_data_in_leaf=300,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    lambda_l2=5.0, max_bin=255, verbose=-1, num_threads=N_THREADS, seed=0)
t0 = time.time()
dtr = lgb.Dataset(X_tr, y_tr)
dva = lgb.Dataset(X_va, y_va, reference=dtr)
model = lgb.train(PARAMS, dtr, num_boost_round=10000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(200)])
p_lgb = model.predict(X_va, num_iteration=model.best_iteration)
c_lgb = cos_uncenter(p_lgb, y_va)
print(f"LGBM(leaves64): cos={c_lgb:.6f} iter={model.best_iteration} ({time.time()-t0:.1f}s)", flush=True)

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
y_tr_mean = y_tr.mean()
y_tr_std = y_tr.std() + 1e-6
mlp = MLP(X_tr.shape[1]).to(device)
mlp.load_state_dict(torch.load(r"D:\mscapital-kaggle\output\mlp_best.pt", map_location=device))
mlp.eval()
with torch.no_grad():
    p_mlp = mlp(torch.from_numpy(X_va_s).to(device)).cpu().numpy() * y_tr_std + y_tr_mean
c_mlp = cos_uncenter(p_mlp, y_va)
print(f"MLP: cos={c_mlp:.6f}", flush=True)

# --- 相关性分析 ---
corr = np.corrcoef(p_lgb, p_mlp)[0, 1]
print(f"LGBM vs MLP 预测相关性: {corr:.4f}", flush=True)

# --- 权重网格搜索 (0.0-1.0) ---
best_w, best_c = -1, -1
for w in np.arange(0.0, 1.01, 0.05):
    p = w * p_lgb + (1 - w) * p_mlp
    c = cos_uncenter(p, y_va)
    if c > best_c:
        best_c, best_w = c, w
print(f"\n=== Exp G1 汇总 ===")
print(f"LGBM alone:  {c_lgb:.6f}")
print(f"MLP alone:   {c_mlp:.6f}")
print(f"blend w={best_w:.2f}: {best_c:.6f} ({(best_c-max(c_lgb,c_mlp))*1000:+.3f}k vs best single)")
