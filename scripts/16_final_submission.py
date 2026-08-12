# -*- coding: utf-8 -*-
"""
Final: 三模型融合提交 — 全量训练(0-70月) + test预测
LGBM + XGB + MLP-ens(3seed×30ep), 权重 (0.1, 0.4, 0.5) [G3 CV1验证]
"""
import time
import random
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import lightgbm as lgb
import xgboost as xgb

FEAT = r"D:\mscapital-forecasting\data\processed\train_features.parquet"
N_THREADS = 12
OUT = r"D:\mscapital-kaggle\output\submissions\submission_blend_v1.csv"
W = {"lgb": 0.1, "xgb": 0.4, "mlp": 0.5}
LGB_ITER = 443   # 385 * 1.15
XGB_ITER = 815   # 708 * 1.15

tr = pl.read_parquet(FEAT)
all_feats = [c for c in tr.columns if c not in ("sample_id", "month", "target")]
te = pl.read_parquet(FEAT.replace("train_features", "test_features"))

X_tr = tr.select(all_feats).to_numpy().astype(np.float32)
y_tr = tr["target"].to_numpy().astype(np.float32)
X_te = te.select(all_feats).to_numpy().astype(np.float32)
ids_te = te["sample_id"].to_numpy()
print(f"train {X_tr.shape} test {X_te.shape}", flush=True)

# --- LGBM 全量 ---
t0 = time.time()
params_l = dict(objective="regression", metric="rmse", learning_rate=0.02, num_leaves=64,
                min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                lambda_l2=5.0, max_bin=255, verbose=-1, num_threads=N_THREADS, seed=0)
m_l = lgb.train(params_l, lgb.Dataset(X_tr, y_tr), LGB_ITER)
p_l = m_l.predict(X_te)
print(f"LGBM full: done ({time.time()-t0:.0f}s)", flush=True)

# --- XGB 全量 ---
t0 = time.time()
params_x = dict(objective="reg:squarederror", eta=0.02, max_depth=5,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, nthread=N_THREADS, seed=0)
m_x = xgb.train(params_x, xgb.DMatrix(X_tr, label=y_tr), XGB_ITER)
p_x = m_x.predict(xgb.DMatrix(X_te))
print(f"XGB full: done ({time.time()-t0:.0f}s)", flush=True)

# --- MLP 全量 3 seeds ---
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
X_te_c = np.nan_to_num(X_te, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
mu = X_tr_c.mean(axis=0, keepdims=True); sd = X_tr_c.std(axis=0, keepdims=True) + 1e-6
X_tr_s = ((X_tr_c - mu) / sd).clip(-10, 10)
X_te_s = ((X_te_c - mu) / sd).clip(-10, 10)
y_mean = y_tr.mean(); y_std = y_tr.std() + 1e-6
y_n = (y_tr - y_mean) / y_std
BATCH = 2048
EPOCHS = 30
n = len(X_tr)
Xte = torch.from_numpy(X_te_s).to(device)

def train_mlp(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = MLP(X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.MSELoss()
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb = torch.from_numpy(X_tr_s[idx.numpy()]).to(device)
            yb = torch.from_numpy(y_n[idx.numpy()]).to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        return model(Xte).cpu().numpy() * y_std + y_mean

t0 = time.time()
p_m = np.mean([train_mlp(s) for s in [2026, 7, 123]], axis=0)
print(f"MLP full 3seeds: done ({time.time()-t0:.0f}s)", flush=True)

# --- 融合 ---
p_final = W["lgb"] * p_l + W["xgb"] * p_x + W["mlp"] * p_m
sub = pl.DataFrame({"sample_id": pl.Series(ids_te, dtype=pl.Int32),
                    "prediction": pl.Series(p_final, dtype=pl.Float64)}).sort("sample_id")
sub.write_csv(OUT)
print(f"\nsaved {OUT} ({sub.height:,} rows)")
print(f"pred stats: mean={sub['prediction'].mean():.6f} std={sub['prediction'].std():.6f}")
print(f"NaN: {sub['prediction'].is_null().sum()}")
