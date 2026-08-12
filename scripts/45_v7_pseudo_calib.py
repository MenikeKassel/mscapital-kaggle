# -*- coding: utf-8 -*-
"""
v7 PSEUDO 定标: 表格(R2+micro) PSEUDO pred + RealMLP PSEUDO pred → 融合 cos
输出: v7 PSEUDO 分 → 更新 calibration.md Regime B
"""
import time
import numpy as np
import polars as pl
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.neural_network import MLPRegressor

DATA = r"D:\mscapital-forecasting\data\processed"
P12 = rf"{DATA}\p12_out"

t0 = time.time()
label = pl.read_ipc(rf"{DATA}\..\raw\train\label.feather", memory_map=False)
tr = pl.read_parquet(f"{DATA}/train_features.parquet")
if "month" in tr.columns:
    tr = tr.drop("month")
if "target" in tr.columns:
    tr = tr.drop("target")
tr = tr.join(label.select(["sample_id", "month", "target"]), on="sample_id")
te = pl.read_parquet(f"{DATA}/test_features.parquet")
micro_tr = pl.read_parquet(f"{DATA}/micro_features_train.parquet").rename({"sample_id": "sample_id"})
micro_te = pl.read_parquet(f"{DATA}/micro_features_test.parquet")
tr = tr.join(micro_tr, on="sample_id", how="left")
te = te.join(micro_te, on="sample_id", how="left")

# R2 归一化 (漂移干预: 除以 train 均值)
drop = ["sample_id", "month", "target"]
all_feats = [c for c in tr.columns if c not in drop]
X_all = tr.select(all_feats).to_numpy()
X_te = te.select(all_feats).to_numpy()
for j in range(X_all.shape[1]):
    m = np.nanmean(X_all[:, j])
    X_all[:, j] = np.nan_to_num(X_all[:, j] / (m + 1e-12), nan=0.0)
    X_te[:, j] = np.nan_to_num(X_te[:, j] / (m + 1e-12), nan=0.0)

# PSEUDO fold
sel_t = (tr["month"] <= 32).to_numpy()
sel_v = ((tr["month"] > 32) & (tr["month"] <= 70)).to_numpy()
Xt, Xv = X_all[sel_t], X_all[sel_v]
yt = tr.filter(pl.Series(sel_t))["target"].to_numpy()
yv = tr.filter(pl.Series(sel_v))["target"].to_numpy()
print(f"train {Xt.shape} valid {Xv.shape} ({time.time()-t0:.0f}s)", flush=True)

def cos_u(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

preds = {}
# LGBM
m = LGBMRegressor(n_estimators=2000, learning_rate=0.05, num_leaves=63, min_child_samples=300,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=8, random_state=0, verbose=-1)
m.fit(Xt, yt, eval_set=[(Xv, yv)], callbacks=[__import__("lightgbm").early_stopping(200, verbose=False)])
preds["lgb"] = m.predict(Xv)
print(f"  lgb PSEUDO {cos_u(preds['lgb'], yv):.6f} ({time.time()-t0:.0f}s)", flush=True)

# XGB
m = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=7, min_child_weight=300,
                     subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=8, random_state=0, verbosity=0,
                     early_stopping_rounds=200)
m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
preds["xgb"] = m.predict(Xv)
print(f"  xgb PSEUDO {cos_u(preds['xgb'], yv):.6f} ({time.time()-t0:.0f}s)", flush=True)

# CatBoost
m = CatBoostRegressor(iterations=5000, learning_rate=0.05, depth=6, l2_leaf_reg=5.0, subsample=0.8,
                      colsample_bylevel=0.8, loss_function="RMSE", early_stopping_rounds=200,
                      verbose=0, thread_count=8, random_seed=0)
m.fit(Xt, yt, eval_set=(Xv, yv))
preds["cat"] = m.predict(Xv)
print(f"  cat PSEUDO {cos_u(preds['cat'], yv):.6f} ({time.time()-t0:.0f}s)", flush=True)

# MLP (与 29 脚本完全一致: 标准化 + GELU + AdamW + Cosine + best-state)
import torch
import torch.nn as nn
import random

class _MLP(nn.Module):
    def __init__(self, n_in, h=256, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.GELU(), nn.Dropout(p),
            nn.Linear(h, h), nn.GELU(), nn.Dropout(p),
            nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

def fit_mlp_best(X, y, Xv, yv, epochs=30, seeds=(2026, 7, 123), batch=2048):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xc = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
    Xvc = np.nan_to_num(Xv, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
    mu = Xc.mean(axis=0, keepdims=True); sd = Xc.std(axis=0, keepdims=True) + 1e-6
    Xs = ((Xc - mu) / sd).clip(-10, 10).astype(np.float32)
    Xvs = ((Xvc - mu) / sd).clip(-10, 10).astype(np.float32)
    y_mean, y_std = y.mean(), y.std() + 1e-6
    y_n = (y - y_mean) / y_std
    Xvt = torch.from_numpy(Xvs).to(dev)
    preds = []
    n = len(X)
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = _MLP(X.shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        lossf = nn.MSELoss()
        best_c, best_state = -1, None
        for ep in range(epochs):
            perm = torch.randperm(n)
            model.train()
            for i in range(0, n, batch):
                idxb = perm[i:i + batch]
                xb = torch.from_numpy(Xs[idxb.numpy()]).to(dev)
                yb = torch.from_numpy(y_n[idxb.numpy()]).to(dev)
                opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                pv = model(Xvt).cpu().numpy() * y_std + y_mean
            c = cos_u(pv, yv)
            if c > best_c:
                best_c = c; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        with torch.no_grad():
            pv = model(Xvt).cpu().numpy() * y_std + y_mean
        preds.append(pv)
        print(f"    mlp seed{seed} best {best_c:.6f}", flush=True)
    return np.mean(preds, axis=0)

preds["mlp"] = fit_mlp_best(Xt, yt, Xv, yv)
print(f"  mlp PSEUDO {cos_u(preds['mlp'], yv):.6f} ({time.time()-t0:.0f}s)", flush=True)

# 融合 (temporal 权重: lgb0.2/xgb0/cat0.5/mlp0.3 — v5 口径)
p_tab = 0.2 * preds["lgb"] + 0.5 * preds["cat"] + 0.3 * preds["mlp"]
print(f"  tabular blend PSEUDO {cos_u(p_tab, yv):.6f}", flush=True)

# RealMLP PSEUDO pred
d = np.load(r"D:\mscapital-kaggle\output\rlps_final\realmlp_pseudo_pred.npz")
p_rl, y_rl = d["pred"], d["y"]
print(f"  realmlp PSEUDO {cos_u(p_rl, y_rl):.6f}", flush=True)
print(f"  corr(tab, realmlp) on PSEUDO: {np.corrcoef(p_tab, p_rl)[0,1]:.4f}")

# v7 = 0.8 tab + 0.2 rl
p_v7 = 0.8 * p_tab + 0.2 * p_rl
print(f"\n=== v7 PSEUDO = {cos_u(p_v7, yv):.6f} (LB 0.135) ===")
print(f"=== gap = {cos_u(p_v7, yv) - 0.135:.4f} ===")

# 权重扫描 (PSEUDO 上找最优, 供参考)
print("\n权重扫描 (PSEUDO):")
for wr in [0.1, 0.2, 0.3, 0.4]:
    p = (1 - wr) * p_tab + wr * p_rl
    print(f"  w_rl={wr}: {cos_u(p, yv):.6f}")
