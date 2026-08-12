# -*- coding: utf-8 -*-
"""
P1-2c: 全融合 blend × TCN 融合验证 (PSEUDO fold)
blend (LGB+XGB+Cat+MLP, R2+micro) pred vs TCN pred → 融合网格
"""
import time
import random
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

FEAT = r"D:\mscapital-forecasting\data\processed\train_features.parquet"
MICRO = r"D:\mscapital-forecasting\data\processed\micro_features_train.parquet"
P12 = r"D:\mscapital-forecasting\data\processed\p12_out\p12_valid_pred.npz"
N_THREADS = 12
torch.set_num_threads(N_THREADS)

tr = pl.read_parquet(FEAT)
mi = pl.read_parquet(MICRO)
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

tr_all = make_R2(tr.select(["sample_id"] + all_feats)).join(mi, on="sample_id", how="left")
tr_all = tr_all.with_columns([pl.col(c).fill_null(0.0) for c in micro_cols])
cols = all_feats + micro_cols
X_all = tr_all.select(cols).to_numpy().astype(np.float32)
y_all = tr["target"].to_numpy().astype(np.float32)
m_all = tr["month"].to_numpy().astype(np.int32)

def cos_uncenter(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

sel_t = (m_all <= 32); sel_v = (m_all > 32) & (m_all <= 70)
X, y, Xv, yv = X_all[sel_t], y_all[sel_t], X_all[sel_v], y_all[sel_v]

# 树模型
out = {}
p = dict(objective="regression", metric="rmse", learning_rate=0.02, num_leaves=64,
         min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
         lambda_l2=5.0, max_bin=255, verbose=-1, num_threads=N_THREADS, seed=0)
m = lgb.train(p, lgb.Dataset(X, y), 10000,
              valid_sets=[lgb.Dataset(Xv, yv, reference=lgb.Dataset(X, y))],
              callbacks=[lgb.early_stopping(200)])
out["lgb"] = m.predict(Xv, num_iteration=m.best_iteration)
px = dict(objective="reg:squarederror", eval_metric="rmse", eta=0.02, max_depth=5,
          subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, nthread=N_THREADS, seed=0)
m = xgb.train(px, xgb.DMatrix(X, label=y), 10000, evals=[(xgb.DMatrix(Xv, label=yv), "va")],
              early_stopping_rounds=200, verbose_eval=False)
out["xgb"] = m.predict(xgb.DMatrix(Xv), iteration_range=(0, m.best_iteration + 1))
cb = CatBoostRegressor(iterations=10000, learning_rate=0.02, depth=6, l2_leaf_reg=5.0,
                       subsample=0.8, colsample_bylevel=0.8, loss_function="RMSE",
                       early_stopping_rounds=200, verbose=0, thread_count=N_THREADS, random_seed=0)
cb.fit(X, y, eval_set=(Xv, yv))
out["cat"] = cb.predict(Xv)
print(f"trees done: lgb={cos_uncenter(out['lgb'],yv):.5f} xgb={cos_uncenter(out['xgb'],yv):.5f} cat={cos_uncenter(out['cat'],yv):.5f}", flush=True)

# MLP (best-state, 简化: 1 seed)
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
Xc = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
Xvc = np.nan_to_num(Xv, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
mu = Xc.mean(axis=0, keepdims=True); sd = Xc.std(axis=0, keepdims=True) + 1e-6
Xs = ((Xc - mu) / sd).clip(-10, 10); Xvs = ((Xvc - mu) / sd).clip(-10, 10)
y_mean, y_std = y.mean(), y.std() + 1e-6
y_n = (y - y_mean) / y_std
Xvt = torch.from_numpy(Xvs).to(device)
random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
model = MLP(X.shape[1]).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
lossf = nn.MSELoss()
n = len(y)
best_c, best_state = -1, None
for ep in range(30):
    perm = torch.randperm(n)
    model.train()
    for i in range(0, n, 2048):
        idx = perm[i:i+2048]
        xb = torch.from_numpy(Xs[idx.numpy()]).to(device)
        yb = torch.from_numpy(y_n[idx.numpy()]).to(device)
        opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
    sched.step()
    model.eval()
    with torch.no_grad():
        pv = model(Xvt).cpu().numpy() * y_std + y_mean
    c = cos_uncenter(pv, yv)
    if c > best_c:
        best_c = c; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    out["mlp"] = model(Xvt).cpu().numpy() * y_std + y_mean
print(f"mlp: {cos_uncenter(out['mlp'],yv):.5f}", flush=True)

# blend (temporal 权重 + xgb 微调)
W = {"lgb": 0.2, "xgb": 0.0, "cat": 0.5, "mlp": 0.3}
p_blend = W["lgb"]*out["lgb"] + W["xgb"]*out["xgb"] + W["cat"]*out["cat"] + W["mlp"]*out["mlp"]
c_blend = cos_uncenter(p_blend, yv)
print(f"\nblend (R2+micro): {c_blend:.6f}", flush=True)

# TCN 融合
d = np.load(P12)
p_tcn = d["pred"]
print(f"corr(blend, tcn) = {np.corrcoef(p_blend, p_tcn)[0,1]:.4f}")
best = (-1, None)
for wt in np.arange(0, 0.31, 0.05):
    wb = 1 - wt
    cc = cos_uncenter(wb * p_blend + wt * p_tcn, yv)
    print(f"  w_blend={wb:.2f} w_tcn={wt:.2f}: {cc:.6f} ({cc-c_blend:+.5f})")
    if cc > best[0]:
        best = (cc, (round(wb, 2), round(wt, 2)))
print(f"\nbest: w={best[1]} cos={best[0]:.6f} (vs blend {c_blend:.6f}, Δ={best[0]-c_blend:+.5f})")
