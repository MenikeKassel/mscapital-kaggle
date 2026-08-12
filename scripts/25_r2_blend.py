# -*- coding: utf-8 -*-
"""
P0.5-D: R2 归一化特征 × 完整融合管线验证
R0 vs R2 特征 × (LGBM+XGB+CatBoost+MLP-best-state) 在 PSEUDO/T3/T4 folds
若 PSEUDO R2 融合 > R0 融合 +0.001 → 生成 v4 提交 (全量训练 + R2 test 特征)
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
N_THREADS = 12
torch.set_num_threads(N_THREADS)

tr = pl.read_parquet(FEAT)
all_feats = [c for c in tr.columns if c not in ("sample_id", "month", "target")]
X_all = tr.select(all_feats).to_numpy().astype(np.float32)
y_all = tr["target"].to_numpy().astype(np.float32)
m_all = tr["month"].to_numpy().astype(np.int32)
idx = {c: i for i, c in enumerate(all_feats)}

def make_R2(X):
    """归一化替换 (与 23 脚本一致)"""
    Xr = X.copy()
    def col(n):
        return X[:, idx[n]]
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

def fit_all(X, y, Xv, yv):
    """返回 4 模型预测 dict"""
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
    return out

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

def fit_mlp_best(X, y, Xv, yv, epochs=30, seeds=(2026, 7, 123), batch=2048):
    Xc = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
    Xvc = np.nan_to_num(Xv, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
    mu = Xc.mean(axis=0, keepdims=True); sd = Xc.std(axis=0, keepdims=True) + 1e-6
    Xs = ((Xc - mu) / sd).clip(-10, 10); Xvs = ((Xvc - mu) / sd).clip(-10, 10)
    y_mean, y_std = y.mean(), y.std() + 1e-6
    y_n = (y - y_mean) / y_std
    Xvt = torch.from_numpy(Xvs).to(device)
    preds = []
    n = len(X)
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = MLP(X.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        lossf = nn.MSELoss()
        best_c, best_state = -1, None
        for ep in range(epochs):
            perm = torch.randperm(n)
            model.train()
            for i in range(0, n, batch):
                idxb = perm[i:i+batch]
                xb = torch.from_numpy(Xs[idxb.numpy()]).to(device)
                yb = torch.from_numpy(y_n[idxb.numpy()]).to(device)
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
            preds.append(model(Xvt).cpu().numpy() * y_std + y_mean)
    return np.mean(preds, axis=0)

FOLDS = {"T3": (0, 50, 51, 60), "T4": (0, 50, 61, 70), "PSEUDO": (0, 32, 33, 70)}
W = {"lgb": 0.2, "cat": 0.5, "mlp": 0.3}  # temporal 权重

T0 = time.time()
for fname, (a, b, c0, d0) in FOLDS.items():
    sel_t = (m_all >= a) & (m_all <= b); sel_v = (m_all >= c0) & (m_all <= d0)
    X, y, Xv, yv = X_all[sel_t], y_all[sel_t], X_all[sel_v], y_all[sel_v]
    print(f"\n=== {fname} ===", flush=True)
    for tag, Xtr, Xva in [("R0", X, Xv), ("R2", make_R2(X), make_R2(Xv))]:
        t0 = time.time()
        preds = fit_all(Xtr, y, Xva, yv)
        preds["mlp"] = fit_mlp_best(Xtr, y, Xva, yv)
        p_b = W["lgb"]*preds["lgb"] + W["cat"]*preds["cat"] + W["mlp"]*preds["mlp"]
        c_b = cos_uncenter(p_b, yv)
        singles = {k: cos_uncenter(v, yv) for k, v in preds.items()}
        print(f"  {tag}: blend={c_b:.6f} | lgb={singles['lgb']:.5f} xgb={singles['xgb']:.5f} "
              f"cat={singles['cat']:.5f} mlp={singles['mlp']:.5f} ({time.time()-t0:.0f}s)", flush=True)

print(f"\n=== P0.5-D 汇总 (总 {time.time()-T0:.0f}s) ===")
print("PSEUDO R2 blend 若 > R0 blend +0.001 → 生成 v4 全量提交")
