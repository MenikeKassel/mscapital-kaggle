# -*- coding: utf-8 -*-
"""
P0-3: 融合权重重估 — 用 Pseudo-LB38 (模拟test) + temporal folds 重估权重
对照: 原权重 (xgb 0.1, cat 0.4, mlp 0.5) 基于 CV1 优化 → 可能高估 MLP
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

def cos_uncenter(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def fit_trees(X, y, Xv, yv):
    p_l = dict(objective="regression", metric="rmse", learning_rate=0.02, num_leaves=64,
               min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
               lambda_l2=5.0, max_bin=255, verbose=-1, num_threads=N_THREADS, seed=0)
    m = lgb.train(p_l, lgb.Dataset(X, y), 10000,
                  valid_sets=[lgb.Dataset(Xv, yv, reference=lgb.Dataset(X, y))],
                  callbacks=[lgb.early_stopping(200)])
    pl_lgb = m.predict(Xv, num_iteration=m.best_iteration)
    p_x = dict(objective="reg:squarederror", eval_metric="rmse", eta=0.02, max_depth=5,
               subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, nthread=N_THREADS, seed=0)
    m = xgb.train(p_x, xgb.DMatrix(X, label=y), 10000, evals=[(xgb.DMatrix(Xv, label=yv), "va")],
                  early_stopping_rounds=200, verbose_eval=False)
    pl_xgb = m.predict(xgb.DMatrix(Xv), iteration_range=(0, m.best_iteration + 1))
    cb = CatBoostRegressor(iterations=10000, learning_rate=0.02, depth=6, l2_leaf_reg=5.0,
                           subsample=0.8, colsample_bylevel=0.8, loss_function="RMSE",
                           early_stopping_rounds=200, verbose=0, thread_count=N_THREADS, random_seed=0)
    cb.fit(X, y, eval_set=(Xv, yv))
    return {"lgb": pl_lgb, "xgb": pl_xgb, "cat": cb.predict(Xv)}

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

def fit_mlp(X, y, Xv, seeds=(2026, 7, 123), epochs=30, batch=2048):
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
        for ep in range(epochs):
            perm = torch.randperm(n)
            model.train()
            for i in range(0, n, batch):
                idx = perm[i:i+batch]
                xb = torch.from_numpy(Xs[idx.numpy()]).to(device)
                yb = torch.from_numpy(y_n[idx.numpy()]).to(device)
                opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            preds.append(model(Xvt).cpu().numpy() * y_std + y_mean)
    return np.mean(preds, axis=0)

# 收集每个 fold 的预测 (用于权重搜索 + Pseudo验证)
fold_preds = {}  # fold -> {model: pred}
FOLDS = {"T1": (0,30,31,40), "T2": (0,40,41,50), "T3": (0,50,51,60),
         "T4": (0,50,61,70), "H1": (0,30,41,50), "H2": (0,40,51,60),
         "PSEUDO": (0,32,33,70)}
T0 = time.time()
for fname, (a,b,c0,d0) in FOLDS.items():
    sel_t = (m_all >= a) & (m_all <= b); sel_v = (m_all >= c0) & (m_all <= d0)
    X, y, Xv, yv = X_all[sel_t], y_all[sel_t], X_all[sel_v], y_all[sel_v]
    preds = fit_trees(X, y, Xv, yv)
    preds["mlp"] = fit_mlp(X, y, Xv)
    fold_preds[fname] = (preds, yv)
    print(f"{fname}: lgb={cos_uncenter(preds['lgb'],yv):.5f} xgb={cos_uncenter(preds['xgb'],yv):.5f} "
          f"cat={cos_uncenter(preds['cat'],yv):.5f} mlp={cos_uncenter(preds['mlp'],yv):.5f} ({time.time()-T0:.0f}s)", flush=True)

# 在非Pseudo folds 上网格搜索权重, 在 Pseudo 上验证
print("\n=== 权重搜索 (temporal folds 训练, Pseudo-LB38 验证) ===")
folds_tr = [f for f in FOLDS if f != "PSEUDO"]
best = (-1, None)
for wl in np.arange(0, 0.51, 0.1):
    for wx in np.arange(0, 1.01 - wl, 0.1):
        for wc in np.arange(0, 1.01 - wl - wx, 0.1):
            wm = 1 - wl - wx - wc
            # temporal folds 平均 cos
            cs = []
            for f in folds_tr:
                preds, yv = fold_preds[f]
                p = wl * preds["lgb"] + wx * preds["xgb"] + wc * preds["cat"] + wm * preds["mlp"]
                cs.append(cos_uncenter(p, yv))
            score = np.mean(cs)
            if score > best[0]:
                best = (score, (round(wl,1), round(wx,1), round(wc,1), round(wm,1)))
print(f"temporal-mean 最优权重: {best[1]} (temporal mean={best[0]:.6f})")

# Pseudo 验证
preds_p, yv_p = fold_preds["PSEUDO"]
w_new = best[1]
c_new = cos_uncenter(w_new[0]*preds_p["lgb"] + w_new[1]*preds_p["xgb"] + w_new[2]*preds_p["cat"] + w_new[3]*preds_p["mlp"], yv_p)
w_old = (0.0, 0.1, 0.4, 0.5)
c_old = cos_uncenter(w_old[0]*preds_p["lgb"] + w_old[1]*preds_p["xgb"] + w_old[2]*preds_p["cat"] + w_old[3]*preds_p["mlp"], yv_p)
print(f"\nPseudo-LB38 验证:")
print(f"  旧权重 {w_old}: {c_old:.6f}")
print(f"  新权重 {w_new}: {c_new:.6f} ({c_new-c_old:+.6f})")

# 单模型 Pseudo 参考
for k in ["lgb", "xgb", "cat", "mlp"]:
    print(f"  {k}: {cos_uncenter(preds_p[k], yv_p):.6f}")
