# -*- coding: utf-8 -*-
"""
Exp H1: MLP v2 增强 (512宽 × 50ep × 3seed) + CatBoost 加入 → 五模型融合评估
本地GPU: 512宽50ep约3-4min/seed
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
tr_df = tr.filter(pl.col("month") <= 50)
va_df = tr.filter((pl.col("month") > 50) & (pl.col("month") <= 70))
X_tr = tr_df.select(all_feats).to_numpy().astype(np.float32)
y_tr = tr_df["target"].to_numpy().astype(np.float32)
X_va = va_df.select(all_feats).to_numpy().astype(np.float32)
y_va = va_df["target"].to_numpy().astype(np.float32)

def cos_uncenter(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

preds = {}

# --- LGBM ---
params_l = dict(objective="regression", metric="rmse", learning_rate=0.02, num_leaves=64,
                min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                lambda_l2=5.0, max_bin=255, verbose=-1, num_threads=N_THREADS, seed=0)
m_l = lgb.train(params_l, lgb.Dataset(X_tr, y_tr), 10000,
                valid_sets=[lgb.Dataset(X_va, y_va, reference=lgb.Dataset(X_tr, y_tr))],
                callbacks=[lgb.early_stopping(200)])
preds["lgb"] = m_l.predict(X_va, num_iteration=m_l.best_iteration)
print(f"LGBM: {cos_uncenter(preds['lgb'], y_va):.6f}", flush=True)

# --- XGB ---
params_x = dict(objective="reg:squarederror", eval_metric="rmse", eta=0.02, max_depth=5,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, nthread=N_THREADS, seed=0)
m_x = xgb.train(params_x, xgb.DMatrix(X_tr, label=y_tr), 10000,
                evals=[(xgb.DMatrix(X_va, label=y_va), "va")], early_stopping_rounds=200, verbose_eval=False)
preds["xgb"] = m_x.predict(xgb.DMatrix(X_va), iteration_range=(0, m_x.best_iteration + 1))
print(f"XGB: {cos_uncenter(preds['xgb'], y_va):.6f}", flush=True)

# --- CatBoost ---
t0 = time.time()
cb = CatBoostRegressor(iterations=10000, learning_rate=0.02, depth=6, l2_leaf_reg=5.0,
                       subsample=0.8, colsample_bylevel=0.8, loss_function="RMSE",
                       early_stopping_rounds=200, verbose=0, thread_count=N_THREADS, random_seed=0)
cb.fit(X_tr, y_tr, eval_set=(X_va, y_va))
preds["cat"] = cb.predict(X_va)
print(f"CatBoost: {cos_uncenter(preds['cat'], y_va):.6f} iter={cb.get_best_iteration()} ({time.time()-t0:.0f}s)", flush=True)

# --- MLP v2 (512宽 50ep 3seed) ---
class MLP2(nn.Module):
    def __init__(self, n_in, h=512, p=0.25):
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
X_tr_s = ((X_tr_c - mu) / sd).clip(-10, 10)
X_va_s = ((X_va_c - mu) / sd).clip(-10, 10)
y_tr_mean = y_tr.mean(); y_tr_std = y_tr.std() + 1e-6
y_tr_n = (y_tr - y_tr_mean) / y_tr_std
Xv = torch.from_numpy(X_va_s).to(device)
BATCH, EPOCHS, n = 2048, 50, len(X_tr)

def train_mlp2(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = MLP2(X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.MSELoss()
    best_c, best_state = -1, None
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb = torch.from_numpy(X_tr_s[idx.numpy()]).to(device)
            yb = torch.from_numpy(y_tr_n[idx.numpy()]).to(device)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).cpu().numpy() * y_tr_std + y_tr_mean
        c = cos_uncenter(pv, y_va)
        if c > best_c:
            best_c = c; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_c, model

t0 = time.time()
p_mlp = []
for seed in [2026, 7, 123]:
    c, m = train_mlp2(seed)
    m.eval()
    with torch.no_grad():
        p_mlp.append(m(Xv).cpu().numpy() * y_tr_std + y_tr_mean)
    print(f"MLP2 seed{seed}: {c:.6f} ({time.time()-t0:.0f}s)", flush=True)
preds["mlp"] = np.mean(p_mlp, axis=0)
print(f"MLP2-ens: {cos_uncenter(preds['mlp'], y_va):.6f}", flush=True)

# --- 相关性 + 权重网格 (5模型) ---
names = list(preds)
print("\n相关性矩阵:")
for a in names:
    print(f"  {a}: " + "  ".join(f"{np.corrcoef(preds[a], preds[b])[0,1]:.3f}" for b in names))

best = (-1, None)
for wl in np.arange(0, 1.01, 0.1):
    for wx in np.arange(0, 1.01 - wl, 0.1):
        for wc in np.arange(0, 1.01 - wl - wx, 0.1):
            wm = 1 - wl - wx - wc
            p = wl * preds["lgb"] + wx * preds["xgb"] + wc * preds["cat"] + wm * preds["mlp"]
            c = cos_uncenter(p, y_va)
            if c > best[0]:
                best = (c, (round(wl, 2), round(wx, 2), round(wc, 2), round(wm, 2)))
print(f"\n=== Exp H1 汇总 ===")
for k in names:
    print(f"{k}: {cos_uncenter(preds[k], y_va):.6f}")
print(f"4-model blend w={best[1]}: {best[0]:.6f} ({(best[0]-max(cos_uncenter(preds[k], y_va) for k in names))*1000:+.3f}k vs best single)")
