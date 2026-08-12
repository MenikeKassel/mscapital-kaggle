# -*- coding: utf-8 -*-
"""
Final v5: 表格全量提交 (R2 归一化 + 22 微观特征)
LGBM + CatBoost + MLP-ens 全量训练, temporal 权重 (0.2/0.5/0.3)
"""
import time
import random
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import lightgbm as lgb
from catboost import CatBoostRegressor

FEAT = r"D:\mscapital-forecasting\data\processed\train_features.parquet"
MICRO = r"D:\mscapital-forecasting\data\processed\micro_features_train.parquet"
MICRO_TE = r"D:\mscapital-forecasting\data\processed\micro_features_test.parquet"
N_THREADS = 12
OUT = r"D:\mscapital-kaggle\output\submissions\submission_blend_v5.csv"
W = {"lgb": 0.2, "cat": 0.5, "mlp": 0.3}
LGB_ITER = 443
CAT_ITER = 1823

tr = pl.read_parquet(FEAT)
te = pl.read_parquet(FEAT.replace("train_features", "test_features"))
mi = pl.read_parquet(MICRO)
mi_te = pl.read_parquet(MICRO_TE)
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
te_all = make_R2(te.select(["sample_id"] + all_feats)).join(mi_te, on="sample_id", how="left")
tr_all = tr_all.with_columns([pl.col(c).fill_null(0.0) for c in micro_cols])
te_all = te_all.with_columns([pl.col(c).fill_null(0.0) for c in micro_cols])
cols = all_feats + micro_cols
X_tr = tr_all.select(cols).to_numpy().astype(np.float32)
y_tr = tr["target"].to_numpy().astype(np.float32)
X_te = te_all.select(cols).to_numpy().astype(np.float32)
ids_te = te["sample_id"].to_numpy()
print(f"v5 train {X_tr.shape} test {X_te.shape}", flush=True)

t0 = time.time()
params_l = dict(objective="regression", metric="rmse", learning_rate=0.02, num_leaves=64,
                min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                lambda_l2=5.0, max_bin=255, verbose=-1, num_threads=N_THREADS, seed=0)
m_l = lgb.train(params_l, lgb.Dataset(X_tr, y_tr), LGB_ITER)
p_l = m_l.predict(X_te)
print(f"LGBM full: done ({time.time()-t0:.0f}s)", flush=True)

t0 = time.time()
cb = CatBoostRegressor(iterations=CAT_ITER, learning_rate=0.02, depth=6, l2_leaf_reg=5.0,
                       subsample=0.8, colsample_bylevel=0.8, loss_function="RMSE",
                       verbose=0, thread_count=N_THREADS, random_seed=0)
cb.fit(X_tr, y_tr)
p_c = cb.predict(X_te)
print(f"CatBoost full: done ({time.time()-t0:.0f}s)", flush=True)

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
BATCH, EPOCHS, n = 2048, 30, len(X_tr)
Xte = torch.from_numpy(X_te_s).to(device)

def train_mlp(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = MLP(X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    lossf = nn.MSELoss()
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, BATCH):
            idxb = perm[i:i+BATCH]
            xb = torch.from_numpy(X_tr_s[idxb.numpy()]).to(device)
            yb = torch.from_numpy(y_n[idxb.numpy()]).to(device)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        return model(Xte).cpu().numpy() * y_std + y_mean

t0 = time.time()
p_m = np.mean([train_mlp(s) for s in [2026, 7, 123]], axis=0)
print(f"MLP full 3seeds: done ({time.time()-t0:.0f}s)", flush=True)

p_final = W["lgb"] * p_l + W["cat"] * p_c + W["mlp"] * p_m
sub = pl.DataFrame({"sample_id": pl.Series(ids_te, dtype=pl.Int32),
                    "prediction": pl.Series(p_final, dtype=pl.Float64)}).sort("sample_id")
sub.write_csv(OUT)
print(f"\nsaved {OUT} ({sub.height:,} rows) NaN={sub['prediction'].is_null().sum()}")
np.savez(r"D:\mscapital-forecasting\data\processed\p12_out\v5_table_test_pred.npz", pred=p_final)
print("saved v5_table_test_pred.npz for later TCN fusion")
