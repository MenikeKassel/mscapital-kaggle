# -*- coding: utf-8 -*-
"""
P0-1: Temporal Generalization Matrix (GPT 拍板执行)
模型: LGBM/XGB/CatBoost/MLP-ens/Blend (固定参数不调参)
折叠: T1(m0-30/31-40) T2(m0-40/41-50) T3(m0-50/51-60) T4(m0-50/61-70) H1(m0-30/41-50) H2(m0-40/51-60)
Pseudo-LB38: train m0-32 / valid m33-70 (独立, 模拟test)
输出: fold×model矩阵, mean/std/worst, 月度cos曲线+half-life, 排序Spearman
"""
import time
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy import stats

FEAT = r"D:\mscapital-forecasting\data\processed\train_features.parquet"
N_THREADS = 12
torch.set_num_threads(N_THREADS)

tr = pl.read_parquet(FEAT)
all_feats = [c for c in tr.columns if c not in ("sample_id", "month", "target")]
X_all = tr.select(all_feats).to_numpy().astype(np.float32)
y_all = tr["target"].to_numpy().astype(np.float32)
m_all = tr["month"].to_numpy().astype(np.int32)

FOLDS = {
    "T1": (0, 30, 31, 40), "T2": (0, 40, 41, 50), "T3": (0, 50, 51, 60),
    "T4": (0, 50, 61, 70), "H1": (0, 30, 41, 50), "H2": (0, 40, 51, 60),
    "PSEUDO_LB38": (0, 32, 33, 70),
}
MODELS = ["lgb", "xgb", "cat", "mlp", "blend"]
BLEND_W = {"xgb": 0.1, "cat": 0.4, "mlp": 0.5}

def cos_uncenter(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def fit_model(name, X, y, Xv, yv):
    if name == "lgb":
        p = dict(objective="regression", metric="rmse", learning_rate=0.02, num_leaves=64,
                 min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                 lambda_l2=5.0, max_bin=255, verbose=-1, num_threads=N_THREADS, seed=0)
        m = lgb.train(p, lgb.Dataset(X, y), 10000, valid_sets=[lgb.Dataset(Xv, yv, reference=lgb.Dataset(X, y))],
                      callbacks=[lgb.early_stopping(200)])
        return m.predict(Xv, num_iteration=m.best_iteration)
    if name == "xgb":
        p = dict(objective="reg:squarederror", eval_metric="rmse", eta=0.02, max_depth=5,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, nthread=N_THREADS, seed=0)
        m = xgb.train(p, xgb.DMatrix(X, label=y), 10000, evals=[(xgb.DMatrix(Xv, label=yv), "va")],
                      early_stopping_rounds=200, verbose_eval=False)
        return m.predict(xgb.DMatrix(Xv), iteration_range=(0, m.best_iteration + 1))
    if name == "cat":
        cb = CatBoostRegressor(iterations=10000, learning_rate=0.02, depth=6, l2_leaf_reg=5.0,
                               subsample=0.8, colsample_bylevel=0.8, loss_function="RMSE",
                               early_stopping_rounds=200, verbose=0, thread_count=N_THREADS, random_seed=0)
        cb.fit(X, y, eval_set=(Xv, yv))
        return cb.predict(Xv)
    return None

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

def mlp_prep(X, Xv):
    Xc = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
    Xvc = np.nan_to_num(Xv, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
    mu = Xc.mean(axis=0, keepdims=True); sd = Xc.std(axis=0, keepdims=True) + 1e-6
    return ((Xc - mu) / sd).clip(-10, 10), ((Xvc - mu) / sd).clip(-10, 10)

def fit_mlp_ens(X, y, Xv, seeds=(2026, 7, 123), epochs=30, batch=2048):
    Xs, Xvs = mlp_prep(X, Xv)
    y_mean, y_std = y.mean(), y.std() + 1e-6
    y_n = (y - y_mean) / y_std
    Xvt = torch.from_numpy(Xvs).to(device)
    preds = []
    import random
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = MLP(X.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        lossf = nn.MSELoss()
        n = len(X)
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

# 运行
results = {}      # fold -> model -> (cos, pred)
month_curves = [] # (fold, month, model, cos)
T0 = time.time()
for fname, (t0m, t1m, v0m, v1m) in FOLDS.items():
    m_tr = (m_all >= t0m) & (m_all <= t1m)
    m_va = (m_all >= v0m) & (m_all <= v1m)
    X, y = X_all[m_tr], y_all[m_tr]
    Xv, yv = X_all[m_va], y_all[m_va]
    vm = m_all[m_va]
    print(f"\n=== {fname}: train m{t0m}-{t1m} ({X.shape[0]:,}) valid m{v0m}-{v1m} ({Xv.shape[0]:,}) ===", flush=True)
    fold_res = {}
    for name in ["lgb", "xgb", "cat"]:
        t0 = time.time()
        pv = fit_model(name, X, y, Xv, yv)
        c = cos_uncenter(pv, yv)
        fold_res[name] = (c, pv)
        print(f"  {name}: {c:.6f} ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    pv = np.asarray(fit_mlp_ens(X, y, Xv), dtype=np.float32).ravel()
    c = cos_uncenter(pv, yv)
    fold_res["mlp"] = (c, pv)
    print(f"  mlp: {c:.6f} ({time.time()-t0:.0f}s)", flush=True)
    assert isinstance(fold_res["mlp"], tuple), f"mlp 结果类型异常: {type(fold_res['mlp'])}"
    pv_xgb = np.asarray(fold_res["xgb"][1], dtype=np.float32)
    pv_cat = np.asarray(fold_res["cat"][1], dtype=np.float32)
    pv_mlp = np.asarray(fold_res["mlp"][1], dtype=np.float32)
    pv_b = BLEND_W["xgb"] * pv_xgb + BLEND_W["cat"] * pv_cat + BLEND_W["mlp"] * pv_mlp
    c = cos_uncenter(pv_b, yv)
    fold_res["blend"] = (c, pv_b)
    print(f"  blend: {c:.6f}", flush=True)
    results[fname] = fold_res
    # 月度曲线
    for name, (c, pv) in fold_res.items():
        for mo in sorted(set(vm.tolist())):
            sel = vm == mo
            if sel.sum() < 100:
                continue
            month_curves.append((fname, mo, name, cos_uncenter(pv[sel], yv[sel]), t1m))

# ==== 输出矩阵 ====
print("\n\n========== P0-1 结果 ==========")
print(f"\nfold × model cos 矩阵 (总耗时 {time.time()-T0:.0f}s):")
header = "fold     " + "  ".join(f"{m:>7}" for m in MODELS)
print(header)
mat = {}
for fname in FOLDS:
    row = [results[fname][m][0] for m in MODELS]
    mat[fname] = {m: results[fname][m][0] for m in MODELS}
    print(f"{fname:<8} " + "  ".join(f"{v:.5f}" for v in row))

# mean/std/worst (排除 PSEUDO 的统计, PSEUDO 独立)
folds_main = [f for f in FOLDS if f != "PSEUDO_LB38"]
print("\n统计 (T1-T4+H1-H2):")
print(f"{'model':<6} {'mean':>8} {'std':>8} {'worst':>8} {'best':>8}")
for m in MODELS:
    vals = [mat[f][m] for f in folds_main]
    print(f"{m:<6} {np.mean(vals):.5f} {np.std(vals):.5f} {min(vals):.5f} {max(vals):.5f}")

# 排序稳定性 (Spearman 两两)
print("\n模型排序 Spearman 相关 (跨 fold):")
ranks = {f: {m: i for i, m in enumerate(sorted(mat[f], key=lambda x: -mat[f][x]))} for f in folds_main}
for f1 in folds_main:
    row = "  ".join(f"{stats.spearmanr([ranks[f1][m] for m in MODELS], [ranks[f2][m] for m in MODELS]).statistic:.2f}" for f2 in folds_main)
    print(f"  {f1}: {row}")

# half-life: 用月度曲线拟合 C(h)=C0*exp(-λh)
print("\nAlpha half-life (月度cos vs 距train末端月距):")
import math
for m in MODELS:
    pts = [(mo - t1m, c) for (fname, mo, name, c, t1m) in month_curves if name == m]
    pts = [p for p in pts if p[1] > 1e-6]
    if len(pts) < 5:
        print(f"  {m}: 数据点不足"); continue
    hs = np.array([p[0] for p in pts], dtype=float)
    cs = np.array([p[1] for p in pts], dtype=float)
    logc = np.log(cs)
    slope, intercept = np.polyfit(hs, logc, 1)
    lam = -slope
    half = math.log(2) / lam if lam > 0 else float("inf")
    print(f"  {m}: λ={lam:.4f} T1/2={half:.1f}月 (n={len(pts)})")

# PSEUDO-LB38 独立报告
print("\nPseudo-LB38 (模拟 hidden test):")
for m in MODELS:
    print(f"  {m}: {mat['PSEUDO_LB38'][m]:.6f}")

# 保存月度曲线
import csv
with open(r"D:\mscapital-kaggle\output\temporal_monthly_curves.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["fold", "month", "model", "cos", "train_end"])

    for row in month_curves:
        w.writerow(row)
print("\nsaved output/temporal_monthly_curves.csv")
