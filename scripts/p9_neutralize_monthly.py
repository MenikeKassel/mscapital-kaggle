# -*- coding: utf-8 -*-
"""P9-neutralize: monthly stability check — per-month delta cosine (gamma=1.0 vs 0)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from c05_recipe_e0 import RobustScaleSmoothClip, MLP, cosine_uncentered

RAW = Path(r"D:\mscapital-forecasting\data\raw\train")
FEAT = Path(r"D:\mscapital-forecasting\data\processed\f0726_train.parquet")
CKPT = Path(r"D:\mscapital-kaggle\output\c05_recipe_e0\best_cos.pt")
OUT = Path(r"D:\mscapital-kaggle\output\p9_neutralize")

NUISANCE = ["m_mid_std", "m_mid_std_180", "m_rv", "m_rv_60", "m_rv_180",
            "m_sp_mean_60", "o_vol_sum", "t_vol_sum", "t_transaction_count", "o_order_count"]

df = pl.read_parquet(FEAT)
lab = pl.read_ipc(RAW / "label.feather")
df = df.join(lab.select(["sample_id", "month"]), on="sample_id", how="left")
feat_cols = [c for c in df.columns if c not in ("sample_id", "target", "month")]
X = df.select(feat_cols).to_numpy().astype(np.float32)
y = df["target"].to_numpy().astype(np.float64)
m = df["month"].to_numpy()

tr = m <= 32
ev = ~tr
pp = RobustScaleSmoothClip().fit(X[tr])
Xe = np.nan_to_num(pp.transform(X[ev]).astype(np.float32), nan=0.0)
y_ev, m_ev = y[ev], m[ev]

model = MLP(X.shape[1])
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()
with torch.no_grad():
    p = np.concatenate([model(torch.from_numpy(Xe[i:i + 8192])).numpy()
                        for i in range(0, len(Xe), 8192)]).ravel().astype(np.float64)

z_cols = [c for c in NUISANCE if c in feat_cols]
Z = np.nan_to_num(np.column_stack([X[ev][:, feat_cols.index(c)] for c in z_cols] + [np.abs(p)]),
                  nan=0.0, posinf=0.0, neginf=0.0)
cal = m_ev <= 50
beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(cal.sum()), Z[cal]]), p[cal], rcond=None)
yhat_Z = Z @ beta[1:] + beta[0]
pn = p - 1.0 * yhat_Z  # gamma=1.0

rows = []
npos = 0
for mm in np.unique(m_ev):
    mask = m_ev == mm
    c0 = cosine_uncentered(p[mask], y_ev[mask])
    c1 = cosine_uncentered(pn[mask], y_ev[mask])
    rows.append({"month": int(mm), "n": int(mask.sum()), "cos_base": c0, "cos_neut": c1,
                 "delta": c1 - c0})
    if c1 > c0:
        npos += 1
res = {"months": rows, "n_positive": npos, "n_months": len(rows),
       "frozen_months_positive": sum(1 for r in rows if r["month"] > 50 and r["delta"] > 0),
       "frozen_months": sum(1 for r in rows if r["month"] > 50)}
(OUT / "monthly.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print(f"months: {len(rows)}, delta>0: {npos}/{len(rows)}")
print(f"frozen 51-70: {res['frozen_months_positive']}/{res['frozen_months']} months positive")
deltas = np.array([r["delta"] for r in rows])
print(f"delta mean: {deltas.mean():+.6f}  positive months: {[r['month'] for r in rows if r['delta']>0][:20]}")
