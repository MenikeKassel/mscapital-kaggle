# -*- coding: utf-8 -*-
"""P5-B follow-up: prep Z-feature matrices for the learner-robustness spot-check.

Saves for blocks B51_60 and B61_70:
  X152 (train/hold/eval), Z (train/hold/eval), y, month, canonical pred
— the RealMLP/MLP spot-check then only loads + trains (GPU phase separate).
"""
import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:\mscapital-kaggle\scripts")
sys.path.insert(0, r"D:\mscapital-kaggle\src")

import p5b_scfi as P

OUT = Path(r"D:\mscapital-kaggle\output\p5b_scfi")
t0 = time.time()

data = P.load_data()
print(f"loaded ({time.time()-t0:.0f}s)", flush=True)

# canonical baseline (load_data 不含 canon — main() 里才有)
canon = np.load(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
cids = canon["sample_id"]
cpos = np.searchsorted(cids, data["sample_id"])
ok = np.searchsorted(cids, data["sample_id"], side="right") > cpos
canon_pred = np.full(len(data["sample_id"]), np.nan)
canon_pred[ok] = canon["baseline_oof"][cpos[ok]]
data["canon"] = canon_pred

m = data["month"]
blocks = [
    dict(name="B51_60", tr=np.arange(0, 49), ho=np.arange(49, 51), ev=np.arange(51, 61)),
    dict(name="B61_70", tr=np.arange(0, 59), ho=np.arange(59, 61), ev=np.arange(61, 71)),
]

for blk in blocks:
    tr_mask = np.isin(m, blk["tr"])
    ho_mask = np.isin(m, blk["ho"])
    ev_mask = np.isin(m, blk["ev"])
    Ztr, Ze, R2, (o_cols, t_cols), (madO, madT) = P.nuisance_crossfit(data, blk["tr"], blk["ev"])
    # holdout Z: fit on tr only, scale by train MADs
    Xf = data["Xm"][tr_mask]
    Xh = data["Xm"][ho_mask]
    Otr, Thr = data["Xraw"][tr_mask][:, o_cols], data["Xraw"][tr_mask][:, t_cols]
    Oho, Tho = data["Xraw"][ho_mask][:, o_cols], data["Xraw"][ho_mask][:, t_cols]
    pred_ho_O = P.ridge_fit_predict(Xf, Otr, Xh)
    Zh_O = (Oho - pred_ho_O) / madO
    Xfo = np.hstack([Xf, Otr])
    Xho2 = np.hstack([Xh, Oho])
    pred_ho_T = P.ridge_fit_predict(Xfo, Thr, Xho2)
    Zh_T = (Tho - pred_ho_T) / madT
    Zh = np.hstack([Zh_O, Zh_T])
    nO = len(o_cols)
    Z = np.full((len(m), nO + len(t_cols)), np.nan)
    Z[tr_mask] = Ztr
    Z[ho_mask] = Zh
    Z[ev_mask] = Ze
    assert np.isfinite(Z[tr_mask | ho_mask | ev_mask]).all()
    np.savez(OUT / f"spotcheck_{blk['name']}.npz",
             X152_tr=data["X152"][tr_mask].astype(np.float32),
             X152_ho=data["X152"][ho_mask].astype(np.float32),
             X152_ev=data["X152"][ev_mask].astype(np.float32),
             Z_tr=Z[tr_mask].astype(np.float32),
             Z_ho=Z[ho_mask].astype(np.float32),
             Z_ev=Z[ev_mask].astype(np.float32),
             y_tr=data["target"][tr_mask], y_ho=data["target"][ho_mask],
             y_ev=data["target"][ev_mask],
             canon_ev=data["canon"][ev_mask], canon_ho=data["canon"][ho_mask],
             month_ev=m[ev_mask])
    print(f"{blk['name']} saved ({time.time()-t0:.0f}s)", flush=True)
    del Ztr, Ze, Zh, Z
    gc.collect()

print("done", flush=True)
