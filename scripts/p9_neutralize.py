# -*- coding: utf-8 -*-
"""P9-Neutralize (⑤): Prediction neutralization — strip nuisance exposure from predictions.

p' = p - gamma * yhat_Z,  where yhat_Z = OLS fit of Z -> p on calibration months.
Z = volatility / spread / activity proxies + |p| (P7-01 mechanism: high-vol samples predict worse).

Protocol: p from C-05 best_cos.pt (same PSEUDO fold m0-32/m33-70).
  calibration 33-50 (fit beta) -> frozen 51-70 (judge), also report 33-70.
gamma in {0, .25, .5, .75, 1} (partial neutralization).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

RAW = Path(r"D:\mscapital-forecasting\data\raw\train")
FEAT = Path(r"D:\mscapital-forecasting\data\processed\f0726_train.parquet")
CKPT = Path(r"D:\mscapital-kaggle\output\c05_recipe_e0\best_cos.pt")
OUT = Path(r"D:\mscapital-kaggle\output\p9_neutralize")
OUT.mkdir(parents=True, exist_ok=True)

NUISANCE_PROXIES = [
    "m_mid_std", "m_mid_std_180", "m_rv", "m_rv_60", "m_rv_180",
    "m_sp_mean_60", "o_vol_sum", "t_vol_sum", "t_transaction_count", "o_order_count",
]
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def cosine_uncentered(p, y):
    p = p.reshape(-1).astype(np.float64)
    y = y.reshape(-1).astype(np.float64)
    return float(p @ y / (np.sqrt(p @ p) * np.sqrt(y @ y) + 1e-30))


def main():
    t0 = time.time()
    df = pl.read_parquet(FEAT)
    lab = pl.read_ipc(RAW / "label.feather")
    df = df.join(lab.select(["sample_id", "month"]), on="sample_id", how="left")
    feat_cols = [c for c in df.columns if c not in ("sample_id", "target", "month")]
    X = df.select(feat_cols).to_numpy().astype(np.float32)
    y = df["target"].to_numpy().astype(np.float64)
    m = df["month"].to_numpy()

    ev = m > 32
    X_ev, y_ev, m_ev = X[ev], y[ev], m[ev]

    # C-05 preprocessing (fold-local fit on train) then inference
    from c05_recipe_e0 import RobustScaleSmoothClip, MLP
    tr = m <= 32
    pp = RobustScaleSmoothClip().fit(X[tr])
    Xe = np.nan_to_num(pp.transform(X_ev).astype(np.float32), nan=0.0)
    model = MLP(X.shape[1])
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        p = np.concatenate([model(torch.from_numpy(Xe[i:i + 4096])).numpy()
                            for i in range(0, len(Xe), 4096)])
    p = p.astype(np.float64).ravel()

    # Z matrix: selected proxies + |p|  (nan-safe: f0726 may contain NaN/Inf)
    z_cols = [c for c in NUISANCE_PROXIES if c in feat_cols]
    Z_raw = np.column_stack([X_ev[:, feat_cols.index(c)] for c in z_cols] + [np.abs(p)])
    Z = np.nan_to_num(Z_raw, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"Z columns ({Z.shape[1]}): {z_cols + ['|p|']}")

    cal = m_ev <= 50
    fro = m_ev > 50
    res = {"z_cols": z_cols + ["|p|"], "n_cal": int(cal.sum()), "n_frozen": int(fro.sum())}

    base_frozen = cosine_uncentered(p[fro], y_ev[fro])
    base_all = cosine_uncentered(p, y_ev)
    res["baseline"] = {"cos_frozen_51_70": base_frozen, "cos_eval_33_70": base_all}

    # OLS fit Z -> p on calibration
    Zc = np.column_stack([np.ones(cal.sum()), Z[cal]])
    beta, *_ = np.linalg.lstsq(Zc, p[cal], rcond=None)
    yhat_Z = Z @ beta[1:] + beta[0]

    per_gamma = {}
    for g in GAMMAS:
        pn = p - g * yhat_Z
        per_gamma[g] = {
            "cos_frozen_51_70": cosine_uncentered(pn[fro], y_ev[fro]),
            "cos_eval_33_70": cosine_uncentered(pn, y_ev),
            "delta_frozen": cosine_uncentered(pn[fro], y_ev[fro]) - base_frozen,
        }
    res["per_gamma"] = per_gamma
    res["runtime_s"] = round(time.time() - t0, 1)

    (OUT / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"baseline frozen={base_frozen:.6f} eval33-70={base_all:.6f}")
    for g, d in per_gamma.items():
        print(f"gamma={g:.2f} frozen={d['cos_frozen_51_70']:.6f} (Δ{d['delta_frozen']:+.6f}) "
              f"eval={d['cos_eval_33_70']:.6f}")
    print(f"DONE {time.time()-t0:.0f}s -> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
