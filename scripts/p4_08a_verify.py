# -*- coding: utf-8 -*-
"""P4-08A-verify: cosine-loss MLP diagnostics.

1. corr(cosine_pred, baseline) vs corr(mse_pred, baseline) - are they
   different directions?
2. corr(cosine_pred, y) per outer, month stability
3. alpha stability: re-select alpha on valid itself (overfit check) vs tune
4. monthly delta breakdown
5. relation to LB142-style behavior: does cosine pred favor high-activity?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import polars as pl

from mscapital.models.m01a import select_alpha
from mscapital.metrics import cosine_uncentered, normalize_prediction
from mscapital.residual import CanonicalOOF

OUTER_MONTHS = {"PSEUDO": (21, 32), "H2": (21, 40), "T3": (21, 50), "T4": (21, 50)}
INNER_TRAIN = {"PSEUDO": (21, 26), "H2": (21, 30), "T3": (21, 40), "T4": (21, 40)}
INNER_TUNE = {"PSEUDO": (27, 32), "H2": (31, 40), "T3": (41, 50), "T4": (41, 50)}
CANONICAL = Path(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
F0726 = Path(r"D:\mscapital-kaggle\scripts\kaggle_0726ds\f0726_train_f32.parquet")
MARKET = Path(r"D:\mscapital-forecasting\data\raw\train\market.feather")
OUT = Path(r"D:\mscapital-kaggle\output\p4_08a_verify")
OUT.mkdir(parents=True, exist_ok=True)

EPOCHS = 15
BATCH = 4096
SEED = 42


def cosine_batch_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = pred - pred.mean()
    t = target - target.mean()
    return 1.0 - torch.nn.functional.cosine_similarity(p.unsqueeze(0), t.unsqueeze(0), dim=1).squeeze()


def main() -> None:
    import torch.nn as nn
    from p4_08a_loss_ablation import MLP, train_model, predict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    canonical = CanonicalOOF(**{
        k: np.asarray(np.load(CANONICAL)[k]) for k in
        ("sample_id", "month", "target", "baseline_oof", "source_train_end")
    })
    canonical.validate()
    fe = pl.read_parquet(F0726).sort("sample_id")
    pos = np.searchsorted(fe["sample_id"].to_numpy(), canonical.sample_id)
    names = [c for c in fe.columns if c not in ("sample_id", "target")]
    X = np.nan_to_num(fe.select(names).to_numpy()[pos].astype(np.float64), nan=0.0)
    y = np.asarray(canonical.target, dtype=np.float64)
    cmonth = np.asarray(canonical.month, dtype=int)
    base = np.asarray(canonical.baseline_oof, dtype=np.float64)

    # activity
    act = pl.read_ipc(MARKET, columns=["sample_id"]).group_by("sample_id").len().sort("sample_id")
    apos = np.searchsorted(act["sample_id"].to_numpy(), canonical.sample_id)
    activity = act["len"].to_numpy()[apos].astype(np.float64)

    for loss_name in ("mse", "cosine"):
        print(f"\n===== {loss_name} =====")
        for outer in ("PSEUDO", "H2", "T3", "T4"):
            t0, t1 = INNER_TRAIN[outer]
            u0, u1 = INNER_TUNE[outer]
            v0, v1 = OUTER_MONTHS[outer]
            tr = (cmonth >= t0) & (cmonth <= t1)
            tu = (cmonth >= u0) & (cmonth <= u1)
            va = (cmonth >= v0) & (cmonth <= v1)
            mu = X[tr].mean(axis=0, keepdims=True)
            sd = X[tr].std(axis=0, keepdims=True) + 1e-8
            Xs = (X - mu) / sd
            model, best_ep = train_model(Xs[tr], y[tr], Xs[tu], y[tu], loss_name, device)
            pv = predict(model, Xs[va], device)
            pt = predict(model, Xs[tu], device)
            base_v, _ = normalize_prediction(base[va], "rms")
            sel = select_alpha(base[tu], pt, y[tu])
            alpha = sel["alpha"]
            pn, _ = normalize_prediction(pv, "rms")
            final = base_v + alpha * pn
            print(f"  {outer}: corr(pred,base)={np.corrcoef(pv, base[va])[0,1]:+.3f} "
                  f"corr(pred,y)={np.corrcoef(pv, y[va])[0,1]:+.4f} alpha={alpha:.2f} ep={best_ep}")
            # overfit check: alpha reselected on valid
            sel2 = select_alpha(base[va], pv, y[va])
            delta_tune = cosine_uncentered(final, y[va]) - cosine_uncentered(base_v, y[va])
            delta_oracle = cosine_uncentered(base_v + sel2["alpha"] * pn, y[va]) - cosine_uncentered(base_v, y[va])
            print(f"           delta(tune-alpha)={delta_tune:+.6f} delta(oracle-alpha)={delta_oracle:+.6f} "
                  f"alpha_oracle={sel2['alpha']:.2f}")
            # monthly breakdown
            md = []
            for m in range(v0, v1 + 1):
                mm = (cmonth[va] == m)
                if mm.sum() > 100:
                    d = cosine_uncentered(final[mm], y[va][mm]) - cosine_uncentered(base_v[mm], y[va][mm])
                    md.append((m, d))
            pos_m = sum(1 for _, d in md if d > 0)
            print(f"           monthly: {pos_m}/{len(md)} months positive; "
                  f"largest: {max(md, key=lambda x: x[1])}, smallest: {min(md, key=lambda x: x[1])}")
            # activity split
            av = activity[va]
            for lo_, hi_, lbl in ((0.0, 0.5, "lo"), (0.5, 0.9, "mid"), (0.9, 1.0, "hi")):
                m2 = (av >= np.quantile(av, lo_)) & (av <= np.quantile(av, hi_))
                d = cosine_uncentered(final[m2], y[va][m2]) - cosine_uncentered(base_v[m2], y[va][m2])
                print(f"           {lbl}-act delta={d:+.6f}")
    print("\nwritten to", OUT)


if __name__ == "__main__":
    main()
