# -*- coding: utf-8 -*-
"""P4-08A: loss ablation - same features, same architecture, same splits.

MLP (256-256-64, RealMLP-style, SINGLE model, no ensemble) on the 152-feature
stack, canonical outer protocol:
  MSE loss  vs  cosine loss (per-batch demean)
Only the loss changes. inner-train trains, inner-tune selects epoch + alpha,
outer valid scores delta vs frozen baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn

from mscapital.models.m01a import select_alpha
from mscapital.metrics import cosine_uncentered, normalize_prediction
from mscapital.residual import CanonicalOOF

OUTER_MONTHS = {"PSEUDO": (21, 32), "H2": (21, 40), "T3": (21, 50), "T4": (21, 50)}
INNER_TRAIN = {"PSEUDO": (21, 26), "H2": (21, 30), "T3": (21, 40), "T4": (21, 40)}
INNER_TUNE = {"PSEUDO": (27, 32), "H2": (31, 40), "T3": (41, 50), "T4": (41, 50)}
CANONICAL = Path(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
BASELINE_ROOT = Path(r"D:\mscapital-kaggle\output\c4_protocol_closed_final\clean-baseline-v2")
F0726 = Path(r"D:\mscapital-kaggle\scripts\kaggle_0726ds\f0726_train_f32.parquet")
OUT = Path(r"D:\mscapital-kaggle\output\p4_08a_unc_ablation")
OUT.mkdir(parents=True, exist_ok=True)

EPOCHS = 15
BATCH = 4096
LR = 1e-3
SEED = 42


class MLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.SiLU(), nn.Dropout(0.01),
            nn.Linear(256, 256), nn.SiLU(), nn.Dropout(0.01),
            nn.Linear(256, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)


def cosine_batch_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.nn.functional.cosine_similarity(
        pred.reshape(1, -1), target.reshape(1, -1), dim=1).squeeze()


def train_model(Xtr: np.ndarray, ytr: np.ndarray, Xtu: np.ndarray, ytu: np.ndarray,
                loss_name: str, device: torch.device) -> tuple[MLP, int]:
    torch.manual_seed(SEED)
    d_in = Xtr.shape[1]
    model = MLP(d_in).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xv = torch.tensor(Xtu, dtype=torch.float32, device=device)
    yv = torch.tensor(ytu, dtype=torch.float32, device=device)
    best_ep, best_score = 0, -1e9
    n = len(Xtr)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            p = model(Xt[idx])
            if loss_name == "mse":
                loss = torch.nn.functional.mse_loss(p, yt[idx])
            else:
                loss = cosine_batch_loss(p, yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).cpu().numpy()
        sc = float(cosine_uncentered(pv, ytu))
        if sc > best_score:
            best_score, best_ep = sc, ep
    return model, best_ep


@torch.no_grad()
def predict(model: MLP, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), BATCH * 8):
        xb = torch.tensor(X[i:i + BATCH * 8], dtype=torch.float32, device=device)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs)


def main() -> None:
    import polars as pl
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    canonical = CanonicalOOF(**{
        k: np.asarray(np.load(CANONICAL)[k]) for k in
        ("sample_id", "month", "target", "baseline_oof", "source_train_end")
    })
    canonical.validate()
    fe = pl.read_parquet(F0726).sort("sample_id")
    fids = fe["sample_id"].to_numpy()
    pos = np.searchsorted(fids, canonical.sample_id)
    names = [c for c in fe.columns if c not in ("sample_id", "target")]
    X = fe.select(names).to_numpy()[pos].astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    print(f"X: {X.shape}")

    for loss_name in ("mse", "cosine"):
        rows = []
        for outer in ("PSEUDO", "H2", "T3", "T4"):
            t0, t1 = INNER_TRAIN[outer]
            u0, u1 = INNER_TUNE[outer]
            v0, v1 = OUTER_MONTHS[outer]
            cmonth = np.asarray(canonical.month, dtype=int)
            tr = (cmonth >= t0) & (cmonth <= t1)
            tu = (cmonth >= u0) & (cmonth <= u1)
            va = (cmonth >= v0) & (cmonth <= v1)
            # standardize on train
            mu = X[tr].mean(axis=0, keepdims=True)
            sd = X[tr].std(axis=0, keepdims=True) + 1e-8
            Xs = (X - mu) / sd
            y = np.asarray(canonical.target, dtype=np.float64)
            model, best_ep = train_model(Xs[tr], y[tr], Xs[tu], y[tu], loss_name, device)
            pred_valid = predict(model, Xs[va], device)
            pred_tune = predict(model, Xs[tu], device)
            # baseline + alpha * pred (alpha on tune)
            base_v, _ = normalize_prediction(np.asarray(canonical.baseline_oof)[va], "rms")
            sel = select_alpha(np.asarray(canonical.baseline_oof)[tu], pred_tune, y[tu])
            alpha = sel["alpha"]
            pred_n, _ = normalize_prediction(pred_valid, "rms")
            final = base_v + alpha * pred_n
            delta = float(cosine_uncentered(final, y[va]) - cosine_uncentered(base_v, y[va]))
            rows.append({"outer": outer, "delta": delta, "alpha": alpha, "best_ep": best_ep})
            print(f"[{loss_name}] {outer}: delta={delta:+.6f} alpha={alpha:.2f} ep={best_ep}")
        deltas = np.array([r["delta"] for r in rows])
        print(f"[{loss_name}] mean_delta={deltas.mean():+.6f} pos={int((deltas > 0).sum())}/4")
        import json
        (OUT / f"{loss_name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nwritten to", OUT)


if __name__ == "__main__":
    main()
