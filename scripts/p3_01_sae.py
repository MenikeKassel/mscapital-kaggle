# -*- coding: utf-8 -*-
"""P3-01: Supervised AutoEncoder latent features (JS 2021 1st / DRW 2025 1st).

For each outer fold a small SAE (152 -> 64 -> 32, recon + supervised head) is
trained on the inner-train months ONLY (Yirun's anti-leakage discipline), then
latent (32d) is concatenated with the raw 152 features and the frozen M01-A
residual protocol runs unchanged (CatBoost on residual -> alpha -> outer).

Provides: learned nonlinear latent that no hand-crafted feature has.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from mscapital.models.m01a import run_m01a_outer, summarize_m01a
from mscapital.residual import CanonicalOOF

from p3_common import load_p3_frame, save_p3_features

CANONICAL = Path(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
F0726 = Path(r"D:\mscapital-kaggle\scripts\kaggle_0726ds\f0726_train_f32.parquet")
BASELINE_ROOT = Path(r"D:\mscapital-kaggle\output\c4_protocol_closed_final\clean-baseline-v2")
OUT = Path(r"D:\mscapital-kaggle\output\p3_01_sae_formal")
FEATURE_OUT = Path(r"D:\mscapital-kaggle\output\p3_01_sae_features")

INNER_TRAIN = {
    "PSEUDO": (21, 26),
    "H2": (21, 30),
    "T3": (21, 40),
    "T4": (21, 40),
}
LATENT_DIM = 32
EPOCHS = 8
BATCH = 4096
LR = 1e-3
LAMBDA_PRED = 1.0
NOISE = 0.01
SEED = 2026


class SAE(nn.Module):
    def __init__(self, n_in: int, latent: int = LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, 64), nn.SiLU(), nn.Linear(64, latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 64), nn.SiLU(), nn.Linear(64, n_in)
        )
        self.head = nn.Linear(latent, 1)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), self.head(z), z


def load_f0726_aligned(canonical: CanonicalOOF) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = pl.read_parquet(F0726)
    names = [c for c in df.columns if c not in ("sample_id", "target")]
    df = df.sort("sample_id")
    ids = df["sample_id"].to_numpy()
    order = np.searchsorted(ids, canonical.sample_id)
    if np.any(order >= ids.size) or not np.array_equal(ids[order], canonical.sample_id):
        raise ValueError("f0726 features do not cover canonical sample ids")
    X = df.select(names).to_numpy()[order].astype(np.float32)
    t = df["target"].to_numpy()[order].astype(np.float64)
    if not np.allclose(t, canonical.target, atol=1e-12):
        raise ValueError("f0726 target does not match canonical target")
    return X, t, np.asarray(canonical.month), names


def train_sae(X: np.ndarray, y: np.ndarray, device: torch.device) -> tuple[SAE, np.ndarray, np.ndarray]:
    """Train SAE on inner-train rows; return model + row mean/std (fitted on train only)."""
    mean = np.nanmean(X, axis=0, keepdims=True)
    std = np.nanstd(X, axis=0, keepdims=True) + 1e-8
    X = np.where(np.isnan(X), mean, X)
    Xs = (X - mean) / std
    model = SAE(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()
    n = Xs.shape[0]
    rng = np.random.default_rng(SEED)
    model.train()
    for epoch in range(EPOCHS):
        perm = rng.permutation(n)
        total = 0.0
        for start in range(0, n, BATCH):
            idx = perm[start:start + BATCH]
            xb = torch.from_numpy(Xs[idx]).to(device)
            yb = torch.from_numpy(y[idx].reshape(-1, 1).astype(np.float32)).to(device)
            if NOISE > 0:
                xb = xb + NOISE * torch.randn_like(xb)
            recon, pred, _ = model(xb)
            loss = mse(recon, xb) + LAMBDA_PRED * mse(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        if epoch in (0, EPOCHS - 1):
            print(f"  sae epoch {epoch + 1}/{EPOCHS} loss={total / n:.6f}")
    return model, mean, std


def encode(model: SAE, X: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    X = np.where(np.isnan(X), mean, X)
    Xs = (X - mean) / std
    latents = []
    with torch.no_grad():
        for start in range(0, Xs.shape[0], 65536):
            xb = torch.from_numpy(Xs[start:start + 65536]).to(device)
            _, _, z = model(xb)
            latents.append(z.cpu().numpy())
    return np.concatenate(latents, axis=0)


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    canonical = CanonicalOOF(**{
        k: np.asarray(np.load(CANONICAL)[k]) for k in
        ("sample_id", "month", "target", "baseline_oof", "source_train_end")
    })
    canonical.validate()
    X, t, months, names = load_f0726_aligned(canonical)
    print(f"aligned: {X.shape} rows, {len(names)} features")

    for outer, (t0, t1) in INNER_TRAIN.items():
        print(f"\n=== outer {outer}: SAE on inner-train months {t0}-{t1} ===")
        train_mask = (months >= t0) & (months <= t1)
        model, mean, std = train_sae(X[train_mask], t[train_mask], device)
        latent = encode(model, X, mean, std, device)
        values = np.hstack([X, latent])
        names_all = tuple(names) + tuple(f"sae_latent_{i}" for i in range(LATENT_DIM))
        feat_path = FEATURE_OUT / f"{outer.lower()}_features.parquet"
        save_p3_features(
            feat_path, f"p3-01-sae-{outer.lower()}", names_all,
            canonical.sample_id, canonical.month, canonical.target, values,
        )
        frame = load_p3_frame(feat_path, names_all)
        diag = run_m01a_outer(canonical, frame, BASELINE_ROOT, OUT, outer)
        print(f"  {outer}: delta={diag['delta_vs_baseline']:+.9f} score={diag['final_score']:.9f}")

    summary = summarize_m01a(OUT)
    print("\n=== P3-01 gate ===")
    for row in summary["rows"]:
        print(f"  {row['outer']}: delta={row['delta_vs_baseline']:+.9f}")
    print(f"mean delta={summary['mean_delta']:+.9f} gate={summary['gate']}")


if __name__ == "__main__":
    main()
