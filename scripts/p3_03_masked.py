# -*- coding: utf-8 -*-
"""P3-03: TinyLOBERT-style masked step pretraining (second-level approximation).

LOBERT (arXiv:2511.12563) tokenizes LOB messages and pretrains with masked
prediction. Our processed data is second-level aggregated, so the closest
legal approximation is: per-sample sequence (60 steps x 6 channels), small
transformer encoder, masked-step reconstruction pretraining on inner-train
months ONLY, then mean-pooled 32d latent feeds the frozen residual protocol.

Diagnostics: corr(latent, 152 dynamics features) must be < 0.70 for the
latent to count as new information (pre-registered in the method card).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn

from mscapital.models.m01a import run_m01a_outer, summarize_m01a
from mscapital.residual import CanonicalOOF

from p3_common import load_p3_frame, save_p3_features

CANONICAL = Path(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
F0726 = Path(r"D:\mscapital-kaggle\scripts\kaggle_0726ds\f0726_train_f32.parquet")
GRID_CACHE = Path(r"D:\mscapital-kaggle\output\p3_04_grid_features\second_grid.npz")
BASELINE_ROOT = Path(r"D:\mscapital-kaggle\output\c4_protocol_closed_final\clean-baseline-v2")
OUT = Path(r"D:\mscapital-kaggle\output\p3_03_masked_formal")
FEATURE_OUT = Path(r"D:\mscapital-kaggle\output\p3_03_masked_features")

INNER_TRAIN = {
    "PSEUDO": (21, 26),
    "H2": (21, 30),
    "T3": (21, 40),
    "T4": (21, 40),
}
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
LATENT = 32
MASK_RATIO = 0.20
EPOCHS = 2
BATCH = 1024
LR = 1e-3
SEED = 2026


class TinyLOBERT(nn.Module):
    def __init__(self, n_channels: int = 6, d_model: int = D_MODEL):
        super().__init__()
        self.proj = nn.Linear(n_channels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 60, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=N_HEADS, dim_feedforward=128,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.head = nn.Linear(d_model, n_channels)

    def forward(self, x, mask=None):
        h = self.proj(x) + self.pos
        h = self.encoder(h)
        out = self.head(h)
        return out, h


def load_grid() -> tuple[np.ndarray, np.ndarray]:
    """grid (n,60,6) + canonical-aligned ids from cache or build."""
    if GRID_CACHE.exists():
        with np.load(GRID_CACHE) as src:
            grid = src["grid"]
            ids = src["sample_id"]
        return grid, ids
    from p3_grid_common import build_second_grid
    grid = build_second_grid()
    ids = np.arange(grid.shape[0])
    GRID_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GRID_CACHE, grid=grid, sample_id=ids)
    return grid, ids


def pretrain(X: np.ndarray, device: torch.device) -> TinyLOBERT:
    """Masked-step reconstruction on given rows."""
    model = TinyLOBERT().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    mse = nn.MSELoss()
    n = X.shape[0]
    rng = np.random.default_rng(SEED)
    model.train()
    for epoch in range(EPOCHS):
        perm = rng.permutation(n)
        total = 0.0
        for start in range(0, n, BATCH):
            idx = perm[start:start + BATCH]
            xb = torch.from_numpy(X[idx]).to(device)
            mask = torch.rand(xb.shape[0], 60, 1, device=device) < MASK_RATIO
            xm = xb.clone()
            xm[mask.expand_as(xb)] = 0.0
            out, _ = model(xm)
            loss = mse(out[mask.expand_as(xb)], xb[mask.expand_as(xb)])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        print(f"  pretrain epoch {epoch + 1}/{EPOCHS} loss={total / n:.6f}")
    return model


def encode(model: TinyLOBERT, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    latents = []
    with torch.no_grad():
        for start in range(0, X.shape[0], 8192):
            xb = torch.from_numpy(X[start:start + 8192]).to(device)
            _, h = model(xb)
            latents.append(h.mean(dim=1).cpu().numpy())
    return np.concatenate(latents, axis=0)


def load_f0726(canonical: CanonicalOOF) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import polars as pl
    df = pl.read_parquet(F0726).sort("sample_id")
    ids = df["sample_id"].to_numpy()
    order = np.searchsorted(ids, canonical.sample_id)
    names = [c for c in df.columns if c not in ("sample_id", "target")]
    X = df.select(names).to_numpy()[order].astype(np.float32)
    return X, np.asarray(canonical.month), names


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    canonical = CanonicalOOF(**{
        k: np.asarray(np.load(CANONICAL)[k]) for k in
        ("sample_id", "month", "target", "baseline_oof", "source_train_end")
    })
    canonical.validate()

    grid, gids = load_grid()
    pos = np.searchsorted(np.sort(gids), canonical.sample_id)
    order = np.argsort(gids)
    Xgrid = grid[order][pos]
    print(f"grid aligned: {Xgrid.shape}")

    X152, months, names152 = load_f0726(canonical)

    for outer, (t0, t1) in INNER_TRAIN.items():
        print(f"\n=== outer {outer}: pretrain on months {t0}-{t1} ===")
        train_mask = (months >= t0) & (months <= t1)
        model = pretrain(Xgrid[train_mask], device)
        latent = encode(model, Xgrid, device)
        # diagnostics: latent vs 152-feature correlation
        Lc = np.nan_to_num(latent, nan=0.0)
        Xc = np.nan_to_num(X152, nan=0.0)
        corr = np.corrcoef(Lc, Xc, rowvar=False)[:LATENT, LATENT:]
        print(f"  corr(latent, 152feats): absmax={np.abs(corr).max():.4f} mean={np.abs(corr).mean():.4f}")
        names_all = tuple(f"masked_latent_{i}" for i in range(LATENT))
        feat_path = FEATURE_OUT / f"{outer.lower()}_features.parquet"
        save_p3_features(
            feat_path, f"p3-03-masked-{outer.lower()}", names_all,
            canonical.sample_id, canonical.month, canonical.target, latent,
            extra={"latent_152_absmax_corr": float(np.abs(corr).max())},
        )
        frame = load_p3_frame(feat_path, names_all)
        diag = run_m01a_outer(canonical, frame, BASELINE_ROOT, OUT, outer)
        print(f"  {outer}: delta={diag['delta_vs_baseline']:+.9f} score={diag['final_score']:.9f}")

    summary = summarize_m01a(OUT)
    print("\n=== P3-03 gate ===")
    for row in summary["rows"]:
        print(f"  {row['outer']}: delta={row['delta_vs_baseline']:+.9f}")
    print(f"mean delta={summary['mean_delta']:+.9f} gate={summary['gate']}")


if __name__ == "__main__":
    main()
