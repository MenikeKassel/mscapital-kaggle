# -*- coding: utf-8 -*-
"""P3-04: OrderFusion-style 2.5D spatial-temporal grid (projected).

Grid projection (minimum version): per-second 6-channel event grid (60x6=360,
time structure) + 16 log-price bins x 3 channels (48, price structure) =
408 features -> frozen M01-A residual protocol. If the projection shows
signal, the next step is a real 2.5D CNN encoder on the (60, 16, 3) tensor.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from mscapital.models.m01a import run_m01a_outer, summarize_m01a
from mscapital.residual import CanonicalOOF

from p3_common import load_p3_frame, save_p3_features
from p3_grid_common import build_price_hist, build_second_grid

CANONICAL = Path(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
BASELINE_ROOT = Path(r"D:\mscapital-kaggle\output\c4_protocol_closed_final\clean-baseline-v2")
OUT = Path(r"D:\mscapital-kaggle\output\p3_04_grid_formal")
FEATURE_OUT = Path(r"D:\mscapital-kaggle\output\p3_04_grid_features")


def main() -> None:
    canonical = CanonicalOOF(**{
        k: np.asarray(np.load(CANONICAL)[k]) for k in
        ("sample_id", "month", "target", "baseline_oof", "source_train_end")
    })
    canonical.validate()

    print("building second grid (60x6)...")
    grid = build_second_grid()
    print(f"grid: {grid.shape}")
    print("building price histogram (16x3)...")
    hist, edges = build_price_hist()
    print(f"hist: {hist.shape}")

    ids = np.arange(grid.shape[0])
    # keep only canonical sample ids
    mask = np.isin(ids, canonical.sample_id)
    grid = grid[mask]
    hist = hist[mask]
    ids = ids[mask]
    order = np.argsort(ids)
    grid = grid[order]
    hist = hist[order]
    ids = ids[order]

    # align to canonical order
    pos = np.searchsorted(ids, canonical.sample_id)
    if not np.array_equal(ids[pos], canonical.sample_id):
        raise ValueError("grid does not cover canonical sample ids")

    names_sec = tuple(f"sec_{s}_ch_{c}" for s in range(60) for c in range(6))
    names_px = tuple(f"pxbin_{b}_ch_{c}" for b in range(16) for c in range(3))
    names_all = names_sec + names_px
    values = np.hstack([grid.reshape(grid.shape[0], -1), hist]).astype(np.float32)
    feat_path = FEATURE_OUT / "grid_features.parquet"
    save_p3_features(
        feat_path, "p3-04-grid", names_all,
        canonical.sample_id, canonical.month, canonical.target, values,
        extra={"price_bin_edges": edges.tolist()},
    )
    print(f"features saved: {values.shape}")

    frame = load_p3_frame(feat_path, names_all)
    for outer in ("PSEUDO", "H2", "T3", "T4"):
        diag = run_m01a_outer(canonical, frame, BASELINE_ROOT, OUT, outer)
        print(f"{outer}: delta={diag['delta_vs_baseline']:+.9f} score={diag['final_score']:.9f}")

    summary = summarize_m01a(OUT)
    print("\n=== P3-04 gate ===")
    for row in summary["rows"]:
        print(f"  {row['outer']}: delta={row['delta_vs_baseline']:+.9f}")
    print(f"mean delta={summary['mean_delta']:+.9f} gate={summary['gate']}")


if __name__ == "__main__":
    main()
