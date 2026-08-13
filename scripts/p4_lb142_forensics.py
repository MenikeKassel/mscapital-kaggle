# -*- coding: utf-8 -*-
"""P4-03: LB142 forensics - what structure explains ref-v7 disagreement.

d_i = unit(pred_lb142) - unit(pred_v7) on test. Where is |d| large?
Compare feature distributions (0726 test features) between high-|d| and
low-|d| samples; check price-level / activity / liquidity structure.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl

REF = Path(r"D:\mscapital-forecasting\reference\lb142\submission_ref_lb0142.csv")
V7 = Path(r"D:\mscapital-kaggle\output\submissions\submission_blend_v7_rl20.csv")
F0726_TEST = Path(r"D:\mscapital-kaggle\scripts\kaggle_0726ds\f0726_test_f32.parquet")
OUT = Path(r"D:\mscapital-kaggle\output\p4_lb142_forensics")
OUT.mkdir(parents=True, exist_ok=True)


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    return x / (np.linalg.norm(x) + 1e-12)


def main() -> None:
    ref = pl.read_csv(REF).sort("sample_id")
    v7 = pl.read_csv(V7).sort("sample_id")
    assert ref.height == v7.height
    ids = ref["sample_id"].to_numpy()
    p_ref = ref["prediction"].to_numpy().astype(np.float64)
    p_v7 = v7["prediction"].to_numpy().astype(np.float64)
    print(f"rows={len(ids)} corr(ref, v7)={np.corrcoef(p_ref, p_v7)[0, 1]:.4f}")

    u_ref, u_v7 = unit(p_ref), unit(p_v7)
    d = u_ref - u_v7
    ad = np.abs(d)
    print(f"|d|: p50={np.quantile(ad, .5):.6f} p90={np.quantile(ad, .9):.6f} "
          f"p99={np.quantile(ad, .99):.6f} max={ad.max():.6f}")

    # feature comparison high vs low |d|
    feats = pl.read_parquet(F0726_TEST).sort("sample_id")
    fids = feats["sample_id"].to_numpy()
    pos = np.searchsorted(fids, ids)
    if not np.array_equal(fids[pos], ids):
        raise ValueError("0726 test features do not cover ref ids")
    names = [c for c in feats.columns if c not in ("sample_id", "target")]
    X = feats.select(names).to_numpy()[pos].astype(np.float64)

    hi = ad >= np.quantile(ad, 0.95)
    lo = ad <= np.quantile(ad, 0.05)
    print(f"\nhigh-|d| (5%): {hi.sum()}  low-|d| (5%): {lo.sum()}")
    print("\n=== top discriminating features (|mean diff| / pooled std) ===")
    m_hi = np.nanmean(X[hi], axis=0)
    m_lo = np.nanmean(X[lo], axis=0)
    s_pool = np.sqrt((np.nanstd(X[hi], axis=0) ** 2 + np.nanstd(X[lo], axis=0) ** 2) / 2 + 1e-12)
    eff = np.abs(m_hi - m_lo) / s_pool
    order = np.argsort(eff)[::-1]
    for i in order[:15]:
        print(f"  {names[i]:45s} eff={eff[i]:.3f}  hi={m_hi[i]:+.4f}  lo={m_lo[i]:+.4f}")
    # nan ratio difference
    nan_hi = np.isnan(X[hi]).mean(axis=0)
    nan_lo = np.isnan(X[lo]).mean(axis=0)
    nan_diff = nan_hi - nan_lo
    top_nan = np.argsort(np.abs(nan_diff))[::-1]
    print("\n=== top missingness difference ===")
    for i in top_nan[:8]:
        print(f"  {names[i]:45s} nan_hi={nan_hi[i]:.3f} nan_lo={nan_lo[i]:.3f}")

    np.savez_compressed(OUT / "d_analysis.npz",
                        sample_id=ids, p_ref=p_ref, p_v7=p_v7, d=d, ad=ad,
                        eff=eff, nan_diff=nan_diff)
    (OUT / "report.md").write_text("\n".join([
        "# P4-03 LB142 forensics", "",
        f"- corr(ref, v7): {np.corrcoef(p_ref, p_v7)[0,1]:.4f}",
        f"- |d| p90/p99: {np.quantile(ad, .9):.6f} / {np.quantile(ad, .99):.6f}",
        f"- top discriminator: {names[order[0]]} (eff={eff[order[0]]:.3f})",
        f"- top missingness diff: {names[top_nan[0]]} ({nan_diff[top_nan[0]]:+.3f})", "",
    ]), encoding="utf-8")
    print("\nwritten to", OUT)


if __name__ == "__main__":
    main()
