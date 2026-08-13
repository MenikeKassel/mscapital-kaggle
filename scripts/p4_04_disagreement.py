# -*- coding: utf-8 -*-
"""P4-04: LB142 Disagreement Forensics (deep).

d = unit(ref) - unit(v7) on test. Where do they disagree, structurally?
1. |d| vs target magnitude (unobservable on test -> use 0726 test features as
   proxies: activity/vol already found; add market-like structure via features)
2. |d| by month proxy: test rows have no month... check against v7-only
   structural features: price level cluster (via t_/o_ features), missingness
3. Top features discriminating high-|d| (done in round 1) - now check the
   *direction*: are high-|d| samples also high-|ref| (ref more extreme) or
   high-|v7|? i.e. who is more confident?
4. Split d by price-level proxy (features that track the 0.5 cluster).
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
OUT = Path(r"D:\mscapital-kaggle\output\p4_04_disagreement_forensics")
OUT.mkdir(parents=True, exist_ok=True)


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    return x / (np.linalg.norm(x) + 1e-12)


def main() -> None:
    ref = pl.read_csv(REF).sort("sample_id")
    v7 = pl.read_csv(V7).sort("sample_id")
    ids = ref["sample_id"].to_numpy()
    p_ref = ref["prediction"].to_numpy().astype(np.float64)
    p_v7 = v7["prediction"].to_numpy().astype(np.float64)
    u_ref, u_v7 = unit(p_ref), unit(p_v7)
    d = u_ref - u_v7
    ad = np.abs(d)

    feats = pl.read_parquet(F0726_TEST).sort("sample_id")
    fids = feats["sample_id"].to_numpy()
    pos = np.searchsorted(fids, ids)
    names = [c for c in feats.columns if c not in ("sample_id", "target")]
    X = feats.select(names).to_numpy()[pos].astype(np.float64)

    # 1) who is more extreme: |ref| vs |v7| on high-|d|
    hi = ad >= np.quantile(ad, 0.95)
    lo = ad <= np.quantile(ad, 0.05)
    print("=== who disagrees (|ref| vs |v7| on high-|d|) ===")
    for label, mask in (("high-|d|", hi), ("low-|d|", lo)):
        print(f"  {label}: mean|ref|={np.abs(u_ref[mask]).mean():.5f} "
              f"mean|v7|={np.abs(u_v7[mask]).mean():.5f} "
              f"ref_std={u_ref[mask].std():.5f} v7_std={u_v7[mask].std():.5f}")

    # 2) correlation of |d| with feature activity groups (from round 1)
    act_idx = [names.index(c) for c in names if "o_n_45" == c or "t_sec_autocorr_lag1" == c or "o_sec_row_count_15" == c or "t_price_range_ratio" == c]
    print("\n=== corr(|d|, activity features) ===")
    for i in act_idx:
        v = np.nan_to_num(X[:, i], nan=0.0)
        print(f"  corr(|d|, {names[i]:30s}) = {np.corrcoef(v, ad)[0,1]:+.4f}")

    # 3) |d| vs missingness (data completeness)
    miss = np.isnan(X).mean(axis=1)
    print(f"\ncorr(|d|, missingness) = {np.corrcoef(miss, ad)[0,1]:+.4f}")

    # 4) sign asymmetry: is d biased (ref consistently above/below v7)?
    print(f"\n=== sign asymmetry ===")
    print(f"  d>0: {(d > 0).mean():.4f}  mean(d|d>0)={d[d > 0].mean():.6f}  mean(|d|)={ad.mean():.6f}")
    # signed d vs activity: does ref over-predict on active samples?
    for i in act_idx:
        v = np.nan_to_num(X[:, i], nan=0.0)
        hi_act = v >= np.quantile(v, 0.9)
        lo_act = v <= np.quantile(v, 0.1)
        print(f"  {names[i]:30s}: d_mean hi-act={d[hi_act].mean():+.6f} lo-act={d[lo_act].mean():+.6f}")

    # 5) price-level proxy: low-price cluster via feature? use o_/t_ price level
    px_feats = [c for c in names if "px" in c.lower() or "price" in c.lower()][:6]
    print(f"\n=== |d| vs price-level features ===")
    for c in px_feats:
        i = names.index(c)
        v = np.nan_to_num(X[:, i], nan=0.0)
        print(f"  corr(|d|, {c:35s}) = {np.corrcoef(v, ad)[0,1]:+.4f}")

    np.savez_compressed(OUT / "disagreement.npz",
                        sample_id=ids, d=d, ad=ad, u_ref=u_ref, u_v7=u_v7, miss=miss)
    print("\nwritten to", OUT)


if __name__ == "__main__":
    main()
