# -*- coding: utf-8 -*-
"""P4-01(a)-3 Conditional Test: is market-path discrimination concentrated in
high-activity / high-volatility regimes?

Design:
- activity proxy: from f0726 features (o_ event rate / t_lv_mean) -> quintiles
- volatility proxy: f0726 t_price_volatility / t_px_std -> quintiles
- Within each (activity Q, vol Q) cell: Cohen's d of key market descriptors
  (high-|d| vs low-|d|, top/bottom 10%), plus E[|d|] by cell
- Key question: does market path info separate high/low |d| INSIDE the
  high-activity x high-vol cell (where P4-04 says disagreement lives)?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl

from p4_01a_common import OUT, DESC_PATH, load_d, load_f0726

KEY_FEATURES = [
    "n_snap", "mid_range", "mid_std", "spread_widen_freq",
    "imb_std", "imb_mean", "vol_recent_early", "spread_recent_early",
    "depth_recent_early", "mid_trend", "depth_trend", "spread_trend",
]


def quintile_bins(x: np.ndarray) -> np.ndarray:
    """Rank-based quintiles (robust to degenerate distributions)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.int64)
    ranks[order] = np.arange(len(x))
    return (ranks * 5 // len(x)).clip(0, 4) + 1  # 1..5


def main() -> None:
    ids, d, ad = load_d()
    fids, X152, fnames = load_f0726()
    assert np.array_equal(fids, ids), "f0726 id mismatch"
    desc = pl.read_parquet(DESC_PATH).sort("sample_id")
    assert np.array_equal(desc["sample_id"].to_numpy(), ids), "desc id mismatch"

    # activity proxy: event rate = o_* count-ish feature; use t_lv_mean & o features
    # (f0726 features are 152-dim; pick interpretable proxies by name)
    act_cands = [c for c in fnames if c.startswith("o_") and "count" in c]
    vol_cands = [c for c in fnames if "volatility" in c or "range_ratio" in c]
    print("activity candidates:", act_cands[:5])
    print("vol candidates:", vol_cands[:5])
    if not act_cands:
        act_cands = [c for c in fnames if c.startswith("o_")][:3]
    if not vol_cands:
        vol_cands = [c for c in fnames if c.startswith("t_")][:3]
    act_proxy = act_cands[0]
    vol_proxy = vol_cands[0]
    print(f"using activity proxy={act_proxy}, vol proxy={vol_proxy}")

    ia = fnames.index(act_proxy)
    iv = fnames.index(vol_proxy)
    act_q = quintile_bins(X152[:, ia])
    vol_q = quintile_bins(X152[:, iv])

    # E[|d|] by activity x vol cell
    print("\n=== E[|d|] x 1e4 by activity Q (rows) x vol Q (cols) ===")
    grid = np.zeros((5, 5))
    for a in range(1, 6):
        row = []
        for v in range(1, 6):
            m = (act_q == a) & (vol_q == v)
            grid[a - 1, v - 1] = ad[m].mean() * 1e4
        print("  ".join(f"{grid[a-1, v-1]:7.2f}" for v in range(1, 6)))

    # per-cell discrimination of key market descriptors (top vs bottom 10% |d|)
    print("\n=== per-cell Cohen's d (high|d| vs low|d|, 10%) ===")
    print("cells: rows=activity Q, cols=vol Q")
    for feat in KEY_FEATURES:
        j = desc.columns.index(feat)
        vals = desc[feat].to_numpy().astype(np.float64)
        print(f"\n[{feat}]")
        for a in range(1, 6):
            row_vals = []
            for v in range(1, 6):
                m = (act_q == a) & (vol_q == v)
                idx = np.where(m)[0]
                if len(idx) < 2000:
                    row_vals.append(float("nan"))
                    continue
                ad_m = ad[idx]
                n = max(int(len(idx) * 0.10), 50)
                hi = idx[np.argsort(ad_m)[-n:]]
                lo = idx[np.argsort(ad_m)[:n]]
                xh, xl = vals[hi], vals[lo]
                sp = np.sqrt(((len(xh) - 1) * xh.var(ddof=1) + (len(xl) - 1) * xl.var(ddof=1)) /
                             (len(xh) + len(xl) - 2))
                d_eff = (xh.mean() - xl.mean()) / sp if sp > 0 else 0.0
                row_vals.append(d_eff)
            print("  " + "  ".join(f"{v:+.2f}" if not np.isnan(v) else "  nan" for v in row_vals))

    # summary: mean |d| by activity quintile (should replicate P4-04 monotonicity)
    print("\n=== E[|d|] by activity quintile ===")
    for a in range(1, 6):
        print(f"  Q{a}: {ad[act_q == a].mean() * 1e4:.3f} (x1e4)")
    print("=== E[|d|] by vol quintile ===")
    for v in range(1, 6):
        print(f"  Q{v}: {ad[vol_q == v].mean() * 1e4:.3f} (x1e4)")


if __name__ == "__main__":
    main()
