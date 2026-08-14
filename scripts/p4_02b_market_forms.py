# -*- coding: utf-8 -*-
"""P4-02B: rebuild LB142 v10 market-form primitives from 600s snapshots.

Specific forms our 152 stack lacks: m_ofi (quote-driven flow imbalance),
m_imb/m_imb2 (depth imbalance), m_dmid, m_mid_slope, m_*_gap, at windows
5/15/30/45/60/120/180/240/300.

Pipeline: rebuild -> overlap audit vs 152 (max corr + ridge CV R2) ->
residual signal (corr with y - beta*baseline, 4 fold) -> gate on
low-overlap + residual corr.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import polars as pl

from mscapital.residual import CanonicalOOF

MARKET = Path(r"D:\mscapital-forecasting\data\raw\train\market.feather")
CANONICAL = Path(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
OUT = Path(r"D:\mscapital-kaggle\output\p4_02b_market_forms")
OUT.mkdir(parents=True, exist_ok=True)

WINDOWS = [5, 15, 30, 45, 60, 120, 180, 240, 300]


def build_market_forms() -> tuple[np.ndarray, list[str]]:
    mkt = pl.read_ipc(MARKET, columns=[
        "sample_id", "seconds_before_predict", "ask_price_1", "bid_price_1",
        "ask_volume_1", "bid_volume_1", "ask_price_2", "bid_price_2",
        "ask_volume_2", "bid_volume_2", "transaction_volume",
    ])
    mkt = mkt.with_columns([
        ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2).alias("mid"),
        (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread1"),
        (pl.col("ask_price_2") - pl.col("bid_price_2")).alias("spread2"),
        (pl.col("bid_volume_1") - pl.col("ask_volume_1")).alias("imb"),
        (pl.col("bid_volume_2") - pl.col("ask_volume_2")).alias("imb2"),
    ]).sort(["sample_id", "seconds_before_predict"])
    # OFI: change in (bid_vol - ask_vol) signed by quote move direction
    mkt = mkt.with_columns([
        (pl.col("ask_price_1") - pl.col("ask_price_1").shift(1)).alias("d_ask"),
        (pl.col("bid_price_1") - pl.col("bid_price_1").shift(1)).alias("d_bid"),
        (pl.col("ask_volume_1") - pl.col("ask_volume_1").shift(1)).alias("d_av1"),
        (pl.col("bid_volume_1") - pl.col("bid_volume_1").shift(1)).alias("d_bv1"),
    ])
    # OFI per step (signed quote-flow imbalance), zero first row of each sample
    ofi = (pl.when(pl.col("d_bid") > 0).then(pl.col("d_bv1"))
           .when(pl.col("d_bid") < 0).then(-pl.col("bid_volume_1"))
           .otherwise(pl.col("d_bv1"))) \
        - (pl.when(pl.col("d_ask") > 0).then(pl.col("d_av1"))
           .when(pl.col("d_ask") < 0).then(-pl.col("ask_volume_1"))
           .otherwise(pl.col("d_av1")))
    mkt = mkt.with_columns(ofi.alias("ofi"))
    n = int(mkt["sample_id"].max()) + 1
    n_feat = len(WINDOWS) * 4 + 8
    X = np.zeros((n, n_feat), dtype=np.float32)
    names: list[str] = []
    # windowed aggregates: for window w, rows with seconds < w (i.e. last w seconds)
    for w in WINDOWS:
        seg = mkt.filter(pl.col("seconds_before_predict") < w)
        g = seg.group_by("sample_id").agg(
            pl.col("ofi").sum().alias("ofi_sum"),
            pl.col("imb").mean().alias("imb_mean"),
            pl.col("mid").std().alias("mid_std"),
            pl.col("mid").last().alias("mid_last"),
        )
        ids = g["sample_id"].to_numpy()
        X[ids, len(names)] = np.nan_to_num(g["ofi_sum"].to_numpy(), nan=0.0); names.append(f"m_ofi_sum_{w}")
        X[ids, len(names)] = np.nan_to_num(g["imb_mean"].to_numpy(), nan=0.0); names.append(f"m_imb_mean_{w}")
        X[ids, len(names)] = np.nan_to_num(g["mid_std"].to_numpy(), nan=0.0); names.append(f"m_mid_std_{w}")
        X[ids, len(names)] = np.nan_to_num(g["mid_last"].to_numpy(), nan=0.0); names.append(f"m_mid_last_{w}")
    # full-window (600s) forms
    g = mkt.group_by("sample_id").agg(
        pl.col("ofi").sum().alias("ofi"),
        pl.col("imb").mean().alias("imb"),
        pl.col("imb2").mean().alias("imb2"),
        pl.col("mid").std().alias("ms"),
        pl.col("mid").first().alias("mf"),
        pl.col("mid").last().alias("ml"),
        pl.col("spread1").mean().alias("sp1"),
        pl.col("spread2").mean().alias("sp2"),
        pl.col("mid").alias("series"),
        pl.col("transaction_volume").sum().alias("txv"),
    )
    ids = g["sample_id"].to_numpy()
    X[ids, len(names)] = np.nan_to_num(g["ofi"].to_numpy(), nan=0.0); names.append("m_ofi_sum_600")
    X[ids, len(names)] = np.nan_to_num(g["imb"].to_numpy(), nan=0.0); names.append("m_imb_mean_600")
    X[ids, len(names)] = np.nan_to_num(g["imb2"].to_numpy(), nan=0.0); names.append("m_imb2_mean_600")
    X[ids, len(names)] = np.nan_to_num(g["ms"].to_numpy(), nan=0.0); names.append("m_mid_std_600")
    slope = (g["ml"].to_numpy() - g["mf"].to_numpy()) / (g["mf"].to_numpy() + 1e-12)
    X[ids, len(names)] = np.nan_to_num(slope, nan=0.0); names.append("m_mid_slope_600")
    X[ids, len(names)] = np.nan_to_num(g["sp1"].to_numpy(), nan=0.0); names.append("m_spread1_mean_600")
    X[ids, len(names)] = np.nan_to_num(g["sp2"].to_numpy(), nan=0.0); names.append("m_spread2_mean_600")
    X[ids, len(names)] = np.nan_to_num(g["txv"].to_numpy(), nan=0.0); names.append("m_txvol_600")
    return X, names


def main() -> None:
    canonical = CanonicalOOF(**{
        k: np.asarray(np.load(CANONICAL)[k]) for k in
        ("sample_id", "month", "target", "baseline_oof", "source_train_end")
    })
    canonical.validate()
    print("building market forms...")
    X, names = build_market_forms()
    X = X[canonical.sample_id]
    print(f"X: {X.shape} ({len(names)} feats)")

    # overlap audit vs 152
    fe = pl.read_parquet(r"D:\mscapital-kaggle\scripts\kaggle_0726ds\f0726_train_f32.parquet").sort("sample_id")
    fids = fe["sample_id"].to_numpy()
    pos = np.searchsorted(fids, canonical.sample_id)
    ours = fe.select([c for c in fe.columns if c not in ("sample_id", "target")]).to_numpy()[pos].astype(np.float64)
    ours = np.nan_to_num(ours, nan=0.0)

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    print("\n=== overlap audit (new form vs 152) ===")
    # matrix corr: (n,152) x (n,36) -> corr per pair (standardized dot / n)
    Z_ours = (ours - ours.mean(axis=0)) / (ours.std(axis=0) + 1e-12)
    Z_v = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    corr_mat = (Z_ours.T @ Z_v) / len(X)  # (152, 36)
    maxcorr = np.abs(corr_mat).max(axis=0)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(X), 100_000, replace=False)
    ours_s = ours[sub]
    low_overlap = []
    for j, name in enumerate(names):
        v = X[:, j].astype(np.float64)
        c = float(maxcorr[j])
        if c > 0.95:
            r2 = float("nan")
            tag = "high"
        else:
            k = 100_000 // 5
            r2s = []
            for f in range(5):
                te = sub[f * k:(f + 1) * k] if f < 4 else sub[4 * k:]
                tr = np.setdiff1d(sub, te)
                sc = StandardScaler().fit(ours_s)
                m = Ridge(alpha=1.0).fit(sc.transform(ours[tr]), v[tr])
                r2s.append(1 - np.sum((v[te] - m.predict(sc.transform(ours[te]))) ** 2) / np.sum((v[te] - v[te].mean()) ** 2))
            r2 = float(np.mean(r2s))
            tag = "LOW" if r2 < 0.5 else ("mid" if r2 < 0.9 else "high")
        print(f"  {name:22s} maxcorr={c:.3f} ridgeR2={r2:.3f} [{tag}]")
        if r2 < 0.5:
            low_overlap.append((name, c, r2))

    # residual signal for low-overlap forms
    print(f"\n=== residual signal (low-overlap: {len(low_overlap)}) ===")
    r = canonical.target - 0.0004019 * canonical.baseline_oof
    res = []
    for name, c, r2 in low_overlap:
        v = X[:, names.index(name)].astype(np.float64)
        cc = np.corrcoef(v, r)[0, 1]
        res.append((name, c, r2, cc))
        print(f"  {name:22s} overlapR2={r2:.3f} residual_corr={cc:+.4f}")
    np.savez_compressed(OUT / "market_forms.npz", X=X, names=np.array(names),
                        low_overlap=np.array([(a, b, c) for a, b, c, _ in res], dtype=object) if res else None)
    print("\nwritten to", OUT)


if __name__ == "__main__":
    main()
