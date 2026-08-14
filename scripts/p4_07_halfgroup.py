# -*- coding: utf-8 -*-
"""P4-07: 0.5-price latent group (secondary line).

Checks:
1. is_half_group flag stability: month coverage, std(y|g)/std(y|main) by month
2. group flag as residual feature: [is_half_group] (+28 long-context) via M01-A
3. group-specific scale: std ratio ~0.17 stable? -> normalization bucket?
4. test-side: LB142 vs v7 magnitude on the group
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import polars as pl

from mscapital.models.m01a import run_m01a_outer, summarize_m01a
from mscapital.residual import CanonicalOOF

from p3_common import save_p3_features, load_p3_frame

MARKET = Path(r"D:\mscapital-forecasting\data\raw\train\market.feather")
MARKET_TEST = Path(r"D:\mscapital-forecasting\data\raw\test\market.feather")
LABEL = Path(r"D:\mscapital-forecasting\data\raw\train\label.feather")
CANONICAL = Path(r"D:\mscapital-kaggle\output\canonical_residual_oof\canonical_residual_oof.npz")
BASELINE_ROOT = Path(r"D:\mscapital-kaggle\output\c4_protocol_closed_final\clean-baseline-v2")
REF = Path(r"D:\mscapital-forecasting\reference\lb142\submission_ref_lb0142.csv")
V7 = Path(r"D:\mscapital-kaggle\output\submissions\submission_blend_v7_rl20.csv")
OUT = Path(r"D:\mscapital-kaggle\output\p4_07_halfgroup")
FEATURE_OUT = Path(r"D:\mscapital-kaggle\output\p4_07_features")
OUT.mkdir(parents=True, exist_ok=True)
FEATURE_OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    canonical = CanonicalOOF(**{
        k: np.asarray(np.load(CANONICAL)[k]) for k in
        ("sample_id", "month", "target", "baseline_oof", "source_train_end")
    })
    canonical.validate()

    # is_half_group from market mid median
    mkt = pl.read_ipc(MARKET, columns=["sample_id", "ask_price_1", "bid_price_1"])
    med = mkt.with_columns(((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2).alias("mid")) \
        .group_by("sample_id").agg(pl.col("mid").median().alias("mid_med"))
    low_ids = set(med.filter(pl.col("mid_med") < 0.7)["sample_id"].to_numpy().tolist())
    flag = np.array([1 if int(s) in low_ids else 0 for s in canonical.sample_id], dtype=np.float32)
    print(f"is_half_group: {flag.sum()} / {len(flag)} ({flag.mean()*100:.3f}%)")

    # 1) std ratio by month (stability of the 0.17 scale)
    lab = pl.read_ipc(LABEL)
    y_lab = lab["target"].to_numpy()
    m_lab = lab["month"].to_numpy()
    sid_lab = lab["sample_id"].to_numpy()
    low_set = low_ids
    ratios = []
    for m in range(71):
        mm = m_lab == m
        if mm.sum() < 1000:
            continue
        lo = np.isin(sid_lab[mm], list(low_set))
        if lo.sum() < 10:
            continue
        r = y_lab[mm][lo].std() / y_lab[mm][~lo].std()
        ratios.append((m, float(r)))
    ratios = np.asarray(ratios)
    print(f"std-ratio by month: n={len(ratios)} mean={ratios[:, 1].mean():.3f} "
          f"std={ratios[:, 1].std():.3f} min={ratios[:, 1].min():.3f} max={ratios[:, 1].max():.3f}")

    # 2) group flag as residual feature (alone)
    for name, vals, exp_id in (
        ("flag", flag.reshape(-1, 1), "p4-07-flag"),
    ):
        fp = FEATURE_OUT / f"{name}_features.parquet"
        save_p3_features(fp, exp_id, ("is_half_group",),
                         canonical.sample_id, canonical.month, canonical.target, vals)
        frame = load_p3_frame(fp, ("is_half_group",))
        rdir = OUT / name
        for outer in ("PSEUDO", "H2", "T3", "T4"):
            diag = run_m01a_outer(canonical, frame, BASELINE_ROOT, rdir, outer)
            print(f"{name} {outer}: delta={diag['delta_vs_baseline']:+.9f}")
        summ = summarize_m01a(rdir)
        print(f"{name}: gate={summ['gate']['passed']} mean_delta={summ['mean_delta']:+.6f}")

    # 3) test-side LB142 vs v7 on the group
    mkt_te = pl.read_ipc(MARKET_TEST, columns=["sample_id", "ask_price_1", "bid_price_1"])
    med_te = mkt_te.with_columns(((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2).alias("mid")) \
        .group_by("sample_id").agg(pl.col("mid").median().alias("mid_med"))
    te_low = set(med_te.filter(pl.col("mid_med") < 0.7)["sample_id"].to_numpy().tolist())
    ref = pl.read_csv(REF).sort("sample_id")
    v7 = pl.read_csv(V7).sort("sample_id")
    ids = ref["sample_id"].to_numpy()
    te_flag = np.array([1 if int(s) in te_low else 0 for s in ids])
    p_ref = ref["prediction"].to_numpy()
    p_v7 = v7["prediction"].to_numpy()
    print(f"\ntest group: n={te_flag.sum()} ({te_flag.mean()*100:.2f}%)")
    for lbl, p in (("ref", p_ref), ("v7", p_v7)):
        print(f"  {lbl}: |pred| low={np.abs(p[te_flag == 1]).mean():.6f} "
              f"main={np.abs(p[te_flag == 0]).mean():.6f} ratio={np.abs(p[te_flag==1]).mean()/np.abs(p[te_flag==0]).mean():.3f}")
    print("\nwritten to", OUT)


if __name__ == "__main__":
    main()
