# -*- coding: utf-8 -*-
"""P4-01(a)-2 Distribution Test: do market path descriptors differ between
high-|d| and low-|d| samples?

Design (per GPT spec):
- Groups: Top/Bottom 5% and 10% by |d| (plus random control)
- Effect size: Cohen's d (standardized mean difference), direction
- Stability: 10 random folds -> effect sign consistency + fold stability
  (test has no month field; fold stability stands in for month stability)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl

from p4_01a_common import OUT, DESC_PATH, load_d


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if sp == 0:
        return 0.0
    return (a.mean() - b.mean()) / sp


def main() -> None:
    ids, d, ad = load_d()
    desc = pl.read_parquet(DESC_PATH).sort("sample_id")
    assert np.array_equal(desc["sample_id"].to_numpy(), ids), "id mismatch"

    names = [c for c in desc.columns if c != "sample_id"]
    X = desc.select(names).to_numpy().astype(np.float64)

    rng = np.random.default_rng(42)
    results = []
    for top in (0.05, 0.10):
        n = int(len(ad) * top)
        hi_idx = np.argsort(ad)[-n:]
        lo_idx = np.argsort(ad)[:n]
        hi, lo = ad[hi_idx], ad[lo_idx]
        print(f"\n=== Top/Bottom {top:.0%} |d| (n={n}) ===")
        print(f"|d| high: mean={hi.mean():.6f}  low: mean={lo.mean():.6f}")

        # per-feature effect size + stability across 10 random folds
        fold_effects = np.zeros((10, len(names)))
        fold_hi_mean, fold_lo_mean = np.zeros(10), np.zeros(10)
        idx_all = np.arange(len(ad))
        for f in range(10):
            fold_ids = rng.permutation(idx_all)[: n * 2]  # sample n high + n low
            fold_hi = fold_ids[ad[fold_ids] >= np.quantile(ad[fold_ids], 0.9)]
            fold_lo = fold_ids[ad[fold_ids] <= np.quantile(ad[fold_ids], 0.1)]
            # fall back to simple split if quantile edge cases
            if len(fold_hi) == 0 or len(fold_lo) == 0:
                fold_hi = fold_ids[:n]
                fold_lo = fold_ids[-n:]
            fold_hi_mean[f] = ad[fold_hi].mean()
            fold_lo_mean[f] = ad[fold_lo].mean()
            for j in range(len(names)):
                fold_effects[f, j] = cohens_d(X[fold_hi, j], X[fold_lo, j])

        for j in range(len(names)):
            e = cohens_d(X[hi_idx, j], X[lo_idx, j])
            fe = fold_effects[:, j]
            sign_consistent = np.mean(np.sign(fe) == np.sign(e)) if e != 0 else 0.0
            results.append({
                "top": top, "feature": names[j],
                "d_high_mean": X[hi_idx, j].mean(), "d_low_mean": X[lo_idx, j].mean(),
                "cohens_d": e,
                "fold_sign_consistency": sign_consistent,
                "fold_effect_std": fe.std(),
                "fold_effect_mean": fe.mean(),
            })

    rdf = pl.DataFrame(results)
    rdf = rdf.with_columns(
        (pl.col("cohens_d").abs()).alias("abs_d")
    ).sort(["top", "abs_d"], descending=[False, True])
    rdf.write_csv(OUT / "distribution_test.csv")
    print(f"\nsaved -> {OUT / 'distribution_test.csv'}")

    # summary: top discriminative features per threshold
    for top in (0.05, 0.10):
        sub = rdf.filter(pl.col("top") == top).head(15)
        print(f"\n--- Top 15 discriminative features ({top:.0%}) ---")
        for r in sub.iter_rows(named=True):
            print(f"  {r['feature']:<28s} d={r['cohens_d']:+.4f} "
                  f"fold_sign={r['fold_sign_consistency']:.2f} "
                  f"hi={r['d_high_mean']:.5f} lo={r['d_low_mean']:.5f}")
    # count of stable features
    for top in (0.05, 0.10):
        sub = rdf.filter(pl.col("top") == top)
        n_stable = sub.filter(pl.col("fold_sign_consistency") >= 0.90).height
        n_any = sub.filter(pl.col("fold_sign_consistency") >= 0.70).height
        print(f"\n[{top:.0%}] features with fold sign >=0.90: {n_stable}/{len(names)}; >=0.70: {n_any}/{len(names)}")


if __name__ == "__main__":
    main()
