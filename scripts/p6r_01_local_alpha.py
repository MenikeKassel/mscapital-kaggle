# -*- coding: utf-8 -*-
"""P6R-01: Local Varying-Coefficient Ridge (Conditional Alpha test).

For each query, beta(z_q) is fit by a weighted ridge over the K nearest
historical states (strictly month < query month); the global ridge over the
same bank is the control. If Local does not beat Global, Conditional Alpha is
declared weak.

Preregistered candidates (docs/p6r_preregistration.md §4):
    K in {128, 256} x lambda in {0.01, 0.1} x {local, global}
Features: 40 preregistered core features from the f0726 schema (§7).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mscapital.features.revol_lite import context_feature_names
from mscapital.metrics import cosine_uncentered
from mscapital.residual import load_canonical_oof_artifact, outer_residual
from mscapital.retrieval import (
    StateKNN,
    bank_mask,
    blend_with_scales,
    fit_global_ridge,
    fit_local_ridge,
    fit_state_scaler,
    query_months,
    select_alpha,
    standardize,
)
from mscapital.models.revol_lite import load_revol_lite_frame

SEED = 2026
ALPHA_GRID = np.array([0.05, 0.10, 0.15, 0.20, 0.30])
K_GRID = (128, 256)
LAM_GRID = (0.01, 0.1)

OUTER_EVAL = {"PSEUDO": (33, 70), "H2": (51, 60), "T3": (51, 60), "T4": (61, 70)}
INNER_TRAIN = {"PSEUDO": (21, 26), "H2": (21, 30), "T3": (21, 40), "T4": (21, 40)}
TUNE = {"PSEUDO": (27, 32), "H2": (31, 40), "T3": (41, 50), "T4": (41, 50)}

CORE_FEATURES: tuple[str, ...] = (
    # OFI (6)
    "m_ofi_sum", "m_ofi_sum_60", "m_ofi_sum_180", "m_ofi_weighted_300",
    "m_ofi_ewm_120", "x_m_ofi_long_short_diff",
    # trade imbalance / buy-sell pressure (6)
    "t_buy_sell_vol_ratio", "t_avg_signed_vol", "t_avg_signed_vol_30",
    "t_large_buy_90", "t_large_sell_95", "x_large_trade_imbalance",
    # cancel pressure (3)
    "o_sec_cancel_new_ratio", "o_sec_cancel_new_ratio_30", "o_sec_cancel_volume",
    # aggressive flow (3)
    "o_market_ratio", "o_sv_30", "t_sv_weighted_15",
    # order/trade intensity (5)
    "o_vol_sum", "o_sec_new_count", "t_lv_mean", "t_lv_mean_30", "t_sec_vol_mean",
    # spread / depth / imbalance (7)
    "m_sp_mean_60", "x_sec_cancel_spread", "o_bid_depth", "o_ask_depth",
    "m_vol_weighted_60", "m_imb_last", "m_imb_mean_60",
    # microprice / price pressure (5)
    "x_vwap_mid_ratio", "m_mid_last", "m_mid_std", "m_rv", "m_rv_60",
    # execution / conversion (2)
    "x_trans_order_vol_ratio", "x_tx_order_rate_ratio",
    # velocity / pressure-over-depth (3)
    "t_avg_time_gap", "x_t_vol_weight_ratio_30", "x_o_vol_weight_ratio_15",
)


def take_features(features_path: Path, sample_id: np.ndarray) -> np.ndarray:
    """40 core feature columns for the requested sample ids, row-aligned.

    NaN/inf policy (preregistered): NaN is imputed per feature with the
    inner-train mean after standardization (NaN -> 0 in z-space); inf is
    clipped to the finite range of the inner-train months.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(features_path, columns=["sample_id"] + list(CORE_FEATURES))
    ids = table["sample_id"].to_numpy()
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    positions = np.searchsorted(sorted_ids, np.asarray(sample_id))
    if np.any(positions >= sorted_ids.size) or not np.array_equal(
        sorted_ids[positions], np.asarray(sample_id)
    ):
        raise ValueError("f0726 feature artifact does not cover canonical sample IDs")
    values = np.column_stack([table[name].to_numpy() for name in CORE_FEATURES]).astype(np.float64)
    return values[order][positions]


def take_context(frame, sample_id: np.ndarray) -> np.ndarray:
    requested = np.asarray(sample_id).reshape(-1)
    order = np.argsort(frame.sample_id)
    ids = frame.sample_id[order]
    positions = np.searchsorted(ids, requested)
    if np.any(positions >= ids.size) or not np.array_equal(ids[positions], requested):
        raise ValueError("ReVol-lite feature artifact does not cover canonical sample IDs")
    cols = [frame.feature_names.index(name) for name in context_feature_names()]
    return frame.values[order[positions]][:, cols].astype(np.float64)


def load_c4_baseline(root: Path, outer: str) -> dict[str, np.ndarray]:
    directory = root / outer
    with np.load(directory / "predictions.npz") as source:
        return {key: np.asarray(source[key]) for key in ("sample_id", "month", "target", "pred")}


def align_ids(ids_src: np.ndarray, ids_dst: np.ndarray) -> np.ndarray:
    order = np.argsort(ids_dst, kind="stable")
    sorted_dst = ids_dst[order]
    positions = np.searchsorted(sorted_dst, ids_src)
    if np.any(positions >= sorted_dst.size) or not np.array_equal(
        sorted_dst[positions], ids_src
    ):
        raise ValueError("alignment failed: source ids not fully covered by destination")
    return order[positions]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def beta_diagnostics(
    betas: np.ndarray,
    z_q: np.ndarray,
    month_q: np.ndarray,
    pc1: np.ndarray,
) -> dict[str, Any]:
    """Per-feature aggregate beta diagnostics over the query set."""
    d = betas.shape[1]
    rows = []
    for j in range(d):
        b = betas[:, j]
        # sign stability across months (same-sign fraction of monthly means)
        monthly_means = np.array(
            [b[month_q == m].mean() for m in sorted(np.unique(month_q).tolist())]
        )
        sign_stability = float(np.mean(np.sign(monthly_means) == np.sign(np.median(monthly_means))))
        reversal_freq = 1.0 - sign_stability
        # beta vs context dims
        corr_z = [pearson(b, z_q[:, c]) for c in range(z_q.shape[1])]
        # beta by state bucket (PC1 quartiles)
        edges = np.quantile(pc1, [0.25, 0.5, 0.75])
        buckets = np.digitize(pc1, edges)
        beta_by_state = [float(b[buckets == qq].mean()) if (buckets == qq).any() else None for qq in range(4)]
        rows.append(
            {
                "feature": CORE_FEATURES[j],
                "beta_mean": float(b.mean()),
                "beta_std": float(b.std()),
                "beta_p10": float(np.quantile(b, 0.10)),
                "beta_p90": float(np.quantile(b, 0.90)),
                "sign_stability": sign_stability,
                "sign_reversal_freq": reversal_freq,
                "abs_beta_mean": float(np.abs(b).mean()),
                "corr_z_" + str(0): corr_z[0],
                "corr_z_mean": float(np.mean(corr_z)),
                "corr_z_max_abs": float(np.max(np.abs(corr_z))),
                "beta_state_q0": beta_by_state[0],
                "beta_state_q1": beta_by_state[1],
                "beta_state_q2": beta_by_state[2],
                "beta_state_q3": beta_by_state[3],
            }
        )
    return rows


def run_outer(
    canonical,
    z_std: np.ndarray,
    x_std: np.ndarray,
    r_resid: np.ndarray,
    pc1: np.ndarray,
    c4: dict[str, np.ndarray],
    c4_idx: np.ndarray,
    outer: str,
    out_root: Path,
) -> dict[str, Any]:
    months = np.asarray(canonical.month)
    target = np.asarray(canonical.target, dtype=np.float64)
    baseline = np.asarray(canonical.baseline_oof, dtype=np.float64)
    n = months.size
    eval_lo, eval_hi = OUTER_EVAL[outer]
    tune_lo, tune_hi = TUNE[outer]
    tune_mask = (months >= tune_lo) & (months <= tune_hi)
    fr_mask = (months >= eval_lo) & (months <= eval_hi)
    q_months = query_months(months, tune_lo)

    results: dict[str, Any] = {}
    pred_cols: dict[str, Any] = {"sample_id": canonical.sample_id, "month": months}
    all_beta_rows: dict[str, list[dict[str, Any]]] = {}

    for lam in LAM_GRID:
        for k in K_GRID:
            for kind in ("local", "global"):
                key = f"{kind}_K{k}_lam{lam}"
                t0 = time.perf_counter()
                r_hat = np.full(n, np.nan)
                betas_all = np.full((n, x_std.shape[1]), np.nan)
                for m in q_months:
                    qmask = months == m
                    bmask = bank_mask(months, m)
                    z_bank, x_bank = z_std[bmask], x_std[bmask]
                    r_bank = r_resid[bmask]
                    z_q, x_q = z_std[qmask], x_std[qmask]
                    if kind == "local":
                        betas = fit_local_ridge(z_q, z_bank, x_bank, r_bank, k, lam)
                        r_hat[qmask] = (x_q * betas).sum(axis=1)
                    else:
                        beta_g = fit_global_ridge(x_bank, r_bank, lam)
                        r_hat[qmask] = x_q @ beta_g
                        betas = np.tile(beta_g, (z_q.shape[0], 1))
                    betas_all[qmask] = betas
                sel = select_alpha(baseline[tune_mask], r_hat[tune_mask], target[tune_mask], ALPHA_GRID)
                y_hat_c4 = blend_with_scales(c4["pred"], r_hat[c4_idx], sel["baseline_rms"], sel["residual_rms"], sel["alpha"])
                base_c4 = cosine_uncentered(c4["pred"], c4["target"])
                final_c4 = cosine_uncentered(y_hat_c4, c4["target"])
                y_hat_canon = blend_with_scales(baseline[fr_mask], r_hat[fr_mask], sel["baseline_rms"], sel["residual_rms"], sel["alpha"])
                base_canon = cosine_uncentered(baseline[fr_mask], target[fr_mask])
                final_canon = cosine_uncentered(y_hat_canon, target[fr_mask])
                r_fr = r_resid[fr_mask]
                rhat_fr = r_hat[fr_mask]
                nmse = 1.0 - float(np.mean((rhat_fr - r_fr) ** 2)) / max(float(np.mean(r_fr**2)), 1e-30)
                metrics = {
                    "outer": outer, "kind": kind, "k": k, "lam": lam,
                    "alpha": sel["alpha"], "tune_score": sel["score"],
                    "tune_baseline_score": sel["baseline_score"],
                    "baseline_c4": base_c4, "final_c4": final_c4, "delta_c4": final_c4 - base_c4,
                    "baseline_canon": base_canon, "final_canon": final_canon, "delta_canon": final_canon - base_canon,
                    "corr_rhat_y": pearson(rhat_fr, target[fr_mask]),
                    "corr_rhat_r": pearson(rhat_fr, r_fr),
                    "corr_rhat_y0": pearson(rhat_fr, baseline[fr_mask]),
                    "normalized_mse": nmse,
                    "runtime_seconds": time.perf_counter() - t0,
                }
                results[key] = metrics
                pred_cols[f"r_hat_{key}"] = r_hat
                frozen_betas = betas_all[fr_mask]
                z_fr = z_std[fr_mask]
                m_fr = months[fr_mask]
                pc_fr = pc1[fr_mask]
                all_beta_rows[key] = beta_diagnostics(frozen_betas, z_fr, m_fr, pc_fr)
                print(f"    {outer} {key}: tune_alpha={sel['alpha']:.2f} delta_c4={metrics['delta_c4']:+.6f} "
                      f"delta_canon={metrics['delta_canon']:+.6f} corr_rhat_r={metrics['corr_rhat_r']:+.4f} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)

    # persist
    (out_root / outer).mkdir(parents=True, exist_ok=True)
    pl.DataFrame(pred_cols).write_parquet(out_root / outer / "predictions.parquet")
    for key, rows in all_beta_rows.items():
        pl.DataFrame(rows).write_csv(out_root / outer / f"beta_{key}.csv")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="P6R-01 Local Varying-Coefficient Ridge")
    parser.add_argument("--canonical", type=Path, default=Path("output/canonical_residual_oof/canonical_residual_oof.npz"))
    parser.add_argument("--features-context", type=Path, default=Path("output/e01_revol_lite_features/revol_lite_train.parquet"))
    parser.add_argument("--features-152", type=Path, default=Path(r"D:\mscapital-forecasting\data\processed\f0726_train.parquet"))
    parser.add_argument("--c4-root", type=Path, default=Path("output/c4_protocol_closed_final/clean-baseline-v2"))
    parser.add_argument("--output-root", type=Path, default=Path("output/p6r_01"))
    parser.add_argument("--outer", default="all", help="single outer name or all")
    args = parser.parse_args()

    t_start = time.perf_counter()
    canonical = load_canonical_oof_artifact(args.canonical)
    frame = load_revol_lite_frame(args.features_context)
    z_all = take_context(frame, canonical.sample_id)
    x_all = take_features(args.features_152, canonical.sample_id)
    months = np.asarray(canonical.month)
    target = np.asarray(canonical.target, dtype=np.float64)
    baseline = np.asarray(canonical.baseline_oof, dtype=np.float64)

    outers = ("PSEUDO", "H2", "T3", "T4") if args.outer == "all" else (args.outer,)
    fold_rows = []
    all_results: dict[str, Any] = {"beta": {}}
    for outer in outers:
        print(f"[P6R-01] outer={outer} start", flush=True)
        view = outer_residual(canonical, outer)
        beta = float(view["beta"])
        all_results["beta"][outer] = beta
        r_resid = target - beta * baseline
        it_lo, it_hi = INNER_TRAIN[outer]
        it_mask = (months >= it_lo) & (months <= it_hi)
        mu_z, sd_z = fit_state_scaler(z_all[it_mask])
        z_std = standardize(z_all, mu_z, sd_z)
        # X NaN/inf policy (preregistered): impute NaN with inner-train column
        # means, clip inf to inner-train finite range, then standardize.
        x_col_mean = np.nanmean(x_all[it_mask], axis=0)
        x_imp = np.where(np.isnan(x_all), x_col_mean, x_all)
        x_lo = np.nanmin(x_imp[it_mask], axis=0)
        x_hi = np.nanmax(x_imp[it_mask], axis=0)
        x_imp = np.clip(x_imp, x_lo, x_hi)
        mu_x, sd_x = fit_state_scaler(x_imp[it_mask])
        x_std = standardize(x_imp, mu_x, sd_x)
        if not np.isfinite(x_std).all():
            raise ValueError(f"{outer}: X standardization produced non-finite values")
        # PC1 of the state space for beta-by-state bucketing (fit on inner-train)
        cov = np.cov(z_std[it_mask].T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        pc1 = z_std @ eigvecs[:, -1]
        c4 = load_c4_baseline(args.c4_root, outer)
        c4_idx = align_ids(c4["sample_id"], canonical.sample_id)
        if not np.array_equal(c4["month"], months[c4_idx]) or not np.array_equal(c4["target"], target[c4_idx]):
            raise ValueError(f"{outer}: c4 frozen baseline rows do not align with canonical OOF")
        results = run_outer(canonical, z_std, x_std, r_resid, pc1, c4, c4_idx, outer, args.output_root)
        all_results[outer] = results
        for key, mets in results.items():
            fold_rows.append(mets)

    pl.DataFrame(fold_rows).write_csv(args.output_root / "fold_metrics.csv")
    config = {
        "experiment": "p6r-01-local-varying-coefficient-ridge",
        "preregistration": "docs/p6r_preregistration.md",
        "K": list(K_GRID), "lambda": list(LAM_GRID), "kinds": ["local", "global"],
        "features": list(CORE_FEATURES),
        "alpha_grid": list(map(float, ALPHA_GRID)),
        "seed": SEED,
    }
    (args.output_root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output_root / "metrics.json").write_text(json.dumps(all_results, indent=2, default=float), encoding="utf-8")
    print(f"[P6R-01] total {time.perf_counter() - t_start:.0f}s -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
