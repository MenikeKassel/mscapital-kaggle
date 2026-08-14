# -*- coding: utf-8 -*-
"""P6R-00: Retrieval Residual Mean.

Strict temporal KNN over the E02 context state (11 features) predicting the
canonical Clean Baseline v2 residual; final blend follows the E01 convention
y_hat = RMS(y0) + alpha * RMS(r_hat) with alpha selected on tune months only.

Preregistered candidates (docs/p6r_preregistration.md §3):
    K in {64, 128, 256, 512} x metric in {euclidean, cosine}
No candidate is added after looking at frozen results.

Outputs (per brief §11): output/p6r_00/{config.json, metrics.json,
monthly_metrics.csv, fold_metrics.csv, diagnostics.json, predictions.parquet,
neighbor_diagnostics.parquet}
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
    fit_state_scaler,
    query_months,
    retrieve_residual_mean,
    select_alpha,
    standardize,
    support_score,
)
from mscapital.models.revol_lite import load_revol_lite_frame

SEED = 2026
ALPHA_GRID = np.array([0.05, 0.10, 0.15, 0.20, 0.30])
K_GRID = (64, 128, 256, 512)
METRICS = ("euclidean", "cosine")
REF_K = 256  # reference config for neighbour-month entropy diagnostics

OUTER_EVAL = {"PSEUDO": (33, 70), "H2": (51, 60), "T3": (51, 60), "T4": (61, 70)}
INNER_TRAIN = {"PSEUDO": (21, 26), "H2": (21, 30), "T3": (21, 40), "T4": (21, 40)}
TUNE = {"PSEUDO": (27, 32), "H2": (31, 40), "T3": (41, 50), "T4": (41, 50)}
VISIBLE_END = {"PSEUDO": 32, "H2": 40, "T3": 50, "T4": 50}


def take_context(frame, sample_id: np.ndarray) -> np.ndarray:
    """E02 context features (11) for the requested sample ids, row-aligned."""
    requested = np.asarray(sample_id).reshape(-1)
    order = np.argsort(frame.sample_id)
    ids = frame.sample_id[order]
    positions = np.searchsorted(ids, requested)
    if np.any(positions >= ids.size) or not np.array_equal(ids[positions], requested):
        raise ValueError("ReVol-lite feature artifact does not cover canonical sample IDs")
    cols = [frame.feature_names.index(name) for name in context_feature_names()]
    return frame.values[order[positions]][:, cols].astype(np.float64)


def load_c4_baseline(root: Path, outer: str) -> dict[str, np.ndarray]:
    """Frozen production baseline for one outer."""
    directory = root / outer
    with np.load(directory / "predictions.npz") as source:
        return {key: np.asarray(source[key]) for key in ("sample_id", "month", "target", "pred")}


def align_ids(ids_src: np.ndarray, ids_dst: np.ndarray) -> np.ndarray:
    """Positions in ids_dst (sorted) for each id in ids_src; asserts full coverage."""
    order = np.argsort(ids_dst, kind="stable")
    sorted_dst = ids_dst[order]
    positions = np.searchsorted(sorted_dst, ids_src)
    if np.any(positions >= sorted_dst.size) or not np.array_equal(
        sorted_dst[positions], ids_src
    ):
        raise ValueError("alignment failed: source ids not fully covered by destination")
    return order[positions]


def neighbor_month_stats_vectorized(months: np.ndarray, idx: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-query neighbour-month entropy and unique-month count (chunked bincount)."""
    n_labels = hi - lo + 1
    n = idx.shape[0]
    entropy = np.full(n, np.nan, dtype=np.float64)
    uniques = np.full(n, -1, dtype=np.int16)
    neighbor_months = months[idx].astype(np.int64)
    chunk = 8192
    for start in range(0, n, chunk):
        block = neighbor_months[start : start + chunk]
        n_c = block.shape[0]
        flat = (block - lo).reshape(-1)
        qids = np.repeat(np.arange(n_c), block.shape[1])
        counts = np.bincount(qids * n_labels + flat, minlength=n_c * n_labels).reshape(n_c, n_labels)
        p = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        nz = p > 0
        with np.errstate(divide="ignore"):
            logp = np.log(p)
        entropy[start : start + n_c] = -(p * np.where(nz, logp, 0.0)).sum(axis=1)
        uniques[start : start + n_c] = (counts > 0).sum(axis=1)
    return entropy, uniques


def monthly_deltas(month: np.ndarray, y0: np.ndarray, y_hat: np.ndarray, y: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for m in sorted(np.unique(month).tolist()):
        mask = month == m
        if mask.sum() < 50:
            continue
        b, c, t = y0[mask], y_hat[mask], y[mask]
        rows.append(
            {
                "month": int(m),
                "baseline_cosine": cosine_uncentered(b, t),
                "candidate_cosine": cosine_uncentered(c, t),
                "delta": cosine_uncentered(c, t) - cosine_uncentered(b, t),
                "baseline_dot": float(np.dot(b, t)),
                "baseline_sq": float(np.dot(b, b)),
                "candidate_dot": float(np.dot(c, t)),
                "candidate_sq": float(np.dot(c, c)),
                "target_sq": float(np.dot(t, t)),
            }
        )
    return rows


def bootstrap_delta(rows: list[dict[str, float]], n_bootstrap: int = 5000, seed: int = SEED) -> dict[str, float]:
    if not rows:
        return {"mean": 0.0, "lower_95": 0.0, "upper_95": 0.0}
    rng = np.random.default_rng(seed)
    n = len(rows)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    dot = np.asarray([r["candidate_dot"] for r in rows])[indices].sum(1)
    sq = np.asarray([r["candidate_sq"] for r in rows])[indices].sum(1)
    tsq = np.asarray([r["target_sq"] for r in rows])[indices].sum(1)
    cand = np.divide(dot, np.sqrt(np.maximum(sq, 0.0) * np.maximum(tsq, 0.0)), out=np.zeros_like(dot), where=tsq != 0.0)
    bdot = np.asarray([r["baseline_dot"] for r in rows])[indices].sum(1)
    bsq = np.asarray([r["baseline_sq"] for r in rows])[indices].sum(1)
    base = np.divide(bdot, np.sqrt(np.maximum(bsq, 0.0) * np.maximum(tsq, 0.0)), out=np.zeros_like(bdot), where=tsq != 0.0)
    delta = cand - base
    return {
        "mean": float(delta.mean()),
        "lower_95": float(np.quantile(delta, 0.025)),
        "upper_95": float(np.quantile(delta, 0.975)),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run_outer(
    canonical,
    z_std: np.ndarray,
    r_resid: np.ndarray,
    c4: dict[str, np.ndarray],
    c4_idx: np.ndarray,
    outer: str,
    out_root: Path,
    *,
    skip_query: bool = False,
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
    diag_frames: list[Any] = []

    for metric in METRICS:
        for k in K_GRID:
            key = f"{metric}_K{k}"
            t0 = time.perf_counter()
            if skip_query:
                table = pl.read_parquet(out_root / outer / "predictions.parquet")
                r_hat = np.asarray(table[f"r_hat_{key}"].to_numpy(), dtype=np.float64)
                kth = np.full(n, np.nan)
                meand = np.full(n, np.nan)
                entr = np.full(n, np.nan)
                nuniq = np.full(n, -1, dtype=np.int16)
                diag = pl.read_parquet(out_root / outer / "neighbor_diagnostics.parquet").filter(
                    (pl.col("metric") == metric) & (pl.col("k") == k)
                )
                if diag.height:
                    pos = np.searchsorted(np.sort(canonical.sample_id), diag["sample_id"].to_numpy())
                    kth[pos] = diag["kth_dist"].to_numpy()
                    meand[pos] = diag["mean_dist"].to_numpy()
                    if "entropy" in diag.columns:
                        entr[pos] = diag["entropy"].to_numpy()
                        nuniq[pos] = diag["n_unique"].to_numpy()
            else:
                r_hat = np.full(n, np.nan)
                kth = np.full(n, np.nan)
                meand = np.full(n, np.nan)
                entr = np.full(n, np.nan)
                nuniq = np.full(n, -1, dtype=np.int16)
                for m in q_months:
                    qmask = months == m
                    bmask = bank_mask(months, m)
                    knn = StateKNN(metric).fit(z_std[bmask], k)
                    idx, dists, w = knn.query(z_std[qmask], k)
                    r_hat[qmask] = retrieve_residual_mean(idx, w, r_resid)
                    kth[qmask] = dists[:, -1]
                    meand[qmask] = dists.mean(axis=1)
                    if k == REF_K:
                        e, u = neighbor_month_stats_vectorized(months, idx, 21, 70)
                        entr[qmask] = e
                        nuniq[qmask] = u
            sel = select_alpha(baseline[tune_mask], r_hat[tune_mask], target[tune_mask], ALPHA_GRID)
            y_hat_canon = blend_with_scales(baseline[fr_mask], r_hat[fr_mask], sel["baseline_rms"], sel["residual_rms"], sel["alpha"])
            base_canon = cosine_uncentered(baseline[fr_mask], target[fr_mask])
            final_canon = cosine_uncentered(y_hat_canon, target[fr_mask])
            r_hat_c4 = r_hat[c4_idx]
            y_hat_c4 = blend_with_scales(c4["pred"], r_hat_c4, sel["baseline_rms"], sel["residual_rms"], sel["alpha"])
            base_c4 = cosine_uncentered(c4["pred"], c4["target"])
            final_c4 = cosine_uncentered(y_hat_c4, c4["target"])
            r_fr = r_resid[fr_mask]
            y_fr, y0_fr = target[fr_mask], baseline[fr_mask]
            rhat_fr = r_hat[fr_mask]
            nmse = 1.0 - float(np.mean((rhat_fr - r_fr) ** 2)) / max(float(np.mean(r_fr**2)), 1e-30)
            monthly = monthly_deltas(c4["month"], c4["pred"], y_hat_c4, c4["target"])
            deltas = np.asarray([row["delta"] for row in monthly])
            positive = deltas[deltas > 0.0]
            concentration = 1.0 if positive.sum() <= 0.0 else float(np.sort(positive)[-3:].sum() / positive.sum())
            slope = 0.0 if len(monthly) < 2 else float(np.polyfit([r["month"] for r in monthly], deltas, 1)[0])
            d_ref = float(np.median(kth[tune_mask])) if np.isfinite(kth[tune_mask]).any() else 1.0
            support = support_score(kth, d_ref)
            metrics = {
                "outer": outer, "metric": metric, "k": k,
                "alpha": sel["alpha"], "alpha_scores": sel["scores"],
                "tune_score": sel["score"], "tune_baseline_score": sel["baseline_score"],
                "baseline_rms": sel["baseline_rms"], "residual_rms": sel["residual_rms"],
                "baseline_c4": base_c4, "final_c4": final_c4, "delta_c4": final_c4 - base_c4,
                "baseline_canon": base_canon, "final_canon": final_canon, "delta_canon": final_canon - base_canon,
                "corr_rhat_y": pearson(rhat_fr, y_fr),
                "corr_rhat_r": pearson(rhat_fr, r_fr),
                "corr_rhat_y0": pearson(rhat_fr, y0_fr),
                "normalized_mse": nmse,
                "positive_month_ratio": float(np.mean(deltas > 0.0)) if deltas.size else 0.0,
                "worst_month_delta": float(deltas.min()) if deltas.size else 0.0,
                "mean_month_delta": float(deltas.mean()) if deltas.size else 0.0,
                "delta_slope_per_month": slope,
                "top3_positive_concentration": concentration,
                "bootstrap": bootstrap_delta(monthly),
                "neighbor_kth_p50": float(np.nanmedian(kth[fr_mask])),
                "neighbor_kth_p90": float(np.nanquantile(kth[fr_mask], 0.90)),
                "neighbor_mean_dist_p50": float(np.nanmedian(meand[fr_mask])),
                "d_ref_support": d_ref,
                "runtime_seconds": time.perf_counter() - t0,
            }
            results[key] = metrics
            pred_cols[f"r_hat_{key}"] = r_hat
            pred_cols[f"y_hat_c4_{key}"] = np.full(n, np.nan)
            pred_cols[f"y_hat_c4_{key}"][c4_idx] = y_hat_c4
            if not skip_query:
                ok = np.isfinite(kth)
                diag_frames.append(
                    pl.DataFrame(
                        {
                            "sample_id": canonical.sample_id[ok],
                            "month": months[ok],
                            "metric": metric,
                            "k": k,
                            "mean_dist": meand[ok],
                            "kth_dist": kth[ok],
                            "entropy": entr[ok],
                            "n_unique": nuniq[ok],
                            "support": support[ok],
                        }
                    )
                )
            print(f"    {outer} {key}: tune_alpha={sel['alpha']:.2f} "
                  f"delta_c4={metrics['delta_c4']:+.6f} delta_canon={metrics['delta_canon']:+.6f} "
                  f"corr_rhat_r={metrics['corr_rhat_r']:+.4f} ({time.perf_counter()-t0:.0f}s)", flush=True)

    if not skip_query:
        # V2 variant (capped bank) for the tune-best config
        best_key = max(results, key=lambda key: results[key]["tune_score"])
        best = results[best_key]
        k, metric = best["k"], best["metric"]
        cap = VISIBLE_END[outer]
        r_hat_v2 = np.full(n, np.nan)
        for m in q_months:
            qmask = months == m
            bmask = bank_mask(months, m, cap_month=cap)
            if not bmask.any():
                continue
            knn = StateKNN(metric).fit(z_std[bmask], k)
            idx, dists, w = knn.query(z_std[qmask], k)
            r_hat_v2[qmask] = retrieve_residual_mean(idx, w, r_resid)
        sel_v2 = select_alpha(baseline[tune_mask], r_hat_v2[tune_mask], target[tune_mask], ALPHA_GRID)
        y_hat_v2 = blend_with_scales(c4["pred"], r_hat_v2[c4_idx], sel_v2["baseline_rms"], sel_v2["residual_rms"], sel_v2["alpha"])
        results["v2_capped"] = {
            "variant": f"capped_{cap}_bank_of_{best_key}",
            "delta_c4": cosine_uncentered(y_hat_v2, c4["target"]) - base_c4,
            "alpha": sel_v2["alpha"],
        }

        # beta=1 residual variant for the same config
        r1 = target - baseline
        r_hat_1 = np.full(n, np.nan)
        for m in q_months:
            qmask = months == m
            bmask = bank_mask(months, m)
            knn = StateKNN(metric).fit(z_std[bmask], k)
            idx, dists, w = knn.query(z_std[qmask], k)
            r_hat_1[qmask] = retrieve_residual_mean(idx, w, r1)
        sel_1 = select_alpha(baseline[tune_mask], r_hat_1[tune_mask], target[tune_mask], ALPHA_GRID)
        y_hat_1 = blend_with_scales(c4["pred"], r_hat_1[c4_idx], sel_1["baseline_rms"], sel_1["residual_rms"], sel_1["alpha"])
        results["beta1_variant"] = {
            "variant": f"residual_y_minus_y0_of_{best_key}",
            "delta_c4": cosine_uncentered(y_hat_1, c4["target"]) - base_c4,
            "alpha": sel_1["alpha"],
        }

    # persist
    (out_root / outer).mkdir(parents=True, exist_ok=True)
    if not skip_query:
        pl.DataFrame(pred_cols).write_parquet(out_root / outer / "predictions.parquet")
        pl.concat(diag_frames).write_parquet(out_root / outer / "neighbor_diagnostics.parquet")
    monthly_all = []
    for key, mets in results.items():
        if key in ("v2_capped", "beta1_variant"):
            continue
        monthly_all.extend(
            {"config": key, **row}
            for row in monthly_deltas(c4["month"], c4["pred"], pred_cols[f"y_hat_c4_{key}"][c4_idx], c4["target"])
        )
    pl.DataFrame(monthly_all).write_csv(out_root / outer / "monthly_metrics.csv")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="P6R-00 Retrieval Residual Mean")
    parser.add_argument("--canonical", type=Path, default=Path("output/canonical_residual_oof/canonical_residual_oof.npz"))
    parser.add_argument("--features", type=Path, default=Path("output/e01_revol_lite_features/revol_lite_train.parquet"))
    parser.add_argument("--c4-root", type=Path, default=Path("output/c4_protocol_closed_final/clean-baseline-v2"))
    parser.add_argument("--output-root", type=Path, default=Path("output/p6r_00"))
    parser.add_argument("--outer", default="all", help="single outer name or all")
    args = parser.parse_args()

    t_start = time.perf_counter()
    canonical = load_canonical_oof_artifact(args.canonical)
    frame = load_revol_lite_frame(args.features)
    z_all = take_context(frame, canonical.sample_id)
    months = np.asarray(canonical.month)
    target = np.asarray(canonical.target, dtype=np.float64)
    baseline = np.asarray(canonical.baseline_oof, dtype=np.float64)

    outers = ("PSEUDO", "H2", "T3", "T4") if args.outer == "all" else (args.outer,)
    fold_rows = []
    all_results: dict[str, Any] = {"beta": {}, "gate": {}}
    for outer in outers:
        print(f"[P6R-00] outer={outer} start", flush=True)
        view = outer_residual(canonical, outer)
        beta = float(view["beta"])
        all_results["beta"][outer] = beta
        r_resid = target - beta * baseline
        it_lo, it_hi = INNER_TRAIN[outer]
        it_mask = (months >= it_lo) & (months <= it_hi)
        mu, sd = fit_state_scaler(z_all[it_mask])
        z_std = standardize(z_all, mu, sd)
        c4 = load_c4_baseline(args.c4_root, outer)
        c4_idx = align_ids(c4["sample_id"], canonical.sample_id)
        if not np.array_equal(c4["month"], months[c4_idx]) or not np.array_equal(c4["target"], target[c4_idx]):
            raise ValueError(f"{outer}: c4 frozen baseline rows do not align with canonical OOF")
        skip = (args.output_root / outer / "predictions.parquet").exists()
        results = run_outer(canonical, z_std, r_resid, c4, c4_idx, outer, args.output_root, skip_query=skip)
        all_results[outer] = results
        for key, mets in results.items():
            if key in ("v2_capped", "beta1_variant"):
                continue
            fold_rows.append(mets)
        config_keys = [k for k in results if not k.startswith(("v2_", "beta1"))]
        best = max(config_keys, key=lambda k: results[k]["tune_score"])
        print(f"[P6R-00] outer={outer} done: tune-best={best} delta_c4={results[best]['delta_c4']:+.6f}", flush=True)

    pl.DataFrame(fold_rows).write_csv(args.output_root / "fold_metrics.csv")

    # continuation gate (preregistration §3)
    gate: dict[str, Any] = {}
    if "PSEUDO" in all_results:
        pseudo = all_results["PSEUDO"]
        configs = [k for k in pseudo if not k.startswith(("v2_", "beta1"))]
        gate_configs = [
            k for k in configs
            if pseudo[k]["k"] >= 128
            and pseudo[k]["corr_rhat_r"] > 0.0
            and pseudo[k]["bootstrap"]["lower_95"] > 0.0
            and pseudo[k]["alpha"] >= 0.10
        ]
        gate["candidate_configs"] = gate_configs
        gate["pseudo_delta_c4_positive"] = any(pseudo[k]["delta_c4"] > 0.0 for k in gate_configs)
        pos_outers = sum(
            1
            for outer in outers
            if any(all_results[outer][k]["delta_c4"] > 0.0 for k in gate_configs)
        )
        gate["positive_outers_at_least_3"] = pos_outers >= 3
        worst = min(
            (all_results[outer][k]["delta_c4"] for outer in outers for k in gate_configs),
            default=0.0,
        )
        gate["worst_fold_delta"] = worst
        gate["no_catastrophic_fold"] = worst > -0.0005
        if gate_configs:
            best_key = max(gate_configs, key=lambda k: pseudo[k]["tune_score"])
            metric = best_key.split("_K")[0]
            k_deltas = {k: pseudo[f"{metric}_K{k}"]["delta_c4"] for k in K_GRID}
            same_sign = sum(1 for d in k_deltas.values() if (d > 0.0) == (pseudo[best_key]["delta_c4"] > 0.0))
            gate["k_robustness"] = {"metric": metric, "deltas": k_deltas, "same_sign_count": same_sign, "ok": same_sign >= 3}
        else:
            gate["k_robustness"] = {"ok": False}
        gate["passed"] = bool(
            gate_configs
            and gate["pseudo_delta_c4_positive"]
            and gate["positive_outers_at_least_3"]
            and gate["no_catastrophic_fold"]
            and gate["k_robustness"]["ok"]
        )
    all_results["gate"] = gate
    all_results["runtime_seconds"] = time.perf_counter() - t_start
    config = {
        "experiment": "p6r-00-retrieval-residual-mean",
        "preregistration": "docs/p6r_preregistration.md",
        "state": list(context_feature_names()),
        "K": list(K_GRID),
        "metrics": list(METRICS),
        "alpha_grid": list(map(float, ALPHA_GRID)),
        "weights": "gaussian exp(-0.5 (d/d_K)^2)",
        "bank": "V1 expanding month<query (V2 capped variant for tune-best config)",
        "residual": "y - beta*y0 (beta per outer from visible view)",
        "seed": SEED,
        "canonical_rows": int(canonical.sample_id.size),
        "canonical_months": [int(months.min()), int(months.max())],
    }
    (args.output_root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output_root / "metrics.json").write_text(json.dumps(all_results, indent=2, default=float), encoding="utf-8")
    (args.output_root / "diagnostics.json").write_text(json.dumps({"gate": gate, "beta": all_results["beta"]}, indent=2, default=float), encoding="utf-8")
    print(f"[P6R-00] gate: {json.dumps(gate, indent=2, default=float)}", flush=True)
    print(f"[P6R-00] total {time.perf_counter() - t_start:.0f}s -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
