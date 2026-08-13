"""Monthly and state-conditioned stability audit for candidate predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .features.revol_lite import context_feature_names
from .metrics import cosine_uncentered
from .models.revol_lite import load_revol_lite_frame


def _cosine_sums(dot: object, pred_sq: object, target_sq: object) -> np.ndarray:
    dot_a, pred_a, target_a = (np.asarray(value, dtype=float) for value in (dot, pred_sq, target_sq))
    denom = np.sqrt(np.maximum(pred_a, 0.0) * np.maximum(target_a, 0.0))
    return np.divide(dot_a, denom, out=np.zeros_like(dot_a), where=denom != 0.0)


def _monthly_summary(pred: np.ndarray, baseline: np.ndarray, target: np.ndarray, month: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for value in sorted(np.unique(month).tolist()):
        mask = month == value
        y, p, b = target[mask], pred[mask], baseline[mask]
        rows.append({
            "month": int(value), "rows": int(mask.sum()),
            "baseline_cosine": cosine_uncentered(b, y), "candidate_cosine": cosine_uncentered(p, y),
            "delta": cosine_uncentered(p, y) - cosine_uncentered(b, y),
            "candidate_dot": float(np.dot(p, y)), "candidate_sq": float(np.dot(p, p)),
            "baseline_dot": float(np.dot(b, y)), "baseline_sq": float(np.dot(b, b)),
            "target_sq": float(np.dot(y, y)),
        })
    return rows


def _bootstrap_delta(rows: list[dict[str, Any]], n_bootstrap: int = 5000, seed: int = 2026) -> dict[str, float]:
    if not rows:
        return {"mean": 0.0, "lower_95": 0.0, "upper_95": 0.0, "n_bootstrap": n_bootstrap, "seed": seed}
    rng = np.random.default_rng(seed)
    n = len(rows)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    candidate = _cosine_sums(
        np.asarray([r["candidate_dot"] for r in rows])[indices].sum(1),
        np.asarray([r["candidate_sq"] for r in rows])[indices].sum(1),
        np.asarray([r["target_sq"] for r in rows])[indices].sum(1),
    )
    baseline = _cosine_sums(
        np.asarray([r["baseline_dot"] for r in rows])[indices].sum(1),
        np.asarray([r["baseline_sq"] for r in rows])[indices].sum(1),
        np.asarray([r["target_sq"] for r in rows])[indices].sum(1),
    )
    delta = candidate - baseline
    return {"mean": float(delta.mean()), "lower_95": float(np.quantile(delta, .025)), "upper_95": float(np.quantile(delta, .975)), "n_bootstrap": n_bootstrap, "seed": seed}


def _rows_for_ids(frame: Any, ids: np.ndarray) -> np.ndarray:
    order = np.argsort(frame.sample_id)
    sorted_ids = frame.sample_id[order]
    requested = np.asarray(ids).reshape(-1)
    positions = np.searchsorted(sorted_ids, requested)
    if np.any(positions >= sorted_ids.size) or not np.array_equal(sorted_ids[positions], requested):
        raise ValueError("feature artifact does not cover requested sample IDs")
    return order[positions]


def _state_summary(frame: Any, inner_ids: np.ndarray, outer_ids: np.ndarray, pred: np.ndarray, baseline: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    inner_rows, outer_rows = _rows_for_ids(frame, inner_ids), _rows_for_ids(frame, outer_ids)
    result: dict[str, Any] = {}
    for name in context_feature_names():
        index = frame.feature_names.index(name)
        edges = np.quantile(frame.values[inner_rows, index].astype(float), [.25, .50, .75])
        bins = np.digitize(frame.values[outer_rows, index].astype(float), edges)
        bucket_rows = []
        for bucket in range(4):
            mask = bins == bucket
            if not mask.any():
                bucket_rows.append({"quartile": bucket, "rows": 0, "delta": None})
                continue
            candidate, base = cosine_uncentered(pred[mask], target[mask]), cosine_uncentered(baseline[mask], target[mask])
            bucket_rows.append({"quartile": bucket, "rows": int(mask.sum()), "candidate_cosine": candidate, "baseline_cosine": base, "delta": candidate - base})
        result[name] = {"edges": edges.tolist(), "buckets": bucket_rows}
    return result


def audit_candidate_stability(artifact_root: str | Path, features_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    root = Path(artifact_root) / "e01-revol-lite"
    outers = ("PSEUDO", "H2", "T3", "T4")
    artifacts, manifests = {}, {}
    for outer in outers:
        directory = root / outer
        manifests[outer] = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        with np.load(directory / "predictions.npz") as source:
            artifacts[outer] = {key: np.asarray(source[key]) for key in source.files}
    outer_rows = []
    for outer in outers:
        data = artifacts[outer]
        base, candidate = cosine_uncentered(data["baseline_pred"], data["target"]), cosine_uncentered(data["pred"], data["target"])
        outer_rows.append({"outer": outer, "baseline_score": base, "candidate_score": candidate, "delta": candidate - base})
    pseudo = artifacts["PSEUDO"]
    pseudo_mask = (pseudo["month"] >= 33) & (pseudo["month"] <= 70)
    if not pseudo_mask.any():
        raise ValueError("PSEUDO candidate artifact has no month 33-70 rows for the primary stability audit")
    monthly = _monthly_summary(
        pseudo["pred"][pseudo_mask], pseudo["baseline_pred"][pseudo_mask],
        pseudo["target"][pseudo_mask], pseudo["month"][pseudo_mask],
    )
    deltas = np.asarray([row["delta"] for row in monthly], dtype=float)
    positive = deltas[deltas > 0.0]
    concentration = 1.0 if positive.sum() <= 0.0 else float(np.sort(positive)[-3:].sum() / positive.sum())
    slope = 0.0 if len(monthly) < 2 else float(np.polyfit([r["month"] for r in monthly], deltas, 1)[0])
    frame = load_revol_lite_frame(features_path)
    with np.load(root / "PSEUDO" / "inner_predictions.npz") as source:
        inner_ids = np.asarray(source["sample_id"])
    stability = {
        "positive_month_ratio": float(np.mean(deltas > 0.0)), "worst_month_delta": float(deltas.min()),
        "median_month_delta": float(np.median(deltas)), "mean_month_delta": float(deltas.mean()),
        "delta_slope_per_month": slope, "top3_positive_month_concentration": concentration,
        "bootstrap": _bootstrap_delta(monthly),
    }
    basic = {
        "pseudo_delta_at_least_0_0015": bool(outer_rows[0]["delta"] >= .0015),
        "positive_outers": int(sum(row["delta"] > 0 for row in outer_rows)),
        "worst_delta": float(min(row["delta"] for row in outer_rows)),
        "finite_ok": bool(all(np.isfinite(data["pred"]).all() for data in artifacts.values())),
        "drift_ok": bool(all(.67 <= manifests[o].get("diagnostics", {}).get("drift", {}).get("std_test_over_valid", 0.0) <= 1.50 and .50 <= manifests[o].get("diagnostics", {}).get("drift", {}).get("abs_p99_test_over_valid", 0.0) <= 2.00 for o in outers)),
    }
    bootstrap = stability["bootstrap"]
    gate = {**basic, "bootstrap_lower_95_gt_0": bool(bootstrap["lower_95"] > 0), "positive_month_ratio_at_least_0_50": bool(stability["positive_month_ratio"] >= .50), "top3_concentration_at_most_0_50": bool(concentration <= .50)}
    gate["positive_outers_at_least_3"] = bool(basic["positive_outers"] >= 3)
    gate["worst_delta_at_least_minus_0_0005"] = bool(basic["worst_delta"] >= -0.0005)
    gate["passed"] = bool(
        gate["pseudo_delta_at_least_0_0015"]
        and gate["positive_outers_at_least_3"]
        and gate["worst_delta_at_least_minus_0_0005"]
        and gate["finite_ok"]
        and gate["drift_ok"]
        and gate["bootstrap_lower_95_gt_0"]
        and gate["positive_month_ratio_at_least_0_50"]
        and gate["top3_concentration_at_most_0_50"]
    )
    result = {"candidate": "e01-revol-lite", "outer": outer_rows, "monthly": monthly, "stability": stability, "state": _state_summary(frame, inner_ids, pseudo["sample_id"], pseudo["pred"], pseudo["baseline_pred"], pseudo["target"]), "gate": gate}
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "monthly_stats.npz",
        month=np.asarray([row["month"] for row in monthly], dtype=np.int16),
        delta=np.asarray([row["delta"] for row in monthly], dtype=np.float64),
        candidate_dot=np.asarray([row["candidate_dot"] for row in monthly], dtype=np.float64),
        candidate_sq=np.asarray([row["candidate_sq"] for row in monthly], dtype=np.float64),
        baseline_dot=np.asarray([row["baseline_dot"] for row in monthly], dtype=np.float64),
        baseline_sq=np.asarray([row["baseline_sq"] for row in monthly], dtype=np.float64),
        target_sq=np.asarray([row["target_sq"] for row in monthly], dtype=np.float64),
    )
    (output / "manifest.json").write_text(json.dumps({
        "experiment_id": "e03-candidate-stability",
        "status": "complete",
        "candidate": "e01-revol-lite",
        "primary_months": [33, 70],
        "bootstrap": {"n_bootstrap": 5000, "seed": 2026},
        "context_feature_count": len(context_feature_names()),
        "gate": gate,
    }, indent=2), encoding="utf-8")
    (output / "stability.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "report.md").write_text("\n".join(["# E03 Stability audit - E01 ReVol-lite", "", f"- mean monthly delta: `{stability['mean_month_delta']:+.9f}`", f"- positive month ratio: `{stability['positive_month_ratio']:.3f}`", f"- bootstrap 95% CI: `[{bootstrap['lower_95']:+.9f}, {bootstrap['upper_95']:+.9f}]`", f"- top-3 positive concentration: `{concentration:.3f}`", "", f"- final combined gate: **{gate['passed']}**", ""]), encoding="utf-8")
    return result
