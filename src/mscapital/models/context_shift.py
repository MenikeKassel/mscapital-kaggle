"""E02 Reconditionor-lite residual/context diagnostic."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ..features.revol_lite import context_feature_names
from ..metrics import cosine_uncentered
from ..residual import CanonicalOOF, outer_residual
from ..splits import MonthRange
from .revol_lite import load_revol_lite_frame


CONTEXT_FOLDS: tuple[tuple[str, MonthRange, MonthRange], ...] = (
    ("C31_40", MonthRange(21, 30), MonthRange(31, 40)),
    ("C41_50", MonthRange(21, 40), MonthRange(41, 50)),
    ("C51_60", MonthRange(21, 50), MonthRange(51, 60)),
    ("C61_70", MonthRange(21, 60), MonthRange(61, 70)),
)


def _take_context(frame: Any, sample_id: np.ndarray) -> np.ndarray:
    requested = np.asarray(sample_id).reshape(-1)
    order = np.argsort(frame.sample_id)
    ids = frame.sample_id[order]
    positions = np.searchsorted(ids, requested)
    if np.any(positions >= ids.size) or not np.array_equal(ids[positions], requested):
        raise ValueError("ReVol-lite feature artifact does not cover canonical sample IDs")
    context_indices = [frame.feature_names.index(name) for name in context_feature_names()]
    return frame.values[order[positions]][:, context_indices].astype(np.float64)


def _fit_beta(pred: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.dot(pred, pred))
    return 0.0 if denominator == 0.0 else float(np.dot(pred, target) / denominator)


def _monthly_stats(pred: np.ndarray, target: np.ndarray, month: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for value in sorted(np.unique(month).tolist()):
        mask = month == value
        rows.append({
            "month": int(value), "dot": float(np.dot(pred[mask], target[mask])),
            "pred_sq": float(np.dot(pred[mask], pred[mask])), "target_sq": float(np.dot(target[mask], target[mask])),
        })
    return rows


def _bootstrap_cosine(rows: list[dict[str, float]], *, seed: int = 2026, n_bootstrap: int = 5000) -> dict[str, float]:
    if not rows:
        return {"lower_95": 0.0, "upper_95": 0.0, "mean": 0.0, "n_bootstrap": n_bootstrap, "seed": seed}
    rng = np.random.default_rng(seed)
    n = len(rows)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    dot = np.asarray([row["dot"] for row in rows])[indices].sum(axis=1)
    pred_sq = np.asarray([row["pred_sq"] for row in rows])[indices].sum(axis=1)
    target_sq = np.asarray([row["target_sq"] for row in rows])[indices].sum(axis=1)
    denominator = np.sqrt(np.maximum(pred_sq, 0.0) * np.maximum(target_sq, 0.0))
    scores = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator != 0.0)
    return {
        "lower_95": float(np.quantile(scores, 0.025)), "upper_95": float(np.quantile(scores, 0.975)),
        "mean": float(scores.mean()), "n_bootstrap": n_bootstrap, "seed": seed,
    }


def _context_bias(x_train: np.ndarray, x_valid: np.ndarray, actual: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, name in enumerate(context_feature_names()):
        edges = np.quantile(x_train[:, index], [0.25, 0.50, 0.75])
        buckets = np.digitize(x_valid[:, index], edges, right=False)
        rows = []
        for bucket in range(4):
            mask = buckets == bucket
            rows.append({
                "quartile": bucket, "rows": int(mask.sum()),
                "mean_actual_residual": None if not mask.any() else float(actual[mask].mean()),
                "mean_prediction": None if not mask.any() else float(pred[mask].mean()),
                "bias_actual_minus_prediction": None if not mask.any() else float((actual[mask] - pred[mask]).mean()),
            })
        result[name] = {"edges": edges.tolist(), "buckets": rows}
    return result


def diagnose_context_shift(canonical: CanonicalOOF, features_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Run forward-only linear/nonlinear residual context diagnostics."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for E02 context diagnostics") from exc

    frame = load_revol_lite_frame(features_path)
    canonical.validate()
    frame_rows = _take_context(frame, canonical.sample_id)
    months = np.asarray(canonical.month)
    target = np.asarray(canonical.target, dtype=np.float64)
    baseline = np.asarray(canonical.baseline_oof, dtype=np.float64)
    fold_results: list[dict[str, Any]] = []
    fold_arrays: list[dict[str, np.ndarray]] = []
    pooled: dict[str, list[np.ndarray]] = {"ridge": [], "histgb": [], "actual": [], "month": []}
    for name, train_range, valid_range in CONTEXT_FOLDS:
        train_mask = train_range.contains(months)
        valid_mask = valid_range.contains(months)
        if not train_mask.any() or not valid_mask.any() or np.any(train_mask & valid_mask):
            raise ValueError(f"{name}: invalid context diagnostic split")
        beta = _fit_beta(baseline[train_mask], target[train_mask])
        train_residual = target[train_mask] - beta * baseline[train_mask]
        valid_residual = target[valid_mask] - beta * baseline[valid_mask]
        scaler = StandardScaler().fit(frame_rows[train_mask])
        x_train = scaler.transform(frame_rows[train_mask])
        x_valid = scaler.transform(frame_rows[valid_mask])
        ridge = Ridge(alpha=10.0).fit(x_train, train_residual)
        histgb = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=1000, l2_regularization=10.0, random_state=2026,
            early_stopping=False,
        ).fit(x_train, train_residual)
        predictions = {"ridge": ridge.predict(x_valid), "histgb": histgb.predict(x_valid)}
        fold_arrays.append({
            "sample_id": np.asarray(canonical.sample_id[valid_mask]),
            "month": np.asarray(months[valid_mask]),
            "actual_residual": np.asarray(valid_residual),
            "ridge_pred": np.asarray(predictions["ridge"]),
            "histgb_pred": np.asarray(predictions["histgb"]),
        })
        zero_mse = float(np.mean(valid_residual ** 2))
        fold_payload: dict[str, Any] = {"fold": name, "train": train_range.as_tuple(), "valid": valid_range.as_tuple(), "beta": beta, "models": {}}
        for model_name, prediction in predictions.items():
            prediction = np.asarray(prediction, dtype=np.float64)
            mse = float(np.mean((prediction - valid_residual) ** 2))
            fold_payload["models"][model_name] = {
                "residual_cosine": cosine_uncentered(prediction, valid_residual),
                "normalized_mse_improvement": 0.0 if zero_mse == 0.0 else 1.0 - mse / zero_mse,
                "prediction_rms": float(np.sqrt(np.mean(prediction ** 2))),
                "prediction_mean": float(prediction.mean()),
                "prediction_std": float(prediction.std()),
            }
            pooled[model_name].append(prediction)
        pooled["actual"].append(valid_residual)
        pooled["month"].append(months[valid_mask])
        fold_payload["prediction_correlation_ridge_histgb"] = float(np.corrcoef(predictions["ridge"], predictions["histgb"])[0, 1]) if np.std(predictions["ridge"]) and np.std(predictions["histgb"]) else 0.0
        fold_payload["context_bias_histgb"] = _context_bias(frame_rows[train_mask], frame_rows[valid_mask], valid_residual, predictions["histgb"])
        fold_results.append(fold_payload)
    pooled_result: dict[str, Any] = {}
    for model_name in ("ridge", "histgb"):
        prediction = np.concatenate(pooled[model_name])
        actual = np.concatenate(pooled["actual"])
        months_pooled = np.concatenate(pooled["month"])
        mse = float(np.mean((prediction - actual) ** 2))
        zero_mse = float(np.mean(actual ** 2))
        pooled_result[model_name] = {
            "residual_cosine": cosine_uncentered(prediction, actual),
            "normalized_mse_improvement": 0.0 if zero_mse == 0.0 else 1.0 - mse / zero_mse,
            "prediction_rms": float(np.sqrt(np.mean(prediction ** 2))),
            "prediction_correlation": float(np.corrcoef(prediction, actual)[0, 1]) if np.std(prediction) and np.std(actual) else 0.0,
            "bootstrap": _bootstrap_cosine(_monthly_stats(prediction, actual, months_pooled)),
        }
    hist_folds = np.asarray([row["models"]["histgb"]["residual_cosine"] for row in fold_results])
    gate = {
        "histgb_pooled_cosine_at_least_0_01": bool(pooled_result["histgb"]["residual_cosine"] >= 0.01),
        "histgb_bootstrap_lower_95_gt_0": bool(pooled_result["histgb"]["bootstrap"]["lower_95"] > 0.0),
        "positive_folds_at_least_3": bool((hist_folds > 0.0).sum() >= 3),
        "worst_fold_at_least_minus_0_002": bool(hist_folds.min() >= -0.002),
        "finite_ok": bool(
            all(
                np.isfinite(np.concatenate(values)).all()
                for values in pooled.values()
                if values
            )
        ),
    }
    gate["passed"] = bool(
        gate["histgb_pooled_cosine_at_least_0_01"]
        and gate["histgb_bootstrap_lower_95_gt_0"]
        and gate["positive_folds_at_least_3"]
        and gate["worst_fold_at_least_minus_0_002"]
        and gate["finite_ok"]
    )
    result = {"method": "E02 Reconditionor-lite", "context_features": list(context_feature_names()), "folds": fold_results, "pooled": pooled_result, "gate": gate}
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    fold_dir = output / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for payload, arrays in zip(fold_results, fold_arrays):
        np.savez_compressed(fold_dir / f"{payload['fold']}.npz", **arrays)
    manifest = {
        "experiment_id": "e02-reconditionor-lite",
        "status": "complete",
        "protocol": "protocol-v2",
        "feature_count": len(context_feature_names()),
        "context_features": list(context_feature_names()),
        "folds": [payload["fold"] for payload in fold_results],
        "config_hash": hashlib.sha256(json.dumps({"ridge_alpha": 10.0, "histgb": {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 1000, "l2_regularization": 10.0, "random_state": 2026, "early_stopping": False}}, sort_keys=True).encode()).hexdigest(),
        "canonical_rows": int(canonical.sample_id.size),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "context_shift.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "report.md").write_text("\n".join([
        "# E02 Reconditionor-lite context shift", "",
        f"- HistGB pooled residual cosine: `{pooled_result['histgb']['residual_cosine']:.9f}`",
        f"- HistGB bootstrap 95% CI: `[{pooled_result['histgb']['bootstrap']['lower_95']:+.9f}, {pooled_result['histgb']['bootstrap']['upper_95']:+.9f}]`",
        f"- gate passed: **{gate['passed']}**", "",
    ]), encoding="utf-8")
    return result
