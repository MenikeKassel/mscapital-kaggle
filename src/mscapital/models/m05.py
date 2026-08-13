"""Historical-only market-state prototype KNN residual experiment."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import ExperimentManifest, array_hash, feature_hash
from ..diagnostics import drift_report, prediction_diagnostics
from ..metrics import cosine_uncentered
from ..residual import CanonicalOOF, outer_residual
from ..splits import NESTED_SPLITS, MonthRange
from .m01a import ALPHA_GRID, M01AConfig, select_alpha, _load_outer_baseline

STATE_FEATURE_NAMES = (
    "m_mid_std", "m_rv", "m_sp_mean_60", "m_imb_mean_60", "m_ofi_sum_60",
    "m_mid_range_60", "m_mid_std_180", "m_imb_mean_180", "m_ofi_sum_180",
    "t_buy_ratio_15", "t_px_std_15", "t_avg_time_gap", "o_buy_ratio_15",
    "o_market_ratio_15", "o_avg_time_gap", "x_trans_order_buy_diff_15",
)
K_GRID = (8, 16, 32, 64)
INNER_SPLITS = {
    "PSEUDO": (MonthRange(21, 26), MonthRange(27, 32)),
    "H2": (MonthRange(21, 30), MonthRange(31, 40)),
    "T3": (MonthRange(21, 40), MonthRange(41, 50)),
    "T4": (MonthRange(21, 40), MonthRange(41, 50)),
}


@dataclass(frozen=True)
class MarketStateFrame:
    sample_id: np.ndarray
    month: np.ndarray
    target: np.ndarray
    values: np.ndarray
    feature_names: tuple[str, ...] = STATE_FEATURE_NAMES

    def validate(self) -> None:
        n = self.sample_id.size
        if self.month.size != n or self.target.size != n or self.values.shape != (n, len(self.feature_names)):
            raise ValueError("Market-State arrays are not aligned")
        if np.unique(self.sample_id).size != n:
            raise ValueError("Market-State sample_id must be unique")
        if not np.isfinite(self.target).all() or np.isinf(self.values).any():
            raise ValueError("Market-State targets must be finite and state values cannot contain infinities")


def build_market_state_file(source_path: str | Path, labels_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    import pyarrow.parquet as pq
    import pyarrow as pa

    source = pq.read_table(source_path, columns=["sample_id", *STATE_FEATURE_NAMES])
    labels = pq.read_table(labels_path, columns=["sample_id", "month", "target"]) if str(labels_path).lower().endswith(".parquet") else None
    if labels is None:
        import pyarrow.feather as feather
        labels = feather.read_table(labels_path, columns=["sample_id", "month", "target"])
    src = {name: source[name].to_numpy(zero_copy_only=False) for name in source.column_names}
    lab = {name: labels[name].to_numpy(zero_copy_only=False) for name in labels.column_names}
    order = np.argsort(src["sample_id"], kind="mergesort")
    src = {name: np.asarray(value)[order] for name, value in src.items()}
    label_order = np.argsort(lab["sample_id"], kind="mergesort")
    lab = {name: np.asarray(value)[label_order] for name, value in lab.items()}
    if np.unique(lab["sample_id"]).size != lab["sample_id"].size:
        raise ValueError("Market-State labels contain duplicate sample_id values")
    pos = np.searchsorted(lab["sample_id"], src["sample_id"])
    if np.any(pos >= lab["sample_id"].size) or not np.array_equal(lab["sample_id"][pos], src["sample_id"]):
        raise ValueError("Market-State source and labels do not align")
    frame = MarketStateFrame(src["sample_id"].astype(np.int64), lab["month"][pos], lab["target"][pos].astype(np.float64), np.column_stack([src[name] for name in STATE_FEATURE_NAMES]).astype(np.float32))
    frame.validate()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"sample_id": frame.sample_id, "month": frame.month, "target": frame.target, **{name: frame.values[:, i] for i, name in enumerate(STATE_FEATURE_NAMES)}}), output, compression="zstd")
    diagnostics = {"rows": int(frame.sample_id.size), "feature_names": list(STATE_FEATURE_NAMES), "feature_hash": feature_hash(list(STATE_FEATURE_NAMES)), "artifact_hashes": {"sample_id": array_hash(frame.sample_id), "month": array_hash(frame.month), "target": array_hash(frame.target), "values": array_hash(frame.values)}}
    result_hash = hashlib.sha256(json.dumps(diagnostics, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    ExperimentManifest(experiment_id="m05-market-state-features", status="complete", config_hash=result_hash, feature_hash=diagnostics["feature_hash"], train_months=(0, int(frame.month.max())), diagnostics=diagnostics).write(output.parent)
    (output.parent / "report.md").write_text(f"# M05 Market-State\n\n- rows: `{frame.sample_id.size}`\n- features: `16`\n", encoding="utf-8")
    return diagnostics | {"output": str(output), "result_hash": result_hash}


def load_market_state_frame(path: str | Path) -> MarketStateFrame:
    import pyarrow.parquet as pq
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "m05-market-state-features" or manifest.get("status") != "complete":
        raise ValueError("M05 feature manifest identity/status is invalid")
    table = pq.read_table(path, columns=["sample_id", "month", "target", *STATE_FEATURE_NAMES])
    cols = {name: table[name].to_numpy(zero_copy_only=False) for name in table.column_names}
    order = np.argsort(cols["sample_id"], kind="mergesort")
    cols = {name: np.asarray(value)[order] for name, value in cols.items()}
    frame = MarketStateFrame(cols["sample_id"], cols["month"], cols["target"].astype(np.float64), np.column_stack([cols[name] for name in STATE_FEATURE_NAMES]).astype(np.float32))
    frame.validate()
    diagnostics = manifest.get("diagnostics", {})
    expected_hashes = diagnostics.get("artifact_hashes", {})
    actual_hashes = {
        "sample_id": array_hash(frame.sample_id), "month": array_hash(frame.month),
        "target": array_hash(frame.target), "values": array_hash(frame.values),
    }
    if expected_hashes != actual_hashes:
        raise ValueError("M05 state values hash mismatch")
    return frame


def _fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lo = np.nanpercentile(x, 1, axis=0); hi = np.nanpercentile(x, 99, axis=0)
    clipped = np.clip(x, lo, hi); med = np.nanmedian(clipped, axis=0)
    q25 = np.nanpercentile(clipped, 25, axis=0); q75 = np.nanpercentile(clipped, 75, axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    return lo, hi, med, scale


def _transform(x: np.ndarray, state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    lo, hi, med, scale = state
    value = np.where(np.isfinite(x), x, med)
    return np.clip((np.clip(value, lo, hi) - med) / scale, -5.0, 5.0)


def _prototypes(x: np.ndarray, residual: np.ndarray, months: np.ndarray, state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.cluster import MiniBatchKMeans
    z = _transform(x, state); centers=[]; values=[]; sources=[]
    for month in sorted(np.unique(months).tolist()):
        mask = months == month
        data = z[mask]; y = residual[mask]
        n_clusters = min(64, max(1, data.shape[0]))
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=2026, n_init=1, batch_size=min(1024, max(1, data.shape[0])))
        labels = km.fit_predict(data)
        for cluster in range(n_clusters):
            sel = labels == cluster
            if sel.any():
                centers.append(km.cluster_centers_[cluster]); values.append(float(y[sel].mean())); sources.append(int(month))
    return np.asarray(centers), np.asarray(values), np.asarray(sources, dtype=np.int16)


def _predict(x: np.ndarray, query_month: np.ndarray, prototypes: tuple[np.ndarray, np.ndarray, np.ndarray], state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], k: int) -> np.ndarray:
    centers, values, sources = prototypes; z = _transform(x, state); out=np.zeros(z.shape[0], dtype=np.float64)
    for i, month in enumerate(query_month):
        allowed = sources < month
        if not allowed.any(): continue
        d = np.sqrt(np.sum((centers[allowed] - z[i]) ** 2, axis=1)); take=np.argsort(d)[:min(k, d.size)]
        w = 1.0 / np.maximum(d[take], 1e-6) ** 2; out[i] = float(np.dot(w, values[allowed][take]) / w.sum())
    return out


def fit_m05_selection(canonical: CanonicalOOF, features: MarketStateFrame, outer: str, config: M01AConfig = M01AConfig()) -> dict[str, Any]:
    features.validate(); view=outer_residual(canonical, outer); ids=view["sample_id"]; months=view["month"]; target=view["target"]; residual=view["residual"]; baseline=view["baseline_oof"]
    rows=np.searchsorted(features.sample_id, ids)
    if np.any(rows >= features.sample_id.size) or not np.array_equal(features.sample_id[rows], ids) or not np.array_equal(features.month[rows], months) or not np.array_equal(features.target[rows], target): raise ValueError("M05 OOF/state labels do not align")
    train_range,tune_range=INNER_SPLITS[outer]; train=train_range.contains(months); tune=tune_range.contains(months)
    state=_fit_scaler(features.values[rows][train]); proto=_prototypes(features.values[rows][train], residual[train], months[train], state)
    best=None
    for k in K_GRID:
        pred=_predict(features.values[rows][tune], months[tune], proto, state, k)
        selected=select_alpha(baseline[tune], pred, target[tune])
        candidate=(selected["score"], -k, -selected["alpha"], k, pred, selected)
        if best is None or candidate[:3] > best[:3]: best=candidate
    _,_,_,k, tune_pred, selected=best
    refit_state=_fit_scaler(features.values[rows]); refit_proto=_prototypes(features.values[rows], residual, months, refit_state)
    return {"outer": outer, "beta": float(view["beta"]), "k": int(k), "alpha": float(selected["alpha"]), "baseline_scale": float(selected["baseline_scale"]), "residual_scale": float(selected["residual_scale"]), "tune_score": float(selected["score"]), "tune_baseline_score": float(selected["baseline_score"]), "tune_prediction": selected["prediction"], "tune_residual_prediction": tune_pred, "tune_sample_id": ids[tune], "tune_month": months[tune], "tune_target": target[tune], "tune_baseline_oof": baseline[tune], "refit_state": refit_state, "refit_proto": refit_proto}


def run_m05_outer(canonical: CanonicalOOF, features: MarketStateFrame, baseline_root: str | Path, output_root: str | Path, outer: str, config: M01AConfig = M01AConfig()) -> dict[str, Any]:
    started = time.perf_counter()
    sel = fit_m05_selection(canonical, features, outer, config)
    baseline = _load_outer_baseline(baseline_root, outer)
    rows = np.searchsorted(features.sample_id, baseline["sample_id"])
    if np.any(rows >= features.sample_id.size) or not np.array_equal(features.sample_id[rows], baseline["sample_id"]):
        raise ValueError(f"{outer}: M05 state and frozen baseline IDs do not align")
    if not np.array_equal(features.month[rows], baseline["month"]) or not np.array_equal(features.target[rows], baseline["target"]):
        raise ValueError(f"{outer}: M05 state and frozen baseline labels do not align")
    residual_pred = _predict(features.values[rows], baseline["month"], sel["refit_proto"], sel["refit_state"], sel["k"])
    final = baseline["pred"] / sel["baseline_scale"] + sel["alpha"] * residual_pred / max(sel["residual_scale"], 1e-12)
    if not np.isfinite(final).all():
        raise ValueError(f"{outer}: M05 predictions must be finite")
    score = cosine_uncentered(final, baseline["target"])
    base_score = cosine_uncentered(baseline["pred"], baseline["target"])
    diag = prediction_diagnostics(final, baseline["target"], reference=baseline["pred"])
    diag.update({"outer": outer, "beta": sel["beta"], "k": sel["k"], "alpha": sel["alpha"],
                 "best_iteration": None, "baseline_score": base_score, "final_score": score,
                 "delta_vs_baseline": score - base_score, "drift": drift_report(sel["tune_prediction"], final),
                 "rows": int(final.size), "finite_ok": True, "lb142_prediction_corr": None,
                 "lb142_status": "unavailable"})
    out = Path(output_root) / "m05" / outer
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "inner_predictions.npz", sample_id=sel["tune_sample_id"], month=sel["tune_month"],
                        target=sel["tune_target"], baseline_oof=sel["tune_baseline_oof"],
                        residual_pred=sel["tune_residual_prediction"], pred=sel["tune_prediction"])
    np.savez_compressed(out / "predictions.npz", sample_id=baseline["sample_id"], month=baseline["month"],
                        target=baseline["target"], baseline_pred=baseline["pred"], residual_pred=residual_pred,
                        pred=final, split=np.full(final.size, f"{outer}:m05"))
    lo, hi, med, scale = sel["refit_state"]
    centers, values, sources = sel["refit_proto"]
    np.savez_compressed(out / "state.npz", lo=lo, hi=hi, median=med, iqr=scale,
                        centers=centers, prototype_residual=values, source_month=sources)
    state_payload = np.concatenate([lo, hi, med, scale, centers.reshape(-1), values, sources.astype(np.float64)])
    diag["state_hash"] = array_hash(state_payload)
    diag["prediction_hash"] = array_hash(final)
    cfg = hashlib.sha256(json.dumps(config.__dict__, sort_keys=True).encode()).hexdigest()
    ExperimentManifest(experiment_id=f"m05-{outer.lower()}", status="complete", config_hash=cfg,
        feature_hash=feature_hash(list(STATE_FEATURE_NAMES)), train_months=NESTED_SPLITS[outer].refit_train.as_tuple(),
        valid_months=NESTED_SPLITS[outer].outer_valid.as_tuple(), best_step=sel["k"],
        scores={"cosine_uncentered": score}, diagnostics=diag,
        data_fingerprints={"state_values": array_hash(features.values), "baseline_pred": array_hash(baseline["pred"]),
                           "canonical": array_hash(canonical.baseline_oof)},
        runtime_seconds=time.perf_counter() - started).write(out)
    (out / "training_history.json").write_text(json.dumps({"outer": outer, "k": sel["k"], "alpha": sel["alpha"],
        "beta": sel["beta"], "source_month_rule": "source_month < query_month"}, indent=2), encoding="utf-8")
    (out / "report.md").write_text(f"# M05 - {outer}\n\n- score: `{score:.9f}`\n- delta: `{diag['delta_vs_baseline']:+.9f}`\n- k / alpha: `{sel['k']}` / `{sel['alpha']:.2f}`\n", encoding="utf-8")
    return diag


def summarize_m05(artifact_root: str | Path) -> dict[str, Any]:
    rows = []
    for outer in ("PSEUDO", "H2", "T3", "T4"):
        d = Path(artifact_root) / "m05" / outer
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        diag = manifest.get("diagnostics", {})
        split = NESTED_SPLITS[outer]
        if manifest.get("status") != "complete" or manifest.get("experiment_id") != f"m05-{outer.lower()}":
            raise ValueError(f"{outer}: invalid M05 manifest")
        if manifest.get("train_months") != list(split.refit_train.as_tuple()) or manifest.get("valid_months") != list(split.outer_valid.as_tuple()):
            raise ValueError(f"{outer}: manifest month ranges mismatch")
        with np.load(d / "predictions.npz") as src:
            a = {k: np.asarray(src[k]) for k in src.files}
        required = {"sample_id", "month", "target", "baseline_pred", "residual_pred", "pred", "split"}
        if set(a) != required:
            raise ValueError(f"{outer}: invalid M05 prediction schema")
        n = a["pred"].size
        if any(v.reshape(-1).size != n for v in a.values()) or np.unique(a["sample_id"]).size != n:
            raise ValueError(f"{outer}: invalid M05 prediction alignment")
        if set(np.asarray(a["month"], dtype=int).tolist()) != set(range(split.outer_valid.start, split.outer_valid.end + 1)):
            raise ValueError(f"{outer}: prediction months mismatch")
        if not np.array_equal(a["split"], np.full(n, f"{outer}:m05")):
            raise ValueError(f"{outer}: prediction split mismatch")
        if not all(np.isfinite(a[k]).all() for k in ("target", "baseline_pred", "residual_pred", "pred")):
            raise ValueError(f"{outer}: non-finite M05 predictions")
        score = cosine_uncentered(a["pred"], a["target"])
        base = cosine_uncentered(a["baseline_pred"], a["target"])
        if not np.isclose(score, diag.get("final_score"), atol=1e-12) or not np.isclose(base, diag.get("baseline_score"), atol=1e-12):
            raise ValueError(f"{outer}: M05 score mismatch")
        if diag.get("prediction_hash") != array_hash(a["pred"]):
            raise ValueError(f"{outer}: prediction hash mismatch")
        rows.append(diag | {"outer": outer, "final_score": score, "baseline_score": base,
                            "delta_vs_baseline": score - base, "finite_ok": True})
    deltas = np.array([r["delta_vs_baseline"] for r in rows], dtype=float)
    drift_ok = all(0.67 <= r.get("drift", {}).get("std_test_over_valid", 0.0) <= 1.50 and
                   0.50 <= r.get("drift", {}).get("abs_p99_test_over_valid", 0.0) <= 2.00 for r in rows)
    gate = {"pseudo_delta_at_least_0_0015": bool(deltas[0] >= 0.0015), "positive_outers": int((deltas > 0).sum()),
            "worst_delta": float(deltas.min()), "drift_ok": bool(drift_ok), "finite_ok": True,
            "passed": bool(deltas[0] >= 0.0015 and (deltas > 0).sum() >= 3 and deltas.min() >= -0.0005 and drift_ok)}
    return {"method": "M05 Historical Market-State KNN", "rows": rows, "mean_delta": float(deltas.mean()), "gate": gate}
