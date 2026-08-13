"""M01-A Event Flow residual training under the frozen temporal protocol."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..artifacts import ExperimentManifest, array_hash, feature_hash
from ..diagnostics import drift_report, prediction_diagnostics
from ..features.event_flow import event_flow_feature_names
from ..metrics import cosine_uncentered, normalize_prediction
from ..residual import CanonicalOOF, outer_residual
from ..splits import MonthRange, NESTED_SPLITS
from .residual_catboost import CatBoostResidualRegressor


ALPHA_GRID = np.round(np.arange(0.0, 0.3001, 0.01), 2)
RESIDUAL_INNER_SPLITS = {
    "PSEUDO": (MonthRange(21, 26), MonthRange(27, 32)),
    "H2": (MonthRange(21, 30), MonthRange(31, 40)),
    "T3": (MonthRange(21, 40), MonthRange(41, 50)),
    "T4": (MonthRange(21, 40), MonthRange(41, 50)),
}


@dataclass(frozen=True)
class M01AConfig:
    max_iterations: int = 3000
    early_stopping_rounds: int = 200
    random_seed: int = 2026
    learning_rate: float = 0.02
    depth: int = 6
    l2_leaf_reg: float = 5.0
    subsample: float = 0.8
    colsample_bylevel: float = 0.8

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "M01AConfig":
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in allowed})

    def __post_init__(self) -> None:
        if self.max_iterations <= 0 or self.early_stopping_rounds <= 0:
            raise ValueError("M01-A iterations and early stopping must be positive")
        if not 0.0 < self.learning_rate:
            raise ValueError("M01-A learning_rate must be positive")


@dataclass(frozen=True)
class EventFlowFrame:
    sample_id: np.ndarray
    month: np.ndarray
    target: np.ndarray
    values: np.ndarray
    feature_names: tuple[str, ...]

    def validate(self) -> None:
        n = np.asarray(self.sample_id).reshape(-1).size
        if any(np.asarray(value).reshape(-1).size != n for value in (self.month, self.target)):
            raise ValueError("Event Flow identifiers, months and targets must align")
        if np.asarray(self.values).shape != (n, len(self.feature_names)):
            raise ValueError("Event Flow matrix shape does not match rows/features")
        if np.unique(self.sample_id).size != n:
            raise ValueError("Event Flow sample_id must be unique")
        if not np.isfinite(self.target).all() or not np.isfinite(self.values).all():
            raise ValueError("Event Flow targets and features must be finite")


@dataclass
class M01ASelection:
    outer: str
    beta: float
    best_iteration: int
    alpha: float
    baseline_scale: float
    residual_scale: float
    tune_score: float
    tune_baseline_score: float
    tune_prediction: np.ndarray
    tune_residual_prediction: np.ndarray
    tune_sample_id: np.ndarray
    tune_month: np.ndarray
    tune_target: np.ndarray
    tune_baseline_oof: np.ndarray
    refit_model: Any


ModelFactory = Callable[[int, int], Any]


def load_event_flow_frame(path: str | Path) -> EventFlowFrame:
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("polars is required to load M01-A features") from exc
    path = Path(path)
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("M01-A Event Flow manifest.json is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "m01-a-event-flow-features" or manifest.get("status") != "complete":
        raise ValueError("M01-A Event Flow manifest identity/status is invalid")
    frame = pl.read_parquet(path)
    names = event_flow_feature_names()
    required = {"sample_id", "month", "target", *names}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Event Flow artifact is missing columns: {sorted(missing)}")
    frame = frame.sort("sample_id")
    raw_target = frame["target"].to_numpy()
    result = EventFlowFrame(
        frame["sample_id"].to_numpy(), frame["month"].to_numpy(),
        raw_target.astype(np.float64),
        frame.select(names).to_numpy().astype(np.float32), tuple(names),
    )
    result.validate()
    if manifest.get("feature_hash") != feature_hash(list(names)):
        raise ValueError("M01-A Event Flow feature hash is invalid")
    if manifest.get("diagnostics", {}).get("rows") != result.sample_id.size:
        raise ValueError("M01-A Event Flow manifest row count is invalid")
    expected_hashes = {
        "sample_id": array_hash(result.sample_id),
        "month": array_hash(result.month),
        "target": array_hash(raw_target),
        "values": array_hash(result.values),
    }
    if manifest.get("diagnostics", {}).get("artifact_hashes") != expected_hashes:
        raise ValueError("M01-A Event Flow artifact hashes are invalid")
    return result


def _take_features(frame: EventFlowFrame, sample_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    requested = np.asarray(sample_id).reshape(-1)
    order = np.argsort(frame.sample_id)
    sorted_ids = np.asarray(frame.sample_id)[order]
    positions = np.searchsorted(sorted_ids, requested)
    if np.any(positions >= sorted_ids.size) or not np.array_equal(sorted_ids[positions], requested):
        raise ValueError("Event Flow artifact does not cover every requested sample_id")
    rows = order[positions]
    return frame.values[rows], frame.month[rows], frame.target[rows]


def select_alpha(
    baseline: object,
    residual_prediction: object,
    target: object,
    *,
    alpha_grid: np.ndarray = ALPHA_GRID,
) -> dict[str, Any]:
    base_norm, base_scale = normalize_prediction(baseline, "rms")
    residual_norm, residual_scale = normalize_prediction(residual_prediction, "rms")
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    scores = np.array(
        [cosine_uncentered(base_norm + float(alpha) * residual_norm, y) for alpha in alpha_grid]
    )
    best_index = int(np.argmax(scores))
    alpha = float(alpha_grid[best_index])
    return {
        "alpha": alpha,
        "score": float(scores[best_index]),
        "baseline_score": cosine_uncentered(base_norm, y),
        "baseline_scale": float(base_scale),
        "residual_scale": float(residual_scale),
        "prediction": base_norm + alpha * residual_norm,
        "scores": scores,
    }


def _default_model_factory(config: M01AConfig) -> ModelFactory:
    def factory(iterations: int, early_stopping_rounds: int) -> CatBoostResidualRegressor:
        return CatBoostResidualRegressor(
            max_iterations=iterations,
            early_stopping_rounds=early_stopping_rounds,
            random_seed=config.random_seed,
            learning_rate=config.learning_rate,
            depth=config.depth,
            l2_leaf_reg=config.l2_leaf_reg,
            subsample=config.subsample,
            colsample_bylevel=config.colsample_bylevel,
        )

    return factory


def fit_m01a_selection(
    canonical: CanonicalOOF,
    features: EventFlowFrame,
    outer: str,
    *,
    config: M01AConfig = M01AConfig(),
    model_factory: ModelFactory | None = None,
) -> M01ASelection:
    """Select CatBoost length and alpha using historical OOF only."""

    features.validate()
    view = outer_residual(canonical, outer)
    sample_id = np.asarray(view["sample_id"])
    months = np.asarray(view["month"])
    target = np.asarray(view["target"], dtype=np.float64)
    residual_target = np.asarray(view["residual"], dtype=np.float64)
    baseline = np.asarray(view["baseline_oof"], dtype=np.float64)
    x, feature_months, feature_target = _take_features(features, sample_id)
    if not np.array_equal(months, feature_months) or not np.array_equal(target, feature_target):
        raise ValueError("canonical OOF and Event Flow labels must align exactly")
    train_range, tune_range = RESIDUAL_INNER_SPLITS[outer]
    train_mask = train_range.contains(months)
    tune_mask = tune_range.contains(months)
    if not train_mask.any() or not tune_mask.any() or np.any(train_mask & tune_mask):
        raise ValueError(f"{outer}: invalid residual inner split")
    factory = model_factory or _default_model_factory(config)
    inner_model = factory(config.max_iterations, config.early_stopping_rounds)
    inner_model.fit(
        x[train_mask], residual_target[train_mask],
        eval_set=(x[tune_mask], residual_target[tune_mask]),
    )
    tune_residual_prediction = np.asarray(inner_model.predict(x[tune_mask]), dtype=np.float64)
    zero_based_best = inner_model.best_iteration
    best_iteration = config.max_iterations if zero_based_best is None else int(zero_based_best) + 1
    alpha_result = select_alpha(
        baseline[tune_mask], tune_residual_prediction, target[tune_mask]
    )
    refit_model = factory(best_iteration, 0)
    refit_model.fit(x, residual_target, eval_set=None)
    return M01ASelection(
        outer=outer, beta=float(view["beta"]), best_iteration=best_iteration,
        alpha=float(alpha_result["alpha"]),
        baseline_scale=float(alpha_result["baseline_scale"]),
        residual_scale=float(alpha_result["residual_scale"]),
        tune_score=float(alpha_result["score"]),
        tune_baseline_score=float(alpha_result["baseline_score"]),
        tune_prediction=np.asarray(alpha_result["prediction"]),
        tune_residual_prediction=tune_residual_prediction,
        tune_sample_id=sample_id[tune_mask], tune_month=months[tune_mask],
        tune_target=target[tune_mask], refit_model=refit_model,
        tune_baseline_oof=baseline[tune_mask],
    )


def _apply_selected_scale(value: object, scale: float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return array.copy() if scale == 0.0 else array / scale


def _load_outer_baseline(directory: str | Path, outer: str) -> dict[str, np.ndarray]:
    directory = Path(directory) / outer
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    split = NESTED_SPLITS[outer]
    expected = {
        "experiment_id": f"clean-baseline-v2-{outer.lower()}",
        "status": "frozen",
        "train_months": list(split.refit_train.as_tuple()),
        "valid_months": list(split.outer_valid.as_tuple()),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{outer}: frozen baseline manifest {key} mismatch")
    path = directory / "predictions.npz"
    with np.load(path) as source:
        required = {"sample_id", "month", "target", "pred", "split"}
        if not required.issubset(source.files):
            raise ValueError(f"{outer}: frozen baseline predictions schema is incomplete")
        artifact = {key: np.asarray(source[key]) for key in required}
    if set(artifact["month"].tolist()) != set(range(split.outer_valid.start, split.outer_valid.end + 1)):
        raise ValueError(f"{outer}: frozen baseline months mismatch")
    if not np.all(np.char.endswith(artifact["split"].astype(str), ":production_default")):
        raise ValueError(f"{outer}: predictions.npz is not the frozen production default")
    if not np.isfinite(artifact["target"]).all() or not np.isfinite(artifact["pred"]).all():
        raise ValueError(f"{outer}: frozen baseline arrays must be finite")
    if manifest.get("diagnostics", {}).get("rows") != int(artifact["pred"].size):
        raise ValueError(f"{outer}: frozen baseline row count mismatch")
    replay_score = cosine_uncentered(artifact["pred"], artifact["target"])
    expected_score = manifest.get("scores", {}).get("production_schema_replay_cosine")
    if expected_score is None or not np.isclose(replay_score, expected_score, rtol=0.0, atol=1e-12):
        raise ValueError(f"{outer}: frozen baseline production replay score mismatch")
    return artifact


def run_m01a_outer(
    canonical: CanonicalOOF,
    features: EventFlowFrame,
    baseline_root: str | Path,
    output_root: str | Path,
    outer: str,
    *,
    config: M01AConfig = M01AConfig(),
    model_factory: ModelFactory | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    selection = fit_m01a_selection(
        canonical, features, outer, config=config, model_factory=model_factory
    )
    baseline = _load_outer_baseline(baseline_root, outer)
    x_outer, feature_month, feature_target = _take_features(features, baseline["sample_id"])
    if not np.array_equal(feature_month, baseline["month"]) or not np.array_equal(
        feature_target, baseline["target"]
    ):
        raise ValueError(f"{outer}: Event Flow and frozen baseline labels must align")
    residual_prediction = np.asarray(selection.refit_model.predict(x_outer), dtype=np.float64)
    final = _apply_selected_scale(baseline["pred"], selection.baseline_scale) + selection.alpha * _apply_selected_scale(
        residual_prediction, selection.residual_scale
    )
    if not np.isfinite(final).all():
        raise ValueError(f"{outer}: final M01-A predictions must be finite")
    baseline_score = cosine_uncentered(baseline["pred"], baseline["target"])
    final_score = cosine_uncentered(final, baseline["target"])
    diagnostics = prediction_diagnostics(final, baseline["target"], reference=baseline["pred"])
    diagnostics.update(
        {
            "outer": outer, "beta": selection.beta,
            "alpha": selection.alpha, "best_iteration": selection.best_iteration,
            "baseline_scale": selection.baseline_scale,
            "residual_scale": selection.residual_scale,
            "inner_tune_score": selection.tune_score,
            "inner_tune_baseline_score": selection.tune_baseline_score,
            "baseline_score": baseline_score, "final_score": final_score,
            "delta_vs_baseline": final_score - baseline_score,
            "drift": drift_report(selection.tune_prediction, final),
            "rows": int(final.size),
            "lb142_prediction_corr": None,
            "lb142_status": "no_outer_aligned_reference_provided",
        }
    )
    output = Path(output_root) / "m01-a" / outer
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "inner_predictions.npz",
        sample_id=selection.tune_sample_id, month=selection.tune_month,
        target=selection.tune_target, baseline_oof=np.asarray(
            selection.tune_baseline_oof
        ),
        residual_pred=selection.tune_residual_prediction, pred=selection.tune_prediction,
    )
    np.savez_compressed(
        output / "predictions.npz",
        sample_id=baseline["sample_id"], month=baseline["month"], target=baseline["target"],
        baseline_pred=baseline["pred"], residual_pred=residual_prediction, pred=final,
        split=np.full(final.size, f"{outer}:m01-a"),
    )
    history = {
        "outer": outer, "beta": selection.beta, "best_iteration": selection.best_iteration,
        "alpha": selection.alpha, "alpha_grid": ALPHA_GRID.tolist(),
        "baseline_scale": selection.baseline_scale, "residual_scale": selection.residual_scale,
        "inner_tune_score": selection.tune_score,
    }
    (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    config_payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest = ExperimentManifest(
        experiment_id=f"m01-a-{outer.lower()}", status="complete",
        config_hash=hashlib.sha256(config_payload).hexdigest(),
        data_fingerprints={
            "canonical_sample_id": array_hash(canonical.sample_id),
            "canonical_month": array_hash(canonical.month),
            "canonical_target": array_hash(canonical.target),
            "canonical_baseline_oof": array_hash(canonical.baseline_oof),
            "event_flow_values": array_hash(features.values),
            "frozen_outer_baseline": array_hash(baseline["pred"]),
        },
        feature_hash=feature_hash(list(features.feature_names)),
        train_months=(21, int(np.max(outer_residual(canonical, outer)["month"]))),
        valid_months=NESTED_SPLITS[outer].outer_valid.as_tuple(),
        best_step=selection.best_iteration, scores={"cosine_uncentered": final_score},
        diagnostics=diagnostics, runtime_seconds=time.perf_counter() - started,
    )
    manifest.write(output)
    (output / "report.md").write_text(
        "\n".join(
            [
                f"# M01-A - {outer}", "",
                f"- score: `{final_score:.9f}`",
                f"- delta vs frozen baseline: `{final_score - baseline_score:+.9f}`",
                f"- beta / alpha: `{selection.beta:.9g}` / `{selection.alpha:.2f}`",
                f"- best iteration: `{selection.best_iteration}`", "",
            ]
        ), encoding="utf-8",
    )
    return diagnostics | {"output": str(output)}


def summarize_m01a(artifact_root: str | Path) -> dict[str, Any]:
    rows = []
    for outer in ("PSEUDO", "H2", "T3", "T4"):
        manifest = json.loads(
            (Path(artifact_root) / "m01-a" / outer / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("status") != "complete":
            raise ValueError(f"{outer}: M01-A artifact is incomplete")
        rows.append(manifest["diagnostics"])
    deltas = np.array([row["delta_vs_baseline"] for row in rows], dtype=float)
    drift_ok = all(
        0.67 <= row["drift"]["std_test_over_valid"] <= 1.50
        and 0.50 <= row["drift"]["abs_p99_test_over_valid"] <= 2.00
        for row in rows
    )
    gate = {
        "pseudo_delta_at_least_0_0015": rows[0]["delta_vs_baseline"] >= 0.0015,
        "positive_outers": int((deltas > 0.0).sum()),
        "worst_delta": float(deltas.min()),
        "drift_ok": drift_ok,
    }
    gate["passed"] = bool(
        gate["pseudo_delta_at_least_0_0015"]
        and gate["positive_outers"] >= 3
        and gate["worst_delta"] >= -0.0005
        and gate["drift_ok"]
    )
    return {"rows": rows, "mean_delta": float(deltas.mean()), "gate": gate}
