"""Leakage-safe C3 reproduction of the legacy v5 table ensemble.

The legacy representation is intentionally narrow: eight directional R2
replacements over the 90 official features, plus the 22 microstructure
features.  Each nested fold searches stopping points on its inner tune block,
then reinitializes and refits from history only before touching outer valid.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..artifacts import ExperimentManifest, feature_hash, git_sha, save_predictions
from ..diagnostics import prediction_diagnostics
from ..metrics import cosine_uncentered
from ..splits import NESTED_SPLITS, TRAINING_SPLITS


R2_REPLACEMENTS: dict[str, tuple[str, str, float]] = {
    "m_sp_mean": ("m_sp_mean", "m_mid_mean", 1e-8),
    "m_depth_mean": ("m_depth_mean", "m_txv_sum_60", 1.0),
    "m_rv": ("m_rv", "m_mid_std", 1e-8),
    "o_vol_sum": ("o_vol_sum", "t_vol_sum", 1.0),
    "o_n_120": ("o_n_120", "t_n_120", 1.0),
    "m_txv_sum_180": ("m_txv_sum_180", "m_txv_sum_60", 1.0),
    "m_sp_mean_60": ("m_sp_mean_60", "m_mid_mean_60", 1e-8),
    "m_sp_mean_180": ("m_sp_mean_180", "m_mid_mean_180", 1e-8),
}
TABLE_COMPONENT_WEIGHTS = {"lgb": 0.2, "cat": 0.5, "mlp": 0.3}


def _stable_file_fingerprint(path: str | Path, sample_bytes: int = 1 << 20) -> str:
    target = Path(path)
    stat = target.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("ascii"))
    with target.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if stat.st_size > sample_bytes:
            handle.seek(max(0, stat.st_size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def _environment_versions() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for distribution in ("numpy", "pandas", "lightgbm", "catboost", "torch", "pyarrow"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = "unavailable"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }


@dataclass(frozen=True)
class CleanTableConfig:
    seed: int = 0
    lgb_iterations: int = 10_000
    cat_iterations: int = 10_000
    early_stopping_rounds: int = 200
    learning_rate: float = 0.02
    mlp_epochs: int = 30
    mlp_seeds: tuple[int, ...] = (2026, 7, 123)
    mlp_hidden: int = 256
    mlp_dropout: float = 0.2
    mlp_batch_size: int = 2048
    eval_batch_size: int = 8192
    threads: int = 8
    device: str = "auto"
    max_rows_per_month: int | None = None

    def __post_init__(self) -> None:
        if min(self.lgb_iterations, self.cat_iterations, self.mlp_epochs) <= 0:
            raise ValueError("all training lengths must be positive")
        if not self.mlp_seeds:
            raise ValueError("at least one MLP seed is required")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "CleanTableConfig":
        fields = {field.name for field in cls.__dataclass_fields__.values()}
        values = {key: value for key, value in mapping.items() if key in fields}
        if "mlp_seeds" in values:
            values["mlp_seeds"] = tuple(int(value) for value in values["mlp_seeds"])
        return cls(**values)


@dataclass(frozen=True)
class TableFrame:
    sample_id: np.ndarray
    month: np.ndarray
    target: np.ndarray
    values: np.ndarray
    feature_names: tuple[str, ...]


def apply_r2(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    missing = sorted({name for triple in R2_REPLACEMENTS.values() for name in triple[:2]} - set(values))
    if missing:
        raise ValueError(f"missing R2 source columns: {missing}")
    result = {name: np.asarray(column).copy() for name, column in values.items()}
    for output, (numerator, denominator, offset) in R2_REPLACEMENTS.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            result[output] = np.asarray(values[numerator], dtype=np.float64) / (
                np.asarray(values[denominator], dtype=np.float64) + offset
            )
    return result


def load_table_frame(
    features_path: str | Path,
    micro_path: str | Path,
    labels_path: str | Path | None = None,
    *,
    max_rows_per_month: int | None = None,
) -> TableFrame:
    import pandas as pd

    features = pd.read_parquet(features_path)
    micro = pd.read_parquet(micro_path)
    required = {"sample_id", "month", "target"}
    if not required.issubset(features):
        raise ValueError(f"feature parquet is missing {sorted(required - set(features))}")
    if features["sample_id"].duplicated().any() or micro["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique in both table inputs")
    if labels_path is not None:
        labels = pd.read_feather(labels_path)[["sample_id", "month", "target"]]
        if labels["sample_id"].duplicated().any():
            raise ValueError("label sample_id must be unique")
        left = features[["sample_id", "month", "target"]].sort_values("sample_id").reset_index(drop=True)
        right = labels.sort_values("sample_id").reset_index(drop=True)
        if not np.array_equal(left["sample_id"].to_numpy(), right["sample_id"].to_numpy()):
            raise ValueError("feature and label sample_id do not match")
        if not np.array_equal(left["month"].to_numpy(), right["month"].to_numpy()):
            raise ValueError("feature and label month do not match")
        if not np.array_equal(left["target"].to_numpy(), right["target"].to_numpy()):
            raise ValueError("feature and label target do not match exactly")
    official = [name for name in features.columns if name not in required]
    micro_names = [name for name in micro.columns if name != "sample_id"]
    if len(official) != 90 or len(micro_names) != 22:
        raise ValueError(f"expected 90 official + 22 micro features, got {len(official)} + {len(micro_names)}")
    r2_frame = features[["sample_id", *official]].copy()
    for output, (numerator, denominator, offset) in R2_REPLACEMENTS.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            r2_frame[output] = (
                r2_frame[numerator].to_numpy(dtype=np.float64)
                / (r2_frame[denominator].to_numpy(dtype=np.float64) + offset)
            )
    merged = r2_frame.merge(micro, on="sample_id", how="left", validate="one_to_one", sort=False)
    merged[micro_names] = merged[micro_names].fillna(0.0)
    order = np.argsort(features["sample_id"].to_numpy(), kind="stable")
    frame = TableFrame(
        sample_id=features["sample_id"].to_numpy()[order].astype(np.int64),
        month=features["month"].to_numpy()[order].astype(np.int16),
        target=features["target"].to_numpy()[order].astype(np.float32),
        values=merged[official + micro_names].to_numpy(dtype=np.float32)[order],
        feature_names=tuple(official + micro_names),
    )
    if max_rows_per_month is None:
        return frame
    if max_rows_per_month <= 0:
        raise ValueError("max_rows_per_month must be positive")
    keep = np.zeros(frame.month.size, dtype=bool)
    for month in np.unique(frame.month):
        keep[np.flatnonzero(frame.month == month)[:max_rows_per_month]] = True
    return TableFrame(frame.sample_id[keep], frame.month[keep], frame.target[keep], frame.values[keep], frame.feature_names)


class StandardClip:
    """Legacy MLP standardization, fitted only on the current training history."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "StandardClip":
        clean = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
        self.mean = clean.mean(axis=0, keepdims=True)
        self.std = clean.std(axis=0, keepdims=True) + 1e-6
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardClip must be fitted before transform")
        clean = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)
        return np.clip((clean - self.mean) / self.std, -10.0, 10.0).astype(np.float32)

    @property
    def state_hash(self) -> str:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardClip has no fitted state")
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.mean).tobytes())
        digest.update(np.ascontiguousarray(self.std).tobytes())
        return digest.hexdigest()


def _lgb_metric(pred: np.ndarray, dataset: Any) -> tuple[str, float, bool]:
    return "cosine_uncentered", cosine_uncentered(pred, dataset.get_label()), True


def _lgb_params(cfg: CleanTableConfig) -> dict[str, Any]:
    return {
        "objective": "regression", "metric": "None", "learning_rate": cfg.learning_rate,
        "num_leaves": 64, "min_data_in_leaf": 300, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5, "lambda_l2": 5.0,
        "max_bin": 255, "verbose": -1, "num_threads": cfg.threads, "seed": cfg.seed,
    }


def _table_blend(predictions: Mapping[str, np.ndarray]) -> np.ndarray:
    if set(TABLE_COMPONENT_WEIGHTS) - set(predictions):
        raise ValueError("table blend requires lgb, cat, and mlp predictions")
    return sum(TABLE_COMPONENT_WEIGHTS[name] * predictions[name] for name in TABLE_COMPONENT_WEIGHTS)


class _CatCosine:
    def is_max_optimal(self) -> bool:
        return True

    def evaluate(self, approxes: list[np.ndarray], target: np.ndarray, weight: Any) -> tuple[float, float]:
        pred = np.asarray(approxes[0], dtype=np.float64)
        truth = np.asarray(target, dtype=np.float64)
        return float(pred @ truth), float(np.linalg.norm(pred) * np.linalg.norm(truth))

    def get_final_error(self, error: float, weight: float) -> float:
        return float(error / weight) if weight else 0.0


def _fit_lgb_inner(x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray, cfg: CleanTableConfig):
    import lightgbm as lgb

    params = _lgb_params(cfg)
    history: dict[str, Any] = {}
    train = lgb.Dataset(x, y)
    valid = lgb.Dataset(xv, yv, reference=train)
    model = lgb.train(
        params, train, cfg.lgb_iterations, valid_sets=[valid], feval=_lgb_metric,
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False), lgb.record_evaluation(history)],
    )
    return model.predict(xv, num_iteration=model.best_iteration), int(model.best_iteration), history


def _fit_lgb_refit(x: np.ndarray, y: np.ndarray, xv: np.ndarray, steps: int, cfg: CleanTableConfig) -> np.ndarray:
    import lightgbm as lgb

    return lgb.train(_lgb_params(cfg), lgb.Dataset(x, y), steps).predict(xv)


def _catboost_class():
    try:
        from catboost import CatBoostRegressor
    except RuntimeError as exc:  # pragma: no cover - workstation packaging issue
        raise RuntimeError("CatBoost runtime could not be loaded") from exc
    return CatBoostRegressor


def _cat_params(cfg: CleanTableConfig, iterations: int) -> dict[str, Any]:
    return {
        "iterations": iterations, "learning_rate": cfg.learning_rate, "depth": 6,
        "l2_leaf_reg": 5.0, "subsample": 0.8, "colsample_bylevel": 0.8,
        "loss_function": "RMSE", "verbose": 0, "thread_count": cfg.threads,
        "random_seed": cfg.seed, "allow_writing_files": False,
    }


def _fit_cat_inner(x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray, cfg: CleanTableConfig):
    model = _catboost_class()(
        **_cat_params(cfg, cfg.cat_iterations), eval_metric=_CatCosine(),
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
    model.fit(x, y, eval_set=(xv, yv))
    best = int(model.get_best_iteration()) + 1
    return model.predict(xv), best, model.get_evals_result()


def _fit_cat_refit(x: np.ndarray, y: np.ndarray, xv: np.ndarray, steps: int, cfg: CleanTableConfig) -> np.ndarray:
    model = _catboost_class()(**_cat_params(cfg, steps))
    model.fit(x, y)
    return model.predict(xv)


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Clean Table MLP requires torch") from exc
    return torch, nn


def _mlp_model(n_features: int, cfg: CleanTableConfig):
    _, nn = _require_torch()
    return nn.Sequential(
        nn.Linear(n_features, cfg.mlp_hidden), nn.GELU(), nn.Dropout(cfg.mlp_dropout),
        nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden), nn.GELU(), nn.Dropout(cfg.mlp_dropout),
        nn.Linear(cfg.mlp_hidden, 1),
    )


def _device(cfg: CleanTableConfig):
    torch, _ = _require_torch()
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def _seed_everything(seed: int) -> None:
    torch, _ = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict_mlp(model: Any, values: np.ndarray, device: Any, cfg: CleanTableConfig) -> np.ndarray:
    torch, _ = _require_torch()
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), cfg.eval_batch_size):
            batch = torch.from_numpy(values[start:start + cfg.eval_batch_size]).to(device)
            result.append(model(batch).squeeze(-1).cpu().numpy())
    return np.concatenate(result)


def _train_mlp_seed(
    x: np.ndarray, y: np.ndarray, xv: np.ndarray, cfg: CleanTableConfig, seed: int,
    *, yv: np.ndarray | None = None, epochs: int | None = None,
) -> tuple[np.ndarray, int, list[dict[str, float]]]:
    torch, nn = _require_torch()
    device = _device(cfg)
    _seed_everything(seed)
    model = _mlp_model(x.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    total_epochs = int(epochs or cfg.mlp_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.mlp_epochs)
    loss_fn = nn.MSELoss()
    y_mean = float(np.mean(y))
    y_std = float(np.std(y) + 1e-6)
    y_scaled = ((y - y_mean) / y_std).astype(np.float32)
    history: list[dict[str, float]] = []
    best_score = -np.inf
    best_step = total_epochs
    best_state: dict[str, Any] | None = None
    for epoch in range(1, total_epochs + 1):
        permutation = torch.randperm(len(x))
        model.train()
        loss_sum = 0.0
        for start in range(0, len(x), cfg.mlp_batch_size):
            indexes = permutation[start:start + cfg.mlp_batch_size].numpy()
            xb = torch.from_numpy(x[indexes]).to(device)
            yb = torch.from_numpy(y_scaled[indexes]).to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(indexes)
        scheduler.step()
        pred = _predict_mlp(model, xv, device, cfg) * y_std + y_mean
        score = cosine_uncentered(pred, yv) if yv is not None else float("nan")
        history.append({"epoch": epoch, "loss": loss_sum / len(x), "cosine_uncentered": score})
        if yv is not None and score > best_score:
            best_score, best_step = score, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    pred = _predict_mlp(model, xv, device, cfg) * y_std + y_mean
    return pred, best_step, history


def _component_diagnostics(predictions: Mapping[str, np.ndarray], target: np.ndarray) -> dict[str, float]:
    return {name: cosine_uncentered(pred, target) for name, pred in predictions.items()}


def run_outer(
    frame: TableFrame,
    outer_name: str,
    cfg: CleanTableConfig,
    output_dir: str | Path,
    *,
    experiment_id: str = "clean-table-v2",
    data_paths: tuple[str | Path, ...] = (),
    legacy_pseudo_path: str | Path | None = None,
) -> dict[str, Any]:
    if outer_name not in TRAINING_SPLITS:
        raise KeyError(f"unknown outer split: {outer_name}")
    started = time.time()
    source_git_sha = os.environ.get("MSCAP_GIT_SHA") or git_sha()
    split = TRAINING_SPLITS[outer_name]
    masks = {
        "inner_train": split.inner_train.contains(frame.month),
        "inner_tune": split.inner_tune.contains(frame.month),
        "refit": split.refit_train.contains(frame.month),
        "outer": split.outer_valid.contains(frame.month),
    }
    for name, mask in masks.items():
        if not np.any(mask):
            raise ValueError(f"{outer_name} {name} partition is empty")
    x, y = frame.values[masks["inner_train"]], frame.target[masks["inner_train"]]
    xv, yv = frame.values[masks["inner_tune"]], frame.target[masks["inner_tune"]]
    inner: dict[str, np.ndarray] = {}
    history: dict[str, Any] = {}
    steps: dict[str, Any] = {}
    inner["lgb"], steps["lgb"], history["lgb"] = _fit_lgb_inner(x, y, xv, yv, cfg)
    inner["cat"], steps["cat"], history["cat"] = _fit_cat_inner(x, y, xv, yv, cfg)
    inner_scaler = StandardClip().fit(x)
    x_mlp, xv_mlp = inner_scaler.transform(x), inner_scaler.transform(xv)
    mlp_inner = []
    steps["mlp"] = {}
    history["mlp"] = {}
    for seed in cfg.mlp_seeds:
        pred, best, seed_history = _train_mlp_seed(x_mlp, y, xv_mlp, cfg, seed, yv=yv)
        mlp_inner.append(pred)
        steps["mlp"][str(seed)] = best
        history["mlp"][str(seed)] = seed_history
    inner["mlp"] = np.mean(mlp_inner, axis=0)
    inner["blend"] = _table_blend(inner)

    xr, yr = frame.values[masks["refit"]], frame.target[masks["refit"]]
    xo, yo = frame.values[masks["outer"]], frame.target[masks["outer"]]
    outer = {
        "lgb": _fit_lgb_refit(xr, yr, xo, steps["lgb"], cfg),
        "cat": _fit_cat_refit(xr, yr, xo, steps["cat"], cfg),
    }
    refit_scaler = StandardClip().fit(xr)
    xr_mlp, xo_mlp = refit_scaler.transform(xr), refit_scaler.transform(xo)
    mlp_outer = []
    refit_history: dict[str, Any] = {}
    for seed in cfg.mlp_seeds:
        pred, _, seed_history = _train_mlp_seed(
            xr_mlp, yr, xo_mlp, cfg, seed, epochs=int(steps["mlp"][str(seed)])
        )
        mlp_outer.append(pred)
        refit_history[str(seed)] = seed_history
    outer["mlp"] = np.mean(mlp_outer, axis=0)
    outer["blend"] = _table_blend(outer)
    if not all(np.isfinite(pred).all() for pred in outer.values()):
        raise ValueError("outer predictions must be finite")

    output = Path(output_dir) / outer_name
    output.mkdir(parents=True, exist_ok=True)
    save_predictions(
        output / "inner_predictions.npz", sample_id=frame.sample_id[masks["inner_tune"]],
        month=frame.month[masks["inner_tune"]], target=yv, pred=inner["blend"], split=np.full(yv.size, "inner_tune"),
    )
    np.savez_compressed(output / "inner_components.npz", **inner)
    save_predictions(
        output / "predictions.npz", sample_id=frame.sample_id[masks["outer"]], month=frame.month[masks["outer"]],
        target=yo, pred=outer["blend"], split=np.full(yo.size, outer_name),
    )
    np.savez_compressed(output / "components.npz", **outer)
    diagnostics = prediction_diagnostics(outer["blend"], yo)
    pred_std = float(np.std(outer["blend"]))
    target_std = float(np.std(yo))
    diagnostics.update({
        "pearson": 0.0 if pred_std == 0.0 or target_std == 0.0 else float(np.corrcoef(outer["blend"], yo)[0, 1]),
        "pred_mean": float(np.mean(outer["blend"])),
        "pred_std": pred_std,
        "target_std": target_std,
        "nan_or_inf": int((~np.isfinite(outer["blend"])).sum()),
    })
    diagnostics["components"] = _component_diagnostics(outer, yo)
    diagnostics["inner_components"] = _component_diagnostics(inner, yv)
    diagnostics["mlp_inner_scaler_hash"] = inner_scaler.state_hash
    diagnostics["mlp_refit_scaler_hash"] = refit_scaler.state_hash
    if outer_name == "PSEUDO" and legacy_pseudo_path is not None:
        legacy = np.load(legacy_pseudo_path)
        if "y" not in legacy or not np.array_equal(np.asarray(legacy["y"]), yo):
            raise ValueError("legacy PSEUDO target does not match clean outer target exactly")
        diagnostics["legacy_score"] = cosine_uncentered(legacy["pred"], legacy["y"])
        diagnostics["clean_vs_legacy_prediction_corr"] = float(np.corrcoef(outer["blend"], legacy["pred"])[0, 1])
    runtime = time.time() - started
    payload = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest = ExperimentManifest(
        experiment_id=f"{experiment_id}-{outer_name.lower()}", status="complete", git_sha=source_git_sha,
        config_hash=hashlib.sha256(payload).hexdigest(), train_months=split.refit_train.as_tuple(),
        valid_months=split.outer_valid.as_tuple(), feature_hash=feature_hash(frame.feature_names),
        best_step=max(steps["lgb"], steps["cat"], *steps["mlp"].values()),
        best_progress=None, runtime_seconds=runtime,
        scores={"cosine_uncentered": float(diagnostics["cosine_uncentered"])},
        diagnostics=diagnostics | {"best_steps": steps, "n_features": len(frame.feature_names), "r2_replacements": sorted(R2_REPLACEMENTS)},
        environment=_environment_versions(),
    )
    manifest.data_fingerprints = {
        Path(path).name: _stable_file_fingerprint(path) for path in data_paths if Path(path).exists()
    }
    manifest.write(output)
    (output / "training_history.json").write_text(
        json.dumps({"inner": history, "refit_mlp": refit_history, "best_steps": steps}, indent=2), encoding="utf-8"
    )
    (output / "report.md").write_text(
        f"# Clean Table v2 - {outer_name}\n\n"
        f"- cosine_uncentered: `{diagnostics['cosine_uncentered']:.9f}`\n"
        f"- Pearson: `{diagnostics['pearson']:.9f}`\n"
        f"- samples: `{yo.size}`\n"
        f"- best steps: `{json.dumps(steps, sort_keys=True)}`\n"
        f"- runtime seconds: `{runtime:.1f}`\n",
        encoding="utf-8",
    )
    return {"outer": outer_name, "score": diagnostics["cosine_uncentered"], "diagnostics": diagnostics, "best_steps": steps}


def summarize_outer(
    artifact_root: str | Path,
    experiment_id: str,
    legacy_pseudo_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(artifact_root) / experiment_id
    rows = []
    for outer_name, split in NESTED_SPLITS.items():
        manifest_path = root / outer_name / "manifest.json"
        predictions_path = root / outer_name / "predictions.npz"
        if not manifest_path.exists() or not predictions_path.exists():
            raise FileNotFoundError(f"missing completed {outer_name} artifact")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pred = np.load(predictions_path)
        if tuple(manifest["valid_months"]) != split.outer_valid.as_tuple():
            raise ValueError(f"{outer_name} valid months do not match registry")
        if set(np.unique(pred["month"])) != set(range(split.outer_valid.start, split.outer_valid.end + 1)):
            raise ValueError(f"{outer_name} artifact does not cover every registered valid month")
        if not np.isfinite(pred["pred"]).all():
            raise ValueError(f"{outer_name} predictions are not finite")
        diagnostics = manifest["diagnostics"]
        rows.append({
            "outer": outer_name, "score": manifest["scores"]["cosine_uncentered"],
            "pearson": diagnostics["pearson"], "pred_mean": diagnostics["pred_mean"],
            "pred_std": diagnostics["pred_std"], "target_std": diagnostics["target_std"],
            "best_steps": diagnostics["best_steps"], "runtime_seconds": manifest["runtime_seconds"],
        })
    report = {
        "experiment_id": experiment_id, "rows": rows,
        "mean_score": float(np.mean([row["score"] for row in rows])),
        "note": "Outer folds are correlated temporal stress tests, not independent samples.",
    }
    if legacy_pseudo_path is not None:
        clean = np.load(root / "PSEUDO" / "predictions.npz")
        legacy = np.load(legacy_pseudo_path)
        if "y" not in legacy or not np.array_equal(clean["target"], legacy["y"]):
            raise ValueError("legacy PSEUDO target does not match clean PSEUDO target exactly")
        if not np.isfinite(legacy["pred"]).all():
            raise ValueError("legacy PSEUDO prediction must be finite")
        report["legacy_pseudo"] = {
            "score": cosine_uncentered(legacy["pred"], legacy["y"]),
            "clean_vs_legacy_prediction_corr": float(np.corrcoef(clean["pred"], legacy["pred"])[0, 1]),
        }
    json_path = Path(artifact_root) / f"{experiment_id.replace('-', '_')}_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Clean Table v2", "", "| Outer | Cosine | Pearson | Pred std | Target std |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['outer']} | {row['score']:.9f} | {row['pearson']:.9f} | {row['pred_std']:.9f} | {row['target_std']:.9f} |")
    lines += ["", f"Arithmetic mean: `{report['mean_score']:.9f}`.", "", report["note"]]
    if "legacy_pseudo" in report:
        lines += [
            "", "## Historical PSEUDO comparison", "",
            f"- legacy v5 score: `{report['legacy_pseudo']['score']:.9f}`",
            f"- clean-vs-legacy prediction correlation: `{report['legacy_pseudo']['clean_vs_legacy_prediction_corr']:.9f}`",
        ]
    json_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def verify_legacy_anchor(table_path: str | Path, realmlp_path: str | Path) -> dict[str, float]:
    table = np.load(table_path)
    realmlp = np.load(realmlp_path)
    for name, artifact in (("table", table), ("realmlp", realmlp)):
        if not {"pred", "y"}.issubset(artifact.files):
            raise ValueError(f"{name} artifact must contain pred and y")
        if not np.isfinite(artifact["pred"]).all() or not np.isfinite(artifact["y"]).all():
            raise ValueError(f"{name} artifact must be finite")
    if not np.array_equal(table["y"], realmlp["y"]):
        raise ValueError("legacy table and RealMLP targets do not match exactly")
    scores = {
        "table": cosine_uncentered(table["pred"], table["y"]),
        "realmlp": cosine_uncentered(realmlp["pred"], realmlp["y"]),
        "v7_raw": cosine_uncentered(0.8 * table["pred"] + 0.2 * realmlp["pred"], table["y"]),
        "prediction_corr": float(np.corrcoef(table["pred"], realmlp["pred"])[0, 1]),
    }
    expected = {"table": 0.1348707123, "realmlp": 0.1385597007, "v7_raw": 0.1396834806}
    for name, target in expected.items():
        if not np.isclose(scores[name], target, atol=5e-10, rtol=0.0):
            raise ValueError(f"legacy {name} score changed: {scores[name]:.12f} != {target:.12f}")
    return scores
