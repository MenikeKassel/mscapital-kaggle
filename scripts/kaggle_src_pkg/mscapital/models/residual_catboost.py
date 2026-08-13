"""CatBoost residual adapter with protocol-v2 defaults."""

from __future__ import annotations

from dataclasses import dataclass
import sys
import types
from typing import Any

import numpy as np


def _load_catboost_regressor() -> Any:
    """Load CatBoost even when an optional Polars binary is CPU-incompatible.

    CatBoost only needs Polars type markers for the numpy training path used by
    M01-A.  Some Windows environments ship a Polars wheel whose startup CPU
    probe raises before CatBoost can import; a minimal marker module preserves
    CatBoost's normal numpy behavior without changing model parameters.
    """
    try:
        import polars  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        error_text = str(exc).lower()
        if not any(token in error_text for token in ("polars", "cpu", "feature flag", "sse")):
            raise
        marker = types.ModuleType("polars")
        marker.DataFrame = type("DataFrame", (), {})
        marker.Series = type("Series", (), {})
        marker.Object = object
        marker.__version__ = "0.0.0"
        sys.modules["polars"] = marker
    try:
        from catboost import CatBoostRegressor
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "CatBoost is unavailable or incompatible in this environment; "
            "install mscapital[models] before running M01 training."
        ) from exc
    return CatBoostRegressor


@dataclass
class CatBoostResidualRegressor:
    max_iterations: int = 3000
    early_stopping_rounds: int = 200
    random_seed: int = 0
    learning_rate: float = 0.02
    depth: int = 6
    l2_leaf_reg: float = 5.0
    subsample: float = 0.8
    colsample_bylevel: float = 0.8
    model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray, *, eval_set: tuple[np.ndarray, np.ndarray] | None = None) -> "CatBoostResidualRegressor":
        CatBoostRegressor = _load_catboost_regressor()
        kwargs: dict[str, Any] = dict(
            iterations=self.max_iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            bootstrap_type="Bernoulli",
            loss_function="RMSE",
            random_seed=self.random_seed,
            verbose=False,
            allow_writing_files=False,
        )
        try:
            kwargs["subsample"] = self.subsample
            kwargs["colsample_bylevel"] = self.colsample_bylevel
            if self.early_stopping_rounds > 0:
                kwargs["od_type"] = "Iter"
                kwargs["od_wait"] = self.early_stopping_rounds
            self.model = CatBoostRegressor(**kwargs)
        except TypeError:
            # Older CatBoost builds do not expose every regularization alias.
            kwargs.pop("subsample", None)
            kwargs.pop("colsample_bylevel", None)
            self.model = CatBoostRegressor(**kwargs)
        self.model.fit(X, y, eval_set=eval_set, use_best_model=eval_set is not None)
        return self

    @property
    def best_iteration(self) -> int | None:
        if self.model is None:
            return None
        value = self.model.get_best_iteration()
        return None if value is None or value < 0 else int(value)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit must be called before predict")
        return np.asarray(self.model.predict(X), dtype=np.float64).reshape(-1)
