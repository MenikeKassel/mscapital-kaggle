"""Fold-safe numeric preprocessing.

The transformer intentionally has no global fit path.  ``fit`` must receive
the training fold, and ``transform`` is pure with respect to validation/test
data.  Missing values are represented by a dedicated quantile-bin sentinel;
finite out-of-domain values are clipped to the nearest training edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class _ColumnState:
    name: str
    mode: str
    median: float = 0.0
    scale: float = 1.0
    edges: np.ndarray | None = None
    missing_bin: int | None = None


class FoldSafePreprocessor:
    """Numeric fold-safe transformer with optional train-only correlation filter."""

    def __init__(
        self,
        feature_names: Sequence[str],
        *,
        quantile_columns: Iterable[str] = (),
        quantile_bins: int = 20,
        correlation_threshold: float | None = None,
        protected_columns: Iterable[str] = ("sample_id", "month", "target"),
    ) -> None:
        self.feature_names = tuple(feature_names)
        self.quantile_columns = frozenset(quantile_columns)
        self.quantile_bins = int(quantile_bins)
        self.correlation_threshold = correlation_threshold
        self.protected_columns = frozenset(protected_columns)
        self._states: list[_ColumnState] = []
        self.selected_features: tuple[str, ...] = ()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "FoldSafePreprocessor":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("X must be a 2-D array matching feature_names")
        if not np.isfinite(X[~np.isnan(X)]).all():
            raise ValueError("X contains infinities")
        if self.correlation_threshold is not None and y is None:
            raise ValueError("y is required for target-aware feature selection")
        keep = np.ones(X.shape[1], dtype=bool)
        if self.correlation_threshold is not None:
            target = np.asarray(y, dtype=np.float64).reshape(-1)
            if target.shape[0] != X.shape[0]:
                raise ValueError("y length does not match X")
            for j, name in enumerate(self.feature_names):
                if name in self.protected_columns:
                    continue
                column = X[:, j]
                valid = np.isfinite(column) & np.isfinite(target)
                if valid.sum() < 2 or np.std(column[valid]) == 0 or np.std(target[valid]) == 0:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(column[valid], target[valid])[0, 1])
                # A threshold of 0.99 means retain only columns whose absolute
                # target correlation is below the threshold; this is a pruning
                # guard, not a univariate alpha selector.
                if abs(corr) >= self.correlation_threshold:
                    keep[j] = False
        self.selected_features = tuple(
            name for name, selected in zip(self.feature_names, keep) if selected
        )
        self._states = []
        for j, name in enumerate(self.feature_names):
            column = X[:, j]
            finite = column[np.isfinite(column)]
            median = float(np.median(finite)) if finite.size else 0.0
            if name in self.quantile_columns:
                if finite.size == 0:
                    edges = np.zeros(self.quantile_bins + 1, dtype=np.float64)
                else:
                    edges = np.quantile(
                        finite,
                        np.linspace(0.0, 1.0, self.quantile_bins + 1),
                    )
                    edges = np.maximum.accumulate(edges)
                self._states.append(
                    _ColumnState(
                        name=name,
                        mode="quantile",
                        median=median,
                        edges=edges,
                        missing_bin=self.quantile_bins,
                    )
                )
            else:
                q1, q3 = np.quantile(finite, [0.25, 0.75]) if finite.size else (0.0, 1.0)
                scale = float(q3 - q1)
                if scale == 0.0 or not np.isfinite(scale):
                    scale = 1.0
                self._states.append(
                    _ColumnState(name=name, mode="robust", median=median, scale=scale)
                )
        self._keep_mask = keep
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit must be called before transform")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("X must be a 2-D array matching feature_names")
        output = np.empty_like(X, dtype=np.float32)
        for j, state in enumerate(self._states):
            column = X[:, j]
            if state.mode == "quantile":
                values = np.full(column.shape, float(state.missing_bin), dtype=np.float64)
                valid = np.isfinite(column)
                if valid.any():
                    edges = state.edges
                    assert edges is not None
                    # searchsorted against interior edges gives 0..bins-1;
                    # duplicate edges are harmless and deterministic.
                    values[valid] = np.searchsorted(edges[1:-1], column[valid], side="right")
                output[:, j] = values.astype(np.float32)
            else:
                values = np.where(np.isfinite(column), column, state.median)
                output[:, j] = ((values - state.median) / state.scale).astype(np.float32)
        return output[:, self._keep_mask]

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def state(self) -> tuple[_ColumnState, ...]:
        if not self._fitted:
            raise RuntimeError("fit must be called before reading state")
        return tuple(self._states)


class FoldSafeCategoricalEncoder:
    """Train-only vocabulary for categorical/event-code columns.

    Unknown and missing values share one sentinel after the training vocabulary
    has been frozen.  It is deliberately separate from the numeric transformer
    so mixed Arrow schemas can be encoded without coercing strings to floats.
    """

    def __init__(self) -> None:
        self._mapping: dict[str, int] | None = None
        self.unknown_code: int | None = None

    def fit(self, values: Iterable[object]) -> "FoldSafeCategoricalEncoder":
        mapping: dict[str, int] = {}
        for value in values:
            if value is None:
                continue
            try:
                if bool(np.isnan(value)):
                    continue
            except TypeError:
                pass
            key = str(value)
            if key not in mapping:
                mapping[key] = len(mapping)
        self._mapping = mapping
        self.unknown_code = len(mapping)
        return self

    def transform(self, values: Iterable[object]) -> np.ndarray:
        if self._mapping is None or self.unknown_code is None:
            raise RuntimeError("fit must be called before transform")
        result = []
        for value in values:
            try:
                missing = value is None or bool(np.isnan(value))
            except TypeError:
                missing = value is None
            key = str(value)
            result.append(self.unknown_code if missing else self._mapping.get(key, self.unknown_code))
        return np.asarray(result, dtype=np.int32)

    @property
    def vocabulary(self) -> tuple[str, ...]:
        if self._mapping is None:
            raise RuntimeError("fit must be called before reading vocabulary")
        return tuple(self._mapping)
