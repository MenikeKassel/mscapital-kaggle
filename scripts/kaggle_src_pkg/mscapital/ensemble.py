"""Nested-calibration-compatible prediction ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .metrics import cosine_uncentered, normalize_prediction


@dataclass(frozen=True)
class CalibrationResult:
    method: str
    weight: float
    scale_a: float
    scale_b: float
    score: float


class EnsembleCalibrator:
    """Fit a two-component blend on inner temporal data only.

    ``fit`` stores the scales learned from the supplied inner predictions.
    Calling ``transform`` never re-estimates anything from outer or test data.
    """

    def __init__(self, methods: tuple[str, ...] = ("raw", "std", "rms")) -> None:
        self.methods = methods
        self.result: CalibrationResult | None = None

    def fit(
        self,
        pred_a: object,
        pred_b: object,
        target: object,
        *,
        weight_grid: np.ndarray | None = None,
    ) -> "EnsembleCalibrator":
        a, b, y = (np.asarray(v, dtype=np.float64).reshape(-1) for v in (pred_a, pred_b, target))
        if not (a.shape == b.shape == y.shape):
            raise ValueError("ensemble inputs must have the same shape")
        grid = np.arange(0.0, 1.0001, 0.01) if weight_grid is None else np.asarray(weight_grid)
        best: CalibrationResult | None = None
        for method in self.methods:
            aa, scale_a = normalize_prediction(a, method)
            bb, scale_b = normalize_prediction(b, method)
            for weight in grid:
                pred = (1.0 - float(weight)) * aa + float(weight) * bb
                score = cosine_uncentered(pred, y)
                candidate = CalibrationResult(method, float(weight), scale_a, scale_b, score)
                if best is None or candidate.score > best.score:
                    best = candidate
        assert best is not None
        self.result = best
        return self

    def transform(self, pred_a: object, pred_b: object) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("fit must be called before transform")
        a = np.asarray(pred_a, dtype=float).reshape(-1)
        b = np.asarray(pred_b, dtype=float).reshape(-1)
        # Use the inner learned scales, even when a component is constant in the
        # outer set.  This is the critical no-re-fit invariant.
        if self.result.scale_a != 0:
            a = np.asarray(pred_a, dtype=float).reshape(-1) / self.result.scale_a
        if self.result.scale_b != 0:
            b = np.asarray(pred_b, dtype=float).reshape(-1) / self.result.scale_b
        return (1.0 - self.result.weight) * a + self.result.weight * b

    def score(self, pred_a: object, pred_b: object, target: object) -> float:
        return cosine_uncentered(self.transform(pred_a, pred_b), target)


@dataclass(frozen=True)
class NestedBlendFold:
    """Predictions for one outer fold and its strictly earlier inner tune."""

    name: str
    inner_a: np.ndarray
    inner_b: np.ndarray
    inner_target: np.ndarray
    outer_a: np.ndarray
    outer_b: np.ndarray
    outer_target: np.ndarray


def evaluate_nested_blend(
    folds: Sequence[NestedBlendFold],
    *,
    methods: tuple[str, ...] = ("raw", "std", "rms"),
) -> list[dict[str, float | str]]:
    """Calibrate on each inner segment and score once on its outer segment."""

    results: list[dict[str, float | str]] = []
    for fold in folds:
        calibrator = EnsembleCalibrator(methods=methods).fit(
            fold.inner_a, fold.inner_b, fold.inner_target
        )
        assert calibrator.result is not None
        results.append(
            {
                "fold": fold.name,
                "method": calibrator.result.method,
                "weight": calibrator.result.weight,
                "inner_score": calibrator.result.score,
                "outer_score": calibrator.score(
                    fold.outer_a, fold.outer_b, fold.outer_target
                ),
            }
        )
    return results
