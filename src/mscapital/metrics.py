"""Competition metric and scale-safe ensemble primitives."""

from __future__ import annotations

import numpy as np


def _vector(value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("metric vectors must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("metric vectors must contain only finite values")
    return arr


def cosine_uncentered(pred: object, target: object) -> float:
    """Return the MSCapital cosine without subtracting either mean.

    A zero vector has no direction.  Returning 0.0 keeps diagnostics and grid
    searches deterministic and prevents a zero prediction from appearing
    competitive due to a numerical epsilon.
    """

    p, y = _vector(pred), _vector(target)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {y.shape}")
    denominator = float(np.linalg.norm(p) * np.linalg.norm(y))
    return 0.0 if denominator == 0.0 else float(np.dot(p, y) / denominator)


def cosine_centered(pred: object, target: object) -> float:
    p, y = _vector(pred), _vector(target)
    return cosine_uncentered(p - p.mean(), y - y.mean())


def rms_scale(pred: object) -> float:
    p = _vector(pred)
    return float(np.sqrt(np.mean(np.square(p))))


def std_scale(pred: object) -> float:
    p = _vector(pred)
    return float(np.std(p))


def normalize_prediction(pred: object, method: str) -> tuple[np.ndarray, float]:
    p = _vector(pred)
    if method == "raw":
        scale = 1.0
    elif method == "std":
        scale = std_scale(p)
    elif method == "rms":
        scale = rms_scale(p)
    else:
        raise ValueError(f"unknown normalization method: {method}")
    return (p.copy() if scale == 0.0 else p / scale), scale
