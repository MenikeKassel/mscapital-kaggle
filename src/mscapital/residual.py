"""Canonical rolling OOF and outer-fold residual construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .metrics import cosine_uncentered
from .splits import ROLLING_WINDOWS, visible_oof_end


@dataclass(frozen=True)
class OOFBlock:
    name: str
    sample_id: np.ndarray
    month: np.ndarray
    target: np.ndarray
    baseline_oof: np.ndarray
    source_train_end: int

    def validate(self) -> None:
        arrays = [self.sample_id, self.month, self.target, self.baseline_oof]
        if len({np.asarray(a).reshape(-1).shape for a in arrays}) != 1:
            raise ValueError(f"{self.name}: OOF arrays must have equal length")
        months = np.asarray(self.month).reshape(-1)
        if months.size == 0 or months.min() <= self.source_train_end:
            raise ValueError(f"{self.name}: OOF contains a month not after source_train_end")


@dataclass(frozen=True)
class CanonicalOOF:
    sample_id: np.ndarray
    month: np.ndarray
    target: np.ndarray
    baseline_oof: np.ndarray
    source_train_end: np.ndarray

    def validate(self) -> None:
        arrays = [self.sample_id, self.month, self.target, self.baseline_oof, self.source_train_end]
        if len({np.asarray(a).reshape(-1).shape for a in arrays}) != 1:
            raise ValueError("canonical OOF arrays must have equal length")
        sample_ids = np.asarray(self.sample_id).reshape(-1)
        if np.unique(sample_ids).size != sample_ids.size:
            raise ValueError("canonical OOF contains duplicate sample_id values")
        if not np.all(np.asarray(self.source_train_end) < np.asarray(self.month)):
            raise ValueError("canonical OOF contains future-leaking source_train_end")

    def save(self, path: str | Path) -> Path:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            sample_id=np.asarray(self.sample_id),
            month=np.asarray(self.month),
            target=np.asarray(self.target),
            baseline_oof=np.asarray(self.baseline_oof),
            source_train_end=np.asarray(self.source_train_end),
        )
        return target


def build_canonical_oof(blocks: Iterable[OOFBlock]) -> CanonicalOOF:
    blocks = tuple(blocks)
    if not blocks:
        raise ValueError("at least one OOF block is required")
    for block in blocks:
        block.validate()
    merged = CanonicalOOF(
        sample_id=np.concatenate([np.asarray(b.sample_id).reshape(-1) for b in blocks]),
        month=np.concatenate([np.asarray(b.month).reshape(-1) for b in blocks]),
        target=np.concatenate([np.asarray(b.target).reshape(-1) for b in blocks]),
        baseline_oof=np.concatenate([np.asarray(b.baseline_oof).reshape(-1) for b in blocks]),
        source_train_end=np.concatenate([
            np.full(np.asarray(b.sample_id).reshape(-1).shape, b.source_train_end, dtype=np.int16)
            for b in blocks
        ]),
    )
    order = np.lexsort((merged.sample_id, merged.month))
    canonical = CanonicalOOF(
        sample_id=merged.sample_id[order],
        month=merged.month[order],
        target=merged.target[order],
        baseline_oof=merged.baseline_oof[order],
        source_train_end=merged.source_train_end[order],
    )
    canonical.validate()
    return canonical


def rolling_window_spec() -> tuple[tuple[str, tuple[int, int], tuple[int, int]], ...]:
    return tuple((name, train.as_tuple(), valid.as_tuple()) for name, train, valid in ROLLING_WINDOWS)


def outer_residual(
    canonical: CanonicalOOF,
    outer_name: str,
) -> dict[str, np.ndarray | float]:
    """Return the historical residual view visible to one outer fold."""

    canonical.validate()
    end = visible_oof_end(outer_name)
    mask = np.asarray(canonical.month) <= end
    if not mask.any():
        raise ValueError(f"no canonical OOF rows visible to {outer_name}")
    p = np.asarray(canonical.baseline_oof, dtype=np.float64)[mask]
    y = np.asarray(canonical.target, dtype=np.float64)[mask]
    denominator = float(np.dot(p, p))
    beta = 0.0 if denominator == 0 else float(np.dot(p, y) / denominator)
    residual = y - beta * p
    score = cosine_uncentered(beta * p, y)
    return {
        "sample_id": np.asarray(canonical.sample_id)[mask],
        "month": np.asarray(canonical.month)[mask],
        "target": y,
        "baseline_oof": p,
        "residual": residual,
        "beta": beta,
        "baseline_cosine": score,
    }
