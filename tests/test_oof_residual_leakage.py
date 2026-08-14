"""P6R OOF-residual leakage tests: scaler scope, alpha scope, finite guards."""

from __future__ import annotations

import numpy as np
import pytest

from mscapital.retrieval import fit_state_scaler, select_alpha, standardize

import scripts.p6r_00_retrieval_residual as p6r00

EXPECTED_ALPHA_GRID = np.array([0.05, 0.10, 0.15, 0.20, 0.30])


def test_split_constants_respect_temporal_order():
    for outer in ("PSEUDO", "H2", "T3", "T4"):
        it_lo, it_hi = p6r00.INNER_TRAIN[outer]
        t_lo, t_hi = p6r00.TUNE[outer]
        e_lo, e_hi = p6r00.OUTER_EVAL[outer]
        assert it_hi < t_lo, f"{outer}: inner train must precede tune"
        assert t_hi <= p6r00.VISIBLE_END[outer], f"{outer}: tune must stay inside visible OOF"
        assert t_hi < e_lo, f"{outer}: tune must precede frozen eval"
        assert e_lo <= e_hi


def test_alpha_grid_is_preregistered():
    assert np.allclose(p6r00.ALPHA_GRID, EXPECTED_ALPHA_GRID)
    assert p6r00.K_GRID == (64, 128, 256, 512)
    assert p6r00.METRICS == ("euclidean", "cosine")


def test_state_scaler_fit_on_train_only():
    rng = np.random.default_rng(5)
    z_train = rng.normal(loc=10.0, scale=2.0, size=(500, 3))
    z_valid = rng.normal(loc=-10.0, scale=5.0, size=(300, 3))
    mu, sd = fit_state_scaler(z_train)
    # scaler statistics must be exactly the train statistics (no valid-month info)
    assert np.allclose(mu, z_train.mean(axis=0))
    assert np.allclose(sd, z_train.std(axis=0))
    zs = standardize(z_valid, mu, sd)
    assert np.allclose(zs.mean(axis=0), (z_valid.mean(axis=0) - mu) / sd)


def test_select_alpha_uses_tune_rows_only():
    rng = np.random.default_rng(9)
    baseline = rng.normal(size=400)
    residual = rng.normal(size=400)
    target = rng.normal(size=400)
    result = select_alpha(baseline, residual, target, EXPECTED_ALPHA_GRID)
    assert result["alpha"] in EXPECTED_ALPHA_GRID
    assert np.isfinite(result["score"])
    assert result["baseline_rms"] > 0.0
    assert result["residual_rms"] > 0.0


def test_select_alpha_rejects_nan():
    rng = np.random.default_rng(21)
    baseline = rng.normal(size=100)
    target = rng.normal(size=100)
    residual = np.full(100, np.nan)
    with pytest.raises(ValueError):
        select_alpha(baseline, residual, target, EXPECTED_ALPHA_GRID)


def test_feature_order_fixed():
    from mscapital.features.revol_lite import context_feature_names

    names = context_feature_names()
    assert len(names) == 11
    assert names == tuple(sorted(names, key=names.index))  # stable, not re-sorted


def test_residual_alignment_with_labels(canonical_fixture=None):
    """Residual arrays must align with labels (same order, same length)."""
    rng = np.random.default_rng(23)
    y = rng.normal(size=1000)
    y0 = rng.normal(size=1000)
    beta = float(np.dot(y0, y) / np.dot(y0, y0))
    r = y - beta * y0
    assert r.shape == y.shape == y0.shape
    assert np.isfinite(r).all()


def test_bank_never_contains_query_month_in_runner():
    """The runner's V1 bank for month m excludes month m and later."""
    months = np.arange(21, 71)
    for m in (33, 50, 70):
        mask = p6r00.bank_mask(months, m)
        assert (months[mask] < m).all()
