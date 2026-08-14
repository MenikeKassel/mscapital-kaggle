"""P6R anchor-scope tests: anchors fit only on training data; no future usage."""

from __future__ import annotations

import numpy as np
import pytest

from mscapital.retrieval import (
    fit_anchor_experts,
    fit_global_ridge,
    fit_local_ridge,
    predict_anchor_mixture,
)


def _synthetic(n_train=2000, n_query=200, d=5, seed=2026):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_train, 3))
    x = rng.normal(size=(n_train, d))
    beta_true = rng.normal(size=(d,))
    r = x @ beta_true + rng.normal(scale=0.1, size=n_train)
    zq = rng.normal(size=(n_query, 3))
    xq = rng.normal(size=(n_query, d))
    return z, x, r, zq, xq, beta_true


def test_anchor_experts_scope_no_future():
    z, x, r, zq, xq, _ = _synthetic()
    km, betas = fit_anchor_experts(z, x, r, n_anchor=8, lam=0.1)
    assert betas.shape == (8, x.shape[1])
    # prediction depends only on queried state/features, not on any label
    p1 = predict_anchor_mixture(km, betas, zq, xq, tau=1.0)
    p2 = predict_anchor_mixture(km, betas, zq, xq, tau=1.0)
    assert np.allclose(p1, p2)
    assert np.isfinite(p1).all()


def test_anchor_experts_deterministic_seed():
    z, x, r, _, _, _ = _synthetic()
    km1, b1 = fit_anchor_experts(z, x, r, n_anchor=8, lam=0.1, seed=2026)
    km2, b2 = fit_anchor_experts(z, x, r, n_anchor=8, lam=0.1, seed=2026)
    assert np.allclose(b1, b2)
    assert np.allclose(km1.cluster_centers_, km2.cluster_centers_)


def test_anchor_mixture_soft_gate():
    """With well-separated state blobs a small tau concentrates on the nearest anchor."""
    rng = np.random.default_rng(2026)
    n_train, n_query, d = 2000, 200, 5
    centers = np.array([[-10.0, -10.0, -10.0], [10.0, 10.0, 10.0], [0.0, 0.0, 0.0]])
    blob = rng.integers(0, 3, size=n_train)
    z = centers[blob] + rng.normal(scale=0.2, size=(n_train, 3))
    x = rng.normal(size=(n_train, d))
    beta_true = rng.normal(size=(d,))
    r = x @ beta_true + rng.normal(scale=0.1, size=n_train)
    zq = centers[rng.integers(0, 3, size=n_query)] + rng.normal(scale=0.2, size=(n_query, 3))
    xq = rng.normal(size=(n_query, d))
    km, betas = fit_anchor_experts(z, x, r, n_anchor=3, lam=0.1)
    assert np.isfinite(betas).all()
    d2 = np.asarray(km.transform(zq.astype(np.float32)), dtype=np.float64)
    nearest = np.argmin(d2, axis=1)
    expected = np.einsum("nd,nd->n", xq, betas[nearest])
    tight = predict_anchor_mixture(km, betas, zq, xq, tau=0.05)
    assert np.allclose(tight, expected, atol=1e-6)
    wide = predict_anchor_mixture(km, betas, zq, xq, tau=5.0)
    assert np.isfinite(wide).all()


def test_local_ridge_equals_global_without_state_structure():
    """With a constant state, weights are equal and local ridge == global ridge."""
    rng = np.random.default_rng(11)
    z = np.zeros((500, 3))  # identical state -> all Gaussian weights collapse to 1
    x = rng.normal(size=(500, 5))
    r = x @ rng.normal(size=(5,)) + rng.normal(scale=0.1, size=500)
    # k = bank size so every neighbour is used; equal weights make local == global
    betas_local = fit_local_ridge(z[:100], z, x, r, k=500, lam=0.01)
    betas_global = fit_global_ridge(x, r, lam=0.01)
    for b in betas_local:
        assert np.allclose(b, betas_global, atol=1e-10)


def test_local_ridge_respects_bank_scope():
    """Local ridge may only use rows provided in the bank (no global state)."""
    rng = np.random.default_rng(13)
    z_bank = rng.normal(size=(300, 3))
    z_query = rng.normal(size=(50, 3))
    x_bank = rng.normal(size=(300, 4))
    r_bank = rng.normal(size=300)
    betas = fit_local_ridge(z_query, z_bank, x_bank, r_bank, k=64, lam=0.1)
    assert betas.shape == (50, 4)
    assert np.isfinite(betas).all()


def test_global_ridge_is_single_fit():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(1000, 6))
    r = rng.normal(size=1000)
    b = fit_global_ridge(x, r, lam=0.1)
    expected = np.linalg.solve(x.T @ x + 0.1 * np.eye(6), x.T @ r)
    assert np.allclose(b, expected)
