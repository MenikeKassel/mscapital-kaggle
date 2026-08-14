"""P6R leakage tests: no future months may enter the retrieval bank."""

from __future__ import annotations

import numpy as np
import pytest

from mscapital.retrieval import StateKNN, bank_mask, query_months, retrieve_residual_mean


def test_bank_mask_excludes_query_month_and_future():
    months = np.arange(21, 71)
    assert not bank_mask(months, 21).any()  # first OOF month has an empty bank
    for m in (30, 55, 70):
        mask = bank_mask(months, m)
        assert months[mask].max() < m, f"bank for query month {m} contains future rows"
        assert (months[mask] < m).all()


def test_bank_mask_cap_month():
    months = np.arange(21, 71)
    mask = bank_mask(months, 70, cap_month=50)
    assert months[mask].max() == 50
    assert (months[mask] <= 50).all()


def test_query_months_ascending_with_nonempty_bank():
    months = np.arange(21, 71)
    q = query_months(months, 27)
    assert q == sorted(q)
    assert q[0] >= 27
    for m in q:
        assert (months < m).any()


def test_state_knn_never_retrieves_future_month():
    rng = np.random.default_rng(2026)
    # state is month identity + noise, so nearest neighbours are same-month rows
    months = np.repeat(np.arange(21, 40), 100)
    n = months.size
    z = np.column_stack([months.astype(float), rng.normal(size=n)])
    query_mask = months == 35
    bank_mask_rows = months < 35
    knn = StateKNN("euclidean").fit(z[bank_mask_rows], 64)
    idx, _, _ = knn.query(z[query_mask], 64)
    neighbor_months = months[bank_mask_rows][idx]
    assert (neighbor_months < 35).all(), "retrieval returned a same/future-month neighbour"


def test_cosine_knn_monotonic_with_euclidean_on_unit_sphere():
    from mscapital.retrieval import unit_normalize

    rng = np.random.default_rng(7)
    bank = rng.normal(size=(500, 11))
    q = rng.normal(size=(50, 11))
    idx_c, _, _ = StateKNN("cosine").fit(bank, 16).query(q, 16)
    # cosine distance == Euclidean on the unit sphere, so cosine KNN must equal
    # Euclidean KNN applied to unit-normalized rows
    idx_e, _, _ = StateKNN("euclidean").fit(unit_normalize(bank), 16).query(unit_normalize(q), 16)
    assert np.array_equal(idx_c, idx_e)


def test_retrieve_residual_mean_weighted():
    residuals = np.array([1.0, 2.0, 4.0])
    idx = np.array([[0, 1], [1, 2]])
    weights = np.array([[1.0, 3.0], [1.0, 1.0]])
    out = retrieve_residual_mean(idx, weights, residuals)
    assert np.allclose(out, [(1.0 + 6.0) / 4.0, (2.0 + 4.0) / 2.0])


def test_retrieve_residual_mean_handles_zero_weights():
    residuals = np.array([1.0, 2.0])
    idx = np.array([[0, 1]])
    weights = np.array([[0.0, 0.0]])
    out = retrieve_residual_mean(idx, weights, residuals)
    assert np.allclose(out, [0.0])
    assert np.isfinite(out).all()


def test_gaussian_weights_range():
    from mscapital.retrieval import gaussian_weights

    w = gaussian_weights(np.array([0.1, 0.5, 1.0]))
    assert w[0] > w[1] > w[2] > 0.0
    assert np.isclose(w[2], np.exp(-0.5))
