"""P6R retrieval primitives: temporal banks, KNN retrieval, local ridge, anchors.

Protocol contract (docs/p6r_preregistration.md, docs/p6r_repo_audit.md):
- every bank row satisfies ``month < query month`` (asserted; tests enforce it)
- state scaler is fit only on inner-train months (never on validation)
- alpha selection only on tune months (never on frozen outer validation)
- cosine retrieval is implemented as Euclidean on unit-normalized state,
  which is monotonic (1 - cos(a,b) = ||a/||a|| - b/||b||||^2 / 2) and lets all
  metrics share the same kd-tree path.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .metrics import cosine_uncentered

EPS = 1e-12


# --------------------------------------------------------------------------
# Temporal banks
# --------------------------------------------------------------------------

def bank_mask(month: np.ndarray, query_month: int, cap_month: int | None = None) -> np.ndarray:
    """Rows usable as neighbours for a query month.

    Strict temporal contract: ``month < query_month`` always holds. An
    optional ``cap_month`` (V2 variant) additionally caps the bank at the
    outer fold's visible OOF end.
    """
    month = np.asarray(month)
    mask = month < query_month
    if cap_month is not None:
        mask = mask & (month <= cap_month)
    return mask


def query_months(month: np.ndarray, first_month: int) -> list[int]:
    """Ascending query months with a non-empty historical bank."""
    months = sorted({int(m) for m in np.unique(np.asarray(month)) if int(m) >= first_month})
    return [m for m in months if int(np.asarray(month).min()) < m]


# --------------------------------------------------------------------------
# State scaling
# --------------------------------------------------------------------------

def fit_state_scaler(z_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std from training months only (leakage-safe)."""
    z = np.asarray(z_train, dtype=np.float64)
    mu = z.mean(axis=0)
    sd = z.std(axis=0)
    sd = np.where(sd < 1e-10, 1.0, sd)
    return mu, sd


def standardize(z: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (np.asarray(z, dtype=np.float64) - mu) / sd


def unit_normalize(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.maximum(norm, EPS)


# --------------------------------------------------------------------------
# KNN retrieval
# --------------------------------------------------------------------------

def gaussian_weights(dists: np.ndarray) -> np.ndarray:
    """w_i = exp(-0.5 (d_i / d_K)^2); d_K = distance to the K-th neighbour.

    Degenerate (zero) d_K falls back to equal weights so the kernel stays
    well conditioned.
    """
    dists = np.asarray(dists, dtype=np.float64)
    if dists.ndim == 1:
        dk = max(float(dists[-1]), EPS)
        return np.exp(-0.5 * (dists / dk) ** 2)
    dk = np.maximum(dists[:, -1:], EPS)
    return np.exp(-0.5 * (dists / dk) ** 2)


class StateKNN:
    """K-nearest-neighbour retrieval over a state bank.

    Uses FAISS IndexFlatL2 when available (exact brute force, SIMD); falls
    back to sklearn NearestNeighbors. FAISS returns squared L2 distances,
    which are converted back to L2 so weights and diagnostics are identical
    across backends.
    """

    def __init__(self, metric: str = "euclidean", n_jobs: int = -1) -> None:
        if metric not in ("euclidean", "cosine"):
            raise ValueError(f"unknown retrieval metric: {metric}")
        self.metric = metric
        self.n_jobs = n_jobs
        self._nn: Any = None
        self._faiss: Any = None
        self.n_samples_fit: int = 0

    def fit(self, z_bank: np.ndarray, k: int) -> "StateKNN":
        z = np.ascontiguousarray(z_bank, dtype=np.float32)
        if self.metric == "cosine":
            z = np.ascontiguousarray(unit_normalize(z).astype(np.float32))
        self.n_samples_fit = z.shape[0]
        try:
            import faiss  # type: ignore

            self._faiss = faiss.IndexFlatL2(z.shape[1])
            self._faiss.add(z)
            self._nn = None
        except ImportError:
            from sklearn.neighbors import NearestNeighbors

            self._nn = NearestNeighbors(
                n_neighbors=min(k, self.n_samples_fit),
                metric="euclidean",
                algorithm="auto",
                n_jobs=self.n_jobs,
            )
            self._nn.fit(z)
        return self

    def query(self, z_query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._nn is None and self._faiss is None:
            raise RuntimeError("fit must be called before query")
        z = np.ascontiguousarray(z_query, dtype=np.float32)
        if self.metric == "cosine":
            z = np.ascontiguousarray(unit_normalize(z).astype(np.float32))
        n = min(k, self.n_samples_fit)
        if self._faiss is not None:
            dists_sq, idx = self._faiss.search(z, n)
            idx = idx.astype(np.int64)
            dists = np.sqrt(np.maximum(dists_sq, 0.0))
        else:
            dists, idx = self._nn.kneighbors(z, n_neighbors=n)
        weights = gaussian_weights(dists)
        return idx, dists, weights


def retrieve_residual_mean(idx: np.ndarray, weights: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """r_hat(q) = weighted mean of neighbour residuals."""
    w = np.asarray(weights, dtype=np.float64)
    r = np.asarray(residuals, dtype=np.float64)
    return (w * r[idx]).sum(axis=1) / np.maximum(w.sum(axis=1), EPS)


def neighbor_month_stats(neighbor_months: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-query neighbour-month entropy and unique-month count."""
    entropies = np.zeros(idx.shape[0], dtype=np.float64)
    uniques = np.zeros(idx.shape[0], dtype=np.int16)
    for i in range(idx.shape[0]):
        counts = np.bincount(neighbor_months[idx[i]].astype(np.int64))
        if counts.sum() == 0:
            continue
        p = counts[counts > 0] / counts.sum()
        entropies[i] = -float((p * np.log(p)).sum())
        uniques[i] = int((counts > 0).sum())
    return entropies, uniques


def support_score(kth_dists: np.ndarray, ref_dist: float) -> np.ndarray:
    """Distance-based support g(z) = exp(-0.5 (d_K / d_ref)^2) in [0, 1].

    ``d_ref`` is the reference scale (e.g. median kth distance over the tune
    queries). Diagnostic only in P6R-00; full gating arrives in P6R-03.
    """
    return np.exp(-0.5 * (np.asarray(kth_dists, dtype=np.float64) / max(float(ref_dist), EPS)) ** 2)


# --------------------------------------------------------------------------
# Local varying-coefficient ridge (P6R-01)
# --------------------------------------------------------------------------

def fit_local_ridge(
    z_query: np.ndarray,
    z_bank: np.ndarray,
    x_bank: np.ndarray,
    r_bank: np.ndarray,
    k: int,
    lam: float,
    metric: str = "euclidean",
) -> np.ndarray:
    """Per-query weighted ridge beta(z_q) (P6R-01).

    beta(z_q) = argmin_b sum_i w_i(z_q) (r_i - x_i^T b)^2 + lam ||b||^2
    Returns (n_queries, d) coefficient matrix. ``x_bank`` must already be
    standardized with a scaler fit on inner-train months only.
    """
    knn = StateKNN(metric).fit(z_bank, k)
    idx, dists, w = knn.query(z_query, k)
    x = np.asarray(x_bank, dtype=np.float64)
    r = np.asarray(r_bank, dtype=np.float64)
    d = x.shape[1]
    betas = np.empty((z_query.shape[0], d), dtype=np.float64)
    for i in range(z_query.shape[0]):
        ids = idx[i]
        xw = x[ids] * w[i][:, None]
        a = xw.T @ x[ids] + lam * np.eye(d)
        b = xw.T @ r[ids]
        betas[i] = np.linalg.solve(a, b)
    return betas


def fit_global_ridge(x_bank: np.ndarray, r_bank: np.ndarray, lam: float) -> np.ndarray:
    """One equal-weight ridge over the whole bank (P6R-01 control)."""
    x = np.asarray(x_bank, dtype=np.float64)
    r = np.asarray(r_bank, dtype=np.float64)
    d = x.shape[1]
    a = x.T @ x + lam * np.eye(d)
    b = x.T @ r
    return np.linalg.solve(a, b)


# --------------------------------------------------------------------------
# Retrieval anchors / local experts (P6R-02; implemented, not executed here)
# --------------------------------------------------------------------------

def fit_anchor_experts(
    z_train: np.ndarray,
    x_train: np.ndarray,
    r_train: np.ndarray,
    n_anchor: int,
    lam: float,
    *,
    seed: int = 2026,
    batch_size: int = 4096,
) -> tuple[Any, np.ndarray]:
    """Cluster state space, fit one ridge expert per anchor (P6R-02)."""
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(
        n_clusters=n_anchor, random_state=seed, batch_size=batch_size, n_init=3
    )
    labels = km.fit_predict(np.asarray(z_train, dtype=np.float32))
    x = np.asarray(x_train, dtype=np.float64)
    r = np.asarray(r_train, dtype=np.float64)
    d = x.shape[1]
    betas = np.empty((n_anchor, d), dtype=np.float64)
    for c in range(n_anchor):
        mask = labels == c
        a = x[mask].T @ x[mask] + lam * np.eye(d)
        b = x[mask].T @ r[mask]
        betas[c] = np.linalg.solve(a, b)
    return km, betas


def predict_anchor_mixture(
    km: Any, betas: np.ndarray, z_query: np.ndarray, x_query: np.ndarray, tau: float
) -> np.ndarray:
    """Soft distance gate: r_hat = sum_k p_k(z) (x^T beta_k)."""
    z = np.asarray(z_query, dtype=np.float32)
    d2 = np.asarray(km.transform(z), dtype=np.float64)  # (n, n_anchor) squared dist
    p = np.exp(-d2 / (2.0 * max(float(tau), EPS) ** 2))
    total = p.sum(axis=1, keepdims=True)
    p = np.divide(p, total, out=np.zeros_like(p), where=total > 0.0)
    x = np.asarray(x_query, dtype=np.float64)
    return np.einsum("nk,kd,nd->n", p, np.asarray(betas, dtype=np.float64), x)


# --------------------------------------------------------------------------
# Alpha selection (E01 convention: RMS-normalized blend)
# --------------------------------------------------------------------------

def select_alpha(
    baseline: np.ndarray,
    residual_pred: np.ndarray,
    target: np.ndarray,
    alpha_grid: np.ndarray,
) -> dict[str, Any]:
    """Select alpha on tune months; returns scales to reuse on frozen rows."""
    base = np.asarray(baseline, dtype=np.float64).reshape(-1)
    res = np.asarray(residual_pred, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    base_rms = float(np.sqrt(np.mean(base**2)))
    res_rms = float(np.sqrt(np.mean(res**2)))
    base_n = base / base_rms if base_rms > 0.0 else base
    res_n = res / res_rms if res_rms > 0.0 else res
    scores = np.array(
        [cosine_uncentered(base_n + float(a) * res_n, y) for a in alpha_grid]
    )
    best = int(np.argmax(scores))
    return {
        "alpha": float(alpha_grid[best]),
        "score": float(scores[best]),
        "baseline_score": cosine_uncentered(base_n, y),
        "baseline_rms": base_rms,
        "residual_rms": res_rms,
        "scores": {str(float(a)): float(s) for a, s in zip(alpha_grid, scores)},
    }


def blend_with_scales(
    baseline: np.ndarray,
    residual_pred: np.ndarray,
    base_rms: float,
    res_rms: float,
    alpha: float,
) -> np.ndarray:
    base = np.asarray(baseline, dtype=np.float64).reshape(-1)
    res = np.asarray(residual_pred, dtype=np.float64).reshape(-1)
    base_n = base / base_rms if base_rms > 0.0 else base
    res_n = res / res_rms if res_rms > 0.0 else res
    return base_n + float(alpha) * res_n
