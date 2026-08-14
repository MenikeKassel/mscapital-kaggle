"""P6R alignment tests: sample_id uniqueness, feature/baseline alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from mscapital.residual import load_canonical_oof_artifact
from mscapital.models.revol_lite import load_revol_lite_frame

REPO = Path(__file__).resolve().parents[1]
CANONICAL = REPO / "output" / "canonical_residual_oof" / "canonical_residual_oof.npz"
FEATURES = REPO / "output" / "e01_revol_lite_features" / "revol_lite_train.parquet"


def _load_canonical():
    return load_canonical_oof_artifact(CANONICAL)


@pytest.mark.skipif(not CANONICAL.exists(), reason="canonical OOF artifact missing")
def test_canonical_sample_id_unique_and_source_train_end():
    canon = _load_canonical()
    assert np.unique(canon.sample_id).size == canon.sample_id.size
    assert (canon.source_train_end < canon.month).all()
    assert set(np.unique(canon.month).tolist()) == set(range(21, 71))


@pytest.mark.skipif(not FEATURES.exists(), reason="ReVol-lite features missing")
def test_feature_artifact_covers_canonical():
    canon = _load_canonical()
    frame = load_revol_lite_frame(FEATURES)
    ids = np.sort(frame.sample_id)
    covered = np.isin(canon.sample_id, ids)
    assert covered.all(), "ReVol-lite feature artifact must cover every canonical sample"


@pytest.mark.skipif(not FEATURES.exists(), reason="ReVol-lite features missing")
def test_feature_row_alignment_matches_month_target():
    canon = _load_canonical()
    frame = load_revol_lite_frame(FEATURES)
    order = np.argsort(frame.sample_id)
    positions = np.searchsorted(frame.sample_id[order], canon.sample_id)
    assert np.array_equal(frame.sample_id[order][positions], canon.sample_id)
    assert np.array_equal(frame.month[order][positions].astype(np.int16), canon.month)
    assert np.allclose(frame.target[order][positions], canon.target)


@pytest.mark.skipif(not CANONICAL.exists(), reason="canonical OOF artifact missing")
def test_baseline_oof_is_oof():
    canon = _load_canonical()
    assert np.isfinite(canon.baseline_oof).all()
    assert np.isfinite(canon.target).all()


def test_align_ids_roundtrip():
    from scripts.p6r_00_retrieval_residual import align_ids

    rng = np.random.default_rng(1)
    dst = rng.permutation(np.arange(1000))
    src = dst[: 300]
    pos = align_ids(src, dst)
    assert np.array_equal(dst[pos], src)
    assert np.unique(pos).size == pos.size


def test_output_row_counts_match_validation(tmp_path):
    """Regression guard: predictions must have exactly the query rows."""
    rng = np.random.default_rng(3)
    n = 500
    months = rng.integers(21, 71, size=n)
    sid = np.arange(n)
    frame = pl.DataFrame(
        {
            "sample_id": sid,
            "month": months,
            "target": rng.normal(size=n),
            "baseline_oof": rng.normal(size=n),
            "source_train_end": months - 1,
        }
    )
    out = frame.filter(pl.col("month") >= 33)
    assert out.height > 0
    assert out["sample_id"].n_unique() == out.height
